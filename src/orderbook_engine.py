#!/usr/bin/env python3
"""
ORDERBOOK DEPTH ANALYSIS ENGINE
Real-time L2 orderbook analysis for Kraken Futures trading bot.
Detects liquidity walls, imbalances, stop clusters, and generates trading signals.

Free Kraken Futures orderbook API - no auth required!
"""

import time
import requests
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from loguru import logger
import threading


@dataclass
class OrderbookSnapshot:
    """Single orderbook snapshot with analysis."""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, qty), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    spread_pct: float
    imbalance: float  # Bid volume / Ask volume
    bid_depth: float  # Total bid volume
    ask_depth: float  # Total ask volume


@dataclass 
class LiquidityWall:
    """Detected liquidity wall."""
    price: float
    size: float
    side: str  # 'bid' or 'ask'
    strength: float  # Size relative to average
    distance_pct: float  # Distance from mid price


class OrderbookEngine:
    """Analyzes L2 orderbook depth for trading signals."""
    
    def __init__(self):
        self.cache = {}  # pair → orderbook data
        self.cache_ttl = 60  # Refresh every 60 seconds (orderbooks change fast)
        self.history = {}  # Track orderbook changes over time
        self.request_count = 0
        self.max_requests_per_cycle = 20  # Don't hammer the API
        self.last_reset_time = time.time()
        
        # Rate limiting
        self.request_times = []  # Track request timestamps for rate limiting
        self.max_requests_per_minute = 20
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Kraken pair mapping (futures vs spot)
        self.futures_pairs = {
            'XBTUSD': 'PF_XBTUSD',
            'ETHUSD': 'PF_ETHUSD', 
            'SOLUSD': 'PF_SOLUSD',
            'ADAUSD': 'PF_ADAUSD',
            'XRPUSD': 'PF_XRPUSD',
            'DOTUSD': 'PF_DOTUSD',
            'LINKUSD': 'PF_LINKUSD',
            'LTCUSD': 'PF_LTCUSD',
            'BCHUSD': 'PF_BCHUSD',
            'AVAXUSD': 'PF_AVAXUSD',
            'ATOMUSD': 'PF_ATOMUSD',
            'UNIUSD': 'PF_UNIUSD',
            # Add more as available on Kraken Futures
        }
        
        # Spot pairs fallback (when futures not available)
        self.spot_pairs = {
            'XBTUSD': 'XXBTZUSD',
            'ETHUSD': 'XETHZUSD',
            'SOLUSD': 'SOLUSD',
            'ADAUSD': 'ADAUSD',
            'XRPUSD': 'XRPZUSD',
            'DOTUSD': 'DOTUSD',
            'LINKUSD': 'LINKUSD',
            'LTCUSD': 'XLTCZUSD',
            'BCHUSD': 'BCHUSD',
            'AVAXUSD': 'AVAXUSD',
            'ATOMUSD': 'ATOMUSD',
            'UNIUSD': 'UNIUSD',
            'NEARUSD': 'NEARUSD',
            'AAVEUSD': 'AAVEUSD',
            'XLMUSD': 'XLMZUSD',
            'DOGEUSD': 'DOGEUSD',
            'FILUSD': 'FILUSD',
            # Add remaining pairs
        }
    
    def _can_make_request(self) -> bool:
        """Check if we can make another request without hitting rate limits."""
        now = time.time()
        
        # Clean old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Reset daily counter if needed
        if now - self.last_reset_time > 300:  # Every 5 minutes
            self.request_count = 0
            self.last_reset_time = now
        
        # Check limits
        if len(self.request_times) >= self.max_requests_per_minute:
            return False
        if self.request_count >= self.max_requests_per_cycle:
            return False
            
        return True
    
    def _fetch_futures_orderbook(self, pair: str) -> Optional[Dict]:
        """Fetch orderbook from Kraken Futures (free public endpoint)."""
        if not self._can_make_request():
            return None
            
        futures_symbol = self.futures_pairs.get(pair)
        if not futures_symbol:
            return None
            
        try:
            url = f'https://futures.kraken.com/derivatives/api/v3/orderbook?symbol={futures_symbol}'
            
            self.request_times.append(time.time())
            self.request_count += 1
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('result') != 'success':
                return None
                
            orderbook = data.get('orderBook', {})
            bids = [[float(p), float(q)] for p, q in orderbook.get('bids', [])]
            asks = [[float(p), float(q)] for p, q in orderbook.get('asks', [])]
            
            return {'bids': bids, 'asks': asks}
            
        except Exception as e:
            logger.debug(f"Failed to fetch Kraken Futures orderbook for {pair}: {e}")
            return None
    
    def _fetch_spot_orderbook(self, pair: str) -> Optional[Dict]:
        """Fetch orderbook from Kraken Spot (backup)."""
        if not self._can_make_request():
            return None
            
        spot_symbol = self.spot_pairs.get(pair)
        if not spot_symbol:
            return None
        
        try:
            url = f'https://api.kraken.com/0/public/Depth?pair={spot_symbol}&count=25'
            
            self.request_times.append(time.time())
            self.request_count += 1
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('error'):
                return None
                
            result = data.get('result', {})
            pair_data = next(iter(result.values()), {})
            
            bids = [[float(p), float(q)] for p, q, _ in pair_data.get('bids', [])]
            asks = [[float(p), float(q)] for p, q, _ in pair_data.get('asks', [])]
            
            return {'bids': bids, 'asks': asks}
            
        except Exception as e:
            logger.debug(f"Failed to fetch Kraken Spot orderbook for {pair}: {e}")
            return None
    
    def get_orderbook_data(self, pair: str, use_cache: bool = True) -> Optional[Dict]:
        """Get orderbook data with caching and fallback."""
        with self._lock:
            # Check cache first
            if use_cache and pair in self.cache:
                cached_data, cached_time = self.cache[pair]
                if time.time() - cached_time < self.cache_ttl:
                    return cached_data
            
            # Use Kraken Spot orderbook (clean [price, qty] format)
            # Futures PF_ orderbook has inverted bid format — unreliable
            data = self._fetch_spot_orderbook(pair)
            
            # Fallback to futures if spot unavailable
            if not data:
                data = self._fetch_futures_orderbook(pair)
            
            if data and data.get('bids') and data.get('asks'):
                # Cache the result
                self.cache[pair] = (data, time.time())
                
                # Add 1-second delay to be respectful
                time.sleep(1.0)
                
                return data
                
            return None
    
    def calc_imbalance(self, bids: List[List[float]], asks: List[List[float]], 
                      depth_pct: float = 0.02) -> float:
        """
        Compare total bid volume vs ask volume within X% of mid price.
        Imbalance > 2.0 = strong buy pressure (bids dominate)
        Imbalance < 0.5 = strong sell pressure (asks dominate)
        """
        if not bids or not asks:
            return 1.0
            
        mid = (bids[0][0] + asks[0][0]) / 2
        
        bid_vol = sum(qty for price, qty in bids if price >= mid * (1 - depth_pct))
        ask_vol = sum(qty for price, qty in asks if price <= mid * (1 + depth_pct))
        
        return bid_vol / ask_vol if ask_vol > 0 else 3.0
    
    def detect_walls(self, bids: List[List[float]], asks: List[List[float]], 
                    threshold_mult: float = 5.0) -> List[LiquidityWall]:
        """
        Detect unusually large orders (>5x average order size).
        These act as support (bid walls) or resistance (ask walls).
        """
        if not bids or not asks:
            return []
            
        # Calculate average size from top orders
        all_sizes = [qty for _, qty in bids[:20]] + [qty for _, qty in asks[:20]]
        if not all_sizes:
            return []
            
        avg_size = np.mean(all_sizes)
        threshold = avg_size * threshold_mult
        mid = (bids[0][0] + asks[0][0]) / 2
        
        walls = []
        
        # Check bid walls (support)
        for price, qty in bids[:25]:
            if qty > threshold:
                walls.append(LiquidityWall(
                    price=price,
                    size=qty,
                    side='bid',
                    strength=qty / avg_size,
                    distance_pct=abs(price - mid) / mid * 100
                ))
        
        # Check ask walls (resistance) 
        for price, qty in asks[:25]:
            if qty > threshold:
                walls.append(LiquidityWall(
                    price=price,
                    size=qty,
                    side='ask',
                    strength=qty / avg_size,
                    distance_pct=abs(price - mid) / mid * 100
                ))
        
        return walls
    
    def calc_spread_metrics(self, bids: List[List[float]], asks: List[List[float]]) -> Dict[str, float]:
        """
        Calculate spread and related metrics.
        Tight spread = liquid, confident market
        Wide spread = uncertain, potential move coming
        """
        if not bids or not asks:
            return {}
            
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = (best_ask - best_bid) / best_bid * 100  # as percentage
        
        return {
            'spread_pct': spread,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': (best_bid + best_ask) / 2,
        }
    
    def detect_stop_clusters(self, bids: List[List[float]], asks: List[List[float]], 
                           current_price: float) -> Dict[str, Any]:
        """
        Detect potential stop loss clusters by finding gaps in liquidity.
        Stops tend to cluster at round numbers and recent highs/lows.
        """
        if not bids or not asks:
            return {}
        
        # Look for gaps in the orderbook (areas with very low liquidity)
        bid_gaps = []
        ask_gaps = []
        
        # Analyze bid side (potential stop losses below price)
        for i in range(len(bids) - 1):
            current_size = bids[i][1] 
            next_size = bids[i + 1][1]
            price_gap = (bids[i][0] - bids[i + 1][0]) / bids[i][0]
            
            # Look for thin areas (small size) with price gaps
            if price_gap > 0.002 and min(current_size, next_size) < np.mean([b[1] for b in bids[:10]]) * 0.3:
                bid_gaps.append({
                    'price': bids[i + 1][0],
                    'gap_pct': price_gap * 100,
                    'size': next_size,
                    'distance_pct': (current_price - bids[i + 1][0]) / current_price * 100
                })
        
        # Analyze ask side (potential stop losses above price)  
        for i in range(len(asks) - 1):
            current_size = asks[i][1]
            next_size = asks[i + 1][1] 
            price_gap = (asks[i + 1][0] - asks[i][0]) / asks[i][0]
            
            if price_gap > 0.002 and min(current_size, next_size) < np.mean([a[1] for a in asks[:10]]) * 0.3:
                ask_gaps.append({
                    'price': asks[i + 1][0],
                    'gap_pct': price_gap * 100,
                    'size': next_size,
                    'distance_pct': (asks[i + 1][0] - current_price) / current_price * 100
                })
        
        return {
            'bid_gaps': sorted(bid_gaps, key=lambda x: x['distance_pct'])[:5],  # Closest gaps
            'ask_gaps': sorted(ask_gaps, key=lambda x: x['distance_pct'])[:5]
        }
    
    def detect_absorption(self, pair: str) -> Optional[Dict]:
        """
        Compare current orderbook to previous snapshot.
        Detect walls being absorbed or price breaking through walls.
        """
        if pair not in self.history or len(self.history[pair]) < 2:
            return None
        
        prev = self.history[pair][-2]
        curr = self.history[pair][-1]
        
        absorbed_walls = []
        
        # Check if big walls from previous snapshot have disappeared
        for prev_wall in prev.get('walls', []):
            if prev_wall['strength'] > 7:  # Only track big walls
                # Look for similar wall in current snapshot
                wall_exists = any(
                    abs(curr_wall['price'] - prev_wall['price']) / prev_wall['price'] < 0.01
                    and curr_wall['side'] == prev_wall['side']
                    and curr_wall['strength'] > prev_wall['strength'] * 0.5
                    for curr_wall in curr.get('walls', [])
                )
                
                if not wall_exists:
                    absorbed_walls.append({
                        'price': prev_wall['price'],
                        'side': prev_wall['side'],
                        'strength': prev_wall['strength'],
                        'action': 'absorbed'  # Wall absorbed without price moving through
                    })
        
        return {
            'absorbed_walls': absorbed_walls,
            'price_change_pct': (curr['mid_price'] - prev['mid_price']) / prev['mid_price'] * 100
        } if absorbed_walls else None
    
    def calc_depth_momentum(self, pair: str) -> float:
        """
        Track how bid/ask depth is changing over recent snapshots.
        Positive = bullish momentum building (bids growing, asks shrinking)
        Negative = bearish momentum building (asks growing, bids shrinking)
        """
        if pair not in self.history or len(self.history[pair]) < 3:
            return 0.0
        
        recent = self.history[pair][-3:]  # Last 3 snapshots
        
        # Calculate depth changes
        bid_changes = []
        ask_changes = []
        
        for i in range(1, len(recent)):
            prev_bid = recent[i-1]['bid_depth']
            curr_bid = recent[i]['bid_depth']
            bid_change = (curr_bid - prev_bid) / prev_bid if prev_bid > 0 else 0
            bid_changes.append(bid_change)
            
            prev_ask = recent[i-1]['ask_depth'] 
            curr_ask = recent[i]['ask_depth']
            ask_change = (curr_ask - prev_ask) / prev_ask if prev_ask > 0 else 0
            ask_changes.append(ask_change)
        
        # Average the changes
        avg_bid_change = np.mean(bid_changes) if bid_changes else 0
        avg_ask_change = np.mean(ask_changes) if ask_changes else 0
        
        # Momentum = bid growth - ask growth
        momentum = avg_bid_change - avg_ask_change
        
        return momentum * 100  # Scale to percentage points
    
    def analyze_orderbook(self, pair: str, current_price: float) -> Optional[Dict]:
        """Perform complete orderbook analysis for a pair."""
        data = self.get_orderbook_data(pair)
        if not data:
            return None
        
        bids = data['bids']
        asks = data['asks']
        
        if not bids or not asks:
            return None
        
        # Calculate metrics
        imbalance = self.calc_imbalance(bids, asks)
        walls = self.detect_walls(bids, asks)
        spread_metrics = self.calc_spread_metrics(bids, asks)
        stop_clusters = self.detect_stop_clusters(bids, asks, current_price)
        
        # Calculate depth (total volume in top levels)
        bid_depth = sum(qty for _, qty in bids[:10])
        ask_depth = sum(qty for _, qty in asks[:10])
        
        # Create snapshot for history
        snapshot = {
            'timestamp': time.time(),
            'imbalance': imbalance,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'walls': [{'price': w.price, 'size': w.size, 'side': w.side, 'strength': w.strength} for w in walls],
            'spread_pct': spread_metrics.get('spread_pct', 0),
            'mid_price': spread_metrics.get('mid_price', current_price)
        }
        
        # Update history (keep last 12 snapshots = 1 hour at 5-min intervals)
        if pair not in self.history:
            self.history[pair] = []
        self.history[pair].append(snapshot)
        if len(self.history[pair]) > 12:
            self.history[pair].pop(0)
        
        # Calculate momentum and absorption
        depth_momentum = self.calc_depth_momentum(pair)
        absorption = self.detect_absorption(pair)
        
        # Generate signal
        signal = "neutral"
        signal_score = 0
        
        # Strong imbalance signals
        if imbalance > 3.0:
            signal = "strong_buy"
            signal_score = min(10, imbalance * 2)
        elif imbalance > 2.0:
            signal = "buy"  
            signal_score = (imbalance - 1) * 3
        elif imbalance < 0.3:
            signal = "strong_sell"
            signal_score = min(-10, (1/imbalance - 1) * -2)
        elif imbalance < 0.5:
            signal = "sell"
            signal_score = (1/imbalance - 1) * -3
        
        # Adjust for momentum
        if depth_momentum > 5:
            signal_score += 2
        elif depth_momentum < -5:
            signal_score -= 2
        
        # Adjust for spread (wide spread = uncertainty)
        spread = spread_metrics.get('spread_pct', 0)
        if spread > 0.5:  # Very wide spread
            signal_score *= 0.7  # Reduce confidence
        
        return {
            'pair': pair,
            'imbalance': imbalance,
            'walls': walls,
            'spread_pct': spread,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'depth_momentum': depth_momentum,
            'stop_clusters': stop_clusters,
            'absorption': absorption,
            'signal': signal,
            'signal_score': signal_score,
            'mid_price': spread_metrics.get('mid_price', current_price),
            'timestamp': time.time()
        }
    
    def get_pairs_to_scan(self, active_signals: List[Dict], all_pairs: List[str], 
                         active_positions: Dict) -> List[str]:
        """Smart pair selection for orderbook scanning."""
        to_scan = set()
        
        # 1. Pairs with active signals (confirm before trading)
        for sig in active_signals:
            to_scan.add(sig.get('pair', ''))
        
        # 2. Pairs with open positions (monitor for exit)
        for pair in active_positions.keys():
            to_scan.add(pair)
        
        # 3. Top volatile pairs (rotate through remaining)
        # For now, just add some high-volume pairs
        priority_pairs = ['XBTUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 'ADAUSD']
        for pair in priority_pairs:
            if pair in all_pairs:
                to_scan.add(pair)
        
        # 4. Rotate through remaining pairs (add 5 per cycle)
        remaining = [p for p in all_pairs if p not in to_scan]
        cycle_offset = (int(time.time()) // 300) % len(remaining) if remaining else 0  # 5-min cycles
        for i in range(min(5, len(remaining))):
            idx = (cycle_offset + i) % len(remaining)
            to_scan.add(remaining[idx])
        
        # Cap at 15 pairs per cycle to respect rate limits
        result = list(to_scan)[:15]
        return result
    
    def get_orderbook_signals(self, pairs_to_scan: Optional[List[str]] = None, 
                            market_data: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Get orderbook signals for specified pairs.
        Returns dict of pair -> analysis results.
        """
        if not pairs_to_scan:
            return {}
        
        results = {}
        
        for pair in pairs_to_scan:
            # Get current price from market data if available
            current_price = None
            if market_data and pair in market_data:
                current_price = market_data[pair].get('price')
            
            if not current_price:
                # Skip if we don't have price data
                continue
            
            try:
                analysis = self.analyze_orderbook(pair, current_price)
                if analysis:
                    results[pair] = analysis
            except Exception as e:
                logger.debug(f"Failed to analyze orderbook for {pair}: {e}")
                continue
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status and statistics."""
        return {
            'cached_pairs': len(self.cache),
            'pairs_with_history': len(self.history),
            'requests_this_cycle': self.request_count,
            'request_rate_ok': self._can_make_request(),
            'last_reset': self.last_reset_time,
            'supported_futures_pairs': len(self.futures_pairs),
            'supported_spot_pairs': len(self.spot_pairs)
        }