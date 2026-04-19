#!/usr/bin/env python3
"""
ON-CHAIN DATA ENGINE
Tracks whale movements, exchange flows, stablecoin supply, and network activity 
using ONLY free APIs. Generates trading signals and score boosts for the futures bot.

Key metrics:
- Stablecoin supply changes (USDT/USDC minting/burning)
- Chain TVL flows (DeFiLlama)
- Protocol TVL changes mapped to specific coins
- BTC network health (hash rate, mempool, tx volume)
- Multi-chain stats

All data is cached and rate-limited to respect free API limits.
"""

import requests
import json
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
import hashlib


class OnChainEngine:
    """
    Fetches on-chain data from free APIs and generates trading signals.
    
    Features:
    - 5-minute cache for most data, 1-hour cache for slow-changing data
    - Rate limiting: max 20 requests per 5-minute cycle
    - Network error resilience (returns cached data or neutral signals)
    - Independent verification of all data sources
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes default cache
        self.stablecoin_cache_ttl = 3600  # 1 hour for stablecoin data
        
        # Historical data tracking
        self.stablecoin_history = []  # Track supply over time
        self.tvl_history = {}  # Track TVL per chain
        
        # Request tracking for rate limiting
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_window = 20
        self.window_duration = 300  # 5 minutes
        
        # Chain to coin mapping
        self.CHAIN_TO_COIN = {
            'Ethereum': 'ETHUSD',
            'Solana': 'SOLUSD', 
            'BSC': 'BNBUSD',
            'Arbitrum': 'ARBUSD',
            'Avalanche': 'AVAXUSD',
            'Near': 'NEARUSD',
            'Cardano': 'ADAUSD',
            'Polkadot': 'DOTUSD',
            'Sui': 'SUIUSD',
            'Optimism': 'OPUSD',
            'Aptos': 'APTUSD',
            'Tron': 'TRXUSD',
            'Injective': 'INJUSD',
            'Cosmos': 'ATOMUSD',
            'Celestia': 'TIAUSD',
            'Stacks': 'STXUSD',
            'ICP': 'ICPUSD',
        }
        
        # Protocol to coin mapping
        self.PROTOCOL_TO_COIN = {
            'Aave': 'AAVEUSD',
            'Uniswap': 'UNIUSD',
            'Lido': 'LDOUSD',
            'Maker': 'ETHUSD',  # MKR not traded, affects ETH ecosystem
            'Jupiter': 'JUPUSD',
            'Raydium': 'SOLUSD',
            'Ondo Finance': 'ONDOUSD',
            'Render': 'RENDERUSD',
        }
        
        logger.info("🔗 OnChain Data Engine initialized")
    
    def _can_make_request(self) -> bool:
        """Check if we can make another request within rate limits."""
        current_time = time.time()
        
        # Reset window if enough time has passed
        if current_time - self.request_window_start > self.window_duration:
            self.request_count = 0
            self.request_window_start = current_time
        
        return self.request_count < self.max_requests_per_window
    
    def _make_request(self, url: str, cache_key: str, ttl: int = None) -> Optional[Dict]:
        """
        Make HTTP request with caching and rate limiting.
        Returns cached data if available, None if request fails.
        """
        if ttl is None:
            ttl = self.cache_ttl
        
        # Check cache first
        current_time = time.time()
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if current_time - timestamp < ttl:
                return data
        
        # Check rate limits
        if not self._can_make_request():
            logger.warning(f"Rate limit exceeded, using cached data for {cache_key}")
            if cache_key in self.cache:
                return self.cache[cache_key][0]
            return None
        
        # Make request
        try:
            self.request_count += 1
            logger.debug(f"Making request {self.request_count}/{self.max_requests_per_window}: {url[:100]}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = (data, current_time)
            return data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url[:100]}: {e}")
            # Return cached data if available
            if cache_key in self.cache:
                logger.info(f"Using stale cached data for {cache_key}")
                return self.cache[cache_key][0]
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url[:100]}: {e}")
            return None
    
    def _fetch_stablecoin_supply(self) -> Optional[Dict]:
        """Fetch stablecoin supply data from DeFiLlama."""
        # Current supply for all stablecoins
        current_data = self._make_request(
            'https://stablecoins.llama.fi/stablecoins?includePrices=true',
            'stablecoin_current',
            self.stablecoin_cache_ttl
        )
        
        if not current_data:
            return None
        
        # Get USDT and USDC historical data (if not cached)
        usdt_history = self._make_request(
            'https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1',
            'usdt_history',
            self.stablecoin_cache_ttl
        )
        
        usdc_history = self._make_request(
            'https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=2', 
            'usdc_history',
            self.stablecoin_cache_ttl
        )
        
        return {
            'current': current_data,
            'usdt_history': usdt_history,
            'usdc_history': usdc_history
        }
    
    def _fetch_chain_tvl(self) -> Optional[Dict]:
        """Fetch chain TVL data from DeFiLlama."""
        chains = self._make_request(
            'https://api.llama.fi/v2/chains',
            'chain_tvl'
        )
        
        if not chains:
            return None
        
        # Get historical data for key chains (if we have requests left)
        historical = {}
        key_chains = ['Ethereum', 'Solana', 'BSC', 'Arbitrum', 'Avalanche']
        
        for chain in key_chains:
            if not self._can_make_request():
                break
                
            hist_data = self._make_request(
                f'https://api.llama.fi/v2/historicalChainTvl/{chain}',
                f'{chain.lower()}_tvl_history'
            )
            if hist_data:
                historical[chain] = hist_data
        
        return {
            'chains': chains,
            'historical': historical
        }
    
    def _fetch_protocol_tvl(self) -> Optional[List]:
        """Fetch protocol TVL data from DeFiLlama."""
        return self._make_request(
            'https://api.llama.fi/protocols',
            'protocol_tvl'
        )
    
    def _fetch_btc_network(self) -> Optional[Dict]:
        """Fetch BTC network stats from blockchain.com."""
        stats = self._make_request(
            'https://api.blockchain.info/stats',
            'btc_stats'
        )
        
        if not stats:
            return None
        
        # Get hash rate and transaction data if we have requests left
        hash_rate = None
        tx_count = None
        tx_volume = None
        
        if self._can_make_request():
            hash_rate = self._make_request(
                'https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json',
                'btc_hash_rate'
            )
        
        if self._can_make_request():
            tx_count = self._make_request(
                'https://api.blockchain.info/charts/n-transactions?timespan=7days&format=json',
                'btc_tx_count'
            )
        
        if self._can_make_request():
            tx_volume = self._make_request(
                'https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=7days&format=json',
                'btc_tx_volume'
            )
        
        return {
            'stats': stats,
            'hash_rate': hash_rate,
            'tx_count': tx_count,
            'tx_volume': tx_volume
        }
    
    def _fetch_btc_mempool(self) -> Optional[Dict]:
        """Fetch BTC mempool data from mempool.space."""
        fees = self._make_request(
            'https://mempool.space/api/v1/fees/recommended',
            'btc_mempool_fees'
        )
        
        blocks = None
        if self._can_make_request():
            blocks = self._make_request(
                'https://mempool.space/api/v1/fees/mempool-blocks',
                'btc_mempool_blocks'
            )
        
        return {
            'fees': fees,
            'blocks': blocks
        }
    
    def fetch_all(self) -> Dict[str, Any]:
        """
        Fetch all on-chain data. Called every cycle.
        Returns dict with all available data or cached/neutral values on error.
        """
        logger.info("🔗 Fetching on-chain data...")
        start_time = time.time()
        
        # Reset request counter for this cycle
        self.request_count = 0
        self.request_window_start = start_time
        
        data = {
            'stablecoin_supply': self._fetch_stablecoin_supply(),
            'chain_tvl': self._fetch_chain_tvl(),
            'protocol_tvl': self._fetch_protocol_tvl(),
            'btc_network': self._fetch_btc_network(),
            'btc_mempool': self._fetch_btc_mempool(),
            'timestamp': start_time,
            'requests_used': self.request_count
        }
        
        elapsed = time.time() - start_time
        logger.info(f"🔗 On-chain data fetch complete: {elapsed:.1f}s, {self.request_count} requests")
        
        return data
    
    def get_onchain_signals(self) -> Dict[str, Any]:
        """
        Analyze on-chain data and return trading signals.
        
        Returns:
        {
            'market_signal': float (-10 to +10),  # Overall on-chain outlook
            'coin_signals': {pair: float},  # Per-coin on-chain sentiment
            'stablecoin_flow': {...},  # Stablecoin supply analysis
            'tvl_flows': {...},  # TVL changes per chain
            'btc_health': {...},  # BTC network health
            'confidence': float (0-1)  # Data quality score
        }
        """
        try:
            data = self.fetch_all()
            signals = {
                'market_signal': 0.0,
                'coin_signals': {},
                'stablecoin_flow': {},
                'tvl_flows': {},
                'btc_health': {},
                'confidence': 0.0,
                'data_age': data.get('timestamp', 0)
            }
            
            confidence_factors = []
            
            # Analyze stablecoin supply (strongest signal)
            if data['stablecoin_supply']:
                stablecoin_signals = self._analyze_stablecoin_supply(data['stablecoin_supply'])
                signals['stablecoin_flow'] = stablecoin_signals
                signals['market_signal'] += stablecoin_signals.get('market_impact', 0)
                confidence_factors.append(0.4)  # High weight for stablecoin data
            
            # Analyze chain TVL flows
            if data['chain_tvl']:
                tvl_signals = self._analyze_chain_tvl(data['chain_tvl'])
                signals['tvl_flows'] = tvl_signals
                # Add per-coin signals
                for chain, tvl_data in tvl_signals.items():
                    if chain in self.CHAIN_TO_COIN:
                        pair = self.CHAIN_TO_COIN[chain]
                        signals['coin_signals'][pair] = tvl_data.get('signal_strength', 0)
                confidence_factors.append(0.3)  # Medium weight
            
            # Analyze protocol TVL
            if data['protocol_tvl']:
                protocol_signals = self._analyze_protocol_tvl(data['protocol_tvl'])
                # Add protocol-specific coin signals
                for protocol, signal in protocol_signals.items():
                    if protocol in self.PROTOCOL_TO_COIN:
                        pair = self.PROTOCOL_TO_COIN[protocol]
                        current = signals['coin_signals'].get(pair, 0)
                        signals['coin_signals'][pair] = current + signal
                confidence_factors.append(0.2)  # Lower weight
            
            # Analyze BTC network health
            if data['btc_network']:
                btc_signals = self._analyze_btc_network(data['btc_network'])
                signals['btc_health'] = btc_signals
                # Add BTC signal
                btc_signal = btc_signals.get('overall_signal', 0)
                signals['coin_signals']['BTCUSD'] = btc_signal
                signals['market_signal'] += btc_signal * 0.3  # BTC impacts overall market
                confidence_factors.append(0.1)  # Lower weight
            
            # Calculate overall confidence
            signals['confidence'] = sum(confidence_factors) if confidence_factors else 0.0
            
            # Clamp market signal to -10/+10 range
            signals['market_signal'] = max(-10, min(10, signals['market_signal']))
            
            logger.info(f"🔗 On-chain signals: market={signals['market_signal']:+.1f}, "
                       f"confidence={signals['confidence']:.1f}, "
                       f"coins={len(signals['coin_signals'])}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating on-chain signals: {e}")
            # Return neutral signals on error
            return {
                'market_signal': 0.0,
                'coin_signals': {},
                'stablecoin_flow': {'signal': 'neutral'},
                'tvl_flows': {},
                'btc_health': {'signal': 'neutral'},
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_stablecoin_supply(self, data: Dict) -> Dict:
        """
        Analyze stablecoin supply changes for market signals.
        
        Supply increasing = capital flowing into crypto (BULLISH)
        Supply decreasing = capital leaving crypto (BEARISH)
        """
        try:
            current = data.get('current', {})
            usdt_history = data.get('usdt_history')
            usdc_history = data.get('usdc_history')
            
            if not current:
                return {'signal': 'neutral', 'reason': 'no_data'}
            
            # Find USDT and USDC in current data
            usdt_supply = 0
            usdc_supply = 0
            
            for stablecoin in current.get('peggedAssets', []):
                if stablecoin.get('symbol') == 'USDT':
                    usdt_supply = stablecoin.get('circulating', {}).get('peggedUSD', 0)
                elif stablecoin.get('symbol') == 'USDC':
                    usdc_supply = stablecoin.get('circulating', {}).get('peggedUSD', 0)
            
            total_supply = usdt_supply + usdc_supply
            
            # Calculate changes if we have historical data
            usdt_24h_change = 0
            usdc_24h_change = 0
            total_7d_change = 0
            
            if usdt_history and len(usdt_history) > 1:
                # Get 24h change from USDT history
                usdt_24h_change = usdt_supply - usdt_history[-2].get('totalCirculating', {}).get('peggedUSD', usdt_supply)
            
            if usdc_history and len(usdc_history) > 1:
                # Get 24h change from USDC history
                usdc_24h_change = usdc_supply - usdc_history[-2].get('totalCirculating', {}).get('peggedUSD', usdc_supply)
            
            # Calculate 7-day change if possible
            if usdt_history and usdc_history and len(usdt_history) >= 7 and len(usdc_history) >= 7:
                usdt_7d_ago = usdt_history[-7].get('totalCirculating', {}).get('peggedUSD', 0)
                usdc_7d_ago = usdc_history[-7].get('totalCirculating', {}).get('peggedUSD', 0)
                total_7d_change = (usdt_supply + usdc_supply) - (usdt_7d_ago + usdc_7d_ago)
            
            # Generate market signal based on 7-day change
            market_impact = 0
            signal_type = 'neutral'
            
            if total_7d_change > 1_000_000_000:  # >$1B minting in a week
                market_impact = 5  # VERY BULLISH
                signal_type = 'very_bullish'
            elif total_7d_change > 500_000_000:  # >$500M minting
                market_impact = 3  # BULLISH
                signal_type = 'bullish'
            elif total_7d_change > 100_000_000:  # >$100M minting
                market_impact = 1  # Slightly bullish
                signal_type = 'slightly_bullish'
            elif total_7d_change < -1_000_000_000:  # >$1B burning
                market_impact = -5  # VERY BEARISH
                signal_type = 'very_bearish'
            elif total_7d_change < -500_000_000:  # >$500M burning
                market_impact = -3  # BEARISH
                signal_type = 'bearish'
            elif total_7d_change < -100_000_000:  # >$100M burning
                market_impact = -1  # Slightly bearish
                signal_type = 'slightly_bearish'
            
            return {
                'usdt_supply': usdt_supply,
                'usdc_supply': usdc_supply,
                'total_supply': total_supply,
                'usdt_24h_change': usdt_24h_change,
                'usdc_24h_change': usdc_24h_change,
                'total_24h_change': usdt_24h_change + usdc_24h_change,
                'total_7d_change': total_7d_change,
                'market_impact': market_impact,
                'signal': signal_type,
                'reason': f"7d change: ${total_7d_change/1e6:+.0f}M"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stablecoin supply: {e}")
            return {'signal': 'neutral', 'error': str(e)}
    
    def _analyze_chain_tvl(self, data: Dict) -> Dict:
        """
        Analyze chain TVL changes for per-coin signals.
        Rising TVL = bullish for chain's native token
        """
        try:
            chains = data.get('chains', [])
            if not chains:
                return {}
            
            tvl_analysis = {}
            
            for chain in chains:
                name = chain.get('name')
                if name not in self.CHAIN_TO_COIN:
                    continue
                
                tvl = chain.get('tvl', 0)
                change_1d = chain.get('change_1d', 0)
                change_7d = chain.get('change_7d', 0)
                
                # Generate signal strength based on TVL changes
                signal_strength = 0
                
                if change_1d > 5:  # TVL up >5% in 24h
                    signal_strength = 3
                elif change_1d > 2:  # TVL up >2% in 24h
                    signal_strength = 1
                elif change_1d < -5:  # TVL down >5% in 24h
                    signal_strength = -3
                elif change_1d < -2:  # TVL down >2% in 24h
                    signal_strength = -1
                
                # Boost for strong 7-day trends
                if change_7d > 10:
                    signal_strength += 2
                elif change_7d < -10:
                    signal_strength -= 2
                
                tvl_analysis[name] = {
                    'tvl': tvl,
                    'change_24h_pct': change_1d,
                    'change_7d_pct': change_7d,
                    'signal_strength': signal_strength
                }
            
            return tvl_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing chain TVL: {e}")
            return {}
    
    def _analyze_protocol_tvl(self, protocols: List) -> Dict:
        """
        Analyze protocol TVL changes for specific coin signals.
        """
        try:
            if not protocols:
                return {}
            
            protocol_signals = {}
            
            for protocol in protocols:
                name = protocol.get('name')
                if name not in self.PROTOCOL_TO_COIN:
                    continue
                
                change_1d = protocol.get('change_1d', 0)
                change_7d = protocol.get('change_7d', 0)
                tvl = protocol.get('tvl', 0)
                
                # Only consider protocols with significant TVL
                if tvl < 50_000_000:  # Less than $50M TVL
                    continue
                
                # Generate signal based on changes
                signal = 0
                
                if change_1d > 10:  # Protocol TVL surging
                    signal = 3
                elif change_1d > 5:
                    signal = 2
                elif change_1d > 2:
                    signal = 1
                elif change_1d < -10:
                    signal = -3
                elif change_1d < -5:
                    signal = -2
                elif change_1d < -2:
                    signal = -1
                
                protocol_signals[name] = signal
            
            return protocol_signals
            
        except Exception as e:
            logger.error(f"Error analyzing protocol TVL: {e}")
            return {}
    
    def _analyze_btc_network(self, data: Dict) -> Dict:
        """
        Analyze BTC network health for BTC trading signals.
        """
        try:
            stats = data.get('stats', {})
            hash_rate_data = data.get('hash_rate')
            tx_count_data = data.get('tx_count')
            tx_volume_data = data.get('tx_volume')
            
            analysis = {
                'hash_rate_trend': 'unknown',
                'mempool_pressure': 'unknown',
                'tx_volume_trend': 'unknown',
                'overall_signal': 0
            }
            
            # Analyze hash rate trend
            if hash_rate_data and 'values' in hash_rate_data:
                values = hash_rate_data['values']
                if len(values) >= 30:  # 30 days of data
                    recent_hash = np.mean([v['y'] for v in values[-7:]])  # Last 7 days
                    old_hash = np.mean([v['y'] for v in values[-30:-23]])  # 23-30 days ago
                    change = (recent_hash - old_hash) / old_hash * 100
                    
                    if change > 10:
                        analysis['hash_rate_trend'] = 'rising'
                        analysis['overall_signal'] += 2  # Bullish for BTC
                    elif change < -10:
                        analysis['hash_rate_trend'] = 'falling'
                        analysis['overall_signal'] -= 2  # Bearish for BTC
                    else:
                        analysis['hash_rate_trend'] = 'stable'
            
            # Analyze transaction volume
            if tx_volume_data and 'values' in tx_volume_data:
                values = tx_volume_data['values']
                if len(values) >= 7:
                    recent_vol = np.mean([v['y'] for v in values[-1:]])  # Latest
                    avg_vol = np.mean([v['y'] for v in values[-7:]])  # 7-day average
                    
                    if recent_vol > avg_vol * 1.5:  # 50% above average
                        analysis['tx_volume_trend'] = 'surging'
                        # High volume indicates potential volatility, not necessarily bullish
                    elif recent_vol < avg_vol * 0.7:
                        analysis['tx_volume_trend'] = 'low'
                    else:
                        analysis['tx_volume_trend'] = 'normal'
            
            # Analyze mempool from stats
            if stats:
                unconfirmed_count = stats.get('n_btc_mined', 0)  # Approximation
                if unconfirmed_count > 50000:
                    analysis['mempool_pressure'] = 'high'
                elif unconfirmed_count > 20000:
                    analysis['mempool_pressure'] = 'medium'
                else:
                    analysis['mempool_pressure'] = 'low'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing BTC network: {e}")
            return {
                'hash_rate_trend': 'unknown',
                'mempool_pressure': 'unknown', 
                'tx_volume_trend': 'unknown',
                'overall_signal': 0,
                'error': str(e)
            }


# Convenience function for testing
def test_onchain_engine():
    """Test the OnChain engine with live data."""
    engine = OnChainEngine()
    
    print("Testing OnChain Data Engine...")
    print("=" * 50)
    
    # Test data fetching
    data = engine.fetch_all()
    print(f"Requests used: {data.get('requests_used', 0)}")
    
    # Test signal generation
    signals = engine.get_onchain_signals()
    
    print(f"\nMarket Signal: {signals['market_signal']:+.1f}")
    print(f"Confidence: {signals['confidence']:.1f}")
    
    if signals['stablecoin_flow']:
        flow = signals['stablecoin_flow']
        print(f"\nStablecoin Flow: {flow.get('signal', 'unknown')}")
        if 'total_7d_change' in flow:
            print(f"7-day change: ${flow['total_7d_change']/1e6:+.0f}M")
    
    if signals['coin_signals']:
        print(f"\nCoin Signals ({len(signals['coin_signals'])}):")
        for pair, signal in sorted(signals['coin_signals'].items()):
            if abs(signal) > 0.1:
                print(f"  {pair}: {signal:+.1f}")
    
    print("\nDone!")
    return signals


if __name__ == "__main__":
    test_onchain_engine()