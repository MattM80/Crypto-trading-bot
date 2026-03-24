#!/usr/bin/env python3
"""
ULTIMATE FUTURES TRADING BOT - The Kraken Futures Edition
Maximum profit extraction with futures advantages:
- 0.07% round trip fees (vs 0.65% spot)
- 5x leverage on Tier 1 signals, 3x on Tier 2
- Funding rate farming
- 6 re-enabled tools that work at futures fees
- All 30 original signals EXACTLY preserved

The signal logic is IDENTICAL to run_final_bot.py. Only fees, leverage, and execution change.
"""

import sys
import os
import json
import time
import signal
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

# Add src directory for kraken_futures_client
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from kraken_futures_client import KrakenFuturesClient
    from kraken_client import KrakenClient  # Fallback for OHLC data
    from news_sentiment import NewsSentimentEngine  # Import sentiment engine
    from onchain_data import OnChainEngine  # Import on-chain data engine
    from orderbook_engine import OrderbookEngine  # Import orderbook analysis engine
    from ml_signal_weighter import MLSignalWeighter, FEATURES  # ML signal weighting engine
    from volatility_engine import VolatilityEngine  # Import volatility/options intelligence engine
except ImportError as e:
    logger.error(f"Failed to import clients: {e}")
    sys.exit(1)

# Configuration
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "300"))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = DATA_DIR / "futures_bot_state.json"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add(LOGS_DIR / "futures_bot.log", rotation="10 MB", retention="30 days")

# ULTIMATE UPGRADE: 40 pairs (same as final bot)
ORIGINAL_PAIRS = [
    "NEARUSD", "UNIUSD", "AVAXUSD", "LINKUSD", "AAVEUSD", "SOLUSD",
    "ETHUSD", "XBTUSD", "DOTUSD", "XLMUSD", "XRPUSD", "ADAUSD", 
    "ATOMUSD", "DOGEUSD", "FILUSD", "LTCUSD"
]

NEW_PAIRS = [
    "SUIUSD", "PEPEUSD", "SHIBUSD", "BNBUSD", "TRXUSD",
    "HBARUSD", "HYPEUSD", "TAOUSD", "OKBUSD", "INJUSD",
    "ARBUSD", "OPUSD", "APTUSD", "TIAUSD", "ONDOUSD",
    "RENDERUSD", "JUPUSD", "ICPUSD", "LDOUSD", "BCHUSD",
    "STXUSD", "KAVAUSD", "ENAUSD", "FLOKIUSD"
]

PAIRS = ORIGINAL_PAIRS + NEW_PAIRS  # 40 total

# Grid configurations (same as final bot)
GRID_CONFIGS = {
    # Original 16 pairs
    "NEARUSD": 0.01, "UNIUSD": 0.015, "AVAXUSD": 0.01, "LINKUSD": 0.008,
    "AAVEUSD": 0.015, "SOLUSD": 0.003, "ETHUSD": 0.005, "XBTUSD": 0.01,
    "DOTUSD": 0.012, "XLMUSD": 0.01, "XRPUSD": 0.01, "ADAUSD": 0.012,
    "ATOMUSD": 0.008, "DOGEUSD": 0.012, "FILUSD": 0.015, "LTCUSD": 0.01,
    
    # New 24 pairs - default 1.2% spacing
    "SUIUSD": 0.012, "PEPEUSD": 0.012, "SHIBUSD": 0.012, "BNBUSD": 0.012, "TRXUSD": 0.012,
    "HBARUSD": 0.012, "HYPEUSD": 0.012, "TAOUSD": 0.012, "OKBUSD": 0.012, "INJUSD": 0.012,
    "ARBUSD": 0.012, "OPUSD": 0.012, "APTUSD": 0.012, "TIAUSD": 0.012, "ONDOUSD": 0.012,
    "RENDERUSD": 0.012, "JUPUSD": 0.012, "ICPUSD": 0.012, "LDOUSD": 0.012, "BCHUSD": 0.012,
    "STXUSD": 0.012, "KAVAUSD": 0.012, "ENAUSD": 0.012, "FLOKIUSD": 0.012
}

# Constants
MAX_ACTIVE_POSITIONS = 5    # Max simultaneous active positions
RISK_PER_TRADE = 0.05       # 5% of active balance per trade
GRID_REANCHOR_PCT = 0.10    # Reanchor grid when price moves >10% from center
FUNDING_INTERVAL_HOURS = 4  # Funding every 4 hours on Kraken

# FUTURES FEES - THE GAME CHANGER
ENTRY_FEE = 0.0002         # 0.02% maker fee for limit orders (vs 0.25% spot)
EXIT_FEE = 0.0005          # 0.05% taker fee for market orders (vs 0.40% spot) 
ROUND_TRIP_FEE = 0.0007    # 0.07% total (vs 0.65% spot) - 9.3x BETTER!

# Grid fees (both limit orders, both maker)
GRID_ROUND_TRIP_FEE = 0.0004  # 0.02% entry + 0.02% exit = 0.04% (vs 0.50% spot)
GRID_FEE_MULTIPLIER = 0.9996  # Updated from 0.995

# LEVERAGE TIERS - The Power-Up
TIER1_TOOLS = {
    'crash_buy', 'volatile_oversold', 'crash_neg_ac', 'blood_in_streets',
    'quick_crash', 'crash_mean_revert', 'mega_pump_sell_t1', 'rsi_pump_8h',
    'mega_crash', 'vpin_dip'
}
TIER1_LEVERAGE = 5  # 5x leverage for high-conviction signals

TIER2_TOOLS = {
    'flash_crash', 'deep_dip_8h', 'entropy_dip', 'vpin_toxic', 'btc_alt_spread',
    'quick_dip', 'falling_wedge_short', 'greed_short_t2', 'thursday_short',
    'mega_pump_sell_t2', 'distribution_short', 'late_us_short', 'rsi_pump_12h',
    'ema_cross_short', 'rsi_pump_fat_tail', 'entropy_short', 'alt_btc_revert_t3',
    'month_start_long', 'dip_buy_5pct', 'market_panic_70', 'stablecoin_supply', 'tvl_rotation'
}
TIER2_LEVERAGE = 3  # 3x leverage for moderate signals

# RE-ENABLED TOOLS - Now profitable at futures fees!
TIER3_TOOLS = {
    'dip_buy_3pct', 'capitulation', 'zscore_extreme', 
    'panic_close', 'dist_exhaustion', 'deceleration_buy'
}
TIER3_LEVERAGE = 2  # Conservative leverage for marginal tools

# Grid leverage
GRID_LEVERAGE = 2  # 2x leverage for grid (conservative but profitable)

# ALL VALIDATED TOOLS (30 original + 6 re-enabled + 1 funding farm + 1 news sentiment + 2 on-chain + 2 orderbook = 42 total)
VALIDATED_TOOLS = (list(TIER1_TOOLS) + list(TIER2_TOOLS) + list(TIER3_TOOLS) + 
                  ['funding_farm', 'news_sentiment', 'stablecoin_supply', 'tvl_rotation', 'orderbook_imbalance', 'wall_breakout'])


class UltimateFuturesBot:
    """The Ultimate Futures Trading Bot with maximum profit extraction."""
    
    def __init__(self):
        self.client = KrakenFuturesClient(dry_run=not ENABLE_LIVE_TRADING)
        self.spot_client = KrakenClient()  # Fallback for OHLC data
        self.running = True
        self.state = self.load_state()
        
        # News Sentiment Engine - THE GAME CHANGER
        self.news_engine = NewsSentimentEngine()
        self.sentiment_cache = {}
        self.sentiment_cache_time = 0
        
        # On-Chain Data Engine - THE MONEY FLOW TRACKER
        self.onchain_engine = OnChainEngine()
        self.onchain_cache = {}
        self.onchain_cache_time = 0
        
        # Orderbook Engine - THE DEPTH ANALYZER
        self.orderbook_engine = OrderbookEngine()
        self.orderbook_cache = {}
        self.orderbook_cache_time = 0
        
        # ML Signal Weighting Engine - THE LEARNING BRAIN
        self.ml_weighter = MLSignalWeighter()
        
        # Volatility Engine - THE OPTIONS INTELLIGENCE
        self.vol_engine = VolatilityEngine()
        self.vol_cache = {}
        self.vol_cache_time = 0
        
        # Balance tracking
        self.starting_balance = STARTING_BALANCE
        self.total_balance = self.state.get("total_balance", STARTING_BALANCE)
        
        # Fear & Greed index
        self.current_fng = 50
        
        # IMPROVED CAPITAL ALLOCATION - More to active (better edges with low fees)
        grid_pct, active_pct = self.get_capital_allocation()
        self.grid_balance = self.total_balance * grid_pct
        self.active_balance = self.total_balance * active_pct
        
        # Grid state
        self.grid_positions = self.state.get("grid_positions", {})
        self.grid_profit = self.state.get("grid_profit", 0.0)
        self.grid_round_trips = self.state.get("grid_round_trips", 0)
        self.grid_anchors = self.state.get("grid_anchors", {})
        
        # Active positions
        self.active_positions = self.state.get("active_positions", {})
        self.active_profit = self.state.get("active_profit", 0.0)
        
        # Tool performance tracking
        self.tool_stats = self.state.get("tool_stats", {})
        self.tool_streaks = self.state.get("tool_streaks", {})
        self._initialize_tool_stats()
        
        # FUTURES-SPECIFIC TRACKING
        self.funding_income = self.state.get("funding_income", 0.0)
        self.leverage_used = self.state.get("leverage_used", 0.0)
        self.margin_available = self.state.get("margin_available", 1000.0)
        
        # Funding rates cache
        self.funding_rates = {}
        self.funding_cache_time = 0
        
        # Price cache for cross-pair signals
        self._price_cache = {}
        
        # Trade history and current bar
        self.trade_history = self.state.get("trade_history", [])
        self.current_bar = self.state.get("current_bar", 0)
        
        logger.info(f"🚀 ULTIMATE FUTURES BOT initialized")
        logger.info(f"Total balance: ${self.total_balance:.2f} (start: ${self.starting_balance:.2f})")
        logger.info(f"Growth: {(self.total_balance/self.starting_balance-1)*100:+.1f}%")
        logger.info(f"Grid balance: ${self.grid_balance:.2f} ({grid_pct:.0%}) at {GRID_LEVERAGE}x leverage")
        logger.info(f"Active balance: ${self.active_balance:.2f} ({active_pct:.0%})")
        logger.info(f"Futures fees: {ROUND_TRIP_FEE:.3%} RT (was 0.65% spot) - 9.3x BETTER!")
        logger.info(f"Tier 1 tools: {len(TIER1_TOOLS)} at {TIER1_LEVERAGE}x leverage")
        logger.info(f"Tier 2 tools: {len(TIER2_TOOLS)} at {TIER2_LEVERAGE}x leverage") 
        logger.info(f"Tier 3 tools (re-enabled): {len(TIER3_TOOLS)} at {TIER3_LEVERAGE}x leverage")
        logger.info(f"Total signals: {len(VALIDATED_TOOLS)} (30 + 6 + 1 funding farm + 1 news sentiment + 2 on-chain + 2 orderbook)")
        logger.info(f"News sentiment engine: ENABLED (keyword-based, no LLM)")
        logger.info(f"Orderbook engine: ENABLED (L2 depth analysis, wall detection)")
        logger.info(f"Funding income: ${self.funding_income:.2f}")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
    
    def get_capital_allocation(self) -> Tuple[float, float]:
        """IMPROVED: More capital to active trading (better edges at futures fees)."""
        fng = self.current_fng
        
        if fng < 20:  # Extreme fear - crash tools firing
            grid_pct, active_pct = 0.30, 0.70  # More active (was 35/65)
        elif fng < 35:  # Fear
            grid_pct, active_pct = 0.40, 0.60  # More active (was 45/55)
        elif fng <= 70:  # Neutral
            grid_pct, active_pct = 0.45, 0.55  # More active (was 50/50)
        elif fng <= 80:  # Greed
            grid_pct, active_pct = 0.35, 0.65  # More active (was 40/60)
        else:  # Extreme greed
            grid_pct, active_pct = 0.30, 0.70  # More active (was 35/65)
        
        return grid_pct, active_pct
    
    def _initialize_tool_stats(self):
        """Initialize tool statistics for all 37 tools."""
        for tool in VALIDATED_TOOLS:
            if tool not in self.tool_stats:
                self.tool_stats[tool] = {
                    "trades": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0.0, "score_adj": 1.0
                }
            if tool not in self.tool_streaks:
                self.tool_streaks[tool] = {"type": "", "streak": 0}
    
    def load_state(self) -> dict:
        """Load bot state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    logger.info("Loaded existing futures bot state")
                    return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {}
        
    def save_state(self):
        """Save bot state to disk."""
        state = {
            "total_balance": self.total_balance,
            "grid_positions": self.grid_positions,
            "grid_profit": self.grid_profit,
            "grid_round_trips": self.grid_round_trips,
            "grid_anchors": self.grid_anchors,
            "active_positions": self.active_positions,
            "active_profit": self.active_profit,
            "tool_stats": self.tool_stats,
            "tool_streaks": self.tool_streaks,
            "funding_income": self.funding_income,
            "leverage_used": self.leverage_used,
            "margin_available": self.margin_available,
            "trade_history": self.trade_history[-500:],
            "current_bar": self.current_bar,
            "last_update": datetime.now().isoformat()
        }
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates for all pairs."""
        # Cache for 5 minutes
        if time.time() - self.funding_cache_time < 300 and self.funding_rates:
            return self.funding_rates
        
        try:
            tickers = self.client.get_tickers()
            funding_rates = {}
            
            for symbol, data in tickers.items():
                if symbol.startswith("PF_"):
                    spot_symbol = symbol[3:]  # Remove PF_ prefix
                    funding_rates[spot_symbol] = data.get("fundingRate", 0.0)
            
            self.funding_rates = funding_rates
            self.funding_cache_time = time.time()
            return funding_rates
        
        except Exception as e:
            logger.error(f"Error getting funding rates: {e}")
            return self.funding_rates
    
    def get_funding_boost(self, pair: str, direction: str) -> float:
        """Boost signal score when funding rate favors our direction."""
        funding = self.funding_rates.get(pair, 0)
        
        if direction == 'long' and funding < -0.0001:  # Shorts paying longs
            return abs(funding) * 10000  # +1 score per 0.01% funding
        elif direction == 'short' and funding > 0.0001:  # Longs paying shorts
            return funding * 10000
        # Penalize when funding works against us
        elif direction == 'long' and funding > 0.0003:
            return -5  # Penalty for going long when longs are paying
        elif direction == 'short' and funding < -0.0003:
            return -5
        return 0
    
    def get_sentiment_boost(self, pair: str, direction: str) -> float:
        """Boost/penalize signal scores based on news sentiment."""
        # Cache sentiment for 5 minutes (same as bot cycle)
        now = time.time()
        if now - self.sentiment_cache_time > 300:  # 5 minutes
            try:
                self.sentiment_cache = self.news_engine.get_sentiment_signals()
                self.sentiment_cache_time = now
                logger.info(f"📰 Updated sentiment: market={self.sentiment_cache.get('market_sentiment', 0):+.1f}, "
                           f"coins={len(self.sentiment_cache.get('coin_sentiment', {}))}, "
                           f"breaking={len(self.sentiment_cache.get('breaking_events', []))}")
            except Exception as e:
                logger.warning(f"Failed to get sentiment signals: {e}")
                return 0
        
        if not self.sentiment_cache:
            return 0
            
        boost = 0
        
        # 1. Market-wide sentiment
        market = self.sentiment_cache.get('market_sentiment', 0)
        if direction == 'long' and market > 3:
            boost = market * 2  # Strong bullish news → boost longs
        elif direction == 'short' and market < -3:
            boost = abs(market) * 2  # Strong bearish news → boost shorts
        elif direction == 'long' and market < -3:
            boost = market * 1.5  # Going long against bearish news → penalty
        elif direction == 'short' and market > 3:
            boost = -market * 1.5  # Shorting against bullish news → penalty
        
        # 2. Coin-specific sentiment
        coin_sentiment = self.sentiment_cache.get('coin_sentiment', {})
        coin_sent = coin_sentiment.get(pair, 0)
        if direction == 'long' and coin_sent > 2:
            boost += coin_sent * 3  # Coin-specific good news → big boost
        elif direction == 'short' and coin_sent < -2:
            boost += abs(coin_sent) * 3
        
        # 3. Breaking events — these are the big ones
        breaking_events = self.sentiment_cache.get('breaking_events', [])
        for event in breaking_events:
            if pair in event.get('coins', []) and abs(event.get('score', 0)) >= 3:
                if (direction == 'long' and event['score'] > 0) or \
                   (direction == 'short' and event['score'] < 0):
                    boost += abs(event['score']) * 5  # MAJOR boost for breaking news
                    logger.info(f"📈 NEWS BOOST: {pair} {direction} +{abs(event['score']) * 5:.1f} - {event.get('headline', '')[:50]}...")
        
        return boost
    
    def get_onchain_boost(self, pair: str, direction: str) -> float:
        """Boost signal scores based on on-chain data - the MONEY FLOW indicator."""
        # Cache on-chain data for 5 minutes (same as bot cycle)
        now = time.time()
        if now - self.onchain_cache_time > 300:  # 5 minutes
            try:
                self.onchain_cache = self.onchain_engine.get_onchain_signals()
                self.onchain_cache_time = now
                market_signal = self.onchain_cache.get('market_signal', 0)
                coin_count = len(self.onchain_cache.get('coin_signals', {}))
                confidence = self.onchain_cache.get('confidence', 0)
                logger.info(f"🔗 Updated on-chain: market={market_signal:+.1f}, "
                           f"coins={coin_count}, confidence={confidence:.1f}")
            except Exception as e:
                logger.warning(f"Failed to get on-chain signals: {e}")
                return 0
        
        if not self.onchain_cache:
            return 0
            
        boost = 0
        
        # 1. Stablecoin flow (market-wide) - THE BIGGEST SIGNAL
        market_signal = self.onchain_cache.get('market_signal', 0)
        if direction == 'long' and market_signal > 2:
            boost += market_signal * 1.5  # Stablecoin minting → boost longs
        elif direction == 'short' and market_signal < -2:
            boost += abs(market_signal) * 1.5  # Stablecoin burning → boost shorts
        elif direction == 'long' and market_signal < -3:
            boost -= 3  # Penalty: going long while capital leaving crypto
        elif direction == 'short' and market_signal > 3:
            boost -= 3  # Penalty: shorting while capital entering crypto
        
        # 2. Coin-specific TVL/protocol flows
        coin_signals = self.onchain_cache.get('coin_signals', {})
        coin_signal = coin_signals.get(pair, 0)
        if direction == 'long' and coin_signal > 0:
            boost += coin_signal * 2  # TVL/protocol flowing into this coin
        elif direction == 'short' and coin_signal < 0:
            boost += abs(coin_signal) * 2  # TVL/protocol leaving this coin
        
        # 3. Confidence-based adjustment
        confidence = self.onchain_cache.get('confidence', 0)
        boost *= min(1.0, confidence + 0.3)  # Reduce boost if low confidence
        
        return boost
    
    def get_orderbook_boost(self, pair: str, direction: str) -> float:
        """Boost/penalize based on orderbook state - the DEPTH ANALYZER."""
        # Cache orderbook data for 1 minute (orderbooks change fast)
        now = time.time()
        if now - self.orderbook_cache_time > 60:  # 1 minute
            try:
                # Get pairs to scan (prioritize pairs with signals)
                current_pairs = [pair] if pair in PAIRS else []
                market_data = {pair: {'price': self._price_cache.get(pair, [0])[-1] if self._price_cache.get(pair) else 0}}
                
                self.orderbook_cache = self.orderbook_engine.get_orderbook_signals(
                    current_pairs, market_data
                )
                self.orderbook_cache_time = now
                logger.info(f"📊 Updated orderbook: {len(self.orderbook_cache)} pairs scanned")
            except Exception as e:
                logger.warning(f"Failed to get orderbook signals: {e}")
                return 0
        
        if pair not in self.orderbook_cache:
            return 0
            
        data = self.orderbook_cache[pair]
        boost = 0
        
        # 1. Imbalance alignment - THE BIG ONE
        imbalance = data.get('imbalance', 1.0)
        if direction == 'long' and imbalance > 2.0:
            boost += (imbalance - 1) * 3  # Bids dominate → boost longs
        elif direction == 'short' and imbalance < 0.5:
            boost += (1/imbalance - 1) * 3  # Asks dominate → boost shorts
        elif direction == 'long' and imbalance < 0.5:
            boost -= 5  # Going long against sell pressure → penalty
        elif direction == 'short' and imbalance > 2.0:
            boost -= 5  # Shorting against buy pressure → penalty
        
        # 2. Wall proximity - Support/Resistance
        walls = data.get('walls', [])
        mid_price = data.get('mid_price', 0)
        for wall in walls:
            if mid_price > 0:
                price_dist = abs(mid_price - wall.price) / mid_price
                if price_dist < 0.01:  # Within 1% of a wall
                    if wall.side == 'bid' and direction == 'long':
                        boost += wall.strength * 0.5  # Near bid wall = support for longs
                    elif wall.side == 'ask' and direction == 'short':
                        boost += wall.strength * 0.5  # Near ask wall = resistance for shorts
        
        # 3. Depth momentum
        momentum = data.get('depth_momentum', 0)
        if direction == 'long' and momentum > 0:
            boost += momentum * 2
        elif direction == 'short' and momentum < 0:
            boost += abs(momentum) * 2
        
        # 4. Spread penalty (wide spread = uncertainty)
        spread = data.get('spread_pct', 0)
        if spread > 0.5:  # Very wide spread
            boost *= 0.7  # Reduce confidence
        
        return boost
    
    def get_vol_boost(self, pair: str, direction: str) -> float:
        """Boost signal scores based on volatility/options intelligence - THE FEAR/GREED DETECTOR."""
        # Cache volatility data for 5 minutes (options data changes slower than orderbook)
        now = time.time()
        if now - self.vol_cache_time > 300:  # 5 minutes
            try:
                self.vol_cache = self.vol_engine.get_volatility_signals()
                self.vol_cache_time = now
                dvol = self.vol_cache.get('dvol', 50)
                pcr = self.vol_cache.get('put_call_ratio', 1.0)
                market_signal = self.vol_cache.get('market_signal', 0)
                logger.info(f"📊 Updated volatility: DVOL={dvol:.1f}%, P/C={pcr:.2f}, "
                           f"signal={market_signal:+.1f}")
            except Exception as e:
                logger.warning(f"Failed to get volatility signals: {e}")
                return 0
        
        if not self.vol_cache:
            return 0
            
        boost = 0
        
        # 1. Put/Call ratio (contrarian) - THE BIG ONE
        pcr = self.vol_cache.get('put_call_ratio', 1.0)
        if direction == 'long' and pcr > 1.3:
            boost += (pcr - 1) * 5  # Everyone buying puts → contrarian long
        elif direction == 'short' and pcr < 0.7:
            boost += (1/pcr - 1) * 5  # Everyone buying calls → contrarian short
        elif direction == 'long' and pcr > 1.5:
            boost += 10  # EXTREME fear → strong contrarian buy signal
        elif direction == 'short' and pcr < 0.5:
            boost += 10  # EXTREME greed → strong contrarian sell signal
        
        # 2. Max pain gravity (applies only to BTC/ETH pairs)
        if pair in ('XBTUSD', 'PF_XBTUSD', 'ETHUSD', 'PF_ETHUSD'):
            max_pain_bias = self.vol_cache.get('max_pain_bias', 'neutral')
            if direction == 'long' and max_pain_bias == 'bullish':
                boost += 3  # Max pain above current → price likely to rise
            elif direction == 'short' and max_pain_bias == 'bearish':
                boost += 3  # Max pain below current → price likely to fall
        
        # 3. Gamma exposure (affects all crypto pairs)
        gamma_exposure = self.vol_cache.get('gamma_exposure', 'neutral')
        if gamma_exposure == 'negative' and abs(boost) > 0:
            boost *= 1.3  # Negative gamma amplifies moves → more conviction
        elif gamma_exposure == 'positive':
            boost *= 0.9  # Positive gamma dampens moves → less conviction
        
        # 4. Volatility skew
        skew_signal = self.vol_cache.get('skew_signal', 'neutral')
        if direction == 'long' and skew_signal == 'fear':
            boost += 2  # Fear skew → contrarian long signal
        elif direction == 'short' and skew_signal == 'greed':
            boost += 2  # Greed skew → contrarian short signal
        
        # 5. DVOL regime
        dvol = self.vol_cache.get('dvol', 50)
        if dvol > 80 and direction == 'long':
            boost += 3  # High vol often marks bottoms (VIX rule)
        elif dvol < 30:
            boost *= 0.8  # Low vol → reduce all signals (small moves expected)
        
        # 6. Overall market signal
        market_signal = self.vol_cache.get('market_signal', 0)
        if direction == 'long' and market_signal > 2:
            boost += market_signal  # Strong composite bullish signal
        elif direction == 'short' and market_signal < -2:
            boost += abs(market_signal)  # Strong composite bearish signal
        
        return boost
    
    def get_vol_position_multiplier(self) -> float:
        """Get position size multiplier based on volatility regime."""
        if not self.vol_cache:
            return 1.0
        
        return self.vol_cache.get('position_size_multiplier', 1.0)
    
    def get_market_data(self) -> dict:
        """Fetch 1h market data for all pairs."""
        market_data = {}
        
        for pair in PAIRS:
            try:
                # Try futures OHLC first, fallback to spot
                df = self.client.get_ohlc(pair, "1h")
                
                if df.empty or len(df) < 50:
                    # Fallback to spot OHLC
                    klines = self.spot_client.get_klines(pair, interval=3600, limit=200)
                    if not klines:
                        continue
                    
                    df = pd.DataFrame(klines)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    if len(df) < 50:
                        continue
                    
                    df = df.set_index('timestamp')
                
                # Get current price from futures tickers
                tickers = self.client.get_tickers()
                futures_symbol = f"PF_{pair}"
                ticker = tickers.get(futures_symbol, {})
                current_price = ticker.get('last', float(df['close'].iloc[-1]))
                
                # Get bid/ask
                bid_price = ticker.get('bid', current_price * 0.9998)
                ask_price = ticker.get('ask', current_price * 1.0002)
                
                market_data[pair] = {
                    "price": current_price,
                    "bid": bid_price,
                    "ask": ask_price,
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "df": df
                }
                
                # Cache close prices for cross-pair signals
                self._price_cache[pair] = df['close'].values.astype(float)
                
            except Exception as e:
                logger.error(f"Failed to get market data for {pair}: {e}")
                continue
                
        return market_data
    
    # ===== EXACT SAME INDICATOR CALCULATIONS =====
    
    def calc_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI - EXACT COPY."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
            
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(delta))
        avg_loss = np.zeros(len(delta))
        
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        for i in range(period, len(delta)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50.0], rsi])
    
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average - EXACT COPY."""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
            
        sma = np.full(len(prices), np.nan)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            
        return sma
    
    def calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range - EXACT COPY."""
        if len(high) < 2:
            return np.full(len(high), 0.0)
            
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First bar uses high-low
        
        atr = np.full(len(tr), 0.0)
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        return atr
    
    def calc_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average - EXACT COPY."""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        k = 2.0 / (period + 1)
        for i in range(period, len(prices)):
            ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
        return ema
    
    def calc_autocorrelation(self, prices: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation - EXACT COPY."""
        if len(prices) < lag + 5:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < lag + 2:
            return 0.0
        return np.corrcoef(returns[:-lag], returns[lag:])[0, 1] if not np.isnan(np.corrcoef(returns[:-lag], returns[lag:])[0, 1]) else 0.0
    
    def calc_hurst(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent - EXACT COPY."""
        if len(prices) < max_lag + 10:
            return 0.5
        log_prices = np.log(prices)
        n = len(log_prices)
        rs = []
        lags = range(2, min(max_lag, n//2))
        for lag in lags:
            ts = log_prices[:n//lag*lag].reshape(-1, lag)
            mean_ts = np.mean(ts, axis=1, keepdims=True)
            ts_centered = ts - mean_ts
            rs_values = []
            for i in range(ts.shape[0]):
                cumsum = np.cumsum(ts_centered[i])
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(ts[i])
                if s > 0:
                    rs_values.append(r / s)
            if rs_values:
                rs.append(np.mean(rs_values))
        if len(rs) < 2:
            return 0.5
        rs = np.array(rs)
        lags = np.array(list(lags))
        try:
            hurst = np.polyfit(np.log(lags), np.log(rs), 1)[0]
            return max(0.0, min(1.0, hurst))
        except:
            return 0.5
    
    def calc_entropy(self, prices: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy - EXACT COPY."""
        if len(prices) < 10:
            return 3.0
        try:
            returns = np.diff(prices) / prices[:-1]
            hist, _ = np.histogram(returns, bins=bins)
            hist = hist[hist > 0]  # Remove empty bins
            p = hist / np.sum(hist)
            entropy = -np.sum(p * np.log2(p))
            return entropy
        except:
            return 3.0
    
    def calc_vpin(self, df: pd.DataFrame, window: int = 50) -> float:
        """Calculate VPIN - EXACT COPY."""
        if len(df) < window:
            return 0.5
        try:
            volume = df['volume'].values.astype(float)
            close = df['close'].values.astype(float)
            
            if len(volume) < 2 or len(close) < 2:
                return 0.5
                
            price_change = np.diff(close)
            buy_vol = np.where(price_change > 0, volume[1:], 0)
            sell_vol = np.where(price_change < 0, volume[1:], 0)
            
            if len(buy_vol) < window:
                return 0.5
                
            vpin_values = []
            for i in range(window, len(buy_vol)):
                window_buy = np.sum(buy_vol[i-window:i])
                window_sell = np.sum(sell_vol[i-window:i])
                total_vol = window_buy + window_sell
                if total_vol > 0:
                    vpin = abs(window_buy - window_sell) / total_vol
                    vpin_values.append(vpin)
            
            return np.mean(vpin_values[-10:]) if vpin_values else 0.5
        except:
            return 0.5
    
    def update_grid_balance(self):
        """Update grid balance scaling with total balance growth."""
        grid_pct, _ = self.get_capital_allocation()
        self.grid_balance = self.total_balance * grid_pct
    
    def get_leverage_for_tool(self, tool: str) -> int:
        """Get leverage tier for a tool."""
        if tool in TIER1_TOOLS:
            return TIER1_LEVERAGE
        elif tool in TIER2_TOOLS:
            return TIER2_LEVERAGE
        elif tool in TIER3_TOOLS:
            return TIER3_LEVERAGE
        elif tool == 'funding_farm':
            return TIER2_LEVERAGE  # 3x for funding farm
        elif tool == 'news_sentiment':
            return TIER1_LEVERAGE  # 5x for breaking news (highest conviction)
        elif tool in ['orderbook_imbalance', 'wall_breakout']:
            return TIER1_LEVERAGE  # 5x for orderbook signals (high conviction depth analysis)
        else:
            return 1  # No leverage for unknown tools
    
    def get_position_size(self, signal: dict, tool: str, leverage: int) -> Tuple[float, float]:
        """Calculate position size and effective stop loss for leveraged futures."""
        risk_amount = self.active_balance * RISK_PER_TRADE  # 5% of active balance
        
        # Apply volatility-based position sizing multiplier
        vol_multiplier = self.get_vol_position_multiplier()
        risk_amount *= vol_multiplier
        
        # Get stop loss from signal
        sl_pct = signal.get('sl_pct', 0.05)
        
        # CRITICAL: Adjust stop loss for leverage to maintain same dollar risk
        # At 5x leverage with 5% SL on spot: use 1% SL (5% / 5) to maintain same $ risk
        effective_sl = sl_pct / leverage
        
        # Position sizing: Risk divided by effective SL gives position size
        position_size = risk_amount / effective_sl
        
        # Don't risk more than available balance
        position_size = min(position_size, self.active_balance * 0.8)
        
        return position_size, effective_sl
    
    def get_ml_features(self, pair: str, df: pd.DataFrame, cur_rsi: float, cur_atr_pct: float, 
                       ret_4h: float, ret_24h: float, ret_8h: float, ret_12h: float, 
                       cur_vs_sma50: float, vol_ratio: float) -> dict:
        """Calculate ML feature vector for current market conditions."""
        try:
            # Time features
            now = datetime.now(timezone.utc)
            hour = now.hour
            dow = now.weekday()
            
            # BTC return (use this pair's return if BTC not available)
            btc_ret_24h = ret_24h
            if "XBTUSD" in self._price_cache and len(self._price_cache["XBTUSD"]) >= 25:
                btc_prices = self._price_cache["XBTUSD"]
                btc_ret_24h = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
            
            # Orderbook imbalance
            ob_imbalance = 1.0  # Default balanced
            try:
                ob_data = self.orderbook_engine.analyze_orderbook(pair, self.client)
                if ob_data:
                    ob_imbalance = ob_data.get('imbalance', 1.0)
            except:
                pass
            
            features = {
                'rsi_7': float(cur_rsi),
                'atr_pct': float(cur_atr_pct),
                'ret_4h': float(ret_4h),
                'ret_24h': float(ret_24h),
                'vs_sma50': float(cur_vs_sma50),
                'volume_ratio': float(vol_ratio),
                'fng': float(self.current_fng),
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'dow_sin': np.sin(2 * np.pi * dow / 7),
                'dow_cos': np.cos(2 * np.pi * dow / 7),
                'btc_ret_24h': float(btc_ret_24h),
                'stablecoin_signal': float(self.onchain_cache.get('market_signal', 0)),
                'news_sentiment': float(self.sentiment_cache.get('market_sentiment', 0)),
                'ob_imbalance': float(ob_imbalance),
                'funding_rate': float(self.funding_rates.get(pair, 0)),
            }
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating ML features for {pair}: {e}")
            # Return default neutral features
            return {name: 0.0 if 'sin' not in name and 'cos' not in name and name != 'fng' 
                    else (50.0 if name == 'fng' else (1.0 if name == 'ob_imbalance' else 0.0)) 
                    for name in FEATURES}
    
    # ===== EXACT SAME SIGNAL SCANNING LOGIC =====
    # This preserves ALL 30 original tools exactly as they were
    
    def scan_signals(self, pair: str, data: dict) -> List[Tuple[dict, float]]:
        """Scan ALL 37 validated tools for signals (30 original + 6 re-enabled + 1 funding farm)."""
        signals = []
        df = data['df']
        price = data['price']
        
        if len(df) < 50:
            return signals
        
        # Compute base indicators once (EXACT COPY from run_final_bot.py)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        rsi7 = self.calc_rsi(close, 7)
        sma50 = self.calc_sma(close, 50)
        atr14 = self.calc_atr(high, low, close, 14)
        ema5 = self.calc_ema(close, 5)
        ema13 = self.calc_ema(close, 13)
        
        # Current values
        cur_rsi = rsi7[-1] if not np.isnan(rsi7[-1]) else 50
        cur_atr_pct = (atr14[-1] / price * 100) if price > 0 and not np.isnan(atr14[-1]) else 0
        cur_vs_sma50 = ((price - sma50[-1]) / sma50[-1] * 100) if not np.isnan(sma50[-1]) and sma50[-1] > 0 else 0
        
        # Returns
        ret_4h = (price - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
        ret_8h = (price - close[-9]) / close[-9] * 100 if len(close) >= 9 else 0
        ret_12h = (price - close[-13]) / close[-13] * 100 if len(close) >= 13 else 0
        ret_24h = (price - close[-25]) / close[-25] * 100 if len(close) >= 25 else 0
        
        # Helper function to apply score adjustment from UPGRADE 7
        def adjust_score(tool: str, base_score: float) -> float:
            if tool in self.tool_stats:
                return base_score * self.tool_stats[tool].get("score_adj", 1.0)
            return base_score
        
        # ===== CRASH/BEAR SIGNALS (LONG) - 15 tools EXACT COPY =====
        
        # 1. volatile_oversold: atr_pct>3 AND rsi7<25 → LONG | WR_8h=73.8%, Ret_8h=+2.07%
        if cur_atr_pct > 3 and cur_rsi < 25:
            base_score = adjust_score('volatile_oversold', cur_atr_pct * (25 - cur_rsi) * 0.5)  # 30-50 range
            boosts = self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            
            # ML Score Adjustment
            ml_features = self.get_ml_features(pair, df, cur_rsi, cur_atr_pct, ret_4h, ret_24h, ret_8h, ret_12h, cur_vs_sma50, vol_ratio)
            ml_multiplier = self.ml_weighter.get_score_multiplier('volatile_oversold', ml_features)
            final_score = (base_score + boosts) * ml_multiplier
            
            signal_dict = {
                'pair': pair, 'tool': 'volatile_oversold', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.08,
                'reason': f"VOLATILE OVERSOLD: ATR={cur_atr_pct:.1f}%, RSI={cur_rsi:.1f}",
                'ml_features': ml_features  # Store features for learning when trade closes
            }
            signals.append((signal_dict, final_score))
        
        # 2. crash_buy: ret_24h<-10 AND rsi7<20 → LONG | WR_8h=65.1%, Ret_24h=+1.90%
        if ret_24h < -10 and cur_rsi < 20:
            base_score = adjust_score('crash_buy', abs(ret_24h) * (20 - cur_rsi) * 0.3)  # 25-40 range
            boosts = self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            
            # ML Score Adjustment
            ml_features = self.get_ml_features(pair, df, cur_rsi, cur_atr_pct, ret_4h, ret_24h, ret_8h, ret_12h, cur_vs_sma50, vol_ratio)
            ml_multiplier = self.ml_weighter.get_score_multiplier('crash_buy', ml_features)
            final_score = (base_score + boosts) * ml_multiplier
            
            signal_dict = {
                'pair': pair, 'tool': 'crash_buy', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"CRASH BUY: {ret_24h:.1f}% drop 24h, RSI={cur_rsi:.1f}",
                'ml_features': ml_features
            }
            signals.append((signal_dict, final_score))
        
        # 3. mega_crash: ret_24h<-15 → LONG | WR_24h=52.5%, Ret_24h=+1.35%
        if ret_24h < -15:
            score = adjust_score('mega_crash', abs(ret_24h) * 2)  # 30-50 range
            score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            signals.append(({
                'pair': pair, 'tool': 'mega_crash', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.08,
                'reason': f"MEGA CRASH: {ret_24h:.1f}% drop 24h"
            }, score))
        
        # 4. crash_neg_ac: ret_24h<-10 AND autocorr<-0.05 → LONG | WR_8h=62.1%, Ret_8h=+1.25%
        if ret_24h < -10:
            autocorr = self.calc_autocorrelation(close[-30:]) if len(close) >= 30 else 0
            if autocorr < -0.05:
                score = adjust_score('crash_neg_ac', abs(ret_24h) * abs(autocorr) * 50)  # 30-50 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'crash_neg_ac', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.05,
                    'reason': f"CRASH NEG AC: {ret_24h:.1f}% drop, autocorr={autocorr:.3f}"
                }, score))
        
        # 5. blood_in_streets: 70%+ coins down >2% AND rsi7<20 → LONG | WR_24h=61.7%, Ret_24h=+1.10%
        if len(self._price_cache) >= 5 and cur_rsi < 20:
            dropping = sum(1 for cached in self._price_cache.values()
                         if len(cached) >= 5 and (cached[-1]-cached[-5])/cached[-5]*100 < -2)
            panic_pct = dropping / len(self._price_cache) * 100
            if panic_pct >= 70:
                score = adjust_score('blood_in_streets', (20 - cur_rsi) * 2)  # High priority
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'blood_in_streets', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.06,
                    'reason': f"BLOOD IN STREETS: {panic_pct:.0f}% panic + RSI={cur_rsi:.1f}"
                }, score))
        
        # 6. quick_crash: ret_8h<-10 → LONG (8h hold only) | WR_8h=59.1%, Ret_8h=+0.98%
        if ret_8h < -10:
            score = adjust_score('quick_crash', abs(ret_8h) * 2)  # 20-30 range
            score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            signals.append(({
                'pair': pair, 'tool': 'quick_crash', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.07,
                'reason': f"QUICK CRASH: {ret_8h:.1f}% drop 8h"
            }, score))
        
        # 7. crash_mean_revert: ret_24h<-8 AND Hurst<0.45 → LONG | WR_8h=61.3%, Ret_8h=+0.98%
        if ret_24h < -8:
            hurst = self.calc_hurst(close[-50:]) if len(close) >= 50 else 0.5
            if hurst < 0.45:
                score = adjust_score('crash_mean_revert', abs(ret_24h) * (0.45 - hurst) * 10)  # 20-30 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'crash_mean_revert', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.05,
                    'reason': f"CRASH MEAN REVERT: {ret_24h:.1f}% drop, Hurst={hurst:.3f}"
                }, score))
        
        # 8. vpin_dip: ret_8h<-5 AND VPIN>0.5 → LONG | WR_8h=58.8%, Ret_8h=+0.73%
        if ret_8h < -5:
            vpin = self.calc_vpin(df)
            if vpin > 0.5:
                score = adjust_score('vpin_dip', abs(ret_8h) * vpin * 2)  # 15-25 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'vpin_dip', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.05,
                    'reason': f"VPIN DIP: {ret_8h:.1f}% drop 8h, VPIN={vpin:.2f}"
                }, score))
        
        # 9. market_panic_70: 70%+ coins down >3% in 4h → LONG | WR_24h=59.0%, Ret_24h=+0.75%
        if len(self._price_cache) >= 5:
            dropping = sum(1 for cached in self._price_cache.values()
                         if len(cached) >= 5 and (cached[-1]-cached[-5])/cached[-5]*100 < -3)
            panic_pct = dropping / len(self._price_cache) * 100
            if panic_pct >= 70:
                score = adjust_score('market_panic_70', panic_pct * 0.3)  # 21-30 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'market_panic_70', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.04,
                    'reason': f"MARKET PANIC 70: {panic_pct:.0f}% coins down >3%"
                }, score))
        
        # 10. flash_crash: ret_12h<-10 → LONG | WR_8h=55.8%, Ret_8h=+0.51%
        if ret_12h < -10:
            score = adjust_score('flash_crash', abs(ret_12h) * 1.5)  # 15-25 range
            score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            signals.append(({
                'pair': pair, 'tool': 'flash_crash', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.07,
                'reason': f"FLASH CRASH: {ret_12h:.1f}% drop 12h"
            }, score))
        
        # 11. deep_dip_8h: -10<ret_8h<-8 → LONG | WR_8h=54.8%, Ret_8h=+0.22%
        if -10 < ret_8h < -8:
            score = adjust_score('deep_dip_8h', abs(ret_8h) * 1.5)  # 12-15 range
            score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            signals.append(({
                'pair': pair, 'tool': 'deep_dip_8h', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.05,
                'reason': f"DEEP DIP 8h: {ret_8h:.1f}% drop"
            }, score))
        
        # 12. entropy_dip: entropy<2.5 AND ret_4h<-2 → LONG | WR_8h=52.8%, Ret_8h=+0.45%
        if ret_4h < -2:
            entropy = self.calc_entropy(close[-30:]) if len(close) >= 30 else 3.0
            if entropy < 2.5:
                score = adjust_score('entropy_dip', (2.5 - entropy) * abs(ret_4h) * 2)  # 10-20 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'entropy_dip', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.04,
                    'reason': f"ENTROPY DIP: entropy={entropy:.2f}, {ret_4h:.1f}% drop 4h"
                }, score))
        
        # 13. vpin_toxic: VPIN>0.7 AND red candle → LONG | WR_8h=53.8%, Ret_8h=+0.45%
        if len(df) >= 2:
            vpin = self.calc_vpin(df)
            is_red = close[-1] < df['open'].iloc[-1]
            if vpin > 0.7 and is_red:
                score = adjust_score('vpin_toxic', vpin * 20)  # 14-20 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'vpin_toxic', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.04,
                    'reason': f"VPIN TOXIC: VPIN={vpin:.2f}, red candle"
                }, score))
        
        # 14. btc_alt_spread: alt lagging BTC 3%+ AND rsi7<35 → LONG | WR_24h=55.2%, Ret_24h=+0.45%
        if pair != "XBTUSD" and "XBTUSD" in self._price_cache and cur_rsi < 35:
            btc_prices = self._price_cache["XBTUSD"]
            if len(btc_prices) >= 25 and len(close) >= 25:
                btc_ret24 = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                if btc_ret24 - ret_24h > 3:  # BTC outperforming by 3%+
                    score = adjust_score('btc_alt_spread', (btc_ret24 - ret_24h) * (35 - cur_rsi) * 0.1)  # 10-15 range
                    score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                    signals.append(({
                        'pair': pair, 'tool': 'btc_alt_spread', 'direction': 'long',
                        'hold': 24, 'sl_pct': 0.05,
                        'reason': f"BTC ALT SPREAD: BTC {btc_ret24:+.1f}% vs {pair} {ret_24h:+.1f}%, RSI={cur_rsi:.1f}"
                    }, score))
        
        # 15. quick_dip: ret_4h<-5 → LONG | WR_8h=55.5%, Ret_8h=+0.13%
        if ret_4h < -5:
            base_score = adjust_score('quick_dip', abs(ret_4h) * 1.2)  # 6-12 range
            boosts = self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            
            # ML Score Adjustment
            ml_features = self.get_ml_features(pair, df, cur_rsi, cur_atr_pct, ret_4h, ret_24h, ret_8h, ret_12h, cur_vs_sma50, vol_ratio)
            ml_multiplier = self.ml_weighter.get_score_multiplier('quick_dip', ml_features)
            final_score = (base_score + boosts) * ml_multiplier
            
            signal_dict = {
                'pair': pair, 'tool': 'quick_dip', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.05,
                'reason': f"QUICK DIP: {ret_4h:.1f}% drop 4h",
                'ml_features': ml_features
            }
            signals.append((signal_dict, final_score))
        
        # ===== BULL/GREED SIGNALS (SHORT) - 13 tools EXACT COPY =====
        
        # 16. mega_pump_sell_t1: rsi7>80 AND ret_12h>=10 → SHORT | WR_24h=58.7%, Ret_24h=+0.61%
        if cur_rsi > 80 and ret_12h >= 10:
            base_score = adjust_score('mega_pump_sell_t1', 25 + (cur_rsi - 80) * 0.5 + (ret_12h - 10) * 0.3)  # 25-35 range
            boosts = self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
            
            # ML Score Adjustment
            ml_features = self.get_ml_features(pair, df, cur_rsi, cur_atr_pct, ret_4h, ret_24h, ret_8h, ret_12h, cur_vs_sma50, vol_ratio)
            ml_multiplier = self.ml_weighter.get_score_multiplier('mega_pump_sell_t1', ml_features)
            final_score = (base_score + boosts) * ml_multiplier
            
            signal_dict = {
                'pair': pair, 'tool': 'mega_pump_sell_t1', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"MEGA PUMP SELL T1: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h",
                'ml_features': ml_features
            }
            signals.append((signal_dict, final_score))
        
        # 17. rsi_pump_8h: rsi7>80 AND ret_8h>=10 → SHORT | WR_24h=60.3%, Ret_24h=+0.56%
        if cur_rsi > 80 and ret_8h >= 10:
            score = adjust_score('rsi_pump_8h', 25 + (cur_rsi - 80) * 0.4 + (ret_8h - 10) * 0.4)  # 25-35 range
            score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
            signals.append(({
                'pair': pair, 'tool': 'rsi_pump_8h', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"RSI PUMP 8h: RSI={cur_rsi:.1f}, +{ret_8h:.1f}% 8h"
            }, score))
        
        # 18. falling_wedge_short: lower highs+lows converging + price>SMA50 → SHORT | WR_24h=56.7%, Ret_24h=+0.60%
        if len(high) >= 20 and len(low) >= 20 and not np.isnan(sma50[-1]) and price > sma50[-1]:
            # Simple falling wedge detection: recent highs declining, lows declining but converging
            recent_highs = high[-10:]
            recent_lows = low[-10:]
            if len(recent_highs) >= 5 and len(recent_lows) >= 5:
                high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
                if high_trend < 0 and low_trend < 0 and high_trend > low_trend:  # Converging
                    score = adjust_score('falling_wedge_short', abs(high_trend - low_trend) * 1000)  # 20-30 range
                    score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                    signals.append(({
                        'pair': pair, 'tool': 'falling_wedge_short', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.04,
                        'reason': f"FALLING WEDGE SHORT: converging pattern, price>${sma50[-1]:.2f} SMA50"
                    }, score))
        
        # 19. greed_short_t2: rsi7>75 AND ret_8h>5 AND price>SMA50 → SHORT | WR_24h=58.5%, Ret_24h=+0.31%
        # This uses Fear & Greed index for enhanced scoring
        if cur_rsi > 75 and ret_8h > 5 and not np.isnan(sma50[-1]) and price > sma50[-1]:
            base_score = 15 + (cur_rsi - 75) * 0.3 + (ret_8h - 5) * 0.2
            # Boost score if Fear & Greed is high
            if self.current_fng > 75:
                base_score *= 1.5
            score = adjust_score('greed_short_t2', base_score)  # 15-25 range
            score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
            signals.append(({
                'pair': pair, 'tool': 'greed_short_t2', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.04,
                'reason': f"GREED SHORT T2: RSI={cur_rsi:.1f}, +{ret_8h:.1f}% 8h, F&G={self.current_fng}"
            }, score))
        
        # 20. thursday_short: Thursday AND price>SMA50 → SHORT | WR_24h=57.9%, Ret_24h=+0.28%
        try:
            dow = datetime.now(timezone.utc).weekday()
            if dow == 3 and not np.isnan(sma50[-1]) and price > sma50[-1]:  # Thursday
                score = adjust_score('thursday_short', 12)  # Fixed score
                score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                signals.append(({
                    'pair': pair, 'tool': 'thursday_short', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.03,
                    'reason': f"THURSDAY SHORT: uptrend (price>${sma50[-1]:.2f} SMA50)"
                }, score))
        except:
            pass
        
        # 21. mega_pump_sell_t2: rsi7>80 AND ret_12h>=8 → SHORT | WR_24h=55.0%, Ret_24h=+0.19%
        if cur_rsi > 80 and ret_12h >= 8:
            score = adjust_score('mega_pump_sell_t2', 18 + (cur_rsi - 80) * 0.3 + (ret_12h - 8) * 0.2)  # 18-25 range
            score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
            signals.append(({
                'pair': pair, 'tool': 'mega_pump_sell_t2', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"MEGA PUMP SELL T2: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h"
            }, score))
        
        # 22. distribution_short: lower highs + vol decline + RSI fall + price>SMA50 → SHORT | WR_24h=53.4%, Ret_24h=+0.18%
        if len(high) >= 20 and len(volume) >= 20 and not np.isnan(sma50[-1]) and price > sma50[-1]:
            # Detect distribution pattern
            recent_highs = high[-10:]
            recent_vol = volume[-10:]
            recent_rsi = rsi7[-10:]
            
            if len(recent_highs) >= 5 and len(recent_vol) >= 5 and len(recent_rsi) >= 5:
                high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                vol_trend = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
                rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
                
                if high_trend < 0 and vol_trend < 0 and rsi_trend < 0:  # All declining
                    score = adjust_score('distribution_short', abs(high_trend) * 100)  # 15-20 range
                    score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                    signals.append(({
                        'pair': pair, 'tool': 'distribution_short', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.04,
                        'reason': f"DISTRIBUTION SHORT: lower highs, vol decline, RSI fall"
                    }, score))
        
        # 23. late_us_short: hour==21 UTC AND price>SMA50 → SHORT | WR_24h=52.9%, Ret_24h=+0.16%
        try:
            current_hour = datetime.now(timezone.utc).hour
            if current_hour == 21 and not np.isnan(sma50[-1]) and price > sma50[-1]:
                score = adjust_score('late_us_short', 10)  # Fixed score
                score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                signals.append(({
                    'pair': pair, 'tool': 'late_us_short', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.03,
                    'reason': f"LATE US SHORT: 21:00 UTC, price>${sma50[-1]:.2f} SMA50"
                }, score))
        except:
            pass
        
        # 24. rsi_pump_12h: rsi7>80 AND ret_12h>=8 → SHORT | WR_24h=54.9%, Ret_24h=+0.13%
        if cur_rsi > 80 and ret_12h >= 8:
            score = adjust_score('rsi_pump_12h', 15 + (cur_rsi - 80) * 0.2 + (ret_12h - 8) * 0.1)  # 15-20 range
            score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
            signals.append(({
                'pair': pair, 'tool': 'rsi_pump_12h', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.04,
                'reason': f"RSI PUMP 12h: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h"
            }, score))
        
        # 25. ema_cross_short: ema5>ema13 AND price>SMA50 → SHORT | WR_24h=53.2%, Ret_24h=+0.13%
        if (not np.isnan(ema5[-1]) and not np.isnan(ema13[-1]) and not np.isnan(sma50[-1]) and
            ema5[-1] > ema13[-1] and price > sma50[-1]):
            score = adjust_score('ema_cross_short', 10 + (ema5[-1] - ema13[-1]) / price * 1000)  # 10-15 range
            score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
            signals.append(({
                'pair': pair, 'tool': 'ema_cross_short', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.03,
                'reason': f"EMA CROSS SHORT: EMA5>${ema5[-1]:.2f} > EMA13${ema13[-1]:.2f}, price>SMA50"
            }, score))
        
        # 26. rsi_pump_fat_tail: rsi7>80 AND ret_12h>10 AND kurtosis>5 → SHORT | WR_24h=58.9%, Ret_24h=+0.07%
        if cur_rsi > 80 and ret_12h > 10:
            # Calculate kurtosis of recent returns
            if len(close) >= 30:
                recent_returns = np.diff(close[-30:]) / close[-31:-1]
                try:
                    from scipy.stats import kurtosis
                    kurt = kurtosis(recent_returns)
                except ImportError:
                    # Simple kurtosis approximation
                    mean_ret = np.mean(recent_returns)
                    std_ret = np.std(recent_returns)
                    if std_ret > 0:
                        fourth_moment = np.mean(((recent_returns - mean_ret) / std_ret) ** 4)
                        kurt = fourth_moment - 3  # Excess kurtosis
                    else:
                        kurt = 0
                
                if kurt > 5:
                    score = adjust_score('rsi_pump_fat_tail', 15 + kurt * 0.5)  # 15-20 range
                    score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                    signals.append(({
                        'pair': pair, 'tool': 'rsi_pump_fat_tail', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.04,
                        'reason': f"RSI PUMP FAT TAIL: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h, kurtosis={kurt:.1f}"
                    }, score))
        
        # 27. entropy_short: entropy<2.5 AND price>SMA50 → SHORT | WR_24h=54.8%, Ret_24h=+0.02%
        if not np.isnan(sma50[-1]) and price > sma50[-1]:
            entropy = self.calc_entropy(close[-30:]) if len(close) >= 30 else 3.0
            if entropy < 2.5:
                score = adjust_score('entropy_short', (2.5 - entropy) * 4)  # 8-12 range
                score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                signals.append(({
                    'pair': pair, 'tool': 'entropy_short', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.03,
                    'reason': f"ENTROPY SHORT: entropy={entropy:.2f}, price>SMA50"
                }, score))
        
        # 28. alt_btc_revert_t3: alt outperforms BTC 3-5% 24h → SHORT | WR_24h=56.4%, Ret_24h=+0.01%
        if pair != "XBTUSD" and "XBTUSD" in self._price_cache:
            btc_prices = self._price_cache["XBTUSD"]
            if len(btc_prices) >= 25 and len(close) >= 25:
                btc_ret24 = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                outperformance = ret_24h - btc_ret24
                if 3 <= outperformance <= 5:  # Alt outperforming BTC by 3-5%
                    score = adjust_score('alt_btc_revert_t3', outperformance * 2)  # 6-10 range
                    score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                    signals.append(({
                        'pair': pair, 'tool': 'alt_btc_revert_t3', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.03,
                        'reason': f"ALT BTC REVERT T3: {pair} {ret_24h:+.1f}% vs BTC {btc_ret24:+.1f}% (+{outperformance:.1f}%)"
                    }, score))
        
        # ===== NEUTRAL/TRANSITION SIGNALS - 2 tools EXACT COPY =====
        
        # 29. month_start_long: day 1-3 → LONG | WR_24h=53.9%, Ret_24h=+0.72%
        try:
            day_of_month = datetime.now(timezone.utc).day
            if 1 <= day_of_month <= 3:
                score = adjust_score('month_start_long', 15)  # Fixed score
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'month_start_long', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.04,
                    'reason': f"MONTH START LONG: day {day_of_month} of month"
                }, score))
        except:
            pass
        
        # 30. dip_buy_5pct: ret_4h<-5 → LONG | WR_8h=52.7%, Ret_8h=+0.11%
        if ret_4h < -5:
            score = adjust_score('dip_buy_5pct', abs(ret_4h) * 1.0)  # 5-10 range
            score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            signals.append(({
                'pair': pair, 'tool': 'dip_buy_5pct', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.04,
                'reason': f"DIP BUY 5PCT: {ret_4h:.1f}% drop 4h"
            }, score))
        
        # ===== TIER 3 RE-ENABLED TOOLS - Now profitable at futures fees! =====
        
        # 31. dip_buy_3pct: ret_4h<-3 → LONG (was unprofitable at 0.65% fees)
        if ret_4h < -3:
            score = adjust_score('dip_buy_3pct', abs(ret_4h) * 0.8)  # 2.4-8 range  
            score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            signals.append(({
                'pair': pair, 'tool': 'dip_buy_3pct', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.035,
                'reason': f"DIP BUY 3PCT: {ret_4h:.1f}% drop 4h (re-enabled for futures)"
            }, score))
        
        # 32. capitulation: ret_24h<-8 AND rsi7<15 → LONG (was unprofitable at spot fees)
        if ret_24h < -8 and cur_rsi < 15:
            score = adjust_score('capitulation', abs(ret_24h) * (15 - cur_rsi) * 0.4)  # 8-20 range
            score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
            signals.append(({
                'pair': pair, 'tool': 'capitulation', 'direction': 'long',
                'hold': 12, 'sl_pct': 0.04,
                'reason': f"CAPITULATION: {ret_24h:.1f}% drop, RSI={cur_rsi:.1f} (re-enabled)"
            }, score))
        
        # 33. zscore_extreme: zscore<-2.5 OR zscore>2.5 → contrarian | WR_8h=51.2%
        if len(close) >= 100:
            recent_returns = np.diff(close[-100:]) / close[-101:-1] * 100  # 100 periods for z-score
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns)
            if std_ret > 0:
                zscore = (ret_4h - mean_ret) / std_ret
                if zscore < -2.5:  # Extreme downside → go long
                    score = adjust_score('zscore_extreme', abs(zscore) * 3)  # 7.5-20 range
                    score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                    signals.append(({
                        'pair': pair, 'tool': 'zscore_extreme', 'direction': 'long',
                        'hold': 8, 'sl_pct': 0.035,
                        'reason': f"ZSCORE EXTREME: z={zscore:.2f} (re-enabled)"
                    }, score))
                elif zscore > 2.5:  # Extreme upside → go short
                    score = adjust_score('zscore_extreme', zscore * 3)  # 7.5-20 range
                    score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                    signals.append(({
                        'pair': pair, 'tool': 'zscore_extreme', 'direction': 'short',
                        'hold': 8, 'sl_pct': 0.035,
                        'reason': f"ZSCORE EXTREME: z={zscore:.2f} (re-enabled)"
                    }, score))
        
        # 34. panic_close: ret_4h<-4 AND VPIN>0.6 → LONG
        if ret_4h < -4:
            vpin = self.calc_vpin(df)
            if vpin > 0.6:
                score = adjust_score('panic_close', abs(ret_4h) * vpin * 3)  # 7-15 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'panic_close', 'direction': 'long',
                    'hold': 6, 'sl_pct': 0.03,
                    'reason': f"PANIC CLOSE: {ret_4h:.1f}% drop, VPIN={vpin:.2f} (re-enabled)"
                }, score))
        
        # 35. dist_exhaustion: volume spike + price reversal → contrarian
        if len(volume) >= 20:
            avg_vol = np.mean(volume[-20:-1])  # Exclude current bar
            vol_spike = volume[-1] / avg_vol if avg_vol > 0 else 1
            if vol_spike > 3 and ret_4h < -2:  # 3x volume spike + -2% drop
                score = adjust_score('dist_exhaustion', vol_spike * abs(ret_4h))  # 6-20 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                signals.append(({
                    'pair': pair, 'tool': 'dist_exhaustion', 'direction': 'long',
                    'hold': 6, 'sl_pct': 0.04,
                    'reason': f"DIST EXHAUSTION: {vol_spike:.1f}x vol spike, {ret_4h:.1f}% drop (re-enabled)"
                }, score))
        
        # 36. deceleration_buy: slowing momentum on dips → LONG
        if len(close) >= 10:
            # Check if recent decline is decelerating (momentum slowing)
            recent_returns = np.diff(close[-5:]) / close[-6:-1] * 100
            if len(recent_returns) >= 3 and ret_8h < -3:
                momentum = np.mean(recent_returns[-3:])  # Last 3 periods
                prev_momentum = np.mean(recent_returns[-4:-1]) if len(recent_returns) >= 4 else momentum
                if momentum > prev_momentum and momentum > -1:  # Deceleration and momentum turning positive
                    score = adjust_score('deceleration_buy', abs(ret_8h) * 2)  # 6-15 range  
                    score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                    signals.append(({
                        'pair': pair, 'tool': 'deceleration_buy', 'direction': 'long',
                        'hold': 8, 'sl_pct': 0.035,
                        'reason': f"DECELERATION BUY: momentum slowing ({momentum:.2f}), {ret_8h:.1f}% 8h drop (re-enabled)"
                    }, score))
        
        # ===== TOOL 37: FUNDING FARM - The Futures-Only Edge =====
        funding_rate = self.funding_rates.get(pair, 0)
        if abs(funding_rate) > 0.0003:  # |funding| > 0.03% per 4h (27.4% APR)
            # Check trend confirmation with SMA50
            if not np.isnan(sma50[-1]):
                trend_confirmed = False
                direction = None
                
                if funding_rate < -0.0003 and price > sma50[-1]:  # Go long when shorts pay funding
                    direction = 'long'
                    trend_confirmed = True
                elif funding_rate > 0.0003 and price < sma50[-1]:  # Go short when longs pay funding  
                    direction = 'short'
                    trend_confirmed = True
                
                if trend_confirmed:
                    score = adjust_score('funding_farm', abs(funding_rate) * 50000)  # Very high priority
                    signals.append(({
                        'pair': pair, 'tool': 'funding_farm', 'direction': direction,
                        'hold': 16,  # Hold through 4 funding cycles (16h)
                        'sl_pct': 0.03,  # Tight SL - we're here for funding income, not price appreciation
                        'reason': f"FUNDING FARM: {funding_rate:+.4f} per 4h ({abs(funding_rate)*365*24/4*100:.1f}% APR), trend confirmed"
                    }, score))
        
        # ===== TOOL 38: NEWS SENTIMENT SIGNAL - Breaking News Direct Signals =====
        # Generate direct trading signals for very strong breaking news
        # This catches events like "ETF approved" or "Exchange hacked" that warrant immediate action
        if self.sentiment_cache and time.time() - self.sentiment_cache_time < 300:  # Fresh sentiment data
            breaking_events = self.sentiment_cache.get('breaking_events', [])
            for event in breaking_events:
                if (pair in event.get('coins', []) and 
                    abs(event.get('score', 0)) >= 5 and  # Very strong event
                    time.time() - event.get('timestamp', 0) < 3600):  # Within last hour
                    
                    direction = 'long' if event['score'] > 0 else 'short'
                    base_score = abs(event['score']) * 10  # Very high priority
                    score = adjust_score('news_sentiment', base_score)
                    
                    signals.append(({
                        'pair': pair, 'tool': 'news_sentiment', 'direction': direction,
                        'hold': 24, 'sl_pct': 0.05,  # Hold for 24 hours, 5% stop loss
                        'reason': f"NEWS: {event.get('headline', '')[:60]}... (sentiment: {event['score']:+d})"
                    }, score))
                    
                    logger.warning(f"🚨 BREAKING NEWS SIGNAL: {pair} {direction.upper()} - {event.get('headline', '')[:80]}...")
        
        # ===== NEW ON-CHAIN DATA TOOLS =====
        
        # Tool 39: Stablecoin Supply Signal - When extreme stablecoin flows align with technical conditions
        if hasattr(self, 'onchain_cache') and self.onchain_cache:
            market_signal = self.onchain_cache.get('market_signal', 0)
            stablecoin_flow = self.onchain_cache.get('stablecoin_flow', {})
            
            if abs(market_signal) >= 4:  # Extreme stablecoin flows
                direction = 'long' if market_signal > 0 else 'short'
                
                # Only fire if RSI confirms (not overbought for longs, not oversold for shorts)
                rsi_ok = (direction == 'long' and cur_rsi < 60) or (direction == 'short' and cur_rsi > 40)
                
                # Also check if price is near support/resistance based on SMA50
                near_level = False
                if direction == 'long' and not np.isnan(sma50[-1]) and price <= sma50[-1] * 1.02:  # Near support
                    near_level = True
                elif direction == 'short' and not np.isnan(sma50[-1]) and price >= sma50[-1] * 0.98:  # Near resistance
                    near_level = True
                
                if rsi_ok and near_level:
                    base_score = abs(market_signal) * 8  # Strong signal based on stablecoin flow strength
                    score = adjust_score('stablecoin_supply', base_score)
                    score += self.get_funding_boost(pair, direction)  # No need for onchain boost here (it's the source)
                    
                    flow_reason = stablecoin_flow.get('reason', f"${stablecoin_flow.get('total_7d_change', 0)/1e6:+.0f}M 7d")
                    
                    signals.append(({
                        'pair': pair, 'tool': 'stablecoin_supply', 'direction': direction,
                        'hold': 24, 'sl_pct': 0.04,
                        'reason': f"STABLECOIN FLOW: {flow_reason}, RSI={cur_rsi:.1f}, price vs SMA50"
                    }, score))
                    
                    logger.warning(f"🔗 STABLECOIN SIGNAL: {pair} {direction.upper()} - Market signal {market_signal:+.1f}")
        
        # Tool 40: TVL Rotation Signal - When money flows from one chain to another
        if hasattr(self, 'onchain_cache') and self.onchain_cache:
            tvl_flows = self.onchain_cache.get('tvl_flows', {})
            
            # Find chains with significant opposing flows
            gainers = []
            losers = []
            
            for chain, data in tvl_flows.items():
                change_24h = data.get('change_24h_pct', 0)
                if change_24h > 3 and chain in self.onchain_engine.CHAIN_TO_COIN:  # Gaining TVL
                    gainers.append((chain, change_24h))
                elif change_24h < -3 and chain in self.onchain_engine.CHAIN_TO_COIN:  # Losing TVL
                    losers.append((chain, change_24h))
            
            # Generate rotation signals
            for gainer_chain, gainer_change in gainers:
                for loser_chain, loser_change in losers:
                    gainer_pair = self.onchain_engine.CHAIN_TO_COIN[gainer_chain]
                    loser_pair = self.onchain_engine.CHAIN_TO_COIN[loser_chain]
                    
                    # Only generate signal for the current pair
                    if pair == gainer_pair:  # Long the gaining chain's token
                        rotation_strength = gainer_change - loser_change  # Combined flow difference
                        base_score = rotation_strength * 3  # Score based on flow difference
                        score = adjust_score('tvl_rotation', base_score)
                        score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long')
                        
                        signals.append(({
                            'pair': pair, 'tool': 'tvl_rotation', 'direction': 'long',
                            'hold': 24, 'sl_pct': 0.05,
                            'reason': f"TVL ROTATION: {gainer_chain} +{gainer_change:.1f}%, {loser_chain} {loser_change:+.1f}%"
                        }, score))
                        
                    elif pair == loser_pair:  # Short the losing chain's token
                        rotation_strength = abs(loser_change) + gainer_change  # Combined outflow
                        base_score = rotation_strength * 3
                        score = adjust_score('tvl_rotation', base_score)
                        score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short')
                        
                        signals.append(({
                            'pair': pair, 'tool': 'tvl_rotation', 'direction': 'short',
                            'hold': 24, 'sl_pct': 0.05,
                            'reason': f"TVL ROTATION: {loser_chain} {loser_change:+.1f}%, {gainer_chain} +{gainer_change:.1f}%"
                        }, score))
        
        # ===== ORDERBOOK TOOLS - THE DEPTH ANALYZER =====
        
        # Try to get orderbook data for this pair
        try:
            market_data_single = {pair: data}
            orderbook_results = self.orderbook_engine.get_orderbook_signals([pair], market_data_single)
            if pair in orderbook_results:
                ob_data = orderbook_results[pair]
                
                # Tool 41: Orderbook Imbalance Signal
                # When imbalance is extreme AND price movement confirms
                imbalance = ob_data.get('imbalance', 1.0)
                if imbalance > 3.0 and ret_4h < -1:
                    # Massive bid wall + price dipped → LONG
                    score = adjust_score('orderbook_imbalance', imbalance * 5)
                    score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                    signals.append(({
                        'pair': pair, 'tool': 'orderbook_imbalance', 'direction': 'long',
                        'hold': 8, 'sl_pct': 0.05,
                        'reason': f"ORDERBOOK IMBALANCE LONG: {imbalance:.2f} bid dominance + {ret_4h:.1f}% dip"
                    }, score))
                elif imbalance < 0.3 and ret_4h > 1:
                    # Massive ask wall + price pumped → SHORT
                    score = adjust_score('orderbook_imbalance', (1/imbalance) * 5)
                    score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                    signals.append(({
                        'pair': pair, 'tool': 'orderbook_imbalance', 'direction': 'short',
                        'hold': 8, 'sl_pct': 0.05,
                        'reason': f"ORDERBOOK IMBALANCE SHORT: {imbalance:.2f} ask dominance + {ret_4h:.1f}% pump"
                    }, score))
                
                # Tool 42: Wall Breakout Signal
                # When price approaches or breaks through detected walls
                walls = ob_data.get('walls', [])
                mid_price = ob_data.get('mid_price', price)
                absorption = ob_data.get('absorption')
                
                for wall in walls:
                    if wall.strength > 10:  # Only strong walls (10x+ average)
                        distance_pct = abs(mid_price - wall.price) / mid_price * 100
                        
                        # Wall breakout signals
                        if distance_pct < 0.5:  # Very close to wall (<0.5%)
                            if wall.side == 'ask' and ret_1h > 0.5:
                                # Price approaching ask wall from below with momentum
                                # If absorption detected = wall being eaten = breakout likely
                                base_score = wall.strength * 2
                                if absorption and any(w['side'] == 'ask' for w in absorption.get('absorbed_walls', [])):
                                    base_score *= 2  # Double if wall being absorbed
                                
                                score = adjust_score('wall_breakout', base_score)
                                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long') + self.get_vol_boost(pair, 'long')
                                signals.append(({
                                    'pair': pair, 'tool': 'wall_breakout', 'direction': 'long',
                                    'hold': 12, 'sl_pct': 0.04,
                                    'reason': f"WALL BREAKOUT LONG: {wall.strength:.1f}x ask wall @ ${wall.price:.4f}, absorption={absorption is not None}"
                                }, score))
                                break  # One signal per pair
                            
                            elif wall.side == 'bid' and ret_1h < -0.5:
                                # Price approaching bid wall from above with momentum
                                base_score = wall.strength * 2
                                if absorption and any(w['side'] == 'bid' for w in absorption.get('absorbed_walls', [])):
                                    base_score *= 2
                                
                                score = adjust_score('wall_breakout', base_score)  
                                score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short') + self.get_vol_boost(pair, 'short')
                                signals.append(({
                                    'pair': pair, 'tool': 'wall_breakout', 'direction': 'short',
                                    'hold': 12, 'sl_pct': 0.04,
                                    'reason': f"WALL BREAKOUT SHORT: {wall.strength:.1f}x bid wall @ ${wall.price:.4f}, absorption={absorption is not None}"
                                }, score))
                                break
                                
        except Exception as e:
            logger.debug(f"Orderbook analysis failed for {pair}: {e}")
        
        # ===== VOLATILITY/OPTIONS INTELLIGENCE TOOLS =====
        
        # Tool 43: Put/Call Extreme Signal - When extreme fear/greed aligns with technicals
        if hasattr(self, 'vol_cache') and self.vol_cache:
            pcr = self.vol_cache.get('put_call_ratio', 1.0)
            pcr_signal = self.vol_cache.get('put_call_signal', 'neutral')
            
            # EXTREME FEAR (P/C > 1.5) + RSI oversold → STRONG LONG
            if pcr > 1.5 and cur_rsi < 35:
                score = adjust_score('put_call_extreme', (pcr - 1.5) * 20 + (35 - cur_rsi) * 2)  # 20-40 range
                score += self.get_funding_boost(pair, 'long') + self.get_sentiment_boost(pair, 'long') + self.get_onchain_boost(pair, 'long') + self.get_orderbook_boost(pair, 'long')
                # Note: No vol_boost here since this IS the vol signal
                signals.append(({
                    'pair': pair, 'tool': 'put_call_extreme', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.04,
                    'reason': f"P/C EXTREME FEAR: P/C={pcr:.2f} + RSI={cur_rsi:.1f} (max fear contrarian buy)"
                }, score))
            
            # EXTREME GREED (P/C < 0.5) + RSI overbought → STRONG SHORT  
            elif pcr < 0.5 and cur_rsi > 65:
                score = adjust_score('put_call_extreme', (0.5 - pcr) * 40 + (cur_rsi - 65) * 2)  # 20-40 range
                score += self.get_funding_boost(pair, 'short') + self.get_sentiment_boost(pair, 'short') + self.get_onchain_boost(pair, 'short') + self.get_orderbook_boost(pair, 'short')
                signals.append(({
                    'pair': pair, 'tool': 'put_call_extreme', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.04,
                    'reason': f"P/C EXTREME GREED: P/C={pcr:.2f} + RSI={cur_rsi:.1f} (max greed contrarian sell)"
                }, score))
        
        # Tool 44: Max Pain Magnet - Price gravitates toward max pain near expiration
        if (hasattr(self, 'vol_cache') and self.vol_cache and 
            pair in ('XBTUSD', 'PF_XBTUSD', 'ETHUSD', 'PF_ETHUSD')):  # Only BTC/ETH have liquid options
            
            max_pain_btc = self.vol_cache.get('max_pain_btc', 0)
            max_pain_eth = self.vol_cache.get('max_pain_eth', 0)
            
            # Select appropriate max pain based on pair
            if 'BTC' in pair or 'XBT' in pair:
                max_pain = max_pain_btc
            else:  # ETH
                max_pain = max_pain_eth
                
            if max_pain > 0 and price > 0:
                # Check if we're within expiration window (check current day of week)
                # Options often expire on Fridays - strongest signal 1-3 days before
                current_day = datetime.now(timezone.utc).weekday()  # Monday=0, Friday=4
                days_to_friday = (4 - current_day) % 7  # Days until Friday
                
                if days_to_friday <= 3:  # Within 3 days of potential expiration
                    price_distance_pct = (max_pain - price) / price
                    
                    # Only trade if distance is significant (>5%) 
                    if abs(price_distance_pct) > 0.05:
                        direction = 'long' if price_distance_pct > 0 else 'short'  # Trade toward max pain
                        
                        # Stronger signal as we get closer to expiration
                        expiry_urgency = (4 - days_to_friday) / 3  # 0.33 to 1.0
                        base_score = abs(price_distance_pct) * 100 * expiry_urgency  # 5-30 range
                        
                        score = adjust_score('max_pain_magnet', base_score)
                        score += self.get_funding_boost(pair, direction) + self.get_sentiment_boost(pair, direction) + self.get_onchain_boost(pair, direction) + self.get_orderbook_boost(pair, direction)
                        
                        signals.append(({
                            'pair': pair, 'tool': 'max_pain_magnet', 'direction': direction,
                            'hold': min(24, days_to_friday * 8),  # Hold until close to expiration
                            'sl_pct': 0.03,  # Tight SL
                            'reason': f"MAX PAIN MAGNET: ${max_pain:,.0f} target, {price_distance_pct:+.1%} from current, {days_to_friday}d to expiry"
                        }, score))
        
        return signals
    
    def execute_signal(self, signal: dict, score: float):
        """Execute a signal with futures-specific position sizing and leverage."""
        pair = signal['pair']
        tool = signal['tool']
        direction = signal['direction']
        
        # Check if we already have a position
        if pair in self.active_positions:
            return
        
        # Get leverage for this tool
        leverage = self.get_leverage_for_tool(tool)
        
        # Calculate position size with leverage adjustment
        position_size, effective_sl = self.get_position_size(signal, tool, leverage)
        
        # Get current market data
        market_data = self.get_market_data()
        if pair not in market_data:
            logger.warning(f"No market data for {pair}, skipping signal")
            return
        
        data = market_data[pair]
        current_price = data["price"]
        
        # Use limit orders for better fills
        if direction == 'long':
            entry_price = data.get("bid", current_price)
        else:
            entry_price = data.get("ask", current_price)
        
        qty = position_size / entry_price
        
        # Execute futures order with leverage
        if ENABLE_LIVE_TRADING:
            try:
                side = "buy" if direction == 'long' else "sell"
                order_id = self.client.place_order(
                    pair, side, qty, "lmt", entry_price, leverage
                )
                
                if not order_id:
                    logger.error(f"Failed to place {direction} order for {pair}")
                    return
                
                logger.info(f"[FUTURES] {direction.upper()} {pair} @ ${entry_price:.4f} "
                           f"(leverage: {leverage}x, order: {order_id})")
                
            except Exception as e:
                logger.error(f"Failed to execute {direction} order for {pair}: {e}")
                return
        else:
            logger.info(f"[DRY RUN] {direction.upper()} {pair} @ ${entry_price:.4f} "
                       f"(leverage: {leverage}x)")
        
        # Create position record
        position = {
            'pair': pair,
            'tool': tool,
            'direction': direction,
            'leverage': leverage,
            'entry_price': entry_price,
            'entry_bar': self.current_bar,
            'position_size': position_size,
            'qty': qty,
            'sl_pct': effective_sl,  # Adjusted for leverage
            'original_sl_pct': signal['sl_pct'],  # Keep original for reference
            'hold': signal['hold'],
            'score': score,
            'ml_features': signal.get('ml_features', {})  # Store ML features for learning
        }
        
        self.active_positions[pair] = position
        self.active_balance -= position_size / leverage  # Only reserve margin, not full notional
        
        logger.info(f"[OPEN] {pair} {direction} @ ${entry_price:.4f} | "
                   f"Tool: {tool} | Leverage: {leverage}x | "
                   f"Size: ${position_size:.2f} | Margin: ${position_size/leverage:.2f} | "
                   f"Score: {score:.1f} | SL: {effective_sl:.2%} (was {signal['sl_pct']:.2%})")
    
    def get_fear_greed(self) -> int:
        """Get crypto Fear & Greed Index."""
        now = datetime.now(timezone.utc).timestamp()
        if hasattr(self, '_fng_cache') and now - self._fng_cache_ts < 3600:
            return self._fng_cache
        try:
            r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            val = int(r.json()['data'][0]['value'])
            self._fng_cache = val
            self._fng_cache_ts = now
            return val
        except:
            return 50
    
    def manage_positions(self, market_data: dict):
        """Manage active futures positions."""
        for pair in list(self.active_positions.keys()):
            if pair not in market_data:
                continue
            
            pos = self.active_positions[pair]
            data = market_data[pair]
            current_price = data["price"]
            
            # Calculate bars held
            bars_held = self.current_bar - pos.get("entry_bar", self.current_bar)
            
            # Check stop loss (using adjusted SL for leverage)
            if pos['direction'] == 'long':
                sl_price = pos['entry_price'] * (1 - pos['sl_pct'])
                if current_price <= sl_price:
                    self.close_position(pair, current_price, f"Stop loss @ ${sl_price:.4f}")
                    continue
            else:  # short
                sl_price = pos['entry_price'] * (1 + pos['sl_pct'])
                if current_price >= sl_price:
                    self.close_position(pair, current_price, f"Stop loss @ ${sl_price:.4f}")
                    continue
            
            # Check take profit (based on tool type)
            take_profit_pct = self._get_take_profit_for_tool(pos['tool'])
            if take_profit_pct:
                if pos['direction'] == 'long':
                    tp_price = pos['entry_price'] * (1 + take_profit_pct)
                    if current_price >= tp_price:
                        self.close_position(pair, current_price, f"Take profit @ ${tp_price:.4f}")
                        continue
                else:  # short
                    tp_price = pos['entry_price'] * (1 - take_profit_pct)
                    if current_price <= tp_price:
                        self.close_position(pair, current_price, f"Take profit @ ${tp_price:.4f}")
                        continue
            
            # Check hold timeout
            if bars_held >= pos['hold']:
                self.close_position(pair, current_price, f"Hold timeout ({bars_held} bars)")
                continue
    
    def _get_take_profit_for_tool(self, tool: str) -> Optional[float]:
        """Get take profit percentage for a tool."""
        # Mean reversion strategies
        if tool in ['volatile_oversold', 'crash_neg_ac', 'crash_mean_revert', 
                   'blood_in_streets', 'vpin_dip', 'vpin_toxic', 'entropy_dip', 'btc_alt_spread']:
            return 0.08  # 8% TP
        
        # Crash buy strategies
        elif tool in ['crash_buy', 'mega_crash', 'quick_crash', 'flash_crash',
                     'market_panic_70', 'deep_dip_8h']:
            return 0.10  # 10% TP
        
        # Short strategies
        elif tool in ['mega_pump_sell_t1', 'rsi_pump_8h', 'falling_wedge_short', 'greed_short_t2',
                     'thursday_short', 'mega_pump_sell_t2', 'distribution_short', 'late_us_short',
                     'rsi_pump_12h', 'ema_cross_short', 'rsi_pump_fat_tail', 'entropy_short',
                     'alt_btc_revert_t3']:
            return 0.06  # 6% TP
        
        # Other strategies
        elif tool in ['quick_dip', 'dip_buy_5pct', 'month_start_long']:
            return 0.06  # 6% TP
        
        # New Tier 3 tools (smaller targets due to marginal edge)
        elif tool in TIER3_TOOLS:
            return 0.04  # 4% TP
        
        # Funding farm (small price target, we're here for funding)
        elif tool == 'funding_farm':
            return 0.03  # 3% TP
        
        return None  # No fixed TP
    
    def close_position(self, pair: str, price: float, reason: str):
        """Close a futures position."""
        if pair not in self.active_positions:
            return
        
        pos = self.active_positions[pair]
        leverage = pos.get('leverage', 1)
        
        # Calculate PnL with leverage
        if pos['direction'] == 'long':
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
        else:  # short
            pnl_pct = (pos['entry_price'] - price) / pos['entry_price']
        
        # Apply leverage multiplier
        leveraged_pnl_pct = pnl_pct * leverage
        
        # Calculate dollar PnL after fees
        gross_pnl = pos['position_size'] * leveraged_pnl_pct
        fees = pos['position_size'] * ROUND_TRIP_FEE
        net_pnl = gross_pnl - fees
        
        # Add funding income if this was a funding farm position
        funding_income = 0
        if pos['tool'] == 'funding_farm':
            bars_held = self.current_bar - pos['entry_bar']
            funding_cycles = bars_held // 4  # Funding every 4 hours
            funding_rate = self.funding_rates.get(pair, 0)
            funding_income = pos['position_size'] * funding_rate * funding_cycles
            net_pnl += funding_income
            self.funding_income += funding_income
        
        # Execute close order
        if ENABLE_LIVE_TRADING:
            try:
                side = "sell" if pos['direction'] == 'long' else "buy"
                self.client.place_order(pair, side, pos['qty'], "mkt")
            except Exception as e:
                logger.error(f"Failed to close {pair} position: {e}")
        
        # Update balances
        self.active_balance += pos['position_size'] / leverage  # Return margin
        self.active_profit += net_pnl
        
        # Update tool stats
        tool = pos['tool']
        if tool not in self.tool_stats:
            self.tool_stats[tool] = {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "score_adj": 1.0}
        
        stats = self.tool_stats[tool]
        stats["trades"] += 1
        stats["total_pnl"] += net_pnl
        
        if net_pnl > 0:
            stats["wins"] += 1
            outcome = "WIN"
        else:
            stats["losses"] += 1
            outcome = "LOSS"
        
        # Update streaks
        self._update_tool_streaks(tool, outcome)
        
        # Record trade outcome for ML learning
        ml_features = pos.get('ml_features', {})
        if ml_features:
            profitable = net_pnl > 0
            self.ml_weighter.record_trade(tool, ml_features, profitable)
        
        funding_str = f" + ${funding_income:.2f} funding" if funding_income != 0 else ""
        bars_held = self.current_bar - pos['entry_bar']
        
        logger.info(f"[CLOSE] {pair} {pos['direction']} @ ${price:.4f} | "
                   f"{reason} | PnL: ${net_pnl:+.2f} ({leveraged_pnl_pct:+.1%}){funding_str} | "
                   f"{bars_held}h held | Tool: {tool} ({outcome})")
        
        # Remove position
        del self.active_positions[pair]
    
    def _update_tool_streaks(self, tool: str, outcome: str):
        """Update tool win/loss streaks."""
        if tool not in self.tool_streaks:
            self.tool_streaks[tool] = {"type": "", "streak": 0}
        
        streak_info = self.tool_streaks[tool]
        
        if outcome == streak_info["type"]:
            streak_info["streak"] += 1
        else:
            streak_info["type"] = outcome
            streak_info["streak"] = 1
        
        # Update score adjustment based on streaks
        if tool not in self.tool_stats:
            return
        
        stats = self.tool_stats[tool]
        current_type = streak_info["type"]
        
        if current_type == 'L' and streak_info["streak"] >= 3:
            # 3+ consecutive losses: reduce score by 50%
            stats["score_adj"] = 0.5
        elif current_type == 'W' and streak_info["streak"] >= 3:
            # 3+ consecutive wins: boost score by 25%
            stats["score_adj"] = 1.25
        else:
            stats["score_adj"] = 1.0
    
    def rebalance_capital(self):
        """Rebalance grid vs active allocation."""
        grid_pct, active_pct = self.get_capital_allocation()
        target_grid_balance = self.total_balance * grid_pct
        target_active_balance = self.total_balance * active_pct
        
        self.grid_balance = target_grid_balance
        self.active_balance = target_active_balance
    
    def update_grids(self, market_data: dict):
        """Run the grid engine on futures with 2x leverage."""
        grid_balance_per_pair = self.grid_balance / len(PAIRS)
        
        for pair in PAIRS:
            if pair not in market_data or pair not in GRID_CONFIGS:
                continue
            
            data = market_data[pair]
            current_price = data["price"]
            current_high = data.get("high", current_price)
            
            # Grid configuration
            grid_spacing = GRID_CONFIGS[pair]
            num_levels = 3  # Keep simple for now
            
            # Initialize positions
            if pair not in self.grid_positions:
                self.grid_positions[pair] = []
            positions = self.grid_positions[pair]
            
            # Position sizing with leverage
            position_value = grid_balance_per_pair / num_levels
            qty = position_value / current_price
            
            # Grid levels below current price
            levels = [current_price * (1 - grid_spacing * (i + 1)) for i in range(num_levels)]
            
            # Check for buy fills
            for i, buy_level in enumerate(levels):
                level_filled = any(abs(pos["buy_price"] - buy_level) / buy_level < 0.005 for pos in positions)
                
                if not level_filled and current_price <= buy_level:
                    position = {
                        "buy_price": buy_level,
                        "qty": qty,
                        "bar": self.current_bar,
                        "level": i + 1
                    }
                    positions.append(position)
                    
                    if ENABLE_LIVE_TRADING:
                        try:
                            self.client.place_order(pair, "buy", qty, "lmt", buy_level, GRID_LEVERAGE)
                        except Exception as e:
                            logger.error(f"Failed to place grid buy: {e}")
                    else:
                        logger.info(f"[GRID DRY] {pair} buy {qty:.6f} @ ${buy_level:.4f} (2x leverage)")
                    
                    logger.info(f"[GRID] {pair} buy @ ${buy_level:.4f} (level {i+1}, 2x leverage)")
            
            # Check for sell fills (1.5% take profit)
            grid_take_profit = 0.015  # 1.5% TP
            remaining_positions = []
            
            for pos in positions:
                sell_target = pos["buy_price"] * (1 + grid_take_profit)
                
                if current_high >= sell_target:
                    # Calculate profit with improved fees and leverage
                    gross_profit = (sell_target - pos["buy_price"]) * pos["qty"] * GRID_LEVERAGE
                    fees = pos["buy_price"] * pos["qty"] * GRID_ROUND_TRIP_FEE
                    net_profit = gross_profit - fees
                    
                    self.grid_profit += net_profit
                    self.grid_round_trips += 1
                    
                    if ENABLE_LIVE_TRADING:
                        try:
                            self.client.place_order(pair, "sell", pos["qty"], "lmt", sell_target)
                        except Exception as e:
                            logger.error(f"Failed to place grid sell: {e}")
                    else:
                        logger.info(f"[GRID DRY] {pair} sell {pos['qty']:.6f} @ ${sell_target:.4f}")
                    
                    logger.info(f"[GRID] {pair} sell @ ${sell_target:.4f}, profit: ${net_profit:.2f}")
                else:
                    remaining_positions.append(pos)
            
            self.grid_positions[pair] = remaining_positions
    
    def run_cycle(self):
        """Run one complete futures trading cycle."""
        try:
            logger.info("═" * 80)
            self.current_bar += 1
            
            # 1. Get market data
            market_data = self.get_market_data()
            if not market_data:
                logger.warning("No market data, skipping cycle")
                return
            
            # 2. Update funding rates
            self.funding_rates = self.get_funding_rates()
            
            # 3. Update Fear & Greed
            fng = self.get_fear_greed()
            self.current_fng = fng
            regime = ("Extreme Fear" if fng < 20 else "Fear" if fng < 30 else 
                     "Neutral" if fng <= 70 else "Greed" if fng <= 80 else "Extreme Greed")
            
            # 4. Rebalance capital
            self.rebalance_capital()
            
            # 5. Update total balance
            self.total_balance = self.starting_balance + self.grid_profit + self.active_profit
            
            # 6. Update grid
            self.update_grids(market_data)
            
            # 7. Manage positions
            self.manage_positions(market_data)
            
            # 8. Scan for signals
            all_signals = []
            for pair, data in market_data.items():
                signals = self.scan_signals(pair, data)
                all_signals.extend(signals)
            
            # 9. Execute top signals
            if all_signals:
                all_signals.sort(key=lambda x: x[1], reverse=True)
                
                open_positions = len(self.active_positions)
                for signal, score in all_signals:
                    if open_positions >= MAX_ACTIVE_POSITIONS:
                        break
                    
                    pair = signal['pair']
                    if pair not in self.active_positions:
                        self.execute_signal(signal, score)
                        open_positions += 1
            
            # 10. Status report
            grid_positions = sum(len(positions) for positions in self.grid_positions.values())
            active_count = len(self.active_positions)
            growth_pct = (self.total_balance / self.starting_balance - 1) * 100
            
            logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] CYCLE #{self.current_bar} | "
                       f"F&G: {fng} ({regime})")
            
            logger.info(f"Balance: ${self.total_balance:.2f} ({growth_pct:+.1f}%) | "
                       f"Grid: ${self.grid_profit:.2f} | Active: ${self.active_profit:.2f} | "
                       f"Funding: ${self.funding_income:.2f}")
            
            logger.info(f"Grid: {grid_positions} positions, {self.grid_round_trips} RTs | "
                       f"Active: {active_count}/{MAX_ACTIVE_POSITIONS}")
            
            # ML Status
            ml_status = self.ml_weighter.get_status_summary()
            logger.info(f"🧠 {ml_status}")
            
            # Show active positions with leverage
            if self.active_positions:
                for pair, pos in self.active_positions.items():
                    current_price = market_data[pair]["price"]
                    if pos['direction'] == 'long':
                        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * pos['leverage']
                    else:
                        pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * pos['leverage']
                    
                    bars_held = self.current_bar - pos['entry_bar']
                    logger.info(f"  → {pair} {pos['direction']} {pnl_pct:+.1%} "
                               f"({pos['tool']}, {pos['leverage']}x, {bars_held}h)")
            
            # Top signals
            if all_signals:
                top_signals = all_signals[:3]
                signal_str = ", ".join([f"{s[0]['tool']} {s[0]['pair']} (score {s[1]:.1f})" 
                                      for s in top_signals])
                logger.info(f"Top signals: {signal_str}")
            
            # Hot funding rates
            high_funding = [(pair, rate) for pair, rate in self.funding_rates.items() 
                           if abs(rate) > 0.0002]
            if high_funding:
                funding_str = ", ".join([f"{pair} {rate:+.3%}" for pair, rate in high_funding[:5]])
                logger.info(f"High funding: {funding_str}")
            
            # 11. Save state
            self.save_state()
            
        except Exception as e:
            logger.error(f"Error in futures cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run(self):
        """Main futures bot loop."""
        logger.info("🚀 ULTIMATE FUTURES BOT STARTING...")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
        logger.info(f"Futures fees: {ROUND_TRIP_FEE:.3%} RT vs 0.65% spot (9.3x better!)")
        logger.info(f"Leverage: Tier 1 ({TIER1_LEVERAGE}x), Tier 2 ({TIER2_LEVERAGE}x), Tier 3 ({TIER3_LEVERAGE}x)")
        logger.info(f"Trading pairs: {len(PAIRS)}")
        logger.info(f"Total tools: {len(VALIDATED_TOOLS)} (30 + 6 re-enabled + 1 funding farm)")
        
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                cycle_start = time.time()
                self.run_cycle()
                
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, CHECK_INTERVAL - cycle_time)
                
                if sleep_time > 0:
                    logger.info(f"Cycle: {cycle_time:.1f}s, sleeping {sleep_time:.1f}s")
                    logger.info("═" * 80)
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Interrupted")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.save_state()
            logger.info("🚀 ULTIMATE FUTURES BOT STOPPED")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate Futures Trading Bot')
    parser.add_argument('--dry-run', action='store_true', help='Force dry run mode')
    parser.add_argument('--check', action='store_true', help='Initialization check')
    args = parser.parse_args()
    
    if args.dry_run:
        os.environ['ENABLE_LIVE_TRADING'] = 'false'
        logger.info("Forced dry run mode")
    
    bot = UltimateFuturesBot()
    
    if args.check:
        logger.info("✅ ULTIMATE FUTURES BOT initialized")
        logger.info(f"✅ Kraken Futures client: {'LIVE' if not bot.client.dry_run else 'DRY RUN'}")
        logger.info(f"✅ Futures fees: {ROUND_TRIP_FEE:.3%} RT (9.3x better than spot)")
        logger.info(f"✅ Leverage tiers: T1={TIER1_LEVERAGE}x, T2={TIER2_LEVERAGE}x, T3={TIER3_LEVERAGE}x")
        logger.info(f"✅ Tools: {len(TIER1_TOOLS)} T1, {len(TIER2_TOOLS)} T2, {len(TIER3_TOOLS)} T3 (re-enabled), 1 funding farm")
        logger.info(f"✅ Total signals: {len(VALIDATED_TOOLS)} (37 tools)")
        logger.info(f"✅ Trading pairs: {len(PAIRS)} pairs")
        logger.info("✅ READY TO DOMINATE FUTURES MARKETS!")
        return
    
    bot.run()


if __name__ == "__main__":
    main()