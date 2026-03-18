#!/usr/bin/env python3
"""
THE ALL-SEEING EYE - Master Trading Bot
One brain. One capital pool. Many signal sources.

The brain scans everything every cycle, scores every opportunity, 
and deploys capital to the best ones. No regime detection - just
evaluate ALL signals from ALL tools on ALL pairs every cycle.

Signal Sources:
1. Grid Engine (passive income, separate capital)
2. Crash Buy (BEST EDGE: 59% WR, +6.17% avg)
3. Volatile Oversold Buy (SECOND BEST: 50% WR, +12.11% avg)
4. Downtrend Relief Rally (THIRD BEST: 68% WR, +3.56% avg)
5. Overbought Sell (decent edge: 43% WR)
6. Dip Buy (frequent small edge: 44% WR)  
7. Pump Sell (frequent small edge: 41% WR)
"""

import sys
import os
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

# Add src directory for kraken_client
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from kraken_client import KrakenClient
except ImportError as e:
    logger.error(f"Failed to import KrakenClient: {e}")
    sys.exit(1)

# Configuration
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "300"))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = DATA_DIR / "master_bot_state.json"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add(LOGS_DIR / "master_bot.log", rotation="10 MB", retention="30 days")

# All pairs to trade
PAIRS = [
    "NEARUSD", "UNIUSD", "AVAXUSD", "LINKUSD", "AAVEUSD", "SOLUSD",
    "ETHUSD", "XBTUSD", "DOTUSD", "XLMUSD", "XRPUSD", "ADAUSD", 
    "ATOMUSD", "DOGEUSD", "FILUSD", "LTCUSD"
]

# Grid configurations (Tool 1)
GRID_CONFIGS = {
    "NEARUSD": 0.01, "UNIUSD": 0.015, "AVAXUSD": 0.01, "LINKUSD": 0.008,
    "AAVEUSD": 0.015, "SOLUSD": 0.003, "ETHUSD": 0.005, "XBTUSD": 0.01,
    "DOTUSD": 0.012, "XLMUSD": 0.01, "XRPUSD": 0.01, "ADAUSD": 0.012,
    "ATOMUSD": 0.008, "DOGEUSD": 0.012, "FILUSD": 0.015, "LTCUSD": 0.01,
}

# Constants
MAX_ACTIVE_POSITIONS = 5  # Max simultaneous active positions
GRID_CAPITAL_PCT = 0.40   # 40% of balance for grid
ACTIVE_CAPITAL_PCT = 0.60 # 60% of balance for active trading
RISK_PER_TRADE = 0.05     # 5% of active balance per trade
GRID_TAKE_PROFIT = 0.015  # 1.5% take profit for grid


class AllSeeingEye:
    """The All-Seeing Eye - One brain that sees all opportunities."""
    
    def __init__(self):
        self.client = KrakenClient()
        self.running = True
        self.state = self.load_state()
        
        # Balance tracking
        self.total_balance = self.state.get("total_balance", STARTING_BALANCE)
        self.grid_balance = self.total_balance * GRID_CAPITAL_PCT
        self.active_balance = self.total_balance * ACTIVE_CAPITAL_PCT
        
        # Grid state (passive income)
        self.grid_positions = self.state.get("grid_positions", {})  # pair -> [{"buy_price": x, "qty": y, "bar": z}]
        self.grid_profit = self.state.get("grid_profit", 0.0)
        self.grid_round_trips = self.state.get("grid_round_trips", 0)
        
        # Active positions (signal-based)
        self.active_positions = self.state.get("active_positions", {})  # pair -> position_data
        self.active_profit = self.state.get("active_profit", 0.0)
        
        # Tool performance tracking
        self.tool_stats = self.state.get("tool_stats", {})
        # Ensure all tools have stats entries
        for tool in ["crash_buy", "volatile_oversold", "relief_rally", "mega_pump_sell", "green_exhaustion",
                      "dip_buy", "pump_sell", "mega_crash", "flash_crash", "quick_crash",
                      "deep_dip_8h", "deep_dip_12h", "deep_dip_24h", "quick_dip",
                      "btc_eth_diverge", "rsi_divergence", "thursday_short", "crash_neg_ac", "crash_mean_revert", "hurst_trend", "vpin_toxic", "vpin_dip", "entropy_dip", "triple_math"]:
            if tool not in self.tool_stats:
                self.tool_stats[tool] = {"trades": 0, "wins": 0, "pnl": 0.0}
        
        # Price cache for cross-pair signals (e.g. BTC/ETH divergence)
        self._price_cache = {}  # pair -> close_array
        
        # Trade history
        self.trade_history = self.state.get("trade_history", [])
        self.current_bar = self.state.get("current_bar", 0)
        
        logger.info(f"All-Seeing Eye initialized")
        logger.info(f"Total balance: ${self.total_balance:.2f}")
        logger.info(f"Grid balance: ${self.grid_balance:.2f} (40%)")
        logger.info(f"Active balance: ${self.active_balance:.2f} (60%)")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
        
    def load_state(self) -> dict:
        """Load bot state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    logger.info("Loaded existing state")
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
            "active_positions": self.active_positions,
            "active_profit": self.active_profit,
            "tool_stats": self.tool_stats,
            "trade_history": self.trade_history[-500:],  # Keep last 500
            "current_bar": self.current_bar,
            "last_update": datetime.now().isoformat()
        }
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def get_market_data(self) -> dict:
        """Fetch 1h market data for all pairs."""
        market_data = {}
        
        for pair in PAIRS:
            try:
                # Get 1h klines (last 200 bars for indicators)
                klines = self.client.get_klines(pair, interval="1h", limit=200)
                if not klines:
                    continue
                    
                df = pd.DataFrame(klines)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                if len(df) < 50:  # Need enough data for indicators
                    continue
                    
                # Get current ticker
                ticker = self.client.get_ticker(pair)
                current_price = float(ticker["price"]) if ticker else float(df['close'].iloc[-1])
                
                market_data[pair] = {
                    "price": current_price,
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "df": df
                }
                # Cache close prices for cross-pair signals
                self._price_cache[pair] = df['close'].values.astype(float)
                
            except Exception as e:
                logger.error(f"Failed to get data for {pair}: {e}")
                
        logger.info(f"Retrieved data for {len(market_data)}/{len(PAIRS)} pairs")
        return market_data
        
    def calc_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
            
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(delta))
        avg_loss = np.zeros(len(delta))
        
        # Initial averages
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Smoothed averages
        for i in range(period, len(delta)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad to match input length
        return np.concatenate([[50.0], rsi])
        
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
            
        sma = np.full(len(prices), np.nan)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            
        return sma
        
    def calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        if len(high) < 2:
            return np.full(len(high), 0.0)
            
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First bar uses high-low
        
        if len(tr) < period:
            return tr
            
        atr = np.full(len(tr), np.nan)
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        return atr
        
    def scan_signals(self, pair: str, data: dict) -> List[Tuple[dict, float]]:
        """Scan all tools for signals on this pair. Return [(signal, score), ...]"""
        signals = []
        df = data['df']
        price = data['price']
        
        if len(df) < 50:
            return signals
            
        # Compute features once
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        rsi7 = self.calc_rsi(close, 7)
        sma50 = self.calc_sma(close, 50)
        atr14 = self.calc_atr(high, low, close, 14)
        
        cur_rsi = rsi7[-1]
        cur_atr_pct = atr14[-1] / price * 100 if price > 0 and not np.isnan(atr14[-1]) else 0
        cur_vs_sma50 = (price - sma50[-1]) / sma50[-1] * 100 if not np.isnan(sma50[-1]) and sma50[-1] > 0 else 0
        ret_4h = (price - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
        ret_24h = (price - close[-25]) / close[-25] * 100 if len(close) >= 25 else 0
        
        # Tool 2: Crash Buy (BEST EDGE)
        if ret_24h < -10 and cur_rsi < 20:
            score = (20 - cur_rsi) * 2
            signals.append(({
                'pair': pair, 'tool': 'crash_buy', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"CRASH BUY: {ret_24h:.1f}% drop 24h, RSI={cur_rsi:.1f}"
            }, score))
        
        # Tool 3: Volatile Oversold (SECOND BEST)
        if cur_atr_pct > 3 and cur_rsi < 25:
            score = cur_atr_pct * (25 - cur_rsi)
            signals.append(({
                'pair': pair, 'tool': 'volatile_oversold', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.08,
                'reason': f"VOLATILE OVERSOLD: ATR={cur_atr_pct:.1f}%, RSI={cur_rsi:.1f}"
            }, score))
        
        # Tool 4: Downtrend Relief Rally (THIRD BEST)
        if cur_rsi > 75 and cur_vs_sma50 < 0:
            score = (cur_rsi - 75) * 1.5
            signals.append(({
                'pair': pair, 'tool': 'relief_rally', 'direction': 'long',
                'hold': 12, 'sl_pct': 0.03,
                'reason': f"RELIEF RALLY: RSI={cur_rsi:.1f}, below SMA50 by {cur_vs_sma50:.1f}%"
            }, score))
        

        
        # Tool 6: Dip Buy
        if ret_4h < -3:
            score = abs(ret_4h) * 2
            signals.append(({
                'pair': pair, 'tool': 'dip_buy', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.99,
                'reason': f"DIP BUY: {ret_4h:.1f}% drop in 4h"
            }, score))
        
        # Compute additional features for pump/crash tools
        ret_8h = (price - close[-9]) / close[-9] * 100 if len(close) >= 9 else 0
        ret_12h = (price - close[-13]) / close[-13] * 100 if len(close) >= 13 else 0
        
        # Tool 7a: Mega Pump Sell — RSI>80 + 8% pump 8h → 72% WR, +2.82% avg
        if cur_rsi > 80 and ret_8h > 8:
            score = ret_8h * 3 + (cur_rsi - 80)
            signals.append(({
                'pair': pair, 'tool': 'mega_pump_sell', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.99,  # No SL
                'reason': f"MEGA PUMP SELL: RSI={cur_rsi:.1f}, +{ret_8h:.1f}% 8h — 72% WR"
            }, score))
        

        
        # ── NEW TOOLS FROM DEEP QUANT ──
        
        # Tool 8: Mega Crash Buy (>15% drop 24h) — 80% WR, +12.87% avg 24h
        if ret_24h < -15:
            score = abs(ret_24h) * 3  # Highest priority signal
            signals.append(({
                'pair': pair, 'tool': 'mega_crash', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.08,
                'reason': f"MEGA CRASH: {ret_24h:.1f}% drop 24h — 80% WR historically"
            }, score))
        
        # Tool 9: Flash Crash Buy (>10% in 12h) — 77% WR, +7.87% avg 24h
        if ret_12h < -10:
            score = abs(ret_12h) * 2.5
            signals.append(({
                'pair': pair, 'tool': 'flash_crash', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.07,
                'reason': f"FLASH CRASH: {ret_12h:.1f}% drop 12h — 77% WR"
            }, score))
        
        # Tool 10: Quick Crash (>10% in 8h) — 69% WR, +7.05% avg 24h
        if ret_8h < -10:
            score = abs(ret_8h) * 2
            signals.append(({
                'pair': pair, 'tool': 'quick_crash', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.07,
                'reason': f"QUICK CRASH: {ret_8h:.1f}% drop 8h — 69% WR"
            }, score))
        
        # Tool 11: Deep Dip (>8% in various timeframes) — 61-66% WR
        for tf_name, ret_val, tf_label in [("8h", ret_8h, "8h"), ("12h", ret_12h, "12h"), ("24h", ret_24h, "24h")]:
            if ret_val < -8 and ret_val >= -10:  # 8-10% drop (don't overlap with 10%+ tools)
                score = abs(ret_val) * 1.5
                signals.append(({
                    'pair': pair, 'tool': f'deep_dip_{tf_label}', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.05,
                    'reason': f"DEEP DIP: {ret_val:.1f}% drop {tf_label} — 64% WR"
                }, score))
        
        # Tool 12: Quick Dip (>5% in 4h) — 58% WR, +3.15% avg 24h
        if ret_4h < -5:
            score = abs(ret_4h) * 2
            signals.append(({
                'pair': pair, 'tool': 'quick_dip', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.99,
                'reason': f"QUICK DIP: {ret_4h:.1f}% drop 4h — 58% WR"
            }, score))
        
        # Tool 13: BTC/ETH Divergence — when BTC outperforms ETH by 3%+, short (sell ETH catchup)
        if pair == "ETHUSD" and "XBTUSD" in self._price_cache:
            btc_prices = self._price_cache.get("XBTUSD")
            if btc_prices is not None and len(btc_prices) >= 25 and len(close) >= 25:
                btc_ret24 = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                eth_ret24 = ret_24h
                if btc_ret24 - eth_ret24 > 3:
                    score = (btc_ret24 - eth_ret24) * 2
                    signals.append(({
                        'pair': pair, 'tool': 'btc_eth_diverge', 'direction': 'short',
                        'hold': 8, 'sl_pct': 0.03,
                        'reason': f"BTC/ETH DIVERGE: BTC {btc_ret24:+.1f}% vs ETH {eth_ret24:+.1f}% 24h"
                    }, score))
        
        # Tool 14: RSI Divergence (bullish) — price lower low but RSI higher low
        if len(close) >= 30:
            rsi14 = self.calc_rsi(close, 14)
            if not np.isnan(rsi14[-1]) and not np.isnan(rsi14[-14]):
                recent_price_low = np.min(close[-14:])
                prior_price_low = np.min(close[-28:-14]) if len(close) >= 28 else recent_price_low
                recent_rsi_low = np.min(rsi14[-14:])
                prior_rsi_low = np.min(rsi14[-28:-14]) if len(rsi14) >= 28 else recent_rsi_low
                
                if recent_price_low < prior_price_low and recent_rsi_low > prior_rsi_low and rsi14[-1] < 35:
                    score = (prior_rsi_low - recent_rsi_low + 10) * 0.5  # Lower score, but valid
                    signals.append(({
                        'pair': pair, 'tool': 'rsi_divergence', 'direction': 'short',
                        'hold': 8, 'sl_pct': 0.03,
                        'reason': f"RSI DIVERGENCE: price lower low, RSI higher low"
                    }, score))
        
        # Tool 15: Day-of-week filter (Thursday/Sunday short bias)
        try:
            dow = datetime.now(timezone.utc).weekday()
            if dow == 3 and cur_rsi > 50:
                score = 3
                signals.append(({
                    'pair': pair, 'tool': 'thursday_short', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.03,
                    'reason': f"THURSDAY SHORT: consistent weekly pattern"
                }, score))
        except:
            pass
        
        # ── DEEP QUANT V2 TOOLS ──
        
        # Tool 16: Market Panic (90%+ coins dumping >3% in 4h) — 64% WR, +3.39% 24h
        dropping = 0; total_pairs = 0
        for p2, cached in self._price_cache.items():
            if len(cached) >= 5:
                r2 = (cached[-1] - cached[-5]) / cached[-5] * 100
                total_pairs += 1
                if r2 < -3: dropping += 1
        if total_pairs >= 5:
            panic_pct = dropping / total_pairs * 100
            if panic_pct >= 90:
                score = panic_pct * 0.5
                signals.append(({
                    'pair': pair, 'tool': 'market_panic_90', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.05,
                    'reason': f"MARKET PANIC: {panic_pct:.0f}% coins down >3% — blood in streets"
                }, score))
            elif panic_pct >= 80:
                score = panic_pct * 0.4
                signals.append(({
                    'pair': pair, 'tool': 'market_panic_80', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.05,
                    'reason': f"MARKET PANIC: {panic_pct:.0f}% coins down >3%"
                }, score))
            elif panic_pct >= 70:
                score = panic_pct * 0.3
                signals.append(({
                    'pair': pair, 'tool': 'market_panic_70', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.04,
                    'reason': f"MARKET DIP: {panic_pct:.0f}% coins down >3%"
                }, score))
        
        # Tool 17: Whale Buy (5x+ volume on green candle) — +1.02% 8h
        if len(df) >= 21:
            vol = df['volume'].values.astype(float)
            opn = df['open'].values.astype(float)
            avg_vol = np.mean(vol[-21:-1]) if len(vol) >= 21 else 0
            if avg_vol > 0:
                vol_ratio = vol[-1] / avg_vol
                is_green = close[-1] > opn[-1]
                is_red = close[-1] < opn[-1]
                
                if vol_ratio >= 5 and is_green:
                    score = vol_ratio * 2
                    signals.append(({
                        'pair': pair, 'tool': 'whale_buy', 'direction': 'long',
                        'hold': 8, 'sl_pct': 0.03,
                        'reason': f"WHALE BUY: {vol_ratio:.1f}x volume on green candle"
                    }, score))
                
                # Tool 18: Capitulation (8x+ volume on red candle) — 60% WR
                if vol_ratio >= 8 and is_red:
                    score = vol_ratio * 1.5
                    signals.append(({
                        'pair': pair, 'tool': 'capitulation', 'direction': 'long',
                        'hold': 8, 'sl_pct': 0.05,
                        'reason': f"CAPITULATION: {vol_ratio:.1f}x volume on red candle — selling exhaustion"
                    }, score))
        
        # Tool 19: 7 Green Exhaustion (7 consecutive green candles → short) — 58% WR
        if len(close) >= 8 and len(df) >= 8:
            opn = df['open'].values.astype(float)
            all_green = all(close[-j-1] > opn[-j-1] for j in range(1, 8))
            cur_red = close[-1] < opn[-1]
            if all_green and cur_red:
                score = 15  # High confidence
                signals.append(({
                    'pair': pair, 'tool': 'green_exhaustion', 'direction': 'short',
                    'hold': 8, 'sl_pct': 0.03,
                    'reason': f"GREEN EXHAUSTION: 7 green then red — reversal signal"
                }, score))
        
        # Tool 20: Z-score -3σ (extreme statistical deviation) — 53% WR
        if len(close) >= 49:
            window = close[-48:]
            mu = np.mean(window); sigma = np.std(window)
            if sigma > 0:
                z = (close[-1] - mu) / sigma
                if z < -3:
                    score = abs(z) * 5
                    signals.append(({
                        'pair': pair, 'tool': 'zscore_extreme', 'direction': 'long',
                        'hold': 24, 'sl_pct': 0.05,
                        'reason': f"Z-SCORE EXTREME: {z:.1f}σ below 48h mean"
                    }, score))
        
        # Tool 21: Blood in Streets (market panic + this coin RSI<20) — 57% WR, +1.51% 24h
        if total_pairs >= 5:
            dropping_2pct = sum(1 for p2, cached in self._price_cache.items()
                              if len(cached) >= 5 and (cached[-1]-cached[-5])/cached[-5]*100 < -2)
            if dropping_2pct / total_pairs >= 0.7 and cur_rsi < 20:
                score = (20 - cur_rsi) * 3  # Very high priority
                signals.append(({
                    'pair': pair, 'tool': 'blood_in_streets', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.06,
                    'reason': f"BLOOD IN STREETS: {dropping_2pct/total_pairs*100:.0f}% panic + RSI={cur_rsi:.1f}"
                }, score))
        
        # Tool 22: FOMO Ride (80%+ coins pumping >1% in 4h) — +0.47% 8h, 2848 occurrences
        if total_pairs >= 5:
            pumping = sum(1 for p2, cached in self._price_cache.items()
                        if len(cached) >= 5 and (cached[-1]-cached[-5])/cached[-5]*100 > 1)
            if pumping / total_pairs >= 0.8:
                score = 5
                signals.append(({
                    'pair': pair, 'tool': 'fomo_ride', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.03,
                    'reason': f"FOMO RIDE: {pumping/total_pairs*100:.0f}% coins pumping — momentum"
                }, score))
        
        # ── DEEP QUANT V3: MATHEMATICAL TOOLS ──
        
        # Pre-compute math features (only if we have enough data)
        if len(close) >= 100:
            returns = np.diff(close[-100:]) / close[-100:-1]
            
            # Hurst exponent (fast variance ratio method)
            def _hurst(r, w=50):
                if len(r) < w: return 0.5
                r2 = r[-w:]
                v1 = np.var(r2)
                v2 = np.var(r2[::2]) if len(r2) >= 4 else v1
                if v1 <= 0 or v2 <= 0: return 0.5
                vr = v2 / v1
                return max(0, min(1, 0.5 + np.log(max(vr, 0.01)) / (2 * np.log(2))))
            
            # Shannon entropy
            def _entropy(r, bins=15):
                if len(r) < 10: return 3.0
                hist, _ = np.histogram(r, bins=bins, density=True)
                hist = hist[hist > 0]
                if len(hist) == 0: return 3.0
                probs = hist / hist.sum()
                return -np.sum(probs * np.log2(probs))
            
            # Autocorrelation
            def _ac(r, lag=1):
                if len(r) < lag + 5: return 0
                return float(pd.Series(r).autocorr(lag=lag))
            
            # VPIN proxy
            def _vpin(c_arr, v_arr, w=20):
                if len(c_arr) < w + 1: return 0
                rets = np.diff(c_arr) / c_arr[:-1]
                bv = np.where(rets > 0, v_arr[1:], 0)
                sv = np.where(rets < 0, v_arr[1:], 0)
                rb = np.sum(bv[-w:]); rs = np.sum(sv[-w:])
                t = rb + rs
                return abs(rb - rs) / t if t > 0 else 0
            
            H = _hurst(returns)
            ent = _entropy(returns[-30:])
            ac1 = _ac(returns, 1)
            vp = _vpin(close[-30:], df['volume'].values[-30:].astype(float))
            
            # Tool 23: Crash + Negative Autocorrelation — THE BEST EDGE (78% WR, +3.07% 8h)
            if ret_24h < -10 and ac1 < -0.05:
                score = abs(ret_24h) * (abs(ac1) + 0.1) * 10  # Very high priority
                signals.append(({
                    'pair': pair, 'tool': 'crash_neg_ac', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.08,
                    'reason': f"CRASH+NEG_AC: {ret_24h:.1f}% drop, AC1={ac1:.3f} — 78% WR, +3%/8h"
                }, score))
            
            # Tool 24: Crash + Mean Reverting Hurst — 62% WR, +1.35% 8h
            if ret_24h < -8 and H < 0.45:
                score = abs(ret_24h) * (0.5 - H) * 8
                signals.append(({
                    'pair': pair, 'tool': 'crash_mean_revert', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.06,
                    'reason': f"CRASH+HURST: {ret_24h:.1f}% drop, H={H:.3f} (mean-reverting) — 62% WR"
                }, score))
            
            # Tool 25: Strong Trend Follow (Hurst > 0.65 + pump) — 46% WR, +0.87% 8h
            if H > 0.65 and ret_4h > 2:
                score = H * ret_4h * 3
                signals.append(({
                    'pair': pair, 'tool': 'hurst_trend', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.03,
                    'reason': f"HURST TREND: H={H:.3f} (trending), +{ret_4h:.1f}% 4h"
                }, score))
            
            # Tool 26: VPIN Toxic Flow Buy (VPIN > 0.7 + red candle) — 59% WR, +0.84% 8h
            if vp > 0.7 and close[-1] < df['open'].values[-1]:
                score = vp * 10
                signals.append(({
                    'pair': pair, 'tool': 'vpin_toxic', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.04,
                    'reason': f"VPIN TOXIC: VPIN={vp:.3f} + red candle — informed selling exhausted"
                }, score))
            
            # Tool 27: VPIN Dip (>5% drop + VPIN > 0.5) — 59% WR, +0.77% 8h
            if ret_8h < -5 and vp > 0.5:
                score = abs(ret_8h) * vp * 3
                signals.append(({
                    'pair': pair, 'tool': 'vpin_dip', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.05,
                    'reason': f"VPIN DIP: {ret_8h:.1f}% drop 8h, VPIN={vp:.3f}"
                }, score))
            
            # Tool 28: Predictable Dip (low entropy + dip) — 54% WR, +0.77% 8h
            if ent < 2.5 and ret_4h < -2:
                score = (3.0 - ent) * abs(ret_4h) * 2
                signals.append(({
                    'pair': pair, 'tool': 'entropy_dip', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.03,
                    'reason': f"PREDICTABLE DIP: entropy={ent:.2f} (low), {ret_4h:.1f}% dip — 54% WR"
                }, score))
            
            # Tool 29: Drop + Low Entropy + VPIN (triple math confirmation) — strong combo
            if ret_8h < -5 and ent < 2.5 and vp > 0.5:
                score = abs(ret_8h) * (3.0 - ent) * vp * 5
                signals.append(({
                    'pair': pair, 'tool': 'triple_math', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.06,
                    'reason': f"TRIPLE MATH: {ret_8h:.1f}% drop, ent={ent:.2f}, VPIN={vp:.3f}"
                }, score))
        
        return signals
        
    def update_grids(self, market_data: dict):
        """Update grid engine (Tool 1) - runs continuously."""
        grid_balance_per_pair = self.grid_balance / len(PAIRS)
        
        for pair in PAIRS:
            if pair not in market_data or pair not in GRID_CONFIGS:
                continue
                
            data = market_data[pair]
            current_price = data["price"]
            current_low = data.get("low", current_price)
            current_high = data.get("high", current_price)
            grid_pct = GRID_CONFIGS[pair]
            
            # Initialize grid if needed
            if pair not in self.grid_positions:
                self.grid_positions[pair] = []
                
            positions = self.grid_positions[pair]
            
            # SMA50 filter: Only run grid when price is above SMA50 (uptrend)
            df = data.get('df')
            if df is not None and len(df) >= 50:
                close_arr = df['close'].values
                sma50 = np.mean(close_arr[-50:]) if len(close_arr) >= 50 else current_price
                
                if current_price < sma50:
                    # Close all grid inventory for this pair (sell at market)
                    if positions and ENABLE_LIVE_TRADING:
                        for pos in positions:
                            self._place_order(pair, "sell", pos["qty"], current_price, "market")
                            pnl = (current_price - pos["buy_price"]) * pos["qty"] * 0.998  # Minus fees
                            self.grid_profit += pnl
                            logger.info(f"[GRID SMA50] {pair} closed @ ${current_price:.4f}, PnL: ${pnl:.2f}")
                    
                    # Clear positions and skip new orders
                    self.grid_positions[pair] = []
                    continue
            
            # Check for new buy fills (3 levels below price)
            for level in range(1, 4):  # 3 levels
                buy_price = current_price * (1 - grid_pct * level)
                qty = (grid_balance_per_pair / 3) / buy_price
                
                # Check if low touched this level
                if current_low <= buy_price:
                    # Check if we don't already have a position at this level
                    level_exists = any(abs(pos["buy_price"] - buy_price) / buy_price < 0.005 
                                     for pos in positions)
                    
                    if not level_exists:
                        positions.append({
                            "buy_price": buy_price,
                            "qty": qty,
                            "bar": self.current_bar
                        })
                        
                        if ENABLE_LIVE_TRADING:
                            self._place_order(pair, "buy", qty, buy_price, "limit")
                        
                        logger.info(f"[GRID] {pair} buy filled @ ${buy_price:.4f}, qty: {qty:.6f}")
            
            # Check for sell fills (take profit)
            remaining_positions = []
            for pos in positions:
                sell_target = pos["buy_price"] * (1 + GRID_TAKE_PROFIT)
                
                if current_high >= sell_target:
                    # Sell filled - book profit
                    pnl = (sell_target - pos["buy_price"]) * pos["qty"] * 0.998  # Minus fees
                    self.grid_profit += pnl
                    self.grid_round_trips += 1
                    
                    if ENABLE_LIVE_TRADING:
                        self._place_order(pair, "sell", pos["qty"], sell_target, "market")
                    
                    logger.info(f"[GRID] {pair} round-trip: ${pnl:.2f} profit")
                else:
                    remaining_positions.append(pos)
            
            self.grid_positions[pair] = remaining_positions
            
    def get_bars_held(self, position: dict) -> int:
        """Calculate how many bars a position has been held."""
        return self.current_bar - position.get("entry_bar", self.current_bar)
        
    def manage_positions(self, market_data: dict):
        """Check all active positions for exits."""
        for pair in list(self.active_positions.keys()):
            if pair not in market_data:
                continue
                
            pos = self.active_positions[pair]
            data = market_data[pair]
            price = data["price"]
            bars_held = self.get_bars_held(pos)
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            # Stop loss
            if pos['direction'] == 'long' and price <= pos['entry'] * (1 - pos['sl_pct']):
                should_close = True
                close_reason = "STOP LOSS"
            elif pos['direction'] == 'short' and price >= pos['entry'] * (1 + pos['sl_pct']):
                should_close = True
                close_reason = "STOP LOSS"
            # Hold timeout
            elif bars_held >= pos['hold']:
                should_close = True
                close_reason = "HOLD COMPLETE"
            
            if should_close:
                self.close_position(pair, price, close_reason)
                
    def execute_signal(self, signal: dict, score: float):
        """Execute a signal by opening a position."""
        pair = signal['pair']
        direction = signal['direction']
        tool = signal['tool']
        
        # Calculate position size
        risk_amount = self.active_balance * RISK_PER_TRADE
        sl_pct = signal['sl_pct']
        
        # For margin shorts, use leverage=2
        leverage = 2 if direction == 'short' else 1
        
        # Get current price from market data (passed separately)
        try:
            market_data = self.get_market_data()  # Quick refresh for current price
            if pair not in market_data:
                logger.warning(f"No market data for {pair}, skipping signal")
                return
                
            current_price = market_data[pair]["price"]
            qty = risk_amount / (current_price * sl_pct)
            
            # Apply leverage for shorts
            if direction == 'short':
                qty *= leverage
            
            # Check minimum order size
            min_size = self.client.get_min_order_volume(pair) if ENABLE_LIVE_TRADING else 0.001
            if qty < min_size:
                logger.warning(f"Position size {qty:.6f} below minimum {min_size} for {pair}")
                return
            
            # Place order
            if ENABLE_LIVE_TRADING:
                side = "buy" if direction == "long" else "sell"
                success = self._place_order(pair, side, qty, current_price, "market", leverage if direction == 'short' else None)
                if not success:
                    return
            
            # Record position
            self.active_positions[pair] = {
                'tool': tool,
                'direction': direction,
                'entry': current_price,
                'qty': qty,
                'entry_bar': self.current_bar,
                'sl_pct': sl_pct,
                'hold': signal['hold'],
                'reason': signal['reason']
            }
            
            logger.info(f"[EXECUTE] {direction.upper()} {qty:.6f} {pair} @ ${current_price:.4f} ({tool}, hold {signal['hold']}h, SL {sl_pct*100:.0f}%)")
            logger.info(f"[REASON] {signal['reason']}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            
    def close_position(self, pair: str, price: float, reason: str):
        """Close an active position."""
        if pair not in self.active_positions:
            return
            
        pos = self.active_positions[pair]
        tool = pos['tool']
        direction = pos['direction']
        entry_price = pos['entry']
        qty = pos['qty']
        
        # Calculate P&L
        if direction == 'long':
            pnl = (price - entry_price) * qty
        else:  # short
            pnl = (entry_price - price) * qty
            
        pnl *= 0.998  # Account for fees/slippage
        
        # Place closing order
        if ENABLE_LIVE_TRADING:
            close_side = "sell" if direction == "long" else "buy"
            leverage = 2 if direction == 'short' else None
            self._place_order(pair, close_side, qty, price, "market", leverage)
        
        # Update stats
        self.active_profit += pnl
        self.tool_stats[tool]["trades"] += 1
        self.tool_stats[tool]["pnl"] += pnl
        if pnl > 0:
            self.tool_stats[tool]["wins"] += 1
            
        # Record trade
        trade_record = {
            "pair": pair,
            "tool": tool,
            "direction": direction,
            "entry": entry_price,
            "exit": price,
            "qty": qty,
            "pnl": pnl,
            "reason": reason,
            "bars_held": self.get_bars_held(pos),
            "timestamp": datetime.now().isoformat()
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"[CLOSE] {pair} {direction} {tool}: ${pnl:.2f} ({reason})")
        
        del self.active_positions[pair]
        
    def _place_order(self, symbol: str, side: str, quantity: float, price: float, 
                    order_type: str = "market", leverage: Optional[int] = None) -> bool:
        """Place an order via Kraken."""
        try:
            result = self.client.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price if order_type == "limit" else None,
                leverage=leverage
            )
            
            if result.get("error"):
                logger.error(f"Order failed: {result['error']}")
                return False
                
            logger.debug(f"[ORDER] {symbol} {side} {quantity:.6f} @ ${price:.4f} ({order_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
            
    def log_status(self):
        """Log comprehensive status."""
        # Count signals found this cycle
        all_signals = []
        market_data = self.get_market_data()
        for pair, data in market_data.items():
            signals = self.scan_signals(pair, data)
            all_signals.extend(signals)
        
        # Sort by score for display
        all_signals.sort(key=lambda x: x[1], reverse=True)
        
        # Log top signals
        if all_signals:
            logger.info(f"[SCAN] {len(market_data)} pairs scanned, {len(all_signals)} signals found:")
            for i, (signal, score) in enumerate(all_signals[:3]):  # Top 3
                logger.info(f"  #{i+1} {signal['pair']} {signal['tool'].upper()} score={score:.1f} ({signal['reason']})")
        else:
            logger.info(f"[SCAN] {len(market_data)} pairs scanned, 0 signals found")
        
        # Grid status
        total_grid_positions = sum(len(positions) for positions in self.grid_positions.values())
        active_grid_pairs = len([p for p, pos in self.grid_positions.items() if pos])
        logger.info(f"[GRID] {self.grid_round_trips} total RTs, ${self.grid_profit:.2f} realized | Active grids: {active_grid_pairs} pairs")
        
        # Active positions
        if self.active_positions:
            pos_summary = []
            for pair, pos in self.active_positions.items():
                if pair in market_data:
                    current_price = market_data[pair]["price"]
                    if pos['direction'] == 'long':
                        pnl_pct = (current_price - pos['entry']) / pos['entry'] * 100
                    else:
                        pnl_pct = (pos['entry'] - current_price) / pos['entry'] * 100
                    pos_summary.append(f"{pair} {pos['direction']} {pnl_pct:+.1f}%")
            logger.info(f"[POSITIONS] {len(self.active_positions)} active | {' | '.join(pos_summary)}")
        else:
            logger.info("[POSITIONS] 0 active positions")
        
        # P&L summary
        total_profit = self.grid_profit + self.active_profit
        total_return = (total_profit / STARTING_BALANCE) * 100
        logger.info(f"[P&L] Grid ${self.grid_profit:.2f} | Active ${self.active_profit:.2f} | Total ${total_profit:.2f} ({total_return:+.1f}%)")
        
        # Tool stats
        tool_summary = []
        for tool, stats in self.tool_stats.items():
            if stats["trades"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                tool_summary.append(f"{tool}: {stats['trades']}T {stats['wins']}W ${stats['pnl']:+.0f}")
        if tool_summary:
            logger.info(f"[TOOL STATS] {' | '.join(tool_summary)}")
            
    def run_cycle(self):
        """Run one complete trading cycle - THE BRAIN."""
        try:
            logger.info("=" * 60)
            self.current_bar += 1
            
            # 1. Fetch 1h data for all 16 pairs
            market_data = self.get_market_data()
            if not market_data:
                logger.warning("No market data, skipping cycle")
                return
            
            # 2. Update grid engine (passive, always running)
            self.update_grids(market_data)
            
            # 3. Manage existing active positions (check exits)
            self.manage_positions(market_data)
            
            # 4. Scan ALL signal sources across ALL pairs
            all_signals = []
            for pair, data in market_data.items():
                signals = self.scan_signals(pair, data)  # Returns list of (signal_dict, score)
                all_signals.extend(signals)
            
            # 5. Rank by score (highest first)
            all_signals.sort(key=lambda x: x[1], reverse=True)
            
            # 6. Execute top signals (limited by available positions and capital)
            for signal, score in all_signals:
                if len(self.active_positions) >= MAX_ACTIVE_POSITIONS:
                    break
                if signal['pair'] in self.active_positions:
                    continue  # 1 position per pair
                self.execute_signal(signal, score)
            
            # 7. Log everything
            self.log_status()
            
            # 8. Save state
            self.save_state()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def run(self):
        """Main bot loop."""
        logger.info("🔥 THE ALL-SEEING EYE AWAKENS...")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        logger.info(f"Max active positions: {MAX_ACTIVE_POSITIONS}")
        logger.info(f"Risk per trade: {RISK_PER_TRADE*100:.0f}% of active balance")
        
        # Signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received, stopping bot...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                cycle_start = time.time()
                
                self.run_cycle()
                
                # Sleep for remaining interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, CHECK_INTERVAL - cycle_time)
                
                if sleep_time > 0:
                    logger.info(f"Cycle took {cycle_time:.1f}s, sleeping {sleep_time:.1f}s")
                    logger.info("-" * 60)
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Cycle took {cycle_time:.1f}s, longer than {CHECK_INTERVAL}s interval!")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info("Saving final state...")
            self.save_state()
            logger.info("🔥 THE ALL-SEEING EYE SLEEPS...")


if __name__ == "__main__":
    bot = AllSeeingEye()
    bot.run()