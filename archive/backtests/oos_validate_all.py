#!/usr/bin/env python3
"""
Out-of-Sample Validation for ALL Trading Tools
Tests each tool from run_master_bot.py scan_signals() against real Binance 4h data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

class OOSValidator:
    def __init__(self, data_dir="data/binance_real"):
        self.data_dir = data_dir
        self.pairs = []
        self.data_cache = {}
        self.results = []
        
        # Load all available pairs
        for f in os.listdir(data_dir):
            if f.endswith('_4h.csv'):
                pair = f.replace('_4h.csv', '')
                self.pairs.append(pair)
                df = pd.read_csv(f"{data_dir}/{f}")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data_cache[pair] = df
                
        print(f"Loaded {len(self.pairs)} pairs: {', '.join(self.pairs)}")
        
    def calc_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI exactly like the master bot."""
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
    
    def calc_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        k = 2.0 / (period + 1)
        for i in range(period, len(prices)):
            ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
        return ema
        
    def calc_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float):
        """Calculate Bollinger Bands."""
        sma = self.calc_sma(prices, period)
        std = np.full(len(prices), np.nan)
        
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
            
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return sma, upper, lower
    
    def is_thursday(self, timestamp):
        """Check if timestamp is Thursday (day 3, 0=Monday)."""
        if hasattr(timestamp, 'weekday'):
            return timestamp.weekday() == 3
        else:
            # Convert numpy.datetime64 to pandas timestamp
            ts = pd.to_datetime(timestamp)
            return ts.weekday() == 3
        
    def is_sunday(self, timestamp):
        """Check if timestamp is Sunday (day 6)."""
        if hasattr(timestamp, 'weekday'):
            return timestamp.weekday() == 6
        else:
            ts = pd.to_datetime(timestamp)
            return ts.weekday() == 6
        
    def is_month_start(self, timestamp):
        """Check if timestamp is within first 3 days of month."""
        if hasattr(timestamp, 'day'):
            return timestamp.day <= 3
        else:
            ts = pd.to_datetime(timestamp)
            return ts.day <= 3
        
    def is_late_us_hours(self, timestamp):
        """Check if timestamp is late US hours (20:00-23:59 EST)."""
        # Convert to EST and check hour
        # For simplicity, assume UTC timestamps and convert
        if hasattr(timestamp, 'hour'):
            hour_utc = timestamp.hour
        else:
            ts = pd.to_datetime(timestamp)
            hour_utc = ts.hour
        # EST is UTC-5, so 20:00 EST = 01:00 UTC next day
        # This is approximate - real implementation would handle timezones properly
        return hour_utc >= 1 and hour_utc <= 4
        
    def test_single_pair_tools(self, pair):
        """Test all single-pair tools on one pair."""
        df = self.data_cache[pair]
        results = []
        
        # Pre-compute indicators for entire series
        close = df['close'].values
        high = df['high'].values  
        low = df['low'].values
        volume = df['volume'].values
        timestamps = df['timestamp'].values
        
        rsi7 = self.calc_rsi(close, 7)
        rsi14 = self.calc_rsi(close, 14)  # Some tools might use 14
        sma50 = self.calc_sma(close, 50)
        sma20 = self.calc_sma(close, 20)
        sma12 = self.calc_sma(close, 12)
        ema21 = self.calc_ema(close, 21)
        atr14 = self.calc_atr(high, low, close, 14)
        
        # Bollinger Bands
        bb_sma, bb_upper, bb_lower = self.calc_bollinger_bands(close, 20, 2.0)
        bb_squeeze_sma15, bb_squeeze_upper15, bb_squeeze_lower15 = self.calc_bollinger_bands(close, 15, 1.5)
        bb_squeeze_sma30, bb_squeeze_upper30, bb_squeeze_lower30 = self.calc_bollinger_bands(close, 30, 1.5)
        
        # Volume moving average
        vol_sma20 = self.calc_sma(volume, 20)
        
        # Walk through data starting at bar 100 (enough lookback)
        start_idx = max(100, 50)  # Need at least 50 for SMA50
        
        for i in range(start_idx, len(close) - 6):  # Need 6 bars forward for 24h return
            price = close[i]
            cur_rsi = rsi7[i]
            cur_rsi14 = rsi14[i]
            cur_atr_pct = atr14[i] / price * 100 if price > 0 and not np.isnan(atr14[i]) else 0
            cur_vs_sma50 = (price - sma50[i]) / sma50[i] * 100 if not np.isnan(sma50[i]) and sma50[i] > 0 else 0
            
            # Calculate returns (on 4h data, these are the correct lookbacks)
            ret_4h = (price - close[i-1]) / close[i-1] * 100 if i >= 1 and close[i-1] > 0 else 0
            ret_8h = (price - close[i-2]) / close[i-2] * 100 if i >= 2 and close[i-2] > 0 else 0  
            ret_12h = (price - close[i-3]) / close[i-3] * 100 if i >= 3 and close[i-3] > 0 else 0
            ret_24h = (price - close[i-6]) / close[i-6] * 100 if i >= 6 and close[i-6] > 0 else 0
            
            # Forward returns for validation (8h = 2 bars, 24h = 6 bars)
            fwd_8h = (close[i+2] - price) / price * 100 if i+2 < len(close) else 0
            fwd_24h = (close[i+6] - price) / price * 100 if i+6 < len(close) else 0
            
            # Volume ratio
            vol_ratio = volume[i] / vol_sma20[i] if not np.isnan(vol_sma20[i]) and vol_sma20[i] > 0 else 1
            
            # Current timestamp
            ts = pd.to_datetime(timestamps[i])
            
            # Higher timeframe trend (simplified)
            if i >= 16:
                sma12_curr = np.mean(close[i-11:i+1])
                sma12_prev = np.mean(close[i-15:i-3])
                higher_tf_bullish = sma12_curr > sma12_prev
                higher_tf_bearish = sma12_curr < sma12_prev
            else:
                higher_tf_bullish = False
                higher_tf_bearish = False
            
            # ===== TEST INDIVIDUAL TOOLS =====
            
            # Tool 2: Crash Buy
            if ret_24h < -10 and cur_rsi < 20:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'crash_buy', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"ret_24h={ret_24h:.1f}, rsi={cur_rsi:.1f}"
                })
            
            # Tool 3: Volatile Oversold  
            if cur_atr_pct > 3 and cur_rsi < 25:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'volatile_oversold', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"atr_pct={cur_atr_pct:.1f}, rsi={cur_rsi:.1f}"
                })
            
            # Tool 4: Relief Rally
            if cur_rsi > 75 and cur_vs_sma50 < 0:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'relief_rally', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"rsi={cur_rsi:.1f}, vs_sma50={cur_vs_sma50:.1f}"
                })
            
            # Tool 6: Dip Buy
            if ret_4h < -3:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'dip_buy', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"ret_4h={ret_4h:.1f}"
                })
            
            # Tool 7a: Mega Pump Sell (RSI > 80 + pump in 12h)
            if cur_rsi > 80 and i >= 3:
                ret_12h_pump = (price - close[i-3]) / close[i-3] * 100 if close[i-3] > 0 else 0
                if ret_12h_pump >= 10:  # Tier 1
                    win_8h = 1 if fwd_8h < 0 else 0  # SHORT signal
                    win_24h = 1 if fwd_24h < 0 else 0
                    results.append({
                        'tool': 'mega_pump_sell_t1', 'pair': pair, 'direction': 'short',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"rsi={cur_rsi:.1f}, pump_12h={ret_12h_pump:.1f}"
                    })
                elif ret_12h_pump >= 8:  # Tier 2
                    win_8h = 1 if fwd_8h < 0 else 0
                    win_24h = 1 if fwd_24h < 0 else 0
                    results.append({
                        'tool': 'mega_pump_sell_t2', 'pair': pair, 'direction': 'short',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"rsi={cur_rsi:.1f}, pump_12h={ret_12h_pump:.1f}"
                    })
            
            # Tool 8: Mega Crash (>15% drop 24h)
            if ret_24h < -15:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'mega_crash', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"ret_24h={ret_24h:.1f}"
                })
            
            # Tool 9: Flash Crash (>10% in 12h)
            if ret_12h < -10:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'flash_crash', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"ret_12h={ret_12h:.1f}"
                })
            
            # Tool 10: Quick Crash (>10% in 8h)
            if ret_8h < -10:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'quick_crash', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"ret_8h={ret_8h:.1f}"
                })
            
            # Tool 11: Deep Dip (8-10% drops)
            for tf_name, ret_val in [("8h", ret_8h), ("12h", ret_12h), ("24h", ret_24h)]:
                if ret_val < -8 and ret_val >= -10:  # 8-10% drop
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': f'deep_dip_{tf_name}', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"ret_{tf_name}={ret_val:.1f}"
                    })
            
            # Tool 12: Quick Dip (>5% in 4h)
            if ret_4h < -5:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'quick_dip', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"ret_4h={ret_4h:.1f}"
                })
            
            # Tool 14: RSI Divergence (simplified - price makes new low, RSI doesn't)
            if i >= 20:
                lookback = 10
                price_low = np.min(close[i-lookback:i+1]) == close[i]
                rsi_low = np.min(rsi7[i-lookback:i+1]) == rsi7[i]
                if price_low and not rsi_low and cur_rsi < 40:
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'rsi_divergence', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"price_low={price_low}, rsi_low={rsi_low}, rsi={cur_rsi:.1f}"
                    })
            
            # Tool 15: Thursday Short
            if self.is_thursday(ts):
                win_8h = 1 if fwd_8h < 0 else 0
                win_24h = 1 if fwd_24h < 0 else 0
                results.append({
                    'tool': 'thursday_short', 'pair': pair, 'direction': 'short',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"thursday={ts.strftime('%A')}"
                })
            
            # Tool 17: Whale Buy (high volume + price drop)
            if vol_ratio > 2.5 and ret_4h < -2:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'whale_buy', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"vol_ratio={vol_ratio:.1f}, ret_4h={ret_4h:.1f}"
                })
            
            # Tool 18: Capitulation (extreme oversold)
            if cur_rsi < 15:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'capitulation', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"rsi={cur_rsi:.1f}"
                })
            
            # Tool 19: Green Exhaustion (extreme overbought)
            if cur_rsi > 85:
                win_8h = 1 if fwd_8h < 0 else 0
                win_24h = 1 if fwd_24h < 0 else 0
                results.append({
                    'tool': 'green_exhaustion', 'pair': pair, 'direction': 'short',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"rsi={cur_rsi:.1f}"
                })
            
            # Tool 20: Z-Score Extreme (simplified - use 20-period rolling z-score)
            if i >= 20:
                window = close[i-19:i+1]
                z_score = (price - np.mean(window)) / np.std(window) if np.std(window) > 0 else 0
                if z_score < -2.5:  # Extreme undervaluation
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'zscore_extreme', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"z_score={z_score:.2f}"
                    })
            
            # Tool 30: Panic Close (gap down + high volume)
            if i >= 1:
                gap_down = (close[i] - close[i-1]) / close[i-1] * 100
                if gap_down < -5 and vol_ratio > 2:
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'panic_close', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"gap={gap_down:.1f}, vol_ratio={vol_ratio:.1f}"
                    })
            
            # Tool 39: Volume Climax (extreme volume + price move)
            if vol_ratio > 4 and abs(ret_4h) > 5:
                direction = 'long' if ret_4h < 0 else 'short'  # Contrarian
                if direction == 'long':
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                else:
                    win_8h = 1 if fwd_8h < 0 else 0
                    win_24h = 1 if fwd_24h < 0 else 0
                results.append({
                    'tool': 'volume_climax', 'pair': pair, 'direction': direction,
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"vol_ratio={vol_ratio:.1f}, ret_4h={ret_4h:.1f}"
                })
            
            # Tool 42: Breakout Detect (price > SMA50 after being below)
            if i >= 1 and not np.isnan(sma50[i]) and not np.isnan(sma50[i-1]):
                prev_below = close[i-1] < sma50[i-1]
                now_above = close[i] > sma50[i]
                if prev_below and now_above:
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'breakout_detect', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"prev_below={prev_below}, now_above={now_above}"
                    })
            
            # Tool 42b: High Breakout 50 (new 50-bar high)
            if i >= 50:
                is_high = close[i] == np.max(close[i-49:i+1])
                if is_high:
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'high_breakout_50', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"50_bar_high"
                    })
            
            # Tool 46: Distribution Short (high volume at resistance)
            if i >= 20 and not np.isnan(sma50[i]):
                resistance = np.max(close[i-19:i])
                near_resistance = abs(price - resistance) / resistance < 0.02
                if near_resistance and vol_ratio > 2 and cur_vs_sma50 > 5:
                    win_8h = 1 if fwd_8h < 0 else 0
                    win_24h = 1 if fwd_24h < 0 else 0
                    results.append({
                        'tool': 'distribution_short', 'pair': pair, 'direction': 'short',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"near_resistance={near_resistance}, vol_ratio={vol_ratio:.1f}"
                    })
            
            # Tool 47: SMA50 Ext Tiers (distance from SMA50)
            if not np.isnan(sma50[i]):
                if cur_vs_sma50 < -15:  # Tier 1: >15% below SMA50
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'sma50_ext_t1', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"vs_sma50={cur_vs_sma50:.1f}"
                    })
                elif cur_vs_sma50 < -10:  # Tier 2: 10-15% below
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'sma50_ext_t2', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"vs_sma50={cur_vs_sma50:.1f}"
                    })
            
            # Tool 48: EMA Cross Short (EMA21 crosses below SMA50)
            if i >= 1 and not np.isnan(ema21[i]) and not np.isnan(sma50[i]) and not np.isnan(ema21[i-1]) and not np.isnan(sma50[i-1]):
                prev_above = ema21[i-1] > sma50[i-1]
                now_below = ema21[i] < sma50[i]
                if prev_above and now_below:
                    win_8h = 1 if fwd_8h < 0 else 0
                    win_24h = 1 if fwd_24h < 0 else 0
                    results.append({
                        'tool': 'ema_cross_short', 'pair': pair, 'direction': 'short',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"ema21_cross_below_sma50"
                    })
            
            # Tool 49: Month Start Short
            if self.is_month_start(ts):
                win_8h = 1 if fwd_8h < 0 else 0
                win_24h = 1 if fwd_24h < 0 else 0
                results.append({
                    'tool': 'month_start_short', 'pair': pair, 'direction': 'short',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"month_start_day={ts.day}"
                })
            
            # Tool 50: Sunday Short
            if self.is_sunday(ts):
                win_8h = 1 if fwd_8h < 0 else 0
                win_24h = 1 if fwd_24h < 0 else 0
                results.append({
                    'tool': 'sunday_short', 'pair': pair, 'direction': 'short',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"sunday={ts.strftime('%A')}"
                })
            
            # Tool 51: BB Above Long (price above upper Bollinger Band)
            if not np.isnan(bb_upper[i]) and close[i] > bb_upper[i]:
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'bb_above_long', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"price={price:.2f} > bb_upper={bb_upper[i]:.2f}"
                })
            
            # Tool 52: High Breakout 30 (new 30-bar high)
            if i >= 30:
                is_high_30 = close[i] == np.max(close[i-29:i+1])
                if is_high_30:
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'high_breakout_30', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"30_bar_high"
                    })
            
            # Tool 53: BB Squeeze (Bollinger Bands contracting)
            # Simplified: when BB width is at minimum over lookback period
            if i >= 20 and not np.isnan(bb_upper[i]) and not np.isnan(bb_lower[i]):
                bb_width = (bb_upper[i] - bb_lower[i]) / bb_sma[i] if bb_sma[i] > 0 else 0
                min_width = np.min([(bb_upper[j] - bb_lower[j]) / bb_sma[j] for j in range(i-19, i+1) 
                                  if not np.isnan(bb_upper[j]) and not np.isnan(bb_lower[j]) and bb_sma[j] > 0])
                if bb_width == min_width and bb_width < 0.05:  # 5% width threshold
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'bb_squeeze', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"bb_width={bb_width:.4f}"
                    })
            
            # Tool 55: Late US Short (simplified time-based)
            if self.is_late_us_hours(ts):
                win_8h = 1 if fwd_8h < 0 else 0
                win_24h = 1 if fwd_24h < 0 else 0
                results.append({
                    'tool': 'late_us_short', 'pair': pair, 'direction': 'short',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"late_us_hour={ts.hour}"
                })
            
            # Tool 56: High Breakout 50 NV (no volume filter)
            if i >= 50:
                is_high_50_nv = close[i] == np.max(close[i-49:i+1])
                if is_high_50_nv:
                    win_8h = 1 if fwd_8h > 0 else 0
                    win_24h = 1 if fwd_24h > 0 else 0
                    results.append({
                        'tool': 'high_breakout_50_nv', 'pair': pair, 'direction': 'long',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"50_bar_high_nv"
                    })
            
            # Tool 57: Falling Wedge Short (simplified: declining highs + RSI oversold)
            if i >= 10 and cur_rsi < 30:
                recent_highs = high[i-9:i+1]
                if len(recent_highs) >= 3:
                    # Check if highs are generally declining
                    highs_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                    if highs_trend < 0:  # Declining trend
                        win_8h = 1 if fwd_8h < 0 else 0
                        win_24h = 1 if fwd_24h < 0 else 0
                        results.append({
                            'tool': 'falling_wedge_short', 'pair': pair, 'direction': 'short',
                            'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                            'win_8h': win_8h, 'win_24h': win_24h,
                            'trigger': f"highs_trend={highs_trend:.4f}, rsi={cur_rsi:.1f}"
                        })
        
        return results
        
    def test_all_pairs(self):
        """Test all single-pair tools across all pairs."""
        all_results = []
        
        for pair in self.pairs:
            print(f"Testing {pair}...")
            pair_results = self.test_single_pair_tools(pair)
            all_results.extend(pair_results)
            
        return all_results
        
    def analyze_results(self, results):
        """Analyze and summarize results by tool."""
        if not results:
            print("No signals found!")
            return []
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Group by tool
        summary = []
        for tool in df['tool'].unique():
            tool_df = df[df['tool'] == tool]
            
            signal_count = len(tool_df)
            win_rate_8h = tool_df['win_8h'].mean() * 100
            win_rate_24h = tool_df['win_24h'].mean() * 100
            avg_fwd_8h = tool_df['fwd_8h'].mean()
            avg_fwd_24h = tool_df['fwd_24h'].mean()
            
            # Net expected value (simplified)
            net_8h = avg_fwd_8h
            net_24h = avg_fwd_24h
            
            # Determine if tool passes criteria
            direction = tool_df['direction'].iloc[0]
            if direction == 'long':
                passes_8h = net_8h > 0 and win_rate_8h > 50 and signal_count >= 10
                passes_24h = net_24h > 0 and win_rate_24h > 50 and signal_count >= 10
            else:  # short
                passes_8h = net_8h < 0 and win_rate_8h > 45 and signal_count >= 10  # Note: negative return is good for shorts
                passes_24h = net_24h < 0 and win_rate_24h > 45 and signal_count >= 10
                
            status_8h = 'PASS' if passes_8h else ('MARGINAL' if (net_8h > 0 if direction == 'long' else net_8h < 0) and signal_count >= 5 else 'FAIL')
            status_24h = 'PASS' if passes_24h else ('MARGINAL' if (net_24h > 0 if direction == 'long' else net_24h < 0) and signal_count >= 5 else 'FAIL')
            
            summary.append({
                'tool': tool,
                'direction': direction,
                'signals': signal_count,
                'win_rate_8h': win_rate_8h,
                'win_rate_24h': win_rate_24h,
                'avg_fwd_8h': avg_fwd_8h,
                'avg_fwd_24h': avg_fwd_24h,
                'net_8h': net_8h,
                'net_24h': net_24h,
                'status_8h': status_8h,
                'status_24h': status_24h
            })
            
        # Sort by net 24h return (absolute value for shorts)
        summary_df = pd.DataFrame(summary)
        summary_df['sort_key'] = summary_df.apply(lambda x: x['net_24h'] if x['direction'] == 'long' else -x['net_24h'], axis=1)
        summary_df = summary_df.sort_values('sort_key', ascending=False)
        
        return summary_df
        
    def run_validation(self):
        """Run complete validation."""
        print("Starting Out-of-Sample Validation...")
        print("=" * 60)
        
        # Test all single-pair tools
        results = self.test_all_pairs()
        
        print(f"\nFound {len(results)} total signals across all tools and pairs")
        
        # Analyze results
        summary = self.analyze_results(results)
        
        # Print summary table
        print("\nVALIDATION RESULTS SUMMARY")
        print("=" * 100)
        print(f"{'Tool':<25} {'Dir':<5} {'Sigs':<6} {'WR_8h':<7} {'WR_24h':<8} {'Avg_8h':<8} {'Avg_24h':<9} {'Net_8h':<8} {'Net_24h':<9} {'Status_8h':<10} {'Status_24h':<10}")
        print("-" * 100)
        
        for _, row in summary.iterrows():
            print(f"{row['tool']:<25} {row['direction']:<5} {row['signals']:<6} "
                  f"{row['win_rate_8h']:<7.1f} {row['win_rate_24h']:<8.1f} "
                  f"{row['avg_fwd_8h']:<8.2f} {row['avg_fwd_24h']:<9.2f} "
                  f"{row['net_8h']:<8.2f} {row['net_24h']:<9.2f} "
                  f"{row['status_8h']:<10} {row['status_24h']:<10}")
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv('data/oos_validation_detailed.csv', index=False)
        summary.to_csv('data/oos_validation_summary.csv', index=False)
        
        print(f"\nDetailed results saved to: data/oos_validation_detailed.csv")
        print(f"Summary saved to: data/oos_validation_summary.csv")
        
        return results, summary

if __name__ == "__main__":
    validator = OOSValidator()
    results, summary = validator.run_validation()