#!/usr/bin/env python3
"""
BULL/GREED + BREAKOUT/MOMENTUM Tools Testing Framework

Tests all specified trading tools on real 1h Binance data with walk-forward validation:
- In-sample: bars 0-4380  
- Out-of-sample: bars 4380-8760
- Fee-adjusted returns: -0.52% round-trip cost
- Forward returns at +8 bars (8h) and +24 bars (24h)

EXACT REPLICATION of run_master_bot.py signal logic with efficient numpy operations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BullMomentumTester:
    def __init__(self, data_dir: str = "data/binance_1h"):
        self.data_dir = Path(data_dir)
        self.pairs = [
            "NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", 
            "SOLUSDT", "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", 
            "XRPUSDT", "ADAUSDT", "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"
        ]
        self.fee_pct = 0.0052  # 0.52% round-trip
        self.oos_start = 4380  # Out-of-sample start
        self.data = {}
        self.results = []
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all pairs data"""
        print("Loading data...")
        for pair in self.pairs:
            file_path = self.data_dir / f"{pair}_1h.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                print(f"Loaded {pair}: {len(df)} bars")
                self.data[pair] = df
            else:
                print(f"WARNING: Missing {pair}_1h.csv")
        
        print(f"Total pairs loaded: {len(self.data)}")
        return self.data
    
    # ==================== TECHNICAL INDICATORS ====================
    
    def calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI calculation (vectorized)"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Smoothed averages
        rsi = np.full(len(prices), 50.0)
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    
    def calc_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period).mean().values
    
    def calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        if len(high) < 2:
            return np.zeros(len(high))
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return pd.Series(tr).rolling(window=period, min_periods=1).mean().values
    
    def calc_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands (SMA, upper, lower)"""
        sma = self.calc_sma(prices, period)
        std = pd.Series(prices).rolling(window=period, min_periods=1).std().values
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return sma, upper, lower
    
    def calc_autocorr(self, returns: np.ndarray, lag: int = 1) -> float:
        """Autocorrelation"""
        if len(returns) < lag + 5:
            return 0.0
        try:
            return float(pd.Series(returns).autocorr(lag=lag))
        except:
            return 0.0
    
    def calc_entropy(self, returns: np.ndarray, bins: int = 15) -> float:
        """Shannon entropy"""
        if len(returns) < 10:
            return 3.0
        try:
            hist, _ = np.histogram(returns, bins=bins, density=True)
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 3.0
            probs = hist / hist.sum()
            return -np.sum(probs * np.log2(probs))
        except:
            return 3.0
    
    def calc_hurst(self, returns: np.ndarray, window: int = 50) -> float:
        """Hurst exponent (variance ratio method)"""
        if len(returns) < window:
            return 0.5
        try:
            r = returns[-window:]
            v1 = np.var(r)
            v2 = np.var(r[::2]) if len(r) >= 4 else v1
            if v1 <= 0 or v2 <= 0:
                return 0.5
            vr = v2 / v1
            return max(0, min(1, 0.5 + np.log(max(vr, 0.01)) / (2 * np.log(2))))
        except:
            return 0.5
    
    def calc_kurtosis(self, returns: np.ndarray) -> float:
        """Kurtosis"""
        if len(returns) < 10:
            return 0.0
        try:
            return float(pd.Series(returns).kurtosis())
        except:
            return 0.0
    
    def calc_skew(self, returns: np.ndarray) -> float:
        """Skewness"""
        if len(returns) < 10:
            return 0.0
        try:
            return float(pd.Series(returns).skew())
        except:
            return 0.0
    
    # ==================== TOOL TESTING FUNCTIONS ====================
    
    def get_cross_pair_data(self, pair: str, cross_pair: str, bar_idx: int) -> Optional[float]:
        """Get cross-pair price for relative performance calculations"""
        if cross_pair not in self.data:
            return None
        cross_df = self.data[cross_pair]
        if bar_idx >= len(cross_df):
            return None
        return cross_df.iloc[bar_idx]['close']
    
    def test_tool(self, pair: str, tool_name: str, signal_func, direction: str, 
                  test_oos_only: bool = True, optimize_params: bool = False) -> Dict:
        """Test a single tool on one pair"""
        df = self.data[pair]
        
        # Only test on OOS by default
        if test_oos_only:
            test_df = df.iloc[self.oos_start:].copy()
            start_idx = self.oos_start
        else:
            test_df = df.copy()  
            start_idx = 0
        
        signals = []
        
        # Generate signals using the signal function
        for i in range(start_idx, len(df) - 24):  # Need 24 bars for forward returns
            signal = signal_func(df, i)
            if signal:
                signals.append(i)
        
        if len(signals) == 0:
            return {
                'tool': tool_name, 'pair': pair, 'direction': direction,
                'signals': 0, 'wr_8h': 0, 'wr_24h': 0, 
                'avg_return_8h': 0, 'avg_return_24h': 0, 'status': 'NO_SIGNALS'
            }
        
        # Calculate forward returns with fees
        wins_8h = 0
        wins_24h = 0
        returns_8h = []
        returns_24h = []
        
        for sig_idx in signals:
            if sig_idx + 24 >= len(df):
                continue
                
            entry_price = df.iloc[sig_idx]['close']
            exit_8h_price = df.iloc[sig_idx + 8]['close']
            exit_24h_price = df.iloc[sig_idx + 24]['close']
            
            # Calculate raw returns
            ret_8h = (exit_8h_price - entry_price) / entry_price
            ret_24h = (exit_24h_price - entry_price) / entry_price
            
            # Apply direction
            if direction == 'short':
                ret_8h = -ret_8h
                ret_24h = -ret_24h
            
            # Apply fees
            ret_8h_net = ret_8h - self.fee_pct
            ret_24h_net = ret_24h - self.fee_pct
            
            returns_8h.append(ret_8h_net)
            returns_24h.append(ret_24h_net)
            
            if ret_8h_net > 0:
                wins_8h += 1
            if ret_24h_net > 0:
                wins_24h += 1
        
        n_signals = len(returns_8h)
        wr_8h = wins_8h / n_signals if n_signals > 0 else 0
        wr_24h = wins_24h / n_signals if n_signals > 0 else 0
        avg_ret_8h = np.mean(returns_8h) if returns_8h else 0
        avg_ret_24h = np.mean(returns_24h) if returns_24h else 0
        
        # Determine status
        if wr_8h > 0.5 or wr_24h > 0.5:
            status = "PASS"
        else:
            status = "FAIL"
        
        return {
            'tool': tool_name, 'pair': pair, 'direction': direction,
            'signals': n_signals, 'wr_8h': wr_8h, 'wr_24h': wr_24h,
            'avg_return_8h': avg_ret_8h * 100, 'avg_return_24h': avg_ret_24h * 100,
            'status': status
        }
    
    # ==================== BULL/GREED SHORT TOOLS ====================
    
    def mega_pump_sell_t1(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 1: mega_pump_sell T1 - rsi7 > 80 AND ret_12h >= 10 → SHORT"""
        if i < 13:
            return False
        close = df['close'].values
        rsi7 = self.calc_rsi(close[:i+1], 7)
        ret_12h = (close[i] - close[i-13]) / close[i-13] * 100
        return rsi7[i] > 80 and ret_12h >= 10
    
    def mega_pump_sell_t2(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 2: mega_pump_sell T2 - rsi7 > 80 AND ret_12h >= 8 → SHORT"""
        if i < 13:
            return False
        close = df['close'].values
        rsi7 = self.calc_rsi(close[:i+1], 7)
        ret_12h = (close[i] - close[i-13]) / close[i-13] * 100
        return rsi7[i] > 80 and ret_12h >= 8
    
    def rsi_pump_8h(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 3: rsi_pump_8h - rsi7 > 80 AND ret_8h >= 10 (only if mega_pump didn't fire) → SHORT"""
        if i < 9:
            return False
        close = df['close'].values
        rsi7 = self.calc_rsi(close[:i+1], 7)
        ret_8h = (close[i] - close[i-9]) / close[i-9] * 100
        ret_12h = (close[i] - close[i-13]) / close[i-13] * 100 if i >= 13 else 0
        
        # Only fire if mega_pump conditions not met
        mega_pump_fired = (rsi7[i] > 80 and ret_12h >= 8)
        return not mega_pump_fired and rsi7[i] > 80 and ret_8h >= 10
    
    def rsi_pump_12h(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 4: rsi_pump_12h - rsi7 > 80 AND ret_12h >= 8 (fallback) → SHORT"""
        if i < 13:
            return False
        close = df['close'].values
        rsi7 = self.calc_rsi(close[:i+1], 7)
        ret_12h = (close[i] - close[i-13]) / close[i-13] * 100
        ret_8h = (close[i] - close[i-9]) / close[i-9] * 100 if i >= 9 else 0
        
        # Fallback - only if other RSI pump tools didn't fire
        mega_pump_fired = (rsi7[i] > 80 and ret_12h >= 10)
        rsi_8h_fired = (rsi7[i] > 80 and ret_8h >= 10)
        return not (mega_pump_fired or rsi_8h_fired) and rsi7[i] > 80 and ret_12h >= 8
    
    def greed_short_t2(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 5: greed_short T2 - rsi7 > 75 AND ret_8h > 5 AND price > sma50 → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        rsi7 = self.calc_rsi(close[:i+1], 7)
        sma50 = self.calc_sma(close[:i+1], 50)
        ret_8h = (close[i] - close[i-9]) / close[i-9] * 100 if i >= 9 else 0
        return rsi7[i] > 75 and ret_8h > 5 and close[i] > sma50[i]
    
    def sma50_ext_8(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 6: sma50_ext_8 - cur_vs_sma50 > 8% → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        return cur_vs_sma50 > 8
    
    def sma50_ext_10(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 7: sma50_ext_10 - cur_vs_sma50 > 10% → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        return cur_vs_sma50 > 10
    
    def sma50_ext_12(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 8: sma50_ext_12 - cur_vs_sma50 > 12% → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        return cur_vs_sma50 > 12
    
    def sma50_ext_15(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 9: sma50_ext_15 - cur_vs_sma50 > 15% → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        return cur_vs_sma50 > 15
    
    def thursday_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 10: thursday_short - day_of_week == Thursday AND price > sma50 → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        
        # Get day of week from timestamp (assuming hourly data)
        timestamp = pd.to_datetime(df.iloc[i]['timestamp'])
        dow = timestamp.weekday()  # 0=Monday, 3=Thursday
        
        return dow == 3 and close[i] > sma50[i]
    
    def sunday_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 11: sunday_short - day_of_week == Sunday AND price > sma50 → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        
        timestamp = pd.to_datetime(df.iloc[i]['timestamp'])
        dow = timestamp.weekday()  # 6=Sunday
        
        return dow == 6 and close[i] > sma50[i]
    
    def month_start_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 12: month_start_short - day 1-3 AND price > sma50 → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        
        timestamp = pd.to_datetime(df.iloc[i]['timestamp'])
        day = timestamp.day
        
        return day <= 3 and close[i] > sma50[i]
    
    def late_us_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 13: late_us_short - hour == 21 UTC AND price > sma50 → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        
        timestamp = pd.to_datetime(df.iloc[i]['timestamp'])
        hour = timestamp.hour
        
        return hour == 21 and close[i] > sma50[i]
    
    def ema_cross_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 14: ema_cross_short - ema5 > ema13 AND price > sma50 → SHORT"""
        if i < 50:
            return False
        close = df['close'].values
        ema5 = self.calc_ema(close[:i+1], 5)
        ema13 = self.calc_ema(close[:i+1], 13)
        sma50 = self.calc_sma(close[:i+1], 50)
        
        return ema5[i] > ema13[i] and close[i] > sma50[i]
    
    def distribution_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 15: distribution_short - lower highs + vol declining 30% + RSI falling 8+ + price > sma50 + rsi14 > 60 → SHORT"""
        if i < 60:
            return False
        
        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values.astype(float)
        
        sma50 = self.calc_sma(close[:i+1], 50)
        rsi14 = self.calc_rsi(close[:i+1], 14)
        
        if close[i] <= sma50[i] or rsi14[i] <= 60:
            return False
        
        # Lower highs over last 10 bars
        recent_highs = high[i-9:i+1]
        lower_highs = all(recent_highs[j] <= recent_highs[j-1] for j in range(1, len(recent_highs))) if len(recent_highs) >= 2 else False
        
        # Volume declining 30%
        vol_now = np.mean(volume[i-4:i+1])
        vol_before = np.mean(volume[i-19:i-15]) 
        vol_declining = (vol_now / vol_before < 0.7) if vol_before > 0 else False
        
        # RSI falling 8+
        rsi_now = rsi14[i]
        rsi_before = rsi14[i-5] if i >= 5 else rsi_now
        rsi_falling = (rsi_before - rsi_now >= 8)
        
        return lower_highs and vol_declining and rsi_falling
    
    def falling_wedge_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 16: falling_wedge_short - lower highs + lower lows + converging + price > sma50 → SHORT"""
        if i < 60:
            return False
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        
        if close[i] <= sma50[i]:
            return False
        
        # Look at last 20 bars for pattern
        highs = high[i-19:i+1]
        lows = low[i-19:i+1]
        
        # Lower highs trend
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0] if len(highs) > 1 else 0
        
        # Lower lows trend  
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0] if len(lows) > 1 else 0
        
        # Converging: high slope more negative than low slope
        converging = high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope)
        
        return converging
    
    def green_exhaustion(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 17: green_exhaustion - 7 green then 1 red candle → SHORT"""
        if i < 8:
            return False
        
        close = df['close'].values
        open_vals = df['open'].values.astype(float)
        
        # Check 7 consecutive green candles followed by current red
        all_green = all(close[i-j] > open_vals[i-j] for j in range(1, 8))
        cur_red = close[i] < open_vals[i]
        
        return all_green and cur_red
    
    def entropy_short(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 18: entropy_short - entropy < 2.5 AND price > sma50 → SHORT"""
        if i < 100:
            return False
        
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        
        if close[i] <= sma50[i]:
            return False
        
        # Calculate entropy on recent returns
        returns = np.diff(close[i-30:i+1]) / close[i-30:i]
        entropy = self.calc_entropy(returns)
        
        return entropy < 2.5
    
    # ==================== CROSS-PAIR SHORT TOOLS ====================
    
    def alt_btc_revert_t1(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 19: alt_btc_revert T1 - alt outperforms BTC by 8%+ in 24h → SHORT"""
        if i < 25:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':  # Skip BTC itself
            return False
        
        close = df['close'].values
        alt_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100
        
        # Get BTC price at same time
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        spread = alt_ret_24h - btc_ret_24h
        
        return spread >= 8
    
    def alt_btc_revert_t2(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 20: alt_btc_revert T2 - alt outperforms BTC by 5-8% in 24h → SHORT"""
        if i < 25:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        spread = alt_ret_24h - btc_ret_24h
        
        return 5 <= spread < 8
    
    def alt_btc_revert_t3(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 21: alt_btc_revert_t3 - alt outperforms BTC by 3-5% in 24h → SHORT"""
        if i < 25:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        spread = alt_ret_24h - btc_ret_24h
        
        return 3 <= spread < 5
    
    def btc_eth_diverge(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 22: btc_eth_diverge - BTC outperforms ETH by 3%+ in 24h → SHORT ETH only"""
        if i < 25:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair != 'ETHUSDT':  # Only applies to ETH
            return False
        
        close = df['close'].values
        eth_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        spread = btc_ret_24h - eth_ret_24h
        
        return spread >= 3
    
    # ==================== COMBO SHORT TOOLS ====================
    
    def sma50_ext_neg_ac(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 23: sma50_ext_neg_ac - cur_vs_sma50 > 8% AND autocorr < -0.1 → SHORT"""
        if i < 100:
            return False
        
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        
        if cur_vs_sma50 <= 8:
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        autocorr = self.calc_autocorr(returns, lag=1)
        
        return autocorr < -0.1
    
    def sma50_ext_fat_tail(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 24: sma50_ext_fat_tail - cur_vs_sma50 > 10% AND kurtosis > 5 → SHORT"""
        if i < 100:
            return False
        
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        
        if cur_vs_sma50 <= 10:
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        kurtosis = self.calc_kurtosis(returns)
        
        return kurtosis > 5
    
    def sma50_ext_kurt(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 25: sma50_ext_kurt - cur_vs_sma50 > 8% AND kurtosis > 5 → SHORT"""
        if i < 100:
            return False
        
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        
        if cur_vs_sma50 <= 8:
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        kurtosis = self.calc_kurtosis(returns)
        
        return kurtosis > 5
    
    def alt_btc_neg_ac(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 26: alt_btc_neg_ac - alt outperforms BTC 8% AND autocorr < -0.05 → SHORT"""
        if i < 100:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100 if i >= 25 else 0
        
        btc_close = self.get_cross_pair_data('BTCUSDT', 'BTCUSDT', i)
        btc_close_24h = self.get_cross_pair_data('BTCUSDT', 'BTCUSDT', i-25)
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        spread = alt_ret_24h - btc_ret_24h
        
        if spread < 8:
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        autocorr = self.calc_autocorr(returns, lag=1)
        
        return autocorr < -0.05
    
    def alt_btc_neg_ac_5(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 27: alt_btc_neg_ac_5 - alt outperforms BTC 5% AND autocorr < -0.15 → SHORT"""
        if i < 100:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100 if i >= 25 else 0
        
        btc_close = self.get_cross_pair_data('BTCUSDT', 'BTCUSDT', i)
        btc_close_24h = self.get_cross_pair_data('BTCUSDT', 'BTCUSDT', i-25)
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        spread = alt_ret_24h - btc_ret_24h
        
        if spread < 5:
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        autocorr = self.calc_autocorr(returns, lag=1)
        
        return autocorr < -0.15
    
    def rsi_pump_fat_tail(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 28: rsi_pump_fat_tail - rsi7 > 80 AND ret_12h > 10% AND kurtosis > 5 → SHORT"""
        if i < 100:
            return False
        
        close = df['close'].values
        rsi7 = self.calc_rsi(close[:i+1], 7)
        ret_12h = (close[i] - close[i-13]) / close[i-13] * 100 if i >= 13 else 0
        
        if rsi7[i] <= 80 or ret_12h <= 10:
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        kurtosis = self.calc_kurtosis(returns)
        
        return kurtosis > 5
    
    def green_exhaust_kurt(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 29: green_exhaust_kurt - 7 green then red AND kurtosis > 3 → SHORT"""
        if i < 100:
            return False
        
        close = df['close'].values
        open_vals = df['open'].values.astype(float)
        
        # Check 7 consecutive green candles followed by current red
        if i < 8:
            return False
        
        all_green = all(close[i-j] > open_vals[i-j] for j in range(1, 8))
        cur_red = close[i] < open_vals[i]
        
        if not (all_green and cur_red):
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        kurtosis = self.calc_kurtosis(returns)
        
        return kurtosis > 3
    
    # ==================== BREAKOUT/MOMENTUM TOOLS ====================
    
    def breakout_detect(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 30: breakout_detect - BB squeeze (bandwidth at 20-bar low) + price > upper BB + vol > 2x → LONG"""
        if i < 40:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        # Calculate Bollinger Bands
        sma, upper_bb, lower_bb = self.calc_bollinger_bands(close[:i+1], 20, 2.0)
        
        # BB squeeze: bandwidth at 20-bar low
        bandwidth = (upper_bb - lower_bb) / sma
        min_bandwidth = np.min(bandwidth[max(0, i-19):i+1])
        is_squeeze = bandwidth[i] <= min_bandwidth
        
        # Price > upper BB
        price_above_bb = close[i] > upper_bb[i]
        
        # Volume > 2x average
        avg_vol = np.mean(volume[max(0, i-19):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        return is_squeeze and price_above_bb and vol_spike
    
    def high_breakout_50(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 31: high_breakout_50 - price > 50-bar high AND vol > 2x → LONG"""
        if i < 50:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        # Price > 50-bar high
        high_50 = np.max(close[i-50:i])
        price_breakout = close[i] > high_50
        
        # Volume > 2x
        avg_vol = np.mean(volume[max(0, i-19):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        return price_breakout and vol_spike
    
    def high_breakout_50_nv(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 32: high_breakout_50_nv - price > 50-bar high (no vol filter) → LONG"""
        if i < 50:
            return False
        
        close = df['close'].values
        high_50 = np.max(close[i-50:i])
        return close[i] > high_50
    
    def high_breakout_30(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 33: high_breakout_30 - price > 30-bar high AND vol > 2x → LONG"""
        if i < 30:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        high_30 = np.max(close[i-30:i])
        price_breakout = close[i] > high_30
        
        avg_vol = np.mean(volume[max(0, i-19):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        return price_breakout and vol_spike
    
    def bb_above_long_t1(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 34: bb_above_long T1 - price > upper_bb * 1.02 → LONG"""
        if i < 20:
            return False
        
        close = df['close'].values
        _, upper_bb, _ = self.calc_bollinger_bands(close[:i+1], 20, 2.0)
        
        return close[i] > upper_bb[i] * 1.02
    
    def bb_above_long_t2(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 35: bb_above_long T2 - price > upper_bb * 1.01 → LONG"""
        if i < 20:
            return False
        
        close = df['close'].values
        _, upper_bb, _ = self.calc_bollinger_bands(close[:i+1], 20, 2.0)
        
        return close[i] > upper_bb[i] * 1.01
    
    def bb_squeeze_15(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 36: bb_squeeze_15 - BB(15) squeeze + breakout + 2x vol → LONG"""
        if i < 30:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        sma, upper_bb, lower_bb = self.calc_bollinger_bands(close[:i+1], 15, 2.0)
        
        # BB squeeze
        bandwidth = (upper_bb - lower_bb) / sma
        min_bandwidth = np.min(bandwidth[max(0, i-14):i+1])
        is_squeeze = bandwidth[i] <= min_bandwidth
        
        # Breakout
        price_breakout = close[i] > upper_bb[i]
        
        # Volume
        avg_vol = np.mean(volume[max(0, i-14):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        return is_squeeze and price_breakout and vol_spike
    
    def bb_squeeze_30(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 37: bb_squeeze_30 - BB(30) squeeze + breakout + 2x vol → LONG"""
        if i < 50:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        sma, upper_bb, lower_bb = self.calc_bollinger_bands(close[:i+1], 30, 2.0)
        
        bandwidth = (upper_bb - lower_bb) / sma
        min_bandwidth = np.min(bandwidth[max(0, i-29):i+1])
        is_squeeze = bandwidth[i] <= min_bandwidth
        
        price_breakout = close[i] > upper_bb[i]
        
        avg_vol = np.mean(volume[max(0, i-29):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        return is_squeeze and price_breakout and vol_spike
    
    def hurst_trend(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 38: hurst_trend - Hurst > 0.65 AND ret_4h > 2% → LONG"""
        if i < 100:
            return False
        
        close = df['close'].values
        ret_4h = (close[i] - close[i-5]) / close[i-5] * 100 if i >= 5 else 0
        
        if ret_4h <= 2:
            return False
        
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        hurst = self.calc_hurst(returns)
        
        return hurst > 0.65
    
    def fomo_ride(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 39: fomo_ride - 80%+ of all pairs pumping >1% in 4h → LONG (cross-pair)"""
        if i < 5:
            return False
        
        # Count how many pairs are pumping
        pumping_pairs = 0
        total_pairs = 0
        
        for pair_name in self.pairs:
            if pair_name not in self.data:
                continue
                
            pair_df = self.data[pair_name]
            if i >= len(pair_df):
                continue
            
            pair_close = pair_df['close'].values
            if i >= 5:
                pair_ret_4h = (pair_close[i] - pair_close[i-5]) / pair_close[i-5] * 100
                total_pairs += 1
                if pair_ret_4h > 1:
                    pumping_pairs += 1
        
        if total_pairs < 5:
            return False
        
        fomo_pct = pumping_pairs / total_pairs
        return fomo_pct >= 0.8
    
    def btc_lead_lag_buy(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 40: btc_lead_lag_buy - BTC ret_3h >= 2% AND alt ret_3h < 0.5% → LONG alt"""
        if i < 3:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_3h = (close[i] - close[i-3]) / close[i-3] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-3 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_3h = btc_df.iloc[i-3]['close']
        
        btc_ret_3h = (btc_close - btc_close_3h) / btc_close_3h * 100
        
        return btc_ret_3h >= 2 and alt_ret_3h < 0.5
    
    def btc_lag_3h(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 41: btc_lag_3h - BTC ret_3h >= 1.5% AND alt ret_3h < 0.5% → LONG"""
        if i < 3:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_3h = (close[i] - close[i-3]) / close[i-3] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-3 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_3h = btc_df.iloc[i-3]['close']
        
        btc_ret_3h = (btc_close - btc_close_3h) / btc_close_3h * 100
        
        return btc_ret_3h >= 1.5 and alt_ret_3h < 0.5
    
    def btc_lag_1h(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 42: btc_lag_1h - BTC ret_1h >= 1% AND alt ret_1h < 0.3% → LONG"""
        if i < 1:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_1h = (close[i] - close[i-1]) / close[i-1] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-1 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_1h = btc_df.iloc[i-1]['close']
        
        btc_ret_1h = (btc_close - btc_close_1h) / btc_close_1h * 100
        
        return btc_ret_1h >= 1 and alt_ret_1h < 0.3
    
    # ==================== COMBO LONG TOOLS ====================
    
    def bb_break_pos_ac(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 43: bb_break_pos_ac - BB squeeze breakout + autocorr > 0.1 + 2x vol → LONG"""
        if i < 100:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        # BB squeeze breakout
        sma, upper_bb, lower_bb = self.calc_bollinger_bands(close[:i+1], 20, 2.0)
        bandwidth = (upper_bb - lower_bb) / sma
        min_bandwidth = np.min(bandwidth[max(0, i-19):i+1])
        is_squeeze = bandwidth[i] <= min_bandwidth
        price_breakout = close[i] > upper_bb[i]
        
        # Volume
        avg_vol = np.mean(volume[max(0, i-19):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        # Positive autocorrelation
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        autocorr = self.calc_autocorr(returns, lag=1)
        
        return is_squeeze and price_breakout and vol_spike and autocorr > 0.1
    
    def high_break_pos_ac(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 44: high_break_pos_ac - 50-bar high breakout + autocorr > 0.1 + 2x vol → LONG"""
        if i < 100:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        # 50-bar high breakout
        high_50 = np.max(close[i-50:i])
        price_breakout = close[i] > high_50
        
        # Volume
        avg_vol = np.mean(volume[max(0, i-19):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        # Positive autocorrelation
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        autocorr = self.calc_autocorr(returns, lag=1)
        
        return price_breakout and vol_spike and autocorr > 0.1
    
    def high_break_skew(self, df: pd.DataFrame, i: int) -> bool:
        """Tool 45: high_break_skew - 50-bar high breakout + skew > 1.0 + 2x vol → LONG"""
        if i < 100:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        # 50-bar high breakout
        high_50 = np.max(close[i-50:i])
        price_breakout = close[i] > high_50
        
        # Volume
        avg_vol = np.mean(volume[max(0, i-19):i+1])
        vol_spike = volume[i] > 2 * avg_vol if avg_vol > 0 else False
        
        # Positive skew
        returns = np.diff(close[i-50:i+1]) / close[i-50:i]
        skew = self.calc_skew(returns)
        
        return price_breakout and vol_spike and skew > 1.0
    
    # ==================== NEW TOOL IDEAS ====================
    
    def correlation_breakdown_long(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: correlation breakdown - alt-BTC correlation < 0.2 AND alt underperforming → LONG alt"""
        if i < 48:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        
        # Alt underperforming
        if alt_ret_24h >= btc_ret_24h:
            return False
        
        # Calculate rolling correlation using 24-bar window
        alt_returns = np.diff(close[i-24:i+1]) / close[i-24:i]
        btc_returns = []
        
        for j in range(i-24, i):
            if j+1 >= len(btc_df):
                continue
            btc_j = btc_df.iloc[j]['close']
            btc_j1 = btc_df.iloc[j+1]['close']
            btc_returns.append((btc_j1 - btc_j) / btc_j)
        
        if len(btc_returns) < len(alt_returns):
            return False
        
        try:
            correlation = np.corrcoef(alt_returns, btc_returns[:len(alt_returns)])[0, 1]
            return correlation < 0.2 and not np.isnan(correlation)
        except:
            return False
    
    def correlation_breakdown_short(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: correlation breakdown - alt-BTC correlation < 0.2 AND alt outperforming → SHORT alt"""
        if i < 48:
            return False
        
        pair = getattr(df, '_pair_name', None)
        if pair == 'BTCUSDT':
            return False
        
        close = df['close'].values
        alt_ret_24h = (close[i] - close[i-25]) / close[i-25] * 100
        
        if 'BTCUSDT' not in self.data:
            return False
        btc_df = self.data['BTCUSDT']
        if i >= len(btc_df) or i-25 < 0:
            return False
        
        btc_close = btc_df.iloc[i]['close']
        btc_close_24h = btc_df.iloc[i-25]['close']
        btc_ret_24h = (btc_close - btc_close_24h) / btc_close_24h * 100
        
        # Alt outperforming
        if alt_ret_24h <= btc_ret_24h:
            return False
        
        # Calculate rolling correlation using 24-bar window
        alt_returns = np.diff(close[i-24:i+1]) / close[i-24:i]
        btc_returns = []
        
        for j in range(i-24, i):
            if j+1 >= len(btc_df):
                continue
            btc_j = btc_df.iloc[j]['close']
            btc_j1 = btc_df.iloc[j+1]['close']
            btc_returns.append((btc_j1 - btc_j) / btc_j)
        
        if len(btc_returns) < len(alt_returns):
            return False
        
        try:
            correlation = np.corrcoef(alt_returns, btc_returns[:len(alt_returns)])[0, 1]
            return correlation < 0.2 and not np.isnan(correlation)
        except:
            return False
    
    def relative_strength_long(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: relative strength rotation - LONG bottom 3 performers"""
        if i < 25:
            return False
        
        # Calculate 24h returns for all pairs
        pair_returns = []
        current_pair = getattr(df, '_pair_name', None)
        
        for pair_name in self.pairs:
            if pair_name not in self.data:
                continue
            pair_df = self.data[pair_name]
            if i >= len(pair_df) or i < 25:
                continue
            
            pair_close = pair_df['close'].values
            ret_24h = (pair_close[i] - pair_close[i-25]) / pair_close[i-25] * 100
            pair_returns.append((pair_name, ret_24h))
        
        if len(pair_returns) < 6:  # Need at least 6 pairs to rank
            return False
        
        # Sort by return (ascending)
        pair_returns.sort(key=lambda x: x[1])
        
        # Check if current pair is in bottom 3
        bottom_3 = [p[0] for p in pair_returns[:3]]
        return current_pair in bottom_3
    
    def relative_strength_short(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: relative strength rotation - SHORT top 3 performers"""
        if i < 25:
            return False
        
        pair_returns = []
        current_pair = getattr(df, '_pair_name', None)
        
        for pair_name in self.pairs:
            if pair_name not in self.data:
                continue
            pair_df = self.data[pair_name]
            if i >= len(pair_df) or i < 25:
                continue
            
            pair_close = pair_df['close'].values
            ret_24h = (pair_close[i] - pair_close[i-25]) / pair_close[i-25] * 100
            pair_returns.append((pair_name, ret_24h))
        
        if len(pair_returns) < 6:
            return False
        
        # Sort by return (descending)
        pair_returns.sort(key=lambda x: x[1], reverse=True)
        
        # Check if current pair is in top 3
        top_3 = [p[0] for p in pair_returns[:3]]
        return current_pair in top_3
    
    def wick_absorption_long(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: wick absorption - long lower wick (>70% of range) → LONG"""
        if i < 1:
            return False
        
        high = df['high'].values[i]
        low = df['low'].values[i]
        open_val = df['open'].values[i]
        close = df['close'].values[i]
        
        total_range = high - low
        if total_range == 0:
            return False
        
        lower_wick = min(open_val, close) - low
        lower_wick_pct = lower_wick / total_range
        
        return lower_wick_pct > 0.7
    
    def wick_absorption_short(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: wick absorption - long upper wick (>70% of range) → SHORT"""
        if i < 1:
            return False
        
        high = df['high'].values[i]
        low = df['low'].values[i]
        open_val = df['open'].values[i]
        close = df['close'].values[i]
        
        total_range = high - low
        if total_range == 0:
            return False
        
        upper_wick = high - max(open_val, close)
        upper_wick_pct = upper_wick / total_range
        
        return upper_wick_pct > 0.7
    
    def volatility_squeeze_breakout_long(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: volatility squeeze breakout - ATR squeeze + BB breakout → LONG"""
        if i < 200:
            return False
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # ATR squeeze
        atr14 = self.calc_atr(high[:i+1], low[:i+1], close[:i+1], 14)
        atr_percentile = np.percentile(atr14[max(0, i-200):i+1], 20)
        is_squeeze = atr14[i] <= atr_percentile
        
        # BB breakout
        _, upper_bb, _ = self.calc_bollinger_bands(close[:i+1], 20, 2.0)
        price_breakout = close[i] > upper_bb[i]
        
        return is_squeeze and price_breakout
    
    def volatility_squeeze_breakout_short(self, df: pd.DataFrame, i: int) -> bool:
        """New tool: volatility squeeze breakout - ATR squeeze + BB breakdown → SHORT"""
        if i < 200:
            return False
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # ATR squeeze
        atr14 = self.calc_atr(high[:i+1], low[:i+1], close[:i+1], 14)
        atr_percentile = np.percentile(atr14[max(0, i-200):i+1], 20)
        is_squeeze = atr14[i] <= atr_percentile
        
        # BB breakdown
        _, _, lower_bb = self.calc_bollinger_bands(close[:i+1], 20, 2.0)
        price_breakdown = close[i] < lower_bb[i]
        
        return is_squeeze and price_breakdown
    
    def run_all_tests(self):
        """Run all tool tests across all pairs"""
        print("Starting comprehensive tool testing...")
        
        # Define all tools to test
        tools_to_test = [
            # BULL/GREED SHORT TOOLS
            ('mega_pump_sell_t1', self.mega_pump_sell_t1, 'short'),
            ('mega_pump_sell_t2', self.mega_pump_sell_t2, 'short'),
            ('rsi_pump_8h', self.rsi_pump_8h, 'short'),
            ('rsi_pump_12h', self.rsi_pump_12h, 'short'),
            ('greed_short_t2', self.greed_short_t2, 'short'),
            ('sma50_ext_8', self.sma50_ext_8, 'short'),
            ('sma50_ext_10', self.sma50_ext_10, 'short'),
            ('sma50_ext_12', self.sma50_ext_12, 'short'),
            ('sma50_ext_15', self.sma50_ext_15, 'short'),
            ('thursday_short', self.thursday_short, 'short'),
            ('sunday_short', self.sunday_short, 'short'),
            ('month_start_short', self.month_start_short, 'short'),
            ('late_us_short', self.late_us_short, 'short'),
            ('ema_cross_short', self.ema_cross_short, 'short'),
            ('distribution_short', self.distribution_short, 'short'),
            ('falling_wedge_short', self.falling_wedge_short, 'short'),
            ('green_exhaustion', self.green_exhaustion, 'short'),
            ('entropy_short', self.entropy_short, 'short'),
            
            # CROSS-PAIR SHORT TOOLS
            ('alt_btc_revert_t1', self.alt_btc_revert_t1, 'short'),
            ('alt_btc_revert_t2', self.alt_btc_revert_t2, 'short'),
            ('alt_btc_revert_t3', self.alt_btc_revert_t3, 'short'),
            ('btc_eth_diverge', self.btc_eth_diverge, 'short'),
            
            # COMBO SHORT TOOLS
            ('sma50_ext_neg_ac', self.sma50_ext_neg_ac, 'short'),
            ('sma50_ext_fat_tail', self.sma50_ext_fat_tail, 'short'),
            ('sma50_ext_kurt', self.sma50_ext_kurt, 'short'),
            ('alt_btc_neg_ac', self.alt_btc_neg_ac, 'short'),
            ('alt_btc_neg_ac_5', self.alt_btc_neg_ac_5, 'short'),
            ('rsi_pump_fat_tail', self.rsi_pump_fat_tail, 'short'),
            ('green_exhaust_kurt', self.green_exhaust_kurt, 'short'),
            
            # BREAKOUT/MOMENTUM LONG TOOLS
            ('breakout_detect', self.breakout_detect, 'long'),
            ('high_breakout_50', self.high_breakout_50, 'long'),
            ('high_breakout_50_nv', self.high_breakout_50_nv, 'long'),
            ('high_breakout_30', self.high_breakout_30, 'long'),
            ('bb_above_long_t1', self.bb_above_long_t1, 'long'),
            ('bb_above_long_t2', self.bb_above_long_t2, 'long'),
            ('bb_squeeze_15', self.bb_squeeze_15, 'long'),
            ('bb_squeeze_30', self.bb_squeeze_30, 'long'),
            ('hurst_trend', self.hurst_trend, 'long'),
            ('fomo_ride', self.fomo_ride, 'long'),
            ('btc_lead_lag_buy', self.btc_lead_lag_buy, 'long'),
            ('btc_lag_3h', self.btc_lag_3h, 'long'),
            ('btc_lag_1h', self.btc_lag_1h, 'long'),
            
            # COMBO LONG TOOLS
            ('bb_break_pos_ac', self.bb_break_pos_ac, 'long'),
            ('high_break_pos_ac', self.high_break_pos_ac, 'long'),
            ('high_break_skew', self.high_break_skew, 'long'),
            
            # NEW TOOL IDEAS
            ('correlation_breakdown_long', self.correlation_breakdown_long, 'long'),
            ('correlation_breakdown_short', self.correlation_breakdown_short, 'short'),
            ('relative_strength_long', self.relative_strength_long, 'long'),
            ('relative_strength_short', self.relative_strength_short, 'short'),
            ('wick_absorption_long', self.wick_absorption_long, 'long'),
            ('wick_absorption_short', self.wick_absorption_short, 'short'),
            ('volatility_squeeze_breakout_long', self.volatility_squeeze_breakout_long, 'long'),
            ('volatility_squeeze_breakout_short', self.volatility_squeeze_breakout_short, 'short'),
        ]
        
        # Test each tool on each pair
        total_tools = len(tools_to_test)
        for idx, (tool_name, tool_func, direction) in enumerate(tools_to_test, 1):
            print(f"\n[{idx}/{total_tools}] Testing {tool_name} ({direction.upper()})...")
            
            for pair in self.pairs:
                if pair not in self.data:
                    continue
                
                # Add pair name to dataframe for cross-pair tools
                df = self.data[pair].copy()
                df._pair_name = pair
                    
                result = self.test_tool(pair, tool_name, 
                                      lambda df_inner, i: tool_func(df_inner, i), 
                                      direction)
                self.results.append(result)
                
                if result['signals'] > 0:
                    print(f"  {pair}: {result['signals']} signals, "
                          f"WR_8h={result['wr_8h']:.1%}, WR_24h={result['wr_24h']:.1%}, "
                          f"Ret_8h={result['avg_return_8h']:+.2f}%, Ret_24h={result['avg_return_24h']:+.2f}% "
                          f"[{result['status']}]")
    
    def generate_report(self):
        """Generate comprehensive report"""
        if not self.results:
            print("No results to report!")
            return
        
        # Create summary by tool
        tool_summary = {}
        for result in self.results:
            tool = result['tool']
            if tool not in tool_summary:
                tool_summary[tool] = {
                    'direction': result['direction'],
                    'total_signals': 0,
                    'pairs_tested': 0,
                    'pairs_with_signals': 0,
                    'total_wr_8h_weighted': 0,
                    'total_wr_24h_weighted': 0,
                    'total_ret_8h_weighted': 0,
                    'total_ret_24h_weighted': 0,
                    'pass_count': 0,
                }
            
            summary = tool_summary[tool]
            summary['pairs_tested'] += 1
            
            if result['signals'] > 0:
                summary['pairs_with_signals'] += 1
                summary['total_signals'] += result['signals']
                
                # Weight by number of signals
                weight = result['signals']
                summary['total_wr_8h_weighted'] += result['wr_8h'] * weight
                summary['total_wr_24h_weighted'] += result['wr_24h'] * weight
                summary['total_ret_8h_weighted'] += result['avg_return_8h'] * weight
                summary['total_ret_24h_weighted'] += result['avg_return_24h'] * weight
                
                if result['status'] == 'PASS':
                    summary['pass_count'] += 1
        
        # Calculate weighted averages
        for tool, summary in tool_summary.items():
            if summary['total_signals'] > 0:
                summary['avg_wr_8h'] = summary['total_wr_8h_weighted'] / summary['total_signals']
                summary['avg_wr_24h'] = summary['total_wr_24h_weighted'] / summary['total_signals']
                summary['avg_ret_8h'] = summary['total_ret_8h_weighted'] / summary['total_signals']
                summary['avg_ret_24h'] = summary['total_ret_24h_weighted'] / summary['total_signals']
            else:
                summary['avg_wr_8h'] = 0
                summary['avg_wr_24h'] = 0
                summary['avg_ret_8h'] = 0
                summary['avg_ret_24h'] = 0
        
        # Write report
        report_path = Path("data/bull_momentum_tools_1h_report.md")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# BULL/GREED + BREAKOUT/MOMENTUM Tools - 1H Testing Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Test Configuration\n")
            f.write(f"- **Data:** Real 1h Binance candles, {len(self.pairs)} pairs\n")
            f.write(f"- **Walk-forward:** Bars 0-4380 = in-sample, bars 4380-8760 = out-of-sample\n") 
            f.write(f"- **Fee-adjusted:** -{self.fee_pct*100:.2f}% round-trip subtracted from every return\n")
            f.write(f"- **Forward returns:** +8 bars (8h) and +24 bars (24h)\n")
            f.write(f"- **Win condition:** (forward_return - fees) > 0\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Tool | Direction | Signals | Pairs w/ Signals | WR_8h | WR_24h | Avg_Ret_8h | Avg_Ret_24h | Status |\n")
            f.write("|------|-----------|---------|------------------|-------|--------|------------|-------------|--------|\n")
            
            # Sort by total signals descending
            sorted_tools = sorted(tool_summary.items(), key=lambda x: x[1]['total_signals'], reverse=True)
            
            for tool, summary in sorted_tools:
                direction = summary['direction'].upper()
                signals = summary['total_signals']
                pairs_with_signals = summary['pairs_with_signals']
                wr_8h = summary['avg_wr_8h']
                wr_24h = summary['avg_wr_24h']
                ret_8h = summary['avg_ret_8h']
                ret_24h = summary['avg_ret_24h']
                
                # Determine overall status
                if signals == 0:
                    status = "NO_SIGNALS"
                elif wr_8h > 0.5 or wr_24h > 0.5:
                    status = "PASS"
                else:
                    status = "FAIL"
                
                f.write(f"| {tool} | {direction} | {signals} | {pairs_with_signals} | "
                       f"{wr_8h:.1%} | {wr_24h:.1%} | {ret_8h:+.2f}% | {ret_24h:+.2f}% | {status} |\n")
        
        print(f"\nReport written to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, s in sorted_tools if (s['avg_wr_8h'] > 0.5 or s['avg_wr_24h'] > 0.5) and s['total_signals'] > 0)
        failed = sum(1 for _, s in sorted_tools if (s['avg_wr_8h'] <= 0.5 and s['avg_wr_24h'] <= 0.5) and s['total_signals'] > 0)
        no_signals = sum(1 for _, s in sorted_tools if s['total_signals'] == 0)
        
        print(f"Tools PASSED: {passed}")
        print(f"Tools FAILED: {failed}")
        print(f"Tools NO_SIGNALS: {no_signals}")
        print(f"Total tools tested: {len(sorted_tools)}")
        
        if passed > 0:
            print(f"\nBest performing tools:")
            for tool, summary in sorted_tools[:5]:
                if (summary['avg_wr_8h'] > 0.5 or summary['avg_wr_24h'] > 0.5) and summary['total_signals'] > 0:
                    print(f"  {tool}: {summary['total_signals']} signals, "
                          f"WR_8h={summary['avg_wr_8h']:.1%}, WR_24h={summary['avg_wr_24h']:.1%}, "
                          f"Ret_8h={summary['avg_ret_8h']:+.2f}%, Ret_24h={summary['avg_ret_24h']:+.2f}%")


def main():
    print("BULL/GREED + BREAKOUT/MOMENTUM Tools Testing Framework")
    print("="*60)
    
    tester = BullMomentumTester()
    
    # Load data
    tester.load_data()
    
    # Run tests  
    tester.run_all_tests()
    
    # Generate report
    tester.generate_report()
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()