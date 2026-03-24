#!/usr/bin/env python3
"""
Test and fix 30 CRASH/BEAR/MEAN-REVERSION tools with OOS validation.
Uses real 1h Binance data across 16 pairs, 8760 bars each.

Walk-forward: bars 0-4380 = in-sample, bars 4380-8760 = out-of-sample
Fee-adjusted: subtract 0.52% round-trip from every return
Forward returns at +8 bars (8h) and +24 bars (24h)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/Users/lucasaust/code/Crypto-trading-bot/data/binance_1h")
OUTPUT_DIR = Path("/Users/lucasaust/code/Crypto-trading-bot/data")
PAIRS = ["NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", "SOLUSDT",
         "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", "XRPUSDT", "ADAUSDT", 
         "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"]

IN_SAMPLE_END = 4380
TOTAL_BARS = 8760
FEE_PCT = 0.0052  # 0.52% round-trip
FORWARD_8H = 8
FORWARD_24H = 24

class TechnicalIndicators:
    """Technical indicators matching run_master_bot.py exactly"""
    
    @staticmethod
    def calc_rsi(prices: np.ndarray, period: int = 7) -> np.ndarray:
        """7-period RSI as used in master bot"""
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
    
    @staticmethod
    def calc_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
            
        sma = np.full(len(prices), np.nan)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            
        return sma
    
    @staticmethod 
    def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """14-period ATR"""
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

class CrashToolsTester:
    """Test all 30 crash/bear/mean-reversion tools"""
    
    def __init__(self):
        self.data = {}
        self.results = []
        self.load_data()
    
    def load_data(self):
        """Load all pair data"""
        print("Loading data for 16 pairs...")
        for pair in PAIRS:
            file_path = DATA_DIR / f"{pair}_1h.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data[pair] = df
                print(f"  {pair}: {len(df)} bars")
            else:
                print(f"  ERROR: {file_path} not found")
    
    def compute_features(self, df):
        """Compute all technical features for a dataframe"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_prices = df['open'].values
        
        # Basic indicators
        rsi7 = TechnicalIndicators.calc_rsi(close, 7)
        rsi14 = TechnicalIndicators.calc_rsi(close, 14) 
        sma50 = TechnicalIndicators.calc_sma(close, 50)
        atr14 = TechnicalIndicators.calc_atr(high, low, close, 14)
        
        # Returns (matching master bot exactly)
        ret_4h = np.full(len(close), 0.0)
        ret_8h = np.full(len(close), 0.0) 
        ret_12h = np.full(len(close), 0.0)
        ret_24h = np.full(len(close), 0.0)
        
        for i in range(len(close)):
            if i >= 5:  # 4h return uses close[i-5]
                ret_4h[i] = (close[i] - close[i-5]) / close[i-5] * 100
            if i >= 9:  # 8h return uses close[i-9] 
                ret_8h[i] = (close[i] - close[i-9]) / close[i-9] * 100
            if i >= 13:  # 12h return uses close[i-13]
                ret_12h[i] = (close[i] - close[i-13]) / close[i-13] * 100
            if i >= 25:  # 24h return uses close[i-25]
                ret_24h[i] = (close[i] - close[i-25]) / close[i-25] * 100
        
        # ATR percentage
        atr_pct = np.where(close > 0, atr14 / close * 100, 0)
        
        # SMA50 relative position
        vs_sma50 = np.where((sma50 > 0) & ~np.isnan(sma50), 
                           (close - sma50) / sma50 * 100, 0)
        
        # Volume ratio (20-period lookback)
        vol_ratio = np.full(len(volume), 1.0)
        for i in range(20, len(volume)):
            avg_vol = np.mean(volume[i-20:i])
            if avg_vol > 0:
                vol_ratio[i] = volume[i] / avg_vol
        
        # Mathematical features (for math tools)
        autocorr = np.full(len(close), 0.0)
        hurst = np.full(len(close), 0.5)
        entropy = np.full(len(close), 3.0)
        vpin = np.full(len(close), 0.0)
        skew = np.full(len(close), 0.0)
        kurtosis = np.full(len(close), 0.0)
        
        # Compute rolling features for bars with enough history
        for i in range(100, len(close)):
            window = close[i-100:i]
            returns = np.diff(window) / window[:-1]
            
            # Autocorrelation (lag 1)
            if len(returns) > 1:
                autocorr[i] = np.corrcoef(returns[:-1], returns[1:])[0,1] if len(returns) > 2 else 0
                if np.isnan(autocorr[i]):
                    autocorr[i] = 0
            
            # Hurst exponent (simple variance ratio)
            if len(returns) > 20:
                try:
                    v1 = np.var(returns)
                    v2 = np.var(returns[::2]) if len(returns) >= 4 else v1
                    if v1 > 0 and v2 > 0:
                        vr = v2 / v1 
                        hurst[i] = max(0, min(1, 0.5 + np.log(max(vr, 0.01)) / (2 * np.log(2))))
                except:
                    hurst[i] = 0.5
            
            # Shannon entropy 
            if len(returns) > 10:
                try:
                    hist, _ = np.histogram(returns, bins=15, density=True)
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        probs = hist / hist.sum()
                        entropy[i] = -np.sum(probs * np.log2(probs))
                except:
                    entropy[i] = 3.0
            
            # VPIN proxy
            if len(returns) > 20:
                try:
                    bv = np.where(returns > 0, volume[i-len(returns)+1:i], 0)
                    sv = np.where(returns < 0, volume[i-len(returns)+1:i], 0)
                    rb = np.sum(bv[-20:])
                    rs = np.sum(sv[-20:])
                    t = rb + rs
                    if t > 0:
                        vpin[i] = abs(rb - rs) / t
                except:
                    vpin[i] = 0.0
            
            # Skewness and kurtosis (50-bar window)
            if i >= 50:
                try:
                    ret_50 = returns[-50:] if len(returns) >= 50 else returns
                    if len(ret_50) > 5:
                        skew[i] = pd.Series(ret_50).skew()
                        kurtosis[i] = pd.Series(ret_50).kurtosis()
                        if np.isnan(skew[i]): skew[i] = 0
                        if np.isnan(kurtosis[i]): kurtosis[i] = 0
                except:
                    pass
        
        # Bar range and position
        bar_range = (high - low) / close * 100
        range_pos = np.where(high != low, (close - low) / (high - low), 0.5)
        
        # Red/green candles
        is_red = close < open_prices
        is_green = close > open_prices
        
        return {
            'close': close, 'high': high, 'low': low, 'volume': volume, 'open': open_prices,
            'rsi7': rsi7, 'rsi14': rsi14, 'sma50': sma50, 'atr14': atr14, 'atr_pct': atr_pct,
            'ret_4h': ret_4h, 'ret_8h': ret_8h, 'ret_12h': ret_12h, 'ret_24h': ret_24h,
            'vs_sma50': vs_sma50, 'vol_ratio': vol_ratio, 'bar_range': bar_range, 
            'range_pos': range_pos, 'is_red': is_red, 'is_green': is_green,
            'autocorr': autocorr, 'hurst': hurst, 'entropy': entropy, 'vpin': vpin,
            'skew': skew, 'kurtosis': kurtosis
        }
    
    def get_cross_pair_metrics(self, bar_idx):
        """Compute cross-pair metrics for market panic tools"""
        dropping_3pct = 0
        dropping_2pct = 0
        total_pairs = 0
        
        btc_ret_4h = 0.0
        
        for pair, df in self.data.items():
            if bar_idx < len(df):
                features = self.compute_features(df.iloc[:bar_idx+1])
                ret_4h = features['ret_4h'][bar_idx] if bar_idx < len(features['ret_4h']) else 0
                
                if abs(ret_4h) < 100:  # Sanity check
                    total_pairs += 1
                    if ret_4h < -3:
                        dropping_3pct += 1
                    if ret_4h < -2:
                        dropping_2pct += 1
                    
                    if pair == "BTCUSDT":
                        btc_ret_4h = ret_4h
        
        panic_3pct = (dropping_3pct / total_pairs * 100) if total_pairs > 0 else 0
        panic_2pct = (dropping_2pct / total_pairs * 100) if total_pairs > 0 else 0
        
        return panic_3pct, panic_2pct, btc_ret_4h
    
    def test_tool(self, pair, tool_name, signal_func, hold_hours=24, original_params=None, new_params=None):
        """Test a single tool on OOS data with fee adjustment"""
        df = self.data[pair]
        features = self.compute_features(df)
        
        signals = []
        oos_start = IN_SAMPLE_END
        
        # Generate signals on OOS period only
        for i in range(oos_start, len(df) - max(FORWARD_8H, FORWARD_24H)):
            # Get cross-pair metrics if needed
            panic_3pct, panic_2pct, btc_ret_4h = 0, 0, 0
            if any(x in tool_name for x in ['market_panic', 'blood_in_streets', 'btc_alt_spread']):
                panic_3pct, panic_2pct, btc_ret_4h = self.get_cross_pair_metrics(i)
            
            signal = signal_func(features, i, panic_3pct, panic_2pct, btc_ret_4h)
            if signal:
                signals.append((i, signal))
        
        if len(signals) == 0:
            return {
                'tool': tool_name, 'pair': pair, 'direction': 'long', 'signals': 0,
                'wr_8h': 0, 'wr_24h': 0, 'avg_ret_8h': 0, 'avg_ret_24h': 0, 
                'net_ret_8h': 0, 'net_ret_24h': 0, 'status': 'NO_SIGNALS',
                'original_params': original_params, 'new_params': new_params
            }
        
        # Evaluate forward returns
        returns_8h = []
        returns_24h = []
        
        for i, signal in signals:
            direction = signal.get('direction', 'long')
            
            # 8h forward return
            future_8h = i + FORWARD_8H
            if future_8h < len(features['close']):
                ret_8h = (features['close'][future_8h] - features['close'][i]) / features['close'][i]
                if direction == 'short':
                    ret_8h = -ret_8h
                returns_8h.append(ret_8h)
            
            # 24h forward return  
            future_24h = i + FORWARD_24H
            if future_24h < len(features['close']):
                ret_24h = (features['close'][future_24h] - features['close'][i]) / features['close'][i]
                if direction == 'short':
                    ret_24h = -ret_24h
                returns_24h.append(ret_24h)
        
        # Fee-adjusted performance
        def calc_stats(returns):
            if len(returns) == 0:
                return 0, 0, 0
            
            returns_after_fees = [r - FEE_PCT for r in returns]
            wins = sum(1 for r in returns_after_fees if r > 0)
            wr = wins / len(returns_after_fees) * 100
            avg_return = np.mean(returns_after_fees) * 100
            
            return len(returns_after_fees), wr, avg_return
        
        count_8h, wr_8h, avg_ret_8h = calc_stats(returns_8h)
        count_24h, wr_24h, avg_ret_24h = calc_stats(returns_24h)
        
        return {
            'tool': tool_name, 'pair': pair, 'direction': signal.get('direction', 'long'),
            'signals': len(signals), 'wr_8h': wr_8h, 'wr_24h': wr_24h,
            'avg_ret_8h': avg_ret_8h, 'avg_ret_24h': avg_ret_24h,
            'net_ret_8h': avg_ret_8h * len(signals), 'net_ret_24h': avg_ret_24h * len(signals),
            'status': 'PASS' if (wr_8h > 50 and avg_ret_8h > 0) or (wr_24h > 50 and avg_ret_24h > 0) else 'FAIL',
            'original_params': original_params, 'new_params': new_params
        }
    
    def optimize_tool_params(self, pair, tool_name, base_signal_func, param_ranges):
        """Optimize parameters on in-sample data, test best on OOS"""
        df = self.data[pair]
        features = self.compute_features(df)
        
        best_score = -999
        best_params = None
        best_result = None
        
        # Test parameter combinations on in-sample
        for params in param_ranges:
            signal_func = lambda f, i, p3, p2, btc: base_signal_func(f, i, p3, p2, btc, **params)
            
            signals = []
            for i in range(100, IN_SAMPLE_END - max(FORWARD_8H, FORWARD_24H)):
                panic_3pct, panic_2pct, btc_ret_4h = self.get_cross_pair_metrics(i)
                signal = signal_func(features, i, panic_3pct, panic_2pct, btc_ret_4h)
                if signal:
                    signals.append((i, signal))
            
            if len(signals) < 5:  # Need minimum signals
                continue
                
            # Evaluate on in-sample
            returns_24h = []
            for i, signal in signals:
                direction = signal.get('direction', 'long')
                future_24h = i + FORWARD_24H
                if future_24h < len(features['close']):
                    ret = (features['close'][future_24h] - features['close'][i]) / features['close'][i]
                    if direction == 'short':
                        ret = -ret
                    returns_24h.append(ret - FEE_PCT)
            
            if len(returns_24h) > 0:
                wins = sum(1 for r in returns_24h if r > 0)
                wr = wins / len(returns_24h) * 100
                avg_ret = np.mean(returns_24h) * 100
                score = wr * 0.7 + avg_ret * 0.3  # Weighted score
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        # Test best params on OOS
        if best_params:
            optimized_signal_func = lambda f, i, p3, p2, btc: base_signal_func(f, i, p3, p2, btc, **best_params)
            return self.test_tool(pair, tool_name, optimized_signal_func, new_params=best_params)
        
        return None
    
    def run_all_tests(self):
        """Test all 30 crash/bear/mean-reversion tools"""
        print(f"\nTesting 30 crash/bear/mean-reversion tools on OOS data...")
        print(f"OOS period: bars {IN_SAMPLE_END}-{TOTAL_BARS} across {len(PAIRS)} pairs\n")
        
        # Define all 30 tools
        tools = self.define_all_tools()
        
        for tool_name, tool_config in tools.items():
            print(f"Testing {tool_name}...")
            
            for pair in PAIRS:
                if pair not in self.data:
                    continue
                    
                signal_func = tool_config['signal_func']
                original_params = tool_config.get('original_params')
                
                # Test original parameters
                result = self.test_tool(pair, tool_name, signal_func, 
                                      tool_config.get('hold', 24), original_params)
                
                # If tool fails, try parameter optimization
                if result['status'] == 'FAIL' and 'param_ranges' in tool_config:
                    print(f"  {pair}: FAILED, optimizing parameters...")
                    optimized = self.optimize_tool_params(pair, tool_name, 
                                                        tool_config['base_signal_func'],
                                                        tool_config['param_ranges'])
                    if optimized and optimized['status'] == 'PASS':
                        result = optimized
                        print(f"    Optimization SUCCESS: {result['wr_24h']:.1f}% WR, {result['avg_ret_24h']:.2f}% avg")
                    else:
                        print(f"    Optimization FAILED")
                
                self.results.append(result)
                
                if result['signals'] > 0:
                    print(f"  {pair}: {result['signals']} signals, {result['wr_24h']:.1f}% WR, {result['avg_ret_24h']:.2f}% avg - {result['status']}")
    
    def define_all_tools(self):
        """Define all 30 crash/bear/mean-reversion tools"""
        return {
            # Crash/Dip Longs (1-24)
            'crash_buy': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_24h'][i] < -10 and f['rsi7'][i] < 20 else None,
                'original_params': {'ret_threshold': -10, 'rsi_threshold': 20},
                'param_ranges': [
                    {'ret_threshold': -8, 'rsi_threshold': 25},
                    {'ret_threshold': -12, 'rsi_threshold': 15},
                    {'ret_threshold': -10, 'rsi_threshold': 25}
                ],
                'base_signal_func': lambda f, i, p3, p2, btc, ret_threshold=-10, rsi_threshold=20: 
                    {'direction': 'long'} if f['ret_24h'][i] < ret_threshold and f['rsi7'][i] < rsi_threshold else None
            },
            
            'volatile_oversold': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['atr_pct'][i] > 3 and f['rsi7'][i] < 25 else None,
                'original_params': {'atr_threshold': 3, 'rsi_threshold': 25}
            },
            
            'dip_buy': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_4h'][i] < -3 else None,
                'original_params': {'ret_threshold': -3}
            },
            
            'mega_crash': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_24h'][i] < -15 else None,
                'original_params': {'ret_threshold': -15}
            },
            
            'flash_crash': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_12h'][i] < -10 else None,
                'original_params': {'ret_threshold': -10}
            },
            
            'quick_crash': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_8h'][i] < -10 else None,
                'original_params': {'ret_threshold': -10},
                'param_ranges': [
                    {'ret_threshold': -10, 'rsi_threshold': 40},
                    {'ret_threshold': -8, 'rsi_threshold': 35},
                    {'ret_threshold': -12, 'rsi_threshold': 45}
                ],
                'base_signal_func': lambda f, i, p3, p2, btc, ret_threshold=-10, rsi_threshold=100: 
                    {'direction': 'long'} if f['ret_8h'][i] < ret_threshold and f['rsi7'][i] < rsi_threshold else None,
                'hold': 8
            },
            
            'deep_dip_8h': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if -10 < f['ret_8h'][i] < -8 else None,
                'original_params': {'ret_min': -10, 'ret_max': -8}
            },
            
            'deep_dip_12h': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if -10 < f['ret_12h'][i] < -8 else None,
                'original_params': {'ret_min': -10, 'ret_max': -8}
            },
            
            'deep_dip_24h': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if -10 < f['ret_24h'][i] < -8 else None,
                'original_params': {'ret_min': -10, 'ret_max': -8}
            },
            
            'quick_dip': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_4h'][i] < -5 else None,
                'original_params': {'ret_threshold': -5}
            },
            
            'capitulation': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['vol_ratio'][i] >= 8 and f['is_red'][i] else None,
                'original_params': {'vol_threshold': 8}
            },
            
            'zscore_extreme': {
                'signal_func': lambda f, i, p3, p2, btc: self._zscore_signal(f, i),
                'original_params': {'zscore_threshold': -3}
            },
            
            'panic_close': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['bar_range'][i] > 3 and f['range_pos'][i] < 0.25 else None,
                'original_params': {'range_threshold': 3, 'position_threshold': 0.25}
            },
            
            'dist_exhaustion': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['skew'][i] < -1 and f['ret_4h'][i] < -3 else None,
                'original_params': {'skew_threshold': -1, 'ret_threshold': -3}
            },
            
            'fat_tail_revert': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['kurtosis'][i] > 5 and f['ret_4h'][i] < -3 else None,
                'original_params': {'kurt_threshold': 5, 'ret_threshold': -3}
            },
            
            'math_capitulation': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['skew'][i] < -1 and f['kurtosis'][i] > 3 and f['rsi7'][i] < 25 else None,
                'original_params': {'skew_threshold': -1, 'kurt_threshold': 3, 'rsi_threshold': 25}
            },
            
            'mega_align': {
                'signal_func': lambda f, i, p3, p2, btc: self._mega_align_signal(f, i),
                'original_params': {'rsi_threshold': 20, 'skew_threshold': -0.5, 'kurt_threshold': 2}
            },
            
            'efficiency_capitulation': {
                'signal_func': lambda f, i, p3, p2, btc: self._efficiency_signal(f, i),
                'original_params': {'efficiency_threshold': 0.4, 'range_pos_threshold': 0.10, 'vol_trend_threshold': 1.5, 'ret_threshold': -3}
            },
            
            'deceleration_buy': {
                'signal_func': lambda f, i, p3, p2, btc: self._deceleration_signal(f, i),
                'original_params': {'accel_threshold': 0.01, 'ret_threshold': -2}
            },
            
            'volume_climax': {
                'signal_func': lambda f, i, p3, p2, btc: self._volume_climax_signal(f, i),
                'original_params': {'vol_trend_threshold': 1.5, 'ret_threshold': -2}
            },
            
            'crash_neg_ac': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_24h'][i] < -10 and f['autocorr'][i] < -0.05 else None,
                'original_params': {'ret_threshold': -10, 'ac_threshold': -0.05}
            },
            
            'crash_mean_revert': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_24h'][i] < -8 and f['hurst'][i] < 0.45 else None,
                'original_params': {'ret_threshold': -8, 'hurst_threshold': 0.45}
            },
            
            'vpin_toxic': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['vpin'][i] > 0.7 and f['is_red'][i] else None,
                'original_params': {'vpin_threshold': 0.7}
            },
            
            'vpin_dip': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_8h'][i] < -5 and f['vpin'][i] > 0.5 else None,
                'original_params': {'ret_threshold': -5, 'vpin_threshold': 0.5}
            },
            
            'entropy_dip': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['entropy'][i] < 2.5 and f['ret_4h'][i] < -2 else None,
                'original_params': {'entropy_threshold': 2.5, 'ret_threshold': -2}
            },
            
            'triple_math': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['ret_8h'][i] < -5 and f['entropy'][i] < 2.5 and f['vpin'][i] > 0.5 else None,
                'original_params': {'ret_threshold': -5, 'entropy_threshold': 2.5, 'vpin_threshold': 0.5}
            },
            
            # Cross-pair crash/dip (25-27)
            'market_panic_90': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if p3 >= 90 else None,
                'original_params': {'panic_threshold': 90}
            },
            
            'market_panic_80': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if p3 >= 80 else None,
                'original_params': {'panic_threshold': 80}
            },
            
            'market_panic_70': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if p3 >= 70 else None,
                'original_params': {'panic_threshold': 70}
            },
            
            'blood_in_streets': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if p2 >= 70 and f['rsi7'][i] < 20 else None,
                'original_params': {'panic_threshold': 70, 'rsi_threshold': 20}
            },
            
            'btc_alt_spread': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if (f['ret_4h'][i] - btc) < -3 and f['rsi7'][i] < 35 else None,
                'original_params': {'spread_threshold': -3, 'rsi_threshold': 35}
            },
            
            # Mean Reversion (28-30)
            'relief_rally': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['rsi7'][i] > 75 and f['vs_sma50'][i] < 0 else None,
                'original_params': {'rsi_threshold': 75},
                'param_ranges': [
                    {'rsi_threshold': 70, 'sma_threshold': -2},
                    {'rsi_threshold': 75, 'sma_threshold': -5},
                    {'rsi_threshold': 80, 'sma_threshold': -1}
                ],
                'base_signal_func': lambda f, i, p3, p2, btc, rsi_threshold=75, sma_threshold=0: 
                    {'direction': 'long'} if f['rsi7'][i] > rsi_threshold and f['vs_sma50'][i] < sma_threshold else None,
                'hold': 12
            },
            
            'rsi_divergence': {
                'signal_func': lambda f, i, p3, p2, btc: self._rsi_divergence_signal(f, i),
                'original_params': {'lookback': 14, 'rsi_threshold': 35},
                'hold': 8
            },
            
            'whale_buy': {
                'signal_func': lambda f, i, p3, p2, btc: {'direction': 'long'} if f['vol_ratio'][i] >= 5 and f['is_green'][i] else None,
                'original_params': {'vol_threshold': 5}
            }
        }
    
    # Helper methods for complex signals
    def _zscore_signal(self, f, i):
        """Z-score extreme signal"""
        if i < 48:
            return None
        window = f['close'][i-48:i]
        mu = np.mean(window)
        sigma = np.std(window) 
        if sigma > 0:
            z = (f['close'][i] - mu) / sigma
            if z < -3:
                return {'direction': 'long'}
        return None
    
    def _mega_align_signal(self, f, i):
        """Mega align signal: RSI<20 + skew<-0.5 + kurt>2 + vol spike + 5+ down bars"""
        if i < 10:
            return None
        
        # Check basic conditions
        if not (f['rsi7'][i] < 20 and f['skew'][i] < -0.5 and f['kurtosis'][i] > 2):
            return None
        
        # Volume spike (>2x average)
        if f['vol_ratio'][i] < 2:
            return None
        
        # 5+ consecutive down bars
        down_count = 0
        for j in range(max(0, i-10), i):
            if j < len(f['close'])-1 and f['close'][j+1] < f['close'][j]:
                down_count += 1
            else:
                down_count = 0
                
        if down_count >= 5:
            return {'direction': 'long'}
        return None
    
    def _efficiency_signal(self, f, i):
        """Efficiency capitulation signal"""
        if i < 20:
            return None
        
        # Compute efficiency (price move / distance traveled)
        price_move = abs(f['close'][i] - f['close'][i-10])
        distance = sum(abs(f['close'][j] - f['close'][j-1]) for j in range(i-9, i+1))
        efficiency = price_move / distance if distance > 0 else 0
        
        # Volume trend (current vs 10-bar average) 
        vol_trend = f['vol_ratio'][i]
        
        if (efficiency > 0.4 and f['range_pos'][i] < 0.10 and 
            vol_trend > 1.5 and f['ret_4h'][i] < -3):
            return {'direction': 'long'}
        return None
    
    def _deceleration_signal(self, f, i):
        """Deceleration buy signal"""
        if i < 5:
            return None
        
        # Compute price acceleration
        if i >= 4:
            v1 = f['close'][i] - f['close'][i-2]  # Recent velocity
            v2 = f['close'][i-2] - f['close'][i-4]  # Prior velocity
            acceleration = v1 - v2
            
            if acceleration > 0.01 and f['ret_4h'][i] < -2:
                return {'direction': 'long'}
        return None
    
    def _volume_climax_signal(self, f, i):
        """Volume climax signal"""
        # Volume trend >1.5x and dip
        if f['vol_ratio'][i] > 1.5 and f['ret_4h'][i] < -2:
            return {'direction': 'long'}
        return None
    
    def _rsi_divergence_signal(self, f, i):
        """RSI divergence signal: price lower low but RSI higher low"""
        if i < 28:
            return None
        
        # Find recent and prior lows
        recent_price_low = np.min(f['close'][i-14:i+1])
        prior_price_low = np.min(f['close'][i-28:i-14])
        
        recent_rsi_low = np.min(f['rsi14'][i-14:i+1])
        prior_rsi_low = np.min(f['rsi14'][i-28:i-14])
        
        if (recent_price_low < prior_price_low and 
            recent_rsi_low > prior_rsi_low and 
            f['rsi14'][i] < 35):
            return {'direction': 'short'}  # As coded in bot
        return None
    
    def generate_report(self):
        """Generate comprehensive results report"""
        if not self.results:
            print("No results to report")
            return
        
        # Aggregate results by tool
        tool_summary = {}
        for result in self.results:
            tool = result['tool']
            if tool not in tool_summary:
                tool_summary[tool] = {
                    'signals': 0, 'total_wr_8h': 0, 'total_wr_24h': 0,
                    'total_ret_8h': 0, 'total_ret_24h': 0, 'pair_count': 0,
                    'passed_pairs': 0, 'direction': result['direction'],
                    'original_params': result['original_params'],
                    'new_params': result.get('new_params')
                }
            
            ts = tool_summary[tool]
            ts['signals'] += result['signals']
            ts['total_wr_8h'] += result['wr_8h'] * result['signals'] if result['signals'] > 0 else 0
            ts['total_wr_24h'] += result['wr_24h'] * result['signals'] if result['signals'] > 0 else 0
            ts['total_ret_8h'] += result['avg_ret_8h'] * result['signals'] if result['signals'] > 0 else 0
            ts['total_ret_24h'] += result['avg_ret_24h'] * result['signals'] if result['signals'] > 0 else 0
            ts['pair_count'] += 1
            if result['status'] == 'PASS':
                ts['passed_pairs'] += 1
        
        # Calculate aggregated metrics
        for tool in tool_summary:
            ts = tool_summary[tool]
            if ts['signals'] > 0:
                ts['avg_wr_8h'] = ts['total_wr_8h'] / ts['signals']
                ts['avg_wr_24h'] = ts['total_wr_24h'] / ts['signals']
                ts['avg_ret_8h'] = ts['total_ret_8h'] / ts['signals'] 
                ts['avg_ret_24h'] = ts['total_ret_24h'] / ts['signals']
                ts['net_ret_8h'] = ts['avg_ret_8h'] * ts['signals'] / 100
                ts['net_ret_24h'] = ts['avg_ret_24h'] * ts['signals'] / 100
            else:
                ts.update({'avg_wr_8h': 0, 'avg_wr_24h': 0, 'avg_ret_8h': 0, 
                          'avg_ret_24h': 0, 'net_ret_8h': 0, 'net_ret_24h': 0})
        
        # Generate markdown report
        report_lines = [
            "# CRASH/BEAR/MEAN-REVERSION Tools - OOS Validation Report",
            "",
            f"**Test Period:** Bars {IN_SAMPLE_END}-{TOTAL_BARS} (Out-of-Sample)", 
            f"**Pairs Tested:** {len(PAIRS)}",
            f"**Fee Adjustment:** {FEE_PCT*100:.2f}% round-trip subtracted from all returns",
            f"**Forward Returns:** +8 bars (8h) and +24 bars (24h)",
            "",
            "## Summary Table",
            "",
            "| Tool | Dir | Signals | WR 8h | WR 24h | Avg Ret 8h | Avg Ret 24h | Net 8h | Net 24h | Status | Fixes |",
            "|------|-----|---------|--------|--------|------------|-------------|--------|---------|--------|-------|",
        ]
        
        # Sort tools by net 24h return (best first)
        sorted_tools = sorted(tool_summary.items(), key=lambda x: x[1]['net_ret_24h'], reverse=True)
        
        for tool_name, ts in sorted_tools:
            status = "PASS" if (ts['avg_wr_8h'] > 50 and ts['avg_ret_8h'] > 0) or (ts['avg_wr_24h'] > 50 and ts['avg_ret_24h'] > 0) else "FAIL"
            if ts['signals'] == 0:
                status = "NO_SIGNALS"
            
            fixes = ""
            if ts['new_params'] and ts['new_params'] != ts['original_params']:
                fixes = "OPTIMIZED"
            
            report_lines.append(
                f"| {tool_name} | {ts['direction'].upper()} | {ts['signals']} | "
                f"{ts['avg_wr_8h']:.1f}% | {ts['avg_wr_24h']:.1f}% | "
                f"{ts['avg_ret_8h']:.2f}% | {ts['avg_ret_24h']:.2f}% | "
                f"{ts['net_ret_8h']:.2f} | {ts['net_ret_24h']:.2f} | "
                f"{status} | {fixes} |"
            )
        
        # Add parameter changes section
        report_lines.extend(["", "## Parameter Changes", ""])
        
        for tool_name, ts in sorted_tools:
            if ts['new_params'] and ts['new_params'] != ts['original_params']:
                report_lines.extend([
                    f"### {tool_name}",
                    f"- **Original:** {ts['original_params']}",
                    f"- **Optimized:** {ts['new_params']}",
                    ""
                ])
        
        # Write report
        report_content = "\n".join(report_lines)
        
        report_file = OUTPUT_DIR / "crash_tools_1h_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"\n{report_content}")
        print(f"\nFull report written to: {report_file}")

def main():
    """Main execution"""
    tester = CrashToolsTester()
    
    if len(tester.data) == 0:
        print("ERROR: No data files found. Check DATA_DIR path.")
        return
    
    print(f"Loaded {len(tester.data)} pairs with {TOTAL_BARS} bars each")
    print(f"Walk-forward split: bars 0-{IN_SAMPLE_END} (in-sample), {IN_SAMPLE_END}-{TOTAL_BARS} (OOS)")
    
    # Run all tests
    tester.run_all_tests()
    
    # Generate report
    tester.generate_report()

if __name__ == "__main__":
    main()