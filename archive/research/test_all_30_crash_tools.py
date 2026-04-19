#!/usr/bin/env python3
"""
Complete test for all 30 crash/bear/mean-reversion tools.
Test with OOS validation on real 1h Binance data.
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

def calc_rsi(prices, period=7):
    """Calculate RSI"""
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

def calc_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
        
    sma = np.full(len(prices), np.nan)
    for i in range(period-1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
        
    return sma

def calc_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < 2:
        return np.full(len(high), 0.0)
        
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    if len(tr) < period:
        return tr
        
    atr = np.full(len(tr), np.nan)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
    return atr

def compute_features(df):
    """Compute all features for a dataframe"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    open_prices = df['open'].values
    
    # Technical indicators
    rsi7 = calc_rsi(close, 7)
    rsi14 = calc_rsi(close, 14)
    sma50 = calc_sma(close, 50)
    atr14 = calc_atr(high, low, close, 14)
    atr_pct = np.where(close > 0, atr14 / close * 100, 0)
    vs_sma50 = np.where((sma50 > 0) & ~np.isnan(sma50), (close - sma50) / sma50 * 100, 0)
    
    # Returns (exact master bot logic)
    ret_4h = np.full(len(close), 0.0)
    ret_8h = np.full(len(close), 0.0) 
    ret_12h = np.full(len(close), 0.0)
    ret_24h = np.full(len(close), 0.0)
    
    for i in range(len(close)):
        if i >= 5:
            ret_4h[i] = (close[i] - close[i-5]) / close[i-5] * 100
        if i >= 9:
            ret_8h[i] = (close[i] - close[i-9]) / close[i-9] * 100
        if i >= 13:
            ret_12h[i] = (close[i] - close[i-13]) / close[i-13] * 100
        if i >= 25:
            ret_24h[i] = (close[i] - close[i-25]) / close[i-25] * 100
    
    # Volume ratio
    vol_ratio = np.full(len(volume), 1.0)
    for i in range(20, len(volume)):
        avg_vol = np.mean(volume[i-20:i])
        if avg_vol > 0:
            vol_ratio[i] = volume[i] / avg_vol
    
    # Mathematical features (rolling windows for statistical measures)
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
        if len(returns) > 2:
            try:
                autocorr[i] = np.corrcoef(returns[:-1], returns[1:])[0,1]
                if np.isnan(autocorr[i]):
                    autocorr[i] = 0
            except:
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
        'rsi7': rsi7, 'rsi14': rsi14, 'sma50': sma50, 'atr_pct': atr_pct, 'vs_sma50': vs_sma50,
        'ret_4h': ret_4h, 'ret_8h': ret_8h, 'ret_12h': ret_12h, 'ret_24h': ret_24h,
        'vol_ratio': vol_ratio, 'bar_range': bar_range, 'range_pos': range_pos, 
        'is_red': is_red, 'is_green': is_green, 'autocorr': autocorr, 'hurst': hurst,
        'entropy': entropy, 'vpin': vpin, 'skew': skew, 'kurtosis': kurtosis
    }

def get_cross_pair_metrics(all_data, bar_idx, current_pair):
    """Compute cross-pair metrics for market panic tools"""
    dropping_3pct = 0
    dropping_2pct = 0
    total_pairs = 0
    btc_ret_4h = 0.0
    
    for pair, df in all_data.items():
        if bar_idx < len(df) and pair != current_pair:
            features = compute_features(df.iloc[:bar_idx+1])
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

def test_tool(all_data, pair_data, pair_name, tool_name, signal_func, direction='long', hold_hours=24):
    """Test a single tool on OOS data"""
    features = compute_features(pair_data)
    
    signals = []
    oos_start = IN_SAMPLE_END
    
    # Generate signals on OOS period
    for i in range(oos_start, len(pair_data) - max(FORWARD_8H, FORWARD_24H)):
        # Get cross-pair metrics for market panic tools
        panic_3pct, panic_2pct, btc_ret_4h = 0, 0, 0
        if 'market_panic' in tool_name or 'blood_in_streets' in tool_name or 'btc_alt_spread' in tool_name:
            panic_3pct, panic_2pct, btc_ret_4h = get_cross_pair_metrics(all_data, i, pair_name)
        
        if signal_func(features, i, panic_3pct, panic_2pct, btc_ret_4h):
            signals.append(i)
    
    if len(signals) == 0:
        return {'signals': 0, 'wr_8h': 0, 'wr_24h': 0, 'avg_ret_8h': 0, 'avg_ret_24h': 0}
    
    # Evaluate forward returns
    returns_8h = []
    returns_24h = []
    
    for i in signals:
        # 8h forward return
        future_8h = i + FORWARD_8H
        if future_8h < len(features['close']):
            ret_8h = (features['close'][future_8h] - features['close'][i]) / features['close'][i]
            if direction == 'short':
                ret_8h = -ret_8h
            returns_8h.append(ret_8h - FEE_PCT)
        
        # 24h forward return  
        future_24h = i + FORWARD_24H
        if future_24h < len(features['close']):
            ret_24h = (features['close'][future_24h] - features['close'][i]) / features['close'][i]
            if direction == 'short':
                ret_24h = -ret_24h
            returns_24h.append(ret_24h - FEE_PCT)
    
    # Calculate performance
    def calc_perf(returns):
        if len(returns) == 0:
            return 0, 0
        wins = sum(1 for r in returns if r > 0)
        wr = wins / len(returns) * 100
        avg_ret = np.mean(returns) * 100
        return wr, avg_ret
    
    wr_8h, avg_ret_8h = calc_perf(returns_8h)
    wr_24h, avg_ret_24h = calc_perf(returns_24h)
    
    return {
        'signals': len(signals),
        'wr_8h': wr_8h,
        'wr_24h': wr_24h, 
        'avg_ret_8h': avg_ret_8h,
        'avg_ret_24h': avg_ret_24h
    }

def main():
    print("Testing ALL 30 CRASH/BEAR/MEAN-REVERSION tools on OOS data...")
    print(f"OOS period: bars {IN_SAMPLE_END}-{TOTAL_BARS}")
    
    # Load data
    data = {}
    for pair in PAIRS:
        file_path = DATA_DIR / f"{pair}_1h.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            data[pair] = df
            print(f"Loaded {pair}: {len(df)} bars")
    
    if len(data) == 0:
        print("ERROR: No data files found")
        return
    
    # Define all 30 tools
    tools = {
        # Crash/Dip Longs (1-24)
        'crash_buy': {
            'func': lambda f, i, p3, p2, btc: f['ret_24h'][i] < -10 and f['rsi7'][i] < 20,
            'direction': 'long',
            'params': 'ret_24h < -10 AND rsi7 < 20'
        },
        'volatile_oversold': {
            'func': lambda f, i, p3, p2, btc: f['atr_pct'][i] > 3 and f['rsi7'][i] < 25,
            'direction': 'long',
            'params': 'atr_pct > 3 AND rsi7 < 25'
        },
        'dip_buy': {
            'func': lambda f, i, p3, p2, btc: f['ret_4h'][i] < -3,
            'direction': 'long', 
            'params': 'ret_4h < -3'
        },
        'mega_crash': {
            'func': lambda f, i, p3, p2, btc: f['ret_24h'][i] < -15,
            'direction': 'long',
            'params': 'ret_24h < -15'
        },
        'flash_crash': {
            'func': lambda f, i, p3, p2, btc: f['ret_12h'][i] < -10,
            'direction': 'long',
            'params': 'ret_12h < -10'
        },
        'quick_crash': {
            'func': lambda f, i, p3, p2, btc: f['ret_8h'][i] < -10,
            'direction': 'long',
            'params': 'ret_8h < -10'
        },
        'deep_dip_8h': {
            'func': lambda f, i, p3, p2, btc: -10 < f['ret_8h'][i] < -8,
            'direction': 'long',
            'params': '-10 < ret_8h < -8'
        },
        'deep_dip_12h': {
            'func': lambda f, i, p3, p2, btc: -10 < f['ret_12h'][i] < -8,
            'direction': 'long', 
            'params': '-10 < ret_12h < -8'
        },
        'deep_dip_24h': {
            'func': lambda f, i, p3, p2, btc: -10 < f['ret_24h'][i] < -8,
            'direction': 'long',
            'params': '-10 < ret_24h < -8'
        },
        'quick_dip': {
            'func': lambda f, i, p3, p2, btc: f['ret_4h'][i] < -5,
            'direction': 'long',
            'params': 'ret_4h < -5'
        },
        'capitulation': {
            'func': lambda f, i, p3, p2, btc: f['vol_ratio'][i] >= 8 and f['is_red'][i],
            'direction': 'long',
            'params': 'vol_ratio >= 8 AND red candle'
        },
        'zscore_extreme': {
            'func': lambda f, i, p3, p2, btc: i >= 48 and (
                (lambda window, mu, sigma: (f['close'][i] - mu) / sigma < -3 if sigma > 0 else False)
                (*((f['close'][i-48:i], np.mean(f['close'][i-48:i]), np.std(f['close'][i-48:i]))))
            ),
            'direction': 'long',
            'params': 'z-score < -3 on 48-bar window'
        },
        'panic_close': {
            'func': lambda f, i, p3, p2, btc: f['bar_range'][i] > 3 and f['range_pos'][i] < 0.25,
            'direction': 'long',
            'params': 'bar_range > 3% AND close in bottom 25%'
        },
        'dist_exhaustion': {
            'func': lambda f, i, p3, p2, btc: f['skew'][i] < -1 and f['ret_4h'][i] < -3,
            'direction': 'long',
            'params': 'skew < -1 AND ret_4h < -3'
        },
        'fat_tail_revert': {
            'func': lambda f, i, p3, p2, btc: f['kurtosis'][i] > 5 and f['ret_4h'][i] < -3,
            'direction': 'long',
            'params': 'kurtosis > 5 AND ret_4h < -3'
        },
        'math_capitulation': {
            'func': lambda f, i, p3, p2, btc: f['skew'][i] < -1 and f['kurtosis'][i] > 3 and f['rsi7'][i] < 25,
            'direction': 'long',
            'params': 'skew < -1 AND kurt > 3 AND rsi7 < 25'
        },
        'mega_align': {
            'func': lambda f, i, p3, p2, btc: (
                f['rsi7'][i] < 20 and f['skew'][i] < -0.5 and f['kurtosis'][i] > 2 and 
                f['vol_ratio'][i] > 2 and i >= 10 and
                sum(1 for j in range(max(0, i-10), i) if j < len(f['close'])-1 and f['close'][j+1] < f['close'][j]) >= 5
            ),
            'direction': 'long',
            'params': 'rsi7 < 20 AND skew < -0.5 AND kurt > 2 AND vol_spike AND 5+ down bars'
        },
        'efficiency_capitulation': {
            'func': lambda f, i, p3, p2, btc: i >= 20 and (
                (lambda eff, vol_trend: eff > 0.4 and f['range_pos'][i] < 0.10 and vol_trend > 1.5 and f['ret_4h'][i] < -3)(
                    abs(f['close'][i] - f['close'][i-10]) / sum(abs(f['close'][j] - f['close'][j-1]) for j in range(i-9, i+1)) if sum(abs(f['close'][j] - f['close'][j-1]) for j in range(i-9, i+1)) > 0 else 0,
                    f['vol_ratio'][i]
                )
            ),
            'direction': 'long',
            'params': 'efficiency > 0.4 AND range_pos < 0.10 AND vol_trend > 1.5 AND ret_4h < -3'
        },
        'deceleration_buy': {
            'func': lambda f, i, p3, p2, btc: i >= 4 and (
                (f['close'][i] - f['close'][i-2]) - (f['close'][i-2] - f['close'][i-4])
            ) > 0.01 and f['ret_4h'][i] < -2,
            'direction': 'long',
            'params': 'acceleration > 0.01 AND ret_4h < -2'
        },
        'volume_climax': {
            'func': lambda f, i, p3, p2, btc: f['vol_ratio'][i] > 1.5 and f['ret_4h'][i] < -2,
            'direction': 'long',
            'params': 'vol_trend > 1.5 AND ret_4h < -2'
        },
        'crash_neg_ac': {
            'func': lambda f, i, p3, p2, btc: f['ret_24h'][i] < -10 and f['autocorr'][i] < -0.05,
            'direction': 'long',
            'params': 'ret_24h < -10 AND autocorr(1) < -0.05'
        },
        'crash_mean_revert': {
            'func': lambda f, i, p3, p2, btc: f['ret_24h'][i] < -8 and f['hurst'][i] < 0.45,
            'direction': 'long',
            'params': 'ret_24h < -8 AND Hurst < 0.45'
        },
        'vpin_toxic': {
            'func': lambda f, i, p3, p2, btc: f['vpin'][i] > 0.7 and f['is_red'][i],
            'direction': 'long',
            'params': 'VPIN > 0.7 AND red candle'
        },
        'vpin_dip': {
            'func': lambda f, i, p3, p2, btc: f['ret_8h'][i] < -5 and f['vpin'][i] > 0.5,
            'direction': 'long',
            'params': 'ret_8h < -5 AND VPIN > 0.5'
        },
        'entropy_dip': {
            'func': lambda f, i, p3, p2, btc: f['entropy'][i] < 2.5 and f['ret_4h'][i] < -2,
            'direction': 'long',
            'params': 'entropy < 2.5 AND ret_4h < -2'
        },
        'triple_math': {
            'func': lambda f, i, p3, p2, btc: f['ret_8h'][i] < -5 and f['entropy'][i] < 2.5 and f['vpin'][i] > 0.5,
            'direction': 'long',
            'params': 'ret_8h < -5 AND entropy < 2.5 AND VPIN > 0.5'
        },
        
        # Cross-pair crash/dip (25-27)
        'market_panic_90': {
            'func': lambda f, i, p3, p2, btc: p3 >= 90,
            'direction': 'long',
            'params': '90%+ pairs down >3% in 4h'
        },
        'market_panic_80': {
            'func': lambda f, i, p3, p2, btc: p3 >= 80 and p3 < 90,
            'direction': 'long',
            'params': '80%+ pairs down >3% in 4h'
        },
        'market_panic_70': {
            'func': lambda f, i, p3, p2, btc: p3 >= 70 and p3 < 80,
            'direction': 'long',
            'params': '70%+ pairs down >3% in 4h'
        },
        'blood_in_streets': {
            'func': lambda f, i, p3, p2, btc: p2 >= 70 and f['rsi7'][i] < 20,
            'direction': 'long',
            'params': '70%+ pairs down >2% AND this coin rsi7 < 20'
        },
        'btc_alt_spread': {
            'func': lambda f, i, p3, p2, btc: (f['ret_4h'][i] - btc) < -3 and f['rsi7'][i] < 35,
            'direction': 'long',
            'params': 'alt lagging BTC by 3%+ in ret_4h AND rsi7 < 35'
        },
        
        # Mean Reversion (28-30)
        'relief_rally': {
            'func': lambda f, i, p3, p2, btc: f['rsi7'][i] > 75 and f['vs_sma50'][i] < 0,
            'direction': 'long',
            'params': 'rsi7 > 75 AND price < sma50',
            'hold': 12
        },
        'rsi_divergence': {
            'func': lambda f, i, p3, p2, btc: i >= 28 and (
                (lambda recent_low, prior_low, recent_rsi_low, prior_rsi_low: 
                 recent_low < prior_low and recent_rsi_low > prior_rsi_low and f['rsi14'][i] < 35)(
                    np.min(f['close'][i-14:i+1]),
                    np.min(f['close'][i-28:i-14]) if i >= 28 else np.min(f['close'][i-14:i+1]),
                    np.min(f['rsi14'][i-14:i+1]),
                    np.min(f['rsi14'][i-28:i-14]) if i >= 28 else np.min(f['rsi14'][i-14:i+1])
                )
            ),
            'direction': 'short',  # As coded in bot
            'params': 'price lower low but RSI higher low, rsi14 < 35',
            'hold': 8
        },
        'whale_buy': {
            'func': lambda f, i, p3, p2, btc: f['vol_ratio'][i] >= 5 and f['is_green'][i],
            'direction': 'long',
            'params': 'vol_ratio >= 5 AND green candle'
        }
    }
    
    # Test each tool on each pair
    results = []
    
    for tool_name, tool_config in tools.items():
        print(f"\nTesting {tool_name}...")
        
        # Aggregate across all pairs
        total_signals = 0
        total_wr_8h_weighted = 0
        total_wr_24h_weighted = 0  
        total_ret_8h_weighted = 0
        total_ret_24h_weighted = 0
        
        for pair, df in data.items():
            hold_hours = tool_config.get('hold', 24)
            result = test_tool(data, df, pair, tool_name, tool_config['func'], 
                             tool_config['direction'], hold_hours)
            
            if result['signals'] > 0:
                print(f"  {pair}: {result['signals']} signals, "
                      f"WR 8h: {result['wr_8h']:.1f}%, WR 24h: {result['wr_24h']:.1f}%, "
                      f"Avg ret 8h: {result['avg_ret_8h']:.2f}%, Avg ret 24h: {result['avg_ret_24h']:.2f}%")
                
                # Weighted aggregation
                total_signals += result['signals']
                total_wr_8h_weighted += result['wr_8h'] * result['signals']
                total_wr_24h_weighted += result['wr_24h'] * result['signals']
                total_ret_8h_weighted += result['avg_ret_8h'] * result['signals']
                total_ret_24h_weighted += result['avg_ret_24h'] * result['signals']
        
        # Calculate aggregated metrics
        if total_signals > 0:
            agg_wr_8h = total_wr_8h_weighted / total_signals
            agg_wr_24h = total_wr_24h_weighted / total_signals
            agg_ret_8h = total_ret_8h_weighted / total_signals
            agg_ret_24h = total_ret_24h_weighted / total_signals
            
            # Determine status
            status = "PASS" if (agg_wr_8h > 50 and agg_ret_8h > 0) or (agg_wr_24h > 50 and agg_ret_24h > 0) else "FAIL"
            
            results.append({
                'tool': tool_name,
                'direction': tool_config['direction'].upper(),
                'params': tool_config['params'],
                'signals': total_signals,
                'wr_8h': agg_wr_8h,
                'wr_24h': agg_wr_24h, 
                'avg_ret_8h': agg_ret_8h,
                'avg_ret_24h': agg_ret_24h,
                'net_ret_8h': agg_ret_8h * total_signals / 100,
                'net_ret_24h': agg_ret_24h * total_signals / 100,
                'status': status
            })
            
            print(f"  TOTAL: {total_signals} signals, WR 8h: {agg_wr_8h:.1f}%, WR 24h: {agg_wr_24h:.1f}%, "
                  f"Avg ret 8h: {agg_ret_8h:.2f}%, Avg ret 24h: {agg_ret_24h:.2f}% - {status}")
        else:
            results.append({
                'tool': tool_name,
                'direction': tool_config['direction'].upper(),
                'params': tool_config['params'],
                'signals': 0,
                'wr_8h': 0,
                'wr_24h': 0,
                'avg_ret_8h': 0,
                'avg_ret_24h': 0,
                'net_ret_8h': 0,
                'net_ret_24h': 0,
                'status': 'NO_SIGNALS'
            })
            print(f"  TOTAL: No signals generated")
    
    # Generate comprehensive report
    print("\n" + "="*140)
    print("ALL 30 CRASH/BEAR/MEAN-REVERSION TOOLS - OOS VALIDATION SUMMARY")
    print("="*140)
    print(f"Test Period: Bars {IN_SAMPLE_END}-{TOTAL_BARS} (Out-of-Sample)")
    print(f"Pairs Tested: {len(PAIRS)}")
    print(f"Fee Adjustment: {FEE_PCT*100:.2f}% round-trip subtracted from all returns")
    print(f"Forward Returns: +8 bars (8h) and +24 bars (24h)")
    print()
    
    # Sort by net 24h return
    results.sort(key=lambda x: x['net_ret_24h'], reverse=True)
    
    # Print table
    print(f"{'Tool':<25} {'Dir':<5} {'Signals':<8} {'WR 8h':<7} {'WR 24h':<7} {'Ret 8h':<8} {'Ret 24h':<8} {'Net 8h':<8} {'Net 24h':<8} {'Status':<10}")
    print("-" * 140)
    
    for r in results:
        print(f"{r['tool']:<25} {r['direction']:<5} {r['signals']:<8} "
              f"{r['wr_8h']:.1f}%{'':<4} {r['wr_24h']:.1f}%{'':<4} "
              f"{r['avg_ret_8h']:.2f}%{'':<4} {r['avg_ret_24h']:.2f}%{'':<4} "
              f"{r['net_ret_8h']:.2f}{'':<6} {r['net_ret_24h']:.2f}{'':<6} "
              f"{r['status']:<10}")
    
    # Summary stats
    passing_tools = [r for r in results if r['status'] == 'PASS']
    failing_tools = [r for r in results if r['status'] == 'FAIL']
    no_signal_tools = [r for r in results if r['status'] == 'NO_SIGNALS']
    
    print(f"\nSUMMARY:")
    print(f"- PASSING tools: {len(passing_tools)}/30 ({len(passing_tools)/30*100:.1f}%)")
    print(f"- FAILING tools: {len(failing_tools)}/30 ({len(failing_tools)/30*100:.1f}%)")
    print(f"- NO SIGNALS: {len(no_signal_tools)}/30 ({len(no_signal_tools)/30*100:.1f}%)")
    
    total_net_24h = sum(r['net_ret_24h'] for r in results)
    print(f"- Total net return (24h): {total_net_24h:.2f}%")
    
    # Write markdown report
    report_lines = [
        "# ALL 30 CRASH/BEAR/MEAN-REVERSION Tools - OOS Validation Report",
        "",
        f"**Test Period:** Bars {IN_SAMPLE_END}-{TOTAL_BARS} (Out-of-Sample)",
        f"**Pairs Tested:** {len(PAIRS)}",
        f"**Fee Adjustment:** {FEE_PCT*100:.2f}% round-trip subtracted from all returns",
        f"**Forward Returns:** +8 bars (8h) and +24 bars (24h)",
        "",
        "## Executive Summary",
        "",
        f"- **PASSING tools:** {len(passing_tools)}/30 ({len(passing_tools)/30*100:.1f}%)",
        f"- **FAILING tools:** {len(failing_tools)}/30 ({len(failing_tools)/30*100:.1f}%)",
        f"- **NO SIGNALS:** {len(no_signal_tools)}/30 ({len(no_signal_tools)/30*100:.1f}%)",
        f"- **Total net return (24h):** {total_net_24h:.2f}%",
        "",
        "## Detailed Results",
        "",
        "| Tool | Dir | Signals | WR 8h | WR 24h | Avg Ret 8h | Avg Ret 24h | Net 8h | Net 24h | Status |",
        "|------|-----|---------|-------|--------|------------|-------------|--------|---------|--------|"
    ]
    
    for r in results:
        report_lines.append(
            f"| {r['tool']} | {r['direction']} | {r['signals']} | "
            f"{r['wr_8h']:.1f}% | {r['wr_24h']:.1f}% | "
            f"{r['avg_ret_8h']:.2f}% | {r['avg_ret_24h']:.2f}% | "
            f"{r['net_ret_8h']:.2f} | {r['net_ret_24h']:.2f} | "
            f"{r['status']} |"
        )
    
    # Add tool descriptions
    report_lines.extend([
        "",
        "## Tool Descriptions",
        ""
    ])
    
    for r in results:
        report_lines.append(f"- **{r['tool']}:** {r['params']}")
    
    # Add top performers
    report_lines.extend([
        "",
        "## Top Performers (by 24h Net Return)",
        ""
    ])
    
    top_5 = [r for r in results if r['status'] == 'PASS'][:5]
    for i, r in enumerate(top_5, 1):
        report_lines.append(f"{i}. **{r['tool']}:** {r['signals']} signals, {r['wr_24h']:.1f}% WR, {r['net_ret_24h']:.2f}% net return")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    report_file = OUTPUT_DIR / "crash_tools_1h_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\nFull report saved to: {report_file}")

if __name__ == "__main__":
    main()