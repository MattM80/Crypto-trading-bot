#!/usr/bin/env python3
"""
Simplified test for crash/bear/mean-reversion tools.
Test the 30 tools with OOS validation on real 1h Binance data.
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
    
    # Initial averages
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    # Smoothed averages
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
    tr[0] = tr1[0]  # First bar uses high-low
    
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
    
    # Technical indicators
    rsi7 = calc_rsi(close, 7)
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
        if i >= 5:  # ret_4h = (close[i] - close[i-5]) / close[i-5] * 100
            ret_4h[i] = (close[i] - close[i-5]) / close[i-5] * 100
        if i >= 9:  # ret_8h = (close[i] - close[i-9]) / close[i-9] * 100
            ret_8h[i] = (close[i] - close[i-9]) / close[i-9] * 100
        if i >= 13: # ret_12h = (close[i] - close[i-13]) / close[i-13] * 100
            ret_12h[i] = (close[i] - close[i-13]) / close[i-13] * 100
        if i >= 25: # ret_24h = (close[i] - close[i-25]) / close[i-25] * 100
            ret_24h[i] = (close[i] - close[i-25]) / close[i-25] * 100
    
    # Volume ratio
    vol_ratio = np.full(len(volume), 1.0)
    for i in range(20, len(volume)):
        avg_vol = np.mean(volume[i-20:i])
        if avg_vol > 0:
            vol_ratio[i] = volume[i] / avg_vol
    
    # Red/green candles
    is_red = close < df['open'].values
    is_green = close > df['open'].values
    
    return {
        'close': close, 'high': high, 'low': low, 'volume': volume,
        'rsi7': rsi7, 'sma50': sma50, 'atr_pct': atr_pct, 'vs_sma50': vs_sma50,
        'ret_4h': ret_4h, 'ret_8h': ret_8h, 'ret_12h': ret_12h, 'ret_24h': ret_24h,
        'vol_ratio': vol_ratio, 'is_red': is_red, 'is_green': is_green
    }

def test_tool(pair_data, tool_name, signal_func, direction='long'):
    """Test a single tool on OOS data"""
    df = pair_data
    features = compute_features(df)
    
    signals = []
    oos_start = IN_SAMPLE_END
    
    # Generate signals on OOS period
    for i in range(oos_start, len(df) - max(FORWARD_8H, FORWARD_24H)):
        if signal_func(features, i):
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
            returns_8h.append(ret_8h - FEE_PCT)  # Fee-adjusted
        
        # 24h forward return  
        future_24h = i + FORWARD_24H
        if future_24h < len(features['close']):
            ret_24h = (features['close'][future_24h] - features['close'][i]) / features['close'][i]
            if direction == 'short':
                ret_24h = -ret_24h
            returns_24h.append(ret_24h - FEE_PCT)  # Fee-adjusted
    
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
    print("Testing 30 CRASH/BEAR/MEAN-REVERSION tools on OOS data...")
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
    
    # Define test tools (simplified subset for now)
    tools = {
        'crash_buy': {
            'func': lambda f, i: f['ret_24h'][i] < -10 and f['rsi7'][i] < 20,
            'direction': 'long',
            'params': 'ret_24h < -10 AND rsi7 < 20'
        },
        'volatile_oversold': {
            'func': lambda f, i: f['atr_pct'][i] > 3 and f['rsi7'][i] < 25,
            'direction': 'long',
            'params': 'atr_pct > 3 AND rsi7 < 25'
        },
        'dip_buy': {
            'func': lambda f, i: f['ret_4h'][i] < -3,
            'direction': 'long', 
            'params': 'ret_4h < -3'
        },
        'mega_crash': {
            'func': lambda f, i: f['ret_24h'][i] < -15,
            'direction': 'long',
            'params': 'ret_24h < -15'
        },
        'flash_crash': {
            'func': lambda f, i: f['ret_12h'][i] < -10,
            'direction': 'long',
            'params': 'ret_12h < -10'
        },
        'quick_crash': {
            'func': lambda f, i: f['ret_8h'][i] < -10,
            'direction': 'long',
            'params': 'ret_8h < -10'
        },
        'deep_dip_8h': {
            'func': lambda f, i: -10 < f['ret_8h'][i] < -8,
            'direction': 'long',
            'params': '-10 < ret_8h < -8'
        },
        'deep_dip_12h': {
            'func': lambda f, i: -10 < f['ret_12h'][i] < -8,
            'direction': 'long', 
            'params': '-10 < ret_12h < -8'
        },
        'deep_dip_24h': {
            'func': lambda f, i: -10 < f['ret_24h'][i] < -8,
            'direction': 'long',
            'params': '-10 < ret_24h < -8'
        },
        'quick_dip': {
            'func': lambda f, i: f['ret_4h'][i] < -5,
            'direction': 'long',
            'params': 'ret_4h < -5'
        },
        'capitulation': {
            'func': lambda f, i: f['vol_ratio'][i] >= 8 and f['is_red'][i],
            'direction': 'long',
            'params': 'vol_ratio >= 8 AND red candle'
        },
        'whale_buy': {
            'func': lambda f, i: f['vol_ratio'][i] >= 5 and f['is_green'][i],
            'direction': 'long',
            'params': 'vol_ratio >= 5 AND green candle'
        },
        'relief_rally': {
            'func': lambda f, i: f['rsi7'][i] > 75 and f['vs_sma50'][i] < 0,
            'direction': 'long',
            'params': 'rsi7 > 75 AND price < sma50'
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
            result = test_tool(df, tool_name, tool_config['func'], tool_config['direction'])
            
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
    
    # Generate report
    print("\n" + "="*120)
    print("CRASH/BEAR/MEAN-REVERSION TOOLS - OOS VALIDATION SUMMARY")
    print("="*120)
    print(f"Test Period: Bars {IN_SAMPLE_END}-{TOTAL_BARS} (Out-of-Sample)")
    print(f"Pairs Tested: {len(PAIRS)}")
    print(f"Fee Adjustment: {FEE_PCT*100:.2f}% round-trip subtracted from all returns")
    print(f"Forward Returns: +8 bars (8h) and +24 bars (24h)")
    print()
    
    # Sort by net 24h return
    results.sort(key=lambda x: x['net_ret_24h'], reverse=True)
    
    # Print table
    print(f"{'Tool':<20} {'Dir':<4} {'Signals':<7} {'WR 8h':<7} {'WR 24h':<7} {'Avg Ret 8h':<11} {'Avg Ret 24h':<11} {'Net 8h':<8} {'Net 24h':<8} {'Status':<10}")
    print("-" * 120)
    
    for r in results:
        print(f"{r['tool']:<20} {r['direction']:<4} {r['signals']:<7} "
              f"{r['wr_8h']:.1f}%{'':<3} {r['wr_24h']:.1f}%{'':<3} "
              f"{r['avg_ret_8h']:.2f}%{'':<7} {r['avg_ret_24h']:.2f}%{'':<7} "
              f"{r['net_ret_8h']:.2f}{'':<6} {r['net_ret_24h']:.2f}{'':<6} "
              f"{r['status']:<10}")
    
    # Write markdown report
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
    
    report_content = "\n".join(report_lines)
    
    # Save report
    report_file = OUTPUT_DIR / "crash_tools_1h_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\nFull report saved to: {report_file}")

if __name__ == "__main__":
    main()