#!/usr/bin/env python3
"""
Optimized Out-of-Sample Validation
Tests the most important trading tools efficiently.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Pairs to test (focusing on major pairs for speed)
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
DATA_DIR = Path("data/binance_historical")

def calc_rsi_vectorized(prices, period=7):
    """Vectorized RSI calculation for better performance."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use pandas for rolling calculations (faster)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.rolling(window=period).mean()
    avg_loss = loss_series.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Pad to match original length
    return np.concatenate([[50.0], rsi.fillna(50).values])

def process_pair_signals(df, pair_name):
    """Process all signals for a single pair efficiently."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    n = len(close)
    
    # Pre-calculate all indicators
    rsi7 = calc_rsi_vectorized(close, 7)
    sma50 = pd.Series(close).rolling(window=50).mean().fillna(method='bfill').values
    
    # Pre-calculate all returns
    ret_4h = np.zeros(n)
    ret_8h = np.zeros(n)  
    ret_12h = np.zeros(n)
    ret_24h = np.zeros(n)
    
    for i in range(1, n):
        if i >= 2:
            ret_4h[i] = (close[i] - close[i-2]) / close[i-2] * 100
        if i >= 3:
            ret_8h[i] = (close[i] - close[i-3]) / close[i-3] * 100
        if i >= 4:
            ret_12h[i] = (close[i] - close[i-4]) / close[i-4] * 100
        if i >= 7:
            ret_24h[i] = (close[i] - close[i-7]) / close[i-7] * 100
    
    # ATR calculation (vectorized)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                              np.abs(low[1:] - close[:-1])))
    atr_14 = pd.Series(tr).rolling(window=14).mean().values
    atr_pct = np.concatenate([[0], atr_14]) / close * 100
    
    # Collect signals
    signals = []
    
    print(f"  Processing {pair_name}...")
    
    # Process each bar
    for i in range(50, n - 10):  # Leave room for forward returns
        
        # Tool 2: Crash Buy
        if ret_24h[i] < -10 and rsi7[i] < 20:
            signals.append({
                'pair': pair_name,
                'tool': 'crash_buy',
                'direction': 'long',
                'bar': i,
                'price': close[i],
                'trigger': f"{ret_24h[i]:.1f}% drop, RSI={rsi7[i]:.1f}"
            })
        
        # Tool 3: Volatile Oversold
        if atr_pct[i] > 3 and rsi7[i] < 25:
            signals.append({
                'pair': pair_name,
                'tool': 'volatile_oversold',
                'direction': 'long',
                'bar': i,
                'price': close[i],
                'trigger': f"ATR={atr_pct[i]:.1f}%, RSI={rsi7[i]:.1f}"
            })
        
        # Tool 4: Relief Rally
        vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100 if sma50[i] > 0 else 0
        if rsi7[i] > 75 and vs_sma50 < 0:
            signals.append({
                'pair': pair_name,
                'tool': 'relief_rally',
                'direction': 'long',
                'bar': i,
                'price': close[i],
                'trigger': f"RSI={rsi7[i]:.1f}, below SMA50"
            })
        
        # Tool 6: Dip Buy
        if ret_4h[i] < -3:
            signals.append({
                'pair': pair_name,
                'tool': 'dip_buy',
                'direction': 'long',
                'bar': i,
                'price': close[i],
                'trigger': f"{ret_4h[i]:.1f}% dip 4h"
            })
        
        # Tool 7: RSI Pump Short
        if rsi7[i] > 80 and ret_12h[i] > 8:
            signals.append({
                'pair': pair_name,
                'tool': 'rsi_pump_short',
                'direction': 'short',
                'bar': i,
                'price': close[i],
                'trigger': f"RSI={rsi7[i]:.1f}, +{ret_12h[i]:.1f}% 12h"
            })
        
        # Tool 8: Mega Crash
        if ret_24h[i] < -15:
            signals.append({
                'pair': pair_name,
                'tool': 'mega_crash',
                'direction': 'long',
                'bar': i,
                'price': close[i],
                'trigger': f"{ret_24h[i]:.1f}% crash 24h"
            })
        
        # Tool 9: Flash Crash
        if ret_12h[i] < -10:
            signals.append({
                'pair': pair_name,
                'tool': 'flash_crash',
                'direction': 'long',
                'bar': i,
                'price': close[i],
                'trigger': f"{ret_12h[i]:.1f}% crash 12h"
            })
        
        # Tool 10: Quick Crash  
        if ret_8h[i] < -10:
            signals.append({
                'pair': pair_name,
                'tool': 'quick_crash',
                'direction': 'long',
                'bar': i,
                'price': close[i],
                'trigger': f"{ret_8h[i]:.1f}% crash 8h"
            })
    
    # Calculate forward returns for all signals
    for signal in signals:
        i = signal['bar']
        
        # 8h forward (2 bars)
        if i + 2 < n and close[i] > 0:
            ret_8h = (close[i + 2] - close[i]) / close[i] * 100
            signal['ret_8h_forward'] = np.clip(ret_8h, -100, 1000)  # Clip extreme values
        else:
            signal['ret_8h_forward'] = 0
            
        # 24h forward (6 bars)
        if i + 6 < n and close[i] > 0:
            ret_24h = (close[i + 6] - close[i]) / close[i] * 100
            signal['ret_24h_forward'] = np.clip(ret_24h, -100, 1000)  # Clip extreme values
        else:
            signal['ret_24h_forward'] = 0
    
    return signals

def analyze_tool_performance(all_signals):
    """Analyze performance by tool."""
    tool_stats = {}
    
    for signal in all_signals:
        tool = signal['tool']
        direction = signal['direction']
        
        if tool not in tool_stats:
            tool_stats[tool] = {
                'direction': direction,
                'signals': []
            }
        
        tool_stats[tool]['signals'].append(signal)
    
    # Calculate metrics
    results = []
    
    for tool, data in tool_stats.items():
        signals = data['signals']
        direction = data['direction']
        n_signals = len(signals)
        
        if n_signals == 0:
            continue
        
        # Get forward returns
        returns_8h = [s['ret_8h_forward'] for s in signals]
        returns_24h = [s['ret_24h_forward'] for s in signals]
        
        # Adjust for direction (shorts profit from negative moves)
        if direction == 'short':
            adj_ret_8h = [-r for r in returns_8h]
            adj_ret_24h = [-r for r in returns_24h]
        else:
            adj_ret_8h = returns_8h
            adj_ret_24h = returns_24h
        
        # Calculate metrics (handle NaN values)
        adj_ret_8h_clean = [r for r in adj_ret_8h if not np.isnan(r)]
        adj_ret_24h_clean = [r for r in adj_ret_24h if not np.isnan(r)]
        
        wins_8h = sum(1 for r in adj_ret_8h_clean if r > 0)
        wr_8h = wins_8h / len(adj_ret_8h_clean) * 100 if len(adj_ret_8h_clean) > 0 else 0
        avg_ret_8h = np.mean(adj_ret_8h_clean) if len(adj_ret_8h_clean) > 0 else 0
        net_exp_8h = avg_ret_8h * (wr_8h / 100)
        
        wins_24h = sum(1 for r in adj_ret_24h_clean if r > 0)
        wr_24h = wins_24h / len(adj_ret_24h_clean) * 100 if len(adj_ret_24h_clean) > 0 else 0
        avg_ret_24h = np.mean(adj_ret_24h_clean) if len(adj_ret_24h_clean) > 0 else 0
        net_exp_24h = avg_ret_24h * (wr_24h / 100)
        
        # Pass/Fail criteria
        min_wr = 50 if direction == 'long' else 45
        min_signals = 10
        
        pass_condition = (net_exp_24h > 0 and 
                         wr_24h > min_wr and 
                         n_signals >= min_signals)
        
        status = "PASS" if pass_condition else "FAIL"
        
        results.append({
            'tool': tool,
            'direction': direction,
            'n_signals': n_signals,
            'wr_8h': wr_8h,
            'avg_ret_8h': avg_ret_8h,
            'net_exp_8h': net_exp_8h,
            'wr_24h': wr_24h,
            'avg_ret_24h': avg_ret_24h,
            'net_exp_24h': net_exp_24h,
            'status': status
        })
    
    return results

def main():
    """Run optimized validation."""
    print("Optimized Out-of-Sample Validation")
    print("=" * 50)
    
    all_signals = []
    
    for pair in PAIRS:
        file_path = DATA_DIR / f"{pair}_4h.csv"
        
        if not file_path.exists():
            print(f"Skipping {pair} - file not found")
            continue
            
        try:
            # Load data
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Loaded {pair}: {len(df)} bars")
            
            # Process signals
            pair_signals = process_pair_signals(df, pair)
            all_signals.extend(pair_signals)
            
            print(f"  Found {len(pair_signals)} signals")
            
        except Exception as e:
            print(f"Error processing {pair}: {e}")
            continue
    
    print(f"\nTotal signals collected: {len(all_signals)}")
    
    # Analyze results
    results = analyze_tool_performance(all_signals)
    
    # Sort by net expected return (24h)
    results.sort(key=lambda x: x['net_exp_24h'], reverse=True)
    
    # Print results table
    print("\n" + "=" * 110)
    print("VALIDATION RESULTS")
    print("=" * 110)
    print(f"{'Tool':<18} {'Dir':<5} {'Sigs':<6} {'WR_8h':<7} {'Avg_8h':<8} {'Net_8h':<8} {'WR_24h':<7} {'Avg_24h':<8} {'Net_24h':<8} {'Status':<6}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['tool']:<18} {r['direction']:<5} {r['n_signals']:<6} "
              f"{r['wr_8h']:<7.1f}% {r['avg_ret_8h']:<8.2f}% {r['net_exp_8h']:<8.3f}% "
              f"{r['wr_24h']:<7.1f}% {r['avg_ret_24h']:<8.2f}% {r['net_exp_24h']:<8.3f}% "
              f"{r['status']:<6}")
    
    # Summary
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = len(results) - passed
    
    print("-" * 110)
    print(f"SUMMARY: {passed} tools PASSED, {failed} tools FAILED out of {len(results)} tested")
    print("=" * 110)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    output_file = "data/validation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save all signals for detailed analysis
    signals_df = pd.DataFrame(all_signals)
    signals_file = "data/validation_signals.csv"
    signals_df.to_csv(signals_file, index=False)
    print(f"All signals saved to: {signals_file}")
    
    # Quick analysis of best/worst tools
    print(f"\n🎯 BEST PERFORMING TOOLS:")
    for r in results[:3]:
        if r['status'] == 'PASS':
            print(f"  {r['tool']}: {r['net_exp_24h']:.3f}% net return, {r['wr_24h']:.1f}% win rate, {r['n_signals']} signals")
    
    print(f"\n❌ TOOLS NEEDING OPTIMIZATION:")
    failed_tools = [r for r in results if r['status'] == 'FAIL']
    for r in failed_tools[:3]:
        print(f"  {r['tool']}: {r['net_exp_24h']:.3f}% net return, {r['wr_24h']:.1f}% win rate (needs fixing)")

if __name__ == "__main__":
    main()