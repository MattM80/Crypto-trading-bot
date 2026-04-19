#!/usr/bin/env python3
"""
Simple test of a few key tools to establish the framework works
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calc_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
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

def calc_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average"""
    return pd.Series(prices).rolling(window=period, min_periods=1).mean().values

def test_tool(df, tool_func, direction, start_idx=4380, fee_pct=0.0052):
    """Test a tool and return results"""
    signals = []
    
    # Find signals
    for i in range(start_idx, len(df) - 24):
        if tool_func(df, i):
            signals.append(i)
    
    if len(signals) == 0:
        return {'signals': 0, 'wr_8h': 0, 'wr_24h': 0, 'avg_ret_8h': 0, 'avg_ret_24h': 0}
    
    # Calculate returns
    returns_8h = []
    returns_24h = []
    
    for sig_idx in signals:
        if sig_idx + 24 >= len(df):
            continue
            
        entry_price = df.iloc[sig_idx]['close']
        exit_8h_price = df.iloc[sig_idx + 8]['close']
        exit_24h_price = df.iloc[sig_idx + 24]['close']
        
        ret_8h = (exit_8h_price - entry_price) / entry_price
        ret_24h = (exit_24h_price - entry_price) / entry_price
        
        # Apply direction
        if direction == 'short':
            ret_8h = -ret_8h
            ret_24h = -ret_24h
        
        # Apply fees
        ret_8h_net = ret_8h - fee_pct
        ret_24h_net = ret_24h - fee_pct
        
        returns_8h.append(ret_8h_net)
        returns_24h.append(ret_24h_net)
    
    wins_8h = sum(1 for r in returns_8h if r > 0)
    wins_24h = sum(1 for r in returns_24h if r > 0)
    
    return {
        'signals': len(returns_8h),
        'wr_8h': wins_8h / len(returns_8h),
        'wr_24h': wins_24h / len(returns_24h),
        'avg_ret_8h': np.mean(returns_8h) * 100,
        'avg_ret_24h': np.mean(returns_24h) * 100
    }

# Tool definitions with relaxed parameters
def rsi_pump_relaxed(df, i):
    """RSI > 75 AND ret_8h >= 5 → SHORT"""
    if i < 9:
        return False
    close = df['close'].values
    rsi7 = calc_rsi(close[:i+1], 7)
    ret_8h = (close[i] - close[i-9]) / close[i-9] * 100
    return rsi7[i] > 75 and ret_8h >= 5

def sma_ext_relaxed(df, i):
    """price > sma50 by 3%+ → SHORT"""
    if i < 50:
        return False
    close = df['close'].values
    sma50 = calc_sma(close[:i+1], 50)
    ext_pct = (close[i] - sma50[i]) / sma50[i] * 100
    return ext_pct > 3

def high_breakout_relaxed(df, i):
    """price > 30-bar high → LONG"""
    if i < 30:
        return False
    close = df['close'].values
    high_30 = np.max(close[i-30:i])
    return close[i] > high_30

def dip_buy_simple(df, i):
    """ret_4h < -2% → LONG"""
    if i < 5:
        return False
    close = df['close'].values
    ret_4h = (close[i] - close[i-5]) / close[i-5] * 100
    return ret_4h < -2

def main():
    print("Simple Bull/Momentum Tools Test")
    print("="*40)
    
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]
    
    tools = [
        ("rsi_pump_relaxed", rsi_pump_relaxed, "short"),
        ("sma_ext_relaxed", sma_ext_relaxed, "short"),  
        ("high_breakout_relaxed", high_breakout_relaxed, "long"),
        ("dip_buy_simple", dip_buy_simple, "long")
    ]
    
    results = []
    
    for pair in pairs:
        print(f"\nTesting {pair}...")
        file_path = f"data/binance_1h/{pair}_1h.csv"
        
        try:
            df = pd.read_csv(file_path)
            print(f"  Loaded {len(df)} bars")
            
            for tool_name, tool_func, direction in tools:
                result = test_tool(df, tool_func, direction)
                result['pair'] = pair
                result['tool'] = tool_name
                result['direction'] = direction
                results.append(result)
                
                if result['signals'] > 0:
                    print(f"  {tool_name} ({direction}): {result['signals']} signals, "
                          f"WR_8h={result['wr_8h']:.1%}, WR_24h={result['wr_24h']:.1%}, "
                          f"Ret_8h={result['avg_ret_8h']:+.2f}%, Ret_24h={result['avg_ret_24h']:+.2f}%")
                else:
                    print(f"  {tool_name} ({direction}): No signals")
                    
        except Exception as e:
            print(f"  Error loading {pair}: {e}")
    
    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY BY TOOL")
    print("="*60)
    
    tool_summary = {}
    for result in results:
        tool = result['tool']
        if tool not in tool_summary:
            tool_summary[tool] = {
                'direction': result['direction'],
                'total_signals': 0,
                'pairs_with_signals': 0,
                'total_wr_8h': 0,
                'total_wr_24h': 0,
                'signal_weights': []
            }
        
        summary = tool_summary[tool]
        if result['signals'] > 0:
            summary['pairs_with_signals'] += 1
            summary['total_signals'] += result['signals']
            summary['total_wr_8h'] += result['wr_8h'] * result['signals']
            summary['total_wr_24h'] += result['wr_24h'] * result['signals']
            summary['signal_weights'].append(result['signals'])
    
    for tool, summary in tool_summary.items():
        if summary['total_signals'] > 0:
            avg_wr_8h = summary['total_wr_8h'] / summary['total_signals']
            avg_wr_24h = summary['total_wr_24h'] / summary['total_signals']
            status = "PASS" if avg_wr_8h > 0.5 or avg_wr_24h > 0.5 else "FAIL"
        else:
            avg_wr_8h = avg_wr_24h = 0
            status = "NO_SIGNALS"
        
        print(f"{tool} ({summary['direction'].upper()}): {summary['total_signals']} signals, "
              f"{summary['pairs_with_signals']} pairs, WR_8h={avg_wr_8h:.1%}, WR_24h={avg_wr_24h:.1%} [{status}]")
    
    # Write results
    with open('data/simple_test_results.md', 'w') as f:
        f.write("# Simple Bull/Momentum Tools Test Results\n\n")
        f.write("## Tool Performance\n\n")
        f.write("| Tool | Direction | Total Signals | Pairs w/ Signals | Avg WR_8h | Avg WR_24h | Status |\n")
        f.write("|------|-----------|---------------|------------------|-----------|------------|--------|\n")
        
        for tool, summary in tool_summary.items():
            if summary['total_signals'] > 0:
                avg_wr_8h = summary['total_wr_8h'] / summary['total_signals']
                avg_wr_24h = summary['total_wr_24h'] / summary['total_signals']
                status = "PASS" if avg_wr_8h > 0.5 or avg_wr_24h > 0.5 else "FAIL"
            else:
                avg_wr_8h = avg_wr_24h = 0
                status = "NO_SIGNALS"
            
            f.write(f"| {tool} | {summary['direction'].upper()} | {summary['total_signals']} | "
                   f"{summary['pairs_with_signals']} | {avg_wr_8h:.1%} | {avg_wr_24h:.1%} | {status} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for result in results:
            f.write(f"- **{result['pair']} - {result['tool']} ({result['direction']}):** "
                   f"{result['signals']} signals, WR_8h={result['wr_8h']:.1%}, WR_24h={result['wr_24h']:.1%}, "
                   f"Ret_8h={result['avg_ret_8h']:+.2f}%, Ret_24h={result['avg_ret_24h']:+.2f}%\n")
    
    print(f"\nResults written to: data/simple_test_results.md")

if __name__ == "__main__":
    main()