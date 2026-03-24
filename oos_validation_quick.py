#!/usr/bin/env python3
"""
Quick Out-of-Sample Validation - Testing core tools first
Focuses on the most important tools (Tools 2-12) for faster execution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data directory
DATA_DIR = Path("data/binance_historical")

# Map Kraken pairs to Binance pairs
PAIR_MAPPING = {
    "XBTUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT", 
    "NEARUSD": "NEARUSDT",
    "SOLUSD": "SOLUSDT",
    "AVAXUSD": "AVAXUSDT",
    "LINKUSD": "LINKUSDT",
}

def calc_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate RSI using Wilder's smoothing method."""
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

def calc_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
        
    sma = np.full(len(prices), np.nan)
    for i in range(period-1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
        
    return sma

def scan_core_tools(pair: str, df: pd.DataFrame, idx: int) -> list:
    """Scan for core trading tool signals."""
    signals = []
    
    if idx < 50:  # Need enough history
        return signals
    
    # Extract data up to current bar (no lookahead bias)
    close = df['close'].values[:idx+1]
    high = df['high'].values[:idx+1]
    low = df['low'].values[:idx+1]
    volume = df['volume'].values[:idx+1]
    
    if len(close) < 50:
        return signals
        
    # Current values
    price = close[-1]
    
    # Calculate indicators
    rsi7 = calc_rsi(close, 7)
    sma50 = calc_sma(close, 50)
    
    # Current indicator values
    cur_rsi = rsi7[-1]
    cur_vs_sma50 = (price - sma50[-1]) / sma50[-1] * 100 if not np.isnan(sma50[-1]) and sma50[-1] > 0 else 0
    
    # Calculate returns (backward looking, no bias)
    ret_4h = (price - close[-2]) / close[-2] * 100 if len(close) >= 2 and close[-2] > 0 else 0
    ret_8h = (price - close[-3]) / close[-3] * 100 if len(close) >= 3 and close[-3] > 0 else 0
    ret_12h = (price - close[-4]) / close[-4] * 100 if len(close) >= 4 and close[-4] > 0 else 0
    ret_24h = (price - close[-7]) / close[-7] * 100 if len(close) >= 7 and close[-7] > 0 else 0
    
    # Calculate ATR for volatility tools
    cur_atr_pct = 0
    if len(close) >= 15:
        tr1 = high[-14:] - low[-14:]
        tr2 = np.abs(high[-14:] - close[-15:-1])
        tr3 = np.abs(low[-14:] - close[-15:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr14 = np.mean(tr)
        cur_atr_pct = atr14 / price * 100 if price > 0 else 0
    
    # CORE TOOLS TO TEST
    
    # Tool 2: Crash Buy (BEST EDGE)
    if ret_24h < -10 and cur_rsi < 20:
        signals.append({
            'tool': 'crash_buy',
            'direction': 'long',
            'reason': f"CRASH BUY: {ret_24h:.1f}% drop 24h, RSI={cur_rsi:.1f}"
        })
    
    # Tool 3: Volatile Oversold
    if cur_atr_pct > 3 and cur_rsi < 25:
        signals.append({
            'tool': 'volatile_oversold',
            'direction': 'long',
            'reason': f"VOLATILE OVERSOLD: ATR={cur_atr_pct:.1f}%, RSI={cur_rsi:.1f}"
        })
    
    # Tool 4: Downtrend Relief Rally
    if cur_rsi > 75 and cur_vs_sma50 < 0:
        signals.append({
            'tool': 'relief_rally',
            'direction': 'long',
            'reason': f"RELIEF RALLY: RSI={cur_rsi:.1f}, below SMA50"
        })
    
    # Tool 6: Dip Buy
    if ret_4h < -3:
        signals.append({
            'tool': 'dip_buy',
            'direction': 'long',
            'reason': f"DIP BUY: {ret_4h:.1f}% drop 4h"
        })
    
    # Tool 7: RSI Pump Short
    if cur_rsi > 80 and ret_12h > 8:
        signals.append({
            'tool': 'rsi_pump_short',
            'direction': 'short',
            'reason': f"RSI PUMP SHORT: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h"
        })
    
    # Tool 8: Mega Crash Buy
    if ret_24h < -15:
        signals.append({
            'tool': 'mega_crash',
            'direction': 'long',
            'reason': f"MEGA CRASH: {ret_24h:.1f}% drop 24h"
        })
    
    # Tool 9: Flash Crash Buy
    if ret_12h < -10:
        signals.append({
            'tool': 'flash_crash',
            'direction': 'long',
            'reason': f"FLASH CRASH: {ret_12h:.1f}% drop 12h"
        })
    
    # Tool 10: Quick Crash
    if ret_8h < -10:
        signals.append({
            'tool': 'quick_crash',
            'direction': 'long',
            'reason': f"QUICK CRASH: {ret_8h:.1f}% drop 8h"
        })
    
    # Tool 11: Deep Dip
    if -10 < ret_8h < -8:
        signals.append({
            'tool': 'deep_dip_8h',
            'direction': 'long',
            'reason': f"DEEP DIP 8H: {ret_8h:.1f}% drop"
        })
    
    # Tool 12: Quick Dip
    if ret_4h < -5:
        signals.append({
            'tool': 'quick_dip',
            'direction': 'long',
            'reason': f"QUICK DIP: {ret_4h:.1f}% drop 4h"
        })
    
    return signals

def main():
    """Run quick validation on core tools."""
    print("Quick Out-of-Sample Validation - Core Tools")
    print("="*50)
    
    # Load data
    data = {}
    for kraken_pair, binance_pair in PAIR_MAPPING.items():
        file_path = DATA_DIR / f"{binance_pair}_4h.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                data[kraken_pair] = df
                print(f"Loaded {kraken_pair}: {len(df)} bars")
            except Exception as e:
                print(f"Error loading {binance_pair}: {e}")
    
    if not data:
        print("No data loaded!")
        return
    
    print(f"Loaded {len(data)} pairs, starting validation...")
    
    # Collect all signals
    all_signals = []
    
    for pair, df in data.items():
        print(f"Processing {pair}...")
        pair_signals = 0
        
        # Process each bar (leaving room for forward returns)
        for idx in range(50, len(df) - 10):
            signals = scan_core_tools(pair, df, idx)
            
            for signal in signals:
                # Calculate forward returns
                close = df['close'].values
                
                ret_8h = 0
                ret_24h = 0
                if idx + 2 < len(close):
                    ret_8h = (close[idx + 2] - close[idx]) / close[idx] * 100
                if idx + 6 < len(close):
                    ret_24h = (close[idx + 6] - close[idx]) / close[idx] * 100
                
                # Record signal
                signal_record = {
                    'pair': pair,
                    'timestamp': df['timestamp'].iloc[idx],
                    'tool': signal['tool'],
                    'direction': signal['direction'],
                    'reason': signal['reason'],
                    'price': close[idx],
                    'ret_8h': ret_8h,
                    'ret_24h': ret_24h,
                }
                
                all_signals.append(signal_record)
                pair_signals += 1
        
        print(f"  Generated {pair_signals} signals")
    
    print(f"\nTotal signals: {len(all_signals)}")
    
    # Analyze by tool
    tool_results = {}
    for signal in all_signals:
        tool = signal['tool']
        if tool not in tool_results:
            tool_results[tool] = []
        tool_results[tool].append(signal)
    
    print(f"\nResults by Tool:")
    print("-" * 80)
    print(f"{'Tool':<20} {'Dir':<6} {'Sigs':<6} {'WR_8h':<7} {'Avg_8h':<8} {'WR_24h':<7} {'Avg_24h':<8} {'Status':<6}")
    print("-" * 80)
    
    summary = []
    for tool, signals in tool_results.items():
        if not signals:
            continue
        
        n_signals = len(signals)
        direction = signals[0]['direction']
        
        # Get returns
        returns_8h = [s['ret_8h'] for s in signals]
        returns_24h = [s['ret_24h'] for s in signals]
        
        # Adjust for direction
        if direction == 'short':
            adj_ret_8h = [-r for r in returns_8h]
            adj_ret_24h = [-r for r in returns_24h]
        else:
            adj_ret_8h = returns_8h
            adj_ret_24h = returns_24h
        
        # Calculate metrics
        wins_8h = sum(1 for r in adj_ret_8h if r > 0)
        wr_8h = wins_8h / n_signals * 100
        avg_ret_8h = np.mean(adj_ret_8h)
        
        wins_24h = sum(1 for r in adj_ret_24h if r > 0)
        wr_24h = wins_24h / n_signals * 100
        avg_ret_24h = np.mean(adj_ret_24h)
        
        # Status
        min_wr = 50 if direction == 'long' else 45
        status = "PASS" if wr_24h > min_wr and avg_ret_24h > 0 and n_signals >= 10 else "FAIL"
        
        print(f"{tool:<20} {direction:<6} {n_signals:<6} {wr_8h:<7.1f}% {avg_ret_8h:<8.2f}% {wr_24h:<7.1f}% {avg_ret_24h:<8.2f}% {status:<6}")
        
        summary.append({
            'tool': tool,
            'direction': direction,
            'n_signals': n_signals,
            'wr_8h': wr_8h,
            'avg_ret_8h': avg_ret_8h,
            'wr_24h': wr_24h,
            'avg_ret_24h': avg_ret_24h,
            'status': status
        })
    
    # Summary
    passed = sum(1 for s in summary if s['status'] == 'PASS')
    failed = len(summary) - passed
    print("-" * 80)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED out of {len(summary)} tools tested")
    
    # Save results
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("data/quick_validation_results.csv", index=False)
    print(f"Results saved to: data/quick_validation_results.csv")

if __name__ == "__main__":
    main()