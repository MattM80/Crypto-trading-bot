#!/usr/bin/env python3
"""
Quick test of a few tools to debug the system
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

def test_mega_pump_sell_t1(df: pd.DataFrame, i: int) -> bool:
    """Tool 1: mega_pump_sell T1 - rsi7 > 80 AND ret_12h >= 10 → SHORT"""
    if i < 13:
        return False
    close = df['close'].values
    rsi7 = calc_rsi(close[:i+1], 7)
    ret_12h = (close[i] - close[i-13]) / close[i-13] * 100
    return rsi7[i] > 80 and ret_12h >= 10

def test_sma50_ext_8(df: pd.DataFrame, i: int) -> bool:
    """Tool 6: sma50_ext_8 - cur_vs_sma50 > 8% → SHORT"""
    if i < 50:
        return False
    close = df['close'].values
    sma50 = calc_sma(close[:i+1], 50)
    cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
    return cur_vs_sma50 > 8

def run_test():
    print("Loading BTCUSDT data...")
    df = pd.read_csv('data/binance_1h/BTCUSDT_1h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} bars")
    
    print("\nTesting mega_pump_sell_t1...")
    signals = []
    oos_start = 4380
    
    for i in range(oos_start, len(df) - 24):
        if test_mega_pump_sell_t1(df, i):
            signals.append(i)
    
    print(f"Found {len(signals)} signals")
    
    if len(signals) > 0:
        print("First few signal bars:", signals[:5])
        
        # Test returns calculation
        wins_8h = 0
        returns_8h = []
        fee_pct = 0.0052
        
        for sig_idx in signals[:10]:  # Test first 10
            if sig_idx + 24 >= len(df):
                continue
            
            entry_price = df.iloc[sig_idx]['close']
            exit_8h_price = df.iloc[sig_idx + 8]['close']
            
            ret_8h = (exit_8h_price - entry_price) / entry_price
            ret_8h_short = -ret_8h  # SHORT direction
            ret_8h_net = ret_8h_short - fee_pct  # Apply fees
            
            returns_8h.append(ret_8h_net)
            if ret_8h_net > 0:
                wins_8h += 1
        
        print(f"Sample returns: {[f'{r*100:.2f}%' for r in returns_8h]}")
        print(f"Win rate: {wins_8h/len(returns_8h):.1%}")
    
    print("\nTesting sma50_ext_8...")
    signals2 = []
    
    for i in range(oos_start, min(oos_start + 1000, len(df) - 24)):  # Test 1000 bars
        if test_sma50_ext_8(df, i):
            signals2.append(i)
    
    print(f"Found {len(signals2)} signals in 1000 bars")
    
    print("Quick test complete!")

if __name__ == "__main__":
    run_test()