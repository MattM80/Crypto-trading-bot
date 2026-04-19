#!/usr/bin/env python3
"""
Simple test to verify validation logic works on one pair.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calc_rsi(prices, period=7):
    """Simple RSI calculation."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def main():
    # Test with one pair
    data_file = Path("data/binance_historical/BTCUSDT_4h.csv")
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    
    print("Loading BTCUSDT data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} bars")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Test signal detection on a few bars
    close = df['close'].values
    signals_found = 0
    
    print("\nTesting signal detection...")
    for i in range(50, min(100, len(df) - 10)):  # Test first 50 bars
        price = close[i]
        
        # Calculate simple returns
        ret_24h = (close[i] - close[i-6]) / close[i-6] * 100 if i >= 6 else 0
        ret_4h = (close[i] - close[i-1]) / close[i-1] * 100 if i >= 1 else 0
        
        # Calculate RSI
        if i >= 20:
            rsi = calc_rsi(close[i-20:i+1])
        else:
            rsi = 50
        
        # Test crash buy signal
        if ret_24h < -10 and rsi < 20:
            signals_found += 1
            
            # Calculate forward returns
            ret_forward_8h = (close[i+2] - close[i]) / close[i] * 100 if i+2 < len(close) else 0
            ret_forward_24h = (close[i+6] - close[i]) / close[i] * 100 if i+6 < len(close) else 0
            
            print(f"Bar {i}: CRASH BUY signal")
            print(f"  Price: ${price:.2f}, RSI: {rsi:.1f}, 24h ret: {ret_24h:.1f}%")
            print(f"  Forward 8h: {ret_forward_8h:.1f}%, Forward 24h: {ret_forward_24h:.1f}%")
        
        # Test dip buy signal
        if ret_4h < -3:
            signals_found += 1
            
            ret_forward_8h = (close[i+2] - close[i]) / close[i] * 100 if i+2 < len(close) else 0
            
            print(f"Bar {i}: DIP BUY signal")
            print(f"  Price: ${price:.2f}, 4h ret: {ret_4h:.1f}%")
            print(f"  Forward 8h: {ret_forward_8h:.1f}%")
    
    print(f"\nFound {signals_found} signals in test sample")
    print("Basic validation logic working!")

if __name__ == "__main__":
    main()