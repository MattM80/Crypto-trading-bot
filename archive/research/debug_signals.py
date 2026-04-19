#!/usr/bin/env python3
"""Debug signal generation to see why strategies aren't triggering"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategies_v2 import LiquidationCascadeStrategy, MomentumBreakoutStrategy
import requests
from datetime import datetime, timezone, timedelta

def download_sample_data(symbol="XBTUSD", days=7):
    """Download a small sample of data for debugging"""
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - (days * 86400)
    
    resp = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": symbol, "interval": 60, "since": start_ts},
        timeout=30
    )
    
    result = resp.json().get("result", {})
    candles = None
    for key, val in result.items():
        if isinstance(val, list) and key != 'last':
            candles = val
            break
            
    if not candles:
        return pd.DataFrame()
        
    data = []
    for c in candles[:100]:  # Just first 100 bars for debugging
        data.append({
            'timestamp': datetime.fromtimestamp(int(c[0]), tz=timezone.utc),
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'volume': float(c[6]),
            'time': int(c[0])
        })
        
    df = pd.DataFrame(data)
    return df

def debug_liquidation_cascade():
    """Debug the LiquidationCascade strategy"""
    print("=== DEBUGGING LIQUIDATION CASCADE ===")
    
    strategy = LiquidationCascadeStrategy()
    df = download_sample_data("XBTUSD", days=7)
    
    if df.empty:
        print("No data downloaded!")
        return
        
    print(f"Downloaded {len(df)} bars")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
    
    # Calculate some indicators manually to see what's happening
    atr = strategy.calculate_atr(df)
    vwap = strategy.calculate_vwap(df, 20)
    avg_volume = df['volume'].rolling(20).mean()
    
    print(f"\nIndicators calculated:")
    print(f"ATR: {atr.tail(5).values}")
    print(f"VWAP: {vwap.tail(5).values}")
    print(f"Avg Volume: {avg_volume.tail(5).values}")
    
    # Look for drops and volume spikes
    print(f"\nLooking for signals...")
    
    signals_found = 0
    for i in range(50, len(df)):
        current_price = df.iloc[i]['close']
        current_volume = df.iloc[i]['volume']
        current_atr = atr.iloc[i]
        current_vwap = vwap.iloc[i]
        avg_vol = avg_volume.iloc[i]
        
        # Check drop
        lookback = 6
        recent_high = df.iloc[i-lookback:i+1]['high'].max()
        drop_pct = (recent_high - current_price) / recent_high * 100
        
        # Check volume
        volume_ratio = current_volume / avg_vol if avg_vol > 0 else 0
        
        # Check conditions
        drop_condition = drop_pct >= 0.8
        volume_condition = volume_ratio >= 1.2
        below_vwap = current_price < current_vwap
        
        if drop_pct > 0.3 or volume_ratio > 1.1:  # Log interesting moves
            print(f"Bar {i}: drop={drop_pct:.2f}% (need 0.8%), "
                  f"vol_ratio={volume_ratio:.2f}x (need 1.2x), "
                  f"price vs vwap: ${current_price:.2f} vs ${current_vwap:.2f}, "
                  f"below_vwap={below_vwap}")
            
        if drop_condition and volume_condition and below_vwap:
            signals_found += 1
            print(f"*** SIGNAL FOUND at bar {i} ***")
            
    print(f"Total signals found: {signals_found}")
    
    # Test the actual strategy
    print(f"\n=== TESTING ACTUAL STRATEGY ===")
    data = {"XBTUSD": df}
    
    for i in range(10):  # Try 10 times with different data slices
        test_data = {"XBTUSD": df.iloc[:50+i*5].copy()}
        signal = strategy.generate_signal(test_data)
        if signal:
            print(f"Strategy generated signal: {signal}")
            return
            
    print("Strategy generated no signals")

def debug_momentum_breakout():
    """Debug the MomentumBreakout strategy"""
    print("\n=== DEBUGGING MOMENTUM BREAKOUT ===")
    
    strategy = MomentumBreakoutStrategy()
    df = download_sample_data("XBTUSD", days=7)
    
    if df.empty:
        print("No data downloaded!")
        return
        
    print(f"Downloaded {len(df)} bars")
    
    # Calculate indicators
    sma_trend = strategy.calculate_sma(df, 50)
    avg_volume = df['volume'].rolling(20).mean()
    
    print(f"SMA trend: {sma_trend.tail(5).values}")
    
    signals_found = 0
    for i in range(60, len(df)):
        current_price = df.iloc[i]['close']
        current_high = df.iloc[i]['high']
        current_volume = df.iloc[i]['volume']
        current_sma = sma_trend.iloc[i]
        avg_vol = avg_volume.iloc[i]
        
        # Find breakout level
        lookback_start = max(0, i - 20)
        lookback_high = df.iloc[lookback_start:i]['high'].max()
        
        # Check conditions
        breakout_condition = current_high > lookback_high
        trend_condition = current_price >= current_sma
        volume_ratio = current_volume / avg_vol if avg_vol > 0 else 0
        volume_condition = volume_ratio >= 1.3
        
        if breakout_condition or volume_ratio > 1.2:
            print(f"Bar {i}: breakout={breakout_condition} (high={current_high:.2f} vs {lookback_high:.2f}), "
                  f"trend={trend_condition} (price={current_price:.2f} vs sma={current_sma:.2f}), "
                  f"vol_ratio={volume_ratio:.2f}x")
                  
        if breakout_condition and trend_condition and volume_condition:
            signals_found += 1
            print(f"*** MOMENTUM SIGNAL FOUND at bar {i} ***")
            
    print(f"Total momentum signals found: {signals_found}")

if __name__ == "__main__":
    debug_liquidation_cascade()
    debug_momentum_breakout()