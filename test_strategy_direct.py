#!/usr/bin/env python3
"""Test strategies directly to see where they're failing"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategies_v2 import LiquidationCascadeStrategy, MomentumBreakoutStrategy
import requests
from datetime import datetime, timezone, timedelta

def download_sample_data(symbol="XBTUSD", days=7):
    """Download a small sample of data"""
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
    for c in candles[:200]:  # More bars
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

def test_liquidation_strategy_step_by_step():
    """Test liquidation strategy step by step"""
    print("=== STEP BY STEP LIQUIDATION CASCADE TEST ===")
    
    strategy = LiquidationCascadeStrategy()
    df = download_sample_data("XBTUSD", days=10)
    
    print(f"Downloaded {len(df)} bars")
    print(f"Strategy config: {strategy.config}")
    
    # Test with progressively larger datasets
    for test_size in [50, 60, 70, 80, 90, 100]:
        if test_size > len(df):
            continue
            
        test_df = df.iloc[:test_size].copy()
        data = {"XBTUSD": test_df}
        
        print(f"\n--- Testing with {test_size} bars ---")
        
        # Check each condition manually
        current_idx = len(test_df) - 1
        current_price = test_df.iloc[current_idx]['close']
        current_volume = test_df.iloc[current_idx]['volume']
        
        # Calculate indicators like the strategy does
        atr = strategy.calculate_atr(test_df)
        vwap = strategy.calculate_vwap(test_df, strategy.config['vwap_periods'])
        avg_volume = test_df['volume'].rolling(20).mean()
        
        current_atr = atr.iloc[current_idx]
        current_vwap = vwap.iloc[current_idx]
        avg_vol = avg_volume.iloc[current_idx]
        
        # Check all conditions
        print(f"  Current price: ${current_price:.2f}")
        print(f"  Current ATR: {current_atr:.2f}")
        print(f"  Current VWAP: ${current_vwap:.2f}")
        print(f"  Avg volume: {avg_vol:.2f}")
        
        # NaN check
        if pd.isna(current_atr) or pd.isna(current_vwap) or pd.isna(avg_vol):
            print(f"  ❌ NaN values detected")
            continue
            
        if avg_vol == 0:
            print(f"  ❌ Zero average volume")
            continue
            
        # Volatility filter
        atr_pct = current_atr / current_price * 100
        min_atr_pct = strategy.config['min_atr_pct']
        if atr_pct < min_atr_pct:
            print(f"  ❌ ATR too low: {atr_pct:.3f}% < {min_atr_pct}%")
            continue
        else:
            print(f"  ✓ ATR ok: {atr_pct:.3f}% >= {min_atr_pct}%")
            
        # Drop check
        lookback = strategy.config['lookback_bars']
        recent_high = test_df.iloc[current_idx - lookback:current_idx + 1]['high'].max()
        drop_pct = (recent_high - current_price) / recent_high * 100
        drop_needed = strategy.config['drop_threshold_pct']
        
        if drop_pct < drop_needed:
            print(f"  ❌ Drop too small: {drop_pct:.2f}% < {drop_needed}%")
            continue
        else:
            print(f"  ✓ Drop ok: {drop_pct:.2f}% >= {drop_needed}%")
            
        # Volume check
        volume_ratio = current_volume / avg_vol
        volume_needed = strategy.config['volume_spike_multiplier']
        
        if volume_ratio < volume_needed:
            print(f"  ❌ Volume too low: {volume_ratio:.2f}x < {volume_needed}x")
            continue
        else:
            print(f"  ✓ Volume ok: {volume_ratio:.2f}x >= {volume_needed}x")
            
        # VWAP check
        below_vwap = current_price < current_vwap
        if not below_vwap:
            print(f"  ❌ Not below VWAP: ${current_price:.2f} >= ${current_vwap:.2f}")
            continue
        else:
            print(f"  ✓ Below VWAP: ${current_price:.2f} < ${current_vwap:.2f}")
            
        # Risk/reward check
        stop_loss = current_price - (current_atr * strategy.config['stop_loss_atr_mult'])
        take_profit = current_price + (current_atr * strategy.config['take_profit_atr_mult'])
        risk = current_price - stop_loss
        reward = take_profit - current_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < 2.0:
            print(f"  ❌ R:R too low: {rr_ratio:.2f} < 2.0")
            print(f"      Risk: ${risk:.2f}, Reward: ${reward:.2f}")
            continue
        else:
            print(f"  ✓ R:R ok: {rr_ratio:.2f} >= 2.0")
            
        print(f"  🎯 ALL CONDITIONS MET! Should generate signal.")
        
        # Now test the actual strategy
        signal = strategy.generate_signal(data)
        if signal:
            print(f"  ✅ Strategy generated signal: {signal.reason}")
            return signal
        else:
            print(f"  ❌ Strategy generated NO signal (bug!)")
            
    print("\nNo signals found in any test size")
    return None

def test_momentum_strategy_step_by_step():
    """Test momentum strategy step by step"""
    print("\n=== STEP BY STEP MOMENTUM BREAKOUT TEST ===")
    
    strategy = MomentumBreakoutStrategy()
    df = download_sample_data("XBTUSD", days=10)
    
    print(f"Downloaded {len(df)} bars")
    print(f"Strategy config: {strategy.config}")
    
    # Test with larger dataset to get SMA working
    test_df = df.iloc[:150].copy() if len(df) >= 150 else df
    data = {"XBTUSD": test_df}
    
    print(f"\n--- Testing with {len(test_df)} bars ---")
    
    current_idx = len(test_df) - 1
    current_price = test_df.iloc[current_idx]['close']
    current_high = test_df.iloc[current_idx]['high']
    current_volume = test_df.iloc[current_idx]['volume']
    
    # Calculate indicators
    atr = strategy.calculate_atr(test_df)
    sma_trend = strategy.calculate_sma(test_df, strategy.config['trend_sma_period'])
    avg_volume = test_df['volume'].rolling(20).mean()
    
    current_atr = atr.iloc[current_idx]
    current_sma = sma_trend.iloc[current_idx]
    avg_vol = avg_volume.iloc[current_idx]
    
    print(f"  Current price: ${current_price:.2f}")
    print(f"  Current high: ${current_high:.2f}")
    print(f"  Current ATR: {current_atr:.2f}")
    print(f"  Current SMA: ${current_sma:.2f}")
    print(f"  Avg volume: {avg_vol:.2f}")
    
    # Check conditions one by one
    if pd.isna(current_atr) or pd.isna(current_sma) or pd.isna(avg_vol) or avg_vol == 0:
        print(f"  ❌ NaN or zero values")
        return None
        
    # Volatility filter
    atr_pct = current_atr / current_price * 100
    min_atr_pct = strategy.config['min_atr_pct']
    if atr_pct < min_atr_pct:
        print(f"  ❌ ATR too low: {atr_pct:.3f}% < {min_atr_pct}%")
        return None
    else:
        print(f"  ✓ ATR ok: {atr_pct:.3f}% >= {min_atr_pct}%")
        
    # Trend filter
    if current_price < current_sma:
        print(f"  ❌ Not in uptrend: ${current_price:.2f} < ${current_sma:.2f}")
        return None
    else:
        print(f"  ✓ In uptrend: ${current_price:.2f} >= ${current_sma:.2f}")
        
    # Breakout check
    lookback_start = max(0, current_idx - strategy.config['breakout_lookback'])
    lookback_high = test_df.iloc[lookback_start:current_idx]['high'].max()
    breakout_condition = current_high > lookback_high
    
    if not breakout_condition:
        print(f"  ❌ No breakout: ${current_high:.2f} <= ${lookback_high:.2f}")
        return None
    else:
        print(f"  ✓ Breakout: ${current_high:.2f} > ${lookback_high:.2f}")
        
    # Volume check
    volume_ratio = current_volume / avg_vol
    volume_needed = strategy.config['volume_confirmation']
    
    if volume_ratio < volume_needed:
        print(f"  ❌ Volume too low: {volume_ratio:.2f}x < {volume_needed}x")
        return None
    else:
        print(f"  ✓ Volume ok: {volume_ratio:.2f}x >= {volume_needed}x")
        
    print(f"  🎯 ALL CONDITIONS MET! Should generate signal.")
    
    # Test actual strategy
    signal = strategy.generate_signal(data)
    if signal:
        print(f"  ✅ Strategy generated signal: {signal.reason}")
        return signal
    else:
        print(f"  ❌ Strategy generated NO signal (bug!)")
        return None

if __name__ == "__main__":
    test_liquidation_strategy_step_by_step()
    test_momentum_strategy_step_by_step()