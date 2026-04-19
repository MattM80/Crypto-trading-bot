#!/usr/bin/env python3
"""Quick test of Round 3 validation framework"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h_extended"

def test_data_loading():
    print("Testing data loading...")
    try:
        df = pd.read_csv(DATA_DIR / "BTCUSDT_1h.csv")
        print(f"✅ Loaded BTCUSDT: {len(df)} rows")
        print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def simple_bb_test():
    """Test one simple tool"""
    print("\nTesting BB mean reversion tool...")
    
    try:
        df = pd.read_csv(DATA_DIR / "BTCUSDT_1h.csv")
        close = df['close'].values
        high = df['high'].values  
        low = df['low'].values
        volume = df['volume'].values
        
        print(f"Data loaded: {len(close)} bars")
        
        # Test BB calculation
        period = 20
        if len(close) >= period:
            sma = np.mean(close[-period:])
            std = np.std(close[-period:])
            bb_pos = (close[-1] - sma) / std if std > 0 else 0
            print(f"Latest BB position: {bb_pos:.2f}")
            
            # Test ATR calculation
            tr_list = []
            for i in range(1, min(len(close), 15)):
                tr = max(
                    high[-i] - low[-i],
                    abs(high[-i] - close[-i-1]),
                    abs(low[-i] - close[-i-1])
                )
                tr_list.append(tr)
            atr = np.mean(tr_list)
            atr_pct = (atr / close[-1]) * 100
            print(f"ATR percentage: {atr_pct:.2f}%")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in BB test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== ROUND 3 VALIDATION - QUICK TEST ===")
    
    if test_data_loading():
        simple_bb_test()
    
    print("\nTest complete.")