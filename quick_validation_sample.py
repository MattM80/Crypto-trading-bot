#!/usr/bin/env python3
"""
Quick validation on just BTC to see immediate results
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h"

def load_data(pair: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{pair}_1h.csv"
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['returns'] = df['close'].pct_change()
    return df

def simple_hurst(prices, max_lag=15):
    """Fast Hurst calculation"""
    log_prices = np.log(prices[-80:])  # Smaller window
    lags = range(2, min(max_lag, len(log_prices) // 3))
    
    rs_values = []
    for lag in lags:
        n_periods = len(log_prices) // lag
        if n_periods < 3:
            continue
            
        rs_list = []
        for i in range(n_periods):
            period = log_prices[i*lag:(i+1)*lag]
            if len(period) < lag:
                continue
            mean_val = np.mean(period)
            cumsum_dev = np.cumsum(period - mean_val)
            R = np.max(cumsum_dev) - np.min(cumsum_dev)
            S = np.std(period)
            if S > 0:
                rs_list.append(R / S)
        
        if len(rs_list) >= 2:
            rs_values.append((lag, np.mean(rs_list)))
    
    if len(rs_values) < 3:
        return np.nan
        
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values if x[1] > 0])
    
    if len(log_lags) != len(log_rs) or len(log_rs) < 3:
        return np.nan
        
    slope, _, _, _, _ = stats.linregress(log_lags[:len(log_rs)], log_rs)
    return slope

def quick_test():
    print("Quick validation test on BTCUSDT...")
    
    data = load_data("BTCUSDT")
    print(f"Loaded {len(data)} bars of BTC data")
    
    # Test Hurst calculation speed
    close = data['close'].values
    
    signals_found = 0
    
    # Sample every 50th bar for speed
    for i in range(100, len(data), 50):
        if i + 100 > len(data):
            break
            
        window_close = close[:i]
        
        # Quick Hurst test
        if len(window_close) >= 100:
            hurst = simple_hurst(window_close)
            
            if not np.isnan(hurst):
                current_price = window_close[-1]
                prev_price_24h = window_close[-24] if len(window_close) >= 24 else window_close[-1]
                momentum_24h = (current_price - prev_price_24h) / prev_price_24h * 100
                
                # Simple signal logic
                if hurst > 0.55 and abs(momentum_24h) > 4:
                    signals_found += 1
                    if signals_found <= 5:  # Show first 5
                        direction = "long" if momentum_24h > 0 else "short"
                        print(f"  Signal {signals_found}: Bar {i}, Hurst={hurst:.3f}, Momentum={momentum_24h:.1f}% -> {direction}")
                elif hurst < 0.45:
                    # Could add mean reversion logic here
                    pass
    
    print(f"\nFound {signals_found} Hurst regime signals in sample")
    print("This suggests the tools are working and generating signals.")
    print("\nNext step: Full validation (currently running in background)")

if __name__ == "__main__":
    quick_test()