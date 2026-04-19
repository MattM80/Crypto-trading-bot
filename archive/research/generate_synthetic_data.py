#!/usr/bin/env python3
"""
Generate synthetic historical crypto data for backtesting.
Creates realistic OHLCV data that mimics crypto market characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Binance symbols (USDT pairs) 
PAIRS = [
    "NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", "SOLUSDT",
    "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", "XRPUSDT", "ADAUSDT", 
    "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"
]

# Realistic starting prices for each pair (approximate 2024 levels)
STARTING_PRICES = {
    "BTCUSDT": 42000,
    "ETHUSDT": 2500, 
    "SOLUSDT": 60,
    "AVAXUSDT": 35,
    "LINKUSDT": 14,
    "DOGEUSDT": 0.08,
    "ADAUSDT": 0.45,
    "XRPUSDT": 0.55,
    "DOTUSDT": 6.5,
    "ATOMUSDT": 9,
    "NEARUSDT": 1.8,
    "UNIUSDT": 6.5,
    "AAVEUSDT": 95,
    "LTCUSDT": 70,
    "FILUSDT": 4.5,
    "XLMUSDT": 0.11
}

OUTPUT_DIR = Path("data/binance_historical")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_crypto_price_series(start_price: float, n_bars: int, 
                                volatility: float = 0.05, 
                                trend_strength: float = 0.001) -> pd.DataFrame:
    """Generate realistic crypto OHLCV data using GBM with volatility clustering."""
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps (4-hour intervals)
    end_time = datetime(2024, 12, 31, 20, 0, 0, tzinfo=timezone.utc)
    start_time = end_time - timedelta(hours=4 * (n_bars - 1))
    timestamps = pd.date_range(start_time, end_time, periods=n_bars)
    
    # Generate price movements with volatility clustering
    returns = np.zeros(n_bars)
    vol = np.zeros(n_bars)
    vol[0] = volatility
    
    # GARCH-like volatility clustering
    for i in range(1, n_bars):
        # Volatility clustering: vol[t] = a*vol[t-1] + b*|return[t-1]| + c
        vol[i] = 0.85 * vol[i-1] + 0.10 * abs(returns[i-1]) + 0.02
        vol[i] = np.clip(vol[i], 0.005, 0.15)  # Clamp between 0.5% and 15%
        
        # Add trend component and random shocks
        trend = trend_strength * (np.sin(i / 100) + 0.5 * np.sin(i / 30))
        returns[i] = trend + vol[i] * np.random.normal(0, 1)
    
    # Generate close prices from returns
    prices = np.zeros(n_bars)
    prices[0] = start_price
    for i in range(1, n_bars):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    # Generate OHLC from close prices
    df = pd.DataFrame({'timestamp': timestamps, 'close': prices})
    
    # Add realistic OHLC spreads
    df['open'] = df['close'].shift(1).fillna(start_price)
    
    # High/Low with realistic ranges
    range_pcts = np.random.exponential(0.02, n_bars)  # Exponential distribution for ranges
    range_pcts = np.clip(range_pcts, 0.001, 0.1)  # 0.1% to 10% ranges
    
    # Random position within the range for close
    close_positions = np.random.uniform(0.2, 0.8, n_bars)
    
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + range_pcts * (1 - close_positions))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - range_pcts * close_positions)
    
    # Generate volume (correlated with price moves)
    price_changes = df['close'].pct_change().fillna(0)
    volume_base = 1000000  # Base volume
    volume_multiplier = 1 + 3 * np.abs(price_changes)  # Higher volume on big moves
    df['volume'] = volume_base * volume_multiplier * np.random.lognormal(0, 0.5, n_bars)
    
    # Add some extreme events (crashes and pumps) randomly
    n_events = max(1, n_bars // 500)  # ~2 events per year
    event_indices = np.random.choice(range(100, n_bars-100), n_events, replace=False)
    
    for idx in event_indices:
        if np.random.random() < 0.5:  # Crash
            crash_magnitude = np.random.uniform(0.15, 0.35)  # 15-35% crash
            df.loc[idx:idx+5, 'close'] *= (1 - crash_magnitude * np.exp(-np.arange(6)/3))
            df.loc[idx:idx+5, 'volume'] *= np.random.uniform(5, 15)  # Volume spike
        else:  # Pump
            pump_magnitude = np.random.uniform(0.20, 0.50)  # 20-50% pump
            df.loc[idx:idx+3, 'close'] *= (1 + pump_magnitude * np.exp(-np.arange(4)/2))
            df.loc[idx:idx+3, 'volume'] *= np.random.uniform(3, 8)  # Volume spike
    
    # Recalculate OHLC after events
    for i in range(1, len(df)):
        if i in event_indices:
            df.loc[i, 'open'] = df.loc[i-1, 'close']
            # Update high/low to be consistent
            df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close']) * (1 + np.random.uniform(0.001, 0.02))
            df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close']) * (1 - np.random.uniform(0.001, 0.02))
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Round to appropriate precision
    price_precision = 4 if start_price < 1 else (2 if start_price < 100 else 0)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].round(price_precision)
    df['volume'] = df['volume'].round(0)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def main():
    """Generate synthetic data for all pairs."""
    print(f"Generating 12 months of 4-hour synthetic data for {len(PAIRS)} pairs...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    n_bars = int(365 * 24 / 4)  # ~2190 4-hour bars in a year
    
    for symbol in PAIRS:
        try:
            start_price = STARTING_PRICES[symbol]
            
            # Adjust volatility and trend based on asset type
            if symbol == "BTCUSDT":
                vol, trend = 0.04, 0.0005  # Lower vol for BTC
            elif symbol in ["ETHUSDT", "SOLUSDT"]:
                vol, trend = 0.055, 0.001  # Medium vol for major alts
            else:
                vol, trend = 0.07, 0.002  # Higher vol for smaller alts
            
            print(f"Generating {symbol}...")
            df = generate_crypto_price_series(start_price, n_bars, vol, trend)
            
            # Save to CSV
            output_file = OUTPUT_DIR / f"{symbol}_4h.csv"
            df.to_csv(output_file, index=False)
            print(f"  Generated {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Price range: ${df['low'].min():.4f} - ${df['high'].max():.4f}")
            print(f"  Saved to {output_file}")
            
        except Exception as e:
            print(f"Error generating {symbol}: {e}")
            continue
    
    print(f"\nCompleted: synthetic data generated for all pairs")
    print("Note: This is synthetic data designed to mimic crypto market behavior for backtesting.")

if __name__ == "__main__":
    main()