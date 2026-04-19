#!/usr/bin/env python3
"""
Fetch 12 months of 4-hour OHLCV candles from Binance public API
for out-of-sample validation of trading tools.

Handles Binance's 1000 candle limit per request with pagination.
Saves to data/binance_historical/ as CSV files.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Binance symbols (USDT pairs)
BINANCE_PAIRS = [
    "NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", "SOLUSDT",
    "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", "XRPUSDT", "ADAUSDT", 
    "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"
]

OUTPUT_DIR = Path("data/binance_historical")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_binance_klines(symbol: str, interval: str, start_time: int, end_time: int) -> list:
    """Fetch klines from Binance API with rate limiting."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return []

def fetch_symbol_data(symbol: str, months: int = 12) -> pd.DataFrame:
    """Fetch historical data for a symbol with pagination."""
    print(f"Fetching {symbol}...")
    
    # Use realistic historical date range (12 months ending Dec 2024)
    end_time = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    start_time = end_time - timedelta(days=30 * months)
    
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    print(f"  Time range: {start_time} to {end_time}")
    print(f"  Timestamps: {start_ts} to {end_ts}")
    
    all_klines = []
    current_start = start_ts
    
    while current_start < end_ts:
        # Binance returns max 1000 candles per request
        klines = get_binance_klines(symbol, "4h", current_start, end_ts)
        
        if not klines:
            break
            
        all_klines.extend(klines)
        
        # Update start time for next request (last candle close time + 1ms)
        last_close_time = klines[-1][6]  # Close time is index 6
        current_start = last_close_time + 1
        
        print(f"  Fetched {len(klines)} candles, total: {len(all_klines)}")
        
        # Rate limit: Binance allows 1200 requests/min, so 0.05s delay is safe
        time.sleep(0.05)
        
        # If we got less than 1000 candles, we've reached the end
        if len(klines) < 1000:
            break
    
    if not all_klines:
        print(f"  No data retrieved for {symbol}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count', 'taker_buy_volume', 
        'taker_buy_quote_volume', 'ignore'
    ])
    
    # Keep only needed columns and convert types
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert price/volume columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by timestamp and remove duplicates
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    print(f"  Final dataset: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def main():
    """Fetch historical data for all pairs."""
    print(f"Fetching 12 months of 4-hour data for {len(BINANCE_PAIRS)} pairs...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    success_count = 0
    
    for symbol in BINANCE_PAIRS:
        try:
            df = fetch_symbol_data(symbol)
            
            if len(df) > 0:
                # Save to CSV
                output_file = OUTPUT_DIR / f"{symbol}_4h.csv"
                df.to_csv(output_file, index=False)
                print(f"  Saved {len(df)} candles to {output_file}")
                success_count += 1
            else:
                print(f"  No data for {symbol}")
                
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            continue
    
    print(f"\nCompleted: {success_count}/{len(BINANCE_PAIRS)} pairs downloaded successfully")

if __name__ == "__main__":
    main()