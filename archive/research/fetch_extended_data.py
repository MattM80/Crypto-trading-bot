#!/usr/bin/env python3
"""Fetch 3 years of 1h Binance data for all pairs."""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent / "data" / "binance_1h_extended"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT",
    "AVAXUSDT", "ATOMUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "UNIUSDT",
    "FILUSDT", "NEARUSDT", "AAVEUSDT", "XLMUSDT"
]

def fetch_klines(symbol, interval="1h", start_time=None, end_time=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time:
        params["endTime"] = int(end_time.timestamp() * 1000)
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_full_history(symbol, start_date, end_date):
    """Fetch all 1h candles between start and end."""
    all_data = []
    current = start_date
    
    while current < end_date:
        try:
            klines = fetch_klines(symbol, start_time=current, end_time=end_date)
            if not klines:
                break
            
            for k in klines:
                all_data.append({
                    'timestamp': pd.Timestamp(k[0], unit='ms', tz='UTC'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            # Move to after last candle
            last_ts = klines[-1][0]
            current = datetime.utcfromtimestamp(last_ts / 1000) + timedelta(hours=1)
            
            if len(klines) < 1000:
                break
            
            time.sleep(0.2)  # Rate limit
        except Exception as e:
            print(f"  Error at {current}: {e}")
            time.sleep(2)
            continue
    
    return all_data


if __name__ == "__main__":
    # 3 years back from now
    end_date = datetime(2026, 3, 27)
    start_date = datetime(2023, 3, 27)
    
    for pair in PAIRS:
        outfile = DATA_DIR / f"{pair}_1h.csv"
        if outfile.exists():
            existing = pd.read_csv(outfile)
            print(f"{pair}: already have {len(existing)} bars, skipping")
            continue
        
        print(f"Fetching {pair}...", end=" ", flush=True)
        data = fetch_full_history(pair, start_date, end_date)
        
        if data:
            df = pd.DataFrame(data)
            df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
            df.to_csv(outfile, index=False)
            print(f"{len(df)} bars ({df.iloc[0].timestamp} → {df.iloc[-1].timestamp})")
        else:
            print("NO DATA")
        
        time.sleep(0.5)
    
    print("\nDone!")
