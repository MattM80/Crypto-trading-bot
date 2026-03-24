#!/usr/bin/env python3
"""Fetch 12 months of REAL 4h candles from Binance public API."""
import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone, timedelta

PAIRS = [
    "NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", "SOLUSDT",
    "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", "XRPUSDT", "ADAUSDT",
    "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"
]

OUT_DIR = "data/binance_real"
os.makedirs(OUT_DIR, exist_ok=True)

# 12 months back from now
end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
start_ms = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp() * 1000)

def fetch_pair(symbol):
    all_candles = []
    current_start = start_ms
    while current_start < end_ms:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "4h",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 451:
            # US IP blocked, try binance.us
            url = "https://api.binance.us/api/v3/klines"
            resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if not data:
            break
        all_candles.extend(data)
        # Next batch starts after last candle
        current_start = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.2)  # Rate limit
    
    if not all_candles:
        return None
    
    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df

for pair in PAIRS:
    outfile = f"{OUT_DIR}/{pair}_4h.csv"
    if os.path.exists(outfile):
        existing = pd.read_csv(outfile)
        print(f"  {pair}: already have {len(existing)} candles, skipping")
        continue
    print(f"Fetching {pair}...")
    df = fetch_pair(pair)
    if df is not None:
        df.to_csv(outfile, index=False)
        print(f"  {pair}: {len(df)} candles, {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    else:
        print(f"  {pair}: FAILED")
    time.sleep(0.5)

print("\nDone!")
