#!/usr/bin/env python3
"""
Backtest for bull/neutral/greed market strategies.
Tests: pullback_buy, breakout_detect, ema_crossover, trend_continuation,
       greed_short, distribution_short, overextension_short
"""
import requests
import time as _time
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────
PAIRS = [
    "NEARUSD", "UNIUSD", "AVAXUSD", "LINKUSD", "AAVEUSD", "SOLUSD",
    "ETHUSD", "XBTUSD", "DOTUSD", "XLMUSD", "XRPUSD", "ADAUSD",
    "ATOMUSD", "DOGEUSD", "FILUSD", "LTCUSD"
]

KRAKEN_BASE = "https://api.kraken.com/0"
CACHE_DIR = Path(__file__).parent / "data" / "ohlcv_cache"
RESULTS_FILE = Path(__file__).parent / "data" / "bull_backtest_results.txt"
DAYS = 90
INTERVAL = 60  # 1h candles
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005

# Kraken pair name mappings (API sometimes uses different names)
PAIR_MAP = {
    "XBTUSD": "XXBTZUSD",
    "ETHUSD": "XETHZUSD",
    "XRPUSD": "XXRPZUSD",
    "XLMUSD": "XXLMZUSD",
    "LTCUSD": "XLTCZUSD",
    "ADAUSD": "ADAUSD",
    "DOTUSD": "DOTUSD",
    "LINKUSD": "LINKUSD",
    "SOLUSD": "SOLUSD",
    "AVAXUSD": "AVAXUSD",
    "AAVEUSD": "AAVEUSD",
    "UNIUSD": "UNIUSD",
    "NEARUSD": "NEARUSD",
    "ATOMUSD": "ATOMUSD",
    "DOGEUSD": "XDGUSD",
    "FILUSD": "FILUSD",
}

# ── Data Download ───────────────────────────────────────────────
def download_pair(pair: str, days: int = DAYS) -> pd.DataFrame:
    """Download OHLCV data from Kraken, paging back to get `days` worth."""
    cache_file = CACHE_DIR / f"{pair}_60m_backtest_{days}d.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        age_hours = (_time.time() - df['time'].max()) / 3600
        if age_hours < 24 and len(df) > days * 20:
            print(f"  {pair}: Using cache ({len(df)} bars)")
            return df

    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - (days * 86400)
    all_candles = []
    since = start_ts

    # Try the pair as-is first, then mapped name
    names_to_try = [pair]
    if pair in PAIR_MAP and PAIR_MAP[pair] != pair:
        names_to_try.append(PAIR_MAP[pair])

    api_pair = None
    while since < end_ts:
        for attempt_pair in (names_to_try if api_pair is None else [api_pair]):
            try:
                resp = requests.get(
                    f"{KRAKEN_BASE}/public/OHLC",
                    params={"pair": attempt_pair, "interval": INTERVAL, "since": since},
                    timeout=30
                )
                data = resp.json()
                if data.get("error") and len(data["error"]) > 0 and "Unknown" in str(data["error"]):
                    continue
                result = data.get("result", {})
                candles = None
                for key, val in result.items():
                    if isinstance(val, list) and len(val) > 0:
                        candles = val
                        api_pair = attempt_pair
                        break
                if candles:
                    break
            except Exception as e:
                print(f"  Error fetching {attempt_pair}: {e}")
                continue
        else:
            if api_pair is None:
                print(f"  Could not find valid Kraken pair for {pair}")
                return pd.DataFrame()
            break

        if not candles:
            break

        for c in candles:
            ts = int(c[0])
            if ts >= start_ts and ts <= end_ts:
                all_candles.append({
                    'time': ts,
                    'open': float(c[1]), 'high': float(c[2]),
                    'low': float(c[3]), 'close': float(c[4]),
                    'volume': float(c[6])
                })
        last_ts = int(candles[-1][0])
        if last_ts <= since:
            break
        since = last_ts
        _time.sleep(1.2)  # rate limit

    df = pd.DataFrame(all_candles)
    if not df.empty:
        df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)

    print(f"  {pair}: Downloaded {len(df)} bars (~{len(df)//24}d)")
    return df


# ── Indicators ──────────────────────────────────────────────────
def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder smoothed RSI."""
    rsi = np.full(len(close), np.nan)
    if len(close) < period + 1:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def calc_sma(close: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < period:
        return out
    cs = np.cumsum(close)
    out[period-1:] = (cs[period-1:] - np.concatenate([[0], cs[:-period]])) / period
    return out


def calc_ema(close: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < period:
        return out
    k = 2.0 / (period + 1)
    out[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        out[i] = close[i] * k + out[i-1] * (1 - k)
    return out


def calc_bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    mid = calc_sma(close, period)
    std = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i-period+1:i+1], ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    bandwidth = np.where(mid > 0, (upper - lower) / mid, np.nan)
    return mid, upper, lower, bandwidth


def compute_indicators(df: pd.DataFrame) -> dict:
    """Pre-compute all indicators for the full series."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    rsi = calc_rsi(close, 14)
    sma20 = calc_sma(close, 20)
    sma50 = calc_sma(close, 50)
    ema9 = calc_ema(close, 9)
    ema21 = calc_ema(close, 21)
    bb_mid, bb_upper, bb_lower, bb_bandwidth = calc_bollinger(close, 20, 2.0)

    # Rolling volume average (20-period)
    vol_avg20 = calc_sma(volume, 20)

    return {
        'close': close, 'high': high, 'low': low, 'volume': volume,
        'rsi': rsi, 'sma20': sma20, 'sma50': sma50,
        'ema9': ema9, 'ema21': ema21,
        'bb_mid': bb_mid, 'bb_upper': bb_upper, 'bb_lower': bb_lower,
        'bb_bandwidth': bb_bandwidth, 'vol_avg20': vol_avg20,
    }


# ── Strategy Checks ────────────────────────────────────────────
def check_pullback_buy(ind: dict, i: int) -> bool:
    """Price > SMA50, RSI 40-60, bounced off SMA20 in last 3 bars."""
    if i < 50:
        return False
    c, rsi, sma50, sma20, low = ind['close'], ind['rsi'], ind['sma50'], ind['sma20'], ind['low']
    if np.isnan(sma50[i]) or np.isnan(sma20[i]) or np.isnan(rsi[i]):
        return False
    if not (c[i] > sma50[i] and 40 <= rsi[i] <= 60):
        return False
    # Touched SMA20 in last 3 bars
    touched = False
    for j in range(max(0, i-2), i+1):
        if not np.isnan(sma20[j]) and low[j] <= sma20[j]:
            touched = True
            break
    return touched and c[i] > sma20[i]


def check_breakout_detect(ind: dict, i: int) -> bool:
    """BB squeeze then price breaks upper band with volume."""
    if i < 50:
        return False
    bw = ind['bb_bandwidth']
    c, bb_up, vol, vol_avg = ind['close'], ind['bb_upper'], ind['volume'], ind['vol_avg20']
    if np.isnan(bw[i]) or np.isnan(bb_up[i]) or np.isnan(vol_avg[i]) or vol_avg[i] <= 0:
        return False
    # Squeeze: current bandwidth <= 1.1x the min of last 20 bandwidths
    lookback = bw[max(0, i-20):i]
    valid = lookback[~np.isnan(lookback)]
    if len(valid) < 10:
        return False
    bw_low = np.min(valid)
    is_squeeze = bw[i] <= bw_low * 1.1
    breaks_upper = c[i] > bb_up[i]
    vol_ratio = vol[i] / vol_avg[i]
    return is_squeeze and breaks_upper and vol_ratio > 1.5


def check_ema_crossover(ind: dict, i: int) -> bool:
    """EMA9 crosses above EMA21 in last 3 bars, vol > 1.3x, price > SMA50."""
    if i < 50:
        return False
    ema9, ema21, sma50 = ind['ema9'], ind['ema21'], ind['sma50']
    c, vol, vol_avg = ind['close'], ind['volume'], ind['vol_avg20']
    if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(sma50[i]) or np.isnan(vol_avg[i]):
        return False
    if vol_avg[i] <= 0:
        return False
    # Check crossover in last 3 bars
    cross_up = False
    for j in range(1, min(4, i)):
        idx = i - j
        if idx < 1:
            break
        if (not np.isnan(ema9[idx-1]) and not np.isnan(ema21[idx-1]) and
            not np.isnan(ema9[idx]) and not np.isnan(ema21[idx])):
            if ema9[idx-1] <= ema21[idx-1] and ema9[idx] > ema21[idx]:
                cross_up = True
                break
    still_above = ema9[i] > ema21[i]
    vol_ratio = vol[i] / vol_avg[i]
    return cross_up and still_above and vol_ratio > 1.3 and c[i] > sma50[i]


def check_trend_continuation(ind: dict, i: int) -> bool:
    """Price > SMA50 (rising), RSI 50-70, ret_4h > 1%."""
    if i < 54:
        return False
    c, rsi, sma50 = ind['close'], ind['rsi'], ind['sma50']
    if np.isnan(sma50[i]) or np.isnan(sma50[i-10]) or np.isnan(rsi[i]):
        return False
    price_above = c[i] > sma50[i]
    sma_rising = sma50[i] > sma50[i-10]
    rsi_ok = 50 <= rsi[i] <= 70
    # ret_4h = 4-bar return
    ret_4h = (c[i] - c[i-4]) / c[i-4] * 100 if c[i-4] > 0 else 0
    return price_above and sma_rising and rsi_ok and ret_4h > 1.0


def check_greed_short(ind: dict, i: int) -> bool:
    """RSI > 75, ret_8h > 5%, price above BB upper.
    Note: We can't check F&G in backtest, so we use RSI > 75 + ret_8h > 5% as proxy."""
    if i < 50:
        return False
    c, rsi, bb_up = ind['close'], ind['rsi'], ind['bb_upper']
    if np.isnan(rsi[i]) or np.isnan(bb_up[i]):
        return False
    ret_8h = (c[i] - c[i-8]) / c[i-8] * 100 if i >= 8 and c[i-8] > 0 else 0
    return rsi[i] > 75 and ret_8h > 5 and c[i] > bb_up[i]


def check_distribution_short(ind: dict, i: int) -> bool:
    """Lower highs + declining volume + RSI divergence."""
    if i < 50:
        return False
    high, vol, rsi, c = ind['high'], ind['volume'], ind['rsi'], ind['close']
    if np.isnan(rsi[i]) or np.isnan(rsi[i-10]):
        return False
    # Lower highs
    h1, h5, h10 = high[i], high[i-5], high[i-10]
    lower_highs = h1 < h5 < h10
    # Declining volume (20%+)
    vol_recent = np.mean(vol[i-4:i+1])
    vol_prior = np.mean(vol[i-9:i-4])
    vol_declining = vol_prior > 0 and vol_recent < vol_prior * 0.8
    # RSI divergence: RSI fallen 5+ points while price flat
    rsi_falling = rsi[i] < rsi[i-10] - 5
    price_flat = c[i] >= c[i-10] * 0.98
    return lower_highs and vol_declining and rsi_falling and price_flat


def check_overextension_short(ind: dict, i: int) -> bool:
    """Price > 8% above SMA50, RSI > 70."""
    if i < 50:
        return False
    c, rsi, sma50 = ind['close'], ind['rsi'], ind['sma50']
    if np.isnan(sma50[i]) or np.isnan(rsi[i]) or sma50[i] <= 0:
        return False
    extension = (c[i] - sma50[i]) / sma50[i]
    return extension > 0.08 and rsi[i] > 70


# Strategy definitions: (name, check_fn, direction, hold_bars, sl_pct)
STRATEGIES = [
    ("pullback_buy",        check_pullback_buy,        "long",  24, 0.03),
    ("breakout_detect",     check_breakout_detect,      "long",  24, 0.04),
    ("ema_crossover",       check_ema_crossover,        "long",  24, 0.03),
    ("trend_continuation",  check_trend_continuation,   "long",  12, 0.025),
    ("greed_short",         check_greed_short,          "short", 24, 0.06),
    ("distribution_short",  check_distribution_short,   "short", 24, 0.04),
    ("overextension_short", check_overextension_short,  "short", 24, 0.05),
]


# ── Backtest Engine ─────────────────────────────────────────────
def simulate_trade(ind: dict, entry_idx: int, direction: str, hold_bars: int, sl_pct: float):
    """Simulate a single trade with SL and fixed hold period."""
    c = ind['close']
    h = ind['high']
    l = ind['low']
    n = len(c)
    entry_price = c[entry_idx]
    if entry_price <= 0:
        return None

    # Apply entry slippage + fee
    if direction == "long":
        effective_entry = entry_price * (1 + SLIPPAGE)
        sl_price = effective_entry * (1 - sl_pct)
    else:
        effective_entry = entry_price * (1 - SLIPPAGE)
        sl_price = effective_entry * (1 + sl_pct)

    exit_idx = min(entry_idx + hold_bars, n - 1)

    # Check SL bar by bar
    for j in range(entry_idx + 1, exit_idx + 1):
        if direction == "long":
            if l[j] <= sl_price:
                exit_price = sl_price * (1 - SLIPPAGE)
                pnl = (exit_price - effective_entry) / effective_entry - 2 * MAKER_FEE
                return {'exit_idx': j, 'exit_price': exit_price, 'pnl_pct': pnl * 100, 'reason': 'SL'}
        else:
            if h[j] >= sl_price:
                exit_price = sl_price * (1 + SLIPPAGE)
                pnl = (effective_entry - exit_price) / effective_entry - 2 * MAKER_FEE
                return {'exit_idx': j, 'exit_price': exit_price, 'pnl_pct': pnl * 100, 'reason': 'SL'}

    # Time exit
    exit_price = c[exit_idx]
    if direction == "long":
        exit_price_eff = exit_price * (1 - SLIPPAGE)
        pnl = (exit_price_eff - effective_entry) / effective_entry - 2 * MAKER_FEE
    else:
        exit_price_eff = exit_price * (1 + SLIPPAGE)
        pnl = (effective_entry - exit_price_eff) / effective_entry - 2 * MAKER_FEE
    return {'exit_idx': exit_idx, 'exit_price': exit_price, 'pnl_pct': pnl * 100, 'reason': 'TIMEOUT'}


def backtest_pair(pair: str, df: pd.DataFrame) -> list:
    """Run all strategies on a single pair."""
    if len(df) < 60:
        return []

    ind = compute_indicators(df)
    n = len(df)
    trades = []

    # Track last entry per strategy to avoid overlapping trades (cooldown = hold period)
    last_entry = {s[0]: -999 for s in STRATEGIES}

    for i in range(55, n - 1):  # need room for exit
        for name, check_fn, direction, hold_bars, sl_pct in STRATEGIES:
            # Skip if still in a trade for this strategy
            if i - last_entry[name] < hold_bars:
                continue
            # Ensure enough bars for exit
            if i + hold_bars >= n:
                continue
            if check_fn(ind, i):
                result = simulate_trade(ind, i, direction, hold_bars, sl_pct)
                if result is not None:
                    trades.append({
                        'pair': pair,
                        'tool': name,
                        'direction': direction,
                        'entry_bar': i,
                        'entry_price': ind['close'][i],
                        'exit_price': result['exit_price'],
                        'pnl_pct': result['pnl_pct'],
                        'hold_bars': result['exit_idx'] - i,
                        'reason': result['reason'],
                    })
                    last_entry[name] = i
    return trades


# ── Summary ─────────────────────────────────────────────────────
def summarize(trades: list, label: str) -> str:
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {label}")
    lines.append(f"{'='*60}")

    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    total_pnl = sum(pnls)
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    best = max(pnls)
    worst = min(pnls)

    # Count by reason
    sl_count = sum(1 for t in trades if t['reason'] == 'SL')
    timeout_count = sum(1 for t in trades if t['reason'] == 'TIMEOUT')

    lines.append(f"  Total trades:    {total_trades}")
    lines.append(f"  Win rate:        {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    lines.append(f"  Avg win:         +{avg_win:.2f}%")
    lines.append(f"  Avg loss:        {avg_loss:.2f}%")
    lines.append(f"  Total PnL:       {'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%")
    lines.append(f"  Profit factor:   {profit_factor:.2f}")
    lines.append(f"  Best trade:      +{best:.2f}%")
    lines.append(f"  Worst trade:     {worst:.2f}%")
    lines.append(f"  Exits:           {sl_count} SL, {timeout_count} timeout")

    # Per-pair breakdown
    pairs_seen = sorted(set(t['pair'] for t in trades))
    if len(pairs_seen) > 1:
        lines.append(f"\n  Per-pair breakdown:")
        for p in pairs_seen:
            pt = [t['pnl_pct'] for t in trades if t['pair'] == p]
            pw = [x for x in pt if x > 0]
            lines.append(f"    {p:12s}  {len(pt):3d} trades  WR={len(pw)/len(pt)*100:.0f}%  PnL={sum(pt):+.2f}%")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BULL/NEUTRAL/GREED STRATEGY BACKTEST")
    print(f"  {DAYS} days, {len(PAIRS)} pairs, 1h candles")
    print("=" * 60)
    print()

    all_trades = []

    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        df = download_pair(pair, DAYS)
        if df.empty or len(df) < 60:
            print(f"  Skipping {pair}: insufficient data ({len(df)} bars)")
            continue

        pair_trades = backtest_pair(pair, df)
        all_trades.extend(pair_trades)
        # Quick pair summary
        for name in [s[0] for s in STRATEGIES]:
            st = [t for t in pair_trades if t['tool'] == name]
            if st:
                pnl = sum(t['pnl_pct'] for t in st)
                wr = sum(1 for t in st if t['pnl_pct'] > 0) / len(st) * 100
                print(f"    {name:25s} {len(st):3d} trades  WR={wr:.0f}%  PnL={pnl:+.2f}%")

    # ── Generate Report ─────────────────────────────────────────
    output = []
    output.append("=" * 60)
    output.append("  BULL/NEUTRAL/GREED STRATEGY BACKTEST RESULTS")
    output.append(f"  Period: {DAYS} days | Pairs: {len(PAIRS)} | Interval: 1h")
    output.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output.append(f"  Fees: {MAKER_FEE*100:.2f}% maker + {SLIPPAGE*100:.2f}% slippage per side")
    output.append("=" * 60)

    # Per-strategy summary
    for name, _, direction, hold_bars, sl_pct in STRATEGIES:
        strades = [t for t in all_trades if t['tool'] == name]
        label = f"{name} ({direction.upper()}, hold={hold_bars}h, SL={sl_pct*100:.1f}%)"
        output.append(summarize(strades, label))

    # Combined summary
    output.append(summarize(all_trades, "ALL STRATEGIES COMBINED"))

    # Long vs Short breakdown
    longs = [t for t in all_trades if t['direction'] == 'long']
    shorts = [t for t in all_trades if t['direction'] == 'short']
    output.append(summarize(longs, "ALL LONG STRATEGIES"))
    output.append(summarize(shorts, "ALL SHORT STRATEGIES"))

    report = "\n".join(output)
    print("\n" + report)

    # Save to file
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    print(f"\n✅ Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
