#!/usr/bin/env python3
"""
Portfolio simulation: run ALL 37 edge configs simultaneously on real data,
max 3 positions at a time, $300 starting balance, realistic fees.
This is the most honest test — exactly what the bot would have done.
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005
MAX_POSITIONS = 3
RISK_PCT = 0.03
INITIAL_BALANCE = 300

EDGE_CONFIGS = [
    ("UNIUSD", 15, "short", 10, 2.5, 4.0, 50, 36, 15.6),
    ("FILUSD", 60, "short", 10, 1.5, 5.0, 30, 36, 15.2),
    ("ATOMUSD", 240, "short", 10, 2.0, 5.0, 30, 72, 12.5),
    ("ADAUSD", 240, "long", 20, 1.5, 3.0, 30, 36, 11.7),
    ("LINKUSD", 15, "short", 20, 1.5, 5.0, 50, 72, 11.5),
    ("ATOMUSD", 60, "long", 10, 2.0, 4.0, 30, 72, 11.4),
    ("ADAUSD", 15, "short", 10, 1.5, 3.0, 50, 48, 10.0),
    ("SOLUSD", 240, "long", 10, 2.0, 3.0, 50, 48, 9.4),
    ("XRPUSD", 60, "long", 20, 2.5, 4.0, 30, 48, 9.3),
    ("ETHUSD", 240, "long", 15, 2.0, 3.0, 30, 36, 9.2),
    ("AVAXUSD", 240, "long", 10, 2.0, 3.0, 50, 36, 9.2),
    ("AVAXUSD", 60, "short", 15, 1.5, 5.0, 50, 48, 9.2),
    ("NEARUSD", 60, "long", 10, 2.0, 5.0, 30, 36, 9.1),
    ("ETHUSD", 60, "short", 20, 1.5, 5.0, 50, 48, 9.1),
    ("ETHUSD", 60, "long", 15, 2.0, 5.0, 30, 48, 8.6),
    ("XBTUSD", 240, "long", 20, 2.0, 3.0, 30, 48, 8.6),
    ("AAVEUSD", 15, "short", 20, 2.0, 4.0, 30, 36, 8.5),
    ("XRPUSD", 15, "short", 10, 2.0, 3.0, 50, 36, 8.2),
    ("ATOMUSD", 15, "short", 10, 2.5, 5.0, 30, 72, 7.9),
    ("XBTUSD", 15, "short", 10, 2.0, 3.0, 30, 72, 7.7),
    ("FILUSD", 60, "long", 15, 2.5, 4.0, 50, 72, 7.6),
    ("XLMUSD", 60, "long", 10, 2.0, 3.0, 30, 36, 7.6),
    ("DOGEUSD", 15, "short", 15, 1.5, 4.0, 50, 72, 7.4),
    ("UNIUSD", 60, "short", 20, 2.0, 5.0, 30, 48, 7.0),
    ("SOLUSD", 60, "short", 15, 1.5, 5.0, 50, 36, 6.6),
    ("XRPUSD", 60, "short", 20, 1.5, 5.0, 50, 72, 6.2),
    ("FILUSD", 240, "short", 15, 1.5, 5.0, 50, 48, 6.2),
    ("ATOMUSD", 240, "long", 20, 1.5, 5.0, 50, 36, 6.0),
    ("XBTUSD", 60, "long", 10, 2.0, 3.0, 50, 48, 5.9),
    ("DOTUSD", 240, "long", 15, 1.5, 4.0, 30, 36, 5.6),
    ("AAVEUSD", 60, "long", 10, 2.0, 5.0, 30, 36, 5.5),
    ("LTCUSD", 15, "short", 20, 2.5, 5.0, 30, 72, 5.5),
    ("DOTUSD", 60, "short", 15, 1.5, 4.0, 30, 36, 5.4),
    ("AVAXUSD", 60, "long", 10, 2.5, 5.0, 50, 48, 4.7),
    ("DOTUSD", 60, "long", 15, 2.5, 5.0, 30, 36, 4.2),
    ("DOGEUSD", 60, "long", 10, 2.0, 5.0, 30, 48, 4.2),
    ("LINKUSD", 240, "long", 10, 2.0, 4.0, 30, 48, 4.1),
]


def download(pair, interval_min, days):
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - (days * 86400)
    all_candles = []
    since = start_ts
    while since < end_ts:
        try:
            resp = requests.get(f"{KRAKEN_BASE}/public/OHLC",
                params={"pair": pair, "interval": interval_min, "since": since}, timeout=30)
            result = resp.json().get("result", {})
        except:
            break
        candles = None
        for key, val in result.items():
            if isinstance(val, list):
                candles = val
                break
        if not candles:
            break
        for c in candles:
            ts = int(c[0])
            if ts > start_ts and ts <= end_ts:
                all_candles.append({
                    'time': ts, 'open': float(c[1]), 'high': float(c[2]),
                    'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[6])
                })
        last_ts = int(candles[-1][0])
        if last_ts <= since:
            break
        since = last_ts
        _time.sleep(1.2)
    df = pd.DataFrame(all_candles)
    if not df.empty:
        df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
    return df


def check_signal_at_bar(df, bar_idx, direction, breakout_bars, sl_atr_mult,
                        tp_atr_mult, sma_period):
    """Check if a breakout signal fires at a specific bar. Returns signal dict or None."""
    min_needed = max(breakout_bars, sma_period, 20)
    if bar_idx < min_needed:
        return None

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # ATR
    h = high[:bar_idx+1].astype(float)
    l = low[:bar_idx+1].astype(float)
    c = close[:bar_idx+1].astype(float)
    if len(c) < 15:
        return None
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_val = float(np.mean(tr[-14:])) if len(tr) >= 14 else 0
    if atr_val <= 0:
        return None

    # SMA
    if len(c) < sma_period:
        return None
    sma_val = float(np.mean(c[-sma_period:]))

    cur_close = float(c[-1])
    cur_vol = float(volume[bar_idx])
    vol_avg = float(np.mean(volume[max(0,bar_idx-20):bar_idx])) if bar_idx >= 20 else 0

    if direction == "long":
        if cur_close <= sma_val:
            return None
        prev_high = float(np.max(high[bar_idx-breakout_bars:bar_idx]))
        if cur_close <= prev_high:
            return None
        entry = cur_close * (1 + SLIPPAGE)
        recent_low = float(np.min(low[bar_idx-breakout_bars:bar_idx]))
        sl = max(recent_low * 0.998, entry - sl_atr_mult * atr_val)
        tp = entry + tp_atr_mult * atr_val
        risk = entry - sl
        reward = tp - entry
    else:  # short
        if cur_close >= sma_val:
            return None
        prev_low = float(np.min(low[bar_idx-breakout_bars:bar_idx]))
        if cur_close >= prev_low:
            return None
        entry = cur_close * (1 - SLIPPAGE)
        recent_high = float(np.max(high[bar_idx-breakout_bars:bar_idx]))
        sl = min(recent_high * 1.002, entry + sl_atr_mult * atr_val)
        tp = entry - tp_atr_mult * atr_val
        risk = sl - entry
        reward = entry - tp

    if risk <= 0 or reward / risk < 1.5:
        return None

    return {
        "entry": entry, "sl": sl, "tp": tp, "atr": atr_val,
        "risk": risk, "reward": reward, "rr": reward / risk,
    }


def main():
    # Determine unique (pair, tf) combos we need
    needed = set()
    for cfg in EDGE_CONFIGS:
        needed.add((cfg[0], cfg[1]))

    # Kraken returns max 720 bars. Calculate max days per timeframe.
    tf_days = {15: 7, 60: 30, 240: 90}

    print("Downloading data for portfolio simulation...")
    data_cache = {}
    for pair, tf_min in sorted(needed):
        days = tf_days.get(tf_min, 30)
        key = f"{pair}_{tf_min}"
        print(f"  {pair} {tf_min}m ({days}d)...", end=" ", flush=True)
        df = download(pair, tf_min, days)
        print(f"{len(df)} bars")
        data_cache[key] = df

    print(f"\nDownloaded {len(data_cache)} datasets.")

    # ═══════════════════════════════════════════════════
    # PORTFOLIO SIMULATION — bar-by-bar, time-synchronized
    # ═══════════════════════════════════════════════════

    # We simulate using the LOWEST common timeframe approach:
    # For each bar on each (pair, tf), check all configs that match.
    # Process in chronological order using timestamps.

    # Build a master timeline: list of (timestamp, pair, tf, bar_idx)
    events = []
    for key, df in data_cache.items():
        pair, tf_str = key.rsplit("_", 1)
        tf_min = int(tf_str)
        for idx in range(len(df)):
            ts = int(df.iloc[idx]['time'])
            events.append((ts, pair, tf_min, idx))

    events.sort(key=lambda x: x[0])

    # State
    balance = float(INITIAL_BALANCE)
    positions = []  # list of dicts: {pair, direction, entry, qty, sl, tp, entry_ts, timeout_bars, bars_held, config_key}
    trades = []
    peak_balance = balance
    max_dd = 0
    last_ts_per_key = {}

    for ts, pair, tf_min, bar_idx in events:
        key = f"{pair}_{tf_min}"
        df = data_cache[key]

        # Avoid processing same bar twice
        if last_ts_per_key.get(key, 0) >= ts:
            continue
        last_ts_per_key[key] = ts

        cur_high = float(df.iloc[bar_idx]['high'])
        cur_low = float(df.iloc[bar_idx]['low'])
        cur_close = float(df.iloc[bar_idx]['close'])

        # ── Manage existing positions for this pair/tf ──
        remaining = []
        for pos in positions:
            if pos['pair'] != pair or pos['tf'] != tf_min:
                remaining.append(pos)
                continue

            pos['bars_held'] += 1
            exit_price = None
            reason = None

            if pos['direction'] == 'long':
                if cur_low <= pos['sl']:
                    exit_price = max(pos['sl'], cur_low) * (1 - SLIPPAGE)
                    reason = "SL"
                elif cur_high >= pos['tp']:
                    exit_price = min(pos['tp'], cur_high) * (1 - SLIPPAGE)
                    reason = "TP"
                elif pos['bars_held'] >= pos['timeout']:
                    exit_price = cur_close * (1 - SLIPPAGE)
                    reason = "TIMEOUT"
            else:  # short
                if cur_high >= pos['sl']:
                    exit_price = min(pos['sl'], cur_high) * (1 + SLIPPAGE)
                    reason = "SL"
                elif cur_low <= pos['tp']:
                    exit_price = max(pos['tp'], cur_low) * (1 + SLIPPAGE)
                    reason = "TP"
                elif pos['bars_held'] >= pos['timeout']:
                    exit_price = cur_close * (1 + SLIPPAGE)
                    reason = "TIMEOUT"

            if exit_price is not None:
                if pos['direction'] == 'long':
                    pnl = (exit_price - pos['entry']) * pos['qty']
                else:
                    pnl = (pos['entry'] - exit_price) * pos['qty']
                fees = pos['entry'] * pos['qty'] * MAKER_FEE + exit_price * pos['qty'] * MAKER_FEE
                net_pnl = pnl - fees

                if pos['direction'] == 'long':
                    balance += pos['entry'] * pos['qty'] + net_pnl
                else:
                    balance += net_pnl

                trades.append({
                    'pair': pos['pair'], 'dir': pos['direction'], 'tf': pos['tf'],
                    'entry': pos['entry'], 'exit': exit_price, 'pnl': net_pnl,
                    'reason': reason, 'bars': pos['bars_held'],
                })
                peak_balance = max(peak_balance, balance)
                dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
                max_dd = max(max_dd, dd)
            else:
                remaining.append(pos)

        positions = remaining

        # ── Check for new entries ──
        if len(positions) >= MAX_POSITIONS:
            continue
        # Don't enter if drawdown too high
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        if dd > 0.15:
            continue

        # Check all configs matching this pair/tf
        active_symbols = set(p['pair'] for p in positions)
        candidates = []

        for cfg in EDGE_CONFIGS:
            c_pair, c_tf, c_dir, c_bb, c_sl, c_tp, c_sma, c_tout, c_wf = cfg
            if c_pair != pair or c_tf != tf_min:
                continue
            if c_pair in active_symbols:
                continue

            sig = check_signal_at_bar(df, bar_idx, c_dir, c_bb, c_sl, c_tp, c_sma)
            if sig:
                score = c_wf * sig['rr']
                candidates.append((cfg, sig, score))

        if not candidates:
            continue

        # Take best candidate
        candidates.sort(key=lambda x: x[2], reverse=True)
        cfg, sig, score = candidates[0]
        c_pair, c_tf, c_dir, c_bb, c_sl, c_tp, c_sma, c_tout, c_wf = cfg

        # Position sizing: 3% risk
        risk_dollars = balance * RISK_PCT
        qty = risk_dollars / sig['risk']
        cost = qty * sig['entry']

        if c_dir == 'long':
            if cost > balance * 0.95:
                qty = (balance * 0.95) / sig['entry']
                cost = qty * sig['entry']
            if cost < 3:
                continue
            balance -= cost
        else:
            # Short: risk-based sizing, don't deduct full notional
            if risk_dollars > balance * 0.30:
                continue

        positions.append({
            'pair': c_pair, 'direction': c_dir, 'tf': c_tf,
            'entry': sig['entry'], 'qty': qty, 'sl': sig['sl'], 'tp': sig['tp'],
            'timeout': c_tout, 'bars_held': 0, 'config_key': f"{c_pair}_{c_tf}_{c_dir}",
        })

    # Close remaining positions at last price
    for pos in positions:
        key = f"{pos['pair']}_{pos['tf']}"
        df = data_cache.get(key)
        if df is not None and len(df) > 0:
            last_close = float(df.iloc[-1]['close'])
            if pos['direction'] == 'long':
                pnl = (last_close - pos['entry']) * pos['qty']
                fees = pos['entry'] * pos['qty'] * MAKER_FEE + last_close * pos['qty'] * MAKER_FEE
                balance += pos['entry'] * pos['qty'] + (pnl - fees)
            else:
                pnl = (pos['entry'] - last_close) * pos['qty']
                fees = pos['entry'] * pos['qty'] * MAKER_FEE + last_close * pos['qty'] * MAKER_FEE
                balance += (pnl - fees)
            trades.append({
                'pair': pos['pair'], 'dir': pos['direction'], 'tf': pos['tf'],
                'entry': pos['entry'], 'exit': last_close, 'pnl': pnl - fees,
                'reason': 'END', 'bars': pos['bars_held'],
            })

    # ═══════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PORTFOLIO SIMULATION RESULTS")
    print("  37 configs, max 3 positions, $300 start, realistic fees")
    print("=" * 70)

    total_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001

    print(f"\n  Starting balance:  ${INITIAL_BALANCE:.2f}")
    print(f"  Final balance:     ${balance:.2f}")
    print(f"  Total return:      {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:+.2f}%")
    print(f"  Total PnL:         ${total_pnl:+.2f}")
    print(f"  Max drawdown:      {max_dd * 100:.2f}%")
    print(f"\n  Total trades:      {total_trades}")
    print(f"  Wins:              {len(wins)}")
    print(f"  Losses:            {len(losses)}")
    print(f"  Win rate:          {len(wins)/total_trades*100:.1f}%" if total_trades > 0 else "  Win rate:          N/A")
    print(f"  Profit factor:     {gp/gl:.2f}" if gl > 0.001 else "  Profit factor:     N/A")
    print(f"  Avg win:           ${gp/len(wins):.2f}" if wins else "  Avg win:           N/A")
    print(f"  Avg loss:          ${-gl/len(losses):.2f}" if losses else "  Avg loss:          N/A")

    # By direction
    longs = [t for t in trades if t['dir'] == 'long']
    shorts = [t for t in trades if t['dir'] == 'short']
    print(f"\n  Long trades:       {len(longs)} (wins: {sum(1 for t in longs if t['pnl']>0)})")
    print(f"  Short trades:      {len(shorts)} (wins: {sum(1 for t in shorts if t['pnl']>0)})")

    # By exit reason
    reasons = defaultdict(int)
    for t in trades:
        reasons[t['reason']] += 1
    print(f"\n  Exits: {dict(reasons)}")

    # Individual trades
    print(f"\n  {'Pair':10s} {'Dir':6s} {'TF':>4s} {'Entry':>10s} {'Exit':>10s} {'PnL':>8s} {'Reason':>8s} {'Bars':>5s}")
    print("  " + "-" * 65)
    for t in trades:
        print(f"  {t['pair']:10s} {t['dir']:6s} {t['tf']:>4d}m ${t['entry']:>9.4f} ${t['exit']:>9.4f} "
              f"${t['pnl']:>+7.2f} {t['reason']:>8s} {t['bars']:>5d}")

    # Compound projection
    if total_trades > 0 and total_pnl > 0:
        # Figure out the time span
        all_tfs = set(t['tf'] for t in trades)
        monthly_ret = (balance / INITIAL_BALANCE - 1)  # This is over the data period
        print(f"\n  Monthly return estimate: ~{monthly_ret*100:.1f}%")
        print(f"\n  Compound projection (if this holds):")
        for months in [3, 6, 12, 24, 36, 60, 120]:
            projected = INITIAL_BALANCE * (1 + monthly_ret) ** months
            print(f"    {months:>3d} months: ${projected:>12,.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
