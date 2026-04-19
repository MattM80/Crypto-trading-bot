#!/usr/bin/env python3
"""
Fix the edge. The problem: 33% win rate, too many false breakouts.

Fixes to test:
1. Top configs only (WF return > 8%) — sharper edge, less noise
2. Volume confirmation — require 1.2x+ avg volume on breakout
3. Multi-bar confirmation — require 2 closes beyond breakout level
4. Higher risk per trade (5%) — fewer but bigger bets
5. Tighter SL (1.5x ATR) with wider TP (keep asymmetry)
6. Momentum filter — price must be moving strongly (ROC > 0.5%)
7. Combination of all the above
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005
INITIAL_BALANCE = 300

ALL_CONFIGS = [
    # (pair, tf, dir, bb, sl_mult, tp_mult, sma, timeout, wf_ret)
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


def check_signal(df, bar_idx, direction, breakout_bars, sl_atr_mult, tp_atr_mult,
                 sma_period, require_volume=False, require_momentum=False,
                 require_confirmation=False):
    """Check breakout signal with optional filters."""
    min_needed = max(breakout_bars, sma_period, 20)
    if bar_idx < min_needed + 2:  # +2 for confirmation check
        return None

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    c = close[:bar_idx+1].astype(float)
    h = high[:bar_idx+1].astype(float)
    l = low[:bar_idx+1].astype(float)
    v = volume[:bar_idx+1].astype(float)

    if len(c) < max(15, sma_period):
        return None

    # ATR
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_val = float(np.mean(tr[-14:])) if len(tr) >= 14 else 0
    if atr_val <= 0:
        return None

    sma_val = float(np.mean(c[-sma_period:]))
    cur_close = float(c[-1])
    prev_close = float(c[-2])

    # Volume check
    if require_volume:
        vol_avg = float(np.mean(v[-21:-1])) if len(v) >= 21 else 0
        cur_vol = float(v[-1])
        if vol_avg <= 0 or cur_vol < vol_avg * 1.2:
            return None

    # Momentum check (rate of change > 0.5% over 3 bars)
    if require_momentum:
        if len(c) >= 4:
            roc = (cur_close - float(c[-4])) / float(c[-4]) * 100
            if direction == "long" and roc < 0.5:
                return None
            if direction == "short" and roc > -0.5:
                return None

    if direction == "long":
        if cur_close <= sma_val:
            return None
        prev_high = float(np.max(h[-breakout_bars-1:-1]))
        if cur_close <= prev_high:
            return None

        # Confirmation: previous bar also closed above breakout
        if require_confirmation:
            prev_prev_high = float(np.max(h[-breakout_bars-2:-2]))
            if prev_close <= prev_prev_high:
                return None

        entry = cur_close * (1 + SLIPPAGE)
        recent_low = float(np.min(l[-breakout_bars-1:-1]))
        sl = max(recent_low * 0.998, entry - sl_atr_mult * atr_val)
        tp = entry + tp_atr_mult * atr_val
        risk = entry - sl
        reward = tp - entry
    else:
        if cur_close >= sma_val:
            return None
        prev_low = float(np.min(l[-breakout_bars-1:-1]))
        if cur_close >= prev_low:
            return None

        if require_confirmation:
            prev_prev_low = float(np.min(l[-breakout_bars-2:-2]))
            if prev_close >= prev_prev_low:
                return None

        entry = cur_close * (1 - SLIPPAGE)
        recent_high = float(np.max(h[-breakout_bars-1:-1]))
        sl = min(recent_high * 1.002, entry + sl_atr_mult * atr_val)
        tp = entry - tp_atr_mult * atr_val
        risk = sl - entry
        reward = entry - tp

    if risk <= 0 or reward / risk < 1.5:
        return None

    return {"entry": entry, "sl": sl, "tp": tp, "risk": risk, "reward": reward,
            "rr": reward / risk, "atr": atr_val}


def simulate_portfolio(data_cache, configs, max_positions=3, risk_pct=0.03,
                       require_volume=False, require_momentum=False,
                       require_confirmation=False, min_rr=1.5):
    """Run portfolio simulation with given parameters."""
    events = []
    for key, df in data_cache.items():
        pair, tf_str = key.rsplit("_", 1)
        tf_min = int(tf_str)
        for idx in range(len(df)):
            ts = int(df.iloc[idx]['time'])
            events.append((ts, pair, tf_min, idx))
    events.sort(key=lambda x: x[0])

    balance = float(INITIAL_BALANCE)
    positions = []
    trades = []
    peak_balance = balance
    max_dd = 0
    last_ts_per_key = {}

    for ts, pair, tf_min, bar_idx in events:
        key = f"{pair}_{tf_min}"
        df = data_cache[key]
        if last_ts_per_key.get(key, 0) >= ts:
            continue
        last_ts_per_key[key] = ts

        cur_high = float(df.iloc[bar_idx]['high'])
        cur_low = float(df.iloc[bar_idx]['low'])
        cur_close = float(df.iloc[bar_idx]['close'])

        # Manage positions
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
            else:
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
                trades.append({'pnl': net_pnl, 'reason': reason, 'pair': pos['pair'],
                              'dir': pos['direction']})
                peak_balance = max(peak_balance, balance)
                dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
                max_dd = max(max_dd, dd)
            else:
                remaining.append(pos)
        positions = remaining

        # New entries
        if len(positions) >= max_positions:
            continue
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        if dd > 0.15:
            continue

        active_symbols = set(p['pair'] for p in positions)
        candidates = []
        for cfg in configs:
            c_pair, c_tf, c_dir, c_bb, c_sl, c_tp, c_sma, c_tout, c_wf = cfg
            if c_pair != pair or c_tf != tf_min or c_pair in active_symbols:
                continue
            sig = check_signal(df, bar_idx, c_dir, c_bb, c_sl, c_tp, c_sma,
                             require_volume=require_volume,
                             require_momentum=require_momentum,
                             require_confirmation=require_confirmation)
            if sig and sig['rr'] >= min_rr:
                score = c_wf * sig['rr']
                candidates.append((cfg, sig, score))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[2], reverse=True)
        cfg, sig, score = candidates[0]
        c_pair, c_tf, c_dir, c_bb, c_sl, c_tp, c_sma, c_tout, c_wf = cfg

        risk_dollars = balance * risk_pct
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
            if risk_dollars > balance * 0.30:
                continue

        positions.append({
            'pair': c_pair, 'direction': c_dir, 'tf': c_tf,
            'entry': sig['entry'], 'qty': qty, 'sl': sig['sl'], 'tp': sig['tp'],
            'timeout': c_tout, 'bars_held': 0,
        })

    # Close remaining
    for pos in positions:
        key = f"{pos['pair']}_{pos['tf']}"
        df = data_cache.get(key)
        if df is not None and len(df) > 0:
            lc = float(df.iloc[-1]['close'])
            if pos['direction'] == 'long':
                pnl = (lc - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - lc) * pos['qty']
            fees = pos['entry'] * pos['qty'] * MAKER_FEE + lc * pos['qty'] * MAKER_FEE
            balance += (pos['entry'] * pos['qty'] + pnl - fees) if pos['direction'] == 'long' else (pnl - fees)
            trades.append({'pnl': pnl - fees, 'reason': 'END', 'pair': pos['pair'],
                          'dir': pos['direction']})

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001

    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'wr': round(len(wins)/len(trades)*100, 1) if trades else 0,
        'pf': round(gp/gl, 2) if gl > 0.001 else 999,
        'pnl': round(total_pnl, 2),
        'ret': round((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
        'final': round(balance, 2),
        'max_dd': round(max_dd * 100, 2),
        'avg_win': round(gp/len(wins), 2) if wins else 0,
        'avg_loss': round(-gl/len(losses), 2) if losses else 0,
        'exits': {r: sum(1 for t in trades if t['reason']==r) for r in set(t['reason'] for t in trades)},
        'by_dir': {'long': sum(1 for t in trades if t['dir']=='long'),
                   'short': sum(1 for t in trades if t['dir']=='short')},
    }


def main():
    needed = set()
    for cfg in ALL_CONFIGS:
        needed.add((cfg[0], cfg[1]))
    tf_days = {15: 7, 60: 30, 240: 90}

    print("Downloading data...")
    data_cache = {}
    for pair, tf_min in sorted(needed):
        days = tf_days.get(tf_min, 30)
        df = download(pair, tf_min, days)
        if len(df) > 0:
            data_cache[f"{pair}_{tf_min}"] = df
            print(f"  {pair} {tf_min}m: {len(df)} bars")

    TOP_CONFIGS = [c for c in ALL_CONFIGS if c[8] >= 8.0]  # WF return >= 8%
    ELITE_CONFIGS = [c for c in ALL_CONFIGS if c[8] >= 10.0]  # WF return >= 10%

    print(f"\nConfigs: ALL={len(ALL_CONFIGS)}, TOP(≥8%)={len(TOP_CONFIGS)}, ELITE(≥10%)={len(ELITE_CONFIGS)}")

    # ═══════════════════════════════════════════
    # TEST MATRIX
    # ═══════════════════════════════════════════
    tests = [
        # (name, configs, max_pos, risk_pct, vol, momentum, confirm, min_rr)
        ("BASELINE (original)", ALL_CONFIGS, 3, 0.03, False, False, False, 1.5),
        
        # Filter by config quality
        ("TOP configs only (WF≥8%)", TOP_CONFIGS, 3, 0.03, False, False, False, 1.5),
        ("ELITE configs (WF≥10%)", ELITE_CONFIGS, 3, 0.03, False, False, False, 1.5),
        
        # Risk adjustments
        ("TOP + 5% risk", TOP_CONFIGS, 3, 0.05, False, False, False, 1.5),
        ("ELITE + 5% risk", ELITE_CONFIGS, 3, 0.05, False, False, False, 1.5),
        ("TOP + 8% risk", TOP_CONFIGS, 3, 0.08, False, False, False, 1.5),
        
        # Filters
        ("TOP + volume filter", TOP_CONFIGS, 3, 0.03, True, False, False, 1.5),
        ("TOP + momentum filter", TOP_CONFIGS, 3, 0.03, False, True, False, 1.5),
        ("TOP + confirmation", TOP_CONFIGS, 3, 0.03, False, False, True, 1.5),
        ("TOP + vol + momentum", TOP_CONFIGS, 3, 0.03, True, True, False, 1.5),
        ("TOP + all filters", TOP_CONFIGS, 3, 0.03, True, True, True, 1.5),
        
        # Filters + higher risk
        ("TOP + vol + 5% risk", TOP_CONFIGS, 3, 0.05, True, False, False, 1.5),
        ("TOP + momentum + 5%", TOP_CONFIGS, 3, 0.05, False, True, False, 1.5),
        ("TOP + confirm + 5%", TOP_CONFIGS, 3, 0.05, False, False, True, 1.5),
        ("TOP + all filters + 5%", TOP_CONFIGS, 3, 0.05, True, True, True, 1.5),
        ("ELITE + vol + 5%", ELITE_CONFIGS, 3, 0.05, True, False, False, 1.5),
        ("ELITE + momentum + 5%", ELITE_CONFIGS, 3, 0.05, False, True, False, 1.5),
        ("ELITE + confirm + 5%", ELITE_CONFIGS, 3, 0.05, False, False, True, 1.5),
        
        # Higher R:R minimum
        ("TOP + R:R≥2.0 + 5%", TOP_CONFIGS, 3, 0.05, False, False, False, 2.0),
        ("TOP + R:R≥2.5 + 5%", TOP_CONFIGS, 3, 0.05, False, False, False, 2.5),
        
        # More positions
        ("TOP + 5 positions + 5%", TOP_CONFIGS, 5, 0.05, False, False, False, 1.5),
        ("TOP + 5 pos + vol + 5%", TOP_CONFIGS, 5, 0.05, True, False, False, 1.5),
        
        # Nuclear: ELITE + all filters + high risk + more positions
        ("NUCLEAR: ELITE+all+8%+5pos", ELITE_CONFIGS, 5, 0.08, True, True, True, 1.5),
        ("NUCLEAR: TOP+all+8%+5pos", TOP_CONFIGS, 5, 0.08, True, True, True, 1.5),
        ("NUCLEAR: TOP+confirm+8%+5pos", TOP_CONFIGS, 5, 0.08, False, False, True, 1.5),
    ]

    print(f"\n{'='*100}")
    print(f"  TESTING {len(tests)} CONFIGURATIONS")
    print(f"{'='*100}")
    print(f"\n  {'Name':<35s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'PnL':>8s} {'Return':>8s} {'DD':>6s} {'AvgW':>7s} {'AvgL':>7s} Exits")
    print("  " + "-" * 100)

    results = []
    for name, configs, max_pos, risk, vol, mom, conf, min_rr in tests:
        r = simulate_portfolio(data_cache, configs, max_pos, risk, vol, mom, conf, min_rr)
        results.append((name, r))
        exits_str = f"TP={r['exits'].get('TP',0)} SL={r['exits'].get('SL',0)} TO={r['exits'].get('TIMEOUT',0)}"
        color = "\033[92m" if r['ret'] > 0 else "\033[91m"
        reset = "\033[0m"
        print(f"  {name:<35s} {r['trades']:>5d}T {r['wr']:>5.1f}% {r['pf']:>5.2f} "
              f"{color}${r['pnl']:>+7.2f} {r['ret']:>+7.2f}%{reset} {r['max_dd']:>5.2f}% "
              f"${r['avg_win']:>6.2f} ${r['avg_loss']:>6.2f} {exits_str}")

    # Show top 5
    results.sort(key=lambda x: x[1]['ret'], reverse=True)
    print(f"\n{'='*100}")
    print(f"  TOP 5 CONFIGURATIONS")
    print(f"{'='*100}")
    for name, r in results[:5]:
        print(f"  {name}")
        print(f"    {r['trades']}T | {r['wr']}% WR | PF {r['pf']} | ${r['pnl']:+.2f} | "
              f"{r['ret']:+.2f}% | DD {r['max_dd']}% | AvgW ${r['avg_win']} / AvgL ${r['avg_loss']}")
        print(f"    Exits: {r['exits']} | Dirs: {r['by_dir']}")


if __name__ == "__main__":
    main()
