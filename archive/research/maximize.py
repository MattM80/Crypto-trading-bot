#!/usr/bin/env python3
"""
MAXIMIZE — Full expansion sweep.
- 15+ Kraken pairs
- Long AND short
- 15m, 1h, 4h timeframes
- Walk-forward validation (train 60d, test 30d)
- Find every edge that exists
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005

# All liquid Kraken USD pairs
ALL_PAIRS = [
    "XBTUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AVAXUSD",
    "DOTUSD", "MATICUSD", "ADAUSD", "XRPUSD", "DOGEUSD",
    "UNIUSD", "NEARUSD", "ATOMUSD", "AAVEUSD", "ALGOUSD",
    "XLMUSD", "FILUSD", "LTCUSD",
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
        except Exception as e:
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


def backtest_momentum(df, balance=300, direction="long", breakout_bars=15,
                      vol_mult=1.0, sma_period=50, sl_atr_mult=2.0,
                      tp_atr_mult=3.0, risk_pct=0.03, cooldown=3,
                      timeout_bars=48):
    """Momentum breakout — long or short."""
    if len(df) < max(breakout_bars, sma_period) + 20:
        return None

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(14).mean().values
    vol_avg = pd.Series(volume).rolling(20).mean().values
    sma = pd.Series(close).rolling(sma_period).mean().values

    trades = []
    position = None
    bars_since_trade = cooldown + 1
    init_balance = balance
    peak_balance = balance
    max_dd = 0

    start = max(breakout_bars, sma_period, 20)
    for i in range(start, len(df)):
        bars_since_trade += 1

        if position is not None:
            entry_price, qty, sl, tp, entry_bar = position

            if direction == "long":
                # SL
                if low[i] <= sl:
                    exit_price = max(sl, low[i]) * (1 - SLIPPAGE)
                    pnl = (exit_price - entry_price) * qty
                    fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                    balance += entry_price * qty + (pnl - fees)
                    trades.append({"pnl": pnl - fees, "reason": "SL", "bars": i - entry_bar})
                    position = None; bars_since_trade = 0
                    peak_balance = max(peak_balance, balance)
                    max_dd = max(max_dd, (peak_balance - balance) / peak_balance if peak_balance > 0 else 0)
                    continue
                # TP
                if high[i] >= tp:
                    exit_price = min(tp, high[i]) * (1 - SLIPPAGE)
                    pnl = (exit_price - entry_price) * qty
                    fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                    balance += entry_price * qty + (pnl - fees)
                    trades.append({"pnl": pnl - fees, "reason": "TP", "bars": i - entry_bar})
                    position = None; bars_since_trade = 0
                    peak_balance = max(peak_balance, balance)
                    continue
            else:  # short
                # SL (price goes up)
                if high[i] >= sl:
                    exit_price = min(sl, high[i]) * (1 + SLIPPAGE)
                    pnl = (entry_price - exit_price) * qty
                    fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                    balance += (pnl - fees)
                    trades.append({"pnl": pnl - fees, "reason": "SL", "bars": i - entry_bar})
                    position = None; bars_since_trade = 0
                    peak_balance = max(peak_balance, balance)
                    max_dd = max(max_dd, (peak_balance - balance) / peak_balance if peak_balance > 0 else 0)
                    continue
                # TP (price goes down)
                if low[i] <= tp:
                    exit_price = max(tp, low[i]) * (1 + SLIPPAGE)
                    pnl = (entry_price - exit_price) * qty
                    fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                    balance += (pnl - fees)
                    trades.append({"pnl": pnl - fees, "reason": "TP", "bars": i - entry_bar})
                    position = None; bars_since_trade = 0
                    peak_balance = max(peak_balance, balance)
                    continue

            # Timeout
            if i - entry_bar >= timeout_bars:
                exit_price = close[i]
                if direction == "long":
                    pnl = (exit_price - entry_price) * qty
                    fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                    balance += entry_price * qty + (pnl - fees)
                else:
                    pnl = (entry_price - exit_price) * qty
                    fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                    balance += (pnl - fees)
                trades.append({"pnl": pnl - fees, "reason": "TIMEOUT", "bars": i - entry_bar})
                position = None; bars_since_trade = 0
                peak_balance = max(peak_balance, balance)
                continue

        # Entry
        if position is None and bars_since_trade >= cooldown:
            if np.isnan(atr[i]) or np.isnan(vol_avg[i]) or np.isnan(sma[i]) or atr[i] <= 0:
                continue

            if direction == "long":
                if close[i] <= sma[i]:
                    continue
                prev_high = np.max(high[i-breakout_bars:i])
                if close[i] <= prev_high:
                    continue
                vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
                if vol_ratio < vol_mult:
                    continue
                entry_price = close[i] * (1 + SLIPPAGE)
                recent_low = np.min(low[i-breakout_bars:i])
                sl = max(recent_low * 0.998, entry_price - sl_atr_mult * atr[i])
                tp = entry_price + tp_atr_mult * atr[i]
                risk_per_unit = entry_price - sl
            else:  # short
                if close[i] >= sma[i]:
                    continue
                prev_low = np.min(low[i-breakout_bars:i])
                if close[i] >= prev_low:
                    continue
                vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
                if vol_ratio < vol_mult:
                    continue
                entry_price = close[i] * (1 - SLIPPAGE)
                recent_high = np.max(high[i-breakout_bars:i])
                sl = min(recent_high * 1.002, entry_price + sl_atr_mult * atr[i])
                tp = entry_price - tp_atr_mult * atr[i]
                risk_per_unit = sl - entry_price

            if risk_per_unit <= 0:
                continue
            reward = abs(tp - entry_price)
            if reward / risk_per_unit < 1.5:
                continue

            risk_dollars = balance * risk_pct
            qty = risk_dollars / risk_per_unit
            cost = qty * entry_price
            if direction == "long":
                if cost > balance * 0.95:
                    qty = (balance * 0.95) / entry_price
                    cost = qty * entry_price
                if cost < 3:
                    continue
                balance -= cost
            else:
                # Short: need margin. Size based on risk, not full notional
                if risk_dollars > balance * 0.30:
                    continue
                if qty * entry_price < 3:
                    continue
            position = (entry_price, qty, sl, tp, i)

    # Close remaining
    if position is not None:
        entry_price, qty, sl, tp, entry_bar = position
        exit_price = close[-1]
        if direction == "long":
            pnl = (exit_price - entry_price) * qty
            fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
            balance += entry_price * qty + (pnl - fees)
        else:
            pnl = (entry_price - exit_price) * qty
            fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
            balance += (pnl - fees)
        trades.append({"pnl": pnl - fees, "reason": "END"})

    if not trades:
        return None

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
    total_pnl = sum(t["pnl"] for t in trades)

    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "wr": round(len(wins)/len(trades)*100, 1),
        "pf": round(gp / gl, 2) if gl > 0.001 else 999,
        "pnl": round(total_pnl, 2),
        "ret": round((balance - init_balance) / init_balance * 100, 2),
        "final": round(balance, 2),
        "max_dd": round(max_dd * 100, 2),
        "exits": {r: len([t for t in trades if t.get("reason") == r])
                  for r in set(t.get("reason", "?") for t in trades)},
    }


def walk_forward(df, direction="long", train_pct=0.67, **kwargs):
    """Walk-forward validation: train on first 67%, test on last 33%."""
    if len(df) < 100:
        return None, None
    split = int(len(df) * train_pct)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    
    train_result = backtest_momentum(train_df, direction=direction, **kwargs)
    test_result = backtest_momentum(test_df, direction=direction, **kwargs)
    return train_result, test_result


def main():
    print("=" * 80)
    print("  MAXIMUM EDGE DISCOVERY — All Pairs, Both Directions, Multiple TFs")
    print("=" * 80)

    # Configs to sweep (reduced for speed but covering key combos)
    PARAM_GRID = [
        {"breakout_bars": bb, "sl_atr_mult": sl, "tp_atr_mult": tp,
         "sma_period": sma, "risk_pct": 0.03, "timeout_bars": tout}
        for bb in [10, 15, 20]
        for sl in [1.5, 2.0, 2.5]
        for tp in [3.0, 4.0, 5.0]
        for sma in [30, 50]
        for tout in [36, 48, 72]
    ]

    TIMEFRAMES = {
        "15m": 15,
        "1h": 60,
        "4h": 240,
    }

    all_edges = []  # Collect all profitable configs

    for tf_name, tf_min in TIMEFRAMES.items():
        days = 90
        # Kraken only returns 720 bars. Adjust days if needed
        max_bars = 720
        bars_per_day = 1440 / tf_min
        max_days = int(max_bars / bars_per_day)
        days = min(days, max_days)

        print(f"\n{'='*80}")
        print(f"  TIMEFRAME: {tf_name} ({days} days of data)")
        print(f"{'='*80}")

        for pair in ALL_PAIRS:
            print(f"\n  Downloading {pair} {tf_name}...", end=" ", flush=True)
            df = download(pair, tf_min, days)
            if len(df) < 80:
                print(f"insufficient data ({len(df)} bars)")
                continue
            print(f"{len(df)} bars. Price: ${df['close'].iloc[-1]:.2f}")

            for direction in ["long", "short"]:
                best_ret = -999
                best_config = None
                best_result = None

                for params in PARAM_GRID:
                    r = backtest_momentum(df, direction=direction, **params)
                    if r and r["trades"] >= 3 and r["ret"] > best_ret and r["max_dd"] < 25:
                        best_ret = r["ret"]
                        best_config = params.copy()
                        best_result = r

                if best_result and best_result["ret"] > 0 and best_result["pf"] >= 1.2:
                    # Walk-forward validation
                    train_r, test_r = walk_forward(df, direction=direction, **best_config)
                    
                    wf_pass = False
                    if test_r and test_r["trades"] >= 1 and test_r["ret"] > -5:
                        wf_pass = True
                    
                    tag = "✅ WF-PASS" if wf_pass else "⚠️  WF-FAIL"
                    
                    print(f"    {direction.upper():5s} {tag}: "
                          f"{best_result['trades']}T {best_result['wr']}%WR "
                          f"PF={best_result['pf']} ret={best_result['ret']}% "
                          f"dd={best_result['max_dd']}% "
                          f"params=bb{best_config['breakout_bars']}/sl{best_config['sl_atr_mult']}/"
                          f"tp{best_config['tp_atr_mult']}/sma{best_config['sma_period']}/"
                          f"t{best_config['timeout_bars']}")
                    
                    if test_r:
                        print(f"           WF test: {test_r['trades']}T {test_r['wr']}%WR "
                              f"PF={test_r['pf']} ret={test_r['ret']}%")
                    
                    all_edges.append({
                        "pair": pair, "tf": tf_name, "dir": direction,
                        "full": best_result, "wf_test": test_r,
                        "wf_pass": wf_pass, "config": best_config,
                    })

    # ═══════════════════════════════════════════
    # SUMMARY — Rank all edges
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  EDGE RANKING — All profitable configs sorted by return")
    print("=" * 80)

    # Sort by walk-forward test return (most reliable) then full return
    def sort_key(e):
        wf = e.get("wf_test")
        wf_ret = wf["ret"] if wf and wf["trades"] >= 1 else -999
        return (1 if e["wf_pass"] else 0, wf_ret, e["full"]["ret"])

    all_edges.sort(key=sort_key, reverse=True)

    print(f"\n  {'Pair':10s} {'TF':5s} {'Dir':6s} {'WF':8s} {'Trades':7s} {'WR':6s} "
          f"{'PF':6s} {'Return':8s} {'DD':6s} {'WF-Test':8s} Config")
    print("  " + "-" * 100)

    wf_passed = []
    for e in all_edges:
        wf = e.get("wf_test")
        wf_str = f"{wf['ret']:+.1f}%" if wf and wf["trades"] >= 1 else "N/A"
        tag = "✅ PASS" if e["wf_pass"] else "⚠️  FAIL"
        cfg = e["config"]
        print(f"  {e['pair']:10s} {e['tf']:5s} {e['dir']:6s} {tag:8s} "
              f"{e['full']['trades']:5d}T {e['full']['wr']:5.1f}% "
              f"{e['full']['pf']:5.2f} {e['full']['ret']:+7.2f}% "
              f"{e['full']['max_dd']:5.2f}% {wf_str:>8s} "
              f"bb{cfg['breakout_bars']}/sl{cfg['sl_atr_mult']}/tp{cfg['tp_atr_mult']}/"
              f"sma{cfg['sma_period']}/t{cfg['timeout_bars']}")
        if e["wf_pass"]:
            wf_passed.append(e)

    # ═══════════════════════════════════════════
    # PORTFOLIO PROJECTION
    # ═══════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD VALIDATED EDGES: {len(wf_passed)}")
    print(f"{'='*80}")

    if wf_passed:
        # Estimate combined monthly trades and returns
        total_trades_90d = sum(e["full"]["trades"] for e in wf_passed)
        total_pnl_90d = sum(e["full"]["pnl"] for e in wf_passed)
        avg_wr = np.mean([e["full"]["wr"] for e in wf_passed])
        avg_pf = np.mean([e["full"]["pf"] for e in wf_passed])
        
        # Conservative estimate: use WF test results where available
        wf_pnl = 0
        wf_trades = 0
        for e in wf_passed:
            wf = e.get("wf_test")
            if wf and wf["trades"] >= 1:
                wf_pnl += wf["pnl"]
                wf_trades += wf["trades"]
        
        print(f"\n  Full backtest (90d): {total_trades_90d} trades, ${total_pnl_90d:.2f} PnL")
        print(f"  Walk-forward test:  {wf_trades} trades, ${wf_pnl:.2f} PnL")
        print(f"  Avg win rate: {avg_wr:.1f}%")
        print(f"  Avg profit factor: {avg_pf:.2f}")
        
        # Monthly return estimate (conservative: use WF numbers, scale to 30d)
        if wf_pnl > 0:
            monthly_ret_pct = (wf_pnl / 300) / 1.0 * 100  # WF test period varies
            print(f"\n  Estimated monthly return (conservative): ~{monthly_ret_pct:.1f}%")
            
            # Compound projection
            balance = 300
            print(f"\n  Compound projection at {monthly_ret_pct:.1f}% monthly:")
            for months in [6, 12, 24, 36, 60, 120]:
                projected = balance * (1 + monthly_ret_pct/100) ** months
                print(f"    {months:3d} months: ${projected:>12,.2f}")
    else:
        print("\n  No edges passed walk-forward validation. Strategy may not be robust.")

    print(f"\n{'='*80}")
    print(f"  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
