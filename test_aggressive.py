#!/usr/bin/env python3
"""
AGGRESSIVE: Grid always on. RSI on ALL pairs regardless of regime. Shorts too.
Stack bread. 83 days, 16 pairs, $300.
"""
import requests, time as _time, numpy as np, pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

CC_BASE = "https://min-api.cryptocompare.com/data/v2"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005
BALANCE = 300
MAX_ACTIVE = 5
RISK_PCT = 0.03

PAIRS = ["BTC", "ETH", "SOL", "LINK", "AVAX", "DOT", "ADA", "XRP",
         "DOGE", "UNI", "NEAR", "ATOM", "AAVE", "XLM", "FIL", "LTC"]

GRID_CONFIGS = {
    "NEAR": 0.01, "UNI": 0.015, "AVAX": 0.01, "LINK": 0.008,
    "AAVE": 0.015, "SOL": 0.003, "ETH": 0.005, "BTC": 0.01,
    "DOT": 0.012, "XLM": 0.01, "XRP": 0.01, "ADA": 0.012,
    "ATOM": 0.008, "DOGE": 0.012, "FIL": 0.015, "LTC": 0.01,
}

def download_cc(symbol, limit=2000):
    try:
        resp = requests.get(f"{CC_BASE}/histohour",
            params={"fsym": symbol, "tsym": "USD", "limit": limit}, timeout=30)
        data = resp.json().get("Data", {}).get("Data", [])
        rows = [{'time': d['time'], 'open': d['open'], 'high': d['high'],
                 'low': d['low'], 'close': d['close'], 'volume': d.get('volumeto', 0)}
                for d in data if d.get('close', 0) > 0]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()


def calc_rsi(close_arr, period):
    s = pd.Series(close_arr)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def main():
    print("=" * 80)
    print("  AGGRESSIVE MODE: Grid + RSI everywhere, shorts enabled")
    print("=" * 80)

    print("\nDownloading 83 days hourly...")
    all_data = {}
    for sym in PAIRS:
        df = download_cc(sym, 2000)
        if len(df) > 100:
            all_data[sym] = df
            chg = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            print(f"  {sym:6s}: {len(df)} bars, ${df['close'].iloc[0]:.2f}→${df['close'].iloc[-1]:.2f} ({chg:+.1f}%)")

    min_len = min(len(df) for df in all_data.values())
    days = min_len / 24
    print(f"\n{min_len} bars (~{days:.0f} days), {len(all_data)} pairs")

    # ═══════════════════════════════════════
    # STATE
    # ═══════════════════════════════════════
    balance = float(BALANCE)
    grid_balance = balance * 0.40  # 40% to grids
    active_balance = balance * 0.60  # 60% to active trades

    # Grids: always on for all pairs
    grids = {}  # sym -> {"buys": [(price, qty)], "filled": [(price, qty)], "profit": 0}
    grid_profit = 0
    grid_roundtrips = 0

    # Active: RSI long + short on all pairs
    active_pos = {}  # sym -> position dict
    active_trades = []

    equity_curve = []
    peak = balance
    max_dd = 0

    # Initialize grids for ALL pairs
    n_grid_pairs = len(all_data)
    grid_alloc_per = grid_balance / n_grid_pairs

    for sym in all_data:
        df = all_data[sym]
        first_price = float(df['close'].values[60])
        grid_pct = GRID_CONFIGS.get(sym, 0.01)
        grids[sym] = {"buys": [], "filled": [], "profit": 0}
        for lvl in range(1, 4):  # 3 levels
            bp = first_price * (1 - grid_pct * lvl)
            qty = (grid_alloc_per / 3) / bp
            grids[sym]["buys"].append((bp, qty))

    for bar in range(61, min_len):
        # ── GRID UPDATES (ALL pairs, always) ──
        for sym in all_data:
            df = all_data[sym]
            cur_close = float(df['close'].values[bar])
            cur_low = float(df['low'].values[bar])
            cur_high = float(df['high'].values[bar])
            prev_close = float(df['close'].values[bar-1])
            grid_pct = GRID_CONFIGS.get(sym, 0.01)
            g = grids[sym]

            # Sell fills
            new_filled = []
            for bp, qty in g["filled"]:
                sell_target = bp * 1.015  # 1.5% TP
                if cur_high >= sell_target:
                    fee = sell_target * qty * MAKER_FEE
                    profit = sell_target * qty - fee - bp * qty * (1 + MAKER_FEE)
                    g["profit"] += profit
                    grid_profit += profit
                    grid_roundtrips += 1
                    grid_balance += sell_target * qty - fee
                    # Re-place buy at same level
                    g["buys"].append((bp, qty))
                else:
                    new_filled.append((bp, qty))
            g["filled"] = new_filled

            # Buy fills
            new_buys = []
            for bp, qty in g["buys"]:
                if cur_low <= bp:
                    cost = bp * qty * (1 + MAKER_FEE)
                    if grid_balance >= cost:
                        grid_balance -= cost
                        g["filled"].append((bp, qty))
                    else:
                        new_buys.append((bp, qty))
                else:
                    new_buys.append((bp, qty))
            g["buys"] = new_buys

            # Re-center grid if price moved far from grid levels
            if len(g["buys"]) == 0 and len(g["filled"]) == 0:
                for lvl in range(1, 4):
                    bp = cur_close * (1 - grid_pct * lvl)
                    qty = (grid_alloc_per / 3) / bp
                    g["buys"].append((bp, qty))

        # ── MANAGE ACTIVE RSI POSITIONS ──
        for sym in list(active_pos.keys()):
            pos = active_pos[sym]
            df = all_data[sym]
            cur_price = float(df['close'].values[bar])
            cur_high = float(df['high'].values[bar])
            cur_low = float(df['low'].values[bar])
            bars_held = bar - pos['bar']

            rsi_vals = calc_rsi(df['close'].values[:bar+1], 7)
            cur_rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50

            exit_price = None
            reason = None

            if pos['direction'] == 'long':
                if cur_low <= pos['entry'] * 0.97:
                    exit_price = pos['entry'] * 0.97; reason = "SL"
                elif cur_rsi >= 50:
                    exit_price = cur_price; reason = "RSI_REVERT"
                elif bars_held >= 8:
                    exit_price = cur_price; reason = "TIMEOUT"
            else:  # short
                if cur_high >= pos['entry'] * 1.03:
                    exit_price = pos['entry'] * 1.03; reason = "SL"
                elif cur_rsi <= 50:
                    exit_price = cur_price; reason = "RSI_REVERT"
                elif bars_held >= 8:
                    exit_price = cur_price; reason = "TIMEOUT"

            if exit_price:
                if pos['direction'] == 'long':
                    pnl = (exit_price - pos['entry']) * pos['qty']
                    fees = pos['entry'] * pos['qty'] * MAKER_FEE + exit_price * pos['qty'] * MAKER_FEE
                    active_balance += pos['entry'] * pos['qty'] + (pnl - fees)
                else:
                    pnl = (pos['entry'] - exit_price) * pos['qty']
                    fees = pos['entry'] * pos['qty'] * MAKER_FEE + exit_price * pos['qty'] * MAKER_FEE
                    active_balance += (pnl - fees)
                active_trades.append({'sym': sym, 'pnl': pnl - fees, 'dir': pos['direction'],
                                     'reason': reason, 'bars': bars_held})
                del active_pos[sym]

        # ── NEW RSI SIGNALS (ALL pairs, both directions) ──
        if len(active_pos) < MAX_ACTIVE:
            # Score all pairs by RSI extremeness
            candidates = []
            for sym in all_data:
                if sym in active_pos: continue
                df = all_data[sym]
                rsi_vals = calc_rsi(df['close'].values[:bar+1], 7)
                cur_rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
                cur_price = float(df['close'].values[bar])

                if cur_rsi <= 25:
                    # Oversold → buy
                    extremeness = 25 - cur_rsi  # Higher = more oversold
                    candidates.append((sym, 'long', cur_price, cur_rsi, extremeness))
                elif cur_rsi >= 75:
                    # Overbought → short
                    extremeness = cur_rsi - 75
                    candidates.append((sym, 'short', cur_price, cur_rsi, extremeness))

            # Sort by extremeness (most extreme first)
            candidates.sort(key=lambda x: x[4], reverse=True)

            for sym, direction, cur_price, cur_rsi, _ in candidates:
                if len(active_pos) >= MAX_ACTIVE: break
                if sym in active_pos: continue

                entry = cur_price * (1 + SLIPPAGE) if direction == 'long' else cur_price * (1 - SLIPPAGE)
                risk = entry * 0.03
                qty = (active_balance * RISK_PCT) / risk

                if direction == 'long':
                    cost = qty * entry
                    if cost <= active_balance * 0.20 and cost > 1:
                        active_balance -= cost
                        active_pos[sym] = {'entry': entry, 'qty': qty, 'bar': bar, 'direction': 'long'}
                else:
                    # Short: margin trade
                    if qty * entry > 1 and active_balance * RISK_PCT < active_balance * 0.20:
                        active_pos[sym] = {'entry': entry, 'qty': qty, 'bar': bar, 'direction': 'short'}

        # ── EQUITY ──
        unrealized = 0
        for sym, pos in active_pos.items():
            cur = float(all_data[sym]['close'].values[bar])
            if pos['direction'] == 'long':
                unrealized += (cur - pos['entry']) * pos['qty']
            else:
                unrealized += (pos['entry'] - cur) * pos['qty']
        grid_unrealized = 0
        for sym, g in grids.items():
            cur = float(all_data[sym]['close'].values[bar])
            for bp, qty in g["filled"]:
                grid_unrealized += (cur - bp) * qty

        equity = active_balance + grid_balance + unrealized + grid_unrealized
        equity_curve.append(equity)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # ── CLOSE EVERYTHING ──
    for sym, pos in list(active_pos.items()):
        cur = float(all_data[sym]['close'].values[-1])
        if pos['direction'] == 'long':
            pnl = (cur - pos['entry']) * pos['qty']
            fees = pos['entry'] * pos['qty'] * MAKER_FEE + cur * pos['qty'] * MAKER_FEE
            active_balance += pos['entry'] * pos['qty'] + (pnl - fees)
        else:
            pnl = (pos['entry'] - cur) * pos['qty']
            fees = pos['entry'] * pos['qty'] * MAKER_FEE + cur * pos['qty'] * MAKER_FEE
            active_balance += (pnl - fees)
        active_trades.append({'sym': sym, 'pnl': pnl - fees, 'dir': pos['direction'],
                             'reason': 'END', 'bars': min_len - pos['bar']})

    for sym, g in grids.items():
        cur = float(all_data[sym]['close'].values[-1])
        for bp, qty in g["filled"]:
            fee = cur * qty * MAKER_FEE
            grid_balance += cur * qty - fee
            grid_profit += (cur - bp) * qty - bp * qty * MAKER_FEE - fee

    # ═══════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════
    final = active_balance + grid_balance
    total_pnl = final - BALANCE

    wins = [t for t in active_trades if t['pnl'] > 0]
    losses = [t for t in active_trades if t['pnl'] <= 0]
    active_pnl = sum(t['pnl'] for t in active_trades)
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001

    color = "\033[92m" if total_pnl > 0 else "\033[91m"
    R = "\033[0m"

    print(f"\n{'='*80}")
    print(f"  RESULTS — {min_len} bars (~{days:.0f} days), {len(all_data)} pairs")
    print(f"{'='*80}")

    print(f"\n  GRID (always on, all {len(grids)} pairs)")
    print(f"    Round-trips: {grid_roundtrips}")
    print(f"    Realized: ${grid_profit:+.2f}")

    print(f"\n  RSI (all pairs, longs + shorts)")
    print(f"    Trades: {len(active_trades)}")
    print(f"    Wins: {len(wins)} / Losses: {len(losses)}")
    if active_trades:
        print(f"    Win rate: {len(wins)/len(active_trades)*100:.1f}%")
        print(f"    PF: {gp/gl:.2f}")
    print(f"    PnL: ${active_pnl:+.2f}")

    # By direction
    longs = [t for t in active_trades if t['dir'] == 'long']
    shorts = [t for t in active_trades if t['dir'] == 'short']
    if longs:
        lw = sum(1 for t in longs if t['pnl'] > 0)
        print(f"    Longs:  {len(longs)}T {lw}W/{len(longs)-lw}L ${sum(t['pnl'] for t in longs):+.2f}")
    if shorts:
        sw = sum(1 for t in shorts if t['pnl'] > 0)
        print(f"    Shorts: {len(shorts)}T {sw}W/{len(shorts)-sw}L ${sum(t['pnl'] for t in shorts):+.2f}")

    # By reason
    reasons = defaultdict(lambda: [0, 0])
    for t in active_trades:
        reasons[t['reason']][0] += 1
        reasons[t['reason']][1] += t['pnl']
    print(f"    Exits:")
    for r, (cnt, pnl) in sorted(reasons.items(), key=lambda x: -x[1][0]):
        print(f"      {r:15s}: {cnt:3d}T ${pnl:+.2f}")

    print(f"\n  {color}COMBINED{R}")
    print(f"    Start:   ${BALANCE:.2f}")
    print(f"    Grid:    ${grid_profit:+.2f}")
    print(f"    RSI:     ${active_pnl:+.2f}")
    print(f"    {color}TOTAL:   ${total_pnl:+.2f} ({total_pnl/BALANCE*100:+.2f}%){R}")
    print(f"    {color}FINAL:   ${final:.2f}{R}")
    print(f"    Max DD:  {max_dd*100:.1f}%")

    if total_pnl > 0:
        monthly = total_pnl / days * 30
        mpct = monthly / BALANCE * 100
        print(f"\n  Monthly: ~${monthly:.2f} (~{mpct:.1f}%)")
        print(f"  Compound:")
        for m in [6, 12, 24, 36, 60, 120]:
            print(f"    {m:>3d}mo: ${BALANCE * (1 + mpct/100)**m:>12,.2f}")

    # Trade log
    if active_trades:
        print(f"\n  RSI TRADE LOG ({len(active_trades)} trades)")
        print(f"  {'Sym':6s} {'Dir':6s} {'PnL':>8s} {'Reason':>15s} {'Bars':>5s}")
        for t in active_trades:
            c = "\033[92m" if t['pnl'] > 0 else "\033[91m"
            print(f"  {t['sym']:6s} {t['dir']:6s} {c}${t['pnl']:>+7.2f}{R} {t['reason']:>15s} {t['bars']:>5d}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
