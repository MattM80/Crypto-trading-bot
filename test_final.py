#!/usr/bin/env python3
"""
FINAL TEST: Per-pair adaptive ecosystem, 83 days hourly, $300.
Each pair gets its own regime. Strategy follows regime.
This is the kill-or-ship test.
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
    "NEAR": {"grid_pct": 0.01, "tp_pct": 0.015, "levels": 3},
    "UNI": {"grid_pct": 0.015, "tp_pct": 0.015, "levels": 3},
    "AVAX": {"grid_pct": 0.01, "tp_pct": 0.015, "levels": 3},
    "LINK": {"grid_pct": 0.008, "tp_pct": 0.015, "levels": 3},
    "AAVE": {"grid_pct": 0.015, "tp_pct": 0.015, "levels": 3},
    "SOL": {"grid_pct": 0.003, "tp_pct": 0.015, "levels": 3},
    "ETH": {"grid_pct": 0.005, "tp_pct": 0.015, "levels": 3},
    "BTC": {"grid_pct": 0.01, "tp_pct": 0.015, "levels": 3},
    "DOT": {"grid_pct": 0.012, "tp_pct": 0.015, "levels": 3},
    "XLM": {"grid_pct": 0.01, "tp_pct": 0.015, "levels": 3},
    "XRP": {"grid_pct": 0.01, "tp_pct": 0.015, "levels": 3},
    "ADA": {"grid_pct": 0.012, "tp_pct": 0.015, "levels": 3},
    "ATOM": {"grid_pct": 0.008, "tp_pct": 0.015, "levels": 3},
    "DOGE": {"grid_pct": 0.012, "tp_pct": 0.015, "levels": 3},
    "FIL": {"grid_pct": 0.015, "tp_pct": 0.015, "levels": 3},
    "LTC": {"grid_pct": 0.01, "tp_pct": 0.015, "levels": 3},
}

RSI_CONFIGS = {
    "ATOM": {"period": 7, "oversold": 25, "overbought": 75, "hold": 5},
    "AVAX": {"period": 14, "oversold": 25, "overbought": 70, "hold": 8},
    "XRP": {"period": 9, "oversold": 25, "overbought": 75, "hold": 3},
    "NEAR": {"period": 9, "oversold": 25, "overbought": 75, "hold": 8},
    "FIL": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "BTC": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "ETH": {"period": 14, "oversold": 25, "overbought": 75, "hold": 12},
    "SOL": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "LINK": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "ADA": {"period": 14, "oversold": 25, "overbought": 75, "hold": 12},
    "DOT": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "UNI": {"period": 9, "oversold": 25, "overbought": 75, "hold": 8},
    "AAVE": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "DOGE": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "XLM": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
    "LTC": {"period": 14, "oversold": 25, "overbought": 75, "hold": 8},
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


def detect_regime(close, high, low, idx):
    """Per-pair regime at bar idx."""
    if idx < 60:
        return "ranging"
    c = close[max(0, idx-60):idx+1].astype(float)
    h = high[max(0, idx-60):idx+1].astype(float)
    l = low[max(0, idx-60):idx+1].astype(float)
    
    # ATR volatility
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    if len(tr) >= 14:
        atr_recent = float(np.mean(tr[-14:]))
        atr_median = float(np.median(tr))
        if atr_median > 0 and atr_recent > 1.5 * atr_median:
            return "volatile"
    
    # Trend: 20-bar return
    if len(c) >= 21:
        ret = (c[-1] - c[-21]) / c[-21]
        if ret > 0.05: return "trending_up"
        if ret < -0.05: return "trending_down"
    
    return "ranging"


def main():
    print("=" * 80)
    print("  FINAL TEST: Per-Pair Adaptive Ecosystem")
    print("  83 days hourly, $300, each coin independent")
    print("=" * 80)
    
    print("\nDownloading hourly data (83 days)...")
    all_data = {}
    for sym in PAIRS:
        df = download_cc(sym, 2000)
        if len(df) > 100:
            all_data[sym] = df
            print(f"  {sym:6s}: {len(df)} bars, ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    
    min_len = min(len(df) for df in all_data.values())
    days = min_len / 24
    print(f"\nSimulating {min_len} bars (~{days:.0f} days) across {len(all_data)} pairs")
    
    # ═══════════════════════════════════════
    # STATE
    # ═══════════════════════════════════════
    balance = float(BALANCE)
    
    # Grid state per pair: {sym: {"buys": [(price, qty)], "filled": [(price, qty)], "profit": float}}
    grids = {}
    grid_total_profit = 0
    grid_total_roundtrips = 0
    
    # Active positions: {sym: {"entry": float, "qty": float, "bar": int, "strategy": str, "direction": str}}
    active_pos = {}
    active_trades = []
    
    # Per-pair regime tracking
    pair_regimes = {sym: "unknown" for sym in all_data}
    regime_history = defaultdict(list)  # sym -> [(bar, regime)]
    
    equity_curve = []
    peak_balance = balance
    max_dd = 0
    
    # Capital tracking
    capital_in_grid = 0
    capital_in_active = 0
    
    for bar in range(60, min_len):
        # ── DETECT REGIME PER PAIR ──
        regimes = {}
        for sym, df in all_data.items():
            regimes[sym] = detect_regime(df['close'].values, df['high'].values, df['low'].values, bar)
        
        # Track regime transitions
        for sym in all_data:
            old = pair_regimes[sym]
            new = regimes[sym]
            if old != new:
                regime_history[sym].append((bar, f"{old}→{new}"))
                
                # Close positions for this pair on regime change
                if sym in active_pos:
                    pos = active_pos[sym]
                    cur_price = float(all_data[sym]['close'].values[bar])
                    if pos['direction'] == 'long':
                        pnl = (cur_price - pos['entry']) * pos['qty']
                        fees = pos['entry'] * pos['qty'] * MAKER_FEE + cur_price * pos['qty'] * MAKER_FEE
                        balance += pos['entry'] * pos['qty'] + (pnl - fees)
                    else:
                        pnl = (pos['entry'] - cur_price) * pos['qty']
                        fees = pos['entry'] * pos['qty'] * MAKER_FEE + cur_price * pos['qty'] * MAKER_FEE
                        balance += (pnl - fees)
                    active_trades.append({'sym': sym, 'pnl': pnl - fees, 'strategy': pos['strategy'],
                                         'reason': f'regime_{old}→{new}', 'bars': bar - pos['bar']})
                    del active_pos[sym]
                
                # Close grid for this pair if leaving trending_up
                if sym in grids and new != "trending_up":
                    g = grids[sym]
                    cur_price = float(all_data[sym]['close'].values[bar])
                    for bp, qty in g.get("filled", []):
                        pnl = (cur_price - bp) * qty
                        fees = bp * qty * MAKER_FEE + cur_price * qty * MAKER_FEE
                        balance += bp * qty + (pnl - fees)
                        grid_total_profit += (pnl - fees)
                    del grids[sym]
        
        pair_regimes = regimes
        
        # Count regimes
        regime_counts = defaultdict(int)
        for r in regimes.values():
            regime_counts[r] += 1
        
        # ── MANAGE ACTIVE POSITIONS ──
        for sym in list(active_pos.keys()):
            pos = active_pos[sym]
            df = all_data[sym]
            cur_price = float(df['close'].values[bar])
            cur_high = float(df['high'].values[bar])
            cur_low = float(df['low'].values[bar])
            bars_held = bar - pos['bar']
            
            exit_price = None
            reason = None
            
            if pos['strategy'] == 'rsi':
                cfg = RSI_CONFIGS.get(sym, {"period": 14, "oversold": 25, "overbought": 75, "hold": 8})
                rsi_vals = calc_rsi(df['close'].values[:bar+1], cfg['period'])
                cur_rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
                
                if pos['direction'] == 'long':
                    if cur_low <= pos['entry'] * 0.97:
                        exit_price = pos['entry'] * 0.97; reason = "SL"
                    elif cur_rsi >= 50:
                        exit_price = cur_price; reason = "RSI_REVERT"
                    elif bars_held >= cfg['hold']:
                        exit_price = cur_price; reason = "TIMEOUT"
                else:
                    if cur_high >= pos['entry'] * 1.03:
                        exit_price = pos['entry'] * 1.03; reason = "SL"
                    elif cur_rsi <= 50:
                        exit_price = cur_price; reason = "RSI_REVERT"
                    elif bars_held >= cfg['hold']:
                        exit_price = cur_price; reason = "TIMEOUT"
            
            elif pos['strategy'] == 'momentum':
                if pos['direction'] == 'long':
                    if cur_low <= pos.get('sl', pos['entry'] * 0.96):
                        exit_price = pos.get('sl', pos['entry'] * 0.96); reason = "SL"
                    elif cur_high >= pos.get('tp', pos['entry'] * 1.09):
                        exit_price = pos.get('tp', pos['entry'] * 1.09); reason = "TP"
                    elif bars_held >= 48:
                        exit_price = cur_price; reason = "TIMEOUT"
            
            if exit_price:
                if pos['direction'] == 'long':
                    pnl = (exit_price - pos['entry']) * pos['qty']
                    fees = pos['entry'] * pos['qty'] * MAKER_FEE + exit_price * pos['qty'] * MAKER_FEE
                    balance += pos['entry'] * pos['qty'] + (pnl - fees)
                else:
                    pnl = (pos['entry'] - exit_price) * pos['qty']
                    fees = pos['entry'] * pos['qty'] * MAKER_FEE + exit_price * pos['qty'] * MAKER_FEE
                    balance += (pnl - fees)
                active_trades.append({'sym': sym, 'pnl': pnl - fees, 'strategy': pos['strategy'],
                                     'reason': reason, 'bars': bars_held, 'dir': pos['direction']})
                del active_pos[sym]
        
        # ── GRID UPDATES (trending_up pairs) ──
        trending_up_pairs = [s for s, r in regimes.items() if r == "trending_up"]
        
        for sym in trending_up_pairs:
            df = all_data[sym]
            cur_close = float(df['close'].values[bar])
            cur_low = float(df['low'].values[bar])
            cur_high = float(df['high'].values[bar])
            
            cfg = GRID_CONFIGS.get(sym, {"grid_pct": 0.01, "tp_pct": 0.015, "levels": 3})
            
            if sym not in grids:
                # Initialize grid
                n_grid_pairs = max(len(trending_up_pairs), 1)
                alloc = min(balance * 0.15, balance / n_grid_pairs * 0.4)
                if alloc < 3: continue
                
                grids[sym] = {"buys": [], "filled": [], "profit": 0, "alloc": alloc}
                mid = cur_close
                for lvl in range(1, cfg['levels'] + 1):
                    bp = mid * (1 - cfg['grid_pct'] * lvl)
                    qty = (alloc / cfg['levels']) / bp
                    grids[sym]["buys"].append((bp, qty))
            
            g = grids[sym]
            
            # Sell fills
            new_filled = []
            for bp, qty in g["filled"]:
                sell_target = bp * (1 + cfg['tp_pct'])
                if cur_high >= sell_target:
                    fee = sell_target * qty * MAKER_FEE
                    profit = sell_target * qty - fee - bp * qty * (1 + MAKER_FEE)
                    g["profit"] += profit
                    grid_total_profit += profit
                    grid_total_roundtrips += 1
                    balance += sell_target * qty - fee
                    # Re-place buy
                    g["buys"].append((bp, qty))
                else:
                    new_filled.append((bp, qty))
            g["filled"] = new_filled
            
            # Buy fills
            new_buys = []
            for bp, qty in g["buys"]:
                if cur_low <= bp:
                    fee = bp * qty * MAKER_FEE
                    if balance >= bp * qty + fee:
                        balance -= bp * qty + fee
                        g["filled"].append((bp, qty))
                    else:
                        new_buys.append((bp, qty))
                else:
                    new_buys.append((bp, qty))
            g["buys"] = new_buys
        
        # ── RSI SIGNALS (trending_down pairs) ──
        trending_down_pairs = [s for s, r in regimes.items() if r == "trending_down"]
        
        if len(active_pos) < MAX_ACTIVE:
            for sym in trending_down_pairs:
                if sym in active_pos or len(active_pos) >= MAX_ACTIVE:
                    continue
                df = all_data[sym]
                cfg = RSI_CONFIGS.get(sym)
                if not cfg: continue
                
                rsi_vals = calc_rsi(df['close'].values[:bar+1], cfg['period'])
                cur_rsi = rsi_vals[-1] if len(rsi_vals) > 0 and not np.isnan(rsi_vals[-1]) else 50
                cur_price = float(df['close'].values[bar])
                
                if cur_rsi <= cfg['oversold']:
                    entry = cur_price * (1 + SLIPPAGE)
                    risk = entry * 0.03
                    qty = (balance * RISK_PCT) / risk
                    cost = qty * entry
                    if cost <= balance * 0.25 and cost > 2:
                        balance -= cost
                        active_pos[sym] = {'entry': entry, 'qty': qty, 'bar': bar,
                                          'strategy': 'rsi', 'direction': 'long'}
        
        # ── MOMENTUM SIGNALS (volatile pairs) ──
        volatile_pairs = [s for s, r in regimes.items() if r == "volatile"]
        
        if len(active_pos) < MAX_ACTIVE:
            for sym in volatile_pairs:
                if sym in active_pos or len(active_pos) >= MAX_ACTIVE:
                    continue
                df = all_data[sym]
                close = df['close'].values[:bar+1].astype(float)
                high = df['high'].values[:bar+1].astype(float)
                low = df['low'].values[:bar+1].astype(float)
                
                if len(close) < 35: continue
                
                sma30 = float(np.mean(close[-30:]))
                if close[-1] <= sma30: continue
                
                prev_high = float(np.max(high[-16:-1]))
                if close[-1] <= prev_high: continue
                
                # ATR
                prev_c = np.roll(close, 1); prev_c[0] = close[0]
                tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
                atr = float(np.mean(tr[-14:]))
                if atr <= 0: continue
                
                entry = close[-1] * (1 + SLIPPAGE)
                sl = max(float(np.min(low[-16:-1])) * 0.998, entry - 2 * atr)
                tp = entry + 3 * atr
                risk = entry - sl
                if risk <= 0 or (tp - entry) / risk < 1.5: continue
                
                qty = (balance * RISK_PCT) / risk
                cost = qty * entry
                if cost <= balance * 0.25 and cost > 2:
                    balance -= cost
                    active_pos[sym] = {'entry': entry, 'qty': qty, 'bar': bar,
                                      'strategy': 'momentum', 'direction': 'long',
                                      'sl': sl, 'tp': tp}
        
        # ── EQUITY TRACKING ──
        unrealized = 0
        for sym, pos in active_pos.items():
            cur = float(all_data[sym]['close'].values[bar])
            if pos['direction'] == 'long':
                unrealized += (cur - pos['entry']) * pos['qty']
            else:
                unrealized += (pos['entry'] - cur) * pos['qty']
        for sym, g in grids.items():
            cur = float(all_data[sym]['close'].values[bar])
            for bp, qty in g["filled"]:
                unrealized += (cur - bp) * qty
        
        equity = balance + unrealized
        equity_curve.append(equity)
        peak_balance = max(peak_balance, equity)
        dd = (peak_balance - equity) / peak_balance if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)
    
    # ── CLOSE EVERYTHING AT END ──
    for sym, pos in list(active_pos.items()):
        cur = float(all_data[sym]['close'].values[-1])
        if pos['direction'] == 'long':
            pnl = (cur - pos['entry']) * pos['qty']
            fees = pos['entry'] * pos['qty'] * MAKER_FEE + cur * pos['qty'] * MAKER_FEE
            balance += pos['entry'] * pos['qty'] + (pnl - fees)
        else:
            pnl = (pos['entry'] - cur) * pos['qty']
            fees = pos['entry'] * pos['qty'] * MAKER_FEE + cur * pos['qty'] * MAKER_FEE
            balance += (pnl - fees)
        active_trades.append({'sym': sym, 'pnl': pnl - fees, 'strategy': pos['strategy'],
                             'reason': 'END', 'bars': min_len - pos['bar'], 'dir': pos['direction']})
    
    for sym, g in list(grids.items()):
        cur = float(all_data[sym]['close'].values[-1])
        for bp, qty in g["filled"]:
            pnl = (cur - bp) * qty
            fees = bp * qty * MAKER_FEE + cur * qty * MAKER_FEE
            balance += bp * qty + (pnl - fees)
            grid_total_profit += (pnl - fees)
    
    # ═══════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════
    
    total_pnl = balance - BALANCE
    
    print(f"\n{'='*80}")
    print(f"  RESULTS — {min_len} bars (~{days:.0f} days), {len(all_data)} pairs")
    print(f"{'='*80}")
    
    # Grid
    print(f"\n  GRID (trending_up pairs only)")
    print(f"    Round-trips: {grid_total_roundtrips}")
    print(f"    Realized: ${grid_total_profit:+.2f}")
    
    # Active strategies
    wins = [t for t in active_trades if t['pnl'] > 0]
    losses = [t for t in active_trades if t['pnl'] <= 0]
    active_pnl = sum(t['pnl'] for t in active_trades)
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    
    print(f"\n  ACTIVE STRATEGIES (RSI + Momentum)")
    print(f"    Trades: {len(active_trades)}")
    print(f"    Wins: {len(wins)} / Losses: {len(losses)}")
    print(f"    Win rate: {len(wins)/len(active_trades)*100:.1f}%" if active_trades else "    Win rate: N/A")
    print(f"    PF: {gp/gl:.2f}" if gl > 0.001 else "    PF: N/A")
    print(f"    PnL: ${active_pnl:+.2f}")
    
    # By strategy
    for strat in ['rsi', 'momentum']:
        st = [t for t in active_trades if t['strategy'] == strat]
        if st:
            sw = sum(1 for t in st if t['pnl'] > 0)
            sp = sum(t['pnl'] for t in st)
            print(f"    {strat.upper():10s}: {len(st)}T {sw}W/{len(st)-sw}L ${sp:+.2f}")
    
    # By exit reason
    reasons = defaultdict(lambda: [0, 0])  # reason -> [count, total_pnl]
    for t in active_trades:
        r = t.get('reason', '?')
        reasons[r][0] += 1
        reasons[r][1] += t['pnl']
    print(f"    Exit reasons:")
    for r, (cnt, pnl) in sorted(reasons.items(), key=lambda x: -x[1][0]):
        print(f"      {r:20s}: {cnt:3d}T ${pnl:+.2f}")
    
    # Combined
    color = "\033[92m" if total_pnl > 0 else "\033[91m"
    reset = "\033[0m"
    print(f"\n  COMBINED")
    print(f"    Starting:  ${BALANCE:.2f}")
    print(f"    Grid:      ${grid_total_profit:+.2f}")
    print(f"    Active:    ${active_pnl:+.2f}")
    print(f"    {color}TOTAL:     ${total_pnl:+.2f} ({total_pnl/BALANCE*100:+.2f}%){reset}")
    print(f"    {color}FINAL:     ${balance:.2f}{reset}")
    print(f"    Max DD:    {max_dd*100:.1f}%")
    
    # Regime distribution
    total_bars = sum(len(v) for v in regime_history.values())
    print(f"\n  REGIME TRANSITIONS: {sum(len(v) for v in regime_history.values())} total across all pairs")
    
    # Compound projection
    if total_pnl > 0:
        monthly = total_pnl / days * 30
        monthly_pct = monthly / BALANCE * 100
        print(f"\n  PROJECTION (if sustained)")
        print(f"    Monthly: ~${monthly:.2f} (~{monthly_pct:.1f}%)")
        for m in [6, 12, 24, 36, 60]:
            proj = BALANCE * (1 + monthly_pct/100) ** m
            print(f"    {m:>3d}mo: ${proj:>12,.2f}")
    
    # Trade log (last 30)
    print(f"\n  TRADE LOG (showing all {len(active_trades)})")
    print(f"  {'Sym':6s} {'Dir':6s} {'Strat':10s} {'PnL':>8s} {'Reason':>15s} {'Bars':>5s}")
    for t in active_trades:
        c = "\033[92m" if t['pnl'] > 0 else "\033[91m"
        print(f"  {t['sym']:6s} {t.get('dir','?'):6s} {t['strategy']:10s} "
              f"{c}${t['pnl']:>+7.2f}\033[0m {t['reason']:>15s} {t['bars']:>5d}")
    
    # ═══════════════════════════════════════
    # ALSO TEST: What if we just did grid-only vs RSI-only vs adaptive?
    # ═══════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMPARISON: Adaptive vs Pure Strategies")
    print(f"{'='*80}")
    print(f"  Adaptive (per-pair regime): ${total_pnl:+.2f} ({total_pnl/BALANCE*100:+.2f}%)")
    print(f"    Grid component: ${grid_total_profit:+.2f}")
    print(f"    RSI component:  ${sum(t['pnl'] for t in active_trades if t['strategy']=='rsi'):+.2f}")
    print(f"    Mom component:  ${sum(t['pnl'] for t in active_trades if t['strategy']=='momentum'):+.2f}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
