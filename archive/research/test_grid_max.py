#!/usr/bin/env python3
"""
ALL-IN GRID: Full $300 across max pairs. Find the optimal config. Build the money machine.
"""
import requests, time as _time, numpy as np, pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
BALANCE = 300

ALL_PAIRS = [
    "XBTUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AVAXUSD",
    "DOTUSD", "ADAUSD", "XRPUSD", "DOGEUSD",
    "UNIUSD", "NEARUSD", "ATOMUSD", "AAVEUSD", "XLMUSD", "FILUSD", "LTCUSD",
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
        except: break
        candles = None
        for key, val in result.items():
            if isinstance(val, list): candles = val; break
        if not candles: break
        for c in candles:
            ts = int(c[0])
            if ts > start_ts and ts <= end_ts:
                all_candles.append({'time': ts, 'open': float(c[1]), 'high': float(c[2]),
                    'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[6])})
        last_ts = int(candles[-1][0])
        if last_ts <= since: break
        since = last_ts
        _time.sleep(1.0)
    df = pd.DataFrame(all_candles)
    if not df.empty:
        df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
    return df


def sim_grid(df, allocation, grid_pct, tp_pct, levels, max_inventory_pct=0.80):
    """Simulate grid trading. Returns detailed results."""
    if len(df) < 10:
        return None
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    cash = allocation
    inventory = []  # (buy_price, qty)
    inventory_cost = 0
    roundtrips = 0
    total_profit = 0
    total_fees = 0
    max_inventory_val = allocation * max_inventory_pct
    daily_profits = defaultdict(float)
    
    for i in range(1, len(df)):
        mid = close[i-1]
        cur_low = low[i]
        cur_high = high[i]
        
        # Sell fills
        new_inv = []
        for bp, qty in inventory:
            sell_target = bp * (1 + tp_pct)
            if cur_high >= sell_target:
                fee = sell_target * qty * MAKER_FEE
                revenue = sell_target * qty - fee
                profit = revenue - bp * qty * (1 + MAKER_FEE)
                total_profit += profit
                total_fees += fee
                cash += revenue
                inventory_cost -= bp * qty
                roundtrips += 1
                day = i // 24
                daily_profits[day] += profit
            else:
                new_inv.append((bp, qty))
        inventory = new_inv
        
        # Buy fills
        for level in range(1, levels + 1):
            buy_price = mid * (1 - grid_pct * level)
            if cur_low <= buy_price and inventory_cost < max_inventory_val:
                alloc = min(cash * 0.25, (max_inventory_val - inventory_cost) / levels)
                if alloc < 1:
                    continue
                fee = alloc * MAKER_FEE
                qty = (alloc - fee) / buy_price
                if cash >= alloc:
                    cash -= alloc
                    inventory.append((buy_price, qty))
                    inventory_cost += buy_price * qty
                    total_fees += fee
    
    # Value remaining inventory
    last_price = close[-1]
    unrealized = sum((last_price - bp) * qty for bp, qty in inventory)
    inv_value = sum(bp * qty for bp, qty in inventory)
    final = cash + inv_value + unrealized
    
    days = len(df) / 24
    daily_avg = total_profit / days if days > 0 else 0
    monthly_proj = daily_avg * 30
    monthly_pct = (monthly_proj / allocation * 100) if allocation > 0 else 0
    
    return {
        'roundtrips': roundtrips,
        'realized': round(total_profit, 2),
        'unrealized': round(unrealized, 2),
        'fees': round(total_fees, 2),
        'final': round(final, 2),
        'ret': round((final - allocation) / allocation * 100, 2),
        'daily_avg': round(daily_avg, 4),
        'monthly_proj': round(monthly_proj, 2),
        'monthly_pct': round(monthly_pct, 2),
        'inventory_items': len(inventory),
        'inventory_value': round(inv_value, 2),
        'profit_per_rt': round(total_profit / roundtrips, 4) if roundtrips > 0 else 0,
    }


def main():
    print("=" * 80)
    print("  ALL-IN GRID OPTIMIZATION — $300, every pair, find the money")
    print("=" * 80)
    
    # Download 1h data for all pairs
    print("\nDownloading 30 days of 1h data...")
    data = {}
    for pair in ALL_PAIRS:
        df = download(pair, 60, 30)
        if len(df) > 50:
            data[pair] = df
            volatility = (df['high'].max() - df['low'].min()) / df['close'].mean() * 100
            avg_range = ((df['high'] - df['low']) / df['close']).mean() * 100
            print(f"  {pair:10s}: {len(df)} bars, ${df['close'].iloc[-1]:>10.4f}, "
                  f"avg bar range: {avg_range:.2f}%, total range: {volatility:.1f}%")
    
    # ═══════════════════════════════════════
    # FIND BEST GRID CONFIG PER PAIR
    # ═══════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  GRID SWEEP — Finding optimal params per pair ($30 per pair)")
    print(f"{'='*80}")
    
    best_per_pair = {}
    
    for pair in sorted(data.keys()):
        df = data[pair]
        best = None
        best_monthly = -999
        
        for grid_pct in [0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02]:
            for tp_pct in [0.003, 0.005, 0.008, 0.01, 0.015]:
                for levels in [3, 5, 7]:
                    r = sim_grid(df, 30, grid_pct, tp_pct, levels)
                    if r and r['roundtrips'] >= 5 and r['monthly_pct'] > best_monthly:
                        best_monthly = r['monthly_pct']
                        best = {**r, 'grid_pct': grid_pct, 'tp_pct': tp_pct, 'levels': levels}
        
        if best and best['monthly_pct'] > 0:
            best_per_pair[pair] = best
            color = "\033[92m" if best['ret'] > 0 else "\033[91m"
            print(f"  {pair:10s}: grid={best['grid_pct']*100:.1f}% tp={best['tp_pct']*100:.1f}% "
                  f"lvl={best['levels']} | {best['roundtrips']}RT "
                  f"{color}ret={best['ret']:+.2f}% monthly~{best['monthly_pct']:.1f}%\033[0m "
                  f"${best['realized']:+.2f} realized, ${best['profit_per_rt']:.4f}/RT")
        else:
            print(f"  {pair:10s}: No profitable grid config found")
    
    # ═══════════════════════════════════════
    # RANK AND SELECT TOP PAIRS
    # ═══════════════════════════════════════
    ranked = sorted(best_per_pair.items(), key=lambda x: x[1]['monthly_pct'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"  RANKED BY MONTHLY RETURN")
    print(f"{'='*80}")
    for pair, r in ranked:
        print(f"  {pair:10s}: ~{r['monthly_pct']:.1f}%/mo, {r['roundtrips']}RT, "
              f"${r['realized']:+.2f}, grid={r['grid_pct']*100:.1f}%/tp={r['tp_pct']*100:.1f}%")
    
    # ═══════════════════════════════════════
    # PORTFOLIO SIMULATION: TOP N PAIRS, FULL $300
    # ═══════════════════════════════════════
    for n_pairs in [4, 6, 8, 10, len(ranked)]:
        if n_pairs > len(ranked):
            continue
        
        selected = ranked[:n_pairs]
        alloc_per = BALANCE / n_pairs
        
        total_realized = 0
        total_unrealized = 0
        total_roundtrips = 0
        total_fees = 0
        
        for pair, orig_result in selected:
            cfg = orig_result
            r = sim_grid(data[pair], alloc_per, cfg['grid_pct'], cfg['tp_pct'], cfg['levels'])
            if r:
                total_realized += r['realized']
                total_unrealized += r['unrealized']
                total_roundtrips += r['roundtrips']
                total_fees += r['fees']
        
        total_pnl = total_realized + total_unrealized
        monthly = total_realized / (len(data[ranked[0][0]]) / 24) * 30
        monthly_pct = monthly / BALANCE * 100
        
        color = "\033[92m" if total_pnl > 0 else "\033[91m"
        print(f"\n  TOP {n_pairs} PAIRS × ${alloc_per:.0f} each = ${BALANCE:.0f}")
        print(f"    {total_roundtrips} round-trips | "
              f"{color}Realized: ${total_realized:+.2f} | Unrealized: ${total_unrealized:+.2f} | "
              f"Total: ${total_pnl:+.2f} ({total_pnl/BALANCE*100:+.2f}%)\033[0m | "
              f"Fees: ${total_fees:.2f}")
        print(f"    Monthly projection: ~${monthly:.2f}/mo (~{monthly_pct:.1f}%/mo)")
        
        if monthly_pct > 0:
            print(f"    Compound projection:")
            bal = BALANCE
            for m in [3, 6, 12, 24, 36, 60, 120]:
                bal = BALANCE * (1 + monthly_pct/100) ** m
                print(f"      {m:>3d}mo: ${bal:>10,.2f}")
    
    # ═══════════════════════════════════════
    # ALSO TEST 15m TIMEFRAME (more oscillations = more roundtrips?)
    # ═══════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  15-MINUTE TIMEFRAME TEST (7 days, more granular)")
    print(f"{'='*80}")
    
    # Pick top 6 pairs and test on 15m
    top6 = [p for p, _ in ranked[:6]]
    data_15m = {}
    for pair in top6:
        df = download(pair, 15, 7)
        if len(df) > 50:
            data_15m[pair] = df
            print(f"  {pair}: {len(df)} bars")
    
    for pair in sorted(data_15m.keys()):
        df = data_15m[pair]
        best = None
        best_monthly = -999
        for grid_pct in [0.002, 0.003, 0.005, 0.008]:
            for tp_pct in [0.002, 0.003, 0.005, 0.008]:
                for levels in [3, 5, 7]:
                    r = sim_grid(df, 50, grid_pct, tp_pct, levels)
                    if r and r['roundtrips'] >= 5 and r['monthly_pct'] > best_monthly:
                        best_monthly = r['monthly_pct']
                        best = {**r, 'grid_pct': grid_pct, 'tp_pct': tp_pct, 'levels': levels}
        
        if best:
            color = "\033[92m" if best['ret'] > 0 else "\033[91m"
            print(f"  {pair:10s} 15m: grid={best['grid_pct']*100:.1f}% tp={best['tp_pct']*100:.1f}% "
                  f"lvl={best['levels']} | {best['roundtrips']}RT "
                  f"{color}monthly~{best['monthly_pct']:.1f}%\033[0m "
                  f"${best['realized']:+.2f}")
    
    print(f"\n{'='*80}")
    print(f"  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
