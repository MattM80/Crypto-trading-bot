#!/usr/bin/env python3
"""
Parameter sweep: test different SL/TP ratios, confidence levels, and timeframes
to find a genuine edge in the honest backtester.
"""
import sys
import os
import itertools
from pathlib import Path

# Suppress all logging for speed
os.environ["LOGURU_LEVEL"] = "ERROR"

import pandas as pd
import logging
logging.disable(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Suppress loguru
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from backtest_live import RealisticBacktester, print_report
from strategies import create_strategy

# Also suppress loguru in imported modules
import backtest_live
import risk_manager
backtest_live.logger.remove()
risk_manager.logger.remove()

CACHE_DIR = PROJECT_ROOT / "data" / "ohlcv_cache"

# Use the biggest cached files we have
PAIR_FILES = {
    "ETHUSD": "ETHUSD_15m_1770688800_1773280800.parquet",
    "XLMUSD": "XLMUSD_15m_1770688800_1773280800.parquet",
    "SOLUSD": "SOLUSD_15m_1770688800_1773280800.parquet",
}

# Load data once
DATA = {}
for sym, fname in PAIR_FILES.items():
    p = CACHE_DIR / fname
    if p.exists():
        df = pd.read_parquet(p)
        if len(df) >= 200:
            DATA[sym] = df
            print(f"  Loaded {sym}: {len(df)} bars")

if not DATA:
    print("No data loaded!")
    sys.exit(1)

# Parameter grid
PARAMS = {
    "atr_sl_mult": [1.5, 2.0, 2.5],
    "atr_tp_mult": [2.0, 3.0, 4.0, 5.0],
    "min_confidence": [0.40, 0.50, 0.60, 0.70],
}

# Generate all combinations
keys = list(PARAMS.keys())
combos = list(itertools.product(*[PARAMS[k] for k in keys]))
print(f"\nTesting {len(combos)} parameter combinations across {len(DATA)} pairs...\n")

results = []
for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    sl = params["atr_sl_mult"]
    tp = params["atr_tp_mult"]
    conf = params["min_confidence"]

    # Skip combos where TP <= SL (need R:R > 1)
    if tp <= sl:
        continue

    rr = tp / sl

    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    total_losses = 0
    total_fees = 0.0

    for sym, df in DATA.items():
        bt = RealisticBacktester(
            strategy_type="adaptive",
            initial_balance=1000.0,
            use_limit_orders=True,
            max_open=3,
            risk_per_trade=0.02,
        )
        # Override strategy with custom params
        bt.strategy = create_strategy("adaptive", **params)
        stats = bt.run(sym, df)

        total_trades += stats.get("total_trades", 0)
        total_pnl += stats.get("total_pnl", 0)
        total_wins += stats.get("wins", 0)
        total_losses += stats.get("losses", 0)
        total_fees += stats.get("total_fees_paid", 0)

    if total_trades < 5:
        continue

    wr = total_wins / total_trades if total_trades > 0 else 0
    ev = total_pnl / total_trades if total_trades > 0 else 0
    pf = (total_pnl + abs(total_pnl)) / (2 * abs(total_pnl)) if total_pnl != 0 else 0  # rough PF

    # Calculate proper profit factor
    win_pnl = total_pnl + sum(abs(total_pnl) for _ in [1])  # placeholder, compute properly
    # Actually just use the aggregate stats
    results.append({
        "sl": sl, "tp": tp, "conf": conf, "rr": rr,
        "trades": total_trades, "wins": total_wins, "losses": total_losses,
        "wr": wr, "pnl": total_pnl, "ev": ev, "fees": total_fees,
    })

    marker = "***" if ev > 0 else "   "
    print(f"{marker} SL={sl:.1f}x TP={tp:.1f}x conf={conf:.2f} R:R={rr:.1f} | "
          f"trades={total_trades:3d} WR={wr:.1%} P&L=${total_pnl:+.2f} EV=${ev:+.2f} {marker}")

# Sort by EV
print("\n" + "=" * 80)
print("  TOP 10 PARAMETER COMBOS (by EV per trade)")
print("=" * 80)
results.sort(key=lambda x: x["ev"], reverse=True)
for r in results[:10]:
    marker = "✓ EDGE" if r["ev"] > 0 else "✗ no edge"
    print(f"  SL={r['sl']:.1f}x TP={r['tp']:.1f}x conf={r['conf']:.2f} R:R={r['rr']:.1f} | "
          f"trades={r['trades']:3d} WR={r['wr']:.1%} P&L=${r['pnl']:+.2f} EV=${r['ev']:+.2f} | {marker}")

print("\n" + "=" * 80)
print("  BOTTOM 5 (worst performers)")
print("=" * 80)
for r in results[-5:]:
    print(f"  SL={r['sl']:.1f}x TP={r['tp']:.1f}x conf={r['conf']:.2f} R:R={r['rr']:.1f} | "
          f"trades={r['trades']:3d} WR={r['wr']:.1%} P&L=${r['pnl']:+.2f} EV=${r['ev']:+.2f}")

# Summary
positive = [r for r in results if r["ev"] > 0]
print(f"\n  {len(positive)}/{len(results)} combos have positive EV")
if positive:
    best = positive[0]
    print(f"\n  BEST: SL={best['sl']:.1f}x TP={best['tp']:.1f}x conf={best['conf']:.2f}")
    print(f"  → {best['trades']} trades, {best['wr']:.1%} WR, ${best['pnl']:+.2f} P&L, ${best['ev']:+.2f} EV/trade")
