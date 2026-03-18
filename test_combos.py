#!/usr/bin/env python3
"""Targeted parameter tests — strategy types + regime multiplier overrides."""
import sys
import os
from pathlib import Path

os.environ["LOGURU_LEVEL"] = "CRITICAL"
import logging
logging.disable(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger
logger.remove()

from backtest_live import RealisticBacktester, print_report
from strategies import create_strategy, AdaptiveQuantStrategy
import risk_manager as rm_mod
rm_mod.logger.remove()

import pandas as pd

CACHE_DIR = PROJECT_ROOT / "data" / "ohlcv_cache"

data = {}
pairs = {
    "ETHUSD": "ETHUSD_15m_1770688800_1773280800.parquet",
    "XLMUSD": "XLMUSD_15m_1770688800_1773280800.parquet",
    "SOLUSD": "SOLUSD_15m_1770688800_1773280800.parquet",
}
for sym, fname in pairs.items():
    p = CACHE_DIR / fname
    if p.exists():
        df = pd.read_parquet(p)
        if len(df) >= 200:
            data[sym] = df
            print(f"Loaded {sym}: {len(df)} bars", flush=True)


def run_combo(label, strategy_type="adaptive", strategy_kwargs=None, risk_pct=0.02):
    """Run a single test combo across all pairs."""
    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    total_losses = 0

    for sym, df in data.items():
        bt = RealisticBacktester(
            strategy_type=strategy_type,
            initial_balance=1000.0,
            use_limit_orders=True,
            max_open=3,
            risk_per_trade=risk_pct,
        )
        if strategy_kwargs:
            bt.strategy = create_strategy(strategy_type, **strategy_kwargs)
        stats = bt.run(sym, df)
        total_trades += stats.get("total_trades", 0)
        total_pnl += stats.get("total_pnl", 0)
        total_wins += stats.get("wins", 0)
        total_losses += stats.get("losses", 0)

    wr = total_wins / total_trades if total_trades > 0 else 0
    ev = total_pnl / total_trades if total_trades > 0 else 0
    verdict = "*** EDGE ***" if ev > 0 and total_trades >= 10 else "maybe" if ev > 0 else "no edge"

    print(f"{label:<25} | {total_trades:>4} {total_wins:>4} {wr:>6.1%} ${total_pnl:>+8.2f} ${ev:>+7.2f} | {verdict}",
          flush=True)
    return {"label": label, "trades": total_trades, "pnl": total_pnl, "ev": ev, "wr": wr}


print(f"\n{'Label':<25} | {'Trd':>4} {'Wins':>4} {'WR':>6} {'P&L':>9} {'EV/trd':>8} | Verdict", flush=True)
print("-" * 80, flush=True)

results = []

# 1. Baseline: AdaptiveQuant with defaults
results.append(run_combo("adaptive_default"))

# 2. Different strategy types
for st in ["trend_momentum", "mean_reversion", "scalp"]:
    results.append(run_combo(f"strat_{st}", strategy_type=st))

# 3. Confidence levels
results.append(run_combo("conf_0.60", strategy_kwargs={"min_confidence": 0.60}))
results.append(run_combo("conf_0.70", strategy_kwargs={"min_confidence": 0.70}))
results.append(run_combo("conf_0.40", strategy_kwargs={"min_confidence": 0.40}))

# 4. Risk per trade levels
results.append(run_combo("risk_1pct", risk_pct=0.01))
results.append(run_combo("risk_3pct", risk_pct=0.03))

# Summary
print("\n" + "=" * 80, flush=True)
results.sort(key=lambda x: x["ev"], reverse=True)
print("RANKED BY EV PER TRADE:", flush=True)
for r in results:
    marker = "<<<" if r["ev"] > 0 else ""
    print(f"  {r['label']:<25} EV=${r['ev']:+.2f}  P&L=${r['pnl']:+.2f}  WR={r['wr']:.1%}  trades={r['trades']}  {marker}", flush=True)
