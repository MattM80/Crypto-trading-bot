#!/usr/bin/env python3
"""
Edge Scanner — tests every viable Kraken pair to find which ones the strategy
has a real, measurable edge on. This is the foundation for scaling: 
more profitable symbols = more trades/day = faster compounding.

Also projects compound growth at various account sizes.

Usage:
    python scan_edges.py                   # scan all pairs, 15m, 30 days
    python scan_edges.py --timeframe 1h    # test hourly
    python scan_edges.py --project 5000    # project growth from $5K
"""
import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import requests
from loguru import logger

# Suppress verbose logging during scan
logger.remove()
logger.add(sys.stderr, level="ERROR")

from backtest_live import RealisticBacktester, _download_ohlcv, INTERVAL_MAP

# All major USD pairs on Kraken worth testing
ALL_PAIRS = [
    "XBTUSD",  "ETHUSD",  "SOLUSD",  "XRPUSD",  "DOGEUSD",
    "ADAUSD",  "AVAXUSD", "LINKUSD", "DOTUSD",  "MATICUSD",
    "UNIUSD",  "ATOMUSD", "LTCUSD",  "BCHUSD",
    "AAVEUSD", "FILUSD",  "APEUSD",
    "TRXUSD",  "XLMUSD",  "ALGOUSD",
    "NEARUSD", "FETUSD",  "INJUSD",  "OPUSD",   "ARBUSD",
]


def scan_pair(pair: str, interval: int, since: int, until: int, balance: float) -> Dict:
    """Backtest a single pair and return its stats."""
    try:
        df = _download_ohlcv(pair, interval, since, until)
        if df.empty or len(df) < 150:
            return {"symbol": pair, "error": "insufficient data", "total_trades": 0}

        bt = RealisticBacktester(
            strategy_type="adaptive",
            initial_balance=balance,
            use_limit_orders=True,
            max_open=3,
            risk_per_trade=0.02,
        )
        stats = bt.run(pair, df)
        stats["symbol"] = pair
        return stats
    except Exception as e:
        logger.warning(f"Failed to scan {pair}: {e}")
        return {"symbol": pair, "error": str(e), "total_trades": 0}


def project_growth(
    starting_balance: float,
    profitable_symbols: List[Dict],
    months: int = 12,
) -> List[Dict]:
    """
    Project compound growth based on backtest edge data.
    
    Key insight: each profitable symbol contributes ~N trades/day independently.
    More symbols = more trades/day = faster compounding.
    
    We compound daily: each day's profits increase the balance for the next day,
    which proportionally increases position sizes (since our sizing is %-based).
    """
    if not profitable_symbols:
        return []

    # Calculate combined daily edge
    total_ev_per_trade = 0
    total_trades_per_day = 0
    for s in profitable_symbols:
        trades = s.get("total_trades", 0)
        days = max(s.get("days_tested", 1), 1)
        ev = s.get("ev_per_trade", 0)
        trades_per_day = trades / days
        total_trades_per_day += trades_per_day
        total_ev_per_trade += ev * trades_per_day  # weighted EV contribution

    if total_trades_per_day <= 0:
        return []

    avg_ev = total_ev_per_trade / total_trades_per_day  # average EV per trade
    # EV is calibrated on $1000 balance with 2% risk per trade.
    # At different balances, EV scales linearly (since position sizes scale).
    ev_per_dollar_per_trade = avg_ev / 1000.0  # EV per $1 of balance per trade

    projections = []
    balance = starting_balance
    
    for month in range(1, months + 1):
        for day in range(30):  # ~30 days per month
            # Daily trades = total from all profitable symbols
            n_trades = total_trades_per_day
            # Each trade's EV scales with current balance
            daily_ev = balance * ev_per_dollar_per_trade * n_trades
            # Apply some randomness factor (not all days are equal)
            # Use conservative 70% efficiency (some days have fewer setups)
            daily_ev *= 0.70
            balance += daily_ev

        projections.append({
            "month": month,
            "balance": round(balance, 2),
            "monthly_income": round(balance - (projections[-1]["balance"] if projections else starting_balance), 2),
            "total_return_pct": round(((balance - starting_balance) / starting_balance) * 100, 1),
        })

    return projections


def main():
    parser = argparse.ArgumentParser(description="Scan Kraken pairs for trading edge")
    parser.add_argument("--timeframe", default="15m", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance for backtest")
    parser.add_argument("--project", type=float, default=0, help="Project growth from this balance (0 = skip)")
    parser.add_argument("--pairs", default="", help="Comma-separated pairs (empty = all)")
    args = parser.parse_args()

    interval = INTERVAL_MAP.get(args.timeframe, 15)
    now = datetime.now(tz=timezone.utc)
    since = int((now - timedelta(days=args.days)).timestamp())
    until = int(now.timestamp())

    pairs = [s.strip() for s in args.pairs.split(",") if s.strip()] if args.pairs else ALL_PAIRS

    print(f"\n{'='*70}")
    print(f"  EDGE SCANNER — {len(pairs)} pairs × {args.timeframe} × {args.days} days")
    print(f"{'='*70}\n")

    results = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Scanning {pair}...", end=" ", flush=True)
        stats = scan_pair(pair, interval, since, until, args.balance)
        
        trades = stats.get("total_trades", 0)
        if trades == 0:
            print("no trades")
        else:
            wr = stats.get("win_rate", 0)
            pf = stats.get("profit_factor", 0)
            ev = stats.get("ev_per_trade", 0)
            verdict = stats.get("verdict", "")
            tag = "✓" if "STRONG" in verdict else "~" if "MARGINAL" in verdict else "✗"
            print(f"{trades} trades, WR={wr:.0%}, PF={pf:.2f}, EV=${ev:+.2f}  {tag}")
        
        results.append(stats)
        time.sleep(0.5)  # be nice to Kraken API

    # Sort by profit factor
    scored = [r for r in results if r.get("total_trades", 0) >= 3]
    scored.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)

    # Categorize
    strong = [r for r in scored if "STRONG" in r.get("verdict", "")]
    marginal = [r for r in scored if "MARGINAL" in r.get("verdict", "")]
    weak = [r for r in scored if "WEAK" in r.get("verdict", "")]
    no_edge = [r for r in scored if "NO EDGE" in r.get("verdict", "") or r.get("ev_per_trade", 0) <= 0]

    print(f"\n{'='*70}")
    print(f"  SCAN RESULTS SUMMARY")
    print(f"{'='*70}")
    
    if strong:
        print(f"\n  STRONG EDGE ({len(strong)} symbols) — trade these:")
        for s in strong:
            print(f"    {s['symbol']:12s}  {s['total_trades']:3d} trades  WR={s['win_rate']:.0%}  PF={s['profit_factor']:.2f}  EV=${s['ev_per_trade']:+.2f}")

    if marginal:
        print(f"\n  MARGINAL EDGE ({len(marginal)} symbols) — trade cautiously:")
        for s in marginal:
            print(f"    {s['symbol']:12s}  {s['total_trades']:3d} trades  WR={s['win_rate']:.0%}  PF={s['profit_factor']:.2f}  EV=${s['ev_per_trade']:+.2f}")

    if weak:
        print(f"\n  WEAK EDGE ({len(weak)} symbols) — monitor only:")
        for s in weak:
            print(f"    {s['symbol']:12s}  {s['total_trades']:3d} trades  WR={s['win_rate']:.0%}  PF={s['profit_factor']:.2f}  EV=${s['ev_per_trade']:+.2f}")

    if no_edge:
        print(f"\n  NO EDGE ({len(no_edge)} symbols) — avoid:")
        for s in no_edge:
            trades = s.get("total_trades", 0)
            if trades > 0:
                print(f"    {s['symbol']:12s}  {trades:3d} trades  WR={s.get('win_rate',0):.0%}  PF={s.get('profit_factor',0):.2f}  EV=${s.get('ev_per_trade',0):+.2f}")
            else:
                print(f"    {s['symbol']:12s}  no trades / insufficient data")

    # Trading setup recommendation
    profitable = strong + marginal
    if profitable:
        total_trades = sum(s.get("total_trades", 0) for s in profitable)
        total_days = max(max(s.get("days_tested", 1) for s in profitable), 1)
        trades_per_day = total_trades / total_days
        avg_ev = sum(s.get("ev_per_trade", 0) for s in profitable) / len(profitable)

        symbol_list = ",".join(s["symbol"] for s in profitable)

        print(f"\n{'='*70}")
        print(f"  RECOMMENDED CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Symbols:          {symbol_list}")
        print(f"  Timeframe:        {args.timeframe}")
        print(f"  Est. trades/day:  {trades_per_day:.1f}")
        print(f"  Avg EV/trade:     ${avg_ev:+.2f} (on ${args.balance:,.0f} balance)")
        print(f"  Est. daily P&L:   ${avg_ev * trades_per_day:+.2f} (on ${args.balance:,.0f})")
        print(f"  Est. monthly:     ${avg_ev * trades_per_day * 30:+.2f} (on ${args.balance:,.0f})")
        print(f"{'='*70}")

        # Growth projections
        proj_balance = args.project if args.project > 0 else args.balance
        print(f"\n{'='*70}")
        print(f"  COMPOUND GROWTH PROJECTION (starting ${proj_balance:,.0f})")
        print(f"  Based on: {len(profitable)} symbols, {trades_per_day:.1f} trades/day, 70% efficiency")
        print(f"{'='*70}")

        projections = project_growth(proj_balance, profitable, months=12)
        for p in projections:
            print(f"  Month {p['month']:2d}:  Balance ${p['balance']:>10,.2f}  |  Income ${p['monthly_income']:>8,.2f}  |  Total return {p['total_return_pct']:>6.1f}%")

        # Milestone projections at larger balances
        print(f"\n  SCALING PATH:")
        for scale_bal in [1000, 2500, 5000, 10000, 25000, 50000]:
            scale_proj = project_growth(scale_bal, profitable, months=1)
            if scale_proj:
                mo = scale_proj[0]
                print(f"    ${scale_bal:>8,} → ${mo['monthly_income']:>8,.2f}/month")

    # Save results
    results_file = PROJECT_ROOT / "data" / "edge_scan_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "timeframe": args.timeframe,
            "days": args.days,
            "results": results,
            "profitable_symbols": [s["symbol"] for s in profitable] if profitable else [],
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print()


if __name__ == "__main__":
    main()
