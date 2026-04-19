#!/usr/bin/env python3
"""Run backtest directly on cached parquet data to avoid slow re-downloads."""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest_live import RealisticBacktester, print_report

CACHE_DIR = PROJECT_ROOT / "data" / "ohlcv_cache"

# Pairs and their best cached 15m files
PAIRS = {
    "SOLUSD": "SOLUSD_15m_1770688800_1773280800.parquet",
    "ETHUSD": "ETHUSD_15m_1770688800_1773280800.parquet",
    "UNIUSD": "UNIUSD_15m_1770688800_1773280800.parquet",
    "XLMUSD": "XLMUSD_15m_1770688800_1773280800.parquet",
    "XRPUSD": "XRPUSD_15m_1770673213_1773265213.parquet",
    "XBTUSD": "XBTUSD_15m_1770673213_1773265213.parquet",
}

all_stats = {}
for symbol, cache_name in PAIRS.items():
    cache_path = CACHE_DIR / cache_name
    if not cache_path.exists():
        print(f"  {symbol}: cache file not found: {cache_name}")
        continue

    df = pd.read_parquet(cache_path)
    if df.empty or len(df) < 200:
        print(f"  {symbol}: insufficient data ({len(df)} bars)")
        continue

    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    print(f"Loading {symbol}: {len(df)} bars, {days:.1f} days")

    bt = RealisticBacktester(
        strategy_type="adaptive",
        initial_balance=1000.0,
        use_limit_orders=True,
        max_open=3,
        risk_per_trade=0.02,
    )
    stats = bt.run(symbol, df)
    all_stats[symbol] = stats
    print_report(stats, symbol)

if len(all_stats) > 1:
    print("\n" + "=" * 70)
    print("  COMBINED SUMMARY")
    print("=" * 70)
    total_trades = sum(s.get("total_trades", 0) for s in all_stats.values())
    total_pnl = sum(s.get("total_pnl", 0) for s in all_stats.values())
    total_fees = sum(s.get("total_fees_paid", 0) for s in all_stats.values())
    total_wins = sum(s.get("wins", 0) for s in all_stats.values())
    total_losses = sum(s.get("losses", 0) for s in all_stats.values())
    avg_ev = total_pnl / total_trades if total_trades > 0 else 0
    print(f"  Total Trades:     {total_trades}")
    print(f"  Wins/Losses:      {total_wins}/{total_losses}")
    print(f"  Win Rate:         {total_wins/total_trades:.1%}" if total_trades > 0 else "  Win Rate: N/A")
    print(f"  Combined P&L:     ${total_pnl:+.2f}")
    print(f"  Combined Fees:    ${total_fees:.2f}")
    print(f"  EV per Trade:     ${avg_ev:+.2f}")
    print("=" * 70)
