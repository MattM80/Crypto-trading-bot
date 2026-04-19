"""Kraken performance report (realized P&L net of fees).

This script uses Kraken's *own* trade history to compute realized P&L and fees.
It is meant to answer: "how did the bot do" including fees.

Usage examples:
  python .\report_kraken_performance.py
  python .\report_kraken_performance.py --pair XRPUSD --hours 24

Requires API permissions:
  - Query Trades
  - Query Funds (optional, for balances)

Environment:
  - KRAKEN_API_KEY
  - KRAKEN_PRIVATE_KEY
  - KRAKEN_SYMBOLS (optional, used as default pair)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Import KrakenClient without importing the src package (src/__init__.py has side effects).
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
KRAKEN_CLIENT_PATH = SRC_DIR / "kraken_client.py"

_spec = importlib.util.spec_from_file_location("kraken_client", KRAKEN_CLIENT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load kraken_client from {KRAKEN_CLIENT_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
KrakenClient = _mod.KrakenClient


@dataclass
class Trade:
    txid: str
    pair: str
    side: str  # buy/sell
    price: float
    vol: float
    cost: float
    fee: float
    timestamp: float


def _parse_trades(raw: Dict) -> List[Trade]:
    trades = []
    if not raw:
        return trades

    trade_map = raw.get("trades") or {}
    for txid, t in trade_map.items():
        try:
            trades.append(
                Trade(
                    txid=str(txid),
                    pair=str(t.get("pair") or ""),
                    side=str(t.get("type") or "").lower(),
                    price=float(t.get("price") or 0),
                    vol=float(t.get("vol") or 0),
                    cost=float(t.get("cost") or 0),
                    fee=float(t.get("fee") or 0),
                    timestamp=float(t.get("time") or 0),
                )
            )
        except Exception:
            continue

    trades.sort(key=lambda x: x.timestamp)
    return trades


def _fifo_realized_pnl(trades: List[Trade], pair: str) -> Tuple[float, float, float, float]:
    """Compute realized PnL in quote currency using FIFO lots.

    Returns (realized_pnl, total_fees, buys_quote, sells_quote).

    Assumptions:
    - Trade.cost is quote amount (USD for XRP/USD).
    - Trade.fee is also in quote.
    - For buys: cash outflow = cost + fee.
    - For sells: cash inflow = cost - fee.
    - Cost basis includes buy fees.
    """

    lots: List[Tuple[float, float]] = []  # (qty, cost_basis_quote_total)
    realized = 0.0
    fees = 0.0
    buys_quote = 0.0
    sells_quote = 0.0

    for tr in trades:
        if tr.pair.upper() != pair.upper():
            continue
        if tr.vol <= 0:
            continue

        fees += tr.fee

        if tr.side == "buy":
            total_cost = tr.cost + tr.fee
            lots.append((tr.vol, total_cost))
            buys_quote += total_cost
            continue

        if tr.side == "sell":
            remaining = tr.vol
            proceeds = tr.cost - tr.fee
            sells_quote += proceeds

            # Match FIFO
            matched_cost = 0.0
            while remaining > 1e-12 and lots:
                lot_qty, lot_cost = lots[0]
                if lot_qty <= 1e-12:
                    lots.pop(0)
                    continue

                take = min(remaining, lot_qty)
                # Allocate proportional cost basis
                proportion = take / lot_qty
                cost_taken = lot_cost * proportion

                matched_cost += cost_taken

                new_qty = lot_qty - take
                new_cost = lot_cost - cost_taken
                lots[0] = (new_qty, new_cost)
                if lots[0][0] <= 1e-12:
                    lots.pop(0)

                remaining -= take

            # If we sold more than we have (shouldn't happen on spot), treat unmatched as zero cost
            realized += proceeds - matched_cost
            continue

    return realized, fees, buys_quote, sells_quote


def _ts_to_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=None, help="Kraken pair, e.g. XRPUSD")
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours")
    args = parser.parse_args()

    default_pair = None
    env_syms = (os.getenv("KRAKEN_SYMBOLS") or "").strip()
    if env_syms:
        default_pair = env_syms.split(",")[0].strip()

    pair = (args.pair or default_pair or "XRPUSD").strip()

    now = time.time()
    start = int(now - float(args.hours) * 3600)

    client = KrakenClient(sandbox=False)

    raw = client.get_trades_history(start=start)
    if not raw:
        print("No trade history returned. Check API permissions: 'Query Trades'.")
        return 2

    trades = _parse_trades(raw)
    relevant = [t for t in trades if t.pair.upper() == pair.upper()]

    print(f"Pair: {pair}")
    print(f"Window: last {args.hours:g} hours (since {_ts_to_str(start)})")
    print(f"Trades returned: {len(trades)} (relevant: {len(relevant)})")

    realized, fees, buys_quote, sells_quote = _fifo_realized_pnl(trades, pair)

    print("-")
    print(f"Realized P&L (net fees): {realized:+.4f} (quote currency)")
    print(f"Total fees:             {fees:.4f} (quote currency)")
    print(f"Total buy outflow:      {buys_quote:.4f} (quote currency)")
    print(f"Total sell inflow:      {sells_quote:.4f} (quote currency)")

    # Optional: mark-to-market value of leftover inventory (if any)
    try:
        ticker = client.get_ticker(pair)
        last = float((ticker or {}).get("price") or 0)
    except Exception:
        last = 0.0

    # compute leftover qty and cost basis
    # Re-run quickly to derive leftover lots
    lots: List[Tuple[float, float]] = []
    for tr in trades:
        if tr.pair.upper() != pair.upper():
            continue
        if tr.side == "buy" and tr.vol > 0:
            lots.append((tr.vol, tr.cost + tr.fee))
        elif tr.side == "sell" and tr.vol > 0:
            remaining = tr.vol
            while remaining > 1e-12 and lots:
                lot_qty, lot_cost = lots[0]
                take = min(remaining, lot_qty)
                proportion = take / lot_qty if lot_qty else 1.0
                cost_taken = lot_cost * proportion
                lots[0] = (lot_qty - take, lot_cost - cost_taken)
                if lots[0][0] <= 1e-12:
                    lots.pop(0)
                remaining -= take

    leftover_qty = sum(q for q, _ in lots)
    leftover_cost = sum(c for _, c in lots)

    if leftover_qty > 1e-9 and last > 0:
        mtm_value = leftover_qty * last
        unrealized = mtm_value - leftover_cost
        print("-")
        print(f"Leftover inventory:     {leftover_qty:.8f} base")
        print(f"Leftover cost basis:    {leftover_cost:.4f} quote")
        print(f"Mark-to-market value:   {mtm_value:.4f} quote (last={last:.6f})")
        print(f"Unrealized P&L:         {unrealized:+.4f} quote")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
