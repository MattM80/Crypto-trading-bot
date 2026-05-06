"""NinjaTrader signal export.

Writes JSONL lines to logs/ninjatrader_signals.jsonl. One signal per line.
Designed to be read by an NT side-script or eyeballed by the user.

Fields:
    ts              ISO-8601 UTC timestamp
    spot_pair       Bot's Kraken pair (e.g. "XBTUSD")
    nt_symbol       Suggested NinjaTrader symbol (micro futures preferred)
    side            "long" | "short"
    score           Signal score
    tool            Bot tool name
    regime          Optional regime bucket
    stop_pct        Suggested stop distance
    target_pct      Suggested take-profit distance
    entry_price     Spot reference price
    notional_usd    Bot's spot-side notional (for sizing context, not NT size)
    notes           Free-form

No network calls. Pure append-only file writer. Safe in the bot hot path.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

# Spot pair → preferred NinjaTrader symbol (CME micros where available).
# User must confirm exact symbol roots in NT (MBT = Micro Bitcoin, MET = Micro Ether).
SPOT_TO_NT = {
    "XBTUSD": "MBT",
    "BTCUSD": "MBT",
    "ETHUSD": "MET",
    "XETHZUSD": "MET",
}

DEFAULT_LOG_PATH = os.path.join("logs", "ninjatrader_signals.jsonl")


def to_nt_symbol(spot_pair: str) -> str:
    """Return NT symbol hint for a spot pair, or the pair itself if unmapped."""
    if not spot_pair:
        return ""
    return SPOT_TO_NT.get(spot_pair.upper(), spot_pair)


def export_signal(
    spot_pair: str,
    side: str,
    score: float,
    tool: str,
    entry_price: float,
    stop_pct: float = 0.0,
    target_pct: float = 0.0,
    notional_usd: float = 0.0,
    regime: Optional[str] = None,
    notes: str = "",
    log_path: str = DEFAULT_LOG_PATH,
) -> bool:
    """Append a signal line. Returns True on success, False on I/O error."""
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "spot_pair": spot_pair,
            "nt_symbol": to_nt_symbol(spot_pair),
            "nt_tradable": (spot_pair or "").upper() in SPOT_TO_NT,
            "side": side,
            "score": round(float(score), 2),
            "tool": tool,
            "regime": regime or "",
            "stop_pct": round(float(stop_pct), 4),
            "target_pct": round(float(target_pct), 4),
            "entry_price": float(entry_price),
            "notional_usd": round(float(notional_usd), 2),
            "notes": notes,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return True
    except Exception:
        return False
