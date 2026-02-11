"""
Trade Journal & Adaptive Learning System
=========================================
Records full indicator state with each trade entry/exit.
Analyzes which features predict profitable trades.
Provides learnable feature weights that improve with trading history.
Persists learned state to disk for continuous improvement across restarts.

Learning approach (online Bayesian-inspired):
- Each indicator feature has a weight that modifies its contribution to confidence.
- After each trade exit, features present in winning trades get boosted,
  features present in losing trades get dampened.
- Weights converge over time to reflect which indicators actually predict profit.
- Per-pair and per-regime statistics drive dynamic confidence thresholds.
"""

import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_JOURNAL_FILE = DATA_DIR / "trade_journal.json"

# ---------------------------------------------------------------------------
# Default feature weights (initial priors from TrendMomentum signal analysis)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_WEIGHTS = {
    "ema_cross": 1.0,
    "ema_aligned": 1.0,
    "macd_signal": 1.0,
    "macd_histogram": 1.0,
    "rsi_favorable": 1.0,
    "adx_strong": 1.0,
    "di_aligned": 1.0,
    "volume_confirmed": 1.0,
    "obv_aligned": 1.0,
    "vwap_favorable": 1.0,
    "bb_squeeze": 1.0,
    "rsi_divergence": 1.0,
    "htf_aligned": 1.0,
    "btc_favorable": 1.0,
}

LEARNING_RATE = 0.05          # Weight adjustment speed (0.01 slow … 0.10 fast)
MIN_TRADES_FOR_LEARNING = 10  # Minimum completed trades before adjusting weights
WEIGHT_FLOOR = 0.20           # Never let a weight go below this
WEIGHT_CEILING = 3.0          # Never let a weight go above this


class TradeJournal:
    """
    Adaptive trade journal that learns from trading history.

    Records indicator state at entry, links to exit outcomes,
    and adjusts feature weights for better signal quality over time.
    """

    def __init__(self, path: Path = DEFAULT_JOURNAL_FILE):
        self.path = path
        self.feature_weights: Dict[str, float] = dict(DEFAULT_FEATURE_WEIGHTS)
        self.open_entries: Dict[str, Dict] = {}           # trade_id -> entry record
        self.completed_trades: List[Dict] = []
        self.pair_stats: Dict[str, Dict] = {}
        self.regime_stats: Dict[str, Dict] = {}
        # Feature correlation tracking
        self.feature_outcomes: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_pair_stats(self, symbol: str) -> Dict:
        if symbol not in self.pair_stats:
            self.pair_stats[symbol] = {
                "wins": 0, "losses": 0, "total_pnl": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
            }
        return self.pair_stats[symbol]

    def _ensure_regime_stats(self, regime: str) -> Dict:
        if regime not in self.regime_stats:
            self.regime_stats[regime] = {
                "wins": 0, "losses": 0, "total_pnl": 0.0,
            }
        return self.regime_stats[regime]

    def _ensure_feature_outcomes(self, feature: str) -> Dict:
        if feature not in self.feature_outcomes:
            self.feature_outcomes[feature] = {
                "win_present": 0, "win_absent": 0,
                "loss_present": 0, "loss_absent": 0,
            }
        return self.feature_outcomes[feature]

    # ------------------------------------------------------------------
    # Record trade lifecycle
    # ------------------------------------------------------------------

    def record_entry(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        regime: str,
        features_active: Dict[str, bool],
        indicator_snapshot: Dict[str, float],
        htf_trend: str = "unknown",
        btc_trend: str = "unknown",
    ) -> None:
        """Record a trade entry with full indicator state."""
        self.open_entries[trade_id] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "entry_time": datetime.now().isoformat(),
            "regime": regime,
            "features_active": features_active,
            "indicator_snapshot": indicator_snapshot,
            "htf_trend": htf_trend,
            "btc_trend": btc_trend,
        }

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        exit_reason: str,
    ) -> None:
        """Record a trade exit and update learning statistics."""
        entry = self.open_entries.pop(trade_id, None)
        if entry is None:
            return  # No matching entry — nothing to learn from

        is_win = pnl > 0
        symbol = entry["symbol"]
        regime = entry["regime"]
        features = entry.get("features_active", {})

        # Build completed trade record
        trade = {
            **entry,
            "exit_price": exit_price,
            "exit_time": datetime.now().isoformat(),
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "exit_reason": exit_reason,
            "is_win": is_win,
        }
        self.completed_trades.append(trade)

        # --- Update pair stats ---
        ps = self._ensure_pair_stats(symbol)
        if is_win:
            ps["wins"] += 1
            n = ps["wins"]
            ps["avg_win"] = ps["avg_win"] * (n - 1) / n + pnl / n
        else:
            ps["losses"] += 1
            n = ps["losses"]
            ps["avg_loss"] = ps["avg_loss"] * (n - 1) / n + abs(pnl) / n
        ps["total_pnl"] += pnl

        # --- Update regime stats ---
        rs = self._ensure_regime_stats(regime)
        if is_win:
            rs["wins"] += 1
        else:
            rs["losses"] += 1
        rs["total_pnl"] += pnl

        # --- Update feature outcome tracking ---
        for feature_name in DEFAULT_FEATURE_WEIGHTS:
            fo = self._ensure_feature_outcomes(feature_name)
            present = bool(features.get(feature_name, False))
            if is_win:
                fo["win_present" if present else "win_absent"] += 1
            else:
                fo["loss_present" if present else "loss_absent"] += 1

        # --- Update weights if enough data ---
        if len(self.completed_trades) >= MIN_TRADES_FOR_LEARNING:
            self._update_weights()

    # ------------------------------------------------------------------
    # Weight learning
    # ------------------------------------------------------------------

    def _update_weights(self) -> None:
        """Update feature weights based on outcome statistics.

        For each feature compute a "lift" score:
            lift = P(win | feature present) / P(win | feature absent)

        lift > 1 → feature predicts wins → increase weight
        lift < 1 → feature anti-predicts → decrease weight
        """
        for feature_name in DEFAULT_FEATURE_WEIGHTS:
            fo = self._ensure_feature_outcomes(feature_name)
            wp = fo["win_present"]
            lp = fo["loss_present"]
            wa = fo["win_absent"]
            la = fo["loss_absent"]

            total_present = wp + lp
            total_absent = wa + la

            if total_present < 3 or total_absent < 3:
                continue  # Not enough data for this feature

            p_win_present = wp / total_present
            p_win_absent = wa / total_absent

            if p_win_absent == 0:
                lift = 2.0  # Feature is clearly helpful
            else:
                lift = p_win_present / p_win_absent

            current = self.feature_weights.get(feature_name, 1.0)
            target = max(WEIGHT_FLOOR, min(WEIGHT_CEILING, lift))
            new_weight = current + LEARNING_RATE * (target - current)
            new_weight = max(WEIGHT_FLOOR, min(WEIGHT_CEILING, new_weight))
            self.feature_weights[feature_name] = new_weight

    # ------------------------------------------------------------------
    # Query methods (used by strategy and risk manager)
    # ------------------------------------------------------------------

    def get_pair_win_rate(self, symbol: str) -> Optional[float]:
        """Win rate for a specific pair.  None if < 5 trades."""
        ps = self.pair_stats.get(symbol)
        if not ps:
            return None
        total = ps["wins"] + ps["losses"]
        return ps["wins"] / total if total >= 5 else None

    def get_pair_kelly(self, symbol: str) -> Optional[float]:
        """Kelly fraction for a specific pair.  None if < 10 trades."""
        ps = self.pair_stats.get(symbol)
        if not ps:
            return None
        total = ps["wins"] + ps["losses"]
        if total < 10:
            return None
        win_rate = ps["wins"] / total
        avg_win = ps["avg_win"] if ps["avg_win"] > 0 else 1.0
        avg_loss = ps["avg_loss"] if ps["avg_loss"] > 0 else 1.0
        payoff_ratio = avg_win / avg_loss
        if payoff_ratio == 0:
            return None
        kelly = win_rate - (1 - win_rate) / payoff_ratio
        return kelly

    def get_regime_win_rate(self, regime: str) -> Optional[float]:
        """Win rate for a specific regime.  None if < 5 trades."""
        rs = self.regime_stats.get(regime)
        if not rs:
            return None
        total = rs["wins"] + rs["losses"]
        return rs["wins"] / total if total >= 5 else None

    def get_overall_kelly(self) -> Optional[float]:
        """Kelly fraction across all completed trades.  None if < 15 trades."""
        if len(self.completed_trades) < 15:
            return None
        wins = [t for t in self.completed_trades if t.get("is_win")]
        losses = [t for t in self.completed_trades if not t.get("is_win")]
        if not wins or not losses:
            return None
        win_rate = len(wins) / len(self.completed_trades)
        avg_win = sum(t["pnl"] for t in wins) / len(wins)
        avg_loss = sum(abs(t["pnl"]) for t in losses) / len(losses)
        if avg_loss == 0:
            return None
        payoff_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / payoff_ratio
        return kelly

    def get_dynamic_confidence_threshold(self, symbol: str, regime: str) -> float:
        """Dynamic confidence threshold based on pair + regime performance.

        Well-performing pair/regime combos get a slightly lower threshold
        (take more trades).  Poor combos get a higher threshold (be picky).
        """
        base = 0.45

        pair_wr = self.get_pair_win_rate(symbol)
        regime_wr = self.get_regime_win_rate(regime)

        adjustment = 0.0
        if pair_wr is not None:
            if pair_wr > 0.55:
                adjustment -= 0.03
            elif pair_wr < 0.35:
                adjustment += 0.05

        if regime_wr is not None:
            if regime_wr > 0.55:
                adjustment -= 0.02
            elif regime_wr < 0.35:
                adjustment += 0.03

        return max(0.35, min(0.60, base + adjustment))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = None) -> bool:
        """Persist journal state to disk (atomic write)."""
        path = path or self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "saved_at": datetime.now().isoformat(),
                "feature_weights": self.feature_weights,
                "open_entries": self.open_entries,
                "completed_trades": self.completed_trades[-1000:],
                "pair_stats": dict(self.pair_stats),
                "regime_stats": dict(self.regime_stats),
                "feature_outcomes": {k: dict(v) for k, v in self.feature_outcomes.items()},
            }
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix="journal_"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f, indent=2, default=str)
                os.replace(tmp_path, str(path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            logger.debug(f"Trade journal saved → {path} ({len(self.completed_trades)} trades)")
            return True
        except Exception as e:
            logger.error(f"Failed to save trade journal: {e}")
            return False

    def load(self, path: Path = None) -> bool:
        """Load journal state from disk."""
        path = path or self.path
        if not path.exists():
            logger.info("No trade journal found — starting fresh (learning begins after first trades)")
            return False
        try:
            with open(path) as f:
                state = json.load(f)

            self.feature_weights = state.get("feature_weights", dict(DEFAULT_FEATURE_WEIGHTS))
            self.open_entries = state.get("open_entries", {})
            self.completed_trades = state.get("completed_trades", [])

            for symbol, stats in state.get("pair_stats", {}).items():
                self.pair_stats[symbol] = stats
            for regime, stats in state.get("regime_stats", {}).items():
                self.regime_stats[regime] = stats
            for feature, outcomes in state.get("feature_outcomes", {}).items():
                self.feature_outcomes[feature] = outcomes

            # Ensure all default features exist in weights
            for f in DEFAULT_FEATURE_WEIGHTS:
                if f not in self.feature_weights:
                    self.feature_weights[f] = DEFAULT_FEATURE_WEIGHTS[f]

            logger.info(
                f"Trade journal restored: {len(self.completed_trades)} trades | "
                f"Learned weights: {', '.join(f'{k}={v:.2f}' for k, v in sorted(self.feature_weights.items()))}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load trade journal: {e}")
            return False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Human-readable summary of learned patterns."""
        n = len(self.completed_trades)
        if n == 0:
            return "No completed trades yet — learning begins after first trades."

        wins = sum(1 for t in self.completed_trades if t.get("is_win"))
        wr = wins / n if n > 0 else 0
        total_pnl = sum(t.get("pnl", 0) for t in self.completed_trades)

        sorted_features = sorted(
            self.feature_weights.items(), key=lambda x: x[1], reverse=True,
        )
        top_3 = ", ".join(f"{k}={v:.2f}" for k, v in sorted_features[:3])
        bottom_3 = ", ".join(f"{k}={v:.2f}" for k, v in sorted_features[-3:])

        return (
            f"Trades: {n} | Win rate: {wr:.1%} | Total P&L: ${total_pnl:.2f}\n"
            f"Top features: {top_3}\n"
            f"Weakest features: {bottom_3}"
        )
