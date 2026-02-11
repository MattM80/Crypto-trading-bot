"""
JSON-file state persistence for the trading bot.

Saves and restores:
  - Open / pending positions
  - Trade history
  - Risk-manager counters (consecutive losses, cooldown, balances)

The state file is written atomically (write-to-tmp then rename) so a
crash mid-write won't corrupt the file.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


STATE_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_STATE_FILE = STATE_DIR / "bot_state.json"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------

def _position_to_dict(pos) -> Dict:
    """Convert a Position dataclass to a JSON-safe dict."""
    return {
        "id": pos.id,
        "symbol": pos.symbol,
        "entry_price": float(pos.entry_price),
        "quantity": float(pos.quantity),
        "stop_loss": float(pos.stop_loss),
        "take_profit": float(pos.take_profit),
        "entry_time": pos.entry_time,
        "side": pos.side,
        "status": pos.status,
        "entry_order_id": pos.entry_order_id,
        "exit_order_id": getattr(pos, "exit_order_id", None),
        "exit_reason": getattr(pos, "exit_reason", None),
        "highest_price": getattr(pos, "highest_price", None),
        "lowest_price": getattr(pos, "lowest_price", None),
        "trailing_stop_active": getattr(pos, "trailing_stop_active", False),
        "atr_at_entry": getattr(pos, "atr_at_entry", 0.0),
    }


def _dict_to_position(d: Dict):
    """Reconstruct a Position from a dict.  Import here to avoid circular imports."""
    from risk_manager import Position

    return Position(
        id=d["id"],
        symbol=d["symbol"],
        entry_price=float(d["entry_price"]),
        quantity=float(d["quantity"]),
        stop_loss=float(d["stop_loss"]),
        take_profit=float(d["take_profit"]),
        entry_time=d["entry_time"],
        side=d["side"],
        status=d.get("status", "OPEN"),
        entry_order_id=d.get("entry_order_id"),
        exit_order_id=d.get("exit_order_id"),
        exit_reason=d.get("exit_reason"),
        highest_price=d.get("highest_price"),
        lowest_price=d.get("lowest_price"),
        trailing_stop_active=d.get("trailing_stop_active", False),
        atr_at_entry=float(d.get("atr_at_entry", 0.0)),
    )


# ------------------------------------------------------------------
# Save / Load
# ------------------------------------------------------------------

def save_state(
    risk_manager,
    extra: Optional[Dict] = None,
    path: Path = DEFAULT_STATE_FILE,
) -> bool:
    """Persist the current bot state to a JSON file.

    ``risk_manager`` is expected to be a ``RiskManager`` instance.
    ``extra`` is an optional dict of additional top-level keys to store
    (e.g. ``{"kraken_equity_baseline": 1234.56}``).
    """
    try:
        _ensure_dir(path)

        # Collect non-CLOSED positions
        positions_data: Dict[str, List[Dict]] = {}
        for symbol, pos_list in risk_manager.positions.items():
            active = [
                _position_to_dict(p)
                for p in pos_list
                if p.status in ("OPEN", "PENDING_ENTRY", "PENDING_EXIT")
            ]
            if active:
                positions_data[symbol] = active

        state = {
            "saved_at": datetime.now().isoformat(),
            "positions": positions_data,
            "trade_history": risk_manager.trade_history[-500:],  # keep last 500
            "current_balance": risk_manager.current_balance,
            "peak_balance": risk_manager.peak_balance,
            "consecutive_losses": risk_manager._consecutive_losses,
            "cooldown_until": (
                risk_manager._cooldown_until.isoformat()
                if risk_manager._cooldown_until
                else None
            ),
        }

        if extra:
            state.update(extra)

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix="state_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp_path, str(path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.debug(f"State saved → {path} ({len(positions_data)} symbols with active positions)")
        return True

    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        return False


def load_state(
    risk_manager,
    path: Path = DEFAULT_STATE_FILE,
) -> Optional[Dict]:
    """Restore bot state from the JSON file.

    - Restores positions, trade history, and risk counters into *risk_manager*.
    - Returns the full state dict (including any ``extra`` keys) so the caller
      can restore bot-level fields like ``_kraken_equity_baseline``.
    - Returns ``None`` if no state file exists or it cannot be read.
    """
    if not path.exists():
        logger.info("No previous state file found — starting fresh.")
        return None

    try:
        with open(path) as f:
            state = json.load(f)

        # Restore positions
        positions_data = state.get("positions", {})
        restored_count = 0
        for symbol, pos_dicts in positions_data.items():
            for d in pos_dicts:
                try:
                    pos = _dict_to_position(d)
                    if symbol not in risk_manager.positions:
                        risk_manager.positions[symbol] = []
                    risk_manager.positions[symbol].append(pos)
                    restored_count += 1
                except Exception as e:
                    logger.warning(f"Could not restore position {d.get('id')}: {e}")

        # Restore trade history
        risk_manager.trade_history = state.get("trade_history", [])

        # Restore counters
        risk_manager.current_balance = float(state.get("current_balance", risk_manager.current_balance))
        risk_manager.peak_balance = float(state.get("peak_balance", risk_manager.peak_balance))
        risk_manager._consecutive_losses = int(state.get("consecutive_losses", 0))

        cooldown_str = state.get("cooldown_until")
        if cooldown_str:
            try:
                risk_manager._cooldown_until = datetime.fromisoformat(cooldown_str)
            except Exception:
                risk_manager._cooldown_until = None

        logger.info(
            f"State restored from {path} — "
            f"{restored_count} active positions, "
            f"{len(risk_manager.trade_history)} trade history records"
        )
        return state

    except Exception as e:
        logger.error(f"Failed to load state from {path}: {e}")
        return None
