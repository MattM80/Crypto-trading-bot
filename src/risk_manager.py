"""
Risk management system for the trading bot.

Key improvements over the original:
  - ATR-aware position sizing (scale down in high volatility)
  - Consecutive-loss cooldown (pause trading after N losses in a row)
  - Trailing stop support (lock in profits as price moves favourably)
  - Win-rate / profit-factor gate (refuse new trades when recent performance is bad)
  - Improved drawdown calculation using running peak balance
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import pandas as pd
import uuid
import math


@dataclass
class Position:
    """Active position tracking"""
    id: str
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    side: str  # "BUY" or "SELL"
    status: str = "OPEN"  # OPEN | PENDING_ENTRY | PENDING_EXIT | CLOSED
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    exit_reason: Optional[str] = None
    # Trailing stop fields
    highest_price: Optional[float] = None  # For BUY positions (track peak)
    lowest_price: Optional[float] = None   # For SELL positions (track trough)
    trailing_stop_active: bool = False
    atr_at_entry: float = 0.0  # ATR when the position was opened


class RiskManager:
    """Manages portfolio risk and position sizing"""

    def __init__(
        self,
        initial_balance: float,
        max_position_size: float = 0.02,
        max_drawdown: float = 0.10,
        max_open_positions: int = 3,
        allow_multiple_positions_per_symbol: bool = False,
        # New parameters
        consecutive_loss_limit: int = 3,
        cooldown_minutes: int = 60,
        trailing_stop_activation: float = 0.5,   # activate after 50% of TP distance
        trailing_stop_callback: float = 0.4,      # trail at 40% of ATR
        min_win_rate_last_n: int = 10,
        min_win_rate_threshold: float = 0.25,
        max_risk_per_trade_pct: float = 0.02,     # 2% of account per trade risk
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_open_positions = max_open_positions
        self.allow_multiple_positions_per_symbol = allow_multiple_positions_per_symbol
        self.positions: Dict[str, List[Position]] = {}
        self.trade_history: List[Dict] = []

        # Consecutive-loss cooldown
        self.consecutive_loss_limit = consecutive_loss_limit
        self.cooldown_minutes = cooldown_minutes
        self._consecutive_losses = 0
        self._cooldown_until: Optional[datetime] = None

        # Trailing stop
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_callback = trailing_stop_callback

        # Performance gate
        self.min_win_rate_last_n = min_win_rate_last_n
        self.min_win_rate_threshold = min_win_rate_threshold

        # Per-trade risk cap
        self.max_risk_per_trade_pct = max_risk_per_trade_pct

        # Adaptive Kelly sizing (set externally by trading bot when journal available)
        self._kelly_fraction: Optional[float] = None  # None = use default risk%
        self.kelly_min_risk = 0.005   # 0.5 % floor
        self.kelly_max_risk = 0.03    # 3.0 % ceiling

    # -----------------------------------------------------------------
    # Position sizing
    # -----------------------------------------------------------------

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_percent: Optional[float] = None,
        account_balance: Optional[float] = None,
        atr: float = 0.0,
    ) -> float:
        """
        Calculate position size using fixed-fractional risk model,
        optionally enhanced with Kelly Criterion sizing.

        When a Kelly fraction is available (from the trade journal),
        quarter-Kelly is used and bounded between 0.5 % and 3.0 %
        of account balance.  This dynamically sizes positions larger
        when the bot is objectively winning and smaller when losing.

        risk_amount = balance * risk_percent
        position_size = risk_amount / |entry - stop_loss|
        """
        if risk_percent is None:
            # Kelly-adaptive risk when available
            if self._kelly_fraction is not None and self._kelly_fraction > 0:
                quarter_kelly = self._kelly_fraction * 0.25
                risk_percent = max(self.kelly_min_risk, min(self.kelly_max_risk, quarter_kelly))
            else:
                risk_percent = self.max_risk_per_trade_pct

        effective_balance = self.current_balance if account_balance is None else float(account_balance)
        risk_amount = effective_balance * risk_percent

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero, cannot calculate position size")
            return 0

        position_size = risk_amount / risk_per_unit

        # Hard cap: scale with account size (small accounts get more leverage room)
        # <$1000: 25% max notional, $1000-$5000: 15%, >$5000: 10%
        if effective_balance < 1000:
            max_notional_pct = 0.25
        elif effective_balance < 5000:
            max_notional_pct = 0.15
        else:
            max_notional_pct = 0.10
        max_nominal = effective_balance * max_notional_pct / entry_price
        position_size = min(position_size, max_nominal)

        # Volatility scaling: shrink further when ATR is unusually high
        if atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            if atr_pct > 0.03:  # >3% ATR -- very volatile
                scale = 0.03 / atr_pct  # linear scale down
                position_size *= max(scale, 0.25)  # never shrink below 25%

        return max(position_size, 0)

    # -----------------------------------------------------------------
    # Trade gating
    # -----------------------------------------------------------------

    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """Check if a new position can be opened"""

        # Cooldown check
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.now()).total_seconds() / 60
            return False, f"Cooldown active ({remaining:.0f}m remaining after {self.consecutive_loss_limit} consecutive losses)"

        # Active position count
        active_statuses = {"OPEN", "PENDING_ENTRY", "PENDING_EXIT"}
        total_active = sum(
            1
            for positions in self.positions.values()
            for p in positions
            if p.status in active_statuses
        )
        if total_active >= self.max_open_positions:
            return False, f"Max open positions reached ({self.max_open_positions})"

        # Per-symbol check
        if not self.allow_multiple_positions_per_symbol:
            if symbol in self.positions and any(
                p.status in active_statuses for p in self.positions[symbol]
            ):
                return False, f"Position already open for {symbol}"

        # Drawdown check
        drawdown = self.calculate_drawdown()
        if drawdown >= self.max_drawdown:
            return False, f"Maximum drawdown reached ({drawdown:.2%})"

        # Recent win-rate gate
        if len(self.trade_history) >= self.min_win_rate_last_n:
            recent = self.trade_history[-self.min_win_rate_last_n:]
            wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
            wr = wins / len(recent)
            if wr < self.min_win_rate_threshold:
                return False, f"Recent win rate too low ({wr:.0%} over last {self.min_win_rate_last_n} trades)"

        return True, "OK"

    # -----------------------------------------------------------------
    # Position recording
    # -----------------------------------------------------------------

    def record_position(self, position: Position) -> None:
        """Record a new open position"""
        if not getattr(position, "id", None):
            position.id = str(uuid.uuid4())

        # Initialize trailing stop tracking
        if position.side == "BUY":
            position.highest_price = position.entry_price
        else:
            position.lowest_price = position.entry_price

        if position.symbol not in self.positions:
            self.positions[position.symbol] = []
        self.positions[position.symbol].append(position)
        logger.info(f"Position recorded: {position.symbol} @ {position.entry_price:.2f}")

    # -----------------------------------------------------------------
    # Trailing stop management
    # -----------------------------------------------------------------

    def update_trailing_stop(self, position: Position, current_price: float) -> None:
        """
        Update trailing stop for a position.

        Activation:  once price has moved trailing_stop_activation of the
                     distance toward take_profit, activate the trail.
        Trail:       move stop to lock in some profit (callback = ATR * factor).
        """
        if position.status != "OPEN":
            return

        atr_cb = position.atr_at_entry * self.trailing_stop_callback if position.atr_at_entry > 0 else 0

        if position.side == "BUY":
            if position.highest_price is None:
                position.highest_price = current_price
            position.highest_price = max(position.highest_price, current_price)

            # Check activation
            entry_to_tp = position.take_profit - position.entry_price
            if entry_to_tp > 0:
                progress = (current_price - position.entry_price) / entry_to_tp
                if progress >= self.trailing_stop_activation:
                    position.trailing_stop_active = True

            if position.trailing_stop_active and atr_cb > 0:
                new_sl = position.highest_price - atr_cb
                if new_sl > position.stop_loss:
                    position.stop_loss = new_sl

        else:  # SELL
            if position.lowest_price is None:
                position.lowest_price = current_price
            position.lowest_price = min(position.lowest_price, current_price)

            entry_to_tp = position.entry_price - position.take_profit
            if entry_to_tp > 0:
                progress = (position.entry_price - current_price) / entry_to_tp
                if progress >= self.trailing_stop_activation:
                    position.trailing_stop_active = True

            if position.trailing_stop_active and atr_cb > 0:
                new_sl = position.lowest_price + atr_cb
                if new_sl < position.stop_loss:
                    position.stop_loss = new_sl

    # -----------------------------------------------------------------
    # Closing positions
    # -----------------------------------------------------------------

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "Manual",
        position_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Close a position and record P&L"""
        if symbol not in self.positions or not self.positions[symbol]:
            logger.warning(f"No open position for {symbol}")
            return None

        active_statuses = {"OPEN", "PENDING_EXIT"}
        position: Optional[Position] = None
        if position_id:
            for p in reversed(self.positions[symbol]):
                if p.id == position_id and p.status in active_statuses:
                    position = p
                    break
        else:
            for p in reversed(self.positions[symbol]):
                if p.status in active_statuses:
                    position = p
                    break

        if position is None:
            logger.warning(f"Position for {symbol} is not open")
            return None

        # Calculate P&L
        if position.side == "BUY":
            pnl = (exit_price - position.entry_price) * position.quantity
            pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:  # SELL
            pnl = (position.entry_price - exit_price) * position.quantity
            pnl_percent = ((position.entry_price - exit_price) / position.entry_price) * 100

        # Update balance
        self.current_balance += pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)

        # Track consecutive losses / wins
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.consecutive_loss_limit:
                self._cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
                logger.warning(
                    f"COOLDOWN activated: {self._consecutive_losses} consecutive losses. "
                    f"Pausing new entries for {self.cooldown_minutes} minutes."
                )
        else:
            self._consecutive_losses = 0
            self._cooldown_until = None

        # Record trade
        trade_record = {
            "position_id": position.id,
            "symbol": symbol,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "side": position.side,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "reason": exit_reason,
            "entry_time": position.entry_time,
            "exit_time": datetime.now().isoformat(),
            "trailing_stop_used": position.trailing_stop_active,
            "atr_at_entry": position.atr_at_entry,
        }
        self.trade_history.append(trade_record)

        # Update position
        position.status = "CLOSED"
        position.exit_reason = exit_reason

        logger.info(f"Position closed: {symbol}, P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")

        return trade_record

    # -----------------------------------------------------------------
    # Drawdown & stats
    # -----------------------------------------------------------------

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak balance"""
        if self.peak_balance == 0:
            return 0
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        return max(drawdown, 0)

    def get_portfolio_stats(self) -> Dict:
        """Get portfolio statistics"""
        if not self.trade_history:
            return {
                "balance": self.current_balance,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_percent": 0,
                "max_drawdown": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "consecutive_losses": self._consecutive_losses,
            }

        trades_df = pd.DataFrame(self.trade_history)

        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        gross_profit = float(winning_trades["pnl"].sum()) if len(winning_trades) > 0 else 0
        gross_loss = abs(float(losing_trades["pnl"].sum())) if len(losing_trades) > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        stats = {
            "balance": self.current_balance,
            "total_trades": len(trades_df),
            "wins": len(winning_trades),
            "losses": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            "total_pnl": trades_df["pnl"].sum(),
            "total_pnl_percent": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            "max_drawdown": self.calculate_drawdown(),
            "avg_win": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
            "profit_factor": profit_factor,
            "consecutive_losses": self._consecutive_losses,
        }

        return stats
