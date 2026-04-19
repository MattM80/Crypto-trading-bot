"""
Risk management system for the trading bot.

Production-grade features:
  - Scalable position sizing (linear with account balance, no hardcoded caps)
  - Daily income tracking with adaptive aggression
  - Tiered drawdown response (reduce size before halting)
  - ATR-aware position sizing with volatility scaling
  - Consecutive-loss cooldown with progressive recovery
  - Trailing stop support with progressive tightening
  - Win-rate / profit-factor gate
  - Running peak balance drawdown tracking
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
    partial_tp_taken: bool = False  # Whether first partial take-profit has been taken
    original_quantity: float = 0.0  # Quantity before partial exits


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
        self._consecutive_wins = 0
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
        self.kelly_min_risk = 0.008   # 0.8 % floor
        self.kelly_max_risk = 0.05    # 5.0 % ceiling

        # --- Daily income tracking ---
        self._daily_trades: List[Dict] = []   # trades closed today
        self._daily_gross_profit: float = 0.0
        self._daily_gross_loss: float = 0.0
        self._current_day: Optional[str] = None  # YYYY-MM-DD

        # --- Drawdown tier state ---
        # 0=normal, 1=caution, 2=reduced, 3=minimal
        self._drawdown_tier: int = 0
        self._drawdown_tier_multipliers = [1.0, 0.65, 0.35, 0.15]

    # -----------------------------------------------------------------
    # Daily tracking (resets automatically on new calendar day)
    # -----------------------------------------------------------------

    def _check_day_reset(self) -> None:
        """Reset daily counters if the calendar day changed."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._current_day != today:
            self._current_day = today
            self._daily_trades = []
            self._daily_gross_profit = 0.0
            self._daily_gross_loss = 0.0

    def get_daily_stats(self) -> Dict:
        """Return today's realized trading stats."""
        self._check_day_reset()
        net = self._daily_gross_profit - abs(self._daily_gross_loss)
        wins = sum(1 for t in self._daily_trades if t.get("pnl", 0) > 0)
        losses = len(self._daily_trades) - wins
        return {
            "date": self._current_day,
            "trades": len(self._daily_trades),
            "wins": wins,
            "losses": losses,
            "gross_profit": self._daily_gross_profit,
            "gross_loss": abs(self._daily_gross_loss),
            "net_pnl": net,
            "win_rate": wins / len(self._daily_trades) if self._daily_trades else 0.0,
        }

    # -----------------------------------------------------------------
    # Drawdown tier system
    # -----------------------------------------------------------------

    def _update_drawdown_tier(self) -> None:
        """Update drawdown tier based on current vs peak balance.

        Tier 0 (< 3% DD):  Full risk — normal operation
        Tier 1 (3-7% DD):  Caution — reduce position size to 65%
        Tier 2 (7-12% DD): Reduced — 35% size, skip low-confidence signals
        Tier 3 (>12% DD):  Minimal — 15% size, only highest-conviction trades
        """
        dd = self.calculate_drawdown()
        if dd < 0.03:
            self._drawdown_tier = 0
        elif dd < 0.07:
            self._drawdown_tier = 1
        elif dd < 0.12:
            self._drawdown_tier = 2
        else:
            self._drawdown_tier = 3

    @property
    def drawdown_tier(self) -> int:
        return self._drawdown_tier

    @property
    def drawdown_size_multiplier(self) -> float:
        return self._drawdown_tier_multipliers[min(self._drawdown_tier, 3)]

    # -----------------------------------------------------------------
    # Position sizing — scales linearly with balance
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
        Calculate position size using fixed-fractional risk model.

        Scales linearly with account balance — putting more money in
        means proportionally larger positions (and proportionally
        larger dollar returns) while keeping risk percentage constant.

        Enhancements:
        - Kelly Criterion from trade history
        - Anti-martingale streak sizing
        - Tiered drawdown response
        - ATR volatility scaling
        """
        if risk_percent is None:
            if self._kelly_fraction is not None and self._kelly_fraction > 0:
                half_kelly = self._kelly_fraction * 0.5
                risk_percent = max(self.kelly_min_risk, min(self.kelly_max_risk, half_kelly))
            else:
                risk_percent = self.max_risk_per_trade_pct

        # Anti-martingale: boost risk after consecutive wins
        if self._consecutive_wins >= 3:
            streak_bonus = min(self._consecutive_wins - 2, 3) * 0.005
            risk_percent = min(risk_percent + streak_bonus, self.kelly_max_risk)
        elif self._consecutive_losses >= 2:
            risk_percent *= 0.75

        # Apply drawdown tier multiplier (gradually reduce, don't slam to zero)
        self._update_drawdown_tier()
        risk_percent *= self.drawdown_size_multiplier

        effective_balance = self.current_balance if account_balance is None else float(account_balance)
        if effective_balance <= 0:
            return 0

        risk_amount = effective_balance * risk_percent

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero, cannot calculate position size")
            return 0

        position_size = risk_amount / risk_per_unit

        # Scale-friendly max notional: always 20% of balance per position
        # This scales linearly — $500 account -> $100 max, $50k -> $10k max
        max_notional_pct = 0.20
        max_nominal = effective_balance * max_notional_pct / entry_price
        position_size = min(position_size, max_nominal)

        # Volatility scaling: shrink further when ATR is unusually high
        if atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            if atr_pct > 0.04:
                scale = 0.04 / atr_pct
                position_size *= max(scale, 0.30)

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

        # Drawdown check — tiered, not binary
        self._update_drawdown_tier()
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

        # Store original quantity for partial TP tracking
        position.original_quantity = position.quantity

        if position.symbol not in self.positions:
            self.positions[position.symbol] = []
        self.positions[position.symbol].append(position)
        logger.info(f"Position recorded: {position.symbol} @ {position.entry_price:.2f}")

    def get_partial_tp_price(self, position: Position) -> Optional[float]:
        """Calculate the partial take-profit level (halfway between entry and TP).

        Returns None if ATR data isn't available or partial TP doesn't make sense.
        """
        if position.partial_tp_taken:
            return None
        if position.atr_at_entry <= 0:
            return None

        # Partial TP at 1.5x ATR (half the distance to full 3x ATR TP)
        atr = position.atr_at_entry
        if position.side == "BUY":
            return position.entry_price + 1.5 * atr
        else:
            return position.entry_price - 1.5 * atr

    # -----------------------------------------------------------------
    # Trailing stop management
    # -----------------------------------------------------------------

    def update_trailing_stop(self, position: Position, current_price: float) -> None:
        """
        Update trailing stop for a position.

        Activation:  once price has moved trailing_stop_activation of the
                     distance toward take_profit, activate the trail.
        Trail:       move stop to lock in profit using ATR-based callback.
        Progressive: as price moves further in profit, tighten the trail.
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
                # Progressive tightening: as profit grows, trail tighter
                profit_pct = (position.highest_price - position.entry_price) / position.entry_price
                if profit_pct > 0.04:  # >4% profit: trail at 60% of normal callback
                    effective_cb = atr_cb * 0.60
                elif profit_pct > 0.02:  # >2% profit: trail at 80% of normal callback
                    effective_cb = atr_cb * 0.80
                else:
                    effective_cb = atr_cb
                new_sl = position.highest_price - effective_cb
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
                profit_pct = (position.entry_price - position.lowest_price) / position.entry_price
                if profit_pct > 0.04:
                    effective_cb = atr_cb * 0.60
                elif profit_pct > 0.02:
                    effective_cb = atr_cb * 0.80
                else:
                    effective_cb = atr_cb
                new_sl = position.lowest_price + effective_cb
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

        # Update drawdown tier
        self._update_drawdown_tier()

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

        # Track daily stats (must come after trade_record is built)
        self._check_day_reset()
        self._daily_trades.append(trade_record)
        if pnl > 0:
            self._daily_gross_profit += pnl
        else:
            self._daily_gross_loss += abs(pnl)

        # Track consecutive losses / wins
        if pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            if self._consecutive_losses >= self.consecutive_loss_limit:
                cooldown_mult = 1 + (self._consecutive_losses - self.consecutive_loss_limit) * 0.5
                cooldown_mins = int(self.cooldown_minutes * cooldown_mult)
                self._cooldown_until = datetime.now() + timedelta(minutes=cooldown_mins)
                logger.warning(
                    f"COOLDOWN activated: {self._consecutive_losses} consecutive losses. "
                    f"Pausing new entries for {cooldown_mins} minutes."
                )
        else:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            self._cooldown_until = None

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
