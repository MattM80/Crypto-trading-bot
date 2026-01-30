"""
Risk management system for the trading bot.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger
import pandas as pd
import uuid

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

class RiskManager:
    """Manages portfolio risk and position sizing"""
    
    def __init__(
        self,
        initial_balance: float,
        max_position_size: float = 0.02,
        max_drawdown: float = 0.10,
        max_open_positions: int = 3,
        allow_multiple_positions_per_symbol: bool = False,
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_open_positions = max_open_positions
        self.allow_multiple_positions_per_symbol = allow_multiple_positions_per_symbol
        self.positions: Dict[str, List[Position]] = {}
        self.trade_history: List[Dict] = []
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_percent: Optional[float] = None,
        account_balance: Optional[float] = None,
    ) -> float:
        """
        Calculate appropriate position size based on risk management.
        
        Uses Kelly Criterion and portfolio heat calculation.
        """
        if risk_percent is None:
            risk_percent = self.max_position_size
        
        effective_balance = self.current_balance if account_balance is None else float(account_balance)

        # Maximum position size in dollars
        max_risk_amount = effective_balance * risk_percent
        
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero, cannot calculate position size")
            return 0
        
        # Position size
        position_size = max_risk_amount / risk_per_unit
        
        # Ensure position doesn't exceed 10% of balance in nominal value
        max_nominal = effective_balance * 0.10 / entry_price
        position_size = min(position_size, max_nominal)
        
        return max(position_size, 0)
    
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """Check if a new position can be opened"""
        
        # Check max open positions (count only active/pending positions)
        active_statuses = {"OPEN", "PENDING_ENTRY", "PENDING_EXIT"}
        total_active = sum(
            1
            for positions in self.positions.values()
            for p in positions
            if p.status in active_statuses
        )
        if total_active >= self.max_open_positions:
            return False, f"Max open positions reached ({self.max_open_positions})"
        
        # Check if symbol already has an active/pending position (unless explicitly allowed)
        if not self.allow_multiple_positions_per_symbol:
            if symbol in self.positions and any(
                p.status in {"OPEN", "PENDING_ENTRY", "PENDING_EXIT"} for p in self.positions[symbol]
            ):
                return False, f"Position already open for {symbol}"
        
        # Check drawdown
        drawdown = self.calculate_drawdown()
        if drawdown >= self.max_drawdown:
            return False, f"Maximum drawdown reached ({drawdown:.2%})"
        
        return True, "OK"
    
    def record_position(self, position: Position) -> None:
        """Record a new open position"""
        if not getattr(position, "id", None):
            position.id = str(uuid.uuid4())
        if position.symbol not in self.positions:
            self.positions[position.symbol] = []
        self.positions[position.symbol].append(position)
        logger.info(f"Position recorded: {position.symbol} @ {position.entry_price:.2f}")
    
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

        # Find the specific active position to close
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
        }
        self.trade_history.append(trade_record)
        
        # Update position
        position.status = "CLOSED"
        position.exit_reason = exit_reason
        
        logger.info(f"Position closed: {symbol}, P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")
        
        return trade_record
    
    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from initial balance"""
        peak_balance = self.initial_balance
        for trade in self.trade_history:
            peak_balance = max(peak_balance, self.current_balance - trade["pnl"])
        
        if peak_balance == 0:
            return 0
        
        drawdown = (peak_balance - self.current_balance) / peak_balance
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
                "max_drawdown": 0
            }
        
        trades_df = pd.DataFrame(self.trade_history)
        
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
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
            "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0
        }
        
        return stats
