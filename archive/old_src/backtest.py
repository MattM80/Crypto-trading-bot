"""
Backtesting framework to test strategies on historical data.
"""
from typing import Dict, List
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta
import json

from strategies import create_strategy
from risk_manager import RiskManager, Position

class BacktestEngine:
    """Backtest trading strategies on historical data"""
    
    def __init__(
        self,
        strategy_type: str,
        initial_balance: float = 10000,
        **strategy_kwargs
    ):
        self.strategy = create_strategy(strategy_type, **strategy_kwargs)
        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            max_position_size=0.02,
            max_drawdown=0.10,
            max_open_positions=3
        )
        self.initial_balance = initial_balance
        self.trades = []
    
    def backtest(
        self,
        symbol: str,
        data: pd.DataFrame,
        commission: float = 0.001
    ) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
            commission: Trading commission as decimal
        
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest for {symbol}...")
        
        if "close" not in data.columns:
            logger.error("Data must contain 'close' column")
            return {}
        
        # Generate signals for entire dataset
        all_signals = self.strategy.generate_signals(data, symbol)
        
        # Group signals by buy and sell patterns
        entry_signals = []
        for signal in all_signals:
            if signal.action == "BUY":
                entry_signals.append(signal)
        
        # Simulate trades
        for signal in entry_signals:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
                risk_percent=0.02
            )
            
            if position_size <= 0:
                continue
            
            # Create position
            position = Position(
                symbol=symbol,
                entry_price=signal.entry_price,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now().isoformat(),
                side=signal.action
            )
            
            # Simulate price movement and exits
            entry_idx = data.index[data["close"] >= signal.entry_price].min()
            if pd.isna(entry_idx):
                continue
            
            # Look ahead for exit
            future_data = data.loc[entry_idx:].iloc[1:]
            
            if len(future_data) == 0:
                continue
            
            exit_price = None
            exit_reason = "End of data"
            
            for idx, row in future_data.iterrows():
                # Check stop loss
                if row["low"] <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "Stop Loss"
                    break
                
                # Check take profit
                if row["high"] >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "Take Profit"
                    break
            
            # If no exit found, use final close
            if exit_price is None:
                exit_price = future_data.iloc[-1]["close"]
            
            # Record trade with commission
            commission_cost = (position.entry_price * position.quantity * commission) + \
                            (exit_price * position.quantity * commission)
            
            pnl = (exit_price - position.entry_price) * position.quantity - commission_cost
            
            trade = {
                "symbol": symbol,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "quantity": position.quantity,
                "pnl": pnl,
                "pnl_percent": (pnl / (position.entry_price * position.quantity)) * 100,
                "reason": exit_reason,
                "commission": commission_cost
            }
            
            self.trades.append(trade)
            self.risk_manager.current_balance += pnl
        
        return self._calculate_stats()
    
    def _calculate_stats(self) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_percent": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "avg_trade_duration": 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
        total_pnl = trades_df["pnl"].sum()
        total_wins = winning_trades["pnl"].sum()
        total_losses = abs(losing_trades["pnl"].sum())
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        stats = {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0,
            "total_pnl": total_pnl,
            "total_pnl_percent": ((self.risk_manager.current_balance - self.initial_balance) / self.initial_balance) * 100,
            "profit_factor": profit_factor,
            "max_drawdown": self.risk_manager.calculate_drawdown() * 100,
            "avg_win": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
            "largest_win": winning_trades["pnl"].max() if len(winning_trades) > 0 else 0,
            "largest_loss": losing_trades["pnl"].min() if len(losing_trades) > 0 else 0,
            "final_balance": self.risk_manager.current_balance
        }
        
        return stats

def print_backtest_report(stats: Dict, strategy_type: str, symbol: str):
    """Print formatted backtest report"""
    print("\n" + "=" * 70)
    print(f"BACKTEST REPORT - {strategy_type.upper()} Strategy on {symbol}")
    print("=" * 70)
    print(f"Total Trades:        {stats['total_trades']}")
    print(f"Winning Trades:      {stats['winning_trades']}")
    print(f"Losing Trades:       {stats['losing_trades']}")
    print(f"Win Rate:            {stats['win_rate']:.2f}%")
    print(f"Total P&L:           ${stats['total_pnl']:.2f}")
    print(f"Total P&L %:         {stats['total_pnl_percent']:.2f}%")
    print(f"Profit Factor:       {stats['profit_factor']:.2f}")
    print(f"Max Drawdown:        {stats['max_drawdown']:.2f}%")
    print(f"Average Win:         ${stats['avg_win']:.2f}")
    print(f"Average Loss:        ${stats['avg_loss']:.2f}")
    print(f"Largest Win:         ${stats['largest_win']:.2f}")
    print(f"Largest Loss:        ${stats['largest_loss']:.2f}")
    print(f"Final Balance:       ${stats['final_balance']:.2f}")
    print("=" * 70 + "\n")
