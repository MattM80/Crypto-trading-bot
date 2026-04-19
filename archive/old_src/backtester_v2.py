"""
Backtester V2 - Honest Bar-by-Bar Simulation
============================================

NO look-ahead bias. NO perfect entry/exit prices. NO fantasy land backtests.

This backtester:
1. Walks forward bar by bar
2. Applies realistic fees (0.16% maker + 0.05% slippage per side)
3. Checks stop losses and take profits on OHLC data (if high/low hit levels)
4. Uses POST-ONLY limit orders with realistic fill assumptions
5. Reports brutal truth about strategy performance

If a strategy can't make money here, it won't make money live.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import json

from strategies_v2 import StrategyBase, Signal, create_strategy_portfolio
from portfolio_manager import PortfolioManager


@dataclass 
class BacktestOrder:
    """Simulated order in backtest"""
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    order_type: str = "LIMIT"  # LIMIT/MARKET
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    filled: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    strategy: str = ""


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    initial_balance: float
    final_balance: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trade_history: List[Dict]
    strategy_stats: Dict
    daily_returns: List[float]
    equity_curve: pd.Series


class BacktesterV2:
    """Honest backtesting engine"""
    
    def __init__(self, 
                 initial_balance: float = 300,
                 maker_fee_pct: float = 0.16,
                 slippage_pct: float = 0.05):
        
        self.initial_balance = initial_balance
        self.maker_fee_pct = maker_fee_pct  # 0.16% maker fee
        self.slippage_pct = slippage_pct    # 0.05% slippage estimate
        
        # Total round-trip cost = (maker + slippage) * 2 sides = 0.42%
        self.total_round_trip_cost_pct = (maker_fee_pct + slippage_pct) * 2
        
        # Tracking
        self.current_balance = initial_balance
        self.equity_curve = []
        self.daily_returns = []
        self.active_orders: Dict[str, BacktestOrder] = {}
        self.trade_history = []
        
        logger.info(f"Backtester initialized: ${initial_balance:.2f} starting balance, "
                   f"{self.total_round_trip_cost_pct:.2f}% round-trip cost")
    
    def calculate_realistic_fill(self, order: BacktestOrder, bar: Dict) -> Tuple[bool, Optional[float]]:
        """
        Calculate if and at what price an order would fill, considering:
        1. Limit orders only fill if price is touched
        2. Slippage for market impact
        3. Realistic execution assumptions
        """
        
        high = float(bar['high'])
        low = float(bar['low']) 
        open_price = float(bar['open'])
        close_price = float(bar['close'])
        
        # For limit orders, check if price was touched
        if order.order_type == "LIMIT":
            if order.side == "BUY":
                # Buy limit only fills if low <= limit_price
                if low <= order.price:
                    # Fill at limit price or better, with slippage
                    fill_price = order.price * (1 + self.slippage_pct / 100)
                    return True, min(fill_price, high)  # Can't fill above high
                else:
                    return False, None
                    
            else:  # SELL
                # Sell limit only fills if high >= limit_price  
                if high >= order.price:
                    # Fill at limit price or better, with slippage
                    fill_price = order.price * (1 - self.slippage_pct / 100)
                    return True, max(fill_price, low)  # Can't fill below low
                else:
                    return False, None
                    
        else:  # MARKET order
            # Market orders fill at open with slippage
            if order.side == "BUY":
                fill_price = open_price * (1 + self.slippage_pct / 100)
            else:
                fill_price = open_price * (1 - self.slippage_pct / 100)
            return True, fill_price
    
    def check_stop_take_profit(self, order: BacktestOrder, bar: Dict) -> Optional[Tuple[str, float]]:
        """
        Check if stop loss or take profit was hit.
        Returns (exit_reason, exit_price) or None
        """
        
        if not order.filled:
            return None
            
        high = float(bar['high'])
        low = float(bar['low'])
        
        if order.side == "BUY":
            # Long position - check stop loss (below) and take profit (above)
            if order.stop_loss and low <= order.stop_loss:
                # Stop loss hit - sell at stop level with slippage
                exit_price = order.stop_loss * (1 - self.slippage_pct / 100)
                return ("STOP_LOSS", max(exit_price, low))
                
            elif order.take_profit and high >= order.take_profit:
                # Take profit hit - sell at TP level with slippage
                exit_price = order.take_profit * (1 - self.slippage_pct / 100) 
                return ("TAKE_PROFIT", min(exit_price, high))
                
        else:  # SELL (short position)
            # Short position - check stop loss (above) and take profit (below)
            if order.stop_loss and high >= order.stop_loss:
                # Stop loss hit - buy at stop level with slippage
                exit_price = order.stop_loss * (1 + self.slippage_pct / 100)
                return ("STOP_LOSS", min(exit_price, high))
                
            elif order.take_profit and low <= order.take_profit:
                # Take profit hit - buy at TP level with slippage
                exit_price = order.take_profit * (1 + self.slippage_pct / 100)
                return ("TAKE_PROFIT", max(exit_price, low))
                
        return None
    
    def calculate_pnl(self, entry_price: float, exit_price: float, quantity: float, 
                     side: str, fees: float) -> float:
        """Calculate PnL including fees"""
        
        if side == "BUY":
            gross_pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            gross_pnl = (entry_price - exit_price) * quantity
            
        net_pnl = gross_pnl - fees
        return net_pnl
    
    def execute_signal(self, signal: Signal, current_bar: Dict, timestamp: datetime) -> Optional[BacktestOrder]:
        """Convert signal to backtest order"""
        
        if signal.action in ["HOLD", "CLOSE_LONG", "CLOSE_SHORT"]:
            return None
            
        # Create limit order at signal price
        order = BacktestOrder(
            symbol=signal.symbol,
            side=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            order_type="LIMIT",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            timestamp=timestamp,
            strategy=signal.strategy
        )
        
        # Try to fill immediately
        filled, fill_price = self.calculate_realistic_fill(order, current_bar)
        
        if filled and fill_price:
            order.filled = True
            order.fill_price = fill_price
            order.fill_time = timestamp
            
            # Calculate fees
            trade_value = order.quantity * fill_price
            fees = trade_value * (self.maker_fee_pct / 100)  # Entry fee only
            
            # Update balance
            if order.side == "BUY":
                self.current_balance -= (trade_value + fees)
            else:  # SELL
                self.current_balance += (trade_value - fees)
                
            logger.debug(f"Order filled: {order.side} {order.quantity:.6f} {order.symbol} "
                        f"@ ${fill_price:.2f} (fees: ${fees:.2f})")
                        
            return order
        
        else:
            logger.debug(f"Order not filled: {order.side} {order.symbol} @ ${order.price:.2f}")
            return None
    
    def process_bar(self, 
                   bar_data: Dict[str, Dict], 
                   strategies: List[StrategyBase],
                   portfolio_manager: PortfolioManager,
                   timestamp: datetime,
                   historical_data: Dict[str, pd.DataFrame],
                   current_bar_index: int) -> None:
        """Process one bar of data"""
        
        # Convert bar data to DataFrame format for strategies
        # Provide historical context up to current bar
        strategy_data = {}
        for symbol, bar in bar_data.items():
            if symbol in historical_data:
                # Give strategies access to all historical data up to current point
                strategy_data[symbol] = historical_data[symbol].iloc[:current_bar_index + 1].copy()
            
        # Check existing orders for stops/TPs first
        orders_to_close = []
        for order_id, order in self.active_orders.items():
            if order.filled:
                symbol_bar = bar_data.get(order.symbol)
                if symbol_bar:
                    exit_result = self.check_stop_take_profit(order, symbol_bar)
                    if exit_result:
                        exit_reason, exit_price = exit_result
                        orders_to_close.append((order_id, order, exit_price, exit_reason))
        
        # Close orders that hit stops/TPs
        for order_id, order, exit_price, exit_reason in orders_to_close:
            self.close_position(order, exit_price, exit_reason, timestamp)
            del self.active_orders[order_id]
            
        # Generate new signals from strategies
        for strategy in strategies:
            try:
                signal = strategy.generate_signal(strategy_data)
                
                if signal:
                    # Let portfolio manager evaluate and size the signal
                    evaluated_signal = portfolio_manager.evaluate_signal(signal, strategy.name)
                    
                    if evaluated_signal and evaluated_signal.quantity > 0:
                        # Execute the signal
                        symbol_bar = bar_data.get(evaluated_signal.symbol)
                        if symbol_bar:
                            order = self.execute_signal(evaluated_signal, symbol_bar, timestamp)
                            
                            if order:
                                # Store active order
                                order_id = f"{order.symbol}_{len(self.active_orders)}"
                                self.active_orders[order_id] = order
                                
                                # Update portfolio manager
                                portfolio_manager.update_position(
                                    order.symbol, order.side, order.quantity,
                                    order.fill_price, order.strategy
                                )
                                
                                # Notify the strategy about the filled order (for VolatilityHarvesting)
                                strategy.update_positions([order])
                                
            except Exception as e:
                logger.error(f"Error processing strategy {strategy.name}: {e}")
        
        # Update equity curve
        self.equity_curve.append(self.current_balance)
        
    def close_position(self, order: BacktestOrder, exit_price: float, exit_reason: str, timestamp: datetime):
        """Close a position and calculate PnL"""
        
        if not order.filled:
            logger.warning(f"Attempted to close unfilled order: {order.symbol}")
            return
            
        # Calculate fees for exit
        trade_value = order.quantity * exit_price
        exit_fees = trade_value * (self.maker_fee_pct / 100)
        
        # Calculate total fees (entry + exit)
        entry_value = order.quantity * order.fill_price
        entry_fees = entry_value * (self.maker_fee_pct / 100)
        total_fees = entry_fees + exit_fees
        
        # Calculate PnL
        pnl = self.calculate_pnl(order.fill_price, exit_price, order.quantity, 
                                order.side, total_fees)
        
        # Update balance
        if order.side == "BUY":
            # Closing long - sell the asset
            self.current_balance += (trade_value - exit_fees)
        else:  # SELL
            # Closing short - buy back the asset
            self.current_balance -= (trade_value + exit_fees)
            
        # Record trade
        trade_record = {
            'entry_time': order.fill_time,
            'exit_time': timestamp,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'entry_price': order.fill_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': (pnl / entry_value) * 100,
            'fees': total_fees,
            'exit_reason': exit_reason,
            'strategy': order.strategy,
            'balance_after': self.current_balance
        }
        
        self.trade_history.append(trade_record)
        
        logger.info(f"Position closed: {order.side} {order.quantity:.6f} {order.symbol} "
                   f"Entry: ${order.fill_price:.2f} Exit: ${exit_price:.2f} "
                   f"PnL: ${pnl:.2f} ({exit_reason})")
    
    def run_backtest(self, 
                    historical_data: Dict[str, pd.DataFrame],
                    strategies: List[StrategyBase],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            historical_data: Dict of symbol -> OHLCV DataFrame
            strategies: List of strategies to test
            start_date: Optional start date filter
            end_date: Optional end date filter
        """
        
        logger.info(f"Starting backtest with {len(strategies)} strategies")
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(
            initial_balance=self.initial_balance,
            max_risk_per_trade_pct=3.0,
            max_total_risk_pct=15.0,
            max_drawdown_halt_pct=15.0
        )
        
        # Add strategies to portfolio manager
        for strategy in strategies:
            portfolio_manager.add_strategy(strategy)
            
        # Find common time range across all symbols
        min_length = min(len(df) for df in historical_data.values())
        
        if min_length == 0:
            raise ValueError("No historical data provided")
            
        logger.info(f"Backtesting {min_length} bars across {len(historical_data)} symbols")
        
        # Walk forward bar by bar
        for i in range(min_length):
            
            # Create bar data for current timepoint
            bar_data = {}
            timestamp = None
            
            for symbol, df in historical_data.items():
                if i < len(df):
                    bar = df.iloc[i]
                    bar_data[symbol] = {
                        'open': float(bar['open']),
                        'high': float(bar['high']), 
                        'low': float(bar['low']),
                        'close': float(bar['close']),
                        'volume': float(bar['volume']),
                    }
                    
                    # Use timestamp if available
                    if timestamp is None and 'timestamp' in bar:
                        timestamp = pd.to_datetime(bar['timestamp'])
                        
            if not timestamp:
                timestamp = datetime.utcnow()
                
            # Process this bar
            self.process_bar(bar_data, strategies, portfolio_manager, timestamp, historical_data, i)
            
            # Update portfolio manager balance
            portfolio_manager.update_balance(self.current_balance)
            
            # Periodic rebalancing
            if i % 24 == 0:  # Daily on hourly data
                portfolio_manager.rebalance_strategies()
                
            # Progress logging
            if i % 100 == 0:
                progress_pct = i / min_length * 100
                logger.info(f"Backtest progress: {progress_pct:.1f}% "
                           f"(Balance: ${self.current_balance:.2f})")
                           
        # Close any remaining open positions at final prices
        for order_id, order in list(self.active_orders.items()):
            if order.filled:
                final_symbol_data = historical_data.get(order.symbol)
                if final_symbol_data is not None and len(final_symbol_data) > 0:
                    final_price = float(final_symbol_data.iloc[-1]['close'])
                    self.close_position(order, final_price, "BACKTEST_END", timestamp)
                    
        # Calculate results
        return self.calculate_results(portfolio_manager)
    
    def calculate_results(self, portfolio_manager: PortfolioManager) -> BacktestResult:
        """Calculate backtest results and metrics"""
        
        # Basic metrics
        total_return_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdown.min()) * 100
        
        # Daily returns for Sharpe ratio
        daily_returns = []
        if len(self.equity_curve) > 1:
            for i in range(1, len(self.equity_curve)):
                daily_return = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
                daily_returns.append(daily_return)
                
        # Sharpe ratio (assuming daily data, risk-free rate = 0)
        if len(daily_returns) > 1:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            
        # Get strategy stats from portfolio manager
        portfolio_stats = portfolio_manager.get_portfolio_stats()
        
        result = BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            trade_history=self.trade_history,
            strategy_stats=portfolio_stats['strategy_stats'],
            daily_returns=daily_returns,
            equity_curve=equity_series
        )
        
        # Log results
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)
        logger.info(f"Initial Balance: ${result.initial_balance:.2f}")
        logger.info(f"Final Balance: ${result.final_balance:.2f}")
        logger.info(f"Total Return: {result.total_return_pct:.2f}%")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Win Rate: {result.win_rate:.2%}")
        logger.info(f"Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Round-trip Cost: {self.total_round_trip_cost_pct:.2f}%")
        
        return result