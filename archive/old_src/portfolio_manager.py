"""
Portfolio Manager V2 - Orchestrates Multiple Strategies
======================================================

Manages allocation across strategies, handles position sizing, risk limits,
and coordinates between different trading approaches.

Key responsibilities:
1. Allocate capital across strategies based on performance
2. Calculate position sizes within risk limits
3. Handle strategy conflicts and prioritization
4. Track overall portfolio performance and risk
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from strategies_v2 import StrategyBase, Signal


@dataclass
class StrategyAllocation:
    """Track allocation and performance for each strategy"""
    name: str
    allocation_pct: float  # % of total capital allocated
    active_positions: Dict = field(default_factory=dict)
    total_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    last_trade_time: Optional[datetime] = None
    consecutive_losses: int = 0
    max_allocation_pct: float = 0.5  # Max % of capital per strategy


class PortfolioManager:
    """Manages multiple strategies and capital allocation"""
    
    def __init__(self, 
                 initial_balance: float = 300,
                 max_risk_per_trade_pct: float = 3.0,
                 max_total_risk_pct: float = 15.0,
                 max_drawdown_halt_pct: float = 15.0):
        
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Risk limits
        self.max_risk_per_trade_pct = max_risk_per_trade_pct  # 3% per trade
        self.max_total_risk_pct = max_total_risk_pct  # 15% total exposure
        self.max_drawdown_halt_pct = max_drawdown_halt_pct  # 15% max drawdown
        
        # Strategy tracking
        self.strategy_allocations: Dict[str, StrategyAllocation] = {}
        self.active_positions: Dict[str, Dict] = {}  # symbol -> position info
        self.total_risk_exposure = 0.0
        
        # Portfolio performance tracking
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.trade_history = []
        
    def add_strategy(self, strategy: StrategyBase, initial_allocation_pct: float = 0.33):
        """Add a strategy to the portfolio"""
        if strategy.name in self.strategy_allocations:
            logger.warning(f"Strategy {strategy.name} already exists")
            return
            
        self.strategy_allocations[strategy.name] = StrategyAllocation(
            name=strategy.name,
            allocation_pct=initial_allocation_pct,
            max_allocation_pct=0.5 if strategy.name == "VolatilityHarvesting" else 0.3
        )
        
        logger.info(f"Added strategy {strategy.name} with {initial_allocation_pct:.1%} allocation")
        
    def update_balance(self, new_balance: float):
        """Update current balance and track performance"""
        old_balance = self.current_balance
        self.current_balance = new_balance
        
        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            
        # Calculate drawdown
        current_drawdown = (self.peak_balance - new_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        logger.info(f"Balance updated: ${old_balance:.2f} -> ${new_balance:.2f} "
                   f"(Drawdown: {current_drawdown:.1%})")
                   
    def check_drawdown_halt(self) -> bool:
        """Check if we should halt trading due to drawdown"""
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        if current_drawdown >= self.max_drawdown_halt_pct / 100:
            logger.error(f"TRADING HALTED: Drawdown {current_drawdown:.1%} exceeds limit "
                        f"{self.max_drawdown_halt_pct:.1%}")
            return True
            
        return False
        
    def calculate_position_size(self, signal: Signal, strategy_name: str) -> float:
        """
        Calculate appropriate position size based on:
        1. Strategy allocation
        2. Risk per trade limits  
        3. Current exposure
        4. Account minimums
        """
        
        if self.check_drawdown_halt():
            return 0.0
            
        strategy_alloc = self.strategy_allocations.get(strategy_name)
        if not strategy_alloc:
            logger.error(f"Unknown strategy: {strategy_name}")
            return 0.0
            
        # Calculate available capital for this strategy
        strategy_capital = self.current_balance * strategy_alloc.allocation_pct
        
        # Different position sizing logic based on strategy type
        if strategy_name == "VolatilityHarvesting":
            # For rebalancing, use the exact amount needed
            return signal.quantity
            
        elif strategy_name == "CrossPairMeanReversion":
            # For pair trading, use fixed % of allocated capital
            pair_capital = strategy_capital * 0.5  # 50% of strategy allocation
            position_size = pair_capital / signal.price
            
        else:  # LiquidationCascade and others
            # Risk-based position sizing using stop loss
            if signal.stop_loss is None:
                # No stop loss - use fixed % of strategy capital
                position_value = strategy_capital * 0.1  # 10% of strategy allocation
                position_size = position_value / signal.price
            else:
                # Calculate position size based on risk per trade
                risk_per_trade = self.current_balance * (self.max_risk_per_trade_pct / 100)
                price_risk = abs(signal.price - signal.stop_loss)
                
                if price_risk <= 0:
                    logger.warning(f"Invalid stop loss for {signal.symbol}: price={signal.price}, sl={signal.stop_loss}")
                    return 0.0
                    
                position_size = risk_per_trade / price_risk
                
                # Cap by strategy allocation
                max_position_value = strategy_capital * 0.3  # Max 30% of strategy capital per trade
                max_position_size = max_position_value / signal.price
                position_size = min(position_size, max_position_size)
                
        # Apply minimum position size (Kraken minimums)
        min_position_value = self.get_min_position_value(signal.symbol)
        min_position_size = min_position_value / signal.price
        
        if position_size < min_position_size:
            logger.warning(f"Position size ${position_size * signal.price:.2f} below minimum "
                          f"${min_position_value:.2f} for {signal.symbol}")
            return 0.0
            
        # Check total exposure limits
        position_value = position_size * signal.price
        if self.total_risk_exposure + position_value > self.current_balance * (self.max_total_risk_pct / 100):
            logger.warning(f"Position would exceed total risk limit: "
                          f"${self.total_risk_exposure + position_value:.2f} > "
                          f"${self.current_balance * self.max_total_risk_pct / 100:.2f}")
            return 0.0
            
        logger.info(f"Position size for {signal.symbol}: {position_size:.6f} "
                   f"(${position_value:.2f}, {position_value/self.current_balance:.1%} of balance)")
                   
        return position_size
        
    def get_min_position_value(self, symbol: str) -> float:
        """Get minimum position value for a symbol (Kraken minimums)"""
        minimums = {
            'XBTUSD': 10.0,   # $10 minimum
            'ETHUSD': 10.0,   # $10 minimum  
            'SOLUSD': 10.0,   # $10 minimum
            'ADAUSD': 10.0,   # $10 minimum
            'DOTUSD': 10.0,   # $10 minimum
        }
        return minimums.get(symbol, 10.0)
        
    def evaluate_signal(self, signal: Signal, strategy_name: str) -> Optional[Signal]:
        """
        Evaluate and potentially modify a signal before execution.
        Returns None if signal should be rejected.
        """
        
        if self.check_drawdown_halt():
            return None
            
        # Calculate appropriate position size
        position_size = self.calculate_position_size(signal, strategy_name)
        if position_size <= 0:
            return None
            
        # Update signal with calculated position size
        signal.quantity = position_size
        
        # Check for conflicting signals from other strategies
        if self.has_conflicting_position(signal):
            logger.info(f"Conflicting position exists for {signal.symbol}, rejecting signal")
            return None
            
        return signal
        
    def has_conflicting_position(self, signal: Signal) -> bool:
        """Check if signal conflicts with existing positions"""
        
        existing_position = self.active_positions.get(signal.symbol)
        if not existing_position:
            return False
            
        # Check if trying to trade opposite direction
        existing_side = existing_position.get('side', 'NONE')
        
        if existing_side == 'BUY' and signal.action == 'SELL':
            return True
        elif existing_side == 'SELL' and signal.action == 'BUY':
            return True
            
        return False
        
    def record_trade(self, symbol: str, side: str, quantity: float, price: float, 
                    strategy_name: str, pnl: float = 0.0):
        """Record a completed trade"""
        
        trade_record = {
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'strategy': strategy_name,
            'pnl': pnl,
            'balance_after': self.current_balance
        }
        
        self.trade_history.append(trade_record)
        
        # Update strategy stats
        if strategy_name in self.strategy_allocations:
            strategy_alloc = self.strategy_allocations[strategy_name]
            strategy_alloc.total_pnl += pnl
            strategy_alloc.trade_count += 1
            strategy_alloc.last_trade_time = datetime.utcnow()
            
            if pnl > 0:
                strategy_alloc.win_count += 1
                strategy_alloc.consecutive_losses = 0
            else:
                strategy_alloc.consecutive_losses += 1
                
        # Update portfolio stats
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            
        logger.info(f"Trade recorded: {side} {quantity:.6f} {symbol} @ ${price:.2f} "
                   f"PnL: ${pnl:.2f} ({strategy_name})")
                   
    def update_position(self, symbol: str, side: str, quantity: float, price: float, 
                       strategy_name: str):
        """Update active position tracking"""
        
        if symbol not in self.active_positions:
            self.active_positions[symbol] = {}
            
        self.active_positions[symbol].update({
            'side': side,
            'quantity': quantity,
            'entry_price': price,
            'strategy': strategy_name,
            'timestamp': datetime.utcnow()
        })
        
        # Update risk exposure
        position_value = quantity * price
        if side == 'BUY':
            self.total_risk_exposure += position_value
        else:
            self.total_risk_exposure -= position_value
            
    def close_position(self, symbol: str, exit_price: float, pnl: float):
        """Close an active position"""
        
        if symbol not in self.active_positions:
            logger.warning(f"Attempted to close non-existent position: {symbol}")
            return
            
        position = self.active_positions[symbol]
        strategy_name = position.get('strategy', 'Unknown')
        
        # Record the trade
        self.record_trade(
            symbol=symbol,
            side=f"CLOSE_{position['side']}", 
            quantity=position['quantity'],
            price=exit_price,
            strategy_name=strategy_name,
            pnl=pnl
        )
        
        # Update risk exposure
        position_value = position['quantity'] * exit_price
        if position['side'] == 'BUY':
            self.total_risk_exposure -= position_value
        else:
            self.total_risk_exposure += position_value
            
        # Remove position
        del self.active_positions[symbol]
        
    def rebalance_strategies(self):
        """Adjust strategy allocations based on performance"""
        
        if len(self.strategy_allocations) == 0:
            return
            
        total_allocation = 0.0
        
        for name, strategy_alloc in self.strategy_allocations.items():
            
            # Reduce allocation for strategies with consecutive losses
            if strategy_alloc.consecutive_losses >= 3:
                penalty = min(0.1, strategy_alloc.consecutive_losses * 0.03)
                strategy_alloc.allocation_pct = max(0.1, strategy_alloc.allocation_pct - penalty)
                
            # Increase allocation for profitable strategies (within limits)
            elif strategy_alloc.trade_count > 5 and strategy_alloc.total_pnl > 0:
                win_rate = strategy_alloc.win_count / strategy_alloc.trade_count
                if win_rate > 0.6:
                    bonus = min(0.05, win_rate * 0.1)
                    strategy_alloc.allocation_pct = min(
                        strategy_alloc.max_allocation_pct, 
                        strategy_alloc.allocation_pct + bonus
                    )
                    
            total_allocation += strategy_alloc.allocation_pct
            
        # Normalize allocations to sum to 1.0
        if total_allocation > 0:
            for strategy_alloc in self.strategy_allocations.values():
                strategy_alloc.allocation_pct /= total_allocation
                
    def get_portfolio_stats(self) -> Dict:
        """Get current portfolio performance statistics"""
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        total_return_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        strategy_stats = {}
        for name, strategy_alloc in self.strategy_allocations.items():
            win_rate_strat = strategy_alloc.win_count / strategy_alloc.trade_count if strategy_alloc.trade_count > 0 else 0.0
            strategy_stats[name] = {
                'allocation_pct': strategy_alloc.allocation_pct,
                'total_pnl': strategy_alloc.total_pnl,
                'trade_count': strategy_alloc.trade_count,
                'win_rate': win_rate_strat,
                'consecutive_losses': strategy_alloc.consecutive_losses
            }
            
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance, 
            'total_return_pct': total_return_pct,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'max_drawdown_pct': self.max_drawdown * 100,
            'active_positions': len(self.active_positions),
            'total_risk_exposure': self.total_risk_exposure,
            'strategy_stats': strategy_stats
        }