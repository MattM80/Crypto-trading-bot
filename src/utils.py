"""
Utility functions for the trading bot.
"""
from datetime import datetime, timedelta
from typing import Dict, List
import json

def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """Format value as percentage"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"

def calculate_roi(initial: float, final: float) -> float:
    """Calculate Return on Investment"""
    if initial == 0:
        return 0
    return ((final - initial) / initial) * 100

def calculate_profit_factor(winning_trades: float, losing_trades: float) -> float:
    """Calculate profit factor"""
    if losing_trades == 0:
        return 0
    return winning_trades / abs(losing_trades)

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio
    measures risk-adjusted returns
    """
    import numpy as np
    
    if len(returns) < 2:
        return 0
    
    returns_arr = np.array(returns)
    excess_returns = returns_arr - (risk_free_rate / 252)  # Daily risk-free rate
    
    if returns_arr.std() == 0:
        return 0
    
    return np.sqrt(252) * (excess_returns.mean() / returns_arr.std())

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio
    Similar to Sharpe but only penalizes downside volatility
    """
    import numpy as np
    
    if len(returns) < 2:
        return 0
    
    returns_arr = np.array(returns)
    excess_returns = returns_arr - (risk_free_rate / 252)
    
    # Downside volatility (only negative returns)
    downside = np.sqrt(np.mean(np.minimum(excess_returns, 0) ** 2))
    
    if downside == 0:
        return 0
    
    return np.sqrt(252) * (excess_returns.mean() / downside)

def calculate_max_consecutive_losses(trades: List[Dict]) -> int:
    """Calculate maximum consecutive losing trades"""
    if not trades:
        return 0
    
    max_streak = 0
    current_streak = 0
    
    for trade in trades:
        if trade['pnl'] < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def calculate_recovery_factor(total_pnl: float, max_drawdown: float) -> float:
    """Calculate recovery factor - higher is better"""
    if max_drawdown == 0:
        return 0
    return total_pnl / abs(max_drawdown)

def hours_since(timestamp: str) -> float:
    """Calculate hours since timestamp"""
    trade_time = datetime.fromisoformat(timestamp)
    hours = (datetime.now() - trade_time).total_seconds() / 3600
    return hours

def format_trade_summary(trades: List[Dict]) -> str:
    """Format trades into human-readable summary"""
    if not trades:
        return "No trades yet"
    
    import pandas as pd
    df = pd.DataFrame(trades)
    
    wins = len(df[df['pnl'] > 0])
    losses = len(df[df['pnl'] < 0])
    total_pnl = df['pnl'].sum()
    
    return f"""
Trade Summary:
─────────────
Total Trades:  {len(trades)}
Winning:       {wins}
Losing:        {losses}
Win Rate:      {(wins/len(trades)*100):.1f}%
Total P&L:     ${total_pnl:.2f}
Avg Win:       ${df[df['pnl'] > 0]['pnl'].mean():.2f}
Avg Loss:      ${df[df['pnl'] < 0]['pnl'].mean():.2f}
"""

class PerformanceTracker:
    """Track performance metrics over time"""
    
    def __init__(self):
        self.daily_returns = []
        self.daily_pnl = {}
        self.trades = []
    
    def add_trade(self, trade: Dict) -> None:
        """Record a trade"""
        self.trades.append(trade)
        
        # Record daily P&L
        date = datetime.now().date()
        if date not in self.daily_pnl:
            self.daily_pnl[date] = 0
        self.daily_pnl[date] += trade['pnl']
    
    def get_daily_return_percent(self, initial_balance: float) -> Dict:
        """Get daily return percentages"""
        daily_returns = {}
        for date, pnl in self.daily_pnl.items():
            daily_returns[date] = (pnl / initial_balance) * 100
        return daily_returns
    
    def get_cumulative_pnl(self) -> List[float]:
        """Get cumulative P&L over time"""
        cumulative = []
        total = 0
        for trade in self.trades:
            total += trade['pnl']
            cumulative.append(total)
        return cumulative
    
    def get_statistics(self, initial_balance: float) -> Dict:
        """Get comprehensive performance statistics"""
        if not self.trades:
            return {}
        
        import pandas as pd
        df = pd.DataFrame(self.trades)
        
        returns = df['pnl'].values / initial_balance
        
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'win_rate': (len(df[df['pnl'] > 0]) / len(self.trades)) * 100,
            'total_pnl': df['pnl'].sum(),
            'avg_trade': df['pnl'].mean(),
            'std_dev': df['pnl'].std(),
            'sharpe_ratio': calculate_sharpe_ratio(returns.tolist()),
            'sortino_ratio': calculate_sortino_ratio(returns.tolist()),
            'max_consecutive_losses': calculate_max_consecutive_losses(self.trades),
            'daily_pnl': self.daily_pnl
        }
        
        return stats
