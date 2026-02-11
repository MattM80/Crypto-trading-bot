"""
Configuration settings for the crypto trading bot.
"""
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ExchangeConfig:
    """Exchange connection settings"""
    name: str = "binance"
    testnet: bool = True  # Use testnet by default for safety
    api_key: str = ""
    api_secret: str = ""
    request_timeout: int = 30

@dataclass
class RiskManagement:
    """Risk management parameters"""
    max_position_size: float = 0.02  # Max 2% of portfolio per trade
    max_drawdown: float = 0.10  # 10% maximum drawdown
    stop_loss_percent: float = 0.015  # 1.5% stop loss (ATR will override when available)
    take_profit_percent: float = 0.04  # 4% take profit (ATR will override when available)
    max_open_positions: int = 5  # Max concurrent positions
    position_scaling: bool = True  # Scale positions based on volatility
    
@dataclass
class TradingStrategy:
    """Trading strategy parameters"""
    strategy_type: str = "adaptive"  # "adaptive", "trend_momentum", "scalp", "grid", "mean_reversion", "arbitrage"
    symbols: List[str] = None
    timeframe: str = "5m"  # Timeframe: 1m, 5m, 15m, 1h
    
    # Grid Trading
    grid_levels: int = 10  # Number of grid levels
    grid_range_percent: float = 0.05  # 5% range around current price
    
    # Mean Reversion
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Arbitrage
    min_profit_percent: float = 0.01  # 1% minimum profit for arbitrage

@dataclass
class BotConfig:
    """Main bot configuration"""
    exchange: ExchangeConfig = None
    risk_management: RiskManagement = None
    trading_strategy: TradingStrategy = None
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_dashboard: bool = True
    dashboard_port: int = 8000
    
    # Backtesting
    backtest_mode: bool = False
    backtest_start_date: str = "2024-01-01"
    backtest_end_date: str = "2024-12-31"
    backtest_starting_balance: float = 1000.0
    
    def __post_init__(self):
        if self.exchange is None:
            self.exchange = ExchangeConfig()
        if self.risk_management is None:
            self.risk_management = RiskManagement()
        if self.trading_strategy is None:
            self.trading_strategy = TradingStrategy(symbols=["BTCUSDT", "ETHUSDT"])

# Default configuration
DEFAULT_CONFIG = BotConfig()
