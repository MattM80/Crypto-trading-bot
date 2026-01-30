"""
Crypto Trading Bot - Low Risk Day Trading System

A professional-grade cryptocurrency trading bot with emphasis on risk management.
Supports multiple strategies: grid trading, mean reversion, and statistical arbitrage.

Key Features:
- Automatic position sizing based on risk
- Stop loss and take profit on every trade
- Maximum drawdown protection
- Real-time monitoring dashboard
- Backtesting framework
- Testnet support (safe for practice)

Example Usage:
    from config.config import BotConfig, DEFAULT_CONFIG
    from src.trading_bot import TradingBot
    import asyncio
    
    bot = TradingBot(DEFAULT_CONFIG)
    asyncio.run(bot.run_trading_loop(interval=60))

IMPORTANT: Always use testnet first! Never trade real money without testing!
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__title__ = "Crypto Trading Bot"

# Package imports
from src.trading_bot import TradingBot
from src.exchange_client import ExchangeClient
from src.strategies import GridTradingStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy
from src.risk_manager import RiskManager, Position
from config.config import BotConfig

__all__ = [
    'TradingBot',
    'ExchangeClient',
    'GridTradingStrategy',
    'MeanReversionStrategy',
    'StatisticalArbitrageStrategy',
    'RiskManager',
    'Position',
    'BotConfig',
]
