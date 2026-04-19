"""
Example usage of the trading bot - shows how to run different strategies.
"""
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pathlib import Path

from dotenv import load_dotenv
from config.config import BotConfig, ExchangeConfig, RiskManagement, TradingStrategy
from trading_bot import TradingBot
from backtest import BacktestEngine, print_backtest_report
from exchange_client import ExchangeClient
import asyncio

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def example_grid_trading():
    """Example: Run grid trading strategy"""
    print("\n=== Grid Trading Strategy (Kraken) ===\n")
    
    config = BotConfig()
    config.exchange = ExchangeConfig(testnet=True)
    config.trading_strategy = TradingStrategy(
        strategy_type="grid",
        symbols=["XBTUSD", "ETHUSD"],  # Kraken symbols
        timeframe="5m",
        grid_levels=10,
        grid_range_percent=0.05
    )
    config.risk_management = RiskManagement(
        max_position_size=0.02,
        max_drawdown=0.10,
        stop_loss_percent=0.02,
        take_profit_percent=0.05
    )
    
    bot = TradingBot(config, use_kraken=True)  # Use Kraken
    
    # Run for 5 minutes (demo)
    try:
        asyncio.run(bot.run_trading_loop(interval=60))
    except KeyboardInterrupt:
        print("\nBot stopped")

def example_mean_reversion():
    """Example: Run mean reversion strategy"""
    print("\n=== Mean Reversion Strategy ===\n")
    
    config = BotConfig()
    config.exchange = ExchangeConfig(testnet=True)
    config.trading_strategy = TradingStrategy(
        strategy_type="mean_reversion",
        symbols=["BTCUSDT"],
        timeframe="15m",
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        bollinger_period=20,
        bollinger_std=2.0
    )
    config.risk_management = RiskManagement(
        max_position_size=0.02,
        max_drawdown=0.10,
        stop_loss_percent=0.02,
        take_profit_percent=0.05
    )
    
    bot = TradingBot(config)
    
    try:
        asyncio.run(bot.run_trading_loop(interval=60))
    except KeyboardInterrupt:
        print("\nBot stopped")

def example_backtest():
    """Example: Backtest a strategy"""
    print("\n=== Backtesting Strategy ===\n")
    
    # Create sample data (in real use, load from exchange or CSV)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
    prices = 100 + (pd.Series(range(1000)).pct_change().fillna(0).cumsum() * 10)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': [1000000] * 1000
    })
    
    # Run backtest
    engine = BacktestEngine(
        strategy_type="mean_reversion",
        initial_balance=10000,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        bb_period=20,
        bb_std=2.0
    )
    
    stats = engine.backtest("BTCUSDT", data)
    print_backtest_report(stats, "mean_reversion", "BTCUSDT")

def example_check_account():
    """Example: Check exchange account balance"""
    print("\n=== Account Balance ===\n")
    
    client = ExchangeClient(testnet=True)
    balance = client.get_account_balance()
    
    print("Account Balances (Testnet):")
    for asset, balance_info in balance.items():
        if balance_info['total'] > 0:
            print(f"  {asset}: {balance_info['total']:.8f}")

def main():
    """Run examples"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          Crypto Trading Bot - Examples                       ║
    ║  Low-Risk Day Trading with Comprehensive Risk Management     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("""
    Choose an example:
    1. Grid Trading Strategy (Best for ranging markets)
    2. Mean Reversion Strategy (Best for volatile markets)
    3. Backtest Strategy (Test strategy on historical data)
    4. Check Account Balance (Verify testnet connection)
    5. Exit
    """)
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == '1':
        example_grid_trading()
    elif choice == '2':
        example_mean_reversion()
    elif choice == '3':
        example_backtest()
    elif choice == '4':
        example_check_account()
    elif choice == '5':
        print("Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
