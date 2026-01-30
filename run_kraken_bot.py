"""
Run the trading bot with Kraken exchange.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

# Load .env from current directory explicitly
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

from config.config import BotConfig, ExchangeConfig, RiskManagement, TradingStrategy
from trading_bot import TradingBot
import asyncio

def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        print(f"⚠️  Invalid {name}={raw!r}; using default {default}")
        return float(default)

def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        print(f"⚠️  Invalid {name}={raw!r}; using default {default}")
        return int(default)

def _csv_env(name: str, default: str) -> list:
    raw = os.getenv(name, "").strip()
    if raw == "":
        raw = default
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]

def main():
    """Run bot with Kraken"""
    
    # Check for API keys
    api_key = os.getenv("KRAKEN_API_KEY", "").strip()
    private_key = os.getenv("KRAKEN_PRIVATE_KEY", "").strip()
    
    if not api_key or not private_key:
        print("❌ ERROR: Kraken API credentials not found in .env")
        print(f"\nChecked path: {env_path}")
        print(f"File exists: {os.path.exists(env_path)}")
        print("\nMake sure your .env file contains:")
        print("  KRAKEN_API_KEY=your_key")
        print("  KRAKEN_PRIVATE_KEY=your_key")
        return
    
    print(f"✅ Found Kraken credentials")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Private Key: {private_key[:20]}...\n")

    if os.getenv("ENABLE_LIVE_TRADING", "").strip().lower() not in {"1", "true", "yes", "y", "on"}:
        print("⚠️  Live trading is currently DISABLED (safe mode).")
        print("    To enable real orders, set: ENABLE_LIVE_TRADING=true in your .env")
        print("    You can also set MAX_SIGNALS_PER_CYCLE=1 to avoid rapid order bursts.\n")
    
    # Create bot configuration
    config = BotConfig()
    config.exchange = ExchangeConfig(name="kraken", testnet=True)

    strategy_type = (os.getenv("STRATEGY_TYPE", "grid") or "grid").strip().lower()
    symbols = _csv_env("KRAKEN_SYMBOLS", "XRPUSD")
    timeframe = (os.getenv("TIMEFRAME", "5m") or "5m").strip()
    grid_levels = _int_env("GRID_LEVELS", 8)
    grid_range = _float_env("GRID_RANGE_PERCENT", 0.02)

    config.trading_strategy = TradingStrategy(
        strategy_type=strategy_type,
        symbols=symbols,  # Kraken symbols
        timeframe=timeframe,
        grid_levels=grid_levels,
        grid_range_percent=grid_range,
    )
    config.risk_management = RiskManagement(
        max_position_size=0.02,
        max_drawdown=0.10,
        # Defaults tuned for more frequent exits.
        # You can override these in .env with STOP_LOSS_PERCENT and TAKE_PROFIT_PERCENT.
        stop_loss_percent=_float_env("STOP_LOSS_PERCENT", 0.01),
        take_profit_percent=_float_env("TAKE_PROFIT_PERCENT", 0.01),
        max_open_positions=_int_env("MAX_OPEN_POSITIONS", 2),
    )
    
    # Create bot with Kraken LIVE (Kraken has no sandbox for spot trading)
    # WARNING: This will trade with REAL MONEY
    # Recommendation: Test with backtest first, then use small amounts
    bot = TradingBot(config, use_kraken=True)
    
    print("🚀 Starting bot with Kraken LIVE (spot has no sandbox)...\n")
    
    # Run the bot
    try:
        asyncio.run(bot.run_trading_loop())
    except KeyboardInterrupt:
        print("\n\n⛔ Bot stopped by user")
    except Exception as e:
        print(f"\n\n❌ Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
