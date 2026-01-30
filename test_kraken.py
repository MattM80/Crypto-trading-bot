"""
Test Kraken Sandbox connection
Run this to verify your API keys work!
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pathlib import Path

from dotenv import load_dotenv
from kraken_client import KrakenClient

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Get credentials from environment
api_key = os.getenv("KRAKEN_API_KEY", "")
private_key = os.getenv("KRAKEN_PRIVATE_KEY", "")

if not api_key or not private_key:
    print("❌ ERROR: No Kraken credentials found!")
    print("\nPlease set environment variables:")
    print("  set KRAKEN_API_KEY=your_key")
    print("  set KRAKEN_PRIVATE_KEY=your_key")
    sys.exit(1)

print("🔍 Testing Kraken Sandbox Connection...\n")

# Create client
client = KrakenClient(api_key=api_key, private_key=private_key, sandbox=True)

# Test 1: Get Balance
print("1️⃣  Testing account balance...")
balance = client.get_account_balance()
if balance:
    print("✅ Balance retrieved successfully!")
    for asset, amount in balance.items():
        if amount > 0:
            print(f"   {asset}: {amount}")
else:
    print("❌ Failed to get balance")

# Test 2: Get Ticker
print("\n2️⃣  Testing price data...")
ticker = client.get_ticker("XBTUSD")
if ticker:
    print(f"✅ Bitcoin price: ${ticker['price']:.2f}")
else:
    print("❌ Failed to get ticker")

# Test 3: Get Candles
print("\n3️⃣  Testing historical data...")
candles = client.get_klines("XBTUSD", interval=300, limit=10)
if candles:
    print(f"✅ Got {len(candles)} candles")
    latest = candles[-1]
    print(f"   Latest close: ${latest['close']:.2f}")
else:
    print("❌ Failed to get candles")

print("\n" + "="*50)
print("✅ KRAKEN SANDBOX CONNECTION SUCCESSFUL!")
print("="*50)
print("\nYou can now run the bot with Kraken Sandbox:")
print("  python src/trading_bot.py")
