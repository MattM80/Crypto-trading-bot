"""
Simple test to check Kraken API connection
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pathlib import Path

from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from kraken_client import KrakenClient

def main():
    print("Testing Kraken API Connection\n")
    print("=" * 60)
    
    # Get API keys
    api_key = os.getenv("KRAKEN_API_KEY")
    private_key = os.getenv("KRAKEN_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("❌ No API credentials found in .env")
        return
    
    print(f"✅ API Key found: {api_key[:20]}...")
    print(f"✅ Private Key found: {private_key[:20]}...\n")
    
    client = KrakenClient(api_key, private_key)
    
    # Test 1: Get Server Time (public endpoint - no signature needed)
    print("Test 1: Getting server time (public endpoint)...")
    try:
        response = client.session.get(f"{client.base_url}/0/public/Time")
        result = response.json()
        print(f"Response: {result}")
        if result.get("result"):
            print("✅ Public endpoint works!\n")
        else:
            print(f"❌ Error: {result.get('error')}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 2: Get Account Balance (private endpoint)
    print("Test 2: Getting account balance (private endpoint)...")
    try:
        balance = client.get_account_balance()
        if balance:
            print(f"✅ Account balance retrieved!")
            for asset, amount in balance.items():
                print(f"   {asset}: {amount}")
        else:
            print("❌ Could not retrieve balance\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
