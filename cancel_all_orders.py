"""
Script to cancel ALL open orders on Kraken.
Useful for clearing "stuck" limit orders.
"""
import os
import sys
# Add src to path so we can import kraken_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

# Load .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

from kraken_client import KrakenClient

def main():
    print("--- Kraken Bulk Order Canceller ---")
    
    api_key = os.getenv("KRAKEN_API_KEY", "").strip()
    private_key = os.getenv("KRAKEN_PRIVATE_KEY", "").strip()

    if not api_key or not private_key:
        print("❌ Error: API keys not found in .env")
        return

    # Initialize client
    client = KrakenClient(api_key=api_key, private_key=private_key, sandbox=False)
    
    print("Fetching open orders...")
    response = client.get_open_orders()
    
    # Handle Kraken response structure
    open_orders = {}
    if isinstance(response, dict):
        open_orders = response.get('open', {})
    
    if not open_orders:
        print("✅ No open orders found.")
        return

    count = len(open_orders)
    print(f"Found {count} open orders.")
    
    for txid, details in open_orders.items():
        desc = details.get('descr', {}).get('order', txid)
        print(f"Cancelling {desc} ({txid})...")
        client.cancel_order(txid)
        
    print("-----------------------------------")
    print(f"✅ Cancelled {count} orders.")
    print("You can now restart the bot to place fresh orders.")

if __name__ == "__main__":
    main()
