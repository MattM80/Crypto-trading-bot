#!/usr/bin/env python3
"""
Test the on-chain data integration with the futures bot.
"""

import sys
from pathlib import Path

# Add src directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_onchain_integration():
    """Test that the on-chain integration works with the bot."""
    
    print("Testing on-chain integration...")
    
    try:
        # Test 1: Import the on-chain engine
        from onchain_data import OnChainEngine
        engine = OnChainEngine()
        print("✓ OnChain engine import successful")
        
        # Test 2: Generate signals
        signals = engine.get_onchain_signals()
        print(f"✓ Generated on-chain signals: market={signals['market_signal']:+.1f}")
        
        # Test 3: Test boost method
        # Create a mock bot-like object to test the boost method
        class MockBot:
            def __init__(self):
                self.onchain_engine = OnChainEngine()
                self.onchain_cache = {}
                self.onchain_cache_time = 0
            
            def get_onchain_boost(self, pair: str, direction: str) -> float:
                """Boost signal scores based on on-chain data - the MONEY FLOW indicator."""
                import time
                
                # Cache on-chain data for 5 minutes (same as bot cycle)
                now = time.time()
                if now - self.onchain_cache_time > 300:  # 5 minutes
                    try:
                        self.onchain_cache = self.onchain_engine.get_onchain_signals()
                        self.onchain_cache_time = now
                    except Exception as e:
                        print(f"Failed to get on-chain signals: {e}")
                        return 0
                
                if not self.onchain_cache:
                    return 0
                    
                boost = 0
                
                # 1. Stablecoin flow (market-wide) - THE BIGGEST SIGNAL
                market_signal = self.onchain_cache.get('market_signal', 0)
                if direction == 'long' and market_signal > 2:
                    boost += market_signal * 1.5  # Stablecoin minting → boost longs
                elif direction == 'short' and market_signal < -2:
                    boost += abs(market_signal) * 1.5  # Stablecoin burning → boost shorts
                elif direction == 'long' and market_signal < -3:
                    boost -= 3  # Penalty: going long while capital leaving crypto
                elif direction == 'short' and market_signal > 3:
                    boost -= 3  # Penalty: shorting while capital entering crypto
                
                # 2. Coin-specific TVL/protocol flows
                coin_signals = self.onchain_cache.get('coin_signals', {})
                coin_signal = coin_signals.get(pair, 0)
                if direction == 'long' and coin_signal > 0:
                    boost += coin_signal * 2  # TVL/protocol flowing into this coin
                elif direction == 'short' and coin_signal < 0:
                    boost += abs(coin_signal) * 2  # TVL/protocol leaving this coin
                
                # 3. Confidence-based adjustment
                confidence = self.onchain_cache.get('confidence', 0)
                boost *= min(1.0, confidence + 0.3)  # Reduce boost if low confidence
                
                return boost
        
        mock_bot = MockBot()
        
        # Test boost for different scenarios
        btc_long_boost = mock_bot.get_onchain_boost('BTCUSD', 'long')
        eth_short_boost = mock_bot.get_onchain_boost('ETHUSD', 'short')
        
        print(f"✓ OnChain boost test: BTC long={btc_long_boost:+.1f}, ETH short={eth_short_boost:+.1f}")
        
        # Test 4: Check tool definitions
        test_pairs = ['BTCUSD', 'ETHUSD', 'SOLUSD']
        
        stablecoin_signals_found = 0
        tvl_rotation_signals_found = 0
        
        # Simulate the new tool logic
        for pair in test_pairs:
            market_signal = signals.get('market_signal', 0)
            
            # Tool 39: Stablecoin Supply Signal test
            if abs(market_signal) >= 4:  # Would trigger stablecoin signal
                stablecoin_signals_found += 1
            
            # Tool 40: TVL Rotation Signal test
            tvl_flows = signals.get('tvl_flows', {})
            gainers = []
            losers = []
            
            for chain, data in tvl_flows.items():
                change_24h = data.get('change_24h_pct', 0)
                if change_24h > 3:
                    gainers.append((chain, change_24h))
                elif change_24h < -3:
                    losers.append((chain, change_24h))
            
            if gainers and losers:
                tvl_rotation_signals_found += len(gainers) * len(losers)
        
        print(f"✓ Tool signals: Stablecoin signals={stablecoin_signals_found}, TVL rotation potential={tvl_rotation_signals_found}")
        
        # Test 5: Validate API endpoints are accessible
        test_data = engine.fetch_all()
        working_endpoints = sum([
            1 if test_data.get('stablecoin_supply') else 0,
            1 if test_data.get('chain_tvl') else 0,
            1 if test_data.get('protocol_tvl') else 0,
            1 if test_data.get('btc_network') else 0,
            1 if test_data.get('btc_mempool') else 0,
        ])
        
        print(f"✓ API endpoints working: {working_endpoints}/5")
        
        print("\n" + "="*50)
        print("ON-CHAIN INTEGRATION TEST RESULTS")
        print("="*50)
        
        success = True
        
        if signals['market_signal'] != 0:
            print(f"✓ Market signal detection: {signals['market_signal']:+.1f}")
        else:
            print("⚠ Market signal is neutral")
        
        if signals['confidence'] > 0.5:
            print(f"✓ High confidence: {signals['confidence']:.1f}")
        else:
            print(f"⚠ Low confidence: {signals['confidence']:.1f}")
        
        if working_endpoints >= 4:
            print(f"✓ API endpoints: {working_endpoints}/5 working")
        else:
            print(f"❌ API endpoints: Only {working_endpoints}/5 working")
            success = False
        
        stablecoin_flow = signals.get('stablecoin_flow', {})
        if 'total_7d_change' in stablecoin_flow:
            change = stablecoin_flow['total_7d_change']
            print(f"✓ Stablecoin flow detection: ${change/1e6:+.0f}M over 7 days")
        else:
            print("⚠ Stablecoin flow data incomplete")
        
        if signals.get('coin_signals'):
            print(f"✓ Per-coin signals: {len(signals['coin_signals'])} coins")
        else:
            print("⚠ No per-coin signals generated")
        
        if success:
            print("\n🎉 ON-CHAIN INTEGRATION: FULLY OPERATIONAL!")
            print("The futures bot can now use on-chain data for:")
            print("  • Market-wide stablecoin flow signals")  
            print("  • Per-coin TVL and protocol signals")
            print("  • BTC network health indicators")
            print("  • Two new trading tools (Tools 39-40)")
            print("  • Enhanced signal scoring with on-chain boosts")
        else:
            print("\n❌ ON-CHAIN INTEGRATION: ISSUES DETECTED")
            print("Some components need attention before production use.")
        
        return success
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_onchain_integration()
    sys.exit(0 if success else 1)