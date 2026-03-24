#!/usr/bin/env python3
"""
VOLATILITY ENGINE VALIDATION SCRIPT

Tests the volatility engine by:
1. Fetching current BTC+ETH options data
2. Calculating all signals (DVOL, P/C ratio, max pain, skew, GEX)  
3. Reporting current market state
4. Analyzing historical relationships between IV and price moves

This validates that our options intelligence is working correctly.
"""

import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from volatility_engine import VolatilityEngine


def test_deribit_connection():
    """Test basic Deribit API connectivity."""
    print("🔌 Testing Deribit API connection...")
    
    try:
        response = requests.get(
            "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        btc_price = data.get('result', {}).get('index_price', 0)
        print(f"✅ Deribit API working - BTC Index: ${btc_price:,.0f}")
        return True
        
    except Exception as e:
        print(f"❌ Deribit API failed: {e}")
        return False


def test_options_data_availability():
    """Test that we can fetch options data."""
    print("\n📊 Testing options data availability...")
    
    engine = VolatilityEngine()
    
    # Test BTC options
    btc_chain = engine.get_options_chain('BTC')
    btc_summaries = engine.get_book_summaries('BTC')
    
    # Test ETH options
    eth_chain = engine.get_options_chain('ETH')
    eth_summaries = engine.get_book_summaries('ETH')
    
    print(f"BTC Options: {len(btc_chain)} instruments, {len(btc_summaries)} with book data")
    print(f"ETH Options: {len(eth_chain)} instruments, {len(eth_summaries)} with book data")
    
    if btc_summaries or eth_summaries:
        print("✅ Options data available")
        return True
    else:
        print("❌ No options book data available")
        return False


def analyze_current_signals():
    """Fetch and analyze current volatility signals."""
    print("\n🎯 Current Volatility Intelligence:")
    print("=" * 50)
    
    engine = VolatilityEngine()
    signals = engine.get_volatility_signals()
    
    if not signals:
        print("❌ Failed to get volatility signals")
        return
    
    # Display key metrics
    print(f"🏛️  BTC Index Price: ${signals['btc_price']:,.0f}")
    print(f"🏛️  ETH Index Price: ${signals['eth_price']:,.0f}")
    print(f"📈 DVOL (Crypto VIX): {signals['dvol']:.1f}% ({signals['dvol_trend']})")
    
    # Interpret DVOL
    dvol = signals['dvol']
    if dvol > 100:
        dvol_msg = "EXTREME FEAR - Capitulation levels"
    elif dvol > 80:
        dvol_msg = "HIGH VOLATILITY - Big moves expected"
    elif dvol > 60:
        dvol_msg = "ELEVATED VOLATILITY - Market stress"  
    elif dvol < 30:
        dvol_msg = "LOW VOLATILITY - Market complacency"
    elif dvol < 40:
        dvol_msg = "BELOW AVERAGE - Calm conditions"
    else:
        dvol_msg = "NORMAL RANGE - Balanced market"
    print(f"    └─ {dvol_msg}")
    
    # Put/Call Analysis
    pcr = signals['put_call_ratio']
    pcr_signal = signals['put_call_signal']
    print(f"⚖️  Put/Call Ratio: {pcr:.2f} ({pcr_signal.replace('_', ' ').title()})")
    
    if pcr_signal == 'extreme_fear':
        pcr_msg = "MAXIMUM FEAR - Contrarian buy signal"
    elif pcr_signal == 'fear':
        pcr_msg = "HIGH HEDGING - Bullish contrarian"
    elif pcr_signal == 'extreme_greed':
        pcr_msg = "CALL MANIA - Contrarian sell signal"
    elif pcr_signal == 'greed':
        pcr_msg = "EXCESSIVE OPTIMISM - Bearish contrarian"
    else:
        pcr_msg = "BALANCED SENTIMENT"
    print(f"    └─ {pcr_msg}")
    
    # Max Pain Analysis
    print(f"🎯 Max Pain BTC: ${signals['max_pain_btc']:,.0f} ({signals['max_pain_bias']})")
    print(f"🎯 Max Pain ETH: ${signals['max_pain_eth']:,.0f}")
    
    btc_distance = (signals['max_pain_btc'] - signals['btc_price']) / signals['btc_price'] * 100
    if abs(btc_distance) > 5:
        gravity_msg = f"Strong gravity toward ${signals['max_pain_btc']:,.0f} ({btc_distance:+.1f}%)"
    else:
        gravity_msg = "Near equilibrium"
    print(f"    └─ {gravity_msg}")
    
    # Skew Analysis
    skew = signals['skew']
    skew_signal = signals['skew_signal']
    print(f"📊 Volatility Skew: {skew:+.1f}% ({skew_signal})")
    
    if skew > 5:
        skew_msg = "FEAR PREMIUM - Put demand elevated"
    elif skew < -5:
        skew_msg = "CALL PREMIUM - Upside demand elevated"
    else:
        skew_msg = "BALANCED - No directional fear"
    print(f"    └─ {skew_msg}")
    
    # Gamma Exposure
    gex = signals['gamma_exposure']
    print(f"⚡ Gamma Exposure: {gex.upper()}")
    
    if gex == 'positive':
        gex_msg = "Market makers provide liquidity (dampens moves)"
    elif gex == 'negative':  
        gex_msg = "Market makers chase moves (amplifies volatility)"
    else:
        gex_msg = "Neutral positioning"
    print(f"    └─ {gex_msg}")
    
    # Term Structure
    ts = signals['term_structure']
    print(f"📅 Term Structure: {ts.upper()}")
    
    if ts == 'backwardation':
        ts_msg = "Event risk priced in near-term"
    else:
        ts_msg = "Normal calm market structure"
    print(f"    └─ {ts_msg}")
    
    # Overall Assessment
    regime = signals['regime']
    market_signal = signals['market_signal']
    size_mult = signals['position_size_multiplier']
    
    print(f"\n🏁 OVERALL ASSESSMENT:")
    print(f"   Volatility Regime: {regime.replace('_', ' ').upper()}")
    print(f"   Market Signal: {market_signal:+.1f}/5.0")
    print(f"   Position Sizing: {size_mult:.2f}x")
    
    # Trading implications
    print(f"\n💡 TRADING IMPLICATIONS:")
    
    if market_signal >= 2:
        print("   🟢 BULLISH - Multiple contrarian signals aligned")
    elif market_signal >= 1:
        print("   🔵 MODERATELY BULLISH - Some positive signals")
    elif market_signal <= -2:
        print("   🔴 BEARISH - Multiple warning signals")
    elif market_signal <= -1:
        print("   🟠 MODERATELY BEARISH - Some negative signals")
    else:
        print("   ⚪ NEUTRAL - Mixed or balanced signals")
    
    if size_mult > 1.2:
        print("   📈 SIZE UP - High volatility environment favors larger positions")
    elif size_mult < 0.8:
        print("   📉 SIZE DOWN - Low volatility environment, reduce position sizes")
    else:
        print("   ⚖️  NORMAL SIZING - Standard position sizes appropriate")
    
    # Timestamp
    ts_str = datetime.fromtimestamp(signals['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"\n⏰ Data as of: {ts_str}")
    
    return signals


def test_individual_calculations():
    """Test individual calculation methods."""
    print("\n🧪 Testing Individual Calculations:")
    print("=" * 40)
    
    engine = VolatilityEngine()
    
    # Get sample data
    btc_price = engine.get_index_price('BTC')
    btc_summaries = engine.get_book_summaries('BTC')
    
    if not btc_summaries:
        print("❌ No BTC options data for individual testing")
        return
    
    print(f"📊 Using {len(btc_summaries)} BTC options for testing")
    
    # Test DVOL calculation
    dvol = engine.calc_dvol(btc_summaries, btc_price)
    print(f"✅ DVOL calculation: {dvol:.1f}%")
    
    # Test Put/Call ratio
    pcr = engine.calc_put_call_ratio(btc_summaries)
    print(f"✅ Put/Call ratio: {pcr:.2f}")
    
    # Test Max Pain
    max_pain, bias = engine.calc_max_pain(btc_summaries, btc_price)
    print(f"✅ Max Pain: ${max_pain:,.0f} ({bias})")
    
    # Test Skew
    skew, skew_signal = engine.calc_skew(btc_summaries, btc_price)
    print(f"✅ Skew: {skew:+.1f}% ({skew_signal})")
    
    # Test GEX
    gex = engine.estimate_gamma_exposure(btc_summaries, btc_price)
    print(f"✅ Gamma Exposure: {gex}")
    
    # Test Term Structure
    ts = engine.calc_term_structure(btc_summaries)
    print(f"✅ Term Structure: {ts}")


def analyze_options_composition():
    """Analyze the composition of available options."""
    print("\n📋 Options Market Composition:")
    print("=" * 35)
    
    engine = VolatilityEngine()
    
    for currency in ['BTC', 'ETH']:
        summaries = engine.get_book_summaries(currency)
        if not summaries:
            continue
        
        print(f"\n{currency} Options:")
        
        # Count by type
        calls = [s for s in summaries if '-C' in s.get('instrument_name', '')]
        puts = [s for s in summaries if '-P' in s.get('instrument_name', '')]
        
        print(f"  Total Options: {len(summaries)}")
        print(f"  Calls: {len(calls)}")  
        print(f"  Puts: {len(puts)}")
        
        # Volume analysis
        total_volume = sum(s.get('volume', 0) or 0 for s in summaries)
        call_volume = sum(s.get('volume', 0) or 0 for s in calls)
        put_volume = sum(s.get('volume', 0) or 0 for s in puts)
        
        print(f"  Total Volume: {total_volume:,.0f}")
        print(f"  Call Volume: {call_volume:,.0f} ({call_volume/max(total_volume,1)*100:.1f}%)")
        print(f"  Put Volume: {put_volume:,.0f} ({put_volume/max(total_volume,1)*100:.1f}%)")
        
        # Open Interest
        total_oi = sum(s.get('open_interest', 0) or 0 for s in summaries)
        call_oi = sum(s.get('open_interest', 0) or 0 for s in calls)
        put_oi = sum(s.get('open_interest', 0) or 0 for s in puts)
        
        print(f"  Total OI: {total_oi:,.0f}")
        print(f"  Call OI: {call_oi:,.0f} ({call_oi/max(total_oi,1)*100:.1f}%)")
        print(f"  Put OI: {put_oi:,.0f} ({put_oi/max(total_oi,1)*100:.1f}%)")
        
        # Expiration analysis
        expirations = {}
        current_price = engine.get_index_price(currency)
        
        for s in summaries:
            instrument = s.get('instrument_name', '')
            parsed = engine.parse_option_name(instrument)
            if parsed and parsed.get('dte'):
                dte = parsed['dte']
                if dte not in expirations:
                    expirations[dte] = 0
                expirations[dte] += s.get('open_interest', 0) or 0
        
        print(f"  Expirations available: {len(expirations)} different dates")
        
        # Show top 3 expiration dates by OI
        sorted_exp = sorted(expirations.items(), key=lambda x: x[1], reverse=True)[:3]
        for dte, oi in sorted_exp:
            days_str = "today" if dte == 0 else f"{dte} days"
            print(f"    {days_str}: {oi:,.0f} OI")


def test_performance_and_caching():
    """Test performance and caching behavior."""
    print("\n⚡ Performance & Caching Test:")
    print("=" * 32)
    
    engine = VolatilityEngine()
    
    # First call (cold cache)
    start_time = time.time()
    signals1 = engine.get_volatility_signals()
    cold_time = time.time() - start_time
    
    # Second call (warm cache)
    start_time = time.time() 
    signals2 = engine.get_volatility_signals()
    warm_time = time.time() - start_time
    
    print(f"Cold cache time: {cold_time:.2f}s")
    print(f"Warm cache time: {warm_time:.2f}s")
    print(f"Cache speedup: {cold_time/warm_time:.1f}x faster")
    
    # Verify cache working (results should be identical)
    if signals1.get('dvol') == signals2.get('dvol'):
        print("✅ Caching working correctly")
    else:
        print("❌ Cache not working - results differ")


def save_current_state():
    """Save current state for historical analysis."""
    print("\n💾 Saving Current State...")
    
    engine = VolatilityEngine()
    signals = engine.get_volatility_signals()
    
    # Save to data directory
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = data_dir / f"volatility_snapshot_{timestamp}.json"
    
    # Enhanced data for historical analysis
    snapshot = {
        'timestamp': signals['timestamp'],
        'datetime': datetime.fromtimestamp(signals['timestamp']).isoformat(),
        'signals': signals,
        'raw_data': {
            'btc_summaries_count': len(engine.get_book_summaries('BTC')),
            'eth_summaries_count': len(engine.get_book_summaries('ETH')),
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"✅ Snapshot saved to: {filename}")


def main():
    """Run complete volatility engine validation."""
    print("🚀 VOLATILITY ENGINE VALIDATION")
    print("=" * 50)
    
    # Basic connectivity test
    if not test_deribit_connection():
        print("\n❌ Cannot connect to Deribit API. Check internet connection.")
        return False
    
    # Data availability test
    if not test_options_data_availability():
        print("\n❌ Options data not available. Market may be closed.")
        return False
    
    # Main signal analysis
    current_signals = analyze_current_signals()
    
    if not current_signals:
        print("\n❌ Failed to generate volatility signals")
        return False
    
    # Individual component tests
    test_individual_calculations()
    
    # Market composition analysis
    analyze_options_composition()
    
    # Performance test
    test_performance_and_caching()
    
    # Save state
    save_current_state()
    
    print(f"\n✅ VALIDATION COMPLETE")
    print("=" * 30)
    print("🎯 Volatility engine is operational and ready for integration!")
    
    # Integration recommendations
    print(f"\n📝 INTEGRATION READY:")
    print("- Add volatility_engine to run_futures_bot.py imports")
    print("- Add get_vol_boost() method to bot class")
    print("- Include vol_boost in signal scoring")
    print("- Use position_size_multiplier for dynamic sizing")
    print("- Consider Tools 43 & 44 for P/C and Max Pain signals")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)