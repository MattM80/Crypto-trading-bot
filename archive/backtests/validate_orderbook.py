#!/usr/bin/env python3
"""
ORDERBOOK VALIDATION SCRIPT
Test the orderbook analysis engine, collect statistics, and validate signals.
Runs for 30 minutes to build correlation data.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
from typing import Dict, List

# Add src directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from orderbook_engine import OrderbookEngine
    from kraken_futures_client import KrakenFuturesClient
    from kraken_client import KrakenClient
except ImportError as e:
    logger.error(f"Failed to import: {e}")
    sys.exit(1)

# Test pairs (mix of futures and spot available)
TEST_PAIRS = [
    "XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "XRPUSD", 
    "DOTUSD", "LINKUSD", "LTCUSD", "AVAXUSD", "ATOMUSD",
    "UNIUSD", "NEARUSD", "AAVEUSD", "XLMUSD", "DOGEUSD"
]

def get_current_prices() -> Dict[str, float]:
    """Get current prices for all test pairs."""
    prices = {}
    
    # Try Kraken Futures first for supported pairs
    futures_client = KrakenFuturesClient(dry_run=True)
    spot_client = KrakenClient()
    
    for pair in TEST_PAIRS:
        try:
            # Try to get recent price data
            data = spot_client.get_ohlc_data(pair, interval=1, since=None)
            if data and len(data) > 0:
                prices[pair] = float(data[-1]['close'])
                logger.info(f"Got price for {pair}: ${prices[pair]:.4f}")
        except Exception as e:
            logger.warning(f"Failed to get price for {pair}: {e}")
    
    return prices


def analyze_imbalance_distribution(results: Dict[str, Dict]) -> Dict:
    """Analyze the distribution of orderbook imbalances."""
    imbalances = [data['imbalance'] for data in results.values() if 'imbalance' in data]
    
    if not imbalances:
        return {}
    
    stats = {
        'count': len(imbalances),
        'mean': np.mean(imbalances),
        'median': np.median(imbalances),
        'std': np.std(imbalances),
        'min': np.min(imbalances),
        'max': np.max(imbalances),
        'extreme_buy': sum(1 for x in imbalances if x > 2.0),
        'strong_buy': sum(1 for x in imbalances if 1.5 <= x <= 2.0),
        'neutral': sum(1 for x in imbalances if 0.5 <= x <= 1.5),
        'strong_sell': sum(1 for x in imbalances if 0.33 <= x < 0.5),
        'extreme_sell': sum(1 for x in imbalances if x < 0.33)
    }
    
    # Calculate percentages
    total = stats['count']
    if total > 0:
        for key in ['extreme_buy', 'strong_buy', 'neutral', 'strong_sell', 'extreme_sell']:
            stats[f'{key}_pct'] = stats[key] / total * 100
    
    return stats


def analyze_wall_distribution(results: Dict[str, Dict]) -> Dict:
    """Analyze liquidity walls across pairs."""
    all_walls = []
    pairs_with_walls = 0
    
    for pair, data in results.items():
        walls = data.get('walls', [])
        if walls:
            pairs_with_walls += 1
            all_walls.extend(walls)
    
    if not all_walls:
        return {'total_walls': 0, 'pairs_with_walls': 0}
    
    # Categorize by strength
    weak_walls = sum(1 for w in all_walls if w.strength < 7)
    medium_walls = sum(1 for w in all_walls if 7 <= w.strength < 15)
    strong_walls = sum(1 for w in all_walls if w.strength >= 15)
    
    # Categorize by side
    bid_walls = sum(1 for w in all_walls if w.side == 'bid')
    ask_walls = sum(1 for w in all_walls if w.side == 'ask')
    
    # Categorize by distance from price
    close_walls = sum(1 for w in all_walls if w.distance_pct < 1.0)
    medium_walls_dist = sum(1 for w in all_walls if 1.0 <= w.distance_pct < 5.0)
    far_walls = sum(1 for w in all_walls if w.distance_pct >= 5.0)
    
    return {
        'total_walls': len(all_walls),
        'pairs_with_walls': pairs_with_walls,
        'weak_walls': weak_walls,
        'medium_walls': medium_walls,
        'strong_walls': strong_walls,
        'bid_walls': bid_walls,
        'ask_walls': ask_walls,
        'close_walls': close_walls,
        'medium_walls_dist': medium_walls_dist,
        'far_walls': far_walls,
        'avg_strength': np.mean([w.strength for w in all_walls]),
        'avg_distance_pct': np.mean([w.distance_pct for w in all_walls])
    }


def analyze_spread_distribution(results: Dict[str, Dict]) -> Dict:
    """Analyze spread patterns across pairs."""
    spreads = [data['spread_pct'] for data in results.values() if 'spread_pct' in data and data['spread_pct'] > 0]
    
    if not spreads:
        return {}
    
    return {
        'count': len(spreads),
        'mean_spread_pct': np.mean(spreads),
        'median_spread_pct': np.median(spreads),
        'min_spread_pct': np.min(spreads),
        'max_spread_pct': np.max(spreads),
        'tight_spreads': sum(1 for x in spreads if x < 0.05),  # <0.05%
        'normal_spreads': sum(1 for x in spreads if 0.05 <= x < 0.20),
        'wide_spreads': sum(1 for x in spreads if x >= 0.20)
    }


def identify_strongest_signals(results: Dict[str, Dict]) -> List[Dict]:
    """Identify pairs with the strongest current orderbook signals."""
    signals = []
    
    for pair, data in results.items():
        if 'signal_score' in data and abs(data['signal_score']) >= 3:
            signals.append({
                'pair': pair,
                'signal': data.get('signal', 'neutral'),
                'score': data.get('signal_score', 0),
                'imbalance': data.get('imbalance', 1.0),
                'walls': len(data.get('walls', [])),
                'spread_pct': data.get('spread_pct', 0),
                'depth_momentum': data.get('depth_momentum', 0)
            })
    
    # Sort by absolute score (strongest signals first)
    signals.sort(key=lambda x: abs(x['score']), reverse=True)
    return signals


def track_price_changes(engine: OrderbookEngine, pairs: List[str], 
                       duration_minutes: int = 30) -> List[Dict]:
    """Track how imbalance changes relate to price movements."""
    tracking_data = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    logger.info(f"🔍 Starting {duration_minutes}-minute correlation tracking...")
    
    cycle = 0
    prices = get_current_prices()
    
    while time.time() < end_time and prices:
        cycle += 1
        cycle_start = time.time()
        
        logger.info(f"Correlation cycle {cycle}")
        
        # Get orderbook data
        results = engine.get_orderbook_signals(pairs, {'pair': {'price': price} for pair, price in prices.items()})
        
        # Get new prices
        new_prices = get_current_prices()
        
        # Record imbalance vs price change correlations
        for pair in pairs:
            if (pair in results and pair in prices and pair in new_prices):
                old_price = prices[pair]
                new_price = new_prices[pair]
                price_change_pct = (new_price - old_price) / old_price * 100
                
                data = results[pair]
                tracking_data.append({
                    'timestamp': time.time(),
                    'pair': pair,
                    'cycle': cycle,
                    'price_change_pct': price_change_pct,
                    'imbalance': data.get('imbalance', 1.0),
                    'depth_momentum': data.get('depth_momentum', 0),
                    'signal_score': data.get('signal_score', 0),
                    'walls': len(data.get('walls', [])),
                    'spread_pct': data.get('spread_pct', 0)
                })
        
        prices = new_prices  # Update for next cycle
        
        # Wait for next cycle (aim for 5-minute intervals)
        cycle_time = time.time() - cycle_start
        sleep_time = max(0, 300 - cycle_time)  # 5 minutes
        
        if sleep_time > 0:
            logger.info(f"Cycle took {cycle_time:.1f}s, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
    
    return tracking_data


def analyze_correlations(tracking_data: List[Dict]) -> Dict:
    """Analyze correlations between orderbook signals and price movements."""
    if len(tracking_data) < 10:
        return {'error': 'Not enough data points'}
    
    df = pd.DataFrame(tracking_data)
    
    # Group by next price movement (5-minute forward looking)
    df = df.sort_values(['pair', 'timestamp'])
    df['next_price_change'] = df.groupby('pair')['price_change_pct'].shift(-1)
    
    # Remove last observation (no forward price)
    df = df.dropna(subset=['next_price_change'])
    
    if len(df) < 5:
        return {'error': 'Not enough valid data points'}
    
    results = {}
    
    # Overall correlation
    try:
        imbalance_correlation = df['imbalance'].corr(df['next_price_change'])
        momentum_correlation = df['depth_momentum'].corr(df['next_price_change'])
        score_correlation = df['signal_score'].corr(df['next_price_change'])
        
        results['correlations'] = {
            'imbalance_vs_price': imbalance_correlation,
            'momentum_vs_price': momentum_correlation,
            'signal_score_vs_price': score_correlation
        }
    except:
        results['correlations'] = {'error': 'Failed to calculate correlations'}
    
    # Analyze extreme imbalances
    extreme_high = df[df['imbalance'] > 2.0]
    extreme_low = df[df['imbalance'] < 0.5]
    
    results['extreme_analysis'] = {
        'high_imbalance_count': len(extreme_high),
        'high_imbalance_avg_next_move': extreme_high['next_price_change'].mean() if len(extreme_high) > 0 else 0,
        'low_imbalance_count': len(extreme_low),
        'low_imbalance_avg_next_move': extreme_low['next_price_change'].mean() if len(extreme_low) > 0 else 0,
    }
    
    # Strong signals analysis
    strong_buy = df[df['signal_score'] > 5]
    strong_sell = df[df['signal_score'] < -5]
    
    results['signal_analysis'] = {
        'strong_buy_count': len(strong_buy),
        'strong_buy_avg_next_move': strong_buy['next_price_change'].mean() if len(strong_buy) > 0 else 0,
        'strong_sell_count': len(strong_sell),
        'strong_sell_avg_next_move': strong_sell['next_price_change'].mean() if len(strong_sell) > 0 else 0,
    }
    
    return results


def main():
    """Main validation function."""
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")
    logger.info("🚀 ORDERBOOK VALIDATION STARTING")
    
    # Initialize engine
    engine = OrderbookEngine()
    
    # Get current prices
    logger.info("📊 Getting current market prices...")
    prices = get_current_prices()
    
    if not prices:
        logger.error("❌ Failed to get any prices, exiting")
        return
    
    available_pairs = list(prices.keys())
    logger.info(f"✅ Got prices for {len(available_pairs)} pairs: {', '.join(available_pairs)}")
    
    # Single snapshot analysis
    logger.info("📈 Analyzing current orderbook state...")
    market_data = {pair: {'price': price} for pair, price in prices.items()}
    current_results = engine.get_orderbook_signals(available_pairs[:10], market_data)
    
    if not current_results:
        logger.error("❌ Failed to get any orderbook data, exiting")
        return
    
    logger.info(f"✅ Got orderbook data for {len(current_results)} pairs")
    
    # Analyze current state
    imbalance_stats = analyze_imbalance_distribution(current_results)
    wall_stats = analyze_wall_distribution(current_results)
    spread_stats = analyze_spread_distribution(current_results)
    strongest_signals = identify_strongest_signals(current_results)
    
    # Print current analysis
    print("\n" + "=" * 80)
    print("📊 CURRENT ORDERBOOK ANALYSIS")
    print("=" * 80)
    
    if imbalance_stats:
        print(f"\n📈 IMBALANCE DISTRIBUTION ({imbalance_stats['count']} pairs):")
        print(f"  Mean: {imbalance_stats['mean']:.2f} | Median: {imbalance_stats['median']:.2f}")
        print(f"  Extreme Buy (>2.0): {imbalance_stats['extreme_buy']} ({imbalance_stats.get('extreme_buy_pct', 0):.1f}%)")
        print(f"  Strong Buy (1.5-2.0): {imbalance_stats['strong_buy']} ({imbalance_stats.get('strong_buy_pct', 0):.1f}%)")
        print(f"  Neutral (0.5-1.5): {imbalance_stats['neutral']} ({imbalance_stats.get('neutral_pct', 0):.1f}%)")
        print(f"  Strong Sell (0.33-0.5): {imbalance_stats['strong_sell']} ({imbalance_stats.get('strong_sell_pct', 0):.1f}%)")
        print(f"  Extreme Sell (<0.33): {imbalance_stats['extreme_sell']} ({imbalance_stats.get('extreme_sell_pct', 0):.1f}%)")
    
    if wall_stats['total_walls'] > 0:
        print(f"\n🧱 LIQUIDITY WALLS ({wall_stats['total_walls']} total):")
        print(f"  Pairs with walls: {wall_stats['pairs_with_walls']}")
        print(f"  Bid walls: {wall_stats['bid_walls']} | Ask walls: {wall_stats['ask_walls']}")
        print(f"  Strong walls (>15x): {wall_stats['strong_walls']}")
        print(f"  Close walls (<1%): {wall_stats['close_walls']}")
        print(f"  Avg strength: {wall_stats['avg_strength']:.1f}x | Avg distance: {wall_stats['avg_distance_pct']:.2f}%")
    
    if spread_stats:
        print(f"\n📏 SPREAD ANALYSIS ({spread_stats['count']} pairs):")
        print(f"  Mean spread: {spread_stats['mean_spread_pct']:.3f}%")
        print(f"  Tight spreads (<0.05%): {spread_stats['tight_spreads']}")
        print(f"  Normal spreads (0.05-0.20%): {spread_stats['normal_spreads']}")
        print(f"  Wide spreads (>0.20%): {spread_stats['wide_spreads']}")
    
    if strongest_signals:
        print(f"\n⚡ STRONGEST SIGNALS ({len(strongest_signals)}):")
        for signal in strongest_signals[:5]:
            print(f"  {signal['pair']}: {signal['signal']} (score {signal['score']:.1f}, "
                  f"imbalance {signal['imbalance']:.2f}, {signal['walls']} walls)")
    
    # Engine status
    status = engine.get_status()
    print(f"\n🔧 ENGINE STATUS:")
    print(f"  Cached pairs: {status['cached_pairs']}")
    print(f"  Requests this cycle: {status['requests_this_cycle']}")
    print(f"  Rate limit OK: {status['request_rate_ok']}")
    print(f"  Futures pairs supported: {status['supported_futures_pairs']}")
    print(f"  Spot pairs supported: {status['supported_spot_pairs']}")
    
    # Ask for correlation tracking
    print("\n" + "=" * 80)
    response = input("🕐 Run 30-minute correlation tracking? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # Track correlations
        tracking_data = track_price_changes(engine, available_pairs[:8], 30)  # Limit to 8 pairs
        
        if tracking_data:
            correlations = analyze_correlations(tracking_data)
            
            print("\n" + "=" * 80)
            print("🔍 CORRELATION ANALYSIS")
            print("=" * 80)
            
            if 'correlations' in correlations:
                corr = correlations['correlations']
                if 'error' not in corr:
                    print(f"\n📈 CORRELATIONS:")
                    print(f"  Imbalance vs Next Price Move: {corr.get('imbalance_vs_price', 0):.3f}")
                    print(f"  Depth Momentum vs Next Price Move: {corr.get('momentum_vs_price', 0):.3f}")
                    print(f"  Signal Score vs Next Price Move: {corr.get('signal_score_vs_price', 0):.3f}")
            
            if 'extreme_analysis' in correlations:
                ext = correlations['extreme_analysis']
                print(f"\n🔥 EXTREME IMBALANCE ANALYSIS:")
                print(f"  High imbalance (>2.0): {ext['high_imbalance_count']} samples, "
                      f"avg next move: {ext['high_imbalance_avg_next_move']:+.2f}%")
                print(f"  Low imbalance (<0.5): {ext['low_imbalance_count']} samples, "
                      f"avg next move: {ext['low_imbalance_avg_next_move']:+.2f}%")
            
            if 'signal_analysis' in correlations:
                sig = correlations['signal_analysis']
                print(f"\n⚡ STRONG SIGNAL ANALYSIS:")
                print(f"  Strong buy signals: {sig['strong_buy_count']} samples, "
                      f"avg next move: {sig['strong_buy_avg_next_move']:+.2f}%")
                print(f"  Strong sell signals: {sig['strong_sell_count']} samples, "
                      f"avg next move: {sig['strong_sell_avg_next_move']:+.2f}%")
            
            # Save tracking data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orderbook_tracking_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'tracking_data': tracking_data,
                    'analysis': correlations,
                    'current_analysis': {
                        'imbalance_stats': imbalance_stats,
                        'wall_stats': wall_stats,
                        'spread_stats': spread_stats,
                        'strongest_signals': strongest_signals
                    }
                }, f, indent=2, default=str)
            
            logger.info(f"💾 Saved tracking data to {filename}")
    
    print("\n✅ ORDERBOOK VALIDATION COMPLETE")


if __name__ == "__main__":
    main()