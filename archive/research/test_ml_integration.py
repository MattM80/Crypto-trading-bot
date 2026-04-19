#!/usr/bin/env python3
"""
Test ML Integration
Verify that the ML signal weighting is working correctly.
"""

import sys
from pathlib import Path

# Add src directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_signal_weighter import MLSignalWeighter, FEATURES
import numpy as np


def test_ml_weighter():
    """Test the ML signal weighter functionality."""
    print("🧠 Testing ML Signal Weighting Engine...")
    
    # Initialize the weighter (should load existing models)
    ml_weighter = MLSignalWeighter()
    
    print(f"Status: {ml_weighter.get_status_summary()}")
    
    # Test feature calculation
    test_features = {
        'rsi_7': 45.0,
        'atr_pct': 2.5,
        'ret_4h': -3.2,
        'ret_24h': -7.8,
        'vs_sma50': -2.1,
        'volume_ratio': 1.8,
        'fng': 35.0,
        'hour_sin': np.sin(2 * np.pi * 14 / 24),  # 2 PM
        'hour_cos': np.cos(2 * np.pi * 14 / 24),
        'dow_sin': np.sin(2 * np.pi * 2 / 7),  # Tuesday
        'dow_cos': np.cos(2 * np.pi * 2 / 7),
        'btc_ret_24h': -5.4,
        'stablecoin_signal': 1.2,
        'news_sentiment': -2.0,
        'ob_imbalance': 1.5,
        'funding_rate': -0.0002,
    }
    
    print("\n🔧 Testing score multipliers for key tools:")
    
    key_tools = ['crash_buy', 'volatile_oversold', 'mega_pump_sell_t1', 'quick_dip']
    
    for tool in key_tools:
        prob = ml_weighter.predict_win_probability(tool, test_features)
        multiplier = ml_weighter.get_score_multiplier(tool, test_features)
        
        print(f"{tool:20} | Win Prob: {prob:.3f} | Score Multiplier: {multiplier:.2f}x")
        
        # Test feature importance
        importance = ml_weighter.get_feature_importance(tool)
        top_features = list(importance.keys())[:3] if importance else []
        print(f"                     Top features: {', '.join(top_features)}")
    
    # Test recording a trade outcome
    print("\n📊 Recording test trade for crash_buy...")
    ml_weighter.record_trade('crash_buy', test_features, profitable=True)
    
    print("Updated status:", ml_weighter.get_status_summary())
    
    # Show model stats
    print("\n📈 Model Statistics:")
    model_stats = ml_weighter.get_model_stats()
    for tool, stats in sorted(model_stats.items()):
        if stats['samples'] > 0:
            active = "✅" if stats['is_active'] else "⏳"
            print(f"{active} {tool:20} | {stats['samples']:3d} samples | WR: {stats['win_rate']:.1%}")


if __name__ == "__main__":
    test_ml_weighter()