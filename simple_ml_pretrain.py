#!/usr/bin/env python3
"""
Simple ML Pre-training Script
Create baseline ML models with some dummy data to get started.
The models will learn from real trades once the bot runs.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Add src directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_signal_weighter import MLSignalWeighter, FEATURES


def main():
    """Create simple baseline ML models."""
    print("🚀 Creating baseline ML models...")
    
    # Initialize ML weighter
    ml_weighter = MLSignalWeighter()
    
    # All 30 validated tools from VALIDATED_TOOLS.md
    VALIDATED_TOOLS = [
        # Crash/Bear Tools (LONG) - 15 tools
        'volatile_oversold', 'crash_buy', 'mega_crash', 'crash_neg_ac', 'blood_in_streets',
        'quick_crash', 'crash_mean_revert', 'vpin_dip', 'market_panic_70', 'flash_crash',
        'deep_dip_8h', 'entropy_dip', 'vpin_toxic', 'btc_alt_spread', 'quick_dip',
        
        # Bull/Greed Tools (SHORT) - 13 tools  
        'mega_pump_sell_t1', 'rsi_pump_8h', 'falling_wedge_short', 'greed_short_t2',
        'thursday_short', 'mega_pump_sell_t2', 'distribution_short', 'late_us_short',
        'rsi_pump_12h', 'ema_cross_short', 'rsi_pump_fat_tail', 'entropy_short',
        'alt_btc_revert_t3',
        
        # Neutral/Transition Tools - 2 tools
        'month_start_long', 'dip_buy_5pct'
    ]
    
    # Create some dummy training data for each tool
    total_trades = 0
    
    for tool_name in VALIDATED_TOOLS:
        print(f"Creating baseline for {tool_name}...")
        
        # Generate 10 dummy trades per tool to initialize models
        for _ in range(10):
            # Generate random features (in normalized range)
            features = {
                'rsi_7': np.random.uniform(20, 80),  # RSI
                'atr_pct': np.random.uniform(0.5, 5.0),  # Volatility %
                'ret_4h': np.random.uniform(-8, 8),  # 4h return %
                'ret_24h': np.random.uniform(-20, 20),  # 24h return %
                'vs_sma50': np.random.uniform(-15, 15),  # Distance from SMA50 %
                'volume_ratio': np.random.uniform(0.5, 3.0),  # Volume ratio
                'fng': np.random.uniform(10, 90),  # Fear & Greed Index
                'hour_sin': np.sin(2 * np.pi * np.random.uniform(0, 24) / 24),
                'hour_cos': np.cos(2 * np.pi * np.random.uniform(0, 24) / 24),
                'dow_sin': np.sin(2 * np.pi * np.random.uniform(0, 7) / 7),
                'dow_cos': np.cos(2 * np.pi * np.random.uniform(0, 7) / 7),
                'btc_ret_24h': np.random.uniform(-15, 15),  # BTC 24h return %
                'stablecoin_signal': np.random.uniform(-3, 3),  # Stablecoin flow
                'news_sentiment': np.random.uniform(-5, 5),  # News sentiment
                'ob_imbalance': np.random.uniform(0.3, 3.0),  # Orderbook imbalance
                'funding_rate': np.random.uniform(-0.001, 0.001),  # Funding rate
            }
            
            # Generate outcome - make crash tools slightly more likely to be profitable
            if 'crash' in tool_name or 'dip' in tool_name or 'blood' in tool_name:
                profitable = np.random.random() > 0.45  # 55% win rate
            elif 'pump' in tool_name or 'short' in tool_name:
                profitable = np.random.random() > 0.48  # 52% win rate  
            else:
                profitable = np.random.random() > 0.47  # 53% win rate
            
            # Record the trade
            ml_weighter.record_trade(tool_name, features, profitable)
            total_trades += 1
    
    # Save the models
    ml_weighter._save_models()
    
    print(f"✅ Created baseline ML models!")
    print(f"📊 Generated {total_trades} dummy trades across {len(VALIDATED_TOOLS)} tools")
    
    # Print model stats
    model_stats = ml_weighter.get_model_stats()
    active_models = sum(1 for stats in model_stats.values() if stats['is_active'])
    
    print(f"🧠 {ml_weighter.get_status_summary()}")
    print(f"Models ready to learn from real trades!")


if __name__ == "__main__":
    main()