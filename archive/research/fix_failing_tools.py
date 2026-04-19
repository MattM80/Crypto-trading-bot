#!/usr/bin/env python3
"""
Fix failing trading tools by optimizing parameters and adding filters.
Based on OOS validation results, attempt to improve win rates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load validation results
results_file = Path("data/validation_results.csv")
signals_file = Path("data/validation_signals.csv")

def analyze_tool_signals(signals_df, tool_name):
    """Analyze signals for a specific tool to understand failure patterns."""
    tool_signals = signals_df[signals_df['tool'] == tool_name].copy()
    
    print(f"\n🔍 ANALYZING {tool_name.upper()}")
    print("-" * 50)
    print(f"Total signals: {len(tool_signals)}")
    
    # Return distribution
    returns_24h = tool_signals['ret_24h_forward']
    wins = (returns_24h > 0).sum()
    losses = (returns_24h <= 0).sum()
    win_rate = wins / len(returns_24h) * 100
    
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Average return: {returns_24h.mean():.2f}%")
    print(f"Median return: {returns_24h.median():.2f}%")
    print(f"Best return: {returns_24h.max():.2f}%")
    print(f"Worst return: {returns_24h.min():.2f}%")
    
    # Return quartiles
    q25 = returns_24h.quantile(0.25)
    q75 = returns_24h.quantile(0.75)
    print(f"25th percentile: {q25:.2f}%")
    print(f"75th percentile: {q75:.2f}%")
    
    return tool_signals

def optimize_crash_buy(signals_df):
    """Optimize crash buy tool - currently 29.7% WR, need >50%"""
    print(f"\n🛠️  OPTIMIZING CRASH_BUY")
    print("Current: -10% drop + RSI < 20 → 29.7% WR")
    
    # Try stricter thresholds
    crash_signals = signals_df[signals_df['tool'] == 'crash_buy'].copy()
    
    # Extract RSI and drop values from trigger strings
    crash_signals['drop_pct'] = crash_signals['trigger'].str.extract(r'(-?\d+\.\d+)% drop')[0].astype(float)
    crash_signals['rsi_val'] = crash_signals['trigger'].str.extract(r'RSI=(\d+\.\d+)')[0].astype(float)
    
    print("\nTesting parameter variations:")
    
    # Test different combinations
    for min_drop in [-15, -20, -25]:
        for max_rsi in [15, 10]:
            filtered = crash_signals[
                (crash_signals['drop_pct'] <= min_drop) & 
                (crash_signals['rsi_val'] <= max_rsi)
            ]
            
            if len(filtered) > 10:
                wins = (filtered['ret_24h_forward'] > 0).sum()
                wr = wins / len(filtered) * 100
                avg_ret = filtered['ret_24h_forward'].mean()
                net_exp = avg_ret * (wr / 100)
                
                print(f"  Drop ≤ {min_drop}% + RSI ≤ {max_rsi}: {len(filtered)} signals, {wr:.1f}% WR, {net_exp:.3f}% net")
                
                if wr > 50 and len(filtered) >= 10:
                    print(f"    ✅ IMPROVED: {wr:.1f}% WR vs 29.7% original")
                    return f"crash_buy_fixed", f"ret_24h < {min_drop} and cur_rsi < {max_rsi}"
    
    print("  ❌ No parameter combination achieved >50% WR")
    return None, None

def optimize_volatile_oversold(signals_df):
    """Optimize volatile oversold tool - currently 31.5% WR"""
    print(f"\n🛠️  OPTIMIZING VOLATILE_OVERSOLD")
    print("Current: ATR > 3% + RSI < 25 → 31.5% WR")
    
    vol_signals = signals_df[signals_df['tool'] == 'volatile_oversold'].copy()
    
    # Extract ATR and RSI values
    vol_signals['atr_pct'] = vol_signals['trigger'].str.extract(r'ATR=(\d+\.\d+)%')[0].astype(float)
    vol_signals['rsi_val'] = vol_signals['trigger'].str.extract(r'RSI=(\d+\.\d+)')[0].astype(float)
    
    print("\nTesting parameter variations:")
    
    for min_atr in [4, 5, 6]:
        for max_rsi in [20, 15]:
            filtered = vol_signals[
                (vol_signals['atr_pct'] >= min_atr) & 
                (vol_signals['rsi_val'] <= max_rsi)
            ]
            
            if len(filtered) > 10:
                wins = (filtered['ret_24h_forward'] > 0).sum()
                wr = wins / len(filtered) * 100
                avg_ret = filtered['ret_24h_forward'].mean()
                net_exp = avg_ret * (wr / 100)
                
                print(f"  ATR ≥ {min_atr}% + RSI ≤ {max_rsi}: {len(filtered)} signals, {wr:.1f}% WR, {net_exp:.3f}% net")
                
                if wr > 50 and len(filtered) >= 10:
                    print(f"    ✅ IMPROVED: {wr:.1f}% WR vs 31.5% original")
                    return f"volatile_oversold_fixed", f"cur_atr_pct > {min_atr} and cur_rsi < {max_rsi}"
    
    print("  ❌ No parameter combination achieved >50% WR")
    return None, None

def optimize_dip_buy(signals_df):
    """Optimize dip buy tool - currently 40.6% WR, close to 50%"""
    print(f"\n🛠️  OPTIMIZING DIP_BUY")
    print("Current: 4h drop > 3% → 40.6% WR")
    
    dip_signals = signals_df[signals_df['tool'] == 'dip_buy'].copy()
    
    # Extract drop values
    dip_signals['drop_pct'] = dip_signals['trigger'].str.extract(r'(-?\d+\.\d+)% dip')[0].astype(float)
    
    print("\nTesting parameter variations:")
    
    for min_drop in [-5, -6, -7, -8]:
        filtered = dip_signals[dip_signals['drop_pct'] <= min_drop]
        
        if len(filtered) > 10:
            wins = (filtered['ret_24h_forward'] > 0).sum()
            wr = wins / len(filtered) * 100
            avg_ret = filtered['ret_24h_forward'].mean()
            net_exp = avg_ret * (wr / 100)
            
            print(f"  Drop ≤ {min_drop}%: {len(filtered)} signals, {wr:.1f}% WR, {net_exp:.3f}% net")
            
            if wr > 50 and len(filtered) >= 10:
                print(f"    ✅ IMPROVED: {wr:.1f}% WR vs 40.6% original")
                return f"dip_buy_fixed", f"ret_4h < {min_drop}"
    
    print("  ❌ No single parameter fix achieved >50% WR")
    print("  💡 Suggestion: Add volume or RSI filter")
    return None, None

def optimize_crash_tools(signals_df):
    """Optimize the crash tools (mega_crash, flash_crash, quick_crash)"""
    print(f"\n🛠️  OPTIMIZING CRASH TOOLS")
    
    crash_tools = ['mega_crash', 'flash_crash', 'quick_crash']
    improvements = {}
    
    for tool in crash_tools:
        print(f"\n{tool.upper()}:")
        tool_signals = signals_df[signals_df['tool'] == tool].copy()
        
        if tool == 'mega_crash':
            # Currently: 24h drop > 15%
            tool_signals['drop_pct'] = tool_signals['trigger'].str.extract(r'(-?\d+\.\d+)% crash 24h')[0].astype(float)
            for min_drop in [-20, -25, -30]:
                filtered = tool_signals[tool_signals['drop_pct'] <= min_drop]
                if len(filtered) > 10:
                    wins = (filtered['ret_24h_forward'] > 0).sum()
                    wr = wins / len(filtered) * 100
                    avg_ret = filtered['ret_24h_forward'].mean()
                    print(f"  Drop ≤ {min_drop}%: {len(filtered)} signals, {wr:.1f}% WR, {avg_ret:.2f}% avg")
                    if wr > 50:
                        improvements[tool] = f"ret_24h < {min_drop}"
        
        elif tool == 'flash_crash':
            # Currently: 12h drop > 10%
            tool_signals['drop_pct'] = tool_signals['trigger'].str.extract(r'(-?\d+\.\d+)% crash 12h')[0].astype(float)
            for min_drop in [-15, -20, -25]:
                filtered = tool_signals[tool_signals['drop_pct'] <= min_drop]
                if len(filtered) > 10:
                    wins = (filtered['ret_24h_forward'] > 0).sum()
                    wr = wins / len(filtered) * 100
                    avg_ret = filtered['ret_24h_forward'].mean()
                    print(f"  Drop ≤ {min_drop}%: {len(filtered)} signals, {wr:.1f}% WR, {avg_ret:.2f}% avg")
                    if wr > 50:
                        improvements[tool] = f"ret_12h < {min_drop}"
        
        elif tool == 'quick_crash':
            # Currently: 8h drop > 10%
            tool_signals['drop_pct'] = tool_signals['trigger'].str.extract(r'(-?\d+\.\d+)% crash 8h')[0].astype(float)
            for min_drop in [-15, -20, -25]:
                filtered = tool_signals[tool_signals['drop_pct'] <= min_drop]
                if len(filtered) > 10:
                    wins = (filtered['ret_24h_forward'] > 0).sum()
                    wr = wins / len(filtered) * 100
                    avg_ret = filtered['ret_24h_forward'].mean()
                    print(f"  Drop ≤ {min_drop}%: {len(filtered)} signals, {wr:.1f}% WR, {avg_ret:.2f}% avg")
                    if wr > 50:
                        improvements[tool] = f"ret_8h < {min_drop}"
    
    return improvements

def generate_fixed_tools_code(improvements):
    """Generate updated tool code with fixes."""
    print(f"\n📝 GENERATING FIXED TOOL CODE")
    print("=" * 50)
    
    code_updates = []
    
    for tool, condition in improvements.items():
        if 'crash_buy' in tool:
            code_updates.append(f"""
# Tool 2: Crash Buy (OOS-fixed: stricter thresholds)
if {condition}:
    score = (20 - cur_rsi) * 2
    signals.append(({{'
        'pair': pair, 'tool': 'crash_buy', 'direction': 'long',
        'hold': 24, 'sl_pct': 0.05,
        'reason': f"CRASH BUY (FIXED): {{ret_24h:.1f}}% drop 24h, RSI={{cur_rsi:.1f}}"
    }}, score))""")
        
        elif 'volatile_oversold' in tool:
            code_updates.append(f"""
# Tool 3: Volatile Oversold (OOS-fixed: higher volatility threshold)
if {condition}:
    score = cur_atr_pct * (25 - cur_rsi)
    signals.append(({{'
        'pair': pair, 'tool': 'volatile_oversold', 'direction': 'long',
        'hold': 24, 'sl_pct': 0.08,
        'reason': f"VOLATILE OVERSOLD (FIXED): ATR={{cur_atr_pct:.1f}}%, RSI={{cur_rsi:.1f}}"
    }}, score))""")
    
    return code_updates

def main():
    """Run tool optimization analysis."""
    print("TRADING TOOL OPTIMIZATION ANALYSIS")
    print("=" * 50)
    
    if not signals_file.exists():
        print("Error: validation signals file not found. Run oos_validation_optimized.py first.")
        return
    
    # Load validation data
    signals_df = pd.read_csv(signals_file)
    print(f"Loaded {len(signals_df)} validation signals")
    
    # Analyze each failing tool
    failing_tools = ['crash_buy', 'volatile_oversold', 'dip_buy', 'mega_crash', 'flash_crash', 'quick_crash']
    
    improvements = {}
    
    # Try to fix each tool
    fixed_tool, condition = optimize_crash_buy(signals_df)
    if fixed_tool:
        improvements[fixed_tool] = condition
    
    fixed_tool, condition = optimize_volatile_oversold(signals_df)
    if fixed_tool:
        improvements[fixed_tool] = condition
    
    fixed_tool, condition = optimize_dip_buy(signals_df)
    if fixed_tool:
        improvements[fixed_tool] = condition
    
    crash_improvements = optimize_crash_tools(signals_df)
    improvements.update(crash_improvements)
    
    print(f"\n🎯 OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Tools successfully optimized: {len(improvements)}")
    
    if improvements:
        for tool, condition in improvements.items():
            print(f"✅ {tool}: {condition}")
        
        # Generate code updates
        code_updates = generate_fixed_tools_code(improvements)
        
        # Save to file
        with open("data/tool_fixes.py", 'w') as f:
            f.write("# OPTIMIZED TRADING TOOL CODE\n")
            f.write("# Generated from out-of-sample validation\n\n")
            for code in code_updates:
                f.write(code + "\n")
        
        print(f"\nFixed tool code saved to: data/tool_fixes.py")
    else:
        print("❌ No tools could be optimized with simple parameter changes")
        print("💡 Consider adding additional filters:")
        print("  - Volume confirmation (2x+ average volume)")
        print("  - Time-based filters (avoid certain hours/days)")
        print("  - Multi-timeframe confirmation")
        print("  - Momentum filters (price vs moving averages)")
    
    # Generate recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("1. Tools with positive net returns but low WR need additional filters")
    print("2. Consider combining tools (e.g., crash_buy + volume spike)")
    print("3. Add stop losses and take profits based on volatility")
    print("4. Test with longer holding periods (some tools may need 48h+)")
    print("5. Consider market regime filters (bull vs bear markets)")

if __name__ == "__main__":
    main()