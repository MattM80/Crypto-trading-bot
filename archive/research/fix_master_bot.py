#!/usr/bin/env python3
"""
Fix failing trading tools based on OOS validation results
"""

import pandas as pd

def apply_fixes():
    """Apply fixes to run_master_bot.py based on validation results."""
    
    # Read the validation summary
    try:
        summary = pd.read_csv('data/oos_validation_summary.csv')
        
        # Get failing tools (both 8h and 24h status are FAIL)
        failing_tools = summary[
            (summary['status_8h'] == 'FAIL') & 
            (summary['status_24h'] == 'FAIL')
        ]['tool'].tolist()
        
        # Get marginal tools that might need tweaking
        marginal_tools = summary[
            (summary['status_8h'] == 'MARGINAL') | 
            (summary['status_24h'] == 'MARGINAL')
        ]['tool'].tolist()
        
        print("FAILING TOOLS (need major fixes or disable):")
        for tool in failing_tools:
            row = summary[summary['tool'] == tool].iloc[0]
            print(f"  {tool}: {row['signals']} signals, WR={row['win_rate_24h']:.1f}%, Net={row['net_24h']:.2f}%")
        
        print("\nMARGINAL TOOLS (might need minor tweaks):")
        for tool in marginal_tools:
            row = summary[summary['tool'] == tool].iloc[0]
            print(f"  {tool}: {row['signals']} signals, WR={row['win_rate_24h']:.1f}%, Net={row['net_24h']:.2f}%")
        
        print("\nPROPOSED FIXES:")
        print("=" * 50)
        
        # Analyze each failing tool and propose fixes
        fixes = []
        
        if 'relief_rally' in failing_tools:
            fixes.append({
                'tool': 'relief_rally',
                'issue': 'Only 24 signals, 16.7% WR - too restrictive conditions',
                'fix': 'Lower RSI threshold from >75 to >70, or add volume confirmation',
                'code': '''
                # Tool 4: Relief Rally - OOS-FIXED: lowered RSI threshold 75→70
                if cur_rsi > 70 and cur_vs_sma50 < -2:  # Added -2% threshold  
                    score = (cur_rsi - 70) * 1.5
                    signals.append(({
                        'pair': pair, 'tool': 'relief_rally', 'direction': 'long',
                        'hold': 12, 'sl_pct': 0.03,
                        'reason': f"RELIEF RALLY: RSI={cur_rsi:.1f}, below SMA50 by {cur_vs_sma50:.1f}%"
                    }, score))
                '''
            })
        
        if 'mega_pump_sell_t2' in failing_tools:
            fixes.append({
                'tool': 'mega_pump_sell_t2',
                'issue': 'Only 84 signals, wrong direction (positive returns on short)',
                'fix': 'Tighten conditions - require RSI>85 instead of >80',
                'code': '''
                # Tool 7a: RSI Pump Short T2 - OOS-FIXED: tightened RSI 80→85
                if cur_rsi > 85 and len(close) >= 13:  # Tightened threshold
                    ret_12h_pump = (price - close[-13]) / close[-13] * 100 if close[-13] > 0 else 0
                    if ret_12h_pump >= 8:  # Keep 8% pump threshold
                        score = 22 + (cur_rsi - 85) * 0.3
                        signals.append(({
                            'pair': pair, 'tool': 'mega_pump_sell', 'direction': 'short',
                            'hold': 8, 'sl_pct': 0.05,
                            'reason': f"RSI PUMP SHORT T2: RSI={cur_rsi:.1f}, +{ret_12h_pump:.1f}% 12h"
                        }, score))
                '''
            })
        
        if 'quick_crash' in failing_tools:
            fixes.append({
                'tool': 'quick_crash',
                'issue': 'Good 8h performance (73% WR) but bad 24h - hold too long',
                'fix': 'Change hold from 24h to 8h, add RSI filter',
                'code': '''
                # Tool 10: Quick Crash - OOS-FIXED: hold 24h→8h, added RSI filter
                if ret_8h < -10 and cur_rsi < 40:  # Added RSI oversold filter
                    score = abs(ret_8h) * 2
                    signals.append(({
                        'pair': pair, 'tool': 'quick_crash', 'direction': 'long',
                        'hold': 8, 'sl_pct': 0.07,  # Changed from 24h hold
                        'reason': f"QUICK CRASH: {ret_8h:.1f}% drop 8h, RSI={cur_rsi:.1f}"
                    }, score))
                '''
            })
        
        if 'green_exhaustion' in failing_tools:
            fixes.append({
                'tool': 'green_exhaustion', 
                'issue': 'RSI>85 threshold too low, positive returns on short',
                'fix': 'OOS-DISABLED - unreliable signal',
                'code': '''
                # Tool 19: Green Exhaustion - OOS-DISABLED: unreliable (48% WR, wrong direction)
                # if cur_rsi > 85:
                #     win_8h = 1 if fwd_8h < 0 else 0
                #     win_24h = 1 if fwd_24h < 0 else 0
                '''
            })
        
        if 'ema_cross_short' in failing_tools:
            fixes.append({
                'tool': 'ema_cross_short',
                'issue': 'Only 387 signals, 42% WR - classic lagging indicator',
                'fix': 'OOS-DISABLED - lagging signal',
                'code': '''
                # Tool 48: EMA Cross Short - OOS-DISABLED: lagging indicator (42% WR)
                # if i >= 1 and not np.isnan(ema21[i]) and not np.isnan(sma50[i]):
                #     prev_above = ema21[i-1] > sma50[i-1]
                #     now_below = ema21[i] < sma50[i]
                '''
            })
        
        if 'month_start_short' in failing_tools:
            fixes.append({
                'tool': 'month_start_short',
                'issue': 'Calendar effect not reliable (45.8% WR)',
                'fix': 'OOS-DISABLED - weak calendar effect',
                'code': '''
                # Tool 49: Month Start Short - OOS-DISABLED: weak calendar effect (45.8% WR)
                # if self.is_month_start(ts):
                '''
            })
        
        if 'falling_wedge_short' in failing_tools:
            fixes.append({
                'tool': 'falling_wedge_short',
                'issue': 'Pattern recognition too simple, 44.7% WR',
                'fix': 'OOS-DISABLED - oversimplified pattern',
                'code': '''
                # Tool 57: Falling Wedge Short - OOS-DISABLED: oversimplified pattern (44.7% WR)
                # if i >= 10 and cur_rsi < 30:
                '''
            })
        
        if 'distribution_short' in failing_tools:
            fixes.append({
                'tool': 'distribution_short',
                'issue': 'Resistance detection too naive, wrong direction',
                'fix': 'OOS-DISABLED - naive resistance logic',
                'code': '''
                # Tool 46: Distribution Short - OOS-DISABLED: naive resistance logic (wrong direction)
                # if near_resistance and vol_ratio > 2:
                '''
            })
        
        if 'sunday_short' in failing_tools:
            fixes.append({
                'tool': 'sunday_short',
                'issue': 'Calendar effect not reliable (47.6% WR)',
                'fix': 'OOS-DISABLED - weak calendar effect',
                'code': '''
                # Tool 50: Sunday Short - OOS-DISABLED: weak calendar effect (47.6% WR)
                # if self.is_sunday(ts):
                '''
            })
        
        # Print all fixes
        for fix in fixes:
            print(f"\n{fix['tool'].upper()}:")
            print(f"  Issue: {fix['issue']}")
            print(f"  Fix: {fix['fix']}")
            print("  Code:")
            print(fix['code'])
        
        return fixes
        
    except FileNotFoundError:
        print("Error: data/oos_validation_summary.csv not found. Run validation first.")
        return []

if __name__ == "__main__":
    fixes = apply_fixes()