#!/usr/bin/env python3
"""
Re-validate the FIXED trading tools
Test only the tools that were modified to see if fixes worked
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

class OOSReValidator:
    def __init__(self, data_dir="data/binance_real"):
        self.data_dir = data_dir
        self.pairs = []
        self.data_cache = {}
        self.results = []
        
        # Load all available pairs
        for f in os.listdir(data_dir):
            if f.endswith('_4h.csv'):
                pair = f.replace('_4h.csv', '')
                self.pairs.append(pair)
                df = pd.read_csv(f"{data_dir}/{f}")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data_cache[pair] = df
                
        print(f"Loaded {len(self.pairs)} pairs for re-validation")
        
    def calc_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI exactly like the master bot."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
            
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(delta))
        avg_loss = np.zeros(len(delta))
        
        # Initial averages
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Smoothed averages
        for i in range(period, len(delta)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad to match input length
        return np.concatenate([[50.0], rsi])
        
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
            
        sma = np.full(len(prices), np.nan)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            
        return sma
        
    def test_fixed_tools(self, pair):
        """Test only the tools that were fixed."""
        df = self.data_cache[pair]
        results = []
        
        # Pre-compute indicators for entire series
        close = df['close'].values
        high = df['high'].values  
        low = df['low'].values
        volume = df['volume'].values
        timestamps = df['timestamp'].values
        
        rsi7 = self.calc_rsi(close, 7)
        sma50 = self.calc_sma(close, 50)
        
        # Walk through data starting at bar 100 (enough lookback)
        start_idx = max(100, 50)
        
        for i in range(start_idx, len(close) - 6):  # Need 6 bars forward for 24h return
            price = close[i]
            cur_rsi = rsi7[i]
            cur_vs_sma50 = (price - sma50[i]) / sma50[i] * 100 if not np.isnan(sma50[i]) and sma50[i] > 0 else 0
            
            # Calculate returns (on 4h data, these are the correct lookbacks)
            ret_8h = (price - close[i-2]) / close[i-2] * 100 if i >= 2 and close[i-2] > 0 else 0
            ret_12h = (price - close[i-3]) / close[i-3] * 100 if i >= 3 and close[i-3] > 0 else 0
            
            # Forward returns for validation (8h = 2 bars, 24h = 6 bars)
            fwd_8h = (close[i+2] - price) / price * 100 if i+2 < len(close) else 0
            fwd_24h = (close[i+6] - price) / price * 100 if i+6 < len(close) else 0
            
            # ===== TEST FIXED TOOLS ONLY =====
            
            # Relief Rally - FIXED: lowered RSI threshold 75→70, added -2% SMA50 filter
            if cur_rsi > 70 and cur_vs_sma50 < -2:  # FIXED conditions
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'relief_rally_fixed', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"rsi={cur_rsi:.1f}, vs_sma50={cur_vs_sma50:.1f}"
                })
            
            # Mega Pump Sell T2 - FIXED: tightened RSI 80→85
            if cur_rsi > 85 and i >= 3:  # FIXED: 85 instead of 80
                ret_12h_pump = (price - close[i-3]) / close[i-3] * 100 if close[i-3] > 0 else 0
                if ret_12h_pump >= 8:
                    win_8h = 1 if fwd_8h < 0 else 0  # SHORT signal
                    win_24h = 1 if fwd_24h < 0 else 0
                    results.append({
                        'tool': 'mega_pump_sell_t2_fixed', 'pair': pair, 'direction': 'short',
                        'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                        'win_8h': win_8h, 'win_24h': win_24h,
                        'trigger': f"rsi={cur_rsi:.1f}, pump_12h={ret_12h_pump:.1f}"
                    })
            
            # Quick Crash - FIXED: hold 24h→8h, added RSI filter
            if ret_8h < -10 and cur_rsi < 40:  # FIXED: added RSI filter
                win_8h = 1 if fwd_8h > 0 else 0
                win_24h = 1 if fwd_24h > 0 else 0
                results.append({
                    'tool': 'quick_crash_fixed', 'pair': pair, 'direction': 'long',
                    'bar': i, 'price': price, 'fwd_8h': fwd_8h, 'fwd_24h': fwd_24h,
                    'win_8h': win_8h, 'win_24h': win_24h,
                    'trigger': f"ret_8h={ret_8h:.1f}, rsi={cur_rsi:.1f}"
                })
            
        return results
        
    def test_all_pairs(self):
        """Test fixed tools across all pairs."""
        all_results = []
        
        for pair in self.pairs:
            print(f"Re-testing fixed tools on {pair}...")
            pair_results = self.test_fixed_tools(pair)
            all_results.extend(pair_results)
            
        return all_results
        
    def analyze_results(self, results):
        """Analyze and summarize results by tool."""
        if not results:
            print("No signals found!")
            return []
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Group by tool
        summary = []
        for tool in df['tool'].unique():
            tool_df = df[df['tool'] == tool]
            
            signal_count = len(tool_df)
            win_rate_8h = tool_df['win_8h'].mean() * 100
            win_rate_24h = tool_df['win_24h'].mean() * 100
            avg_fwd_8h = tool_df['fwd_8h'].mean()
            avg_fwd_24h = tool_df['fwd_24h'].mean()
            
            # Net expected value (simplified)
            net_8h = avg_fwd_8h
            net_24h = avg_fwd_24h
            
            # Determine if tool passes criteria
            direction = tool_df['direction'].iloc[0]
            if direction == 'long':
                passes_8h = net_8h > 0 and win_rate_8h > 50 and signal_count >= 10
                passes_24h = net_24h > 0 and win_rate_24h > 50 and signal_count >= 10
            else:  # short
                passes_8h = net_8h < 0 and win_rate_8h > 45 and signal_count >= 10  # Note: negative return is good for shorts
                passes_24h = net_24h < 0 and win_rate_24h > 45 and signal_count >= 10
                
            status_8h = 'PASS' if passes_8h else ('MARGINAL' if (net_8h > 0 if direction == 'long' else net_8h < 0) and signal_count >= 5 else 'FAIL')
            status_24h = 'PASS' if passes_24h else ('MARGINAL' if (net_24h > 0 if direction == 'long' else net_24h < 0) and signal_count >= 5 else 'FAIL')
            
            summary.append({
                'tool': tool,
                'direction': direction,
                'signals': signal_count,
                'win_rate_8h': win_rate_8h,
                'win_rate_24h': win_rate_24h,
                'avg_fwd_8h': avg_fwd_8h,
                'avg_fwd_24h': avg_fwd_24h,
                'net_8h': net_8h,
                'net_24h': net_24h,
                'status_8h': status_8h,
                'status_24h': status_24h
            })
            
        # Sort by net 24h return (absolute value for shorts)
        summary_df = pd.DataFrame(summary)
        if not summary_df.empty:
            summary_df['sort_key'] = summary_df.apply(lambda x: x['net_24h'] if x['direction'] == 'long' else -x['net_24h'], axis=1)
            summary_df = summary_df.sort_values('sort_key', ascending=False)
        
        return summary_df
        
    def run_revalidation(self):
        """Run re-validation on fixed tools only."""
        print("Starting Re-validation of Fixed Tools...")
        print("=" * 60)
        
        # Test fixed tools
        results = self.test_all_pairs()
        
        print(f"\nFound {len(results)} total signals from FIXED tools")
        
        # Analyze results
        summary = self.analyze_results(results)
        
        if not summary.empty:
            # Print summary table
            print("\nFIXED TOOLS RE-VALIDATION RESULTS")
            print("=" * 100)
            print(f"{'Tool':<30} {'Dir':<5} {'Sigs':<6} {'WR_8h':<7} {'WR_24h':<8} {'Avg_8h':<8} {'Avg_24h':<9} {'Status_8h':<10} {'Status_24h':<10}")
            print("-" * 100)
            
            for _, row in summary.iterrows():
                print(f"{row['tool']:<30} {row['direction']:<5} {row['signals']:<6} "
                      f"{row['win_rate_8h']:<7.1f} {row['win_rate_24h']:<8.1f} "
                      f"{row['avg_fwd_8h']:<8.2f} {row['avg_fwd_24h']:<9.2f} "
                      f"{row['status_8h']:<10} {row['status_24h']:<10}")
        else:
            print("No results to display - no signals found for fixed tools")
        
        return results, summary

if __name__ == "__main__":
    revalidator = OOSReValidator()
    results, summary = revalidator.run_revalidation()