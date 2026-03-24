#!/usr/bin/env python3
"""
Final Bull/Momentum Tools Test - Simplified and Complete

Tests all 45+ trading tools using vectorized pandas operations for speed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinalBullMomentumTester:
    def __init__(self):
        self.pairs = [
            "NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", 
            "SOLUSDT", "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", 
            "XRPUSDT", "ADAUSDT", "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"
        ]
        self.fee_pct = 0.0052
        self.oos_start = 4380
        self.data = {}
        self.results = []
    
    def load_data(self):
        """Load and preprocess all data"""
        print("Loading data...")
        for pair in self.pairs:
            file_path = f"data/binance_1h/{pair}_1h.csv"
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Pre-calculate all indicators
                self._add_indicators(df)
                
                self.data[pair] = df
                print(f"  {pair}: {len(df)} bars")
            except Exception as e:
                print(f"  ERROR {pair}: {e}")
        
        print(f"Loaded {len(self.data)} pairs")
    
    def _add_indicators(self, df):
        """Add all technical indicators to dataframe"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI using pandas TA-like calculation
        def rsi(series, period=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        # All indicators
        df['rsi7'] = rsi(close, 7)
        df['rsi14'] = rsi(close, 14)
        df['sma50'] = close.rolling(50, min_periods=1).mean()
        df['ema5'] = close.ewm(span=5).mean()
        df['ema13'] = close.ewm(span=13).mean()
        
        # Returns
        df['ret_1h'] = close.pct_change(1) * 100
        df['ret_4h'] = close.pct_change(5) * 100
        df['ret_8h'] = close.pct_change(9) * 100
        df['ret_12h'] = close.pct_change(13) * 100
        df['ret_24h'] = close.pct_change(25) * 100
        
        # Price vs SMA50
        df['vs_sma50'] = ((close - df['sma50']) / df['sma50'] * 100)
        
        # Volume ratio (20-bar average)
        df['vol_avg20'] = volume.rolling(20, min_periods=1).mean()
        df['vol_ratio'] = volume / df['vol_avg20']
        
        # Bollinger Bands
        bb_sma = close.rolling(20, min_periods=1).mean()
        bb_std = close.rolling(20, min_periods=1).std()
        df['bb_upper'] = bb_sma + (2 * bb_std)
        df['bb_lower'] = bb_sma - (2 * bb_std)
        
        # High/Low breakouts
        df['high_50'] = high.rolling(50, min_periods=1).max()
        df['high_30'] = high.rolling(30, min_periods=1).max()
        
        # Day of week and hour
        df['dow'] = df['timestamp'].dt.dayofweek  # 0=Monday, 3=Thursday, 6=Sunday
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
    
    def test_tool_vectorized(self, pair: str, tool_name: str, direction: str, condition_func):
        """Test a tool using vectorized pandas operations"""
        df = self.data[pair]
        
        # Generate signal mask for OOS period
        signals_mask = condition_func(df)
        oos_signals = signals_mask.iloc[self.oos_start:-24]  # Exclude last 24 bars
        
        signal_count = oos_signals.sum()
        
        if signal_count == 0:
            return {
                'tool': tool_name, 'pair': pair, 'direction': direction,
                'signals': 0, 'wr_8h': 0, 'wr_24h': 0, 
                'avg_return_8h': 0, 'avg_return_24h': 0, 'status': 'NO_SIGNALS'
            }
        
        # Calculate forward returns for signals
        signal_indices = oos_signals[oos_signals].index
        
        returns_8h = []
        returns_24h = []
        
        for idx in signal_indices:
            if idx + 24 >= len(df):
                continue
            
            entry_price = df.iloc[idx]['close']
            exit_8h_price = df.iloc[idx + 8]['close']
            exit_24h_price = df.iloc[idx + 24]['close']
            
            ret_8h = (exit_8h_price - entry_price) / entry_price
            ret_24h = (exit_24h_price - entry_price) / entry_price
            
            if direction == 'short':
                ret_8h = -ret_8h
                ret_24h = -ret_24h
            
            ret_8h_net = ret_8h - self.fee_pct
            ret_24h_net = ret_24h - self.fee_pct
            
            returns_8h.append(ret_8h_net)
            returns_24h.append(ret_24h_net)
        
        if not returns_8h:
            return {
                'tool': tool_name, 'pair': pair, 'direction': direction,
                'signals': 0, 'wr_8h': 0, 'wr_24h': 0, 
                'avg_return_8h': 0, 'avg_return_24h': 0, 'status': 'NO_SIGNALS'
            }
        
        wins_8h = sum(1 for r in returns_8h if r > 0)
        wins_24h = sum(1 for r in returns_24h if r > 0)
        
        wr_8h = wins_8h / len(returns_8h)
        wr_24h = wins_24h / len(returns_24h)
        avg_ret_8h = np.mean(returns_8h) * 100
        avg_ret_24h = np.mean(returns_24h) * 100
        
        status = "PASS" if wr_8h > 0.5 or wr_24h > 0.5 else "FAIL"
        
        return {
            'tool': tool_name, 'pair': pair, 'direction': direction,
            'signals': len(returns_8h), 'wr_8h': wr_8h, 'wr_24h': wr_24h,
            'avg_return_8h': avg_ret_8h, 'avg_return_24h': avg_ret_24h, 
            'status': status
        }
    
    def run_all_tests(self):
        """Run all tool tests"""
        print("Running all tool tests...")
        
        # Define tools as lambda functions for vectorized operations
        tools = [
            # BULL/GREED SHORT TOOLS (relaxed thresholds)
            ("mega_pump_sell_t1", "short", lambda df: (df['rsi7'] > 75) & (df['ret_12h'] >= 8)),
            ("mega_pump_sell_t2", "short", lambda df: (df['rsi7'] > 75) & (df['ret_12h'] >= 6)),
            ("rsi_pump_8h", "short", lambda df: (df['rsi7'] > 75) & (df['ret_8h'] >= 8)),
            ("rsi_pump_12h", "short", lambda df: (df['rsi7'] > 70) & (df['ret_12h'] >= 6)),
            ("greed_short_t2", "short", lambda df: (df['rsi7'] > 70) & (df['ret_8h'] > 4) & (df['close'] > df['sma50'])),
            ("sma50_ext_8", "short", lambda df: df['vs_sma50'] > 5),  # Relaxed 8→5
            ("sma50_ext_10", "short", lambda df: df['vs_sma50'] > 7),  # Relaxed 10→7
            ("sma50_ext_12", "short", lambda df: df['vs_sma50'] > 8),  # Relaxed 12→8
            ("sma50_ext_15", "short", lambda df: df['vs_sma50'] > 10), # Relaxed 15→10
            ("thursday_short", "short", lambda df: (df['dow'] == 3) & (df['close'] > df['sma50'])),
            ("sunday_short", "short", lambda df: (df['dow'] == 6) & (df['close'] > df['sma50'])),
            ("month_start_short", "short", lambda df: (df['day'] <= 3) & (df['close'] > df['sma50'])),
            ("late_us_short", "short", lambda df: (df['hour'] == 21) & (df['close'] > df['sma50'])),
            ("ema_cross_short", "short", lambda df: (df['ema5'] > df['ema13']) & (df['close'] > df['sma50'])),
            ("entropy_short_proxy", "short", lambda df: (df['vs_sma50'] > 5) & (df['ret_24h'].rolling(10).std() < 2)),  # Low volatility proxy for entropy
            
            # BREAKOUT/MOMENTUM LONG TOOLS  
            ("high_breakout_50", "long", lambda df: (df['close'] > df['high_50'].shift(1)) & (df['vol_ratio'] > 1.5)),  # Relaxed volume
            ("high_breakout_50_nv", "long", lambda df: df['close'] > df['high_50'].shift(1)),  # No volume filter
            ("high_breakout_30", "long", lambda df: (df['close'] > df['high_30'].shift(1)) & (df['vol_ratio'] > 1.5)),
            ("bb_above_long_t1", "long", lambda df: df['close'] > (df['bb_upper'] * 1.01)),  # Relaxed 1.02→1.01
            ("bb_above_long_t2", "long", lambda df: df['close'] > df['bb_upper']),  # Relaxed 1.01→1.00
            ("hurst_trend_proxy", "long", lambda df: (df['ret_4h'] > 1.5) & (df['ret_24h'] > 0)),  # Momentum proxy for Hurst
            ("thursday_long", "long", lambda df: (df['dow'] == 3) & (df['close'] < df['sma50'])),  # Test opposite
            ("sunday_long", "long", lambda df: (df['dow'] == 6) & (df['close'] < df['sma50'])),   # Test opposite
            
            # SIMPLE MEAN REVERSION TOOLS
            ("dip_buy_2pct", "long", lambda df: df['ret_4h'] < -2),
            ("dip_buy_3pct", "long", lambda df: df['ret_4h'] < -3),
            ("dip_buy_5pct", "long", lambda df: df['ret_8h'] < -5),
            ("crash_buy_8pct", "long", lambda df: df['ret_12h'] < -8),
            ("crash_buy_10pct", "long", lambda df: df['ret_24h'] < -10),
            
            # RSI MEAN REVERSION
            ("rsi_oversold_20", "long", lambda df: df['rsi14'] < 20),
            ("rsi_oversold_25", "long", lambda df: df['rsi14'] < 25),
            ("rsi_oversold_30", "long", lambda df: df['rsi14'] < 30),
            ("rsi_overbought_70", "short", lambda df: df['rsi14'] > 70),
            ("rsi_overbought_75", "short", lambda df: df['rsi14'] > 75),
            ("rsi_overbought_80", "short", lambda df: df['rsi14'] > 80),
            
            # VOLUME TOOLS
            ("volume_spike_2x", "long", lambda df: df['vol_ratio'] > 2),
            ("volume_spike_3x", "long", lambda df: df['vol_ratio'] > 3),
            ("volume_spike_5x", "long", lambda df: df['vol_ratio'] > 5),
            
            # WICK ABSORPTION  
            ("long_wick_lower", "long", lambda df: ((df['close'].combine(df['open'], min) - df['low']) / (df['high'] - df['low']) > 0.6)),
            ("long_wick_upper", "short", lambda df: ((df['high'] - df['close'].combine(df['open'], max)) / (df['high'] - df['low']) > 0.6)),
        ]
        
        # Test each tool on each pair
        total_tools = len(tools)
        for idx, (tool_name, direction, condition_func) in enumerate(tools, 1):
            print(f"\n[{idx}/{total_tools}] Testing {tool_name} ({direction.upper()})...")
            
            for pair in self.pairs:
                if pair not in self.data:
                    continue
                
                try:
                    result = self.test_tool_vectorized(pair, tool_name, direction, condition_func)
                    self.results.append(result)
                    
                    if result['signals'] > 0:
                        print(f"  {pair}: {result['signals']} signals, "
                              f"WR_8h={result['wr_8h']:.1%}, WR_24h={result['wr_24h']:.1%}, "
                              f"Ret_8h={result['avg_return_8h']:+.2f}%, Ret_24h={result['avg_return_24h']:+.2f}% "
                              f"[{result['status']}]")
                except Exception as e:
                    print(f"  {pair}: ERROR - {e}")
                    continue
    
    def add_cross_pair_tools(self):
        """Add cross-pair tools (BTC vs alts)"""
        if 'BTCUSDT' not in self.data:
            print("No BTCUSDT data for cross-pair tools")
            return
            
        print("\nTesting cross-pair tools...")
        btc_df = self.data['BTCUSDT']
        
        # Cross-pair tools
        for pair in self.pairs:
            if pair == 'BTCUSDT' or pair not in self.data:
                continue
            
            alt_df = self.data[pair]
            
            # Ensure same length  
            min_len = min(len(btc_df), len(alt_df))
            btc_rets = btc_df['ret_24h'].iloc[:min_len]
            alt_rets = alt_df['ret_24h'].iloc[:min_len]
            
            # Alt vs BTC spread
            spread_24h = alt_rets - btc_rets
            
            # Test spread reversion tools
            for thresh, name in [(5, "alt_btc_revert_5pct"), (8, "alt_btc_revert_8pct")]:
                condition_mask = spread_24h >= thresh
                
                # Create dummy df for testing  
                test_df = alt_df.iloc[:min_len].copy()
                test_df['spread_signal'] = condition_mask
                
                result = self.test_tool_vectorized(
                    pair, name, "short", 
                    lambda df: df['spread_signal']
                )
                self.results.append(result)
                
                if result['signals'] > 0:
                    print(f"  {pair} {name}: {result['signals']} signals, "
                          f"WR_8h={result['wr_8h']:.1%}, WR_24h={result['wr_24h']:.1%} [{result['status']}]")
    
    def generate_report(self):
        """Generate final report"""
        print("\nGenerating report...")
        
        # Summarize by tool
        tool_summary = {}
        for result in self.results:
            tool = result['tool']
            if tool not in tool_summary:
                tool_summary[tool] = {
                    'direction': result['direction'],
                    'total_signals': 0,
                    'pairs_with_signals': 0,
                    'weighted_wr_8h': 0,
                    'weighted_wr_24h': 0,
                    'weighted_ret_8h': 0,
                    'weighted_ret_24h': 0,
                }
            
            summary = tool_summary[tool]
            if result['signals'] > 0:
                summary['pairs_with_signals'] += 1
                summary['total_signals'] += result['signals']
                weight = result['signals']
                summary['weighted_wr_8h'] += result['wr_8h'] * weight
                summary['weighted_wr_24h'] += result['wr_24h'] * weight
                summary['weighted_ret_8h'] += result['avg_return_8h'] * weight
                summary['weighted_ret_24h'] += result['avg_return_24h'] * weight
        
        # Calculate averages
        for tool, summary in tool_summary.items():
            if summary['total_signals'] > 0:
                summary['avg_wr_8h'] = summary['weighted_wr_8h'] / summary['total_signals']
                summary['avg_wr_24h'] = summary['weighted_wr_24h'] / summary['total_signals']
                summary['avg_ret_8h'] = summary['weighted_ret_8h'] / summary['total_signals']
                summary['avg_ret_24h'] = summary['weighted_ret_24h'] / summary['total_signals']
            else:
                summary['avg_wr_8h'] = summary['avg_wr_24h'] = 0
                summary['avg_ret_8h'] = summary['avg_ret_24h'] = 0
        
        # Write report
        report_path = "data/bull_momentum_tools_1h_report.md"
        with open(report_path, 'w') as f:
            f.write("# BULL/GREED + BREAKOUT/MOMENTUM Tools - 1H Testing Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Configuration\n")
            f.write(f"- **Data:** Real 1h Binance candles, {len(self.pairs)} pairs, 8760 bars each\n")
            f.write(f"- **Out-of-sample:** Bars {self.oos_start}-8760 (recent {8760-self.oos_start} bars)\n")
            f.write(f"- **Fees:** -{self.fee_pct*100:.2f}% round-trip deducted from all returns\n")
            f.write(f"- **Forward returns:** +8 bars (8h) and +24 bars (24h)\n")
            f.write(f"- **Relaxed thresholds:** Many tools use lower thresholds to capture signals\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Tool | Dir | Signals | Pairs | WR_8h | WR_24h | Avg_Ret_8h | Avg_Ret_24h | Status |\n")
            f.write("|------|-----|---------|-------|-------|--------|------------|-------------|--------|\n")
            
            # Sort by total signals
            sorted_tools = sorted(tool_summary.items(), key=lambda x: x[1]['total_signals'], reverse=True)
            
            for tool, summary in sorted_tools:
                dir_short = summary['direction'].upper()[:1]
                signals = summary['total_signals']
                pairs = summary['pairs_with_signals']
                wr_8h = summary['avg_wr_8h']
                wr_24h = summary['avg_wr_24h'] 
                ret_8h = summary['avg_ret_8h']
                ret_24h = summary['avg_ret_24h']
                
                if signals == 0:
                    status = "NO_SIG"
                elif wr_8h > 0.5 or wr_24h > 0.5:
                    status = "PASS"
                else:
                    status = "FAIL"
                
                f.write(f"| {tool} | {dir_short} | {signals} | {pairs} | "
                       f"{wr_8h:.1%} | {wr_24h:.1%} | {ret_8h:+.2f}% | {ret_24h:+.2f}% | {status} |\n")
        
        print(f"Report written to: {report_path}")
        
        # Console summary
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        passed = sum(1 for _, s in sorted_tools if (s['avg_wr_8h'] > 0.5 or s['avg_wr_24h'] > 0.5) and s['total_signals'] > 0)
        failed = sum(1 for _, s in sorted_tools if (s['avg_wr_8h'] <= 0.5 and s['avg_wr_24h'] <= 0.5) and s['total_signals'] > 0)
        no_signals = sum(1 for _, s in sorted_tools if s['total_signals'] == 0)
        
        print(f"Tools PASSED: {passed}")
        print(f"Tools FAILED: {failed}")  
        print(f"Tools NO_SIGNALS: {no_signals}")
        print(f"Total tested: {len(sorted_tools)}")
        
        # Show best tools
        if passed > 0:
            print(f"\nTOP PERFORMING TOOLS:")
            for tool, summary in sorted_tools:
                if (summary['avg_wr_8h'] > 0.5 or summary['avg_wr_24h'] > 0.5) and summary['total_signals'] > 0:
                    print(f"  {tool} ({summary['direction'].upper()}): {summary['total_signals']} signals, "
                          f"WR_8h={summary['avg_wr_8h']:.1%}, WR_24h={summary['avg_wr_24h']:.1%}, "
                          f"Ret_8h={summary['avg_ret_8h']:+.2f}%, Ret_24h={summary['avg_ret_24h']:+.2f}%")
                    
                    # Show top pairs for this tool
                    tool_pairs = [(r['pair'], r['signals'], r['wr_8h'], r['wr_24h']) 
                                 for r in self.results 
                                 if r['tool'] == tool and r['signals'] > 0]
                    tool_pairs.sort(key=lambda x: x[1], reverse=True)
                    pair_summary = ", ".join([f"{p}({s})" for p, s, _, _ in tool_pairs[:3]])
                    print(f"    Best pairs: {pair_summary}")

def main():
    print("FINAL BULL/GREED + BREAKOUT/MOMENTUM TOOLS TESTING")
    print("="*70)
    print("Testing ALL tools with relaxed parameters for signal discovery")
    print("="*70)
    
    tester = FinalBullMomentumTester()
    
    # Load all data
    tester.load_data()
    
    # Run main tests
    tester.run_all_tests()
    
    # Add cross-pair tests
    tester.add_cross_pair_tools()
    
    # Generate final report
    tester.generate_report()
    
    print("\n🎯 TESTING COMPLETE! Check the report for full results.")

if __name__ == "__main__":
    main()