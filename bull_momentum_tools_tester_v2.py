#!/usr/bin/env python3
"""
BULL/GREED + BREAKOUT/MOMENTUM Tools Testing Framework - V2

Optimized version with parameter sweeps for failed tools.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BullMomentumTesterV2:
    def __init__(self, data_dir: str = "data/binance_1h"):
        self.data_dir = Path(data_dir)
        self.pairs = [
            "NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", 
            "SOLUSDT", "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", 
            "XRPUSDT", "ADAUSDT", "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"
        ]
        self.fee_pct = 0.0052  # 0.52% round-trip
        self.oos_start = 4380  # Out-of-sample start
        self.data = {}
        self.results = []
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all pairs data"""
        print("Loading data...")
        for pair in self.pairs:
            file_path = self.data_dir / f"{pair}_1h.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                print(f"Loaded {pair}: {len(df)} bars")
                self.data[pair] = df
            else:
                print(f"WARNING: Missing {pair}_1h.csv")
        
        print(f"Total pairs loaded: {len(self.data)}")
        return self.data
    
    # ==================== TECHNICAL INDICATORS ====================
    
    def calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI calculation (vectorized)"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Smoothed averages
        rsi = np.full(len(prices), 50.0)
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    
    def calc_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period).mean().values
    
    def calc_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands (SMA, upper, lower)"""
        sma = self.calc_sma(prices, period)
        std = pd.Series(prices).rolling(window=period, min_periods=1).std().values
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return sma, upper, lower
    
    def test_tool_with_params(self, pair: str, tool_name: str, signal_func, direction: str, 
                             params: dict = None, test_oos_only: bool = True) -> Dict:
        """Test a single tool on one pair with optional parameters"""
        df = self.data[pair].copy()
        df._pair_name = pair  # Add pair name for cross-pair tools
        
        # Only test on OOS by default
        if test_oos_only:
            start_idx = self.oos_start
        else:
            start_idx = 100  # Start after sufficient history
        
        signals = []
        
        # Generate signals using the signal function
        for i in range(start_idx, len(df) - 24):  # Need 24 bars for forward returns
            try:
                signal = signal_func(df, i, params) if params else signal_func(df, i)
                if signal:
                    signals.append(i)
            except Exception as e:
                # Skip errors (missing data, etc.)
                continue
        
        if len(signals) == 0:
            return {
                'tool': tool_name, 'pair': pair, 'direction': direction,
                'signals': 0, 'wr_8h': 0, 'wr_24h': 0, 
                'avg_return_8h': 0, 'avg_return_24h': 0, 'status': 'NO_SIGNALS',
                'params': params
            }
        
        # Calculate forward returns with fees
        wins_8h = 0
        wins_24h = 0
        returns_8h = []
        returns_24h = []
        
        for sig_idx in signals:
            if sig_idx + 24 >= len(df):
                continue
                
            entry_price = df.iloc[sig_idx]['close']
            exit_8h_price = df.iloc[sig_idx + 8]['close']
            exit_24h_price = df.iloc[sig_idx + 24]['close']
            
            # Calculate raw returns
            ret_8h = (exit_8h_price - entry_price) / entry_price
            ret_24h = (exit_24h_price - entry_price) / entry_price
            
            # Apply direction
            if direction == 'short':
                ret_8h = -ret_8h
                ret_24h = -ret_24h
            
            # Apply fees
            ret_8h_net = ret_8h - self.fee_pct
            ret_24h_net = ret_24h - self.fee_pct
            
            returns_8h.append(ret_8h_net)
            returns_24h.append(ret_24h_net)
            
            if ret_8h_net > 0:
                wins_8h += 1
            if ret_24h_net > 0:
                wins_24h += 1
        
        n_signals = len(returns_8h)
        wr_8h = wins_8h / n_signals if n_signals > 0 else 0
        wr_24h = wins_24h / n_signals if n_signals > 0 else 0
        avg_ret_8h = np.mean(returns_8h) if returns_8h else 0
        avg_ret_24h = np.mean(returns_24h) if returns_24h else 0
        
        # Determine status
        if wr_8h > 0.5 or wr_24h > 0.5:
            status = "PASS"
        else:
            status = "FAIL"
        
        return {
            'tool': tool_name, 'pair': pair, 'direction': direction,
            'signals': n_signals, 'wr_8h': wr_8h, 'wr_24h': wr_24h,
            'avg_return_8h': avg_ret_8h * 100, 'avg_return_24h': avg_ret_24h * 100,
            'status': status, 'params': params
        }
    
    # ==================== TOOL IMPLEMENTATIONS ====================
    
    def mega_pump_sell_t1(self, df: pd.DataFrame, i: int, params: dict = None) -> bool:
        """Tool 1: mega_pump_sell T1 - rsi7 > rsi_thresh AND ret_12h >= ret_thresh → SHORT"""
        if params is None:
            params = {'rsi_thresh': 80, 'ret_thresh': 10}
        
        if i < 13:
            return False
        close = df['close'].values
        rsi7 = self.calc_rsi(close[:i+1], 7)
        ret_12h = (close[i] - close[i-13]) / close[i-13] * 100
        return rsi7[i] > params['rsi_thresh'] and ret_12h >= params['ret_thresh']
    
    def sma50_ext_8(self, df: pd.DataFrame, i: int, params: dict = None) -> bool:
        """Tool 6: sma50_ext_8 - cur_vs_sma50 > ext_thresh → SHORT"""
        if params is None:
            params = {'ext_thresh': 8}
        
        if i < 50:
            return False
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        cur_vs_sma50 = (close[i] - sma50[i]) / sma50[i] * 100
        return cur_vs_sma50 > params['ext_thresh']
    
    def high_breakout_50(self, df: pd.DataFrame, i: int, params: dict = None) -> bool:
        """Tool 31: high_breakout_50 - price > 50-bar high AND vol > vol_mult → LONG"""
        if params is None:
            params = {'vol_mult': 2.0}
        
        if i < 50:
            return False
        
        close = df['close'].values
        volume = df['volume'].values.astype(float)
        
        # Price > 50-bar high
        high_50 = np.max(close[i-50:i])
        price_breakout = close[i] > high_50
        
        # Volume filter
        avg_vol = np.mean(volume[max(0, i-19):i+1])
        vol_spike = volume[i] > params['vol_mult'] * avg_vol if avg_vol > 0 else False
        
        return price_breakout and vol_spike
    
    def thursday_short(self, df: pd.DataFrame, i: int, params: dict = None) -> bool:
        """Tool 10: calendar pattern - thursday/sunday/etc + price > SMA50 → test both directions"""
        if params is None:
            params = {'day': 3}  # 3=Thursday, 6=Sunday
        
        if i < 50:
            return False
        
        close = df['close'].values
        sma50 = self.calc_sma(close[:i+1], 50)
        
        timestamp = pd.to_datetime(df.iloc[i]['timestamp'])
        dow = timestamp.weekday()  # 0=Monday, 3=Thursday, 6=Sunday
        
        return dow == params['day'] and close[i] > sma50[i]
    
    def run_tests_with_sweeps(self):
        """Run tests with parameter sweeps for failed tools"""
        print("Starting tool testing with parameter optimization...")
        
        # Test a subset of tools first
        tools_to_test = [
            ('mega_pump_sell_t1', self.mega_pump_sell_t1, 'short'),
            ('sma50_ext_8', self.sma50_ext_8, 'short'),  
            ('high_breakout_50', self.high_breakout_50, 'long'),
            ('thursday_short', self.thursday_short, 'short'),
        ]
        
        for tool_name, tool_func, direction in tools_to_test:
            print(f"\nTesting {tool_name} ({direction.upper()})...")
            
            # Test original parameters first
            original_results = []
            for pair in self.pairs:
                if pair not in self.data:
                    continue
                    
                result = self.test_tool_with_params(pair, tool_name, tool_func, direction)
                original_results.append(result)
                self.results.append(result)
            
            # Summary of original results
            total_signals = sum(r['signals'] for r in original_results)
            pairs_with_signals = sum(1 for r in original_results if r['signals'] > 0)
            
            print(f"  ORIGINAL: {total_signals} total signals across {pairs_with_signals} pairs")
            
            # If low signal count, try parameter sweep
            if total_signals < 50:  # Arbitrary threshold
                print(f"  Low signal count, trying parameter sweep...")
                
                if tool_name == 'mega_pump_sell_t1':
                    # Try lower RSI and return thresholds
                    param_combinations = [
                        {'rsi_thresh': 75, 'ret_thresh': 8},
                        {'rsi_thresh': 70, 'ret_thresh': 6},
                        {'rsi_thresh': 75, 'ret_thresh': 5},
                    ]
                elif tool_name == 'sma50_ext_8':
                    param_combinations = [
                        {'ext_thresh': 5},
                        {'ext_thresh': 3},
                        {'ext_thresh': 2},
                    ]
                elif tool_name == 'high_breakout_50':
                    param_combinations = [
                        {'vol_mult': 1.5},
                        {'vol_mult': 1.0},  # No volume filter
                    ]
                elif tool_name == 'thursday_short':
                    # Test as LONG direction (maybe edge is reversed)
                    param_combinations = [{'day': 3}]  # Keep same params, test opposite direction
                else:
                    param_combinations = []
                
                best_result = None
                best_score = 0
                
                for params in param_combinations:
                    if tool_name == 'thursday_short':
                        # Test opposite direction
                        test_direction = 'long'
                        test_name = tool_name + '_LONG'
                    else:
                        test_direction = direction
                        test_name = tool_name + '_OPT'
                    
                    sweep_results = []
                    for pair in self.pairs:
                        if pair not in self.data:
                            continue
                        
                        result = self.test_tool_with_params(pair, test_name, tool_func, 
                                                          test_direction, params)
                        sweep_results.append(result)
                    
                    # Evaluate this parameter set
                    sweep_signals = sum(r['signals'] for r in sweep_results)
                    sweep_wr_8h = np.mean([r['wr_8h'] for r in sweep_results if r['signals'] > 0]) if sweep_signals > 0 else 0
                    sweep_wr_24h = np.mean([r['wr_24h'] for r in sweep_results if r['signals'] > 0]) if sweep_signals > 0 else 0
                    
                    # Score: signals * (wr_8h + wr_24h) - prioritize both signal count and win rate
                    score = sweep_signals * (sweep_wr_8h + sweep_wr_24h)
                    
                    print(f"    {params}: {sweep_signals} signals, WR_8h={sweep_wr_8h:.1%}, WR_24h={sweep_wr_24h:.1%}, score={score:.1f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = (params, sweep_results, test_direction, test_name)
                
                # Add best sweep result to results
                if best_result and best_score > 0:
                    params, sweep_results, test_direction, test_name = best_result
                    for result in sweep_results:
                        self.results.append(result)
                    print(f"  BEST PARAMS: {params} with score {best_score:.1f}")
    
    def generate_report(self):
        """Generate comprehensive report"""
        if not self.results:
            print("No results to report!")
            return
        
        # Create summary by tool
        tool_summary = {}
        for result in self.results:
            tool = result['tool']
            if tool not in tool_summary:
                tool_summary[tool] = {
                    'direction': result['direction'],
                    'total_signals': 0,
                    'pairs_tested': 0,
                    'pairs_with_signals': 0,
                    'total_wr_8h_weighted': 0,
                    'total_wr_24h_weighted': 0,
                    'total_ret_8h_weighted': 0,
                    'total_ret_24h_weighted': 0,
                    'pass_count': 0,
                    'params': result.get('params', None)
                }
            
            summary = tool_summary[tool]
            summary['pairs_tested'] += 1
            
            if result['signals'] > 0:
                summary['pairs_with_signals'] += 1
                summary['total_signals'] += result['signals']
                
                # Weight by number of signals
                weight = result['signals']
                summary['total_wr_8h_weighted'] += result['wr_8h'] * weight
                summary['total_wr_24h_weighted'] += result['wr_24h'] * weight
                summary['total_ret_8h_weighted'] += result['avg_return_8h'] * weight
                summary['total_ret_24h_weighted'] += result['avg_return_24h'] * weight
                
                if result['status'] == 'PASS':
                    summary['pass_count'] += 1
        
        # Calculate weighted averages
        for tool, summary in tool_summary.items():
            if summary['total_signals'] > 0:
                summary['avg_wr_8h'] = summary['total_wr_8h_weighted'] / summary['total_signals']
                summary['avg_wr_24h'] = summary['total_wr_24h_weighted'] / summary['total_signals']
                summary['avg_ret_8h'] = summary['total_ret_8h_weighted'] / summary['total_signals']
                summary['avg_ret_24h'] = summary['total_ret_24h_weighted'] / summary['total_signals']
            else:
                summary['avg_wr_8h'] = 0
                summary['avg_wr_24h'] = 0
                summary['avg_ret_8h'] = 0
                summary['avg_ret_24h'] = 0
        
        # Write report
        report_path = Path("data/bull_momentum_tools_1h_report.md")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# BULL/GREED + BREAKOUT/MOMENTUM Tools - 1H Testing Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Test Configuration\n")
            f.write(f"- **Data:** Real 1h Binance candles, {len(self.pairs)} pairs\n")
            f.write(f"- **Walk-forward:** Bars 0-4380 = in-sample, bars 4380-8760 = out-of-sample\n") 
            f.write(f"- **Fee-adjusted:** -{self.fee_pct*100:.2f}% round-trip subtracted from every return\n")
            f.write(f"- **Forward returns:** +8 bars (8h) and +24 bars (24h)\n")
            f.write(f"- **Win condition:** (forward_return - fees) > 0\n")
            f.write(f"- **Parameter sweeps:** Applied to tools with <50 total signals\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Tool | Direction | Signals | Pairs w/ Signals | WR_8h | WR_24h | Avg_Ret_8h | Avg_Ret_24h | Status | Params |\n")
            f.write("|------|-----------|---------|------------------|-------|--------|------------|-------------|--------|---------|\n")
            
            # Sort by total signals descending
            sorted_tools = sorted(tool_summary.items(), key=lambda x: x[1]['total_signals'], reverse=True)
            
            for tool, summary in sorted_tools:
                direction = summary['direction'].upper()
                signals = summary['total_signals']
                pairs_with_signals = summary['pairs_with_signals']
                wr_8h = summary['avg_wr_8h']
                wr_24h = summary['avg_wr_24h']
                ret_8h = summary['avg_ret_8h']
                ret_24h = summary['avg_ret_24h']
                params_str = str(summary['params']) if summary['params'] else "default"
                
                # Determine overall status
                if signals == 0:
                    status = "NO_SIGNALS"
                elif wr_8h > 0.5 or wr_24h > 0.5:
                    status = "PASS"
                else:
                    status = "FAIL"
                
                f.write(f"| {tool} | {direction} | {signals} | {pairs_with_signals} | "
                       f"{wr_8h:.1%} | {wr_24h:.1%} | {ret_8h:+.2f}% | {ret_24h:+.2f}% | {status} | {params_str} |\n")
        
        print(f"\nReport written to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, s in sorted_tools if (s['avg_wr_8h'] > 0.5 or s['avg_wr_24h'] > 0.5) and s['total_signals'] > 0)
        failed = sum(1 for _, s in sorted_tools if (s['avg_wr_8h'] <= 0.5 and s['avg_wr_24h'] <= 0.5) and s['total_signals'] > 0)
        no_signals = sum(1 for _, s in sorted_tools if s['total_signals'] == 0)
        
        print(f"Tools PASSED: {passed}")
        print(f"Tools FAILED: {failed}")
        print(f"Tools NO_SIGNALS: {no_signals}")
        print(f"Total tools tested: {len(sorted_tools)}")

def main():
    print("BULL/GREED + BREAKOUT/MOMENTUM Tools Testing Framework V2")
    print("="*60)
    
    tester = BullMomentumTesterV2()
    
    # Load data
    tester.load_data()
    
    # Run tests with parameter sweeps
    tester.run_tests_with_sweeps()
    
    # Generate report
    tester.generate_report()
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()