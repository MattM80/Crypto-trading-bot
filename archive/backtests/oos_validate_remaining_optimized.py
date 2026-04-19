#!/usr/bin/env python3
"""
Optimized Out-of-Sample Validation for REMAINING Trading Tools
Tests cross-pair, statistical/math, and combo tools from run_master_bot.py against real Binance 4h data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OOSRemainingValidator:
    def __init__(self, data_dir="data/binance_real"):
        self.data_dir = data_dir
        self.pairs = []
        self.data_cache = {}
        self.results = {}
        
        # Load all available pairs
        print("Loading Binance real data...")
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('_4h.csv'):
                pair = f.replace('_4h.csv', '')
                self.pairs.append(pair)
                df = pd.read_csv(f"{data_dir}/{f}")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Pre-compute commonly used arrays
                self.data_cache[pair] = {
                    'df': df,
                    'close': df['close'].values.astype(float),
                    'open': df['open'].values.astype(float),
                    'high': df['high'].values.astype(float),
                    'low': df['low'].values.astype(float),
                    'volume': df['volume'].values.astype(float),
                }
                
        print(f"Loaded {len(self.pairs)} pairs: {', '.join(self.pairs)}")
        print(f"Data range: {self.data_cache[self.pairs[0]]['df']['timestamp'].iloc[0]} to {self.data_cache[self.pairs[0]]['df']['timestamp'].iloc[-1]}")
        
        # Verify all pairs have same length
        lengths = [len(self.data_cache[pair]['close']) for pair in self.pairs]
        if len(set(lengths)) > 1:
            print(f"WARNING: Pairs have different lengths: {dict(zip(self.pairs, lengths))}")
        else:
            print(f"All pairs have {lengths[0]} bars each")
    
    # ---- HELPER FUNCTIONS (optimized) ----
    
    def calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI - vectorized version."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
            
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Use pandas ewm for smoother calculation
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        
        avg_gain = gain_series.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss_series.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with initial value
        result = np.full(len(prices), 50.0)
        result[period:] = rsi.values[period-1:]
        return result
    
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average using pandas for speed."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        return pd.Series(prices).rolling(period).mean().values
    
    # Math helper functions (exact from run_master_bot.py)
    def hurst(self, returns, w=50):
        if len(returns) < w: return 0.5
        r2 = returns[-w:]
        v1 = np.var(r2)
        v2 = np.var(r2[::2]) if len(r2) >= 4 else v1
        if v1 <= 0 or v2 <= 0: return 0.5
        vr = v2 / v1
        return max(0, min(1, 0.5 + np.log(max(vr, 0.01)) / (2 * np.log(2))))
    
    def entropy(self, returns, bins=15):
        if len(returns) < 10: return 3.0
        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0: return 3.0
        probs = hist / hist.sum()
        return -np.sum(probs * np.log2(probs))
    
    def autocorr(self, returns, lag=1):
        if len(returns) < lag + 5: return 0
        return float(pd.Series(returns).autocorr(lag=lag))
    
    def vpin(self, close_arr, vol_arr, w=20):
        if len(close_arr) < w + 1: return 0
        rets = np.diff(close_arr) / close_arr[:-1]
        bv = np.where(rets > 0, vol_arr[1:], 0)
        sv = np.where(rets < 0, vol_arr[1:], 0)
        rb = np.sum(bv[-w:]); rs = np.sum(sv[-w:])
        t = rb + rs
        return abs(rb - rs) / t if t > 0 else 0
    
    def validate_tools(self):
        """Run validation on all remaining tools."""
        print("\nStarting validation of remaining tools...")
        
        min_length = min(len(self.data_cache[pair]['close']) for pair in self.pairs)
        start_bar = 100
        end_bar = min_length - 10
        
        print(f"Processing {end_bar - start_bar} bars from {start_bar} to {end_bar}")
        
        # Process in chunks to save memory and show progress
        chunk_size = 200
        all_signals = []
        
        for chunk_start in range(start_bar, end_bar, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_bar)
            print(f"Processing bars {chunk_start} to {chunk_end} ({chunk_end - chunk_start} bars)...")
            
            chunk_signals = []
            for bar_idx in range(chunk_start, chunk_end):
                # Get all pair data for cross-pair tools
                all_pair_data = {}
                for pair in self.pairs:
                    close = self.data_cache[pair]['close'][:bar_idx+1]
                    if len(close) >= bar_idx + 1:
                        all_pair_data[pair] = {
                            'close': close,
                            'open': self.data_cache[pair]['open'][:bar_idx+1],
                            'high': self.data_cache[pair]['high'][:bar_idx+1],
                            'low': self.data_cache[pair]['low'][:bar_idx+1],
                            'volume': self.data_cache[pair]['volume'][:bar_idx+1],
                        }
                
                if len(all_pair_data) == len(self.pairs):
                    bar_signals = self.scan_tools_optimized(all_pair_data, bar_idx)
                    chunk_signals.extend(bar_signals)
            
            all_signals.extend(chunk_signals)
            print(f"  Generated {len(chunk_signals)} signals in this chunk")
        
        print(f"\nGenerated {len(all_signals)} total signals")
        
        # Calculate performance
        self.analyze_signals_optimized(all_signals)
        self.generate_report()
    
    def scan_tools_optimized(self, all_pair_data, bar_idx):
        """Optimized signal scanning."""
        signals = []
        total_pairs = len(all_pair_data)
        
        # Pre-calculate cross-market stats
        btc_data = all_pair_data.get('BTCUSDT')
        eth_data = all_pair_data.get('ETHUSDT')
        
        # Count coins for market panic/fomo
        dropping_3pct = 0
        dropping_2pct = 0
        pumping_1pct = 0
        
        for pair, data in all_pair_data.items():
            close = data['close']
            if len(close) >= 5:
                ret_4h = (close[-1] - close[-5]) / close[-5] * 100
                if ret_4h < -3.0: dropping_3pct += 1
                if ret_4h < -2.0: dropping_2pct += 1
                if ret_4h > 1.0: pumping_1pct += 1
        
        # Cross-pair tools
        
        # Tool 13: btc_eth_diverge
        if btc_data and eth_data and len(btc_data['close']) >= 25:
            btc_ret_24h = (btc_data['close'][-1] - btc_data['close'][-25]) / btc_data['close'][-25] * 100
            eth_ret_24h = (eth_data['close'][-1] - eth_data['close'][-25]) / eth_data['close'][-25] * 100
            if btc_ret_24h - eth_ret_24h >= 3.0:
                signals.append(('ETHUSDT', 'btc_eth_diverge', 'short', bar_idx, 24))
        
        # Tool 16: market_panic (90%, 80%, 70%)
        if total_pairs >= 5:
            drop_pct = dropping_3pct / total_pairs
            if drop_pct >= 0.9:
                for pair in all_pair_data.keys():
                    signals.append((pair, 'market_panic_90', 'long', bar_idx, 24))
            elif drop_pct >= 0.8:
                for pair in all_pair_data.keys():
                    signals.append((pair, 'market_panic_80', 'long', bar_idx, 24))
            elif drop_pct >= 0.7:
                for pair in all_pair_data.keys():
                    signals.append((pair, 'market_panic_70', 'long', bar_idx, 16))
        
        # Tool 21: blood_in_streets
        if dropping_2pct / total_pairs >= 0.7:
            for pair, data in all_pair_data.items():
                if len(data['close']) >= 15:
                    rsi = self.calc_rsi(data['close'][-15:], 7)
                    if rsi[-1] < 20:
                        signals.append((pair, 'blood_in_streets', 'long', bar_idx, 24))
        
        # Tool 22: fomo_ride
        if pumping_1pct / total_pairs >= 0.8:
            for pair in all_pair_data.keys():
                signals.append((pair, 'fomo_ride', 'long', bar_idx, 8))
        
        # Individual pair tools
        for pair, data in all_pair_data.items():
            close = data['close']
            if len(close) < 100: continue
            
            # Calculate key metrics
            price = close[-1]
            ret_4h = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
            ret_8h = (close[-1] - close[-9]) / close[-9] * 100 if len(close) >= 9 else 0
            ret_24h = (close[-1] - close[-25]) / close[-25] * 100 if len(close) >= 25 else 0
            
            # Math features (only calculate if needed)
            returns = np.diff(close[-100:]) / close[-100:-1]
            H = self.hurst(returns)
            ent = self.entropy(returns[-30:]) if len(returns) >= 30 else 3.0
            ac1 = self.autocorr(returns, 1)
            
            # RSI and SMA50
            rsi = self.calc_rsi(close, 14)[-1]
            sma50 = self.calc_sma(close, 50)
            cur_vs_sma50 = ((price / sma50[-1]) - 1) * 100 if not np.isnan(sma50[-1]) else 0
            
            # Statistical tools
            if ret_24h < -10 and ac1 < -0.05:
                signals.append((pair, 'crash_neg_ac', 'long', bar_idx, 24))
            
            if ret_24h < -8 and H < 0.45:
                signals.append((pair, 'crash_mean_revert', 'long', bar_idx, 24))
            
            if H > 0.65 and ret_4h > 2:
                signals.append((pair, 'hurst_trend', 'long', bar_idx, 8))
            
            # VPIN (only calc if needed)
            if ret_8h < -5 or close[-1] < data['open'][-1]:
                vp = self.vpin(close[-30:], data['volume'][-30:])
                if vp > 0.7 and close[-1] < data['open'][-1]:
                    signals.append((pair, 'vpin_toxic', 'long', bar_idx, 8))
                if ret_8h < -5 and vp > 0.5:
                    signals.append((pair, 'vpin_dip', 'long', bar_idx, 8))
            
            # Entropy tools
            if ent < 2.5:
                if not np.isnan(sma50[-1]) and price > sma50[-1]:
                    signals.append((pair, 'entropy_short', 'short', bar_idx, 8))
                if ret_4h < -2:
                    signals.append((pair, 'entropy_dip', 'long', bar_idx, 8))
            
            # Alt vs BTC tools
            if pair != 'BTCUSDT' and btc_data and len(btc_data['close']) >= 25:
                btc_ret_24h = (btc_data['close'][-1] - btc_data['close'][-25]) / btc_data['close'][-25] * 100
                spread_24h = ret_24h - btc_ret_24h
                
                if spread_24h >= 8:
                    signals.append((pair, 'alt_btc_revert_t1', 'short', bar_idx, 8))
                elif spread_24h >= 5:
                    signals.append((pair, 'alt_btc_revert_t2', 'short', bar_idx, 8))
                elif 3.0 <= spread_24h < 5.0:
                    signals.append((pair, 'alt_btc_revert_t3', 'short', bar_idx, 8))
                
                # Combo tools with alt/btc
                if spread_24h >= 8 and ac1 < -0.05:
                    signals.append((pair, 'alt_btc_neg_ac', 'short', bar_idx, 8))
                elif spread_24h >= 5 and ac1 < -0.15:
                    signals.append((pair, 'alt_btc_neg_ac_5', 'short', bar_idx, 8))
            
            # SMA50 combo tools
            if not np.isnan(sma50[-1]):
                returns_50 = np.diff(close[-50:]) / close[-50:-1] if len(close) >= 50 else np.array([])
                combo_kurt = float(pd.Series(returns_50).kurtosis()) if len(returns_50) > 10 else 0
                if np.isnan(combo_kurt): combo_kurt = 0
                
                if cur_vs_sma50 > 8 and ac1 < -0.1:
                    signals.append((pair, 'sma50_ext_neg_ac', 'short', bar_idx, 8))
                if cur_vs_sma50 > 10 and combo_kurt > 5:
                    signals.append((pair, 'sma50_ext_fat_tail', 'short', bar_idx, 8))
                if cur_vs_sma50 > 8 and combo_kurt > 5:
                    signals.append((pair, 'sma50_ext_kurt', 'short', bar_idx, 8))
        
        return signals
    
    def analyze_signals_optimized(self, signals):
        """Optimized signal analysis."""
        print("\nAnalyzing signal performance...")
        
        for signal_tuple in signals:
            pair, tool, direction, bar_idx, hold_hours = signal_tuple
            
            if tool not in self.results:
                self.results[tool] = {
                    'signals': 0,
                    'direction': direction,
                    'returns_8h': [],
                    'returns_24h': [],
                    'wins_8h': 0,
                    'wins_24h': 0
                }
            
            # Calculate forward returns
            close_data = self.data_cache[pair]['close']
            if bar_idx + 6 < len(close_data):
                current_price = close_data[bar_idx]
                price_8h = close_data[bar_idx + 2]
                price_24h = close_data[bar_idx + 6]
                
                if direction == 'long':
                    ret_8h = (price_8h - current_price) / current_price * 100
                    ret_24h = (price_24h - current_price) / current_price * 100
                else:
                    ret_8h = (current_price - price_8h) / current_price * 100
                    ret_24h = (current_price - price_24h) / current_price * 100
                
                self.results[tool]['signals'] += 1
                self.results[tool]['returns_8h'].append(ret_8h)
                self.results[tool]['returns_24h'].append(ret_24h)
                
                if ret_8h > 0: self.results[tool]['wins_8h'] += 1
                if ret_24h > 0: self.results[tool]['wins_24h'] += 1
        
        # Calculate final metrics
        for tool, data in self.results.items():
            if data['signals'] > 0:
                data['wr_8h'] = data['wins_8h'] / data['signals'] * 100
                data['wr_24h'] = data['wins_24h'] / data['signals'] * 100
                data['net_8h'] = np.mean(data['returns_8h']) if data['returns_8h'] else 0
                data['net_24h'] = np.mean(data['returns_24h']) if data['returns_24h'] else 0
    
    def generate_report(self):
        """Generate validation report."""
        print("\nGenerating validation report...")
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['net_24h'], reverse=True)
        
        # Classification
        passed = []
        marginal = []
        failed = []
        
        for tool, data in sorted_results:
            if data['signals'] >= 10:
                target_wr = 50.0 if data['direction'] == 'long' else 45.0
                if data['net_24h'] > 0 and data['wr_24h'] >= target_wr:
                    passed.append((tool, data))
                elif data['net_24h'] > 0:
                    marginal.append((tool, data))
                else:
                    failed.append((tool, data))
            elif data['signals'] >= 5:
                marginal.append((tool, data))
            else:
                failed.append((tool, data))
        
        # Print results
        print(f"\n{'='*80}")
        print("CROSS-PAIR, STATISTICAL & COMBO TOOLS VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"{'Tool':<25} {'Dir':<5} {'Signals':<8} {'WR_8h':<6} {'Net_8h':<7} {'WR_24h':<6} {'Net_24h':<7} {'Status':<10}")
        print("-" * 80)
        
        for tool, data in sorted_results:
            status = "PASS" if (tool, data) in passed else "MARGINAL" if (tool, data) in marginal else "FAIL"
            print(f"{tool:<25} {data['direction']:<5} {data['signals']:<8} "
                  f"{data['wr_8h']:<6.1f} {data['net_8h']:<+7.2f} "
                  f"{data['wr_24h']:<6.1f} {data['net_24h']:<+7.2f} {status:<10}")
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"PASSED:   {len(passed):2d} tools")
        print(f"MARGINAL: {len(marginal):2d} tools")
        print(f"FAILED:   {len(failed):2d} tools") 
        print(f"TOTAL:    {len(sorted_results):2d} tools tested")
        
        # Append to report file
        report_path = "data/oos_report.md"
        with open(report_path, 'a') as f:
            f.write(f"\n\n## Cross-Pair, Statistical & Combo Tools\n\n")
            f.write(f"**Validation Date**: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Tools Tested**: {len(sorted_results)} remaining tools\n")
            f.write(f"**Results**: {len(passed)} PASSED, {len(marginal)} MARGINAL, {len(failed)} FAILED\n\n")
            
            if passed:
                f.write("### 🟢 PASSED Tools\n\n")
                f.write("| Tool | Direction | Signals | WR_24h | Net_24h | Status |\n")
                f.write("|------|-----------|---------|---------|---------|--------|\n")
                for tool, data in passed:
                    f.write(f"| {tool} | {data['direction']} | {data['signals']} | "
                           f"{data['wr_24h']:.1f}% | {data['net_24h']:+.2f}% | ✅ Validated |\n")
            
            if marginal:
                f.write("\n### 🟡 MARGINAL Tools\n\n")
                f.write("| Tool | Direction | Signals | WR_24h | Net_24h | Issue |\n")
                f.write("|------|-----------|---------|---------|---------|-------|\n")
                for tool, data in marginal:
                    target_wr = 50.0 if data['direction'] == 'long' else 45.0
                    issue = f"WR below {target_wr:.0f}%" if data['wr_24h'] < target_wr else "Low signal count"
                    f.write(f"| {tool} | {data['direction']} | {data['signals']} | "
                           f"{data['wr_24h']:.1f}% | {data['net_24h']:+.2f}% | {issue} |\n")
            
            if failed:
                f.write("\n### 🔴 FAILED Tools\n\n")
                f.write("| Tool | Direction | Signals | WR_24h | Net_24h | Action |\n")
                f.write("|------|-----------|---------|---------|---------|--------|\n")
                for tool, data in failed:
                    action = "🚫 DISABLE" if data['net_24h'] <= 0 else "🔧 NEEDS FIX"
                    f.write(f"| {tool} | {data['direction']} | {data['signals']} | "
                           f"{data['wr_24h']:.1f}% | {data['net_24h']:+.2f}% | {action} |\n")
            
            f.write("\n---\n")
            f.write(f"*Remaining tools validation completed on {datetime.now().strftime('%B %d, %Y')}*\n")
        
        print(f"\nReport appended to {report_path}")
        return len(passed), len(marginal), len(failed)

def main():
    validator = OOSRemainingValidator()
    validator.validate_tools()

if __name__ == "__main__":
    main()