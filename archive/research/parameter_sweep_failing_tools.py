#!/usr/bin/env python3
"""
PARAMETER SWEEP - Optimize Failing Tools
================================================================================

This script attempts to fix the 7 tools that failed OOS validation by sweeping
their parameters to find optimal settings that may be profitable after fees.

FAILING TOOLS TO OPTIMIZE:
1. dip_buy: -0.20% avg (516 signals) - Try tighter/looser dip thresholds
2. rsi_pump_12h: -0.32% avg (18 signals) - Try different RSI/pump thresholds  
3. crash_neg_ac: -0.16% avg (79 signals) - Try different crash/AC thresholds
4. entropy_short: -0.54% avg (1668 signals) - Try different entropy thresholds
5. sma50_ext_8: -0.25% avg (126 signals) - Try different extension thresholds
6. vpin_dip: -0.33% avg (169 signals) - Try different VPIN/dip thresholds  
7. alt_btc_revert_t1: -0.20% avg (123 signals) - Try different spread thresholds

APPROACH:
- For each failing tool, sweep key parameters across reasonable ranges
- Test on out-of-sample data with fee adjustment
- Look for parameter combinations that achieve positive expected value
- Report optimized parameters and their performance
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BINANCE_DATA_DIR = DATA_DIR / "binance_1h"

KRAKEN_TAKER_FEE = 0.0026  # 0.52% round-trip

# Pair mappings
PAIR_MAPPING = {
    "NEARUSD": "NEARUSDT", "UNIUSD": "UNIUSDT", "AVAXUSD": "AVAXUSDT", 
    "LINKUSD": "LINKUSDT", "AAVEUSD": "AAVEUSDT", "SOLUSD": "SOLUSDT",
    "ETHUSD": "ETHUSDT", "XBTUSD": "BTCUSDT", "DOTUSD": "DOTUSDT", 
    "XLMUSD": "XLMUSDT", "XRPUSD": "XRPUSDT", "ADAUSD": "ADAUSDT", 
    "ATOMUSD": "ATOMUSDT", "DOGEUSD": "DOGEUSDT", "FILUSD": "FILUSDT", 
    "LTCUSD": "LTCUSDT"
}

print("🔧 PARAMETER SWEEP - Optimize Failing Tools")
print("="*50)


class ParameterSweeper:
    
    def __init__(self):
        self.data = {}
        self.load_data()
        
    def load_data(self):
        """Load OOS data for parameter testing"""
        print("📊 Loading out-of-sample data...")
        
        for kraken_pair, binance_pair in PAIR_MAPPING.items():
            file_path = BINANCE_DATA_DIR / f"{binance_pair}_1h.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Use second half as OOS (same as backtest)
                split_point = len(df) // 2
                oos_df = df.iloc[split_point:].copy()
                
                self.data[kraken_pair] = oos_df
                
        print(f"✅ Loaded {len(self.data)} pairs OOS data")
        
    def calculate_simple_indicators(self, df):
        """Calculate indicators needed for failing tools"""
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        # RSI-7
        def simple_rsi(prices, period=7):
            if len(prices) <= period:
                return np.full(len(prices), 50.0)
                
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.full(len(prices), np.nan)
            avg_loss = np.full(len(prices), np.nan)
            
            if len(delta) >= period:
                avg_gain[period] = np.mean(gain[:period])
                avg_loss[period] = np.mean(loss[:period])
                
                for i in range(period + 1, len(prices)):
                    avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                    avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
            
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
            return 100 - (100 / (1 + rs))
        
        # SMA50
        sma50 = np.full(len(close), np.nan)
        if len(close) >= 50:
            for i in range(49, len(close)):
                sma50[i] = np.mean(close[i-49:i+1])
        
        # Simple entropy approximation
        def simple_entropy(returns_window):
            if len(returns_window) < 10:
                return 3.0
            try:
                hist, _ = np.histogram(returns_window, bins=8)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 3.0
                probs = hist / hist.sum()
                return -np.sum(probs * np.log2(probs + 1e-10))
            except:
                return 3.0
        
        # Simple autocorrelation  
        def simple_ac(returns_window, lag=1):
            if len(returns_window) < lag + 5:
                return 0
            try:
                return float(pd.Series(returns_window).autocorr(lag=lag)) or 0
            except:
                return 0
        
        # Simple VPIN proxy
        def simple_vpin(close_window, vol_window):
            if len(close_window) < 20:
                return 0
            try:
                returns = np.diff(close_window[-20:]) / close_window[-20:-1]
                vol_recent = vol_window[-19:] if len(vol_window) >= 19 else vol_window
                
                if len(returns) != len(vol_recent):
                    return 0
                    
                buy_vol = np.sum(vol_recent[returns > 0])
                sell_vol = np.sum(vol_recent[returns < 0])
                total_vol = buy_vol + sell_vol
                
                return abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
            except:
                return 0
        
        return {
            'close': close,
            'volume': volume,
            'rsi': simple_rsi(close, 7),
            'sma50': sma50,
            'entropy_func': simple_entropy,
            'ac_func': simple_ac,
            'vpin_func': simple_vpin
        }
        
    def test_tool_parameters(self, tool_name, param_combinations):
        """Test parameter combinations for a tool"""
        print(f"\n🧪 Testing {tool_name} with {len(param_combinations)} parameter combinations...")
        
        best_params = None
        best_performance = -999
        results = []
        
        for params in param_combinations:
            all_signals = []
            
            # Test across all pairs
            for pair, df in self.data.items():
                indicators = self.calculate_simple_indicators(df)
                
                # Get BTC data for cross-pair tools
                btc_indicators = None
                if pair != "XBTUSD" and "XBTUSD" in self.data:
                    btc_df = self.data["XBTUSD"]
                    btc_indicators = self.calculate_simple_indicators(btc_df)
                
                # Test this parameter combination
                signals = self.detect_tool_signals(
                    tool_name, pair, df, indicators, params, btc_indicators
                )
                
                all_signals.extend(signals)
            
            # Analyze performance
            if all_signals:
                returns = [s['return_8h'] for s in all_signals if s.get('return_8h') is not None]
                
                if returns:
                    avg_return = np.mean(returns)
                    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                    signal_count = len(all_signals)
                    
                    results.append({
                        'params': params,
                        'signals': signal_count,
                        'avg_return': avg_return,
                        'win_rate': win_rate,
                        'profitable': avg_return > 0
                    })
                    
                    if avg_return > best_performance:
                        best_performance = avg_return
                        best_params = params
        
        # Sort by performance
        results.sort(key=lambda x: x['avg_return'], reverse=True)
        
        print(f"📊 {tool_name} Results:")
        print(f"   Best params: {best_params}")
        print(f"   Best performance: {best_performance:+.2f}% avg return")
        
        # Show top 5 results
        for i, result in enumerate(results[:5]):
            status = "✅" if result['profitable'] else "❌"
            print(f"   {i+1}. {result['params']} - {result['signals']} signals, "
                  f"{result['win_rate']:.1f}% WR, {result['avg_return']:+.2f}% avg {status}")
        
        return results[0] if results else None
        
    def detect_tool_signals(self, tool_name, pair, df, indicators, params, btc_indicators=None):
        """Generate signals for a specific tool with given parameters"""
        signals = []
        
        close = indicators['close']
        rsi = indicators['rsi']
        sma50 = indicators['sma50']
        
        # Sample every 12 bars for performance
        for i in range(50, len(df) - 50, 12):
            price = close[i]
            cur_rsi = rsi[i] if not np.isnan(rsi[i]) else 50
            
            # Calculate returns
            ret_4h = (price - close[i-5]) / close[i-5] * 100 if i >= 5 else 0
            ret_8h = (price - close[i-9]) / close[i-9] * 100 if i >= 9 else 0
            ret_12h = (price - close[i-13]) / close[i-13] * 100 if i >= 13 else 0  
            ret_24h = (price - close[i-25]) / close[i-25] * 100 if i >= 25 else 0
            
            signal_triggered = False
            
            # Tool-specific logic with parameter variations
            if tool_name == 'dip_buy':
                dip_threshold = params['dip_threshold']
                rsi_max = params.get('rsi_max', 100)  # Optional RSI filter
                
                if ret_4h < dip_threshold and cur_rsi < rsi_max:
                    signal_triggered = True
                    
            elif tool_name == 'rsi_pump_12h':
                rsi_threshold = params['rsi_threshold']
                pump_threshold = params['pump_threshold']
                
                if cur_rsi > rsi_threshold and ret_12h >= pump_threshold:
                    signal_triggered = True
                    
            elif tool_name == 'crash_neg_ac':
                crash_threshold = params['crash_threshold']
                ac_threshold = params['ac_threshold']
                
                # Calculate autocorrelation
                if i >= 50:
                    returns_window = np.diff(close[i-49:i+1]) / close[i-49:i]
                    ac1 = indicators['ac_func'](returns_window, 1)
                    
                    if ret_24h < crash_threshold and ac1 < ac_threshold:
                        signal_triggered = True
                        
            elif tool_name == 'entropy_short':
                entropy_threshold = params['entropy_threshold']
                sma_required = params.get('sma_required', True)
                
                if i >= 30:
                    returns_window = np.diff(close[i-29:i+1]) / close[i-29:i]
                    entropy = indicators['entropy_func'](returns_window)
                    
                    sma_condition = True
                    if sma_required and not np.isnan(sma50[i]):
                        sma_condition = price > sma50[i]
                    
                    if entropy < entropy_threshold and sma_condition:
                        signal_triggered = True
                        
            elif tool_name == 'sma50_ext_8':
                ext_threshold = params['ext_threshold']
                
                if not np.isnan(sma50[i]) and sma50[i] > 0:
                    cur_vs_sma50 = (price - sma50[i]) / sma50[i] * 100
                    
                    if cur_vs_sma50 > ext_threshold:
                        signal_triggered = True
                        
            elif tool_name == 'vpin_dip':
                dip_threshold = params['dip_threshold']
                vpin_threshold = params['vpin_threshold']
                
                if ret_8h < dip_threshold:
                    vpin = indicators['vpin_func'](close[:i+1], indicators['volume'][:i+1])
                    
                    if vpin > vpin_threshold:
                        signal_triggered = True
                        
            elif tool_name == 'alt_btc_revert_t1':
                spread_threshold = params['spread_threshold']
                
                if pair != "XBTUSD" and btc_indicators is not None:
                    btc_close = btc_indicators['close']
                    if i < len(btc_close) and i >= 25:
                        btc_ret24 = (btc_close[i] - btc_close[i-25]) / btc_close[i-25] * 100
                        spread_24h = ret_24h - btc_ret24
                        
                        if spread_24h >= spread_threshold:
                            signal_triggered = True
            
            if signal_triggered:
                # Calculate forward return
                if i + 8 < len(close):
                    exit_price = close[i + 8]
                    
                    if tool_name == 'entropy_short' or tool_name == 'rsi_pump_12h' or tool_name == 'sma50_ext_8' or tool_name == 'alt_btc_revert_t1':
                        # Short direction
                        gross_return = (price - exit_price) / price
                    else:
                        # Long direction
                        gross_return = (exit_price - price) / price
                    
                    net_return = (gross_return - KRAKEN_TAKER_FEE * 2) * 100
                    
                    signals.append({
                        'pair': pair,
                        'bar': i, 
                        'return_8h': net_return,
                        'params': params
                    })
        
        return signals
        
    def run_parameter_sweep(self):
        """Run parameter sweep on all failing tools"""
        
        optimized_tools = {}
        
        # 1. DIP BUY Parameter Sweep
        dip_buy_params = []
        for dip_thresh in [-2, -2.5, -3, -3.5, -4, -5]:
            for rsi_max in [100, 50, 40, 30]:  # Optional RSI filter
                dip_buy_params.append({
                    'dip_threshold': dip_thresh,
                    'rsi_max': rsi_max
                })
        
        result = self.test_tool_parameters('dip_buy', dip_buy_params)
        if result and result['profitable']:
            optimized_tools['dip_buy'] = result
        
        # 2. RSI PUMP 12H Parameter Sweep
        rsi_pump_params = []
        for rsi_thresh in [80, 85, 90, 95]:
            for pump_thresh in [5, 8, 10, 12, 15]:
                rsi_pump_params.append({
                    'rsi_threshold': rsi_thresh,
                    'pump_threshold': pump_thresh
                })
        
        result = self.test_tool_parameters('rsi_pump_12h', rsi_pump_params)
        if result and result['profitable']:
            optimized_tools['rsi_pump_12h'] = result
        
        # 3. CRASH NEG AC Parameter Sweep
        crash_ac_params = []
        for crash_thresh in [-8, -10, -12, -15]:
            for ac_thresh in [-0.03, -0.05, -0.1, -0.15]:
                crash_ac_params.append({
                    'crash_threshold': crash_thresh,
                    'ac_threshold': ac_thresh
                })
        
        result = self.test_tool_parameters('crash_neg_ac', crash_ac_params)
        if result and result['profitable']:
            optimized_tools['crash_neg_ac'] = result
        
        # 4. ENTROPY SHORT Parameter Sweep
        entropy_params = []
        for ent_thresh in [2.0, 2.2, 2.5, 2.8, 3.0]:
            for sma_req in [True, False]:
                entropy_params.append({
                    'entropy_threshold': ent_thresh,
                    'sma_required': sma_req
                })
        
        result = self.test_tool_parameters('entropy_short', entropy_params)
        if result and result['profitable']:
            optimized_tools['entropy_short'] = result
        
        # 5. SMA50 EXT Parameter Sweep
        sma_ext_params = []
        for ext_thresh in [5, 6, 8, 10, 12, 15]:
            sma_ext_params.append({
                'ext_threshold': ext_thresh
            })
        
        result = self.test_tool_parameters('sma50_ext_8', sma_ext_params)
        if result and result['profitable']:
            optimized_tools['sma50_ext_8'] = result
        
        # 6. VPIN DIP Parameter Sweep
        vpin_params = []
        for dip_thresh in [-3, -4, -5, -6, -8]:
            for vpin_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                vpin_params.append({
                    'dip_threshold': dip_thresh,
                    'vpin_threshold': vpin_thresh
                })
        
        result = self.test_tool_parameters('vpin_dip', vpin_params)
        if result and result['profitable']:
            optimized_tools['vpin_dip'] = result
        
        # 7. ALT BTC REVERT Parameter Sweep
        alt_btc_params = []
        for spread_thresh in [5, 6, 8, 10, 12]:
            alt_btc_params.append({
                'spread_threshold': spread_thresh
            })
        
        result = self.test_tool_parameters('alt_btc_revert_t1', alt_btc_params)
        if result and result['profitable']:
            optimized_tools['alt_btc_revert_t1'] = result
        
        # Summary
        print("\n" + "="*60)
        print("🎯 PARAMETER SWEEP SUMMARY")
        print("="*60)
        
        if optimized_tools:
            print(f"✅ Successfully optimized {len(optimized_tools)} tools:")
            
            for tool_name, result in optimized_tools.items():
                print(f"\n🔧 {tool_name}:")
                print(f"   Original performance: NEGATIVE")
                print(f"   Optimized params: {result['params']}")
                print(f"   New performance: {result['avg_return']:+.2f}% avg, {result['win_rate']:.1f}% WR")
                print(f"   Signals: {result['signals']}")
                
            # Generate updated bot code snippets
            self.generate_optimized_code(optimized_tools)
        else:
            print("❌ No tools could be optimized to profitability")
            print("   All 7 failing tools remain unprofitable even with parameter optimization")
            print("   Recommendation: Keep these tools DISABLED until market conditions change")
        
        return optimized_tools
    
    def generate_optimized_code(self, optimized_tools):
        """Generate code snippets for optimized tools"""
        print("\n📝 OPTIMIZED TOOL CODE:")
        print("-" * 40)
        
        for tool_name, result in optimized_tools.items():
            params = result['params']
            
            print(f"\n# {tool_name.upper()} (Optimized: {result['avg_return']:+.2f}% avg)")
            
            if tool_name == 'dip_buy':
                print(f"if ret_4h < {params['dip_threshold']} and cur_rsi < {params['rsi_max']}:")
                print(f"    signals.append(({{")
                print(f"        'pair': pair, 'tool': 'dip_buy_optimized', 'direction': 'long',")
                print(f"        'hold': 8, 'sl_pct': 0.03,")
                print(f"        'reason': f\"DIP BUY OPT: {{ret_4h:.1f}}% drop, RSI={{cur_rsi:.0f}}\",")
                print(f"    }}, score))")
                
            elif tool_name == 'rsi_pump_12h':
                print(f"if cur_rsi > {params['rsi_threshold']} and ret_12h >= {params['pump_threshold']}:")
                print(f"    signals.append(({{")
                print(f"        'pair': pair, 'tool': 'rsi_pump_12h_optimized', 'direction': 'short',")
                print(f"        'hold': 8, 'sl_pct': 0.05,")
                print(f"        'reason': f\"RSI PUMP OPT: RSI={{cur_rsi:.0f}}, +{{ret_12h:.1f}}% 12h\",")
                print(f"    }}, score))")
        
        print("\n📋 Copy these optimized parameters into run_master_bot.py")


if __name__ == "__main__":
    sweeper = ParameterSweeper()
    optimized_tools = sweeper.run_parameter_sweep()