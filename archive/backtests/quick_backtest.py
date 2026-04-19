#!/usr/bin/env python3
"""
QUICK BACKTEST - Focused validation of key tools with progress tracking
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BINANCE_DATA_DIR = DATA_DIR / "binance_1h"

# Trading parameters
KRAKEN_TAKER_FEE = 0.0026  # 0.26% each way = 0.52% round-trip
STARTING_BALANCE = 300

# Pair mappings (focus on top 5 pairs for quick test)
PAIR_MAPPING = {
    "XBTUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT",
    "NEARUSD": "NEARUSDT", "AVAXUSD": "AVAXUSDT"
}

print("🦅 QUICK BACKTEST - Key Tool Validation")
print("="*50)

class QuickBacktest:
    
    def __init__(self):
        self.data = {}
        self.results = {}
        
    def load_data(self):
        """Load top 5 pairs"""
        print("📊 Loading data...")
        
        for kraken_pair, binance_pair in PAIR_MAPPING.items():
            file_path = BINANCE_DATA_DIR / f"{binance_pair}_1h.csv"
            if not file_path.exists():
                print(f"❌ Missing: {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.data[kraken_pair] = df
            print(f"✅ {kraken_pair:8}: {len(df):,} bars")
            
        print(f"📈 Loaded {len(self.data)} pairs")
        
    def calculate_simple_indicators(self, df):
        """Calculate basic indicators efficiently"""
        close = df['close'].values.astype(float)
        
        # Simple RSI-7
        def simple_rsi(prices, period=7):
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.full(len(prices), np.nan)
            avg_loss = np.full(len(prices), np.nan)
            
            if len(delta) >= period:
                avg_gain[period] = np.mean(gain[:period])
                avg_loss[period] = np.mean(loss[:period])
                
                for i in range(period + 1, min(len(prices), period + 1000)):  # Limit calculation
                    avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                    avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
            
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
            return 100 - (100 / (1 + rs))
        
        rsi = simple_rsi(close, 7)
        
        # Simple SMA50
        sma50 = np.full(len(close), np.nan)
        if len(close) >= 50:
            for i in range(49, min(len(close), 2000)):  # Limit calculation
                sma50[i] = np.mean(close[i-49:i+1])
        
        return {
            'close': close,
            'rsi': rsi,
            'sma50': sma50
        }
    
    def test_key_tools(self):
        """Test the most important tools quickly"""
        print("\n🔍 Testing key tools...")
        
        key_tools = {
            'crash_buy': lambda ret_24h, rsi: ret_24h < -10 and rsi < 20,
            'dip_buy': lambda ret_4h, rsi: ret_4h < -3,
            'rsi_pump_short': lambda ret_12h, rsi: rsi > 85 and ret_12h >= 10,
            'mega_crash': lambda ret_24h, rsi: ret_24h < -15,
        }
        
        results = {}
        
        for tool_name, condition_func in key_tools.items():
            print(f"\n🧪 Testing: {tool_name}")
            
            all_signals = []
            
            # Test on each pair
            for pair, df in self.data.items():
                print(f"   Scanning {pair}...", end='')
                
                indicators = self.calculate_simple_indicators(df)
                close = indicators['close']
                rsi = indicators['rsi']
                
                signals_found = 0
                
                # Scan every 24th bar for speed (daily instead of hourly)
                for i in range(50, len(df) - 50, 24):
                    price = close[i]
                    cur_rsi = rsi[i] if not np.isnan(rsi[i]) else 50
                    
                    # Calculate returns
                    ret_4h = (price - close[i-5]) / close[i-5] * 100 if i >= 5 else 0
                    ret_12h = (price - close[i-13]) / close[i-13] * 100 if i >= 13 else 0
                    ret_24h = (price - close[i-25]) / close[i-25] * 100 if i >= 25 else 0
                    
                    # Check tool condition
                    signal_triggered = False
                    
                    if tool_name == 'crash_buy':
                        signal_triggered = condition_func(ret_24h, cur_rsi)
                    elif tool_name == 'dip_buy':
                        signal_triggered = condition_func(ret_4h, cur_rsi)
                    elif tool_name == 'rsi_pump_short':
                        signal_triggered = condition_func(ret_12h, cur_rsi)
                    elif tool_name == 'mega_crash':
                        signal_triggered = condition_func(ret_24h, cur_rsi)
                    
                    if signal_triggered:
                        # Calculate forward returns (8h and 24h)
                        returns_8h = None
                        returns_24h = None
                        
                        if i + 8 < len(close):
                            exit_price_8h = close[i + 8]
                            gross_return_8h = (exit_price_8h - price) / price
                            returns_8h = (gross_return_8h - KRAKEN_TAKER_FEE * 2) * 100
                            
                        if i + 24 < len(close):
                            exit_price_24h = close[i + 24]
                            gross_return_24h = (exit_price_24h - price) / price
                            returns_24h = (gross_return_24h - KRAKEN_TAKER_FEE * 2) * 100
                        
                        all_signals.append({
                            'pair': pair,
                            'bar': i,
                            'return_8h': returns_8h,
                            'return_24h': returns_24h,
                            'ret_24h_trigger': ret_24h,
                            'rsi_trigger': cur_rsi
                        })
                        
                        signals_found += 1
                
                print(f" {signals_found} signals")
            
            # Analyze results
            if all_signals:
                valid_8h = [s['return_8h'] for s in all_signals if s['return_8h'] is not None]
                valid_24h = [s['return_24h'] for s in all_signals if s['return_24h'] is not None]
                
                def calc_stats(returns):
                    if not returns:
                        return 0, 0, 0
                    wins = [r for r in returns if r > 0]
                    win_rate = len(wins) / len(returns) * 100
                    avg_return = np.mean(returns)
                    return len(returns), win_rate, avg_return
                
                count_8h, wr_8h, avg_8h = calc_stats(valid_8h)
                count_24h, wr_24h, avg_24h = calc_stats(valid_24h)
                
                results[tool_name] = {
                    'total_signals': len(all_signals),
                    '8h_count': count_8h,
                    '8h_win_rate': wr_8h,
                    '8h_avg_return': avg_8h,
                    '24h_count': count_24h,
                    '24h_win_rate': wr_24h,
                    '24h_avg_return': avg_24h,
                    'passes': avg_8h > 0 or avg_24h > 0
                }
                
                status = "✅ PASS" if results[tool_name]['passes'] else "❌ FAIL"
                print(f"   📊 {len(all_signals)} signals, 8h: {wr_8h:.1f}% WR {avg_8h:+.2f}% avg, 24h: {wr_24h:.1f}% WR {avg_24h:+.2f}% avg - {status}")
            else:
                results[tool_name] = {'total_signals': 0, 'passes': False}
                print(f"   ❌ No signals found")
        
        self.results = results
        
        # Summary
        print(f"\n📊 SUMMARY:")
        passing = sum(1 for r in results.values() if r.get('passes', False))
        print(f"   Passing tools: {passing}/{len(results)}")
        
        for tool, result in results.items():
            if result.get('passes', False):
                print(f"   ✅ {tool:15} - {result['8h_avg_return']:+.2f}% avg (8h)")
            else:
                print(f"   ❌ {tool:15} - {result.get('8h_avg_return', 0):+.2f}% avg (8h)")
    
    def run_quick_portfolio_sim(self):
        """Quick portfolio simulation with passing tools"""
        print(f"\n💰 Quick Portfolio Simulation")
        
        passing_tools = [name for name, result in self.results.items() if result.get('passes', False)]
        
        if not passing_tools:
            print("❌ No passing tools for simulation")
            return
        
        print(f"🔧 Using {len(passing_tools)} validated tools: {passing_tools}")
        
        # Simplified simulation on BTCUSDT only
        if "XBTUSD" not in self.data:
            print("❌ No BTC data for simulation")
            return
            
        df = self.data["XBTUSD"]
        indicators = self.calculate_simple_indicators(df)
        close = indicators['close']
        rsi = indicators['rsi']
        
        balance = STARTING_BALANCE
        positions = []
        trades_completed = 0
        total_pnl = 0
        
        # Simulate daily (every 24 bars)
        for i in range(100, len(df) - 100, 24):
            price = close[i]
            cur_rsi = rsi[i] if not np.isnan(rsi[i]) else 50
            
            # Calculate returns
            ret_4h = (price - close[i-5]) / close[i-5] * 100 if i >= 5 else 0
            ret_12h = (price - close[i-13]) / close[i-13] * 100 if i >= 13 else 0  
            ret_24h = (price - close[i-25]) / close[i-25] * 100 if i >= 25 else 0
            
            # Check for signals from passing tools
            signal_triggered = False
            
            if 'crash_buy' in passing_tools and ret_24h < -10 and cur_rsi < 20:
                signal_triggered = True
                direction = 'long'
                hold_period = 24
                
            elif 'dip_buy' in passing_tools and ret_4h < -3:
                signal_triggered = True
                direction = 'long'
                hold_period = 8
                
            elif 'rsi_pump_short' in passing_tools and cur_rsi > 85 and ret_12h >= 10:
                signal_triggered = True
                direction = 'short'
                hold_period = 8
                
            elif 'mega_crash' in passing_tools and ret_24h < -15:
                signal_triggered = True
                direction = 'long' 
                hold_period = 24
            
            # Open new position
            if signal_triggered and len(positions) == 0:  # Max 1 position for simplicity
                position_size = balance * 0.05  # 5% risk
                positions.append({
                    'entry_bar': i,
                    'entry_price': price,
                    'direction': direction,
                    'hold_period': hold_period,
                    'position_size': position_size
                })
            
            # Check for exits
            positions_to_close = []
            for pos in positions:
                bars_held = i - pos['entry_bar']
                
                # Exit after hold period
                if bars_held >= pos['hold_period']:
                    if pos['direction'] == 'long':
                        gross_return = (price - pos['entry_price']) / pos['entry_price']
                    else:
                        gross_return = (pos['entry_price'] - price) / pos['entry_price']
                    
                    net_return = gross_return - (KRAKEN_TAKER_FEE * 2)
                    trade_pnl = net_return * pos['position_size']
                    
                    total_pnl += trade_pnl
                    trades_completed += 1
                    positions_to_close.append(pos)
            
            # Remove closed positions
            for pos in positions_to_close:
                positions.remove(pos)
        
        # Results
        final_balance = balance + total_pnl
        total_return = (final_balance - balance) / balance * 100
        
        print(f"💰 PORTFOLIO RESULTS:")
        print(f"   Starting: ${balance:.0f}")
        print(f"   Final:    ${final_balance:.0f}")
        print(f"   Return:   {total_return:+.1f}%")
        print(f"   Trades:   {trades_completed}")
        
    def run(self):
        """Run the quick backtest"""
        self.load_data()
        self.test_key_tools()
        self.run_quick_portfolio_sim()
        
        print(f"\n🎉 Quick backtest complete!")

if __name__ == "__main__":
    backtest = QuickBacktest()
    backtest.run()