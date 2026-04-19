#!/usr/bin/env python3
"""
Bull Market Tools Validation - Regime Aware Testing

This validates bull market tools ONLY during actual bull regime periods,
not across all market conditions like previous tests.

Key insight: Tools should only fire when F&G is in greed territory.
We simulate this with: 30-day BTC return > 8% AND BTC above 50-period SMA.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RegimeAwareTester:
    def __init__(self, data_path="/Users/lucasaust/code/Crypto-trading-bot/data/binance_1h_extended"):
        self.data_path = data_path
        self.pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'LINKUSDT', 
                     'DOTUSDT', 'AVAXUSDT', 'ATOMUSDT', 'AAVEUSDT', 'UNIUSDT',
                     'LTCUSDT', 'XRPUSDT', 'DOGEUSDT', 'FILUSDT', 'XLMUSDT', 'NEARUSDT']
        self.data = {}
        self.btc_data = None
        self.load_data()
        
    def load_data(self):
        """Load all pair data and prepare BTC regime detection"""
        print("Loading extended dataset...")
        for pair in self.pairs:
            file_path = f"{self.data_path}/{pair}_1h.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df = df.sort_index()
                
                # Add indicators
                self.add_indicators(df)
                self.data[pair] = df
                
                if pair == 'BTCUSDT':
                    self.btc_data = df
                    
        print(f"Loaded {len(self.data)} pairs")
        print(f"Date range: {self.btc_data.index[0]} to {self.btc_data.index[-1]}")
        print(f"Total bars per pair: ~{len(self.btc_data)}")

    def add_indicators(self, df):
        """Add technical indicators"""
        # EMAs/SMAs
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Returns
        df['ret_1d'] = df['close'].pct_change(24)
        df['ret_2d'] = df['close'].pct_change(48)
        df['ret_7d'] = df['close'].pct_change(168)
        df['ret_30d'] = df['close'].pct_change(720)
        
        # Rolling stats
        df['high_1w'] = df['high'].rolling(168).max()
        df['mean_4w'] = df['close'].rolling(672).mean()
        df['std_4w'] = df['close'].rolling(672).std()
        
        return df

    def identify_bull_regime(self):
        """Identify bull regime bars using BTC 30-day return > 8% AND price > 50 SMA"""
        if self.btc_data is None:
            raise ValueError("BTC data not loaded")
            
        bull_bars = (
            (self.btc_data['ret_30d'] > 0.08) &  # 30-day return > 8%
            (self.btc_data['close'] > self.btc_data['sma_50'])  # Above 50 SMA
        )
        
        bull_periods = bull_bars[bull_bars].index
        print(f"Bull regime bars identified: {len(bull_periods)} out of {len(self.btc_data)} ({100*len(bull_periods)/len(self.btc_data):.1f}%)")
        
        # Show major bull periods
        bull_months = bull_periods.to_period('M').value_counts().sort_index()
        print("Bull periods by month:")
        for month, count in bull_months.items():
            if count > 100:  # Show months with significant bull activity
                price_start = self.btc_data.loc[self.btc_data.index.to_period('M') == month]['close'].iloc[0]
                price_end = self.btc_data.loc[self.btc_data.index.to_period('M') == month]['close'].iloc[-1]
                print(f"  {month}: {count} bars, BTC ${price_start:.0f}→${price_end:.0f} ({100*(price_end/price_start-1):+.1f}%)")
        
        return bull_periods

    def calc_hurst(self, series, max_lag=100):
        """Calculate Hurst exponent for trend detection"""
        if len(series) < max_lag:
            return 0.5
            
        lags = range(10, min(max_lag, len(series)//2), 5)
        rs_values, lag_values = [], []
        
        for lag in lags:
            if lag >= len(series):
                continue
            sub = series[-lag:]
            if len(sub) < 10:
                continue
                
            mean = np.mean(sub)
            dev = np.cumsum(sub - mean)
            R = np.max(dev) - np.min(dev)
            S = np.std(sub)
            
            if S > 0 and R > 0:
                rs_values.append(np.log(R/S))
                lag_values.append(np.log(lag))
                
        if len(rs_values) < 3:
            return 0.5
            
        slope, _, _, _, _ = stats.linregress(lag_values, rs_values)
        return np.clip(slope, 0, 1)

    def trend_structure_long(self, df, i):
        """Higher highs + higher lows over 4 weekly windows, buy RSI dip"""
        if i < 672:  # Need 4 weeks of data
            return False
            
        # Check for higher highs and higher lows over 4 weekly windows
        weekly_highs = []
        weekly_lows = []
        
        for week in range(4):
            start_idx = i - (week + 1) * 168
            end_idx = i - week * 168
            if start_idx < 0:
                return False
                
            week_high = df['high'].iloc[start_idx:end_idx].max()
            week_low = df['low'].iloc[start_idx:end_idx].min()
            weekly_highs.append(week_high)
            weekly_lows.append(week_low)
        
        # Check if we have ascending highs and lows (most recent first)
        weekly_highs.reverse()
        weekly_lows.reverse()
        
        higher_highs = all(weekly_highs[i] >= weekly_highs[i-1] for i in range(1, 4))
        higher_lows = all(weekly_lows[i] >= weekly_lows[i-1] for i in range(1, 4))
        
        if not (higher_highs and higher_lows):
            return False
            
        # Buy on RSI dip within structure
        current_rsi = df['rsi'].iloc[i]
        return 25 <= current_rsi <= 45

    def weekly_momentum_pullback(self, df, i):
        """Strong 2-4 week uptrend + 3-15% pullback + RSI 30-55"""
        if i < 672:
            return False
            
        # Strong 2-4 week uptrend
        ret_2w = df['close'].iloc[i] / df['close'].iloc[i-336] - 1
        ret_4w = df['close'].iloc[i] / df['close'].iloc[i-672] - 1
        
        if not (ret_2w > 0.15 and ret_4w > 0.25):  # Strong uptrend
            return False
            
        # 3-15% pullback from recent high
        recent_high = df['high'].iloc[i-168:i].max()  # 1 week lookback
        current_price = df['close'].iloc[i]
        pullback = (recent_high - current_price) / recent_high
        
        if not (0.03 <= pullback <= 0.15):
            return False
            
        # Price still above 200h SMA
        if current_price <= df['sma_200'].iloc[i]:
            return False
            
        # RSI in buy zone
        current_rsi = df['rsi'].iloc[i]
        return 30 <= current_rsi <= 55

    def golden_cross_swing(self, df, i):
        """50h EMA crosses above 200h EMA with momentum confirmation"""
        if i < 400:
            return False
            
        # Golden cross: 50 EMA crossed above 200 EMA recently
        ema_50_now = df['ema_50'].iloc[i]
        ema_200_now = df['ema_200'].iloc[i]
        ema_50_prev = df['ema_50'].iloc[i-24]  # 1 day ago
        ema_200_prev = df['ema_200'].iloc[i-24]
        
        # Check if cross happened in last 7 days
        cross_happened = False
        for j in range(1, 168):  # Check last week
            if i - j < 200:
                break
            ema_50_then = df['ema_50'].iloc[i-j]
            ema_200_then = df['ema_200'].iloc[i-j]
            ema_50_before = df['ema_50'].iloc[i-j-1] if i-j-1 >= 0 else 0
            ema_200_before = df['ema_200'].iloc[i-j-1] if i-j-1 >= 0 else 0
            
            if (ema_50_before <= ema_200_before and ema_50_then > ema_200_then):
                cross_happened = True
                break
                
        if not cross_happened:
            return False
            
        # Currently 50 > 200
        if ema_50_now <= ema_200_now:
            return False
            
        # Momentum confirmation: positive 7-day return
        ret_7d = df['close'].iloc[i] / df['close'].iloc[i-168] - 1
        
        return ret_7d > 0.05

    def accumulation_breakout(self, df, i):
        """2-4 week consolidation -> volume breakout"""
        if i < 672:
            return False
            
        # Check for consolidation period (2-4 weeks)
        lookback_2w = df['close'].iloc[i-336:i]
        lookback_4w = df['close'].iloc[i-672:i]
        
        high_4w = lookback_4w.max()
        low_4w = lookback_4w.min()
        range_4w = (high_4w - low_4w) / low_4w
        
        # Consolidation: 3-20% range over 4 weeks
        if not (0.03 <= range_4w <= 0.20):
            return False
            
        # Breakout: price breaks above 2-week high
        high_2w = lookback_2w.max()
        current_price = df['close'].iloc[i]
        
        if current_price <= high_2w * 1.02:  # 2% breakout threshold
            return False
            
        # Volume surge (if volume data available)
        try:
            avg_vol_2w = df['volume'].iloc[i-336:i].mean()
            current_vol = df['volume'].iloc[i]
            volume_surge = current_vol > avg_vol_2w * 1.5
        except:
            volume_surge = True  # Assume volume is good if not available
            
        return volume_surge

    def dip_in_uptrend(self, df, i):
        """Positive 4-week return + 5-25% drawdown + RSI < 40 + above 200h SMA"""
        if i < 720:
            return False
            
        # Positive 4-week return
        ret_4w = df['close'].iloc[i] / df['close'].iloc[i-672] - 1
        if ret_4w <= 0.1:  # Need at least 10% gain over 4 weeks
            return False
            
        # 5-25% drawdown from 1-week high
        high_1w = df['high'].iloc[i-168:i].max()
        current_price = df['close'].iloc[i]
        drawdown = (high_1w - current_price) / high_1w
        
        if not (0.05 <= drawdown <= 0.25):
            return False
            
        # Still above 200h SMA
        if current_price <= df['sma_200'].iloc[i]:
            return False
            
        # Oversold RSI
        current_rsi = df['rsi'].iloc[i]
        return current_rsi < 40

    def mean_reversion_weekly(self, df, i):
        """Price > 1.5 std below 4-week mean + not crashing + RSI < 40"""
        if i < 672:
            return False
            
        current_price = df['close'].iloc[i]
        mean_4w = df['mean_4w'].iloc[i]
        std_4w = df['std_4w'].iloc[i]
        
        if pd.isna(mean_4w) or pd.isna(std_4w) or std_4w == 0:
            return False
            
        # Price below 1.5 standard deviations
        z_score = (current_price - mean_4w) / std_4w
        if z_score > -1.5:
            return False
            
        # Not still crashing (48h return > -10%)
        ret_48h = df['close'].iloc[i] / df['close'].iloc[i-48] - 1
        if ret_48h <= -0.10:
            return False
            
        # Oversold RSI
        current_rsi = df['rsi'].iloc[i]
        return current_rsi < 40

    def hurst_trend_long(self, df, i):
        """Hurst exponent > 0.6 (trending) + momentum + volume"""
        if i < 500:
            return False
            
        # Calculate Hurst exponent on recent price series
        price_series = np.log(df['close'].iloc[i-200:i]).values
        hurst = self.calc_hurst(price_series)
        
        if hurst <= 0.6:  # Need trending market
            return False
            
        # Positive momentum
        ret_7d = df['close'].iloc[i] / df['close'].iloc[i-168] - 1
        if ret_7d <= 0.02:
            return False
            
        # Above key moving average
        current_price = df['close'].iloc[i]
        return current_price > df['ema_50'].iloc[i]

    def ou_mean_reversion_long(self, df, i):
        """Ornstein-Uhlenbeck z-score < -1.5 in mean-reverting market"""
        if i < 500:
            return False
            
        # Calculate OU parameters on recent data
        price_series = np.log(df['close'].iloc[i-200:i]).values
        
        # Simple OU parameter estimation
        prices = price_series[:-1]
        price_changes = np.diff(price_series)
        
        if len(prices) != len(price_changes):
            return False
            
        try:
            # Linear regression: dX = alpha * (theta - X) * dt + sigma * dW
            # Approximated as: dX = a + b*X + error
            slope, intercept, r_value, p_value, std_err = stats.linregress(prices, price_changes)
            
            # OU mean reversion strength
            mean_reversion_strength = -slope
            if mean_reversion_strength <= 0.01:  # Need some mean reversion
                return False
                
            # Calculate current z-score relative to long-term mean
            long_term_mean = np.mean(price_series)
            long_term_std = np.std(price_series)
            
            if long_term_std == 0:
                return False
                
            current_z = (price_series[-1] - long_term_mean) / long_term_std
            
            # Buy when deeply oversold in mean-reverting regime
            return current_z < -1.5
            
        except:
            return False

    def backtest_tool(self, tool_func, tool_name, bull_periods, step_size=8):
        """Backtest a tool only during bull regime periods"""
        print(f"\n=== Testing {tool_name} ===")
        
        all_trades = []
        
        for pair in self.pairs:
            if pair not in self.data:
                continue
                
            df = self.data[pair]
            pair_trades = []
            
            # Only test on bars that align with bull periods (using BTC regime)
            test_indices = []
            for i in range(720, len(df), step_size):  # Start after warmup, step by 8
                current_time = df.index[i]
                
                # Check if this time aligns with a bull regime period
                if current_time in bull_periods:
                    test_indices.append(i)
            
            print(f"{pair}: Testing {len(test_indices)} bull regime bars")
            
            for i in test_indices:
                if tool_func(df, i):
                    # Entry
                    entry_price = df['close'].iloc[i]
                    entry_time = df.index[i]
                    
                    # Find exit using trailing stop
                    exit_price, exit_time, exit_reason = self.find_exit(df, i, entry_price)
                    
                    if exit_price is None:
                        continue
                        
                    # Calculate trade metrics
                    gross_return = (exit_price / entry_price) - 1
                    net_return = gross_return - 0.0065  # 0.65% round-trip fees
                    
                    trade = {
                        'pair': pair,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'gross_return': gross_return,
                        'net_return': net_return,
                        'exit_reason': exit_reason,
                        'hold_hours': (exit_time - entry_time).total_seconds() / 3600
                    }
                    
                    pair_trades.append(trade)
                    all_trades.extend(pair_trades)
            
        return self.analyze_results(all_trades, tool_name)

    def find_exit(self, df, entry_idx, entry_price):
        """Find exit using 8% trailing stop, 12% hard stop, 336h max hold"""
        max_hold_hours = 336
        trail_pct = 0.08
        hard_stop_pct = 0.12
        
        highest_price = entry_price
        trail_stop = entry_price * (1 - trail_pct)
        hard_stop = entry_price * (1 - hard_stop_pct)
        
        for i in range(entry_idx + 1, min(entry_idx + max_hold_hours + 1, len(df))):
            current_price = df['close'].iloc[i]
            current_time = df.index[i]
            
            # Update trailing stop
            if current_price > highest_price:
                highest_price = current_price
                trail_stop = highest_price * (1 - trail_pct)
            
            # Check exits
            if current_price <= hard_stop:
                return current_price, current_time, "hard_stop"
            elif current_price <= trail_stop:
                return current_price, current_time, "trail_stop"
        
        # Max hold reached
        if entry_idx + max_hold_hours < len(df):
            final_price = df['close'].iloc[entry_idx + max_hold_hours]
            final_time = df.index[entry_idx + max_hold_hours]
            return final_price, final_time, "max_hold"
        
        return None, None, None

    def analyze_results(self, trades, tool_name):
        """Analyze backtest results"""
        if not trades:
            return {
                'tool_name': tool_name,
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'profit_factor': 0,
                'max_dd': 0,
                'sharpe': 0,
                'passed': False,
                'reason': 'No signals generated'
            }
        
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(trades)
        wins = len(df_trades[df_trades['net_return'] > 0])
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        avg_return = df_trades['net_return'].mean()
        
        # Profit factor
        gross_gains = df_trades[df_trades['net_return'] > 0]['net_return'].sum()
        gross_losses = abs(df_trades[df_trades['net_return'] < 0]['net_return'].sum())
        profit_factor = gross_gains / gross_losses if gross_losses > 0 else float('inf')
        
        # Max drawdown
        cumulative = (1 + df_trades['net_return']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        max_dd = drawdown.min()
        
        # Sharpe approximation
        sharpe = (avg_return * np.sqrt(8760)) / df_trades['net_return'].std() if df_trades['net_return'].std() > 0 else 0
        
        # Pass criteria: 30+ trades AND (win_rate > 55% OR profit_factor > 1.5)
        passed = (total_trades >= 30) and (win_rate > 0.55 or profit_factor > 1.5)
        
        reason = ""
        if total_trades < 30:
            reason = f"Insufficient signals ({total_trades} < 30)"
        elif win_rate <= 0.55 and profit_factor <= 1.5:
            reason = f"Poor performance (WR: {win_rate:.1%}, PF: {profit_factor:.2f})"
        else:
            reason = "PASSED validation criteria"
        
        results = {
            'tool_name': tool_name,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'profit_factor': profit_factor,
            'max_dd': max_dd,
            'sharpe': sharpe,
            'passed': passed,
            'reason': reason
        }
        
        print(f"Results for {tool_name}:")
        print(f"  Signals: {total_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Return: {avg_return:.2%}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Max DD: {max_dd:.2%}")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Status: {reason}")
        
        return results

def main():
    """Run the regime-aware validation"""
    print("🐂 BULL MARKET TOOLS VALIDATION - REGIME AWARE 🐂")
    print("=" * 60)
    
    tester = RegimeAwareTester()
    
    # Identify bull regime periods
    bull_periods = tester.identify_bull_regime()
    
    # Test all tools
    tools_to_test = [
        (tester.trend_structure_long, "trend_structure_long"),
        (tester.weekly_momentum_pullback, "weekly_momentum_pullback"),
        (tester.golden_cross_swing, "golden_cross_swing"),
        (tester.accumulation_breakout, "accumulation_breakout"),
        (tester.dip_in_uptrend, "dip_in_uptrend"),
        (tester.mean_reversion_weekly, "mean_reversion_weekly"),
        (tester.hurst_trend_long, "hurst_trend_long"),
        (tester.ou_mean_reversion_long, "ou_mean_reversion_long"),
    ]
    
    all_results = []
    
    for tool_func, tool_name in tools_to_test:
        try:
            result = tester.backtest_tool(tool_func, tool_name, bull_periods)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR testing {tool_name}: {e}")
            all_results.append({
                'tool_name': tool_name,
                'total_trades': 0,
                'passed': False,
                'reason': f'Error: {str(e)}'
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed_tools = [r for r in all_results if r['passed']]
    
    print(f"Tools tested: {len(all_results)}")
    print(f"Tools passed: {len(passed_tools)}")
    
    if passed_tools:
        print(f"\n✅ PASSING TOOLS:")
        for tool in passed_tools:
            print(f"  {tool['tool_name']}: {tool['total_trades']} signals, "
                  f"{tool['win_rate']:.1%} WR, PF {tool['profit_factor']:.2f}")
    else:
        print(f"\n❌ NO TOOLS PASSED VALIDATION")
        
    print(f"\n📊 ALL RESULTS:")
    for tool in all_results:
        status = "✅" if tool['passed'] else "❌"
        print(f"  {status} {tool['tool_name']}: {tool['reason']}")
    
    # Save results
    with open('/Users/lucasaust/code/Crypto-trading-bot/regime_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results, passed_tools

if __name__ == "__main__":
    results, survivors = main()