#!/usr/bin/env python3
"""
Out-of-Sample Validation for REMAINING Trading Tools
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
        self.results = []
        
        # Load all available pairs
        print("Loading Binance real data...")
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('_4h.csv'):
                pair = f.replace('_4h.csv', '')
                self.pairs.append(pair)
                df = pd.read_csv(f"{data_dir}/{f}")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data_cache[pair] = df
                
        print(f"Loaded {len(self.pairs)} pairs: {', '.join(self.pairs)}")
        print(f"Data range: {self.data_cache[self.pairs[0]]['timestamp'].iloc[0]} to {self.data_cache[self.pairs[0]]['timestamp'].iloc[-1]}")
        
        # Verify all pairs have same length
        lengths = [len(self.data_cache[pair]) for pair in self.pairs]
        if len(set(lengths)) > 1:
            print(f"WARNING: Pairs have different lengths: {dict(zip(self.pairs, lengths))}")
        else:
            print(f"All pairs have {lengths[0]} bars each")
    
    # ---- HELPER FUNCTIONS (exact copies from master bot) ----
    
    def calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
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
        
        # Pad with initial value
        result = np.full(len(prices), 50.0)
        result[period:] = rsi[period-1:]
        return result
    
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        result = np.full(len(prices), np.nan)
        for i in range(period-1, len(prices)):
            result[i] = np.mean(prices[i-period+1:i+1])
        return result
    
    def calc_bollinger(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands."""
        sma = self.calc_sma(prices, period)
        std = np.full(len(prices), np.nan)
        
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        bandwidth = (upper - lower) / sma  # Normalized bandwidth
        
        return sma, upper, lower, bandwidth
    
    # Math helper functions (exact from run_master_bot.py lines ~714-755)
    def hurst(self, returns, w=50):
        """Hurst exponent (variance ratio method)"""
        if len(returns) < w:
            return 0.5
        r2 = returns[-w:]
        v1 = np.var(r2)
        v2 = np.var(r2[::2]) if len(r2) >= 4 else v1
        if v1 <= 0 or v2 <= 0:
            return 0.5
        vr = v2 / v1
        return max(0, min(1, 0.5 + np.log(max(vr, 0.01)) / (2 * np.log(2))))
    
    def entropy(self, returns, bins=15):
        """Shannon entropy"""
        if len(returns) < 10:
            return 3.0
        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 3.0
        probs = hist / hist.sum()
        return -np.sum(probs * np.log2(probs))
    
    def autocorr(self, returns, lag=1):
        """Autocorrelation"""
        if len(returns) < lag + 5:
            return 0
        return float(pd.Series(returns).autocorr(lag=lag))
    
    def vpin(self, close_arr, vol_arr, w=20):
        """VPIN proxy"""
        if len(close_arr) < w + 1:
            return 0
        rets = np.diff(close_arr) / close_arr[:-1]
        bv = np.where(rets > 0, vol_arr[1:], 0)
        sv = np.where(rets < 0, vol_arr[1:], 0)
        rb = np.sum(bv[-w:])
        rs = np.sum(sv[-w:])
        t = rb + rs
        return abs(rb - rs) / t if t > 0 else 0
    
    # ---- VALIDATION LOGIC ----
    
    def validate_tools(self):
        """Run validation on all remaining tools."""
        print("\nStarting validation of remaining tools...")
        
        # Get the minimum data length across all pairs
        min_length = min(len(self.data_cache[pair]) for pair in self.pairs)
        
        # Start validation from bar 100 (need history for indicators)
        start_bar = 100
        end_bar = min_length - 10  # Leave buffer for forward returns
        
        signals = []
        
        for bar_idx in range(start_bar, end_bar):
            if bar_idx % 500 == 0:
                print(f"Processing bar {bar_idx}/{end_bar}...")
            
            # For cross-pair tools, we need ALL pairs data at this timestamp
            all_pair_data = {}
            for pair in self.pairs:
                df = self.data_cache[pair]
                if bar_idx >= len(df):
                    continue
                all_pair_data[pair] = {
                    'df': df.iloc[:bar_idx+1].copy(),
                    'close': df['close'].iloc[:bar_idx+1].values.astype(float),
                    'open': df['open'].iloc[:bar_idx+1].values.astype(float),
                    'high': df['high'].iloc[:bar_idx+1].values.astype(float), 
                    'low': df['low'].iloc[:bar_idx+1].values.astype(float),
                    'volume': df['volume'].iloc[:bar_idx+1].values.astype(float),
                    'timestamp': df['timestamp'].iloc[bar_idx]
                }
            
            # Skip if we don't have data for all pairs
            if len(all_pair_data) != len(self.pairs):
                continue
                
            # Generate signals for this bar
            bar_signals = self.scan_remaining_tools(all_pair_data, bar_idx)
            signals.extend(bar_signals)
        
        print(f"\nGenerated {len(signals)} total signals")
        
        # Calculate forward returns and analyze performance
        self.analyze_signals(signals)
        
        # Generate report
        self.generate_report()
    
    def scan_remaining_tools(self, all_pair_data, bar_idx):
        """Scan for signals from remaining tools at given bar."""
        signals = []
        
        # ---- CROSS-PAIR TOOLS ----
        
        # Get market-wide statistics
        total_pairs = len(all_pair_data)
        btc_data = all_pair_data.get('BTCUSDT')
        eth_data = all_pair_data.get('ETHUSDT')
        
        # Tool 13: btc_eth_diverge
        if btc_data and eth_data and len(btc_data['close']) >= 25:
            btc_ret_24h = (btc_data['close'][-1] - btc_data['close'][-25]) / btc_data['close'][-25] * 100
            eth_ret_24h = (eth_data['close'][-1] - eth_data['close'][-25]) / eth_data['close'][-25] * 100
            if btc_ret_24h - eth_ret_24h >= 3.0:  # BTC outperforms ETH by 3%+
                signals.append(self.create_signal(
                    'ETHUSDT', 'btc_eth_diverge', 'short', bar_idx, 24, 0.05,
                    f"BTC outperforms ETH: BTC +{btc_ret_24h:.1f}%, ETH +{eth_ret_24h:.1f}%"
                ))
        
        # Tool 16: market_panic (3 tiers: 90%, 80%, 70%)
        if total_pairs >= 5:
            dropping_3pct = 0
            for pair, data in all_pair_data.items():
                if len(data['close']) >= 5:
                    ret_4h = (data['close'][-1] - data['close'][-5]) / data['close'][-5] * 100
                    if ret_4h < -3.0:
                        dropping_3pct += 1
            
            drop_pct = dropping_3pct / total_pairs
            if drop_pct >= 0.9:
                # 90% tier
                for pair in all_pair_data.keys():
                    signals.append(self.create_signal(
                        pair, 'market_panic_90', 'long', bar_idx, 24, 0.06,
                        f"Market panic: {drop_pct*100:.0f}% coins dropping >3%"
                    ))
            elif drop_pct >= 0.8:
                # 80% tier
                for pair in all_pair_data.keys():
                    signals.append(self.create_signal(
                        pair, 'market_panic_80', 'long', bar_idx, 24, 0.05,
                        f"Market panic: {drop_pct*100:.0f}% coins dropping >3%"
                    ))
            elif drop_pct >= 0.7:
                # 70% tier
                for pair in all_pair_data.keys():
                    signals.append(self.create_signal(
                        pair, 'market_panic_70', 'long', bar_idx, 16, 0.04,
                        f"Market panic: {drop_pct*100:.0f}% coins dropping >3%"
                    ))
        
        # Tool 21: blood_in_streets
        if total_pairs >= 5:
            dropping_2pct = 0
            for pair, data in all_pair_data.items():
                if len(data['close']) >= 5:
                    ret_4h = (data['close'][-1] - data['close'][-5]) / data['close'][-5] * 100
                    if ret_4h < -2.0:
                        dropping_2pct += 1
            
            if dropping_2pct / total_pairs >= 0.7:  # 70%+ of coins down >2%
                for pair, data in all_pair_data.items():
                    if len(data['close']) >= 15:
                        rsi = self.calc_rsi(data['close'][-15:], 7)
                        if rsi[-1] < 20:  # Individual coin oversold
                            signals.append(self.create_signal(
                                pair, 'blood_in_streets', 'long', bar_idx, 24, 0.06,
                                f"Blood in streets: {dropping_2pct/total_pairs*100:.0f}% panic + RSI={rsi[-1]:.1f}"
                            ))
        
        # Tool 22: fomo_ride
        if total_pairs >= 5:
            pumping_1pct = 0
            for pair, data in all_pair_data.items():
                if len(data['close']) >= 5:
                    ret_4h = (data['close'][-1] - data['close'][-5]) / data['close'][-5] * 100
                    if ret_4h > 1.0:
                        pumping_1pct += 1
            
            if pumping_1pct / total_pairs >= 0.8:  # 80%+ coins pumping >1%
                for pair in all_pair_data.keys():
                    signals.append(self.create_signal(
                        pair, 'fomo_ride', 'long', bar_idx, 8, 0.03,
                        f"FOMO ride: {pumping_1pct/total_pairs*100:.0f}% coins pumping"
                    ))
        
        # Tool 33: btc_alt_spread
        if btc_data and len(btc_data['close']) >= 5:
            btc_ret_4h = (btc_data['close'][-1] - btc_data['close'][-5]) / btc_data['close'][-5] * 100
            for pair, data in all_pair_data.items():
                if pair != 'BTCUSDT' and len(data['close']) >= 15:
                    alt_ret_4h = (data['close'][-1] - data['close'][-5]) / data['close'][-5] * 100
                    spread = alt_ret_4h - btc_ret_4h
                    if spread < -3.0:  # Alt lagging BTC by 3%+
                        rsi = self.calc_rsi(data['close'], 14)
                        if rsi[-1] < 35:  # AND RSI < 35
                            signals.append(self.create_signal(
                                pair, 'btc_alt_spread', 'long', bar_idx, 24, 0.05,
                                f"Alt lagging BTC: spread {spread:+.1f}%, RSI={rsi[-1]:.1f}"
                            ))
        
        # Tool 34: alt_btc_revert (2 tiers)
        if btc_data and len(btc_data['close']) >= 25:
            btc_ret_24h = (btc_data['close'][-1] - btc_data['close'][-25]) / btc_data['close'][-25] * 100
            for pair, data in all_pair_data.items():
                if pair != 'BTCUSDT' and len(data['close']) >= 25:
                    alt_ret_24h = (data['close'][-1] - data['close'][-25]) / data['close'][-25] * 100
                    spread = alt_ret_24h - btc_ret_24h
                    if spread >= 8.0:  # Tier 1: 8%+
                        signals.append(self.create_signal(
                            pair, 'alt_btc_revert_t1', 'short', bar_idx, 8, 0.05,
                            f"Alt outperforms BTC T1: spread {spread:+.1f}%"
                        ))
                    elif spread >= 5.0:  # Tier 2: 5%+
                        signals.append(self.create_signal(
                            pair, 'alt_btc_revert_t2', 'short', bar_idx, 8, 0.05,
                            f"Alt outperforms BTC T2: spread {spread:+.1f}%"
                        ))
        
        # Tool 42c: btc_lead_lag_buy (BTC pumped 2%+ in 1 bar, alt < 0.5%)
        if btc_data and len(btc_data['close']) >= 2:
            btc_ret_1bar = (btc_data['close'][-1] - btc_data['close'][-2]) / btc_data['close'][-2] * 100
            if btc_ret_1bar >= 2.0:  # BTC pumped 2%+ in last bar
                for pair, data in all_pair_data.items():
                    if pair != 'BTCUSDT' and len(data['close']) >= 2:
                        alt_ret_1bar = (data['close'][-1] - data['close'][-2]) / data['close'][-2] * 100
                        if alt_ret_1bar < 0.5:  # Alt hasn't caught up
                            signals.append(self.create_signal(
                                pair, 'btc_lead_lag_buy', 'long', bar_idx, 8, 0.03,
                                f"BTC lead-lag: BTC +{btc_ret_1bar:.1f}%, {pair} +{alt_ret_1bar:.1f}%"
                            ))
        
        # Tool 58: alt_btc_revert_t3 (3-5% spread, only if T1/T2 didn't fire)
        if btc_data and len(btc_data['close']) >= 25:
            btc_ret_24h = (btc_data['close'][-1] - btc_data['close'][-25]) / btc_data['close'][-25] * 100
            for pair, data in all_pair_data.items():
                if pair != 'BTCUSDT' and len(data['close']) >= 25:
                    alt_ret_24h = (data['close'][-1] - data['close'][-25]) / data['close'][-25] * 100
                    spread = alt_ret_24h - btc_ret_24h
                    # Only fire if in 3-5% range (T1/T2 didn't fire)
                    if 3.0 <= spread < 5.0:
                        signals.append(self.create_signal(
                            pair, 'alt_btc_revert_t3', 'short', bar_idx, 8, 0.04,
                            f"Alt outperforms BTC T3: spread {spread:+.1f}%"
                        ))
        
        # ---- INDIVIDUAL PAIR STATISTICAL/MATH TOOLS ----
        
        for pair, data in all_pair_data.items():
            df = data['df']
            close = data['close']
            volume = data['volume']
            
            if len(close) < 100:  # Need sufficient history
                continue
                
            # Calculate returns for math features
            returns = np.diff(close[-100:]) / close[-100:-1]
            
            # Calculate forward returns for signal
            price = close[-1]
            ret_4h = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
            ret_8h = (close[-1] - close[-9]) / close[-9] * 100 if len(close) >= 9 else 0
            ret_24h = (close[-1] - close[-25]) / close[-25] * 100 if len(close) >= 25 else 0
            
            # Math features
            H = self.hurst(returns)
            ent = self.entropy(returns[-30:]) if len(returns) >= 30 else 3.0
            ac1 = self.autocorr(returns, 1)
            vp = self.vpin(close[-30:], volume[-30:]) if len(close) >= 30 else 0
            
            # RSI
            rsi = self.calc_rsi(close, 14)
            cur_rsi = rsi[-1]
            
            # SMA50 for entropy_short
            sma50 = self.calc_sma(close, 50)
            cur_vs_sma50 = ((price / sma50[-1]) - 1) * 100 if not np.isnan(sma50[-1]) else 0
            
            # Open array for red candle detection
            opn = data['open']
            
            # ---- STATISTICAL/MATH TOOLS ----
            
            # Tool 23: crash_neg_ac
            if ret_24h < -10 and ac1 < -0.05:
                signals.append(self.create_signal(
                    pair, 'crash_neg_ac', 'long', bar_idx, 24, 0.08,
                    f"Crash+neg AC: {ret_24h:.1f}% drop, AC1={ac1:.3f}"
                ))
            
            # Tool 24: crash_mean_revert
            if ret_24h < -8 and H < 0.45:
                signals.append(self.create_signal(
                    pair, 'crash_mean_revert', 'long', bar_idx, 24, 0.06,
                    f"Crash+Hurst: {ret_24h:.1f}% drop, H={H:.3f}"
                ))
            
            # Tool 25: hurst_trend
            if H > 0.65 and ret_4h > 2:
                signals.append(self.create_signal(
                    pair, 'hurst_trend', 'long', bar_idx, 8, 0.03,
                    f"Hurst trend: H={H:.3f}, +{ret_4h:.1f}% 4h"
                ))
            
            # Tool 26: vpin_toxic
            if vp > 0.7 and close[-1] < opn[-1]:  # Red candle
                signals.append(self.create_signal(
                    pair, 'vpin_toxic', 'long', bar_idx, 8, 0.04,
                    f"VPIN toxic: VPIN={vp:.3f}, red candle"
                ))
            
            # Tool 27: vpin_dip
            if ret_8h < -5 and vp > 0.5:
                signals.append(self.create_signal(
                    pair, 'vpin_dip', 'long', bar_idx, 8, 0.05,
                    f"VPIN dip: {ret_8h:.1f}% drop, VPIN={vp:.3f}"
                ))
            
            # Tool 28a: entropy_short
            if ent < 2.5 and not np.isnan(sma50[-1]) and price > sma50[-1]:
                signals.append(self.create_signal(
                    pair, 'entropy_short', 'short', bar_idx, 8, 0.05,
                    f"Entropy short: ent={ent:.2f}, price>SMA50"
                ))
            
            # Tool 28: entropy_dip
            if ent < 2.5 and ret_4h < -2:
                signals.append(self.create_signal(
                    pair, 'entropy_dip', 'long', bar_idx, 8, 0.03,
                    f"Entropy dip: ent={ent:.2f}, {ret_4h:.1f}% dip"
                ))
            
            # Tool 29: triple_math
            if ret_8h < -5 and ent < 2.5 and vp > 0.5:
                signals.append(self.create_signal(
                    pair, 'triple_math', 'long', bar_idx, 24, 0.06,
                    f"Triple math: {ret_8h:.1f}% drop, ent={ent:.2f}, VPIN={vp:.3f}"
                ))
            
            # ---- COMBO TOOLS ----
            
            # Pre-compute combo stats
            returns_50 = np.diff(close[-50:]) / close[-50:-1] if len(close) >= 50 else np.array([])
            combo_skew = float(pd.Series(returns_50).skew()) if len(returns_50) > 10 else 0
            combo_kurt = float(pd.Series(returns_50).kurtosis()) if len(returns_50) > 10 else 0
            if np.isnan(combo_skew): combo_skew = 0
            if np.isnan(combo_kurt): combo_kurt = 0
            
            # Combo 1: sma50_ext_neg_ac
            if not np.isnan(sma50[-1]) and cur_vs_sma50 > 8 and ac1 < -0.1:
                signals.append(self.create_signal(
                    pair, 'sma50_ext_neg_ac', 'short', bar_idx, 8, 0.06,
                    f"SMA50+neg AC: {cur_vs_sma50:.1f}% ext, AC1={ac1:.3f}"
                ))
            
            # Combo 3: sma50_ext_fat_tail
            if not np.isnan(sma50[-1]) and cur_vs_sma50 > 10 and combo_kurt > 5:
                signals.append(self.create_signal(
                    pair, 'sma50_ext_fat_tail', 'short', bar_idx, 8, 0.06,
                    f"SMA50+fat tail: {cur_vs_sma50:.1f}% ext, kurt={combo_kurt:.1f}"
                ))
            
            # Combo 6: sma50_ext_kurt
            if not np.isnan(sma50[-1]) and cur_vs_sma50 > 8 and combo_kurt > 5:
                signals.append(self.create_signal(
                    pair, 'sma50_ext_kurt', 'short', bar_idx, 8, 0.05,
                    f"SMA50+kurt: {cur_vs_sma50:.1f}% ext, kurt={combo_kurt:.1f}"
                ))
            
            # Combo 2: alt_btc_neg_ac (alt outperforms BTC 8% + neg AC)
            if pair != 'BTCUSDT' and btc_data and len(btc_data['close']) >= 25:
                btc_ret_24h_combo = (btc_data['close'][-1] - btc_data['close'][-25]) / btc_data['close'][-25] * 100
                spread_combo = ret_24h - btc_ret_24h_combo
                if spread_combo >= 8 and ac1 < -0.05:
                    signals.append(self.create_signal(
                        pair, 'alt_btc_neg_ac', 'short', bar_idx, 8, 0.06,
                        f"Alt/BTC+neg AC: {spread_combo:+.1f}% spread, AC1={ac1:.3f}"
                    ))
            
            # Combo 5: alt_btc_neg_ac_5
            if pair != 'BTCUSDT' and btc_data and len(btc_data['close']) >= 25:
                btc_ret_24h_combo = (btc_data['close'][-1] - btc_data['close'][-25]) / btc_data['close'][-25] * 100
                spread_combo = ret_24h - btc_ret_24h_combo
                if spread_combo >= 5 and ac1 < -0.15:
                    signals.append(self.create_signal(
                        pair, 'alt_btc_neg_ac_5', 'short', bar_idx, 8, 0.05,
                        f"Alt/BTC5%+neg AC: {spread_combo:+.1f}% spread, AC1={ac1:.3f}"
                    ))
            
            # Combo 4: rsi_pump_fat_tail
            if cur_rsi > 80 and len(close) >= 13:
                ret_12h_combo = (price - close[-13]) / close[-13] * 100 if close[-13] > 0 else 0
                if ret_12h_combo > 10 and combo_kurt > 5:
                    signals.append(self.create_signal(
                        pair, 'rsi_pump_fat_tail', 'short', bar_idx, 8, 0.06,
                        f"RSI pump+fat tail: RSI={cur_rsi:.1f}, +{ret_12h_combo:.1f}% 12h, kurt={combo_kurt:.1f}"
                    ))
            
            # Combo 10: green_exhaust_kurt (7 green then red + kurt > 3)
            if len(close) >= 8 and len(opn) >= 8:
                all_green = all(close[-j-1] > opn[-j-1] for j in range(1, 8))
                cur_red = close[-1] < opn[-1]
                if all_green and cur_red and combo_kurt > 3:
                    signals.append(self.create_signal(
                        pair, 'green_exhaust_kurt', 'short', bar_idx, 8, 0.04,
                        f"Green exhaust+kurt: 7 green then red, kurt={combo_kurt:.1f}"
                    ))
            
            # BB calculations for breakout combos
            if len(close) >= 50:
                bb_mid, bb_upper, bb_lower, bb_bandwidth = self.calc_bollinger(close, 20, 2.0)
                vol_ratio = volume[-1] / np.mean(volume[-21:-1]) if len(volume) >= 21 else 1.0
                
                # Check for BB squeeze
                recent_bw = bb_bandwidth[-21:-1] if len(bb_bandwidth) >= 21 else bb_bandwidth[:-1]
                valid_bw = recent_bw[~np.isnan(recent_bw)]
                is_squeeze = False
                breaks_upper = False
                if len(valid_bw) >= 10 and not np.isnan(bb_bandwidth[-1]) and not np.isnan(bb_upper[-1]):
                    bw_low = np.min(valid_bw)
                    is_squeeze = bb_bandwidth[-1] <= bw_low * 1.1
                    breaks_upper = price > bb_upper[-1]
                
                # Combo 7: bb_break_pos_ac (BB squeeze breakout + pos AC + 2x vol)
                if is_squeeze and breaks_upper and vol_ratio > 2.0 and ac1 > 0.1:
                    signals.append(self.create_signal(
                        pair, 'bb_break_pos_ac', 'long', bar_idx, 24, 0.04,
                        f"BB break+pos AC: squeeze breakout, AC1={ac1:.3f}, vol={vol_ratio:.1f}x"
                    ))
                
                # 50-bar high for other combos
                high_50 = np.max(data['high'][-50:]) if len(data['high']) >= 50 else 0
                is_high_break = price > high_50
                
                # Combo 8: high_break_pos_ac (50-bar high + pos AC + 2x vol)
                if is_high_break and vol_ratio > 2.0 and ac1 > 0.1:
                    signals.append(self.create_signal(
                        pair, 'high_break_pos_ac', 'long', bar_idx, 24, 0.04,
                        f"High break+pos AC: 50-bar high, AC1={ac1:.3f}, vol={vol_ratio:.1f}x"
                    ))
                
                # Combo 9: high_break_skew (50-bar high + skew > 1.0 + 2x vol)
                if is_high_break and vol_ratio > 2.0 and combo_skew > 1.0:
                    signals.append(self.create_signal(
                        pair, 'high_break_skew', 'long', bar_idx, 24, 0.04,
                        f"High break+skew: 50-bar high, skew={combo_skew:.1f}, vol={vol_ratio:.1f}x"
                    ))
        
        return signals
    
    def create_signal(self, pair, tool, direction, bar_idx, hold_hours, sl_pct, reason):
        """Create a signal dictionary."""
        return {
            'pair': pair,
            'tool': tool,
            'direction': direction,
            'bar_idx': bar_idx,
            'hold_hours': hold_hours,
            'sl_pct': sl_pct,
            'reason': reason,
            'timestamp': self.data_cache[pair]['timestamp'].iloc[bar_idx]
        }
    
    def analyze_signals(self, signals):
        """Calculate forward returns and performance metrics for all signals."""
        print("\nAnalyzing signal performance...")
        
        results = {}
        
        for signal in signals:
            pair = signal['pair']
            tool = signal['tool']
            direction = signal['direction']
            bar_idx = signal['bar_idx']
            hold_hours = signal['hold_hours']
            
            if tool not in results:
                results[tool] = {
                    'signals': 0,
                    'direction': direction,
                    'returns_8h': [],
                    'returns_24h': [],
                    'wins_8h': 0,
                    'wins_24h': 0
                }
            
            # Get forward returns
            df = self.data_cache[pair]
            if bar_idx + 6 < len(df) and bar_idx + 2 < len(df):
                current_price = df['close'].iloc[bar_idx]
                price_8h = df['close'].iloc[bar_idx + 2]  # 2 bars = 8h
                price_24h = df['close'].iloc[bar_idx + 6] if bar_idx + 6 < len(df) else current_price
                
                # Calculate returns based on direction
                if direction == 'long':
                    ret_8h = (price_8h - current_price) / current_price * 100
                    ret_24h = (price_24h - current_price) / current_price * 100
                else:  # short
                    ret_8h = (current_price - price_8h) / current_price * 100
                    ret_24h = (current_price - price_24h) / current_price * 100
                
                results[tool]['signals'] += 1
                results[tool]['returns_8h'].append(ret_8h)
                results[tool]['returns_24h'].append(ret_24h)
                
                if ret_8h > 0:
                    results[tool]['wins_8h'] += 1
                if ret_24h > 0:
                    results[tool]['wins_24h'] += 1
        
        # Calculate final metrics
        for tool, data in results.items():
            if data['signals'] > 0:
                data['wr_8h'] = data['wins_8h'] / data['signals'] * 100
                data['wr_24h'] = data['wins_24h'] / data['signals'] * 100
                data['net_8h'] = np.mean(data['returns_8h']) if data['returns_8h'] else 0
                data['net_24h'] = np.mean(data['returns_24h']) if data['returns_24h'] else 0
        
        self.results = results
    
    def generate_report(self):
        """Generate and save the validation report."""
        print("\nGenerating validation report...")
        
        # Sort results by net 24h return
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['net_24h'], reverse=True)
        
        # Classification criteria
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
        
        # Print results table
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
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"PASSED:   {len(passed):2d} tools")
        print(f"MARGINAL: {len(marginal):2d} tools")  
        print(f"FAILED:   {len(failed):2d} tools")
        print(f"TOTAL:    {len(sorted_results):2d} tools tested")
        
        # Write to report file
        report_path = "data/oos_report.md"
        with open(report_path, 'a') as f:
            f.write(f"\n\n## Cross-Pair, Statistical & Combo Tools\n\n")
            f.write(f"**Validation Date**: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Tools Tested**: {len(sorted_results)} remaining tools\n")
            f.write(f"**Results**: {len(passed)} PASSED, {len(marginal)} MARGINAL, {len(failed)} FAILED\n\n")
            
            f.write("### 🟢 PASSED Tools\n\n")
            f.write("| Tool | Direction | Signals | WR_24h | Net_24h | Status |\n")
            f.write("|------|-----------|---------|---------|---------|--------|\n")
            for tool, data in passed:
                f.write(f"| {tool} | {data['direction']} | {data['signals']} | "
                       f"{data['wr_24h']:.1f}% | {data['net_24h']:+.2f}% | ✅ Validated |\n")
            
            f.write("\n### 🟡 MARGINAL Tools\n\n")
            f.write("| Tool | Direction | Signals | WR_24h | Net_24h | Issue |\n")
            f.write("|------|-----------|---------|---------|---------|-------|\n")
            for tool, data in marginal:
                target_wr = 50.0 if data['direction'] == 'long' else 45.0
                issue = f"WR below {target_wr:.0f}%" if data['wr_24h'] < target_wr else "Low signal count"
                f.write(f"| {tool} | {data['direction']} | {data['signals']} | "
                       f"{data['wr_24h']:.1f}% | {data['net_24h']:+.2f}% | {issue} |\n")
            
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
        
        # Apply fixes to master bot (placeholder - would need actual implementation)
        self.apply_fixes_to_master_bot(passed, marginal, failed)
        
        return len(passed), len(marginal), len(failed)
    
    def apply_fixes_to_master_bot(self, passed, marginal, failed):
        """Apply validation results to master bot code."""
        print("\nSuggested fixes for run_master_bot.py:")
        print("-" * 50)
        
        for tool, data in passed:
            print(f"# {tool}: OOS-validated ({data['wr_24h']:.1f}% WR, {data['net_24h']:+.2f}% net)")
        
        for tool, data in failed:
            print(f"# {tool}: OOS-DISABLED ({data['wr_24h']:.1f}% WR, {data['net_24h']:+.2f}% net)")
        
        print("\nManual review recommended for marginal tools.")

def main():
    validator = OOSRemainingValidator()
    validator.validate_tools()

if __name__ == "__main__":
    main()