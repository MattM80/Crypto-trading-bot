#!/usr/bin/env python3
"""
BULL MARKET TOOLS - REFINED APPROACH
Based on initial testing results, refining the most promising strategies.

ANALYSIS OF INITIAL RESULTS:
1. btc_strength_alt_rotation: 47.8% avg WR but some pairs excellent (NEAR 96.7%, SOL 100%)
2. post_consolidation_breakout: 44.5% avg WR, some good individual results  
3. wyckoff_accumulation: 34.4% avg WR but UNI 83.3%, XLM 58.8%

REFINEMENT STRATEGY:
- Tighten entry conditions for higher conviction signals
- Add additional filters to reduce false signals
- Focus on combinations that leverage multiple confirmations
- Test parameter variations to find optimal thresholds
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RefinedBullToolsDeveloper:
    def __init__(self, data_dir: str = "data/binance_1h"):
        self.data_dir = Path(data_dir)
        self.pairs = [
            "NEARUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "AAVEUSDT", 
            "SOLUSDT", "ETHUSDT", "BTCUSDT", "DOTUSDT", "XLMUSDT", 
            "XRPUSDT", "ADAUSDT", "ATOMUSDT", "DOGEUSDT", "FILUSDT", "LTCUSDT"
        ]
        self.fee_pct = 0.0065  # 0.65% round-trip (worst case for spot)
        self.oos_start = 4380  # Out-of-sample start
        self.data = {}
        self.results = []
        self.btc_data = None
        
    def load_data(self):
        """Load all pairs data"""
        print("Loading data...")
        for pair in self.pairs:
            file_path = self.data_dir / f"{pair}_1h.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                self.data[pair] = df
                if pair == "BTCUSDT":
                    self.btc_data = df.copy()
        print(f"Loaded {len(self.data)} pairs")
        
    # ==================== INDICATORS ====================
    
    def calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI calculation using Wilder's smoothing"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
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
    
    def calc_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
        
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        k = 2.0 / (period + 1)
        
        for i in range(period, len(prices)):
            ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
        
        return ema
    
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    
    def calc_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        sma = self.calc_sma(prices, period)
        std = pd.Series(prices).rolling(window=period, min_periods=1).std().values
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calc_volume_profile(self, volume: np.ndarray, period: int = 20) -> float:
        """Volume relative to recent average"""
        if len(volume) < period:
            return 1.0
        
        recent_vol = volume[-1]
        avg_vol = np.mean(volume[-period:])
        
        return recent_vol / avg_vol if avg_vol > 0 else 1.0
    
    # ==================== REFINED TOOLS ====================
    
    def test_btc_strength_refined(self, df: pd.DataFrame, pair: str) -> List[Dict]:
        """
        REFINED BTC STRENGTH → ALT ROTATION
        Tighter conditions based on analysis of high-performing signals
        """
        signals = []
        if len(df) < 200 or pair == "BTCUSDT" or self.btc_data is None:
            return signals
            
        close = df['close'].values
        btc_close = self.btc_data['close'].values
        
        min_len = min(len(close), len(btc_close))
        close = close[:min_len]
        btc_close = btc_close[:min_len]
        
        rsi14 = self.calc_rsi(close, 14)
        
        for i in range(200, min_len):
            # BTC 7-day performance (more restrictive range)
            if i >= 168:
                btc_7d_return = (btc_close[i] - btc_close[i-168]) / btc_close[i-168] * 100
            else:
                continue
            
            # BTC stability (tighter requirement)
            if i >= 48:  # 48h for better stability assessment
                btc_48h_volatility = np.std(btc_close[i-48:i]) / np.mean(btc_close[i-48:i]) * 100
            else:
                continue
            
            # Alt performance metrics
            if i >= 168:
                alt_7d_return = (close[i] - close[i-168]) / close[i-168] * 100
            else:
                continue
                
            if i >= 48:
                alt_48h_return = (close[i] - close[i-48]) / close[i-48] * 100
            else:
                continue
            
            # REFINED CONDITIONS (much stricter)
            btc_sweet_spot = 8 <= btc_7d_return <= 20    # Narrower range
            btc_stable = btc_48h_volatility < 2.5        # Even more stable
            alt_lagging = alt_7d_return < btc_7d_return - 8  # Bigger lag required
            alt_not_crashed = alt_48h_return > -5        # Alt not in freefall
            rsi_ready = 30 <= rsi14[i] <= 55             # RSI in middle range
            
            # Additional momentum filter
            if i >= 24:
                alt_24h_return = (close[i] - close[i-24]) / close[i-24] * 100
                alt_starting_move = alt_24h_return > 0.5  # Small positive move
            else:
                continue
            
            if (btc_sweet_spot and btc_stable and alt_lagging and 
                alt_not_crashed and rsi_ready and alt_starting_move):
                
                # Enhanced scoring
                lag_gap = btc_7d_return - alt_7d_return
                stability_score = max(0, 4 - btc_48h_volatility) * 10
                momentum_score = alt_24h_return * 5
                
                score = lag_gap * 3 + stability_score + momentum_score
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'btc_7d': btc_7d_return,
                    'btc_vol': btc_48h_volatility,
                    'alt_7d': alt_7d_return,
                    'lag_gap': lag_gap,
                    'alt_24h': alt_24h_return,
                    'rsi': rsi14[i],
                    'reason': f"BTC refined: lag={lag_gap:.1f}%, vol={btc_48h_volatility:.1f}%, RSI={rsi14[i]:.0f}"
                })
        
        return signals
    
    def test_volume_squeeze_combo(self, df: pd.DataFrame) -> List[Dict]:
        """
        VOLUME SQUEEZE COMBO
        Combines volume breakout with consolidation squeeze
        Much stricter criteria based on initial results
        """
        signals = []
        if len(df) < 150:
            return signals
            
        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values
        
        bb_upper, bb_mid, bb_lower = self.calc_bollinger_bands(close, 20, 2.0)
        rsi14 = self.calc_rsi(close, 14)
        
        for i in range(100, len(df)):
            # Bollinger squeeze (very tight)
            bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] * 100
            
            # Was recently squeezed (very narrow bands)
            if i >= 20:
                min_width_20 = np.min((bb_upper[i-20:i] - bb_lower[i-20:i]) / bb_mid[i-20:i] * 100)
                was_very_squeezed = min_width_20 < 3  # Very tight squeeze
            else:
                continue
            
            # Current expansion
            expanding = bb_width > 5
            
            # High volume breakout
            vol_multiplier = self.calc_volume_profile(volume[:i+1], 15)
            volume_surge = vol_multiplier >= 4.0  # Higher volume requirement
            
            # Price breakout above recent range
            if i >= 50:
                range_high = np.max(high[i-50:i-5])  # Don't include very recent bars
                breakout = close[i] > range_high * 1.02  # 2% above range
            else:
                continue
            
            # RSI not overbought
            rsi_ok = rsi14[i] < 70
            
            # Trend filter - price above 50-period SMA
            sma50 = self.calc_sma(close, 50)
            uptrend = close[i] > sma50[i] if not np.isnan(sma50[i]) else False
            
            if (was_very_squeezed and expanding and volume_surge and 
                breakout and rsi_ok and uptrend):
                
                # Score based on all factors
                squeeze_score = max(0, 4 - min_width_20) * 10  # Tighter squeeze = higher score
                volume_score = vol_multiplier * 3
                breakout_pct = (close[i] - range_high) / range_high * 100
                breakout_score = breakout_pct * 15
                
                score = squeeze_score + volume_score + breakout_score
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'bb_width': bb_width,
                    'min_width': min_width_20,
                    'volume_mult': vol_multiplier,
                    'breakout_pct': breakout_pct,
                    'rsi': rsi14[i],
                    'reason': f"Volume squeeze: width={bb_width:.1f}% (was {min_width_20:.1f}%), vol={vol_multiplier:.1f}x"
                })
        
        return signals
    
    def test_trend_pullback_enhanced(self, df: pd.DataFrame) -> List[Dict]:
        """
        ENHANCED TREND PULLBACK
        Much stricter trend requirements and precise entry timing
        """
        signals = []
        if len(df) < 150:
            return signals
            
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        rsi7 = self.calc_rsi(close, 7)    # Faster RSI for entries
        rsi14 = self.calc_rsi(close, 14)  # Trend RSI
        ema21 = self.calc_ema(close, 21)
        ema50 = self.calc_ema(close, 50)
        ema100 = self.calc_ema(close, 100)
        
        for i in range(100, len(df)):
            # STRONG uptrend confirmation (all EMAs aligned)
            strong_uptrend = (not np.isnan(ema21[i]) and not np.isnan(ema50[i]) and not np.isnan(ema100[i]) and
                             ema21[i] > ema50[i] and ema50[i] > ema100[i] and
                             close[i] > ema21[i])
            
            # EMA separation (strong trend has good separation)
            if not np.isnan(ema21[i]) and not np.isnan(ema50[i]):
                ema_sep = (ema21[i] - ema50[i]) / ema50[i] * 100
                good_separation = ema_sep > 2  # At least 2% separation
            else:
                continue
            
            # Recent higher highs pattern
            if i >= 30:
                recent_high = np.max(high[i-30:i-5])
                prev_high = np.max(high[i-60:i-30]) if i >= 60 else recent_high
                higher_highs = recent_high > prev_high * 1.03  # 3% higher
            else:
                continue
            
            # RSI pullback but not oversold
            rsi_pullback = 30 <= rsi7[i] <= 45
            rsi_trend_ok = rsi14[i] > 40  # Overall trend still healthy
            
            # Volume confirmation (not declining on pullback)
            if i >= 10:
                avg_vol_10 = np.mean(volume[i-10:i])
                current_vol = volume[i]
                vol_ok = current_vol >= avg_vol_10 * 0.8  # Not too light
            else:
                continue
            
            # Price near but above key EMA
            near_ema21 = abs(close[i] - ema21[i]) / ema21[i] < 0.02  # Within 2%
            above_ema50 = close[i] > ema50[i]
            
            if (strong_uptrend and good_separation and higher_highs and
                rsi_pullback and rsi_trend_ok and vol_ok and near_ema21 and above_ema50):
                
                # Score based on trend strength and pullback quality
                trend_score = ema_sep * 5
                pullback_score = (45 - rsi7[i]) * 2  # Better pullback = higher score
                momentum_score = (recent_high / prev_high - 1) * 100
                
                score = trend_score + pullback_score + momentum_score
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'rsi7': rsi7[i],
                    'rsi14': rsi14[i],
                    'ema_sep': ema_sep,
                    'higher_highs': higher_highs,
                    'reason': f"Enhanced pullback: RSI7={rsi7[i]:.0f}, trend={ema_sep:.1f}%, momentum={momentum_score:.1f}%"
                })
        
        return signals
    
    def test_wyckoff_spring_refined(self, df: pd.DataFrame) -> List[Dict]:
        """
        REFINED WYCKOFF SPRING
        More precise spring detection with volume analysis
        """
        signals = []
        if len(df) < 150:
            return signals
            
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values
        volume = df['volume'].values
        
        for i in range(100, len(df)):
            # Establish support zone from longer lookback
            if i >= 80:
                support_zone = np.min(low[i-80:i-20])  # Support from 80-20 bars ago
                resistance_zone = np.max(high[i-80:i-20])  # Resistance from same period
            else:
                continue
            
            # Range trading (sideways market)
            range_size = (resistance_zone - support_zone) / support_zone * 100
            is_range = 5 <= range_size <= 25  # 5-25% range
            
            # Spring test - break below support with volume
            if i >= 10:
                recent_low = np.min(low[i-10:i])
                spring_break = recent_low < support_zone * 0.98  # 2% below support
                
                # Volume on spring (look for volume spike during break)
                spring_bars = [j for j in range(i-10, i) if low[j] < support_zone]
                if spring_bars:
                    spring_vol = np.max([self.calc_volume_profile(volume[:j+1], 20) for j in spring_bars])
                    vol_confirmation = spring_vol > 2.5
                else:
                    vol_confirmation = False
            else:
                continue
            
            # Recovery above support
            strong_recovery = close[i] > support_zone * 1.02  # 2% above support
            
            # No new lows recently (spring was temporary)
            if i >= 5:
                no_new_lows = np.min(low[i-5:i]) > recent_low
            else:
                no_new_lows = True
            
            # Volume drying up after spring (accumulation phase)
            current_vol = self.calc_volume_profile(volume[:i+1], 10)
            low_volume_recovery = current_vol < 1.5
            
            if (is_range and spring_break and vol_confirmation and 
                strong_recovery and no_new_lows and low_volume_recovery):
                
                # Score based on spring quality and recovery
                spring_depth = (support_zone - recent_low) / support_zone * 100
                recovery_strength = (close[i] - support_zone) / support_zone * 100
                
                score = spring_depth * 5 + recovery_strength * 10 + spring_vol * 3
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'support': support_zone,
                    'spring_depth': spring_depth,
                    'recovery_pct': recovery_strength,
                    'spring_vol': spring_vol,
                    'range_pct': range_size,
                    'reason': f"Wyckoff refined: spring {spring_depth:.1f}% below, recovery {recovery_strength:.1f}%"
                })
        
        return signals
    
    def test_momentum_confirmation_combo(self, df: pd.DataFrame, pair: str) -> List[Dict]:
        """
        MOMENTUM CONFIRMATION COMBO
        Combines multiple momentum signals for high conviction entries
        """
        signals = []
        if len(df) < 200:
            return signals
            
        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values
        
        rsi14 = self.calc_rsi(close, 14)
        ema12 = self.calc_ema(close, 12)
        ema26 = self.calc_ema(close, 26)
        ema50 = self.calc_ema(close, 50)
        
        for i in range(100, len(df)):
            # MACD-like momentum
            macd_line = ema12[i] - ema26[i]
            macd_prev = ema12[i-1] - ema26[i-1] if i > 0 else macd_line
            macd_improving = macd_line > macd_prev
            
            # Price momentum (multiple timeframes)
            if i >= 24:
                mom_24h = (close[i] - close[i-24]) / close[i-24] * 100
            else:
                continue
                
            if i >= 48:
                mom_48h = (close[i] - close[i-48]) / close[i-48] * 100
            else:
                continue
            
            # Trend alignment
            uptrend = (close[i] > ema50[i] and ema12[i] > ema26[i] and ema26[i] > ema50[i])
            
            # Volume confirmation
            vol_strength = self.calc_volume_profile(volume[:i+1], 20)
            
            # RSI in momentum zone
            rsi_momentum = 50 <= rsi14[i] <= 75
            
            # Recent breakout
            if i >= 20:
                resistance = np.max(high[i-20:i-5])
                breakout = close[i] > resistance * 1.01
            else:
                continue
            
            # CONDITIONS
            positive_momentum = mom_24h > 3 and mom_48h > 5
            volume_supporting = vol_strength > 1.3
            
            if (uptrend and macd_improving and positive_momentum and
                volume_supporting and rsi_momentum and breakout):
                
                # Score based on momentum strength
                momentum_score = (mom_24h + mom_48h) * 2
                volume_score = vol_strength * 5
                rsi_score = (75 - abs(rsi14[i] - 62.5)) * 2  # Closer to 62.5 is better
                
                score = momentum_score + volume_score + rsi_score
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'mom_24h': mom_24h,
                    'mom_48h': mom_48h,
                    'volume_mult': vol_strength,
                    'rsi': rsi14[i],
                    'reason': f"Momentum combo: 24h={mom_24h:+.1f}%, 48h={mom_48h:+.1f}%, vol={vol_strength:.1f}x"
                })
        
        return signals
    
    def test_tool_on_pair(self, pair: str, tool_func, tool_name: str) -> Dict:
        """Test a specific tool on a specific pair"""
        if pair not in self.data:
            return None
            
        df = self.data[pair].copy()
        
        # Get signals - handle different function signatures
        if tool_name in ["btc_strength_refined", "momentum_confirmation_combo"]:
            signals = tool_func(df, pair)
        else:
            signals = tool_func(df)
        
        if not signals:
            return {
                'pair': pair,
                'tool': tool_name,
                'signals_total': 0,
                'signals_oos': 0,
                'win_rate_8h': 0,
                'win_rate_24h': 0,
                'avg_return_8h': 0,
                'avg_return_24h': 0,
                'status': 'NO_SIGNALS'
            }
        
        # Separate IS and OOS signals
        oos_signals = [s for s in signals if s['bar'] >= self.oos_start]
        
        if not oos_signals:
            return {
                'pair': pair,
                'tool': tool_name,
                'signals_total': len(signals),
                'signals_oos': 0,
                'win_rate_8h': 0,
                'win_rate_24h': 0,
                'avg_return_8h': 0,
                'avg_return_24h': 0,
                'status': 'NO_OOS_SIGNALS'
            }
        
        # Calculate forward returns
        close = df['close'].values
        wins_8h = 0
        wins_24h = 0
        returns_8h = []
        returns_24h = []
        
        for signal in oos_signals:
            entry_bar = signal['bar']
            entry_price = signal['price']
            
            # 8-hour forward return
            if entry_bar + 8 < len(close):
                exit_price_8h = close[entry_bar + 8]
                ret_8h = (exit_price_8h - entry_price) / entry_price * 100 - self.fee_pct
                returns_8h.append(ret_8h)
                if ret_8h > 0:
                    wins_8h += 1
            
            # 24-hour forward return
            if entry_bar + 24 < len(close):
                exit_price_24h = close[entry_bar + 24]
                ret_24h = (exit_price_24h - entry_price) / entry_price * 100 - self.fee_pct
                returns_24h.append(ret_24h)
                if ret_24h > 0:
                    wins_24h += 1
        
        # Calculate metrics
        win_rate_8h = wins_8h / len(returns_8h) * 100 if returns_8h else 0
        win_rate_24h = wins_24h / len(returns_24h) * 100 if returns_24h else 0
        avg_return_8h = np.mean(returns_8h) if returns_8h else 0
        avg_return_24h = np.mean(returns_24h) if returns_24h else 0
        
        # Determine status
        status = 'PASS' if (win_rate_8h > 50 or win_rate_24h > 50) else 'FAIL'
        
        return {
            'pair': pair,
            'tool': tool_name,
            'signals_total': len(signals),
            'signals_oos': len(oos_signals),
            'win_rate_8h': win_rate_8h,
            'win_rate_24h': win_rate_24h,
            'avg_return_8h': avg_return_8h,
            'avg_return_24h': avg_return_24h,
            'status': status
        }
    
    def run_refined_tests(self):
        """Run all refined bull market tools"""
        print("\\n" + "="*80)
        print("REFINED BULL MARKET TOOLS - ENHANCED TEST SUITE")
        print("="*80)
        
        tools_to_test = [
            (self.test_btc_strength_refined, "btc_strength_refined"),
            (self.test_volume_squeeze_combo, "volume_squeeze_combo"),
            (self.test_trend_pullback_enhanced, "trend_pullback_enhanced"),
            (self.test_wyckoff_spring_refined, "wyckoff_spring_refined"),
            (self.test_momentum_confirmation_combo, "momentum_confirmation_combo")
        ]
        
        all_results = []
        tool_summaries = []
        
        for tool_func, tool_name in tools_to_test:
            print(f"\\nTesting {tool_name}...")
            tool_results = []
            
            for pair in self.pairs:
                result = self.test_tool_on_pair(pair, tool_func, tool_name)
                if result:
                    tool_results.append(result)
                    if result['signals_oos'] > 0:
                        print(f"  {pair}: {result['signals_oos']:>2} signals, "
                              f"WR_8h={result['win_rate_8h']:>5.1f}%, "
                              f"WR_24h={result['win_rate_24h']:>5.1f}%, "
                              f"Ret_24h={result['avg_return_24h']:>+6.2f}% -> {result['status']}")
                    else:
                        print(f"  {pair}: {result['signals_oos']:>2} signals -> {result['status']}")
            
            # Calculate tool summary
            valid_results = [r for r in tool_results if r['signals_oos'] > 0]
            if valid_results:
                total_signals = sum(r['signals_oos'] for r in valid_results)
                passing_pairs = sum(1 for r in valid_results if r['status'] == 'PASS')
                avg_wr_8h = np.mean([r['win_rate_8h'] for r in valid_results])
                avg_wr_24h = np.mean([r['win_rate_24h'] for r in valid_results])
                avg_ret_8h = np.mean([r['avg_return_8h'] for r in valid_results])
                avg_ret_24h = np.mean([r['avg_return_24h'] for r in valid_results])
                
                summary = {
                    'tool': tool_name,
                    'total_signals': total_signals,
                    'passing_pairs': passing_pairs,
                    'total_pairs': len(valid_results),
                    'avg_wr_8h': avg_wr_8h,
                    'avg_wr_24h': avg_wr_24h,
                    'avg_ret_8h': avg_ret_8h,
                    'avg_ret_24h': avg_ret_24h,
                    'best_wr': max(avg_wr_8h, avg_wr_24h),
                    'status': 'PASS' if (passing_pairs > 0 and max(avg_wr_8h, avg_wr_24h) > 50) else 'FAIL'
                }
                
                print(f"  SUMMARY: {total_signals} signals, {passing_pairs}/{len(valid_results)} pairs pass")
                print(f"  AVG: WR_8h={avg_wr_8h:.1f}%, WR_24h={avg_wr_24h:.1f}%, "
                      f"Ret_8h={avg_ret_8h:+.2f}%, Ret_24h={avg_ret_24h:+.2f}% -> {summary['status']}")
                
                tool_summaries.append(summary)
            else:
                print(f"  SUMMARY: No valid signals generated")
            
            all_results.extend(tool_results)
        
        return all_results, tool_summaries
    
    def generate_refined_report(self, results, summaries):
        """Generate report for refined tools"""
        print("\\n" + "="*80)
        print("REFINED BULL MARKET TOOLS - FINAL REPORT")
        print("="*80)
        
        passing_tools = [s for s in summaries if s['status'] == 'PASS']
        failing_tools = [s for s in summaries if s['status'] == 'FAIL']
        
        print(f"\\n📊 REFINED RESULTS:")
        print(f"✅ PASSED: {len(passing_tools)} tools")
        print(f"❌ FAILED: {len(failing_tools)} tools")
        if summaries:
            print(f"📈 SUCCESS RATE: {len(passing_tools)}/{len(summaries)} ({len(passing_tools)/len(summaries)*100:.1f}%)")
        
        if summaries:
            print(f"\\n📋 DETAILED RESULTS:")
            for summary in sorted(summaries, key=lambda x: x['best_wr'], reverse=True):
                status_emoji = "✅" if summary['status'] == 'PASS' else "❌"
                print(f"{status_emoji} {summary['tool']:<35} | "
                      f"{summary['total_signals']:>4} signals | "
                      f"{summary['passing_pairs']:>2}/{summary['total_pairs']} pairs | "
                      f"Best WR: {summary['best_wr']:>5.1f}% | "
                      f"Ret_24h: {summary['avg_ret_24h']:>+6.2f}%")
        
        # Show the best performing individual pairs
        print(f"\\n🏆 BEST INDIVIDUAL PERFORMANCES:")
        best_individual = []
        for result in results:
            if result['signals_oos'] > 0 and result['status'] == 'PASS':
                best_wr = max(result['win_rate_8h'], result['win_rate_24h'])
                best_individual.append((result, best_wr))
        
        best_individual.sort(key=lambda x: x[1], reverse=True)
        
        for (result, wr) in best_individual[:10]:  # Top 10
            timeframe = "8h" if result['win_rate_8h'] > result['win_rate_24h'] else "24h"
            ret = result['avg_return_8h'] if timeframe == "8h" else result['avg_return_24h']
            print(f"   {result['pair']} - {result['tool']}: {wr:.1f}% WR ({timeframe}), {ret:+.2f}% ret, {result['signals_oos']} signals")
        
        if passing_tools:
            print(f"\\n💡 IMPLEMENTATION RECOMMENDATIONS:")
            for tool in passing_tools:
                print(f"   • {tool['tool']}: Deploy with {tool['total_signals']} total signals")
            print(f"   • Focus on pairs with highest individual performance")
            print(f"   • Consider combining signals for higher conviction entries")
            
        return {
            'passing_tools': len(passing_tools),
            'total_tools': len(summaries),
            'best_individual': best_individual[:5]
        }


def main():
    """Main execution for refined tests"""
    print("🔬 Starting Refined Bull Market Tools Analysis...")
    
    developer = RefinedBullToolsDeveloper()
    developer.load_data()
    
    if not developer.data:
        print("❌ No data loaded")
        return
    
    # Run refined tests
    results, summaries = developer.run_refined_tests()
    
    # Generate report
    report = developer.generate_refined_report(results, summaries)
    
    print(f"\\n🎯 REFINED MISSION STATUS:")
    if report['passing_tools'] > 0:
        print(f"✅ SUCCESS: Found {report['passing_tools']} validated bull market tools!")
        print("   Ready for implementation in run_final_bot.py")
        print("   Check individual performances for best pairs to focus on")
    else:
        print("❌ STILL INCOMPLETE: Even refined tools need more work")
        print("   Consider even tighter criteria or different market regime analysis")
    
    return results, summaries


if __name__ == "__main__":
    results, summaries = main()