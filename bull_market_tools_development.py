#!/usr/bin/env python3
"""
BULL MARKET TOOLS DEVELOPMENT - New Approach
Building validated bull market trading tools to complement existing fear/crash tools.

MISSION: Find bull market edge that survives fees and OOS validation.
Previous attempts at breakout longs ALL FAILED due to:
- Fee drag (0.65% round-trip) kills low-edge signals
- Crypto breakouts have too many false signals
- High volatility causes whipsaws

NEW APPROACH - What to try:
1. Trend pullback buys in confirmed uptrends
2. Volume confirmation + breakout continuation 
3. Multi-timeframe momentum alignment
4. Post-consolidation breakouts with tight ranges
5. BTC strength → alt rotation patterns
6. Accumulation detection (Wyckoff patterns)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BullMarketToolsDeveloper:
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
        self.btc_data = None  # For BTC relative strength analysis
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all pairs data"""
        print("Loading data...")
        for pair in self.pairs:
            file_path = self.data_dir / f"{pair}_1h.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"Loaded {pair}: {len(df)} bars")
                self.data[pair] = df
                
                # Store BTC data for relative analysis
                if pair == "BTCUSDT":
                    self.btc_data = df.copy()
            else:
                print(f"WARNING: Missing {pair}_1h.csv")
        
        print(f"Total pairs loaded: {len(self.data)}")
        return self.data
    
    # ==================== TECHNICAL INDICATORS ====================
    
    def calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI calculation using Wilder's smoothing"""
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
    
    def calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        if len(high) < 2:
            return np.zeros(len(high))
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return pd.Series(true_range).rolling(window=period, min_periods=1).mean().values
    
    def calc_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        sma = self.calc_sma(prices, period)
        std = pd.Series(prices).rolling(window=period, min_periods=1).std().values
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def detect_higher_highs_lows(self, high: np.ndarray, low: np.ndarray, lookback: int = 10) -> Tuple[bool, bool]:
        """Detect higher highs and higher lows pattern (uptrend)"""
        if len(high) < lookback * 2:
            return False, False
        
        recent_highs = high[-lookback:]
        recent_lows = low[-lookback:]
        prev_highs = high[-lookback*2:-lookback]
        prev_lows = low[-lookback*2:-lookback]
        
        higher_highs = np.max(recent_highs) > np.max(prev_highs)
        higher_lows = np.min(recent_lows) > np.min(prev_lows)
        
        return higher_highs, higher_lows
    
    def calc_volume_profile(self, volume: np.ndarray, period: int = 20) -> float:
        """Volume relative to recent average"""
        if len(volume) < period:
            return 1.0
        
        recent_vol = volume[-1]
        avg_vol = np.mean(volume[-period:])
        
        return recent_vol / avg_vol if avg_vol > 0 else 1.0
    
    # ==================== NEW BULL MARKET TOOLS ====================
    
    def test_trend_pullback_rsi_dip(self, df: pd.DataFrame) -> List[Dict]:
        """
        TREND PULLBACK RSI DIP
        Logic: Buy RSI dips (25-35) within confirmed uptrends (EMA alignment + higher highs)
        Theory: In strong uptrends, temporary RSI dips create good risk/reward entries
        """
        signals = []
        if len(df) < 100:
            return signals
            
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        rsi14 = self.calc_rsi(close, 14)
        ema21 = self.calc_ema(close, 21)
        ema50 = self.calc_ema(close, 50)
        
        for i in range(50, len(df)):
            # Skip if not enough data
            if np.isnan(ema21[i]) or np.isnan(ema50[i]):
                continue
                
            # Confirm uptrend: EMA21 > EMA50 and price above both
            uptrend_confirmed = (ema21[i] > ema50[i] and 
                                close[i] > ema21[i] and 
                                close[i] > ema50[i])
            
            # Check for higher highs in recent 20 bars
            if i >= 20:
                higher_highs, higher_lows = self.detect_higher_highs_lows(high[i-20:i], low[i-20:i], 10)
            else:
                higher_highs, higher_lows = False, False
            
            # RSI pullback condition
            rsi_dip = 25 <= rsi14[i] <= 35
            
            # Entry condition
            if uptrend_confirmed and higher_highs and rsi_dip:
                # Calculate strength score
                ema_separation = (ema21[i] - ema50[i]) / ema50[i] * 100
                trend_strength = min(ema_separation * 10, 30)  # Cap at 30
                rsi_discount = (35 - rsi14[i]) / 10  # More discount = higher score
                
                score = trend_strength + rsi_discount
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'rsi': rsi14[i],
                    'ema21': ema21[i],
                    'ema50': ema50[i],
                    'reason': f"Trend pullback: RSI={rsi14[i]:.1f}, EMA21>{ema50[i]:.2f}, HH pattern"
                })
        
        return signals
    
    def test_volume_breakout_continuation(self, df: pd.DataFrame) -> List[Dict]:
        """
        VOLUME BREAKOUT CONTINUATION
        Logic: High breakout + volume spike (3x avg) + hold for trend continuation
        Theory: Real breakouts have volume, fake ones don't. Wait for confirmation.
        """
        signals = []
        if len(df) < 100:
            return signals
            
        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values
        
        sma20 = self.calc_sma(close, 20)
        
        for i in range(50, len(df)):
            # 20-day high breakout
            if i >= 20:
                is_breakout = high[i] > np.max(high[i-20:i])
            else:
                is_breakout = False
            
            # Volume confirmation (3x recent average)
            vol_multiplier = self.calc_volume_profile(volume[:i+1], 20)
            volume_surge = vol_multiplier >= 3.0
            
            # Price above SMA20 (trend filter)
            above_sma = close[i] > sma20[i] if not np.isnan(sma20[i]) else False
            
            if is_breakout and volume_surge and above_sma:
                # Score based on volume surge and breakout strength
                breakout_pct = (high[i] - np.max(high[i-20:i-1])) / np.max(high[i-20:i-1]) * 100
                
                score = vol_multiplier * 5 + breakout_pct * 20
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'volume_mult': vol_multiplier,
                    'breakout_pct': breakout_pct,
                    'reason': f"Volume breakout: {vol_multiplier:.1f}x vol, {breakout_pct:.2f}% breakout"
                })
        
        return signals
    
    def test_multi_timeframe_momentum(self, df: pd.DataFrame) -> List[Dict]:
        """
        MULTI-TIMEFRAME MOMENTUM ALIGNMENT
        Logic: Simulate 4H uptrend + 1H entry timing (RSI not overbought)
        Theory: Align with higher timeframe momentum but time entries on lower TF
        """
        signals = []
        if len(df) < 200:
            return signals
            
        close = df['close'].values
        
        # Simulate 4H by taking every 4th bar
        close_4h = close[::4]
        rsi14_1h = self.calc_rsi(close, 14)
        
        # 4H trend (simplified)
        ema21_4h = self.calc_ema(close_4h, 21)
        ema50_4h = self.calc_ema(close_4h, 50)
        
        for i in range(200, len(df)):
            # Map to 4H index
            h4_idx = i // 4
            if h4_idx >= len(ema21_4h) or h4_idx >= len(ema50_4h):
                continue
            
            # 4H uptrend condition
            h4_uptrend = (not np.isnan(ema21_4h[h4_idx]) and 
                         not np.isnan(ema50_4h[h4_idx]) and 
                         ema21_4h[h4_idx] > ema50_4h[h4_idx] and 
                         close_4h[h4_idx] > ema21_4h[h4_idx])
            
            # 1H entry timing: RSI not overbought but not oversold
            rsi_entry_zone = 35 <= rsi14_1h[i] <= 65
            
            # Recent momentum check (1H)
            if i >= 8:
                recent_momentum = (close[i] - close[i-8]) / close[i-8] * 100
                positive_momentum = recent_momentum > 1  # At least 1% move in 8h
            else:
                positive_momentum = False
            
            if h4_uptrend and rsi_entry_zone and positive_momentum:
                # Score based on 4H trend strength and 1H momentum
                h4_separation = (ema21_4h[h4_idx] - ema50_4h[h4_idx]) / ema50_4h[h4_idx] * 100
                
                score = h4_separation * 5 + recent_momentum * 2
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'rsi_1h': rsi14_1h[i],
                    'h4_trend_strength': h4_separation,
                    'momentum_8h': recent_momentum,
                    'reason': f"MTF momentum: 4H trend={h4_separation:.1f}%, 1H RSI={rsi14_1h[i]:.1f}"
                })
        
        return signals
    
    def test_post_consolidation_breakout(self, df: pd.DataFrame) -> List[Dict]:
        """
        POST-CONSOLIDATION BREAKOUT
        Logic: Tight Bollinger Band squeeze → expansion with volume
        Theory: After periods of low volatility, expansion often continues
        """
        signals = []
        if len(df) < 100:
            return signals
            
        close = df['close'].values
        high = df['high'].values
        volume = df['volume'].values
        
        bb_upper, bb_mid, bb_lower = self.calc_bollinger_bands(close, 20, 2.0)
        
        for i in range(50, len(df)):
            # Bollinger Band squeeze detection
            bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] * 100
            
            # Check if it was squeezed recently (narrow bands)
            if i >= 10:
                was_squeezed = np.min((bb_upper[i-10:i] - bb_lower[i-10:i]) / bb_mid[i-10:i] * 100) < 4
            else:
                was_squeezed = False
            
            # Current expansion (bands widening)
            expanding = bb_width > 6
            
            # Breakout above upper band with volume
            breakout = close[i] > bb_upper[i]
            vol_confirm = self.calc_volume_profile(volume[:i+1], 10) > 1.5
            
            if was_squeezed and expanding and breakout and vol_confirm:
                # Score based on expansion and volume
                expansion_strength = bb_width / 4  # Normalize
                vol_strength = self.calc_volume_profile(volume[:i+1], 10)
                
                score = expansion_strength * 5 + vol_strength * 10
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'bb_width': bb_width,
                    'volume_mult': vol_strength,
                    'reason': f"Post-consolidation: BB width={bb_width:.1f}%, vol={vol_strength:.1f}x"
                })
        
        return signals
    
    def test_btc_strength_alt_rotation(self, df: pd.DataFrame, pair: str) -> List[Dict]:
        """
        BTC STRENGTH → ALT ROTATION
        Logic: When BTC is strong and stable (low volatility), alts often catch up
        Theory: Risk-on rotation from BTC to alts during stable BTC periods
        """
        signals = []
        if len(df) < 100 or pair == "BTCUSDT" or self.btc_data is None:
            return signals
            
        close = df['close'].values
        btc_close = self.btc_data['close'].values
        
        # Align lengths
        min_len = min(len(close), len(btc_close))
        close = close[:min_len]
        btc_close = btc_close[:min_len]
        
        for i in range(50, min_len):
            # BTC 7-day performance (strong but not parabolic)
            if i >= 168:  # 7 days of hourly data
                btc_7d_return = (btc_close[i] - btc_close[i-168]) / btc_close[i-168] * 100
            else:
                btc_7d_return = 0
            
            # BTC recent stability (low volatility in last 24h)
            if i >= 24:
                btc_24h_volatility = np.std(btc_close[i-24:i]) / np.mean(btc_close[i-24:i]) * 100
            else:
                btc_24h_volatility = 100
            
            # Alt underperformance vs BTC (lagging)
            if i >= 168:
                alt_7d_return = (close[i] - close[i-168]) / close[i-168] * 100
            else:
                alt_7d_return = 0
            
            # Conditions
            btc_strong = 5 <= btc_7d_return <= 25  # Strong but not parabolic
            btc_stable = btc_24h_volatility < 3   # Low recent volatility
            alt_lagging = alt_7d_return < btc_7d_return - 5  # Alt underperforming by 5%+
            
            # Alt showing signs of strength (recent 24h uptick)
            if i >= 24:
                alt_24h_return = (close[i] - close[i-24]) / close[i-24] * 100
                alt_uptick = alt_24h_return > 2  # Starting to move
            else:
                alt_uptick = False
            
            if btc_strong and btc_stable and alt_lagging and alt_uptick:
                # Score based on BTC stability and alt catch-up potential
                lag_gap = btc_7d_return - alt_7d_return
                stability_score = max(0, 5 - btc_24h_volatility) * 5  # Higher score for more stability
                
                score = lag_gap * 2 + stability_score + alt_24h_return
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'btc_7d': btc_7d_return,
                    'btc_vol': btc_24h_volatility,
                    'alt_7d': alt_7d_return,
                    'alt_24h': alt_24h_return,
                    'lag_gap': lag_gap,
                    'reason': f"BTC rotation: BTC {btc_7d_return:+.1f}% vs ALT {alt_7d_return:+.1f}%, vol={btc_24h_volatility:.1f}%"
                })
        
        return signals
    
    def test_wyckoff_accumulation(self, df: pd.DataFrame) -> List[Dict]:
        """
        WYCKOFF ACCUMULATION PATTERN
        Logic: Spring pattern - test of support with volume, then markup
        Theory: Smart money accumulation creates specific volume/price patterns
        """
        signals = []
        if len(df) < 100:
            return signals
            
        close = df['close'].values
        low = df['low'].values
        volume = df['volume'].values
        
        for i in range(50, len(df)):
            # Support level from recent lows
            if i >= 30:
                support_level = np.min(low[i-30:i-5])  # Support from bars 30-5 ago
                recent_low = np.min(low[i-5:i])        # Recent test
            else:
                continue
            
            # Spring test: brief break below support with volume
            spring_test = recent_low < support_level * 0.98  # 2% break below
            
            # Volume surge on spring test
            if i >= 10:
                vol_on_spring = np.max([self.calc_volume_profile(volume[:j+1], 10) for j in range(i-5, i)]) > 2.0
            else:
                vol_on_spring = False
            
            # Recovery above support
            recovery = close[i] > support_level * 1.01  # 1% above support
            
            # Recent accumulation (sideways action)
            if i >= 20:
                price_range = (np.max(close[i-20:i]) - np.min(close[i-20:i])) / np.mean(close[i-20:i]) * 100
                sideways = price_range < 15  # Less than 15% range
            else:
                sideways = False
            
            if spring_test and vol_on_spring and recovery and sideways:
                # Score based on volume surge and recovery strength
                recovery_strength = (close[i] - support_level) / support_level * 100
                vol_surge = np.max([self.calc_volume_profile(volume[:j+1], 10) for j in range(i-5, i)])
                
                score = recovery_strength * 10 + vol_surge * 5
                
                signals.append({
                    'bar': i,
                    'price': close[i],
                    'score': score,
                    'support': support_level,
                    'recovery_pct': recovery_strength,
                    'volume_surge': vol_surge,
                    'reason': f"Wyckoff spring: support=${support_level:.4f}, recovery={recovery_strength:.1f}%"
                })
        
        return signals
    
    def test_tool_on_pair(self, pair: str, tool_func, tool_name: str) -> Dict:
        """Test a specific tool on a specific pair"""
        if pair not in self.data:
            return None
            
        df = self.data[pair].copy()
        
        # Get signals
        signals = tool_func(df, pair) if tool_name == "btc_strength_alt_rotation" else tool_func(df)
        
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
        is_signals = [s for s in signals if s['bar'] < self.oos_start]
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
        
        # Calculate forward returns for OOS signals
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
        
        # Determine status (pass if either timeframe > 50% WR)
        status = 'PASS' if (win_rate_8h > 50 or win_rate_24h > 50) else 'FAIL'
        
        return {
            'pair': pair,
            'tool': tool_name,
            'signals_total': len(signals),
            'signals_is': len(is_signals),
            'signals_oos': len(oos_signals),
            'win_rate_8h': win_rate_8h,
            'win_rate_24h': win_rate_24h,
            'avg_return_8h': avg_return_8h,
            'avg_return_24h': avg_return_24h,
            'status': status
        }
    
    def run_full_test_suite(self):
        """Run all new bull market tools on all pairs"""
        print("\\n" + "="*80)
        print("BULL MARKET TOOLS DEVELOPMENT - FULL TEST SUITE")
        print("="*80)
        
        # Define tools to test
        tools_to_test = [
            (self.test_trend_pullback_rsi_dip, "trend_pullback_rsi_dip"),
            (self.test_volume_breakout_continuation, "volume_breakout_continuation"),
            (self.test_multi_timeframe_momentum, "multi_timeframe_momentum"),
            (self.test_post_consolidation_breakout, "post_consolidation_breakout"),
            (self.test_btc_strength_alt_rotation, "btc_strength_alt_rotation"),
            (self.test_wyckoff_accumulation, "wyckoff_accumulation")
        ]
        
        all_results = []
        
        for tool_func, tool_name in tools_to_test:
            print(f"\\nTesting {tool_name}...")
            tool_results = []
            
            for pair in self.pairs:
                result = self.test_tool_on_pair(pair, tool_func, tool_name)
                if result:
                    tool_results.append(result)
                    print(f"  {pair}: {result['signals_oos']} OOS signals, "
                          f"WR_24h={result['win_rate_24h']:.1f}%, "
                          f"Ret_24h={result['avg_return_24h']:+.2f}% -> {result['status']}")
            
            # Calculate tool summary
            if tool_results:
                total_signals = sum(r['signals_oos'] for r in tool_results)
                passing_pairs = sum(1 for r in tool_results if r['status'] == 'PASS')
                avg_wr_24h = np.mean([r['win_rate_24h'] for r in tool_results if r['signals_oos'] > 0])
                avg_ret_24h = np.mean([r['avg_return_24h'] for r in tool_results if r['signals_oos'] > 0])
                
                summary = {
                    'tool': tool_name,
                    'total_oos_signals': total_signals,
                    'passing_pairs': passing_pairs,
                    'total_pairs': len([r for r in tool_results if r['signals_oos'] > 0]),
                    'avg_win_rate_24h': avg_wr_24h,
                    'avg_return_24h': avg_ret_24h,
                    'overall_status': 'PASS' if (passing_pairs > 0 and avg_wr_24h > 50) else 'FAIL'
                }
                
                print(f"  SUMMARY: {total_signals} total signals, {passing_pairs}/{len(tool_results)} pairs passing")
                print(f"  AVG: WR_24h={avg_wr_24h:.1f}%, Ret_24h={avg_ret_24h:+.2f}% -> {summary['overall_status']}")
                
                all_results.extend(tool_results)
                self.results.append(summary)
        
        return all_results
    
    def generate_report(self, results):
        """Generate comprehensive test report"""
        print("\\n" + "="*80)
        print("BULL MARKET TOOLS - FINAL REPORT")
        print("="*80)
        
        # Overall summary
        passing_tools = [r for r in self.results if r['overall_status'] == 'PASS']
        failing_tools = [r for r in self.results if r['overall_status'] == 'FAIL']
        
        print(f"\\n📊 OVERALL RESULTS:")
        print(f"✅ PASSED: {len(passing_tools)} tools")
        print(f"❌ FAILED: {len(failing_tools)} tools")
        print(f"📈 SUCCESS RATE: {len(passing_tools)}/{len(self.results)} ({len(passing_tools)/len(self.results)*100:.1f}%)")
        
        # Detailed results by tool
        print(f"\\n📋 DETAILED RESULTS:")
        for summary in sorted(self.results, key=lambda x: x['avg_win_rate_24h'], reverse=True):
            status_emoji = "✅" if summary['overall_status'] == 'PASS' else "❌"
            print(f"{status_emoji} {summary['tool']:<35} | "
                  f"{summary['total_oos_signals']:>4} signals | "
                  f"{summary['passing_pairs']:>2}/{summary['total_pairs']} pairs | "
                  f"WR: {summary['avg_win_rate_24h']:>5.1f}% | "
                  f"Ret: {summary['avg_return_24h']:>+6.2f}%")
        
        # Best performing tools details
        if passing_tools:
            print(f"\\n🏆 TOP PERFORMING TOOLS:")
            for tool in sorted(passing_tools, key=lambda x: x['avg_win_rate_24h'], reverse=True)[:3]:
                print(f"   {tool['tool']}: {tool['avg_win_rate_24h']:.1f}% WR, {tool['avg_return_24h']:+.2f}% Ret")
        
        # Analysis and recommendations
        print(f"\\n💡 ANALYSIS:")
        if passing_tools:
            print("   • Found viable bull market strategies!")
            print("   • Focus on implementing the top performers")
            print("   • Consider combining multiple signals for higher conviction")
        else:
            print("   • No tools survived OOS validation with fees")
            print("   • Bull market edge remains elusive in current data")
            print("   • Consider different approaches or market regimes")
        
        return {
            'passing_tools': len(passing_tools),
            'total_tools': len(self.results),
            'best_tools': passing_tools[:3] if passing_tools else []
        }


def main():
    """Main execution"""
    print("🚀 Starting Bull Market Tools Development...")
    
    developer = BullMarketToolsDeveloper()
    
    # Load data
    developer.load_data()
    
    if not developer.data:
        print("❌ No data loaded. Check data directory.")
        return
    
    # Run full test suite
    results = developer.run_full_test_suite()
    
    # Generate report
    report_summary = developer.generate_report(results)
    
    print(f"\\n🎯 MISSION STATUS:")
    if report_summary['passing_tools'] > 0:
        print(f"✅ SUCCESS: Found {report_summary['passing_tools']} validated bull market tools")
        print("   Ready for implementation in live bot")
    else:
        print("❌ INCOMPLETE: No tools survived validation")
        print("   Need to iterate on approach or explore different market regimes")
    
    return results, developer.results


if __name__ == "__main__":
    results, summaries = main()