#!/usr/bin/env python3
"""
ADVANCED BULL TOOLS - V2 UNCONVENTIONAL APPROACHES
Focus on practical implementations of advanced techniques that retail traders don't use.

Key insight: Make them PRACTICAL, not academic. If it's too complex to implement properly
in time constraints, implement a practical approximation that captures the core insight.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h"

# Validation settings
FEES = 0.0065  # 0.65% round-trip
MIN_SIGNALS = 15
MIN_WIN_RATE = 55
MIN_PROFIT_FACTOR = 1.5

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT", 
    "AVAXUSDT", "ATOMUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "UNIUSDT", 
    "FILUSDT", "NEARUSDT", "AAVEUSDT", "XLMUSDT"
]

class AdvancedBullTools:
    """Advanced bull market tools using unconventional approaches."""
    
    def __init__(self):
        self.data_cache = {}
        self.btc_data = None
        
    def load_data(self, pair: str) -> pd.DataFrame:
        """Load and cache data."""
        if pair not in self.data_cache:
            file_path = DATA_DIR / f"{pair}_1h.csv"
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            self.data_cache[pair] = df
            
            if pair == "BTCUSDT":
                self.btc_data = df
                
        return self.data_cache[pair]
    
    # ================== PRACTICAL IMPLEMENTATIONS ==================
    
    def hurst_regime_detector(self, df: pd.DataFrame) -> dict:
        """
        Hurst exponent regime detection - PRACTICAL VERSION
        
        Core insight: H > 0.5 = trending, H < 0.5 = mean-reverting
        Implementation: Simplified R/S analysis
        """
        if len(df) < 200:
            return {"signal": False, "reason": "Insufficient data"}
        
        close = df['close'].values
        
        # Calculate Hurst exponent for last 100 bars
        def simple_hurst(prices, max_lag=20):
            """Simplified Hurst calculation"""
            log_prices = np.log(prices)
            lags = range(2, min(max_lag, len(log_prices) // 3))
            
            rs_ratios = []
            for lag in lags:
                # Split data into non-overlapping periods
                n_periods = len(log_prices) // lag
                if n_periods < 3:
                    continue
                    
                period_rs = []
                for i in range(n_periods):
                    period_data = log_prices[i*lag:(i+1)*lag] 
                    if len(period_data) < lag:
                        continue
                        
                    # Calculate R/S for this period
                    mean_val = np.mean(period_data)
                    cumsum_dev = np.cumsum(period_data - mean_val)
                    R = np.max(cumsum_dev) - np.min(cumsum_dev)
                    S = np.std(period_data)
                    
                    if S > 0:
                        period_rs.append(R / S)
                
                if len(period_rs) >= 2:
                    rs_ratios.append((lag, np.mean(period_rs)))
            
            if len(rs_ratios) < 4:
                return np.nan
                
            # Linear regression in log-log space
            log_lags = np.log([x[0] for x in rs_ratios])
            log_rs = np.log([x[1] for x in rs_ratios if x[1] > 0])
            
            if len(log_lags) != len(log_rs) or len(log_rs) < 4:
                return np.nan
                
            slope, _, _, _, _ = stats.linregress(log_lags[:len(log_rs)], log_rs)
            return slope
        
        current_hurst = simple_hurst(close[-120:])
        
        if np.isnan(current_hurst):
            return {"signal": False, "reason": "Invalid Hurst"}
        
        # Also calculate previous period for regime change detection
        prev_hurst = simple_hurst(close[-200:-80]) if len(close) >= 200 else current_hurst
        
        # Signal logic
        if current_hurst > 0.58:  # Strong trending regime
            # Get momentum direction
            momentum_12h = (close[-1] - close[-12]) / close[-12] * 100 if len(close) >= 12 else 0
            momentum_24h = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
            
            # Trend acceleration check
            accel = abs(momentum_12h) > abs(momentum_24h) * 0.5 if momentum_24h != 0 else True
            
            if abs(momentum_24h) > 3 and accel:  # Strong trending move with acceleration
                direction = "long" if momentum_24h > 0 else "short"
                
                # Score based on regime strength and momentum
                regime_strength = (current_hurst - 0.5) * 200
                momentum_score = min(abs(momentum_24h) * 2, 20)  # Cap at 20
                regime_change_bonus = 10 if current_hurst > prev_hurst + 0.05 else 0
                
                score = regime_strength + momentum_score + regime_change_bonus
                
                return {
                    "signal": True,
                    "direction": direction,
                    "score": score,
                    "reason": f"HURST TREND: H={current_hurst:.3f}, mom_24h={momentum_24h:.1f}%"
                }
        
        elif current_hurst < 0.42:  # Strong mean-reverting regime
            # Look for oversold/overbought conditions to fade
            rsi_14 = self.calculate_rsi(close, 14)
            if np.isnan(rsi_14):
                return {"signal": False, "reason": "Invalid RSI"}
            
            # Bollinger Bands for extreme detection
            bb_period = 20
            if len(close) >= bb_period:
                sma = np.mean(close[-bb_period:])
                std = np.std(close[-bb_period:])
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                
                current_price = close[-1]
                
                # Mean reversion signals
                extreme_oversold = (rsi_14 < 25 and current_price < bb_lower)
                extreme_overbought = (rsi_14 > 75 and current_price > bb_upper)
                
                if extreme_oversold or extreme_overbought:
                    direction = "long" if extreme_oversold else "short"
                    
                    # Score based on regime strength and extremity
                    regime_strength = (0.5 - current_hurst) * 200
                    rsi_extreme = max(25 - rsi_14, 0) if extreme_oversold else max(rsi_14 - 75, 0)
                    bb_extreme = abs(current_price - sma) / std
                    
                    score = regime_strength + rsi_extreme * 3 + bb_extreme * 10
                    
                    return {
                        "signal": True,
                        "direction": direction,
                        "score": score,
                        "reason": f"HURST REVERT: H={current_hurst:.3f}, RSI={rsi_14:.1f}, BB_dev={bb_extreme:.2f}"
                    }
        
        return {"signal": False, "reason": f"Neutral regime: H={current_hurst:.3f}"}
    
    def correlation_breakdown_detector(self, pair: str, df: pd.DataFrame) -> dict:
        """
        Correlation regime breaks - PRACTICAL VERSION
        
        Core insight: When alt-BTC correlation suddenly drops, independent moves begin
        Implementation: Rolling correlation with regime change detection
        """
        if pair == "BTCUSDT" or self.btc_data is None or len(df) < 200:
            return {"signal": False, "reason": "BTC pair or insufficient data"}
        
        # Align data
        min_len = min(len(df), len(self.btc_data))
        alt_returns = df['returns'].values[-min_len:]
        btc_returns = self.btc_data['returns'].values[-min_len:]
        
        # Remove NaN values
        valid_idx = ~(np.isnan(alt_returns) | np.isnan(btc_returns))
        alt_returns = alt_returns[valid_idx]
        btc_returns = btc_returns[valid_idx]
        
        if len(alt_returns) < 100:
            return {"signal": False, "reason": "Insufficient clean data"}
        
        # Calculate rolling correlations
        window_short = 24  # 1 day
        window_long = 168  # 1 week
        
        if len(alt_returns) < window_long + 24:
            return {"signal": False, "reason": "Need more history"}
        
        # Recent correlation
        recent_corr = np.corrcoef(alt_returns[-window_short:], btc_returns[-window_short:])[0, 1]
        
        # Historical correlation  
        hist_corr = np.corrcoef(alt_returns[-window_long:-24], btc_returns[-window_long:-24])[0, 1]
        
        if np.isnan(recent_corr) or np.isnan(hist_corr):
            return {"signal": False, "reason": "Invalid correlations"}
        
        # Correlation breakdown detection
        corr_drop = hist_corr - recent_corr
        significant_breakdown = corr_drop > 0.3  # Correlation dropped by >0.3
        was_correlated = hist_corr > 0.5  # Was previously correlated
        
        # Alt momentum independence 
        alt_momentum = (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100 if len(df) >= 24 else 0
        btc_momentum = (self.btc_data['close'].iloc[-1] - self.btc_data['close'].iloc[-24]) / self.btc_data['close'].iloc[-24] * 100 if len(self.btc_data) >= 24 else 0
        
        momentum_divergence = abs(alt_momentum - btc_momentum) > 5  # Moving independently
        
        if significant_breakdown and was_correlated and momentum_divergence:
            # Direction based on alt's independent momentum
            direction = "long" if alt_momentum > btc_momentum + 2 else "short" if alt_momentum < btc_momentum - 2 else None
            
            if direction:
                score = corr_drop * 100 + abs(alt_momentum - btc_momentum) * 2
                
                return {
                    "signal": True,
                    "direction": direction,
                    "score": score,
                    "reason": f"CORR BREAKDOWN: {hist_corr:.3f}→{recent_corr:.3f}, alt_mom={alt_momentum:.1f}%"
                }
        
        return {"signal": False, "reason": f"Corr stable: {recent_corr:.3f}"}
    
    def regime_momentum_detector(self, df: pd.DataFrame) -> dict:
        """
        Advanced momentum using regime detection - PRACTICAL VERSION
        
        Core insight: Momentum works differently in different volatility regimes
        Implementation: Volatility regime classification + adaptive momentum
        """
        if len(df) < 200:
            return {"signal": False, "reason": "Insufficient data"}
        
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate volatility regimes
        returns = df['returns'].values
        
        # Rolling volatility (20-period)
        vol_window = 20
        if len(returns) < vol_window * 3:
            return {"signal": False, "reason": "Need more data for volatility"}
        
        rolling_vol = pd.Series(returns).rolling(window=vol_window).std().values
        
        # Volatility regime classification
        current_vol = rolling_vol[-1]
        vol_percentiles = np.nanpercentile(rolling_vol[-200:], [25, 50, 75])
        
        if np.isnan(current_vol) or np.any(np.isnan(vol_percentiles)):
            return {"signal": False, "reason": "Invalid volatility data"}
        
        # Classify regime
        if current_vol <= vol_percentiles[0]:
            vol_regime = "low"
            momentum_threshold = 2  # Lower threshold in low vol
        elif current_vol <= vol_percentiles[2]:
            vol_regime = "medium"
            momentum_threshold = 4  # Medium threshold
        else:
            vol_regime = "high"
            momentum_threshold = 8  # Higher threshold in high vol
        
        # Adaptive momentum calculation
        if vol_regime == "low":
            # In low vol, use shorter periods for sensitivity
            momentum_period = 8
        elif vol_regime == "medium":
            # Medium vol, standard periods
            momentum_period = 12
        else:
            # High vol, longer periods to avoid noise
            momentum_period = 24
        
        if len(close) < momentum_period + 12:
            return {"signal": False, "reason": "Insufficient data for momentum"}
        
        # Calculate momentum
        momentum = (close[-1] - close[-momentum_period]) / close[-momentum_period] * 100
        
        # Volume confirmation - adapted to regime
        vol_multiplier = self.calculate_volume_multiplier(volume, 10)
        
        if vol_regime == "low":
            vol_threshold = 1.2  # Lower volume threshold
        elif vol_regime == "medium":
            vol_threshold = 1.5  # Medium volume threshold  
        else:
            vol_threshold = 2.0  # Higher volume threshold
        
        # Momentum acceleration (change in momentum rate)
        if momentum_period >= 16:
            prev_momentum = (close[-momentum_period//2] - close[-momentum_period]) / close[-momentum_period] * 100
            momentum_accel = abs(momentum) > abs(prev_momentum) * 1.2  # Accelerating
        else:
            momentum_accel = True  # Skip acceleration check for short periods
        
        # Signal conditions
        strong_momentum = abs(momentum) >= momentum_threshold
        volume_confirmation = vol_multiplier >= vol_threshold
        
        # Additional filter: RSI not extreme (avoid buying tops/selling bottoms)
        rsi = self.calculate_rsi(close, 14)
        rsi_ok = 25 < rsi < 75 if not np.isnan(rsi) else True
        
        if strong_momentum and volume_confirmation and momentum_accel and rsi_ok:
            direction = "long" if momentum > 0 else "short"
            
            # Regime-adapted scoring
            base_score = abs(momentum) * 2
            vol_score = vol_multiplier * 10
            regime_bonus = {"low": 20, "medium": 10, "high": 5}[vol_regime]  # Prefer low vol signals
            accel_bonus = 10 if momentum_accel else 0
            
            score = base_score + vol_score + regime_bonus + accel_bonus
            
            return {
                "signal": True,
                "direction": direction,
                "score": score,
                "reason": f"REGIME MOMENTUM: {vol_regime}_vol, mom={momentum:.1f}%, vol={vol_multiplier:.1f}x"
            }
        
        return {"signal": False, "reason": f"Weak signal: {vol_regime}_vol, mom={momentum:.1f}%"}
    
    def volume_profile_reversion(self, df: pd.DataFrame) -> dict:
        """
        Volume profile mean reversion - PRACTICAL VERSION
        
        Core insight: Price tends to revert to high-volume areas (value areas)
        Implementation: VWAP and volume-weighted support/resistance
        """
        if len(df) < 100:
            return {"signal": False, "reason": "Insufficient data"}
        
        # Calculate VWAP over different periods
        def calc_vwap(data, period):
            if len(data) < period:
                return np.nan
            recent = data.tail(period)
            typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
            vwap = (typical_price * recent['volume']).sum() / recent['volume'].sum()
            return vwap
        
        current_price = df['close'].iloc[-1]
        
        # Multiple VWAP periods for confluence
        vwap_short = calc_vwap(df, 24)    # 1 day
        vwap_medium = calc_vwap(df, 72)   # 3 days  
        vwap_long = calc_vwap(df, 168)    # 1 week
        
        if any(np.isnan(x) for x in [vwap_short, vwap_medium, vwap_long]):
            return {"signal": False, "reason": "Invalid VWAP"}
        
        # Distance from VWAPs
        dist_short = (current_price - vwap_short) / vwap_short * 100
        dist_medium = (current_price - vwap_medium) / vwap_medium * 100
        dist_long = (current_price - vwap_long) / vwap_long * 100
        
        # Volume-weighted support/resistance levels
        window = 168  # 1 week
        recent_data = df.tail(window)
        
        # Create volume profile (simplified)
        price_levels = []
        volumes = []
        
        for _, row in recent_data.iterrows():
            # Use candle midpoint as price level
            mid_price = (row['high'] + row['low']) / 2
            price_levels.append(mid_price)
            volumes.append(row['volume'])
        
        # Find high volume areas (POC - Point of Control)
        if len(price_levels) < 10:
            return {"signal": False, "reason": "Insufficient volume data"}
        
        # Volume-weighted average price in recent range
        volume_weights = np.array(volumes)
        price_array = np.array(price_levels)
        
        # Find the highest volume cluster
        vol_weighted_center = np.average(price_array, weights=volume_weights)
        
        # Distance from volume center
        dist_from_poc = (current_price - vol_weighted_center) / vol_weighted_center * 100
        
        # Signal conditions
        far_from_vwaps = abs(dist_short) > 2 or abs(dist_medium) > 3  # Far from VWAP
        far_from_poc = abs(dist_from_poc) > 4  # Far from high-volume area
        
        # VWAP confluence (multiple VWAPs pointing same direction)
        vwap_consensus = (
            (dist_short > 1 and dist_medium > 1.5) or  # Above both VWAPs
            (dist_short < -1 and dist_medium < -1.5)    # Below both VWAPs
        )
        
        # Volume momentum (recent volume increasing)
        vol_momentum = self.calculate_volume_multiplier(df['volume'].values, 5)
        increasing_volume = vol_momentum > 1.3
        
        if far_from_vwaps and far_from_poc and vwap_consensus:
            # Mean reversion signal
            if dist_short > 2 and dist_medium > 2:  # Price above VWAPs
                direction = "short"  # Expect reversion down
            elif dist_short < -2 and dist_medium < -2:  # Price below VWAPs
                direction = "long"  # Expect reversion up
            else:
                return {"signal": False, "reason": "Mixed VWAP signals"}
            
            # Score based on distance and volume
            distance_score = min(abs(dist_medium) * 5, 30)  # Cap at 30
            poc_score = min(abs(dist_from_poc) * 2, 20)     # Cap at 20
            volume_score = vol_momentum * 5
            
            score = distance_score + poc_score + volume_score
            
            return {
                "signal": True,
                "direction": direction,
                "score": score,
                "reason": f"VOLUME REVERT: VWAP_dist={dist_medium:.1f}%, POC_dist={dist_from_poc:.1f}%"
            }
        
        return {"signal": False, "reason": f"Near value: VWAP={dist_medium:.1f}%"}
    
    def cross_asset_rotation(self, pair: str, df: pd.DataFrame) -> dict:
        """
        Cross-asset rotation detector - PRACTICAL VERSION
        
        Core insight: Money flows between assets in predictable patterns
        Implementation: Relative strength vs BTC with momentum confirmation
        """
        if pair == "BTCUSDT" or self.btc_data is None or len(df) < 200:
            return {"signal": False, "reason": "BTC pair or insufficient data"}
        
        # Calculate relative performance metrics
        min_len = min(len(df), len(self.btc_data))
        
        # Different timeframes for relative strength
        periods = [24, 72, 168]  # 1 day, 3 days, 1 week
        rel_performances = {}
        
        for period in periods:
            if min_len < period + 1:
                continue
                
            # Alt performance
            alt_return = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period] * 100
            
            # BTC performance
            btc_return = (self.btc_data['close'].iloc[-1] - self.btc_data['close'].iloc[-period]) / self.btc_data['close'].iloc[-period] * 100
            
            # Relative performance
            rel_perf = alt_return - btc_return
            rel_performances[f"{period}h"] = rel_perf
        
        if len(rel_performances) < 2:
            return {"signal": False, "reason": "Insufficient period data"}
        
        # Rotation detection logic
        short_rel = rel_performances.get("24h", 0)
        med_rel = rel_performances.get("72h", 0)
        long_rel = rel_performances.get("168h", 0)
        
        # Momentum building (getting stronger relative performance)
        if len(rel_performances) >= 3:
            momentum_building = abs(short_rel) > abs(med_rel) * 0.7 and abs(med_rel) > abs(long_rel) * 0.5
        else:
            momentum_building = True
        
        # BTC market conditions
        btc_trend = (self.btc_data['close'].iloc[-1] - self.btc_data['close'].iloc[-72]) / self.btc_data['close'].iloc[-72] * 100
        btc_stable = abs(btc_trend) < 15  # BTC not in extreme move
        
        # Volume confirmation
        alt_vol_mult = self.calculate_volume_multiplier(df['volume'].values, 10)
        volume_increase = alt_vol_mult > 1.4
        
        # Signal conditions for rotation
        strong_relative = abs(short_rel) > 5  # Strong relative performance  
        consistent_direction = (short_rel > 0 and med_rel > 0) or (short_rel < 0 and med_rel < 0)
        
        # Additional filter: alt not overbought/oversold
        alt_rsi = self.calculate_rsi(df['close'].values, 14)
        rsi_reasonable = 30 < alt_rsi < 70 if not np.isnan(alt_rsi) else True
        
        if strong_relative and consistent_direction and momentum_building and volume_increase and rsi_reasonable:
            direction = "long" if short_rel > 0 else "short"
            
            # Scoring
            rel_strength_score = abs(short_rel) * 2
            consistency_score = 20 if consistent_direction else 0
            momentum_score = 15 if momentum_building else 0
            volume_score = alt_vol_mult * 5
            btc_stability_bonus = 10 if btc_stable else 0
            
            score = rel_strength_score + consistency_score + momentum_score + volume_score + btc_stability_bonus
            
            return {
                "signal": True,
                "direction": direction,
                "score": score,
                "reason": f"ROTATION: rel_24h={short_rel:.1f}%, vol={alt_vol_mult:.1f}x, BTC={btc_trend:.1f}%"
            }
        
        return {"signal": False, "reason": f"No rotation: rel={short_rel:.1f}%"}
    
    # ================== HELPER FUNCTIONS ==================
    
    def calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return np.nan
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Simple moving average
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_volume_multiplier(self, volumes: np.array, period: int = 20) -> float:
        """Calculate volume multiplier vs average."""
        if len(volumes) < period + 1:
            return 1.0
            
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-period-1:-1])  # Exclude current bar
        
        if avg_vol == 0:
            return 1.0
            
        return current_vol / avg_vol


def backtest_advanced_tool(tool_func, pair: str, data: pd.DataFrame, 
                          hold_hours: int = 8, stop_loss: float = 0.05) -> dict:
    """Backtest an advanced tool."""
    
    # OOS period: second half
    split_point = len(data) // 2
    oos_data = data.iloc[split_point:].copy().reset_index(drop=True)
    
    trades = []
    tools = AdvancedBullTools()
    
    # Load BTC data if needed
    if pair != "BTCUSDT":
        try:
            tools.load_data("BTCUSDT")
        except:
            pass
    
    # Signal generation loop
    for i in range(200, len(oos_data) - hold_hours - 1):
        window_data = oos_data.iloc[:i+1].copy()
        
        # Get signal
        if tool_func.__name__ in ['correlation_breakdown_detector', 'cross_asset_rotation']:
            signal = tool_func(pair, window_data)
        else:
            signal = tool_func(window_data)
        
        if signal.get("signal", False):
            entry_price = oos_data.iloc[i]['close']
            direction = signal["direction"]
            
            # Calculate exit
            if direction == "long":
                stop_price = entry_price * (1 - stop_loss)
                
                # Check stop loss
                exit_idx = None
                for j in range(i + 1, min(i + hold_hours + 1, len(oos_data))):
                    if oos_data.iloc[j]['low'] <= stop_price:
                        exit_idx = j
                        exit_price = stop_price
                        break
                
                if exit_idx is None:
                    exit_idx = min(i + hold_hours, len(oos_data) - 1)
                    exit_price = oos_data.iloc[exit_idx]['close']
                
                gross_return = (exit_price - entry_price) / entry_price
                
            else:  # short
                stop_price = entry_price * (1 + stop_loss)
                
                # Check stop loss
                exit_idx = None
                for j in range(i + 1, min(i + hold_hours + 1, len(oos_data))):
                    if oos_data.iloc[j]['high'] >= stop_price:
                        exit_idx = j
                        exit_price = stop_price
                        break
                
                if exit_idx is None:
                    exit_idx = min(i + hold_hours, len(oos_data) - 1)
                    exit_price = oos_data.iloc[exit_idx]['close']
                
                gross_return = (entry_price - exit_price) / entry_price
            
            net_return = gross_return - FEES
            
            trades.append({
                'entry_idx': i,
                'exit_idx': exit_idx,
                'net_return': net_return,
                'direction': direction,
                'score': signal.get("score", 0),
                'reason': signal.get("reason", "")
            })
    
    # Calculate metrics
    if not trades:
        return {'pair': pair, 'total_signals': 0, 'win_rate': 0, 'avg_return': 0, 'profit_factor': 0}
    
    returns = [t['net_return'] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    
    win_rate = len(wins) / len(returns) * 100
    avg_return = np.mean(returns) * 100
    
    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 1e-6
    profit_factor = total_wins / total_losses
    
    return {
        'pair': pair,
        'total_signals': len(trades),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'profit_factor': profit_factor,
        'trades': trades
    }


def main():
    """Run validation on advanced bull tools."""
    
    print("ADVANCED BULL TOOLS V2 - UNCONVENTIONAL APPROACHES")
    print("=" * 60)
    
    tools = AdvancedBullTools()
    
    # Tool configurations
    tool_configs = {
        'hurst_regime_detector': (8, 0.05),
        'correlation_breakdown_detector': (12, 0.05), 
        'regime_momentum_detector': (8, 0.06),
        'volume_profile_reversion': (6, 0.04),
        'cross_asset_rotation': (12, 0.05)
    }
    
    results = {}
    
    for tool_name, (hold_hours, stop_loss) in tool_configs.items():
        print(f"\n🔬 Testing {tool_name}...")
        tool_results = []
        
        for pair in PAIRS:
            try:
                data = tools.load_data(pair)
                
                # Get tool function
                tool_func = getattr(tools, tool_name)
                
                result = backtest_advanced_tool(tool_func, pair, data, hold_hours, stop_loss)
                tool_results.append(result)
                
                if result['total_signals'] > 0:
                    print(f"  {pair}: {result['total_signals']} signals, {result['win_rate']:.1f}% WR, {result['avg_return']:+.2f}%")
                
            except Exception as e:
                print(f"  {pair}: Error - {e}")
        
        results[tool_name] = tool_results
    
    # Analysis
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    
    passed_tools = []
    failed_tools = []
    
    for tool_name, tool_results in results.items():
        total_signals = sum(r['total_signals'] for r in tool_results)
        
        if total_signals == 0:
            print(f"\n❌ {tool_name}: No signals")
            failed_tools.append((tool_name, "No signals"))
            continue
        
        if total_signals < MIN_SIGNALS:
            print(f"\n❌ {tool_name}: Only {total_signals} signals")
            failed_tools.append((tool_name, f"Only {total_signals} signals"))
            continue
        
        # Aggregate metrics
        all_results = [r for r in tool_results if r['total_signals'] > 0]
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_return = np.mean([r['avg_return'] for r in all_results])
        avg_pf = np.mean([r['profit_factor'] for r in all_results])
        passing_pairs = sum(1 for r in all_results if r['win_rate'] >= MIN_WIN_RATE or r['profit_factor'] >= MIN_PROFIT_FACTOR)
        
        print(f"\n📊 {tool_name}:")
        print(f"  Signals: {total_signals}")
        print(f"  Win rate: {avg_win_rate:.1f}%") 
        print(f"  Avg return: {avg_return:+.2f}%")
        print(f"  Profit factor: {avg_pf:.2f}")
        print(f"  Passing pairs: {passing_pairs}/{len(all_results)}")
        
        # Check validation
        passes = (avg_win_rate >= MIN_WIN_RATE or avg_pf >= MIN_PROFIT_FACTOR) and total_signals >= MIN_SIGNALS
        
        if passes and passing_pairs >= 1:
            tier = "T2" if avg_win_rate >= 65 and total_signals >= 30 else "T3"
            print(f"  ✅ PASSED ({tier})")
            passed_tools.append((tool_name, tier, avg_win_rate, avg_return, total_signals))
        else:
            print(f"  ❌ FAILED")
            failed_tools.append((tool_name, "Below threshold"))
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n✅ PASSED: {len(passed_tools)}")
    for tool_name, tier, wr, ret, sigs in passed_tools:
        print(f"  🏆 {tool_name} ({tier}): {wr:.1f}% WR, {ret:+.2f}% return, {sigs} signals")
    
    print(f"\n❌ FAILED: {len(failed_tools)}")
    for tool_name, reason in failed_tools:
        print(f"  💀 {tool_name}: {reason}")
    
    # Save results
    import json
    with open("/Users/lucasaust/code/Crypto-trading-bot/advanced_tools_results.json", "w") as f:
        # Prepare for JSON serialization
        serializable_results = {}
        for tool, tool_results in results.items():
            serializable_results[tool] = []
            for result in tool_results:
                clean_result = {k: (float(v) if isinstance(v, np.floating) else 
                                  int(v) if isinstance(v, np.integer) else v) 
                               for k, v in result.items() if k != 'trades'}
                serializable_results[tool].append(clean_result)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to advanced_tools_results.json")
    
    return results, passed_tools, failed_tools


if __name__ == "__main__":
    main()