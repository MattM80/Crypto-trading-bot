#!/usr/bin/env python3
"""
ROUND 3 VALIDATION: Advanced Market Tools & Near-Miss Optimization

Mission: Find MORE edge for crypto trading bot.
Focus areas:
1. Choppy/sideways market tools (60%+ of market time)
2. Improved near-misses from round 2
3. Combo signals (multiple confirmations)
4. Short-side bull market tools (extreme greed)

Validation requirements:
- Data: binance_1h_extended (3yr, 16 pairs)
- OOS: second half
- 0.65% RT fees
- Trailing stops (8% trail, 12% hard stop, max 336h hold)
- Non-overlapping trades (min 48h between entries)
- Step by 8 bars for speed
- Min 30 signals, WR > 55% OR PF > 1.5
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h_extended"
FEES = 0.0065

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT",
         "AVAXUSDT", "ATOMUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "UNIUSDT",
         "FILUSDT", "NEARUSDT", "AAVEUSDT", "XLMUSDT"]

_data_cache = {}

def load_data(pair):
    if pair not in _data_cache:
        df = pd.read_csv(DATA_DIR / f"{pair}_1h.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        _data_cache[pair] = df
    return _data_cache[pair]

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_sma(close, period):
    if len(close) < period: return close[-1]
    return np.mean(close[-period:])

def calc_ema(close, period):
    if len(close) < period: return close[-1]
    mult = 2 / (period + 1)
    result = close[0]
    for price in close[1:]:
        result = (price * mult) + (result * (1 - mult))
    return result

def calc_bb_position(close, period=20):
    """Calculate Bollinger Band position (-2 to +2)"""
    if len(close) < period:
        return 0
    sma = calc_sma(close, period)
    std = np.std(close[-period:])
    if std == 0:
        return 0
    return (close[-1] - sma) / std

def calc_volume_ratio(volume, short=10, long=50):
    """Volume ratio: recent vs longer term"""
    if len(volume) < long:
        return 1.0
    recent = np.mean(volume[-short:])
    longer = np.mean(volume[-long:])
    return recent / longer if longer > 0 else 1.0

def calc_atr_pct(high, low, close, period=14):
    """ATR as percentage of price"""
    if len(close) < period + 1:
        return 0
    tr_list = []
    for i in range(1, min(len(close), period + 1)):
        tr = max(
            high[-i] - low[-i],
            abs(high[-i] - close[-i-1]),
            abs(low[-i] - close[-i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list)
    return (atr / close[-1]) * 100

def is_bull_regime(btc_close, bar_idx):
    """Bull regime: BTC 30d return > 8% AND price > 50 SMA"""
    if bar_idx < 720:  # 30 days = 720 hours
        return False
    ret_30d = (btc_close[bar_idx] - btc_close[bar_idx-720]) / btc_close[bar_idx-720]
    sma50 = calc_sma(btc_close[:bar_idx+1], 50)
    return ret_30d > 0.08 and btc_close[bar_idx] > sma50

def is_bear_regime(btc_close, bar_idx):
    """Bear regime: BTC 30d return < -8%"""
    if bar_idx < 720:
        return False
    ret_30d = (btc_close[bar_idx] - btc_close[bar_idx-720]) / btc_close[bar_idx-720]
    return ret_30d < -0.08

def is_chop_regime(btc_close, bar_idx):
    """Choppy/sideways: Not bull, not bear"""
    return not is_bull_regime(btc_close, bar_idx) and not is_bear_regime(btc_close, bar_idx)

def trailing_exit(close_arr, entry_bar, max_hold, direction, trail_pct=0.08, hard_stop_pct=0.12):
    """Trailing stop exit logic"""
    entry_price = close_arr[entry_bar]
    max_bar = min(entry_bar + max_hold, len(close_arr) - 1)
    
    if direction == 'long':
        hard_stop = entry_price * (1 - hard_stop_pct)
        best_price = entry_price
        
        for bar in range(entry_bar + 1, max_bar + 1):
            current = close_arr[bar]
            
            if current <= hard_stop:
                return current, bar - entry_bar, "hard_stop"
            
            if current > best_price:
                best_price = current
            
            trail_stop = best_price * (1 - trail_pct)
            if current <= trail_stop and best_price > entry_price * 1.02:
                return current, bar - entry_bar, "trail_stop"
        
        return close_arr[max_bar], max_bar - entry_bar, "time_exit"
    
    else:  # short
        hard_stop = entry_price * (1 + hard_stop_pct)
        best_price = entry_price
        
        for bar in range(entry_bar + 1, max_bar + 1):
            current = close_arr[bar]
            
            if current >= hard_stop:
                return current, bar - entry_bar, "hard_stop"
            
            if current < best_price:
                best_price = current
            
            trail_stop = best_price * (1 + trail_pct)
            if current >= trail_stop and best_price < entry_price * 0.98:
                return current, bar - entry_bar, "trail_stop"
        
        return close_arr[max_bar], max_bar - entry_bar, "time_exit"

# ======================== CATEGORY 1: CHOPPY/SIDEWAYS TOOLS ========================

def bb_mean_reversion_chop(close, high, low, volume, rsi):
    """
    Bollinger Band mean reversion during tight ranges (choppy markets)
    Logic: In low volatility, tight BB environments, extreme moves tend to revert
    """
    if len(close) < 100:
        return None
    
    # Must be in choppy environment (low volatility)
    atr_pct = calc_atr_pct(high, low, close, 20)
    if atr_pct > 3.5:  # Too volatile for mean reversion
        return None
    
    # Bollinger band metrics
    bb_pos = calc_bb_position(close, 20)
    
    # Range-bound check: 30d high/low range < 25%
    if len(close) >= 720:
        range_high = np.max(high[-720:])
        range_low = np.min(low[-720:])
        range_pct = (range_high - range_low) / range_low * 100
        if range_pct > 25:
            return None
    
    # Long setup: extreme oversold in tight range
    if bb_pos < -1.8 and rsi < 30:
        vol_ratio = calc_volume_ratio(volume)
        if vol_ratio > 1.2:  # Volume spike on oversold
            return {
                'tool': 'bb_mean_reversion_chop',
                'direction': 'long',
                'score': abs(bb_pos) * 10 + (30 - rsi) + vol_ratio * 5
            }
    
    # Short setup: extreme overbought in tight range  
    elif bb_pos > 1.8 and rsi > 70:
        vol_ratio = calc_volume_ratio(volume)
        if vol_ratio > 1.2:
            return {
                'tool': 'bb_mean_reversion_chop',
                'direction': 'short',
                'score': bb_pos * 10 + (rsi - 70) + vol_ratio * 5
            }
    
    return None

def range_oscillator_strategy(close, high, low, volume, rsi):
    """
    Range-bound oscillator strategy for choppy markets
    Logic: Trade between established support/resistance levels
    """
    if len(close) < 200:
        return None
    
    # Identify range (using 50d lookback)
    lookback = min(1200, len(close) - 1)  # 50 days max
    range_high = np.max(high[-lookback:])
    range_low = np.min(low[-lookback:])
    range_mid = (range_high + range_low) / 2
    range_pct = (range_high - range_low) / range_low * 100
    
    # Must be in defined range (5-30% range)
    if range_pct < 5 or range_pct > 30:
        return None
    
    # Current position in range
    range_position = (close[-1] - range_low) / (range_high - range_low)
    
    # RSI confirmation
    if range_position < 0.25 and rsi < 35:  # Near bottom, oversold
        # Additional confirmation: recent bounce off low
        recent_low = np.min(low[-48:])  # 2 day low
        if recent_low <= range_low * 1.02:  # Within 2% of range low
            return {
                'tool': 'range_oscillator_strategy',
                'direction': 'long',
                'score': (0.25 - range_position) * 100 + (35 - rsi) + range_pct
            }
    
    elif range_position > 0.75 and rsi > 65:  # Near top, overbought
        recent_high = np.max(high[-48:])
        if recent_high >= range_high * 0.98:  # Within 2% of range high
            return {
                'tool': 'range_oscillator_strategy',
                'direction': 'short',
                'score': (range_position - 0.75) * 100 + (rsi - 65) + range_pct
            }
    
    return None

def low_vol_breakout_anticipation(close, high, low, volume, rsi):
    """
    Low volatility breakout anticipation
    Logic: After consolidation, volume spikes often precede breakouts
    """
    if len(close) < 200:
        return None
    
    # Must be in low volatility environment
    atr_pct = calc_atr_pct(high, low, close, 20)
    if atr_pct > 2.5:  # Must be low vol
        return None
    
    # Consolidation check: tight range for at least 7 days
    consolidation_period = min(168, len(close) - 1)  # 7 days
    cons_high = np.max(high[-consolidation_period:])
    cons_low = np.min(low[-consolidation_period:])
    cons_range = (cons_high - cons_low) / cons_low * 100
    
    if cons_range > 12:  # Too wide for consolidation
        return None
    
    # Volume spike detection
    vol_ratio = calc_volume_ratio(volume, 4, 50)  # 4h vs 50h average
    if vol_ratio < 2.0:  # Need significant volume spike
        return None
    
    # Price action: breaking out of consolidation
    breakout_threshold = 0.015  # 1.5% above/below range
    
    if close[-1] > cons_high * (1 + breakout_threshold):  # Upside breakout
        if 30 < rsi < 70:  # Not extreme
            return {
                'tool': 'low_vol_breakout_anticipation',
                'direction': 'long',
                'score': vol_ratio * 10 + (12 - cons_range) * 2 + ((close[-1] - cons_high) / cons_high * 1000)
            }
    
    elif close[-1] < cons_low * (1 - breakout_threshold):  # Downside breakout
        if 30 < rsi < 70:
            return {
                'tool': 'low_vol_breakout_anticipation',
                'direction': 'short',
                'score': vol_ratio * 10 + (12 - cons_range) * 2 + ((cons_low - close[-1]) / cons_low * 1000)
            }
    
    return None

# ======================== CATEGORY 2: IMPROVED NEAR-MISSES ========================

def trend_structure_long_v2(close, high, low, volume, rsi):
    """
    Improved trend structure - optimized parameters based on round 2 feedback
    Original: 60 signals, 38.3% WR, +1.81% avg, PF=1.47
    Improvements: Tighter entry conditions, better trend definition
    """
    if len(close) < 200:
        return None
    
    # More conservative trend definition (stronger trend required)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    sma200 = calc_sma(close, 200)
    
    # Stronger trend structure required
    if not (close[-1] > ema20 > ema50 > sma200):
        return None
    
    # Higher highs and higher lows over longer period (more robust)
    lookback_bars = 96  # 4 days instead of shorter period
    if len(close) < lookback_bars + 50:
        return None
    
    # More stringent HH/HL check
    recent_highs = [np.max(high[i:i+24]) for i in range(len(high)-lookback_bars, len(high)-24, 24)]
    recent_lows = [np.min(low[i:i+24]) for i in range(len(low)-lookback_bars, len(low)-24, 24)]
    
    if len(recent_highs) < 3 or len(recent_lows) < 3:
        return None
    
    # Trend must be accelerating (recent HH > previous HH)
    if not (recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]):
        return None
    
    # Volume confirmation (trend with volume)
    vol_trend = calc_volume_ratio(volume, 24, 168)
    if vol_trend < 1.1:
        return None
    
    # Momentum confirmation (stronger momentum required)
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    if ret_7d < 0.03:  # Need at least 3% weekly momentum
        return None
    
    # RSI not extreme (tighter range)
    if rsi < 35 or rsi > 65:
        return None
    
    # Entry on pullback to shorter EMA (better entry timing)
    ema_distance = (close[-1] - ema20) / ema20
    if abs(ema_distance) > 0.02:  # Must be within 2% of 20 EMA
        return None
    
    return {
        'tool': 'trend_structure_long_v2',
        'direction': 'long', 
        'score': ret_7d * 100 + vol_trend * 10 + (recent_highs[-1] - recent_highs[-2]) / recent_highs[-2] * 500
    }

def weekly_momentum_pullback_v2(close, high, low, volume, rsi):
    """
    Enhanced weekly momentum pullback - parameter optimization
    Original: 57 signals, 40.4% WR, +1.89% avg, PF=1.47  
    Improvements: Better pullback detection, volume confirmation
    """
    if len(close) < 800:
        return None
    
    # Stronger trend requirements (higher thresholds)
    ret_2w = (close[-1] - close[-336]) / close[-336] if len(close) >= 336 else 0
    ret_4w = (close[-1] - close[-672]) / close[-672] if len(close) >= 672 else 0
    
    if ret_2w < 0.12 or ret_4w < 0.15:  # Higher momentum thresholds
        return None
    
    # Better pullback detection
    ret_48h = (close[-1] - close[-48]) / close[-48]
    ret_24h = (close[-1] - close[-24]) / close[-24]
    
    # Pullback must be meaningful but not panic
    if ret_48h > -0.02 or ret_48h < -0.20:  # 2-20% pullback
        return None
    
    if ret_24h > -0.01:  # Recent weakness required
        return None
    
    # Trend structure intact
    sma200 = calc_sma(close, 200)
    if close[-1] < sma200 * 1.02:  # Must be above 200 SMA with buffer
        return None
    
    # Volume analysis: selling volume should be below average (no panic)
    vol_ratio = calc_volume_ratio(volume, 24, 168)
    if vol_ratio > 1.5:  # Too much volume = panic selling
        return None
    
    # RSI oversold but not extreme (refined range)
    if rsi < 25 or rsi > 50:  # Tighter RSI range
        return None
    
    # Momentum still positive on longer timeframe
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    if ret_7d < 0.01:  # Weekly momentum must be positive
        return None
    
    return {
        'tool': 'weekly_momentum_pullback_v2',
        'direction': 'long',
        'score': ret_4w * 100 + abs(ret_48h) * 50 + (50 - rsi) + ret_7d * 30
    }

def dip_in_uptrend_v2(close, high, low, volume, rsi):
    """
    Enhanced dip buying - better trend and dip detection
    Original: 125 signals, 35.2% WR, +1.02% avg, PF=1.23
    Improvements: Stricter trend requirements, optimal dip size
    """
    if len(close) < 800:
        return None
    
    # Much stronger uptrend required
    ret_4w = (close[-1] - close[-672]) / close[-672] if len(close) >= 672 else 0
    ret_8w = (close[-1] - close[-1344]) / close[-1344] if len(close) >= 1344 else 0
    
    if ret_4w < 0.08 or ret_8w < 0.15:  # Higher trend thresholds
        return None
    
    # Better dip characterization
    recent_high = np.max(high[-168:])  # 1 week high
    dip_size = (close[-1] - recent_high) / recent_high
    
    # Optimal dip size (not too small, not too big)
    if dip_size > -0.03 or dip_size < -0.18:  # 3-18% dip range
        return None
    
    # Must be recent dip (within last 24h)
    hours_since_high = 0
    for i in range(1, min(72, len(high))):  # Look back max 3 days
        if high[-i] == recent_high:
            hours_since_high = i
            break
    
    if hours_since_high > 48:  # Dip must be recent
        return None
    
    # Stronger trend structure requirement
    sma50 = calc_sma(close, 50)
    sma200 = calc_sma(close, 200)
    if not (close[-1] > sma50 > sma200):
        return None
    
    # RSI oversold but recovering (more nuanced)
    if rsi > 45:  # Must be oversold
        return None
    
    # Check for recent RSI bounce (recovery signal)
    if len(close) >= 24:
        prev_rsi = calc_rsi(close[:-12])  # RSI 12h ago
        if calc_rsi(close) <= prev_rsi:  # RSI must be recovering
            return None
    
    # Volume should show buying interest
    vol_ratio = calc_volume_ratio(volume, 8, 48)  # Recent vs medium term
    if vol_ratio < 0.8:  # Some volume required (but not panic)
        return None
    
    return {
        'tool': 'dip_in_uptrend_v2', 
        'direction': 'long',
        'score': abs(dip_size) * 100 + ret_4w * 50 + (45 - rsi) + vol_ratio * 10
    }

# ======================== CATEGORY 3: COMBO SIGNALS ========================

def hurst_accumulation_combo(close, high, low, volume, rsi):
    """
    Combo: Hurst trending + accumulation breakout = double confirmation
    Only fire when both conditions align
    """
    if len(close) < 500:
        return None
    
    # PART 1: Hurst trending component
    if len(close) >= 168:
        returns = np.diff(np.log(close[-168:]))
        if len(returns) > 10:
            # Simple Hurst calculation
            lags = range(2, min(15, len(returns) // 3))
            rs_values = []
            for lag in lags:
                n_periods = len(returns) // lag
                if n_periods >= 3:
                    rs_list = []
                    for i in range(n_periods):
                        period_rets = returns[i*lag:(i+1)*lag]
                        if len(period_rets) == lag:
                            mean_ret = np.mean(period_rets)
                            cumsum_dev = np.cumsum(period_rets - mean_ret)
                            R = np.max(cumsum_dev) - np.min(cumsum_dev)
                            S = np.std(period_rets)
                            if S > 0:
                                rs_list.append(R / S)
                    if rs_list:
                        rs_values.append((lag, np.mean(rs_list)))
            
            if len(rs_values) >= 3:
                log_lags = np.log([x[0] for x in rs_values])
                log_rs = np.log([x[1] for x in rs_values if x[1] > 0])
                if len(log_rs) >= 3:
                    H = np.polyfit(log_lags[:len(log_rs)], log_rs, 1)[0]
                    if H < 0.58:  # Not trending enough
                        return None
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None
    
    # PART 2: Accumulation breakout component
    if len(close) >= 500:
        # Range analysis (consolidation period)
        range_high = np.max(high[-336:-48])  # 2 weeks ago to 2 days ago
        range_low = np.min(low[-336:-48])
        range_pct = (range_high - range_low) / range_low * 100
        
        if range_pct > 25 or range_pct < 3:  # Range too wide or too tight
            return None
        
        # Breakout detection
        if close[-1] <= range_high * 1.008:  # Must break above range
            return None
        
        # Volume confirmation for breakout
        vol_recent = np.mean(volume[-24:])
        vol_range = np.mean(volume[-336:-48])
        vol_ratio = vol_recent / vol_range if vol_range > 0 else 1
        
        if vol_ratio < 1.3:  # Need volume confirmation
            return None
    else:
        return None
    
    # PART 3: Combined entry conditions
    sma50 = calc_sma(close, 50)
    ret_24h = (close[-1] - close[-24]) / close[-24]
    
    if close[-1] <= sma50 or ret_24h <= 0.005:  # Must be above SMA with momentum
        return None
    
    if rsi > 75:  # Not too overbought
        return None
    
    return {
        'tool': 'hurst_accumulation_combo',
        'direction': 'long',
        'score': (H - 0.5) * 100 + vol_ratio * 15 + ret_24h * 100 + ((close[-1] - range_high) / range_high * 1000)
    }

def rsi_volume_trend_combo(close, high, low, volume, rsi):
    """
    RSI + volume + trend structure combo for stronger entries
    Triple confirmation system
    """
    if len(close) < 200:
        return None
    
    # PART 1: Trend structure (all EMAs aligned)
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    ema50 = calc_ema(close, 50)
    
    trend_up = ema12 > ema26 > ema50
    trend_down = ema12 < ema26 < ema50
    
    if not (trend_up or trend_down):
        return None
    
    # PART 2: RSI setup
    if trend_up:
        # For longs: RSI pullback but not extreme
        if rsi < 35 or rsi > 55:
            return None
        # RSI should be recovering
        if len(close) >= 12:
            prev_rsi = calc_rsi(close[:-6])
            if rsi <= prev_rsi:  # Must be improving
                return None
    else:  # trend_down
        # For shorts: RSI rally but not extreme  
        if rsi < 45 or rsi > 65:
            return None
        # RSI should be deteriorating
        if len(close) >= 12:
            prev_rsi = calc_rsi(close[:-6])
            if rsi >= prev_rsi:  # Must be weakening
                return None
    
    # PART 3: Volume confirmation
    vol_ratio = calc_volume_ratio(volume, 6, 30)  # Recent vs medium term
    
    if trend_up and vol_ratio < 1.2:  # Need volume on longs
        return None
    elif trend_down and vol_ratio < 1.1:  # Less volume needed for shorts
        return None
    
    # Additional momentum filter
    ret_24h = (close[-1] - close[-24]) / close[-24]
    if trend_up and ret_24h < -0.02:  # Don't chase falling knives
        return None
    elif trend_down and ret_24h > 0.02:  # Don't chase rising daggers
        return None
    
    direction = 'long' if trend_up else 'short'
    score = vol_ratio * 15
    
    if trend_up:
        score += (55 - rsi) + max(0, ret_24h) * 100
    else:
        score += (rsi - 45) + max(0, -ret_24h) * 100
    
    return {
        'tool': 'rsi_volume_trend_combo',
        'direction': direction,
        'score': score
    }

# ======================== CATEGORY 4: SHORT-SIDE BULL MARKET TOOLS ========================

def extreme_greed_short(close, high, low, volume, rsi):
    """
    Short overbought bull market tops during extreme greed (F&G > 80 simulated)
    Logic: Even in bull markets, parabolic moves need to cool off
    """
    if len(close) < 200:
        return None
    
    # Simulate extreme greed: very strong recent momentum + high RSI
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    ret_3d = (close[-1] - close[-72]) / close[-72] if len(close) >= 72 else 0
    ret_24h = (close[-1] - close[-24]) / close[-24]
    
    # Must have parabolic momentum (extreme greed conditions)
    if ret_7d < 0.15 or ret_3d < 0.08 or ret_24h < 0.03:
        return None
    
    # RSI extremely overbought
    if rsi < 78:
        return None
    
    # Volume should be climactic (exhaustion)
    vol_ratio = calc_volume_ratio(volume, 12, 168)
    if vol_ratio < 2.0:  # Need volume climax
        return None
    
    # Bollinger band extreme
    bb_pos = calc_bb_position(close, 20)
    if bb_pos < 2.2:  # Must be well outside upper BB
        return None
    
    # Price action: look for shooting star / doji patterns
    recent_range = (high[-1] - low[-1]) / close[-1] * 100
    if recent_range < 2.0:  # Need some volatility for reversal
        return None
    
    # Check for upper wick (rejection of highs)
    upper_wick = (high[-1] - max(close[-1], close[-2])) / close[-1] * 100
    if upper_wick < 1.0:  # Need upper wick showing rejection
        return None
    
    return {
        'tool': 'extreme_greed_short',
        'direction': 'short',
        'score': rsi + bb_pos * 10 + vol_ratio * 5 + upper_wick * 10 + ret_3d * 50
    }

def parabolic_exhaustion_short(close, high, low, volume, rsi):
    """
    Parabolic move exhaustion shorts
    Logic: When moves become too steep too fast, they often correct
    """
    if len(close) < 100:
        return None
    
    # Calculate acceleration (rate of change of rate of change)
    if len(close) >= 72:
        ret_1d = (close[-1] - close[-24]) / close[-24]
        ret_2d = (close[-25] - close[-48]) / close[-48] if len(close) >= 48 else 0
        ret_3d = (close[-49] - close[-72]) / close[-72] if len(close) >= 72 else 0
        
        # Acceleration must be increasing (parabolic)
        if not (ret_1d > ret_2d > ret_3d and ret_1d > 0.04):
            return None
    else:
        return None
    
    # High ATR (volatility expansion)
    atr_pct = calc_atr_pct(high, low, close, 14)
    if atr_pct < 4.0:  # Need high volatility
        return None
    
    # RSI divergence check (price higher, RSI lower = bearish divergence)
    if len(close) >= 48:
        price_ratio = close[-1] / close[-48]
        old_rsi = calc_rsi(close[:-48])
        if price_ratio > 1.02 and rsi > old_rsi:  # No divergence
            return None
    
    # Volume climax
    vol_ratio = calc_volume_ratio(volume, 8, 100)
    if vol_ratio < 1.8:
        return None
    
    # Must be overbought
    if rsi < 72:
        return None
    
    return {
        'tool': 'parabolic_exhaustion_short',
        'direction': 'short', 
        'score': ret_1d * 100 + atr_pct * 3 + vol_ratio * 8 + (rsi - 50)
    }

def distribution_volume_short(close, high, low, volume, rsi):
    """
    Smart money distribution detection (volume analysis)
    Logic: High volume on up days with poor follow-through = distribution
    """
    if len(close) < 100:
        return None
    
    # Must be in uptrend context (for distribution to make sense)
    ret_14d = (close[-1] - close[-336]) / close[-336] if len(close) >= 336 else 0
    if ret_14d < 0.05:  # Need uptrend context
        return None
    
    # Analyze recent volume patterns (look for distribution signs)
    # High volume up days followed by weak follow-through
    up_volume_days = []
    down_volume_days = []
    
    for i in range(2, min(15, len(close))):  # Last 2 weeks
        day_return = (close[-i] - close[-i-1]) / close[-i-1]
        day_volume = volume[-i]
        
        if day_return > 0.02:  # Up day > 2%
            up_volume_days.append((day_return, day_volume))
        elif day_return < -0.02:  # Down day > 2%
            down_volume_days.append((abs(day_return), day_volume))
    
    if len(up_volume_days) < 2:
        return None
    
    # Check if recent up days had high volume but poor follow through
    avg_up_volume = np.mean([x[1] for x in up_volume_days])
    avg_all_volume = np.mean(volume[-50:])
    
    if avg_up_volume < avg_all_volume * 1.4:  # Up days should have high volume
        return None
    
    # Recent price action: making lower highs despite up volume days
    recent_highs = [np.max(high[-i-5:-i]) for i in range(0, min(30, len(high)-5), 5)]
    if len(recent_highs) >= 3 and not (recent_highs[0] < recent_highs[1]):  # Not making lower highs
        return None
    
    # RSI showing weakness despite price holding up
    if rsi < 55:  # RSI should be high for distribution
        return None
    
    # Current conditions for entry
    ret_48h = (close[-1] - close[-48]) / close[-48] if len(close) >= 48 else 0
    if ret_48h < -0.01:  # Already declining
        return None
    
    vol_ratio = calc_volume_ratio(volume, 5, 25)
    
    return {
        'tool': 'distribution_volume_short',
        'direction': 'short',
        'score': (avg_up_volume / avg_all_volume) * 20 + rsi + vol_ratio * 5
    }

# ======================== VALIDATION FRAMEWORK ========================

ALL_ROUND3_TOOLS = [
    # Category 1: Choppy/Sideways
    bb_mean_reversion_chop,
    range_oscillator_strategy,
    low_vol_breakout_anticipation,
    
    # Category 2: Improved Near-Misses
    trend_structure_long_v2,
    weekly_momentum_pullback_v2,
    dip_in_uptrend_v2,
    
    # Category 3: Combo Signals
    hurst_accumulation_combo,
    rsi_volume_trend_combo,
    
    # Category 4: Short-Side Bull Market
    extreme_greed_short,
    parabolic_exhaustion_short,
    distribution_volume_short,
]

def validate_tool(tool_func, regime_filter='bull', max_hold=336):
    """Validate a single tool with regime filtering"""
    all_trades = []
    btc_df = load_data("BTCUSDT")
    btc_close = btc_df['close'].values
    
    print(f"  Running {tool_func.__name__}...")
    
    for pair in PAIRS:
        df = load_data(pair)
        close = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        vol = df['volume'].values
        
        # OOS: second half of data
        oos_start = len(close) // 2
        i = oos_start
        last_signal = -max_hold
        
        while i < len(close) - max_hold:
            # Non-overlapping trades
            if i - last_signal < 48:
                i += 8
                continue
            
            # Regime filtering
            if regime_filter == 'bull' and not is_bull_regime(btc_close, i):
                i += 8
                continue
            elif regime_filter == 'bear' and not is_bear_regime(btc_close, i):
                i += 8
                continue
            elif regime_filter == 'chop' and not is_chop_regime(btc_close, i):
                i += 8
                continue
            
            # Get signal
            rsi = calc_rsi(close[:i+1])
            sig = tool_func(close[:i+1], high_arr[:i+1], low_arr[:i+1], vol[:i+1], rsi)
            
            if sig is not None:
                # Execute trade with trailing stops
                exit_price, hold_bars, exit_reason = trailing_exit(
                    close, i, max_hold, sig['direction']
                )
                entry_price = close[i]
                
                # Calculate returns
                if sig['direction'] == 'long':
                    raw_return = (exit_price - entry_price) / entry_price
                else:  # short
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - FEES
                
                all_trades.append({
                    'pair': pair,
                    'bar': i,
                    'direction': sig['direction'],
                    'entry': float(entry_price),
                    'exit': float(exit_price),
                    'raw': float(raw_return),
                    'net': float(net_return),
                    'win': net_return > 0,
                    'hold': int(hold_bars),
                    'exit_reason': exit_reason,
                    'score': sig.get('score', 0)
                })
                last_signal = i
            
            i += 8
    
    return all_trades

def report_results(name, trades, regime=''):
    """Generate detailed report for a tool"""
    if not trades:
        print(f"  {name} ({regime}): NO SIGNALS ❌")
        return None
    
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    avg_ret = np.mean([t['net'] for t in trades]) * 100
    total_ret = sum(t['net'] for t in trades) * 100
    
    # Win/loss analysis
    w_trades = [t['net'] for t in trades if t['win']]
    l_trades = [t['net'] for t in trades if not t['win']]
    avg_win = np.mean(w_trades) * 100 if w_trades else 0
    avg_loss = np.mean(l_trades) * 100 if l_trades else 0
    
    # Profit factor
    gross_profit = sum(w_trades) if w_trades else 0
    gross_loss = abs(sum(l_trades)) if l_trades else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    
    # Other metrics
    max_dd = min(t['net'] for t in trades) * 100
    avg_hold = np.mean([t['hold'] for t in trades])
    
    # Exit analysis
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1
    
    # Pair distribution
    pairs = set(t['pair'] for t in trades)
    
    # Pass criteria
    passed = n >= 30 and (wr >= 55 or pf >= 1.5)
    status = "✅ PASSED" if passed else "❌ KILLED"
    
    print(f"\n  {name} ({regime}): {status}")
    print(f"    Signals: {n} | WR: {wr:.1f}% | Avg: {avg_ret:.2f}% | Total: {total_ret:.1f}%")
    print(f"    PF: {pf:.2f} | Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}% | Max DD: {max_dd:.2f}%")
    print(f"    Hold: {avg_hold:.0f}h ({avg_hold/24:.1f}d) | Pairs: {len(pairs)} | Exits: {exits}")
    
    # Top performing pairs
    pair_stats = {}
    for t in trades:
        pair_stats.setdefault(t['pair'], []).append(t['net'])
    
    if pair_stats:
        top = sorted(pair_stats.items(), key=lambda x: np.mean(x[1]), reverse=True)
        print(f"    Top 3 pairs:")
        for p, rets in top[:3]:
            pw = sum(1 for r in rets if r > 0)
            print(f"      {p}: {len(rets)} trades, {pw}/{len(rets)} wins ({pw/len(rets)*100:.0f}%), avg {np.mean(rets)*100:.2f}%")
    
    return {
        'tool': name,
        'regime': regime,
        'signals': n,
        'wr': wr,
        'avg_ret': avg_ret,
        'pf': pf,
        'passed': passed,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_hold': avg_hold,
        'total_return': total_ret
    }

def main():
    print("=" * 80)
    print("ROUND 3 VALIDATION: Advanced Market Tools & Near-Miss Optimization")
    print("=" * 80)
    print("Data: binance_1h_extended (3yr, 16 pairs) | OOS: 2nd half | Fees: 0.65% RT")
    print("Trailing stops: 8% trail, 12% hard stop | Max hold: 336h | Min gap: 48h")
    print("Pass criteria: Min 30 signals AND (WR > 55% OR PF > 1.5)")
    print("=" * 80)
    
    all_results = []
    
    # Category 1: Choppy/Sideways Tools (test in chop regime)
    print(f"\n{'='*60}")
    print("CATEGORY 1: CHOPPY/SIDEWAYS MARKET TOOLS")
    print(f"{'='*60}")
    
    chop_tools = [bb_mean_reversion_chop, range_oscillator_strategy, low_vol_breakout_anticipation]
    
    for tool in chop_tools:
        print(f"\n{'-'*50}")
        print(f"Testing: {tool.__name__}")
        print(f"{'-'*50}")
        trades = validate_tool(tool, regime_filter='chop')
        result = report_results(tool.__name__, trades, 'CHOP')
        if result:
            all_results.append(result)
    
    # Category 2: Improved Near-Misses (test in bull regime)
    print(f"\n{'='*60}")
    print("CATEGORY 2: IMPROVED NEAR-MISSES FROM ROUND 2")
    print(f"{'='*60}")
    
    improved_tools = [trend_structure_long_v2, weekly_momentum_pullback_v2, dip_in_uptrend_v2]
    
    for tool in improved_tools:
        print(f"\n{'-'*50}")
        print(f"Testing: {tool.__name__}")
        print(f"{'-'*50}")
        trades = validate_tool(tool, regime_filter='bull')
        result = report_results(tool.__name__, trades, 'BULL')
        if result:
            all_results.append(result)
    
    # Category 3: Combo Signals (test in bull regime)
    print(f"\n{'='*60}")
    print("CATEGORY 3: COMBO SIGNALS")
    print(f"{'='*60}")
    
    combo_tools = [hurst_accumulation_combo, rsi_volume_trend_combo]
    
    for tool in combo_tools:
        print(f"\n{'-'*50}")
        print(f"Testing: {tool.__name__}")
        print(f"{'-'*50}")
        trades = validate_tool(tool, regime_filter='bull')
        result = report_results(tool.__name__, trades, 'BULL')
        if result:
            all_results.append(result)
    
    # Category 4: Short-Side Bull Market Tools (test in bull regime)  
    print(f"\n{'='*60}")
    print("CATEGORY 4: SHORT-SIDE BULL MARKET TOOLS")
    print(f"{'='*60}")
    
    short_tools = [extreme_greed_short, parabolic_exhaustion_short, distribution_volume_short]
    
    for tool in short_tools:
        print(f"\n{'-'*50}")
        print(f"Testing: {tool.__name__}")
        print(f"{'-'*50}")
        trades = validate_tool(tool, regime_filter='bull')
        result = report_results(tool.__name__, trades, 'BULL_SHORT')
        if result:
            all_results.append(result)
    
    # Final Summary
    print(f"\n{'='*80}")
    print("ROUND 3 FINAL SCORECARD")
    print(f"{'='*80}")
    
    passed = [r for r in all_results if r['passed']]
    killed = [r for r in all_results if not r['passed']]
    
    print(f"\n✅ PASSED TOOLS ({len(passed)}):")
    if passed:
        for r in sorted(passed, key=lambda x: x['avg_ret'], reverse=True):
            print(f"  {r['tool']} ({r['regime']}): {r['signals']} signals, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    else:
        print("  NONE - All tools failed validation ❌")
    
    print(f"\n❌ KILLED TOOLS ({len(killed)}):")
    if killed:
        for r in sorted(killed, key=lambda x: x['avg_ret'], reverse=True):
            print(f"  {r['tool']} ({r['regime']}): {r['signals']} signals, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    else:
        print("  NONE - All tools passed! 🎉")
    
    print(f"\n{'='*80}")
    print("ROUND 3 ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return all_results

if __name__ == "__main__":
    results = main()