#!/usr/bin/env python3
"""
ROUND 3 COMPREHENSIVE VALIDATION
Build on successful hurst_accumulation_combo and create more sophisticated tools
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h_extended"
FEES = 0.0065

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT", "ADAUSDT", "ATOMUSDT", "XRPUSDT"]

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

def is_bull_regime(btc_close, bar_idx):
    if bar_idx < 720:
        return False
    ret_30d = (btc_close[bar_idx] - btc_close[bar_idx-720]) / btc_close[bar_idx-720]
    sma50 = calc_sma(btc_close[:bar_idx+1], 50)
    return ret_30d > 0.08 and btc_close[bar_idx] > sma50

def is_chop_regime(btc_close, bar_idx):
    if bar_idx < 720:
        return False
    ret_30d = (btc_close[bar_idx] - btc_close[bar_idx-720]) / btc_close[bar_idx-720]
    return -0.08 <= ret_30d <= 0.08

def trailing_exit(close_arr, entry_bar, max_hold, direction, trail_pct=0.08, hard_stop_pct=0.12):
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
    else:
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

# =================== IMPROVED ROUND 3 TOOLS ===================

def hurst_accumulation_combo_v2(close, high, low, volume, rsi):
    """Enhanced version of the successful hurst_accumulation_combo"""
    if len(close) < 500:
        return None
    
    # Enhanced Hurst calculation
    if len(close) >= 168:
        returns = np.diff(np.log(close[-120:]))  # 5 day window
        if len(returns) < 30:
            return None
        
        # R/S analysis for Hurst
        mean_ret = np.mean(returns)
        centered_rets = returns - mean_ret
        cumsum_rets = np.cumsum(centered_rets)
        
        R = np.max(cumsum_rets) - np.min(cumsum_rets)
        S = np.std(returns)
        
        if S == 0 or R == 0:
            return None
        
        # Calculate Hurst with rescaling
        lags = [10, 20, 30, 40]
        rs_values = []
        
        for lag in lags:
            if len(returns) >= lag * 2:
                n_windows = len(returns) // lag
                window_rs = []
                
                for i in range(n_windows):
                    window = returns[i*lag:(i+1)*lag]
                    if len(window) == lag:
                        w_mean = np.mean(window)
                        w_cumsum = np.cumsum(window - w_mean)
                        w_R = np.max(w_cumsum) - np.min(w_cumsum)
                        w_S = np.std(window)
                        if w_S > 0:
                            window_rs.append(w_R / w_S)
                
                if window_rs:
                    rs_values.append((lag, np.mean(window_rs)))
        
        if len(rs_values) < 3:
            return None
        
        # Linear regression for Hurst exponent
        log_lags = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values if x[1] > 0])
        
        if len(log_rs) != len(log_lags) or len(log_rs) < 3:
            return None
        
        H = np.polyfit(log_lags, log_rs, 1)[0]
        
        if H < 0.58:  # Trending threshold
            return None
    else:
        return None
    
    # Enhanced accumulation detection
    if len(close) >= 400:
        # Multiple timeframe range analysis
        range_periods = [240, 336, 480]  # 10d, 14d, 20d
        breakout_confirmed = False
        vol_score = 0
        
        for period in range_periods:
            if len(close) >= period + 48:
                range_high = np.max(high[-period:-48])
                range_low = np.min(low[-period:-48])
                range_pct = (range_high - range_low) / range_low * 100
                
                # Optimal range: 3-22%
                if 3 <= range_pct <= 22:
                    # Breakout detection
                    if close[-1] > range_high * 1.006:  # 0.6% breakout
                        breakout_confirmed = True
                        
                        # Volume analysis for this timeframe
                        vol_during = np.mean(volume[-period:-48])
                        vol_recent = np.mean(volume[-24:])
                        vol_breakout = np.mean(volume[-6:])  # Last 6 hours
                        
                        if vol_breakout > vol_during * 1.4 and vol_recent > vol_during * 1.2:
                            vol_score += (vol_breakout / vol_during) * 10
        
        if not breakout_confirmed or vol_score < 15:
            return None
    else:
        return None
    
    # Enhanced trend structure validation
    ema21 = calc_ema(close, 21)
    ema50 = calc_ema(close, 50)
    sma200 = calc_sma(close, 200)
    
    # Must have proper trend alignment
    if not (close[-1] > ema21 > ema50 and ema50 > sma200 * 0.98):
        return None
    
    # Momentum confirmation
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    ret_3d = (close[-1] - close[-72]) / close[-72] if len(close) >= 72 else 0
    
    if ret_7d < 0.01 or ret_3d < -0.02:  # Recent momentum required
        return None
    
    # RSI not extreme (better entry timing)
    if rsi > 75 or rsi < 25:
        return None
    
    # Price position relative to EMAs (entry timing)
    ema21_distance = (close[-1] - ema21) / ema21
    if ema21_distance > 0.03 or ema21_distance < -0.015:  # Not too far from EMA21
        return None
    
    score = H * 150 + vol_score + ret_7d * 100 + ret_3d * 50 + (75 - rsi)
    
    return {
        'tool': 'hurst_accumulation_combo_v2',
        'direction': 'long',
        'score': score
    }

def volume_profile_breakout(close, high, low, volume, rsi):
    """Smart money accumulation using volume profile analysis"""
    if len(close) < 300:
        return None
    
    # Volume-weighted average price over multiple periods
    periods = [96, 168, 240]  # 4d, 7d, 10d
    vwap_signals = 0
    
    for period in periods:
        if len(close) >= period:
            # Calculate VWAP
            typical_price = (high[-period:] + low[-period:] + close[-period:]) / 3
            vwap = np.sum(typical_price * volume[-period:]) / np.sum(volume[-period:])
            
            # Price above VWAP = bullish
            if close[-1] > vwap * 1.002:  # Above VWAP with small buffer
                vwap_signals += 1
    
    if vwap_signals < 2:  # Need multiple timeframe confirmation
        return None
    
    # Volume distribution analysis (look for accumulation)
    if len(volume) >= 168:
        # Compare recent volume distribution vs baseline
        recent_vol = volume[-48:]  # 2 days
        baseline_vol = volume[-168:-48]  # Previous 5 days
        
        # Volume percentiles
        recent_90th = np.percentile(recent_vol, 90)
        baseline_90th = np.percentile(baseline_vol, 90)
        recent_median = np.median(recent_vol)
        baseline_median = np.median(baseline_vol)
        
        # Smart money pattern: elevated median volume, controlled spikes
        if (recent_median > baseline_median * 1.15 and 
            recent_90th < baseline_90th * 1.8):  # No panic volume spikes
            
            # Price action: controlled moves with volume
            price_volatility = np.std(close[-48:]) / np.mean(close[-48:])
            if price_volatility > 0.05:  # Too volatile
                return None
        else:
            return None
    
    # Trend confirmation
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    
    if close[-1] <= ema12 or ema12 <= ema26:
        return None
    
    # RSI in productive range
    if rsi > 70 or rsi < 35:
        return None
    
    # Recent momentum but not parabolic
    ret_24h = (close[-1] - close[-24]) / close[-24]
    ret_72h = (close[-1] - close[-72]) / close[-72] if len(close) >= 72 else 0
    
    if ret_24h > 0.06 or ret_72h > 0.18:  # Not too fast
        return None
    
    if ret_72h < 0:  # Need positive momentum
        return None
    
    volume_score = (recent_median / baseline_median) * 20
    momentum_score = ret_72h * 100
    vwap_score = vwap_signals * 15
    
    return {
        'tool': 'volume_profile_breakout',
        'direction': 'long',
        'score': volume_score + momentum_score + vwap_score + (70 - rsi)
    }

def mean_reversion_chop_v2(close, high, low, volume, rsi):
    """Enhanced mean reversion for choppy markets"""
    if len(close) < 200:
        return None
    
    # Volatility regime detection (must be low vol for mean reversion)
    if len(close) >= 168:
        returns_7d = np.diff(np.log(close[-168:]))
        realized_vol = np.std(returns_7d) * np.sqrt(24 * 365)  # Annualized
        
        if realized_vol > 0.8:  # Too volatile for mean reversion
            return None
    
    # Multiple timeframe mean reversion
    periods = [48, 96, 168]  # 2d, 4d, 7d
    reversion_signals = 0
    total_z_score = 0
    
    for period in periods:
        if len(close) >= period:
            mean_price = np.mean(close[-period:])
            std_price = np.std(close[-period:])
            
            if std_price > 0:
                z_score = (close[-1] - mean_price) / std_price
                total_z_score += abs(z_score)
                
                # Mean reversion thresholds
                if z_score < -1.5:  # Oversold
                    reversion_signals += 1
                elif z_score > 1.5:  # Overbought  
                    reversion_signals -= 1
    
    if abs(reversion_signals) < 2:  # Need strong signal
        return None
    
    # Volume analysis: look for capitulation or exhaustion
    vol_recent = np.mean(volume[-12:])
    vol_baseline = np.mean(volume[-96:])
    vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    # Direction determination
    direction = 'long' if reversion_signals >= 2 else 'short'
    
    if direction == 'long':
        # Oversold conditions
        if rsi > 35 or vol_ratio < 1.1:
            return None
        
        # Look for selling exhaustion
        if len(close) >= 24:
            recent_low = np.min(low[-24:])
            if (close[-1] - recent_low) / recent_low < 0.005:  # Near recent low
                pass
            else:
                return None
    else:
        # Overbought conditions
        if rsi < 65 or vol_ratio < 1.1:
            return None
        
        # Look for buying exhaustion
        if len(close) >= 24:
            recent_high = np.max(high[-24:])
            if (recent_high - close[-1]) / close[-1] < 0.005:  # Near recent high
                pass
            else:
                return None
    
    return {
        'tool': 'mean_reversion_chop_v2',
        'direction': direction,
        'score': total_z_score * 20 + vol_ratio * 10 + abs(50 - rsi)
    }

def trend_continuation_bull(close, high, low, volume, rsi):
    """Trend continuation during bull markets - ride the momentum"""
    if len(close) < 200:
        return None
    
    # Multi-timeframe trend validation
    ema8 = calc_ema(close, 8)
    ema21 = calc_ema(close, 21) 
    ema55 = calc_ema(close, 55)
    sma200 = calc_sma(close, 200)
    
    # Perfect trend alignment required
    if not (close[-1] > ema8 > ema21 > ema55 > sma200):
        return None
    
    # Momentum persistence check
    momentum_periods = [24, 48, 72, 96]  # 1d, 2d, 3d, 4d
    positive_momentum = 0
    
    for period in momentum_periods:
        if len(close) >= period:
            ret = (close[-1] - close[-period]) / close[-period]
            if ret > 0.01:  # At least 1% gain
                positive_momentum += 1
    
    if positive_momentum < 3:  # Need consistent momentum
        return None
    
    # EMA distance analysis (entry timing)
    ema8_dist = (close[-1] - ema8) / ema8
    ema21_dist = (close[-1] - ema21) / ema21
    
    # Look for pullback to shorter EMAs (better entry)
    if ema8_dist > 0.02:  # Too far from EMA8
        return None
    
    # RSI momentum but not extreme
    if rsi < 45 or rsi > 75:
        return None
    
    # Volume confirmation of trend
    vol_recent = np.mean(volume[-24:])
    vol_baseline = np.mean(volume[-168:])
    vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    if vol_ratio < 0.8:  # Need some volume participation
        return None
    
    # Trend strength calculation
    trend_angles = []
    for period in [21, 55]:
        if len(close) >= period + 24:
            old_ema = calc_ema(close[:-24], period)
            new_ema = calc_ema(close, period) 
            angle = (new_ema - old_ema) / old_ema
            trend_angles.append(angle)
    
    avg_trend_strength = np.mean(trend_angles) if trend_angles else 0
    
    if avg_trend_strength < 0.001:  # Trend must be strengthening
        return None
    
    # Weekly momentum for context
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    
    if ret_7d < 0.02:  # Need meaningful weekly gains
        return None
    
    momentum_score = positive_momentum * 15
    trend_score = avg_trend_strength * 1000
    volume_score = vol_ratio * 10
    timing_score = (75 - rsi) if rsi < 70 else 0
    
    return {
        'tool': 'trend_continuation_bull',
        'direction': 'long',
        'score': momentum_score + trend_score + volume_score + timing_score + ret_7d * 50
    }

def breakout_retest_long(close, high, low, volume, rsi):
    """Enter on retest of breakout levels (classic technical pattern)"""
    if len(close) < 400:
        return None
    
    # Identify resistance levels using pivots
    resistance_levels = []
    lookback = min(720, len(high) - 50)  # Up to 30 days
    
    # Find pivot highs (local maxima)
    for i in range(50, lookback - 10):
        if (high[-i] > np.max(high[-i-10:-i+1]) and 
            high[-i] > np.max(high[-i:-i+11])):
            resistance_levels.append(high[-i])
    
    if len(resistance_levels) < 2:
        return None
    
    # Find significant resistance (multiple tests)
    significant_resistance = []
    for level in resistance_levels:
        test_count = 0
        for j in range(len(high) - 168, len(high)):  # Last 7 days
            if abs(high[j] - level) / level < 0.005:  # Within 0.5%
                test_count += 1
        
        if test_count >= 2:  # Multiple tests of this level
            significant_resistance.append(level)
    
    if not significant_resistance:
        return None
    
    # Check for recent breakout above resistance
    current_price = close[-1]
    broken_resistance = None
    
    for level in significant_resistance:
        # Must have broken above this level recently (last 72h)
        if current_price > level * 1.003:  # 0.3% above resistance
            # Check if breakout was recent
            breakout_found = False
            for k in range(1, min(72, len(close))):
                if close[-k] <= level and close[-k+1:].max() > level * 1.003:
                    breakout_found = True
                    broken_resistance = level
                    break
            
            if breakout_found:
                break
    
    if broken_resistance is None:
        return None
    
    # Current price should be retesting the broken resistance (now support)
    support_level = broken_resistance
    distance_from_support = (current_price - support_level) / support_level
    
    # Must be near the support level (retest pattern)
    if distance_from_support < -0.01 or distance_from_support > 0.03:  # -1% to +3%
        return None
    
    # Volume confirmation during breakout
    breakout_volume = None
    for k in range(1, min(72, len(volume))):
        if close[-k] <= support_level and close[-k+1:].max() > support_level * 1.003:
            breakout_volume = np.mean(volume[-k-3:-k+3])  # Volume around breakout
            break
    
    if breakout_volume:
        current_volume = np.mean(volume[-6:])
        baseline_volume = np.mean(volume[-168:])
        
        # Breakout should have had volume
        if breakout_volume < baseline_volume * 1.2:
            return None
    
    # RSI should be healthy for continuation
    if rsi < 35 or rsi > 70:
        return None
    
    # Trend context (must be in uptrend)
    ema50 = calc_ema(close, 50)
    if current_price < ema50:
        return None
    
    # Recent momentum
    ret_48h = (close[-1] - close[-48]) / close[-48] if len(close) >= 48 else 0
    if ret_48h < -0.05:  # Not falling too hard
        return None
    
    breakout_score = (current_price / support_level - 1) * 500
    volume_score = (breakout_volume / baseline_volume) * 15 if breakout_volume else 10
    position_score = max(0, 30 - distance_from_support * 1000)  # Closer to support = better
    
    return {
        'tool': 'breakout_retest_long',
        'direction': 'long', 
        'score': breakout_score + volume_score + position_score + (70 - rsi)
    }

# Tool definitions
COMPREHENSIVE_TOOLS = [
    (hurst_accumulation_combo_v2, 'bull'),
    (volume_profile_breakout, 'bull'),
    (mean_reversion_chop_v2, 'chop'),
    (trend_continuation_bull, 'bull'),
    (breakout_retest_long, 'bull'),
]

def validate_tool(tool_func, regime_filter, max_hold=336):
    all_trades = []
    btc_df = load_data("BTCUSDT")
    btc_close = btc_df['close'].values
    
    print(f"    Validating {tool_func.__name__}...")
    
    for pair in PAIRS:
        df = load_data(pair)
        close = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        vol = df['volume'].values
        
        oos_start = len(close) // 2
        i = oos_start
        last_signal = -max_hold
        signals_this_pair = 0
        
        while i < len(close) - max_hold and signals_this_pair < 25:
            if i - last_signal < 48:
                i += 12  # Medium step size
                continue
            
            # Regime filtering
            if regime_filter == 'bull' and not is_bull_regime(btc_close, i):
                i += 12
                continue
            elif regime_filter == 'chop' and not is_chop_regime(btc_close, i):
                i += 12
                continue
            
            rsi = calc_rsi(close[:i+1])
            sig = tool_func(close[:i+1], high_arr[:i+1], low_arr[:i+1], vol[:i+1], rsi)
            
            if sig is not None:
                signals_this_pair += 1
                exit_price, hold_bars, exit_reason = trailing_exit(close, i, max_hold, sig['direction'])
                entry_price = close[i]
                
                if sig['direction'] == 'long':
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - FEES
                
                all_trades.append({
                    'pair': pair,
                    'direction': sig['direction'],
                    'net': float(net_return),
                    'win': net_return > 0,
                    'hold': int(hold_bars),
                    'exit_reason': exit_reason,
                    'score': sig.get('score', 0)
                })
                last_signal = i
            
            i += 12
        
        if signals_this_pair > 0:
            print(f"      {pair}: {signals_this_pair} signals")
    
    return all_trades

def detailed_report(name, trades, regime):
    if not trades:
        print(f"\n  ❌ {name} ({regime}): NO SIGNALS")
        return None
    
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    avg_ret = np.mean([t['net'] for t in trades]) * 100
    
    w_trades = [t['net'] for t in trades if t['win']]
    l_trades = [t['net'] for t in trades if not t['win']]
    avg_win = np.mean(w_trades) * 100 if w_trades else 0
    avg_loss = np.mean(l_trades) * 100 if l_trades else 0
    
    gross_profit = sum(w_trades) if w_trades else 0
    gross_loss = abs(sum(l_trades)) if l_trades else 0.001
    pf = gross_profit / gross_loss
    
    avg_hold = np.mean([t['hold'] for t in trades])
    total_return = sum(t['net'] for t in trades) * 100
    max_dd = min(t['net'] for t in trades) * 100
    
    # Exit analysis
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1
    
    # Pair analysis
    pair_stats = {}
    for t in trades:
        pair_stats.setdefault(t['pair'], []).append(t['net'])
    
    # Pass criteria
    passed = n >= 30 and (wr >= 55 or pf >= 1.5)
    status = "✅ PASSED" if passed else "❌ KILLED"
    
    print(f"\n  {status} {name} ({regime})")
    print(f"    📊 Overview: {n} signals | {wr:.1f}% WR | {avg_ret:.2f}% avg | PF={pf:.2f}")
    print(f"    💰 Returns: Total={total_return:.1f}% | Win={avg_win:.2f}% | Loss={avg_loss:.2f}% | Max DD={max_dd:.2f}%")
    print(f"    ⏱️  Timing: Avg hold={avg_hold:.0f}h ({avg_hold/24:.1f}d) | Exits: {exits}")
    
    if pair_stats:
        print(f"    🎯 Top pairs:")
        sorted_pairs = sorted(pair_stats.items(), key=lambda x: np.mean(x[1]), reverse=True)
        for pair, rets in sorted_pairs[:3]:
            pair_wins = sum(1 for r in rets if r > 0)
            print(f"      {pair}: {len(rets)} trades, {pair_wins}/{len(rets)} wins ({pair_wins/len(rets)*100:.0f}%), avg {np.mean(rets)*100:.2f}%")
    
    return {
        'tool': name,
        'regime': regime,
        'signals': n,
        'wr': wr,
        'avg_ret': avg_ret,
        'pf': pf,
        'passed': passed,
        'total_return': total_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_hold': avg_hold
    }

def main():
    print("=" * 80)
    print("ROUND 3 COMPREHENSIVE VALIDATION")
    print("Advanced crypto trading tools with sophisticated entry logic")
    print("=" * 80)
    print("Dataset: 8 pairs | OOS: 2nd half | Fees: 0.65% | Trail: 8% | Hard stop: 12%")
    print("Pass criteria: ≥30 signals AND (WR≥55% OR PF≥1.5)")
    print("=" * 80)
    
    all_results = []
    
    for i, (tool_func, regime) in enumerate(COMPREHENSIVE_TOOLS, 1):
        print(f"\n[{i}/{len(COMPREHENSIVE_TOOLS)}] Testing {tool_func.__name__} in {regime} regime")
        print("-" * 60)
        
        trades = validate_tool(tool_func, regime)
        result = detailed_report(tool_func.__name__, trades, regime)
        
        if result:
            all_results.append(result)
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏆 FINAL ROUND 3 RESULTS")
    print(f"{'='*80}")
    
    passed = [r for r in all_results if r['passed']]
    killed = [r for r in all_results if not r['passed']]
    
    if passed:
        print(f"\n✅ PASSED TOOLS ({len(passed)}):")
        for r in sorted(passed, key=lambda x: x['avg_ret'], reverse=True):
            print(f"  🌟 {r['tool']} ({r['regime']}): {r['signals']} sigs, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    else:
        print(f"\n❌ NO TOOLS PASSED VALIDATION")
    
    if killed:
        print(f"\n❌ KILLED TOOLS ({len(killed)}):")
        for r in sorted(killed, key=lambda x: x['avg_ret'], reverse=True):
            reason = "Low signals" if r['signals'] < 30 else "Poor performance"
            print(f"  💀 {r['tool']} ({r['regime']}): {r['signals']} sigs, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f} - {reason}")
    
    print(f"\n{'='*80}")
    return all_results

if __name__ == "__main__":
    results = main()