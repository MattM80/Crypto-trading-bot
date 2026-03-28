#!/usr/bin/env python3
"""
ROUND 3 VALIDATION - OPTIMIZED VERSION
Fast validation of new tools with sample of data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h_extended"
FEES = 0.0065

# Reduced pairs for speed
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT", "ADAUSDT"]

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

# =================== ROUND 3 TOOLS ===================

def trend_structure_long_v2(close, high, low, volume, rsi):
    """Improved trend structure with stricter requirements"""
    if len(close) < 200:
        return None
    
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    sma200 = calc_sma(close, 200)
    
    if not (close[-1] > ema20 > ema50 > sma200):
        return None
    
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    if ret_7d < 0.03:
        return None
    
    if rsi < 35 or rsi > 65:
        return None
    
    ema_distance = (close[-1] - ema20) / ema20
    if abs(ema_distance) > 0.02:
        return None
    
    return {
        'tool': 'trend_structure_long_v2',
        'direction': 'long',
        'score': ret_7d * 100 + (65 - rsi)
    }

def weekly_momentum_pullback_v2(close, high, low, volume, rsi):
    """Enhanced weekly momentum pullback"""
    if len(close) < 800:
        return None
    
    ret_2w = (close[-1] - close[-336]) / close[-336]
    ret_4w = (close[-1] - close[-672]) / close[-672]
    
    if ret_2w < 0.12 or ret_4w < 0.15:
        return None
    
    ret_48h = (close[-1] - close[-48]) / close[-48]
    if ret_48h > -0.02 or ret_48h < -0.20:
        return None
    
    sma200 = calc_sma(close, 200)
    if close[-1] < sma200 * 1.02:
        return None
    
    if rsi < 25 or rsi > 50:
        return None
    
    return {
        'tool': 'weekly_momentum_pullback_v2',
        'direction': 'long',
        'score': ret_4w * 100 + abs(ret_48h) * 50 + (50 - rsi)
    }

def hurst_accumulation_combo(close, high, low, volume, rsi):
    """Combo: Hurst + accumulation"""
    if len(close) < 500:
        return None
    
    # Simple Hurst calculation
    if len(close) >= 168:
        returns = np.diff(np.log(close[-100:]))  # Smaller window for speed
        if len(returns) < 20:
            return None
        
        # Simplified Hurst
        cumsum_returns = np.cumsum(returns - np.mean(returns))
        R = np.max(cumsum_returns) - np.min(cumsum_returns)
        S = np.std(returns)
        
        if S == 0 or R == 0:
            return None
        
        # Rough Hurst estimate
        H = np.log(R/S) / np.log(len(returns)/2)
        
        if H < 0.55:
            return None
    else:
        return None
    
    # Accumulation part
    if len(close) >= 400:
        range_high = np.max(high[-336:-48])
        range_low = np.min(low[-336:-48])
        range_pct = (range_high - range_low) / range_low * 100
        
        if range_pct > 25 or range_pct < 3:
            return None
        
        if close[-1] <= range_high * 1.008:
            return None
        
        vol_recent = np.mean(volume[-24:])
        vol_range = np.mean(volume[-200:])
        vol_ratio = vol_recent / vol_range if vol_range > 0 else 1
        
        if vol_ratio < 1.3:
            return None
    
    sma50 = calc_sma(close, 50)
    if close[-1] <= sma50 or rsi > 75:
        return None
    
    return {
        'tool': 'hurst_accumulation_combo',
        'direction': 'long',
        'score': H * 100 + vol_ratio * 10
    }

def range_oscillator_chop(close, high, low, volume, rsi):
    """Range trading for chop markets"""
    if len(close) < 200:
        return None
    
    # Identify range
    lookback = min(1200, len(close) - 1)
    range_high = np.max(high[-lookback:])
    range_low = np.min(low[-lookback:])
    range_pct = (range_high - range_low) / range_low * 100
    
    if range_pct < 5 or range_pct > 30:
        return None
    
    range_position = (close[-1] - range_low) / (range_high - range_low)
    
    if range_position < 0.25 and rsi < 35:
        recent_low = np.min(low[-48:])
        if recent_low <= range_low * 1.02:
            return {
                'tool': 'range_oscillator_chop',
                'direction': 'long',
                'score': (0.25 - range_position) * 100 + (35 - rsi)
            }
    
    elif range_position > 0.75 and rsi > 65:
        recent_high = np.max(high[-48:])
        if recent_high >= range_high * 0.98:
            return {
                'tool': 'range_oscillator_chop',
                'direction': 'short',
                'score': (range_position - 0.75) * 100 + (rsi - 65)
            }
    
    return None

def extreme_greed_short(close, high, low, volume, rsi):
    """Short extreme greed conditions"""
    if len(close) < 200:
        return None
    
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    ret_3d = (close[-1] - close[-72]) / close[-72] if len(close) >= 72 else 0
    ret_24h = (close[-1] - close[-24]) / close[-24]
    
    if ret_7d < 0.15 or ret_3d < 0.08 or ret_24h < 0.03:
        return None
    
    if rsi < 78:
        return None
    
    vol_recent = np.mean(volume[-12:])
    vol_baseline = np.mean(volume[-168:])
    vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    if vol_ratio < 2.0:
        return None
    
    # BB position
    sma20 = calc_sma(close, 20)
    std20 = np.std(close[-20:])
    bb_pos = (close[-1] - sma20) / std20 if std20 > 0 else 0
    
    if bb_pos < 2.2:
        return None
    
    return {
        'tool': 'extreme_greed_short',
        'direction': 'short',
        'score': rsi + bb_pos * 10 + vol_ratio * 5 + ret_3d * 50
    }

def parabolic_exhaustion_short(close, high, low, volume, rsi):
    """Parabolic exhaustion shorts"""
    if len(close) < 100:
        return None
    
    if len(close) >= 72:
        ret_1d = (close[-1] - close[-24]) / close[-24]
        ret_2d = (close[-25] - close[-48]) / close[-48] if len(close) >= 48 else 0
        ret_3d = (close[-49] - close[-72]) / close[-72] if len(close) >= 72 else 0
        
        if not (ret_1d > ret_2d > ret_3d and ret_1d > 0.04):
            return None
    else:
        return None
    
    # ATR check
    if len(close) >= 15:
        tr_list = []
        for i in range(1, 15):
            tr = max(
                high[-i] - low[-i],
                abs(high[-i] - close[-i-1]),
                abs(low[-i] - close[-i-1])
            )
            tr_list.append(tr)
        atr = np.mean(tr_list)
        atr_pct = (atr / close[-1]) * 100
        
        if atr_pct < 4.0:
            return None
    
    if rsi < 72:
        return None
    
    return {
        'tool': 'parabolic_exhaustion_short',
        'direction': 'short',
        'score': ret_1d * 100 + atr_pct * 3 + (rsi - 50)
    }

# Tool list
TOOLS = [
    (trend_structure_long_v2, 'bull'),
    (weekly_momentum_pullback_v2, 'bull'),
    (hurst_accumulation_combo, 'bull'),
    (range_oscillator_chop, 'chop'),
    (extreme_greed_short, 'bull'),
    (parabolic_exhaustion_short, 'bull'),
]

def validate_tool(tool_func, regime_filter, max_hold=336):
    all_trades = []
    btc_df = load_data("BTCUSDT")
    btc_close = btc_df['close'].values
    
    for pair in PAIRS:
        df = load_data(pair)
        close = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        vol = df['volume'].values
        
        oos_start = len(close) // 2
        i = oos_start
        last_signal = -max_hold
        signals_found = 0
        
        while i < len(close) - max_hold and signals_found < 50:  # Limit for speed
            if i - last_signal < 48:
                i += 16  # Larger steps for speed
                continue
            
            # Regime filtering
            if regime_filter == 'bull' and not is_bull_regime(btc_close, i):
                i += 16
                continue
            elif regime_filter == 'chop' and not is_chop_regime(btc_close, i):
                i += 16
                continue
            
            rsi = calc_rsi(close[:i+1])
            sig = tool_func(close[:i+1], high_arr[:i+1], low_arr[:i+1], vol[:i+1], rsi)
            
            if sig is not None:
                signals_found += 1
                exit_price, hold_bars, exit_reason = trailing_exit(close, i, max_hold, sig['direction'])
                entry_price = close[i]
                
                if sig['direction'] == 'long':
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - FEES
                
                all_trades.append({
                    'pair': pair,
                    'net': float(net_return),
                    'win': net_return > 0,
                    'hold': int(hold_bars),
                    'exit_reason': exit_reason
                })
                last_signal = i
            
            i += 16
    
    return all_trades

def report_results(name, trades, regime):
    if not trades:
        print(f"  {name} ({regime}): NO SIGNALS ❌")
        return None
    
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    avg_ret = np.mean([t['net'] for t in trades]) * 100
    
    w_trades = [t['net'] for t in trades if t['win']]
    l_trades = [t['net'] for t in trades if not t['win']]
    
    gross_profit = sum(w_trades) if w_trades else 0
    gross_loss = abs(sum(l_trades)) if l_trades else 0.001
    pf = gross_profit / gross_loss
    
    passed = n >= 20 and (wr >= 55 or pf >= 1.5)  # Lower threshold for testing
    status = "✅ PASSED" if passed else "❌ KILLED"
    
    print(f"  {name} ({regime}): {status}")
    print(f"    Signals: {n} | WR: {wr:.1f}% | Avg: {avg_ret:.2f}% | PF: {pf:.2f}")
    
    return {
        'tool': name,
        'regime': regime,
        'signals': n,
        'wr': wr,
        'avg_ret': avg_ret,
        'pf': pf,
        'passed': passed
    }

def main():
    print("=" * 70)
    print("ROUND 3 VALIDATION - FAST VERSION")
    print("6 pairs | Large steps | Limited signals per pair")
    print("=" * 70)
    
    all_results = []
    
    for tool_func, regime in TOOLS:
        print(f"\nTesting: {tool_func.__name__} ({regime})")
        trades = validate_tool(tool_func, regime)
        result = report_results(tool_func.__name__, trades, regime)
        if result:
            all_results.append(result)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    passed = [r for r in all_results if r['passed']]
    killed = [r for r in all_results if not r['passed']]
    
    print(f"\n✅ PASSED ({len(passed)}):")
    for r in passed:
        print(f"  {r['tool']}: {r['signals']} sigs, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    
    print(f"\n❌ KILLED ({len(killed)}):")
    for r in killed:
        print(f"  {r['tool']}: {r['signals']} sigs, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    
    return all_results

if __name__ == "__main__":
    results = main()