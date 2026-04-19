#!/usr/bin/env python3
"""
ROUND 3 FINAL VALIDATION
Focus on what works: simplify and optimize the successful patterns
Based on learnings: keep it simple, focus on robust signals
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h_extended"
FEES = 0.0065

# Use all 16 pairs like the original validation
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

# =================== OPTIMIZED ROUND 3 TOOLS ===================

def accumulation_breakout_v3(close, high, low, volume, rsi):
    """
    Enhanced accumulation breakout - based on successful original
    Optimized parameters from round 2 feedback
    """
    if len(close) < 500:
        return None
    
    # Range analysis - optimized parameters
    range_high = np.max(high[-400:-24])  # Look back 400 bars, skip recent 24
    range_low = np.min(low[-400:-24])
    range_pct = (range_high - range_low) / range_low * 100
    
    # Optimal range: 4-18% (tighter than original 3-20%)
    if range_pct > 18 or range_pct < 4:
        return None
    
    # Breakout detection - more conservative
    if close[-1] <= range_high * 1.012:  # Need 1.2% breakout (vs 1% original)
        return None
    
    # Volume confirmation - enhanced
    vol_recent = np.mean(volume[-12:])  # 12h recent volume
    vol_range = np.mean(volume[-400:-24])  # Volume during range
    vol_baseline = np.mean(volume[-168:])  # 7d baseline
    
    vol_ratio = vol_recent / vol_range if vol_range > 0 else 1
    vol_baseline_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    if vol_ratio < 1.6 or vol_baseline_ratio < 1.3:  # Higher volume thresholds
        return None
    
    # RSI not extreme (refined range)
    if rsi > 70 or rsi < 30:
        return None
    
    # Trend context - must be in uptrend
    sma50 = calc_sma(close, 50)
    sma200 = calc_sma(close, 200)
    if close[-1] < sma50 or sma50 < sma200:
        return None
    
    # Recent momentum
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    if ret_7d < 0.02:
        return None
    
    return {
        'tool': 'accumulation_breakout_v3',
        'direction': 'long',
        'score': vol_ratio * 15 + (close[-1] - range_high) / range_high * 1000 + ret_7d * 50
    }

def simple_hurst_trend(close, high, low, volume, rsi):
    """
    Simplified Hurst trend following - keep what works
    Based on the successful hurst_trend_long from regime validation
    """
    if len(close) < 500:
        return None
    
    # Simple Hurst calculation (fast version)
    if len(close) >= 168:
        returns = np.diff(np.log(close[-120:]))  # 5 day window
        
        if len(returns) < 30:
            return None
        
        # Simplified R/S calculation
        mean_ret = np.mean(returns)
        cumsum_rets = np.cumsum(returns - mean_ret)
        R = np.max(cumsum_rets) - np.min(cumsum_rets)
        S = np.std(returns)
        
        if S == 0 or R == 0:
            return None
        
        # Simple Hurst estimate
        H = 0.5 + np.log(R/S) / np.log(len(returns))
        
        if H < 0.58:  # Trending threshold
            return None
    else:
        return None
    
    # Trend confirmation
    sma50 = calc_sma(close, 50)
    ema21 = calc_ema(close, 21)
    
    if close[-1] <= sma50 or close[-1] <= ema21:
        return None
    
    # Recent momentum
    ret_24h = (close[-1] - close[-24]) / close[-24]
    ret_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    
    if ret_24h <= 0.005 or ret_7d <= 0.01:
        return None
    
    # Volume confirmation
    vol_recent = np.mean(volume[-24:])
    vol_baseline = np.mean(volume[-168:])
    vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    if vol_ratio < 1.1:
        return None
    
    # RSI healthy
    if rsi > 72 or rsi < 35:
        return None
    
    return {
        'tool': 'simple_hurst_trend',
        'direction': 'long',
        'score': (H - 0.5) * 200 + ret_7d * 100 + vol_ratio * 10
    }

def momentum_pullback_optimized(close, high, low, volume, rsi):
    """
    Optimized momentum pullback based on round 2 near-miss
    Simplified and tightened parameters
    """
    if len(close) < 800:
        return None
    
    # Strong momentum requirements (higher than v2)
    ret_2w = (close[-1] - close[-336]) / close[-336]
    ret_4w = (close[-1] - close[-672]) / close[-672]
    
    if ret_2w < 0.10 or ret_4w < 0.18:  # Even stronger momentum needed
        return None
    
    # Pullback detection
    ret_48h = (close[-1] - close[-48]) / close[-48]
    ret_24h = (close[-1] - close[-24]) / close[-24]
    
    # Pullback range: 1-15% (tighter than v2's 2-20%)
    if ret_48h > -0.01 or ret_48h < -0.15:
        return None
    
    if ret_24h > 0:  # Must still be pulling back
        return None
    
    # Trend structure - stronger requirements
    sma50 = calc_sma(close, 50)
    sma200 = calc_sma(close, 200)
    ema21 = calc_ema(close, 21)
    
    if not (close[-1] > ema21 > sma50 > sma200):
        return None
    
    # RSI in sweet spot
    if rsi < 30 or rsi > 45:  # Narrower range than v2
        return None
    
    # Volume: not panicky
    vol_recent = np.mean(volume[-24:])
    vol_baseline = np.mean(volume[-168:])
    vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    if vol_ratio > 1.6:  # No panic volume
        return None
    
    # Long-term momentum still strong
    ret_7d = (close[-1] - close[-168]) / close[-168]
    if ret_7d < 0.03:
        return None
    
    return {
        'tool': 'momentum_pullback_optimized',
        'direction': 'long',
        'score': ret_4w * 120 + abs(ret_48h) * 60 + (45 - rsi) + ret_7d * 40
    }

def low_volatility_breakout(close, high, low, volume, rsi):
    """
    Low volatility breakout for choppy markets
    Enter breakouts from tight consolidations
    """
    if len(close) < 300:
        return None
    
    # Volatility analysis
    if len(close) >= 168:
        returns = np.diff(np.log(close[-168:]))
        volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized
        
        if volatility > 0.6:  # Must be low volatility period
            return None
    
    # Consolidation analysis
    consolidation_period = min(240, len(close) - 24)  # 10 days max
    cons_high = np.max(high[-consolidation_period:-12])
    cons_low = np.min(low[-consolidation_period:-12])
    cons_range = (cons_high - cons_low) / cons_low * 100
    
    # Tight consolidation: 2-12%
    if cons_range < 2 or cons_range > 12:
        return None
    
    # Volume spike during breakout
    vol_spike = np.mean(volume[-6:])  # Last 6 hours
    vol_consolidation = np.mean(volume[-consolidation_period:-12])
    vol_ratio = vol_spike / vol_consolidation if vol_consolidation > 0 else 1
    
    if vol_ratio < 1.8:
        return None
    
    # Breakout direction
    breakout_threshold = 0.008  # 0.8%
    
    if close[-1] > cons_high * (1 + breakout_threshold):
        direction = 'long'
        if rsi > 75:  # Not too overbought
            return None
    elif close[-1] < cons_low * (1 - breakout_threshold):
        direction = 'short'
        if rsi < 25:  # Not too oversold
            return None
    else:
        return None
    
    # Context check
    sma100 = calc_sma(close, 100)
    if direction == 'long' and close[-1] < sma100 * 0.98:
        return None
    elif direction == 'short' and close[-1] > sma100 * 1.02:
        return None
    
    return {
        'tool': 'low_volatility_breakout',
        'direction': direction,
        'score': vol_ratio * 15 + (12 - cons_range) * 3 + abs(50 - rsi) * 0.5
    }

def smart_dip_buy(close, high, low, volume, rsi):
    """
    Enhanced dip buying with better trend and dip characterization
    Optimized based on round 2 failure analysis
    """
    if len(close) < 800:
        return None
    
    # Very strong uptrend required (higher than v2)
    ret_4w = (close[-1] - close[-672]) / close[-672]
    ret_8w = (close[-1] - close[-1344]) / close[-1344] if len(close) >= 1344 else ret_4w
    
    if ret_4w < 0.12 or ret_8w < 0.20:  # Much higher thresholds
        return None
    
    # Dip characterization
    recent_high = np.max(high[-120:])  # 5 day high
    dip_size = (close[-1] - recent_high) / recent_high
    
    # Optimal dip: 2-12% (smaller range than v2)
    if dip_size > -0.02 or dip_size < -0.12:
        return None
    
    # Dip must be recent (within 48h)
    high_age = 0
    for i in range(1, min(48, len(high))):
        if high[-i] >= recent_high * 0.999:
            high_age = i
            break
    
    if high_age == 0 or high_age > 48:
        return None
    
    # Strong trend structure
    ema21 = calc_ema(close, 21)
    sma50 = calc_sma(close, 50)
    sma200 = calc_sma(close, 200)
    
    if not (close[-1] > ema21 and ema21 > sma50 > sma200):
        return None
    
    # RSI showing oversold bounce
    if rsi > 40 or rsi < 20:
        return None
    
    # Volume: some interest but not panic
    vol_recent = np.mean(volume[-12:])
    vol_baseline = np.mean(volume[-120:])
    vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    if vol_ratio < 0.9 or vol_ratio > 2.0:
        return None
    
    # 21 EMA support
    ema21_distance = (close[-1] - ema21) / ema21
    if ema21_distance < -0.05:  # Too far below EMA21
        return None
    
    return {
        'tool': 'smart_dip_buy',
        'direction': 'long',
        'score': abs(dip_size) * 150 + ret_4w * 80 + (40 - rsi) + vol_ratio * 8
    }

def volume_breakout_simple(close, high, low, volume, rsi):
    """
    Simple volume breakout - when volume explodes, follow the direction
    Keep it simple and robust
    """
    if len(close) < 200:
        return None
    
    # Volume explosion detection
    vol_recent = np.mean(volume[-4:])  # 4h average
    vol_baseline = np.mean(volume[-96:])  # 4d baseline
    vol_spike = vol_recent / vol_baseline if vol_baseline > 0 else 1
    
    if vol_spike < 2.5:  # Need significant volume spike
        return None
    
    # Price movement with volume
    ret_4h = (close[-1] - close[-4]) / close[-4]
    ret_12h = (close[-1] - close[-12]) / close[-12]
    
    # Direction determination
    if ret_4h > 0.015 and ret_12h > 0.01:  # Upside breakout
        direction = 'long'
        if rsi > 78:  # Not extremely overbought
            return None
    elif ret_4h < -0.015 and ret_12h < -0.01:  # Downside breakout
        direction = 'short'
        if rsi < 22:  # Not extremely oversold
            return None
    else:
        return None
    
    # Trend context
    sma50 = calc_sma(close, 50)
    sma200 = calc_sma(close, 200)
    
    if direction == 'long':
        if close[-1] < sma50 * 0.95:  # Need reasonable trend context
            return None
    else:
        if close[-1] > sma50 * 1.05:
            return None
    
    # Recent consolidation before breakout
    if len(close) >= 72:
        cons_range = (np.max(high[-72:-4]) - np.min(low[-72:-4])) / np.mean(close[-72:-4]) * 100
        if cons_range < 3:  # Some consolidation needed
            return None
    
    return {
        'tool': 'volume_breakout_simple',
        'direction': direction,
        'score': vol_spike * 10 + abs(ret_4h) * 200 + abs(ret_12h) * 100
    }

# Final tool list with regime assignments
FINAL_TOOLS = [
    (accumulation_breakout_v3, 'bull'),
    (simple_hurst_trend, 'bull'),
    (momentum_pullback_optimized, 'bull'),
    (low_volatility_breakout, 'chop'),
    (smart_dip_buy, 'bull'),
    (volume_breakout_simple, 'bull'),
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
        
        # OOS: second half
        oos_start = len(close) // 2
        i = oos_start
        last_signal = -max_hold
        
        while i < len(close) - max_hold:
            if i - last_signal < 48:
                i += 8  # Back to original step size
                continue
            
            # Regime filtering
            if regime_filter == 'bull' and not is_bull_regime(btc_close, i):
                i += 8
                continue
            elif regime_filter == 'chop' and not is_chop_regime(btc_close, i):
                i += 8
                continue
            
            rsi = calc_rsi(close[:i+1])
            sig = tool_func(close[:i+1], high_arr[:i+1], low_arr[:i+1], vol[:i+1], rsi)
            
            if sig is not None:
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
                    'exit_reason': exit_reason
                })
                last_signal = i
            
            i += 8
    
    return all_trades

def final_report(name, trades, regime):
    if not trades:
        print(f"  ❌ {name} ({regime}): NO SIGNALS")
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
    
    # Exit breakdown
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1
    
    # Pass criteria (exact same as original)
    passed = n >= 30 and (wr >= 55 or pf >= 1.5)
    status = "✅ PASSED" if passed else "❌ KILLED"
    
    print(f"\n  {status} {name} ({regime})")
    print(f"    Signals: {n} | WR: {wr:.1f}% | Avg: {avg_ret:.2f}% | Total: {total_return:.1f}%")
    print(f"    PF: {pf:.2f} | Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%")
    print(f"    Hold: {avg_hold:.0f}h ({avg_hold/24:.1f}d) | Exits: {exits}")
    
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
    print("ROUND 3 FINAL VALIDATION")
    print("Optimized tools based on learnings from comprehensive testing")
    print("=" * 80)
    print("Dataset: 16 pairs | OOS: 2nd half | Step: 8 bars | Fees: 0.65%")
    print("Pass criteria: ≥30 signals AND (WR≥55% OR PF≥1.5)")
    print("=" * 80)
    
    all_results = []
    
    for i, (tool_func, regime) in enumerate(FINAL_TOOLS, 1):
        print(f"\n[{i}/{len(FINAL_TOOLS)}] Testing {tool_func.__name__} ({regime})")
        print("-" * 50)
        
        trades = validate_tool(tool_func, regime)
        result = final_report(tool_func.__name__, trades, regime)
        
        if result:
            all_results.append(result)
    
    print(f"\n{'='*80}")
    print("🏆 ROUND 3 FINAL SCORECARD")
    print(f"{'='*80}")
    
    passed = [r for r in all_results if r['passed']]
    killed = [r for r in all_results if not r['passed']]
    
    if passed:
        print(f"\n✅ SURVIVED ROUND 3 ({len(passed)}):")
        for r in sorted(passed, key=lambda x: x['avg_ret'], reverse=True):
            print(f"  🌟 {r['tool']}: {r['signals']} signals, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    else:
        print(f"\n❌ NO TOOLS SURVIVED ROUND 3")
    
    if killed:
        print(f"\n❌ KILLED IN ROUND 3 ({len(killed)}):")
        for r in sorted(killed, key=lambda x: x['avg_ret'], reverse=True):
            print(f"  💀 {r['tool']}: {r['signals']} signals, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    
    print(f"\n{'='*80}")
    print("ROUND 3 VALIDATION COMPLETE")
    print(f"{'='*80}")
    
    return all_results

if __name__ == "__main__":
    results = main()