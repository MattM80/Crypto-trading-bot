#!/usr/bin/env python3
"""
SWING/MACRO BULL TOOLS — Longer hold periods (2-14 days).
The insight: 0.65% fees are nothing on a 15-40% move over 2 weeks.
Stop thinking in hours. Think in days and weeks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h"
FEES = 0.0065

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT",
         "AVAXUSDT", "ATOMUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "UNIUSDT",
         "FILUSDT", "NEARUSDT", "AAVEUSDT", "XLMUSDT"]

# SWING hold periods: 2 days, 5 days, 7 days, 14 days
HOLD_PERIODS = [48, 120, 168, 336]


def load_data(pair):
    df = pd.read_csv(DATA_DIR / f"{pair}_1h.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['returns'] = df['close'].pct_change()
    return df


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def calc_hurst(series, max_lag=100):
    """Simplified R/S Hurst exponent."""
    if len(series) < max_lag:
        return 0.5
    lags = range(10, min(max_lag, len(series) // 2), 5)
    rs_values, lag_values = [], []
    for lag in lags:
        sub = series[-lag:]
        mean = np.mean(sub)
        dev = np.cumsum(sub - mean)
        R = np.max(dev) - np.min(dev)
        S = np.std(sub)
        if S > 0 and R > 0:
            rs_values.append(np.log(R / S))
            lag_values.append(np.log(lag))
    if len(rs_values) < 3:
        return 0.5
    slope, _, _, _, _ = stats.linregress(lag_values, rs_values)
    return np.clip(slope, 0, 1)


def calc_weekly_momentum(close, weeks=1):
    """Return over N weeks (N*168 hours)."""
    bars = weeks * 168
    if len(close) < bars + 1:
        return 0
    return (close[-1] - close[-bars]) / close[-bars]


def calc_drawdown_from_high(close, lookback=336):
    """How far is price from its N-bar high."""
    if len(close) < lookback:
        return 0
    high = np.max(close[-lookback:])
    return (close[-1] - high) / high


def calc_sma(close, period):
    if len(close) < period:
        return close[-1]
    return np.mean(close[-period:])


def calc_ema(close, period):
    if len(close) < period:
        return close[-1]
    multiplier = 2 / (period + 1)
    ema = close[-period]
    for p in close[-period+1:]:
        ema = (p - ema) * multiplier + ema
    return ema


def calc_atr(high, low, close, period=14):
    if len(high) < period + 1:
        return 0
    trs = []
    for i in range(-period, 0):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        trs.append(tr)
    return np.mean(trs)


def volume_trend(volume, period1=48, period2=168):
    """Short-term volume vs long-term volume."""
    if len(volume) < period2:
        return 1.0
    short = np.mean(volume[-period1:])
    long = np.mean(volume[-period2:])
    return short / long if long > 0 else 1.0


# ==================== SWING BULL TOOLS ====================

def weekly_momentum_pullback(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: Strong weekly trend + short-term pullback = swing entry.
    Buy when 2-week momentum is strong but price pulled back in last 48h.
    Hold for continuation. Target: 10-30% over 1-2 weeks.
    """
    if len(close) < 500:
        return None
    
    # Strong multi-week uptrend
    ret_2w = calc_weekly_momentum(close, 2)
    ret_4w = calc_weekly_momentum(close, 4)
    
    if ret_2w < 0.08 or ret_4w < 0.10:  # Need solid uptrend
        return None
    
    # Short-term pullback (48h dip within uptrend)
    ret_48h = (close[-1] - close[-48]) / close[-48]
    if ret_48h > -0.03 or ret_48h < -0.15:  # Need 3-15% pullback
        return None
    
    # Price above 200h SMA (long-term trend intact)
    sma200 = calc_sma(close, 200)
    if close[-1] < sma200:
        return None
    
    # RSI pulled back but not crashed
    if rsi < 30 or rsi > 55:
        return None
    
    # Volume increasing on pullback (institutional buying)
    vol_ratio = volume_trend(volume, 24, 168)
    
    score = ret_4w * 100 + abs(ret_48h) * 50 + vol_ratio * 5
    return {'tool': 'weekly_momentum_pullback', 'direction': 'long', 'score': score}


def trend_structure_long(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: Higher highs + higher lows over 2+ weeks = established uptrend.
    Enter on RSI dip within the structure. Ride for weeks.
    """
    if len(close) < 500:
        return None
    
    # Check trend structure: compare 4 weekly windows
    w1_high = np.max(high[-336:-252]) if len(high) > 336 else 0
    w1_low = np.min(low[-336:-252]) if len(low) > 336 else 0
    w2_high = np.max(high[-252:-168])
    w2_low = np.min(low[-252:-168])
    w3_high = np.max(high[-168:-84])
    w3_low = np.min(low[-168:-84])
    w4_high = np.max(high[-84:])
    w4_low = np.min(low[-84:])
    
    if w1_high == 0:
        return None
    
    # Higher highs
    hh = w2_high > w1_high and w3_high > w2_high
    # Higher lows  
    hl = w2_low > w1_low and w3_low > w2_low
    
    if not (hh and hl):
        return None
    
    # Current price in a dip within the structure
    if rsi > 50 or rsi < 25:
        return None
    
    # Not too far from recent high (pullback, not breakdown)
    dd = calc_drawdown_from_high(close, 168)
    if dd < -0.15 or dd > -0.03:  # 3-15% from high
        return None
    
    # Price above long-term EMA
    ema100 = calc_ema(close, 100)
    if close[-1] < ema100:
        return None
    
    score = (1 if hh else 0) * 20 + (1 if hl else 0) * 20 + abs(dd) * 100 + (50 - rsi) * 0.5
    return {'tool': 'trend_structure_long', 'direction': 'long', 'score': score}


def accumulation_breakout(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: Long consolidation (2-4 weeks) → breakout with volume.
    The longer the base, the bigger the move. Hold for the full swing.
    """
    if len(close) < 500:
        return None
    
    # Detect consolidation: tight range over last 2 weeks
    range_2w_high = np.max(high[-336:-48])  # Exclude last 2 days
    range_2w_low = np.min(low[-336:-48])
    range_pct = (range_2w_high - range_2w_low) / range_2w_low * 100
    
    if range_pct > 20 or range_pct < 3:  # Need consolidation (3-20% range)
        return None
    
    # Price broke above the range in the last 48h
    if close[-1] <= range_2w_high * 1.01:  # Need clear breakout
        return None
    
    # Volume surge on breakout
    vol_short = np.mean(volume[-24:])
    vol_long = np.mean(volume[-336:-48])
    vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0
    
    if vol_ratio < 1.5:  # Need volume confirmation
        return None
    
    # RSI not already overbought
    if rsi > 72:
        return None
    
    # Trend context: price above 200h SMA
    sma200 = calc_sma(close, 200)
    if close[-1] < sma200 * 0.95:  # Allow slight below
        return None
    
    breakout_pct = (close[-1] - range_2w_high) / range_2w_high * 100
    base_duration = range_pct  # Tighter base = more energy
    
    score = vol_ratio * 15 + breakout_pct * 10 + (20 - range_pct) * 2
    return {'tool': 'accumulation_breakout', 'direction': 'long', 'score': score}


def golden_cross_swing(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: 50h EMA crosses above 200h EMA (golden cross on hourly = ~2d/8d cross).
    Classic trend signal but held for weeks, not hours.
    """
    if len(close) < 500:
        return None
    
    ema50_now = calc_ema(close, 50)
    ema200_now = calc_ema(close, 200)
    ema50_prev = calc_ema(close[:-12], 50)  # 12h ago
    ema200_prev = calc_ema(close[:-12], 200)
    
    # Golden cross: 50 just crossed above 200
    if not (ema50_prev <= ema200_prev and ema50_now > ema200_now):
        return None
    
    # Price above both EMAs
    if close[-1] < ema50_now:
        return None
    
    # RSI in healthy range (not overbought)
    if rsi > 68 or rsi < 40:
        return None
    
    # Volume confirmation
    vol_ratio = volume_trend(volume, 48, 336)
    if vol_ratio < 0.8:  # Not declining volume
        return None
    
    # Momentum: positive 1-week return
    ret_1w = calc_weekly_momentum(close, 1)
    if ret_1w < 0:
        return None
    
    spread = (ema50_now - ema200_now) / ema200_now * 100
    score = spread * 20 + ret_1w * 100 + vol_ratio * 10
    return {'tool': 'golden_cross_swing', 'direction': 'long', 'score': score}


def btc_bull_regime_long(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: When BTC is in a confirmed bull trend (above 200h SMA, positive monthly),
    buy alts that are lagging but starting to catch up. Ride the rising tide.
    """
    if btc_close is None or len(close) < 500 or len(btc_close) < 500:
        return None
    
    # BTC in bull regime
    btc_sma200 = calc_sma(btc_close, 200)
    btc_4w = calc_weekly_momentum(btc_close, 4)
    
    if btc_close[-1] < btc_sma200 or btc_4w < 0.05:
        return None
    
    # Alt is lagging: below its own potential
    alt_4w = calc_weekly_momentum(close, 4)
    alt_sma200 = calc_sma(close, 200)
    
    lag = btc_4w - alt_4w
    if lag < 0.05:  # Alt must be lagging BTC
        return None
    
    # Alt showing signs of life (positive 1-week)
    alt_1w = calc_weekly_momentum(close, 1)
    if alt_1w < 0:
        return None
    
    # Alt price near or above 200h SMA (not in death spiral)
    if close[-1] < alt_sma200 * 0.90:
        return None
    
    # RSI not extreme
    if rsi > 65 or rsi < 30:
        return None
    
    score = lag * 100 + alt_1w * 50 + (65 - rsi) * 0.5
    return {'tool': 'btc_bull_regime_long', 'direction': 'long', 'score': score}


def dip_in_uptrend(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: Simple but effective. Strong 4-week uptrend + 
    significant dip (5-20% from high) + RSI oversold = buy the dip.
    Let the macro trend do the work over 1-2 weeks.
    """
    if len(close) < 500:
        return None
    
    # Confirmed uptrend: positive 4-week and 8-week return
    ret_4w = calc_weekly_momentum(close, 4)
    
    if ret_4w < 0.05:
        return None
    
    # Significant dip from recent high
    dd = calc_drawdown_from_high(close, 168)  # From 1-week high
    if dd > -0.05 or dd < -0.25:  # Need 5-25% dip
        return None
    
    # RSI oversold
    if rsi > 40:
        return None
    
    # Price still above 200h SMA (trend not broken)
    sma200 = calc_sma(close, 200)
    if close[-1] < sma200:
        return None
    
    # Volume spike (capitulation / forced selling)
    vol_ratio = volume_trend(volume, 24, 168)
    
    score = abs(dd) * 100 + ret_4w * 50 + (40 - rsi) * 1.0 + vol_ratio * 5
    return {'tool': 'dip_in_uptrend', 'direction': 'long', 'score': score}


def mean_reversion_weekly(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: Price deviated >2 std from 4-week mean.
    At weekly scale, mean reversion is powerful — oversold coins snap back over days.
    """
    if len(close) < 500:
        return None
    
    # 4-week (672h) mean and std
    lookback = 672
    if len(close) < lookback:
        return None
    
    mean_4w = np.mean(close[-lookback:])
    std_4w = np.std(close[-lookback:])
    
    if std_4w == 0:
        return None
    
    z_score = (close[-1] - mean_4w) / std_4w
    
    # Buy when significantly below mean
    if z_score > -1.5:
        return None
    
    # But trend context: not in a total collapse (some positive momentum recently)
    ret_48h = (close[-1] - close[-48]) / close[-48]
    if ret_48h < -0.10:  # Still falling hard, don't catch knife
        return None
    
    # RSI oversold
    if rsi > 40:
        return None
    
    # Volume confirmation
    vol_ratio = volume_trend(volume, 24, 168)
    
    score = abs(z_score) * 20 + (40 - rsi) * 0.5 + vol_ratio * 5
    return {'tool': 'mean_reversion_weekly', 'direction': 'long', 'score': score}


def multi_week_breakout(close, high, low, volume, rsi, btc_close=None):
    """
    MACRO: Price breaks above 4-week high with volume.
    At this timeframe, breakouts actually work because the move is already confirmed.
    """
    if len(close) < 700:
        return None
    
    # 4-week high (excluding last 24h to confirm breakout, not just wick)
    high_4w = np.max(high[-672:-24])
    
    # Current price above 4-week high
    if close[-1] <= high_4w * 1.01:  # Need clear break, not just touching
        return None
    
    # Held above for at least 12h (not a fakeout)
    bars_above = 0
    for i in range(-12, 0):
        if close[i] > high_4w:
            bars_above += 1
    if bars_above < 8:  # At least 8 of last 12 bars above
        return None
    
    # Volume surge
    vol_ratio = volume_trend(volume, 48, 336)
    if vol_ratio < 1.2:
        return None
    
    # Not overbought
    if rsi > 75:
        return None
    
    # EMA alignment (50 > 200 = bullish structure)
    ema50 = calc_ema(close, 50)
    ema200 = calc_ema(close, 200)
    if ema50 < ema200:
        return None
    
    breakout_pct = (close[-1] - high_4w) / high_4w * 100
    score = breakout_pct * 15 + vol_ratio * 10 + bars_above * 3
    return {'tool': 'multi_week_breakout', 'direction': 'long', 'score': score}


# ==================== TRAILING STOP EXIT ====================

def calc_exit_with_trailing(close_series, entry_idx, max_hold, direction, 
                            trailing_pct=0.08, stop_loss_pct=0.10):
    """
    Instead of fixed hold, use trailing stop.
    This lets winners run and cuts losers.
    Returns (exit_price, exit_bar, exit_reason).
    """
    entry_price = close_series[entry_idx]
    best_price = entry_price
    
    for offset in range(1, min(max_hold, len(close_series) - entry_idx)):
        price = close_series[entry_idx + offset]
        
        if direction == 'long':
            if price > best_price:
                best_price = price
            
            # Trailing stop: price dropped trailing_pct from peak
            drawdown = (best_price - price) / best_price
            if drawdown >= trailing_pct:
                return price, offset, 'trailing_stop'
            
            # Hard stop loss
            loss = (entry_price - price) / entry_price
            if loss >= stop_loss_pct:
                return price, offset, 'stop_loss'
        else:
            if price < best_price:
                best_price = price
            drawup = (price - best_price) / best_price
            if drawup >= trailing_pct:
                return price, offset, 'trailing_stop'
            loss = (price - entry_price) / entry_price
            if loss >= stop_loss_pct:
                return price, offset, 'stop_loss'
    
    # Max hold reached
    final_price = close_series[min(entry_idx + max_hold, len(close_series) - 1)]
    return final_price, max_hold, 'max_hold'


# ==================== VALIDATION ====================

ALL_TOOLS = [
    weekly_momentum_pullback,
    trend_structure_long,
    accumulation_breakout,
    golden_cross_swing,
    btc_bull_regime_long,
    dip_in_uptrend,
    mean_reversion_weekly,
    multi_week_breakout,
]


def validate_tool(tool_func, max_hold=168, use_trailing=True):
    """OOS walk-forward validation with trailing stops."""
    all_trades = []
    
    btc_df = load_data("BTCUSDT")
    btc_close = btc_df['close'].values
    
    for pair in PAIRS:
        df = load_data(pair)
        close = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        volume = df['volume'].values
        
        oos_start = len(close) // 2
        i = oos_start
        last_signal_bar = -max_hold
        
        while i < len(close) - max_hold:
            if i - last_signal_bar < min(48, max_hold // 2):  # Min 48h between entries
                i += 8
                continue
            
            rsi = calc_rsi(close[:i+1])
            btc_slice = btc_close[:i+1] if pair != "BTCUSDT" else None
            
            signal = tool_func(
                close[:i+1], high_arr[:i+1], low_arr[:i+1],
                volume[:i+1], rsi, btc_slice
            )
            
            if signal is not None:
                if use_trailing:
                    exit_price, exit_bars, exit_reason = calc_exit_with_trailing(
                        close, i, max_hold, signal['direction'],
                        trailing_pct=0.08, stop_loss_pct=0.12
                    )
                else:
                    exit_price = close[min(i + max_hold, len(close) - 1)]
                    exit_bars = max_hold
                    exit_reason = 'fixed_hold'
                
                entry_price = close[i]
                
                if signal['direction'] == 'long':
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - FEES
                
                all_trades.append({
                    'pair': pair,
                    'bar': i,
                    'direction': signal['direction'],
                    'entry': float(entry_price),
                    'exit': float(exit_price),
                    'raw_return': float(raw_return),
                    'net_return': float(net_return),
                    'score': float(signal['score']),
                    'win': bool(net_return > 0),
                    'hold_bars': int(exit_bars),
                    'exit_reason': exit_reason
                })
                
                last_signal_bar = i
            
            i += 8  # Step by 8h
    
    return all_trades


def print_results(tool_name, trades, max_hold, use_trailing):
    trail_str = "trailing" if use_trailing else "fixed"
    
    if not trades:
        print(f"  {tool_name} ({trail_str}, max={max_hold}h): NO SIGNALS ❌")
        return None
    
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    avg_ret = np.mean([t['net_return'] for t in trades]) * 100
    total_ret = sum(t['net_return'] for t in trades) * 100
    
    winning_trades = [t['net_return'] for t in trades if t['win']]
    losing_trades = [t['net_return'] for t in trades if not t['win']]
    
    avg_win = np.mean(winning_trades) * 100 if winning_trades else 0
    avg_loss = np.mean(losing_trades) * 100 if losing_trades else 0
    
    gross_profit = sum(winning_trades) if winning_trades else 0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    
    max_dd = min(t['net_return'] for t in trades) * 100
    avg_hold = np.mean([t['hold_bars'] for t in trades])
    
    exit_reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    
    pairs_seen = set(t['pair'] for t in trades)
    
    passed = n >= 15 and (wr >= 55 or profit_factor >= 1.5)
    status = "✅ PASSED" if passed else "❌ KILLED"
    
    print(f"\n  {tool_name} ({trail_str}, max={max_hold}h): {status}")
    print(f"    Signals: {n} | Win Rate: {wr:.1f}% | Avg Return: {avg_ret:.2f}%")
    print(f"    Total Return: {total_ret:.1f}% | Profit Factor: {profit_factor:.2f}")
    print(f"    Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}% | Max DD: {max_dd:.2f}%")
    print(f"    Avg Hold: {avg_hold:.0f}h ({avg_hold/24:.1f}d) | Pairs: {len(pairs_seen)}")
    print(f"    Exits: {exit_reasons}")
    
    # Best pairs
    pair_stats = {}
    for t in trades:
        if t['pair'] not in pair_stats:
            pair_stats[t['pair']] = []
        pair_stats[t['pair']].append(t['net_return'])
    
    sorted_pairs = sorted(pair_stats.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print(f"    Best pairs:")
    for p, rets in sorted_pairs[:3]:
        pw = sum(1 for r in rets if r > 0)
        print(f"      {p}: {len(rets)} trades, {pw}/{len(rets)} wins ({pw/len(rets)*100:.0f}%), avg {np.mean(rets)*100:.2f}%")
    
    if sorted_pairs:
        worst = sorted_pairs[-1]
        p, rets = worst
        pw = sum(1 for r in rets if r > 0)
        print(f"    Worst: {p}: {len(rets)} trades, {pw}/{len(rets)} wins ({pw/len(rets)*100:.0f}%), avg {np.mean(rets)*100:.2f}%")
    
    return {
        'tool': tool_name,
        'max_hold': max_hold,
        'trailing': use_trailing,
        'signals': n,
        'win_rate': float(wr),
        'avg_return': float(avg_ret),
        'total_return': float(total_ret),
        'profit_factor': float(profit_factor),
        'max_dd': float(max_dd),
        'avg_hold_hours': float(avg_hold),
        'passed': bool(passed),
        'pairs': len(pairs_seen)
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SWING/MACRO BULL TOOLS — OOS VALIDATION")
    print("Hold: days-to-weeks | Trailing stops | Fees: 0.65% RT")
    print("OOS: second half of dataset")
    print("=" * 70)
    
    results = []
    
    for tool_func in ALL_TOOLS:
        print(f"\n{'─' * 60}")
        print(f"Testing: {tool_func.__name__}")
        print(f"{'─' * 60}")
        
        # Test with trailing stop at different max holds
        for max_hold in [168, 336]:  # 1 week, 2 weeks max
            trades = validate_tool(tool_func, max_hold=max_hold, use_trailing=True)
            result = print_results(tool_func.__name__, trades, max_hold, True)
            if result:
                results.append(result)
        
        # Also test fixed hold at 1 week for comparison
        trades = validate_tool(tool_func, max_hold=168, use_trailing=False)
        result = print_results(tool_func.__name__, trades, 168, False)
        if result:
            results.append(result)
    
    print(f"\n{'=' * 70}")
    print("FINAL SCORECARD — SWING TOOLS")
    print(f"{'=' * 70}")
    
    passed = [r for r in results if r['passed']]
    killed = [r for r in results if not r['passed']]
    
    print(f"\n✅ PASSED ({len(passed)} configs):")
    for r in sorted(passed, key=lambda x: x['avg_return'], reverse=True):
        t = "trailing" if r['trailing'] else "fixed"
        print(f"  {r['tool']} ({t}, max={r['max_hold']}h): {r['signals']} signals, "
              f"{r['win_rate']:.1f}% WR, {r['avg_return']:.2f}% avg, "
              f"PF={r['profit_factor']:.2f}, hold={r['avg_hold_hours']:.0f}h")
    
    print(f"\n❌ KILLED ({len(killed)} configs):")
    for r in sorted(killed, key=lambda x: x['avg_return'], reverse=True)[:10]:
        t = "trailing" if r['trailing'] else "fixed"
        print(f"  {r['tool']} ({t}, max={r['max_hold']}h): {r['signals']} signals, "
              f"{r['win_rate']:.1f}% WR, {r['avg_return']:.2f}% avg")
    
    print(f"\nDone. {len(passed)} passed, {len(killed)} killed.")
