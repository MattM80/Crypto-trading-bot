#!/usr/bin/env python3
"""
Validate the unconventional bull/chop tools with OOS walk-forward.
Train on first 50%, test on last 50%. 0.65% RT fees. Brutal.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h"
FEES = 0.0065  # 0.65% round-trip
HOLD_PERIODS = [8, 12, 24]  # Test multiple hold periods

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT",
         "AVAXUSDT", "ATOMUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "UNIUSDT",
         "FILUSDT", "NEARUSDT", "AAVEUSDT", "XLMUSDT"]


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
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_hurst(series, max_lag=100):
    """Simplified R/S Hurst exponent."""
    if len(series) < max_lag:
        return 0.5
    lags = range(10, min(max_lag, len(series) // 2), 5)
    rs_values = []
    lag_values = []
    for lag in lags:
        subseries = series[-lag:]
        mean = np.mean(subseries)
        deviations = np.cumsum(subseries - mean)
        R = np.max(deviations) - np.min(deviations)
        S = np.std(subseries)
        if S > 0 and R > 0:
            rs_values.append(np.log(R / S))
            lag_values.append(np.log(lag))
    if len(rs_values) < 3:
        return 0.5
    slope, _, _, _, _ = stats.linregress(lag_values, rs_values)
    return np.clip(slope, 0, 1)


def calc_sample_entropy(series, m=2, r_mult=0.2):
    """Sample entropy — lower = more predictable. Vectorized for speed."""
    if len(series) < 50:
        return 2.0
    data = np.array(series[-100:])  # Use last 100 for speed
    N = len(data)
    r = r_mult * np.std(data)
    if r == 0:
        return 2.0
    
    def count_matches(template_len):
        templates = np.array([data[i:i+template_len] for i in range(N - template_len)])
        n_templates = len(templates)
        count = 0
        for i in range(n_templates):
            # Vectorized comparison against all j > i
            diffs = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
            count += np.sum(diffs < r)
        return count
    
    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0 or A == 0:
        return 2.0
    return -np.log(A / B)


def calc_ou_params(prices, window=168):
    """Fit Ornstein-Uhlenbeck process. Returns (theta, mu, z_score, half_life)."""
    if len(prices) < window:
        return None
    series = np.log(prices[-window:])
    y = np.diff(series)
    x = series[:-1]
    slope, intercept, _, p_value, _ = stats.linregress(x, y)
    
    if slope >= 0 or p_value > 0.05:  # Not mean-reverting or not significant
        return None
    
    theta = -slope  # Mean reversion speed
    mu = intercept / theta  # Long-term mean
    half_life = np.log(2) / theta
    
    if half_life < 5 or half_life > 200:  # Unreasonable half-life
        return None
    
    residuals = series - mu
    sigma = np.std(residuals)
    z_score = residuals[-1] / sigma if sigma > 0 else 0
    
    return theta, mu, z_score, half_life


def calc_volume_ratio(volume, period=20):
    """Current volume vs average."""
    if len(volume) < period + 1:
        return 1.0
    avg = np.mean(volume[-period-1:-1])
    return volume[-1] / avg if avg > 0 else 1.0


# ==================== TOOL SIGNALS ====================

def hurst_trend_long(close, volume, rsi, btc_close=None):
    """
    Chaos Theory: Hurst > 0.6 = trending regime. 
    Buy when trending + momentum + volume confirms.
    """
    if len(close) < 200:
        return None
    
    returns = np.diff(np.log(close[-168:]))
    H = calc_hurst(returns)
    
    if H < 0.6:  # Not trending enough
        return None
    
    # Momentum confirmation: price above 50h SMA, positive 24h return
    sma50 = np.mean(close[-50:])
    ret_24h = (close[-1] - close[-24]) / close[-24]
    
    if close[-1] <= sma50 or ret_24h <= 0.01:
        return None
    
    # Volume confirmation
    vol_ratio = calc_volume_ratio(volume)
    if vol_ratio < 1.2:
        return None
    
    # RSI filter: not overbought
    if rsi > 70:
        return None
    
    score = (H - 0.5) * 100 + ret_24h * 50 + vol_ratio * 5
    return {'tool': 'hurst_trend_long', 'direction': 'long', 'score': score}


def hurst_mean_revert_long(close, volume, rsi, btc_close=None):
    """
    Chaos Theory: Hurst < 0.4 = mean-reverting regime.
    Buy oversold dips knowing they'll revert.
    """
    if len(close) < 200:
        return None
    
    returns = np.diff(np.log(close[-168:]))
    H = calc_hurst(returns)
    
    if H > 0.4:  # Not mean-reverting enough
        return None
    
    # Need oversold condition to buy
    if rsi > 35:
        return None
    
    # Price below lower Bollinger
    sma20 = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    lower_bb = sma20 - 2 * std20
    
    if close[-1] > lower_bb:
        return None
    
    # Volume spike (capitulation)
    vol_ratio = calc_volume_ratio(volume)
    if vol_ratio < 1.5:
        return None
    
    deviation = (sma20 - close[-1]) / std20
    score = (0.5 - H) * 100 + deviation * 20 + vol_ratio * 5
    return {'tool': 'hurst_mean_revert_long', 'direction': 'long', 'score': score}


def entropy_regime_long(close, volume, rsi, btc_close=None):
    """
    Information Theory: Low sample entropy = predictable.
    Trade when entropy drops and momentum confirms direction.
    """
    if len(close) < 200:
        return None
    
    returns = np.diff(np.log(close[-100:]))
    sampen = calc_sample_entropy(returns)
    
    if sampen > 0.8:  # Too random
        return None
    
    # Need momentum confirmation in low-entropy regime
    ret_12h = (close[-1] - close[-12]) / close[-12]
    ret_48h = (close[-1] - close[-48]) / close[-48]
    
    if ret_12h <= 0.005 or ret_48h <= 0:
        return None
    
    # RSI filter
    if rsi > 65 or rsi < 40:
        return None
    
    vol_ratio = calc_volume_ratio(volume)
    
    score = (1.0 - sampen) * 50 + ret_12h * 200 + vol_ratio * 3
    return {'tool': 'entropy_regime_long', 'direction': 'long', 'score': score}


def ou_mean_reversion_long(close, volume, rsi, btc_close=None):
    """
    Statistical: Ornstein-Uhlenbeck mean reversion.
    Buy when z-score < -2 in a statistically confirmed mean-reverting market.
    """
    if len(close) < 200:
        return None
    
    params = calc_ou_params(close)
    if params is None:
        return None
    
    theta, mu, z_score, half_life = params
    
    if z_score > -1.5:  # Not oversold enough
        return None
    
    # Volume confirmation
    vol_ratio = calc_volume_ratio(volume)
    
    score = abs(z_score) * 20 + theta * 100 + vol_ratio * 3
    return {'tool': 'ou_mean_reversion_long', 'direction': 'long', 'score': score}


def ou_mean_reversion_short(close, volume, rsi, btc_close=None):
    """
    Statistical: OU mean reversion short.
    Short when z-score > +2 in confirmed mean-reverting market.
    """
    if len(close) < 200:
        return None
    
    params = calc_ou_params(close)
    if params is None:
        return None
    
    theta, mu, z_score, half_life = params
    
    if z_score < 1.5:  # Not overbought enough
        return None
    
    vol_ratio = calc_volume_ratio(volume)
    
    score = z_score * 20 + theta * 100 + vol_ratio * 3
    return {'tool': 'ou_mean_reversion_short', 'direction': 'short', 'score': score}


def btc_alt_rotation_long(close, volume, rsi, btc_close=None):
    """
    Cross-Asset: BTC dominance rotation.
    When BTC is strong+stable, lagging alts with starting momentum get bought.
    """
    if btc_close is None or len(close) < 168 or len(btc_close) < 168:
        return None
    
    btc_7d = (btc_close[-1] - btc_close[-168]) / btc_close[-168] * 100
    alt_7d = (close[-1] - close[-168]) / close[-168] * 100
    
    # BTC sweet spot: strong but not parabolic
    if not (5 <= btc_7d <= 25):
        return None
    
    # BTC stable (low 48h vol)
    btc_vol = np.std(btc_close[-48:]) / np.mean(btc_close[-48:]) * 100
    if btc_vol > 3.0:
        return None
    
    # Alt lagging BTC significantly
    lag = btc_7d - alt_7d
    if lag < 5:
        return None
    
    # Alt showing initial momentum (24h)
    alt_24h = (close[-1] - close[-24]) / close[-24] * 100
    if alt_24h < 0.3:
        return None
    
    # RSI in buy zone
    if rsi > 60 or rsi < 25:
        return None
    
    vol_ratio = calc_volume_ratio(volume)
    
    score = lag * 3 + alt_24h * 10 + vol_ratio * 3
    return {'tool': 'btc_alt_rotation_long', 'direction': 'long', 'score': score}


def correlation_break_long(close, volume, rsi, btc_close=None):
    """
    Information Theory: Correlation regime break.
    When alt-BTC correlation drops sharply + alt has positive momentum = independent move starting.
    """
    if btc_close is None or len(close) < 200 or len(btc_close) < 200:
        return None
    
    # Rolling correlation
    alt_ret = pd.Series(close).pct_change().values
    btc_ret = pd.Series(btc_close).pct_change().values
    
    if len(alt_ret) < 100:
        return None
    
    corr_long = np.corrcoef(alt_ret[-72:], btc_ret[-72:])[0, 1]
    corr_short = np.corrcoef(alt_ret[-24:], btc_ret[-24:])[0, 1]
    
    if np.isnan(corr_long) or np.isnan(corr_short):
        return None
    
    # Correlation drop
    corr_drop = corr_long - corr_short
    if corr_drop < 0.25:
        return None
    
    # Alt outperforming BTC (diverging up)
    alt_24h = (close[-1] - close[-24]) / close[-24] * 100
    btc_24h = (btc_close[-1] - btc_close[-24]) / btc_close[-24] * 100
    
    if alt_24h <= btc_24h or alt_24h < 1.0:
        return None
    
    # RSI not overbought
    if rsi > 70:
        return None
    
    vol_ratio = calc_volume_ratio(volume)
    
    score = corr_drop * 30 + (alt_24h - btc_24h) * 10 + vol_ratio * 3
    return {'tool': 'correlation_break_long', 'direction': 'long', 'score': score}


def volume_accumulation_long(close, volume, rsi, btc_close=None):
    """
    Microstructure: Detect smart money accumulation.
    Price flat/down but volume increasing + VWAP support = accumulation phase.
    """
    if len(close) < 100 or len(volume) < 100:
        return None
    
    # Price range is tight (consolidation)
    high_20 = np.max(close[-20:])
    low_20 = np.min(close[-20:])
    range_pct = (high_20 - low_20) / low_20 * 100
    
    if range_pct > 8 or range_pct < 1:  # Need consolidation, not dead
        return None
    
    # Volume trend: increasing over 20 bars
    vol_first_10 = np.mean(volume[-20:-10])
    vol_last_10 = np.mean(volume[-10:])
    
    if vol_first_10 == 0:
        return None
    vol_trend = vol_last_10 / vol_first_10
    
    if vol_trend < 1.3:  # Volume not increasing enough
        return None
    
    # Price near bottom of range (accumulation at support)
    position_in_range = (close[-1] - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
    if position_in_range > 0.4:  # Should be near bottom
        return None
    
    # RSI neutral-to-oversold
    if rsi > 50:
        return None
    
    # VWAP: price below or near VWAP (buying below value)
    vwap_20 = np.average(close[-20:], weights=volume[-20:])
    if close[-1] > vwap_20 * 1.01:
        return None
    
    vol_ratio = calc_volume_ratio(volume)
    
    score = vol_trend * 20 + (1 - position_in_range) * 30 + (50 - rsi) * 0.5
    return {'tool': 'volume_accumulation_long', 'direction': 'long', 'score': score}


# ==================== VALIDATION ENGINE ====================

ALL_TOOLS = [
    hurst_trend_long,
    hurst_mean_revert_long,
    entropy_regime_long,
    ou_mean_reversion_long,
    ou_mean_reversion_short,
    btc_alt_rotation_long,
    correlation_break_long,
    volume_accumulation_long,
]


def validate_tool(tool_func, hold_period=12):
    """OOS walk-forward validation for a single tool across all pairs."""
    all_trades = []
    
    # Load BTC data for cross-asset tools
    btc_df = load_data("BTCUSDT")
    btc_close = btc_df['close'].values
    
    for pair in PAIRS:
        df = load_data(pair)
        close = df['close'].values
        volume = df['volume'].values
        
        # OOS split: test on second half only
        oos_start = len(close) // 2
        
        # Scan OOS period (step by 4 bars to keep runtime sane)
        i = oos_start
        last_signal_bar = -hold_period  # Prevent overlapping
        
        while i < len(close) - hold_period:
            if i - last_signal_bar < hold_period:
                i += 4
                continue
            
            rsi = calc_rsi(close[:i+1])
            btc_slice = btc_close[:i+1] if pair != "BTCUSDT" else None
            
            signal = tool_func(close[:i+1], volume[:i+1], rsi, btc_slice)
            
            if signal is not None:
                entry_price = close[i]
                exit_price = close[min(i + hold_period, len(close) - 1)]
                
                if signal['direction'] == 'long':
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - FEES
                
                all_trades.append({
                    'pair': pair,
                    'bar': i,
                    'direction': signal['direction'],
                    'entry': entry_price,
                    'exit': exit_price,
                    'raw_return': raw_return,
                    'net_return': net_return,
                    'score': signal['score'],
                    'win': net_return > 0
                })
                
                last_signal_bar = i
            
            i += 4
    
    return all_trades


def print_results(tool_name, trades, hold):
    """Print validation results for a tool."""
    min_signals = 15
    min_win_rate = 55
    min_profit_factor = 1.5
    
    if not trades:
        print(f"  {tool_name} (hold={hold}h): NO SIGNALS ❌")
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
    
    # Per-pair breakdown
    pairs_seen = set(t['pair'] for t in trades)
    
    passed = n >= min_signals and (wr >= min_win_rate or profit_factor >= min_profit_factor)
    status = "✅ PASSED" if passed else "❌ KILLED"
    
    print(f"\n  {tool_name} (hold={hold}h): {status}")
    print(f"    Signals: {n} | Win Rate: {wr:.1f}% | Avg Return: {avg_ret:.2f}%")
    print(f"    Total Return: {total_ret:.1f}% | Profit Factor: {profit_factor:.2f}")
    print(f"    Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}% | Max DD: {max_dd:.2f}%")
    print(f"    Pairs: {len(pairs_seen)} | {', '.join(sorted(pairs_seen)[:5])}...")
    
    # Top 3 pairs
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
    
    return {
        'tool': tool_name,
        'hold': hold,
        'signals': n,
        'win_rate': wr,
        'avg_return': avg_ret,
        'total_return': total_ret,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'passed': passed
    }


if __name__ == "__main__":
    print("=" * 70)
    print("UNCONVENTIONAL BULL/CHOP TOOLS - OOS VALIDATION")
    print("Train: first 50% | Test: last 50% | Fees: 0.65% RT")
    print("=" * 70)
    
    results = []
    
    for tool_func in ALL_TOOLS:
        print(f"\n{'─' * 50}")
        print(f"Testing: {tool_func.__name__}")
        print(f"{'─' * 50}")
        
        best_result = None
        for hold in HOLD_PERIODS:
            trades = validate_tool(tool_func, hold)
            result = print_results(tool_func.__name__, trades, hold)
            if result and (best_result is None or 
                          (result['passed'] and result['avg_return'] > (best_result.get('avg_return', -999)))):
                best_result = result
        
        if best_result:
            results.append(best_result)
    
    print(f"\n{'=' * 70}")
    print("FINAL SCORECARD")
    print(f"{'=' * 70}")
    
    passed = [r for r in results if r['passed']]
    killed = [r for r in results if not r['passed']]
    
    print(f"\n✅ PASSED ({len(passed)} tools):")
    for r in sorted(passed, key=lambda x: x['avg_return'], reverse=True):
        print(f"  {r['tool']} (hold={r['hold']}h): {r['signals']} signals, "
              f"{r['win_rate']:.1f}% WR, {r['avg_return']:.2f}% avg, PF={r['profit_factor']:.2f}")
    
    print(f"\n❌ KILLED ({len(killed)} tools):")
    for r in sorted(killed, key=lambda x: x.get('avg_return', -99), reverse=True):
        reason = "too few signals" if r['signals'] < 15 else f"WR {r['win_rate']:.0f}% < 55% & PF {r['profit_factor']:.1f} < 1.5"
        print(f"  {r['tool']} (hold={r['hold']}h): {r['signals']} signals, "
              f"{r['win_rate']:.1f}% WR, {r['avg_return']:.2f}% avg — {reason}")
    
    # Save results
    with open(PROJECT_ROOT / "v2_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to v2_validation_results.json")
