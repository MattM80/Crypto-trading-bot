#!/usr/bin/env python3
"""
REGIME-AWARE VALIDATION: Test bull tools ONLY during bull regimes.
Uses 3yr extended data. Bull regime = 30d BTC return > 8% AND BTC > 50 SMA.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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
    ema = close[-period]
    for p in close[-period+1:]:
        ema = (p - ema) * mult + ema
    return ema


def calc_hurst(series, max_lag=100):
    if len(series) < max_lag: return 0.5
    lags = range(10, min(max_lag, len(series)//2), 5)
    rs_v, lag_v = [], []
    for lag in lags:
        sub = series[-lag:]
        mean = np.mean(sub)
        dev = np.cumsum(sub - mean)
        R = np.max(dev) - np.min(dev)
        S = np.std(sub)
        if S > 0 and R > 0:
            rs_v.append(np.log(R/S))
            lag_v.append(np.log(lag))
    if len(rs_v) < 3: return 0.5
    slope, _, _, _, _ = stats.linregress(lag_v, rs_v)
    return np.clip(slope, 0, 1)


def calc_ou_params(prices, window=168):
    if len(prices) < window: return None
    series = np.log(prices[-window:])
    y = np.diff(series)
    x = series[:-1]
    slope, intercept, _, p_val, _ = stats.linregress(x, y)
    if slope >= 0 or p_val > 0.05: return None
    theta = -slope
    mu = intercept / theta
    half_life = np.log(2) / theta
    if half_life < 5 or half_life > 200: return None
    residuals = series - mu
    sigma = np.std(residuals)
    z_score = residuals[-1] / sigma if sigma > 0 else 0
    return theta, mu, z_score, half_life


def volume_trend(volume, short_p=48, long_p=168):
    if len(volume) < long_p: return 1.0
    s = np.mean(volume[-short_p:])
    l = np.mean(volume[-long_p:])
    return s / l if l > 0 else 1.0


def is_bull_regime(btc_close, idx):
    """Check if we're in a bull regime at this bar."""
    if idx < 720: return False
    ret_30d = (btc_close[idx] - btc_close[idx-720]) / btc_close[idx-720]
    sma50 = np.mean(btc_close[max(0,idx-50):idx])
    return ret_30d > 0.08 and btc_close[idx] > sma50


def trailing_exit(close, entry_idx, max_hold, direction, trail_pct=0.08, sl_pct=0.12):
    entry = close[entry_idx]
    best = entry
    for off in range(1, min(max_hold, len(close) - entry_idx)):
        p = close[entry_idx + off]
        if direction == 'long':
            if p > best: best = p
            if (best - p) / best >= trail_pct: return p, off, 'trail'
            if (entry - p) / entry >= sl_pct: return p, off, 'stop'
        else:
            if p < best: best = p
            if (p - best) / best >= trail_pct: return p, off, 'trail'
            if (p - entry) / entry >= sl_pct: return p, off, 'stop'
    final = close[min(entry_idx + max_hold, len(close)-1)]
    return final, max_hold, 'max_hold'


# ==================== TOOLS ====================

def trend_structure_long(close, high, low, volume, rsi):
    if len(close) < 500: return None
    w1h = np.max(high[-336:-252]) if len(high)>336 else 0
    w1l = np.min(low[-336:-252]) if len(low)>336 else 0
    w2h = np.max(high[-252:-168])
    w2l = np.min(low[-252:-168])
    w3h = np.max(high[-168:-84])
    w3l = np.min(low[-168:-84])
    if w1h == 0: return None
    hh = w2h > w1h and w3h > w2h
    hl = w2l > w1l and w3l > w2l
    if not (hh and hl): return None
    if rsi > 50 or rsi < 25: return None
    recent_high = np.max(high[-168:])
    dd = (close[-1] - recent_high) / recent_high
    if dd < -0.15 or dd > -0.03: return None
    ema100 = calc_ema(close, 100)
    if close[-1] < ema100: return None
    return {'tool': 'trend_structure_long', 'direction': 'long', 'score': 40 + abs(dd)*100}


def weekly_momentum_pullback(close, high, low, volume, rsi):
    if len(close) < 500: return None
    ret_2w = (close[-1] - close[-336]) / close[-336] if len(close)>=336 else 0
    ret_4w = (close[-1] - close[-672]) / close[-672] if len(close)>=672 else 0
    if ret_2w < 0.08 or ret_4w < 0.10: return None
    ret_48h = (close[-1] - close[-48]) / close[-48]
    if ret_48h > -0.03 or ret_48h < -0.15: return None
    sma200 = calc_sma(close, 200)
    if close[-1] < sma200: return None
    if rsi < 30 or rsi > 55: return None
    return {'tool': 'weekly_momentum_pullback', 'direction': 'long', 'score': ret_4w*100 + abs(ret_48h)*50}


def golden_cross_swing(close, high, low, volume, rsi):
    if len(close) < 500: return None
    ema50_now = calc_ema(close, 50)
    ema200_now = calc_ema(close, 200)
    ema50_prev = calc_ema(close[:-12], 50)
    ema200_prev = calc_ema(close[:-12], 200)
    if not (ema50_prev <= ema200_prev and ema50_now > ema200_now): return None
    if close[-1] < ema50_now: return None
    if rsi > 68 or rsi < 40: return None
    ret_1w = (close[-1] - close[-168]) / close[-168] if len(close)>=168 else 0
    if ret_1w < 0: return None
    return {'tool': 'golden_cross_swing', 'direction': 'long', 'score': 30 + ret_1w*100}


def accumulation_breakout(close, high, low, volume, rsi):
    if len(close) < 500: return None
    rh = np.max(high[-336:-48])
    rl = np.min(low[-336:-48])
    rng = (rh - rl) / rl * 100
    if rng > 20 or rng < 3: return None
    if close[-1] <= rh * 1.01: return None
    vol_s = np.mean(volume[-24:])
    vol_l = np.mean(volume[-336:-48])
    vr = vol_s / vol_l if vol_l > 0 else 1
    if vr < 1.5: return None
    if rsi > 72: return None
    return {'tool': 'accumulation_breakout', 'direction': 'long', 'score': vr*15 + (close[-1]-rh)/rh*1000}


def dip_in_uptrend(close, high, low, volume, rsi):
    if len(close) < 500: return None
    ret_4w = (close[-1] - close[-672]) / close[-672] if len(close)>=672 else 0
    if ret_4w < 0.05: return None
    recent_high = np.max(high[-168:])
    dd = (close[-1] - recent_high) / recent_high
    if dd > -0.05 or dd < -0.25: return None
    if rsi > 40: return None
    sma200 = calc_sma(close, 200)
    if close[-1] < sma200: return None
    return {'tool': 'dip_in_uptrend', 'direction': 'long', 'score': abs(dd)*100 + ret_4w*50}


def mean_reversion_weekly(close, high, low, volume, rsi):
    if len(close) < 700: return None
    lookback = 672
    mean_4w = np.mean(close[-lookback:])
    std_4w = np.std(close[-lookback:])
    if std_4w == 0: return None
    z = (close[-1] - mean_4w) / std_4w
    if z > -1.5: return None
    ret_48h = (close[-1] - close[-48]) / close[-48]
    if ret_48h < -0.10: return None
    if rsi > 40: return None
    return {'tool': 'mean_reversion_weekly', 'direction': 'long', 'score': abs(z)*20}


def hurst_trend_long(close, high, low, volume, rsi):
    if len(close) < 500: return None
    returns = np.diff(np.log(close[-168:]))
    H = calc_hurst(returns)
    if H < 0.6: return None
    sma50 = calc_sma(close, 50)
    ret_24h = (close[-1] - close[-24]) / close[-24]
    if close[-1] <= sma50 or ret_24h <= 0.01: return None
    vr = volume_trend(volume)
    if vr < 1.2: return None
    if rsi > 70: return None
    return {'tool': 'hurst_trend_long', 'direction': 'long', 'score': (H-0.5)*100 + ret_24h*50}


def ou_mean_reversion_long(close, high, low, volume, rsi):
    if len(close) < 500: return None
    params = calc_ou_params(close)
    if params is None: return None
    theta, mu, z_score, half_life = params
    if z_score > -1.5: return None
    return {'tool': 'ou_mean_reversion_long', 'direction': 'long', 'score': abs(z_score)*20 + theta*100}


ALL_TOOLS = [
    trend_structure_long,
    weekly_momentum_pullback,
    golden_cross_swing,
    accumulation_breakout,
    dip_in_uptrend,
    mean_reversion_weekly,
    hurst_trend_long,
    ou_mean_reversion_long,
]


def validate(tool_func, max_hold=336):
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
        
        while i < len(close) - max_hold:
            if i - last_signal < 48:
                i += 8
                continue
            
            # REGIME FILTER: only fire during bull markets
            if not is_bull_regime(btc_close, i):
                i += 8
                continue
            
            rsi = calc_rsi(close[:i+1])
            sig = tool_func(close[:i+1], high_arr[:i+1], low_arr[:i+1], vol[:i+1], rsi)
            
            if sig is not None:
                exit_p, hold_bars, exit_r = trailing_exit(close, i, max_hold, sig['direction'])
                entry_p = close[i]
                
                if sig['direction'] == 'long':
                    raw = (exit_p - entry_p) / entry_p
                else:
                    raw = (entry_p - exit_p) / entry_p
                
                net = raw - FEES
                
                all_trades.append({
                    'pair': pair, 'bar': i, 'direction': sig['direction'],
                    'entry': float(entry_p), 'exit': float(exit_p),
                    'raw': float(raw), 'net': float(net),
                    'win': net > 0, 'hold': int(hold_bars), 'exit_reason': exit_r
                })
                last_signal = i
            
            i += 8
    
    return all_trades


def report(name, trades):
    if not trades:
        print(f"  {name}: NO SIGNALS ❌")
        return None
    
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    avg_ret = np.mean([t['net'] for t in trades]) * 100
    total_ret = sum(t['net'] for t in trades) * 100
    
    w_trades = [t['net'] for t in trades if t['win']]
    l_trades = [t['net'] for t in trades if not t['win']]
    avg_win = np.mean(w_trades)*100 if w_trades else 0
    avg_loss = np.mean(l_trades)*100 if l_trades else 0
    gp = sum(w_trades) if w_trades else 0
    gl = abs(sum(l_trades)) if l_trades else 0.001
    pf = gp / gl if gl > 0 else 999
    max_dd = min(t['net'] for t in trades) * 100
    avg_hold = np.mean([t['hold'] for t in trades])
    
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1
    
    pairs = set(t['pair'] for t in trades)
    passed = n >= 30 and (wr >= 55 or pf >= 1.5)
    status = "✅ PASSED" if passed else "❌ KILLED"
    
    print(f"\n  {name}: {status}")
    print(f"    Signals: {n} | WR: {wr:.1f}% | Avg: {avg_ret:.2f}% | Total: {total_ret:.1f}%")
    print(f"    PF: {pf:.2f} | Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}% | Max DD: {max_dd:.2f}%")
    print(f"    Hold: {avg_hold:.0f}h ({avg_hold/24:.1f}d) | Pairs: {len(pairs)} | Exits: {exits}")
    
    pair_stats = {}
    for t in trades:
        pair_stats.setdefault(t['pair'], []).append(t['net'])
    
    top = sorted(pair_stats.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print(f"    Top 3:")
    for p, rets in top[:3]:
        pw = sum(1 for r in rets if r > 0)
        print(f"      {p}: {len(rets)} trades, {pw}/{len(rets)} wins ({pw/len(rets)*100:.0f}%), avg {np.mean(rets)*100:.2f}%")
    
    return {'tool': name, 'signals': n, 'wr': wr, 'avg_ret': avg_ret, 'pf': pf, 'passed': passed}


if __name__ == "__main__":
    print("=" * 70)
    print("REGIME-AWARE BULL TOOLS VALIDATION")
    print("3yr data | OOS 2nd half | Bull regime only | 0.65% fees | Trailing stops")
    print("=" * 70)
    
    results = []
    for func in ALL_TOOLS:
        print(f"\n{'─'*60}")
        print(f"Testing: {func.__name__}")
        print(f"{'─'*60}")
        trades = validate(func, max_hold=336)
        r = report(func.__name__, trades)
        if r: results.append(r)
    
    print(f"\n{'='*70}")
    print("FINAL SCORECARD")
    print(f"{'='*70}")
    
    passed = [r for r in results if r['passed']]
    killed = [r for r in results if not r['passed']]
    
    print(f"\n✅ PASSED ({len(passed)}):")
    for r in sorted(passed, key=lambda x: x['avg_ret'], reverse=True):
        print(f"  {r['tool']}: {r['signals']} signals, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
    
    print(f"\n❌ KILLED ({len(killed)}):")
    for r in sorted(killed, key=lambda x: x['avg_ret'], reverse=True):
        print(f"  {r['tool']}: {r['signals']} signals, {r['wr']:.1f}% WR, {r['avg_ret']:.2f}% avg, PF={r['pf']:.2f}")
