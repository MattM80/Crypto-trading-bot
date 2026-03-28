#!/usr/bin/env python3
"""
SIMPLE BULL STRATEGY: Just ride the trend.
When market is bullish, buy strong coins, trail stop, hold for weeks.
No fancy indicators. Just trend + trailing stop.
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

_cache = {}

def load_data(pair):
    if pair not in _cache:
        df = pd.read_csv(DATA_DIR / f"{pair}_1h.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        _cache[pair] = df
    return _cache[pair]


def calc_rsi(prices, period=14):
    if len(prices) < period + 1: return 50
    d = np.diff(prices[-period-1:])
    g = np.mean(np.where(d > 0, d, 0))
    l = np.mean(np.where(d < 0, -d, 0))
    if l == 0: return 100
    return 100 - 100 / (1 + g / l)


def is_bull(btc_close, idx):
    if idx < 720: return False
    ret_30d = (btc_close[idx] - btc_close[idx-720]) / btc_close[idx-720]
    sma50 = np.mean(btc_close[max(0,idx-50):idx])
    return ret_30d > 0.08 and btc_close[idx] > sma50


def trailing_exit(close, entry_idx, max_hold, trail_pct, sl_pct):
    entry = close[entry_idx]
    best = entry
    for off in range(1, min(max_hold, len(close) - entry_idx)):
        p = close[entry_idx + off]
        if p > best: best = p
        if (best - p) / best >= trail_pct: return p, off, 'trail'
        if (entry - p) / entry >= sl_pct: return p, off, 'stop'
    final = close[min(entry_idx + max_hold, len(close)-1)]
    return final, max_hold, 'max_hold'


# ==================== SIMPLE BULL STRATEGIES ====================

def simple_buy_uptrend(close, volume, rsi, btc_close, idx):
    """
    Dead simple: price above 50h and 200h SMA, RSI not overbought.
    Just get in and ride.
    """
    if len(close) < 200: return None
    sma50 = np.mean(close[-50:])
    sma200 = np.mean(close[-200:])
    if close[-1] <= sma50 or close[-1] <= sma200: return None
    if sma50 <= sma200: return None  # Want golden cross alignment
    if rsi > 70 or rsi < 35: return None
    # Some recent momentum
    ret_1w = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
    if ret_1w < 0.02: return None
    return {'tool': 'simple_buy_uptrend', 'direction': 'long', 'score': ret_1w * 100}


def buy_weekly_green(close, volume, rsi, btc_close, idx):
    """
    Buy after a green week with volume. Simplest momentum.
    If it went up last week with volume, ride the continuation.
    """
    if len(close) < 200 or len(volume) < 168: return None
    ret_1w = (close[-1] - close[-168]) / close[-168]
    if ret_1w < 0.05: return None  # Need 5%+ green week
    # Volume above average
    vol_ratio = np.mean(volume[-168:]) / np.mean(volume[-336:-168]) if len(volume) >= 336 else 1
    if vol_ratio < 1.1: return None
    # Not overbought
    if rsi > 72: return None
    # Above long-term SMA
    sma200 = np.mean(close[-200:])
    if close[-1] < sma200: return None
    return {'tool': 'buy_weekly_green', 'direction': 'long', 'score': ret_1w * 100 + vol_ratio * 10}


def buy_btc_leading(close, volume, rsi, btc_close, idx):
    """
    BTC pumping, this alt hasn't moved yet. Simple rotation.
    """
    if btc_close is None or len(close) < 168 or len(btc_close) < 168: return None
    btc_1w = (btc_close[-1] - btc_close[-168]) / btc_close[-168]
    alt_1w = (close[-1] - close[-168]) / close[-168]
    if btc_1w < 0.05: return None  # BTC needs to be moving
    lag = btc_1w - alt_1w
    if lag < 0.03: return None  # Alt needs to be lagging
    # Alt showing life (positive 48h)
    alt_48h = (close[-1] - close[-48]) / close[-48] if len(close) >= 48 else 0
    if alt_48h < 0: return None
    if rsi > 65: return None
    sma200 = np.mean(close[-200:]) if len(close) >= 200 else close[-1]
    if close[-1] < sma200 * 0.85: return None  # Not in death spiral
    return {'tool': 'buy_btc_leading', 'direction': 'long', 'score': lag * 100 + alt_48h * 50}


def buy_dip_bull_market(close, volume, rsi, btc_close, idx):
    """
    In a bull market, buy ANY significant dip. The market does the work.
    3-10% pullback from recent high, RSI oversold, in an uptrend.
    """
    if len(close) < 336: return None
    # Must be in uptrend (positive 4-week return)
    ret_4w = (close[-1] - close[-672]) / close[-672] if len(close) >= 672 else 0
    if ret_4w < 0.05: return None
    # Dip from recent high
    recent_high = np.max(close[-168:])
    dd = (close[-1] - recent_high) / recent_high
    if dd > -0.03 or dd < -0.15: return None  # 3-15% dip
    # RSI dipped
    if rsi > 45: return None
    # Still above 200 SMA
    sma200 = np.mean(close[-200:]) if len(close) >= 200 else close[-1]
    if close[-1] < sma200: return None
    return {'tool': 'buy_dip_bull_market', 'direction': 'long', 'score': abs(dd)*100 + (45-rsi)*0.5}


def buy_breakout_simple(close, volume, rsi, btc_close, idx):
    """
    Price at new 30-day high with volume. Simple breakout.
    In a bull market, breakouts work because there's actual buying pressure.
    """
    if len(close) < 720 or len(volume) < 168: return None
    high_30d = np.max(close[-720:-24])  # Exclude last day
    if close[-1] <= high_30d * 1.005: return None  # Need clear break
    # Volume surge
    vol_now = np.mean(volume[-24:])
    vol_avg = np.mean(volume[-720:-24])
    vr = vol_now / vol_avg if vol_avg > 0 else 1
    if vr < 1.3: return None
    if rsi > 78: return None
    return {'tool': 'buy_breakout_simple', 'direction': 'long', 'score': vr * 20 + (close[-1]/high_30d - 1) * 500}


ALL_TOOLS = [
    simple_buy_uptrend,
    buy_weekly_green,
    buy_btc_leading,
    buy_dip_bull_market,
    buy_breakout_simple,
]

# Test multiple trailing stop configs
CONFIGS = [
    # (trail_pct, stop_loss_pct, max_hold_hours, label)
    (0.08, 0.10, 336, "8%trail/10%sl/2wk"),
    (0.10, 0.12, 504, "10%trail/12%sl/3wk"),
    (0.12, 0.15, 672, "12%trail/15%sl/4wk"),
    (0.15, 0.18, 1008, "15%trail/18%sl/6wk"),
]


def validate(tool_func, trail_pct, sl_pct, max_hold):
    trades = []
    btc_df = load_data("BTCUSDT")
    btc_close = btc_df['close'].values
    
    for pair in PAIRS:
        df = load_data(pair)
        close = df['close'].values
        vol = df['volume'].values
        oos_start = len(close) // 2
        i = oos_start
        last_sig = -max_hold
        
        while i < len(close) - max_hold:
            if i - last_sig < 72:  # Min 3 days between entries
                i += 12
                continue
            if not is_bull(btc_close, i):
                i += 12
                continue
            
            rsi = calc_rsi(close[:i+1])
            btc_slice = btc_close[:i+1] if pair != "BTCUSDT" else None
            sig = tool_func(close[:i+1], vol[:i+1], rsi, btc_slice, i)
            
            if sig:
                exit_p, hold, reason = trailing_exit(close, i, max_hold, trail_pct, sl_pct)
                entry_p = close[i]
                raw = (exit_p - entry_p) / entry_p
                net = raw - FEES
                trades.append({
                    'pair': pair, 'net': net, 'raw': raw, 'win': net > 0,
                    'hold': hold, 'exit_reason': reason
                })
                last_sig = i
            i += 12
    
    return trades


def report(name, trades, config_label):
    if not trades:
        print(f"  {name} [{config_label}]: NO SIGNALS")
        return None
    
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    avg = np.mean([t['net'] for t in trades]) * 100
    total = sum(t['net'] for t in trades) * 100
    
    w = [t['net'] for t in trades if t['win']]
    l = [t['net'] for t in trades if not t['win']]
    gp = sum(w) if w else 0
    gl = abs(sum(l)) if l else 0.001
    pf = gp / gl
    avg_w = np.mean(w)*100 if w else 0
    avg_l = np.mean(l)*100 if l else 0
    avg_hold = np.mean([t['hold'] for t in trades])
    
    passed = n >= 30 and (wr >= 55 or pf >= 1.5)
    tag = "✅" if passed else "❌"
    
    exits = {}
    for t in trades:
        exits[t['exit_reason']] = exits.get(t['exit_reason'], 0) + 1
    
    print(f"\n  {tag} {name} [{config_label}]")
    print(f"    {n} signals | {wr:.1f}% WR | {avg:.2f}% avg | {total:.1f}% total | PF={pf:.2f}")
    print(f"    Avg win: {avg_w:.1f}% | Avg loss: {avg_l:.1f}% | Hold: {avg_hold:.0f}h ({avg_hold/24:.0f}d)")
    print(f"    Exits: {exits}")
    
    return {'tool': name, 'config': config_label, 'n': n, 'wr': wr, 'avg': avg, 
            'total': total, 'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l,
            'hold_h': avg_hold, 'passed': passed}


if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLE BULL STRATEGIES — REGIME-AWARE VALIDATION")
    print("Just buy in uptrends, trail stop, ride it. Keep it simple.")
    print("=" * 70)
    
    all_results = []
    
    for func in ALL_TOOLS:
        print(f"\n{'─'*60}")
        print(f"{func.__name__}")
        print(f"{'─'*60}")
        
        for trail, sl, max_h, label in CONFIGS:
            trades = validate(func, trail, sl, max_h)
            r = report(func.__name__, trades, label)
            if r: all_results.append(r)
    
    print(f"\n{'='*70}")
    print("WINNERS")
    print(f"{'='*70}")
    
    passed = [r for r in all_results if r['passed']]
    for r in sorted(passed, key=lambda x: x['total'], reverse=True):
        print(f"  {r['tool']} [{r['config']}]: {r['n']} sig, {r['wr']:.0f}% WR, "
              f"{r['avg']:.2f}%/trade, {r['total']:.0f}% total, PF={r['pf']:.2f}, "
              f"hold {r['hold_h']/24:.0f}d")
    
    if not passed:
        print("  None passed. Showing top 5 by total return:")
        for r in sorted(all_results, key=lambda x: x['total'], reverse=True)[:5]:
            print(f"  {r['tool']} [{r['config']}]: {r['n']} sig, {r['wr']:.0f}% WR, "
                  f"{r['avg']:.2f}%/trade, {r['total']:.0f}% total, PF={r['pf']:.2f}")
