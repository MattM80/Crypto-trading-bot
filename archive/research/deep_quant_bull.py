#!/usr/bin/env python3
"""
DEEP QUANT BULL: Comprehensive signal discovery for BULL / NEUTRAL / GREED markets.

Tests 150+ conditions across 10 categories:
1. Trend Following / Momentum
2. Pullback in Uptrend
3. Breakout / Volatility Expansion
4. Mean Reversion from Strength (short overbought)
5. Distribution Detection
6. Cross-Asset / Relative Strength
7. Mathematical / Statistical
8. Time-Based
9. Volume Profile
10. Combined / Multi-Factor
"""
import requests, numpy as np, pandas as pd, warnings, sys, os, time
from datetime import datetime, timezone
from collections import defaultdict
warnings.filterwarnings('ignore')

CC_BASE = "https://min-api.cryptocompare.com/data/v2"
FEE = 0.42

PAIRS = ["BTC","ETH","SOL","LINK","AVAX","DOT","ADA","XRP",
         "DOGE","UNI","NEAR","ATOM","AAVE","XLM","FIL","LTC"]

# ═══════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════

def dl(sym, n=2000):
    try:
        r = requests.get(f"{CC_BASE}/histohour", params={"fsym":sym,"tsym":"USD","limit":n}, timeout=30)
        d = r.json().get("Data",{}).get("Data",[])
        rows = [{'time':x['time'],'open':x['open'],'high':x['high'],'low':x['low'],
                 'close':x['close'],'volume':x.get('volumeto',0)} for x in d if x.get('close',0)>0]
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def test_edge(all_data, name, cond_fn, min_n=20):
    results = []
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        o = df['open'].values.astype(float)
        for i in range(100, len(c)-25):
            try:
                if cond_fn(c,h,l,v,o,i,sym,all_data):
                    row = {'sym':sym}
                    for hb in [1,4,8,24]:
                        if i+hb < len(c):
                            row[f'L{hb}'] = (c[i+hb]-c[i])/c[i]*100
                            row[f'S{hb}'] = (c[i]-c[i+hb])/c[i]*100
                    results.append(row)
            except:
                continue
    if len(results) < min_n:
        return None
    rdf = pd.DataFrame(results)
    n = len(rdf)
    l8 = rdf.get('L8', pd.Series([0])).mean()
    s8 = rdf.get('S8', pd.Series([0])).mean()
    l24 = rdf.get('L24', pd.Series([0])).mean()
    s24 = rdf.get('S24', pd.Series([0])).mean()
    l1 = rdf.get('L1', pd.Series([0])).mean()
    s1 = rdf.get('S1', pd.Series([0])).mean()
    l4 = rdf.get('L4', pd.Series([0])).mean()
    s4 = rdf.get('S4', pd.Series([0])).mean()
    best8 = max(l8, s8)
    direction = "LONG" if l8 >= s8 else "SHORT"
    if direction == "LONG":
        wr8 = (rdf['L8'] > FEE).mean()*100 if 'L8' in rdf else 0
    else:
        wr8 = (rdf['S8'] > FEE).mean()*100 if 'S8' in rdf else 0
    return {'name':name, 'n':n, 'L1':round(l1,3), 'S1':round(s1,3),
            'L4':round(l4,3), 'S4':round(s4,3),
            'L8':round(l8,3), 'S8':round(s8,3),
            'L24':round(l24,3), 'S24':round(s24,3),
            'best8':round(best8,3), 'dir':direction, 'wr8':round(wr8,1)}

# ═══════════════════════════════════════
# INDICATOR FUNCTIONS
# ═══════════════════════════════════════

def ema(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().values

def sma(data, period):
    return pd.Series(data).rolling(period).mean().values

def rsi(c, p=14):
    s = pd.Series(c); d = s.diff()
    g = d.where(d>0, 0.0); l = (-d).where(d<0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(50).values

def atr_func(h, l, c, p=14):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    return pd.Series(tr).rolling(p).mean().values

def bollinger(c, period=20, num_std=2):
    s = pd.Series(c)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = ((upper - lower) / mid * 100)
    return mid.values, upper.values, lower.values, bandwidth.values

def adx(high, low, close, period=14):
    plus_dm = np.maximum(np.diff(high), 0)
    minus_dm = np.maximum(-np.diff(low), 0)
    mask_plus = plus_dm < minus_dm
    mask_minus = minus_dm < plus_dm
    plus_dm[mask_plus] = 0
    minus_dm[mask_minus] = 0
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    smooth_tr = pd.Series(tr).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    smooth_plus = pd.Series(plus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    smooth_minus = pd.Series(minus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * smooth_plus / np.where(smooth_tr > 0, smooth_tr, 1)
    minus_di = 100 * smooth_minus / np.where(smooth_tr > 0, smooth_tr, 1)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    adx_val = pd.Series(dx).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    return adx_val, plus_di, minus_di

def obv(close, volume):
    """On-Balance Volume"""
    direction = np.sign(np.diff(close))
    direction = np.insert(direction, 0, 0)
    return np.cumsum(direction * volume)

def hurst_exponent(series, max_lag=20):
    lags = range(2, min(max_lag, len(series)//2))
    tau = []; rs_list = []
    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
        if len(chunks) < 2: continue
        rs_values = []
        for chunk in chunks:
            if len(chunk) < 2: continue
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk)
            if S > 0: rs_values.append(R/S)
        if rs_values:
            tau.append(lag); rs_list.append(np.mean(rs_values))
    if len(tau) < 3: return 0.5
    try:
        H = np.polyfit(np.log(tau), np.log(rs_list), 1)[0]
        return max(0, min(1, H))
    except: return 0.5

def shannon_entropy(returns, bins=20):
    if len(returns) < 10: return 3.0
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0: return 3.0
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs))

def autocorrelation(series, lag=1):
    if len(series) < lag + 10: return 0
    s = pd.Series(series)
    r = s.autocorr(lag=lag)
    return float(r) if not np.isnan(r) else 0

def vpin_proxy(close, volume, window=20):
    if len(close) < window + 1: return 0
    returns = np.diff(close) / close[:-1]
    buy_vol = np.where(returns > 0, volume[1:], 0)
    sell_vol = np.where(returns < 0, volume[1:], 0)
    recent_buy = np.sum(buy_vol[-window:])
    recent_sell = np.sum(sell_vol[-window:])
    total = recent_buy + recent_sell
    if total == 0: return 0
    return abs(recent_buy - recent_sell) / total

# ═══════════════════════════════════════
# PRE-COMPUTE CACHE (avoids redundant computation)
# ═══════════════════════════════════════

CACHE = {}

def precompute(all_data):
    """Pre-compute all indicators once per pair."""
    global CACHE
    CACHE = {}
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        o = df['open'].values.astype(float)
        
        d = {}
        # EMAs
        for p in [5, 8, 9, 12, 13, 20, 21, 26, 34, 50]:
            d[f'ema{p}'] = ema(c, p)
        # SMAs
        for p in [10, 20, 30, 50]:
            d[f'sma{p}'] = sma(c, p)
        # RSI
        d['rsi14'] = rsi(c, 14)
        d['rsi7'] = rsi(c, 7)
        # ATR
        d['atr14'] = atr_func(h, l, c, 14)
        # Bollinger
        d['bb_mid'], d['bb_upper'], d['bb_lower'], d['bb_bw'] = bollinger(c, 20, 2)
        # ADX (note: 1 element shorter than close array)
        adx_val, pdi, mdi = adx(h, l, c, 14)
        # Pad to match length
        d['adx'] = np.insert(adx_val, 0, np.nan)
        d['pdi'] = np.insert(pdi, 0, np.nan)
        d['mdi'] = np.insert(mdi, 0, np.nan)
        # OBV
        d['obv'] = obv(c, v)
        # Volume average
        d['vol_avg20'] = sma(v, 20)
        d['vol_avg5'] = sma(v, 5)
        
        CACHE[sym] = d
    return CACHE


def main():
    t0 = time.time()
    
    # Setup output to both stdout and file
    os.makedirs("/Users/lucasaust/code/Crypto-trading-bot/data", exist_ok=True)
    output_path = "/Users/lucasaust/code/Crypto-trading-bot/data/deep_quant_bull_results.txt"
    output_lines = []
    
    def out(s=""):
        print(s)
        output_lines.append(s)
    
    out("=" * 110)
    out("  DEEP QUANT BULL: Signal Discovery for Bull / Neutral / Greed Markets")
    out("  Testing 150+ conditions across 10 categories")
    out("=" * 110)
    
    out("\nDownloading 2000 hourly candles (~83 days) per pair...")
    all_data = {}
    for sym in PAIRS:
        df = dl(sym)
        if len(df) > 200:
            all_data[sym] = df
            out(f"  {sym}: {len(df)} bars")
    out(f"\nLoaded {len(all_data)} pairs")
    
    out("\nPre-computing indicators...")
    precompute(all_data)
    out("Done.\n")
    
    edges = []
    edge_count = 0
    
    def add_edge(name, cond_fn):
        nonlocal edge_count
        edge_count += 1
        r = test_edge(all_data, name, cond_fn)
        if r: edges.append(r)
        if edge_count % 20 == 0:
            out(f"  ... tested {edge_count} conditions, found {len(edges)} valid so far ...")
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 1: TREND FOLLOWING / MOMENTUM
    # ═══════════════════════════════════════════════════════════
    out("━" * 80)
    out("  CATEGORY 1: TREND FOLLOWING / MOMENTUM")
    out("━" * 80)
    
    # --- EMA Crossovers ---
    ema_pairs = [(5,13),(8,21),(9,26),(12,26),(13,34)]
    vol_filters = [None, 1.3, 1.5, 2.0]
    sma50_filter = [False, True]
    rsi_filters = [None, 40, 50, 55]
    
    for fast, slow in ema_pairs:
        for vf in [None, 1.5]:  # Reduced combos to keep runtime sane
            for sf in [False, True]:
                for rf in [None, 50]:
                    tag = f"EMA({fast},{slow})"
                    if vf: tag += f"+vol>{vf}x"
                    if sf: tag += "+>SMA50"
                    if rf: tag += f"+RSI>{rf}"
                    def mk(f_, s_, vf_, sf_, rf_):
                        ef = f'ema{f_}'; es = f'ema{s_}'
                        def fn(c,h,l,v,o,i,sym,ad):
                            d = CACHE[sym]
                            if np.isnan(d[ef][i]) or np.isnan(d[es][i]): return False
                            # Crossover: fast just crossed above slow
                            if not (d[ef][i] > d[es][i] and d[ef][i-1] <= d[es][i-1]): return False
                            if vf_:
                                va = d['vol_avg20'][i]
                                if np.isnan(va) or va <= 0 or v[i] < va * vf_: return False
                            if sf_:
                                s50 = d['sma50'][i]
                                if np.isnan(s50) or c[i] <= s50: return False
                            if rf_:
                                r = d['rsi14'][i]
                                if np.isnan(r) or r < rf_: return False
                            return True
                        return fn
                    add_edge(tag, mk(fast, slow, vf, sf, rf))
    
    # --- SMA Crossovers ---
    sma_pairs_list = [(10,30),(20,50),(10,50)]
    for fast, slow in sma_pairs_list:
        def mk(f_, s_):
            sf = f'sma{f_}'; ss = f'sma{s_}'
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                if np.isnan(d[sf][i]) or np.isnan(d[ss][i]): return False
                return d[sf][i] > d[ss][i] and d[sf][i-1] <= d[ss][i-1]
            return fn
        add_edge(f"SMA({fast},{slow}) crossover", mk(fast, slow))
    
    # --- Price above rising SMA50 ---
    for combo in ["RSI40-60", "RSI50-70", "vol_spike", "pos_mom"]:
        def mk(cmb):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                s50 = d['sma50']
                if np.isnan(s50[i]) or i < 110: return False
                # Price > SMA50 and SMA50 rising
                if c[i] <= s50[i]: return False
                if np.isnan(s50[i-10]) or s50[i] <= s50[i-10]: return False
                r = d['rsi14'][i]
                if cmb == "RSI40-60":
                    return 40 <= r <= 60
                elif cmb == "RSI50-70":
                    return 50 <= r <= 70
                elif cmb == "vol_spike":
                    va = d['vol_avg20'][i]
                    return not np.isnan(va) and va > 0 and v[i] > va * 1.5
                elif cmb == "pos_mom":
                    return c[i] > c[i-4]
                return False
            return fn
        add_edge(f"Price>rising SMA50+{combo}", mk(combo))
    
    # --- ADX-based trend ---
    for thresh in [20, 25, 30]:
        def mk(t):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                a = d['adx'][i]; p = d['pdi'][i]; m = d['mdi'][i]
                if np.isnan(a) or np.isnan(p) or np.isnan(m): return False
                return a > t and p > m
            return fn
        add_edge(f"ADX>{thresh}+DI>-DI (confirmed uptrend)", mk(thresh))
    
    # --- ADX + volume ---
    for thresh in [25, 30]:
        def mk(t):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                a = d['adx'][i]; p = d['pdi'][i]; m = d['mdi'][i]
                if np.isnan(a) or np.isnan(p) or np.isnan(m): return False
                va = d['vol_avg20'][i]
                if np.isnan(va) or va <= 0: return False
                return a > t and p > m and v[i] > va * 1.5
            return fn
        add_edge(f"ADX>{thresh}+DI>-DI+vol>1.5x", mk(thresh))
    
    # --- Consecutive green candles ---
    for n_candles in [3, 4, 5]:
        def mk(nc):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < nc: return False
                for j in range(nc):
                    if c[i-j] <= c[i-j-1]: return False
                return True
            return fn
        add_edge(f"{n_candles} consecutive green candles", mk(n_candles))
    
    # --- Higher highs + higher lows ---
    for n in [5, 10, 15]:
        def mk(nb):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < nb: return False
                for j in range(1, nb):
                    if h[i-j+1] <= h[i-j] or l[i-j+1] <= l[i-j]: return False
                return True
            return fn
        add_edge(f"HH+HL for {n} bars (strong uptrend)", mk(n))
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 2: PULLBACK IN UPTREND
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 2: PULLBACK IN UPTREND")
    out("━" * 80)
    
    # --- Price > SMA50, pullback to SMA20 ---
    def pullback_sma20(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        s50 = d['sma50'][i]; s20 = d['sma20'][i]
        if np.isnan(s50) or np.isnan(s20): return False
        return c[i] > s50 and l[i] <= s20 and c[i] > s20
    add_edge("Pullback to SMA20 in uptrend (>SMA50)", pullback_sma20)
    
    # --- Price > SMA50, RSI dips then recovers ---
    def rsi_dip_recover(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        r = d['rsi14']
        if i < 5: return False
        # RSI was 40-50 in last 3 bars, now > 50
        dipped = any(40 <= r[i-j] <= 50 for j in range(1,4))
        return dipped and r[i] > 50
    add_edge("RSI dip to 40-50 then >50 in uptrend", rsi_dip_recover)
    
    # --- Pullback to EMA21 with volume ---
    def pullback_ema21_vol(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        s50 = d['sma50'][i]; e21 = d['ema21'][i]
        if np.isnan(s50) or np.isnan(e21): return False
        va = d['vol_avg20'][i]
        if np.isnan(va) or va <= 0: return False
        return c[i] > s50 and l[i] <= e21 * 1.005 and c[i] > e21 and v[i] > va * 1.3
    add_edge("Pullback to EMA21+volume in uptrend", pullback_ema21_vol)
    
    # --- Fibonacci retracements ---
    for lookback in [24, 48, 72]:
        for fib_level in [0.236, 0.382, 0.5]:
            def mk(lb, fl):
                def fn(c,h,l,v,o,i,sym,ad):
                    d = CACHE[sym]
                    if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
                    if i < lb: return False
                    swing_high = np.max(h[i-lb:i])
                    swing_low = np.min(l[i-lb:i])
                    if swing_high <= swing_low: return False
                    fib_price = swing_high - fl * (swing_high - swing_low)
                    # Price touches fib level and bounces
                    return l[i] <= fib_price * 1.005 and c[i] > fib_price
                return fn
            add_edge(f"Fib {fib_level:.1%} pullback ({lookback}h swing)", mk(lookback, fib_level))
    
    # --- Falling wedge in uptrend ---
    def falling_wedge_uptrend(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        if i < 15: return False
        # Check for lower highs and lower lows over last 10 bars
        highs = h[i-10:i+1]
        lows = l[i-10:i+1]
        # Slope of highs vs lows
        x = np.arange(11)
        try:
            slope_h = np.polyfit(x, highs, 1)[0]
            slope_l = np.polyfit(x, lows, 1)[0]
        except: return False
        # Falling wedge: both slopes negative but lows slope is flatter (converging)
        return slope_h < 0 and slope_l < 0 and slope_l > slope_h and c[i] > c[i-1]
    add_edge("Falling wedge in uptrend (converging)", falling_wedge_uptrend)
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 3: BREAKOUT / VOLATILITY EXPANSION
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 3: BREAKOUT / VOLATILITY EXPANSION")
    out("━" * 80)
    
    # --- Bollinger Band squeeze then breakout ---
    for bw_lookback in [10, 15, 20, 30]:
        for vol_mult in [None, 1.5, 2.0]:
            tag = f"BB squeeze({bw_lookback}) breakout"
            if vol_mult: tag += f"+vol>{vol_mult}x"
            def mk(bwl, vm):
                def fn(c,h,l,v,o,i,sym,ad):
                    d = CACHE[sym]
                    bw = d['bb_bw']
                    up = d['bb_upper'][i]
                    if np.isnan(bw[i]) or np.isnan(up) or i < bwl + 5: return False
                    # Bandwidth at minimum over lookback
                    recent_bw = bw[i-bwl:i]
                    recent_bw = recent_bw[~np.isnan(recent_bw)]
                    if len(recent_bw) < bwl // 2: return False
                    if bw[i-1] > np.min(recent_bw) * 1.1: return False  # Was near min
                    # Price breaks upper band
                    if c[i] <= up: return False
                    if vm:
                        va = d['vol_avg20'][i]
                        if np.isnan(va) or va <= 0 or v[i] < va * vm: return False
                    return True
                return fn
            add_edge(tag, mk(bw_lookback, vol_mult))
    
    # --- ATR contraction then expansion ---
    def atr_contract_expand(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        a = d['atr14']
        if i < 25 or np.isnan(a[i]): return False
        atr_history = a[i-20:i]
        atr_history = atr_history[~np.isnan(atr_history)]
        if len(atr_history) < 15: return False
        atr_high = np.max(atr_history)
        if atr_high <= 0: return False
        # ATR was < 50% of 20-period high, now expanding
        was_low = np.min(atr_history[-5:]) < atr_high * 0.5
        expanding = a[i] > np.mean(atr_history[-3:]) * 1.3
        return was_low and expanding and c[i] > c[i-1]
    add_edge("ATR contraction→expansion (breakout)", atr_contract_expand)
    
    # --- N-bar high breakout ---
    for n_bars in [10, 20, 30, 50]:
        for vm in [None, 1.5, 2.0]:
            tag = f"Break {n_bars}-bar high"
            if vm: tag += f"+vol>{vm}x"
            def mk(nb, vm_):
                def fn(c,h,l,v,o,i,sym,ad):
                    if i < nb + 1: return False
                    prev_high = np.max(h[i-nb:i])
                    if c[i] <= prev_high: return False
                    if vm_:
                        d = CACHE[sym]
                        va = d['vol_avg20'][i]
                        if np.isnan(va) or va <= 0 or v[i] < va * vm_: return False
                    return True
                return fn
            add_edge(tag, mk(n_bars, vm))
    
    # --- Keltner Channel breakout ---
    def keltner_breakout(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        e20 = d['ema20'][i]; a = d['atr14'][i]
        if np.isnan(e20) or np.isnan(a): return False
        return c[i] > e20 + 2 * a
    add_edge("Keltner breakout: price > EMA20+2*ATR", keltner_breakout)
    
    # --- Range-bound then breakout ---
    for n_range in [12, 24, 48]:
        def mk(nr):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < nr + 2: return False
                range_high = np.max(h[i-nr:i])
                range_low = np.min(l[i-nr:i])
                range_mid = (range_high + range_low) / 2
                if range_mid <= 0: return False
                range_pct = (range_high - range_low) / range_mid * 100
                return range_pct < 3 and c[i] > range_high
            return fn
        add_edge(f"Range({n_range}h)<3% then breakout", mk(n_range))
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 4: MEAN REVERSION FROM STRENGTH (SHORT OVERBOUGHT)
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 4: MEAN REVERSION FROM STRENGTH (SHORT OVERBOUGHT)")
    out("━" * 80)
    
    # --- RSI > 80 after big pump ---
    for pct in [5, 8, 10]:
        for lookback in [8, 12, 24]:
            def mk(p, lb):
                def fn(c,h,l,v,o,i,sym,ad):
                    d = CACHE[sym]
                    r = d['rsi14'][i]
                    if np.isnan(r) or i < lb: return False
                    ret = (c[i] - c[i-lb]) / c[i-lb] * 100
                    return r > 80 and ret > p
                return fn
            add_edge(f"RSI>80+{pct}% pump in {lookback}h (overbought short)", mk(pct, lookback))
    
    # --- Price > upper BB by N% ---
    for pct in [1, 2, 3]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                up = d['bb_upper'][i]
                if np.isnan(up) or up <= 0: return False
                return c[i] > up * (1 + p/100)
            return fn
        add_edge(f"Price > upper BB by {pct}% (overbought)", mk(pct))
    
    # --- Price > SMA50 by X% ---
    for pct in [8, 10, 12, 15]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                s50 = d['sma50'][i]
                if np.isnan(s50) or s50 <= 0: return False
                return (c[i] - s50) / s50 * 100 > p
            return fn
        add_edge(f"Price > SMA50 by {pct}% (extended)", mk(pct))
    
    # --- Consecutive green then red ---
    for n_green in [5, 7, 9]:
        def mk(ng):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < ng + 1: return False
                # N green candles followed by a red
                if c[i] >= o[i]: return False  # Current must be red
                for j in range(1, ng + 1):
                    if c[i-j] <= o[i-j]: return False  # Must be green
                return True
            return fn
        add_edge(f"{n_green} green then red (exhaustion)", mk(n_green))
    
    # --- Volume divergence: new high on declining volume ---
    def vol_divergence_bearish(c,h,l,v,o,i,sym,ad):
        if i < 20: return False
        # Price at 20-bar high
        if c[i] < np.max(c[i-20:i]) * 0.998: return False
        # Volume declining
        vol_recent = np.mean(v[i-5:i+1])
        vol_prior = np.mean(v[i-15:i-5])
        if vol_prior <= 0: return False
        return vol_recent < vol_prior * 0.7
    add_edge("New high on declining volume (bearish div)", vol_divergence_bearish)
    
    # --- RSI bearish divergence ---
    def rsi_bearish_div(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        r = d['rsi14']
        if i < 28 or np.isnan(r[i]) or np.isnan(r[i-14]): return False
        # Price: higher high
        recent_high = np.max(c[i-14:i+1])
        prior_high = np.max(c[i-28:i-14])
        # RSI: lower high
        recent_rsi_high = np.max(r[i-14:i+1])
        prior_rsi_high = np.max(r[i-28:i-14])
        return recent_high > prior_high and recent_rsi_high < prior_rsi_high and r[i] > 60
    add_edge("RSI bearish divergence (higher price, lower RSI)", rsi_bearish_div)
    
    # --- Parabolic acceleration ---
    for n in [3, 4, 5]:
        def mk(nc):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < nc + 1: return False
                # Each candle bigger than the last
                for j in range(nc):
                    curr_range = abs(c[i-j] - o[i-j])
                    prev_range = abs(c[i-j-1] - o[i-j-1])
                    if prev_range <= 0: return False
                    if curr_range <= prev_range: return False
                    if c[i-j] <= o[i-j]: return False  # Must be green
                return True
            return fn
        add_edge(f"Parabolic {n}-bar acceleration (expanding green)", mk(n))
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 5: DISTRIBUTION DETECTION
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 5: DISTRIBUTION DETECTION")
    out("━" * 80)
    
    # --- Lower highs + declining volume ---
    def lower_highs_low_vol(c,h,l,v,o,i,sym,ad):
        if i < 15: return False
        # 3 consecutive lower highs
        if not (h[i] < h[i-3] and h[i-3] < h[i-6]): return False
        vol_recent = np.mean(v[i-5:i+1])
        vol_prior = np.mean(v[i-15:i-5])
        if vol_prior <= 0: return False
        return vol_recent < vol_prior * 0.7
    add_edge("Lower highs + declining volume (distribution)", lower_highs_low_vol)
    
    # --- Range narrowing + volume dropping ---
    def range_narrow_vol_drop(c,h,l,v,o,i,sym,ad):
        if i < 20: return False
        range_recent = np.mean(h[i-5:i+1] - l[i-5:i+1])
        range_prior = np.mean(h[i-15:i-5] - l[i-15:i-5])
        vol_recent = np.mean(v[i-5:i+1])
        vol_prior = np.mean(v[i-15:i-5])
        if range_prior <= 0 or vol_prior <= 0: return False
        return range_recent < range_prior * 0.6 and vol_recent < vol_prior * 0.6
    add_edge("Range narrowing + volume dropping (quiet before storm)", range_narrow_vol_drop)
    
    # --- Large red candle on huge volume after uptrend ---
    for vol_mult in [3, 5, 8]:
        def mk(vm):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                if np.isnan(d['sma50'][i]) or i < 50: return False
                # Was in uptrend
                if c[i-5] <= d['sma50'][i-5]: return False
                # Big red candle
                candle_size = (o[i] - c[i]) / o[i] * 100
                if candle_size < 2: return False  # At least 2% red
                va = d['vol_avg20'][i]
                if np.isnan(va) or va <= 0: return False
                return v[i] > va * vm
            return fn
        add_edge(f"Big red candle + {vol_mult}x volume after uptrend", mk(vol_mult))
    
    # --- Skewness shift ---
    def neg_skew_developing(c,h,l,v,o,i,sym,ad):
        if i < 60: return False
        ret_recent = np.diff(c[i-30:i+1]) / c[i-30:i]
        ret_prior = np.diff(c[i-60:i-29]) / c[i-60:i-29]
        if len(ret_recent) < 20 or len(ret_prior) < 20: return False
        skew_recent = float(pd.Series(ret_recent).skew())
        skew_prior = float(pd.Series(ret_prior).skew())
        if np.isnan(skew_recent) or np.isnan(skew_prior): return False
        return skew_prior > 0.3 and skew_recent < -0.3
    add_edge("Skew shift: positive→negative (distribution)", neg_skew_developing)
    
    # --- Kurtosis spike ---
    def kurtosis_spike(c,h,l,v,o,i,sym,ad):
        if i < 60: return False
        ret_recent = np.diff(c[i-30:i+1]) / c[i-30:i]
        ret_prior = np.diff(c[i-60:i-29]) / c[i-60:i-29]
        if len(ret_recent) < 20 or len(ret_prior) < 20: return False
        kurt_recent = float(pd.Series(ret_recent).kurtosis())
        kurt_prior = float(pd.Series(ret_prior).kurtosis())
        if np.isnan(kurt_recent) or np.isnan(kurt_prior): return False
        return kurt_recent > kurt_prior + 2 and kurt_recent > 3
    add_edge("Kurtosis spike (fat tails developing)", kurtosis_spike)
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 6: CROSS-ASSET / RELATIVE STRENGTH
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 6: CROSS-ASSET / RELATIVE STRENGTH")
    out("━" * 80)
    
    # --- BTC leads alt (buy alt after BTC pump) ---
    for lag in [1, 2, 3, 4]:
        for btc_thresh in [1.0, 1.5, 2.0]:
            def mk(lg, bt):
                def fn(c,h,l,v,o,i,sym,ad):
                    if sym == "BTC" or "BTC" not in ad: return False
                    btc = ad["BTC"]['close'].values
                    if i >= len(btc) or i < lg + 1: return False
                    btc_ret = (btc[i] - btc[i-lg]) / btc[i-lg] * 100
                    alt_ret = (c[i] - c[i-lg]) / c[i-lg] * 100
                    return btc_ret > bt and alt_ret < bt * 0.3
                return fn
            add_edge(f"BTC+{btc_thresh}% in {lag}h, alt lagging (buy alt)", mk(lag, btc_thresh))
    
    # --- Alt outperforms BTC → short alt ---
    for pct in [3, 5, 8]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                if sym == "BTC" or "BTC" not in ad: return False
                btc = ad["BTC"]['close'].values
                if i >= len(btc) or i < 24: return False
                btc_ret = (btc[i] - btc[i-24]) / btc[i-24] * 100
                alt_ret = (c[i] - c[i-24]) / c[i-24] * 100
                return alt_ret - btc_ret > p
            return fn
        add_edge(f"Alt outperforms BTC by {pct}% 24h (short alt)", mk(pct))
    
    # --- Strongest alt → momentum continuation ---
    def strongest_alt_24h(c,h,l,v,o,i,sym,ad):
        if i < 24: return False
        rets = {}
        for s, d in ad.items():
            cl = d['close'].values
            if i < len(cl) and i >= 24:
                rets[s] = (cl[i] - cl[i-24]) / cl[i-24] * 100
        if len(rets) < 5: return False
        ranked = sorted(rets.items(), key=lambda x: x[1], reverse=True)
        return sym == ranked[0][0]
    add_edge("Strongest alt 24h (momentum continue)", strongest_alt_24h)
    
    # --- Weakest alt still above SMA50 → pullback buy ---
    def weakest_above_sma50(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        if i < 24: return False
        rets = {}
        for s, dd in ad.items():
            cl = dd['close'].values
            s50 = CACHE[s]['sma50']
            if i < len(cl) and i >= 24 and not np.isnan(s50[i]) and cl[i] > s50[i]:
                rets[s] = (cl[i] - cl[i-24]) / cl[i-24] * 100
        if len(rets) < 5: return False
        ranked = sorted(rets.items(), key=lambda x: x[1])
        return sym == ranked[0][0]
    add_edge("Weakest alt above SMA50 (pullback buy)", weakest_above_sma50)
    
    # --- BTC dominance proxy: when BTC flat and alts pump ---
    def btc_flat_alt_pump(c,h,l,v,o,i,sym,ad):
        if sym == "BTC" or "BTC" not in ad: return False
        btc = ad["BTC"]['close'].values
        if i >= len(btc) or i < 8: return False
        btc_ret = abs((btc[i] - btc[i-8]) / btc[i-8] * 100)
        alt_ret = (c[i] - c[i-8]) / c[i-8] * 100
        return btc_ret < 0.5 and alt_ret > 2  # BTC flat, alt pumping (alt season signal)
    add_edge("BTC flat + alt pumping (alt season)", btc_flat_alt_pump)
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 7: MATHEMATICAL / STATISTICAL
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 7: MATHEMATICAL / STATISTICAL")
    out("━" * 80)
    
    # --- Hurst + positive momentum ---
    for h_thresh in [0.55, 0.6, 0.65]:
        def mk(ht):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < 100: return False
                rets = np.diff(c[i-100:i+1]) / c[i-100:i]
                H = hurst_exponent(rets[-50:])
                mom = (c[i] - c[i-8]) / c[i-8] * 100
                return H > ht and mom > 1
            return fn
        add_edge(f"Hurst>{h_thresh}+positive mom (trend follow)", mk(h_thresh))
    
    # --- Positive autocorrelation + momentum ---
    for ac_thresh in [0.1, 0.15, 0.2]:
        def mk(at):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < 50: return False
                rets = np.diff(c[i-50:i+1]) / c[i-50:i]
                ac = autocorrelation(rets, 1)
                mom = (c[i] - c[i-4]) / c[i-4] * 100
                return ac > at and mom > 0.5
            return fn
        add_edge(f"Autocorr>{ac_thresh}+positive mom (persistent)", mk(ac_thresh))
    
    # --- Low entropy + above SMA50 ---
    for ent_thresh in [2.0, 2.5, 3.0]:
        def mk(et):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
                if i < 60: return False
                rets = np.diff(c[i-50:i+1]) / c[i-50:i]
                ent = shannon_entropy(rets)
                return ent < et
            return fn
        add_edge(f"Entropy<{ent_thresh}+above SMA50 (predictable bull)", mk(ent_thresh))
    
    # --- VPIN high + green candle ---
    for vpin_thresh in [0.5, 0.6, 0.7]:
        def mk(vt):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < 30: return False
                vp = vpin_proxy(c[i-30:i+1], v[i-30:i+1])
                return vp > vt and c[i] > o[i]
            return fn
        add_edge(f"VPIN>{vpin_thresh}+green (informed buying)", mk(vpin_thresh))
    
    # --- Positive skew developing ---
    def pos_skew_rising(c,h,l,v,o,i,sym,ad):
        if i < 60: return False
        rets = np.diff(c[i-30:i+1]) / c[i-30:i]
        if len(rets) < 20: return False
        skew = float(pd.Series(rets).skew())
        if np.isnan(skew): return False
        mom = (c[i] - c[i-10]) / c[i-10] * 100
        return skew > 0.5 and mom > 0
    add_edge("Positive skew + rising price (bullish distribution)", pos_skew_rising)
    
    # --- Hurst mean-reverting + dip in uptrend ---
    def hurst_mr_dip_uptrend(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        if i < 100: return False
        rets = np.diff(c[i-100:i+1]) / c[i-100:i]
        H = hurst_exponent(rets[-50:])
        dip = (c[i] - c[i-4]) / c[i-4] * 100
        return H < 0.45 and dip < -1.5
    add_edge("Hurst<0.45 + dip in uptrend (mean revert buy)", hurst_mr_dip_uptrend)
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 8: TIME-BASED
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 8: TIME-BASED")
    out("━" * 80)
    
    # --- Hour of day + uptrend ---
    for hour in [0, 4, 8, 9, 13, 14, 15, 16, 21]:  # Key session opens
        def mk(hr):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
                ts = ad[sym]['time'].values[i]
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).hour == hr
            return fn
        session = {0:"Asia open", 4:"Asia mid", 8:"EU open", 9:"EU early",
                   13:"US pre", 14:"US open", 15:"US early", 16:"US mid", 21:"Late US"}
        add_edge(f"Hour={hour} ({session.get(hour,'')}) in uptrend", mk(hour))
    
    # --- Day of week in uptrend ---
    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for dow in range(7):
        def mk(d_):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
                ts = ad[sym]['time'].values[i]
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).weekday() == d_
            return fn
        add_edge(f"Day={day_names[dow]} in uptrend", mk(dow))
    
    # --- Weekend vs weekday in uptrend ---
    def weekend_uptrend(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        ts = ad[sym]['time'].values[i]
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).weekday() >= 5
    add_edge("Weekend in uptrend", weekend_uptrend)
    
    def weekday_uptrend(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        ts = ad[sym]['time'].values[i]
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).weekday() < 5
    add_edge("Weekday in uptrend", weekday_uptrend)
    
    # --- Month start/end ---
    def month_start_uptrend(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        ts = ad[sym]['time'].values[i]
        day = datetime.fromtimestamp(int(ts), tz=timezone.utc).day
        return day <= 3
    add_edge("Month start (day 1-3) in uptrend", month_start_uptrend)
    
    def month_end_uptrend(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        ts = ad[sym]['time'].values[i]
        day = datetime.fromtimestamp(int(ts), tz=timezone.utc).day
        return day >= 28
    add_edge("Month end (day 28+) in uptrend", month_end_uptrend)
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 9: VOLUME PROFILE
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 9: VOLUME PROFILE")
    out("━" * 80)
    
    # --- Volume climax on green ---
    for vm in [3, 5, 8]:
        def mk(mult):
            def fn(c,h,l,v,o,i,sym,ad):
                d = CACHE[sym]
                va = d['vol_avg20'][i]
                if np.isnan(va) or va <= 0: return False
                return v[i] > va * mult and c[i] > o[i]
            return fn
        add_edge(f"Green candle + {vm}x volume (institutional buy)", mk(vm))
    
    # --- OBV making new highs before price ---
    def obv_leads_price(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        ob = d['obv']
        if i < 20: return False
        obv_high = np.max(ob[i-20:i])
        price_high = np.max(c[i-20:i])
        return ob[i] > obv_high and c[i] < price_high * 0.995
    add_edge("OBV new high before price (accumulation)", obv_leads_price)
    
    # --- Volume trend: short > long avg in uptrend ---
    def vol_trend_bullish(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        v5 = d['vol_avg5'][i]; v20 = d['vol_avg20'][i]
        if np.isnan(v5) or np.isnan(v20) or v20 <= 0: return False
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        return v5 > v20 * 1.3
    add_edge("Vol trend: 5bar>1.3x 20bar in uptrend", vol_trend_bullish)
    
    # --- Price up on decreasing volume → weakening ---
    def price_up_vol_down(c,h,l,v,o,i,sym,ad):
        if i < 15: return False
        # Price up over 10 bars
        if c[i] <= c[i-10]: return False
        # Volume declining
        vol_recent = np.mean(v[i-5:i+1])
        vol_prior = np.mean(v[i-10:i-5])
        if vol_prior <= 0: return False
        return vol_recent < vol_prior * 0.6
    add_edge("Price up + volume down (weakening short)", price_up_vol_down)
    
    # --- Buy/sell volume ratio ---
    for ratio in [2.0, 2.5, 3.0]:
        def mk(r):
            def fn(c,h,l,v,o,i,sym,ad):
                if i < 10: return False
                buy_v = sum(v[j] for j in range(i-10, i+1) if c[j] > o[j])
                sell_v = sum(v[j] for j in range(i-10, i+1) if c[j] <= o[j])
                if sell_v <= 0: return False
                return buy_v / sell_v > r
            return fn
        add_edge(f"Buy vol > {ratio}x sell vol (10 bars)", mk(ratio))
    
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 10: COMBINED / MULTI-FACTOR
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 80)
    out("  CATEGORY 10: COMBINED / MULTI-FACTOR")
    out("━" * 80)
    
    # --- Trend + Momentum + Volume ---
    def trifecta_long(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        # Trend: above SMA50
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        # Momentum: RSI 50-70
        r = d['rsi14'][i]
        if np.isnan(r) or r < 50 or r > 70: return False
        # Volume: above average
        va = d['vol_avg20'][i]
        if np.isnan(va) or va <= 0 or v[i] < va * 1.3: return False
        return True
    add_edge("TRIFECTA: >SMA50+RSI50-70+vol>1.3x", trifecta_long)
    
    # --- Trifecta v2: ADX + EMA crossover + volume ---
    def trifecta_v2(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        a = d['adx'][i]; p = d['pdi'][i]; m = d['mdi'][i]
        if np.isnan(a) or a < 25 or p <= m: return False
        e12 = d['ema12'][i]; e26 = d['ema26'][i]
        if np.isnan(e12) or e12 <= e26: return False
        va = d['vol_avg20'][i]
        if np.isnan(va) or va <= 0 or v[i] < va * 1.3: return False
        return True
    add_edge("TRIFECTA v2: ADX>25+EMA12>26+vol", trifecta_v2)
    
    # --- Pullback + Hurst mean-reverting + RSI dip ---
    def pullback_hurst_rsi(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        r = d['rsi14'][i]
        if np.isnan(r) or r < 35 or r > 50: return False
        if i < 100: return False
        rets = np.diff(c[i-100:i+1]) / c[i-100:i]
        H = hurst_exponent(rets[-50:])
        return H < 0.45
    add_edge("Pullback + Hurst<0.45 + RSI 35-50 (MR buy)", pullback_hurst_rsi)
    
    # --- Breakout + VPIN ---
    def breakout_vpin(c,h,l,v,o,i,sym,ad):
        if i < 30: return False
        prev_high = np.max(h[i-20:i])
        if c[i] <= prev_high: return False
        vp = vpin_proxy(c[i-30:i+1], v[i-30:i+1])
        return vp > 0.5
    add_edge("20-bar breakout + VPIN>0.5 (informed breakout)", breakout_vpin)
    
    # --- Cross-asset + Trend ---
    def strongest_in_uptrend(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        if np.isnan(d['sma50'][i]) or c[i] <= d['sma50'][i]: return False
        a = d['adx'][i]
        if np.isnan(a) or a < 20: return False
        if i < 24: return False
        rets = {}
        for s, dd in ad.items():
            cl = dd['close'].values
            if i < len(cl) and i >= 24:
                rets[s] = (cl[i] - cl[i-24]) / cl[i-24] * 100
        if len(rets) < 5: return False
        ranked = sorted(rets.items(), key=lambda x: x[1], reverse=True)
        return sym in [ranked[0][0], ranked[1][0], ranked[2][0]]  # Top 3
    add_edge("Top 3 strongest + uptrend + ADX>20", strongest_in_uptrend)
    
    # --- EMA stack: 5>13>26>50 (perfect trend alignment) ---
    def ema_stack(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        e5 = d['ema5'][i]; e13 = d['ema13'][i]; e26 = d['ema26'][i]; s50 = d['sma50'][i]
        if any(np.isnan(x) for x in [e5, e13, e26, s50]): return False
        return e5 > e13 > e26 > s50
    add_edge("EMA stack: 5>13>26>SMA50 (perfect alignment)", ema_stack)
    
    # --- EMA stack + volume + RSI ---
    def ema_stack_vol_rsi(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        e5 = d['ema5'][i]; e13 = d['ema13'][i]; e26 = d['ema26'][i]; s50 = d['sma50'][i]
        if any(np.isnan(x) for x in [e5, e13, e26, s50]): return False
        if not (e5 > e13 > e26 > s50): return False
        r = d['rsi14'][i]
        if np.isnan(r) or r < 50 or r > 70: return False
        va = d['vol_avg20'][i]
        if np.isnan(va) or va <= 0 or v[i] < va * 1.2: return False
        return True
    add_edge("EMA stack + RSI50-70 + vol>1.2x (full confirm)", ema_stack_vol_rsi)
    
    # --- Multiple MA pullback ---
    def multi_ma_pullback(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        s50 = d['sma50'][i]; e21 = d['ema21'][i]; s20 = d['sma20'][i]
        if any(np.isnan(x) for x in [s50, e21, s20]): return False
        if c[i] <= s50: return False
        # Price near both EMA21 and SMA20 (double support)
        near_ema21 = abs(l[i] - e21) / e21 < 0.01
        near_sma20 = abs(l[i] - s20) / s20 < 0.01
        return (near_ema21 or near_sma20) and c[i] > max(e21, s20)
    add_edge("Multi-MA pullback: bounce off EMA21/SMA20 >SMA50", multi_ma_pullback)
    
    # --- Momentum + Low entropy + VPIN ---
    def mom_entropy_vpin(c,h,l,v,o,i,sym,ad):
        if i < 60: return False
        rets = np.diff(c[i-50:i+1]) / c[i-50:i]
        ent = shannon_entropy(rets)
        if ent > 2.5: return False
        vp = vpin_proxy(c[i-30:i+1], v[i-30:i+1])
        if vp < 0.5: return False
        mom = (c[i] - c[i-8]) / c[i-8] * 100
        return mom > 1 and c[i] > o[i]
    add_edge("Low entropy + VPIN>0.5 + momentum (high conviction)", mom_entropy_vpin)
    
    # --- ADX trend + pullback RSI ---
    def adx_pullback(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        a = d['adx'][i]; p = d['pdi'][i]; m = d['mdi'][i]
        if np.isnan(a) or a < 25 or p <= m: return False
        r = d['rsi14'][i]
        return not np.isnan(r) and 40 <= r <= 55
    add_edge("ADX>25 uptrend + RSI pullback to 40-55", adx_pullback)
    
    # --- BB squeeze + ADX rising ---
    def bb_squeeze_adx(c,h,l,v,o,i,sym,ad):
        d = CACHE[sym]
        bw = d['bb_bw']
        a = d['adx']
        if np.isnan(bw[i]) or np.isnan(a[i]) or i < 25: return False
        bw_hist = bw[i-20:i]
        bw_hist = bw_hist[~np.isnan(bw_hist)]
        if len(bw_hist) < 10: return False
        # BB squeezed
        if bw[i] > np.percentile(bw_hist, 25): return False
        # ADX rising (starting to trend)
        if np.isnan(a[i-5]): return False
        return a[i] > a[i-5] and c[i] > d['bb_mid'][i]
    add_edge("BB squeeze + ADX rising + above mid (trend ignition)", bb_squeeze_adx)
    
    # ═══════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════
    
    edges = [e for e in edges if e is not None]
    edges.sort(key=lambda e: e['best8'], reverse=True)
    
    out(f"\n{'='*110}")
    out(f"  TESTED {edge_count} CONDITIONS | {len(edges)} HAD 20+ TRADES")
    out(f"{'='*110}")
    
    header = f"  {'Pattern':<62s} {'N':>5s} {'L1h':>6s} {'L4h':>6s} {'L8h':>6s} {'S8h':>6s} {'L24h':>7s} {'WR8':>5s} {'Dir':>5s} {'Net8':>6s}"
    out(header)
    out("  " + "-" * 108)
    
    profitable = []
    for e in edges:
        best = e['best8']
        net = best - FEE
        color = "\033[92m" if best > FEE else ("\033[93m" if best > 0 else "\033[91m")
        R = "\033[0m"
        line = (f"  {e['name']:<62s} {e['n']:>5d} "
                f"{color}{e['L1']:>+5.2f}% {e['L4']:>+5.2f}% {e['L8']:>+5.2f}% {e['S8']:>+5.2f}% {e['L24']:>+6.2f}%{R} "
                f"{e['wr8']:>4.0f}% {e['dir']:>5s} {net:>+5.2f}%")
        print(line)
        # Plain version for file
        plain = (f"  {e['name']:<62s} {e['n']:>5d} "
                 f"{e['L1']:>+5.2f}% {e['L4']:>+5.2f}% {e['L8']:>+5.2f}% {e['S8']:>+5.2f}% {e['L24']:>+6.2f}% "
                 f"{e['wr8']:>4.0f}% {e['dir']:>5s} {net:>+5.2f}%")
        output_lines.append(plain)
        if best > FEE and e['n'] >= 20:
            profitable.append(e)
    
    out(f"\n{'='*110}")
    out(f"  PROFITABLE EDGES: {len(profitable)} (beat {FEE}% fee)")
    out(f"{'='*110}")
    for e in profitable:
        net = e['best8'] - FEE
        out(f"  ✅ {e['name']}")
        out(f"     N={e['n']} | {e['dir']} | 8h={e['best8']:+.2f}% net={net:+.2f}% | 24h L={e['L24']:+.2f}% S={e['S24']:+.2f}% | WR={e['wr8']:.0f}%")
        out("")
    
    # --- TOP 20 SUMMARY ---
    out(f"\n{'='*110}")
    out(f"  ★ TOP 20 BULL MARKET EDGES ★")
    out(f"{'='*110}")
    for idx, e in enumerate(profitable[:20], 1):
        net = e['best8'] - FEE
        out(f"  {idx:>2d}. {e['name']}")
        out(f"      Direction: {e['dir']} | Trades: {e['n']} | WinRate: {e['wr8']:.1f}%")
        out(f"      Returns → 1h: {e['L1']:+.2f}% | 4h: {e['L4']:+.2f}% | 8h: {e['best8']:+.2f}% (net {net:+.2f}%) | 24h: {e['L24']:+.2f}%")
        out("")
    
    elapsed = time.time() - t0
    out(f"\n  Completed in {elapsed:.0f}s")
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write("\n".join(output_lines))
    out(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
