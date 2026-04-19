#!/usr/bin/env python3
"""
DEEP QUANT BULL COMBOS: Combination sweep — base signals + mathematical confirmations.

Tests 150+ combos: each top bull signal combined with Hurst, entropy, autocorrelation,
VPIN, skewness, kurtosis, volume metrics, structural breaks, Poisson anomalies.
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
# DATA
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

def test_edge(all_data, name, cond_fn, min_n=15):
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
    best8 = max(l8, s8)
    direction = "LONG" if l8 >= s8 else "SHORT"
    if direction == "LONG":
        wr8 = (rdf['L8'] > FEE).mean()*100 if 'L8' in rdf else 0
    else:
        wr8 = (rdf['S8'] > FEE).mean()*100 if 'S8' in rdf else 0
    return {'name':name, 'n':n, 'L8':round(l8,3), 'S8':round(s8,3),
            'L24':round(l24,3), 'S24':round(s24,3),
            'best8':round(best8,3), 'dir':direction, 'wr8':round(wr8,1),
            'net8': round(best8 - FEE, 3)}

# ═══════════════════════════════════════
# MATHEMATICAL TOOLS (from deep_quant_v3.py)
# ═══════════════════════════════════════

def hurst_exponent(series, max_lag=20):
    lags = range(2, min(max_lag, len(series)//2))
    tau = []; rs = []
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
            tau.append(lag); rs.append(np.mean(rs_values))
    if len(tau) < 3: return 0.5
    try:
        H = np.polyfit(np.log(tau), np.log(rs), 1)[0]
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

def detect_structural_break(series, window=50):
    if len(series) < window * 2: return 0
    first_half = series[-window*2:-window]
    second_half = series[-window:]
    mean1 = np.mean(first_half); std1 = np.std(first_half)
    mean2 = np.mean(second_half)
    if std1 == 0: return 0
    return abs(mean2 - mean1) / std1

def poisson_volume_anomaly(volume, window=48):
    if len(volume) < window + 10: return 0, 0
    baseline = np.mean(volume[-window-10:-10])
    if baseline <= 0: return 0, 0
    recent = np.mean(volume[-5:])
    z = (recent - baseline) / np.sqrt(max(baseline, 1))
    return z, recent / baseline

# ═══════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════

def sma(data, period):
    return pd.Series(data).rolling(period).mean().values

def rsi(c, p=14):
    s = pd.Series(c); d = s.diff()
    g = d.where(d>0, 0.0); l = (-d).where(d<0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(50).values

def bollinger(c, period=20, num_std=2):
    s = pd.Series(c)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = ((upper - lower) / mid * 100)
    return mid.values, upper.values, lower.values, bandwidth.values

def obv(close, volume):
    direction = np.sign(np.diff(close))
    direction = np.insert(direction, 0, 0)
    return np.cumsum(direction * volume)

# ═══════════════════════════════════════
# PRE-COMPUTE MATH FEATURES PER BAR
# ═══════════════════════════════════════

MATH = {}  # MATH[sym] = dict of arrays

def precompute_math(all_data):
    """Pre-compute all mathematical features for each bar to avoid redundant computation."""
    global MATH
    MATH = {}
    
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        v = df['volume'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        n = len(c)
        
        # Pre-allocate arrays
        m = {
            'hurst': np.full(n, np.nan),
            'entropy': np.full(n, np.nan),
            'ac1': np.full(n, np.nan),
            'vpin': np.full(n, np.nan),
            'skew': np.full(n, np.nan),
            'kurt': np.full(n, np.nan),
            'vol_ratio': np.full(n, np.nan),
            'vol_trend': np.full(n, np.nan),
            'break_mag': np.full(n, np.nan),
            'pois_z': np.full(n, np.nan),
            'pois_ratio': np.full(n, np.nan),
            # Standard indicators
            'sma50': sma(c, 50),
            'rsi14': rsi(c, 14),
            'obv': obv(c, v),
            'vol_avg20': sma(v, 20),
        }
        
        # Bollinger
        m['bb_mid'], m['bb_upper'], m['bb_lower'], m['bb_bw'] = bollinger(c, 20, 2)
        
        # Compute rolling math features
        for i in range(100, n):
            # Returns windows
            returns_50 = np.diff(c[i-50:i+1]) / c[i-50:i]
            returns_30 = np.diff(c[i-30:i+1]) / c[i-30:i]
            
            # Hurst
            m['hurst'][i] = hurst_exponent(returns_50)
            
            # Entropy
            m['entropy'][i] = shannon_entropy(returns_30)
            
            # Autocorrelation
            m['ac1'][i] = autocorrelation(returns_50, lag=1)
            
            # VPIN
            m['vpin'][i] = vpin_proxy(c[i-30:i+1], v[i-30:i+1])
            
            # Skewness
            sk = pd.Series(returns_50).skew()
            m['skew'][i] = float(sk) if not np.isnan(sk) else 0
            
            # Kurtosis
            kt = pd.Series(returns_50).kurtosis()
            m['kurt'][i] = float(kt) if not np.isnan(kt) else 0
            
            # Volume metrics
            va20 = np.mean(v[i-20:i])
            m['vol_ratio'][i] = v[i] / va20 if va20 > 0 else 1
            
            va_early = np.mean(v[i-20:i-5])
            va_late = np.mean(v[i-5:i+1])
            m['vol_trend'][i] = va_late / va_early if va_early > 0 else 1
            
            # Structural break
            if i >= 100:
                m['break_mag'][i] = detect_structural_break(c[:i+1], 50)
            
            # Poisson volume
            if i >= 60:
                pz, pr = poisson_volume_anomaly(v[:i+1])
                m['pois_z'][i] = pz
                m['pois_ratio'][i] = pr
        
        MATH[sym] = m
    
    return MATH


# ═══════════════════════════════════════
# BASE SIGNAL HELPERS
# ═══════════════════════════════════════

def base_sma50_ext(c, o, i, sym, pct):
    """Price > SMA50 by pct%"""
    m = MATH[sym]
    s50 = m['sma50'][i]
    if np.isnan(s50) or s50 <= 0: return False
    return (c[i] - s50) / s50 * 100 > pct

def base_alt_outperform_btc(c, o, i, sym, all_data, pct, hours=24):
    """Alt outperforms BTC by pct% in hours"""
    if sym == "BTC" or "BTC" not in all_data: return False
    btc = all_data["BTC"]['close'].values
    if i >= len(btc) or i < hours: return False
    btc_ret = (btc[i] - btc[i-hours]) / btc[i-hours] * 100
    alt_ret = (c[i] - c[i-hours]) / c[i-hours] * 100
    return alt_ret - btc_ret > pct

def base_rsi_pump(c, o, i, sym, rsi_thresh=80, pump_pct=8, lookback=12):
    """RSI > thresh + pump > pct% in lookback hours"""
    m = MATH[sym]
    r = m['rsi14'][i]
    if np.isnan(r) or i < lookback: return False
    ret = (c[i] - c[i-lookback]) / c[i-lookback] * 100
    return r > rsi_thresh and ret > pump_pct

def base_bb_breakout(c, h, l, v, o, i, sym, bw_lookback=20, vol_mult=1.5):
    """BB squeeze breakout + volume"""
    m = MATH[sym]
    bw = m['bb_bw']
    up = m['bb_upper'][i]
    if np.isnan(bw[i]) or np.isnan(up) or i < bw_lookback + 5: return False
    recent_bw = bw[i-bw_lookback:i]
    recent_bw = recent_bw[~np.isnan(recent_bw)]
    if len(recent_bw) < bw_lookback // 2: return False
    if bw[i-1] > np.min(recent_bw) * 1.1: return False
    if c[i] <= up: return False
    va = m['vol_avg20'][i]
    if np.isnan(va) or va <= 0 or v[i] < va * vol_mult: return False
    return True

def base_50bar_high(c, h, l, v, o, i, sym, vol_mult=1.5):
    """50-bar high breakout + volume"""
    if i < 51: return False
    prev_high = np.max(h[i-50:i])
    if c[i] <= prev_high: return False
    m = MATH[sym]
    va = m['vol_avg20'][i]
    if np.isnan(va) or va <= 0 or v[i] < va * vol_mult: return False
    return True

def base_btc_lead_lag(c, o, i, sym, all_data, btc_thresh=1.5, lag=2):
    """BTC pumped, alt lagging"""
    if sym == "BTC" or "BTC" not in all_data: return False
    btc = all_data["BTC"]['close'].values
    if i >= len(btc) or i < lag + 1: return False
    btc_ret = (btc[i] - btc[i-lag]) / btc[i-lag] * 100
    alt_ret = (c[i] - c[i-lag]) / c[i-lag] * 100
    return btc_ret > btc_thresh and alt_ret < btc_thresh * 0.3

def base_green_exhaustion(c, o, i, sym, n_green=7):
    """N green candles then red"""
    if i < n_green + 1: return False
    if c[i] >= o[i]: return False  # current must be red
    for j in range(1, n_green + 1):
        if c[i-j] <= o[i-j]: return False
    return True

def base_low_entropy_above_sma50(c, o, i, sym, ent_thresh=2.5):
    """Low entropy + above SMA50"""
    m = MATH[sym]
    s50 = m['sma50'][i]
    if np.isnan(s50) or c[i] <= s50: return False
    ent = m['entropy'][i]
    if np.isnan(ent): return False
    return ent < ent_thresh


# ═══════════════════════════════════════
# MATH CONFIRMATION HELPERS
# ═══════════════════════════════════════

def math_neg_ac(i, sym, thresh=-0.05):
    m = MATH[sym]
    ac = m['ac1'][i]
    return not np.isnan(ac) and ac < thresh

def math_low_hurst(i, sym, thresh=0.45):
    m = MATH[sym]
    h = m['hurst'][i]
    return not np.isnan(h) and h < thresh

def math_high_hurst(i, sym, thresh=0.55):
    m = MATH[sym]
    h = m['hurst'][i]
    return not np.isnan(h) and h > thresh

def math_high_vpin(i, sym, thresh=0.5):
    m = MATH[sym]
    vp = m['vpin'][i]
    return not np.isnan(vp) and vp > thresh

def math_low_vpin(i, sym, thresh=0.3):
    m = MATH[sym]
    vp = m['vpin'][i]
    return not np.isnan(vp) and vp < thresh

def math_neg_skew(i, sym, thresh=-0.5):
    m = MATH[sym]
    sk = m['skew'][i]
    return not np.isnan(sk) and sk < thresh

def math_pos_skew(i, sym, thresh=0.5):
    m = MATH[sym]
    sk = m['skew'][i]
    return not np.isnan(sk) and sk > thresh

def math_high_kurt(i, sym, thresh=3):
    m = MATH[sym]
    kt = m['kurt'][i]
    return not np.isnan(kt) and kt > thresh

def math_vol_spike(i, sym, thresh=2):
    m = MATH[sym]
    vr = m['vol_ratio'][i]
    return not np.isnan(vr) and vr > thresh

def math_vol_trend_declining(i, sym, thresh=0.8):
    m = MATH[sym]
    vt = m['vol_trend'][i]
    return not np.isnan(vt) and vt < thresh

def math_struct_break(i, sym, thresh=1.5):
    m = MATH[sym]
    brk = m['break_mag'][i]
    return not np.isnan(brk) and brk > thresh

def math_poisson_surge(i, sym, thresh=2):
    m = MATH[sym]
    pz = m['pois_z'][i]
    return not np.isnan(pz) and pz > thresh

def math_pos_ac(i, sym, thresh=0.1):
    m = MATH[sym]
    ac = m['ac1'][i]
    return not np.isnan(ac) and ac > thresh

def math_high_entropy(i, sym, thresh=3.5):
    m = MATH[sym]
    ent = m['entropy'][i]
    return not np.isnan(ent) and ent > thresh

def math_low_entropy(i, sym, thresh=2.5):
    m = MATH[sym]
    ent = m['entropy'][i]
    return not np.isnan(ent) and ent < thresh


def main():
    t0 = time.time()
    
    os.makedirs("/Users/lucasaust/code/Crypto-trading-bot/data", exist_ok=True)
    output_path = "/Users/lucasaust/code/Crypto-trading-bot/data/deep_quant_bull_combos.txt"
    output_lines = []
    
    def out(s=""):
        print(s)
        output_lines.append(s)
    
    out("=" * 120)
    out("  DEEP QUANT BULL COMBOS: Base Signals + Mathematical Confirmations")
    out("  Testing 150+ combinations for high-conviction entries")
    out("=" * 120)
    
    out("\nDownloading 2000 hourly candles per pair...")
    all_data = {}
    for sym in PAIRS:
        df = dl(sym)
        if len(df) > 200:
            all_data[sym] = df
            out(f"  {sym}: {len(df)} bars")
        time.sleep(0.15)
    out(f"\nLoaded {len(all_data)} pairs")
    
    out("\nPre-computing mathematical features (Hurst, entropy, AC, VPIN, skew, kurt, etc.)...")
    precompute_math(all_data)
    out("Done.\n")
    
    edges = []
    base_edges = {}  # Track base signal results for comparison
    edge_count = 0
    
    def add_edge(name, cond_fn, base_name=None):
        nonlocal edge_count
        edge_count += 1
        r = test_edge(all_data, name, cond_fn)
        if r:
            r['base'] = base_name or name
            edges.append(r)
        if edge_count % 25 == 0:
            out(f"  ... tested {edge_count} combinations, found {len(edges)} valid ...")
        return r
    
    # ═══════════════════════════════════════════════════════════
    # BASE SIGNALS (for comparison)
    # ═══════════════════════════════════════════════════════════
    out("━" * 100)
    out("  TESTING BASE SIGNALS (for comparison)")
    out("━" * 100)
    
    for pct in [8, 10, 12]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_sma50_ext(c,o,i,sym,p)
            return fn
        r = add_edge(f"BASE: SMA50 ext >{pct}%", mk(pct), f"SMA50>{pct}%")
        if r: base_edges[f"SMA50>{pct}%"] = r
    
    for pct in [5, 8]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_alt_outperform_btc(c,o,i,sym,ad,p)
            return fn
        r = add_edge(f"BASE: Alt>BTC by {pct}% 24h", mk(pct), f"Alt>BTC {pct}%")
        if r: base_edges[f"Alt>BTC {pct}%"] = r
    
    for pump in [8, 10]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_rsi_pump(c,o,i,sym,80,p,12)
            return fn
        r = add_edge(f"BASE: RSI>80+{pump}% pump", mk(pump), f"RSI80+{pump}%")
        if r: base_edges[f"RSI80+{pump}%"] = r
    
    def bb_base(c,h,l,v,o,i,sym,ad):
        return base_bb_breakout(c,h,l,v,o,i,sym)
    r = add_edge("BASE: BB squeeze breakout+vol", bb_base, "BB_breakout")
    if r: base_edges["BB_breakout"] = r
    
    def h50_base(c,h,l,v,o,i,sym,ad):
        return base_50bar_high(c,h,l,v,o,i,sym)
    r = add_edge("BASE: 50-bar high breakout+vol", h50_base, "50bar_high")
    if r: base_edges["50bar_high"] = r
    
    def ll_base(c,h,l,v,o,i,sym,ad):
        return base_btc_lead_lag(c,o,i,sym,ad)
    r = add_edge("BASE: BTC lead-lag buy", ll_base, "BTC_leadlag")
    if r: base_edges["BTC_leadlag"] = r
    
    def ge_base(c,h,l,v,o,i,sym,ad):
        return base_green_exhaustion(c,o,i,sym,7)
    r = add_edge("BASE: 7 green then red", ge_base, "7green_red")
    if r: base_edges["7green_red"] = r
    
    def le_base(c,h,l,v,o,i,sym,ad):
        return base_low_entropy_above_sma50(c,o,i,sym,2.5)
    r = add_edge("BASE: Low entropy+>SMA50", le_base, "LowEnt_SMA50")
    if r: base_edges["LowEnt_SMA50"] = r
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 1: SMA50 Extension + Math Confirmations
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 1: SMA50 Extension + Math Confirmations (SHORT)")
    out("━" * 100)
    
    for ext_pct in [8, 10, 12]:
        bn = f"SMA50>{ext_pct}%"
        
        # + Negative AC
        for ac_t in [-0.05, -0.1, -0.15]:
            def mk(ep, at):
                def fn(c,h,l,v,o,i,sym,ad):
                    return base_sma50_ext(c,o,i,sym,ep) and math_neg_ac(i,sym,at)
                return fn
            add_edge(f"SMA50>{ext_pct}%+AC<{ac_t}", mk(ext_pct, ac_t), bn)
        
        # + Low Hurst
        for ht in [0.45, 0.4, 0.35]:
            def mk(ep, h):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_sma50_ext(c,o,i,sym,ep) and math_low_hurst(i,sym,h)
                return fn
            add_edge(f"SMA50>{ext_pct}%+H<{ht}", mk(ext_pct, ht), bn)
        
        # + High VPIN
        for vt in [0.5, 0.6, 0.7]:
            def mk(ep, v):
                def fn(c,hh,l,vol,o,i,sym,ad):
                    return base_sma50_ext(c,o,i,sym,ep) and math_high_vpin(i,sym,v)
                return fn
            add_edge(f"SMA50>{ext_pct}%+VPIN>{vt}", mk(ext_pct, vt), bn)
        
        # + Negative skew
        for st in [-0.5, -1.0]:
            def mk(ep, s):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_sma50_ext(c,o,i,sym,ep) and math_neg_skew(i,sym,s)
                return fn
            add_edge(f"SMA50>{ext_pct}%+skew<{st}", mk(ext_pct, st), bn)
        
        # + High kurtosis
        for kt in [3, 5]:
            def mk(ep, k):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_sma50_ext(c,o,i,sym,ep) and math_high_kurt(i,sym,k)
                return fn
            add_edge(f"SMA50>{ext_pct}%+kurt>{kt}", mk(ext_pct, kt), bn)
        
        # + Volume spike
        for vm in [2, 3]:
            def mk(ep, v):
                def fn(c,hh,l,vol,o,i,sym,ad):
                    return base_sma50_ext(c,o,i,sym,ep) and math_vol_spike(i,sym,v)
                return fn
            add_edge(f"SMA50>{ext_pct}%+vol>{vm}x", mk(ext_pct, vm), bn)
        
        # + Structural break
        for bt in [1.5, 2.0]:
            def mk(ep, b):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_sma50_ext(c,o,i,sym,ep) and math_struct_break(i,sym,b)
                return fn
            add_edge(f"SMA50>{ext_pct}%+break>{bt}σ", mk(ext_pct, bt), bn)
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 2: Alt Outperforms BTC + Math
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 2: Alt Outperforms BTC + Math Confirmations (SHORT)")
    out("━" * 100)
    
    for alt_pct in [5, 8]:
        bn = f"Alt>BTC {alt_pct}%"
        
        # + Negative AC on alt
        for at in [-0.05, -0.1, -0.15]:
            def mk(ap, at_):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_alt_outperform_btc(c,o,i,sym,ad,ap) and math_neg_ac(i,sym,at_)
                return fn
            add_edge(f"Alt>BTC{alt_pct}%+AC<{at}", mk(alt_pct, at), bn)
        
        # + Low Hurst on alt
        for ht in [0.45, 0.4]:
            def mk(ap, h):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_alt_outperform_btc(c,o,i,sym,ad,ap) and math_low_hurst(i,sym,h)
                return fn
            add_edge(f"Alt>BTC{alt_pct}%+H<{ht}", mk(alt_pct, ht), bn)
        
        # + High VPIN on alt
        for vt in [0.5, 0.6, 0.7]:
            def mk(ap, v):
                def fn(c,hh,l,vol,o,i,sym,ad):
                    return base_alt_outperform_btc(c,o,i,sym,ad,ap) and math_high_vpin(i,sym,v)
                return fn
            add_edge(f"Alt>BTC{alt_pct}%+VPIN>{vt}", mk(alt_pct, vt), bn)
        
        # + Alt RSI high
        for rt in [70, 75, 80]:
            def mk(ap, r):
                def fn(c,hh,l,v,o,i,sym,ad):
                    if not base_alt_outperform_btc(c,o,i,sym,ad,ap): return False
                    m = MATH[sym]
                    rsi_val = m['rsi14'][i]
                    return not np.isnan(rsi_val) and rsi_val > r
                return fn
            add_edge(f"Alt>BTC{alt_pct}%+RSI>{rt}", mk(alt_pct, rt), bn)
        
        # + BTC volume declining
        def mk(ap):
            def fn(c,hh,l,v,o,i,sym,ad):
                if not base_alt_outperform_btc(c,o,i,sym,ad,ap): return False
                if "BTC" not in ad: return False
                btc_v = ad["BTC"]['volume'].values.astype(float)
                if i >= len(btc_v) or i < 20: return False
                btc_vol_recent = np.mean(btc_v[i-5:i+1])
                btc_vol_prior = np.mean(btc_v[i-15:i-5])
                return btc_vol_prior > 0 and btc_vol_recent < btc_vol_prior * 0.7
            return fn
        add_edge(f"Alt>BTC{alt_pct}%+BTC vol declining", mk(alt_pct), bn)
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 3: RSI > 80 + Pump + Math
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 3: RSI > 80 + Pump + Math Confirmations (SHORT)")
    out("━" * 100)
    
    for pump_pct in [8, 10]:
        bn = f"RSI80+{pump_pct}%"
        
        # + Negative AC
        for at in [-0.05, -0.1, -0.15]:
            def mk(pp, at_):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_rsi_pump(c,o,i,sym,80,pp,12) and math_neg_ac(i,sym,at_)
                return fn
            add_edge(f"RSI80+{pump_pct}%+AC<{at}", mk(pump_pct, at), bn)
        
        # + Declining volume trend
        for vt in [0.7, 0.8]:
            def mk(pp, v):
                def fn(c,hh,l,vol,o,i,sym,ad):
                    return base_rsi_pump(c,o,i,sym,80,pp,12) and math_vol_trend_declining(i,sym,v)
                return fn
            add_edge(f"RSI80+{pump_pct}%+vol_trend<{vt}", mk(pump_pct, vt), bn)
        
        # + High entropy
        for et in [3.5, 4.0]:
            def mk(pp, e):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_rsi_pump(c,o,i,sym,80,pp,12) and math_high_entropy(i,sym,e)
                return fn
            add_edge(f"RSI80+{pump_pct}%+entropy>{et}", mk(pump_pct, et), bn)
        
        # + Negative skew
        for st in [-0.5, -1.0]:
            def mk(pp, s):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_rsi_pump(c,o,i,sym,80,pp,12) and math_neg_skew(i,sym,s)
                return fn
            add_edge(f"RSI80+{pump_pct}%+skew<{st}", mk(pump_pct, st), bn)
        
        # + Poisson volume anomaly
        for pz in [2, 3]:
            def mk(pp, p):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_rsi_pump(c,o,i,sym,80,pp,12) and math_poisson_surge(i,sym,p)
                return fn
            add_edge(f"RSI80+{pump_pct}%+Poisson>{pz}", mk(pump_pct, pz), bn)
        
        # + High kurtosis
        for kt in [3, 5]:
            def mk(pp, k):
                def fn(c,hh,l,v,o,i,sym,ad):
                    return base_rsi_pump(c,o,i,sym,80,pp,12) and math_high_kurt(i,sym,k)
                return fn
            add_edge(f"RSI80+{pump_pct}%+kurt>{kt}", mk(pump_pct, kt), bn)
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 4: Low Entropy + SMA50 + Math
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 4: Low Entropy + SMA50 + Math Confirmations (SHORT)")
    out("━" * 100)
    
    bn = "LowEnt_SMA50"
    
    for at in [-0.05, -0.1]:
        def mk(at_):
            def fn(c,hh,l,v,o,i,sym,ad):
                return base_low_entropy_above_sma50(c,o,i,sym) and math_neg_ac(i,sym,at_)
            return fn
        add_edge(f"LowEnt+SMA50+AC<{at}", mk(at), bn)
    
    for ht in [0.45, 0.4]:
        def mk(h):
            def fn(c,hh,l,v,o,i,sym,ad):
                return base_low_entropy_above_sma50(c,o,i,sym) and math_low_hurst(i,sym,h)
            return fn
        add_edge(f"LowEnt+SMA50+H<{ht}", mk(ht), bn)
    
    for vt in [0.5, 0.6]:
        def mk(v):
            def fn(c,hh,l,vol,o,i,sym,ad):
                return base_low_entropy_above_sma50(c,o,i,sym) and math_high_vpin(i,sym,v)
            return fn
        add_edge(f"LowEnt+SMA50+VPIN>{vt}", mk(vt), bn)
    
    for kt in [3, 5]:
        def mk(k):
            def fn(c,hh,l,v,o,i,sym,ad):
                return base_low_entropy_above_sma50(c,o,i,sym) and math_high_kurt(i,sym,k)
            return fn
        add_edge(f"LowEnt+SMA50+kurt>{kt}", mk(kt), bn)
    
    for st in [-0.5, -1.0]:
        def mk(s):
            def fn(c,hh,l,v,o,i,sym,ad):
                return base_low_entropy_above_sma50(c,o,i,sym) and math_neg_skew(i,sym,s)
            return fn
        add_edge(f"LowEnt+SMA50+skew<{st}", mk(st), bn)
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 5: BB Breakout + Math Confirmations (LONG)
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 5: BB Squeeze Breakout + Math Confirmations (LONG)")
    out("━" * 100)
    
    bn = "BB_breakout"
    
    # + Positive AC
    for at in [0.05, 0.1, 0.15]:
        def mk(a):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_bb_breakout(c,h,l,v,o,i,sym) and math_pos_ac(i,sym,a)
            return fn
        add_edge(f"BB_break+AC>{at}", mk(at), bn)
    
    # + High Hurst
    for ht in [0.55, 0.6, 0.65]:
        def mk(h):
            def fn(c,hh,l,v,o,i,sym,ad):
                return base_bb_breakout(c,hh,l,v,o,i,sym) and math_high_hurst(i,sym,h)
            return fn
        add_edge(f"BB_break+H>{ht}", mk(ht), bn)
    
    # + Low VPIN
    for vt in [0.3, 0.2]:
        def mk(v):
            def fn(c,h,l,vol,o,i,sym,ad):
                return base_bb_breakout(c,h,l,vol,o,i,sym) and math_low_vpin(i,sym,v)
            return fn
        add_edge(f"BB_break+VPIN<{vt}", mk(vt), bn)
    
    # + Positive skew
    for st in [0.5, 1.0]:
        def mk(s):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_bb_breakout(c,h,l,v,o,i,sym) and math_pos_skew(i,sym,s)
            return fn
        add_edge(f"BB_break+skew>{st}", mk(st), bn)
    
    # + Poisson volume surge
    for pz in [2, 3]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_bb_breakout(c,h,l,v,o,i,sym) and math_poisson_surge(i,sym,p)
            return fn
        add_edge(f"BB_break+Poisson>{pz}", mk(pz), bn)
    
    # + Low entropy (predictable breakout)
    for et in [2.5, 3.0]:
        def mk(e):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_bb_breakout(c,h,l,v,o,i,sym) and math_low_entropy(i,sym,e)
            return fn
        add_edge(f"BB_break+entropy<{et}", mk(et), bn)
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 6: 50-bar High Breakout + Math (LONG)
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 6: 50-bar High Breakout + Math Confirmations (LONG)")
    out("━" * 100)
    
    bn = "50bar_high"
    
    for at in [0.05, 0.1]:
        def mk(a):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_50bar_high(c,h,l,v,o,i,sym) and math_pos_ac(i,sym,a)
            return fn
        add_edge(f"50bar_high+AC>{at}", mk(at), bn)
    
    for ht in [0.55, 0.6]:
        def mk(h):
            def fn(c,hh,l,v,o,i,sym,ad):
                return base_50bar_high(c,hh,l,v,o,i,sym) and math_high_hurst(i,sym,h)
            return fn
        add_edge(f"50bar_high+H>{ht}", mk(ht), bn)
    
    for vt in [0.3, 0.2]:
        def mk(v):
            def fn(c,h,l,vol,o,i,sym,ad):
                return base_50bar_high(c,h,l,vol,o,i,sym) and math_low_vpin(i,sym,v)
            return fn
        add_edge(f"50bar_high+VPIN<{vt}", mk(vt), bn)
    
    for st in [0.5, 1.0]:
        def mk(s):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_50bar_high(c,h,l,v,o,i,sym) and math_pos_skew(i,sym,s)
            return fn
        add_edge(f"50bar_high+skew>{st}", mk(st), bn)
    
    # + Recently crossed above SMA50
    def mk_sma_cross():
        def fn(c,h,l,v,o,i,sym,ad):
            if not base_50bar_high(c,h,l,v,o,i,sym): return False
            m = MATH[sym]
            s50 = m['sma50']
            # Check if price crossed above SMA50 within last 10 bars
            for j in range(1, min(11, i)):
                if not np.isnan(s50[i-j]) and c[i-j] < s50[i-j]:
                    return True
            return False
        return fn
    add_edge("50bar_high+SMA50 cross <10bars", mk_sma_cross(), bn)
    
    # + OBV trending up
    def mk_obv_up():
        def fn(c,h,l,v,o,i,sym,ad):
            if not base_50bar_high(c,h,l,v,o,i,sym): return False
            m = MATH[sym]
            ob = m['obv']
            if i < 20: return False
            return ob[i] > ob[i-10] and ob[i-10] > ob[i-20]  # OBV trending up
        return fn
    add_edge("50bar_high+OBV trending up", mk_obv_up(), bn)
    
    for pz in [2, 3]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_50bar_high(c,h,l,v,o,i,sym) and math_poisson_surge(i,sym,p)
            return fn
        add_edge(f"50bar_high+Poisson>{pz}", mk(pz), bn)
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 7: BTC Lead-Lag + Math (LONG)
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 7: BTC Lead-Lag + Math Confirmations (LONG)")
    out("━" * 100)
    
    bn = "BTC_leadlag"
    
    # + Positive AC on BTC
    for at in [0.05, 0.1]:
        def mk(a):
            def fn(c,h,l,v,o,i,sym,ad):
                if not base_btc_lead_lag(c,o,i,sym,ad): return False
                if "BTC" not in MATH: return False
                return math_pos_ac(i,"BTC",a)
            return fn
        add_edge(f"BTC_lag+BTC_AC>{at}", mk(at), bn)
    
    # + High Hurst on BTC
    for ht in [0.55, 0.6]:
        def mk(h):
            def fn(c,hh,l,v,o,i,sym,ad):
                if not base_btc_lead_lag(c,o,i,sym,ad): return False
                if "BTC" not in MATH: return False
                return math_high_hurst(i,"BTC",h)
            return fn
        add_edge(f"BTC_lag+BTC_H>{ht}", mk(ht), bn)
    
    # + Low VPIN on alt
    for vt in [0.3, 0.2]:
        def mk(v):
            def fn(c,h,l,vol,o,i,sym,ad):
                if not base_btc_lead_lag(c,o,i,sym,ad): return False
                return math_low_vpin(i,sym,v)
            return fn
        add_edge(f"BTC_lag+alt_VPIN<{vt}", mk(vt), bn)
    
    # + Alt RSI < 50
    for rt in [50, 45, 40]:
        def mk(r):
            def fn(c,h,l,v,o,i,sym,ad):
                if not base_btc_lead_lag(c,o,i,sym,ad): return False
                m = MATH[sym]
                rsi_val = m['rsi14'][i]
                return not np.isnan(rsi_val) and rsi_val < r
            return fn
        add_edge(f"BTC_lag+alt_RSI<{rt}", mk(rt), bn)
    
    # ═══════════════════════════════════════════════════════════
    # SIGNAL 8: Green Exhaustion (7 green then red) + Math (SHORT)
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  SIGNAL 8: Green Exhaustion + Math Confirmations (SHORT)")
    out("━" * 100)
    
    bn = "7green_red"
    
    # + Negative AC
    for at in [-0.05, -0.1, -0.15]:
        def mk(a):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_green_exhaustion(c,o,i,sym,7) and math_neg_ac(i,sym,a)
            return fn
        add_edge(f"7green_red+AC<{at}", mk(at), bn)
    
    # + High kurtosis
    for kt in [3, 5]:
        def mk(k):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_green_exhaustion(c,o,i,sym,7) and math_high_kurt(i,sym,k)
            return fn
        add_edge(f"7green_red+kurt>{kt}", mk(kt), bn)
    
    # + High VPIN
    for vt in [0.5, 0.6, 0.7]:
        def mk(v):
            def fn(c,h,l,vol,o,i,sym,ad):
                return base_green_exhaustion(c,o,i,sym,7) and math_high_vpin(i,sym,v)
            return fn
        add_edge(f"7green_red+VPIN>{vt}", mk(vt), bn)
    
    # + Declining volume on green candles
    def mk_declining_vol():
        def fn(c,h,l,v,o,i,sym,ad):
            if not base_green_exhaustion(c,o,i,sym,7): return False
            # Volume declining over the green streak
            early_vol = np.mean(v[i-7:i-4])
            late_vol = np.mean(v[i-3:i])
            return early_vol > 0 and late_vol < early_vol * 0.7
        return fn
    add_edge("7green_red+declining_green_vol", mk_declining_vol(), bn)
    
    # + Low Hurst (mean-reverting)
    for ht in [0.45, 0.4]:
        def mk(h):
            def fn(c,hh,l,v,o,i,sym,ad):
                return base_green_exhaustion(c,o,i,sym,7) and math_low_hurst(i,sym,h)
            return fn
        add_edge(f"7green_red+H<{ht}", mk(ht), bn)
    
    # + Negative skew
    for st in [-0.5, -1.0]:
        def mk(s):
            def fn(c,h,l,v,o,i,sym,ad):
                return base_green_exhaustion(c,o,i,sym,7) and math_neg_skew(i,sym,s)
            return fn
        add_edge(f"7green_red+skew<{st}", mk(st), bn)
    
    # ═══════════════════════════════════════════════════════════
    # MULTI-MATH COMBOS (2-3 confirmations stacked)
    # ═══════════════════════════════════════════════════════════
    out("\n" + "━" * 100)
    out("  MULTI-MATH COMBOS: 2-3 confirmations stacked")
    out("━" * 100)
    
    # SMA50 ext + neg AC + high VPIN
    for ep in [8, 10]:
        def mk(e):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_sma50_ext(c,o,i,sym,e) and 
                        math_neg_ac(i,sym,-0.05) and 
                        math_high_vpin(i,sym,0.5))
            return fn
        add_edge(f"SMA50>{ep}%+negAC+highVPIN", mk(ep), f"SMA50>{ep}%")
    
    # SMA50 ext + neg AC + low Hurst
    for ep in [8, 10]:
        def mk(e):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_sma50_ext(c,o,i,sym,e) and 
                        math_neg_ac(i,sym,-0.05) and 
                        math_low_hurst(i,sym,0.45))
            return fn
        add_edge(f"SMA50>{ep}%+negAC+lowH", mk(ep), f"SMA50>{ep}%")
    
    # SMA50 ext + neg skew + high kurt
    for ep in [8, 10]:
        def mk(e):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_sma50_ext(c,o,i,sym,e) and 
                        math_neg_skew(i,sym,-0.5) and 
                        math_high_kurt(i,sym,3))
            return fn
        add_edge(f"SMA50>{ep}%+negSkew+highKurt", mk(ep), f"SMA50>{ep}%")
    
    # SMA50 ext + neg AC + vol spike
    for ep in [8, 10]:
        def mk(e):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_sma50_ext(c,o,i,sym,e) and 
                        math_neg_ac(i,sym,-0.05) and 
                        math_vol_spike(i,sym,2))
            return fn
        add_edge(f"SMA50>{ep}%+negAC+vol>2x", mk(ep), f"SMA50>{ep}%")
    
    # SMA50 ext + low Hurst + high VPIN
    for ep in [8, 10]:
        def mk(e):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_sma50_ext(c,o,i,sym,e) and 
                        math_low_hurst(i,sym,0.45) and 
                        math_high_vpin(i,sym,0.5))
            return fn
        add_edge(f"SMA50>{ep}%+lowH+highVPIN", mk(ep), f"SMA50>{ep}%")
    
    # Triple: SMA50 ext + neg AC + low Hurst + high VPIN
    for ep in [8, 10]:
        def mk(e):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_sma50_ext(c,o,i,sym,e) and 
                        math_neg_ac(i,sym,-0.05) and 
                        math_low_hurst(i,sym,0.45) and
                        math_high_vpin(i,sym,0.5))
            return fn
        add_edge(f"SMA50>{ep}%+negAC+lowH+highVPIN", mk(ep), f"SMA50>{ep}%")
    
    # BB breakout + pos AC + high Hurst
    def mk_bb_ac_h():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_bb_breakout(c,h,l,v,o,i,sym) and 
                    math_pos_ac(i,sym,0.05) and 
                    math_high_hurst(i,sym,0.55))
        return fn
    add_edge("BB_break+posAC+highH", mk_bb_ac_h(), "BB_breakout")
    
    # BB breakout + pos AC + low VPIN
    def mk_bb_ac_vpin():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_bb_breakout(c,h,l,v,o,i,sym) and 
                    math_pos_ac(i,sym,0.05) and 
                    math_low_vpin(i,sym,0.3))
        return fn
    add_edge("BB_break+posAC+lowVPIN", mk_bb_ac_vpin(), "BB_breakout")
    
    # BB breakout + high Hurst + Poisson surge
    def mk_bb_h_pois():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_bb_breakout(c,h,l,v,o,i,sym) and 
                    math_high_hurst(i,sym,0.55) and 
                    math_poisson_surge(i,sym,2))
        return fn
    add_edge("BB_break+highH+Poisson>2", mk_bb_h_pois(), "BB_breakout")
    
    # BB breakout + low entropy + high Hurst
    def mk_bb_ent_h():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_bb_breakout(c,h,l,v,o,i,sym) and 
                    math_low_entropy(i,sym,2.5) and 
                    math_high_hurst(i,sym,0.55))
        return fn
    add_edge("BB_break+lowEntropy+highH", mk_bb_ent_h(), "BB_breakout")
    
    # Alt outperform + neg AC + high RSI
    for ap in [5, 8]:
        def mk(a):
            def fn(c,h,l,v,o,i,sym,ad):
                if not base_alt_outperform_btc(c,o,i,sym,ad,a): return False
                if not math_neg_ac(i,sym,-0.05): return False
                m = MATH[sym]
                rsi_val = m['rsi14'][i]
                return not np.isnan(rsi_val) and rsi_val > 70
            return fn
        add_edge(f"Alt>BTC{ap}%+negAC+RSI>70", mk(ap), f"Alt>BTC {ap}%")
    
    # Alt outperform + high VPIN + neg skew
    for ap in [5, 8]:
        def mk(a):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_alt_outperform_btc(c,o,i,sym,ad,a) and 
                        math_high_vpin(i,sym,0.5) and 
                        math_neg_skew(i,sym,-0.5))
            return fn
        add_edge(f"Alt>BTC{ap}%+highVPIN+negSkew", mk(ap), f"Alt>BTC {ap}%")
    
    # RSI80 + pump + neg AC + declining vol
    for pp in [8, 10]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_rsi_pump(c,o,i,sym,80,p,12) and 
                        math_neg_ac(i,sym,-0.05) and 
                        math_vol_trend_declining(i,sym,0.8))
            return fn
        add_edge(f"RSI80+{pp}%+negAC+decVol", mk(pp), f"RSI80+{pp}%")
    
    # RSI80 + pump + neg skew + high kurt
    for pp in [8, 10]:
        def mk(p):
            def fn(c,h,l,v,o,i,sym,ad):
                return (base_rsi_pump(c,o,i,sym,80,p,12) and 
                        math_neg_skew(i,sym,-0.5) and 
                        math_high_kurt(i,sym,3))
            return fn
        add_edge(f"RSI80+{pp}%+negSkew+highKurt", mk(pp), f"RSI80+{pp}%")
    
    # Green exhaustion + neg AC + high VPIN
    def mk_ge_ac_vpin():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_green_exhaustion(c,o,i,sym,7) and 
                    math_neg_ac(i,sym,-0.05) and 
                    math_high_vpin(i,sym,0.5))
        return fn
    add_edge("7green_red+negAC+highVPIN", mk_ge_ac_vpin(), "7green_red")
    
    # Green exhaustion + high kurt + neg skew
    def mk_ge_kurt_skew():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_green_exhaustion(c,o,i,sym,7) and 
                    math_high_kurt(i,sym,3) and 
                    math_neg_skew(i,sym,-0.5))
        return fn
    add_edge("7green_red+highKurt+negSkew", mk_ge_kurt_skew(), "7green_red")
    
    # Low entropy + SMA50 + neg AC + high VPIN
    def mk_le_ac_vpin():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_low_entropy_above_sma50(c,o,i,sym) and 
                    math_neg_ac(i,sym,-0.05) and 
                    math_high_vpin(i,sym,0.5))
        return fn
    add_edge("LowEnt+SMA50+negAC+highVPIN", mk_le_ac_vpin(), "LowEnt_SMA50")
    
    # Low entropy + SMA50 + low Hurst + neg skew
    def mk_le_h_skew():
        def fn(c,h,l,v,o,i,sym,ad):
            return (base_low_entropy_above_sma50(c,o,i,sym) and 
                    math_low_hurst(i,sym,0.45) and 
                    math_neg_skew(i,sym,-0.5))
        return fn
    add_edge("LowEnt+SMA50+lowH+negSkew", mk_le_h_skew(), "LowEnt_SMA50")
    
    # BTC lead-lag + BTC pos AC + alt low VPIN
    def mk_ll_combo():
        def fn(c,h,l,v,o,i,sym,ad):
            if not base_btc_lead_lag(c,o,i,sym,ad): return False
            if "BTC" not in MATH: return False
            return math_pos_ac(i,"BTC",0.05) and math_low_vpin(i,sym,0.3)
        return fn
    add_edge("BTC_lag+BTC_posAC+alt_lowVPIN", mk_ll_combo(), "BTC_leadlag")
    
    # BTC lead-lag + BTC high Hurst + alt RSI < 50
    def mk_ll_combo2():
        def fn(c,h,l,v,o,i,sym,ad):
            if not base_btc_lead_lag(c,o,i,sym,ad): return False
            if "BTC" not in MATH: return False
            if not math_high_hurst(i,"BTC",0.55): return False
            m = MATH[sym]
            rsi_val = m['rsi14'][i]
            return not np.isnan(rsi_val) and rsi_val < 50
        return fn
    add_edge("BTC_lag+BTC_highH+alt_RSI<50", mk_ll_combo2(), "BTC_leadlag")
    
    # 50-bar high + pos AC + high Hurst + OBV up
    def mk_50h_triple():
        def fn(c,h,l,v,o,i,sym,ad):
            if not base_50bar_high(c,h,l,v,o,i,sym): return False
            if not math_pos_ac(i,sym,0.05): return False
            if not math_high_hurst(i,sym,0.55): return False
            m = MATH[sym]
            ob = m['obv']
            if i < 10: return False
            return ob[i] > ob[i-10]
        return fn
    add_edge("50bar_high+posAC+highH+OBVup", mk_50h_triple(), "50bar_high")
    
    # ═══════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════
    
    edges_valid = [e for e in edges if e is not None]
    edges_valid.sort(key=lambda e: e['best8'], reverse=True)
    
    out(f"\n{'='*120}")
    out(f"  TESTED {edge_count} COMBINATIONS | {len(edges_valid)} HAD 15+ TRADES")
    out(f"{'='*120}")
    
    header = f"  {'Pattern':<60s} {'N':>5s} {'L8h':>7s} {'S8h':>7s} {'L24h':>7s} {'WR8':>5s} {'Dir':>6s} {'Net8':>7s}"
    out(header)
    out("  " + "-" * 105)
    
    for e in edges_valid:
        net = e['net8']
        line = (f"  {e['name']:<60s} {e['n']:>5d} "
                f"{e['L8']:>+6.2f}% {e['S8']:>+6.2f}% {e['L24']:>+6.2f}% "
                f"{e['wr8']:>4.0f}% {e['dir']:>6s} {net:>+6.2f}%")
        out(line)
    
    # ═══ TOP 20 ═══
    profitable = [e for e in edges_valid if e['net8'] > 0]
    out(f"\n{'='*120}")
    out(f"  ★ TOP 20 COMBINATIONS (sorted by net 8h return) ★")
    out(f"{'='*120}")
    for idx, e in enumerate(profitable[:20], 1):
        out(f"  {idx:>2d}. {e['name']}")
        out(f"      Dir: {e['dir']} | N={e['n']} | WR={e['wr8']:.1f}% | "
            f"8h={e['best8']:+.3f}% (net {e['net8']:+.3f}%) | "
            f"L24={e['L24']:+.3f}% S24={e['S24']:+.3f}%")
        out("")
    
    # ═══ BEST COMBO PER BASE SIGNAL ═══
    out(f"\n{'='*120}")
    out(f"  ★ BEST COMBINATION PER BASE SIGNAL ★")
    out(f"{'='*120}")
    
    base_names = set()
    for e in edges_valid:
        if e.get('base'):
            base_names.add(e['base'])
    
    for bn in sorted(base_names):
        combos = [e for e in edges_valid if e.get('base') == bn and not e['name'].startswith("BASE:")]
        if not combos:
            continue
        combos.sort(key=lambda e: e['best8'], reverse=True)
        best = combos[0]
        
        base_result = base_edges.get(bn)
        
        out(f"\n  BASE: {bn}")
        if base_result:
            out(f"    Base:  N={base_result['n']:>5d} | {base_result['dir']} | "
                f"8h={base_result['best8']:+.3f}% net={base_result['net8']:+.3f}% | WR={base_result['wr8']:.1f}%")
        out(f"    Best:  {best['name']}")
        out(f"           N={best['n']:>5d} | {best['dir']} | "
            f"8h={best['best8']:+.3f}% net={best['net8']:+.3f}% | WR={best['wr8']:.1f}%")
        if base_result:
            wr_imp = best['wr8'] - base_result['wr8']
            ret_imp = best['best8'] - base_result['best8']
            out(f"    IMPROVEMENT: WR {wr_imp:+.1f}pp | Return {ret_imp:+.3f}%")
    
    # ═══ IMPROVEMENT vs BASE ═══
    out(f"\n{'='*120}")
    out(f"  ★ ALL IMPROVEMENTS vs BASE SIGNALS ★")
    out(f"{'='*120}")
    out(f"  {'Combo':<55s} {'Base WR':>7s} {'Combo WR':>8s} {'Δ WR':>6s} {'Base 8h':>8s} {'Combo 8h':>9s} {'Δ 8h':>7s}")
    out("  " + "-" * 105)
    
    for e in edges_valid:
        if e['name'].startswith("BASE:"): continue
        bn = e.get('base')
        if not bn or bn not in base_edges: continue
        base = base_edges[bn]
        wr_imp = e['wr8'] - base['wr8']
        ret_imp = e['best8'] - base['best8']
        if ret_imp > 0 or wr_imp > 2:  # Only show improvements
            out(f"  {e['name']:<55s} {base['wr8']:>6.1f}% {e['wr8']:>7.1f}% {wr_imp:>+5.1f}% "
                f"{base['best8']:>+7.3f}% {e['best8']:>+8.3f}% {ret_imp:>+6.3f}%")
    
    elapsed = time.time() - t0
    out(f"\n  Completed in {elapsed:.0f}s | Tested {edge_count} combos | {len(profitable)} profitable")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(output_lines))
    out(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
