#!/usr/bin/env python3
"""
DEEP QUANT V3: Real mathematical edges from statistical physics & probability theory.

Not indicators — MATH.
- Poisson process: model trade arrival rates, detect anomalies
- Hurst exponent: is this pair mean-reverting or trending RIGHT NOW?
- Shannon entropy: measure market uncertainty, trade when it drops
- Autocorrelation structure: find persistence and anti-persistence
- Markov transition probabilities: what's the next state likely to be?
- Order flow toxicity: VPIN-like metric from OHLCV
- Fractal dimension: market complexity signals
- Extreme value theory: model tail risk properly
- Regime change detection: CUSUM/structural breaks
- Cross-correlation lag: which coin leads which?
"""
import requests, numpy as np, pandas as pd, warnings
from datetime import datetime, timezone
from collections import defaultdict
warnings.filterwarnings('ignore')

CC_BASE = "https://min-api.cryptocompare.com/data/v2"
FEE = 0.42

PAIRS = ["BTC","ETH","SOL","LINK","AVAX","DOT","ADA","XRP",
         "DOGE","UNI","NEAR","ATOM","AAVE","XLM","FIL","LTC"]

def dl(sym, n=2000):
    try:
        r = requests.get(f"{CC_BASE}/histohour", params={"fsym":sym,"tsym":"USD","limit":n}, timeout=30)
        d = r.json().get("Data",{}).get("Data",[])
        rows = [{'time':x['time'],'open':x['open'],'high':x['high'],'low':x['low'],
                 'close':x['close'],'volume':x.get('volumeto',0)} for x in d if x.get('close',0)>0]
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

def test_edge(all_data, name, cond_fn, min_n=20):
    results = []
    for sym, df in all_data.items():
        c=df['close'].values.astype(float); h=df['high'].values.astype(float)
        l=df['low'].values.astype(float); v=df['volume'].values.astype(float)
        o=df['open'].values.astype(float)
        for i in range(100, len(c)-25):
            try:
                if cond_fn(c,h,l,v,o,i,sym,all_data):
                    row = {'sym':sym}
                    for hb in [1,4,8,24]:
                        if i+hb<len(c):
                            row[f'L{hb}']=(c[i+hb]-c[i])/c[i]*100
                            row[f'S{hb}']=(c[i]-c[i+hb])/c[i]*100
                    results.append(row)
            except: continue
    if len(results)<min_n: return None
    rdf=pd.DataFrame(results); n=len(rdf)
    l8=rdf.get('L8',pd.Series([0])).mean(); s8=rdf.get('S8',pd.Series([0])).mean()
    l24=rdf.get('L24',pd.Series([0])).mean()
    best8=max(l8,s8); direction="LONG" if l8>=s8 else "SHORT"
    wr8=(rdf['L8']>FEE).mean()*100 if 'L8' in rdf and direction=="LONG" else (rdf['S8']>FEE).mean()*100 if 'S8' in rdf else 0
    return {'name':name,'n':n,'L8':round(l8,3),'S8':round(s8,3),'L24':round(l24,3),
            'best8':round(best8,3),'dir':direction,'wr8':round(wr8,1)}


# ═══════════════════════════════════════
# MATHEMATICAL TOOLS
# ═══════════════════════════════════════

def hurst_exponent(series, max_lag=20):
    """Estimate Hurst exponent. H<0.5 = mean-reverting, H>0.5 = trending, H=0.5 = random."""
    lags = range(2, min(max_lag, len(series)//2))
    tau = []; rs = []
    for lag in lags:
        # R/S analysis
        chunks = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
        if len(chunks) < 2: continue
        rs_values = []
        for chunk in chunks:
            if len(chunk) < 2: continue
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk)
            if S > 0:
                rs_values.append(R/S)
        if rs_values:
            tau.append(lag)
            rs.append(np.mean(rs_values))
    if len(tau) < 3: return 0.5
    log_tau = np.log(tau)
    log_rs = np.log(rs)
    try:
        H = np.polyfit(log_tau, log_rs, 1)[0]
        return max(0, min(1, H))
    except:
        return 0.5

def shannon_entropy(returns, bins=20):
    """Shannon entropy of return distribution. Low = predictable, High = random."""
    if len(returns) < 10: return 3.0  # Default high entropy
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0: return 3.0
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs))

def autocorrelation(series, lag=1):
    """Autocorrelation at given lag. Positive = persistent, Negative = mean-reverting."""
    if len(series) < lag + 10: return 0
    s = pd.Series(series)
    return float(s.autocorr(lag=lag))

def vpin_proxy(close, volume, window=20):
    """Volume-synchronized probability of informed trading (proxy from OHLCV).
    High VPIN = toxic flow = big move coming."""
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
    """CUSUM-like structural break detection. Returns magnitude of break."""
    if len(series) < window * 2: return 0
    first_half = series[-window*2:-window]
    second_half = series[-window:]
    mean1 = np.mean(first_half); std1 = np.std(first_half)
    mean2 = np.mean(second_half)
    if std1 == 0: return 0
    return abs(mean2 - mean1) / std1  # Z-score of the shift

def poisson_volume_anomaly(volume, window=48):
    """Model volume as Poisson process. Detect when arrival rate changes significantly."""
    if len(volume) < window + 10: return 0, 0
    # Normalize volume to "events per hour"
    baseline = np.mean(volume[-window-10:-10])
    if baseline <= 0: return 0, 0
    recent = np.mean(volume[-5:])
    # Poisson: expected = baseline, observed = recent
    # Z-score approximation for Poisson
    z = (recent - baseline) / np.sqrt(max(baseline, 1))
    return z, recent / baseline

def cross_correlation_lead(series_a, series_b, max_lag=12):
    """Find if series_a leads series_b (or vice versa). Returns (best_lag, correlation)."""
    if len(series_a) < max_lag * 3 or len(series_b) < max_lag * 3: return 0, 0
    ret_a = np.diff(series_a) / series_a[:-1]
    ret_b = np.diff(series_b) / series_b[:-1]
    min_len = min(len(ret_a), len(ret_b))
    ret_a = ret_a[-min_len:]; ret_b = ret_b[-min_len:]
    
    best_lag = 0; best_corr = 0
    for lag in range(-max_lag, max_lag+1):
        if lag == 0: continue
        if lag > 0:
            corr = np.corrcoef(ret_a[:-lag], ret_b[lag:])[0,1]
        else:
            corr = np.corrcoef(ret_a[-lag:], ret_b[:lag])[0,1]
        if abs(corr) > abs(best_corr):
            best_corr = corr; best_lag = lag
    return best_lag, best_corr


def main():
    print("=" * 100)
    print("  DEEP QUANT V3: Statistical Physics & Probability Theory")
    print("=" * 100)

    print("\nDownloading data...")
    all_data = {}
    for sym in PAIRS:
        df = dl(sym)
        if len(df) > 200: all_data[sym] = df
    print(f"Loaded {len(all_data)} pairs\n")

    # ═══ Pre-analyze all pairs ═══
    print("Pre-computing mathematical features...")
    pair_features = {}
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        v = df['volume'].values.astype(float)
        returns = np.diff(c) / c[:-1]
        
        H = hurst_exponent(returns[-200:])
        entropy = shannon_entropy(returns[-100:])
        ac1 = autocorrelation(returns, 1)
        ac4 = autocorrelation(returns, 4)
        vpin = vpin_proxy(c[-50:], v[-50:])
        break_mag = detect_structural_break(c, 50)
        pois_z, pois_ratio = poisson_volume_anomaly(v)
        
        pair_features[sym] = {
            'hurst': H, 'entropy': entropy,
            'ac1': ac1, 'ac4': ac4,
            'vpin': vpin, 'break': break_mag,
            'pois_z': pois_z, 'pois_ratio': pois_ratio
        }
        
        print(f"  {sym:6s}: H={H:.3f} entropy={entropy:.2f} ac1={ac1:+.3f} "
              f"VPIN={vpin:.3f} break={break_mag:.2f} poisZ={pois_z:.1f}")

    # ═══ Cross-correlation analysis ═══
    print("\nCross-correlation lead/lag analysis...")
    btc_close = all_data.get("BTC", pd.DataFrame()).get('close', pd.Series()).values.astype(float) if "BTC" in all_data else None
    if btc_close is not None:
        for sym in all_data:
            if sym == "BTC": continue
            alt_close = all_data[sym]['close'].values.astype(float)
            lag, corr = cross_correlation_lead(btc_close, alt_close)
            if abs(corr) > 0.15:
                leader = "BTC leads" if lag > 0 else f"{sym} leads"
                print(f"  BTC↔{sym:5s}: lag={lag:+d}h corr={corr:+.3f} ({leader})")

    edges = []

    # ═══ HURST-BASED STRATEGIES ═══
    print("\nTesting Hurst-based strategies...")
    
    # When Hurst < 0.4 = strongly mean-reverting → buy dips
    for h_thresh in [0.3, 0.35, 0.4]:
        def mk(ht):
            def f(c,h,l,v,o,i,s,ad):
                if i < 100: return False
                returns = np.diff(c[i-100:i+1]) / c[i-100:i]
                H = hurst_exponent(returns)
                # Mean reverting + price below recent average
                cur_ret = (c[i] - c[i-4]) / c[i-4] * 100
                return H < ht and cur_ret < -1  # Mean reverting + just dipped
            return f
        edges.append(test_edge(all_data, f"Hurst<{h_thresh} + dip >1% (mean revert)", mk(h_thresh)))

    # When Hurst > 0.6 = trending → follow momentum
    for h_thresh in [0.6, 0.65, 0.7]:
        def mk(ht):
            def f(c,h,l,v,o,i,s,ad):
                if i < 100: return False
                returns = np.diff(c[i-100:i+1]) / c[i-100:i]
                H = hurst_exponent(returns)
                cur_ret = (c[i] - c[i-4]) / c[i-4] * 100
                return H > ht and cur_ret > 1  # Trending + moving up
            return f
        edges.append(test_edge(all_data, f"Hurst>{h_thresh} + pump >1% (trend follow)", mk(h_thresh)))

    # ═══ ENTROPY-BASED ═══
    print("Testing entropy-based strategies...")
    
    # Low entropy = predictable market → signals are more reliable
    def low_entropy_oversold(c,h,l,v,o,i,s,ad):
        if i < 100: return False
        returns = np.diff(c[i-50:i+1]) / c[i-50:i]
        ent = shannon_entropy(returns)
        from deep_quant_v3 import hurst_exponent as he  # avoid circular
        cur_ret = (c[i] - c[i-8]) / c[i-8] * 100
        return ent < 2.5 and cur_ret < -3  # Low entropy + dip
    edges.append(test_edge(all_data, "Low entropy (<2.5) + 3% dip (predictable bounce)", low_entropy_oversold))

    # High entropy = chaos → stay out or play small
    def high_entropy_extreme(c,h,l,v,o,i,s,ad):
        if i < 100: return False
        returns = np.diff(c[i-50:i+1]) / c[i-50:i]
        ent = shannon_entropy(returns)
        cur_ret = (c[i] - c[i-8]) / c[i-8] * 100
        return ent > 3.5 and cur_ret < -5  # High chaos + big dip = massive overreaction
    edges.append(test_edge(all_data, "High entropy (>3.5) + 5% dip (chaos overreaction)", high_entropy_extreme))

    # ═══ AUTOCORRELATION ═══
    print("Testing autocorrelation strategies...")
    
    # Negative AC = anti-persistent → mean reversion is the play
    def neg_ac_buy_dip(c,h,l,v,o,i,s,ad):
        if i < 50: return False
        returns = np.diff(c[i-50:i+1]) / c[i-50:i]
        ac = autocorrelation(returns, 1)
        cur_ret = (c[i] - c[i-4]) / c[i-4] * 100
        return ac < -0.1 and cur_ret < -2  # Anti-persistent + dipped
    edges.append(test_edge(all_data, "Negative AC1 + 2% dip (anti-persistent bounce)", neg_ac_buy_dip))

    # Positive AC = persistent → ride the trend
    def pos_ac_follow(c,h,l,v,o,i,s,ad):
        if i < 50: return False
        returns = np.diff(c[i-50:i+1]) / c[i-50:i]
        ac = autocorrelation(returns, 1)
        cur_ret = (c[i] - c[i-4]) / c[i-4] * 100
        return ac > 0.1 and cur_ret > 1  # Persistent + moving up
    edges.append(test_edge(all_data, "Positive AC1 + 1% pump (persistent follow)", pos_ac_follow))

    # ═══ VPIN (TOXICITY) ═══
    print("Testing VPIN/toxicity strategies...")
    
    # High VPIN = informed traders active → big move coming
    for vpin_thresh in [0.5, 0.6, 0.7, 0.8]:
        def mk(vt):
            def f(c,h,l,v,o,i,s,ad):
                if i < 30: return False
                vp = vpin_proxy(c[i-30:i+1], v[i-30:i+1])
                cur_ret = (c[i] - c[i-1]) / c[i-1] * 100
                return vp > vt and cur_ret > 0  # Toxic flow + direction = follow
            return f
        edges.append(test_edge(all_data, f"VPIN>{vpin_thresh} + green (smart money follow)", mk(vpin_thresh)))

    # High VPIN + red = informed selling → short
    for vpin_thresh in [0.5, 0.6, 0.7]:
        def mk(vt):
            def f(c,h,l,v,o,i,s,ad):
                if i < 30: return False
                vp = vpin_proxy(c[i-30:i+1], v[i-30:i+1])
                cur_ret = (c[i] - c[i-1]) / c[i-1] * 100
                return vp > vt and cur_ret < 0
            return f
        edges.append(test_edge(all_data, f"VPIN>{vpin_thresh} + red (smart money sell)", mk(vpin_thresh)))

    # ═══ POISSON VOLUME ═══
    print("Testing Poisson volume anomalies...")
    
    # Volume Z-score > 3 (Poisson) = something unusual happening
    for pz in [2, 3, 4, 5]:
        def mk(threshold):
            def f(c,h,l,v,o,i,s,ad):
                if i < 60: return False
                z, ratio = poisson_volume_anomaly(v[:i+1])
                return z > threshold and c[i] < o[i]  # Anomalous volume + red = capitulation
            return f
        edges.append(test_edge(all_data, f"Poisson vol Z>{pz} + red (anomalous selling)", mk(pz)))

    for pz in [2, 3, 4]:
        def mk(threshold):
            def f(c,h,l,v,o,i,s,ad):
                if i < 60: return False
                z, ratio = poisson_volume_anomaly(v[:i+1])
                return z > threshold and c[i] > o[i]  # Anomalous vol + green = smart buying
            return f
        edges.append(test_edge(all_data, f"Poisson vol Z>{pz} + green (anomalous buying)", mk(pz)))

    # ═══ STRUCTURAL BREAK ═══
    print("Testing structural break strategies...")
    
    for break_thresh in [1.5, 2.0, 2.5, 3.0]:
        # Downward break = buy the overreaction
        def mk(bt):
            def f(c,h,l,v,o,i,s,ad):
                if i < 110: return False
                brk = detect_structural_break(c[:i+1], 50)
                cur_ret = (c[i] - c[i-24]) / c[i-24] * 100
                return brk > bt and cur_ret < -5  # Big structural break + down = buy
            return f
        edges.append(test_edge(all_data, f"Structural break >{break_thresh}σ + down (regime buy)", mk(break_thresh)))

    # ═══ COMBINED MATH SIGNALS ═══
    print("Testing combined mathematical signals...")
    
    # The holy grail: mean-reverting (Hurst<0.5) + low entropy + oversold + high volume
    def holy_grail_buy(c,h,l,v,o,i,s,ad):
        if i < 100: return False
        returns = np.diff(c[i-100:i+1]) / c[i-100:i]
        H = hurst_exponent(returns[-50:])
        ent = shannon_entropy(returns[-30:])
        ac = autocorrelation(returns, 1)
        avg_vol = np.mean(v[i-20:i])
        vol_spike = avg_vol > 0 and v[i] > avg_vol * 2
        cur_ret = (c[i] - c[i-8]) / c[i-8] * 100
        return H < 0.45 and ent < 3.0 and ac < 0 and cur_ret < -3 and vol_spike
    edges.append(test_edge(all_data, "HOLY GRAIL: H<0.45 + low entropy + neg AC + dip + vol", holy_grail_buy))

    # Trending + persistent + momentum
    def trend_grail(c,h,l,v,o,i,s,ad):
        if i < 100: return False
        returns = np.diff(c[i-100:i+1]) / c[i-100:i]
        H = hurst_exponent(returns[-50:])
        ac = autocorrelation(returns, 1)
        cur_ret = (c[i] - c[i-8]) / c[i-8] * 100
        sma = np.mean(c[i-50:i])
        return H > 0.55 and ac > 0.05 and cur_ret > 2 and c[i] > sma
    edges.append(test_edge(all_data, "TREND GRAIL: H>0.55 + pos AC + pump + above SMA", trend_grail))

    # ═══ BTC LEADS ALT (with lag) ═══
    print("Testing BTC-lead signals...")
    
    def btc_just_pumped_alt_buy(c,h,l,v,o,i,s,ad):
        if s == "BTC" or "BTC" not in ad: return False
        btc = ad["BTC"]['close'].values
        if i >= len(btc) or i < 4: return False
        # BTC pumped in last 2h, this alt hasn't caught up yet
        btc_ret_2h = (btc[i] - btc[i-2]) / btc[i-2] * 100
        alt_ret_2h = (c[i] - c[i-2]) / c[i-2] * 100
        return btc_ret_2h > 1.5 and alt_ret_2h < 0.5  # BTC up, alt flat
    edges.append(test_edge(all_data, "BTC pumped 1.5% 2h, alt flat (lead-lag buy)", btc_just_pumped_alt_buy))

    def btc_just_dumped_alt_short(c,h,l,v,o,i,s,ad):
        if s == "BTC" or "BTC" not in ad: return False
        btc = ad["BTC"]['close'].values
        if i >= len(btc) or i < 4: return False
        btc_ret_2h = (btc[i] - btc[i-2]) / btc[i-2] * 100
        alt_ret_2h = (c[i] - c[i-2]) / c[i-2] * 100
        return btc_ret_2h < -1.5 and alt_ret_2h > -0.5
    edges.append(test_edge(all_data, "BTC dumped 1.5% 2h, alt flat (lead-lag short)", btc_just_dumped_alt_short))

    # ═══ MARKOV STATE TRANSITIONS ═══
    print("Testing Markov transitions...")
    
    # After 3 down hours + 1 up hour = continuation more likely?
    def markov_3down_1up(c,h,l,v,o,i,s,ad):
        if i < 5: return False
        return (c[i-4]<c[i-5] and c[i-3]<c[i-4] and c[i-2]<c[i-3]  # 3 down
                and c[i-1]>c[i-2]  # 1 up (dead cat)
                and c[i]<c[i-1])   # Current down again = pattern complete
    edges.append(test_edge(all_data, "Markov: 3 down→1 up→down (dead cat short)", markov_3down_1up))

    # After sustained down, first double green = reversal
    def markov_sustained_reversal(c,h,l,v,o,i,s,ad):
        if i < 10: return False
        # At least 6 of last 10 bars were red
        red_count = sum(1 for j in range(i-10, i-2) if c[j] < c[j-1])
        # Last 2 bars are both green
        double_green = c[i] > c[i-1] and c[i-1] > c[i-2]
        return red_count >= 6 and double_green
    edges.append(test_edge(all_data, "Markov: 6+/10 red then double green (reversal)", markov_sustained_reversal))

    # ═══ PRINT RESULTS ═══
    edges = [e for e in edges if e is not None]
    edges.sort(key=lambda e: e['best8'], reverse=True)

    print(f"\n{'='*100}")
    print(f"  {len(edges)} MATHEMATICAL PATTERNS TESTED")
    print(f"{'='*100}")

    print(f"\n  {'Pattern':<60s} {'N':>5s} {'L8h':>6s} {'S8h':>6s} {'L24h':>7s} {'WR8':>5s} {'Dir':>5s}")
    print("  "+"-"*95)

    profitable = []
    for e in edges:
        best=e['best8']
        color="\033[92m" if best>FEE else ("\033[93m" if best>0 else "\033[91m")
        R="\033[0m"
        print(f"  {e['name']:<60s} {e['n']:>5d} {color}{e['L8']:>+5.2f}% {e['S8']:>+5.2f}% {e['L24']:>+6.2f}%{R} {e['wr8']:>4.0f}% {e['dir']:>5s}")
        if best>FEE and e['n']>=20:
            profitable.append(e)

    print(f"\n{'='*100}")
    print(f"  PROFITABLE MATHEMATICAL EDGES ({len(profitable)})")
    print(f"{'='*100}")
    for e in profitable:
        net=e['best8']-FEE
        print(f"  ✅ {e['name']}")
        print(f"     N={e['n']} | {e['dir']} | 8h={e['best8']:+.2f}% net={net:+.2f}% | 24h={e['L24']:+.2f}% | WR={e['wr8']:.0f}%\n")

if __name__=="__main__":
    main()
