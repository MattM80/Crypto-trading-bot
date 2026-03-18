#!/usr/bin/env python3
"""
DEEP QUANT V2: 200+ patterns. Go insane.
Cross-asset signals, sequential patterns, whale detection,
volatility regimes, funding proxies, everything.
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
        r = requests.get(f"{CC_BASE}/histohour", params={"fsym": sym, "tsym": "USD", "limit": n}, timeout=30)
        d = r.json().get("Data",{}).get("Data",[])
        rows = [{'time':x['time'],'open':x['open'],'high':x['high'],'low':x['low'],
                 'close':x['close'],'volume':x.get('volumeto',0)} for x in d if x.get('close',0)>0]
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

def rsi(c, p=7):
    s=pd.Series(c); d=s.diff(); g=d.where(d>0,0); l=-d.where(d<0,0)
    ag=g.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    al=l.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    return (100-100/(1+ag/al.replace(0,np.nan))).values

def test_edge(all_data, name, cond_fn, min_n=20):
    results = []
    for sym, df in all_data.items():
        c=df['close'].values.astype(float); h=df['high'].values.astype(float)
        l=df['low'].values.astype(float); v=df['volume'].values.astype(float)
        o=df['open'].values.astype(float)
        for i in range(60, len(c)-25):
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

def main():
    print("Downloading data..."); all_data={}
    for sym in PAIRS:
        df=dl(sym)
        if len(df)>200: all_data[sym]=df
    print(f"Loaded {len(all_data)} pairs. Mining patterns...\n")

    edges=[]
    
    # ═══ SEQUENTIAL PATTERNS ═══
    print("Sequential patterns...")
    # N red then green (reversal confirmation)
    for n_red in [3,4,5,6,7]:
        def mk(nr):
            def f(c,h,l,v,o,i,s,ad):
                if i<nr+1: return False
                for j in range(1,nr+1):
                    if c[i-j]>=c[i-j-1]: return False  # Must be red
                return c[i]>o[i]  # Current candle is green (reversal)
            return f
        edges.append(test_edge(all_data, f"{n_red} red then green candle (reversal)", mk(n_red)))

    # N green then red
    for n_green in [3,4,5,6,7]:
        def mk(ng):
            def f(c,h,l,v,o,i,s,ad):
                if i<ng+1: return False
                for j in range(1,ng+1):
                    if c[i-j]<=c[i-j-1]: return False
                return c[i]<o[i]
            return f
        edges.append(test_edge(all_data, f"{n_green} green then red candle (top)", mk(n_green)))

    # ═══ WHALE DETECTION (volume anomalies) ═══
    print("Whale detection...")
    for vol_x in [2, 3, 5, 8, 10]:
        # Huge volume spike
        def mk(vx):
            def f(c,h,l,v,o,i,s,ad):
                if i<20: return False
                avg=np.mean(v[i-20:i])
                return avg>0 and v[i]>avg*vx
            return f
        edges.append(test_edge(all_data, f"Volume {vol_x}x spike", mk(vol_x)))
        
        # Huge vol + green candle (whale buying)
        def mk2(vx):
            def f(c,h,l,v,o,i,s,ad):
                if i<20: return False
                avg=np.mean(v[i-20:i])
                return avg>0 and v[i]>avg*vx and c[i]>o[i]
            return f
        edges.append(test_edge(all_data, f"Volume {vol_x}x + green (whale buy)", mk2(vol_x)))
        
        # Huge vol + red candle (whale selling / capitulation)
        def mk3(vx):
            def f(c,h,l,v,o,i,s,ad):
                if i<20: return False
                avg=np.mean(v[i-20:i])
                return avg>0 and v[i]>avg*vx and c[i]<o[i]
            return f
        edges.append(test_edge(all_data, f"Volume {vol_x}x + red (capitulation)", mk3(vol_x)))

    # ═══ MULTI-ASSET MOMENTUM ═══
    print("Cross-asset momentum...")
    # When majority of coins are dropping (market panic)
    for pct_dropping in [60, 70, 80, 90]:
        for drop_thresh in [1, 2, 3]:
            def mk(pd_pct, dt):
                def f(c,h,l,v,o,i,s,ad):
                    if i<4: return False
                    dropping=0; total=0
                    for sym2,df2 in ad.items():
                        cl=df2['close'].values
                        if i<len(cl) and i>=4:
                            ret=(cl[i]-cl[i-4])/cl[i-4]*100
                            total+=1
                            if ret<-dt: dropping+=1
                    return total>=5 and dropping/total*100>=pd_pct
                return f
            edges.append(test_edge(all_data, f"{pct_dropping}% coins down >{drop_thresh}% 4h (panic buy)", mk(pct_dropping, drop_thresh)))

    # When majority pumping (FOMO)
    for pct_pumping in [60, 70, 80]:
        def mk(pp):
            def f(c,h,l,v,o,i,s,ad):
                if i<4: return False
                pumping=0; total=0
                for sym2,df2 in ad.items():
                    cl=df2['close'].values
                    if i<len(cl) and i>=4:
                        ret=(cl[i]-cl[i-4])/cl[i-4]*100
                        total+=1
                        if ret>1: pumping+=1
                return total>=5 and pumping/total*100>=pp
            return f
        edges.append(test_edge(all_data, f"{pct_pumping}% coins up >1% 4h (ride FOMO)", mk(pct_pumping)))

    # ═══ VOLATILITY REGIME SIGNALS ═══
    print("Volatility regimes...")
    # Low vol → high vol transition (breakout)
    def vol_expansion(c,h,l,v,o,i,s,ad):
        if i<30: return False
        pc=np.roll(c[:i+1],1); pc[0]=c[0]
        tr=np.maximum(h[:i+1]-l[:i+1], np.maximum(np.abs(h[:i+1]-pc),np.abs(l[:i+1]-pc)))
        recent=np.mean(tr[i-5:i+1]); prior=np.mean(tr[i-20:i-5])
        return prior>0 and recent>prior*2  # ATR doubled
    edges.append(test_edge(all_data, "ATR doubled in 5 bars (vol expansion)", vol_expansion))

    # Very low vol (squeeze about to break)
    def vol_squeeze(c,h,l,v,o,i,s,ad):
        if i<30: return False
        ranges = (h[i-10:i+1]-l[i-10:i+1])/c[i-10:i+1]*100
        avg_range = np.mean(ranges)
        return avg_range < 0.5  # Very tight candles
    edges.append(test_edge(all_data, "10-bar avg range <0.5% (squeeze)", vol_squeeze))

    # ═══ PRICE STRUCTURE ═══
    print("Price structure patterns...")
    # Higher low + higher high (trend confirmation)
    def higher_lows(c,h,l,v,o,i,s,ad):
        if i<12: return False
        # 3 consecutive higher lows over 12 bars
        l1=np.min(l[i-12:i-8]); l2=np.min(l[i-8:i-4]); l3=np.min(l[i-4:i+1])
        return l3>l2>l1
    edges.append(test_edge(all_data, "3 higher lows (12 bars, trend forming)", higher_lows))

    def lower_highs(c,h,l,v,o,i,s,ad):
        if i<12: return False
        h1=np.max(h[i-12:i-8]); h2=np.max(h[i-8:i-4]); h3=np.max(h[i-4:i+1])
        return h3<h2<h1
    edges.append(test_edge(all_data, "3 lower highs (12 bars, breakdown)", lower_highs))

    # Hammer candle (long wick down, small body, at bottom)
    def hammer(c,h,l,v,o,i,s,ad):
        if i<10: return False
        body=abs(c[i]-o[i]); rng=h[i]-l[i]
        if rng<=0: return False
        lower_wick=min(c[i],o[i])-l[i]
        upper_wick=h[i]-max(c[i],o[i])
        # Hammer: lower wick > 2x body, upper wick small
        is_hammer = lower_wick > body*2 and upper_wick < body*0.5 and rng>0
        # At bottom (RSI<30 or price below 10-bar SMA)
        at_bottom = c[i] < np.mean(c[i-10:i])
        return is_hammer and at_bottom
    edges.append(test_edge(all_data, "Hammer candle at bottom", hammer))

    # Shooting star (long wick up at top)
    def shooting_star(c,h,l,v,o,i,s,ad):
        if i<10: return False
        body=abs(c[i]-o[i]); rng=h[i]-l[i]
        if rng<=0: return False
        upper_wick=h[i]-max(c[i],o[i])
        lower_wick=min(c[i],o[i])-l[i]
        is_star = upper_wick > body*2 and lower_wick < body*0.5
        at_top = c[i] > np.mean(c[i-10:i])
        return is_star and at_top
    edges.append(test_edge(all_data, "Shooting star at top", shooting_star))

    # ═══ BTC DOMINANCE PROXY ═══
    print("BTC dominance signals...")
    # BTC outperforming all alts (dominance rising = risk off)
    def btc_outperform_all(c,h,l,v,o,i,s,ad):
        if s!="BTC" or i<8: return False
        btc_ret=(c[i]-c[i-8])/c[i-8]*100
        outperform=0; total=0
        for sym2,df2 in ad.items():
            if sym2=="BTC": continue
            cl=df2['close'].values
            if i<len(cl) and i>=8:
                alt_ret=(cl[i]-cl[i-8])/cl[i-8]*100
                total+=1
                if btc_ret>alt_ret: outperform+=1
        return total>=5 and outperform/total>0.8  # BTC beating 80%+ of alts
    edges.append(test_edge(all_data, "BTC outperforming 80%+ alts 8h (dominance)", btc_outperform_all))

    # Alts outperforming BTC (alt season starting)
    def alts_outperform(c,h,l,v,o,i,s,ad):
        if s=="BTC" or "BTC" not in ad or i<8: return False
        btc=ad["BTC"]['close'].values
        if i>=len(btc): return False
        btc_ret=(btc[i]-btc[i-8])/btc[i-8]*100
        alt_ret=(c[i]-c[i-8])/c[i-8]*100
        # This specific alt is beating BTC by 3%+
        return alt_ret - btc_ret > 3
    edges.append(test_edge(all_data, "Alt beating BTC by >3% 8h (alt season)", alts_outperform))

    # ═══ MEAN REVERSION: EXTREME Z-SCORES ═══
    print("Z-score extremes...")
    for lookback in [24, 48, 96]:
        for z_thresh in [2, 2.5, 3]:
            def mk(lb, zt):
                def f(c,h,l,v,o,i,s,ad):
                    if i<lb: return False
                    window=c[i-lb:i]
                    mu=np.mean(window); sigma=np.std(window)
                    if sigma<=0: return False
                    z=(c[i]-mu)/sigma
                    return z < -zt  # Extremely below mean
                return f
            edges.append(test_edge(all_data, f"Z-score < -{z_thresh} ({lookback}h lookback)", mk(lookback, z_thresh)))

    # ═══ VOLUME-PRICE DIVERGENCE ═══
    print("Volume-price divergence...")
    # Price up but volume declining (weak rally)
    def price_up_vol_down(c,h,l,v,o,i,s,ad):
        if i<10: return False
        price_up = c[i]>c[i-5]>c[i-10]
        vol_down = np.mean(v[i-5:i+1]) < np.mean(v[i-10:i-5])*0.7
        return price_up and vol_down
    edges.append(test_edge(all_data, "Price rising + volume declining (weak rally → short)", price_up_vol_down))

    # Price down but volume declining (selling exhaustion)
    def price_down_vol_down(c,h,l,v,o,i,s,ad):
        if i<10: return False
        price_down = c[i]<c[i-5]<c[i-10]
        vol_down = np.mean(v[i-5:i+1]) < np.mean(v[i-10:i-5])*0.7
        return price_down and vol_down
    edges.append(test_edge(all_data, "Price falling + volume declining (exhaustion → long)", price_down_vol_down))

    # ═══ COMBINED SIGNALS ═══
    print("Combined signals...")
    # RSI oversold + hammer + volume spike (triple confirmation)
    def triple_bottom(c,h,l,v,o,i,s,ad):
        if i<20: return False
        r=rsi(c[:i+1],7)
        if np.isnan(r[-1]): return False
        oversold = r[-1]<25
        body=abs(c[i]-o[i]); rng=h[i]-l[i]
        is_hammer = rng>0 and (min(c[i],o[i])-l[i])>body*1.5
        avg_vol=np.mean(v[i-20:i])
        vol_spike = avg_vol>0 and v[i]>avg_vol*1.5
        return oversold and is_hammer and vol_spike
    edges.append(test_edge(all_data, "RSI<25 + hammer + vol spike (triple buy)", triple_bottom))

    # RSI overbought + shooting star + volume spike
    def triple_top(c,h,l,v,o,i,s,ad):
        if i<20: return False
        r=rsi(c[:i+1],7)
        if np.isnan(r[-1]): return False
        overbought = r[-1]>75
        body=abs(c[i]-o[i]); rng=h[i]-l[i]
        is_star = rng>0 and (h[i]-max(c[i],o[i]))>body*1.5
        avg_vol=np.mean(v[i-20:i])
        vol_spike = avg_vol>0 and v[i]>avg_vol*1.5
        return overbought and is_star and vol_spike
    edges.append(test_edge(all_data, "RSI>75 + shooting star + vol spike (triple sell)", triple_top))

    # Market panic + specific coin oversold (best entry)
    def panic_plus_oversold(c,h,l,v,o,i,s,ad):
        if i<24: return False
        # 70%+ coins down >2% in 4h
        dropping=0; total=0
        for sym2,df2 in ad.items():
            cl=df2['close'].values
            if i<len(cl) and i>=4:
                ret=(cl[i]-cl[i-4])/cl[i-4]*100
                total+=1
                if ret<-2: dropping+=1
        market_panic = total>=5 and dropping/total>=0.7
        # This specific coin RSI<20
        r=rsi(c[:i+1],7)
        coin_oversold = not np.isnan(r[-1]) and r[-1]<20
        return market_panic and coin_oversold
    edges.append(test_edge(all_data, "70% market panic + RSI<20 (blood in streets)", panic_plus_oversold))

    # ═══ PRINT RESULTS ═══
    edges = [e for e in edges if e is not None]
    edges.sort(key=lambda e: e['best8'], reverse=True)

    print(f"\n{'='*95}")
    print(f"  {len(edges)} PATTERNS TESTED — Sorted by 8h return")
    print(f"{'='*95}")
    print(f"\n  {'Pattern':<55s} {'N':>5s} {'L8h':>6s} {'S8h':>6s} {'L24h':>7s} {'WR8':>5s} {'Dir':>5s}")
    print("  "+"-"*85)

    profitable = []
    for e in edges:
        best = e['best8']
        color = "\033[92m" if best>FEE else ("\033[93m" if best>0 else "\033[91m")
        R = "\033[0m"
        print(f"  {e['name']:<55s} {e['n']:>5d} {color}{e['L8']:>+5.2f}% {e['S8']:>+5.2f}% {e['L24']:>+6.2f}%{R} {e['wr8']:>4.0f}% {e['dir']:>5s}")
        if best > FEE and e['n'] >= 20:
            profitable.append(e)

    print(f"\n{'='*95}")
    print(f"  NEW PROFITABLE TOOLS ({len(profitable)})")
    print(f"{'='*95}")
    for e in profitable:
        net=e['best8']-FEE
        print(f"  ✅ {e['name']}")
        print(f"     N={e['n']} | {e['dir']} | 8h={e['best8']:+.2f}% net={net:+.2f}% | 24h={e['L24']:+.2f}% | WR={e['wr8']:.0f}%\n")

if __name__=="__main__":
    main()
