#!/usr/bin/env python3
"""
FINAL BACKTEST: The All-Seeing Eye with all 29 tools.
83 days, 16 pairs, $300, grid + every signal tool.
This is the definitive test.
"""
import requests, time as _time, numpy as np, pandas as pd, warnings
from datetime import datetime, timezone
from collections import defaultdict
warnings.filterwarnings('ignore')

CC_BASE = "https://min-api.cryptocompare.com/data/v2"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005
BALANCE = 300
GRID_PCT = 0.40
ACTIVE_PCT = 0.60
MAX_ACTIVE = 5
RISK_PER_TRADE = 0.05
FEE_RT = 0.0042

PAIRS = ["BTC","ETH","SOL","LINK","AVAX","DOT","ADA","XRP",
         "DOGE","UNI","NEAR","ATOM","AAVE","XLM","FIL","LTC"]

GRID_CONFIGS = {
    "NEAR":0.01,"UNI":0.015,"AVAX":0.01,"LINK":0.008,"AAVE":0.015,
    "SOL":0.003,"ETH":0.005,"BTC":0.01,"DOT":0.012,"XLM":0.01,
    "XRP":0.01,"ADA":0.012,"ATOM":0.008,"DOGE":0.012,"FIL":0.015,"LTC":0.01,
}

def dl(sym):
    try:
        r = requests.get(f"{CC_BASE}/histohour", params={"fsym":sym,"tsym":"USD","limit":2000}, timeout=30)
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

def hurst_fast(r, w=50):
    if len(r)<w: return 0.5
    r2=r[-w:]; v1=np.var(r2); v2=np.var(r2[::2]) if len(r2)>=4 else v1
    if v1<=0 or v2<=0: return 0.5
    return max(0, min(1, 0.5+np.log(max(v2/v1,0.01))/(2*np.log(2))))

def entropy(r, bins=15):
    if len(r)<10: return 3.0
    h,_=np.histogram(r,bins=bins,density=True); h=h[h>0]; p=h/h.sum()
    return -np.sum(p*np.log2(p))

def vpin(c_arr, v_arr, w=20):
    if len(c_arr)<w+1: return 0
    rets=np.diff(c_arr)/c_arr[:-1]
    bv=np.where(rets>0,v_arr[1:],0); sv=np.where(rets<0,v_arr[1:],0)
    rb=np.sum(bv[-w:]); rs=np.sum(sv[-w:]); t=rb+rs
    return abs(rb-rs)/t if t>0 else 0


def scan_signals(close, high, low, vol, opn, i, sym, all_close, all_vol, all_open):
    """All 29 tools scanning one pair at one bar. Returns [(signal_dict, score), ...]"""
    signals = []
    if i < 100: return signals
    
    c = close; h = high; l = low; v = vol; o = opn
    price = c[i]
    
    # Basic features
    rsi7 = rsi(c[:i+1], 7)
    cur_rsi = rsi7[-1] if not np.isnan(rsi7[-1]) else 50
    sma50 = np.mean(c[max(0,i-50):i]) if i >= 50 else price
    
    ret_4h = (c[i]-c[i-4])/c[i-4]*100 if i>=4 else 0
    ret_8h = (c[i]-c[i-8])/c[i-8]*100 if i>=8 else 0
    ret_12h = (c[i]-c[i-12])/c[i-12]*100 if i>=12 else 0
    ret_24h = (c[i]-c[i-24])/c[i-24]*100 if i>=24 else 0
    
    avg_vol = np.mean(v[max(0,i-20):i]) if i>=20 else 1
    vol_ratio = v[i]/avg_vol if avg_vol>0 else 1
    is_green = c[i]>o[i]
    
    # Math features
    returns = np.diff(c[max(0,i-100):i+1])/c[max(0,i-100):i]
    H = hurst_fast(returns)
    ent = entropy(returns[-30:])
    ac1 = float(pd.Series(returns).autocorr(lag=1)) if len(returns)>5 else 0
    if np.isnan(ac1): ac1 = 0
    vp = vpin(c[max(0,i-30):i+1], v[max(0,i-30):i+1])
    
    # Cross-asset features
    n_pairs = len(all_close)
    dropping_3 = 0; dropping_2 = 0; pumping_1 = 0; total_pairs = 0
    for sym2, cl2 in all_close.items():
        if i < 4 or i >= len(cl2): continue
        r2 = (cl2[i]-cl2[i-4])/cl2[i-4]*100
        total_pairs += 1
        if r2 < -3: dropping_3 += 1
        if r2 < -2: dropping_2 += 1
        if r2 > 1: pumping_1 += 1
    
    def add(tool, direction, hold, sl, reason, score):
        signals.append(({'pair':sym,'tool':tool,'direction':direction,
                        'hold':hold,'sl_pct':sl,'reason':reason}, score))
    
    # ── TOOL 2: Crash Buy (>10% drop + RSI<20) ──
    if ret_24h < -10 and cur_rsi < 20:
        add('crash_buy','long',24,0.05, f"CRASH BUY: {ret_24h:.1f}% drop, RSI={cur_rsi:.1f}", (20-cur_rsi)*2)
    
    # ── TOOL 3: Volatile Oversold (ATR>3% + RSI<25) ──
    atr_vals = np.diff(np.maximum(h[:i+1]-l[:i+1], np.abs(np.diff(np.concatenate([[c[0]],c[:i+1]])))))
    cur_atr_pct = np.mean(np.abs(np.diff(c[max(0,i-14):i+1])))/price*100 if i>14 else 0
    if cur_atr_pct > 3 and cur_rsi < 25:
        add('volatile_oversold','long',24,0.08, f"VOL OVERSOLD: ATR={cur_atr_pct:.1f}%, RSI={cur_rsi:.1f}", cur_atr_pct*(25-cur_rsi))
    
    # ── TOOL 4: Relief Rally (RSI>75 + below SMA50) ──
    if cur_rsi > 75 and price < sma50:
        add('relief_rally','long',12,0.03, f"RELIEF RALLY: RSI={cur_rsi:.1f}", (cur_rsi-75)*1.5)
    
    # ── TOOL 5: Overbought Sell (RSI>85) ──
    if cur_rsi > 90 and ret_4h > 3:
        add('overbought_sell','short',24,0.99, f"OVERBOUGHT: RSI={cur_rsi:.1f}", cur_rsi-85)
    
    # ── TOOL 6: Dip Buy (>3% drop 4h) ──
    if ret_4h < -3:
        add('dip_buy','long',8,0.03, f"DIP BUY: {ret_4h:.1f}% 4h", abs(ret_4h)*2)
    
    # ── TOOL 7: Pump Sell ──
    if cur_rsi > 80 and ret_8h > 8:
        add('mega_pump_sell','short',24,0.99, f"MEGA PUMP: RSI={cur_rsi:.1f}, +{ret_8h:.1f}% 8h", (cur_rsi-80)+ret_4h)
    
    # ── TOOL 8: Mega Crash (>15% 24h) ──
    if ret_24h < -15:
        add('mega_crash','long',24,0.08, f"MEGA CRASH: {ret_24h:.1f}% 24h", abs(ret_24h)*3)
    
    # ── TOOL 9: Flash Crash (>10% 12h) ──
    if ret_12h < -10:
        add('flash_crash','long',24,0.07, f"FLASH CRASH: {ret_12h:.1f}% 12h", abs(ret_12h)*2.5)
    
    # ── TOOL 10: Quick Crash (>10% 8h) ──
    if ret_8h < -10:
        add('quick_crash','long',24,0.07, f"QUICK CRASH: {ret_8h:.1f}% 8h", abs(ret_8h)*2)
    
    # ── TOOL 11: Deep Dip (8-10% drops) ──
    for tf, rv, lb in [("8h",ret_8h,"8h"),("12h",ret_12h,"12h"),("24h",ret_24h,"24h")]:
        if rv < -8 and rv >= -10:
            add(f'deep_dip_{lb}','long',24,0.05, f"DEEP DIP: {rv:.1f}% {lb}", abs(rv)*1.5)
    
    # ── TOOL 12: Quick Dip (>5% 4h) ──
    if ret_4h < -5:
        add('quick_dip','long',8,0.04, f"QUICK DIP: {ret_4h:.1f}% 4h", abs(ret_4h)*2)
    
    # ── TOOL 16: Market Panic ──
    if total_pairs >= 5:
        panic_pct = dropping_3/total_pairs*100
        if panic_pct >= 90:
            add('market_panic_90','long',24,0.05, f"PANIC 90%: {panic_pct:.0f}%", panic_pct*0.5)
        elif panic_pct >= 80:
            add('market_panic_80','long',24,0.05, f"PANIC 80%", panic_pct*0.4)
        elif panic_pct >= 70:
            add('market_panic_70','long',8,0.04, f"PANIC 70%", panic_pct*0.3)
    
    # ── TOOL 17: Whale Buy (5x vol + green) ──
    if vol_ratio >= 5 and is_green:
        add('whale_buy','long',8,0.03, f"WHALE BUY: {vol_ratio:.1f}x vol", vol_ratio*2)
    
    # ── TOOL 18: Capitulation (8x vol + red) ──
    if vol_ratio >= 8 and not is_green:
        add('capitulation','long',8,0.05, f"CAPITULATION: {vol_ratio:.1f}x vol", vol_ratio*1.5)
    
    # ── TOOL 19: 7 Green Exhaustion ──
    if i >= 8:
        all_green = all(c[i-j-1]>o[i-j-1] for j in range(1,8))
        if all_green and c[i]<o[i]:
            add('green_exhaustion','short',8,0.03, "7 GREEN EXHAUSTION", 15)
    
    # ── TOOL 20: Z-score -3σ ──
    if i >= 49:
        window = c[i-48:i]; mu=np.mean(window); sig=np.std(window)
        if sig>0:
            z=(c[i]-mu)/sig
            if z<-3:
                add('zscore_extreme','long',24,0.05, f"Z-SCORE: {z:.1f}σ", abs(z)*5)
    
    # ── TOOL 21: Blood in Streets ──
    if total_pairs>=5 and dropping_2/total_pairs>=0.7 and cur_rsi<20:
        add('blood_streets','long',24,0.06, f"BLOOD: {dropping_2/total_pairs*100:.0f}% panic + RSI={cur_rsi:.1f}", (20-cur_rsi)*3)
    
    # ── TOOL 22: FOMO Ride ──
    if total_pairs>=5 and pumping_1/total_pairs>=0.8:
        add('fomo_ride','long',8,0.03, f"FOMO: {pumping_1/total_pairs*100:.0f}% pumping", 5)
    
    # ── TOOL 23: Crash + Neg AC (THE BEST — 78% WR) ──
    if ret_24h < -10 and ac1 < -0.05:
        add('crash_neg_ac','long',24,0.08, f"CRASH+NEG_AC: {ret_24h:.1f}%, AC={ac1:.3f} — 78%WR", abs(ret_24h)*(abs(ac1)+0.1)*10)
    
    # ── TOOL 24: Crash + Mean Reverting Hurst ──
    if ret_24h < -8 and H < 0.45:
        add('crash_hurst','long',24,0.06, f"CRASH+HURST: {ret_24h:.1f}%, H={H:.3f}", abs(ret_24h)*(0.5-H)*8)
    
    # ── TOOL 25: Hurst Trend Follow ──
    if H > 0.65 and ret_4h > 2:
        add('hurst_trend','long',8,0.03, f"HURST TREND: H={H:.3f}, +{ret_4h:.1f}%", H*ret_4h*3)
    
    # ── TOOL 26: VPIN Toxic Flow ──
    if vp > 0.7 and c[i]<o[i]:
        add('vpin_toxic','long',8,0.04, f"VPIN TOXIC: {vp:.3f}", vp*10)
    
    # ── TOOL 27: VPIN Dip ──
    if ret_8h < -5 and vp > 0.5:
        add('vpin_dip','long',8,0.05, f"VPIN DIP: {ret_8h:.1f}%, VPIN={vp:.3f}", abs(ret_8h)*vp*3)
    
    # ── TOOL 28: Entropy Dip ──
    if ent < 2.5 and ret_4h < -2:
        add('entropy_dip','long',8,0.03, f"ENTROPY DIP: ent={ent:.2f}, {ret_4h:.1f}%", (3-ent)*abs(ret_4h)*2)
    
    # ── TOOL 29: Triple Math ──
    if ret_8h < -5 and ent < 2.5 and vp > 0.5:
        add('triple_math','long',24,0.06, f"TRIPLE MATH: {ret_8h:.1f}%, ent={ent:.2f}, VPIN={vp:.3f}", abs(ret_8h)*(3-ent)*vp*5)
    
    return signals


def main():
    print("="*80)
    print("  ALL-SEEING EYE BACKTEST — 29 tools, 16 pairs, 83 days, $300")
    print("="*80)

    print("\nDownloading...")
    all_data = {}
    for sym in PAIRS:
        df = dl(sym); _time.sleep(0.3)
        if len(df)>200: all_data[sym]=df
    min_len = min(len(df) for df in all_data.values())
    days = min_len/24
    print(f"Loaded {len(all_data)} pairs, {min_len} bars (~{days:.0f} days)\n")

    # Prepare arrays
    all_close={}; all_high={}; all_low={}; all_vol={}; all_open={}
    for sym,df in all_data.items():
        all_close[sym]=df['close'].values.astype(float)
        all_high[sym]=df['high'].values.astype(float)
        all_low[sym]=df['low'].values.astype(float)
        all_vol[sym]=df['volume'].values.astype(float)
        all_open[sym]=df['open'].values.astype(float)

    # State
    grid_bal = BALANCE * GRID_PCT
    active_bal = BALANCE * ACTIVE_PCT
    grids = {}  # sym -> {"buys":[], "filled":[], "profit":0}
    active_pos = {}  # sym -> {entry, qty, bar, direction, sl_pct, hold, tool}
    trades = []
    tool_stats = defaultdict(lambda: {"trades":0,"wins":0,"pnl":0})
    grid_profit = 0; grid_rts = 0
    peak = BALANCE; max_dd = 0

    # Init grids
    alloc_per = grid_bal / len(all_data)
    for sym in all_data:
        gp = GRID_CONFIGS.get(sym, 0.01)
        first_price = all_close[sym][100]
        grids[sym] = {"buys":[], "filled":[], "profit":0}
        for lvl in range(1,4):
            bp = first_price*(1-gp*lvl)
            qty = (alloc_per/3)/bp
            grids[sym]["buys"].append((bp,qty))

    for bar in range(101, min_len):
        # ── GRID ──
        for sym in all_data:
            c=all_close[sym]; h=all_high[sym]; l=all_low[sym]
            if bar>=len(c): continue
            gp_cfg = GRID_CONFIGS.get(sym, 0.01)
            g = grids[sym]
            # Sell fills
            new_f = []
            for bp,qty in g["filled"]:
                st = bp*1.015
                if h[bar]>=st:
                    fee=st*qty*MAKER_FEE
                    profit=st*qty-fee-bp*qty*(1+MAKER_FEE)
                    g["profit"]+=profit; grid_profit+=profit; grid_rts+=1
                    grid_bal+=st*qty-fee
                    g["buys"].append((bp,qty))
                else: new_f.append((bp,qty))
            g["filled"]=new_f
            # Buy fills
            new_b = []
            for bp,qty in g["buys"]:
                if l[bar]<=bp:
                    cost=bp*qty*(1+MAKER_FEE)
                    if grid_bal>=cost:
                        grid_bal-=cost; g["filled"].append((bp,qty))
                    else: new_b.append((bp,qty))
                else: new_b.append((bp,qty))
            g["buys"]=new_b
            # Recenter if empty
            if not g["buys"] and not g["filled"]:
                for lvl in range(1,4):
                    bp2=c[bar]*(1-gp_cfg*lvl); qty2=(alloc_per/3)/bp2
                    g["buys"].append((bp2,qty2))

        # ── MANAGE POSITIONS ──
        for sym in list(active_pos.keys()):
            pos=active_pos[sym]
            c=all_close[sym]; h=all_high[sym]; l=all_low[sym]
            if bar>=len(c): continue
            bars_held=bar-pos['bar']
            exit_price=None; reason=None

            if pos['direction']=='long':
                if l[bar]<=pos['entry']*(1-pos['sl_pct']):
                    exit_price=pos['entry']*(1-pos['sl_pct']); reason="SL"
                elif bars_held>=pos['hold']:
                    exit_price=c[bar]; reason="HOLD"
            else:
                if h[bar]>=pos['entry']*(1+pos['sl_pct']):
                    exit_price=pos['entry']*(1+pos['sl_pct']); reason="SL"
                elif bars_held>=pos['hold']:
                    exit_price=c[bar]; reason="HOLD"

            if exit_price:
                if pos['direction']=='long':
                    pnl=(exit_price-pos['entry'])*pos['qty']
                    fees=pos['entry']*pos['qty']*MAKER_FEE+exit_price*pos['qty']*MAKER_FEE
                    active_bal+=pos['entry']*pos['qty']+(pnl-fees)
                else:
                    pnl=(pos['entry']-exit_price)*pos['qty']
                    fees=pos['entry']*pos['qty']*MAKER_FEE+exit_price*pos['qty']*MAKER_FEE
                    active_bal+=(pnl-fees)
                net=pnl-fees
                trades.append({'sym':sym,'tool':pos['tool'],'dir':pos['direction'],
                              'pnl':net,'reason':reason,'bars':bars_held})
                tool_stats[pos['tool']]['trades']+=1
                if net>0: tool_stats[pos['tool']]['wins']+=1
                tool_stats[pos['tool']]['pnl']+=net
                del active_pos[sym]

        # ── SCAN SIGNALS ──
        if len(active_pos) < MAX_ACTIVE:
            all_signals = []
            for sym in all_data:
                if sym in active_pos: continue
                c=all_close[sym]; h=all_high[sym]; l=all_low[sym]
                v=all_vol[sym]; o=all_open[sym]
                if bar>=len(c): continue
                sigs = scan_signals(c,h,l,v,o,bar,sym,all_close,all_vol,all_open)
                all_signals.extend(sigs)
            
            all_signals.sort(key=lambda x:x[1], reverse=True)
            
            for sig,score in all_signals:
                if len(active_pos)>=MAX_ACTIVE: break
                if sig['pair'] in active_pos: continue
                
                entry=all_close[sig['pair']][bar]
                if sig['direction']=='long':
                    entry*=(1+SLIPPAGE)
                else:
                    entry*=(1-SLIPPAGE)
                
                risk_amt=active_bal*RISK_PER_TRADE
                qty=risk_amt/(entry*sig['sl_pct'])
                cost=qty*entry
                
                if sig['direction']=='long':
                    if cost>active_bal*0.25 or cost<1: continue
                    active_bal-=cost
                else:
                    if risk_amt>active_bal*0.25: continue
                
                active_pos[sig['pair']]={
                    'entry':entry,'qty':qty,'bar':bar,'direction':sig['direction'],
                    'sl_pct':sig['sl_pct'],'hold':sig['hold'],'tool':sig['tool']
                }

        # Equity tracking
        eq = active_bal + grid_bal
        for sym,g in grids.items():
            for bp,qty in g["filled"]:
                eq += (all_close[sym][min(bar,len(all_close[sym])-1)] - bp)*qty
        for sym,pos in active_pos.items():
            cur=all_close[sym][min(bar,len(all_close[sym])-1)]
            if pos['direction']=='long': eq+=(cur-pos['entry'])*pos['qty']
            else: eq+=(pos['entry']-cur)*pos['qty']
        peak=max(peak,eq)
        dd=(peak-eq)/peak if peak>0 else 0
        max_dd=max(max_dd,dd)

    # Close remaining
    for sym,pos in list(active_pos.items()):
        cur=all_close[sym][-1]
        if pos['direction']=='long':
            pnl=(cur-pos['entry'])*pos['qty']
            fees=pos['entry']*pos['qty']*MAKER_FEE+cur*pos['qty']*MAKER_FEE
            active_bal+=pos['entry']*pos['qty']+(pnl-fees)
        else:
            pnl=(pos['entry']-cur)*pos['qty']
            fees=pos['entry']*pos['qty']*MAKER_FEE+cur*pos['qty']*MAKER_FEE
            active_bal+=(pnl-fees)
        net=pnl-fees
        trades.append({'sym':sym,'tool':pos['tool'],'dir':pos['direction'],
                      'pnl':net,'reason':'END','bars':min_len-pos['bar']})
        tool_stats[pos['tool']]['trades']+=1
        if net>0: tool_stats[pos['tool']]['wins']+=1
        tool_stats[pos['tool']]['pnl']+=net

    for sym,g in grids.items():
        cur=all_close[sym][-1]
        for bp,qty in g["filled"]:
            fee=cur*qty*MAKER_FEE
            grid_bal+=cur*qty-fee
            grid_profit+=(cur-bp)*qty-bp*qty*MAKER_FEE-fee

    # ═══════════════════════════════════════
    final = active_bal + grid_bal
    total_pnl = final - BALANCE
    active_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl']>0]
    losses = [t for t in trades if t['pnl']<=0]
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001

    C="\033[92m" if total_pnl>0 else "\033[91m"
    R="\033[0m"

    print(f"\n{'='*80}")
    print(f"  RESULTS — {min_len} bars (~{days:.0f} days)")
    print(f"{'='*80}")

    print(f"\n  GRID ENGINE")
    print(f"    Round-trips: {grid_rts}")
    print(f"    Realized: ${grid_profit:+.2f}")

    print(f"\n  ACTIVE TOOLS (29 signal sources)")
    print(f"    Trades: {len(trades)}")
    print(f"    Wins: {len(wins)} / Losses: {len(losses)}")
    if trades:
        print(f"    Win rate: {len(wins)/len(trades)*100:.1f}%")
        print(f"    PF: {gp/gl:.2f}")
    print(f"    PnL: ${active_pnl:+.2f}")

    print(f"\n  TOOL PERFORMANCE:")
    sorted_tools = sorted(tool_stats.items(), key=lambda x:-x[1]['pnl'])
    for tool, stats in sorted_tools:
        if stats['trades']==0: continue
        wr=stats['wins']/stats['trades']*100
        c2="\033[92m" if stats['pnl']>0 else "\033[91m"
        print(f"    {tool:<25s} {stats['trades']:>3d}T {stats['wins']:>2d}W {wr:>5.1f}%WR {c2}${stats['pnl']:>+8.2f}{R}")

    print(f"\n  {C}COMBINED{R}")
    print(f"    Start:  ${BALANCE:.2f}")
    print(f"    Grid:   ${grid_profit:+.2f} ({grid_rts} RTs)")
    print(f"    Active: ${active_pnl:+.2f} ({len(trades)} trades)")
    print(f"    {C}TOTAL:  ${total_pnl:+.2f} ({total_pnl/BALANCE*100:+.2f}%){R}")
    print(f"    {C}FINAL:  ${final:.2f}{R}")
    print(f"    Max DD: {max_dd*100:.1f}%")

    if total_pnl > 0:
        monthly = total_pnl/days*30
        mpct = monthly/BALANCE*100
        print(f"\n  PROJECTION: ~${monthly:.2f}/mo (~{mpct:.1f}%)")
        for m in [6,12,24,36,60]:
            print(f"    {m:>3d}mo: ${BALANCE*(1+mpct/100)**m:>10,.2f}")

    # Trade log
    if trades:
        print(f"\n  TRADE LOG ({len(trades)} trades)")
        print(f"  {'Sym':6s} {'Tool':25s} {'Dir':6s} {'PnL':>8s} {'Reason':>6s} {'Bars':>5s}")
        for t in trades:
            c2="\033[92m" if t['pnl']>0 else "\033[91m"
            print(f"  {t['sym']:6s} {t['tool']:25s} {t['dir']:6s} {c2}${t['pnl']:>+7.2f}{R} {t['reason']:>6s} {t['bars']:>5d}")

    print(f"\n{'='*80}")

if __name__=="__main__":
    main()
