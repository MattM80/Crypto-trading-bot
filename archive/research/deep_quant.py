#!/usr/bin/env python3
"""
DEEP QUANT: Edges that big firms can't/won't touch.

Big firms need:
- Liquidity (can't trade small alts)
- Speed (can't wait 24h for a trade)
- Scale ($1M+ per trade)

We have the OPPOSITE advantages:
- Can trade tiny altcoins with $10 positions
- Can hold for hours/days (no quarterly reporting)
- No slippage on small orders
- Can exploit inefficiencies too small for institutions

Testing:
1. Microstructure edges (spread patterns, volume anomalies)
2. Cross-asset correlation breakdowns (when BTC/ETH decorrelate)
3. Time-of-day patterns (specific hours have consistent bias)
4. Volatility clustering (vol predicts vol, not direction)
5. Order flow imbalance (volume on up vs down candles)
6. Mean reversion at multiple timescales simultaneously
7. Relative strength momentum (buy strongest, short weakest)
8. Liquidity gaps (price jumps = overreaction = revert)
9. Funding-like signals from spot premium patterns
10. Entropy/information theory (unusual price patterns)
"""
import requests, numpy as np, pandas as pd, warnings
from datetime import datetime, timezone
from collections import defaultdict
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

CC_BASE = "https://min-api.cryptocompare.com/data/v2"
FEE = 0.42  # 0.42% round trip

PAIRS = ["BTC", "ETH", "SOL", "LINK", "AVAX", "DOT", "ADA", "XRP",
         "DOGE", "UNI", "NEAR", "ATOM", "AAVE", "XLM", "FIL", "LTC"]

def dl(sym, tf="hour", n=2000):
    ep = {"hour": "histohour", "day": "histoday", "minute": "histominute"}[tf]
    try:
        r = requests.get(f"{CC_BASE}/{ep}", params={"fsym": sym, "tsym": "USD", "limit": n}, timeout=30)
        d = r.json().get("Data", {}).get("Data", [])
        rows = [{'time': x['time'], 'open': x['open'], 'high': x['high'],
                 'low': x['low'], 'close': x['close'], 'volume': x.get('volumeto', 0)}
                for x in d if x.get('close', 0) > 0]
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def rsi(c, p=7):
    s = pd.Series(c); d = s.diff()
    g = d.where(d>0,0); l = -d.where(d<0,0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).values

def atr(h, l, c, p=14):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    return pd.Series(tr).rolling(p).mean().values

def test_edge(all_data, name, condition_fn, hold_bars=[1,4,8,24], min_n=20):
    """Test an edge across all pairs. Return stats."""
    results = []
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        
        for i in range(60, len(c) - 25):
            try:
                if condition_fn(c, h, l, v, i, sym, all_data):
                    row = {'sym': sym, 'bar': i, 'price': c[i]}
                    for hb in hold_bars:
                        if i + hb < len(c):
                            row[f'long_{hb}h'] = (c[i+hb] - c[i]) / c[i] * 100
                            row[f'short_{hb}h'] = (c[i] - c[i+hb]) / c[i] * 100
                            row[f'max_up_{hb}h'] = (np.max(h[i+1:i+hb+1]) - c[i]) / c[i] * 100
                            row[f'max_down_{hb}h'] = (c[i] - np.min(l[i+1:i+hb+1])) / c[i] * 100
                    results.append(row)
            except:
                continue
    
    if len(results) < min_n:
        return None
    
    rdf = pd.DataFrame(results)
    n = len(rdf)
    
    stats = {'name': name, 'n': n}
    for hb in hold_bars:
        lk = f'long_{hb}h'
        sk = f'short_{hb}h'
        if lk in rdf:
            stats[f'L{hb}'] = round(rdf[lk].mean(), 3)
            stats[f'S{hb}'] = round(rdf[sk].mean(), 3)
            stats[f'WR_L{hb}'] = round((rdf[lk] > FEE).mean() * 100, 1)
            stats[f'WR_S{hb}'] = round((rdf[sk] > FEE).mean() * 100, 1)
            stats[f'maxup_{hb}'] = round(rdf[f'max_up_{hb}h'].mean(), 3)
            stats[f'maxdd_{hb}'] = round(rdf[f'max_down_{hb}h'].mean(), 3)
    
    return stats


def main():
    print("=" * 100)
    print("  DEEP QUANT: Mining edges humans can't see")
    print("=" * 100)
    
    print("\nDownloading 83 days hourly data...")
    all_data = {}
    for sym in PAIRS:
        df = dl(sym)
        if len(df) > 200:
            all_data[sym] = df
    print(f"Loaded {len(all_data)} pairs\n")

    # Pre-compute cross-asset data
    # Align all close prices into a matrix
    min_len = min(len(df) for df in all_data.values())
    price_matrix = {}
    for sym, df in all_data.items():
        price_matrix[sym] = df['close'].values[-min_len:].astype(float)
    
    returns_matrix = {}
    for sym, prices in price_matrix.items():
        returns_matrix[sym] = np.diff(prices) / prices[:-1] * 100

    edges = []

    # ═══════════════════════════════════════
    # EDGE 1: Time-of-Day Patterns
    # Crypto has consistent hourly biases
    # ═══════════════════════════════════════
    print("Testing time-of-day patterns...")
    for hour in range(24):
        def make_cond(h):
            def cond(c, hi, lo, v, i, sym, ad):
                ts = ad[sym]['time'].values[i]
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).hour == h
            return cond
        r = test_edge(all_data, f"Hour={hour:02d} UTC", make_cond(hour))
        if r:
            edges.append(r)

    # ═══════════════════════════════════════
    # EDGE 2: Volume Anomaly
    # When volume spikes 3x+ with small price change = accumulation
    # ═══════════════════════════════════════
    print("Testing volume anomalies...")
    def vol_spike_small_move(c, h, l, v, i, sym, ad):
        if i < 20: return False
        vol_avg = np.mean(v[i-20:i])
        if vol_avg <= 0: return False
        vol_ratio = v[i] / vol_avg
        price_move = abs(c[i] - c[i-1]) / c[i-1] * 100
        return vol_ratio > 3.0 and price_move < 0.5
    edges.append(test_edge(all_data, "Vol 3x+ but price <0.5% (accumulation)", vol_spike_small_move))

    def vol_spike_down(c, h, l, v, i, sym, ad):
        if i < 20: return False
        vol_avg = np.mean(v[i-20:i])
        if vol_avg <= 0: return False
        return v[i] / vol_avg > 2.5 and c[i] < c[i-1]
    edges.append(test_edge(all_data, "Vol 2.5x+ on red candle (capitulation)", vol_spike_down))

    # ═══════════════════════════════════════
    # EDGE 3: BTC/ETH Correlation Breakdown
    # When BTC and ETH diverge, they revert
    # ═══════════════════════════════════════
    print("Testing correlation breakdowns...")
    def btc_eth_diverge_buy_eth(c, h, l, v, i, sym, ad):
        if sym != "ETH" or "BTC" not in ad: return False
        btc = ad["BTC"]['close'].values
        eth = ad["ETH"]['close'].values
        if i >= len(btc) or i >= len(eth) or i < 24: return False
        btc_ret = (btc[i] - btc[i-24]) / btc[i-24] * 100
        eth_ret = (eth[i] - eth[i-24]) / eth[i-24] * 100
        return btc_ret - eth_ret > 3  # BTC outperformed ETH by 3%+ in 24h
    edges.append(test_edge(all_data, "BTC outperformed ETH >3% 24h (buy ETH)", btc_eth_diverge_buy_eth))

    def btc_eth_diverge_buy_btc(c, h, l, v, i, sym, ad):
        if sym != "BTC" or "ETH" not in ad: return False
        btc = ad["BTC"]['close'].values
        eth = ad["ETH"]['close'].values
        if i >= len(btc) or i >= len(eth) or i < 24: return False
        eth_ret = (eth[i] - eth[i-24]) / eth[i-24] * 100
        btc_ret = (btc[i] - btc[i-24]) / btc[i-24] * 100
        return eth_ret - btc_ret > 3
    edges.append(test_edge(all_data, "ETH outperformed BTC >3% 24h (buy BTC)", btc_eth_diverge_buy_btc))

    # ═══════════════════════════════════════
    # EDGE 4: Relative Strength
    # Buy the strongest coin of the day, short the weakest
    # ═══════════════════════════════════════
    print("Testing relative strength...")
    def strongest_24h(c, h, l, v, i, sym, ad):
        if i < 24: return False
        rets = {}
        for s, d in ad.items():
            cl = d['close'].values
            if i < len(cl) and i >= 24:
                rets[s] = (cl[i] - cl[i-24]) / cl[i-24] * 100
        if len(rets) < 5: return False
        ranked = sorted(rets.items(), key=lambda x: x[1], reverse=True)
        return sym == ranked[0][0]  # Is this the strongest?
    edges.append(test_edge(all_data, "Strongest coin last 24h (momentum)", strongest_24h))

    def weakest_24h(c, h, l, v, i, sym, ad):
        if i < 24: return False
        rets = {}
        for s, d in ad.items():
            cl = d['close'].values
            if i < len(cl) and i >= 24:
                rets[s] = (cl[i] - cl[i-24]) / cl[i-24] * 100
        if len(rets) < 5: return False
        ranked = sorted(rets.items(), key=lambda x: x[1])
        return sym == ranked[0][0]  # Is this the weakest?
    edges.append(test_edge(all_data, "Weakest coin last 24h (mean revert buy)", weakest_24h))

    # ═══════════════════════════════════════
    # EDGE 5: Volatility Contraction → Expansion
    # When ATR shrinks then expands = breakout
    # ═══════════════════════════════════════
    print("Testing volatility patterns...")
    def vol_squeeze_breakout(c, h, l, v, i, sym, ad):
        at = atr(h, l, c, 14)
        if np.isnan(at[i]) or np.isnan(at[i-10]) or at[i-10] <= 0: return False
        # ATR was contracting for 10 bars then just expanded
        recent_atr = at[i]
        prev_atr = at[i-10]
        mid_atr = np.mean(at[i-5:i])
        return mid_atr < prev_atr * 0.7 and recent_atr > mid_atr * 1.5
    edges.append(test_edge(all_data, "Vol squeeze then expand (breakout)", vol_squeeze_breakout))

    # ═══════════════════════════════════════
    # EDGE 6: Consecutive Candle Patterns
    # ═══════════════════════════════════════
    print("Testing candle patterns...")
    def three_red_candles(c, h, l, v, i, sym, ad):
        if i < 3: return False
        return c[i] < c[i-1] < c[i-2] < c[i-3]
    edges.append(test_edge(all_data, "3 consecutive red candles (buy bounce)", three_red_candles))

    def five_red_candles(c, h, l, v, i, sym, ad):
        if i < 5: return False
        return all(c[i-j] < c[i-j-1] for j in range(5))
    edges.append(test_edge(all_data, "5 consecutive red candles (buy bounce)", five_red_candles))

    def three_green_candles(c, h, l, v, i, sym, ad):
        if i < 3: return False
        return c[i] > c[i-1] > c[i-2] > c[i-3]
    edges.append(test_edge(all_data, "3 consecutive green candles (sell fade)", three_green_candles))

    # ═══════════════════════════════════════
    # EDGE 7: Order Flow Imbalance
    # Buy volume vs sell volume ratio
    # ═══════════════════════════════════════
    print("Testing order flow...")
    def buy_volume_dominant(c, h, l, v, i, sym, ad):
        if i < 10: return False
        # Estimate buy/sell volume from candle structure
        buy_vol = 0; sell_vol = 0
        for j in range(i-10, i+1):
            if c[j] > ad[sym]['open'].values[j]:
                buy_vol += v[j]
            else:
                sell_vol += v[j]
        if sell_vol <= 0: return False
        return buy_vol / sell_vol > 2.0  # Buyers dominating 2:1
    edges.append(test_edge(all_data, "Buy volume > 2x sell volume (10 bars)", buy_volume_dominant))

    def sell_volume_dominant(c, h, l, v, i, sym, ad):
        if i < 10: return False
        buy_vol = 0; sell_vol = 0
        for j in range(i-10, i+1):
            if c[j] > ad[sym]['open'].values[j]:
                buy_vol += v[j]
            else:
                sell_vol += v[j]
        if buy_vol <= 0: return False
        return sell_vol / buy_vol > 2.0
    edges.append(test_edge(all_data, "Sell volume > 2x buy volume (10 bars)", sell_volume_dominant))

    # ═══════════════════════════════════════
    # EDGE 8: Price Gaps / Liquidity Jumps
    # ═══════════════════════════════════════
    print("Testing price gaps...")
    def gap_down(c, h, l, v, i, sym, ad):
        if i < 1: return False
        gap = (ad[sym]['open'].values[i] - c[i-1]) / c[i-1] * 100
        return gap < -1.0  # Opened >1% below previous close
    edges.append(test_edge(all_data, "Gap down >1% (fill the gap long)", gap_down))

    def gap_up(c, h, l, v, i, sym, ad):
        if i < 1: return False
        gap = (ad[sym]['open'].values[i] - c[i-1]) / c[i-1] * 100
        return gap > 1.0
    edges.append(test_edge(all_data, "Gap up >1% (fade the gap short)", gap_up))

    # ═══════════════════════════════════════
    # EDGE 9: Mean Reversion Extremes (multi-bar)
    # ═══════════════════════════════════════
    print("Testing extreme mean reversion...")
    for drop_pct in [3, 5, 8, 10, 15]:
        for lookback in [4, 8, 12, 24]:
            def make_drop_cond(dp, lb):
                def cond(c, h, l, v, i, sym, ad):
                    if i < lb: return False
                    ret = (c[i] - c[i-lb]) / c[i-lb] * 100
                    return ret < -dp
                return cond
            edges.append(test_edge(all_data, f"Drop >{drop_pct}% in {lookback}h (long)", make_drop_cond(drop_pct, lookback)))

    # ═══════════════════════════════════════
    # EDGE 10: RSI Divergence
    # Price makes new low but RSI makes higher low
    # ═══════════════════════════════════════
    print("Testing RSI divergence...")
    def bullish_divergence(c, h, l, v, i, sym, ad):
        if i < 28: return False
        r = rsi(c[:i+1], 14)
        if len(r) < 28 or np.isnan(r[-1]) or np.isnan(r[-14]): return False
        # Price: lower low in last 14 bars vs prior 14
        recent_low = np.min(c[i-14:i+1])
        prior_low = np.min(c[i-28:i-14])
        # RSI: higher low
        recent_rsi_low = np.min(r[-14:])
        prior_rsi_low = np.min(r[-28:-14])
        return recent_low < prior_low and recent_rsi_low > prior_rsi_low and r[-1] < 35
    edges.append(test_edge(all_data, "Bullish RSI divergence (price lower low, RSI higher low)", bullish_divergence))

    # ═══════════════════════════════════════
    # EDGE 11: Multi-timeframe RSI
    # Oversold on fast AND slow RSI
    # ═══════════════════════════════════════
    print("Testing multi-timeframe RSI...")
    def dual_rsi_oversold(c, h, l, v, i, sym, ad):
        r7 = rsi(c[:i+1], 7)
        r14 = rsi(c[:i+1], 14)
        if np.isnan(r7[-1]) or np.isnan(r14[-1]): return False
        return r7[-1] < 20 and r14[-1] < 30
    edges.append(test_edge(all_data, "RSI7<20 + RSI14<30 (deep dual oversold)", dual_rsi_oversold))

    # ═══════════════════════════════════════
    # EDGE 12: BTC leads, alts follow
    # When BTC moves first, alts catch up
    # ═══════════════════════════════════════
    print("Testing BTC-lead-alt-follow...")
    def btc_pumped_alt_lagging(c, h, l, v, i, sym, ad):
        if sym == "BTC" or "BTC" not in ad: return False
        btc = ad["BTC"]['close'].values
        if i < 4 or i >= len(btc): return False
        btc_ret = (btc[i] - btc[i-4]) / btc[i-4] * 100
        alt_ret = (c[i] - c[i-4]) / c[i-4] * 100
        return btc_ret > 2 and alt_ret < 0.5  # BTC up >2%, alt flat/down
    edges.append(test_edge(all_data, "BTC pumped >2% 4h, alt lagging (buy alt)", btc_pumped_alt_lagging))

    def btc_dumped_alt_lagging(c, h, l, v, i, sym, ad):
        if sym == "BTC" or "BTC" not in ad: return False
        btc = ad["BTC"]['close'].values
        if i < 4 or i >= len(btc): return False
        btc_ret = (btc[i] - btc[i-4]) / btc[i-4] * 100
        alt_ret = (c[i] - c[i-4]) / c[i-4] * 100
        return btc_ret < -2 and alt_ret > -0.5
    edges.append(test_edge(all_data, "BTC dumped >2% 4h, alt hasn't (short alt)", btc_dumped_alt_lagging))

    # ═══════════════════════════════════════
    # EDGE 13: Weekend effect
    # ═══════════════════════════════════════
    print("Testing day-of-week patterns...")
    for dow in range(7):
        day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        def make_dow_cond(d):
            def cond(c, h, l, v, i, sym, ad):
                ts = ad[sym]['time'].values[i]
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).weekday() == d
            return cond
        edges.append(test_edge(all_data, f"Day={day_names[dow]}", make_dow_cond(dow)))

    # ═══════════════════════════════════════
    # EDGE 14: Inside bar breakout
    # ═══════════════════════════════════════
    print("Testing inside bars...")
    def inside_bar_buy(c, h, l, v, i, sym, ad):
        if i < 2: return False
        # Current bar is inside previous bar
        inside = h[i] < h[i-1] and l[i] > l[i-1]
        # Previous bar was also inside (compression)
        double_inside = inside and h[i-1] < h[i-2] and l[i-1] > l[i-2]
        return double_inside
    edges.append(test_edge(all_data, "Double inside bar (compression breakout)", inside_bar_buy))

    # Filter and print
    edges = [e for e in edges if e is not None]

    print(f"\n{'='*100}")
    print(f"  RESULTS: {len(edges)} patterns tested")
    print(f"{'='*100}")

    # Sort by best 8h long return (or short if better)
    def best_return(e):
        l8 = e.get('L8', 0)
        s8 = e.get('S8', 0)
        return max(l8, s8)

    edges.sort(key=best_return, reverse=True)

    print(f"\n  {'Pattern':<55s} {'N':>5s} {'L4h':>6s} {'L8h':>6s} {'L24h':>7s} {'S4h':>6s} {'S8h':>6s} {'WR_L8':>6s} {'MaxUp8':>7s} {'Best':>5s}")
    print("  " + "-" * 110)

    profitable_tools = []

    for e in edges:
        l4 = e.get('L4', 0); l8 = e.get('L8', 0); l24 = e.get('L24', 0)
        s4 = e.get('S4', 0); s8 = e.get('S8', 0)
        wr_l8 = e.get('WR_L8', 0)
        mu8 = e.get('maxup_8', 0)
        best = max(l8, s8)
        profitable = best > FEE
        
        color = "\033[92m" if profitable else ("\033[93m" if best > 0 else "\033[91m")
        R = "\033[0m"
        
        direction = "LONG" if l8 >= s8 else "SHORT"
        
        print(f"  {e['name']:<55s} {e['n']:>5d} "
              f"{color}{l4:>+5.2f}% {l8:>+5.2f}% {l24:>+6.2f}%{R} "
              f"{s4:>+5.2f}% {s8:>+5.2f}% {wr_l8:>5.1f}% {mu8:>+6.2f}% {direction:>5s}")
        
        if profitable and e['n'] >= 30:
            profitable_tools.append((e['name'], e['n'], direction, best, l8, s8, l24, wr_l8))

    print(f"\n{'='*100}")
    print(f"  NEW TOOLS FOR THE ALL-SEEING EYE (profitable after {FEE}% fees, N≥30)")
    print(f"{'='*100}")

    if profitable_tools:
        profitable_tools.sort(key=lambda x: x[3], reverse=True)
        for name, n, direction, best, l8, s8, l24, wr in profitable_tools:
            net = best - FEE
            print(f"  ✅ {name}")
            print(f"     N={n} | {direction} | 8h avg={best:+.2f}% (net {net:+.2f}%) | 24h={l24:+.2f}% | WR={wr:.0f}%")
            print()
    else:
        print("  No new profitable tools found at this threshold.")

    print(f"{'='*100}")


if __name__ == "__main__":
    main()
