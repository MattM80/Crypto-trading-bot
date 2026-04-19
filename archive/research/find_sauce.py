#!/usr/bin/env python3
"""
FIND THE SAUCE.
Analyze every possible edge in the data. Not strategies — PATTERNS.
What conditions predict profitable mean reversion vs stop loss?
"""
import requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

CC_BASE = "https://min-api.cryptocompare.com/data/v2"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005

PAIRS = ["BTC", "ETH", "SOL", "LINK", "AVAX", "DOT", "ADA", "XRP",
         "DOGE", "UNI", "NEAR", "ATOM", "AAVE", "XLM", "FIL", "LTC"]

def download(sym, limit=2000):
    try:
        resp = requests.get(f"{CC_BASE}/histohour",
            params={"fsym": sym, "tsym": "USD", "limit": limit}, timeout=30)
        data = resp.json().get("Data", {}).get("Data", [])
        rows = [{'time': d['time'], 'open': d['open'], 'high': d['high'],
                 'low': d['low'], 'close': d['close'], 'volume': d.get('volumeto', 0)}
                for d in data if d.get('close', 0) > 0]
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def calc_rsi(c, p):
    s = pd.Series(c); d = s.diff()
    g = d.where(d>0,0); l = -d.where(d<0,0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).values

def calc_bb(c, p=20, s=2.0):
    sr = pd.Series(c)
    mid = sr.rolling(p).mean().values
    std = sr.rolling(p).std().values
    return mid + s*std, mid, mid - s*std

def main():
    print("Downloading data...")
    all_data = {}
    for sym in PAIRS:
        df = download(sym)
        if len(df) > 200:
            all_data[sym] = df
    print(f"Loaded {len(all_data)} pairs\n")

    # ═══════════════════════════════════════
    # For every bar, compute a feature vector.
    # Then look ahead N bars and see what happened.
    # Find which features predict profitable entries.
    # ═══════════════════════════════════════

    results = []  # Each entry: {features..., outcome_1h, outcome_4h, outcome_8h, outcome_24h}

    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        
        # Pre-compute indicators
        rsi7 = calc_rsi(c, 7)
        rsi14 = calc_rsi(c, 14)
        bb_upper, bb_mid, bb_lower = calc_bb(c, 20, 2.0)
        
        # ATR
        prev_c = np.roll(c, 1); prev_c[0] = c[0]
        tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
        atr14 = pd.Series(tr).rolling(14).mean().values
        
        # Volume ratio
        vol_sma = pd.Series(v).rolling(20).mean().values
        
        # SMA
        sma20 = pd.Series(c).rolling(20).mean().values
        sma50 = pd.Series(c).rolling(50).mean().values
        
        for i in range(60, len(c) - 25):
            if np.isnan(rsi7[i]) or np.isnan(atr14[i]) or np.isnan(bb_mid[i]):
                continue
            if atr14[i] <= 0 or c[i] <= 0:
                continue
            
            # Features
            cur_rsi7 = rsi7[i]
            cur_rsi14 = rsi14[i]
            price_vs_bb_lower = (c[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
            price_vs_sma20 = (c[i] - sma20[i]) / sma20[i] * 100 if not np.isnan(sma20[i]) and sma20[i] > 0 else 0
            price_vs_sma50 = (c[i] - sma50[i]) / sma50[i] * 100 if not np.isnan(sma50[i]) and sma50[i] > 0 else 0
            vol_ratio = v[i] / vol_sma[i] if not np.isnan(vol_sma[i]) and vol_sma[i] > 0 else 1
            atr_pct = atr14[i] / c[i] * 100
            
            # Recent price action
            ret_4h = (c[i] - c[i-4]) / c[i-4] * 100
            ret_8h = (c[i] - c[i-8]) / c[i-8] * 100
            ret_24h = (c[i] - c[i-24]) / c[i-24] * 100
            
            # Candle structure
            body = abs(c[i] - df['open'].values[i])
            wick_up = h[i] - max(c[i], df['open'].values[i])
            wick_down = min(c[i], df['open'].values[i]) - l[i]
            candle_range = h[i] - l[i]
            body_ratio = body / candle_range if candle_range > 0 else 0
            
            # Volume spike
            vol_spike = v[i] > vol_sma[i] * 1.5 if not np.isnan(vol_sma[i]) else False
            
            # Outcomes (what happens AFTER this bar?)
            # For LONG entry at close[i]:
            fee_cost = c[i] * 0.0042  # 0.42% round trip
            
            outcomes = {}
            for horizon in [1, 2, 4, 8, 12, 24]:
                if i + horizon >= len(c):
                    continue
                future_max = np.max(h[i+1:i+horizon+1])
                future_min = np.min(l[i+1:i+horizon+1])
                future_close = c[i+horizon]
                
                long_profit = (future_close - c[i]) / c[i] * 100
                long_max_gain = (future_max - c[i]) / c[i] * 100
                long_max_drawdown = (c[i] - future_min) / c[i] * 100
                
                short_profit = (c[i] - future_close) / c[i] * 100
                
                outcomes[f'long_{horizon}h'] = long_profit
                outcomes[f'short_{horizon}h'] = short_profit
                outcomes[f'long_max_gain_{horizon}h'] = long_max_gain
                outcomes[f'long_max_dd_{horizon}h'] = long_max_drawdown
            
            results.append({
                'sym': sym, 'bar': i,
                'rsi7': cur_rsi7, 'rsi14': cur_rsi14,
                'bb_pos': price_vs_bb_lower,
                'vs_sma20': price_vs_sma20, 'vs_sma50': price_vs_sma50,
                'vol_ratio': vol_ratio, 'vol_spike': vol_spike,
                'atr_pct': atr_pct,
                'ret_4h': ret_4h, 'ret_8h': ret_8h, 'ret_24h': ret_24h,
                'body_ratio': body_ratio,
                **outcomes
            })

    df_all = pd.DataFrame(results)
    print(f"Analyzed {len(df_all)} bar-states across {len(all_data)} pairs\n")

    # ═══════════════════════════════════════
    # FIND WHAT PREDICTS PROFIT
    # ═══════════════════════════════════════
    
    fee = 0.42  # 0.42% round trip fee

    print("=" * 90)
    print("  PATTERN MINING: What conditions predict profitable trades?")
    print("=" * 90)

    # Test various conditions and see average outcomes
    conditions = [
        # RSI extremes
        ("RSI7 < 15 (deeply oversold)", lambda r: r['rsi7'] < 15),
        ("RSI7 < 20", lambda r: r['rsi7'] < 20),
        ("RSI7 < 25", lambda r: r['rsi7'] < 25),
        ("RSI7 < 30", lambda r: r['rsi7'] < 30),
        ("RSI7 > 70", lambda r: r['rsi7'] > 70),
        ("RSI7 > 75", lambda r: r['rsi7'] > 75),
        ("RSI7 > 80", lambda r: r['rsi7'] > 80),
        ("RSI7 > 85 (deeply overbought)", lambda r: r['rsi7'] > 85),
        
        # RSI + volume
        ("RSI7<20 + volume spike", lambda r: r['rsi7'] < 20 and r['vol_spike']),
        ("RSI7>80 + volume spike", lambda r: r['rsi7'] > 80 and r['vol_spike']),
        
        # RSI + trend alignment
        ("RSI7<25 + price>SMA50 (oversold in uptrend)", lambda r: r['rsi7'] < 25 and r['vs_sma50'] > 0),
        ("RSI7<25 + price<SMA50 (oversold in downtrend)", lambda r: r['rsi7'] < 25 and r['vs_sma50'] < 0),
        ("RSI7>75 + price<SMA50 (overbought in downtrend)", lambda r: r['rsi7'] > 75 and r['vs_sma50'] < 0),
        ("RSI7>75 + price>SMA50 (overbought in uptrend)", lambda r: r['rsi7'] > 75 and r['vs_sma50'] > 0),
        
        # BB extremes
        ("Price at lower BB (bb_pos<0.05)", lambda r: r['bb_pos'] < 0.05),
        ("Price at upper BB (bb_pos>0.95)", lambda r: r['bb_pos'] > 0.95),
        
        # Sharp drops
        ("Dropped >3% in 4h", lambda r: r['ret_4h'] < -3),
        ("Dropped >5% in 8h", lambda r: r['ret_8h'] < -5),
        ("Dropped >3% in 4h + vol spike", lambda r: r['ret_4h'] < -3 and r['vol_spike']),
        ("Dropped >5% in 8h + vol spike", lambda r: r['ret_8h'] < -5 and r['vol_spike']),
        
        # Sharp pumps  
        ("Pumped >3% in 4h", lambda r: r['ret_4h'] > 3),
        ("Pumped >5% in 8h", lambda r: r['ret_8h'] > 5),
        
        # Combinations
        ("RSI7<20 + drop>3% 4h (crash buy)", lambda r: r['rsi7'] < 20 and r['ret_4h'] < -3),
        ("RSI7<15 + drop>3% 4h + vol (mega crash)", lambda r: r['rsi7'] < 15 and r['ret_4h'] < -3 and r['vol_spike']),
        ("RSI7>80 + pump>3% 4h (pump sell)", lambda r: r['rsi7'] > 80 and r['ret_4h'] > 3),
        
        # High volatility entries
        ("ATR>3% + RSI<25 (volatile oversold)", lambda r: r['atr_pct'] > 3 and r['rsi7'] < 25),
        ("ATR>3% + RSI>75 (volatile overbought)", lambda r: r['atr_pct'] > 3 and r['rsi7'] > 75),
        ("ATR>4% + RSI<20 (extreme)", lambda r: r['atr_pct'] > 4 and r['rsi7'] < 20),
        
        # Small body + direction (indecision candles after moves)
        ("Doji after drop (body<30% + ret_4h<-2%)", lambda r: r['body_ratio'] < 0.3 and r['ret_4h'] < -2),
        
        # Multi-timeframe
        ("RSI7<25 + RSI14<35 (both oversold)", lambda r: r['rsi7'] < 25 and r['rsi14'] < 35),
        ("RSI7>75 + RSI14>65 (both overbought)", lambda r: r['rsi7'] > 75 and r['rsi14'] > 65),
        
        # Trend + mean reversion
        ("Drop>5% 24h + RSI7<25 (correction buy)", lambda r: r['ret_24h'] < -5 and r['rsi7'] < 25),
        ("Drop>10% 24h + RSI7<20 (crash buy)", lambda r: r['ret_24h'] < -10 and r['rsi7'] < 20),
        ("Pump>5% 24h + RSI7>75 (rally sell)", lambda r: r['ret_24h'] > 5 and r['rsi7'] > 75),
    ]

    print(f"\n  {'Condition':<50s} {'N':>5s} {'L1h':>6s} {'L4h':>6s} {'L8h':>6s} {'L24h':>6s} {'S1h':>6s} {'S4h':>6s} {'S8h':>6s} {'MaxG4':>6s} {'MaxDD4':>6s} {'W%4h':>5s}")
    print("  " + "-" * 115)

    best_edges = []

    for name, cond_fn in conditions:
        mask = df_all.apply(cond_fn, axis=1)
        subset = df_all[mask]
        n = len(subset)
        if n < 10:
            continue

        l1 = subset['long_1h'].mean() if 'long_1h' in subset else 0
        l4 = subset['long_4h'].mean() if 'long_4h' in subset else 0
        l8 = subset['long_8h'].mean() if 'long_8h' in subset else 0
        l24 = subset['long_24h'].mean() if 'long_24h' in subset else 0
        s1 = subset['short_1h'].mean() if 'short_1h' in subset else 0
        s4 = subset['short_4h'].mean() if 'short_4h' in subset else 0
        s8 = subset['short_8h'].mean() if 'short_8h' in subset else 0
        mg4 = subset['long_max_gain_4h'].mean() if 'long_max_gain_4h' in subset else 0
        mdd4 = subset['long_max_dd_4h'].mean() if 'long_max_dd_4h' in subset else 0
        
        # Win rate: how often is 4h long return > fee?
        if 'long_4h' in subset:
            wr_long = (subset['long_4h'] > fee).mean() * 100
        else:
            wr_long = 0
            
        # Determine best direction for this condition
        best_4h = max(l4, s4)
        best_dir = "LONG" if l4 >= s4 else "SHORT"
        
        # Color coding
        profitable = best_4h > fee
        color = "\033[92m" if profitable else ("\033[93m" if best_4h > 0 else "\033[91m")
        reset = "\033[0m"

        print(f"  {name:<50s} {n:>5d} {color}{l1:>+5.2f}% {l4:>+5.2f}% {l8:>+5.2f}% {l24:>+5.2f}%{reset} "
              f"{s1:>+5.2f}% {s4:>+5.2f}% {s8:>+5.2f}% {mg4:>+5.2f}% {mdd4:>+5.2f}% {wr_long:>4.0f}%")
        
        if profitable:
            best_edges.append((name, n, best_dir, best_4h, l4, s4, wr_long, l8, l24))

    # ═══════════════════════════════════════
    # TOP EDGES (sorted by 4h return after fees)
    # ═══════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  TOP EDGES (profitable after 0.42% fees)")
    print(f"{'='*90}")
    
    best_edges.sort(key=lambda x: max(x[4], x[5]), reverse=True)
    
    for name, n, best_dir, best_4h, l4, s4, wr, l8, l24 in best_edges:
        net_4h = best_4h - fee
        dir_str = f"LONG +{l4:.2f}%" if l4 >= s4 else f"SHORT +{s4:.2f}%"
        print(f"  \033[92m{name:<50s}\033[0m  N={n:>4d}  {dir_str}  net={net_4h:+.2f}%  WR={wr:.0f}%  8h={l8:+.2f}%  24h={l24:+.2f}%")

    # ═══════════════════════════════════════
    # SIMULATE THE BEST EDGES AS A PORTFOLIO
    # ═══════════════════════════════════════
    if best_edges:
        print(f"\n{'='*90}")
        print(f"  SIMULATING TOP EDGES ON $300")
        print(f"{'='*90}")
        
        # Take the top 5 edges, simulate them
        for name, n, best_dir, best_4h, l4, s4, wr, l8, l24 in best_edges[:5]:
            # Estimate monthly trades (n signals over 83 days)
            monthly_trades = n / 83 * 30
            net_per_trade_pct = best_4h - fee
            monthly_return_pct = monthly_trades * net_per_trade_pct * 0.03  # 3% risk per trade
            
            print(f"  {name}")
            print(f"    Signals: {n} in 83d (~{monthly_trades:.0f}/month)")
            print(f"    Edge: {best_4h:.2f}% avg, net {net_per_trade_pct:.2f}% after fees")
            print(f"    Monthly return estimate: ~{monthly_return_pct:.2f}% (at 3% risk/trade)")
            print()

    print(f"\n{'='*90}")
    print(f"  DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
