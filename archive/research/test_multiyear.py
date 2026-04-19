#!/usr/bin/env python3
"""
MULTI-YEAR ADAPTIVE ECOSYSTEM TEST
Tests grid + RSI + BB + momentum across all market regimes.
Uses daily candles for 2 years of data + hourly for recent.
The bot adapts: tracks what's working, allocates more to winners.
"""
import requests, time as _time, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# CryptoCompare gives 2000 bars/request — way more than Kraken's 720
CC_BASE = "https://min-api.cryptocompare.com/data/v2"
KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
BALANCE = 300

# CryptoCompare symbols (slightly different from Kraken)
CC_PAIRS = {
    "BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", 
    "LINK": "LINKUSD", "AVAX": "AVAXUSD", "DOT": "DOTUSD",
    "ADA": "ADAUSD", "XRP": "XRPUSD", "DOGE": "DOGEUSD",
    "UNI": "UNIUSD", "NEAR": "NEARUSD", "ATOM": "ATOMUSD",
    "AAVE": "AAVEUSD", "XLM": "XLMUSD", "FIL": "FILUSD",
    "LTC": "LTCUSD",
}

def download_cc(symbol, timeframe="hour", limit=2000):
    """Download from CryptoCompare (more history than Kraken)."""
    endpoint = {"hour": "histohour", "day": "histoday", "minute": "histominute"}[timeframe]
    try:
        resp = requests.get(f"{CC_BASE}/{endpoint}",
            params={"fsym": symbol, "tsym": "USD", "limit": limit}, timeout=30)
        data = resp.json().get("Data", {}).get("Data", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            if d.get("close", 0) > 0:
                rows.append({
                    'time': d['time'], 'open': d['open'], 'high': d['high'],
                    'low': d['low'], 'close': d['close'], 'volume': d.get('volumeto', 0)
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  Error downloading {symbol}: {e}")
        return pd.DataFrame()


def calc_rsi(close, period=14):
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def calc_bb(close, period=20, std_mult=2.0):
    s = pd.Series(close)
    mid = s.rolling(period).mean().values
    std = s.rolling(period).std().values
    return mid + std_mult * std, mid, mid - std_mult * std


def detect_regime(close, high, low):
    """Detect regime from price data."""
    if len(close) < 60:
        return "unknown"
    c = np.array(close[-60:], dtype=float)
    h = np.array(high[-60:], dtype=float)
    l = np.array(low[-60:], dtype=float)
    
    # Volatility
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_recent = float(np.mean(tr[-14:]))
    atr_median = float(np.median(tr))
    
    if atr_median > 0 and atr_recent > 1.5 * atr_median:
        return "volatile"
    
    # Trend: 20-bar return
    ret_20 = (c[-1] - c[-20]) / c[-20] if c[-20] > 0 else 0
    if ret_20 > 0.05:
        return "trending_up"
    elif ret_20 < -0.05:
        return "trending_down"
    
    return "ranging"


def sim_grid(df, start_idx, end_idx, allocation, grid_pct, tp_pct, levels):
    """Simulate grid on a slice of data."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    cash = allocation
    inventory = []
    inv_cost = 0
    roundtrips = 0
    total_profit = 0
    max_inv = allocation * 0.80
    
    for i in range(max(start_idx + 1, 1), min(end_idx, len(df))):
        mid = close[i-1]
        cur_low, cur_high = low[i], high[i]
        
        # Sell fills
        new_inv = []
        for bp, qty in inventory:
            sell_target = bp * (1 + tp_pct)
            if cur_high >= sell_target:
                fee = sell_target * qty * MAKER_FEE
                profit = sell_target * qty - fee - bp * qty * (1 + MAKER_FEE)
                total_profit += profit
                cash += sell_target * qty - fee
                inv_cost -= bp * qty
                roundtrips += 1
            else:
                new_inv.append((bp, qty))
        inventory = new_inv
        
        # Buy fills
        for level in range(1, levels + 1):
            buy_price = mid * (1 - grid_pct * level)
            if cur_low <= buy_price and inv_cost < max_inv:
                alloc = min(cash * 0.25, (max_inv - inv_cost) / levels)
                if alloc < 0.5: continue
                fee = alloc * MAKER_FEE
                qty = (alloc - fee) / buy_price
                if cash >= alloc:
                    cash -= alloc
                    inventory.append((buy_price, qty))
                    inv_cost += buy_price * qty
    
    unrealized = sum((close[min(end_idx-1, len(close)-1)] - bp) * qty for bp, qty in inventory)
    return total_profit, unrealized, roundtrips


def sim_rsi(df, start_idx, end_idx, allocation, rsi_period=9, oversold=20, overbought=80, hold=5):
    """Simulate RSI mean reversion on a slice."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    balance = allocation
    position = None  # (entry, qty, entry_bar, direction)
    trades = []
    
    rsi = calc_rsi(close[:end_idx], rsi_period)
    
    for i in range(max(start_idx, rsi_period + 5), min(end_idx, len(df))):
        if np.isnan(rsi[i]): continue
        
        # Exit
        if position:
            entry, qty, entry_bar, direction = position
            bars = i - entry_bar
            exit_price = None
            
            if direction == 'long':
                if low[i] <= entry * 0.97: exit_price = entry * 0.97
                elif rsi[i] >= 50 or bars >= hold: exit_price = close[i]
            else:
                if high[i] >= entry * 1.03: exit_price = entry * 1.03
                elif rsi[i] <= 50 or bars >= hold: exit_price = close[i]
            
            if exit_price:
                pnl = ((exit_price - entry) if direction == 'long' else (entry - exit_price)) * qty
                fees = entry * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                if direction == 'long':
                    balance += entry * qty + pnl - fees
                else:
                    balance += pnl - fees
                trades.append(pnl - fees)
                position = None
        
        # Entry
        if not position:
            if rsi[i] <= oversold:
                entry = close[i]
                qty = (balance * 0.03) / (entry * 0.03)
                cost = qty * entry
                if cost <= balance * 0.30 and cost > 1:
                    balance -= cost
                    position = (entry, qty, i, 'long')
            elif rsi[i] >= overbought:
                entry = close[i]
                qty = (balance * 0.03) / (entry * 0.03)
                if qty * entry > 1:
                    position = (entry, qty, i, 'short')
    
    # Close remaining
    if position:
        entry, qty, _, direction = position
        ep = close[min(end_idx-1, len(close)-1)]
        pnl = ((ep - entry) if direction == 'long' else (entry - ep)) * qty
        fees = entry * qty * MAKER_FEE + ep * qty * MAKER_FEE
        if direction == 'long':
            balance += entry * qty + pnl - fees
        else:
            balance += pnl - fees
        trades.append(pnl - fees)
    
    total_pnl = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    return total_pnl, len(trades), wins


def sim_momentum(df, start_idx, end_idx, allocation, bb=15, sl_mult=2.0, tp_mult=3.0, sma_p=50):
    """Simulate momentum breakout on a slice."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    balance = allocation
    position = None
    trades = []
    
    # Pre-calc
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().values
    sma = pd.Series(close).rolling(sma_p).mean().values
    
    for i in range(max(start_idx, sma_p + 5), min(end_idx, len(df))):
        if np.isnan(atr[i]) or np.isnan(sma[i]) or atr[i] <= 0: continue
        
        # Exit
        if position:
            entry, qty, sl, tp, entry_bar = position
            if low[i] <= sl:
                pnl = (sl - entry) * qty
                fees = entry * qty * MAKER_FEE + sl * qty * MAKER_FEE
                balance += entry * qty + pnl - fees
                trades.append(pnl - fees)
                position = None
            elif high[i] >= tp:
                pnl = (tp - entry) * qty
                fees = entry * qty * MAKER_FEE + tp * qty * MAKER_FEE
                balance += entry * qty + pnl - fees
                trades.append(pnl - fees)
                position = None
            elif i - entry_bar >= 48:
                pnl = (close[i] - entry) * qty
                fees = entry * qty * MAKER_FEE + close[i] * qty * MAKER_FEE
                balance += entry * qty + pnl - fees
                trades.append(pnl - fees)
                position = None
        
        # Entry (long only for simplicity)
        if not position:
            if close[i] <= sma[i]: continue
            prev_high = float(np.max(high[i-bb:i]))
            if close[i] <= prev_high: continue
            
            entry = close[i]
            sl = max(float(np.min(low[i-bb:i])) * 0.998, entry - sl_mult * atr[i])
            tp = entry + tp_mult * atr[i]
            risk = entry - sl
            if risk <= 0: continue
            
            qty = (balance * 0.03) / risk
            cost = qty * entry
            if cost > balance * 0.95: qty = (balance * 0.95) / entry
            if qty * entry < 1: continue
            balance -= qty * entry
            position = (entry, qty, sl, tp, i)
    
    if position:
        entry, qty, sl, tp, _ = position
        pnl = (close[min(end_idx-1, len(close)-1)] - entry) * qty
        fees = entry * qty * MAKER_FEE * 2
        balance += entry * qty + pnl - fees
        trades.append(pnl - fees)
    
    total_pnl = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    return total_pnl, len(trades), wins


def main():
    print("=" * 80)
    print("  MULTI-YEAR ADAPTIVE ECOSYSTEM TEST")
    print("  2 years of data, all market regimes, strategy auto-selection")
    print("=" * 80)
    
    # Download 2000 hourly bars (~83 days) from CryptoCompare
    # Also download daily bars (2000 bars = ~5.5 years)
    print("\nDownloading multi-year daily data...")
    daily_data = {}
    for cc_sym, kraken_sym in CC_PAIRS.items():
        df = download_cc(cc_sym, "day", 730)  # 2 years daily
        if len(df) > 100:
            daily_data[kraken_sym] = df
            start_date = datetime.fromtimestamp(df.iloc[0]['time']).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(df.iloc[-1]['time']).strftime('%Y-%m-%d')
            print(f"  {kraken_sym:10s}: {len(df)} days ({start_date} → {end_date}), "
                  f"${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    
    print(f"\nDownloading hourly data (83 days)...")
    hourly_data = {}
    for cc_sym, kraken_sym in CC_PAIRS.items():
        df = download_cc(cc_sym, "hour", 2000)
        if len(df) > 100:
            hourly_data[kraken_sym] = df
            print(f"  {kraken_sym:10s}: {len(df)} bars")
    
    # ═══════════════════════════════════════
    # PHASE 1: Test each strategy across 2-year daily windows
    # Split into 3-month rolling windows, test each strategy
    # ═══════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"  ROLLING 3-MONTH WINDOW TEST (daily data, 2 years)")
    print(f"  Which strategies work in which market conditions?")
    print(f"{'='*80}")
    
    # For grid on daily data, use wider grid spacing
    GRID_DAILY = {"grid_pct": 0.03, "tp_pct": 0.03, "levels": 3}
    
    # Collect performance by regime
    regime_perf = defaultdict(lambda: defaultdict(list))  # regime -> strategy -> [returns]
    
    # Use a few representative pairs
    test_pairs = ["XBTUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "NEARUSD", "ATOMUSD"]
    window_size = 90  # 3 months in days
    
    for pair in test_pairs:
        if pair not in daily_data:
            continue
        df = daily_data[pair]
        if len(df) < window_size + 60:
            continue
        
        print(f"\n  {pair}:")
        
        for start in range(60, len(df) - window_size, window_size // 2):  # 50% overlap
            end = start + window_size
            if end > len(df):
                break
            
            # Detect regime for this window
            close_slice = df['close'].values[:end]
            high_slice = df['high'].values[:end]
            low_slice = df['low'].values[:end]
            regime = detect_regime(close_slice, high_slice, low_slice)
            
            # Price change in this window
            price_change = (df['close'].values[end-1] - df['close'].values[start]) / df['close'].values[start] * 100
            
            # Test grid
            grid_profit, grid_unr, grid_rt = sim_grid(df, start, end, 100, 
                GRID_DAILY["grid_pct"], GRID_DAILY["tp_pct"], GRID_DAILY["levels"])
            grid_ret = (grid_profit + grid_unr) / 100 * 100
            
            # Test RSI
            rsi_pnl, rsi_trades, rsi_wins = sim_rsi(df, start, end, 100)
            rsi_ret = rsi_pnl / 100 * 100
            
            # Test momentum
            mom_pnl, mom_trades, mom_wins = sim_momentum(df, start, end, 100)
            mom_ret = mom_pnl / 100 * 100
            
            regime_perf[regime]["grid"].append(grid_ret)
            regime_perf[regime]["rsi"].append(rsi_ret)
            regime_perf[regime]["momentum"].append(mom_ret)
            
            start_date = datetime.fromtimestamp(df.iloc[start]['time']).strftime('%Y-%m')
            
            best_strat = "grid" if grid_ret >= max(rsi_ret, mom_ret) else ("rsi" if rsi_ret >= mom_ret else "momentum")
            
            print(f"    {start_date} {regime:14s} price={price_change:+6.1f}% | "
                  f"grid={grid_ret:+5.1f}%({grid_rt}RT) "
                  f"rsi={rsi_ret:+5.1f}%({rsi_trades}T) "
                  f"mom={mom_ret:+5.1f}%({mom_trades}T) "
                  f"→ BEST: {best_strat}")
    
    # ═══════════════════════════════════════
    # SUMMARY: Which strategy wins per regime?
    # ═══════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"  STRATEGY PERFORMANCE BY REGIME (averaged across all windows)")
    print(f"{'='*80}")
    
    print(f"\n  {'Regime':<18s} {'Grid':>10s} {'RSI':>10s} {'Momentum':>10s} {'Winner':>12s} {'Windows':>8s}")
    print(f"  {'─'*70}")
    
    optimal_map = {}
    for regime in ["ranging", "trending_up", "trending_down", "volatile"]:
        if regime not in regime_perf:
            continue
        grid_avg = np.mean(regime_perf[regime]["grid"]) if regime_perf[regime]["grid"] else -99
        rsi_avg = np.mean(regime_perf[regime]["rsi"]) if regime_perf[regime]["rsi"] else -99
        mom_avg = np.mean(regime_perf[regime]["momentum"]) if regime_perf[regime]["momentum"] else -99
        
        winner = "grid" if grid_avg >= max(rsi_avg, mom_avg) else ("rsi" if rsi_avg >= mom_avg else "momentum")
        n_windows = len(regime_perf[regime]["grid"])
        optimal_map[regime] = winner
        
        print(f"  {regime:<18s} {grid_avg:>+9.2f}% {rsi_avg:>+9.2f}% {mom_avg:>+9.2f}% {winner:>12s} {n_windows:>8d}")
    
    # ═══════════════════════════════════════
    # PHASE 2: ADAPTIVE SIMULATION on hourly data (83 days)
    # Re-evaluate regime every 24 bars, switch strategy
    # ═══════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"  ADAPTIVE ECOSYSTEM SIMULATION (hourly, ~83 days)")
    print(f"  Re-evaluates regime every 24h, switches to best strategy")
    print(f"{'='*80}")
    
    # Portfolio: split across top 6 pairs
    selected_pairs = [p for p in test_pairs if p in hourly_data]
    if not selected_pairs:
        print("  No hourly data available!")
        return
    
    alloc_per = BALANCE / len(selected_pairs)
    
    total_grid_profit = 0
    total_rsi_profit = 0  
    total_mom_profit = 0
    total_adaptive_profit = 0
    
    for pair in selected_pairs:
        df = hourly_data[pair]
        if len(df) < 200:
            continue
        
        # Run each strategy for the full period as baseline
        grid_p, grid_u, grid_rt = sim_grid(df, 60, len(df), alloc_per, 0.01, 0.015, 3)
        rsi_p, rsi_t, rsi_w = sim_rsi(df, 60, len(df), alloc_per)
        mom_p, mom_t, mom_w = sim_momentum(df, 60, len(df), alloc_per)
        
        # Adaptive: switch every 24 bars based on regime
        adaptive_profit = 0
        adaptive_alloc = alloc_per
        window = 24  # Re-evaluate daily
        
        for chunk_start in range(60, len(df) - window, window):
            chunk_end = min(chunk_start + window, len(df))
            
            # Detect regime
            regime = detect_regime(
                df['close'].values[:chunk_start+1],
                df['high'].values[:chunk_start+1],
                df['low'].values[:chunk_start+1]
            )
            
            # Pick strategy based on what works in this regime
            best = optimal_map.get(regime, "grid")
            
            # Always run grid (it's passive) + the best active strategy
            g_p, g_u, g_rt = sim_grid(df, chunk_start, chunk_end, adaptive_alloc * 0.5, 0.01, 0.015, 3)
            
            if best == "rsi" or best == "grid":
                a_p, a_t, a_w = sim_rsi(df, chunk_start, chunk_end, adaptive_alloc * 0.5)
            elif best == "momentum":
                a_p, a_t, a_w = sim_momentum(df, chunk_start, chunk_end, adaptive_alloc * 0.5)
            else:
                a_p = 0
            
            chunk_profit = g_p + g_u + a_p
            adaptive_profit += chunk_profit
        
        total_grid_profit += grid_p + grid_u
        total_rsi_profit += rsi_p
        total_mom_profit += mom_p
        total_adaptive_profit += adaptive_profit
        
        print(f"  {pair:10s}: grid=${grid_p+grid_u:+.2f}({grid_rt}RT) "
              f"rsi=${rsi_p:+.2f}({rsi_t}T) "
              f"mom=${mom_p:+.2f}({mom_t}T) "
              f"adaptive=${adaptive_profit:+.2f}")
    
    print(f"\n  {'─'*60}")
    print(f"  TOTALS on ${BALANCE}:")
    
    for name, pnl in [("Grid only", total_grid_profit), 
                       ("RSI only", total_rsi_profit),
                       ("Momentum only", total_mom_profit),
                       ("ADAPTIVE (ecosystem)", total_adaptive_profit)]:
        ret = pnl / BALANCE * 100
        color = "\033[92m" if pnl > 0 else "\033[91m"
        final = BALANCE + pnl
        print(f"  {name:25s}: {color}${pnl:+8.2f} ({ret:+.2f}%) → ${final:.2f}\033[0m")
    
    # Best approach
    results = {
        "Grid only": total_grid_profit,
        "RSI only": total_rsi_profit,
        "Momentum only": total_mom_profit,
        "ADAPTIVE": total_adaptive_profit,
    }
    best_name = max(results, key=results.get)
    best_pnl = results[best_name]
    
    print(f"\n  🏆 WINNER: {best_name} (${best_pnl:+.2f})")
    
    if best_pnl > 0:
        days = len(hourly_data[selected_pairs[0]]) / 24
        monthly = best_pnl / days * 30
        monthly_pct = monthly / BALANCE * 100
        print(f"\n  Monthly projection: ~${monthly:.2f}/mo (~{monthly_pct:.1f}%)")
        print(f"  Compound (if sustained):")
        for m in [6, 12, 24, 36, 60]:
            proj = BALANCE * (1 + monthly_pct / 100) ** m
            print(f"    {m:>3d}mo: ${proj:>12,.2f}")
    
    print(f"\n{'='*80}")
    print(f"  CONCLUSION")
    print(f"{'='*80}")
    print(f"\n  The optimal approach based on {len(daily_data)} pairs over ~2 years:")
    print(f"  Regime → Strategy mapping:")
    for regime, strat in optimal_map.items():
        print(f"    {regime:18s} → {strat}")
    print(f"\n  The adaptive ecosystem {'BEATS' if total_adaptive_profit > max(total_grid_profit, total_rsi_profit, total_mom_profit) else 'does NOT beat'} individual strategies.")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
