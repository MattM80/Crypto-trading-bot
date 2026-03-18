#!/usr/bin/env python3
"""
Test a fundamentally different approach: GRID + MEAN REVERSION
Don't predict direction. Harvest the noise.

Approach 1: Simple Grid
- Place buy limit orders at -0.5%, -1.0%, -1.5% below current price
- When a buy fills, immediately place a sell at +0.5% above that buy
- Repeat forever. Each completed round-trip = profit.
- Works in ANY market direction as long as price oscillates.

Approach 2: Bollinger Band Mean Reversion (5m/15m)
- When price touches lower BB, buy. When it touches upper BB, sell.
- Very short holding period (1-5 bars). Quick in, quick out.
- Mean reversion on short timeframes is the strongest crypto edge.

Approach 3: Range Trading
- Detect the 24h range (support/resistance). Buy near support, sell near resistance.
- Works in ranging markets (which is ~70% of the time).
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
TAKER_FEE = 0.0026
SLIPPAGE = 0.0003  # Lower slippage for limit orders

def download(pair, interval_min, days):
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - (days * 86400)
    all_candles = []
    since = start_ts
    while since < end_ts:
        try:
            resp = requests.get(f"{KRAKEN_BASE}/public/OHLC",
                params={"pair": pair, "interval": interval_min, "since": since}, timeout=30)
            result = resp.json().get("result", {})
        except:
            break
        candles = None
        for key, val in result.items():
            if isinstance(val, list):
                candles = val
                break
        if not candles:
            break
        for c in candles:
            ts = int(c[0])
            if ts > start_ts and ts <= end_ts:
                all_candles.append({
                    'time': ts, 'open': float(c[1]), 'high': float(c[2]),
                    'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[6])
                })
        last_ts = int(candles[-1][0])
        if last_ts <= since:
            break
        since = last_ts
        _time.sleep(1.2)
    df = pd.DataFrame(all_candles)
    if not df.empty:
        df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════
# APPROACH 1: Grid Trading
# ═══════════════════════════════════════════════════════════════
def test_grid(df, balance=300, grid_pct=0.005, num_levels=3,
              tp_pct=0.005, max_inventory_pct=0.50,
              use_maker=True):
    """
    Simple grid: place buy orders at grid_pct intervals below mid.
    When a buy fills (low touches the level), place a sell at tp_pct above buy.
    
    This simulates limit orders using OHLC:
    - Buy fills if bar LOW <= buy_price
    - Sell fills if bar HIGH >= sell_price
    """
    if len(df) < 10:
        return None

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    fee = MAKER_FEE if use_maker else TAKER_FEE
    
    init_balance = balance
    cash = balance
    inventory = []  # list of (buy_price, qty)
    total_inventory_cost = 0
    trades = []
    total_fees = 0
    completed_roundtrips = 0

    for i in range(1, len(df)):
        mid = close[i-1]  # Use previous close as reference
        cur_low = low[i]
        cur_high = high[i]
        cur_close = close[i]

        # Check sell fills first (from oldest inventory)
        new_inventory = []
        for buy_price, qty in inventory:
            sell_target = buy_price * (1 + tp_pct)
            if cur_high >= sell_target:
                # Sell fills
                sell_price = sell_target
                sell_fee = sell_price * qty * fee
                revenue = sell_price * qty - sell_fee
                buy_cost = buy_price * qty  # Already paid
                profit = revenue - buy_cost  # buy_cost includes buy fee
                cash += revenue
                total_inventory_cost -= buy_price * qty
                total_fees += sell_fee
                trades.append({'profit': profit, 'buy': buy_price, 'sell': sell_price,
                              'type': 'roundtrip'})
                completed_roundtrips += 1
            else:
                new_inventory.append((buy_price, qty))
        inventory = new_inventory

        # Check buy fills (grid levels below mid)
        max_inventory = init_balance * max_inventory_pct
        for level in range(1, num_levels + 1):
            buy_price = mid * (1 - grid_pct * level)
            
            if cur_low <= buy_price and total_inventory_cost < max_inventory:
                # Buy fills
                # Size: equal allocation per level
                alloc = cash / (num_levels - level + 1) if cash > 10 else 0
                alloc = min(alloc, cash * 0.3)  # Don't use too much per level
                if alloc < 3:
                    continue
                
                buy_fee = alloc * fee
                qty = (alloc - buy_fee) / buy_price
                actual_cost = alloc
                
                if cash >= actual_cost:
                    cash -= actual_cost
                    inventory.append((buy_price, qty))
                    total_inventory_cost += buy_price * qty
                    total_fees += buy_fee

    # Close remaining inventory at last price
    unrealized = 0
    for buy_price, qty in inventory:
        sell_price = close[-1]
        sell_fee = sell_price * qty * fee
        revenue = sell_price * qty - sell_fee
        unrealized += revenue - buy_price * qty
        cash += revenue
        total_fees += sell_fee

    final = cash
    total_profit = sum(t['profit'] for t in trades)

    return {
        'roundtrips': completed_roundtrips,
        'total_profit': round(total_profit, 2),
        'unrealized': round(unrealized, 2),
        'final': round(final, 2),
        'ret': round((final - init_balance) / init_balance * 100, 2),
        'fees': round(total_fees, 2),
        'avg_profit': round(total_profit / completed_roundtrips, 2) if completed_roundtrips > 0 else 0,
        'inventory_at_end': len(inventory),
    }


# ═══════════════════════════════════════════════════════════════
# APPROACH 2: Bollinger Band Mean Reversion
# ═══════════════════════════════════════════════════════════════
def test_bb_reversion(df, balance=300, bb_period=20, bb_std=2.0,
                      hold_bars=5, risk_pct=0.03, max_positions=3):
    """
    Buy when price touches lower BB. Sell when price reaches mid BB or hold_bars timeout.
    Also: sell when price touches upper BB (short via existing holdings).
    """
    if len(df) < bb_period + 10:
        return None

    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    sma = pd.Series(close).rolling(bb_period).mean().values
    std = pd.Series(close).rolling(bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std

    fee = MAKER_FEE + SLIPPAGE
    init_balance = balance
    positions = []  # (entry_price, qty, entry_bar, direction)
    trades = []

    for i in range(bb_period, len(df)):
        if np.isnan(sma[i]) or np.isnan(std[i]) or std[i] == 0:
            continue

        # Exit existing positions
        remaining = []
        for entry, qty, entry_bar, direction in positions:
            bars_held = i - entry_bar
            exit_price = None

            if direction == 'long':
                # Exit at mid BB or timeout
                if close[i] >= sma[i] or bars_held >= hold_bars:
                    exit_price = close[i]
                elif low[i] <= entry * 0.97:  # 3% stop loss
                    exit_price = entry * 0.97
            else:  # short
                if close[i] <= sma[i] or bars_held >= hold_bars:
                    exit_price = close[i]
                elif high[i] >= entry * 1.03:
                    exit_price = entry * 1.03

            if exit_price is not None:
                if direction == 'long':
                    pnl = (exit_price - entry) * qty
                else:
                    pnl = (entry - exit_price) * qty
                fees = entry * qty * fee + exit_price * qty * fee
                net = pnl - fees
                if direction == 'long':
                    balance += entry * qty + net
                else:
                    balance += net
                trades.append({'pnl': net, 'dir': direction, 'bars': bars_held})
            else:
                remaining.append((entry, qty, entry_bar, direction))
        positions = remaining

        # New entries
        if len(positions) >= max_positions:
            continue

        # Buy at lower BB
        if low[i] <= lower[i] and close[i] > lower[i] * 0.99:
            entry = close[i] * (1 + SLIPPAGE)
            risk = entry * 0.03  # 3% risk
            qty = (balance * risk_pct) / risk
            cost = qty * entry
            if cost <= balance * 0.95 and cost > 3:
                balance -= cost
                positions.append((entry, qty, i, 'long'))

        # Sell at upper BB (mean reversion short)
        elif high[i] >= upper[i] and close[i] < upper[i] * 1.01:
            entry = close[i] * (1 - SLIPPAGE)
            risk = entry * 0.03
            qty = (balance * risk_pct) / risk
            if qty * entry > 3:
                positions.append((entry, qty, i, 'short'))

    # Close remaining
    for entry, qty, entry_bar, direction in positions:
        exit_price = close[-1]
        if direction == 'long':
            pnl = (exit_price - entry) * qty
            fees = entry * qty * fee + exit_price * qty * fee
            balance += entry * qty + (pnl - fees)
        else:
            pnl = (entry - exit_price) * qty
            fees = entry * qty * fee + exit_price * qty * fee
            balance += (pnl - fees)
        trades.append({'pnl': pnl - fees, 'dir': direction, 'bars': i - entry_bar})

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001

    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'wr': round(len(wins)/len(trades)*100, 1) if trades else 0,
        'pf': round(gp/gl, 2) if gl > 0.001 else 999,
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'ret': round((balance - init_balance) / init_balance * 100, 2),
        'final': round(balance, 2),
        'avg_bars': round(np.mean([t['bars'] for t in trades]), 1) if trades else 0,
    }


# ═══════════════════════════════════════════════════════════════
# APPROACH 3: RSI Mean Reversion (oversold/overbought on fast TF)
# ═══════════════════════════════════════════════════════════════
def test_rsi_reversion(df, balance=300, rsi_period=7, oversold=25, overbought=75,
                       exit_rsi=50, hold_bars=8, risk_pct=0.03, max_positions=2):
    """Fast RSI mean reversion on short timeframes."""
    if len(df) < rsi_period + 10:
        return None

    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    # Calculate RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).values

    fee = MAKER_FEE + SLIPPAGE
    init_balance = balance
    positions = []
    trades = []

    for i in range(rsi_period + 5, len(df)):
        if np.isnan(rsi[i]):
            continue

        # Exit
        remaining = []
        for entry, qty, entry_bar, direction in positions:
            bars_held = i - entry_bar
            exit_price = None

            if direction == 'long':
                if rsi[i] >= exit_rsi or bars_held >= hold_bars:
                    exit_price = close[i]
                elif low[i] <= entry * 0.97:
                    exit_price = entry * 0.97
            else:
                if rsi[i] <= exit_rsi or bars_held >= hold_bars:
                    exit_price = close[i]
                elif high[i] >= entry * 1.03:
                    exit_price = entry * 1.03

            if exit_price is not None:
                if direction == 'long':
                    pnl = (exit_price - entry) * qty
                else:
                    pnl = (entry - exit_price) * qty
                fees = entry * qty * fee + exit_price * qty * fee
                net = pnl - fees
                if direction == 'long':
                    balance += entry * qty + net
                else:
                    balance += net
                trades.append({'pnl': net, 'dir': direction, 'bars': bars_held,
                              'entry_rsi': rsi[entry_bar], 'exit_rsi': rsi[i]})
            else:
                remaining.append((entry, qty, entry_bar, direction))
        positions = remaining

        if len(positions) >= max_positions:
            continue

        # Buy when oversold
        if rsi[i] <= oversold:
            entry = close[i] * (1 + SLIPPAGE)
            sl_dist = entry * 0.03
            qty = (balance * risk_pct) / sl_dist
            cost = qty * entry
            if cost <= balance * 0.95 and cost > 3:
                balance -= cost
                positions.append((entry, qty, i, 'long'))

        # Sell when overbought
        elif rsi[i] >= overbought:
            entry = close[i] * (1 - SLIPPAGE)
            sl_dist = entry * 0.03
            qty = (balance * risk_pct) / sl_dist
            if qty * entry > 3:
                positions.append((entry, qty, i, 'short'))

    # Close remaining
    for entry, qty, entry_bar, direction in positions:
        exit_price = close[-1]
        if direction == 'long':
            pnl = (exit_price - entry) * qty
            fees = entry * qty * fee + exit_price * qty * fee
            balance += entry * qty + (pnl - fees)
        else:
            pnl = (entry - exit_price) * qty
            fees = entry * qty * fee + exit_price * qty * fee
            balance += (pnl - fees)
        trades.append({'pnl': pnl - fees, 'dir': direction, 'bars': 0})

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001

    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'wr': round(len(wins)/len(trades)*100, 1) if trades else 0,
        'pf': round(gp/gl, 2) if gl > 0.001 else 999,
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'ret': round((balance - init_balance) / init_balance * 100, 2),
        'final': round(balance, 2),
        'avg_bars': round(np.mean([t['bars'] for t in trades]), 1) if trades else 0,
    }


def main():
    PAIRS = ["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "XRPUSD", "ATOMUSD",
             "LINKUSD", "AVAXUSD", "DOTUSD", "NEARUSD", "UNIUSD", "FILUSD"]
    
    # Download 15m data (7 days) and 1h data (30 days)
    print("Downloading data...")
    data_15m = {}
    data_1h = {}
    for pair in PAIRS:
        df = download(pair, 15, 7)
        if len(df) > 50:
            data_15m[pair] = df
            print(f"  {pair} 15m: {len(df)} bars")
        df = download(pair, 60, 30)
        if len(df) > 50:
            data_1h[pair] = df
            print(f"  {pair} 1h: {len(df)} bars")

    # ═══════════════════════════════════════════
    # GRID TRADING
    # ═══════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  GRID TRADING — Harvest the noise")
    print(f"{'='*80}")
    print(f"\n  {'Pair':10s} {'TF':>4s} {'Grid%':>6s} {'TP%':>5s} {'Lvls':>5s} {'RTs':>5s} {'Profit':>8s} {'Ret':>7s} {'Fees':>7s} {'AvgRT':>7s} {'InvEnd':>6s}")
    print("  " + "-" * 80)

    for pair in PAIRS:
        for tf_name, df_dict in [("15m", data_15m), ("1h", data_1h)]:
            if pair not in df_dict:
                continue
            df = df_dict[pair]
            for grid_pct in [0.003, 0.005, 0.008, 0.01, 0.015]:
                for tp_pct in [0.003, 0.005, 0.008, 0.01]:
                    for levels in [2, 3, 5]:
                        r = test_grid(df, grid_pct=grid_pct, tp_pct=tp_pct,
                                     num_levels=levels, max_inventory_pct=0.5)
                        if r and r['roundtrips'] > 0 and r['ret'] > 0:
                            color = "\033[92m"
                            reset = "\033[0m"
                            print(f"  {pair:10s} {tf_name:>4s} {grid_pct*100:>5.1f}% {tp_pct*100:>4.1f}% "
                                  f"{levels:>5d} {r['roundtrips']:>5d} "
                                  f"{color}${r['total_profit']:>+7.2f} {r['ret']:>+6.2f}%{reset} "
                                  f"${r['fees']:>6.2f} ${r['avg_profit']:>6.2f} {r['inventory_at_end']:>6d}")

    # ═══════════════════════════════════════════
    # BB MEAN REVERSION
    # ═══════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  BOLLINGER BAND MEAN REVERSION")
    print(f"{'='*80}")
    print(f"\n  {'Pair':10s} {'TF':>4s} {'BB':>4s} {'Std':>4s} {'Hold':>5s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'PnL':>8s} {'Ret':>7s} {'AvgBars':>7s}")
    print("  " + "-" * 80)

    for pair in PAIRS:
        for tf_name, df_dict in [("15m", data_15m), ("1h", data_1h)]:
            if pair not in df_dict:
                continue
            df = df_dict[pair]
            for bb_p in [10, 15, 20]:
                for bb_s in [1.5, 2.0, 2.5]:
                    for hold in [3, 5, 8, 12]:
                        r = test_bb_reversion(df, bb_period=bb_p, bb_std=bb_s,
                                             hold_bars=hold)
                        if r and r['trades'] >= 3 and r['ret'] > 0:
                            color = "\033[92m"
                            reset = "\033[0m"
                            print(f"  {pair:10s} {tf_name:>4s} {bb_p:>4d} {bb_s:>4.1f} {hold:>5d} "
                                  f"{r['trades']:>5d}T {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                                  f"{color}${r['pnl']:>+7.2f} {r['ret']:>+6.2f}%{reset} {r['avg_bars']:>7.1f}")

    # ═══════════════════════════════════════════
    # RSI MEAN REVERSION
    # ═══════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  RSI MEAN REVERSION (Fast)")
    print(f"{'='*80}")
    print(f"\n  {'Pair':10s} {'TF':>4s} {'RSI':>4s} {'OS':>4s} {'OB':>4s} {'Hold':>5s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'PnL':>8s} {'Ret':>7s}")
    print("  " + "-" * 80)

    for pair in PAIRS:
        for tf_name, df_dict in [("15m", data_15m), ("1h", data_1h)]:
            if pair not in df_dict:
                continue
            df = df_dict[pair]
            for rsi_p in [5, 7, 9, 14]:
                for os_val in [20, 25, 30]:
                    for ob_val in [70, 75, 80]:
                        for hold in [3, 5, 8, 12]:
                            r = test_rsi_reversion(df, rsi_period=rsi_p,
                                                  oversold=os_val, overbought=ob_val,
                                                  hold_bars=hold)
                            if r and r['trades'] >= 3 and r['ret'] > 0:
                                color = "\033[92m"
                                reset = "\033[0m"
                                print(f"  {pair:10s} {tf_name:>4s} {rsi_p:>4d} {os_val:>4d} {ob_val:>4d} {hold:>5d} "
                                      f"{r['trades']:>5d}T {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                                      f"{color}${r['pnl']:>+7.2f} {r['ret']:>+6.2f}%{reset}")

    print(f"\n{'='*80}")
    print(f"  DONE — Green lines = profitable configs")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
