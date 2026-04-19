#!/usr/bin/env python3
"""
Combined portfolio simulation of the master bot ecosystem.
Grid + RSI + BB all running simultaneously on $300, with regime detection.
This is the kill-or-confirm test.
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
SLIPPAGE_LIMIT = 0.0003  # Grid/limit orders
SLIPPAGE_MARKET = 0.0005  # Market orders
INITIAL_BALANCE = 300
GRID_ALLOC_PCT = 0.40  # 40% to grids
ACTIVE_ALLOC_PCT = 0.60  # 60% to active strategies
MAX_ACTIVE_POSITIONS = 5
RISK_PER_TRADE = 0.03  # 3% of active balance

# Per-pair configs (from backtests)
PAIR_CONFIGS = {
    "NEARUSD": {
        "grid": {"grid_pct": 0.01, "tp_pct": 0.01, "levels": 5},
        "rsi": {"period": 9, "oversold": 20, "overbought": 80, "hold": 8},
    },
    "ATOMUSD": {
        "rsi": {"period": 7, "oversold": 20, "overbought": 80, "hold": 5},
    },
    "AVAXUSD": {
        "grid": {"grid_pct": 0.008, "tp_pct": 0.01, "levels": 5},
        "rsi": {"period": 14, "oversold": 20, "overbought": 70, "hold": 8},
    },
    "DOTUSD": {
        "grid": {"grid_pct": 0.015, "tp_pct": 0.01, "levels": 5},
        "bb": {"period": 20, "std": 2.0, "hold": 12},
    },
    "XRPUSD": {
        "rsi": {"period": 9, "oversold": 20, "overbought": 80, "hold": 3},
    },
    "FILUSD": {
        "rsi": {"period": 14, "oversold": 20, "overbought": 80, "hold": 8},
        "bb": {"period": 20, "std": 2.5, "hold": 8},
    },
    "UNIUSD": {
        "grid": {"grid_pct": 0.015, "tp_pct": 0.01, "levels": 5},
        "rsi": {"period": 9, "oversold": 20, "overbought": 75, "hold": 8},
    },
    "XBTUSD": {
        "rsi": {"period": 14, "oversold": 20, "overbought": 80, "hold": 8},
    },
    "ETHUSD": {
        "rsi": {"period": 14, "oversold": 20, "overbought": 80, "hold": 12},
    },
    "SOLUSD": {
        "rsi": {"period": 14, "oversold": 20, "overbought": 80, "hold": 8},
    },
    "LINKUSD": {
        "rsi": {"period": 14, "oversold": 20, "overbought": 75, "hold": 8},
    },
    "ADAUSD": {
        "rsi": {"period": 14, "oversold": 20, "overbought": 80, "hold": 12},
    },
}

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


def calc_rsi(close, period):
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def calc_bb(close, period, std_mult):
    s = pd.Series(close)
    sma = s.rolling(period).mean().values
    std = s.rolling(period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, sma, lower


def detect_regime(close, high, low, sma_period=50, adx_period=14):
    """Simple regime detection."""
    if len(close) < max(sma_period, adx_period * 2) + 5:
        return "unknown"
    
    c = np.array(close, dtype=float)
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    
    # SMA
    sma = float(np.mean(c[-sma_period:]))
    
    # ATR
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_vals = tr[-adx_period:]
    cur_atr = float(np.mean(atr_vals))
    med_atr = float(np.median(tr[-50:])) if len(tr) >= 50 else cur_atr
    
    # Volatility check
    if med_atr > 0 and cur_atr > 1.5 * med_atr:
        return "volatile"
    
    # Simple trend: use 20-bar slope
    if len(c) >= 20:
        slope = (c[-1] - c[-20]) / c[-20]
        if abs(slope) > 0.03:  # >3% move in 20 bars = trending
            return "trending_up" if slope > 0 else "trending_down"
    
    # Price vs SMA
    if c[-1] > sma * 1.02:
        return "trending_up"
    elif c[-1] < sma * 0.98:
        return "trending_down"
    
    return "ranging"


class GridSim:
    """Simulates grid trading for one pair."""
    def __init__(self, pair, config, allocation):
        self.pair = pair
        self.grid_pct = config["grid_pct"]
        self.tp_pct = config["tp_pct"]
        self.levels = config["levels"]
        self.allocation = allocation
        self.buy_orders = []  # (price, qty)
        self.filled_buys = []  # (buy_price, qty)
        self.total_profit = 0
        self.total_fees = 0
        self.roundtrips = 0
        self.initialized = False
        self.inventory_cost = 0  # Total USD locked in inventory
    
    def initialize(self, current_price):
        if self.initialized:
            return
        alloc_per_level = self.allocation / self.levels
        for i in range(1, self.levels + 1):
            buy_price = current_price * (1 - self.grid_pct * i)
            qty = alloc_per_level / buy_price
            self.buy_orders.append((buy_price, qty))
        self.initialized = True
    
    def update(self, cur_low, cur_high, cur_close):
        """Process one bar. Returns (profit_this_bar, fees_this_bar)."""
        if not self.initialized:
            self.initialize(cur_close)
            
        profit = 0
        fees = 0
        
        # Check sell fills first (from filled buys)
        remaining_fills = []
        for buy_price, qty in self.filled_buys:
            sell_target = buy_price * (1 + self.tp_pct)
            if cur_high >= sell_target:
                sell_fee = sell_target * qty * MAKER_FEE
                revenue = sell_target * qty - sell_fee
                cost = buy_price * qty  # What we paid (including buy fee)
                rp = revenue - cost
                profit += rp
                fees += sell_fee
                self.roundtrips += 1
                self.inventory_cost -= buy_price * qty
                
                # Re-place buy order at the same level
                self.buy_orders.append((buy_price, qty))
            else:
                remaining_fills.append((buy_price, qty))
        self.filled_buys = remaining_fills
        
        # Check buy fills
        remaining_orders = []
        for buy_price, qty in self.buy_orders:
            if cur_low <= buy_price:
                buy_fee = buy_price * qty * MAKER_FEE
                # Track cost including fee
                self.filled_buys.append((buy_price * (1 + MAKER_FEE / buy_price), qty))
                self.inventory_cost += buy_price * qty + buy_fee
                fees += buy_fee
            else:
                remaining_orders.append((buy_price, qty))
        self.buy_orders = remaining_orders
        
        self.total_profit += profit
        self.total_fees += fees
        return profit, fees
    
    def get_unrealized(self, current_price):
        """Get unrealized P&L of inventory."""
        unrealized = 0
        for buy_price, qty in self.filled_buys:
            unrealized += (current_price - buy_price) * qty
        return unrealized


def main():
    print("=" * 70)
    print("  MASTER BOT PORTFOLIO SIMULATION")
    print("  Grid + RSI + BB, regime-adaptive, $300 start")
    print("=" * 70)
    
    # Download 1h data for all pairs (30 days)
    print("\nDownloading 1h data for all pairs...")
    all_data = {}
    for pair in PAIR_CONFIGS:
        df = download(pair, 60, 30)
        if len(df) > 50:
            all_data[pair] = df
            print(f"  {pair}: {len(df)} bars, ${df['close'].iloc[-1]:.4f}")
    
    print(f"\nLoaded {len(all_data)} pairs")
    
    # Find common length
    min_len = min(len(df) for df in all_data.values())
    print(f"Simulating {min_len} bars (~{min_len/24:.0f} days)")
    
    # ═══════════════════════════════════════════
    # INITIALIZE
    # ═══════════════════════════════════════════
    
    grid_balance = INITIAL_BALANCE * GRID_ALLOC_PCT  # $120
    active_balance = INITIAL_BALANCE * ACTIVE_ALLOC_PCT  # $180
    
    # Initialize grid managers
    grid_pairs = [p for p in PAIR_CONFIGS if "grid" in PAIR_CONFIGS[p]]
    alloc_per_grid = grid_balance / len(grid_pairs) if grid_pairs else 0
    grids = {}
    for pair in grid_pairs:
        if pair in all_data:
            grids[pair] = GridSim(pair, PAIR_CONFIGS[pair]["grid"], alloc_per_grid)
    
    # Active strategy state
    active_positions = []  # (pair, direction, entry_price, qty, entry_bar, strategy, hold_limit)
    active_trades = []
    peak_active = active_balance
    max_dd = 0
    
    # Tracking
    equity_curve = []
    grid_profit_total = 0
    grid_fees_total = 0
    regime_counts = defaultdict(int)
    
    # ═══════════════════════════════════════════
    # BAR-BY-BAR SIMULATION
    # ═══════════════════════════════════════════
    
    for bar_idx in range(60, min_len):  # Start at 60 for indicator warmup
        
        # ── GRID UPDATES ──
        bar_grid_profit = 0
        for pair, grid in grids.items():
            df = all_data[pair]
            if bar_idx >= len(df):
                continue
            cur_low = float(df.iloc[bar_idx]['low'])
            cur_high = float(df.iloc[bar_idx]['high'])
            cur_close = float(df.iloc[bar_idx]['close'])
            profit, fees = grid.update(cur_low, cur_high, cur_close)
            bar_grid_profit += profit
            grid_profit_total += profit
            grid_fees_total += fees
        
        # ── MANAGE ACTIVE POSITIONS ──
        remaining = []
        for pair, direction, entry, qty, entry_bar, strategy, hold_limit in active_positions:
            df = all_data[pair]
            if bar_idx >= len(df):
                remaining.append((pair, direction, entry, qty, entry_bar, strategy, hold_limit))
                continue
            
            cur_high = float(df.iloc[bar_idx]['high'])
            cur_low = float(df.iloc[bar_idx]['low'])
            cur_close = float(df.iloc[bar_idx]['close'])
            bars_held = bar_idx - entry_bar
            
            exit_price = None
            reason = None
            
            if direction == 'long':
                # 3% stop loss
                if cur_low <= entry * 0.97:
                    exit_price = entry * 0.97
                    reason = "SL"
                # RSI/BB target: check if price reverted to mean
                elif strategy == "rsi":
                    rsi_vals = calc_rsi(df['close'].values[:bar_idx+1], PAIR_CONFIGS[pair]["rsi"]["period"])
                    if not np.isnan(rsi_vals[-1]) and rsi_vals[-1] >= 50:
                        exit_price = cur_close
                        reason = "RSI_REVERT"
                elif strategy == "bb":
                    _, mid, _ = calc_bb(df['close'].values[:bar_idx+1],
                                       PAIR_CONFIGS[pair]["bb"]["period"],
                                       PAIR_CONFIGS[pair]["bb"]["std"])
                    if not np.isnan(mid[-1]) and cur_close >= mid[-1]:
                        exit_price = cur_close
                        reason = "BB_MID"
                # Timeout
                if exit_price is None and bars_held >= hold_limit:
                    exit_price = cur_close
                    reason = "TIMEOUT"
                    
            else:  # short
                if cur_high >= entry * 1.03:
                    exit_price = entry * 1.03
                    reason = "SL"
                elif strategy == "rsi":
                    rsi_vals = calc_rsi(df['close'].values[:bar_idx+1], PAIR_CONFIGS[pair]["rsi"]["period"])
                    if not np.isnan(rsi_vals[-1]) and rsi_vals[-1] <= 50:
                        exit_price = cur_close
                        reason = "RSI_REVERT"
                elif strategy == "bb":
                    _, mid, _ = calc_bb(df['close'].values[:bar_idx+1],
                                       PAIR_CONFIGS[pair]["bb"]["period"],
                                       PAIR_CONFIGS[pair]["bb"]["std"])
                    if not np.isnan(mid[-1]) and cur_close <= mid[-1]:
                        exit_price = cur_close
                        reason = "BB_MID"
                if exit_price is None and bars_held >= hold_limit:
                    exit_price = cur_close
                    reason = "TIMEOUT"
            
            if exit_price is not None:
                if direction == 'long':
                    pnl = (exit_price - entry) * qty
                else:
                    pnl = (entry - exit_price) * qty
                fees = entry * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                net = pnl - fees
                
                if direction == 'long':
                    active_balance += entry * qty + net  # Return capital + pnl
                else:
                    active_balance += net
                
                active_trades.append({
                    'pair': pair, 'dir': direction, 'strategy': strategy,
                    'entry': entry, 'exit': exit_price, 'pnl': net,
                    'reason': reason, 'bars': bars_held,
                })
                peak_active = max(peak_active, active_balance)
                dd = (peak_active - active_balance) / peak_active if peak_active > 0 else 0
                max_dd = max(max_dd, dd)
            else:
                remaining.append((pair, direction, entry, qty, entry_bar, strategy, hold_limit))
        
        active_positions = remaining
        
        # ── GENERATE NEW SIGNALS ──
        if len(active_positions) < MAX_ACTIVE_POSITIONS:
            # Check drawdown
            dd = (peak_active - active_balance) / peak_active if peak_active > 0 else 0
            if dd <= 0.15:
                active_pairs = set(p[0] for p in active_positions)
                
                for pair in all_data:
                    if pair in active_pairs:
                        continue
                    if len(active_positions) >= MAX_ACTIVE_POSITIONS:
                        break
                    
                    df = all_data[pair]
                    if bar_idx >= len(df):
                        continue
                    
                    close_arr = df['close'].values[:bar_idx+1].astype(float)
                    high_arr = df['high'].values[:bar_idx+1].astype(float)
                    low_arr = df['low'].values[:bar_idx+1].astype(float)
                    
                    if len(close_arr) < 60:
                        continue
                    
                    regime = detect_regime(close_arr, high_arr, low_arr)
                    regime_counts[regime] += 1
                    
                    config = PAIR_CONFIGS.get(pair, {})
                    cur_close = float(close_arr[-1])
                    
                    # RSI strategy (ranging + volatile)
                    if "rsi" in config and regime in ("ranging", "volatile", "trending_down", "trending_up"):
                        rsi_cfg = config["rsi"]
                        rsi_vals = calc_rsi(close_arr, rsi_cfg["period"])
                        cur_rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
                        
                        if cur_rsi <= rsi_cfg["oversold"]:
                            entry = cur_close * (1 + SLIPPAGE_MARKET)
                            risk = entry * 0.03
                            qty = (active_balance * RISK_PER_TRADE) / risk
                            cost = qty * entry
                            if cost <= active_balance * 0.30 and cost > 3:
                                active_balance -= cost
                                active_positions.append((pair, 'long', entry, qty, bar_idx,
                                                        'rsi', rsi_cfg["hold"]))
                        
                        elif cur_rsi >= rsi_cfg["overbought"]:
                            entry = cur_close * (1 - SLIPPAGE_MARKET)
                            risk = entry * 0.03
                            qty = (active_balance * RISK_PER_TRADE) / risk
                            if qty * entry > 3 and active_balance * RISK_PER_TRADE < active_balance * 0.30:
                                active_positions.append((pair, 'short', entry, qty, bar_idx,
                                                        'rsi', rsi_cfg["hold"]))
                    
                    # BB strategy (ranging)
                    if "bb" in config and regime in ("ranging", "volatile"):
                        bb_cfg = config["bb"]
                        if len(close_arr) >= bb_cfg["period"] + 5:
                            upper, mid, lower = calc_bb(close_arr, bb_cfg["period"], bb_cfg["std"])
                            cur_upper = upper[-1]
                            cur_lower = lower[-1]
                            
                            if not np.isnan(cur_lower) and cur_close <= cur_lower:
                                entry = cur_close * (1 + SLIPPAGE_MARKET)
                                risk = entry * 0.03
                                qty = (active_balance * RISK_PER_TRADE) / risk
                                cost = qty * entry
                                if cost <= active_balance * 0.30 and cost > 3:
                                    active_balance -= cost
                                    active_positions.append((pair, 'long', entry, qty, bar_idx,
                                                            'bb', bb_cfg["hold"]))
                            
                            elif not np.isnan(cur_upper) and cur_close >= cur_upper:
                                entry = cur_close * (1 - SLIPPAGE_MARKET)
                                risk = entry * 0.03
                                qty = (active_balance * RISK_PER_TRADE) / risk
                                if qty * entry > 3:
                                    active_positions.append((pair, 'short', entry, qty, bar_idx,
                                                            'bb', bb_cfg["hold"]))
        
        # Track equity
        grid_unrealized = sum(g.get_unrealized(float(all_data[p].iloc[min(bar_idx, len(all_data[p])-1)]['close']))
                             for p, g in grids.items())
        total_equity = active_balance + grid_balance + grid_profit_total + grid_unrealized
        
        # Add unrealized active P&L
        for pair, direction, entry, qty, entry_bar, strategy, hold_limit in active_positions:
            df = all_data[pair]
            if bar_idx < len(df):
                cur = float(df.iloc[bar_idx]['close'])
                if direction == 'long':
                    total_equity += (cur - entry) * qty
                else:
                    total_equity += (entry - cur) * qty
        
        equity_curve.append(total_equity)
    
    # ── Close remaining active positions ──
    for pair, direction, entry, qty, entry_bar, strategy, hold_limit in active_positions:
        df = all_data[pair]
        exit_price = float(df.iloc[-1]['close'])
        if direction == 'long':
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty
        fees = entry * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
        net = pnl - fees
        if direction == 'long':
            active_balance += entry * qty + net
        else:
            active_balance += net
        active_trades.append({
            'pair': pair, 'dir': direction, 'strategy': strategy,
            'entry': entry, 'exit': exit_price, 'pnl': net,
            'reason': 'END', 'bars': min_len - entry_bar,
        })
    
    # ═══════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════
    
    # Grid results
    grid_inventory_value = sum(g.get_unrealized(float(all_data[p].iloc[-1]['close']))
                              for p, g in grids.items() if p in all_data)
    grid_total = grid_profit_total + grid_inventory_value
    
    # Active results
    active_wins = [t for t in active_trades if t['pnl'] > 0]
    active_losses = [t for t in active_trades if t['pnl'] <= 0]
    active_pnl = sum(t['pnl'] for t in active_trades)
    active_gp = sum(t['pnl'] for t in active_wins) if active_wins else 0
    active_gl = abs(sum(t['pnl'] for t in active_losses)) if active_losses else 0.001
    
    # Combined
    total_pnl = grid_total + active_pnl
    final_balance = INITIAL_BALANCE + total_pnl
    
    print(f"\n{'='*70}")
    print(f"  PORTFOLIO SIMULATION RESULTS — {min_len} bars (~{min_len/24:.0f} days)")
    print(f"{'='*70}")
    
    print(f"\n  GRID TRADING (40% allocation = ${INITIAL_BALANCE * GRID_ALLOC_PCT:.0f})")
    print(f"  {'─'*50}")
    for pair, grid in grids.items():
        unrealized = grid.get_unrealized(float(all_data[pair].iloc[-1]['close']))
        print(f"    {pair:10s}: {grid.roundtrips:3d} round-trips, "
              f"profit=${grid.total_profit:+.2f}, unrealized=${unrealized:+.2f}, "
              f"fees=${grid.total_fees:.2f}")
    print(f"    {'TOTAL':10s}: {sum(g.roundtrips for g in grids.values())} round-trips, "
          f"realized=${grid_profit_total:+.2f}, unrealized=${grid_inventory_value:+.2f}")
    
    print(f"\n  ACTIVE STRATEGIES (60% allocation = ${INITIAL_BALANCE * ACTIVE_ALLOC_PCT:.0f})")
    print(f"  {'─'*50}")
    print(f"    Trades: {len(active_trades)}")
    print(f"    Wins: {len(active_wins)} | Losses: {len(active_losses)}")
    print(f"    Win rate: {len(active_wins)/len(active_trades)*100:.1f}%" if active_trades else "    Win rate: N/A")
    print(f"    PF: {active_gp/active_gl:.2f}" if active_gl > 0.001 else "    PF: N/A")
    print(f"    PnL: ${active_pnl:+.2f}")
    print(f"    Max DD: {max_dd*100:.1f}%")
    
    # By strategy
    strat_groups = defaultdict(list)
    for t in active_trades:
        strat_groups[t['strategy']].append(t)
    for strat, trades in strat_groups.items():
        w = sum(1 for t in trades if t['pnl'] > 0)
        pnl = sum(t['pnl'] for t in trades)
        print(f"    {strat.upper():6s}: {len(trades)}T, {w}W/{len(trades)-w}L, ${pnl:+.2f}")
    
    # By exit reason
    reasons = defaultdict(int)
    for t in active_trades:
        reasons[t['reason']] += 1
    print(f"    Exits: {dict(reasons)}")
    
    print(f"\n  COMBINED")
    print(f"  {'─'*50}")
    print(f"    Starting balance:  ${INITIAL_BALANCE:.2f}")
    print(f"    Grid P&L:          ${grid_total:+.2f}")
    print(f"    Active P&L:        ${active_pnl:+.2f}")
    
    color = "\033[92m" if total_pnl > 0 else "\033[91m"
    reset = "\033[0m"
    print(f"    {color}Total P&L:          ${total_pnl:+.2f}{reset}")
    print(f"    {color}Final balance:      ${final_balance:.2f}{reset}")
    print(f"    {color}Return:             {total_pnl/INITIAL_BALANCE*100:+.2f}%{reset}")
    
    # Regime distribution
    total_regime = sum(regime_counts.values())
    print(f"\n  REGIME DISTRIBUTION")
    print(f"  {'─'*50}")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {regime:15s}: {count:5d} ({count/total_regime*100:.1f}%)")
    
    # Individual trades
    if active_trades:
        print(f"\n  TRADE LOG")
        print(f"  {'─'*70}")
        print(f"  {'Pair':10s} {'Dir':6s} {'Strat':6s} {'Entry':>10s} {'Exit':>10s} {'PnL':>8s} {'Reason':>10s} {'Bars':>5s}")
        for t in active_trades:
            c = "\033[92m" if t['pnl'] > 0 else "\033[91m"
            print(f"  {t['pair']:10s} {t['dir']:6s} {t['strategy']:6s} "
                  f"${t['entry']:>9.4f} ${t['exit']:>9.4f} "
                  f"{c}${t['pnl']:>+7.2f}\033[0m {t['reason']:>10s} {t['bars']:>5d}")
    
    # Monthly projection
    days_tested = min_len / 24
    if total_pnl > 0 and days_tested > 0:
        monthly_ret = (total_pnl / INITIAL_BALANCE) * (30 / days_tested)
        print(f"\n  PROJECTION (if this rate holds)")
        print(f"  {'─'*50}")
        print(f"    Monthly return: ~{monthly_ret*100:.1f}%")
        bal = INITIAL_BALANCE
        for m in [3, 6, 12, 24, 36]:
            bal = INITIAL_BALANCE * (1 + monthly_ret) ** m
            print(f"    {m:2d} months: ${bal:>10,.2f}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
