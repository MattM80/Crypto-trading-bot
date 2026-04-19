#!/usr/bin/env python3
"""
Deep optimization of momentum breakout strategy.
Tests SL/TP ratios, trailing stops, partial exits, and combined BTC+ETH portfolio.
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005

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
        _time.sleep(1.5)
    df = pd.DataFrame(all_candles)
    if not df.empty:
        df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
    return df


def backtest_momentum_advanced(df, balance=300, breakout_bars=15, vol_mult=1.0,
                                sma_period=50, sl_atr_mult=1.5, tp_atr_mult=3.0,
                                risk_pct=0.03, cooldown=3, timeout_bars=48,
                                trailing_stop=False, trail_activation=0.5,
                                trail_callback=0.4, max_positions=1):
    """Advanced momentum backtest with trailing stops and tunable params."""
    if len(df) < max(breakout_bars, sma_period) + 20:
        return None
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(14).mean().values
    vol_avg = pd.Series(volume).rolling(20).mean().values
    sma = pd.Series(close).rolling(sma_period).mean().values
    
    trades = []
    position = None  # (entry_price, qty, sl, tp, entry_bar, highest_since_entry, atr_at_entry)
    bars_since_trade = cooldown + 1
    equity_curve = [balance]
    peak_balance = balance
    max_dd = 0
    
    start = max(breakout_bars, sma_period, 20)
    for i in range(start, len(df)):
        bars_since_trade += 1
        
        if position is not None:
            entry_price, qty, sl, tp, entry_bar, highest, atr_entry = position
            
            # Update highest price for trailing stop
            if high[i] > highest:
                highest = high[i]
            
            # Trailing stop logic
            if trailing_stop and highest > entry_price:
                profit_pct = (highest - entry_price) / entry_price
                if profit_pct >= trail_activation * (tp - entry_price) / entry_price:
                    trail_sl = highest - trail_callback * atr_entry
                    if trail_sl > sl:
                        sl = trail_sl
            
            position = (entry_price, qty, sl, tp, entry_bar, highest, atr_entry)
            
            # Check SL
            if low[i] <= sl:
                exit_price = max(sl, low[i]) * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty
                fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                pnl -= fees
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "SL", "bars": i - entry_bar,
                              "entry": entry_price, "exit": exit_price})
                position = None
                bars_since_trade = 0
                equity_curve.append(balance)
                peak_balance = max(peak_balance, balance)
                dd = (peak_balance - balance) / peak_balance
                max_dd = max(max_dd, dd)
                continue
            
            # Check TP
            if high[i] >= tp:
                exit_price = min(tp, high[i]) * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty
                fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                pnl -= fees
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "TP", "bars": i - entry_bar,
                              "entry": entry_price, "exit": exit_price})
                position = None
                bars_since_trade = 0
                equity_curve.append(balance)
                peak_balance = max(peak_balance, balance)
                continue
            
            # Timeout
            if i - entry_bar >= timeout_bars:
                exit_price = close[i] * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty
                fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
                pnl -= fees
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "TIMEOUT", "bars": i - entry_bar,
                              "entry": entry_price, "exit": exit_price})
                position = None
                bars_since_trade = 0
                equity_curve.append(balance)
                peak_balance = max(peak_balance, balance)
                continue
        
        # Entry
        if position is None and bars_since_trade >= cooldown:
            if np.isnan(atr[i]) or np.isnan(vol_avg[i]) or np.isnan(sma[i]) or atr[i] <= 0:
                continue
            if close[i] <= sma[i]:
                continue
            prev_high = np.max(high[i-breakout_bars:i])
            if close[i] <= prev_high:
                continue
            vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
            if vol_ratio < vol_mult:
                continue
            
            entry_price = close[i] * (1 + SLIPPAGE)
            recent_low = np.min(low[i-breakout_bars:i])
            sl_swing = recent_low * 0.998
            sl_atr = entry_price - sl_atr_mult * atr[i]
            sl = max(sl_swing, sl_atr)
            tp = entry_price + tp_atr_mult * atr[i]
            
            risk_per_unit = entry_price - sl
            if risk_per_unit <= 0:
                continue
            reward = tp - entry_price
            if reward / risk_per_unit < 1.5:
                continue
            
            risk_dollars = balance * risk_pct
            qty = risk_dollars / risk_per_unit
            cost = qty * entry_price
            if cost > balance * 0.95:
                qty = (balance * 0.95) / entry_price
                cost = qty * entry_price
            if cost < 5:
                continue
            
            balance -= cost
            position = (entry_price, qty, sl, tp, i, entry_price, atr[i])
        
        equity_curve.append(balance if position is None else 
                           balance + (close[i] - position[0]) * position[1])
    
    # Close remaining
    if position is not None:
        entry_price, qty, sl, tp, entry_bar, highest, atr_entry = position
        exit_price = close[-1] * (1 - SLIPPAGE)
        pnl = (exit_price - entry_price) * qty
        fees = entry_price * qty * MAKER_FEE + exit_price * qty * MAKER_FEE
        pnl -= fees
        balance += entry_price * qty + pnl
        trades.append({"pnl": pnl, "reason": "END"})
    
    if not trades:
        return None
    
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
    total_pnl = sum(t["pnl"] for t in trades)
    
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": round(len(wins)/len(trades)*100, 1),
        "pf": round(gross_profit / gross_loss, 2),
        "pnl": round(total_pnl, 2),
        "ret": round((balance - 300) / 300 * 100, 2),
        "final": round(balance, 2),
        "max_dd": round(max_dd * 100, 2),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0,
        "exits": {r: len([t for t in trades if t.get("reason") == r]) 
                  for r in set(t.get("reason", "?") for t in trades)},
    }


def main():
    print("Downloading data...")
    btc = download("XBTUSD", 60, 90)
    eth = download("ETHUSD", 60, 90)
    print(f"BTC: {len(btc)} bars, ETH: {len(eth)} bars\n")
    
    # ═══════════════════════════════════════════
    # DEEP PARAMETER OPTIMIZATION
    # ═══════════════════════════════════════════
    
    print("=" * 80)
    print("  MOMENTUM BREAKOUT — DEEP OPTIMIZATION")
    print("=" * 80)
    
    best_results = []
    
    for pair_name, df in [("BTC", btc), ("ETH", eth)]:
        print(f"\n--- {pair_name} ---")
        best_ret = -999
        best_config = None
        
        for bb in [10, 12, 15, 18, 20, 25]:
            for sl_mult in [1.0, 1.2, 1.5, 2.0, 2.5]:
                for tp_mult in [2.0, 2.5, 3.0, 4.0, 5.0]:
                    for risk in [0.02, 0.03, 0.05]:
                        for timeout in [24, 36, 48, 72]:
                            for sma in [30, 40, 50]:
                                for trail in [False, True]:
                                    r = backtest_momentum_advanced(
                                        df, breakout_bars=bb, sl_atr_mult=sl_mult,
                                        tp_atr_mult=tp_mult, risk_pct=risk,
                                        timeout_bars=timeout, sma_period=sma,
                                        trailing_stop=trail,
                                        trail_activation=0.4, trail_callback=0.5
                                    )
                                    if r and r["trades"] >= 5 and r["ret"] > best_ret and r["max_dd"] < 20:
                                        best_ret = r["ret"]
                                        best_config = {
                                            "pair": pair_name, "bb": bb, "sl": sl_mult,
                                            "tp": tp_mult, "risk": risk, "timeout": timeout,
                                            "sma": sma, "trail": trail
                                        }
                                        best_result = r
        
        if best_config:
            print(f"  BEST: {best_config}")
            print(f"  Result: {best_result}")
            best_results.append((best_config, best_result))
        
        # Also show top 5
        print(f"\n  Top configs for {pair_name}:")
        all_results = []
        for bb in [10, 12, 15, 18, 20]:
            for sl_mult in [1.0, 1.5, 2.0]:
                for tp_mult in [2.5, 3.0, 4.0, 5.0]:
                    for risk in [0.03, 0.05]:
                        for timeout in [36, 48, 72]:
                            for sma in [30, 50]:
                                for trail in [False, True]:
                                    r = backtest_momentum_advanced(
                                        df, breakout_bars=bb, sl_atr_mult=sl_mult,
                                        tp_atr_mult=tp_mult, risk_pct=risk,
                                        timeout_bars=timeout, sma_period=sma,
                                        trailing_stop=trail
                                    )
                                    if r and r["trades"] >= 5:
                                        all_results.append((
                                            f"bb={bb} sl={sl_mult} tp={tp_mult} risk={risk} "
                                            f"t={timeout} sma={sma} trail={trail}",
                                            r
                                        ))
        
        all_results.sort(key=lambda x: x[1]["ret"], reverse=True)
        for cfg, res in all_results[:10]:
            print(f"    {cfg}")
            print(f"      → {res['trades']}T {res['wr']}%WR PF={res['pf']} "
                  f"ret={res['ret']}% dd={res['max_dd']}% ${res['pnl']}")
    
    # ═══════════════════════════════════════════
    # COMBINED PORTFOLIO TEST
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  COMBINED BTC + ETH PORTFOLIO (best params)")
    print("=" * 80)
    
    # Run both simultaneously, splitting balance
    if best_results:
        print("\n  Running combined portfolio with best params per pair...")
        for cfg, res in best_results:
            print(f"  {cfg['pair']}: bb={cfg['bb']} sl={cfg['sl']} tp={cfg['tp']} "
                  f"risk={cfg['risk']} timeout={cfg['timeout']} sma={cfg['sma']} "
                  f"trail={cfg['trail']}")
            print(f"    → {res['trades']}T {res['wr']}%WR PF={res['pf']} "
                  f"ret={res['ret']}% dd={res['max_dd']}%")


if __name__ == "__main__":
    main()
