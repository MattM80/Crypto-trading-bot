#!/usr/bin/env python3
"""
Raw, no-BS backtester. Downloads Kraken data, tests strategies bar-by-bar.
No frameworks, no abstractions, just math.
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016   # 0.16%
SLIPPAGE = 0.0005     # 0.05%
ROUND_TRIP = (MAKER_FEE + SLIPPAGE) * 2  # 0.42%

def download(pair: str, interval_min: int, days: int) -> pd.DataFrame:
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - (days * 86400)
    all_candles = []
    since = start_ts
    while since < end_ts:
        try:
            resp = requests.get(f"{KRAKEN_BASE}/public/OHLC",
                params={"pair": pair, "interval": interval_min, "since": since}, timeout=30)
            result = resp.json().get("result", {})
        except Exception as e:
            print(f"  Download error: {e}")
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
    print(f"  Downloaded {pair}: {len(df)} bars ({days}d @ {interval_min}m)")
    return df

# ═══════════════════════════════════════════════════════════════
# STRATEGY 1: Dip Buy (Mean Reversion on Drops)
# ═══════════════════════════════════════════════════════════════
def backtest_dip_buy(df: pd.DataFrame, balance: float = 300,
                     drop_pct: float = 0.8, lookback: int = 6,
                     vol_mult: float = 1.1, sl_atr_mult: float = 1.5,
                     tp_atr_mult: float = 3.0, risk_pct: float = 0.03,
                     cooldown: int = 3) -> Dict:
    """Buy dips with volume confirmation."""
    if len(df) < 60:
        return {"trades": 0, "pnl": 0, "balance": balance}
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Pre-compute ATR
    tr = np.maximum(high[1:] - low[1:], 
         np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(14).mean().values
    vol_avg = pd.Series(volume).rolling(20).mean().values
    
    trades = []
    position = None  # (entry_price, qty, sl, tp, entry_bar)
    bars_since_trade = cooldown + 1
    
    for i in range(max(lookback, 20), len(df)):
        bars_since_trade += 1
        
        # Check exit first
        if position is not None:
            entry_price, qty, sl, tp, entry_bar = position
            # Check SL
            if low[i] <= sl:
                exit_price = sl * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
                balance += entry_price * qty + pnl  # Return capital + pnl
                trades.append({"pnl": pnl, "reason": "SL", "bars": i - entry_bar})
                position = None
                bars_since_trade = 0
                continue
            # Check TP
            if high[i] >= tp:
                exit_price = tp * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "TP", "bars": i - entry_bar})
                position = None
                bars_since_trade = 0
                continue
            # Time-based exit: 48 bars (48h on 1h)
            if i - entry_bar >= 48:
                exit_price = close[i] * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "TIMEOUT", "bars": i - entry_bar})
                position = None
                bars_since_trade = 0
                continue
        
        # Check entry
        if position is None and bars_since_trade >= cooldown:
            if np.isnan(atr[i]) or np.isnan(vol_avg[i]) or atr[i] <= 0 or vol_avg[i] <= 0:
                continue
            
            recent_high = np.max(high[i-lookback:i+1])
            drop = (recent_high - close[i]) / recent_high * 100
            vol_ratio = volume[i] / vol_avg[i]
            
            if drop >= drop_pct and vol_ratio >= vol_mult:
                entry_price = close[i] * (1 + SLIPPAGE)
                sl = entry_price - sl_atr_mult * atr[i]
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
                
                if cost < 5:  # Min trade size
                    continue
                
                balance -= cost  # Lock up capital
                position = (entry_price, qty, sl, tp, i)
    
    # Close any open position at end
    if position is not None:
        entry_price, qty, sl, tp, entry_bar = position
        exit_price = close[-1] * (1 - SLIPPAGE)
        pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
        balance += entry_price * qty + pnl
        trades.append({"pnl": pnl, "reason": "END", "bars": len(df) - entry_bar})
    
    return summarize(trades, 300, balance)

# ═══════════════════════════════════════════════════════════════
# STRATEGY 2: Momentum Breakout
# ═══════════════════════════════════════════════════════════════
def backtest_momentum(df: pd.DataFrame, balance: float = 300,
                      breakout_bars: int = 20, vol_mult: float = 1.3,
                      sma_period: int = 50, sl_atr_mult: float = 1.5,
                      tp_atr_mult: float = 3.0, risk_pct: float = 0.03,
                      cooldown: int = 5) -> Dict:
    """Buy breakouts above N-bar high with volume confirmation."""
    if len(df) < max(breakout_bars, sma_period) + 20:
        return {"trades": 0, "pnl": 0, "balance": balance}
    
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
    position = None
    bars_since_trade = cooldown + 1
    
    start = max(breakout_bars, sma_period, 20)
    for i in range(start, len(df)):
        bars_since_trade += 1
        
        # Check exit
        if position is not None:
            entry_price, qty, sl, tp, entry_bar = position
            if low[i] <= sl:
                exit_price = sl * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "SL", "bars": i - entry_bar})
                position = None
                bars_since_trade = 0
                continue
            if high[i] >= tp:
                exit_price = tp * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "TP", "bars": i - entry_bar})
                position = None
                bars_since_trade = 0
                continue
            if i - entry_bar >= 48:
                exit_price = close[i] * (1 - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
                balance += entry_price * qty + pnl
                trades.append({"pnl": pnl, "reason": "TIMEOUT", "bars": i - entry_bar})
                position = None
                bars_since_trade = 0
                continue
        
        # Check entry
        if position is None and bars_since_trade >= cooldown:
            if np.isnan(atr[i]) or np.isnan(vol_avg[i]) or np.isnan(sma[i]) or atr[i] <= 0:
                continue
            
            # Price above SMA (uptrend filter)
            if close[i] <= sma[i]:
                continue
            
            # Break above N-bar high
            prev_high = np.max(high[i-breakout_bars:i])
            if close[i] <= prev_high:
                continue
            
            # Volume confirmation
            vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
            if vol_ratio < vol_mult:
                continue
            
            entry_price = close[i] * (1 + SLIPPAGE)
            # SL at recent swing low
            recent_low = np.min(low[i-breakout_bars:i])
            sl_from_swing = recent_low * 0.998  # Slightly below swing low
            sl_from_atr = entry_price - sl_atr_mult * atr[i]
            sl = max(sl_from_swing, sl_from_atr)  # Use tighter of two
            
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
            position = (entry_price, qty, sl, tp, i)
    
    if position is not None:
        entry_price, qty, sl, tp, entry_bar = position
        exit_price = close[-1] * (1 - SLIPPAGE)
        pnl = (exit_price - entry_price) * qty - (entry_price * qty * MAKER_FEE) - (exit_price * qty * MAKER_FEE)
        balance += entry_price * qty + pnl
        trades.append({"pnl": pnl, "reason": "END", "bars": len(df) - entry_bar})
    
    return summarize(trades, 300, balance)

# ═══════════════════════════════════════════════════════════════
# STRATEGY 3: Volatility Harvesting (Shannon's Demon)
# ═══════════════════════════════════════════════════════════════
def backtest_vol_harvest(df: pd.DataFrame, balance: float = 300,
                         target_crypto_pct: float = 0.5,
                         rebalance_threshold: float = 0.05) -> Dict:
    """Rebalance 50/50 crypto/USD portfolio when drift exceeds threshold."""
    if len(df) < 10:
        return {"trades": 0, "pnl": 0, "balance": balance}
    
    close = df['close'].values
    
    # Initial allocation
    crypto_usd_value = balance * target_crypto_pct
    usd_balance = balance * (1 - target_crypto_pct)
    crypto_qty = crypto_usd_value / close[0]
    
    trades = []
    rebalance_count = 0
    total_fees = 0
    
    for i in range(1, len(df)):
        crypto_value = crypto_qty * close[i]
        total_value = crypto_value + usd_balance
        crypto_pct = crypto_value / total_value if total_value > 0 else 0
        
        drift = abs(crypto_pct - target_crypto_pct)
        
        if drift >= rebalance_threshold:
            target_crypto_value = total_value * target_crypto_pct
            delta_usd = crypto_value - target_crypto_value
            
            # delta_usd > 0 means sell crypto, < 0 means buy crypto
            trade_value = abs(delta_usd)
            fee = trade_value * (MAKER_FEE + SLIPPAGE)
            total_fees += fee
            
            if delta_usd > 0:
                # Sell crypto
                sell_qty = delta_usd / close[i]
                crypto_qty -= sell_qty
                usd_balance += delta_usd - fee
                trades.append({"action": "SELL", "value": delta_usd, "fee": fee, "bar": i})
            else:
                # Buy crypto
                buy_usd = abs(delta_usd)
                buy_qty = buy_usd / close[i]
                crypto_qty += buy_qty
                usd_balance -= buy_usd + fee
                trades.append({"action": "BUY", "value": buy_usd, "fee": fee, "bar": i})
            
            rebalance_count += 1
    
    final_value = crypto_qty * close[-1] + usd_balance
    
    # Compare to buy-and-hold
    bh_value = balance  # Just holding USD
    bh_crypto = (balance * target_crypto_pct / close[0]) * close[-1] + balance * (1 - target_crypto_pct)
    
    return {
        "trades": rebalance_count,
        "initial": balance,
        "final": round(final_value, 2),
        "return_pct": round((final_value - balance) / balance * 100, 2),
        "total_fees": round(total_fees, 2),
        "buy_hold_50_50": round(bh_crypto, 2),
        "vs_buy_hold": round((final_value - bh_crypto) / bh_crypto * 100, 2),
        "vs_usd": round((final_value - balance) / balance * 100, 2),
    }

# ═══════════════════════════════════════════════════════════════
# STRATEGY 4: ETH/BTC Ratio Mean Reversion
# ═══════════════════════════════════════════════════════════════
def backtest_cross_pair(eth_df: pd.DataFrame, btc_df: pd.DataFrame,
                        balance: float = 300, lookback: int = 720,
                        entry_sigma: float = 2.0, exit_sigma: float = 0.5,
                        risk_pct: float = 0.03) -> Dict:
    """Trade ETH/BTC ratio mean reversion."""
    min_len = min(len(eth_df), len(btc_df))
    if min_len < lookback + 10:
        return {"trades": 0, "pnl": 0, "balance": balance}
    
    eth_close = eth_df['close'].values[:min_len]
    btc_close = btc_df['close'].values[:min_len]
    ratio = eth_close / btc_close
    
    trades = []
    position = None  # (direction, eth_qty, entry_ratio, entry_bar)
    
    for i in range(lookback, min_len):
        window = ratio[i-lookback:i]
        mean_r = np.mean(window)
        std_r = np.std(window)
        if std_r == 0:
            continue
        z = (ratio[i] - mean_r) / std_r
        
        # Check exit
        if position is not None:
            direction, eth_qty, entry_ratio, entry_bar, entry_eth_price = position
            
            # Exit when z reverts toward mean
            should_exit = False
            if direction == "long_eth" and z >= -exit_sigma:
                should_exit = True
            elif direction == "short_eth" and z <= exit_sigma:
                should_exit = True
            # Timeout: 168 bars (1 week on 1h)
            if i - entry_bar >= 168:
                should_exit = True
            
            if should_exit:
                exit_eth_price = eth_close[i]
                if direction == "long_eth":
                    pnl = (exit_eth_price - entry_eth_price) * eth_qty
                else:
                    pnl = (entry_eth_price - exit_eth_price) * eth_qty
                
                fee = eth_qty * entry_eth_price * MAKER_FEE + eth_qty * exit_eth_price * MAKER_FEE
                pnl -= fee
                balance += pnl
                trades.append({"pnl": pnl, "direction": direction, "z_entry": entry_ratio,
                              "z_exit": z, "bars": i - entry_bar,
                              "reason": "REVERT" if i - entry_bar < 168 else "TIMEOUT"})
                position = None
        
        # Check entry
        if position is None:
            if z < -entry_sigma:
                # ETH cheap vs BTC — buy ETH
                risk_dollars = balance * risk_pct
                eth_qty = risk_dollars / eth_close[i]
                position = ("long_eth", eth_qty, z, i, eth_close[i])
            elif z > entry_sigma:
                # ETH expensive vs BTC — sell ETH
                risk_dollars = balance * risk_pct
                eth_qty = risk_dollars / eth_close[i]
                position = ("short_eth", eth_qty, z, i, eth_close[i])
    
    # Close any open position
    if position is not None:
        direction, eth_qty, entry_z, entry_bar, entry_eth_price = position
        exit_eth_price = eth_close[-1]
        if direction == "long_eth":
            pnl = (exit_eth_price - entry_eth_price) * eth_qty
        else:
            pnl = (entry_eth_price - exit_eth_price) * eth_qty
        fee = eth_qty * entry_eth_price * MAKER_FEE + eth_qty * exit_eth_price * MAKER_FEE
        pnl -= fee
        balance += pnl
        trades.append({"pnl": pnl, "direction": direction, "reason": "END"})
    
    return summarize_generic(trades, 300, balance)


def summarize(trades: List[Dict], initial: float, final: float) -> Dict:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
    
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins)/len(trades)*100, 1) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((final - initial) / initial * 100, 2),
        "final_balance": round(final, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0,
        "by_reason": {r: len([t for t in trades if t.get("reason") == r]) for r in set(t.get("reason", "?") for t in trades)},
    }

def summarize_generic(trades: List[Dict], initial: float, final: float) -> Dict:
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins)/len(trades)*100, 1) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((final - initial) / initial * 100, 2),
        "final_balance": round(final, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
    }


def print_result(name: str, result: Dict):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")


def main():
    print("Downloading 90 days of 1h Kraken data...")
    btc = download("XBTUSD", 60, 90)
    eth = download("ETHUSD", 60, 90)
    sol = download("SOLUSD", 60, 90)
    
    print(f"\nData: BTC={len(btc)} bars, ETH={len(eth)} bars, SOL={len(sol)} bars")
    if len(btc) > 0:
        print(f"BTC range: ${btc['close'].min():.0f} - ${btc['close'].max():.0f}")
    if len(eth) > 0:
        print(f"ETH range: ${eth['close'].min():.0f} - ${eth['close'].max():.0f}")
    if len(sol) > 0:
        print(f"SOL range: ${sol['close'].min():.0f} - ${sol['close'].max():.0f}")
    
    # Test multiple parameter combos for dip buy
    print("\n" + "="*60)
    print("  DIP BUY STRATEGY — Parameter Sweep")
    print("="*60)
    
    for pair_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
        if len(df) < 60:
            print(f"  {pair_name}: insufficient data")
            continue
        for drop in [0.5, 0.8, 1.0, 1.5, 2.0]:
            for vol_m in [1.0, 1.1, 1.3]:
                r = backtest_dip_buy(df, drop_pct=drop, vol_mult=vol_m)
                if r["trades"] > 0:
                    print(f"  {pair_name} drop={drop}% vol={vol_m}x: "
                          f"{r['trades']}T {r['win_rate']}%WR PF={r['profit_factor']} "
                          f"PnL=${r['total_pnl']} ret={r['return_pct']}% "
                          f"exits={r['by_reason']}")
    
    # Momentum breakout sweep
    print("\n" + "="*60)
    print("  MOMENTUM BREAKOUT — Parameter Sweep")
    print("="*60)
    
    for pair_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
        if len(df) < 70:
            continue
        for bb in [10, 15, 20, 30]:
            for vm in [1.0, 1.2, 1.5]:
                r = backtest_momentum(df, breakout_bars=bb, vol_mult=vm)
                if r["trades"] > 0:
                    print(f"  {pair_name} bars={bb} vol={vm}x: "
                          f"{r['trades']}T {r['win_rate']}%WR PF={r['profit_factor']} "
                          f"PnL=${r['total_pnl']} ret={r['return_pct']}% "
                          f"exits={r['by_reason']}")
    
    # Volatility harvesting
    print("\n" + "="*60)
    print("  VOLATILITY HARVESTING (Shannon's Demon)")
    print("="*60)
    
    for pair_name, df in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
        if len(df) < 10:
            continue
        for thresh in [0.03, 0.05, 0.08, 0.10]:
            for target in [0.3, 0.5, 0.7]:
                r = backtest_vol_harvest(df, target_crypto_pct=target, rebalance_threshold=thresh)
                if r["trades"] > 0:
                    print(f"  {pair_name} target={int(target*100)}% thresh={int(thresh*100)}%: "
                          f"{r['trades']}T final=${r['final']} ret={r['return_pct']}% "
                          f"fees=${r['total_fees']} vs_hold={r['vs_buy_hold']}%")
    
    # Cross-pair
    print("\n" + "="*60)
    print("  ETH/BTC CROSS-PAIR MEAN REVERSION")
    print("="*60)
    
    if len(eth) > 100 and len(btc) > 100:
        for lb in [168, 336, 720]:
            for es in [1.5, 2.0, 2.5]:
                r = backtest_cross_pair(eth, btc, lookback=lb, entry_sigma=es)
                if r["trades"] > 0:
                    print(f"  lookback={lb}h sigma={es}: "
                          f"{r['trades']}T {r['win_rate']}%WR PF={r['profit_factor']} "
                          f"PnL=${r['total_pnl']} ret={r['return_pct']}%")
    
    print("\n" + "="*60)
    print("  DONE — Check results above for any strategy with positive PnL")
    print("="*60)

if __name__ == "__main__":
    main()
