#!/usr/bin/env python3
"""Single tool test for Round 3"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h_extended"
FEES = 0.0065

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT"]  # Just 4 pairs for speed

_data_cache = {}

def load_data(pair):
    if pair not in _data_cache:
        df = pd.read_csv(DATA_DIR / f"{pair}_1h.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        _data_cache[pair] = df
    return _data_cache[pair]

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_sma(close, period):
    if len(close) < period: return close[-1]
    return np.mean(close[-period:])

def calc_bb_position(close, period=20):
    if len(close) < period:
        return 0
    sma = calc_sma(close, period)
    std = np.std(close[-period:])
    if std == 0:
        return 0
    return (close[-1] - sma) / std

def calc_volume_ratio(volume, short=10, long=50):
    if len(volume) < long:
        return 1.0
    recent = np.mean(volume[-short:])
    longer = np.mean(volume[-long:])
    return recent / longer if longer > 0 else 1.0

def calc_atr_pct(high, low, close, period=14):
    if len(close) < period + 1:
        return 0
    tr_list = []
    for i in range(1, min(len(close), period + 1)):
        tr = max(
            high[-i] - low[-i],
            abs(high[-i] - close[-i-1]),
            abs(low[-i] - close[-i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list)
    return (atr / close[-1]) * 100

def is_chop_regime(btc_close, bar_idx):
    """Choppy/sideways: -8% < 30d return < 8%"""
    if bar_idx < 720:
        return False
    ret_30d = (btc_close[bar_idx] - btc_close[bar_idx-720]) / btc_close[bar_idx-720]
    return -0.08 <= ret_30d <= 0.08

def trailing_exit(close_arr, entry_bar, max_hold, direction, trail_pct=0.08, hard_stop_pct=0.12):
    entry_price = close_arr[entry_bar]
    max_bar = min(entry_bar + max_hold, len(close_arr) - 1)
    
    if direction == 'long':
        hard_stop = entry_price * (1 - hard_stop_pct)
        best_price = entry_price
        
        for bar in range(entry_bar + 1, max_bar + 1):
            current = close_arr[bar]
            
            if current <= hard_stop:
                return current, bar - entry_bar, "hard_stop"
            
            if current > best_price:
                best_price = current
            
            trail_stop = best_price * (1 - trail_pct)
            if current <= trail_stop and best_price > entry_price * 1.02:
                return current, bar - entry_bar, "trail_stop"
        
        return close_arr[max_bar], max_bar - entry_bar, "time_exit"
    
    else:  # short
        hard_stop = entry_price * (1 + hard_stop_pct)
        best_price = entry_price
        
        for bar in range(entry_bar + 1, max_bar + 1):
            current = close_arr[bar]
            
            if current >= hard_stop:
                return current, bar - entry_bar, "hard_stop"
            
            if current < best_price:
                best_price = current
            
            trail_stop = best_price * (1 + trail_pct)
            if current >= trail_stop and best_price < entry_price * 0.98:
                return current, bar - entry_bar, "trail_stop"
        
        return close_arr[max_bar], max_bar - entry_bar, "time_exit"

def bb_mean_reversion_chop(close, high, low, volume, rsi):
    """BB mean reversion during choppy markets"""
    if len(close) < 100:
        return None
    
    # Must be in choppy environment (low volatility)
    atr_pct = calc_atr_pct(high, low, close, 20)
    if atr_pct > 3.5:
        return None
    
    # Bollinger band metrics
    bb_pos = calc_bb_position(close, 20)
    
    # Range-bound check: 30d high/low range < 25%
    if len(close) >= 720:
        range_high = np.max(high[-720:])
        range_low = np.min(low[-720:])
        range_pct = (range_high - range_low) / range_low * 100
        if range_pct > 25:
            return None
    
    # Long setup: extreme oversold in tight range
    if bb_pos < -1.8 and rsi < 30:
        vol_ratio = calc_volume_ratio(volume)
        if vol_ratio > 1.2:
            return {
                'tool': 'bb_mean_reversion_chop',
                'direction': 'long',
                'score': abs(bb_pos) * 10 + (30 - rsi) + vol_ratio * 5
            }
    
    # Short setup: extreme overbought in tight range  
    elif bb_pos > 1.8 and rsi > 70:
        vol_ratio = calc_volume_ratio(volume)
        if vol_ratio > 1.2:
            return {
                'tool': 'bb_mean_reversion_chop',
                'direction': 'short',
                'score': bb_pos * 10 + (rsi - 70) + vol_ratio * 5
            }
    
    return None

def validate_single_tool():
    print("Testing bb_mean_reversion_chop...")
    
    all_trades = []
    btc_df = load_data("BTCUSDT")
    btc_close = btc_df['close'].values
    
    for pair in PAIRS:
        print(f"  Processing {pair}...")
        df = load_data(pair)
        close = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        vol = df['volume'].values
        
        # OOS: second half of data
        oos_start = len(close) // 2
        i = oos_start
        last_signal = -336
        
        signals_found = 0
        
        while i < len(close) - 336 and signals_found < 20:  # Limit signals for testing
            # Non-overlapping trades
            if i - last_signal < 48:
                i += 8
                continue
            
            # Regime filtering (chop)
            if not is_chop_regime(btc_close, i):
                i += 8
                continue
            
            # Get signal
            rsi = calc_rsi(close[:i+1])
            sig = bb_mean_reversion_chop(close[:i+1], high_arr[:i+1], low_arr[:i+1], vol[:i+1], rsi)
            
            if sig is not None:
                signals_found += 1
                print(f"    Signal #{signals_found}: {pair} {sig['direction']} at bar {i}, price {close[i]:.2f}")
                
                # Execute trade
                exit_price, hold_bars, exit_reason = trailing_exit(close, i, 336, sig['direction'])
                entry_price = close[i]
                
                # Calculate returns
                if sig['direction'] == 'long':
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - FEES
                
                all_trades.append({
                    'pair': pair,
                    'direction': sig['direction'],
                    'entry': float(entry_price),
                    'exit': float(exit_price),
                    'net': float(net_return),
                    'win': net_return > 0,
                    'hold': int(hold_bars),
                    'exit_reason': exit_reason
                })
                last_signal = i
            
            i += 8
    
    return all_trades

def report_results(trades):
    if not trades:
        print("NO SIGNALS ❌")
        return
    
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    avg_ret = np.mean([t['net'] for t in trades]) * 100
    
    w_trades = [t['net'] for t in trades if t['win']]
    l_trades = [t['net'] for t in trades if not t['win']]
    avg_win = np.mean(w_trades) * 100 if w_trades else 0
    avg_loss = np.mean(l_trades) * 100 if l_trades else 0
    
    gross_profit = sum(w_trades) if w_trades else 0
    gross_loss = abs(sum(l_trades)) if l_trades else 0.001
    pf = gross_profit / gross_loss
    
    avg_hold = np.mean([t['hold'] for t in trades])
    
    print(f"\nRESULTS:")
    print(f"  Signals: {n}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Avg Return: {avg_ret:.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg Win: {avg_win:.2f}%")
    print(f"  Avg Loss: {avg_loss:.2f}%")
    print(f"  Avg Hold: {avg_hold:.0f}h ({avg_hold/24:.1f}d)")
    
    passed = n >= 30 and (wr >= 55 or pf >= 1.5)
    status = "✅ PASSED" if passed else "❌ KILLED"
    print(f"  Status: {status}")

if __name__ == "__main__":
    print("=== ROUND 3 SINGLE TOOL TEST ===")
    trades = validate_single_tool()
    report_results(trades)
    print("\nTest complete.")