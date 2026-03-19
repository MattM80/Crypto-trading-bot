#!/usr/bin/env python3
"""
Backtest: Take-Profit & Trailing Stop Impact Analysis

Compares the bot's signal performance WITH vs WITHOUT the new TP/trailing logic.
Downloads real Kraken OHLCV data and simulates bar-by-bar.
"""
import requests
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple

KRAKEN_BASE = "https://api.kraken.com/0"
MAKER_FEE = 0.0016
SLIPPAGE = 0.0005
ROUND_TRIP_COST = (MAKER_FEE + SLIPPAGE) * 2  # ~0.42%

# ─── Data Download ───────────────────────────────────────────

def download(pair: str, interval_min: int = 60, days: int = 90) -> pd.DataFrame:
    """Download OHLCV from Kraken public API."""
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


# ─── Indicators ──────────────────────────────────────────────

def calc_rsi(prices, period=7):
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(len(delta))
    avg_loss = np.zeros(len(delta))
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    for i in range(period, len(delta)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50.0], rsi])

def calc_atr(high, low, close, period=14):
    if len(high) < 2:
        return np.full(len(high), 0.0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    if len(tr) < period:
        return tr
    atr = np.full(len(tr), np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calc_sma(prices, period):
    sma = np.full(len(prices), np.nan)
    for i in range(period-1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    return sma


# ─── Signal Generation (mirrors run_master_bot.py) ──────────

def generate_signals(df: pd.DataFrame) -> List[dict]:
    """Generate signals on a single pair's OHLCV data. Returns list of signal dicts with bar index."""
    if len(df) < 50:
        return []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    opn = df['open'].values
    
    rsi7 = calc_rsi(close, 7)
    sma50 = calc_sma(close, 50)
    atr14 = calc_atr(high, low, close, 14)
    
    signals = []
    
    for i in range(50, len(df)):
        price = close[i]
        cur_rsi = rsi7[i]
        cur_atr = atr14[i] if not np.isnan(atr14[i]) else price * 0.03
        cur_atr_pct = cur_atr / price * 100 if price > 0 else 0
        cur_vs_sma50 = (price - sma50[i]) / sma50[i] * 100 if not np.isnan(sma50[i]) and sma50[i] > 0 else 0
        
        ret_4h = (price - close[i-5]) / close[i-5] * 100 if i >= 5 else 0
        ret_8h = (price - close[i-9]) / close[i-9] * 100 if i >= 9 else 0
        ret_12h = (price - close[i-13]) / close[i-13] * 100 if i >= 13 else 0
        ret_24h = (price - close[i-25]) / close[i-25] * 100 if i >= 25 else 0
        
        # Crash Buy
        if ret_24h < -10 and cur_rsi < 20:
            signals.append({'bar': i, 'tool': 'crash_buy', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.05, 'score': (20 - cur_rsi) * 2})
        
        # Volatile Oversold
        if cur_atr_pct > 3 and cur_rsi < 25:
            signals.append({'bar': i, 'tool': 'volatile_oversold', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.08, 'score': cur_atr_pct * (25 - cur_rsi)})
        
        # Relief Rally
        if cur_rsi > 75 and cur_vs_sma50 < 0:
            signals.append({'bar': i, 'tool': 'relief_rally', 'direction': 'long',
                           'hold': 12, 'sl_pct': 0.03, 'score': (cur_rsi - 75) * 1.5})
        
        # Dip Buy
        if ret_4h < -3:
            signals.append({'bar': i, 'tool': 'dip_buy', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.99, 'score': abs(ret_4h) * 2})
        
        # Mega Crash
        if ret_24h < -15:
            signals.append({'bar': i, 'tool': 'mega_crash', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.08, 'score': abs(ret_24h) * 3})
        
        # Flash Crash
        if ret_12h < -10:
            signals.append({'bar': i, 'tool': 'flash_crash', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.07, 'score': abs(ret_12h) * 2.5})
        
        # Quick Crash
        if ret_8h < -10:
            signals.append({'bar': i, 'tool': 'quick_crash', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.07, 'score': abs(ret_8h) * 2})
        
        # Deep Dip (8-10% drops)
        for tf_label, ret_val in [("8h", ret_8h), ("12h", ret_12h), ("24h", ret_24h)]:
            if ret_val < -8 and ret_val >= -10:
                signals.append({'bar': i, 'tool': f'deep_dip_{tf_label}', 'direction': 'long',
                               'hold': 24, 'sl_pct': 0.05, 'score': abs(ret_val) * 1.5})
        
        # Quick Dip
        if ret_4h < -5:
            signals.append({'bar': i, 'tool': 'quick_dip', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.99, 'score': abs(ret_4h) * 2})
        
        # Panic Close
        bar_range = (high[i] - low[i]) / price * 100 if price > 0 else 0
        bar_close_pos = (price - low[i]) / (high[i] - low[i]) if high[i] > low[i] else 0.5
        if bar_range > 3 and bar_close_pos < 0.25:
            signals.append({'bar': i, 'tool': 'panic_close', 'direction': 'long',
                           'hold': 24, 'sl_pct': 0.05, 'score': bar_range * 5})
        
        # Fat Tail Reversion
        if i >= 50:
            returns_50 = np.diff(close[i-50:i+1]) / close[i-50:i]
            kurt = float(pd.Series(returns_50).kurtosis()) if len(returns_50) > 10 else 0
            if not np.isnan(kurt) and kurt > 5 and ret_4h < -3:
                signals.append({'bar': i, 'tool': 'fat_tail_revert', 'direction': 'long',
                               'hold': 24, 'sl_pct': 0.06, 'score': kurt * abs(ret_4h) * 2})
        
        # Distribution Exhaustion
        if i >= 50:
            returns_50 = np.diff(close[i-50:i+1]) / close[i-50:i]
            skew = float(pd.Series(returns_50).skew()) if len(returns_50) > 10 else 0
            if not np.isnan(skew) and skew < -1 and ret_4h < -3:
                signals.append({'bar': i, 'tool': 'dist_exhaustion', 'direction': 'long',
                               'hold': 24, 'sl_pct': 0.05, 'score': abs(skew) * abs(ret_4h) * 3})
        
        # Deceleration Buy
        if i >= 7:
            vel_recent = (close[i] - close[i-3]) / close[i-3]
            vel_prior = (close[i-3] - close[i-6]) / close[i-6]
            acceleration = vel_recent - vel_prior
            if acceleration > 0.01 and ret_4h < -2:
                signals.append({'bar': i, 'tool': 'deceleration_buy', 'direction': 'long',
                               'hold': 24, 'sl_pct': 0.05, 'score': acceleration * abs(ret_4h) * 50})
        
        # Volume Climax
        if i >= 21:
            vol_first = np.mean(volume[i-20:i-10])
            vol_second = np.mean(volume[i-10:i+1])
            vol_trend = vol_second / vol_first if vol_first > 0 else 1
            if vol_trend > 1.5 and ret_4h < -2:
                signals.append({'bar': i, 'tool': 'volume_climax', 'direction': 'long',
                               'hold': 24, 'sl_pct': 0.99, 'score': vol_trend * abs(ret_4h) * 2})
        
        # Whale Buy (5x volume on green candle)
        if i >= 21:
            avg_vol = np.mean(volume[i-20:i])
            if avg_vol > 0:
                vol_ratio = volume[i] / avg_vol
                if vol_ratio >= 5 and close[i] > opn[i]:
                    signals.append({'bar': i, 'tool': 'whale_buy', 'direction': 'long',
                                   'hold': 8, 'sl_pct': 0.03, 'score': vol_ratio * 2})
        
        # Efficiency Capitulation
        if i >= 25:
            net_move = abs(close[i] - close[i-10])
            total_path = sum(abs(close[i-j] - close[i-j-1]) for j in range(10))
            efficiency = net_move / total_path if total_path > 0 else 0
            recent_high_val = np.max(high[i-24:i+1])
            recent_low_val = np.min(low[i-24:i+1])
            range_pos = (close[i] - recent_low_val) / (recent_high_val - recent_low_val) if recent_high_val > recent_low_val else 0.5
            vol_first = np.mean(volume[i-20:i-10]) if i >= 20 else 1
            vol_second = np.mean(volume[i-10:i+1]) if i >= 10 else 1
            vol_trend = vol_second / vol_first if vol_first > 0 else 1
            if efficiency > 0.4 and range_pos < 0.10 and vol_trend > 1.5 and ret_4h < -3:
                signals.append({'bar': i, 'tool': 'efficiency_capitulation', 'direction': 'long',
                               'hold': 24, 'sl_pct': 0.06, 'score': efficiency * abs(ret_4h) * vol_trend * 5})
        
        # Crash + Negative AC
        if i >= 100:
            returns = np.diff(close[i-100:i+1]) / close[i-100:i]
            ac1 = float(pd.Series(returns).autocorr(lag=1))
            if not np.isnan(ac1) and ret_24h < -10 and ac1 < -0.05:
                signals.append({'bar': i, 'tool': 'crash_neg_ac', 'direction': 'long',
                               'hold': 24, 'sl_pct': 0.08, 'score': abs(ret_24h) * (abs(ac1) + 0.1) * 10})
        
        # Green Exhaustion (7 green then red)
        if i >= 8:
            all_green = all(close[i-j-1] > opn[i-j-1] for j in range(1, 8))
            cur_red = close[i] < opn[i]
            if all_green and cur_red:
                signals.append({'bar': i, 'tool': 'green_exhaustion', 'direction': 'short',
                               'hold': 8, 'sl_pct': 0.03, 'score': 15})
        
        # Mega Pump Sell
        if cur_rsi > 80 and ret_8h > 8:
            signals.append({'bar': i, 'tool': 'mega_pump_sell', 'direction': 'short',
                           'hold': 24, 'sl_pct': 0.99, 'score': ret_8h * 3 + (cur_rsi - 80)})
        
        # Z-score extreme
        if i >= 49:
            window = close[i-48:i+1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 0:
                z = (close[i] - mu) / sigma
                if z < -3:
                    signals.append({'bar': i, 'tool': 'zscore_extreme', 'direction': 'long',
                                   'hold': 24, 'sl_pct': 0.05, 'score': abs(z) * 5})
    
    return signals


# ─── Exit Parameter Logic (mirrors _get_exit_params) ────────

MEAN_REVERSION = {
    'volatile_oversold', 'fat_tail_revert', 'rsi_divergence',
    'dist_exhaustion', 'math_capitulation', 'zscore_extreme',
    'btc_alt_spread', 'alt_btc_revert', 'entropy_dip',
    'vpin_toxic', 'vpin_dip', 'panic_close',
}

CRASH_BUY = {
    'crash_buy', 'crash_neg_ac', 'mega_crash', 'flash_crash',
    'quick_crash', 'relief_rally', 'blood_in_streets',
    'efficiency_capitulation', 'mega_align', 'crash_mean_revert',
    'market_panic_90', 'market_panic_80', 'market_panic_70',
    'capitulation',
}

MOMENTUM = {
    'hurst_trend', 'fomo_ride', 'deceleration_buy',
    'whale_buy', 'volume_climax',
}

DIP_BUY = {'dip_buy', 'deep_dip_24h', 'quick_dip'}

BIG_CRASH = {'mega_crash', 'crash_neg_ac', 'mega_align', 'market_panic_90'}

def get_exit_params(tool, price, atr_val):
    """Return (exit_mode, tp_pct, trail_pct, trail_activate_pct)
    Backtested optimal TP levels (90-day sweep across AAVE/NEAR/AVAX/XRP).
    """
    atr_pct = atr_val / price if price > 0 and not np.isnan(atr_val) else 0.03
    
    if tool in MOMENTUM:
        # NO TP — backtested: any TP hurts momentum runners
        return ('default', None, None, None)
    if tool in MEAN_REVERSION:
        # TP at 8-10%: signals regularly move 5-10%, 3% was too tight
        tp = max(min(atr_pct * 4.0, 0.10), 0.08)
        return ('fixed_tp', tp, None, None)
    if tool in CRASH_BUY:
        tp = 0.12 if tool in BIG_CRASH else 0.10
        return ('fixed_tp', tp, None, None)
    if tool in DIP_BUY:
        # TP at 6%: backtested sweet spot
        return ('fixed_tp', 0.06, None, None)
    # Default: 6-8%
    tp = max(min(atr_pct * 3.0, 0.08), 0.06)
    return ('fixed_tp', tp, None, None)


# ─── Simulation Engine ──────────────────────────────────────

def simulate_trades(df: pd.DataFrame, signals: List[dict], use_tp: bool = False,
                    balance: float = 1000, risk_pct: float = 0.05) -> Dict:
    """Simulate trades bar-by-bar. If use_tp=True, apply TP/trailing logic."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr14 = calc_atr(high, low, close, 14)
    
    trades = []
    position = None  # Only one position at a time per pair
    equity_curve = [balance]
    peak_equity = balance
    max_dd = 0
    
    # Sort signals by bar
    signals = sorted(signals, key=lambda s: s['bar'])
    signal_idx = 0
    
    for i in range(len(df)):
        # Check exit for existing position
        if position is not None:
            price = close[i]
            bar_high = high[i]
            bar_low = low[i]
            bars_held = i - position['entry_bar']
            
            should_close = False
            close_reason = ""
            close_price = price
            
            # Update tracking
            if position['direction'] == 'long':
                position['highest'] = max(position['highest'], bar_high)
            else:
                position['lowest'] = min(position['lowest'], bar_low)
            
            # 1. Stop loss (check intra-bar)
            if position['direction'] == 'long':
                sl_price = position['entry'] * (1 - position['sl_pct'])
                if bar_low <= sl_price:
                    should_close = True
                    close_reason = "STOP LOSS"
                    close_price = sl_price
            else:
                sl_price = position['entry'] * (1 + position['sl_pct'])
                if bar_high >= sl_price:
                    should_close = True
                    close_reason = "STOP LOSS"
                    close_price = sl_price
            
            # 2. Hybrid TP→trailing with floor (only if use_tp=True)
            if not should_close and use_tp:
                tp_pct = position.get('tp_pct')
                
                if tp_pct is not None:
                    if position['direction'] == 'long':
                        tp_price = position['entry'] * (1 + tp_pct)
                        # Activate trailing when TP hit
                        if not position.get('trailing_active', False) and bar_high >= tp_price:
                            position['trailing_active'] = True
                            position['trail_pct'] = max(tp_pct * 0.5, 0.02)
                        # Trailing with TP floor
                        if position.get('trailing_active', False):
                            trail_price = max(position['highest'] * (1 - position['trail_pct']), tp_price)
                            if bar_low <= trail_price:
                                should_close = True
                                close_reason = "TRAILING STOP"
                                close_price = trail_price
                    else:
                        tp_price = position['entry'] * (1 - tp_pct)
                        if not position.get('trailing_active', False) and bar_low <= tp_price:
                            position['trailing_active'] = True
                            position['trail_pct'] = max(tp_pct * 0.5, 0.02)
                        if position.get('trailing_active', False):
                            trail_price = min(position['lowest'] * (1 + position['trail_pct']), tp_price)
                            if bar_high >= trail_price:
                                should_close = True
                                close_reason = "TRAILING STOP"
                                close_price = trail_price
                # No tp_pct (momentum) → no TP, just SL + hold timeout
            
            # 3. Hold timeout
            if not should_close and bars_held >= position['hold']:
                should_close = True
                close_reason = "HOLD TIMEOUT"
                close_price = price
            
            if should_close:
                if position['direction'] == 'long':
                    pnl_pct = (close_price - position['entry']) / position['entry']
                else:
                    pnl_pct = (position['entry'] - close_price) / position['entry']
                pnl_pct -= ROUND_TRIP_COST
                
                pnl_dollar = balance * risk_pct * pnl_pct / position['sl_pct']
                balance += pnl_dollar
                
                trades.append({
                    'tool': position['tool'],
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'exit': close_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_dollar': pnl_dollar,
                    'bars_held': bars_held,
                    'reason': close_reason,
                })
                position = None
        
        # Try to enter new position
        if position is None:
            while signal_idx < len(signals) and signals[signal_idx]['bar'] <= i:
                if signals[signal_idx]['bar'] == i:
                    sig = signals[signal_idx]
                    entry_price = close[i]
                    atr_val = atr14[i] if i < len(atr14) and not np.isnan(atr14[i]) else entry_price * 0.03
                    
                    pos = {
                        'tool': sig['tool'],
                        'direction': sig['direction'],
                        'entry': entry_price,
                        'entry_bar': i,
                        'sl_pct': sig['sl_pct'],
                        'hold': sig['hold'],
                        'highest': entry_price,
                        'lowest': entry_price,
                    }
                    
                    if use_tp:
                        mode, tp, trail, activate = get_exit_params(sig['tool'], entry_price, atr_val)
                        pos['exit_mode'] = mode
                        pos['tp_pct'] = tp
                        pos['trail_pct'] = trail
                        pos['activate_pct'] = activate
                    
                    position = pos
                    break
                signal_idx += 1
            else:
                if signal_idx < len(signals) and signals[signal_idx]['bar'] < i:
                    signal_idx += 1
        
        equity_curve.append(balance)
        if balance > peak_equity:
            peak_equity = balance
        dd = (peak_equity - balance) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    # Force close if still in position
    if position is not None:
        price = close[-1]
        if position['direction'] == 'long':
            pnl_pct = (price - position['entry']) / position['entry']
        else:
            pnl_pct = (position['entry'] - price) / position['entry']
        pnl_pct -= ROUND_TRIP_COST
        pnl_dollar = balance * risk_pct * pnl_pct / position['sl_pct']
        balance += pnl_dollar
        trades.append({
            'tool': position['tool'], 'direction': position['direction'],
            'entry': position['entry'], 'exit': price,
            'pnl_pct': pnl_pct * 100, 'pnl_dollar': pnl_dollar,
            'bars_held': len(df) - 1 - position['entry_bar'],
            'reason': 'FORCE CLOSE',
        })
    
    # Stats
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    
    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'total_pnl': balance - 1000,
        'total_return_pct': (balance - 1000) / 1000 * 100,
        'avg_win_pct': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
        'avg_loss_pct': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
        'max_drawdown_pct': max_dd,
        'avg_bars_held': np.mean([t['bars_held'] for t in trades]) if trades else 0,
        'trade_details': trades,
        'exit_reasons': {},
    }


def count_exit_reasons(trades):
    reasons = {}
    for t in trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1
    return reasons


# ─── Main ────────────────────────────────────────────────────

def main():
    pairs = ["AAVEUSD", "NEARUSD", "AVAXUSD", "XRPUSD"]
    days = 90
    
    print("=" * 80)
    print("  TAKE-PROFIT & TRAILING STOP BACKTEST")
    print(f"  Pairs: {', '.join(pairs)} | Period: {days} days | Interval: 1h")
    print("=" * 80)
    
    all_results = {}
    
    for pair in pairs:
        print(f"\n{'─' * 60}")
        print(f"  Processing {pair}...")
        df = download(pair, 60, days)
        if len(df) < 100:
            print(f"  Skipping {pair}: insufficient data ({len(df)} bars)")
            continue
        
        # Generate signals
        signals = generate_signals(df)
        print(f"  Generated {len(signals)} signals")
        
        if not signals:
            print(f"  No signals for {pair}, skipping")
            continue
        
        # Tool breakdown
        tool_counts = {}
        for s in signals:
            tool_counts[s['tool']] = tool_counts.get(s['tool'], 0) + 1
        print(f"  Signal breakdown: {dict(sorted(tool_counts.items(), key=lambda x: -x[1])[:8])}")
        
        # Run WITHOUT TP
        result_no_tp = simulate_trades(df, signals, use_tp=False)
        result_no_tp['exit_reasons'] = count_exit_reasons(result_no_tp['trade_details'])
        
        # Run WITH TP
        result_with_tp = simulate_trades(df, signals, use_tp=True)
        result_with_tp['exit_reasons'] = count_exit_reasons(result_with_tp['trade_details'])
        
        all_results[pair] = {'no_tp': result_no_tp, 'with_tp': result_with_tp}
        
        # Per-pair comparison
        print(f"\n  {pair} Results:")
        print(f"  {'Metric':<25} {'WITHOUT TP':>15} {'WITH TP':>15} {'Delta':>12}")
        print(f"  {'─' * 67}")
        
        metrics = [
            ('Total Trades', 'trades', 'd'),
            ('Win Rate %', 'win_rate', '.1f'),
            ('Total PnL $', 'total_pnl', '.2f'),
            ('Total Return %', 'total_return_pct', '.2f'),
            ('Avg Win %', 'avg_win_pct', '.2f'),
            ('Avg Loss %', 'avg_loss_pct', '.2f'),
            ('Max Drawdown %', 'max_drawdown_pct', '.2f'),
            ('Avg Bars Held', 'avg_bars_held', '.1f'),
        ]
        
        for label, key, fmt in metrics:
            v1 = result_no_tp[key]
            v2 = result_with_tp[key]
            delta = v2 - v1
            sign = '+' if delta > 0 else ''
            print(f"  {label:<25} {v1:>15{fmt}} {v2:>15{fmt}} {sign}{delta:>11{fmt}}")
        
        # Exit reasons
        print(f"\n  Exit reasons (WITH TP):")
        for reason, count in sorted(result_with_tp['exit_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    # ─── Aggregate Summary ─────────────────────────────────
    if all_results:
        print(f"\n{'=' * 80}")
        print("  AGGREGATE COMPARISON (all pairs combined)")
        print(f"{'=' * 80}")
        
        agg = {'no_tp': {}, 'with_tp': {}}
        for mode in ['no_tp', 'with_tp']:
            total_trades = sum(r[mode]['trades'] for r in all_results.values())
            total_wins = sum(r[mode]['wins'] for r in all_results.values())
            total_pnl = sum(r[mode]['total_pnl'] for r in all_results.values())
            all_wins = []
            all_losses = []
            for r in all_results.values():
                for t in r[mode]['trade_details']:
                    if t['pnl_pct'] > 0:
                        all_wins.append(t['pnl_pct'])
                    else:
                        all_losses.append(t['pnl_pct'])
            
            agg[mode] = {
                'trades': total_trades,
                'wins': total_wins,
                'win_rate': total_wins / total_trades * 100 if total_trades else 0,
                'total_pnl': total_pnl,
                'avg_win': np.mean(all_wins) if all_wins else 0,
                'avg_loss': np.mean(all_losses) if all_losses else 0,
                'max_dd': max(r[mode]['max_drawdown_pct'] for r in all_results.values()),
            }
        
        print(f"\n  {'Metric':<25} {'WITHOUT TP':>15} {'WITH TP':>15} {'Delta':>12}")
        print(f"  {'─' * 67}")
        
        for label, key, fmt in [
            ('Total Trades', 'trades', 'd'),
            ('Total Wins', 'wins', 'd'),
            ('Win Rate %', 'win_rate', '.1f'),
            ('Total PnL $', 'total_pnl', '.2f'),
            ('Avg Win %', 'avg_win', '.2f'),
            ('Avg Loss %', 'avg_loss', '.2f'),
            ('Max Drawdown %', 'max_dd', '.2f'),
        ]:
            v1 = agg['no_tp'][key]
            v2 = agg['with_tp'][key]
            delta = v2 - v1
            sign = '+' if delta > 0 else ''
            print(f"  {label:<25} {v1:>15{fmt}} {v2:>15{fmt}} {sign}{delta:>11{fmt}}")
        
        # Aggregate exit reasons
        print(f"\n  Exit Reasons (WITH TP, all pairs):")
        all_reasons = {}
        for r in all_results.values():
            for reason, count in r['with_tp']['exit_reasons'].items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
            pct = count / agg['with_tp']['trades'] * 100 if agg['with_tp']['trades'] else 0
            print(f"    {reason}: {count} ({pct:.1f}%)")
        
        # Per-tool breakdown (WITH TP)
        print(f"\n  Per-Tool Performance (WITH TP, all pairs):")
        tool_stats = {}
        for r in all_results.values():
            for t in r['with_tp']['trade_details']:
                tool = t['tool']
                if tool not in tool_stats:
                    tool_stats[tool] = {'trades': 0, 'wins': 0, 'pnl_sum': 0}
                tool_stats[tool]['trades'] += 1
                if t['pnl_pct'] > 0:
                    tool_stats[tool]['wins'] += 1
                tool_stats[tool]['pnl_sum'] += t['pnl_pct']
        
        print(f"  {'Tool':<25} {'Trades':>8} {'WR%':>8} {'Avg PnL%':>10}")
        print(f"  {'─' * 51}")
        for tool, stats in sorted(tool_stats.items(), key=lambda x: -x[1]['pnl_sum']):
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] else 0
            avg_pnl = stats['pnl_sum'] / stats['trades'] if stats['trades'] else 0
            print(f"  {tool:<25} {stats['trades']:>8} {wr:>7.1f}% {avg_pnl:>+9.2f}%")
    
    print(f"\n{'=' * 80}")
    print("  DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
