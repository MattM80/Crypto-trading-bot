#!/usr/bin/env python3
"""Diagnose why the strategy produces zero signals on real data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from strategies import (
    create_strategy, calc_ema, calc_rsi, calc_atr, calc_adx, detect_regime,
    calc_macd, calc_obv, calc_bb_width, calc_vwap, calc_volume_sma,
    detect_mean_reversion, fee_aware_ev_filter,
)

# Download 1h data for BTC
base = "https://api.kraken.com/0"
resp = requests.get(f"{base}/public/OHLC", params={"pair": "XBTUSD", "interval": 60}, timeout=30)
result = resp.json().get("result", {})
ohlcv = None
for k, v in result.items():
    if isinstance(v, list) and len(v) > 0:
        ohlcv = v
        break

rows = [{"time": int(r[0]), "open": float(r[1]), "high": float(r[2]),
         "low": float(r[3]), "close": float(r[4]), "volume": float(r[6])} for r in ohlcv]
df = pd.DataFrame(rows)
print(f"Candles: {len(df)}, price range: {df['close'].min():.0f} - {df['close'].max():.0f}")

strategy = create_strategy("adaptive")

# Try with growing windows
signal_count = 0
for i in range(120, len(df)):
    window = df.iloc[:i + 1].copy()
    signals = strategy.generate_signals(window, "XBTUSD")
    if signals:
        signal_count += 1
        sig = signals[0]
        print(f"Bar {i}: {sig.action} conf={sig.confidence:.3f} regime={sig.regime}")
        print(f"  reason: {sig.reason[:150]}")
    if signal_count >= 10:
        break

print(f"\nTotal signals found: {signal_count}")

if signal_count == 0:
    print("\n--- DIAGNOSING ZERO SIGNALS ---\n")
    # Check core indicator values at the last bar
    close = pd.to_numeric(df["close"])
    ema_f = calc_ema(close, 9)
    ema_s = calc_ema(close, 21)
    rsi = calc_rsi(close, 14)
    atr = calc_atr(df, 14)
    adx = calc_adx(df, 14)
    regime = detect_regime(df)
    macd_line, signal_line, macd_hist = calc_macd(close, 12, 26, 9)

    print(f"Last bar close: {close.iloc[-1]:.0f}")
    print(f"EMA 9/21:       {ema_f.iloc[-1]:.0f} / {ema_s.iloc[-1]:.0f}")
    print(f"RSI:            {rsi.iloc[-1]:.1f}")
    print(f"ATR:            {atr.iloc[-1]:.0f}")
    print(f"ADX:            {adx.iloc[-1]:.1f}")
    print(f"MACD hist:      {macd_hist.iloc[-1]:.1f}")
    print(f"Regime:         {regime}")
    print(f"EMA bullish:    {ema_f.iloc[-1] > ema_s.iloc[-1]}")

    # Manually compute what confidence would be
    print("\n--- MANUAL BUY CONFIDENCE TRACE ---\n")
    conf = 0.0
    reasons = []

    # EMA
    if ema_f.iloc[-1] > ema_s.iloc[-1]:
        if ema_f.iloc[-2] <= ema_s.iloc[-2]:
            conf += 0.20
            reasons.append(f"EMA cross: +0.20 = {conf:.2f}")
        else:
            conf += 0.10
            reasons.append(f"EMA aligned: +0.10 = {conf:.2f}")

    # MACD
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
            conf += 0.15
            reasons.append(f"MACD bullish: +0.15 = {conf:.2f}")
        else:
            conf += 0.05
            reasons.append(f"MACD weak: +0.05 = {conf:.2f}")

    # RSI
    rsi_val = rsi.iloc[-1]
    if rsi_val < 70:
        conf += 0.05
        if rsi_val < 40:
            conf += 0.10
            reasons.append(f"RSI favorable ({rsi_val:.0f}): +0.15 = {conf:.2f}")
        else:
            reasons.append(f"RSI OK ({rsi_val:.0f}): +0.05 = {conf:.2f}")
    else:
        conf -= 0.10
        reasons.append(f"RSI overbought: -0.10 = {conf:.2f}")

    # ADX
    if adx.iloc[-1] > 20:
        conf += 0.10
        reasons.append(f"ADX strong ({adx.iloc[-1]:.0f}): +0.10 = {conf:.2f}")

    # Regime penalty
    if regime == "high_volatility":
        old = conf
        conf *= 0.55
        reasons.append(f"HIGH VOL regime: *0.55 ({old:.2f} -> {conf:.2f})")
    elif regime == "trending_down":
        old = conf
        conf *= 0.50
        reasons.append(f"DOWNTREND: *0.50 ({old:.2f} -> {conf:.2f})")
    elif regime == "trending_up":
        old = conf
        conf *= 1.15
        reasons.append(f"UPTREND: *1.15 ({old:.2f} -> {conf:.2f})")

    # Mean reversion check
    mr = detect_mean_reversion(df)
    if mr["buy_boost"] > 0:
        conf += mr["buy_boost"]
        reasons.append(f"MEAN REVERSION: +{mr['buy_boost']:.2f} = {conf:.2f}")
    print(f"MR signal: {mr}")

    for r in reasons:
        print(f"  {r}")

    print(f"\nFinal confidence: {conf:.3f}")
    print(f"Min threshold:    0.42")
    print(f"Would signal:     {'YES' if conf >= 0.42 else 'NO'}")

    # Check R:R
    price = close.iloc[-1]
    atr_val = atr.iloc[-1]
    sl = price - 2.0 * atr_val
    tp = price + 3.0 * atr_val
    risk = price - sl
    reward = tp - price
    rr = reward / risk if risk > 0 else 0
    passes_fee, fee_rr = fee_aware_ev_filter(price, sl, tp, "BUY")
    print(f"\nR:R check:        {rr:.2f} (need >= 1.49)")
    print(f"Fee R:R:          {fee_rr:.2f} (passes: {passes_fee})")

    # Scan ALL bars to find what fraction pass the confidence threshold
    print("\n--- SCANNING ALL BARS ---\n")
    pass_count = 0
    max_conf_seen = 0.0
    for i in range(120, len(df)):
        window = df.iloc[:i + 1].copy()
        close_w = pd.to_numeric(window["close"])
        ema_f_w = calc_ema(close_w, 9)
        ema_s_w = calc_ema(close_w, 21)
        c = 0.0
        if ema_f_w.iloc[-1] > ema_s_w.iloc[-1]:
            c += 0.10
        rsi_w = calc_rsi(close_w, 14)
        if rsi_w.iloc[-1] < 70:
            c += 0.05
        adx_w = calc_adx(window, 14)
        if adx_w.iloc[-1] > 20:
            c += 0.10
        regime_w = detect_regime(window)
        if regime_w == "high_volatility":
            c *= 0.55
        elif regime_w == "trending_down":
            c *= 0.50
        elif regime_w == "trending_up":
            c *= 1.15
        max_conf_seen = max(max_conf_seen, c)
        if c >= 0.42:
            pass_count += 1

    total = len(df) - 120
    print(f"Bars tested:        {total}")
    print(f"Pass threshold:     {pass_count} ({pass_count / total * 100:.1f}%)")
    print(f"Max conf seen:      {max_conf_seen:.3f}")
