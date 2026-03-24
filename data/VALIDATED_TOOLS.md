# VALIDATED TOOLS — Production Ready
# All tested: Real 1h Binance data, walk-forward OOS, 0.52% Kraken fees

## CRASH/BEAR TOOLS (LONG) — 15 tools
1. volatile_oversold: atr_pct>3 AND rsi7<25 → LONG | WR_8h=73.8%, Ret_8h=+2.07%
2. crash_buy: ret_24h<-10 AND rsi7<20 → LONG | WR_8h=65.1%, Ret_24h=+1.90%
3. mega_crash: ret_24h<-15 → LONG | WR_24h=52.5%, Ret_24h=+1.35%
4. crash_neg_ac: ret_24h<-10 AND autocorr<-0.05 → LONG | WR_8h=62.1%, Ret_8h=+1.25%
5. blood_in_streets: 70%+ coins down >2% AND rsi7<20 → LONG | WR_24h=61.7%, Ret_24h=+1.10%
6. quick_crash: ret_8h<-10 → LONG (8h hold only) | WR_8h=59.1%, Ret_8h=+0.98%
7. crash_mean_revert: ret_24h<-8 AND Hurst<0.45 → LONG | WR_8h=61.3%, Ret_8h=+0.98%
8. vpin_dip: ret_8h<-5 AND VPIN>0.5 → LONG | WR_8h=58.8%, Ret_8h=+0.73%
9. market_panic_70: 70%+ coins down >3% in 4h → LONG | WR_24h=59.0%, Ret_24h=+0.75%
10. flash_crash: ret_12h<-10 → LONG | WR_8h=55.8%, Ret_8h=+0.51%
11. deep_dip_8h: -10<ret_8h<-8 → LONG | WR_8h=54.8%, Ret_8h=+0.22%
12. entropy_dip: entropy<2.5 AND ret_4h<-2 → LONG | WR_8h=52.8%, Ret_8h=+0.45%
13. vpin_toxic: VPIN>0.7 AND red candle → LONG | WR_8h=53.8%, Ret_8h=+0.45%
14. btc_alt_spread: alt lagging BTC 3%+ AND rsi7<35 → LONG | WR_24h=55.2%, Ret_24h=+0.45%
15. quick_dip: ret_4h<-5 → LONG | WR_8h=55.5%, Ret_8h=+0.13%

## BULL/GREED TOOLS (SHORT) — 13 tools
16. mega_pump_sell_t1: rsi7>80 AND ret_12h>=10 → SHORT | WR_24h=58.7%, Ret_24h=+0.61%
17. rsi_pump_8h: rsi7>80 AND ret_8h>=10 → SHORT | WR_24h=60.3%, Ret_24h=+0.56%
18. falling_wedge_short: lower highs+lows converging + price>SMA50 → SHORT | WR_24h=56.7%, Ret_24h=+0.60%
19. greed_short_t2: rsi7>75 AND ret_8h>5 AND price>SMA50 → SHORT | WR_24h=58.5%, Ret_24h=+0.31%
20. thursday_short: Thursday AND price>SMA50 → SHORT | WR_24h=57.9%, Ret_24h=+0.28%
21. mega_pump_sell_t2: rsi7>80 AND ret_12h>=8 → SHORT | WR_24h=55.0%, Ret_24h=+0.19%
22. distribution_short: lower highs + vol decline + RSI fall + price>SMA50 → SHORT | WR_24h=53.4%, Ret_24h=+0.18%
23. late_us_short: hour==21 UTC AND price>SMA50 → SHORT | WR_24h=52.9%, Ret_24h=+0.16%
24. rsi_pump_12h: rsi7>80 AND ret_12h>=8 → SHORT | WR_24h=54.9%, Ret_24h=+0.13%
25. ema_cross_short: ema5>ema13 AND price>SMA50 → SHORT | WR_24h=53.2%, Ret_24h=+0.13%
26. rsi_pump_fat_tail: rsi7>80 AND ret_12h>10 AND kurtosis>5 → SHORT | WR_24h=58.9%, Ret_24h=+0.07%
27. entropy_short: entropy<2.5 AND price>SMA50 → SHORT | WR_24h=54.8%, Ret_24h=+0.02%
28. alt_btc_revert_t3: alt outperforms BTC 3-5% 24h → SHORT | WR_24h=56.4%, Ret_24h=+0.01%

## NEUTRAL/TRANSITION TOOLS — 2 tools
29. month_start_long: day 1-3 → LONG | WR_24h=53.9%, Ret_24h=+0.72%
30. dip_buy_5pct: ret_4h<-5 → LONG | WR_8h=52.7%, Ret_8h=+0.11%

## GRID ENGINE — separately validated
- +136% over 12 months, 139 round trips, 0.98% per trip
- 16 pairs, 3 levels each, 1.5% TP (2x in downtrend)
- ATR-based spacing, SMA50 regime detection

## DEAD — DO NOT INCLUDE
ALL breakout longs (BB squeeze, high breakout, bb_above_long)
ALL BTC lead-lag longs (btc_lead_lag_buy, btc_lag_1h, btc_lag_3h)
fomo_ride, hurst_trend, whale_buy, capitulation, volume_climax
dip_buy (3% threshold too broad), zscore_extreme, panic_close
fat_tail_revert, relief_rally, mega_align (too rare)
sunday_short, month_start_short
correlation_breakdown, relative_strength, wick_absorption, volatility_squeeze
All volume spike tools
