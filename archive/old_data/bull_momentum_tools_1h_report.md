# BULL/GREED + BREAKOUT/MOMENTUM Tools - 1H Testing Report

**Generated:** 2026-03-24 11:52:48

## Test Configuration
- **Data:** Real 1h Binance candles, 16 pairs, 8760 bars each
- **Walk-forward:** Bars 0-4380 = in-sample, bars 4380-8760 = out-of-sample
- **Fee-adjusted:** -0.52% round-trip subtracted from every return
- **Forward returns:** +8 bars (8h) and +24 bars (24h)
- **Win condition:** (forward_return - fees) > 0
- **Tools tested:** 70 different trading strategies
- **Note:** Many tools use optimized/relaxed thresholds and statistical proxies for complex indicators

## Results Summary

| Tool | Dir | Signals | Pairs | WR_8h | WR_24h | Avg_Ret_8h | Avg_Ret_24h | Status |
|------|-----|---------|-------|-------|--------|------------|-------------|--------|
| sma50_ext_15 | S | 491 | 14 | 55.4% | 63.1% | -0.93% | -0.87% | ✅ PASS |
| sma50_ext_12 | S | 975 | 15 | 52.9% | 61.7% | -0.63% | -0.53% | ✅ PASS |
| alt_btc_revert_t1_manual | S | 975 | 15 | 52.9% | 61.7% | -0.63% | -0.53% | ✅ PASS |
| sma50_ext_10 | S | 1487 | 16 | 51.9% | 61.7% | -0.35% | -0.09% | ✅ PASS |
| sma50_ext_fat_tail | S | 135 | 7 | 51.1% | 60.7% | -2.48% | -5.20% | ✅ PASS |
| green_exhaustion | S | 184 | 16 | 47.8% | 60.3% | -0.44% | -0.45% | ✅ PASS |
| rsi_pump_8h | S | 506 | 16 | 54.2% | 60.3% | -0.33% | +0.56% | ✅ PASS |
| alt_btc_revert_t2_manual | S | 2270 | 16 | 49.2% | 59.7% | -0.37% | -0.06% | ✅ PASS |
| sma50_ext_neg_ac | S | 171 | 14 | 50.3% | 59.1% | -0.98% | -0.29% | ✅ PASS |
| rsi_pump_fat_tail | S | 163 | 15 | 49.7% | 58.9% | -0.48% | +0.07% | ✅ PASS |
| mega_pump_sell_t1 | S | 656 | 16 | 52.7% | 58.7% | -0.13% | +0.61% | ✅ PASS |
| sma50_ext_kurt | S | 297 | 13 | 50.2% | 58.6% | -1.65% | -2.57% | ✅ PASS |
| greed_short_t2 | S | 2779 | 16 | 48.9% | 58.5% | -0.10% | +0.31% | ✅ PASS |
| alt_btc_neg_ac | S | 98 | 11 | 57.1% | 58.2% | -0.88% | -0.39% | ✅ PASS |
| green_exhaust_kurt | S | 50 | 14 | 48.0% | 58.0% | -0.18% | -1.28% | ✅ PASS |
| thursday_short | S | 3500 | 16 | 51.1% | 57.9% | +0.14% | +0.28% | ✅ PASS |
| sma50_ext_8 | S | 3431 | 16 | 48.1% | 57.5% | -0.33% | -0.01% | ✅ PASS |
| falling_wedge_short | S | 2764 | 16 | 44.6% | 56.7% | +0.00% | +0.60% | ✅ PASS |
| alt_btc_revert_t3_manual | S | 5275 | 16 | 47.8% | 56.4% | -0.27% | +0.01% | ✅ PASS |
| mega_pump_sell_t2 | S | 1388 | 16 | 47.0% | 55.0% | -0.36% | +0.19% | ✅ PASS |
| rsi_pump_12h | S | 1569 | 16 | 47.3% | 54.9% | -0.33% | +0.13% | ✅ PASS |
| entropy_short | S | 1577 | 16 | 46.0% | 54.8% | -0.30% | +0.02% | ✅ PASS |
| crash_buy_10pct | L | 1413 | 16 | 54.0% | 54.3% | +0.18% | +1.26% | ✅ PASS |
| month_start_long | L | 2763 | 16 | 47.0% | 53.9% | +0.00% | +0.72% | ✅ PASS |
| distribution_short | S | 9851 | 16 | 46.6% | 53.4% | -0.13% | +0.18% | ✅ PASS |
| ema_cross_short | S | 24713 | 16 | 45.6% | 53.2% | -0.17% | +0.13% | ✅ PASS |
| late_us_short | S | 1301 | 16 | 45.8% | 52.9% | -0.08% | +0.16% | ✅ PASS |
| dip_buy_5pct | L | 2886 | 16 | 52.7% | 49.1% | +0.11% | +0.25% | ✅ PASS |
| crash_buy_8pct | L | 1016 | 16 | 51.8% | 47.8% | +0.03% | +0.18% | ✅ PASS |
| correlation_breakdown_short | S | 2976 | 16 | 44.8% | 51.1% | -0.49% | -0.39% | ✅ PASS |
| rsi_overbought_80 | S | 3071 | 16 | 43.9% | 50.6% | -0.29% | -0.23% | ✅ PASS |
| rsi_overbought_70 | S | 8213 | 16 | 43.9% | 50.3% | -0.31% | -0.15% | ✅ PASS |
| dip_buy_3pct | L | 4912 | 16 | 50.2% | 46.7% | -0.05% | -0.18% | ✅ PASS |
| rsi_overbought_75 | S | 5090 | 16 | 43.6% | 50.0% | -0.32% | -0.21% | ❌ FAIL |
| relative_strength_short | S | 15705 | 16 | 41.7% | 48.6% | -0.46% | -0.34% | ❌ FAIL |
| breakout_detect | L | 25 | 12 | 36.0% | 48.0% | -0.86% | -1.35% | ❌ FAIL |
| dip_buy_2pct | L | 9275 | 16 | 46.9% | 44.9% | -0.21% | -0.36% | ❌ FAIL |
| sunday_long | L | 5199 | 16 | 41.5% | 46.8% | -0.37% | -0.36% | ❌ FAIL |
| alt_btc_neg_ac_5 | S | 139 | 13 | 45.3% | 46.8% | -0.88% | -0.59% | ❌ FAIL |
| wick_absorption_short | S | 4020 | 16 | 36.7% | 46.2% | -0.54% | -0.31% | ❌ FAIL |
| rsi_oversold_20 | L | 3394 | 16 | 42.2% | 45.8% | -0.43% | -0.49% | ❌ FAIL |
| volatility_squeeze_breakout_short | S | 1115 | 16 | 32.8% | 45.7% | -0.67% | -0.18% | ❌ FAIL |
| rsi_oversold_25 | L | 5915 | 16 | 42.6% | 44.5% | -0.40% | -0.46% | ❌ FAIL |
| btc_lag_1h | L | 8385 | 16 | 43.2% | 44.0% | -0.42% | -0.51% | ❌ FAIL |
| correlation_breakdown_long | L | 8752 | 16 | 42.9% | 43.5% | -0.45% | -0.51% | ❌ FAIL |
| rsi_oversold_30 | L | 9753 | 16 | 42.6% | 43.4% | -0.44% | -0.55% | ❌ FAIL |
| btc_lag_3h | L | 13700 | 16 | 42.7% | 43.3% | -0.42% | -0.55% | ❌ FAIL |
| btc_lead_lag_buy | L | 20071 | 16 | 41.7% | 42.6% | -0.45% | -0.62% | ❌ FAIL |
| bb_above_long_t1 | L | 580 | 16 | 42.4% | 40.3% | -0.38% | -0.60% | ❌ FAIL |
| late_us_long | L | 1595 | 16 | 41.9% | 41.8% | -0.27% | -0.57% | ❌ FAIL |
| sunday_short | S | 4785 | 16 | 41.7% | 41.5% | -0.17% | -0.35% | ❌ FAIL |
| relative_strength_long | L | 15208 | 16 | 40.2% | 40.3% | -0.47% | -0.58% | ❌ FAIL |
| volume_spike_5x | L | 3329 | 16 | 39.5% | 38.8% | -0.63% | -0.86% | ❌ FAIL |
| high_break_skew | L | 676 | 16 | 39.3% | 36.4% | -0.26% | -0.72% | ❌ FAIL |
| volume_spike_2x | L | 10226 | 16 | 38.8% | 39.3% | -0.59% | -0.78% | ❌ FAIL |
| volume_spike_3x | L | 6622 | 16 | 39.2% | 38.7% | -0.61% | -0.85% | ❌ FAIL |
| high_breakout_30 | L | 1335 | 16 | 36.8% | 39.2% | -0.44% | -0.76% | ❌ FAIL |
| thursday_long | L | 6483 | 16 | 36.1% | 38.1% | -0.97% | -1.20% | ❌ FAIL |
| wick_absorption_long | L | 5084 | 16 | 33.0% | 37.9% | -0.73% | -0.92% | ❌ FAIL |
| high_break_pos_ac | L | 776 | 16 | 37.8% | 36.5% | -0.39% | -0.84% | ❌ FAIL |
| bb_squeeze_15 | L | 1248 | 16 | 35.8% | 37.4% | -0.60% | -0.69% | ❌ FAIL |
| bb_squeeze_30 | L | 1248 | 16 | 35.8% | 37.4% | -0.60% | -0.69% | ❌ FAIL |
| bb_break_pos_ac | L | 1248 | 16 | 35.8% | 37.4% | -0.60% | -0.69% | ❌ FAIL |
| high_breakout_50 | L | 965 | 16 | 37.4% | 37.3% | -0.30% | -0.83% | ❌ FAIL |
| fomo_ride | L | 2187 | 16 | 36.6% | 36.9% | -0.65% | -0.77% | ❌ FAIL |
| month_start_short | S | 4149 | 16 | 34.8% | 36.3% | -0.75% | -1.08% | ❌ FAIL |
| high_breakout_50_nv | L | 1924 | 16 | 35.2% | 36.3% | -0.48% | -0.95% | ❌ FAIL |
| bb_above_long_t2 | L | 3310 | 16 | 33.9% | 36.1% | -0.67% | -1.01% | ❌ FAIL |
| volatility_squeeze_breakout_long | L | 833 | 16 | 31.7% | 35.9% | -0.90% | -1.21% | ❌ FAIL |
| hurst_trend | L | 7827 | 16 | 31.4% | 34.1% | -0.85% | -1.13% | ❌ FAIL |

## Tool Analysis

### ✅ PASSED TOOLS (33)

