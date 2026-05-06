# Profit Strategy Lab - 2026-04-28

Scope: Kraken-compatible spot-only long strategies tested on local 1h OHLCV, with blocked names excluded.

Data range: 2023-07-14 13:00:00+00:00 to 2026-03-27 04:00:00+00:00
Pairs: AAVEUSD, ADAUSD, ATOMUSD, AVAXUSD, DOGEUSD, DOTUSD, ETHUSD, FILUSD, LINKUSD, LTCUSD, NEARUSD, SOLUSD, UNIUSD, XBTUSD, XLMUSD, XRPUSD
Assumptions: $300 start, 0.19% cost per side, 10% max position, 55% max total exposure.

## Best Walk-Forward Candidates

| Rank | Strategy | Train Ret | Train PF | Train DD | Test Ret | Test PF | Test DD | Test Trades | Full Ret | Full PF |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `panic_reversal_drop_8_vol_1.5` | 7.04% | 1.36 | 7.43% | 1.26% | 1.19 | 3.93% | 36 | 8.39% | 1.31 |
| 2 | `panic_reversal_drop_8_vol_2.0` | 6.66% | 1.46 | 6.46% | 0.75% | 1.14 | 3.29% | 27 | 7.46% | 1.37 |
| 3 | `panic_reversal_drop_12_vol_2.0` | 2.43% | 1.36 | 5.14% | 0.30% | 1.28 | 1.21% | 7 | 2.74% | 1.35 |
| 4 | `relative_strength_top_3_rebalance_72` | 4.05% | 1.06 | 16.84% | 6.26% | 1.23 | 10.47% | 78 | 10.57% | 1.12 |
| 5 | `trend_donchian_480_vol_1.3` | 2.95% | 1.03 | 14.81% | 2.84% | 1.11 | 6.87% | 139 | 5.87% | 1.05 |
| 6 | `relative_strength_top_2_rebalance_72` | 3.26% | 1.08 | 11.70% | -0.81% | 0.96 | 10.91% | 57 | 2.42% | 1.04 |
| 7 | `trend_donchian_480_vol_1.0` | -1.52% | 0.98 | 17.38% | 3.56% | 1.13 | 7.19% | 143 | 1.98% | 1.02 |
| 8 | `panic_reversal_drop_12_vol_1.5` | 1.73% | 1.21 | 6.40% | -0.58% | 0.71 | 2.01% | 9 | 1.15% | 1.11 |
| 9 | `relative_strength_top_4_rebalance_72` | 0.40% | 1.01 | 19.25% | 5.27% | 1.14 | 12.92% | 97 | 5.69% | 1.05 |
| 10 | `relative_strength_top_3_rebalance_24` | -1.39% | 0.99 | 19.30% | -1.19% | 0.97 | 12.12% | 145 | -2.57% | 0.98 |
| 11 | `relative_strength_top_2_rebalance_24` | -0.57% | 0.99 | 16.07% | -5.24% | 0.84 | 11.97% | 108 | -5.78% | 0.95 |
| 12 | `squeeze_breakout_72_vol_1.2` | -6.33% | 0.78 | 7.24% | -2.26% | 0.85 | 3.61% | 155 | -8.44% | 0.80 |

## Interpretation

Walk-forward candidates passing test PF >= 1.15, positive test return, and test drawdown <= 12%: 3.
A strategy should not be wired live just because it wins in-sample. The live bot should only consume candidates that survive the test segment and remain sane on full-sample behavior.

## Next Live Integration Rule

If a candidate passes, wire only the signal logic first and keep the existing validation-mode caps. The live bot should reject the strategy if BTC regime, liquidity, or native stop placement cannot be confirmed.

Recommended first candidate:

- `panic_reversal_drop_8_vol_1.5` (panic_reversal)
- Test return: 1.26%
- Test profit factor: 1.19
- Test max drawdown: 3.93%
- Test trades: 36

## Family Summary

| Family | Tested | Median Test Ret | Median Test PF | Best Test Ret |
|---|---:|---:|---:|---:|
| panic_reversal | 4 | 0.53% | 1.16 | 1.26% |
| relative_strength | 6 | -1.00% | 0.97 | 6.26% |
| squeeze_breakout | 6 | -6.89% | 0.69 | -2.26% |
| trend_breakout | 6 | -4.84% | 0.89 | 3.56% |
