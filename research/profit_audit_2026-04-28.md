# Profit Audit - 2026-04-28

Scope: active bot in `run_final_bot.py`, current state/logs, and attached historical 1h data. Blocked/old volatile names were excluded from the forward audit: `DRIFTUSD`, `RAVEUSD`, and the current global block list.

## Executive Read

The bot is not hopeless. Almost all realized damage came from blocked names. With blocked names removed, the live journal is close to breakeven, but not yet profitable enough to compound a $300 account.

The current failure mode is capital quality: too many weak long entries, too much account exposure, and missing native stop coverage after pending limit fills. The first objective is not more trades. It is fewer, higher-conviction trades with confirmed TP/SL placement after fill.

## Live Journal Attribution

Completed trades reconstructed from `logs/trade_journal.csv`:

| Segment | Trades | Win Rate | PnL | Profit Factor | Avg PnL % |
|---|---:|---:|---:|---:|---:|
| All completed | 69 | 39.1% | -$90.82 | 0.50 | -1.33% |
| Blocked names only | 23 | 47.8% | -$87.38 | 0.40 | -2.91% |
| Eligible names only | 46 | 34.8% | -$3.44 | 0.90 | -0.55% |
| Eligible since 2026-04-23 | 9 | 33.3% | -$0.15 | 0.93 | -0.54% |

Conclusion: blocking DRIFT/RAVE removed the account killer. The remaining system is near flat, but fees and weak signal churn still keep it below profitable.

## Current State Risk

`data/final_bot_state.json` showed balance `$308.70`, 8 active positions, and roughly `$282` deployed. That is about 91% of the account deployed.

Most active positions are low-score swing longs:

| Tool | Current examples | Issue |
|---|---|---|
| `simple_buy_uptrend` | JITOSOL, LTC, SOL, XBT, ETH | scores about 4.3-7.0, weeks-long hold, large slot usage |
| `buy_btc_leading` | XRP | recent eligible live closes are negative |
| `btc_alt_spread` | ZEC, HYPE | live-positive so far, but local historical test is fee-negative unless filtered tightly |

Stop coverage issue: only one active state row had `_sl_order_id`; most positions had no native Kraken stop recorded. The local watchdog can still close, but native stops are the safety net we want on a small account.

Root cause found: `execute_signal()` created an active position and tried to place TP before the entry order filled. Later, the fill handler skipped the TP/SL placement path because the active position already existed. This could leave real fills without native stops.

## Historical Data Audit

Using `data/binance_1h_extended` from 2023-03-27 to 2026-03-27 across the 16 locally covered pairs, excluding blocked names. This is an approximate strategy-family replay, not a perfect live simulator because live F&G, dynamic Kraken pairs, order book behavior, and all context multipliers are not fully reproduced.

Core result: low-score churn is fee-negative. Standalone tool results showed:

| Tool | Approx Trades | Avg PnL % | Profit Factor | Best Finding |
|---|---:|---:|---:|---|
| `major_pair_breakout` | 119 | +2.74% | 1.62 | Strongest validated long family |
| `mega_crash` | 307 | +1.30% | 1.41 | Better when score >= 40 |
| `flash_crash` | 665 | +0.16% | 1.08 | Better when score >= 30 |
| `buy_breakout_simple` | 315 | +0.20% | 1.03 | Marginal, needs regime/score filter |
| `simple_buy_uptrend` | 797 | -1.01% | 0.86 | Low/mid score buckets are weak |
| `buy_btc_leading` | 410 | -2.02% | 0.73 | Weak standalone; should be probationary |
| `btc_alt_spread` | 5598 | -0.64% | 0.70 | Needs strong live/context filters |
| `dip_buy_5pct` | 2310 | -0.54% | 0.70 | Only high-score crashes worked |
| `vpin_dip` | 1420 | -0.53% | 0.71 | Needs high-score/deeper panic filter |

The current bot was letting scores around 4-7 consume slots. That is exactly the zone the audit does not trust for $300 validation.

## Changes Applied

1. Added default `VALIDATION_ACCOUNT_MODE=true` in `run_final_bot.py`.

2. Tightened small-account deployment defaults:
   - `MAX_ACTIVE_POSITIONS`: 4 by default in validation mode
   - `RISK_PER_TRADE`: 4% by default in validation mode
   - `MAX_POSITION_PCT`: 10% by default in validation mode
   - `MAX_TOTAL_EXPOSURE_PCT`: 55% by default in validation mode

3. Added validation score floors for new long entries. Examples:
   - `simple_buy_uptrend` and `buy_btc_leading`: minimum 8
   - `btc_alt_spread`: minimum 5, but unknown dynamic pairs need at least 8
   - crash tools need stronger scores before consuming capital
   - dynamic pairs without local historical coverage need at least score 8

4. Fixed pending-entry handling:
   - positions reserved for unfilled limit buys are marked `_pending_entry`
   - pending entries are skipped by position-management exit logic
   - TP/SL placement is deferred until Kraken confirms the entry filled
   - fill confirmation now updates the existing reserved position and places TP plus native SL

## Validation Plan

Run the bot in validation mode until it has at least 30 completed eligible trades or 14 calendar days, whichever comes later.

Pass criteria before scaling:

| Metric | Required |
|---|---:|
| Eligible realized PnL | positive |
| Profit factor | >= 1.20 |
| Max drawdown from local peak | <= 8% |
| Trades from blocked names | 0 |
| New fills without `_sl_order_id` after reconciliation | 0 |
| Any one tool's loss contribution | less than 40% of total gross losses |

Scale only after passing. Suggested scale ladder: keep the same percentages, not fixed dollars. Increase capital in steps and reset validation at each step: $300 -> $500 -> $1,000 -> $2,500. Do not increase `MAX_TOTAL_EXPOSURE_PCT` until the system passes at the current level.

## Immediate Operational Note

The code is patched, but any already-running launchd bot process will not use the changes until restarted. Existing live positions also need reconciliation/backfill so native stops are placed under the new fill-handling path.