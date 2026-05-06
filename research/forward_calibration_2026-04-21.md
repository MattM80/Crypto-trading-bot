# Forward Calibration Snapshot

Date: 2026-04-21

## Scope

- Source: `logs/trade_journal.csv`
- Reconstruction method: match `OPEN` and `CLOSE` rows by pair and aggregate partial close PnL until the final close
- Goal: decide whether the pooled meta-model veto and size-scaling thresholds are ready for retuning

## Current Sample

- Completed trades reconstructed: 56 total
- Completed long trades: 47
- Completed short trades: 9
- Long-side realized win rate: 38.3%
- Long-side realized PnL: -$78.91

## Long-Side Observations

- The most common long close outcomes are hold timeouts with no remaining signal conviction.
- The heaviest long trade count came from `btc_alt_spread` (15), `quick_dip` (9), `entropy_dip` (6), and `dip_buy_5pct` (6).
- Historical score buckets are not stable enough to turn into a new global score floor because the sample spans multiple bot versions and pre-dates several recent filters.

Observed historical score buckets on completed long trades:

- `0 <= score < 8`: 18 trades, 27.8% win rate, -$7.29 PnL
- `8 <= score < 12`: 5 trades, 60.0% win rate, +$7.20 PnL
- `12 <= score < 16`: 11 trades, 45.5% win rate, -$11.58 PnL
- `16 <= score < 20`: 1 trade, 0.0% win rate, -$0.09 PnL
- `score >= 20`: 12 trades, 41.7% win rate, -$67.16 PnL

## Enriched Feature Coverage

The new journal fields are too fresh to support feature-level recalibration yet.

- Completed long trades with nonblank `range_pos_24h`: 0
- Completed long trades with nonblank `atr_pct`: 0
- Completed long trades with nonblank `short_pressure_score`: 0
- Completed long trades with nonblank `correlation_group`: 0

This means the latest feature logging is working for future analysis, but the completed long-trade sample still does not contain enough post-upgrade closes to tune those fields.

## Walk-Forward Meta-Model Check

The pooled meta-model was evaluated in a walk-forward style: each long trade was scored only with the information that would have been available before that trade closed, then the model was updated.

- Model-active completed long trades: 7
- Current live veto rule (`prob < 0.38` and `score < 16`) would have vetoed: 0
- Current live kept sample under that rule: 7 trades, -$3.72 PnL, 42.9% win rate

More aggressive in-sample vetoes looked better on this tiny sample, but they are not trustworthy enough to deploy:

- Best in-sample grid point found: `prob < 0.55` and `score < 18`
- That would have vetoed 5 of the 7 model-active long trades
- Vetoed trade bucket PnL: -$7.03
- Remaining kept bucket PnL: +$3.31

The problem is sample size, not signal quality. Seven model-active long closes is too little to justify changing live veto thresholds.

## Size Scaling Check

- Model-active long trades all sat in a narrow probability band around `0.40` to `0.50`
- Average live size multiplier across that sample was about `0.99`
- There is no support yet for widening the size multiplier range

## Recommendation

Do not change the live pooled meta-model thresholds yet.

Keep these live settings unchanged until the journal contains both:

- At least 15 completed long trades scored while the meta-model was active
- At least 20 completed long trades with the enriched feature columns populated

At the next calibration pass, re-check:

- Whether conviction-decay exits are replacing some hold-timeout/no-conviction exits
- Whether pooled probabilities start separating profitable from unprofitable longs more clearly
- Whether enriched features such as `short_pressure_score` and `range_pos_24h` have enough completed trades to support threshold tuning