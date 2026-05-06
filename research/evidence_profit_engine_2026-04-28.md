# Evidence Profit Engine - 2026-04-28

Scope: Kraken spot-only live bot in `run_final_bot.py`. Futures and external execution are off by default. Short detectors remain useful only as defensive overlays for long entries and open-position risk.

## Thesis

The bot's edge should come from a portfolio of detectors, but capital should flow only to detectors that show evidence after fees. The system now treats each long tool as a candidate in a capital competition instead of letting every signal consume account slots.

## Current Live Evidence

Eligible completed journal trades, excluding blocked `DRIFTUSD` and `RAVEUSD`, were still slightly negative: 49 completed trades, about `-$6.27` realized PnL.

The strongest eligible live signals were not broad swing tools. The best realized buckets were:

| Tool | Eligible Trades | PnL | Notes |
|---|---:|---:|---|
| `alt_btc_revert_t3` | 2 | `+$10.22` | Short-side signal only; defensive overlay in spot mode |
| `btc_alt_spread` | 24 | `+$1.23` | Thin live edge, but weak average after fees |
| `falling_wedge_short` | 1 | `+$0.51` | Short-side signal only; defensive overlay in spot mode |

Weak or damaging live buckets included `buy_btc_leading`, `simple_buy_uptrend`, `entropy_dip`, `flash_crash`, and low-score crash/dip variants. Those tools should not get automatic capital just because they generate a signal.

## New Capital Rules

`EVIDENCE_MODE=true` adds a central evidence gate after signal stacking and before entries are ranked.

Long tools must now pass one or more of these bars:

- tool-specific score floor based on live/audit/walk-forward evidence
- positive same-pair/tool/regime context
- stacked agreement from multiple detectors
- positive live edge after fees
- bull-offense regime for slow swing tools

The new walk-forward survivor, `panic_reversal_absorption`, is promoted modestly because it passed train/test/full checks:

- Full return: `+8.39%`
- Full profit factor: `1.31`
- Test return: `+1.26%`
- Test profit factor: `1.19`
- Test drawdown: `3.93%`

## Spot-Only Constraint

`SPOT_ONLY_MODE=true` is the default. That means:

- no Kraken Futures routing
- no external signal export unless explicitly enabled with `ENABLE_EXTERNAL_SIGNAL_EXPORT=true` and spot-only disabled
- short tools still build same-pair bearish pressure and market-wide bear pressure
- long entries are blocked, downsized, or exited when short evidence dominates

## Practical Effect

The system should now trade less often, but with higher capital quality:

- Low-score `simple_buy_uptrend` and `buy_btc_leading` no longer consume validation capital outside strong bull conditions.
- `btc_alt_spread`, `dip_buy_5pct`, `vpin_*`, `deep_dip_8h`, and `flash_crash` need stack/context/live proof instead of firing as isolated weak signals.
- `mega_crash` now requires much stronger scores before entry, reflecting the live loss from a mid-score eligible trade.
- Per-tool evidence metadata is written into the trade journal so future calibration can verify whether the gate is helping.

## Forward Validation Bar

Before increasing risk or account size, require a fresh forward sample:

| Metric | Pass Bar |
|---|---:|
| Eligible realized PnL | `> 0` |
| Profit factor | `>= 1.20` |
| Max drawdown from local peak | `<= 8%` |
| Completed eligible long trades | `>= 30` |
| Blocked-pair trades | `0` |
| New fills without native SL | `0` |
| Evidence-gated journal rows | present on new opens/closes |

The bot is now closer to a daily profit engine structurally, but it should still be judged by forward evidence rather than assumed profitable.