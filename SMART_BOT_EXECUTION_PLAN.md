# Smart Bot Execution Plan

Date: 2026-04-21

Goal: turn the current long-only Kraken bot into a positive-expectancy system by improving trade selection, portfolio allocation, and defensive behavior under the no-shorts constraint.

## Completed

### 1. Harvestable volatility selection
- Changed dynamic pair selection from raw volatility to harvestable volatility.
- Highly volatile names pinned near their 24h lows are penalized instead of automatically prioritized.

### 2. Weak-long collapse filter
- Added collapse-regime gating for weaker long-entry tools.
- Weak dip-buy tools now require rebound confirmation in fast downtrends.

### 3. Runner behavior for volatile winners
- Fixed the post-partial stop anchor bug.
- High-volatility long trades can now keep a runner after the first partial TP instead of auto-closing the remainder at the same target.

### 4. Capital accounting fix
- Fixed active balance calculation so free capital is based on latest total balance minus deployed capital.
- This should reduce misleading sizing and some insufficient-USD behavior.

### 5. Long-only bearish overlay
- Added same-pair short-pressure aggregation from blocked short signals.
- Weak new longs can now be blocked when the same pair is flashing strong bearish short setups.
- Open longs can now have stops tightened or be exited when same-pair bearish pressure becomes strong.
- Pending long entries can now be cancelled when bearish pressure takes over.

### 6. False max-position fix
- Fixed a run-cycle bug where rejected signals still incremented the local open-position counter.
- This should reduce false max_positions_reached saturation and free capacity for valid long entries.

### 7. Live bot restart
- Replaced the old live process with the updated bot.
- Background process PID after restart: 26027
- Startup confirmed with live Kraken connection, state load, journal-backed contextual stats, forward-tool quarantine bootstrap, and pooled meta-model bootstrap.

### 8. Pair-tool-regime expectancy layer
- Added journal-backed expectancy tracking keyed by pair, tool, and entry regime.
- New trades now carry entry regime context and whole-trade PnL is reconstructed across partial exits.
- Signal scoring and risk sizing now use Bayesian-shrunk contextual multipliers instead of only global tool stats.
- Fixed the Kelly sizing path to use tracked trade counts instead of the stale total key.

### 9. Slot reservation and replacement
- Added dynamic slot reservation for trend-leader tools when the broader regime supports momentum and swing continuation.
- Non-trend rebound entries can no longer consume every slot in trend-supportive regimes.
- When the book is full, materially stronger new signals can now rotate out the weakest older non-runner position through the existing exit path.

### 10. Market-wide bear-pressure cash mode
- Added a market-wide bearish-pressure summary based on blocked short breadth and score dominance.
- Active capital allocation now contracts automatically when bearish setups dominate the market.
- Weak new longs and weak pending long buys are now filtered when the market flips into cautious, defensive, or risk-off cash mode.

### 11. Richer feature logging
- Trade journal schema now includes range position, ATR percent, same-pair short-pressure score, liquidity-cap usage, correlation group, and collapse-gate status.
- New entry-context fields are stored on live positions so close rows retain the original decision context.
- Existing journal files are migrated forward on startup so offline analysis can consume one consistent schema.

### 12. Offline meta-model
- Added a pooled logistic meta-model trained from completed journal trades with no external ML dependencies.
- The model now boots from historical journal data on startup and updates online when full trades close.
- Live execution uses the model only conservatively: weak low-probability trades can be vetoed and all active predictions are limited to narrow size scaling.

### 13. Conviction-decay exits
- Added an early exit path for weak long positions that lose same-direction signal support well before their full hold window expires.
- The decay check is throttled and conservative: it ignores runner positions, waits for meaningful hold time, and only exits when the trade is flat-to-red or bearish pressure is also building.
- This should reduce avoidable stop-loss drift on longs that stop working but have not yet reached timeout.

### 14. Forward-only tool quarantine
- Added a recent forward-only long-tool quarantine layer that rebuilds from post-upgrade journal rows instead of legacy mixed-version history.
- Weak recent long tools can now be downweighted or fully quarantined from entry selection, while short-side informational tools remain untouched.
- The layer is currently live but neutral because there are still no completed enriched-feature long closes to score.

### 15. Forward diagnostics surface
- Added structured diagnostics output for the new forward-only tool layer and exit behavior instead of relying only on raw logs.
- The bot now writes per-cycle snapshots to `logs/forward_diagnostics.csv` and daily rollups to `logs/forward_daily_summary.csv`.
- Diagnostics include forward-tool quarantine activity, meta-model veto counts, bearish-overlay rejection counts, conviction-decay exits, and hold-timeout/no-conviction exits.

### 16. Bull-offense sizing mode
- Added a symmetric bullish offense layer so the bot can press validated long edge in strong supportive tape instead of only shrinking in bearish tape.
- In strong bull conditions, trend-leader long tools can now receive a modest size lift plus slightly higher single-position and total-exposure caps.
- The layer stays dormant whenever market short pressure is not normal, so it should not increase risk in mixed or bearish regimes.

## Current State
- Bot is running in background with live trading enabled.
- Account balance on restart: about $340.72 on $300.00 start.
- Regime is still mixed to bearish, so many strongest live signals remain short-side information that cannot be traded directly.
- Contextual expectancy bootstrapped 56 historical outcomes into 81 pair-tool-regime buckets at startup.
- Trade journal schema migrated live with the new feature columns for offline modeling.
- Forward-only tool quarantine bootstrapped live and currently has zero completed enriched-feature long closes, so it is active but neutral.
- Pooled meta-model bootstrapped from 56 completed trades and is active at startup.
- Conviction-decay exits are now active on the live bot to cut weak stale longs earlier.
- Forward diagnostics are live; the new CSV outputs will become informative after the next completed cycle and the next set of post-upgrade closes.
- Bull-offense sizing is live but will only activate when sentiment, breadth, and short-pressure all flip supportive enough to justify pressing long exposure.

## Next Execution Priorities

### Priority 1: Forward calibration
- Let the richer journal fields and pooled meta-model collect fresh forward data; the current completed long-trade sample still has zero populated closes for the new feature columns.
- Do not retune pooled meta-model veto or size-scaling yet; a walk-forward check found only 7 model-active long closes, which is too small for a safe threshold change.
- Revisit veto and size-scaling thresholds after at least 15 model-active long closes and 20 enriched-feature long closes. See `research/forward_calibration_2026-04-21.md`.

### Priority 2: Reintroduce RAVE as a quarantined pair
- Keep RAVE blocked in live trading until a pair-specific profile is implemented and validated.
- Journal audit on 2026-04-22: 26 RAVE opens, 31 RAVE close rows, realized PnL -$2.48 overall.
- Profitable RAVE bucket: `crash_neg_ac` +$10.64, `dip_buy_5pct` +$10.46, `mega_crash` +$6.78, `vpin_toxic` +$2.74, `quick_dip` +$0.73. Combined: +$31.35 across 19 opens and 26 closes.
- Unprofitable RAVE bucket: `deep_dip_8h` -$14.76, `volatile_oversold` -$10.23, `quick_crash` -$8.84. Combined: -$33.83 across 5 opens and 5 closes.
- Constraint: live Kraken account cannot short RAVE, but `logs/rejected_signals.csv` recorded 1,335 blocked RAVE short signals with 71.9 average score. Treat that short-side pressure as a hard risk input even though it is not directly tradeable.

#### RAVE trading thesis
- RAVE should be traded as a rebound-harvest name, not as a blind falling-knife accumulator.
- The current edge appears to come from confirmed crash-reversal entries and fast profit harvesting after the first bounce.
- The current losses come from tools that keep buying during collapse or allow crash-bypass behavior to ignore bearish tape for too long.

#### RAVE pair profile
- Live allow-list for RAVE: `crash_neg_ac`, `mega_crash`, `dip_buy_5pct`.
- Probation-only for RAVE until enough enriched-feature closes exist: `quick_dip`, `vpin_toxic`.
- Hard-disable for RAVE: `deep_dip_8h`, `volatile_oversold`, `quick_crash`.
- No DCA on RAVE.
- Max one active RAVE position at a time.
- Start RAVE at 50% of normal risk and cap it at 0.15% of 24h dollar volume, even if the generic liquidity cap allows more.
- Reduce RAVE max trades per day from 3 to 1 or 2.
- After a losing RAVE trade, enforce a longer hard cooldown than the generic pair rule.

#### RAVE entry rules
- Weak-long tools on RAVE only fire when `collapse_gate == rebound_confirmed`.
- Crash-bypass tools do not get a blind collapse exemption on RAVE. Require either `rebound_confirmed` or a materially higher score plus low same-pair bearish pressure.
- For RAVE, remove or heavily tighten the existing `can_fight_bearish` exception so very strong blocked short pressure wins more often.
- Prefer entries only after `range_pos_24h` has recovered above 0.25 and the 1h rebound is positive.
- If market-wide bear mode is `defensive` or `risk_off`, RAVE should only be tradable through the allow-listed crash-reversal tools.

#### RAVE exit rules
- Take the first partial earlier than the generic plan. RAVE pays by overshooting quickly and then mean-reverting again.
- Disable runner mode for RAVE until forward expectancy turns positive again.
- Tighten conviction-decay timing and shorten the acceptable flat-trade hold window.
- Cancel pending RAVE long entries aggressively if short pressure spikes or price drifts away from the planned entry.

#### RAVE implementation surface
- Wire the pair profile into dynamic pair selection, entry gating, DCA prevention, pending-order cancellation, pair-specific size caps, and exit-parameter selection.
- Reuse the existing `collapse_gate`, same-pair short-pressure, and liquidity-cap fields instead of adding a second parallel risk model.
- Implement the profile as a dedicated pair-policy layer rather than scattering ad hoc `if pair == "RAVEUSD"` checks across unrelated logic.

#### RAVE go/no-go criteria
- Re-enable RAVE live only after a dry-run or tiny-size forward sample of at least 10 completed RAVE trades with enriched feature fields.
- Keep RAVE only if realized expectancy is positive, the bad-tool bucket stays disabled, and stop-loss clusters do not reappear.
- Kill RAVE again if collapse-state entries remain the main loss source or if the quarantined profile still cannot produce positive realized PnL.

### Priority 3: Generalize pair governance across all coins
- Yes, the bot should do a lighter-weight version of this for every tradeable pair.
- No, it should not require a manual RAVE-style memo for every coin. The right shape is an automated pair-policy layer built on top of the existing pair-tool-regime expectancy, forward-tool quarantine, pair cooldowns, pair daily limits, bearish-overlay signals, and liquidity caps.

#### Pair governance model
- Every pair should live in one of four states: `normal`, `watch`, `quarantine`, `blocked`.
- State transitions should be driven by pair-level realized expectancy, stop-loss clustering, blocked-short pressure, collapse-state loss rate, liquidity-cap usage, and sample size.
- Pair decisions should use Bayesian shrinkage or minimum-sample gates so one lucky or unlucky trade does not flip policy too aggressively.

#### What should be pair-specific
- Tool allow-list or deny-list when the pair shows a clear split between profitable and unprofitable entry styles.
- Risk multiplier relative to the base tool risk.
- Max trades per day, hard cooldown length, and whether DCA is allowed.
- Whether runner mode is allowed.
- Whether collapse-state entries require rebound confirmation even for normally exempt crash tools.

#### What should stay global
- The base signal engine and global tool definitions.
- Market-wide bear-pressure logic.
- Meta-model and forward-tool quarantine logic.
- Generic liquidity and drawdown safety limits.

#### Suggested rollout
- Start with an automated report that ranks every pair by realized PnL, stop-loss concentration, blocked-short pressure, and collapse-state outcomes.
- Promote only the worst outliers and strongest winners into explicit pair policies.
- Leave the middle of the distribution on global defaults plus the existing contextual expectancy layer.
- This keeps the system scalable: manual attention goes to names that are either costing money or showing enough unique edge to justify special treatment.

## Validation Checklist
- max_positions_reached should fall materially after the counter fix.
- active_balance should not exceed total_balance in fresh balance snapshots.
- bearish_overlay rejections should appear when strong blocked short setups cluster on a pair.
- stop-loss damage should fall relative to take-profit gains over the next 30 to 50 closes.
- long-side realized expectancy must turn positive before this can be called a winning bot.

## Definition of Success
- Positive realized expectancy on the long side over a fresh forward sample.
- Lower stop-loss concentration on names like RAVE and elimination of DRIFT-style outlier damage.
- Better opportunity capture without false portfolio saturation.
