"""
Monte Carlo forecast — comprehensive strategy comparison on Kraken.

Tests all viable strategies and picks the most profitable:
  1. trend_momentum (10-indicator, high-conviction, wide targets)
  2. adaptive (learnable weights, multi-TF, BTC filter, smart limit orders)

The adaptive strategy gets TWO runs:
  - adaptive (taker fees 0.26%/side)
  - adaptive_maker (limit orders: 0.16%/side — the default entry mode)

Uses realistic Kraken fees, 2% risk, 8 trading pairs.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from strategies import create_strategy

# ── Config ─────────────────────────────────────────────────────────────
STARTING_BALANCE = 5000.0
KRAKEN_TAKER_FEE = 0.0026   # 0.26% per side
KRAKEN_MAKER_FEE = 0.0016   # 0.16% per side (limit orders)
MAX_RISK_PER_TRADE = 0.02
MAX_NOTIONAL_PCT = 0.15
NUM_PAIRS = 8                # 8 pairs now (added ADA, AVAX, LINK)
N_PROFILE_SEEDS = 500
N_SIMS = 10_000
DAYS = 90

# Strategies to compare
STRATEGIES_TO_TEST = [
    {"name": "trend_momentum", "fee": KRAKEN_TAKER_FEE, "label": "trend_momentum (taker)"},
    {"name": "adaptive",       "fee": KRAKEN_TAKER_FEE, "label": "adaptive (taker)"},
    {"name": "adaptive",       "fee": KRAKEN_MAKER_FEE, "label": "adaptive (maker — limit orders)"},
]


def profile_strategy(name: str, n_seeds: int = 500) -> dict:
    """Profile a strategy on synthetic data. Returns stats dict."""
    strategy = create_strategy(name)

    # For adaptive strategy, simulate HTF context (boosts ~15% of signals)
    is_adaptive = hasattr(strategy, 'set_context')
    if is_adaptive:
        from trade_journal import TradeJournal
        tj = TradeJournal()

    outcomes = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        n = 200
        drift = rng.choice([-0.003, -0.001, 0.0, 0.001, 0.003])
        vol = rng.uniform(0.01, 0.04)
        prices = 50000 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        df = pd.DataFrame({
            'open': prices * (1 + rng.normal(0, 0.002, n)),
            'high': prices * (1 + abs(rng.normal(0, 0.008, n))),
            'low':  prices * (1 - abs(rng.normal(0, 0.008, n))),
            'close': prices,
            'volume': rng.uniform(500, 3000, n),
        })

        # For adaptive: create synthetic HTF (1h = ~12x 5m) and BTC data
        if is_adaptive:
            # Subsample to simulate hourly bars
            htf_idx = list(range(0, n, 12))
            if len(htf_idx) >= 55:
                htf_df = df.iloc[htf_idx].reset_index(drop=True)
            else:
                htf_df = df.copy()
            # BTC data: same drift direction (correlated market)
            btc_prices = 60000 * np.exp(np.cumsum(rng.normal(drift * 0.8, vol, n)))
            btc_df = pd.DataFrame({
                'open': btc_prices * (1 + rng.normal(0, 0.002, n)),
                'high': btc_prices * (1 + abs(rng.normal(0, 0.008, n))),
                'low':  btc_prices * (1 - abs(rng.normal(0, 0.008, n))),
                'close': btc_prices,
                'volume': rng.uniform(500, 3000, n),
            })
            strategy.set_context(
                htf_data={'SIM': htf_df},
                btc_data=btc_df,
                trade_journal=tj,
            )

        sigs = strategy.generate_signals(df, 'SIM')
        if not sigs:
            continue

        for sig in sigs:
            entry = sig.entry_price
            sl, tp = sig.stop_loss, sig.take_profit
            if entry <= 0:
                continue

            risk_d = abs(entry - sl) / entry
            reward_d = abs(tp - entry) / entry

            # Forward-walk last 80 bars
            future = prices[-80:]
            hit_tp = hit_sl = False
            for fp in future:
                if sig.action == "BUY":
                    if fp <= sl: hit_sl = True; break
                    if fp >= tp: hit_tp = True; break
                else:
                    if fp >= sl: hit_sl = True; break
                    if fp <= tp: hit_tp = True; break

            if hit_tp:
                outcomes.append(reward_d)
            elif hit_sl:
                outcomes.append(-risk_d)
            else:
                final = future[-1]
                d = ((final - entry) / entry) if sig.action == "BUY" else ((entry - final) / entry)
                outcomes.append(d)

    if not outcomes:
        return None

    arr = np.array(outcomes)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    wr = len(wins) / len(arr)
    aw = float(np.mean(wins)) if len(wins) > 0 else 0
    al = float(np.mean(losses)) if len(losses) > 0 else 0
    spw = len(arr) / n_seeds
    exp = wr * aw + (1 - wr) * al

    return {
        "name": name,
        "n_trades": len(arr),
        "win_rate": wr,
        "avg_win": aw,
        "avg_loss": al,
        "expectancy": exp,
        "signals_per_window": spw,
        "rr": abs(aw / al) if al != 0 else 0,
    }


def run_mc(stats: dict, label: str, fee_per_side: float = KRAKEN_TAKER_FEE) -> dict:
    """Run Monte Carlo simulation using profiled stats."""
    wr = stats["win_rate"]
    aw = stats["avg_win"]
    al = abs(stats["avg_loss"])
    trades_per_day = max(stats["signals_per_window"] * 2, 0.1) * NUM_PAIRS

    rng = np.random.default_rng(42)
    final_bals = np.zeros(N_SIMS)
    max_dds = np.zeros(N_SIMS)
    total_trades = np.zeros(N_SIMS, dtype=int)
    total_wins = np.zeros(N_SIMS, dtype=int)
    total_fees = np.zeros(N_SIMS)

    for sim in range(N_SIMS):
        bal = STARTING_BALANCE
        peak = STARTING_BALANCE
        mdd = nt = nw = 0
        fees = 0.0

        for day in range(DAYS):
            if bal <= 20:
                break

            n_today = rng.poisson(trades_per_day)
            n_today = min(n_today, 6)

            for _ in range(n_today):
                is_win = rng.random() < wr
                if is_win:
                    pnl_pct = abs(rng.normal(aw, aw * 0.25))
                else:
                    pnl_pct = -abs(rng.normal(al, al * 0.25))

                sl_pct = al if al > 0 else 0.02
                risk_amt = bal * MAX_RISK_PER_TRADE
                notional = risk_amt / sl_pct
                notional = min(notional, bal * MAX_NOTIONAL_PCT)

                fee = notional * fee_per_side * 2  # entry + exit
                fees += fee
                pnl = notional * pnl_pct - fee
                bal += pnl
                nt += 1
                if pnl > 0:
                    nw += 1

                peak = max(peak, bal)
                if peak > 0:
                    dd = (peak - bal) / peak
                    mdd = max(mdd, dd)

        final_bals[sim] = bal
        max_dds[sim] = mdd * 100
        total_trades[sim] = nt
        total_wins[sim] = nw
        total_fees[sim] = fees

    pnls = final_bals - STARTING_BALANCE
    pnl_pcts = pnls / STARTING_BALANCE * 100
    active = total_trades > 0

    return {
        "label": label,
        "final_bals": final_bals,
        "pnls": pnls,
        "pnl_pcts": pnl_pcts,
        "max_dds": max_dds,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_fees": total_fees,
        "trades_per_day": trades_per_day,
        "profitable_pct": np.sum(pnls > 0) / N_SIMS * 100,
        "ruin_pct": np.sum(final_bals < 50) / N_SIMS * 100,
    }


def print_result(r: dict, s: dict, fee_per_side: float):
    """Pretty-print one strategy's results."""
    rt_fee = fee_per_side * 2
    print(f"\n  ┌─── {r['label']} ───")
    print(f"  │ Profile: {s['n_trades']} samples, {s['win_rate']*100:.0f}% WR, "
          f"+{s['avg_win']*100:.1f}%/-{abs(s['avg_loss'])*100:.1f}% "
          f"(1:{s['rr']:.1f} R:R)")
    print(f"  │ Expectancy: {s['expectancy']*100:+.2f}%/trade "
          f"(fee: -{rt_fee*100:.2f}% → net {(s['expectancy']-rt_fee)*100:+.2f}%)")
    print(f"  │ ~{r['trades_per_day']:.1f} trades/day across {NUM_PAIRS} pairs")
    print(f"  │")

    fb = r["final_bals"]
    pnls = r["pnls"]
    for pct, lbl in [(5, "Worst 5%"), (25, "25th"), (50, "◆ MEDIAN"), (75, "75th"), (95, "Best 95%")]:
        b = np.percentile(fb, pct)
        p = np.percentile(pnls, pct)
        pp = np.percentile(r["pnl_pcts"], pct)
        print(f"  │  {lbl:12s}  ${b:>8.2f}  ({p:+7.2f} / {pp:+5.1f}%)")

    print(f"  │  {'Average':12s}  ${np.mean(fb):>8.2f}  ({np.mean(pnls):+7.2f} / {np.mean(r['pnl_pcts']):+5.1f}%)")
    print(f"  │")
    print(f"  │  Profit chance:  {r['profitable_pct']:.1f}%")
    print(f"  │  Ruin chance:    {r['ruin_pct']:.1f}%")
    print(f"  │  Avg trades:     {np.mean(r['total_trades']):.0f} / {DAYS}d")
    print(f"  │  Avg fees:       ${np.mean(r['total_fees']):.2f} ({np.mean(r['total_fees'])/STARTING_BALANCE*100:.1f}% of ${STARTING_BALANCE:.0f})")
    print(f"  │  Avg max DD:     {np.mean(r['max_dds']):.1f}% (worst: {np.max(r['max_dds']):.1f}%)")
    print(f"  └{'─'*50}")


# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  $5,000 KRAKEN BOT FORECAST — Strategy Comparison")
print("=" * 65)

best_result = None
best_stats = None
best_fee = KRAKEN_TAKER_FEE

# Cache profiled stats to avoid re-profiling same strategy
profile_cache = {}

for strat_cfg in STRATEGIES_TO_TEST:
    strat_name = strat_cfg["name"]
    fee = strat_cfg["fee"]
    label = strat_cfg["label"]

    if strat_name not in profile_cache:
        print(f"\n  Profiling '{strat_name}'...", end=" ", flush=True)
        t0 = time.time()
        stats = profile_strategy(strat_name, N_PROFILE_SEEDS)
        if stats is None:
            print(f"SKIP — no signals generated")
            profile_cache[strat_name] = None
            continue
        print(f"done ({time.time()-t0:.1f}s, {stats['n_trades']} samples)")
        profile_cache[strat_name] = stats
    else:
        stats = profile_cache[strat_name]
        if stats is None:
            continue

    print(f"  Running {N_SIMS:,} Monte Carlo sims ({label})...", end=" ", flush=True)
    result = run_mc(stats, label, fee_per_side=fee)
    print("done")

    print_result(result, stats, fee)

    med_pnl = np.percentile(result["pnls"], 50)
    if best_result is None or med_pnl > np.percentile(best_result["pnls"], 50):
        best_result = result
        best_stats = stats
        best_fee = fee

# ── Winner + 12-month outlook ──────────────────────────────────────────
if best_result:
    winner = best_result["label"]
    fb = best_result["final_bals"]
    med_90 = np.percentile(best_result["pnl_pcts"], 50) / 100
    p25_90 = np.percentile(best_result["pnl_pcts"], 25) / 100
    p75_90 = np.percentile(best_result["pnl_pcts"], 75) / 100

    print(f"\n{'='*65}")
    print(f"  RECOMMENDED: {winner}")
    rt_fee = best_fee * 2
    print(f"  Net edge/trade: {(best_stats['expectancy']-rt_fee)*100:+.2f}% (after {rt_fee*100:.2f}% fees)")
    print(f"{'='*65}")

    print(f"\n  12-Month Projection (compounding quarterly, ${STARTING_BALANCE:.0f} start):")
    print(f"  {'':20s} {'Pessimistic':>12s} {'MEDIAN':>12s} {'Optimistic':>12s}")
    bl, bm, bh = STARTING_BALANCE, STARTING_BALANCE, STARTING_BALANCE
    for q in range(1, 5):
        bl *= (1 + p25_90); bm *= (1 + med_90); bh *= (1 + p75_90)
        print(f"    Month {q*3:2d}:          ${bl:>10.2f}  ${bm:>10.2f}  ${bh:>10.2f}")

    # Dollar breakdown
    med_profit = np.percentile(best_result["pnls"], 50)
    avg_fees = np.mean(best_result["total_fees"])
    avg_trades = np.mean(best_result["total_trades"])
    print(f"\n  Per-quarter breakdown (${STARTING_BALANCE:.0f} account):")
    print(f"    Trades:    ~{avg_trades:.0f}")
    print(f"    Fees:      ~${avg_fees:.2f}")
    print(f"    Median P&L: ${med_profit:+.2f}")
    if avg_trades > 0:
        print(f"    Per trade:  ${med_profit/avg_trades:+.3f}")

print(f"\n  ─── Reality Check ───")
print(f"  • ${STARTING_BALANCE:.0f} account with proper position sizing.")
print(f"  • Kraken's 0.52% round-trip fee still matters.")
print(f"  • Returns scale roughly linearly with capital.")
print(f"  • These are synthetic simulations, not predictions.")
print(f"  • Crypto is volatile — real drawdowns can be worse.")
print(f"  • SELL signals only close longs on Kraken (no shorting).")
print(f"{'='*65}\n")
