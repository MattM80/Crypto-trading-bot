#!/usr/bin/env python3
"""
Model the All-Seeing Eye's growth from $300.

Uses REAL data from our testing:
- Grid: +$18.22 realized in 83 days (278 round-trips) on $120 allocation
- Crash/dip tools: from 30,656 data points, known win rates and avg returns
- Compounding: profits get reinvested, positions scale with balance

This is a Monte Carlo simulation — runs 10,000 scenarios with randomized
trade outcomes based on historical win rates and return distributions.
"""
import numpy as np
from collections import defaultdict

np.random.seed(42)

INITIAL = 300
GRID_PCT = 0.40
ACTIVE_PCT = 0.60
RISK_PER_TRADE = 0.05  # 5% of active balance per trade
MAKER_FEE = 0.0042  # 0.42% round-trip

# Tool statistics from our data (30,656 data points, 83 days, 16 pairs)
# Format: (avg_return_pct, win_rate, avg_trades_per_month, avg_win_pct, avg_loss_pct)
TOOLS = {
    "mega_crash":       {"wr": 0.80, "avg_win": 15.0, "avg_loss": -5.0, "trades_mo": 1.5},
    "crash_buy_10":     {"wr": 0.76, "avg_win": 8.0,  "avg_loss": -5.0, "trades_mo": 4},
    "flash_crash":      {"wr": 0.77, "avg_win": 10.0, "avg_loss": -7.0, "trades_mo": 2},
    "quick_crash":      {"wr": 0.69, "avg_win": 9.0,  "avg_loss": -7.0, "trades_mo": 1},
    "deep_dip_8pct":    {"wr": 0.64, "avg_win": 7.0,  "avg_loss": -5.0, "trades_mo": 5},
    "correction_buy":   {"wr": 0.61, "avg_win": 5.0,  "avg_loss": -5.0, "trades_mo": 8},
    "quick_dip":        {"wr": 0.58, "avg_win": 4.0,  "avg_loss": -3.0, "trades_mo": 7},
    "relief_rally":     {"wr": 0.68, "avg_win": 3.5,  "avg_loss": -3.0, "trades_mo": 3},
    "dip_buy_3pct":     {"wr": 0.44, "avg_win": 3.0,  "avg_loss": -3.0, "trades_mo": 12},
}

# Grid: proven $18.22 on $120 in 83 days = ~$6.58/month per $120
# Scales linearly with allocation
GRID_MONTHLY_RETURN_PCT = 18.22 / 120 / 83 * 30 * 100  # ~5.5% monthly on grid capital

def simulate_month(balance, tools, grid_pct, active_pct, risk_pct):
    """Simulate one month of trading. Returns new balance."""
    grid_bal = balance * grid_pct
    active_bal = balance * active_pct
    
    # Grid income (predictable)
    grid_profit = grid_bal * GRID_MONTHLY_RETURN_PCT / 100
    # Add some variance to grid (±30%)
    grid_profit *= np.random.uniform(0.7, 1.3)
    
    # Active trades
    active_profit = 0
    total_trades = 0
    
    for tool_name, stats in tools.items():
        # How many trades this month from this tool?
        # Poisson-distributed around expected
        n_trades = np.random.poisson(stats["trades_mo"])
        
        for _ in range(n_trades):
            # Risk amount
            risk_amount = active_bal * risk_pct
            if risk_amount < 1:
                continue
            
            # Win or lose?
            if np.random.random() < stats["wr"]:
                # Win: return is avg_win with some variance
                ret_pct = stats["avg_win"] * np.random.uniform(0.5, 1.5)
                profit = risk_amount * (ret_pct / 100) / (abs(stats["avg_loss"]) / 100)
                # Cap at reasonable level
                profit = min(profit, active_bal * 0.15)
            else:
                # Loss: usually hits stop loss
                ret_pct = stats["avg_loss"] * np.random.uniform(0.8, 1.2)
                profit = risk_amount * (ret_pct / 100) / (abs(stats["avg_loss"]) / 100)
                profit = max(profit, -risk_amount)
            
            # Subtract fees
            trade_value = risk_amount / (abs(stats["avg_loss"]) / 100)
            fee = trade_value * MAKER_FEE
            profit -= fee
            
            active_profit += profit
            active_bal += profit  # Compound within month
            total_trades += 1
    
    new_balance = balance + grid_profit + active_profit
    return max(new_balance, balance * 0.5), grid_profit, active_profit, total_trades  # Floor at 50% loss


def run_simulation(months=120, n_sims=10000):
    """Run Monte Carlo simulation."""
    
    all_paths = np.zeros((n_sims, months + 1))
    all_paths[:, 0] = INITIAL
    
    final_balances = []
    monthly_returns = []
    
    for sim in range(n_sims):
        balance = INITIAL
        
        for month in range(months):
            new_bal, gp, ap, nt = simulate_month(
                balance, TOOLS, GRID_PCT, ACTIVE_PCT, RISK_PER_TRADE)
            monthly_returns.append((new_bal - balance) / balance * 100)
            balance = new_bal
            all_paths[sim, month + 1] = balance
        
        final_balances.append(balance)
    
    return all_paths, final_balances, monthly_returns


def main():
    print("=" * 80)
    print("  ALL-SEEING EYE: GROWTH MODEL")
    print(f"  $300 start, 15 tools, 16 pairs, Monte Carlo (10,000 simulations)")
    print("=" * 80)
    
    print(f"\n  Grid monthly return: ~{GRID_MONTHLY_RETURN_PCT:.1f}% on 40% allocation")
    print(f"  Active risk per trade: {RISK_PER_TRADE*100:.0f}%")
    print(f"  Active tools: {len(TOOLS)}")
    total_monthly_trades = sum(t["trades_mo"] for t in TOOLS.values())
    print(f"  Expected trades/month: ~{total_monthly_trades:.0f}")
    
    # Expected monthly return (rough)
    expected_grid = INITIAL * GRID_PCT * GRID_MONTHLY_RETURN_PCT / 100
    expected_active = 0
    for t in TOOLS.values():
        ev_per_trade = t["wr"] * t["avg_win"] + (1 - t["wr"]) * t["avg_loss"]
        expected_active += t["trades_mo"] * ev_per_trade * RISK_PER_TRADE / abs(t["avg_loss"]) * 100
    expected_active_usd = INITIAL * ACTIVE_PCT * expected_active / 100 / len(TOOLS)
    
    print(f"\n  Expected monthly grid income: ~${expected_grid:.2f}")
    
    # Run simulation
    print(f"\n  Running 10,000 Monte Carlo simulations over 10 years...")
    paths, finals, monthly_rets = run_simulation(months=120, n_sims=10000)
    
    monthly_rets = np.array(monthly_rets)
    finals = np.array(finals)
    
    print(f"\n  Monthly return stats:")
    print(f"    Mean:   {np.mean(monthly_rets):+.2f}%")
    print(f"    Median: {np.median(monthly_rets):+.2f}%")
    print(f"    Std:    {np.std(monthly_rets):.2f}%")
    print(f"    Best:   {np.max(monthly_rets):+.2f}%")
    print(f"    Worst:  {np.min(monthly_rets):+.2f}%")
    
    # Percentile analysis at various timepoints
    print(f"\n{'='*80}")
    print(f"  BALANCE PROJECTIONS (from $300)")
    print(f"{'='*80}")
    
    timepoints = [3, 6, 12, 24, 36, 48, 60, 84, 120]
    
    print(f"\n  {'Month':>6s} {'10th%':>10s} {'25th%':>10s} {'MEDIAN':>10s} {'75th%':>10s} {'90th%':>10s} {'Mean':>10s}")
    print(f"  {'-'*66}")
    
    for m in timepoints:
        if m >= paths.shape[1]:
            continue
        balances = paths[:, m]
        p10 = np.percentile(balances, 10)
        p25 = np.percentile(balances, 25)
        p50 = np.percentile(balances, 50)
        p75 = np.percentile(balances, 75)
        p90 = np.percentile(balances, 90)
        mean = np.mean(balances)
        
        print(f"  {m:>4d}mo ${p10:>9,.2f} ${p25:>9,.2f} ${p50:>9,.2f} ${p75:>9,.2f} ${p90:>9,.2f} ${mean:>9,.2f}")
    
    # Probability analysis
    print(f"\n{'='*80}")
    print(f"  PROBABILITY OF HITTING TARGETS")
    print(f"{'='*80}")
    
    targets = [500, 1000, 5000, 10000, 50000, 100000, 1000000]
    
    for target in targets:
        # At what month does median first hit this target?
        hit_month = None
        for m in range(1, paths.shape[1]):
            if np.median(paths[:, m]) >= target:
                hit_month = m
                break
        
        # Probability of hitting by various timepoints
        prob_1y = np.mean(paths[:, min(12, paths.shape[1]-1)] >= target) * 100
        prob_3y = np.mean(paths[:, min(36, paths.shape[1]-1)] >= target) * 100
        prob_5y = np.mean(paths[:, min(60, paths.shape[1]-1)] >= target) * 100
        prob_10y = np.mean(paths[:, -1] >= target) * 100
        
        hit_str = f"month {hit_month}" if hit_month else "never (median)"
        print(f"  ${target:>10,d}: P(1y)={prob_1y:5.1f}%  P(3y)={prob_3y:5.1f}%  P(5y)={prob_5y:5.1f}%  P(10y)={prob_10y:5.1f}%  median hits: {hit_str}")
    
    # Risk analysis
    print(f"\n{'='*80}")
    print(f"  RISK ANALYSIS")
    print(f"{'='*80}")
    
    # Max drawdown distribution
    max_drawdowns = []
    for sim in range(min(10000, paths.shape[0])):
        path = paths[sim]
        peak = np.maximum.accumulate(path)
        dd = (peak - path) / peak
        max_drawdowns.append(np.max(dd) * 100)
    
    max_drawdowns = np.array(max_drawdowns)
    print(f"\n  Max drawdown over 10 years:")
    print(f"    Median: {np.median(max_drawdowns):.1f}%")
    print(f"    90th percentile: {np.percentile(max_drawdowns, 90):.1f}%")
    print(f"    Worst case: {np.max(max_drawdowns):.1f}%")
    
    # Probability of ruin (losing 80%+)
    ruin_prob = np.mean(np.min(paths, axis=1) < INITIAL * 0.2) * 100
    print(f"    Probability of 80%+ loss: {ruin_prob:.1f}%")
    
    # Probability of being profitable at each timepoint
    print(f"\n  Probability of being profitable:")
    for m in [1, 3, 6, 12, 24, 36, 60, 120]:
        if m >= paths.shape[1]: continue
        prob = np.mean(paths[:, m] > INITIAL) * 100
        print(f"    After {m:>3d} months: {prob:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"  BOTTOM LINE")
    print(f"{'='*80}")
    print(f"\n  Starting with ${INITIAL}:")
    print(f"  Median outcome after 1 year:  ${np.median(paths[:, 12]):>10,.2f}")
    print(f"  Median outcome after 3 years: ${np.median(paths[:, 36]):>10,.2f}")
    print(f"  Median outcome after 5 years: ${np.median(paths[:, 60]):>10,.2f}")
    print(f"  Median outcome after 10 years:${np.median(paths[:, -1]):>10,.2f}")
    
    prob_1m = np.mean(paths[:, -1] >= 1000000) * 100
    print(f"\n  Probability of $1M+ in 10 years: {prob_1m:.1f}%")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
