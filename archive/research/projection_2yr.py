#!/usr/bin/env python3
"""
2-Year Monte Carlo Projection for the Final Bot
Uses REAL OOS-validated numbers + all 8 upgrades
"""
import numpy as np
np.random.seed(42)

N_SIMS = 50000
MONTHS = 24
INITIAL = 300

# ═══════════════════════════════════════════════════════════
# GRID ENGINE — validated: +136% in 12mo on 16 pairs
# Now 40 pairs = 2.5x opportunity surface
# Maker fees: 0.50% RT (was 0.80% taker)
# Fee savings boost: +0.30% per round trip
# ═══════════════════════════════════════════════════════════

# Base: 139 round trips / 6mo on 16 pairs = ~23/mo
# 40 pairs = ~58 round trips/month (scaled by 2.5x)
# Avg profit per RT: 1.5% TP - 0.50% fees = 1.0% net (was 0.98% at old fees)
# Grid capital fraction varies by regime (0.35-0.65, avg ~0.55)
GRID_RT_PER_MONTH_MEAN = 58
GRID_RT_PER_MONTH_STD = 15
GRID_PROFIT_PER_RT_MEAN = 0.010  # 1.0% net after maker fees
GRID_PROFIT_PER_RT_STD = 0.003

# ═══════════════════════════════════════════════════════════
# ACTIVE TRADING — 30 validated tools across 40 pairs
# Tier 1 (10 tools) at 2x margin, Tier 2 (20 tools) at 1x
# Maker entry fees: 0.25% (was 0.40%)
# ═══════════════════════════════════════════════════════════

# Tier 1 tools (2x margin, WR 58-73%, validated edge)
TIER1 = {
    'crash_buy':        {'wr': 0.603, 'avg_win': 0.038, 'avg_loss': -0.025, 'trades_mo': 3.5, 'lev': 2},
    'volatile_oversold':{'wr': 0.738, 'avg_win': 0.041, 'avg_loss': -0.030, 'trades_mo': 1.2, 'lev': 2},
    'crash_neg_ac':     {'wr': 0.589, 'avg_win': 0.025, 'avg_loss': -0.020, 'trades_mo': 1.0, 'lev': 2},
    'mega_crash':       {'wr': 0.525, 'avg_win': 0.027, 'avg_loss': -0.020, 'trades_mo': 4.0, 'lev': 2},
    'blood_in_streets': {'wr': 0.617, 'avg_win': 0.022, 'avg_loss': -0.018, 'trades_mo': 0.6, 'lev': 2},
    'quick_crash':      {'wr': 0.591, 'avg_win': 0.020, 'avg_loss': -0.015, 'trades_mo': 3.0, 'lev': 2},
    'crash_mean_revert':{'wr': 0.549, 'avg_win': 0.016, 'avg_loss': -0.015, 'trades_mo': 1.2, 'lev': 2},
    'mega_pump_sell_t1':{'wr': 0.587, 'avg_win': 0.012, 'avg_loss': -0.010, 'trades_mo': 6.5, 'lev': 2},
    'rsi_pump_8h':      {'wr': 0.603, 'avg_win': 0.011, 'avg_loss': -0.010, 'trades_mo': 5.0, 'lev': 2},
    'vpin_dip':         {'wr': 0.535, 'avg_win': 0.019, 'avg_loss': -0.015, 'trades_mo': 1.9, 'lev': 2},
}

# Tier 2 tools (1x, WR 50-58%, smaller edge)
TIER2 = {
    'flash_crash':      {'wr': 0.558, 'avg_win': 0.010, 'avg_loss': -0.010, 'trades_mo': 4.8},
    'deep_dip_8h':      {'wr': 0.548, 'avg_win': 0.005, 'avg_loss': -0.005, 'trades_mo': 3.3},
    'entropy_dip':      {'wr': 0.528, 'avg_win': 0.009, 'avg_loss': -0.008, 'trades_mo': 2.8},
    'vpin_toxic':       {'wr': 0.538, 'avg_win': 0.009, 'avg_loss': -0.008, 'trades_mo': 0.8},
    'market_panic_70':  {'wr': 0.590, 'avg_win': 0.015, 'avg_loss': -0.012, 'trades_mo': 0.9},
    'btc_alt_spread':   {'wr': 0.552, 'avg_win': 0.009, 'avg_loss': -0.008, 'trades_mo': 1.3},
    'quick_dip':        {'wr': 0.555, 'avg_win': 0.003, 'avg_loss': -0.003, 'trades_mo': 15.0},
    'greed_short_t2':   {'wr': 0.585, 'avg_win': 0.006, 'avg_loss': -0.006, 'trades_mo': 28.0},
    'thursday_short':   {'wr': 0.579, 'avg_win': 0.006, 'avg_loss': -0.005, 'trades_mo': 35.0},
    'mega_pump_sell_t2':{'wr': 0.550, 'avg_win': 0.004, 'avg_loss': -0.005, 'trades_mo': 14.0},
    'falling_wedge':    {'wr': 0.567, 'avg_win': 0.012, 'avg_loss': -0.008, 'trades_mo': 28.0},
    'distribution_short':{'wr': 0.534, 'avg_win': 0.004, 'avg_loss': -0.004, 'trades_mo': 99.0},
    'late_us_short':    {'wr': 0.529, 'avg_win': 0.003, 'avg_loss': -0.003, 'trades_mo': 13.0},
    'rsi_pump_12h':     {'wr': 0.549, 'avg_win': 0.003, 'avg_loss': -0.004, 'trades_mo': 16.0},
    'ema_cross_short':  {'wr': 0.532, 'avg_win': 0.003, 'avg_loss': -0.003, 'trades_mo': 250.0},
    'month_start_long': {'wr': 0.539, 'avg_win': 0.014, 'avg_loss': -0.010, 'trades_mo': 28.0},
    'rsi_pump_fat_tail':{'wr': 0.589, 'avg_win': 0.001, 'avg_loss': -0.005, 'trades_mo': 1.6},
    'entropy_short':    {'wr': 0.548, 'avg_win': 0.001, 'avg_loss': -0.003, 'trades_mo': 16.0},
    'alt_btc_revert_t3':{'wr': 0.564, 'avg_win': 0.001, 'avg_loss': -0.003, 'trades_mo': 53.0},
    'dip_buy_5pct':     {'wr': 0.527, 'avg_win': 0.002, 'avg_loss': -0.003, 'trades_mo': 29.0},
}

# Scale trade counts for 40 pairs (validated on 16)
PAIR_SCALE = 40 / 16  # 2.5x

# Margin cost for Tier 1: 0.02% open + 0.02%/4h * avg 12h hold = 0.08% total
MARGIN_COST = 0.0008
# Fee savings from maker orders (already in the return numbers above which used taker)
# Maker entry saves 0.15% per trade (0.25% vs 0.40%)
MAKER_SAVINGS = 0.0015

def sim_month(balance, regime='mixed'):
    """Simulate one month. Returns new balance."""
    # Dynamic allocation based on regime
    if regime == 'fear':    grid_pct, active_pct = 0.40, 0.60
    elif regime == 'greed': grid_pct, active_pct = 0.45, 0.55
    else:                   grid_pct, active_pct = 0.60, 0.40
    
    grid_bal = balance * grid_pct
    active_bal = balance * active_pct
    
    # Grid income
    n_rt = max(0, int(np.random.normal(GRID_RT_PER_MONTH_MEAN, GRID_RT_PER_MONTH_STD)))
    grid_profit = 0
    for _ in range(n_rt):
        rt_profit_pct = np.random.normal(GRID_PROFIT_PER_RT_MEAN, GRID_PROFIT_PER_RT_STD)
        per_pair_alloc = grid_bal / 40
        grid_profit += per_pair_alloc * rt_profit_pct
    
    # Active trades — Tier 1 (2x leverage)
    active_profit = 0
    risk_per_trade = 0.05  # 5% of active balance
    
    for tool, stats in TIER1.items():
        n_trades = np.random.poisson(max(1, stats['trades_mo'] * PAIR_SCALE * 0.4))  # 0.4 = not all pairs signal
        for _ in range(n_trades):
            trade_size = active_bal * risk_per_trade
            leverage = stats['lev']
            if np.random.random() < stats['wr']:
                ret = abs(np.random.normal(stats['avg_win'], stats['avg_win']*0.3))
            else:
                ret = -abs(np.random.normal(abs(stats['avg_loss']), abs(stats['avg_loss'])*0.3))
            # Add maker savings, subtract margin cost
            ret += MAKER_SAVINGS - MARGIN_COST
            active_profit += trade_size * ret * leverage
    
    # Active trades — Tier 2 (1x)
    for tool, stats in TIER2.items():
        # Cap high-frequency tools to avoid unrealistic counts
        scaled = min(stats['trades_mo'] * PAIR_SCALE * 0.3, 50)
        n_trades = np.random.poisson(max(1, scaled))
        n_trades = min(n_trades, 60)  # Hard cap
        for _ in range(n_trades):
            trade_size = active_bal * risk_per_trade * 0.5  # Half size for Tier 2
            if np.random.random() < stats['wr']:
                ret = abs(np.random.normal(stats['avg_win'], stats['avg_win']*0.3))
            else:
                ret = -abs(np.random.normal(abs(stats['avg_loss']), abs(stats['avg_loss'])*0.3))
            ret += MAKER_SAVINGS  # Maker savings
            active_profit += trade_size * ret
    
    return balance + grid_profit + active_profit

# Run simulations
results = np.zeros((N_SIMS, MONTHS+1))
results[:, 0] = INITIAL

# Regime sequence: random mix weighted toward reality
# Crypto: ~40% chop, ~25% bull, ~20% bear, ~15% extreme
regime_weights = ['mixed']*40 + ['greed']*25 + ['fear']*20 + ['mixed']*15

for sim in range(N_SIMS):
    bal = INITIAL
    for m in range(MONTHS):
        regime = regime_weights[np.random.randint(len(regime_weights))]
        bal = sim_month(bal, regime)
        bal = max(bal, 10)  # Floor at $10 (can't go negative)
        results[sim, m+1] = bal

# Results
final = results[:, -1]
yr1 = results[:, 12]

print("═"*70)
print("  2-YEAR MONTE CARLO PROJECTION — FINAL BOT")
print("  50,000 simulations | $300 starting balance")
print("═"*70)

print(f"\n  {'':40} {'Year 1':>12} {'Year 2':>12}")
print(f"  {'-'*64}")

pcts = [5, 10, 25, 50, 75, 90, 95]
labels = ['Worst 5%', 'Conservative (10th)', '25th percentile', 
          'MEDIAN (50th)', '75th percentile', 'Optimistic (90th)', 'Best 5%']

for p, label in zip(pcts, labels):
    y1 = np.percentile(yr1, p)
    y2 = np.percentile(final, p)
    print(f"  {label:<40} ${y1:>10,.0f}  ${y2:>10,.0f}")

print(f"\n  {'Mean':40} ${np.mean(yr1):>10,.0f}  ${np.mean(final):>10,.0f}")
print(f"  {'Mean monthly return':40} {((np.mean(yr1)/INITIAL)**(1/12)-1)*100:>10.1f}%  {((np.mean(final)/INITIAL)**(1/24)-1)*100:>10.1f}%")

# Risk metrics
bust = np.sum(final < 100) / N_SIMS * 100
double = np.sum(final > 600) / N_SIMS * 100
triple = np.sum(final > 900) / N_SIMS * 100
fivex = np.sum(final > 1500) / N_SIMS * 100
tenx = np.sum(final > 3000) / N_SIMS * 100

print(f"\n  RISK / REWARD ODDS:")
print(f"  {'Probability of losing >66% (below $100)':40} {bust:>10.1f}%")
print(f"  {'Probability of 2x ($600+)':40} {double:>10.1f}%")
print(f"  {'Probability of 3x ($900+)':40} {triple:>10.1f}%")
print(f"  {'Probability of 5x ($1,500+)':40} {fivex:>10.1f}%")
print(f"  {'Probability of 10x ($3,000+)':40} {tenx:>10.1f}%")

# Max drawdown analysis
max_dd = np.zeros(N_SIMS)
for sim in range(N_SIMS):
    running_max = np.maximum.accumulate(results[sim])
    dd = (results[sim] - running_max) / running_max
    max_dd[sim] = np.min(dd)

print(f"\n  DRAWDOWN:")
print(f"  {'Median max drawdown':40} {np.median(max_dd)*100:>10.1f}%")
print(f"  {'Worst 10% max drawdown':40} {np.percentile(max_dd, 10)*100:>10.1f}%")
print(f"  {'Worst 5% max drawdown':40} {np.percentile(max_dd, 5)*100:>10.1f}%")

print(f"\n{'═'*70}")
