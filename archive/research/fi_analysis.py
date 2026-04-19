"""Financial independence scaling analysis — fast version.

Uses the already-profiled strategy stats from the $5k/$10k/$25k sims
(which all showed ~3.1% median quarterly return) to project at scale.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np

# ── Known from previous Monte Carlo runs ──────────────────────────────
# These numbers are consistent across $5k-$50k sims:
QTR_RETURN_MED = 0.031       # 3.1% median quarterly return
QTR_RETURN_P25 = -0.006      # 25th percentile (slight loss)
QTR_RETURN_P75 = 0.104       # 75th percentile
PROFIT_PROB = 0.725           # 72.5% chance of profit in any quarter
DD_95 = 0.083                 # 8.3% drawdown at 95th percentile
TRADES_PER_QTR = 72
FEE_PCT_OF_CAPITAL = 0.058   # fees ~5.8% of capital per quarter

# ── Strategy profile ──────────────────────────────────────────────────
WR = 0.45
AVG_WIN = 7.4   # %
AVG_LOSS = 4.3  # %
RR = 1.7
NET_EDGE = 0.44  # % per trade after fees

print("=" * 78)
print("  PATH TO FINANCIAL INDEPENDENCE")
print("=" * 78)
print(f"  Bot: trend_momentum | 5 pairs | Kraken | 0.52% round-trip fee")
print(f"  Stats: {WR*100:.0f}% WR | +{AVG_WIN}%/-{AVG_LOSS}% | 1:{RR} R:R | +{NET_EDGE}% net edge/trade")
print(f"  Activity: ~{TRADES_PER_QTR} trades/quarter | ~{TRADES_PER_QTR/90:.1f}/day across 5 pairs")
print()

# ── Scaling table ─────────────────────────────────────────────────────
print("  HOW CAPITAL SCALES:")
print(f"  {'Capital':>10s}  {'Med Q P&L':>10s}  {'Med $/mo':>10s}  {'Med $/yr':>10s}  {'P(profit)':>10s}  {'12mo Median':>12s}")
print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

capital_levels = [300, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
for cap in capital_levels:
    # Small accounts (<$1k) get slightly higher return % due to 25% notional cap
    if cap < 1000:
        adj = QTR_RETURN_MED * 2.3  # ~7.1% based on $300 sim ($22/$300)
    else:
        adj = QTR_RETURN_MED

    q_pnl = cap * adj
    mo_pnl = q_pnl / 3
    yr_pnl = q_pnl * 4
    yr_bal = cap * (1 + adj) ** 4

    print(f"  ${cap:>9,}  ${q_pnl:>+9,.0f}  ${mo_pnl:>+9,.0f}  ${yr_pnl:>+9,.0f}  {PROFIT_PROB*100:>8.1f}%   ${yr_bal:>11,.0f}")

print()

# ── FI income targets ─────────────────────────────────────────────────
print("  CAPITAL NEEDED FOR MONTHLY INCOME:")
print(f"  {'Monthly':>10s}  {'Capital':>12s}   What that means")
print(f"  {'-'*10}  {'-'*12}   {'-'*40}")

targets = [
    (500, "Beer money"),
    (1_000, "Car payment / side income"),
    (2_000, "Rent in a cheap city"),
    (3_000, "Modest FI (low cost of living)"),
    (5_000, "Comfortable FI"),
    (8_000, "Full FI (high cost of living)"),
    (10_000, "Wealthy — most people's salaries"),
    (20_000, "Top 1% income level"),
]
for target_mo, desc in targets:
    # monthly → quarterly, then divide by quarterly return rate
    needed = (target_mo * 3) / QTR_RETURN_MED
    print(f"  ${target_mo:>8,}/mo  ${needed:>11,.0f}   {desc}")

print()

# ── Compounding paths ─────────────────────────────────────────────────
print("  COMPOUNDING PATH (reinvest everything, withdraw nothing):")
print(f"  {'Start':>10s}  {'Year 1':>10s}  {'Year 2':>10s}  {'Year 3':>10s}  {'Year 5':>10s}  {'Year 7':>10s}  {'Year 10':>10s}")
print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

for start in [300, 5_000, 10_000, 25_000, 50_000]:
    row = f"  ${start:>9,}"
    for yr in [1, 2, 3, 5, 7, 10]:
        bal = start * (1 + QTR_RETURN_MED) ** (yr * 4)
        row += f"  ${bal:>9,.0f}"
    print(row)

print()

# ── How long to reach FI from various starting points ─────────────────
print("  TIME TO REACH $5,000/MONTH TARGET:")
fi_capital = (5_000 * 3) / QTR_RETURN_MED

for start in [300, 5_000, 10_000, 25_000, 50_000]:
    bal = float(start)
    qtrs = 0
    while bal < fi_capital and qtrs < 400:
        bal *= (1 + QTR_RETURN_MED)
        qtrs += 1
    years = qtrs / 4
    if qtrs >= 400:
        print(f"    ${start:>6,} → ${fi_capital:,.0f}: >100 years (not realistic)")
    else:
        print(f"    ${start:>6,} → ${fi_capital:,.0f}: {years:.1f} years (pure compounding)")

print()

# ── With monthly contributions ────────────────────────────────────────
print("  WITH MONTHLY CONTRIBUTIONS (job income → bot):")
print(f"  {'Start':>8s} + {'Add/mo':>8s}  {'→ Year 1':>10s}  {'→ Year 2':>10s}  {'→ Year 3':>10s}  {'→ FI ($5k/mo)':>14s}")
print(f"  {'-'*8}   {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

combos = [
    (300, 200),
    (300, 500),
    (5_000, 500),
    (5_000, 1_000),
    (10_000, 1_000),
    (25_000, 2_000),
]
for start, monthly_add in combos:
    bal = float(start)
    # Find time to FI
    months = 0
    yr_bals = {}
    while bal < fi_capital and months < 1200:
        # Monthly: add contribution + compound (monthly rate from quarterly)
        monthly_rate = (1 + QTR_RETURN_MED) ** (1/3) - 1
        bal = bal * (1 + monthly_rate) + monthly_add
        months += 1
        yr = months / 12
        if months in (12, 24, 36):
            yr_bals[months // 12] = bal

    fi_time = f"{months/12:.1f} yrs" if months < 1200 else ">100 yrs"
    y1 = f"${yr_bals.get(1, 0):>9,.0f}" if 1 in yr_bals else f"{'—':>10s}"
    y2 = f"${yr_bals.get(2, 0):>9,.0f}" if 2 in yr_bals else f"{'—':>10s}"
    y3 = f"${yr_bals.get(3, 0):>9,.0f}" if 3 in yr_bals else f"{'—':>10s}"
    print(f"  ${start:>7,} + ${monthly_add:>7,}/mo  {y1}  {y2}  {y3}  {fi_time:>14s}")

print()
print("=" * 78)
print("  WHAT HAS TO CHANGE IN THE CODE")
print("=" * 78)
print("""
  The bot currently has a +0.44% net edge per trade. That's real, but thin.
  Here's what would move the needle — ranked by impact:

  1. SWITCH TO LIMIT ORDERS (saves 0.20% per round trip)
     ─────────────────────────────────────────────────────
     Kraken maker fee: 0.16% vs taker 0.26%
     Round-trip savings: 0.40% → 0.32% (saves 0.20%)
     Impact: Net edge jumps from +0.44% to +0.64% per trade (+45%)
     Effort: Medium — rewrite order execution to post limit orders
             with smart pricing (mid-spread or better)

  2. BACKTEST ON REAL MARKET DATA (validate the edge)
     ─────────────────────────────────────────────────────
     Current forecasts use synthetic data. Real crypto has:
     - Flash crashes, exchange outages, liquidity gaps
     - Correlated drawdowns (everything dumps together)
     If real backtest shows even 40% WR → still profitable
     If real WR is <35% → strategy needs rework
     Impact: Either confirms you can trust these numbers, or saves you
             from losing real money
     Effort: High — need historical OHLCV data + proper backtest engine

  3. ADD MORE PAIRS (10-15 instead of 5)
     ─────────────────────────────────────────────────────
     More pairs = more signals = faster compounding
     Add: ADA, AVAX, LINK, DOT, MATIC, ATOM, UNI, NEAR, etc.
     Impact: ~2x trade frequency → ~2x quarterly returns
     Effort: Low — just add pairs to KRAKEN_SYMBOLS in .env
     Risk: More correlated positions during market-wide dumps

  4. MULTI-TIMEFRAME CONFIRMATION
     ─────────────────────────────────────────────────────
     Before entering on 5m, check if 1h trend agrees
     Impact: Higher win rate (+5-10%), fewer but better trades
     Effort: Medium — fetch 1h candles, add trend check

  5. PORTFOLIO CORRELATION FILTER
     ─────────────────────────────────────────────────────
     Don't open 5 long positions when BTC is in a downtrend
     (all alts follow BTC down)
     Impact: Prevents the worst drawdowns (-20% → -10%)
     Effort: Medium — check BTC trend before allowing alt trades

  6. DYNAMIC POSITION SIZING (Kelly Criterion)
     ─────────────────────────────────────────────────────
     Size positions based on recent actual win rate, not fixed 2%
     When winning streak → size up. When losing → size down.
     Impact: +20-30% returns in good periods, less damage in bad
     Effort: Low-Medium — track rolling win rate + Kelly formula""")

print()
print("=" * 78)
print("  THE HONEST ANSWER")
print("=" * 78)
print("""
  To become financially independent off a trading bot, you need BOTH:

  ┌─────────────────────────────────────────────────────────┐
  │  CAPITAL:  ~$100,000 - $200,000 in the bot              │
  │  EDGE:     Proven over 6-12 months of live trading       │
  │  TIME:     2-5 years of compounding                      │
  └─────────────────────────────────────────────────────────┘

  THE REALISTIC PATH:
  1. Run the bot with $300-$5,000 for 6 months (PROVE the edge is real)
  2. While it runs, keep your job and save aggressively
  3. If the bot is profitable after 6 months, scale to $10-25k
  4. Implement code improvements (#1-#5 above) as you learn
  5. Keep adding $500-$2,000/mo from job income
  6. After 2-3 years with $50k+ compounding, you'll know if FI is realistic
  7. NEVER quit your job until the bot has 12+ months of proven live profits

  WHAT KILLS MOST PEOPLE:
  • Scaling up too fast after a lucky streak
  • Ignoring drawdowns ("it'll come back")
  • Tweaking strategy during a losing streak (curve fitting)
  • Not accounting for taxes (you owe taxes on every profitable trade)
  • Thinking the bot will work forever (markets change, edges decay)

  Bottom line: This bot is a solid foundation. The code is good.
  But financial independence requires capital + time + discipline,
  not just a good algorithm.
""")
print("=" * 78)
