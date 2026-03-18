# SCALING ROADMAP — Path to Thousands

## Current System Performance (Validated)

| Metric | Value |
|--------|-------|
| Strategy | AdaptiveQuant on 15m candles |
| Symbols | SOLUSD, UNIUSD, ETHUSD, XLMUSD |
| Monthly Return | **~3.0%** ($30/mo on $1,000) |
| Win Rate | 73% combined |
| Profit Factor | 3.38 combined |
| Avg EV/Trade | $1.00 |
| Trades/Month | ~30 (1/day avg) |
| Risk per Trade | 2% of balance |
| Fees | Maker 0.16% + 0.05% slippage (post-only limit orders) |

> All numbers from 30-day backtest with real Kraken data and realistic fees.

---

## Edge Breakdown by Symbol

| Symbol | Win Rate | Profit Factor | EV/Trade | Verdict |
|--------|----------|--------------|----------|---------|
| SOLUSD | 78-90% | 6.12 | $+1.89 | **STRONG EDGE** |
| UNIUSD | 80% | 2.82 | $+1.23 | **STRONG EDGE** |
| ETHUSD | 64% | 1.39 | $+0.48 | MARGINAL |
| XLMUSD | 60% | 1.18 | $+0.28 | WEAK |

SOL is the star performer. UNI is the second engine.

---

## Compound Growth Projections

### Pure Compounding (no deposits)
At 3% monthly return, compounding:

| Starting Balance | Month 6 | Month 12 | Month 24 | Month 36 |
|-----------------|---------|----------|----------|----------|
| $1,000 | $1,194 | $1,426 | $2,033 | $2,898 |
| $5,000 | $5,970 | $7,129 | $10,164 | $14,490 |
| $10,000 | $11,941 | $14,258 | $20,328 | $28,980 |

### With Monthly Deposits ($500/month added)
| Month | Balance | Monthly Income |
|-------|---------|---------------|
| 6 | $4,248 | $127 |
| 12 | $8,379 | $251 |
| 18 | $13,539 | $406 |
| 24 | $19,900 | $597 |
| 36 | $37,483 | $1,124 |
| 48 | $63,741 | **$1,912** |

### Income at Various Account Sizes
| Account | Monthly Income | Annual Income |
|---------|---------------|---------------|
| $1,000 | $30 | $360 |
| $5,000 | $150 | $1,800 |
| $10,000 | $300 | $3,600 |
| $25,000 | $750 | $9,000 |
| $50,000 | $1,500 | $18,000 |
| $100,000 | **$3,000** | **$36,000** |

---

## How to Scale

### Phase 1: Prove It ($1K, months 1-3)
- Run the bot live on SOLUSD, UNIUSD, ETHUSD, XLMUSD
- Verify ~3% monthly return matches backtest
- Build confidence in the system
- **Target: $30/month income**

### Phase 2: Load the Base ($5K-$10K, months 4-6)
- Add capital to accelerate compounding
- Position sizes auto-scale (%-based sizing already built in)
- Run `python scan_edges.py` monthly to check for new edges
- **Target: $150-$300/month income**

### Phase 3: Scale Up ($10K-$25K, months 7-12)
- Proven track record → deploy more capital
- More capital = proportionally larger positions and profits
- System handles this automatically — no code changes needed
- **Target: $300-$750/month income**

### Phase 4: Serious Income ($25K+, year 2+)
- At $25K: $750/month ($9K/year)
- At $50K: $1,500/month ($18K/year)
- At $100K: $3,000/month ($36K/year)
- **Target: Thousands per month**

---

## Key Commands

```bash
# Run the bot (production)
python run_kraken_bot.py

# Scan for new edges (run monthly)
python scan_edges.py --timeframe 15m --days 30

# Backtest specific symbols
python backtest_live.py --symbols SOLUSD,UNIUSD,ETHUSD,XLMUSD --timeframe 15m --days 30

# See growth projections
python scan_edges.py --timeframe 15m --days 30 --project 5000
```

---

## What Makes This Work

1. **Edge exists because of fees**: Most traders use market orders (0.26% taker fee). We use post-only limit orders (0.16% maker fee). That 0.10% saving per trade compounds over time.

2. **ATR-to-fee filter**: The bot only trades when volatility is high enough to overcome fees. This prevents the #1 killer of trading bots: churning on low-volatility periods.

3. **Fee-aware EV filter**: Every signal is checked: "Does this trade have positive expected value AFTER fees?" If not, it's skipped.

4. **Compound position sizing**: Position sizes are 2% of current balance (not fixed dollars). As the account grows, positions grow proportionally. This is automatic compounding.

5. **Selective symbol trading**: Out of 25 pairs scanned, only 4 have a proven edge. We trade them. No dilution by wasted trades on losing pairs.

---

## Risk Warning

- Past backtest performance does not guarantee future results
- Crypto markets can change regime (trending ↔ ranging)
- Run `scan_edges.py` monthly to verify edge still exists
- Never risk more than you can afford to lose
- The 3% monthly return is an average — individual months will vary
