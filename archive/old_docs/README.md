# Crypto Trading Bot

Automated crypto trading bot for Kraken spot markets. Uses 30+ validated technical signals with a hybrid grid + active trading strategy.

## How It Works

- **Grid Engine (60% of capital):** Passive income from multi-level grid trading across 40 pairs. ATR-adaptive spacing, auto-compounding.
- **Active Signals (40% of capital):** 30 validated tools that buy crash dips and short overbought pumps. Every signal was tested out-of-sample on real 1h Binance data with Kraken fees.
- **Dynamic Allocation:** Capital shifts between grid and active based on Fear & Greed index.
- **Risk Management:** Per-tool position limits, streak tracking, automatic stop-losses, max drawdown protection.

## Signal Categories

| Category | Tools | Strategy |
|----------|-------|----------|
| Crash/Bear | 15 | Buy dips (RSI oversold, volume spikes, capitulation) |
| Bull/Greed | 13 | Short pumps (overbought, distribution, exhaustion) |
| Neutral | 2 | Calendar effects, mean reversion |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/LucasAust/Crypto-trading-bot.git
cd Crypto-trading-bot
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your Kraken API credentials:
- Go to https://www.kraken.com/u/security/api
- Create a new API key with permissions: **Query Funds**, **Trade**, **Query Trades**
- Paste your key and private key into `.env`

### 3. Dry Run (Recommended First)

```bash
python3 run_final_bot.py
```

By default, `ENABLE_LIVE_TRADING=false` — the bot runs all its analysis and signals but doesn't place real orders. Watch the logs to make sure everything looks right.

### 4. Go Live

Once you're comfortable, edit `.env`:

```
ENABLE_LIVE_TRADING=true
```

Then restart:

```bash
python3 run_final_bot.py
```

### 5. Run in Background (Linux/macOS)

```bash
nohup python3 run_final_bot.py >> logs/final_bot.log 2>&1 &
```

Check logs:
```bash
tail -f logs/final_bot.log
```

## Requirements

- Python 3.8+
- Kraken account with API access
- ~$300+ recommended starting balance
- Stable internet connection

No paid APIs needed — uses free Binance data for OHLC and free CoinGecko for Fear & Greed.

## Configuration

All config is via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `KRAKEN_API_KEY` | (required) | Your Kraken API key |
| `KRAKEN_PRIVATE_KEY` | (required) | Your Kraken private key |
| `ENABLE_LIVE_TRADING` | `false` | Set `true` to place real orders |
| `STARTING_BALANCE` | `300` | Starting USD balance for sizing |
| `CHECK_INTERVAL` | `300` | Seconds between cycles |

## File Structure

| File | What It Does |
|------|-------------|
| `run_final_bot.py` | **THE bot.** Run this one. |
| `src/kraken_client.py` | Kraken spot API connector |
| `src/news_sentiment.py` | LLM news sentiment engine |
| `src/onchain_data.py` | On-chain data engine |
| `src/orderbook_engine.py` | Orderbook analysis |
| `src/ml_signal_weighter.py` | ML signal weighting |
| `src/volatility_engine.py` | Volatility intelligence |
| `data/` | Bot state, validated tools data |
| `logs/` | Trading logs |

## Monitoring

The bot logs every cycle with:
- Current balance and P&L
- Active positions with entry prices
- Top signals firing
- Grid status and round trips
- Fear & Greed regime

Example output:
```
[19:02] Balance: $295.44 | Grid: $180.00 (1 positions, 7 round trips) | Active: $0.23 (5/5 positions)
Signals: ema_cross_short XLMUSD (score 23.9), ema_cross_short AAVEUSD (score 16.8)
Regime: Extreme Fear (F&G=11) | Grid Profit: $0.39 | Active Profit: -$4.56
```

## ⚠️ Important Notes

- **Start with dry run.** Always verify signals make sense before going live.
- **Start small.** Don't put in more than you can afford to lose.
- **US users:** Kraken Futures is NOT available in the US. This bot uses Kraken spot.
- **No guarantees.** Past backtest performance doesn't guarantee future results.

## Old Files (Ignore)

These are earlier versions kept for reference. Don't run them:
- `run_production_bot.py` — predecessor (30 tools, no margin)
- `run_futures_bot.py` — Kraken Futures version (not available in US)
- `run_master_bot.py` — deprecated (74 unvalidated tools)
- `run_kraken_bot.py` — original v1
