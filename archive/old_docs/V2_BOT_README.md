# V2 Trading Bot - Clean Rewrite

## Philosophy
**CAPITAL PRESERVATION FIRST** - This bot sits in cash 70-90% of the time. Only trades A+ setups with high conviction. Boring, patient, disciplined.

## Key Features

### 1. Regime-Based Trading
- **Bull**: F&G >50 + BTC above 20-day SMA → Max 3 positions, 25% deployed
- **Neutral**: Mixed signals → Max 2 positions, 15% deployed  
- **Bear**: F&G <25 or BTC below SMA → Max 1 position, 8% deployed

### 2. Only 8 Liquid Pairs
XBTUSD, ETHUSD, SOLUSD, XRPUSD, DOGEUSD, AVAXUSD, LINKUSD, DOTUSD
(All $50M+ daily volume, no more microcap disasters)

### 3. 4 High-Conviction Strategies
1. **Capitulation Buy** - F&G <20, >5% drop, RSI <25, volume spike
2. **Momentum Continuation** - Bull regime only, trending moves
3. **Support Bounce** - Near 200 SMA with oversold RSI
4. **Liquidation Cascade** - >8% drop in 48h with massive volume

### 4. Strict Risk Management
- Time-based exits (24h-7 days depending on regime)
- Trailing stops in bull markets
- Daily max drawdown: -3%
- Monthly max drawdown: -5%
- 6-hour cooldown after stop losses

## Running the Bot

```bash
# Make sure you have .env configured with:
# KRAKEN_API_KEY=your_key
# KRAKEN_PRIVATE_KEY=your_private_key  
# ENABLE_LIVE_TRADING=true
# STARTING_BALANCE=300

# Run the bot
python3 run_v2_bot.py

# Or use the wrapper
python3 start_v2_bot.py
```

## Files Created

- `logs/v2_bot.log` - Main bot log
- `logs/v2_trades.csv` - Trade history
- `logs/v2_balance.csv` - Balance snapshots
- `data/v2_bot_state.json` - Bot state persistence

## Monitoring

The bot logs its status every 5 minutes:
```
Cycle complete - Balance: $308.61 | Regime: BEAR | Positions: 0 | Daily PnL: $0.00
```

In bear markets, expect to see 0 positions most of the time. This is GOOD.

## Safety Features

- Uses existing tested kraken_client.py (no changes)
- Websocket crash monitoring via ws_monitor.py
- Graceful shutdown on SIGINT/SIGTERM  
- Position sync on startup
- Conservative position sizing
- No new dependencies (uses numpy, pandas, requests, loguru, python-dotenv)

## Philosophy Reminder

This is the OPPOSITE of the previous 30+ tool bot that lost $60 on DRIFTUSD. 
We trade LESS, not more. Quality over quantity. Cash is a position.