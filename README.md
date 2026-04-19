# Crypto Trading Bot — Kraken Spot

Live trading bot on Kraken spot. $300 starting balance, launched Mar 27 2026.

## Live Files
- `run_final_bot.py` — The production bot (single file, ~4k lines)
- `src/kraken_client.py` — Kraken REST API client
- `src/ws_monitor.py` — Websocket crash/pump/volume monitor
- `data/final_bot_state.json` — Persistent state
- `logs/` — Trade journal, signal rejections, bot logs
- `cancel_all_orders.py` — Emergency: cancel all open Kraken orders
- `UPGRADE_PLAN.md` — Improvement roadmap (Phase 1-3)

## Run
```bash
cd /Users/lucasaust/code/Crypto-trading-bot
ENABLE_LIVE_TRADING=true nohup python3 run_final_bot.py > /dev/null 2>&1 &
```

## Architecture
Single-file bot scanning 40 Kraken pairs every 5 minutes. 30+ validated signal tools (crash buying, dip buying, spread trading). Limit order entries, smart exits (trailing, fixed TP, stop loss). Real-time websocket monitoring between cycles.

## Archive
Old bots, research scripts, backtests, and docs in `archive/`.
