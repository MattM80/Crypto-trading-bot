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
SPOT_ONLY_MODE=true EVIDENCE_MODE=true VALIDATION_ACCOUNT_MODE=true ENABLE_LIVE_TRADING=true nohup .venv/bin/python run_final_bot.py > logs/nohup_bot.out 2>&1 &
```

`SPOT_ONLY_MODE=true` keeps the system Kraken spot-only. Short detectors still
feed defensive overlays, but they do not route to futures or external execution.
`EVIDENCE_MODE=true` makes long tools compete for capital using live PnL,
walk-forward priors, score floors, stacking, and context after Kraken fees.
`VALIDATION_ACCOUNT_MODE=true` keeps the $300 account on conservative slots,
position caps, and exposure limits while forward evidence accumulates.
`FORWARD_TOOL_STRICT_VALIDATION=true` blocks recent losing long tools in the
enriched forward sample, and `ENABLE_QUALITY_UNIVERSE=true` keeps new validation
longs inside a known, externally checkable asset universe. `ENABLE_ASSET_CONTEXT_GUARD=true`
uses CoinGecko market-cap rank and severe 24h/7d drawdowns as veto-only context;
it never creates a buy signal.
`MARKET_BREADTH_RECOVERY_ENABLED=true` gives the bot a small green-day/rebound
participation lane when broad 4h breadth, BTC context, volume, and the asset's
own 4h/24h recovery agree, even if sentiment is still lagging in fear.
`AUTONOMOUS_PROOF_LADDER_ENABLED=true` lets long tools earn looser gates only
after clean forward proof. Scout-only proof can slightly reduce score floors;
real closed live trades are required before stack/bull gates relax or sizing is
lifted. `OPPORTUNITY_SCOUT_ENABLED=true` records blocked/rejected long ideas in
`logs/opportunity_scout.csv` and labels their forward outcome after the scout
horizon so the bot can search for new edges without immediately risking money.
Fear & Greed comes from CoinMarketCap's crypto Fear & Greed Index. The bot reads
the live public CMC page state first because the official historical API can lag
the visible dial. `COINMARKETCAP_API_KEY` is optional and used only as a fresh
fallback. CNN Fear & Greed is stock-market sentiment, not crypto F&G, and the old
`alternative.me` default is no longer used for live gates.
Even in bull-offense mode, validation defaults keep total deployment capped near
65% and only use a small size lift unless `.env` deliberately raises it.

The bot now loads `.env` at startup, so these flags can live there as well as in
the shell environment.

## Watchdog
Use the macOS launchd watchdog if you want the bot to survive VS Code exits and restart automatically after crashes.

```bash
cd /Users/lucasaust/code/Crypto-trading-bot
chmod +x scripts/launchd_watchdog.sh
scripts/launchd_watchdog.sh install
```

The watchdog defaults to `.venv/bin/python` and exports `SPOT_ONLY_MODE=true`,
`EVIDENCE_MODE=true`, `VALIDATION_ACCOUNT_MODE=true`, futures off, and external
export off. Override with `BOT_PYTHON_BIN` or `BOT_*` environment variables only
when deliberately changing deployment mode.

Useful commands:

```bash
scripts/launchd_watchdog.sh status
scripts/launchd_watchdog.sh restart
scripts/launchd_watchdog.sh uninstall
```

## Architecture
Single-file bot scanning 40 Kraken pairs every 5 minutes. 30+ validated signal tools (crash buying, dip buying, spread trading). Limit order entries, smart exits (trailing, fixed TP, stop loss). Real-time websocket monitoring between cycles.

## Archive
Old bots, research scripts, backtests, and docs in `archive/`.
