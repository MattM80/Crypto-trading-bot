# Daily Operating Instructions (Kraken XRP Bot)

This is a **live Kraken spot trading bot**.
- It can place **real orders**.
- Kraken spot has **no sandbox**.
- Keep trade sizes small.

## 1) One-time setup (do this once)

### A) Make sure your `.env` exists
- File location: `.env` (repo root)
- It must include:
  - `KRAKEN_API_KEY=...`
  - `KRAKEN_PRIVATE_KEY=...`
  - `ENABLE_LIVE_TRADING=true`

Do **not** share `.env` contents with anyone.

### B) Kraken API key permissions
In Kraken API key settings, enable (minimum):
- **Query Funds** (so the bot can read balances)
- **Trade** (so the bot can place/cancel orders)

For the end-of-day performance report script, also enable:
- **Query Trades** (so we can calculate fees + realized P&L)

### C) Windows PowerShell + venv
From the repo folder `C:\Users\Matthew V Mady\Lucas code\crypto-trading-bot`:

1. Create a venv (if you don't have one already):
   - `python -m venv venv`

2. Activate it:
   - `.\venv\Scripts\Activate.ps1`

3. Install dependencies:
   - `pip install -r requirements.txt`

If you ever move the folder to a different path and venv breaks, recreate the venv.

---

## 2) Start-of-day (how to start the bot)

1. Open PowerShell.
2. Go to the repo folder:
   - `cd C:\code\crypto-trading-bot`
3. Activate venv:
   - `.\venv\Scripts\Activate.ps1`
4. Start the bot:
   - `python .\run_kraken_bot.py`

Leave the window open while it runs.

### What you should see
Each cycle you’ll see lines like:
- **Kraken money (REAL):** this is the real Kraken cash/holdings/equity
- **Bot tracker (estimate):** internal stats (not authoritative)

If you see **KILL SWITCH** messages, the bot has paused or stopped for safety.

---

## 3) During the day (what to monitor)

### A) The most important line
Look for:
- **`Kraken money (REAL)`**

It shows:
- **Cash USD**: how much USD is available
- **Holdings**: how much XRP you currently own
- **Total Equity≈**: USD + (XRP valued at latest price)
- **Change since start**: how much equity has changed since the bot started

### B) Warning signs (stop and ask for help)
Stop the bot (Ctrl+C) and ask for help if you see any of these repeatedly:
- Balance fetch errors (permissions/connection)
- Order placement errors (invalid volume/price, permissions)
- Very rapid trading (more than expected)
- Equity dropping fast

### C) Risk controls that should NOT be changed casually
These protect you from blowing up:
- `MAX_NOTIONAL_PER_TRADE`
- `MAX_TOTAL_EXPOSURE`
- `DAILY_LOSS_LIMIT`
- `KILL_SWITCH_STOP_BOT`

If you want more activity, talk to chat first—don’t just raise limits.

### D) How to stop safely
- Press `Ctrl + C` once.
- Wait for it to print final stats and exit.

---

## 4) End-of-day (nightly performance report)

This report uses **Kraken trade history** to compute:
- realized P&L **net of fees**
- total fees paid
- buy outflows and sell inflows

1. Open PowerShell
2. Go to the repo folder:
   - `cd C:\code\crypto-trading-bot`
3. Activate venv:
   - `.\venv\Scripts\Activate.ps1`
4. Run the report (last 24h):
   - `python .\report_kraken_performance.py --pair XRPUSD --hours 24`

If it prints “No trade history returned”, your API key likely needs **Query Trades** enabled.

### How to read the report
- **Realized P&L (net fees)**: what you actually made/lost from completed round-trips
- **Total fees**: what Kraken charged for those trades
- **Unrealized P&L** (optional section): value of leftover XRP minus its cost basis

---

## 5) When to ask chat to change things

Ask chat to adjust settings if:
- You want **more/fewer trades**
- You want **tighter/wider TP/SL**
- You see **too many small losses** (fees may be eating you)
- You see **orders not filling** (limit orders can sit)
- You want to switch pairs (ex: `XRPUSD` → another)

### When you ask, include this info
Copy/paste these 5 things (do NOT paste API keys):
1. Your `.env` settings block (without keys)
2. The last ~50 log lines from the bot
3. The nightly report output
4. What you want to optimize (profit vs safety vs number of trades)
5. Your max acceptable daily loss

### Don’t change these without chat
- Turning off the kill switch
- Raising exposure limits
- Increasing max trades/day dramatically

---

## 6) Quick troubleshooting

### Bot says it can’t fetch balances
- Check Kraken API key permissions: **Query Funds**
- Check internet connection

### Report script says no trade history
- Enable **Query Trades** permission

### Bot seems to disagree with Kraken P&L
- Trust the report script (Kraken fees + trades)
- “Bot tracker (estimate)” is not an exchange reconciliation

---

## 7) What files matter
- `.env` = settings (keep private)
- `logs/trading_bot.log` = full logs (useful for debugging)
- `run_kraken_bot.py` = starts the bot
- `report_kraken_performance.py` = end-of-day P&L/fees report
