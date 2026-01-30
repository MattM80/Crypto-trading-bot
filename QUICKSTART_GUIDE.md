# 🚀 Quick Start Guide - Full System Setup

How to start your crypto trading bot from scratch each time.

---

## 📋 Prerequisites

- Python 3.8+ installed
- Kraken API keys in `.env` file
- Virtual environment already created (one-time setup)

---

## 🎯 Step 1: Open Project Folder

Open PowerShell and navigate to the project:

```powershell
cd "C:\Users\Matthew V Mady\Lucas code\crypto-trading-bot"
```

---

## 🔧 Step 2: Activate Virtual Environment

**IMPORTANT:** You must do this every time before running the bot!

```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your terminal prompt:
```
(venv) PS C:\Users\Matthew V Mady\Lucas code\crypto-trading-bot>
```

**If you get an error:** Run this once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

## ✅ Step 3: Verify API Keys are Set

Make sure your `.env` file has your Kraken credentials:

```
KRAKEN_API_KEY=your_key_here
KRAKEN_PRIVATE_KEY=your_key_here
```

---

## 🤖 Step 4: Start the Bot (Terminal 1)

**Keep this terminal open and running.**

```powershell
python run_kraken_bot.py
```

You should see output like:
```
✅ Found Kraken credentials
   API Key: Pqwgx1nM0O0KMdeWk+Z2V...
   Private Key: Foo/ZFA+LElFnPGMoaPg...

🚀 Starting bot with Kraken Sandbox...

2026-01-21 18:45:25 | INFO | Generated 0 high-confidence signals
```

This means the bot is running! Leave this terminal open.

---

## 📊 Step 5: Start Dashboard (Terminal 2)

**Open a NEW PowerShell window** and do the same:

```powershell
cd "C:\Users\Matthew V Mady\Lucas code\crypto-trading-bot"
venv\Scripts\Activate.ps1
python run_dashboard.py
```

You should see:
```
============================================================
🚀 Starting Trading Bot Dashboard
============================================================

📊 Dashboard running at: http://localhost:8000

Open this URL in your browser to monitor the bot in real-time.
Press Ctrl+C to stop.

============================================================
```

Leave this terminal open too.

---

## 🌐 Step 6: Open Dashboard in Browser

Open your browser and go to:

```
http://localhost:8000
```

You should see the dashboard with bot status and live updates.

---

## 📈 Monitoring Your Bot

### Check Real-Time Logs (Terminal 3)

Open another PowerShell and run:

```powershell
cd "C:\Users\Matthew V Mady\Lucas code\crypto-trading-bot"
Get-Content logs\trading_bot.log -Wait
```

This shows live trading activity as it happens.

### Dashboard Metrics

On the dashboard you can see:
- **Status**: 🟢 Running or 🔴 Stopped
- **Exchange**: Kraken Sandbox
- **Strategy**: Grid Trading
- **Trades**: Active positions and history
- **P&L**: Profit/Loss percentage
- **Win Rate**: Percentage of winning trades

---

## 🛑 Stopping Everything

**To stop the bot:**
1. Go to Terminal 1 (the one running the bot)
2. Press `Ctrl+C`

**To stop the dashboard:**
1. Go to Terminal 2 (the one running the dashboard)
2. Press `Ctrl+C`

---

## 🧪 Optional: Run Backtests (No Real Money Risk)

If you want to test the strategy on historical data first:

```powershell
python examples.py
```

Then choose:
```
Option 3: Backtest a Strategy
```

This simulates trading on past data to see how the bot would have performed.

---

## 📁 File Locations

Important files to know:

| File | Purpose |
|------|---------|
| `.env` | Your Kraken API keys (KEEP SAFE!) |
| `logs/trading_bot.log` | All bot trading logs |
| `run_kraken_bot.py` | Main bot entry point |
| `run_dashboard.py` | Web dashboard |
| `src/trading_bot.py` | Bot engine |
| `src/kraken_client.py` | Kraken API connector |
| `src/strategies.py` | Trading strategies |
| `config/config.py` | Bot settings |

---

## 🔧 Customizing the Bot

### Change Trading Symbols

Edit `run_kraken_bot.py` (around line 39):

```python
config.trading_strategy = TradingStrategy(
    strategy_type="grid",
    symbols=["XBTUSD", "ETHUSD"],  # ← Change these
    timeframe="5m",
    grid_levels=10,
    grid_range_percent=0.05
)
```

Kraken symbols:
- `XBTUSD` = Bitcoin
- `ETHUSD` = Ethereum
- `ADAUSD` = Cardano
- etc.

### Change Risk Settings

In `run_kraken_bot.py` (around line 43):

```python
config.risk_management = RiskManagement(
    max_position_size=0.02,        # Max 2% per trade (← change here)
    max_drawdown=0.10,              # Stop if 10% loss (← or here)
    stop_loss_percent=0.02,         # 2% stop loss
    take_profit_percent=0.05        # 5% take profit
)
```

### Change Strategy Type

Change the strategy in `run_kraken_bot.py` (line 37):

```python
config.trading_strategy = TradingStrategy(
    strategy_type="grid",           # "grid", "mean_reversion", or "arbitrage"
    ...
)
```

---

## ⚠️ Important Safety Tips

1. **Start with small amounts** - Don't trade your entire account on day 1
2. **Let it run overnight** - Trading works 24/7 on crypto
3. **Monitor closely** - Check logs and dashboard regularly
4. **Stop losses are on** - The bot will automatically stop losses at 2% by default
5. **Max drawdown protection** - Bot stops trading if 10% loss is reached
6. **Backup your keys** - Keep your `.env` file safe!

---

## 🐛 Troubleshooting

### "venv not found"
You need to create it first:
```powershell
python -m venv venv
```

### "API Secret required"
Make sure:
1. `.env` file exists in project root
2. Both `KRAKEN_API_KEY` and `KRAKEN_PRIVATE_KEY` are filled in
3. Virtual environment is activated `(venv)` shows in terminal

### "ModuleNotFoundError"
Make sure venv is activated and run:
```powershell
pip install -r requirements.txt
```

### Dashboard won't open
Make sure:
1. Dashboard terminal shows "running at http://localhost:8000"
2. Port 8000 isn't already in use
3. Try `http://127.0.0.1:8000` instead

### Bot shows "0 signals"
This is normal! The bot:
- Needs 5+ minutes of data first
- Only trades when conditions are met
- May not trade every interval

---

## 📊 Expected Output

**Terminal 1 (Bot):**
```
🚀 Starting bot with Kraken Sandbox...

2026-01-21 18:45:25 | INFO | Generated 0 high-confidence signals
2026-01-21 18:45:25 | INFO | Portfolio: $10000.00 | P&L: +0.00% | Trades: 0
```

**Terminal 2 (Dashboard):**
```
🚀 Starting Trading Bot Dashboard

📊 Dashboard running at: http://localhost:8000
```

**Browser (localhost:8000):**
Shows dashboard with status, strategy, and metrics.

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Start project | `cd "C:\Users\Matthew V Mady\Lucas code\crypto-trading-bot"` |
| Activate venv | `venv\Scripts\Activate.ps1` |
| Start bot | `python run_kraken_bot.py` |
| Start dashboard | `python run_dashboard.py` |
| View logs | `Get-Content logs\trading_bot.log -Wait` |
| Backtest | `python examples.py` |
| Deactivate venv | `deactivate` |

---

## 📞 Need Help?

Check these files:
- `README.md` - Full documentation
- `logs/trading_bot.log` - Trading activity logs
- `GETTING_STARTED.md` - Detailed setup guide
- `SYSTEM_OVERVIEW.md` - How everything works

---

## 🎉 You're Ready!

Now you can:
1. Activate venv
2. Start the bot
3. Start the dashboard
4. Monitor trades in real-time
5. Let it trade 24/7

**Happy trading! 🚀**

---

*Last updated: January 21, 2026*
