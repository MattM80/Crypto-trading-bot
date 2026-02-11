# How to Run the Crypto Trading Bot

1. Open **VS Code**
2. Press **Ctrl + `** to open the terminal (backtick key, top-left of keyboard next to the 1)
3. Run these commands one at a time:

```
cd C:\Users\YourName\code\Crypto-trading-bot
git pull origin main
venv\Scripts\activate
pip install -r requirements.txt
python run_kraken_bot.py
```

That's it — the bot will start running. You'll see log output showing what it's doing.

To stop it, press **Ctrl + C** in the terminal.
