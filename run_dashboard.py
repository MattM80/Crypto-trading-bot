"""
Run a simple web dashboard for the trading bot.
Access at: http://localhost:8000
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pathlib import Path

try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
except ImportError:
    pass

from flask import Flask, render_template_string
import json

# Simple Flask app for monitoring
app = Flask(__name__)

# Dashboard HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Trading Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #fff;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-box {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 15px;
            backdrop-filter: blur(10px);
        }
        .stat-label {
            font-size: 12px;
            text-transform: uppercase;
            opacity: 0.8;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }
        .positive { color: #4ade80; }
        .negative { color: #ef4444; }
        .status {
            text-align: center;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Crypto Trading Bot Dashboard</h1>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Status</div>
                <div class="stat-value">🟢 Running</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Exchange</div>
                <div class="stat-value">Kraken</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Strategy</div>
                <div class="stat-value">Grid Trading</div>
            </div>
        </div>
        
        <div class="status">
            <h2>✅ Bot is Running!</h2>
            <p>The trading bot is active and monitoring markets.</p>
            <p>Check the terminal output for trade updates and logs.</p>
            <p style="margin-top: 20px; font-size: 12px; opacity: 0.7;">
                Bot logs: <code>logs/trading_bot.log</code>
            </p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML)

@app.route('/api/status')
def status():
    return {
        "status": "running",
        "exchange": "kraken",
        "strategy": "grid_trading",
        "timestamp": "live"
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Trading Bot Dashboard")
    print("=" * 60)
    print("\n📊 Dashboard running at: http://localhost:8000\n")
    print("Open this URL in your browser to monitor the bot.")
    print("Press Ctrl+C to stop.\n")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
