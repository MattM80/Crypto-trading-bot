"""
Web dashboard for monitoring the trading bot in real-time.
"""
from flask import Flask, jsonify, render_template_string
from loguru import logger
import json
from datetime import datetime
from threading import Thread

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Trading Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #2a5298;
        }
        
        .card h2 {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        
        .card .value {
            font-size: 2em;
            font-weight: bold;
            color: #2a5298;
        }
        
        .card.positive .value {
            color: #27ae60;
        }
        
        .card.negative .value {
            color: #e74c3c;
        }
        
        .trades-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .trades-table th {
            background: #2a5298;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .trades-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        
        .trades-table tr:hover {
            background: #f5f5f5;
        }
        
        .trades-table .positive {
            color: #27ae60;
            font-weight: bold;
        }
        
        .trades-table .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .status.open {
            background: #3498db;
            color: white;
        }
        
        .status.closed {
            background: #95a5a6;
            color: white;
        }
        
        .refresh-btn {
            display: block;
            margin: 20px auto;
            padding: 10px 20px;
            background: #2a5298;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        
        .refresh-btn:hover {
            background: #1e3c72;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Crypto Trading Bot Dashboard</h1>
        
        <div class="grid" id="stats-grid">
            <div class="card">
                <h2>Current Balance</h2>
                <div class="value" id="balance">$0.00</div>
            </div>
            
            <div class="card">
                <h2>Total P&L</h2>
                <div class="value" id="pnl">$0.00</div>
            </div>
            
            <div class="card">
                <h2>Total Trades</h2>
                <div class="value" id="total-trades">0</div>
            </div>
            
            <div class="card">
                <h2>Win Rate</h2>
                <div class="value" id="win-rate">0%</div>
            </div>
            
            <div class="card">
                <h2>Max Drawdown</h2>
                <div class="value" id="max-drawdown">0%</div>
            </div>
            
            <div class="card">
                <h2>Status</h2>
                <div class="value" id="status" style="font-size: 1.2em;">●</div>
            </div>
        </div>
        
        <h2 style="color: white; margin-top: 40px; margin-bottom: 20px;">Recent Trades</h2>
        <table class="trades-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Quantity</th>
                    <th>P&L</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="trades-tbody">
                <tr>
                    <td colspan="6" style="text-align: center; color: #999;">No trades yet</td>
                </tr>
            </tbody>
        </table>
        
        <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
    </div>
    
    <script>
        function refreshData() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('balance').textContent = '$' + data.balance.toFixed(2);
                    document.getElementById('pnl').textContent = '$' + data.total_pnl.toFixed(2);
                    document.getElementById('total-trades').textContent = data.total_trades;
                    document.getElementById('win-rate').textContent = data.win_rate.toFixed(1) + '%';
                    document.getElementById('max-drawdown').textContent = data.max_drawdown.toFixed(1) + '%';
                    document.getElementById('status').textContent = data.is_running ? '🟢' : '🔴';
                });
            
            fetch('/api/trades')
                .then(response => response.json())
                .then(trades => {
                    let tbody = document.getElementById('trades-tbody');
                    tbody.innerHTML = '';
                    
                    if (trades.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #999;">No trades yet</td></tr>';
                        return;
                    }
                    
                    trades.reverse().forEach(trade => {
                        let row = tbody.insertRow();
                        let pnlClass = trade.pnl > 0 ? 'positive' : 'negative';
                        
                        row.innerHTML = `
                            <td>${trade.symbol}</td>
                            <td>$${parseFloat(trade.entry_price).toFixed(2)}</td>
                            <td>$${parseFloat(trade.exit_price).toFixed(2)}</td>
                            <td>${parseFloat(trade.quantity).toFixed(4)}</td>
                            <td class="${pnlClass}">$${trade.pnl.toFixed(2)}</td>
                            <td><span class="status">${trade.status || 'Closed'}</span></td>
                        `;
                    });
                });
        }
        
        // Refresh data every 5 seconds
        setInterval(refreshData, 5000);
        refreshData(); // Initial load
    </script>
</body>
</html>
"""

class Dashboard:
    """Web dashboard for bot monitoring"""
    
    def __init__(self, bot, port=8000):
        self.bot = bot
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/api/stats')
        def get_stats():
            stats = self.bot.risk_manager.get_portfolio_stats()
            return jsonify({
                "balance": stats["balance"],
                "total_pnl": stats["total_pnl"],
                "total_pnl_percent": stats["total_pnl_percent"],
                "total_trades": stats["total_trades"],
                "winning_trades": stats["wins"],
                "losing_trades": stats["losses"],
                "win_rate": stats["win_rate"] * 100,
                "max_drawdown": stats["max_drawdown"] * 100,
                "is_running": self.bot.is_running,
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/trades')
        def get_trades():
            return jsonify(self.bot.risk_manager.trade_history)
        
        @self.app.route('/api/positions')
        def get_positions():
            positions = []
            for symbol, pos_list in self.bot.risk_manager.positions.items():
                for pos in pos_list:
                    positions.append({
                        "symbol": pos.symbol,
                        "entry_price": pos.entry_price,
                        "quantity": pos.quantity,
                        "stop_loss": pos.stop_loss,
                        "take_profit": pos.take_profit,
                        "side": pos.side,
                        "status": pos.status,
                        "entry_time": pos.entry_time
                    })
            return jsonify(positions)
    
    def run(self):
        """Run dashboard server"""
        logger.info(f"Dashboard running at http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)
    
    def run_in_background(self):
        """Run dashboard in a background thread"""
        thread = Thread(target=self.run, daemon=True)
        thread.start()
        return thread
