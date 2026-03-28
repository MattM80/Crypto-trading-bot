#!/usr/bin/env python3
"""
Real-time Kraken websocket monitor for flash crash detection.
Runs in a background thread, tracks prices, and fires callbacks
when crash conditions are detected.

Replaces 5-minute polling with sub-second detection for crash events.
"""

import json
import time
import threading
import numpy as np
from datetime import datetime, timezone
from collections import deque
from loguru import logger

try:
    import websocket
except ImportError:
    websocket = None
    logger.warning("websocket-client not installed, ws_monitor disabled")


class CrashMonitor:
    """Real-time price monitor that detects flash crashes via Kraken websocket."""
    
    # Track last N minutes of trades per pair
    WINDOW_SECONDS = 300  # 5 min rolling window
    
    # Crash thresholds
    CRASH_THRESHOLDS = {
        'flash_5m': -0.03,    # -3% in 5 min
        'flash_1m': -0.02,    # -2% in 1 min  
        'dump_5m': -0.05,     # -5% in 5 min (major)
    }
    
    # Pairs to monitor (Kraken websocket format)
    WS_PAIRS = [
        "XBT/USD", "ETH/USD", "SOL/USD", "LINK/USD", "DOT/USD", "ADA/USD",
        "AVAX/USD", "ATOM/USD", "XRP/USD", "DOGE/USD", "LTC/USD", "UNI/USD",
        "FIL/USD", "NEAR/USD", "AAVE/USD", "XLM/USD"
    ]
    
    # Map WS pair names to bot pair names
    PAIR_MAP = {
        "XBT/USD": "XBTUSD", "ETH/USD": "ETHUSD", "SOL/USD": "SOLUSD",
        "LINK/USD": "LINKUSD", "DOT/USD": "DOTUSD", "ADA/USD": "ADAUSD",
        "AVAX/USD": "AVAXUSD", "ATOM/USD": "ATOMUSD", "XRP/USD": "XRPUSD",
        "DOGE/USD": "DOGEUSD", "LTC/USD": "LTCUSD", "UNI/USD": "UNIUSD",
        "FIL/USD": "FILUSD", "NEAR/USD": "NEARUSD", "AAVE/USD": "AAVEUSD",
        "XLM/USD": "XLMUSD"
    }
    
    def __init__(self, on_crash_callback=None):
        """
        Args:
            on_crash_callback: function(pair, crash_type, drop_pct, current_price)
                Called when a crash is detected.
        """
        self.on_crash = on_crash_callback
        self.running = False
        self.ws = None
        self.thread = None
        
        # Price tracking: {pair: deque of (timestamp, price)}
        self.price_history = {pair: deque(maxlen=5000) for pair in self.PAIR_MAP.values()}
        self.latest_prices = {}  # {bot_pair: latest_price}
        self.alerts_sent = {}  # Prevent spam: {pair: last_alert_time}
        
        # Stats
        self.connected = False
        self.last_message_time = 0
        self.trade_count = 0
    
    def start(self):
        """Start the websocket monitor in a background thread."""
        if websocket is None:
            logger.warning("websocket-client not available, crash monitor disabled")
            return False
        
        if self.running:
            return True
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("🔌 Crash monitor websocket starting...")
        return True
    
    def stop(self):
        """Stop the monitor."""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        self.connected = False
    
    def get_price(self, bot_pair):
        """Get latest real-time price for a pair. Returns None if unavailable."""
        return self.latest_prices.get(bot_pair)
    
    def get_5m_change(self, bot_pair):
        """Get 5-minute price change % for a pair."""
        history = self.price_history.get(bot_pair)
        if not history or len(history) < 2:
            return 0
        
        now = time.time()
        current = history[-1][1]
        
        # Find price from ~5 min ago
        for ts, price in history:
            if now - ts <= self.WINDOW_SECONDS:
                return (current - price) / price
        
        return 0
    
    def _run(self):
        """Main websocket loop with reconnection."""
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    'wss://ws.kraken.com',
                    on_message=self._on_message,
                    on_open=self._on_open,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"Websocket error: {e}")
            
            if self.running:
                logger.info("Websocket disconnected, reconnecting in 5s...")
                self.connected = False
                time.sleep(5)
    
    def _on_open(self, ws):
        """Subscribe to trade feeds for all pairs."""
        self.connected = True
        logger.info(f"🔌 Crash monitor connected, subscribing to {len(self.WS_PAIRS)} pairs")
        
        ws.send(json.dumps({
            'event': 'subscribe',
            'pair': self.WS_PAIRS,
            'subscription': {'name': 'trade'}
        }))
    
    def _on_message(self, ws, message):
        """Process incoming trade messages."""
        try:
            data = json.loads(message)
            
            # Skip system messages
            if isinstance(data, dict):
                return
            
            # Trade data: [channelID, [[price, vol, time, side, type, misc], ...], channelName, pair]
            if isinstance(data, list) and len(data) >= 4:
                ws_pair = data[3]
                bot_pair = self.PAIR_MAP.get(ws_pair)
                if not bot_pair:
                    return
                
                trades = data[1]
                now = time.time()
                self.last_message_time = now
                
                for trade in trades:
                    price = float(trade[0])
                    self.price_history[bot_pair].append((now, price))
                    self.latest_prices[bot_pair] = price
                    self.trade_count += 1
                
                # Check for crash conditions
                self._check_crash(bot_pair)
                
        except Exception as e:
            pass  # Don't spam logs on parse errors
    
    def _on_error(self, ws, error):
        if self.running:
            logger.debug(f"Websocket error: {error}")
    
    def _on_close(self, ws, close_code, close_msg):
        self.connected = False
    
    def _check_crash(self, bot_pair):
        """Check if current price action qualifies as a crash."""
        history = self.price_history.get(bot_pair)
        if not history or len(history) < 10:
            return
        
        now = time.time()
        current_price = history[-1][1]
        
        # Don't spam alerts — 10 min cooldown per pair
        last_alert = self.alerts_sent.get(bot_pair, 0)
        if now - last_alert < 600:
            return
        
        # Check 5-minute drop
        for ts, price in history:
            if now - ts <= 300:  # Within 5 min
                drop_5m = (current_price - price) / price
                
                if drop_5m <= self.CRASH_THRESHOLDS['dump_5m']:
                    self.alerts_sent[bot_pair] = now
                    logger.warning(f"🚨 MAJOR DUMP: {bot_pair} {drop_5m*100:.1f}% in 5min (${price:.2f} → ${current_price:.2f})")
                    if self.on_crash:
                        self.on_crash(bot_pair, 'dump_5m', drop_5m, current_price)
                    return
                
                elif drop_5m <= self.CRASH_THRESHOLDS['flash_5m']:
                    self.alerts_sent[bot_pair] = now
                    logger.warning(f"⚡ FLASH CRASH: {bot_pair} {drop_5m*100:.1f}% in 5min (${price:.2f} → ${current_price:.2f})")
                    if self.on_crash:
                        self.on_crash(bot_pair, 'flash_5m', drop_5m, current_price)
                    return
                
                break  # Only check oldest price in window
        
        # Check 1-minute drop
        for ts, price in reversed(list(history)):
            if now - ts >= 60:
                drop_1m = (current_price - price) / price
                if drop_1m <= self.CRASH_THRESHOLDS['flash_1m']:
                    self.alerts_sent[bot_pair] = now
                    logger.warning(f"⚡ FLASH DIP: {bot_pair} {drop_1m*100:.1f}% in 1min")
                    if self.on_crash:
                        self.on_crash(bot_pair, 'flash_1m', drop_1m, current_price)
                break
