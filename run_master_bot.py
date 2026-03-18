#!/usr/bin/env python3
"""
MASTER ADAPTIVE TRADING BOT - Final Production Version
Detects market regimes and activates the optimal strategy per regime.

2-year backtest results:
- trending_up: Grid +5.88% 
- trending_down: RSI +2.13%
- volatile: Momentum +2.95%
- ranging: Stay flat (no strategy works)

Architecture: ONE job = detect regime → activate right strategy → go to cash when nothing works
"""

import sys
import os
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

# Add src directory for kraken_client
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from kraken_client import KrakenClient
except ImportError as e:
    logger.error(f"Failed to import KrakenClient: {e}")
    sys.exit(1)

# Configuration
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "300"))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = DATA_DIR / "master_bot_state.json"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add(LOGS_DIR / "master_bot.log", rotation="10 MB", retention="30 days")

# Optimal configurations from 2-year backtest
GRID_CONFIGS = {
    "NEARUSD": {"grid_pct": 0.01, "tp_pct": 0.015},
    "UNIUSD": {"grid_pct": 0.015, "tp_pct": 0.015},
    "AVAXUSD": {"grid_pct": 0.01, "tp_pct": 0.015},
    "LINKUSD": {"grid_pct": 0.008, "tp_pct": 0.015},
    "AAVEUSD": {"grid_pct": 0.015, "tp_pct": 0.015},
    "SOLUSD": {"grid_pct": 0.003, "tp_pct": 0.015},
    "ETHUSD": {"grid_pct": 0.005, "tp_pct": 0.015},
    "XBTUSD": {"grid_pct": 0.01, "tp_pct": 0.015},
    "DOTUSD": {"grid_pct": 0.012, "tp_pct": 0.015},
    "XLMUSD": {"grid_pct": 0.01, "tp_pct": 0.015},
}

RSI_CONFIGS = {
    "ATOMUSD": {"period": 7, "oversold": 20, "overbought": 80, "hold": 5},
    "AVAXUSD": {"period": 14, "oversold": 20, "overbought": 70, "hold": 8},
    "XRPUSD": {"period": 9, "oversold": 20, "overbought": 80, "hold": 3},
    "NEARUSD": {"period": 9, "oversold": 20, "overbought": 80, "hold": 8},
    "FILUSD": {"period": 14, "oversold": 20, "overbought": 80, "hold": 8},
    "XBTUSD": {"period": 14, "oversold": 20, "overbought": 80, "hold": 8},
    "ETHUSD": {"period": 14, "oversold": 20, "overbought": 80, "hold": 12},
    "SOLUSD": {"period": 14, "oversold": 20, "overbought": 80, "hold": 8},
}

# Top pairs for each regime
GRID_PAIRS = list(GRID_CONFIGS.keys())[:6]  # Top 6 for grid
RSI_PAIRS = list(RSI_CONFIGS.keys())[:6]    # Top 6 for RSI
MOMENTUM_PAIRS = ["XBTUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "NEARUSD"][:3]  # Top 3 for momentum


class AdaptiveBot:
    """The master bot that detects regimes and activates the optimal strategy."""
    
    def __init__(self):
        self.client = KrakenClient()
        self.running = True
        self.state = self.load_state()
        
        # Core state
        self.balance = self.state.get("balance", STARTING_BALANCE)
        self.current_regime = self.state.get("current_regime", "ranging")
        self.previous_regime = self.state.get("previous_regime", "ranging")
        
        # Strategy states
        self.grid_inventory = self.state.get("grid_inventory", {})  # pair -> [{"buy_price": x, "qty": y}]
        self.active_positions = self.state.get("active_positions", {})  # pair -> position_data
        
        # P&L tracking
        self.grid_profit = self.state.get("grid_profit", 0.0)
        self.active_profit = self.state.get("active_profit", 0.0)
        self.total_trades = self.state.get("total_trades", 0)
        self.winning_trades = self.state.get("winning_trades", 0)
        
        logger.info(f"AdaptiveBot initialized - Balance: ${self.balance:.2f}")
        logger.info(f"Current regime: {self.current_regime}")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
        
    def load_state(self) -> dict:
        """Load bot state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    logger.info("Loaded existing state")
                    return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {}
        
    def save_state(self):
        """Save bot state to disk."""
        state = {
            "balance": self.balance,
            "current_regime": self.current_regime,
            "previous_regime": self.previous_regime,
            "grid_inventory": self.grid_inventory,
            "active_positions": self.active_positions,
            "grid_profit": self.grid_profit,
            "active_profit": self.active_profit,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "last_update": datetime.now().isoformat()
        }
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect market regime using 1h data.
        - Calculate 20-bar price return
        - Calculate ATR vs median ATR (volatility)
        - If ATR > 1.5x median → "volatile"
        - If 20-bar return > +5% → "trending_up"  
        - If 20-bar return < -5% → "trending_down"
        - Else → "ranging"
        """
        if len(df) < 60:
            return "ranging"
            
        try:
            # Calculate 20-bar return
            current_price = df['close'].iloc[-1]
            price_20_bars_ago = df['close'].iloc[-21]  # 20 bars back
            return_20bar = (current_price - price_20_bars_ago) / price_20_bars_ago
            
            # Calculate ATR
            df = df.copy()
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['true_range'].rolling(14).mean()
            
            current_atr = df['atr'].iloc[-1]
            median_atr = df['atr'].rolling(50).median().iloc[-1]
            
            if pd.isna(current_atr) or pd.isna(median_atr):
                return "ranging"
            
            # Regime classification (FAST and DECISIVE)
            if current_atr > median_atr * 1.5:
                return "volatile"
            elif return_20bar > 0.05:  # +5%
                return "trending_up"
            elif return_20bar < -0.05:  # -5%
                return "trending_down"
            else:
                return "ranging"
                
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return "ranging"
            
    def calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate current RSI value."""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        
    def handle_regime_transition(self, new_regime: str, market_data: dict):
        """Handle transitions between regimes - CRITICAL logic."""
        if new_regime == self.current_regime:
            return  # No change
            
        logger.info(f"[REGIME SWITCH] {self.current_regime} → {new_regime}")
        
        # REGIME TRANSITION RULES:
        # 1. Any → ranging: Close all positions, go to cash
        # 2. Any → trending_up: Close RSI/momentum, start grid
        # 3. Any → trending_down: Close grid/momentum, start RSI
        # 4. Any → volatile: Close grid/RSI, start momentum
        
        # First: close positions from old strategy
        if new_regime == "ranging":
            self.close_all_positions(market_data, "Market is ranging")
        elif new_regime == "trending_up":
            self.close_rsi_positions(market_data, "Switching to grid")
            self.close_momentum_positions(market_data, "Switching to grid")
        elif new_regime == "trending_down":
            self.close_grid_positions(market_data, "Switching to RSI")
            self.close_momentum_positions(market_data, "Switching to RSI")
        elif new_regime == "volatile":
            self.close_grid_positions(market_data, "Switching to momentum")
            self.close_rsi_positions(market_data, "Switching to momentum")
            
        self.previous_regime = self.current_regime
        self.current_regime = new_regime
        
    def close_all_positions(self, market_data: dict, reason: str):
        """Close ALL positions - go to cash."""
        logger.info(f"[CASH MODE] {reason}")
        self.close_grid_positions(market_data, reason)
        self.close_rsi_positions(market_data, reason)
        self.close_momentum_positions(market_data, reason)
        
    def close_grid_positions(self, market_data: dict, reason: str):
        """Close grid inventory gracefully."""
        if not self.grid_inventory:
            return
            
        logger.info(f"[GRID CLOSE] {reason}")
        
        for pair, inventory in self.grid_inventory.items():
            if pair not in market_data:
                continue
                
            current_price = market_data[pair]["price"]
            total_qty = sum(pos["qty"] for pos in inventory)
            
            if total_qty > 0:
                if ENABLE_LIVE_TRADING:
                    self._place_market_order(pair, "sell", total_qty, current_price)
                else:
                    logger.info(f"[DRY RUN] Would sell {total_qty:.6f} {pair} @ ${current_price:.4f}")
                    
                # Calculate P&L
                avg_buy_price = sum(pos["buy_price"] * pos["qty"] for pos in inventory) / total_qty
                pnl = (current_price - avg_buy_price) * total_qty * 0.999  # Minus fees
                self.grid_profit += pnl
                logger.info(f"[GRID] Closed {pair} inventory: {total_qty:.6f} @ ${current_price:.4f}, P&L: ${pnl:.2f}")
                
        self.grid_inventory.clear()
        
    def close_rsi_positions(self, market_data: dict, reason: str):
        """Close RSI positions."""
        rsi_positions = {k: v for k, v in self.active_positions.items() if v.get("strategy") == "rsi"}
        
        if not rsi_positions:
            return
            
        logger.info(f"[RSI CLOSE] {reason}")
        
        for pair, position in rsi_positions.items():
            if pair in market_data:
                self._close_position(pair, position, market_data[pair]["price"], reason)
                
    def close_momentum_positions(self, market_data: dict, reason: str):
        """Close momentum positions."""
        momentum_positions = {k: v for k, v in self.active_positions.items() if v.get("strategy") == "momentum"}
        
        if not momentum_positions:
            return
            
        logger.info(f"[MOMENTUM CLOSE] {reason}")
        
        for pair, position in momentum_positions.items():
            if pair in market_data:
                self._close_position(pair, position, market_data[pair]["price"], reason)
                
    def _close_position(self, pair: str, position: dict, current_price: float, reason: str):
        """Close a single position."""
        if ENABLE_LIVE_TRADING:
            close_side = "sell" if position["side"] == "buy" else "buy"
            self._place_market_order(pair, close_side, position["qty"], current_price)
        else:
            logger.info(f"[DRY RUN] Would close {pair} {position['side']} position")
            
        # Calculate P&L
        if position["side"] == "buy":
            pnl = (current_price - position["entry_price"]) * position["qty"]
        else:
            pnl = (position["entry_price"] - current_price) * position["qty"]
            
        pnl *= 0.999  # Minus fees/slippage
        self.active_profit += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            
        logger.info(f"[CLOSE] {pair} {position['strategy']}: P&L ${pnl:.2f} ({reason})")
        
        del self.active_positions[pair]
        
    def run_grid_strategy(self, market_data: dict):
        """Run grid strategy for trending_up regime."""
        if not market_data:
            return
            
        allocation_per_pair = self.balance / len(GRID_PAIRS)
        
        for pair in GRID_PAIRS:
            if pair not in market_data or pair not in GRID_CONFIGS:
                continue
                
            data = market_data[pair]
            current_price = data["price"]
            current_low = data.get("low", current_price)
            current_high = data.get("high", current_price)
            config = GRID_CONFIGS[pair]
            
            # Initialize inventory if needed
            if pair not in self.grid_inventory:
                self.grid_inventory[pair] = []
                
            inventory = self.grid_inventory[pair]
            
            # Check for buy fills (when price touched our grid levels)
            grid_levels = 3  # Number of grid levels
            for level in range(1, grid_levels + 1):
                buy_price = current_price * (1 - config["grid_pct"] * level)
                
                # Simulate fill if low touched this level
                if current_low <= buy_price:
                    qty = (allocation_per_pair / grid_levels) / buy_price
                    
                    # Check if we don't already have inventory at this level
                    level_filled = any(abs(pos["buy_price"] - buy_price) / buy_price < 0.001 for pos in inventory)
                    
                    if not level_filled:
                        if ENABLE_LIVE_TRADING:
                            if self._place_limit_order(pair, "buy", qty, buy_price):
                                inventory.append({"buy_price": buy_price, "qty": qty})
                        else:
                            inventory.append({"buy_price": buy_price, "qty": qty})
                            logger.info(f"[GRID] {pair} buy filled @ ${buy_price:.4f}, qty: {qty:.6f}")
                            
            # Check for sell fills (take profit)
            remaining_inventory = []
            for position in inventory:
                sell_target = position["buy_price"] * (1 + config["tp_pct"])
                
                if current_high >= sell_target:
                    # Sell filled
                    pnl = (sell_target - position["buy_price"]) * position["qty"] * 0.998  # Minus fees
                    self.grid_profit += pnl
                    logger.info(f"[GRID] {pair} round-trip complete: ${pnl:.2f} profit")
                    
                    if ENABLE_LIVE_TRADING:
                        self._place_market_order(pair, "sell", position["qty"], sell_target)
                else:
                    remaining_inventory.append(position)
                    
            self.grid_inventory[pair] = remaining_inventory
            
    def run_rsi_strategy(self, market_data: dict):
        """Run RSI mean reversion for trending_down pairs."""
        if not market_data:
            return
            
        max_positions = 3
        risk_per_trade = 0.03  # 3% risk per trade
        
        # Manage existing RSI positions
        for pair in list(self.active_positions.keys()):
            position = self.active_positions[pair]
            if position.get("strategy") != "rsi":
                continue
                
            position["bars_held"] += 1
            
            if pair in market_data:
                current_price = market_data[pair]["price"]
                df = market_data[pair]["df"]
                config = RSI_CONFIGS.get(pair, RSI_CONFIGS["XBTUSD"])
                current_rsi = self.calculate_rsi(df['close'], config["period"])
                
                should_close = False
                close_reason = ""
                
                # Exit conditions
                if current_rsi >= 50:
                    should_close = True
                    close_reason = f"RSI recovered to {current_rsi:.1f}"
                elif position["bars_held"] >= config["hold"]:
                    should_close = True
                    close_reason = f"Hold time reached ({config['hold']} bars)"
                elif current_price <= position["entry_price"] * 0.97:  # 3% stop loss
                    should_close = True
                    close_reason = "Stop loss hit"
                    
                if should_close:
                    self._close_position(pair, position, current_price, close_reason)
                    
        # Look for new RSI signals
        current_rsi_positions = len([p for p in self.active_positions.values() if p.get("strategy") == "rsi"])
        
        if current_rsi_positions < max_positions:
            for pair in RSI_PAIRS:
                if pair in self.active_positions or pair not in market_data or pair not in RSI_CONFIGS:
                    continue
                    
                data = market_data[pair]
                df = data["df"]
                current_price = data["price"]
                config = RSI_CONFIGS[pair]
                
                current_rsi = self.calculate_rsi(df['close'], config["period"])
                
                # RSI oversold signal (buy the dip in downtrend)
                if current_rsi < config["oversold"]:
                    risk_amount = self.balance * risk_per_trade
                    stop_loss_pct = 0.03
                    qty = risk_amount / (current_price * stop_loss_pct)
                    
                    if ENABLE_LIVE_TRADING:
                        if self._place_market_order(pair, "buy", qty, current_price):
                            self.active_positions[pair] = {
                                "strategy": "rsi",
                                "side": "buy",
                                "entry_price": current_price,
                                "qty": qty,
                                "bars_held": 0,
                                "entry_time": datetime.now().isoformat()
                            }
                    else:
                        self.active_positions[pair] = {
                            "strategy": "rsi",
                            "side": "buy", 
                            "entry_price": current_price,
                            "qty": qty,
                            "bars_held": 0,
                            "entry_time": datetime.now().isoformat()
                        }
                        
                    logger.info(f"[RSI] {pair}: oversold (RSI={current_rsi:.1f}), BUY @ ${current_price:.4f}")
                    break  # One signal per cycle
                    
    def run_momentum_strategy(self, market_data: dict):
        """Run momentum breakout for volatile regime."""
        if not market_data:
            return
            
        max_positions = 3
        risk_per_trade = 0.03
        
        # Manage existing momentum positions
        for pair in list(self.active_positions.keys()):
            position = self.active_positions[pair]
            if position.get("strategy") != "momentum":
                continue
                
            position["bars_held"] += 1
            
            if pair in market_data:
                current_price = market_data[pair]["price"]
                
                # Exit conditions for momentum
                should_close = False
                close_reason = ""
                
                if position["bars_held"] >= 48:  # 48 bar timeout
                    should_close = True
                    close_reason = "Timeout (48 bars)"
                elif position["side"] == "buy" and current_price <= position["stop_loss"]:
                    should_close = True
                    close_reason = "Stop loss hit"
                elif position["side"] == "buy" and current_price >= position["take_profit"]:
                    should_close = True
                    close_reason = "Take profit hit"
                    
                if should_close:
                    self._close_position(pair, position, current_price, close_reason)
                    
        # Look for new momentum signals
        current_momentum_positions = len([p for p in self.active_positions.values() if p.get("strategy") == "momentum"])
        
        if current_momentum_positions < max_positions:
            for pair in MOMENTUM_PAIRS:
                if pair in self.active_positions or pair not in market_data:
                    continue
                    
                data = market_data[pair]
                df = data["df"]
                current_price = data["price"]
                
                if len(df) < 30:
                    continue
                    
                # Calculate indicators
                high_15 = df['high'].rolling(15).max().iloc[-1]
                sma_30 = df['close'].rolling(30).mean().iloc[-1]
                atr = df['close'].diff().abs().rolling(14).mean().iloc[-1]
                
                # Momentum breakout signal
                if (current_price > high_15 and current_price > sma_30 and not pd.isna(atr)):
                    risk_amount = self.balance * risk_per_trade
                    stop_loss = current_price - (2 * atr)
                    take_profit = current_price + (3 * atr)
                    stop_loss_pct = (current_price - stop_loss) / current_price
                    qty = risk_amount / (current_price * stop_loss_pct)
                    
                    if ENABLE_LIVE_TRADING:
                        if self._place_market_order(pair, "buy", qty, current_price):
                            self.active_positions[pair] = {
                                "strategy": "momentum",
                                "side": "buy",
                                "entry_price": current_price,
                                "qty": qty,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "bars_held": 0,
                                "entry_time": datetime.now().isoformat()
                            }
                    else:
                        self.active_positions[pair] = {
                            "strategy": "momentum",
                            "side": "buy",
                            "entry_price": current_price,
                            "qty": qty,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "bars_held": 0,
                            "entry_time": datetime.now().isoformat()
                        }
                        
                    logger.info(f"[MOMENTUM] {pair}: breakout above ${high_15:.4f}, BUY @ ${current_price:.4f}")
                    logger.info(f"[MOMENTUM] SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
                    break
                    
    def _place_market_order(self, pair: str, side: str, qty: float, price: float) -> bool:
        """Place a market order."""
        try:
            min_size = self.client.get_min_order_volume(pair)
            if qty < min_size:
                logger.warning(f"Order size {qty:.6f} below minimum {min_size} for {pair}")
                return False
                
            result = self.client.place_order(
                symbol=pair,
                side=side,
                order_type="market",
                quantity=qty
            )
            
            if result.get("error"):
                logger.error(f"Failed to place {side} order for {pair}: {result['error']}")
                return False
                
            logger.info(f"[ORDER] {pair} {side} {qty:.6f} @ ${price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return False
            
    def _place_limit_order(self, pair: str, side: str, qty: float, price: float) -> bool:
        """Place a limit order."""
        try:
            min_size = self.client.get_min_order_volume(pair)
            if qty < min_size:
                return False
                
            result = self.client.place_order(
                symbol=pair,
                side=side,
                order_type="limit",
                quantity=qty,
                price=price
            )
            
            if result.get("error"):
                logger.error(f"Failed to place limit order: {result['error']}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return False
            
    def get_market_data(self) -> dict:
        """Fetch 1h market data for regime detection and strategy execution."""
        all_pairs = set(GRID_PAIRS + RSI_PAIRS + MOMENTUM_PAIRS)
        market_data = {}
        
        for pair in all_pairs:
            try:
                # Get 1h klines (last 200 bars for regime detection)
                klines = self.client.get_klines(pair, interval="1h", limit=200)
                if not klines:
                    continue
                    
                df = pd.DataFrame(klines)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                if len(df) < 60:  # Need at least 60 bars for regime detection
                    continue
                    
                # Get current ticker
                ticker = self.client.get_ticker(pair)
                current_price = float(ticker["price"]) if ticker else float(df['close'].iloc[-1])
                
                market_data[pair] = {
                    "price": current_price,
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "df": df
                }
                
            except Exception as e:
                logger.error(f"Failed to get data for {pair}: {e}")
                
        return market_data
        
    def log_status(self):
        """Log current bot status - key insight into what the bot is doing and why."""
        # Current regime
        logger.info(f"[REGIME] {self.current_regime}")
        
        # Strategy status
        if self.current_regime == "trending_up":
            grid_pairs_with_inventory = len([p for p in GRID_PAIRS if p in self.grid_inventory and self.grid_inventory[p]])
            total_inventory = sum(len(inv) for inv in self.grid_inventory.values())
            logger.info(f"[GRID] Active on {grid_pairs_with_inventory}/{len(GRID_PAIRS)} pairs, {total_inventory} open positions")
            
        elif self.current_regime == "trending_down":
            rsi_positions = len([p for p in self.active_positions.values() if p.get("strategy") == "rsi"])
            logger.info(f"[RSI] {rsi_positions} active positions (max 3)")
            
        elif self.current_regime == "volatile":
            momentum_positions = len([p for p in self.active_positions.values() if p.get("strategy") == "momentum"])
            logger.info(f"[MOMENTUM] {momentum_positions} active positions (max 3)")
            
        elif self.current_regime == "ranging":
            logger.info("[CASH] Market is ranging, staying flat")
            
        # P&L summary
        total_profit = self.grid_profit + self.active_profit
        total_return = (total_profit / STARTING_BALANCE) * 100 if STARTING_BALANCE > 0 else 0
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        logger.info(f"[P&L] Grid: ${self.grid_profit:.2f} | Active: ${self.active_profit:.2f} | Total: ${total_profit:.2f} ({total_return:+.1f}%)")
        if self.total_trades > 0:
            logger.info(f"[STATS] {self.total_trades} trades, {self.winning_trades} wins ({win_rate:.1f}% win rate)")
            
    def run_cycle(self):
        """Run one complete trading cycle."""
        try:
            logger.info("=" * 50)
            logger.info("Starting trading cycle...")
            
            # 1. Get market data (1h bars for regime detection)
            market_data = self.get_market_data()
            if not market_data:
                logger.warning("No market data received, skipping cycle")
                return
                
            # 2. Detect regime PER PAIR — each coin moves independently
            pair_regimes = {}
            for pair, data in market_data.items():
                regime = self.detect_regime(data["df"])
                pair_regimes[pair] = regime
                
                # Track regime transitions per pair
                prev = self.state.get("pair_regimes", {}).get(pair, "unknown")
                if prev != regime and prev != "unknown":
                    logger.info(f"[SWITCH] {pair}: {prev} → {regime}")
                    # Close any positions for this pair from the old strategy
                    if pair in self.active_positions:
                        pos = self.active_positions[pair]
                        self._close_position(pair, pos, data["price"],
                                           f"Regime switch {prev}→{regime}")
            
            self.state["pair_regimes"] = pair_regimes
            
            regime_summary = " ".join(f"{p}={r}" for p, r in sorted(pair_regimes.items()))
            logger.info(f"[REGIME] {regime_summary}")
            
            # Count regimes
            regime_counts = {}
            for r in pair_regimes.values():
                regime_counts[r] = regime_counts.get(r, 0) + 1
            logger.info(f"[REGIME DIST] {regime_counts}")
            
            # 3. For each pair, run the strategy that matches ITS regime
            #    Grid pairs: only those in trending_up
            #    RSI pairs: only those in trending_down
            #    Momentum pairs: only those in volatile
            #    Ranging pairs: stay flat
            
            grid_pairs = {p: d for p, d in market_data.items() 
                         if pair_regimes.get(p) == "trending_up"}
            rsi_pairs = {p: d for p, d in market_data.items()
                        if pair_regimes.get(p) == "trending_down"}
            momentum_pairs = {p: d for p, d in market_data.items()
                            if pair_regimes.get(p) == "volatile"}
            cash_pairs = {p: d for p, d in market_data.items()
                         if pair_regimes.get(p) == "ranging"}
            
            # Close positions for pairs that switched to ranging
            for pair in cash_pairs:
                if pair in self.active_positions:
                    self._close_position(pair, self.active_positions[pair],
                                       cash_pairs[pair]["price"],
                                       "Regime→ranging, going flat")
            
            # Run strategies on their respective pairs
            if grid_pairs:
                self.run_grid_strategy(grid_pairs)
            if rsi_pairs:
                self.run_rsi_strategy(rsi_pairs)
            if momentum_pairs:
                self.run_momentum_strategy(momentum_pairs)
            
            if not grid_pairs and not rsi_pairs and not momentum_pairs:
                logger.info("[CASH MODE] All pairs ranging, staying flat")
                
            # 5. Log status (human-readable decision log)
            self.log_status()
            
            # 6. Save state
            self.save_state()
            
            logger.info("Trading cycle completed")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def run(self):
        """Main bot loop with signal handling for clean shutdown."""
        logger.info("🦞 ADAPTIVE MASTER BOT STARTING...")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        logger.info(f"Starting balance: ${STARTING_BALANCE}")
        
        # Signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received, stopping bot...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                cycle_start = time.time()
                
                self.run_cycle()
                
                # Sleep for remaining interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, CHECK_INTERVAL - cycle_time)
                
                if sleep_time > 0:
                    logger.info(f"Cycle took {cycle_time:.1f}s, sleeping {sleep_time:.1f}s")
                    logger.info("-" * 50)
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Cycle took {cycle_time:.1f}s, longer than {CHECK_INTERVAL}s interval!")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info("Saving final state...")
            self.save_state()
            logger.info("🦞 ADAPTIVE MASTER BOT STOPPED")


if __name__ == "__main__":
    bot = AdaptiveBot()
    bot.run()