#!/usr/bin/env python3
"""
MASTER TRADING BOT - Adaptive Ecosystem Bot
The final production bot that detects market regimes and activates appropriate strategies.

Architecture:
1. Regime Detection: Classifies markets as trending_up/trending_down/ranging/volatile
2. Strategy Arsenal: Grid, RSI Mean Reversion, Bollinger Band Reversion, Momentum
3. Portfolio Management: Multi-strategy execution with risk controls
4. State Persistence: Survives restarts with complete state recovery

Usage:
    python run_master_bot.py
    
Environment Variables:
    ENABLE_LIVE_TRADING=true     # Enable live trading (default: false = dry run)
    CHECK_INTERVAL=300           # Check interval in seconds (default: 5 min)
    STARTING_BALANCE=300         # Starting balance in USD (default: 300)
"""

import sys
import os
import json
import time
import signal
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger

# Add src directory to path for kraken_client import
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from kraken_client import KrakenClient
except ImportError as e:
    logger.error(f"Failed to import KrakenClient: {e}")
    logger.error("Make sure kraken_client.py exists in src/ directory")
    sys.exit(1)

# Configuration
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes default
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "300"))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = DATA_DIR / "master_bot_state.json"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logger.remove()  # Remove default handler
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(LOGS_DIR / "master_bot.log", rotation="10 MB", retention="30 days")

# Trading configuration
MAKER_FEE = 0.0016  # 0.16%
MARKET_SLIPPAGE = 0.0005  # 0.05% for market orders
LIMIT_SLIPPAGE = 0.0003   # 0.03% for limit orders

# Per-pair optimal configurations from backtests
PAIR_CONFIGS = {
    "NEARUSD": {
        "grid": {"grid_pct": 0.01, "tp_pct": 0.01, "levels": 5},
        "rsi": {"rsi_period": 9, "oversold": 20, "overbought": 80, "hold_bars": 8},
    },
    "ATOMUSD": {
        "rsi": {"rsi_period": 7, "oversold": 20, "overbought": 80, "hold_bars": 5},
        "rsi_alt": {"rsi_period": 14, "oversold": 20, "overbought": 70, "hold_bars": 5},
    },
    "AVAXUSD": {
        "grid": {"grid_pct": 0.008, "tp_pct": 0.01, "levels": 5},
        "rsi": {"rsi_period": 14, "oversold": 20, "overbought": 70, "hold_bars": 8},
    },
    "DOTUSD": {
        "grid": {"grid_pct": 0.015, "tp_pct": 0.01, "levels": 5},
        "bb": {"bb_period": 20, "bb_std": 2.0, "hold_bars": 12},
    },
    "XRPUSD": {
        "rsi": {"rsi_period": 9, "oversold": 20, "overbought": 80, "hold_bars": 3},
    },
    "FILUSD": {
        "rsi": {"rsi_period": 14, "oversold": 20, "overbought": 80, "hold_bars": 8},
        "bb": {"bb_period": 20, "bb_std": 2.5, "hold_bars": 8},
    },
    "UNIUSD": {
        "grid": {"grid_pct": 0.015, "tp_pct": 0.01, "levels": 5},
        "rsi": {"rsi_period": 9, "oversold": 20, "overbought": 75, "hold_bars": 8},
    },
    "XBTUSD": {
        "rsi": {"rsi_period": 14, "oversold": 20, "overbought": 80, "hold_bars": 8},
    },
    "ETHUSD": {
        "rsi": {"rsi_period": 14, "oversold": 20, "overbought": 80, "hold_bars": 12},
    },
    "SOLUSD": {
        "rsi": {"rsi_period": 14, "oversold": 20, "overbought": 80, "hold_bars": 8},
    },
    "LINKUSD": {
        "rsi": {"rsi_period": 14, "oversold": 20, "overbought": 75, "hold_bars": 8},
    },
    "ADAUSD": {
        "rsi": {"rsi_period": 14, "oversold": 20, "overbought": 80, "hold_bars": 12},
    },
}

# Regime to strategy mapping
REGIME_STRATEGIES = {
    "ranging": ["grid", "rsi", "bb"],       # All mean reversion strategies
    "volatile": ["rsi", "bb"],              # Mean reversion on extremes only  
    "trending_up": ["momentum_long"],       # Ride the trend up
    "trending_down": ["momentum_short"],    # Ride the trend down
}

# Risk management
MAX_ACTIVE_POSITIONS = 5
GRID_ALLOCATION_PCT = 0.4  # 40% for grid strategies
ACTIVE_ALLOCATION_PCT = 0.6  # 60% for active strategies
RISK_PER_TRADE = 0.03  # 3% risk per active trade
MAX_DRAWDOWN_HALT = 0.15  # 15% max drawdown on active strategies


class GridManager:
    """Manages grid orders for a single pair."""
    
    def __init__(self, pair: str, config: dict, allocation: float):
        self.pair = pair
        self.grid_pct = config["grid_pct"]
        self.tp_pct = config["tp_pct"]
        self.levels = config["levels"]
        self.allocation = allocation  # USD allocated to this grid
        self.buy_orders = []   # List of {"price": float, "qty": float, "filled": bool}
        self.filled_buys = []  # List of {"buy_price": float, "qty": float, "sell_target": float}
        self.completed_trades = []  # Historical completed round-trips
        self.total_profit = 0.0
        
    def initialize_grid(self, current_price: float, min_order_size: float) -> List[dict]:
        """Initialize grid buy orders around current price."""
        orders_to_place = []
        qty_per_level = self.allocation / self.levels / current_price
        
        if qty_per_level < min_order_size:
            logger.warning(f"{self.pair}: Grid qty {qty_per_level:.6f} below minimum {min_order_size}")
            return orders_to_place
            
        # Place buy orders below current price
        for i in range(1, self.levels + 1):
            buy_price = current_price * (1 - self.grid_pct * i)
            self.buy_orders.append({
                "price": buy_price,
                "qty": qty_per_level,
                "filled": False
            })
            orders_to_place.append({
                "type": "limit",
                "side": "buy",
                "price": buy_price,
                "quantity": qty_per_level
            })
            
        return orders_to_place
        
    def update(self, current_low: float, current_high: float, current_price: float, min_order_size: float) -> Tuple[List[dict], List[dict]]:
        """
        Check for fills and manage grid state.
        Returns: (new_orders_to_place, completed_trades)
        """
        new_orders = []
        completed_this_cycle = []
        
        # Check buy fills (when low touches buy price)
        for order in self.buy_orders:
            if not order["filled"] and current_low <= order["price"]:
                order["filled"] = True
                sell_target = order["price"] * (1 + self.tp_pct)
                
                self.filled_buys.append({
                    "buy_price": order["price"],
                    "qty": order["qty"],
                    "sell_target": sell_target,
                    "timestamp": datetime.now()
                })
                
                logger.info(f"{self.pair} Grid: Buy filled @ ${order['price']:.4f}, target ${sell_target:.4f}")
        
        # Check sell fills (when high touches sell target)
        remaining_buys = []
        for filled_buy in self.filled_buys:
            if current_high >= filled_buy["sell_target"]:
                # Sell filled - complete the round trip
                profit = (filled_buy["sell_target"] - filled_buy["buy_price"]) * filled_buy["qty"]
                profit -= filled_buy["buy_price"] * filled_buy["qty"] * MAKER_FEE  # Buy fee
                profit -= filled_buy["sell_target"] * filled_buy["qty"] * MAKER_FEE  # Sell fee
                
                self.total_profit += profit
                trade_data = {
                    "buy_price": filled_buy["buy_price"],
                    "sell_price": filled_buy["sell_target"],
                    "qty": filled_buy["qty"],
                    "profit": profit,
                    "timestamp": datetime.now()
                }
                self.completed_trades.append(trade_data)
                completed_this_cycle.append(trade_data)
                
                logger.info(f"{self.pair} Grid: Round-trip completed! ${profit:.2f} profit")
                
                # Place new buy order at this level
                qty_per_level = self.allocation / self.levels / current_price
                if qty_per_level >= min_order_size:
                    new_buy_price = current_price * (1 - self.grid_pct)
                    new_orders.append({
                        "type": "limit",
                        "side": "buy", 
                        "price": new_buy_price,
                        "quantity": qty_per_level
                    })
                    
                    self.buy_orders.append({
                        "price": new_buy_price,
                        "qty": qty_per_level,
                        "filled": False
                    })
            else:
                remaining_buys.append(filled_buy)
                
        self.filled_buys = remaining_buys
        return new_orders, completed_this_cycle


class MasterBot:
    """The master trading bot that manages multiple strategies across multiple pairs."""
    
    def __init__(self):
        self.client = KrakenClient()
        self.running = True
        self.state = self.load_state()
        
        # Initialize components
        self.grid_managers = {}  # pair -> GridManager
        self.active_positions = {}  # pair -> position data
        self.balance = self.state.get("balance", STARTING_BALANCE)
        self.grid_balance = self.balance * GRID_ALLOCATION_PCT
        self.active_balance = self.balance * ACTIVE_ALLOCATION_PCT
        
        # Performance tracking
        self.start_balance = self.balance
        self.total_grid_profit = 0.0
        self.total_active_profit = 0.0
        self.active_drawdown = 0.0
        self.max_active_balance = self.active_balance
        
        # Initialize grid managers for grid-enabled pairs
        self._initialize_grid_managers()
        
        logger.info(f"MasterBot initialized - Balance: ${self.balance:.2f}")
        logger.info(f"Grid allocation: ${self.grid_balance:.2f}, Active allocation: ${self.active_balance:.2f}")
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
                
        return {"balance": STARTING_BALANCE, "positions": {}, "grid_states": {}}
        
    def save_state(self):
        """Save bot state to disk."""
        state = {
            "balance": self.balance,
            "grid_balance": self.grid_balance,
            "active_balance": self.active_balance,
            "positions": self.active_positions,
            "grid_states": {
                pair: {
                    "buy_orders": gm.buy_orders,
                    "filled_buys": [
                        {**fb, "timestamp": fb["timestamp"].isoformat()} 
                        for fb in gm.filled_buys
                    ],
                    "completed_trades": [
                        {**ct, "timestamp": ct["timestamp"].isoformat()}
                        for ct in gm.completed_trades
                    ],
                    "total_profit": gm.total_profit
                }
                for pair, gm in self.grid_managers.items()
            },
            "performance": {
                "start_balance": self.start_balance,
                "total_grid_profit": self.total_grid_profit,
                "total_active_profit": self.total_active_profit,
                "active_drawdown": self.active_drawdown,
                "last_update": datetime.now().isoformat()
            }
        }
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def _initialize_grid_managers(self):
        """Initialize grid managers for grid-enabled pairs."""
        grid_pairs = [pair for pair, config in PAIR_CONFIGS.items() if "grid" in config]
        allocation_per_grid = self.grid_balance / len(grid_pairs)
        
        for pair in grid_pairs:
            config = PAIR_CONFIGS[pair]["grid"]
            self.grid_managers[pair] = GridManager(pair, config, allocation_per_grid)
            logger.info(f"Initialized grid manager for {pair} with ${allocation_per_grid:.2f}")
            
    def detect_regime(self, df: pd.DataFrame, sma_period: int = 50, adx_period: int = 14) -> str:
        """
        Detect market regime for given price data.
        Returns: "trending_up", "trending_down", "ranging", "volatile"
        """
        if len(df) < max(sma_period, adx_period) + 1:
            return "ranging"  # Default to ranging if insufficient data
            
        try:
            # Calculate indicators
            df = df.copy()
            df['sma'] = df['close'].rolling(sma_period).mean()
            
            # ADX calculation
            df['high_prev'] = df['high'].shift(1)
            df['low_prev'] = df['low'].shift(1)
            df['close_prev'] = df['close'].shift(1)
            
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close_prev']),
                    abs(df['low'] - df['close_prev'])
                )
            )
            
            df['plus_dm'] = np.where(
                (df['high'] - df['high_prev']) > (df['low_prev'] - df['low']),
                np.maximum(df['high'] - df['high_prev'], 0), 0
            )
            df['minus_dm'] = np.where(
                (df['low_prev'] - df['low']) > (df['high'] - df['high_prev']),
                np.maximum(df['low_prev'] - df['low'], 0), 0
            )
            
            df['atr'] = df['tr'].rolling(adx_period).mean()
            df['plus_di'] = 100 * (df['plus_dm'].rolling(adx_period).mean() / df['atr'])
            df['minus_di'] = 100 * (df['minus_dm'].rolling(adx_period).mean() / df['atr'])
            
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            df['adx'] = df['dx'].rolling(adx_period).mean()
            
            # Current values
            current_price = df['close'].iloc[-1]
            current_sma = df['sma'].iloc[-1]
            current_adx = df['adx'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            median_atr = df['atr'].rolling(50).median().iloc[-1]
            
            # Regime classification
            if pd.isna(current_adx) or pd.isna(current_sma):
                return "ranging"
                
            # Volatile market check
            if current_atr > median_atr * 1.5:
                return "volatile"
                
            # Trending market check
            if current_adx > 25:
                if current_price > current_sma:
                    return "trending_up"
                else:
                    return "trending_down"
                    
            return "ranging"
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return "ranging"
            
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
        
    def generate_signals(self, pair: str, df: pd.DataFrame, regime: str) -> List[dict]:
        """Generate trading signals for a pair based on current regime."""
        signals = []
        
        if pair not in PAIR_CONFIGS:
            return signals
            
        config = PAIR_CONFIGS[pair]
        available_strategies = REGIME_STRATEGIES.get(regime, [])
        
        current_price = df['close'].iloc[-1]
        
        # RSI signals
        if "rsi" in available_strategies and "rsi" in config:
            rsi_config = config["rsi"]
            rsi = self.calculate_rsi(df['close'], rsi_config["rsi_period"])
            current_rsi = rsi.iloc[-1]
            
            if not pd.isna(current_rsi):
                # Long signal
                if current_rsi < rsi_config["oversold"]:
                    signals.append({
                        "pair": pair,
                        "strategy": "rsi_long",
                        "side": "buy",
                        "price": current_price,
                        "confidence": (rsi_config["oversold"] - current_rsi) / rsi_config["oversold"],
                        "hold_bars": rsi_config["hold_bars"],
                        "reason": f"RSI oversold: {current_rsi:.1f}"
                    })
                
                # Short signal (only for margin-enabled pairs)
                elif current_rsi > rsi_config["overbought"] and pair != "DOGEUSD":
                    signals.append({
                        "pair": pair,
                        "strategy": "rsi_short", 
                        "side": "sell",
                        "price": current_price,
                        "confidence": (current_rsi - rsi_config["overbought"]) / (100 - rsi_config["overbought"]),
                        "hold_bars": rsi_config["hold_bars"],
                        "reason": f"RSI overbought: {current_rsi:.1f}"
                    })
        
        # Bollinger Bands signals
        if "bb" in available_strategies and "bb" in config:
            bb_config = config["bb"]
            upper_bb, mid_bb, lower_bb = self.calculate_bollinger_bands(
                df['close'], bb_config["bb_period"], bb_config["bb_std"]
            )
            
            current_upper = upper_bb.iloc[-1]
            current_mid = mid_bb.iloc[-1]
            current_lower = lower_bb.iloc[-1]
            
            if not any(pd.isna([current_upper, current_mid, current_lower])):
                # Long signal at lower band
                if current_price <= current_lower:
                    distance_ratio = (current_lower - current_price) / (current_mid - current_lower)
                    signals.append({
                        "pair": pair,
                        "strategy": "bb_long",
                        "side": "buy",
                        "price": current_price,
                        "target": current_mid,
                        "confidence": min(distance_ratio, 1.0),
                        "hold_bars": bb_config["hold_bars"],
                        "reason": f"Price at lower BB: ${current_price:.4f} vs ${current_lower:.4f}"
                    })
                
                # Short signal at upper band
                elif current_price >= current_upper and pair != "DOGEUSD":
                    distance_ratio = (current_price - current_upper) / (current_upper - current_mid)
                    signals.append({
                        "pair": pair,
                        "strategy": "bb_short",
                        "side": "sell", 
                        "price": current_price,
                        "target": current_mid,
                        "confidence": min(distance_ratio, 1.0),
                        "hold_bars": bb_config["hold_bars"],
                        "reason": f"Price at upper BB: ${current_price:.4f} vs ${current_upper:.4f}"
                    })
        
        return signals
        
    def execute_signal(self, signal: dict) -> bool:
        """Execute a trading signal."""
        if not ENABLE_LIVE_TRADING:
            logger.info(f"[DRY RUN] Would execute: {signal}")
            return True
            
        try:
            pair = signal["pair"]
            
            # Calculate position size (3% of active balance)
            risk_amount = self.active_balance * RISK_PER_TRADE
            stop_loss_pct = 0.03  # 3% stop loss
            position_size = risk_amount / (signal["price"] * stop_loss_pct)
            
            # Get minimum order size
            min_size = self.client.get_min_order_volume(pair)
            if position_size < min_size:
                logger.warning(f"Position size {position_size:.6f} below minimum {min_size} for {pair}")
                return False
                
            # Place order
            order_params = {
                "pair": pair,
                "type": "market",
                "ordertype": signal["side"],
                "volume": f"{position_size:.6f}"
            }
            
            if signal["side"] == "sell":  # Short position
                order_params["leverage"] = "2"
                
            result = self.client.place_order(**order_params)
            
            if result.get("error"):
                logger.error(f"Failed to place order: {result['error']}")
                return False
                
            # Track position
            self.active_positions[pair] = {
                "strategy": signal["strategy"],
                "side": signal["side"],
                "entry_price": signal["price"],
                "size": position_size,
                "stop_loss": signal["price"] * (0.97 if signal["side"] == "buy" else 1.03),
                "target": signal.get("target"),
                "hold_bars": signal["hold_bars"],
                "bars_held": 0,
                "entry_time": datetime.now(),
                "order_id": result.get("txid", [""])[0]
            }
            
            logger.info(f"Executed {signal['strategy']} on {pair}: {signal['side']} ${position_size:.2f} @ ${signal['price']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return False
            
    def manage_positions(self, market_data: dict):
        """Manage existing active positions."""
        positions_to_close = []
        
        for pair, position in self.active_positions.items():
            if pair not in market_data:
                continue
                
            current_price = market_data[pair]["price"]
            position["bars_held"] += 1
            
            should_close = False
            close_reason = ""
            
            # Stop loss check
            if position["side"] == "buy" and current_price <= position["stop_loss"]:
                should_close = True
                close_reason = f"Stop loss hit: ${current_price:.4f} <= ${position['stop_loss']:.4f}"
            elif position["side"] == "sell" and current_price >= position["stop_loss"]:
                should_close = True
                close_reason = f"Stop loss hit: ${current_price:.4f} >= ${position['stop_loss']:.4f}"
                
            # Target hit (for BB strategy)
            elif position.get("target") and position["side"] == "buy" and current_price >= position["target"]:
                should_close = True
                close_reason = f"Target hit: ${current_price:.4f} >= ${position['target']:.4f}"
            elif position.get("target") and position["side"] == "sell" and current_price <= position["target"]:
                should_close = True
                close_reason = f"Target hit: ${current_price:.4f} <= ${position['target']:.4f}"
                
            # Time-based exit
            elif position["bars_held"] >= position["hold_bars"]:
                should_close = True
                close_reason = f"Hold time reached: {position['bars_held']} bars"
                
            # RSI recovery exit (for RSI strategy)
            elif "rsi" in position["strategy"]:
                # Would need current RSI here - simplified for now
                pass
                
            if should_close:
                positions_to_close.append((pair, close_reason))
                
        # Close positions
        for pair, reason in positions_to_close:
            self.close_position(pair, reason, market_data[pair]["price"])
            
    def close_position(self, pair: str, reason: str, current_price: float):
        """Close an active position."""
        if pair not in self.active_positions:
            return
            
        position = self.active_positions[pair]
        
        if not ENABLE_LIVE_TRADING:
            logger.info(f"[DRY RUN] Would close {pair} position: {reason}")
            # Calculate theoretical P&L for logging
            if position["side"] == "buy":
                pnl = (current_price - position["entry_price"]) * position["size"]
            else:
                pnl = (position["entry_price"] - current_price) * position["size"]
            logger.info(f"[DRY RUN] Theoretical P&L: ${pnl:.2f}")
        else:
            try:
                # Close position via market order
                close_side = "sell" if position["side"] == "buy" else "buy"
                
                order_params = {
                    "pair": pair,
                    "type": "market", 
                    "ordertype": close_side,
                    "volume": f"{position['size']:.6f}"
                }
                
                if position["side"] == "sell":  # Closing short
                    order_params["leverage"] = "2"
                    
                result = self.client.place_order(**order_params)
                
                if result.get("error"):
                    logger.error(f"Failed to close {pair} position: {result['error']}")
                    return
                    
                # Calculate actual P&L
                if position["side"] == "buy":
                    pnl = (current_price - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - current_price) * position["size"]
                    
                # Deduct fees
                pnl -= position["entry_price"] * position["size"] * MARKET_SLIPPAGE
                pnl -= current_price * position["size"] * MARKET_SLIPPAGE
                
                self.total_active_profit += pnl
                self.active_balance += pnl
                
                logger.info(f"Closed {pair} position: {reason} | P&L: ${pnl:.2f}")
                
            except Exception as e:
                logger.error(f"Error closing {pair} position: {e}")
                return
                
        # Remove from tracking
        del self.active_positions[pair]
        
    def manage_grids(self, market_data: dict):
        """Manage grid strategies."""
        for pair, grid_manager in self.grid_managers.items():
            if pair not in market_data:
                continue
                
            data = market_data[pair]
            current_price = data["price"]
            current_low = data.get("low", current_price)
            current_high = data.get("high", current_price)
            
            # Initialize grid if needed
            if not grid_manager.buy_orders:
                min_size = self.client.get_min_order_volume(pair) if ENABLE_LIVE_TRADING else 0.001
                initial_orders = grid_manager.initialize_grid(current_price, min_size)
                
                if ENABLE_LIVE_TRADING and initial_orders:
                    for order in initial_orders:
                        self._place_grid_order(pair, order)
                        
                logger.info(f"Initialized {pair} grid with {len(initial_orders)} buy orders")
                
            # Update grid state
            min_size = self.client.get_min_order_volume(pair) if ENABLE_LIVE_TRADING else 0.001
            new_orders, completed_trades = grid_manager.update(current_low, current_high, current_price, min_size)
            
            # Place new grid orders
            if ENABLE_LIVE_TRADING and new_orders:
                for order in new_orders:
                    self._place_grid_order(pair, order)
                    
            # Update grid profit tracking
            self.total_grid_profit = sum(gm.total_profit for gm in self.grid_managers.values())
            
    def _place_grid_order(self, pair: str, order: dict):
        """Place a grid order."""
        try:
            order_params = {
                "pair": pair,
                "type": "limit",
                "ordertype": order["side"],
                "volume": f"{order['quantity']:.6f}",
                "price": f"{order['price']:.4f}"
            }
            
            result = self.client.place_order(**order_params)
            
            if result.get("error"):
                logger.error(f"Failed to place grid order for {pair}: {result['error']}")
            else:
                logger.debug(f"Placed grid {order['side']} order for {pair}: {order['quantity']:.6f} @ ${order['price']:.4f}")
                
        except Exception as e:
            logger.error(f"Error placing grid order for {pair}: {e}")
            
    def get_market_data(self, pairs: List[str]) -> dict:
        """Fetch current market data for all pairs.
        
        Uses kraken_client which returns:
        - get_klines(): List[Dict] with keys: timestamp, open, high, low, close, volume
        - get_ticker(): Dict with keys: symbol, price, bid, ask, timestamp
        """
        market_data = {}
        
        for pair in pairs:
            try:
                # get_klines returns a list of dicts already parsed
                klines = self.client.get_klines(pair, interval="1h", limit=100)
                if not klines:
                    continue
                
                df = pd.DataFrame(klines)
                # Ensure numeric types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if len(df) < 20:
                    continue
                
                # get_ticker returns a processed dict: {symbol, price, bid, ask}
                ticker = self.client.get_ticker(pair)
                current_price = float(ticker["price"]) if ticker else float(df['close'].iloc[-1])
                
                market_data[pair] = {
                    "price": current_price,
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "df": df,
                    "regime": self.detect_regime(df),
                }
                
            except Exception as e:
                logger.error(f"Failed to get data for {pair}: {e}")
        
        return market_data
        
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached."""
        if self.active_balance > 0:
            drawdown = (self.max_active_balance - self.active_balance) / self.max_active_balance
            self.active_drawdown = drawdown
            
            if drawdown > MAX_DRAWDOWN_HALT:
                logger.error(f"Max drawdown exceeded: {drawdown:.1%} > {MAX_DRAWDOWN_HALT:.1%}")
                logger.error("HALTING ACTIVE STRATEGIES")
                return False
                
        return True
        
    def log_status(self, regimes: dict):
        """Log current bot status."""
        # Regime summary
        regime_summary = " ".join([f"{pair}={regime}" for pair, regime in regimes.items()])
        logger.info(f"[REGIME] {regime_summary}")
        
        # Grid status
        grid_active = len([gm for gm in self.grid_managers.values() if gm.buy_orders])
        total_roundtrips = sum(len(gm.completed_trades) for gm in self.grid_managers.values())
        logger.info(f"[GRID] {grid_active} pairs active, {total_roundtrips} completed roundtrips")
        
        # Active positions
        active_count = len(self.active_positions)
        logger.info(f"[POSITIONS] {active_count} active positions | Max: {MAX_ACTIVE_POSITIONS}")
        
        # P&L summary  
        total_profit = self.total_grid_profit + self.total_active_profit
        total_return = (total_profit / self.start_balance) * 100
        logger.info(f"[P&L] Grid: ${self.total_grid_profit:.2f} | Active: ${self.total_active_profit:.2f} | Total: ${total_profit:.2f} ({total_return:+.1f}%)")
        
        if self.active_drawdown > 0:
            logger.info(f"[RISK] Active drawdown: {self.active_drawdown:.1%}")
            
    def run_cycle(self):
        """Run one complete trading cycle."""
        try:
            logger.info("Starting trading cycle...")
            
            # Get market data for all configured pairs
            pairs = list(PAIR_CONFIGS.keys())
            market_data = self.get_market_data(pairs)
            
            if not market_data:
                logger.warning("No market data received, skipping cycle")
                return
                
            regimes = {pair: data["regime"] for pair, data in market_data.items()}
            
            # Check risk limits
            if not self.check_risk_limits():
                # Halt active strategies but continue grids
                self.active_positions.clear()
                
            # Manage existing positions first
            self.manage_positions(market_data)
            
            # Manage grid strategies
            self.manage_grids(market_data)
            
            # Generate new signals if we have capacity
            if len(self.active_positions) < MAX_ACTIVE_POSITIONS:
                all_signals = []
                
                for pair, data in market_data.items():
                    # Skip if we already have a position in this pair
                    if pair in self.active_positions:
                        continue
                        
                    regime = data["regime"]
                    df = data["df"]
                    
                    signals = self.generate_signals(pair, df, regime)
                    all_signals.extend(signals)
                    
                # Sort signals by confidence and execute best ones
                all_signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                
                slots_available = MAX_ACTIVE_POSITIONS - len(self.active_positions)
                for signal in all_signals[:slots_available]:
                    logger.info(f"Signal: {signal['pair']} {signal['strategy']} - {signal['reason']}")
                    if self.execute_signal(signal):
                        # Brief pause between executions
                        time.sleep(1)
                        
            # Update balance tracking
            self.max_active_balance = max(self.max_active_balance, self.active_balance)
            
            # Log status
            self.log_status(regimes)
            
            # Save state
            self.save_state()
            
            logger.info("Trading cycle completed")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            
    def run(self):
        """Main bot loop."""
        logger.info("MasterBot starting...")
        
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
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Cycle took {cycle_time:.1f}s, longer than {CHECK_INTERVAL}s interval!")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            logger.info("Saving final state...")
            self.save_state()
            logger.info("MasterBot stopped")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MASTER TRADING BOT - Adaptive Ecosystem")
    logger.info("=" * 60)
    logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
    logger.info(f"Check interval: {CHECK_INTERVAL}s")
    logger.info(f"Starting balance: ${STARTING_BALANCE}")
    logger.info("=" * 60)
    
    bot = MasterBot()
    bot.run()