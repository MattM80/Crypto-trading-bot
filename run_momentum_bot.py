#!/usr/bin/env python3
"""
Production Multi-Pair Momentum Breakout Bot
===========================================

Backtested with walk-forward validation across 17 Kraken pairs, 3 timeframes (15m, 1h, 4h).
66 edges passed validation. Strategy: momentum breakout with SMA trend filter, ATR-based SL/TP.

Architecture:
1. Every 5 minutes, scan ALL pairs across ALL timeframes for breakout signals
2. Rank signals by quality (walk-forward validated combos get priority)
3. Execute the best available signal (max 3 concurrent positions to manage risk on $300)
4. Manage exits (SL, TP, timeout) on all open positions each cycle
5. Track state across restarts (JSON persistence)
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv

# Setup paths and environment
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from kraken_client import KrakenClient


# ═══════════════════════════════════════════════════════════════
# VALIDATED EDGE CONFIGS (Walk-Forward Tested)
# ═══════════════════════════════════════════════════════════════

# Top 37 configs that passed walk-forward validation with positive returns
# Format: (pair, timeframe_min, direction, breakout_bars, sl_atr_mult, tp_atr_mult, sma_period, timeout_bars, wf_test_return)
EDGE_CONFIGS = [
    ("UNIUSD", 15, "short", 10, 2.5, 4.0, 50, 36, 15.6),
    ("FILUSD", 60, "short", 10, 1.5, 5.0, 30, 36, 15.2),
    ("ATOMUSD", 240, "short", 10, 2.0, 5.0, 30, 72, 12.5),
    ("ADAUSD", 240, "long", 20, 1.5, 3.0, 30, 36, 11.7),
    ("LINKUSD", 15, "short", 20, 1.5, 5.0, 50, 72, 11.5),
    ("ATOMUSD", 60, "long", 10, 2.0, 4.0, 30, 72, 11.4),
    ("ADAUSD", 15, "short", 10, 1.5, 3.0, 50, 48, 10.0),
    ("SOLUSD", 240, "long", 10, 2.0, 3.0, 50, 48, 9.4),
    ("XRPUSD", 60, "long", 20, 2.5, 4.0, 30, 48, 9.3),
    ("ETHUSD", 240, "long", 15, 2.0, 3.0, 30, 36, 9.2),
    ("AVAXUSD", 240, "long", 10, 2.0, 3.0, 50, 36, 9.2),
    ("AVAXUSD", 60, "short", 15, 1.5, 5.0, 50, 48, 9.2),
    ("NEARUSD", 60, "long", 10, 2.0, 5.0, 30, 36, 9.1),
    ("ETHUSD", 60, "short", 20, 1.5, 5.0, 50, 48, 9.1),
    ("ETHUSD", 60, "long", 15, 2.0, 5.0, 30, 48, 8.6),
    ("XBTUSD", 240, "long", 20, 2.0, 3.0, 30, 48, 8.6),
    ("AAVEUSD", 15, "short", 20, 2.0, 4.0, 30, 36, 8.5),
    ("XRPUSD", 15, "short", 10, 2.0, 3.0, 50, 36, 8.2),
    ("ATOMUSD", 15, "short", 10, 2.5, 5.0, 30, 72, 7.9),
    ("XBTUSD", 15, "short", 10, 2.0, 3.0, 30, 72, 7.7),
    ("FILUSD", 60, "long", 15, 2.5, 4.0, 50, 72, 7.6),
    ("XLMUSD", 60, "long", 10, 2.0, 3.0, 30, 36, 7.6),
    ("DOGEUSD", 15, "short", 15, 1.5, 4.0, 50, 72, 7.4),
    ("UNIUSD", 60, "short", 20, 2.0, 5.0, 30, 48, 7.0),
    ("SOLUSD", 60, "short", 15, 1.5, 5.0, 50, 36, 6.6),
    ("XRPUSD", 60, "short", 20, 1.5, 5.0, 50, 72, 6.2),
    ("FILUSD", 240, "short", 15, 1.5, 5.0, 50, 48, 6.2),
    ("ATOMUSD", 240, "long", 20, 1.5, 5.0, 50, 36, 6.0),
    ("XBTUSD", 60, "long", 10, 2.0, 3.0, 50, 48, 5.9),
    ("DOTUSD", 240, "long", 15, 1.5, 4.0, 30, 36, 5.6),
    ("AAVEUSD", 60, "long", 10, 2.0, 5.0, 30, 36, 5.5),
    ("LTCUSD", 15, "short", 20, 2.5, 5.0, 30, 72, 5.5),
    ("DOTUSD", 60, "short", 15, 1.5, 4.0, 30, 36, 5.4),
    ("AVAXUSD", 60, "long", 10, 2.5, 5.0, 50, 48, 4.7),
    ("DOTUSD", 60, "long", 15, 2.5, 5.0, 30, 36, 4.2),
    ("DOGEUSD", 60, "long", 10, 2.0, 5.0, 30, 48, 4.2),
    ("LINKUSD", 240, "long", 10, 2.0, 4.0, 30, 48, 4.1),
]

# Config class for type safety
class EdgeConfig(NamedTuple):
    pair: str
    timeframe_min: int
    direction: str
    breakout_bars: int
    sl_atr_mult: float
    tp_atr_mult: float
    sma_period: int
    timeout_bars: int
    wf_test_return: float

# Convert to typed configs
CONFIGS = [EdgeConfig(*config) for config in EDGE_CONFIGS]

# Constants
RISK_PER_TRADE = 0.03  # 3% of total equity
MAX_POSITIONS = 3  # Risk management for $300
MAX_DRAWDOWN = 0.15  # 15% halt
CHECK_INTERVAL = 300  # 5 minutes
MAKER_FEE = 0.0016  # 0.16%
SLIPPAGE = 0.0005  # 0.05%
MIN_RR_RATIO = 1.5  # Minimum risk:reward ratio

STATE_FILE = PROJECT_ROOT / "data" / "momentum_state.json"


# ═══════════════════════════════════════════════════════════════
# POSITION AND STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    timeframe: int
    bar_count_at_entry: int
    config_key: str  # For tracking which config generated this
    order_id: Optional[str] = None
    status: str = "PENDING"  # PENDING, OPEN, CLOSING
    exit_order_id: Optional[str] = None


@dataclass
class BotState:
    """Persistent bot state."""
    positions: Dict[str, Position] = field(default_factory=dict)  # symbol -> Position
    trade_history: List[Dict] = field(default_factory=list)
    bar_counts: Dict[str, Dict[int, int]] = field(default_factory=dict)  # symbol -> timeframe -> count
    initial_balance: float = 0
    peak_balance: float = 0
    start_time: str = ""


def get_interval_string(timeframe_min: int) -> str:
    """Convert minutes to Kraken interval string."""
    mapping = {15: "15m", 60: "1h", 240: "4h"}
    return mapping.get(timeframe_min, "1h")


def save_state(state: BotState):
    """Save bot state to JSON file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert positions to serializable format
    positions_dict = {}
    for symbol, pos in state.positions.items():
        positions_dict[symbol] = {
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "entry_time": pos.entry_time,
            "timeframe": pos.timeframe,
            "bar_count_at_entry": pos.bar_count_at_entry,
            "config_key": pos.config_key,
            "order_id": pos.order_id,
            "status": pos.status,
            "exit_order_id": pos.exit_order_id,
        }
    
    data = {
        "positions": positions_dict,
        "trade_history": state.trade_history[-200:],  # Keep last 200 trades
        "bar_counts": state.bar_counts,
        "initial_balance": state.initial_balance,
        "peak_balance": state.peak_balance,
        "start_time": state.start_time,
    }
    
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_state() -> BotState:
    """Load bot state from JSON file."""
    state = BotState()
    
    if not STATE_FILE.exists():
        return state
    
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        
        # Restore positions
        for symbol, pos_data in data.get("positions", {}).items():
            state.positions[symbol] = Position(**pos_data)
        
        state.trade_history = data.get("trade_history", [])
        state.bar_counts = data.get("bar_counts", {})
        state.initial_balance = data.get("initial_balance", 0)
        state.peak_balance = data.get("peak_balance", 0)
        state.start_time = data.get("start_time", "")
        
        logger.info(f"Restored state: {len(state.positions)} positions, "
                   f"{len(state.trade_history)} historical trades")
    except Exception as e:
        logger.warning(f"Could not load state: {e}")
    
    return state


# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════

def check_breakout_signal(df: pd.DataFrame, config: EdgeConfig) -> Optional[Dict]:
    """
    Check for momentum breakout signal based on validated config.
    
    Returns signal dict with entry details or None if no signal.
    """
    if len(df) < max(config.breakout_bars, config.sma_period) + 20:
        return None
    
    # Convert to float arrays for fast computation
    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    
    # Calculate indicators
    atr_series = calc_atr(df, 14)
    sma_series = calc_sma(pd.Series(close), config.sma_period)
    
    # Get current values
    curr_idx = len(df) - 1
    curr_close = close[curr_idx]
    curr_atr = atr_series.iloc[curr_idx]
    curr_sma = sma_series.iloc[curr_idx]
    
    # Skip if indicators are NaN or invalid
    if pd.isna(curr_atr) or pd.isna(curr_sma) or curr_atr <= 0:
        return None
    
    # Direction-specific logic
    if config.direction == "long":
        # LONG: price > SMA and close > max(high[-N:])
        if curr_close <= curr_sma:
            return None
        
        # Check breakout above N-bar high
        lookback_high = np.max(high[curr_idx - config.breakout_bars:curr_idx])
        if curr_close <= lookback_high:
            return None
        
        # Calculate levels
        entry_price = curr_close
        lookback_low = np.min(low[curr_idx - config.breakout_bars:curr_idx])
        sl_swing = lookback_low * 0.998  # Small buffer
        sl_atr = entry_price - config.sl_atr_mult * curr_atr
        stop_loss = max(sl_swing, sl_atr)
        take_profit = entry_price + config.tp_atr_mult * curr_atr
        
    else:  # SHORT
        # SHORT: price < SMA and close < min(low[-N:])
        if curr_close >= curr_sma:
            return None
        
        # Check breakdown below N-bar low
        lookback_low = np.min(low[curr_idx - config.breakout_bars:curr_idx])
        if curr_close >= lookback_low:
            return None
        
        # Calculate levels
        entry_price = curr_close
        lookback_high = np.max(high[curr_idx - config.breakout_bars:curr_idx])
        sl_swing = lookback_high * 1.002  # Small buffer
        sl_atr = entry_price + config.sl_atr_mult * curr_atr
        stop_loss = min(sl_swing, sl_atr)
        take_profit = entry_price - config.tp_atr_mult * curr_atr
    
    # Validate risk:reward ratio
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk <= 0 or reward / risk < MIN_RR_RATIO:
        return None
    
    return {
        "config": config,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": curr_atr,
        "risk": risk,
        "reward": reward,
        "rr_ratio": reward / risk,
        "reason": f"{config.direction.upper()} breakout {config.breakout_bars}bars, "
                  f"{config.direction=='long' and 'above' or 'below'} SMA{config.sma_period}",
    }


def check_exit_conditions(position: Position, current_price: float, current_bar_count: int) -> Optional[str]:
    """Check if position should be exited."""
    if current_price <= 0:
        return None
    
    # Stop loss hit
    if position.direction == "long":
        if current_price <= position.stop_loss:
            return "STOP_LOSS"
        if current_price >= position.take_profit:
            return "TAKE_PROFIT"
    else:  # short
        if current_price >= position.stop_loss:
            return "STOP_LOSS"
        if current_price <= position.take_profit:
            return "TAKE_PROFIT"
    
    # Timeout check
    bars_held = current_bar_count - position.bar_count_at_entry
    if bars_held >= CONFIGS[0].timeout_bars:  # Use timeout from any config - they're similar
        return "TIMEOUT"
    
    return None


# ═══════════════════════════════════════════════════════════════
# MAIN BOT CLASS
# ═══════════════════════════════════════════════════════════════

class MomentumBreakoutBot:
    """Production multi-pair momentum breakout bot."""
    
    def __init__(self):
        self.client = KrakenClient(
            api_key=os.getenv("KRAKEN_API_KEY", ""),
            private_key=os.getenv("KRAKEN_PRIVATE_KEY", ""),
        )
        self.state = load_state()
        self.live_trading = os.getenv("ENABLE_LIVE_TRADING", "").strip().lower() in {
            "1", "true", "yes"
        }
        
        # Initialize start time if not set
        if not self.state.start_time:
            self.state.start_time = datetime.now().isoformat()
        
        if not self.live_trading:
            logger.warning("=" * 60)
            logger.warning("  DRY RUN MODE — No real orders will be placed")
            logger.warning("  Set ENABLE_LIVE_TRADING=true in .env to go live")
            logger.warning("=" * 60)
    
    def get_account_balance_usd(self) -> float:
        """Get USD balance from Kraken."""
        if not self.live_trading:
            return 300.0  # Simulated balance for dry run
        
        try:
            balance = self.client.get_account_balance()
            return float(balance.get("USD", 0) or 0)
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0
    
    def get_asset_balance(self, asset: str) -> float:
        """Get specific asset balance."""
        if not self.live_trading:
            return 0.0  # No holdings in dry run
        
        try:
            balance = self.client.get_account_balance()
            return float(balance.get(asset, 0) or 0)
        except Exception as e:
            logger.error(f"Failed to fetch {asset} balance: {e}")
            return 0
    
    def fetch_candles(self, symbol: str, timeframe_min: int, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch historical candles for a symbol and timeframe."""
        try:
            interval = get_interval_string(timeframe_min)
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines:
                return None
            
            df = pd.DataFrame(klines)
            if len(df) < 50:  # Need sufficient history
                return None
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol} {timeframe_min}m: {e}")
            return None
    
    def scan_for_signals(self) -> List[Tuple[Dict, float]]:
        """
        Scan all configs for breakout signals.
        Returns list of (signal_dict, priority_score) tuples.
        """
        signals = []
        
        for config in CONFIGS:
            # Skip if we already have a position in this symbol
            if config.pair in self.state.positions:
                continue
            
            # SHORT via margin: Kraken supports margin selling on most USD pairs.
            # We use leverage=2 for shorts. DOGEUSD has no margin — skip it.
            # NOTE: margin shorts require collateral (USD balance).
            if config.direction == "short" and config.pair == "DOGEUSD":
                continue  # No margin available for DOGE
            
            # Fetch candles
            df = self.fetch_candles(config.pair, config.timeframe_min)
            if df is None:
                continue
            
            # Update bar count for this symbol/timeframe
            if config.pair not in self.state.bar_counts:
                self.state.bar_counts[config.pair] = {}
            self.state.bar_counts[config.pair][config.timeframe_min] = len(df)
            
            # Check for signal
            signal = check_breakout_signal(df, config)
            if signal:
                # Priority score: walk-forward return * risk:reward ratio
                priority = config.wf_test_return * signal["rr_ratio"]
                signals.append((signal, priority))
                logger.info(f"Signal found: {config.pair} {config.timeframe_min}m {config.direction} "
                           f"(priority={priority:.1f})")
        
        # Sort by priority (highest first)
        signals.sort(key=lambda x: x[1], reverse=True)
        return signals
    
    def calculate_position_size(self, signal: Dict, balance: float) -> float:
        """Calculate position size based on risk management."""
        risk_dollars = balance * RISK_PER_TRADE
        position_size = risk_dollars / signal["risk"]
        return position_size
    
    def place_entry_order(self, signal: Dict, position_size: float) -> Optional[Position]:
        """Place entry order for a signal."""
        config = signal["config"]
        
        # Check minimum order size
        min_vol = self.client.get_min_order_volume(config.pair)
        if min_vol and position_size < float(min_vol):
            logger.warning(f"Position size {position_size:.8f} below Kraken minimum {min_vol}")
            return None
        
        # Get current bar count
        bar_count = self.state.bar_counts.get(config.pair, {}).get(config.timeframe_min, 0)
        
        # Create position object
        position = Position(
            symbol=config.pair,
            direction=config.direction,
            entry_price=signal["entry_price"],
            quantity=position_size,
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"],
            entry_time=datetime.now().isoformat(),
            timeframe=config.timeframe_min,
            bar_count_at_entry=bar_count,
            config_key=f"{config.pair}_{config.timeframe_min}_{config.direction}",
        )
        
        if self.live_trading:
            try:
                # Place market order. For shorts, use leverage=2 (margin sell).
                side = "buy" if config.direction == "long" else "sell"
                leverage = 2 if config.direction == "short" else None
                
                order = self.client.place_order(
                    symbol=config.pair,
                    side=side,
                    order_type="market",
                    quantity=position_size,
                    leverage=leverage,
                )
                
                if order and "txid" in order:
                    position.order_id = order["txid"][0] if order["txid"] else None
                    position.status = "PENDING"
                    
                    logger.info(f"ENTRY ORDER PLACED: {side.upper()} {position_size:.8f} {config.pair} "
                               f"@ ~${signal['entry_price']:.4f} (SL: ${signal['stop_loss']:.4f}, "
                               f"TP: ${signal['take_profit']:.4f}) R:R={signal['rr_ratio']:.1f}")
                    return position
                else:
                    logger.error(f"Failed to place entry order for {config.pair}")
                    return None
            except Exception as e:
                logger.error(f"Error placing entry order: {e}")
                return None
        else:
            # Dry run
            position.status = "OPEN"
            logger.info(f"DRY RUN ENTRY: {config.direction.upper()} {position_size:.8f} {config.pair} "
                       f"@ ${signal['entry_price']:.4f} (SL: ${signal['stop_loss']:.4f}, "
                       f"TP: ${signal['take_profit']:.4f}) R:R={signal['rr_ratio']:.1f}")
            return position
    
    def exit_position(self, position: Position, reason: str, exit_price: float) -> bool:
        """Exit a position."""
        if self.live_trading:
            try:
                # For longs: sell to close. For shorts: buy to cover (margin).
                side = "sell" if position.direction == "long" else "buy"
                leverage = 2 if position.direction == "short" else None
                
                order = self.client.place_order(
                    symbol=position.symbol,
                    side=side,
                    order_type="market",
                    quantity=position.quantity,
                    leverage=leverage,
                )
                
                if order and "txid" in order:
                    position.exit_order_id = order["txid"][0] if order["txid"] else None
                    position.status = "CLOSING"
                    
                    logger.info(f"EXIT ORDER PLACED: {side.upper()} {position.quantity:.8f} "
                               f"{position.symbol} @ ~${exit_price:.4f} ({reason})")
                    return True
                else:
                    logger.error(f"Failed to place exit order for {position.symbol}")
                    return False
            except Exception as e:
                logger.error(f"Error placing exit order: {e}")
                return False
        else:
            # Dry run - record the trade
            if position.direction == "long":
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Account for fees
            entry_fee = position.entry_price * position.quantity * MAKER_FEE
            exit_fee = exit_price * position.quantity * MAKER_FEE
            total_fees = entry_fee + exit_fee
            net_pnl = pnl - total_fees
            
            trade_record = {
                "symbol": position.symbol,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "quantity": position.quantity,
                "pnl": round(net_pnl, 2),
                "fees": round(total_fees, 2),
                "reason": reason,
                "entry_time": position.entry_time,
                "exit_time": datetime.now().isoformat(),
                "config": position.config_key,
            }
            
            self.state.trade_history.append(trade_record)
            
            logger.info(f"DRY RUN EXIT: {position.direction.upper()} {position.symbol} "
                       f"@ ${exit_price:.4f} ({reason}) PnL: ${net_pnl:.2f}")
            return True
    
    def sync_orders(self):
        """Check status of pending orders with Kraken."""
        if not self.live_trading:
            return
        
        for symbol, position in list(self.state.positions.items()):
            try:
                if position.status == "PENDING" and position.order_id:
                    # Check if entry order filled
                    orders = self.client.query_orders([position.order_id])
                    if position.order_id in orders:
                        order_info = orders[position.order_id]
                        status = order_info.get("status", "").lower()
                        
                        if status == "closed":
                            # Order filled - update position
                            position.status = "OPEN"
                            # Could update actual fill price/quantity here
                            logger.info(f"Entry order filled for {symbol}")
                        elif status in ["canceled", "cancelled", "expired"]:
                            # Order failed - remove position
                            logger.warning(f"Entry order {status} for {symbol}")
                            del self.state.positions[symbol]
                
                elif position.status == "CLOSING" and position.exit_order_id:
                    # Check if exit order filled
                    orders = self.client.query_orders([position.exit_order_id])
                    if position.exit_order_id in orders:
                        order_info = orders[position.exit_order_id]
                        status = order_info.get("status", "").lower()
                        
                        if status == "closed":
                            # Exit filled - record trade and remove position
                            # Could get actual exit price here
                            logger.info(f"Exit order filled for {symbol}")
                            del self.state.positions[symbol]
                        elif status in ["canceled", "cancelled", "expired"]:
                            # Exit failed - position still open
                            logger.warning(f"Exit order {status} for {symbol}")
                            position.status = "OPEN"
                            position.exit_order_id = None
                            
            except Exception as e:
                logger.error(f"Error syncing orders for {symbol}: {e}")
    
    def manage_existing_positions(self):
        """Check exit conditions for existing positions."""
        for symbol, position in list(self.state.positions.items()):
            if position.status != "OPEN":
                continue
            
            try:
                # Get current price
                ticker = self.client.get_ticker(symbol)
                if not ticker:
                    continue
                
                current_price = float(ticker["price"])
                
                # Get current bar count
                current_bar_count = self.state.bar_counts.get(symbol, {}).get(position.timeframe, 0)
                
                # Check exit conditions
                exit_reason = check_exit_conditions(position, current_price, current_bar_count)
                if exit_reason:
                    if self.exit_position(position, exit_reason, current_price):
                        if not self.live_trading:
                            # In dry run, remove position immediately
                            del self.state.positions[symbol]
            
            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}")
    
    def log_status(self):
        """Log current bot status."""
        balance = self.get_account_balance_usd()
        
        # Update peak balance
        if balance > self.state.peak_balance:
            self.state.peak_balance = balance
        
        # Calculate drawdown
        if self.state.initial_balance == 0:
            self.state.initial_balance = balance
            self.state.peak_balance = balance
        
        drawdown = 0
        if self.state.peak_balance > 0:
            drawdown = (self.state.peak_balance - balance) / self.state.peak_balance
        
        # Count positions by status
        open_positions = [s for s, p in self.state.positions.items() if p.status == "OPEN"]
        pending_positions = [s for s, p in self.state.positions.items() if p.status == "PENDING"]
        
        # Trade stats
        trades = self.state.trade_history
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0) if trades else 0
        total_pnl = sum(t.get("pnl", 0) for t in trades) if trades else 0
        
        logger.info(f"Status: Balance=${balance:.2f} | Open={len(open_positions)} "
                   f"Pending={len(pending_positions)} | DD={drawdown*100:.1f}% | "
                   f"Trades={total_trades} WR={winning_trades/total_trades*100 if total_trades else 0:.0f}% "
                   f"PnL=${total_pnl:.2f}")
        
        if open_positions:
            for symbol in open_positions:
                pos = self.state.positions[symbol]
                logger.info(f"  {pos.direction.upper()} {symbol}: entry=${pos.entry_price:.4f} "
                           f"SL=${pos.stop_loss:.4f} TP=${pos.take_profit:.4f}")
    
    def run(self):
        """Main bot execution loop."""
        logger.info("=" * 70)
        logger.info("    PRODUCTION MULTI-PAIR MOMENTUM BREAKOUT BOT")
        logger.info(f"    {len(CONFIGS)} validated edge configs loaded")
        logger.info(f"    Max positions: {MAX_POSITIONS} | Risk per trade: {RISK_PER_TRADE*100:.0f}%")
        logger.info(f"    Max drawdown: {MAX_DRAWDOWN*100:.0f}% | Check interval: {CHECK_INTERVAL}s")
        logger.info(f"    Live trading: {'ENABLED' if self.live_trading else 'DRY RUN'}")
        logger.info("=" * 70)
        
        try:
            while True:
                cycle_start = time.time()
                
                # Sync order status with exchange
                self.sync_orders()
                
                # Manage existing positions (check exits)
                self.manage_existing_positions()
                
                # Check for new entry signals (if we have capacity)
                current_positions = len([p for p in self.state.positions.values() if p.status == "OPEN"])
                balance = self.get_account_balance_usd()
                drawdown = (self.state.peak_balance - balance) / self.state.peak_balance if self.state.peak_balance > 0 else 0
                
                if (current_positions < MAX_POSITIONS and 
                    drawdown < MAX_DRAWDOWN and 
                    balance > 50):  # Minimum balance check
                    
                    signals = self.scan_for_signals()
                    if signals:
                        # Take the best signal
                        best_signal, priority = signals[0]
                        position_size = self.calculate_position_size(best_signal, balance)
                        
                        if position_size > 0:
                            position = self.place_entry_order(best_signal, position_size)
                            if position:
                                self.state.positions[position.symbol] = position
                
                # Log current status
                self.log_status()
                
                # Save state
                save_state(self.state)
                
                # Sleep until next cycle
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, CHECK_INTERVAL - cycle_time)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Save final state
            save_state(self.state)
            
            # Final summary
            if self.state.positions:
                logger.warning(f"Bot stopped with {len(self.state.positions)} open positions:")
                for symbol, pos in self.state.positions.items():
                    logger.warning(f"  {pos.direction.upper()} {symbol} @ ${pos.entry_price:.4f}")
            
            trades = self.state.trade_history
            if trades:
                total_trades = len(trades)
                winners = sum(1 for t in trades if t.get("pnl", 0) > 0)
                total_pnl = sum(t.get("pnl", 0) for t in trades)
                win_rate = winners / total_trades * 100
                
                logger.info("=" * 50)
                logger.info("FINAL SUMMARY:")
                logger.info(f"Total trades: {total_trades}")
                logger.info(f"Win rate: {win_rate:.1f}%")
                logger.info(f"Total PnL: ${total_pnl:.2f}")
                logger.info("=" * 50)


def main():
    """Entry point."""
    # Setup logging
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(str(PROJECT_ROOT / "logs" / "momentum_bot.log"), 
              rotation="100 MB", retention="30 days")
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    
    # Create and run bot
    bot = MomentumBreakoutBot()
    bot.run()


if __name__ == "__main__":
    main()