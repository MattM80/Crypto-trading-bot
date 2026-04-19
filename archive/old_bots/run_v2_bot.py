#!/usr/bin/env python3
"""
Crypto Trading Bot v2 - Clean Rewrite
CAPITAL PRESERVATION FOCUSED - Sit in cash 70-90% of the time. Only trade A+ setups.

Architecture:
1. Regime Detection (Fear & Greed + BTC price action)
2. 8 liquid pairs only (XBTUSD, ETHUSD, etc.)
3. 4 high-conviction strategies only  
4. Strict risk management
5. Conservative position sizing
6. Clean logging and state persistence

This bot is the opposite of the previous 30+ tool madness.
Boring, patient, disciplined.
"""

import os
import sys
import json
import time
import signal
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Import existing components
sys.path.append(str(Path(__file__).parent / "src"))
from kraken_client import KrakenClient
from ws_monitor import CrashMonitor

# Configuration
PROJECT_ROOT = Path(__file__).parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Ensure directories exist
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
(PROJECT_ROOT / "data").mkdir(exist_ok=True)

# Configure logging
logger.remove()
logger.add(
    PROJECT_ROOT / "logs" / "v2_bot.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="INFO"
)
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level: <5} | {message}")

class RegimeType:
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"  
    BEAR = "BEAR"

class TradingBot:
    """V2 Trading Bot - Capital preservation focused"""
    
    # Trading pairs - ONLY liquid large caps
    PAIRS = [
        "XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", 
        "DOGEUSD", "AVAXUSD", "LINKUSD", "DOTUSD"
    ]
    
    # Regime-based parameters
    REGIME_PARAMS = {
        RegimeType.BULL: {
            "max_positions": 3,
            "max_deployed_pct": 0.25,
            "signal_threshold_mult": 1.0,
            "take_profit_pct": 0.12,  # 12% trailing
            "stop_loss_pct": 0.05,    # -5%
            "max_hold_hours": 168     # 7 days
        },
        RegimeType.NEUTRAL: {
            "max_positions": 2,
            "max_deployed_pct": 0.15,
            "signal_threshold_mult": 1.5,
            "take_profit_pct": 0.08,  # 8% fixed
            "stop_loss_pct": 0.04,    # -4%
            "max_hold_hours": 72      # 3 days
        },
        RegimeType.BEAR: {
            "max_positions": 1,
            "max_deployed_pct": 0.08,
            "signal_threshold_mult": 3.0,
            "take_profit_pct": 0.05,  # 5% fixed
            "stop_loss_pct": 0.03,    # -3%
            "max_hold_hours": 24      # 1 day
        }
    }
    
    # Risk controls
    DAILY_MAX_DRAWDOWN = 0.03    # -3% daily max
    MONTHLY_MAX_DRAWDOWN = 0.05  # -5% monthly max
    CONSECUTIVE_LOSS_LIMIT = 3
    COOLDOWN_HOURS = 6           # Hours to wait after stop loss
    
    def __init__(self):
        """Initialize the trading bot"""
        self.client = KrakenClient()
        self.crash_monitor = None
        self.running = False
        self.state_file = PROJECT_ROOT / "data" / "v2_bot_state.json"
        
        # Cached values per cycle (reset each cycle)
        self._cached_fg_index = None
        self._cached_balance_usd = None
        
        # Initialize state
        self.state = self._load_state()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Performance tracking files
        self.trades_file = PROJECT_ROOT / "logs" / "v2_trades.csv"
        self.balance_file = PROJECT_ROOT / "logs" / "v2_balance.csv"
        
        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()
        
        logger.info("V2 Trading Bot initialized")
        
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        if not self.trades_file.exists():
            with open(self.trades_file, 'w') as f:
                f.write("timestamp,pair,side,entry_price,exit_price,pnl_usd,pnl_pct,strategy,regime,hold_time_hours\n")
                
        if not self.balance_file.exists():
            with open(self.balance_file, 'w') as f:
                f.write("timestamp,total_usd,deployed_pct,regime,positions_count\n")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        
    def _load_state(self) -> Dict:
        """Load bot state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info("Loaded existing state")
                return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                
        # Default state
        return {
            "positions": {},
            "cooldowns": {},
            "daily_pnl": 0.0,
            "monthly_pnl": 0.0,
            "consecutive_losses": 0,
            "last_regime": RegimeType.NEUTRAL,
            "last_regime_time": datetime.now(timezone.utc).isoformat(),
            "daily_start": datetime.now(timezone.utc).date().isoformat(),
            "monthly_start": datetime.now(timezone.utc).replace(day=1).date().isoformat()
        }
        
    def _save_state(self):
        """Save bot state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def _get_fear_greed_index(self) -> Optional[int]:
        """Get Fear & Greed Index from API (cached per cycle)"""
        if self._cached_fg_index is not None:
            return self._cached_fg_index
        try:
            response = requests.get("https://api.alternative.me/fng/", timeout=10)
            data = response.json()
            value = int(data["data"][0]["value"])
            self._cached_fg_index = value
            logger.debug(f"Fear & Greed Index: {value}")
            return value
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed Index: {e}")
            return None
            
    def _get_btc_sma_signal(self) -> bool:
        """Check if BTC is above 20-day SMA"""
        try:
            # Get 25 days of daily data to calculate 20-day SMA
            klines = self.client.get_klines("XBTUSD", interval="1d", limit=25)
            if not klines or len(klines) < 20:
                return False
                
            closes = np.array([float(k['close']) for k in klines])
            sma_20 = np.mean(closes[-20:])
            current_price = closes[-1]
            
            above_sma = current_price > sma_20
            logger.debug(f"BTC: {current_price:.0f}, SMA20: {sma_20:.0f}, Above: {above_sma}")
            return above_sma
            
        except Exception as e:
            logger.warning(f"Failed to get BTC SMA signal: {e}")
            return False
            
    def _detect_regime(self) -> str:
        """Detect market regime using Fear & Greed + BTC price action"""
        fg_index = self._get_fear_greed_index()
        btc_above_sma = self._get_btc_sma_signal()
        
        if fg_index is None:
            logger.warning("Using previous regime due to missing F&G data")
            return self.state.get("last_regime", RegimeType.NEUTRAL)
            
        # Regime logic
        if fg_index > 50 and btc_above_sma:
            regime = RegimeType.BULL
        elif fg_index < 25 or (not btc_above_sma and fg_index < 40):
            regime = RegimeType.BEAR
        else:
            regime = RegimeType.NEUTRAL
            
        # Update state if regime changed
        if regime != self.state.get("last_regime"):
            logger.info(f"Regime changed: {self.state.get('last_regime')} -> {regime}")
            self.state["last_regime"] = regime
            self.state["last_regime_time"] = datetime.now(timezone.utc).isoformat()
            
        return regime
        
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
            
        prices = np.array(prices)
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
        
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)"""
        if len(highs) < period + 1:
            return 0.0
            
        highs, lows, closes = np.array(highs), np.array(lows), np.array(closes)
        
        # Calculate True Range
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate Directional Movement
        dm_plus = np.maximum(highs[1:] - highs[:-1], 0)
        dm_minus = np.maximum(lows[:-1] - lows[1:], 0)
        dm_plus = np.where(dm_plus > dm_minus, dm_plus, 0)
        dm_minus = np.where(dm_minus > dm_plus, dm_minus, 0)
        
        # Smooth the values
        if len(tr) < period:
            return 0.0
            
        atr = np.mean(tr[-period:])
        adm_plus = np.mean(dm_plus[-period:])
        adm_minus = np.mean(dm_minus[-period:])
        
        if atr == 0:
            return 0.0
            
        di_plus = 100 * adm_plus / atr
        di_minus = 100 * adm_minus / atr
        
        if di_plus + di_minus == 0:
            return 0.0
            
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        return dx
        
    def _get_market_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[Dict]:
        """Get market data for a symbol"""
        try:
            klines = self.client.get_klines(symbol, interval=interval, limit=limit)
            if not klines:
                return None
                
            ticker = self.client.get_ticker(symbol)
            if not ticker:
                return None
                
            # Extract OHLCV data
            opens = [float(k['open']) for k in klines]
            highs = [float(k['high']) for k in klines]
            lows = [float(k['low']) for k in klines]
            closes = [float(k['close']) for k in klines]
            volumes = [float(k['volume']) for k in klines]
            
            # Calculate indicators
            rsi = self._calculate_rsi(closes)
            adx = self._calculate_adx(highs, lows, closes)
            
            # Calculate volume average
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            
            # Price change calculations
            current_price = float(ticker['price'])
            price_24h_ago = closes[-24] if len(closes) >= 24 else closes[0]
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago
            
            price_48h_ago = closes[-48] if len(closes) >= 48 else closes[0]
            price_change_48h = (current_price - price_48h_ago) / price_48h_ago
            
            # SMA calculations
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
            sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else current_price
            
            return {
                "symbol": symbol,
                "price": current_price,
                "bid": float(ticker['bid']),
                "ask": float(ticker['ask']),
                "volume": volumes[-1],
                "avg_volume": avg_volume,
                "rsi": rsi,
                "adx": adx,
                "price_change_24h": price_change_24h,
                "price_change_48h": price_change_48h,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "opens": opens,
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "volumes": volumes
            }
            
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")
            return None
            
    def _strategy_capitulation_buy(self, data: Dict, regime: str) -> Optional[Dict]:
        """Strategy 1: Capitulation Buy (bear market bread and butter)"""
        fg_index = self._get_fear_greed_index()
        
        # Requirements
        if fg_index is None or fg_index >= 20:
            return None
            
        if data['price_change_24h'] >= -0.05:  # Must drop >5% in 24h
            return None
            
        if data['rsi'] >= 25:  # RSI must be oversold
            return None
            
        if data['volume'] < data['avg_volume'] * 2:  # Volume spike
            return None
            
        # Calculate score
        fear_intensity = max(0, 20 - fg_index)
        drop_magnitude = abs(data['price_change_24h']) * 100
        score = 80 + (fear_intensity * 2) + (drop_magnitude * 3)
        
        return {
            "strategy": "capitulation_buy",
            "score": score,
            "reason": f"Capitulation buy: F&G={fg_index}, drop={drop_magnitude:.1f}%, RSI={data['rsi']:.1f}"
        }
        
    def _strategy_momentum_continuation(self, data: Dict, regime: str) -> Optional[Dict]:
        """Strategy 2: Momentum Continuation (bull market workhorse)"""
        # Only fires in bull regime
        if regime != RegimeType.BULL:
            return None
            
        # Requirements
        if data['price'] <= data['sma_20'] or data['price'] <= data['sma_50']:
            return None
            
        if data['rsi'] < 40 or data['rsi'] > 65:
            return None
            
        if data['volume'] < data['avg_volume']:
            return None
            
        if data['adx'] < 25:
            return None
            
        # Calculate score
        score = 50 + (data['adx'] - 25) * 2
        
        return {
            "strategy": "momentum_continuation",
            "score": score,
            "reason": f"Momentum: price above SMAs, RSI={data['rsi']:.1f}, ADX={data['adx']:.1f}"
        }
        
    def _strategy_support_bounce(self, data: Dict, regime: str) -> Optional[Dict]:
        """Strategy 3: Support Bounce"""
        # Check proximity to 200 SMA (within 2%)
        sma_distance = abs(data['price'] - data['sma_200']) / data['sma_200']
        if sma_distance > 0.02:
            return None
            
        if data['rsi'] >= 35:
            return None
            
        # Check for bullish candle pattern (recent green candle after red)
        if len(data['closes']) < 3:
            return None
            
        recent_closes = data['closes'][-3:]
        recent_opens = data['opens'][-3:]
        
        # Last candle should be green, previous should be red
        if recent_closes[-1] <= recent_opens[-1]:  # Not a green candle
            return None
        if recent_closes[-2] >= recent_opens[-2]:  # Previous wasn't red
            return None
            
        # Calculate score
        proximity_score = (0.02 - sma_distance) * 10
        score = 60 + proximity_score * 10
        
        return {
            "strategy": "support_bounce",
            "score": score,
            "reason": f"Support bounce: {sma_distance*100:.1f}% from SMA200, RSI={data['rsi']:.1f}"
        }
        
    def _strategy_liquidation_cascade(self, data: Dict, regime: str) -> Optional[Dict]:
        """Strategy 4: Liquidation Cascade Bounce"""
        # Major drop >8% in 48h with massive volume
        if data['price_change_48h'] >= -0.08:
            return None
            
        if data['volume'] < data['avg_volume'] * 3:
            return None
            
        # This is a rare, high-priority signal
        score = 90
        
        return {
            "strategy": "liquidation_cascade",
            "score": score,
            "reason": f"Liquidation cascade: {data['price_change_48h']*100:.1f}% drop in 48h, massive volume"
        }
        
    def _generate_signals(self, regime: str) -> List[Dict]:
        """Generate trading signals for all pairs"""
        signals = []
        
        for symbol in self.PAIRS:
            # Skip if position already open
            if symbol in self.state["positions"]:
                continue
                
            # Skip if in cooldown
            if symbol in self.state["cooldowns"]:
                cooldown_time = datetime.fromisoformat(self.state["cooldowns"][symbol])
                if datetime.now(timezone.utc) < cooldown_time:
                    continue
                else:
                    # Remove expired cooldown
                    del self.state["cooldowns"][symbol]
                    
            # Get market data
            data = self._get_market_data(symbol, interval="1h")
            if not data:
                continue
                
            # Test all strategies
            strategies = [
                self._strategy_capitulation_buy,
                self._strategy_momentum_continuation,
                self._strategy_support_bounce,
                self._strategy_liquidation_cascade
            ]
            
            for strategy_func in strategies:
                signal = strategy_func(data, regime)
                if signal:
                    # Apply regime threshold multiplier
                    threshold_mult = self.REGIME_PARAMS[regime]["signal_threshold_mult"]
                    if signal["score"] >= 50 * threshold_mult:
                        signal.update({
                            "symbol": symbol,
                            "regime": regime,
                            "data": data
                        })
                        signals.append(signal)
                        
        # Sort by score descending
        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals
        
    def _calculate_position_size(self, symbol: str, price: float, regime: str) -> float:
        """Calculate position size based on regime and risk management"""
        try:
            balance = self.client.get_account_balance()
            if not balance or "USD" not in balance:
                logger.warning("No USD balance available")
                return 0.0
                
            usd_balance = float(balance["USD"])
            
            # Get regime parameters
            params = self.REGIME_PARAMS[regime]
            max_deployed_pct = params["max_deployed_pct"]
            
            # Adjust for consecutive losses
            if self.state["consecutive_losses"] >= self.CONSECUTIVE_LOSS_LIMIT:
                max_deployed_pct *= 0.5
                logger.info(f"Halving position size due to {self.state['consecutive_losses']} consecutive losses")
                
            # Calculate base position size
            position_value = usd_balance * max_deployed_pct / params["max_positions"]
            position_size = position_value / price
            
            # Check minimum order size
            min_volume = self.client.get_min_order_volume(symbol) or 0.0001
            if position_size < min_volume:
                logger.warning(f"Position size {position_size} below minimum {min_volume} for {symbol}")
                return 0.0
                
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.0
            
    def _place_entry_order(self, signal: Dict) -> bool:
        """Place entry order for a signal"""
        try:
            symbol = signal["symbol"]
            data = signal["data"]
            regime = signal["regime"]
            
            # Calculate position size
            entry_price = data["ask"]  # Buy at ask
            spread = data["ask"] - data["bid"]
            limit_price = entry_price - (spread * 0.3)  # Better limit price
            
            position_size = self._calculate_position_size(symbol, limit_price, regime)
            if position_size <= 0:
                return False
                
            # Place limit order
            order = self.client.place_order(
                symbol=symbol,
                side="buy",
                order_type="limit",
                quantity=position_size,
                price=limit_price,
                post_only=True  # Maker order for lower fees
            )
            
            if not order or not order.get("txid"):
                logger.warning(f"Failed to place order for {symbol}")
                return False
                
            # Store position in state — marked as pending until fill confirmed
            params = self.REGIME_PARAMS[regime]
            self.state["positions"][symbol] = {
                "entry_price": limit_price,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "strategy": signal["strategy"],
                "size": position_size,
                "regime": regime,
                "order_id": order["txid"][0],
                "pending_entry": True,  # Not confirmed filled yet
                "stop_loss": limit_price * (1 - params["stop_loss_pct"]),
                "take_profit": limit_price * (1 + params["take_profit_pct"]),
                "highest_price_seen": limit_price,
                "trailing_stop_active": False,
                "partial_profit_taken": False
            }
            
            logger.info(f"Placed {signal['strategy']} entry for {symbol} at {limit_price:.4f} (size: {position_size:.6f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            return False
            
    def _manage_position_exits(self, regime: str):
        """Manage exits for open positions"""
        for symbol in list(self.state["positions"].keys()):
            position = self.state["positions"][symbol]
            
            # Skip positions where entry hasn't confirmed yet
            if position.get("pending_entry"):
                continue
            
            # Get current price
            ticker = self.client.get_ticker(symbol)
            if not ticker:
                continue
                
            current_price = float(ticker["price"])
            entry_price = position["entry_price"]
            entry_time = datetime.fromisoformat(position["entry_time"])
            hold_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            
            # Update highest price seen
            if current_price > position["highest_price_seen"]:
                position["highest_price_seen"] = current_price
                
            # Get regime parameters  
            params = self.REGIME_PARAMS[position["regime"]]
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            # 1. Stop loss
            if current_price <= position["stop_loss"]:
                exit_reason = "stop_loss"
                
            # 2. Time-based exit
            elif hold_hours >= params["max_hold_hours"]:
                exit_reason = "time_limit"
                
            # 3. Take profit logic
            elif regime == RegimeType.BULL:
                # Bull: trailing stop after 5% profit
                current_profit_pct = (current_price - entry_price) / entry_price
                if current_profit_pct >= 0.05:
                    if not position["trailing_stop_active"]:
                        position["trailing_stop_active"] = True
                        logger.info(f"Activated trailing stop for {symbol}")
                        
                    # Trail at 12% from high
                    trailing_stop = position["highest_price_seen"] * (1 - params["take_profit_pct"])
                    if current_price <= trailing_stop:
                        exit_reason = "trailing_stop"
                        
            else:
                # Neutral/Bear: fixed take profit
                if current_price >= position["take_profit"]:
                    # Partial profit in neutral/bear
                    if not position["partial_profit_taken"]:
                        # Sell 50% at first TP target
                        self._exit_partial_position(symbol, position, 0.5, "partial_profit")
                        position["partial_profit_taken"] = True
                        continue
                    else:
                        exit_reason = "take_profit"
                        
            # Execute exit if triggered
            if exit_reason:
                self._exit_position(symbol, position, exit_price, exit_reason)
                
    def _exit_partial_position(self, symbol: str, position: Dict, exit_pct: float, reason: str):
        """Exit a partial position"""
        try:
            # Use actual Kraken balance, not state
            base, _ = self.client.get_pair_assets(symbol)
            actual_qty = position["size"]
            if base:
                balance = self.client.get_account_balance()
                if balance:
                    actual_qty = float(balance.get(base, 0))
                    if actual_qty <= 0:
                        logger.warning(f"[PARTIAL EXIT] {symbol}: no {base} on Kraken")
                        return

            exit_size = actual_qty * exit_pct
            
            order = self.client.place_order(
                symbol=symbol,
                side="sell",
                order_type="market",
                quantity=exit_size
            )
            
            if order and order.get("txid"):
                # Re-sync size from Kraken after partial sell
                remaining = actual_qty - exit_size
                position["size"] = remaining
                logger.info(f"Partial exit ({exit_pct*100:.0f}%) for {symbol}: {reason}, {remaining:.6f} remaining")
                
        except Exception as e:
            logger.error(f"Failed to exit partial position: {e}")
            
    def _exit_position(self, symbol: str, position: Dict, exit_price: float, reason: str):
        """Exit a complete position"""
        try:
            # Use ACTUAL Kraken balance for sell quantity — not our state.
            # This is what caused the DRIFT "Insufficient funds" loop.
            base, _ = self.client.get_pair_assets(symbol)
            sell_qty = position["size"]
            
            if base:
                balance = self.client.get_account_balance()
                if balance:
                    actual_qty = float(balance.get(base, 0))
                    if actual_qty <= 0:
                        logger.warning(f"[EXIT] {symbol}: no {base} on Kraken, removing from state")
                        del self.state["positions"][symbol]
                        return
                    if actual_qty < sell_qty:
                        logger.info(f"[EXIT] {symbol}: adjusting sell qty {sell_qty:.6f} → {actual_qty:.6f} (Kraken actual)")
                        sell_qty = actual_qty

            # Place market sell order
            order = self.client.place_order(
                symbol=symbol,
                side="sell",
                order_type="market",
                quantity=sell_qty
            )
            
            if not order or not order.get("txid"):
                logger.error(f"Failed to place exit order for {symbol}")
                return
                
            # Calculate PnL
            entry_price = position["entry_price"]
            pnl_usd = (exit_price - entry_price) * position["size"]
            pnl_pct = (exit_price - entry_price) / entry_price
            
            # Calculate hold time
            entry_time = datetime.fromisoformat(position["entry_time"])
            hold_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            
            # Update performance tracking
            self.state["daily_pnl"] += pnl_usd
            self.state["monthly_pnl"] += pnl_usd
            
            # Update consecutive losses
            if pnl_usd < 0:
                self.state["consecutive_losses"] += 1
                # Add cooldown
                cooldown_until = datetime.now(timezone.utc) + timedelta(hours=self.COOLDOWN_HOURS)
                self.state["cooldowns"][symbol] = cooldown_until.isoformat()
            else:
                self.state["consecutive_losses"] = 0
                
            # Log trade to CSV
            self._log_trade(
                symbol, "sell", entry_price, exit_price, pnl_usd, pnl_pct,
                position["strategy"], position["regime"], hold_hours
            )
            
            # Remove position from state
            del self.state["positions"][symbol]
            
            logger.info(
                f"Exited {symbol} at {exit_price:.4f} ({reason}): "
                f"PnL ${pnl_usd:.2f} ({pnl_pct*100:.1f}%), hold {hold_hours:.1f}h"
            )
            
        except Exception as e:
            logger.error(f"Failed to exit position {symbol}: {e}")
            
    def _log_trade(self, pair: str, side: str, entry_price: float, exit_price: float,
                   pnl_usd: float, pnl_pct: float, strategy: str, regime: str, hold_hours: float):
        """Log trade to CSV file"""
        try:
            with open(self.trades_file, 'a') as f:
                timestamp = datetime.now(timezone.utc).isoformat()
                f.write(f"{timestamp},{pair},{side},{entry_price},{exit_price},"
                       f"{pnl_usd},{pnl_pct},{strategy},{regime},{hold_hours}\n")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            
    def _log_balance_snapshot(self, regime: str):
        """Log balance snapshot to CSV"""
        try:
            balance = self.client.get_account_balance()
            if not balance:
                return
                
            usd_cash = float(balance.get("USD", 0))
            
            # Estimate deployed value from current prices
            deployed_value = 0.0
            for symbol, position in self.state["positions"].items():
                ticker = self.client.get_ticker(symbol)
                if ticker:
                    deployed_value += float(ticker["price"]) * position["size"]
                else:
                    deployed_value += position["entry_price"] * position["size"]
                    
            total_usd = usd_cash + deployed_value
            deployed_pct = deployed_value / total_usd if total_usd > 0 else 0.0
            positions_count = len(self.state["positions"])
            
            with open(self.balance_file, 'a') as f:
                timestamp = datetime.now(timezone.utc).isoformat()
                f.write(f"{timestamp},{total_usd},{deployed_pct},{regime},{positions_count}\n")
                
        except Exception as e:
            logger.error(f"Failed to log balance: {e}")
            
    def _check_drawdown_limits(self) -> bool:
        """Check if drawdown limits are exceeded (based on actual balance)"""
        starting = float(os.getenv("STARTING_BALANCE", "300"))
        
        # Daily drawdown check: -3% of current portfolio
        daily_limit = -self.DAILY_MAX_DRAWDOWN * starting
        if self.state["daily_pnl"] < daily_limit:
            logger.warning(f"Daily drawdown limit hit: ${self.state['daily_pnl']:.2f} (limit: ${daily_limit:.2f})")
            return False
            
        # Monthly drawdown check: -5% of current portfolio
        monthly_limit = -self.MONTHLY_MAX_DRAWDOWN * starting
        if self.state["monthly_pnl"] < monthly_limit:
            logger.warning(f"Monthly drawdown limit hit: ${self.state['monthly_pnl']:.2f} (limit: ${monthly_limit:.2f})")
            return False
            
        return True
        
    def _reset_daily_tracking(self):
        """Reset daily tracking at midnight UTC"""
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.state["daily_start"]:
            logger.info("Resetting daily tracking")
            self.state["daily_pnl"] = 0.0
            self.state["daily_start"] = today
            
    def _reset_monthly_tracking(self):
        """Reset monthly tracking at start of new month"""
        month_start = datetime.now(timezone.utc).replace(day=1).date().isoformat()
        if month_start != self.state["monthly_start"]:
            logger.info("Resetting monthly tracking")
            self.state["monthly_pnl"] = 0.0
            self.state["monthly_start"] = month_start
            
    def _check_realtime_exit(self, pair: str, current_price: float):
        """Check if a position needs to exit based on real-time price.
        
        Called from websocket callbacks — runs on every price update for
        pairs we have positions in. This catches stop losses within seconds
        instead of waiting up to 5 minutes for the next cycle.
        
        Uses a lock to prevent concurrent exit attempts.
        """
        pos = self.state.get("positions", {}).get(pair)
        if not pos or pos.get("pending_entry"):
            return

        # Prevent concurrent exit attempts from rapid websocket updates
        if pos.get("_exiting"):
            return

        entry_price = pos["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price

        # Update highest price seen (for trailing stops)
        if current_price > pos.get("highest_price_seen", 0):
            pos["highest_price_seen"] = current_price

        regime = pos.get("regime", RegimeType.BEAR)
        params = self.REGIME_PARAMS.get(regime, self.REGIME_PARAMS[RegimeType.BEAR])
        exit_reason = None

        # 1. Stop loss — the critical real-time check
        if current_price <= pos["stop_loss"]:
            exit_reason = "stop_loss_rt"

        # 2. Trailing stop (bull mode, after 5% profit)
        elif regime == RegimeType.BULL and pos.get("trailing_stop_active"):
            trailing_stop = pos["highest_price_seen"] * (1 - params["take_profit_pct"])
            if current_price <= trailing_stop:
                exit_reason = "trailing_stop_rt"

        # 3. Hard take profit (neutral/bear)
        elif regime != RegimeType.BULL:
            if current_price >= pos["take_profit"] and pos.get("partial_profit_taken"):
                exit_reason = "take_profit_rt"

        # 4. Activate trailing stop if we just crossed 5% profit in bull
        if regime == RegimeType.BULL and not pos.get("trailing_stop_active") and pnl_pct >= 0.05:
            pos["trailing_stop_active"] = True
            logger.info(f"🎯 [RT] Trailing stop activated for {pair} at +{pnl_pct*100:.1f}%")

        if exit_reason:
            pos["_exiting"] = True  # Guard against re-entry
            logger.info(
                f"⚡ [RT EXIT] {pair} triggered {exit_reason} @ ${current_price:.4f} "
                f"(PnL: {pnl_pct*100:.1f}%)"
            )
            self._exit_position(pair, pos, current_price, exit_reason)

    def _on_crash(self, pair: str, event_type: str, change_pct: float, current_price: float):
        """Handle crash events from websocket monitor.
        
        CrashMonitor calls: on_crash(pair, event_type, change_pct, current_price)
        where pair is already in bot format (e.g. 'XBTUSD').
        """
        try:
            logger.info(f"🚨 Crash event: {pair} {event_type} {change_pct*100:.1f}% @ ${current_price:.2f}")
            
            # Check if we need to emergency exit this position
            self._check_realtime_exit(pair, current_price)
            
            # Only enter if we're mostly in cash and it's a major dump (>5%)
            if len(self.state["positions"]) == 0 and abs(change_pct) > 0.05:
                if pair in self.PAIRS and pair not in self.state.get("cooldowns", {}):
                    data = self._get_market_data(pair, interval="1h")
                    if data:
                        regime = self.state.get("last_regime", RegimeType.BEAR)
                        signal = self._strategy_capitulation_buy(data, regime)
                        if signal:
                            signal.update({"symbol": pair, "regime": regime, "data": data})
                            logger.info(f"⚡ Emergency capitulation signal for {pair}!")
                            self._place_entry_order(signal)
                            
        except Exception as e:
            logger.error(f"Error handling crash event: {e}")

    def _on_pump(self, pair: str, event_type: str, change_pct: float, current_price: float):
        """Handle pump events — update tracking and check exits."""
        try:
            self._check_realtime_exit(pair, current_price)
        except Exception as e:
            logger.error(f"Error handling pump event: {e}")

    def _on_volume_spike(self, pair: str, volume_mult: float, current_price: float):
        """Handle volume spike events — check exits on price update."""
        try:
            self._check_realtime_exit(pair, current_price)
        except Exception as e:
            logger.error(f"Error handling volume spike: {e}")
            
    def _sync_positions_with_exchange(self):
        """Reconcile bot state with what Kraken actually holds.
        
        This is the critical function that prevents position tracking drift.
        Kraken's balance is the source of truth, not our state file.
        """
        try:
            balance = self.client.get_account_balance()
            if not balance:
                logger.error("Cannot sync — failed to fetch Kraken balance")
                return

            # 1. Check each position in our state against Kraken's actual holdings
            for symbol in list(self.state["positions"].keys()):
                pos = self.state["positions"][symbol]
                base, quote = self.client.get_pair_assets(symbol)

                if not base:
                    logger.warning(f"Cannot resolve base asset for {symbol}")
                    continue

                kraken_qty = float(balance.get(base, 0))

                if kraken_qty <= 0:
                    # We think we have a position, but Kraken says we don't.
                    # Could be: order never filled, or position was sold externally.
                    logger.warning(
                        f"[SYNC] {symbol}: state says position of {pos['size']:.6f} "
                        f"but Kraken holds 0 {base}. Removing from state."
                    )
                    # Try to check if the entry order ever filled
                    order_id = pos.get("order_id")
                    if order_id:
                        order_info = self.client.query_orders([order_id])
                        if order_info:
                            status = order_info.get(order_id, {}).get("status", "unknown")
                            logger.info(f"  → Entry order {order_id} status: {status}")

                    del self.state["positions"][symbol]
                else:
                    # Kraken has the asset. Update our size to match reality.
                    if abs(kraken_qty - pos["size"]) / max(pos["size"], 0.0001) > 0.01:
                        logger.info(
                            f"[SYNC] {symbol}: adjusting size {pos['size']:.6f} → {kraken_qty:.6f} "
                            f"to match Kraken"
                        )
                        pos["size"] = kraken_qty

            # 2. Check for assets on Kraken that we DON'T have in state
            #    (manual trades, or state was lost)
            known_bases = set()
            for symbol in self.state["positions"]:
                base, _ = self.client.get_pair_assets(symbol)
                if base:
                    known_bases.add(base)

            skip_assets = {"USD", "USDT", "USDC"}  # Stablecoins, not positions
            for asset, qty_str in balance.items():
                clean_asset = asset
                # Normalize Kraken prefixes (XXBT -> XBT, ZUSD -> USD)
                if len(clean_asset) > 3 and clean_asset[0] in {"X", "Z"}:
                    clean_asset = clean_asset[1:]
                    
                qty = float(qty_str)
                if qty > 0 and clean_asset not in skip_assets and clean_asset not in known_bases:
                    # Check if it's worth more than $1 to avoid dust
                    ticker = self.client.get_ticker(f"{clean_asset}USD")
                    if ticker:
                        value = qty * float(ticker["price"])
                        if value > 1.0:
                            logger.warning(
                                f"[SYNC] Found {qty:.6f} {clean_asset} (${value:.2f}) on Kraken "
                                f"not tracked in state! Manual trade or state loss."
                            )

            # 3. Check open orders — cancel stale ones not tied to positions
            #    BUT skip orders for untracked assets (manual trades / state loss)
            open_orders = self.client.get_open_orders()
            if open_orders and isinstance(open_orders, dict):
                orders_dict = open_orders.get("open", open_orders)
                tracked_order_ids = set()
                for pos in self.state["positions"].values():
                    oid = pos.get("order_id")
                    if oid:
                        tracked_order_ids.add(oid)
                    exit_oid = pos.get("exit_order_id")
                    if exit_oid:
                        tracked_order_ids.add(exit_oid)

                # Build set of assets we found on Kraken but don't track
                untracked_assets = set()
                for asset, qty_str in balance.items():
                    clean_asset = asset
                    if len(clean_asset) > 3 and clean_asset[0] in {"X", "Z"}:
                        clean_asset = clean_asset[1:]
                    qty = float(qty_str)
                    if qty > 0 and clean_asset not in skip_assets and clean_asset not in known_bases:
                        untracked_assets.add(clean_asset.upper())

                for order_id, order_data in orders_dict.items():
                    if order_id not in tracked_order_ids:
                        desc = order_data.get("descr", {})
                        order_str = desc.get("order", "")
                        pair = desc.get("pair", "")
                        # Extract base asset from pair (e.g. "RIVERUSD" -> "RIVER")
                        order_base = pair.replace("USD", "").replace("USDT", "").upper() if pair else ""
                        if order_base and order_base in untracked_assets:
                            logger.info(
                                f"[SYNC] Keeping open order {order_id} for untracked asset "
                                f"{order_base}: {order_str} — not cancelling (manual/external trade)"
                            )
                        else:
                            logger.warning(
                                f"[SYNC] Stale open order {order_id}: {order_str} "
                                f"— cancelling"
                            )
                            self.client.cancel_order(order_id)

            logger.info(
                f"[SYNC] Complete — {len(self.state['positions'])} positions tracked, "
                f"Kraken balance: ${float(balance.get('USD', 0)):.2f} USD"
            )

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    def _check_pending_entry(self, symbol: str) -> bool:
        """Check if a pending limit entry order has filled.
        
        Returns True if position is confirmed (filled or already held).
        Returns False if order is still pending or was cancelled/expired.
        Cleans up state if order didn't fill.
        """
        pos = self.state["positions"].get(symbol)
        if not pos:
            return False

        order_id = pos.get("order_id")
        if not order_id:
            return True  # No order to track, assume filled

        # Check if we actually hold the asset
        balance = self.client.get_account_balance()
        base, _ = self.client.get_pair_assets(symbol)
        if base and balance:
            kraken_qty = float(balance.get(base, 0))
            if kraken_qty > 0:
                pos["size"] = kraken_qty  # Update to actual
                pos.pop("pending_entry", None)
                return True

        # Check order status
        order_info = self.client.query_orders([order_id])
        if not order_info:
            return False

        status = order_info.get(order_id, {}).get("status", "unknown")

        if status == "closed":
            # Order filled
            pos.pop("pending_entry", None)
            logger.info(f"[ENTRY FILLED] {symbol} limit order {order_id} confirmed")
            return True
        elif status == "canceled" or status == "expired":
            # Order didn't fill — remove position from state
            logger.info(f"[ENTRY CANCELLED] {symbol} order {order_id} was {status}, removing")
            del self.state["positions"][symbol]
            return False
        elif status == "open" or status == "pending":
            # Still waiting — check if it's been too long (10 min)
            entry_time = datetime.fromisoformat(pos["entry_time"])
            waited = (datetime.now(timezone.utc) - entry_time).total_seconds()
            if waited > 600:  # 10 minutes
                logger.info(f"[ENTRY TIMEOUT] {symbol} limit order unfilled after {waited/60:.0f}min, cancelling")
                self.client.cancel_order(order_id)
                del self.state["positions"][symbol]
                return False
            return False  # Still waiting
        else:
            logger.warning(f"[ENTRY] {symbol} order {order_id} unknown status: {status}")
            return False
            
    def run_cycle(self):
        """Run one trading cycle"""
        try:
            # Clear per-cycle caches
            self._cached_fg_index = None
            
            logger.debug("Starting trading cycle")
            
            # 1. Reset daily/monthly tracking if needed
            self._reset_daily_tracking()
            self._reset_monthly_tracking()
            
            # 2. Check drawdown limits
            if not self._check_drawdown_limits():
                logger.warning("Drawdown limits exceeded, skipping cycle")
                return
                
            # 3. Sync balance
            balance = self.client.get_account_balance()
            if not balance or "USD" not in balance:
                logger.warning("No balance data available")
                return
                
            # 4. Detect regime
            regime = self._detect_regime()
            
            # 5. Sync with Kraken (source of truth)
            self._sync_positions_with_exchange()
            
            # 6. Check pending limit entries (cancel if stale)
            for symbol in list(self.state["positions"].keys()):
                if self.state["positions"].get(symbol, {}).get("pending_entry"):
                    self._check_pending_entry(symbol)
            
            # 7. Manage existing positions (exits)
            self._manage_position_exits(regime)
            
            # 8. Look for new entries (if we have room)
            params = self.REGIME_PARAMS[regime]
            current_positions = len(self.state["positions"])
            
            if current_positions < params["max_positions"]:
                signals = self._generate_signals(regime)
                
                # Place orders for top signals
                for signal in signals:
                    if len(self.state["positions"]) >= params["max_positions"]:
                        break
                        
                    if self._place_entry_order(signal):
                        break  # Only one entry per cycle
                        
            # 7. Log status
            self._log_balance_snapshot(regime)
            usd_balance = float(balance["USD"])
            
            logger.info(
                f"Cycle complete - Balance: ${usd_balance:.2f} | "
                f"Regime: {regime} | Positions: {len(self.state['positions'])} | "
                f"Daily PnL: ${self.state['daily_pnl']:.2f}"
            )
            
            # 8. Save state
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            
    def start_websocket_monitor(self):
        """Start the websocket crash monitor"""
        try:
            self.crash_monitor = CrashMonitor(
                on_crash_callback=self._on_crash,
                on_pump_callback=self._on_pump,
                on_volume_spike_callback=self._on_volume_spike
            )
            self.crash_monitor.start()
            logger.info("🔌 Websocket crash monitor started")
            
        except Exception as e:
            logger.warning(f"Failed to start websocket monitor: {e}")
            
    def run(self):
        """Main bot loop"""
        logger.info("Starting V2 Trading Bot")
        logger.info(f"Trading pairs: {', '.join(self.PAIRS)}")
        
        # Sync positions on startup
        self._sync_positions_with_exchange()
        
        # Start websocket monitor
        self.start_websocket_monitor()
        
        self.running = True
        cycle_interval = 300  # 5 minutes
        rt_check_interval = 10  # Check exits every 10 seconds via websocket prices
        
        try:
            while self.running:
                cycle_start = time.time()
                
                # Run trading cycle
                self.run_cycle()
                
                # Between cycles: check exits every 10s using websocket prices
                cycle_duration = time.time() - cycle_start
                remaining = max(0, cycle_interval - cycle_duration)
                
                while remaining > 0 and self.running:
                    sleep_chunk = min(rt_check_interval, remaining)
                    time.sleep(sleep_chunk)
                    remaining -= sleep_chunk
                    
                    # Real-time exit checks using websocket latest prices
                    if self.crash_monitor and self.state.get("positions"):
                        for symbol in list(self.state["positions"].keys()):
                            ws_price = self.crash_monitor.get_price(symbol)
                            if ws_price and ws_price > 0:
                                self._check_realtime_exit(symbol, ws_price)
                    
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.running = False
            if self.crash_monitor:
                self.crash_monitor.stop()
            logger.info("V2 Trading Bot stopped")

def main():
    """Main entry point"""
    # Check environment
    if not os.getenv("KRAKEN_API_KEY") or not os.getenv("KRAKEN_PRIVATE_KEY"):
        logger.error("Missing Kraken API credentials in .env file")
        return 1
        
    # Initialize and run bot
    bot = TradingBot()
    
    try:
        bot.run()
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())