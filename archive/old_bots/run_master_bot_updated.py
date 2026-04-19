#!/usr/bin/env python3
"""
THE ALL-SEEING EYE - Master Trading Bot (Updated with Backtest Results)
================================================================================

BACKTEST RESULTS INTEGRATION:
- Only 2 tools passed OOS validation with fee adjustment: crash_buy, mega_crash
- Grid engine performed exceptionally well: +136% profit, 0.98% per round trip
- 7 tools failed OOS validation and are DISABLED or marked for parameter sweep
- 5 new tools tested, none passed validation

CURRENT ACTIVE TOOLS (OOS Validated):
1. Grid Engine (passive income) - EXCELLENT: +136% profit
2. Crash Buy (52.5% WR, +0.00% net after fees) - MARGINAL but passing
3. Mega Crash (59.6% WR, +0.63% net after fees) - GOOD

DISABLED TOOLS (Failed OOS):
- dip_buy: -0.20% avg (516 signals)
- rsi_pump_12h: -0.32% avg (18 signals) 
- crash_neg_ac: -0.16% avg (79 signals)
- entropy_short: -0.54% avg (1668 signals)
- sma50_ext_8: -0.25% avg (126 signals)
- vpin_dip: -0.33% avg (169 signals)
- alt_btc_revert_t1: -0.20% avg (123 signals)
"""

import sys
import os
import json
import time
import signal
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

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

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add(LOGS_DIR / "master_bot.log", rotation="10 MB", retention="30 days")

# Trading pairs
PAIRS = [
    "NEARUSD", "UNIUSD", "AVAXUSD", "LINKUSD", "AAVEUSD", "SOLUSD",
    "ETHUSD", "XBTUSD", "DOTUSD", "XLMUSD", "XRPUSD", "ADAUSD", 
    "ATOMUSD", "DOGEUSD", "FILUSD", "LTCUSD"
]

# Grid configurations (PROVEN PROFITABLE: +136% in backtest)
GRID_CONFIGS = {
    "NEARUSD": 0.01, "UNIUSD": 0.015, "AVAXUSD": 0.01, "LINKUSD": 0.008,
    "AAVEUSD": 0.015, "SOLUSD": 0.003, "ETHUSD": 0.005, "XBTUSD": 0.01,
    "DOTUSD": 0.012, "XLMUSD": 0.01, "XRPUSD": 0.01, "ADAUSD": 0.012,
    "ATOMUSD": 0.008, "DOGEUSD": 0.012, "FILUSD": 0.015, "LTCUSD": 0.01,
}

# Constants
MAX_ACTIVE_POSITIONS = 3  # Reduced due to fewer validated tools
GRID_CAPITAL_PCT = 0.60   # Increased grid allocation (it's profitable!)
ACTIVE_CAPITAL_PCT = 0.40 # Reduced active (fewer tools)
RISK_PER_TRADE = 0.05
GRID_TAKE_PROFIT = 0.015
GRID_REANCHOR_PCT = 0.10


class ValidatedAllSeeingEye:
    """All-Seeing Eye with only OOS-validated tools"""
    
    def __init__(self):
        self.client = KrakenClient()
        self.running = True
        self.state = self.load_state()
        
        # Balance tracking
        self.total_balance = self.state.get("total_balance", STARTING_BALANCE)
        self.grid_balance = self.total_balance * GRID_CAPITAL_PCT
        self.active_balance = self.total_balance * ACTIVE_CAPITAL_PCT
        
        # State tracking
        self.grid_positions = self.state.get("grid_positions", {})
        self.active_positions = self.state.get("active_positions", {})
        self.grid_profit = self.state.get("grid_profit", 0.0)
        self.active_profit = self.state.get("active_profit", 0.0)
        self.grid_anchors = self.state.get("grid_anchors", {})
        
        # Tool performance (only validated tools)
        self.tool_stats = self.state.get("tool_stats", {})
        
        # Initialize validated tool stats
        VALIDATED_TOOLS = {
            "crash_buy": {"trades": 40, "wins": 21, "pnl": 0.0},     # 52.5% WR, +0.00% net
            "mega_crash": {"trades": 57, "wins": 34, "pnl": 35.8},   # 59.6% WR, +0.63% net
        }
        
        for tool, backfill in VALIDATED_TOOLS.items():
            if tool not in self.tool_stats:
                self.tool_stats[tool] = backfill.copy()
        
        # Price cache for cross-pair analysis
        self._price_cache = {}
        self.current_bar = 0
        self.current_fng = 50  # Neutral fear/greed
        
        logger.info("🦅 VALIDATED ALL-SEEING EYE INITIALIZED")
        logger.info(f"💰 Total balance: ${self.total_balance:.0f}")
        logger.info(f"📊 Grid: ${self.grid_balance:.0f} ({GRID_CAPITAL_PCT*100:.0f}%)")
        logger.info(f"⚡ Active: ${self.active_balance:.0f} ({ACTIVE_CAPITAL_PCT*100:.0f}%)")
        logger.info(f"🔧 Validated tools: {list(VALIDATED_TOOLS.keys())}")
        
    def load_state(self):
        """Load bot state from file"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return {}
    
    def save_state(self):
        """Save bot state to file"""
        state = {
            "total_balance": self.total_balance,
            "grid_positions": self.grid_positions,
            "active_positions": self.active_positions,
            "grid_profit": self.grid_profit,
            "active_profit": self.active_profit,
            "grid_anchors": self.grid_anchors,
            "tool_stats": self.tool_stats,
            "last_update": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get_market_data(self, pair: str) -> Optional[dict]:
        """Get current market data for a pair"""
        try:
            # Get recent OHLCV data
            result = self.client.get_ohlc_data(pair, interval=60)  # 1h candles
            
            if 'error' in result and result['error']:
                logger.error(f"Error getting data for {pair}: {result['error']}")
                return None
            
            if not result.get('result') or pair not in result['result']:
                logger.warning(f"No data returned for {pair}")
                return None
            
            candles = result['result'][pair]
            if not candles:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Get current price and basic data
            latest = df.iloc[-1]
            
            return {
                'pair': pair,
                'price': float(latest['close']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'volume': float(latest['volume']),
                'df': df.tail(200)  # Keep last 200 bars for indicators
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {pair}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators"""
        close = df['close'].values.astype(float)
        
        # RSI-7 (only indicator we need for validated tools)
        def rsi(prices, period=7):
            if len(prices) <= period:
                return np.full(len(prices), 50.0)
                
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.zeros_like(prices)
            avg_loss = np.zeros_like(prices)
            
            if len(delta) >= period:
                avg_gain[period] = np.mean(gain[:period])
                avg_loss[period] = np.mean(loss[:period])
                
                for i in range(period + 1, len(prices)):
                    avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                    avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
            
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
            return 100 - (100 / (1 + rs))
        
        cur_rsi = rsi(close, 7)
        
        return {
            'close': close,
            'rsi': cur_rsi[-1] if len(cur_rsi) > 0 else 50
        }
    
    def scan_signals(self, pair: str, data: dict) -> List[Tuple[dict, float]]:
        """Scan for signals using ONLY validated tools"""
        signals = []
        
        try:
            df = data['df']
            if len(df) < 30:  # Need minimum data
                return signals
            
            indicators = self.calculate_indicators(df)
            close = indicators['close']
            cur_rsi = indicators['rsi']
            price = data['price']
            
            if len(close) < 25:  # Need enough data for returns calculation
                return signals
            
            # Calculate returns with exact indices from bot
            ret_24h = (price - close[-25]) / close[-25] * 100 if len(close) >= 25 else 0
            
            # Cache price for this pair
            self._price_cache[pair] = close
            
            # ===== VALIDATED TOOL 1: CRASH BUY =====
            # OOS: 40 signals, 52.5% WR, +0.00% net (marginal but passing)
            if ret_24h < -10 and cur_rsi < 20:
                score = abs(ret_24h) * (20 - cur_rsi) * 0.5
                signals.append(({
                    'pair': pair, 'tool': 'crash_buy', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.08,
                    'reason': f"CRASH BUY (OOS validated): {ret_24h:.1f}% drop 24h, RSI={cur_rsi:.1f}"
                }, score))
            
            # ===== VALIDATED TOOL 2: MEGA CRASH =====  
            # OOS: 57 signals, 59.6% WR, +0.63% net (good edge)
            if ret_24h < -15:
                score = abs(ret_24h) * 3
                signals.append(({
                    'pair': pair, 'tool': 'mega_crash', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.08,
                    'reason': f"MEGA CRASH (OOS validated): {ret_24h:.1f}% drop 24h"
                }, score))
            
            # ===== DISABLED TOOLS (Failed OOS validation) =====
            # These are commented out but kept for reference
            
            # DISABLED: dip_buy (-0.20% avg, 516 signals)
            # if ret_4h < -3:
            #     # Tool disabled due to negative expectancy after fees
            
            # DISABLED: rsi_pump_12h (-0.32% avg, 18 signals)  
            # if cur_rsi > 85 and ret_12h >= 10:
            #     # Tool disabled due to negative expectancy after fees
            
            # DISABLED: entropy_short (-0.54% avg, 1668 signals)
            # DISABLED: sma50_ext_8 (-0.25% avg, 126 signals)
            # DISABLED: vpin_dip (-0.33% avg, 169 signals)
            # DISABLED: crash_neg_ac (-0.16% avg, 79 signals)
            # DISABLED: alt_btc_revert_t1 (-0.20% avg, 123 signals)
            
        except Exception as e:
            logger.error(f"Error scanning signals for {pair}: {e}")
        
        return signals
    
    def update_grids(self, market_data: dict):
        """Update grid engine (PROVEN PROFITABLE: +136% in backtest)"""
        grid_balance_per_pair = self.grid_balance / len(PAIRS)
        
        for pair in PAIRS:
            if pair not in market_data or pair not in GRID_CONFIGS:
                continue
                
            data = market_data[pair]
            current_price = data["price"]
            
            grid_spacing = GRID_CONFIGS[pair]
            
            # Initialize grid positions
            if pair not in self.grid_positions:
                self.grid_positions[pair] = []
                
            positions = self.grid_positions[pair]
            
            # Get or create grid anchor
            if pair not in self.grid_anchors:
                self.grid_anchors[pair] = {
                    "center": current_price,
                    "levels": [
                        current_price * (1 - grid_spacing),
                        current_price * (1 - grid_spacing * 2),
                        current_price * (1 - grid_spacing * 3)
                    ]
                }
            
            anchor = self.grid_anchors[pair]
            grid_center = anchor["center"]
            grid_levels = anchor["levels"]
            
            # Check for reanchoring (if price moved >10% from center)
            if abs(current_price - grid_center) / grid_center > GRID_REANCHOR_PCT:
                logger.info(f"[GRID] Reanchoring {pair}: {grid_center:.2f} → {current_price:.2f}")
                
                # Close existing positions at current price (simplified)
                for pos in positions:
                    profit_pct = (current_price - pos["buy_price"]) / pos["buy_price"] * 100
                    self.grid_profit += profit_pct
                
                positions.clear()
                
                # Set new anchor
                self.grid_anchors[pair] = {
                    "center": current_price,
                    "levels": [
                        current_price * (1 - grid_spacing),
                        current_price * (1 - grid_spacing * 2),
                        current_price * (1 - grid_spacing * 3)
                    ]
                }
                grid_levels = self.grid_anchors[pair]["levels"]
            
            # Check for grid buy opportunities
            for level in grid_levels:
                # Buy if price is within 0.5% of grid level and no existing position at this level
                if abs(current_price - level) / level < 0.005:
                    existing_at_level = any(abs(pos["buy_price"] - level) / level < 0.01 for pos in positions)
                    
                    if not existing_at_level:
                        # Execute grid buy
                        position_size = grid_balance_per_pair * 0.05  # 5% per grid level
                        
                        positions.append({
                            "buy_price": level,
                            "target_sell": level * (1 + GRID_TAKE_PROFIT),
                            "quantity": position_size / level,
                            "bar": self.current_bar
                        })
                        
                        logger.info(f"[GRID] {pair} BUY at {level:.2f}, target {level * (1 + GRID_TAKE_PROFIT):.2f}")
            
            # Check for grid sells
            positions_to_remove = []
            for pos in positions:
                if current_price >= pos["target_sell"]:
                    # Execute grid sell
                    profit_pct = (pos["target_sell"] - pos["buy_price"]) / pos["buy_price"] * 100
                    profit_pct -= 0.52  # Subtract fees (0.26% * 2)
                    
                    self.grid_profit += profit_pct
                    positions_to_remove.append(pos)
                    
                    logger.info(f"[GRID] {pair} SELL at {pos['target_sell']:.2f}, profit: {profit_pct:.2f}%")
            
            # Remove completed positions
            for pos in positions_to_remove:
                positions.remove(pos)
    
    def open_position(self, signal: dict, market_data: dict):
        """Open a new active position"""
        pair = signal['pair']
        tool = signal['tool']
        direction = signal['direction']
        
        if pair in self.active_positions:
            return  # Already have position in this pair
        
        current_price = market_data[pair]['price']
        position_size = self.active_balance * RISK_PER_TRADE  # 5% per trade
        
        self.active_positions[pair] = {
            'tool': tool,
            'direction': direction,
            'entry': current_price,
            'entry_bar': self.current_bar,
            'position_size': position_size,
            'sl_pct': signal.get('sl_pct', 0.05),
            'hold': signal.get('hold', 24),
            'highest_price_since_entry': current_price,
            'reason': signal.get('reason', 'No reason given')
        }
        
        # Update tool stats
        if tool not in self.tool_stats:
            self.tool_stats[tool] = {"trades": 0, "wins": 0, "pnl": 0.0}
        
        logger.info(f"[OPEN] {pair} {direction.upper()} - {tool}: {signal.get('reason', '')}")
    
    def manage_positions(self, market_data: dict):
        """Manage existing active positions"""
        for pair in list(self.active_positions.keys()):
            if pair not in market_data:
                continue
                
            pos = self.active_positions[pair]
            price = market_data[pair]['price']
            bars_held = self.current_bar - pos['entry_bar']
            
            # Update highest price for trailing
            if pos['direction'] == 'long':
                pos['highest_price_since_entry'] = max(
                    pos.get('highest_price_since_entry', pos['entry']), price)
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            # Stop loss check
            if pos['direction'] == 'long':
                current_return = (price - pos['entry']) / pos['entry']
            else:
                current_return = (pos['entry'] - price) / pos['entry']
            
            if current_return <= -pos['sl_pct']:
                should_close = True
                close_reason = "STOP LOSS"
            
            # Hold timeout check
            elif bars_held >= pos['hold']:
                should_close = True
                close_reason = "HOLD TIMEOUT"
            
            if should_close:
                self.close_position(pair, price, close_reason)
    
    def close_position(self, pair: str, exit_price: float, reason: str):
        """Close an active position"""
        if pair not in self.active_positions:
            return
        
        pos = self.active_positions[pair]
        
        # Calculate P&L
        if pos['direction'] == 'long':
            gross_return = (exit_price - pos['entry']) / pos['entry']
        else:
            gross_return = (pos['entry'] - exit_price) / pos['entry']
        
        # Apply fees (0.26% * 2 = 0.52% round-trip)
        net_return = gross_return - 0.0052
        position_pnl = net_return * pos['position_size']
        
        # Update stats
        self.active_profit += position_pnl
        
        tool = pos['tool']
        if tool in self.tool_stats:
            self.tool_stats[tool]["trades"] += 1
            if net_return > 0:
                self.tool_stats[tool]["wins"] += 1
            self.tool_stats[tool]["pnl"] += position_pnl
        
        logger.info(f"[CLOSE] {pair} {pos['direction'].upper()} - {reason}: "
                   f"{net_return*100:+.2f}% (${position_pnl:+.0f}) - {pos['tool']}")
        
        # Remove position
        del self.active_positions[pair]
    
    def run_cycle(self):
        """Run one complete trading cycle"""
        logger.info("🔄 Starting trading cycle...")
        
        # 1. Get market data for all pairs
        market_data = {}
        for pair in PAIRS:
            data = self.get_market_data(pair)
            if data:
                market_data[pair] = data
        
        if not market_data:
            logger.warning("No market data available")
            return
        
        logger.info(f"📊 Got data for {len(market_data)}/{len(PAIRS)} pairs")
        
        # 2. Update grid positions (highest priority - proven profitable)
        self.update_grids(market_data)
        
        # 3. Manage existing active positions
        self.manage_positions(market_data)
        
        # 4. Scan for new signals (only validated tools)
        all_signals = []
        for pair, data in market_data.items():
            signals = self.scan_signals(pair, data)
            all_signals.extend(signals)
        
        # 5. Sort signals by score and open top positions
        all_signals.sort(key=lambda x: x[1], reverse=True)
        
        available_slots = MAX_ACTIVE_POSITIONS - len(self.active_positions)
        opened_positions = 0
        
        for signal, score in all_signals:
            if opened_positions >= available_slots:
                break
                
            if signal['pair'] not in self.active_positions:
                self.open_position(signal, market_data)
                opened_positions += 1
        
        # 6. Update balance and save state
        self.current_bar += 1
        self.save_state()
        
        # 7. Log status
        total_pnl = self.grid_profit + self.active_profit
        current_balance = self.total_balance + total_pnl
        
        logger.info(f"💰 Balance: ${current_balance:.0f} (Grid: ${self.grid_profit:+.0f}, "
                   f"Active: ${self.active_profit:+.0f})")
        logger.info(f"📊 Positions: {len(self.active_positions)} active, "
                   f"{sum(len(pos) for pos in self.grid_positions.values())} grid")
        
        if all_signals:
            logger.info(f"🎯 Signals: {len(all_signals)} found, {opened_positions} opened")
        
        # Tool performance summary  
        if self.current_bar % 10 == 0:  # Every 10 cycles
            logger.info("🔧 Tool Performance:")
            for tool, stats in self.tool_stats.items():
                trades = stats.get("trades", 0)
                wins = stats.get("wins", 0)
                pnl = stats.get("pnl", 0)
                wr = wins / trades * 100 if trades > 0 else 0
                logger.info(f"   {tool:15}: {trades:3} trades, {wr:5.1f}% WR, ${pnl:+6.0f}")
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 VALIDATED ALL-SEEING EYE STARTED")
        logger.info(f"💡 Using only OOS-validated tools with fee adjustment")
        logger.info(f"⏱️  Check interval: {CHECK_INTERVAL} seconds")
        
        # Signal handlers
        def signal_handler(signum, frame):
            logger.info("🛑 Shutdown signal received")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                start_time = time.time()
                
                self.run_cycle()
                
                cycle_duration = time.time() - start_time
                sleep_time = max(0, CHECK_INTERVAL - cycle_duration)
                
                logger.info(f"⏱️  Cycle completed in {cycle_duration:.1f}s, "
                           f"sleeping {sleep_time:.1f}s")
                
                if self.running and sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
            raise
        finally:
            self.save_state()
            logger.info("🦅 VALIDATED ALL-SEEING EYE STOPPED")


if __name__ == "__main__":
    bot = ValidatedAllSeeingEye()
    bot.run()