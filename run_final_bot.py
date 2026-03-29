#!/usr/bin/env python3
"""
FINAL TRADING BOT - THE ULTIMATE VERSION
Maximum profit extraction with 8 major upgrades from the production bot.

UPGRADES:
1. Kraken Pro fees (0.25% maker / 0.40% taker) with LIMIT entries
2. 40 pairs (up from 16)
3. 2x margin on Tier 1 signals
4. Dynamic capital allocation based on Fear & Greed
5. Scaling position sizes with growing balance
6. Smarter grid system (ATR-adaptive, more levels for volatile pairs)
7. Consecutive win/loss tracking per tool
8. Enhanced status logging
"""

import sys
import os
import json
import time
import signal
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

# Add src directory for kraken_client
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from kraken_client import KrakenClient
    from ws_monitor import CrashMonitor
except ImportError as e:
    logger.error(f"Failed to import KrakenClient: {e}")
    sys.exit(1)

# Configuration
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "300"))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = DATA_DIR / "final_bot_state.json"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add(LOGS_DIR / "final_bot.log", rotation="10 MB", retention="30 days")

# UPGRADE 2: 40 pairs (expanded from 16)
ORIGINAL_PAIRS = [
    "NEARUSD", "UNIUSD", "AVAXUSD", "LINKUSD", "AAVEUSD", "SOLUSD",
    "ETHUSD", "XBTUSD", "DOTUSD", "XLMUSD", "XRPUSD", "ADAUSD", 
    "ATOMUSD", "DOGEUSD", "FILUSD", "LTCUSD"
]

NEW_PAIRS = [
    "SUIUSD", "PEPEUSD", "SHIBUSD", "BNBUSD", "TRXUSD",
    "HBARUSD", "HYPEUSD", "TAOUSD", "OKBUSD", "INJUSD",
    "ARBUSD", "OPUSD", "APTUSD", "TIAUSD", "ONDOUSD",
    "RENDERUSD", "JUPUSD", "ICPUSD", "LDOUSD", "BCHUSD",
    "STXUSD", "KAVAUSD", "ENAUSD", "FLOKIUSD"
]

PAIRS = ORIGINAL_PAIRS + NEW_PAIRS  # 40 total

# Dynamic pair selection — refresh volatile pairs every hour
VOLATILITY_REFRESH_INTERVAL = 3600  # 1 hour
MAX_TRADING_PAIRS = 60  # Top N most volatile pairs to scan
MIN_PAIR_VOLUME_USD = 1_000_000  # Min $1M 24h volume — real liquidity only
MIN_PAIR_PRICE_USD = 0.01       # No sub-penny coins — spread/slippage kills you
MAX_POSITION_PCT_OF_VOLUME = 0.005  # Never be more than 0.5% of daily volume

# Coins restricted for US:FL, known rugs, or naming collisions
GEO_BLOCKED_PAIRS = {'BLUAIUSD', 'B3USD', 'GUSD'}

# Grid configurations (ATR-based spacing per pair) - now with 40 pairs
GRID_CONFIGS = {
    # Original 16 pairs
    "NEARUSD": 0.01, "UNIUSD": 0.015, "AVAXUSD": 0.01, "LINKUSD": 0.008,
    "AAVEUSD": 0.015, "SOLUSD": 0.003, "ETHUSD": 0.005, "XBTUSD": 0.01,
    "DOTUSD": 0.012, "XLMUSD": 0.01, "XRPUSD": 0.01, "ADAUSD": 0.012,
    "ATOMUSD": 0.008, "DOGEUSD": 0.012, "FILUSD": 0.015, "LTCUSD": 0.01,
    
    # New 24 pairs - default 1.2% spacing
    "SUIUSD": 0.012, "PEPEUSD": 0.012, "SHIBUSD": 0.012, "BNBUSD": 0.012, "TRXUSD": 0.012,
    "HBARUSD": 0.012, "HYPEUSD": 0.012, "TAOUSD": 0.012, "OKBUSD": 0.012, "INJUSD": 0.012,
    "ARBUSD": 0.012, "OPUSD": 0.012, "APTUSD": 0.012, "TIAUSD": 0.012, "ONDOUSD": 0.012,
    "RENDERUSD": 0.012, "JUPUSD": 0.012, "ICPUSD": 0.012, "LDOUSD": 0.012, "BCHUSD": 0.012,
    "STXUSD": 0.012, "KAVAUSD": 0.012, "ENAUSD": 0.012, "FLOKIUSD": 0.012
}

# Constants
MAX_ACTIVE_POSITIONS = 8    # Max simultaneous active positions
RISK_PER_TRADE = 0.08       # 8% of active balance per trade
GRID_REANCHOR_PCT = 0.10    # Reanchor grid when price moves >10% from center
LIMIT_ORDER_TIMEOUT = 2     # Cancel and re-place limit orders after 2 cycles (10 min)
MAX_LIMIT_RETRIES = 3       # Max re-places before giving up (3 retries = ~15 min total)
PRICE_DRIFT_ABANDON = 0.02  # Abandon pending order if price drifted >2% from original entry

# Multi-timeframe confirmation
ENABLE_MTF = True  # Multi-timeframe confirmation
MTF_BOOST = 1.3    # Score multiplier when HTF aligns
MTF_PENALTY = 0.4  # Score multiplier when HTF conflicts (was 0.6, too mild)

# UPGRADE 1: Kraken Pro fees (corrected)
ENTRY_FEE = 0.0025         # 0.25% maker fee for limit orders
EXIT_FEE = 0.0025          # 0.25% maker fee (post-only limit exits)
ROUND_TRIP_FEE = 0.0065    # Mixed round trip: 0.25% entry + 0.40% exit = 0.65%

# Grid fees are both limit orders (maker/maker)
GRID_ROUND_TRIP_FEE = 0.005  # 0.25% entry + 0.25% exit = 0.50%
GRID_FEE_MULTIPLIER = 0.995  # Updated from 0.996

# UPGRADE 3: Tier 1 tools with 2x margin
TIER1_TOOLS = {
    'crash_buy', 'volatile_oversold', 'crash_neg_ac', 'blood_in_streets',
    'quick_crash', 'crash_mean_revert', 'mega_pump_sell_t1', 'rsi_pump_8h',
    'mega_crash', 'vpin_dip'
}

# Margin costs for 2x leverage
MARGIN_COST_OPEN = 0.0002   # 0.02% to open margin position
MARGIN_COST_PER_BAR = 0.0002  # 0.02% every 4 hours (per bar)

# VALIDATED TOOLS - exact same 30 from production bot
# CRASH/BEAR TOOLS (LONG) - 15 tools
CRASH_BEAR_TOOLS = [
    "volatile_oversold", "crash_buy", "mega_crash", "crash_neg_ac", "blood_in_streets",
    "quick_crash", "crash_mean_revert", "vpin_dip", "market_panic_70", "flash_crash",
    "deep_dip_8h", "entropy_dip", "vpin_toxic", "btc_alt_spread", "quick_dip"
]

# BULL/GREED TOOLS (SHORT) - 13 tools  
BULL_GREED_TOOLS = [
    "mega_pump_sell_t1", "rsi_pump_8h", "falling_wedge_short", "greed_short_t2", "thursday_short",
    "mega_pump_sell_t2", "distribution_short", "late_us_short", "rsi_pump_12h", "ema_cross_short",
    "rsi_pump_fat_tail", "entropy_short", "alt_btc_revert_t3"
]

# NEUTRAL/TRANSITION TOOLS - 2 tools
NEUTRAL_TOOLS = ["month_start_long", "dip_buy_5pct"]

# BULL MOMENTUM TOOLS (LONG) - validated on 3yr OOS, bull regime only
BULL_MOMENTUM_TOOLS = ["accumulation_breakout", "hurst_trend_long"]

# BULL SWING TOOLS (LONG) - simple trend-following, 15% trail, hold weeks
# Validated: 3yr OOS, bull regime only, 0.65% fees
BULL_SWING_TOOLS = [
    "buy_weekly_green",       # 257 sig, 44% WR, +5.92%/trade, PF=1.96
    "buy_breakout_simple",    # 123 sig, 44% WR, +6.54%/trade, PF=2.20
    "simple_buy_uptrend",     # 347 sig, 41% WR, +3.55%/trade, PF=1.52
    "buy_btc_leading",        # 103 sig, 31% WR, +4.64%/trade, PF=1.52
]

# All validated tools combined
VALIDATED_TOOLS = CRASH_BEAR_TOOLS + BULL_GREED_TOOLS + NEUTRAL_TOOLS + BULL_MOMENTUM_TOOLS + BULL_SWING_TOOLS


class FinalTradingBot:
    """Final trading bot with all 8 upgrades."""
    
    def __init__(self):
        self.client = KrakenClient()
        self.running = True
        
        # Real-time crash detection via websocket
        self.crash_monitor = None
        self.pending_crash_signals = []  # Queue of crash events from websocket
        self.pending_pump_signals = []   # Queue of pump events for exit tightening
        self.pending_volume_spikes = []  # Queue of volume spikes
        try:
            self.crash_monitor = CrashMonitor(
                on_crash_callback=self._on_crash_detected,
                on_pump_callback=self._on_pump_detected,
                on_volume_spike_callback=self._on_volume_spike
            )
            if ENABLE_LIVE_TRADING:
                self.crash_monitor.start()
                logger.info("🔌 Real-time monitor: ACTIVE (crash + pump + volume)")
            else:
                logger.info("🔌 Real-time monitor: DISABLED (dry run)")
        except Exception as e:
            logger.warning(f"Real-time monitor failed to start: {e}")
        self.state = self.load_state()
        
        # UPGRADE 5: Balance tracking — syncs from Kraken account
        self.starting_balance = STARTING_BALANCE
        self.total_balance = self._sync_kraken_balance() or self.state.get("total_balance", STARTING_BALANCE)
        
        # Fear & Greed index (initialize before allocation)
        self.current_fng = 50
        
        # Get initial allocation (will be dynamic)
        grid_pct, active_pct = self.get_capital_allocation()
        self.grid_balance = self.total_balance * grid_pct
        self.active_balance = self.total_balance * active_pct
        
        # Grid state
        self.grid_positions = self.state.get("grid_positions", {})
        self.grid_profit = self.state.get("grid_profit", 0.0)
        self.grid_round_trips = self.state.get("grid_round_trips", 0)
        self.grid_anchors = self.state.get("grid_anchors", {})
        
        # Active positions
        self.active_positions = self.state.get("active_positions", {})
        self.active_profit = self.state.get("active_profit", 0.0)
        
        # Tool performance tracking with consecutive wins/losses
        self.tool_stats = self.state.get("tool_stats", {})
        self.tool_streaks = self.state.get("tool_streaks", {})  # UPGRADE 7: track streaks
        self._initialize_tool_stats()
        
        # Price cache for cross-pair signals
        self._price_cache = {}
        
        # Market regime: default to 75% bullish (conservative — don't open weak shorts on cold start)
        self._bullish_4h_pct = 75
        self._avg_rsi_4h = 55.0
        
        # Trade history and current bar
        self.trade_history = self.state.get("trade_history", [])
        self.current_bar = self.state.get("current_bar", 0)
        
        # UPGRADE 1: Pending limit orders tracking
        self.pending_limit_orders = self.state.get("pending_limit_orders", {})  # pair -> order_info
        
        # Pending EXIT orders — don't delete positions until exit fills
        self.pending_exit_orders = self.state.get("pending_exit_orders", {})  # pair -> exit_info
        
        # Trade journal — detailed CSV for post-analysis
        self.trade_journal_path = LOGS_DIR / "trade_journal.csv"
        self._init_trade_journal()
        
        # Daily tracking
        self._current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_stats = {
            "trades_opened": 0, "trades_closed": 0, 
            "wins": 0, "losses": 0, "pnl": 0.0,
            "start_balance": self.total_balance,
            "tool_pnl": {}
        }
        
        # Reconcile state with actual Kraken holdings/orders
        self._reconcile_with_kraken()
        
        logger.info(f"🚀 FINAL Trading Bot initialized with 8 UPGRADES")
        logger.info(f"Total balance: ${self.total_balance:.2f} (start: ${self.starting_balance:.2f})")
        logger.info(f"Growth: {(self.total_balance/self.starting_balance-1)*100:+.1f}%")
        logger.info(f"Grid balance: ${self.grid_balance:.2f} ({grid_pct:.0%})")
        logger.info(f"Active balance: ${self.active_balance:.2f} ({active_pct:.0%})")
        logger.info(f"Trading pairs: {len(PAIRS)} (expanded from 16)")
        logger.info(f"Tier 1 tools with 2x margin: {len(TIER1_TOOLS)}")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
        
    def _sync_kraken_balance(self) -> Optional[float]:
        """Pull total USD-equivalent balance from Kraken account.
        
        Sums USD cash + market value of all held assets via raw Ticker API.
        Returns None on failure so caller can fall back to internal tracking.
        """
        try:
            if not ENABLE_LIVE_TRADING:
                return None
            balances = self.client.get_account_balance()
            if not balances:
                logger.warning("Could not fetch Kraken balance, using internal tracking")
                return None
            
            total_usd = 0.0
            usd_assets = {'USD', 'ZUSD', 'USDT', 'USDC', 'DAI', 'USDG'}
            
            for asset, amount in balances.items():
                if asset in usd_assets:
                    total_usd += amount
                elif amount > 0 and amount * 0.01 > 0:  # Skip dust
                    # Use raw Ticker API with the pair format Kraken expects
                    pair_name = f"{asset}USD"
                    try:
                        result = self.client._request('public/Ticker', {'pair': pair_name})
                        if result:
                            # Result keys may differ from input (e.g., KAVAUSD → KAVAUSD)
                            for key, data in result.items():
                                if 'c' in data:  # 'c' = last trade [price, volume]
                                    last_price = float(data['c'][0])
                                    total_usd += amount * last_price
                                    break
                    except Exception:
                        pass  # Skip assets we can't price (dust)
            
            if total_usd > 0:
                logger.info(f"💰 Kraken account balance: ${total_usd:.2f}")
                return total_usd
            return None
        except Exception as e:
            logger.warning(f"Kraken balance sync failed: {e}")
            return None

    def _reconcile_with_kraken(self):
        """Sync bot state with actual Kraken holdings and open orders.
        
        On startup (and optionally each cycle), pull real data from Kraken:
        - Any held asset not in active_positions → create a position for it
        - Any open buy order not in pending_limit_orders → track it
        - Any open sell order not in pending_exit_orders → track it
        - Any active_position for an asset we don't hold → clean it up
        """
        if not ENABLE_LIVE_TRADING:
            return
        
        try:
            # 1. Get actual holdings
            balances = self.client.get_account_balance()
            if not balances:
                return
            
            usd_assets = {'USD', 'ZUSD', 'USDT', 'USDC', 'DAI', 'USDG'}
            held_assets = {}  # asset -> qty
            for asset, amount in balances.items():
                if asset not in usd_assets and amount > 0:
                    # Normalize asset name to match PAIRS format (e.g., KAVA -> KAVAUSD)
                    pair = f"{asset}USD"
                    # Also handle Kraken's X/Z prefix (XXBT -> XBTUSD, XETH -> ETHUSD)
                    if asset.startswith('X') and len(asset) == 4:
                        pair = f"{asset[1:]}USD"
                    elif asset.startswith('X') and len(asset) > 4:
                        pair = f"{asset}USD"
                    
                    # Check ALL held assets — don't filter by pair list
                    # If we hold it on Kraken, we need to track it regardless
                    if True:
                        # Get current price
                        try:
                            result = self.client._request('public/Ticker', {'pair': pair})
                            if result:
                                for key, data in result.items():
                                    if 'c' in data:
                                        price = float(data['c'][0])
                                        value = amount * price
                                        if value > 1.0:  # Skip dust (<$1)
                                            held_assets[pair] = {
                                                'qty': amount,
                                                'price': price,
                                                'value': value
                                            }
                                        break
                        except Exception:
                            pass
            
            # 2. Get open orders
            open_orders = self.client.get_open_orders()
            kraken_orders = {}  # txid -> order_info
            if isinstance(open_orders, dict) and 'open' in open_orders:
                kraken_orders = open_orders['open']
            elif isinstance(open_orders, dict):
                kraken_orders = open_orders
            
            # Parse open orders
            open_buys = {}   # pair -> order_info
            open_sells = {}  # pair -> order_info
            for txid, order in kraken_orders.items():
                descr = order.get('descr', {})
                pair_raw = descr.get('pair', '')
                side = descr.get('type', '')
                price = float(descr.get('price', 0))
                
                # Normalize pair name
                pair = pair_raw.upper()
                if pair not in PAIRS:
                    # Try common mappings
                    for p in PAIRS:
                        if pair_raw.upper().replace('/', '').endswith(p[-3:]) and pair_raw.upper().replace('/', '').startswith(p[:3]):
                            pair = p
                            break
                
                if pair in PAIRS:
                    info = {
                        'txid': txid,
                        'pair': pair,
                        'side': side,
                        'price': price,
                        'qty': float(order.get('vol', 0)),
                        'descr': descr.get('order', '')
                    }
                    if side == 'buy':
                        open_buys[pair] = info
                    elif side == 'sell':
                        open_sells[pair] = info
            
            # 3. Reconcile: held assets not tracked → add as positions
            for pair, holding in held_assets.items():
                if pair not in self.active_positions:
                    # Check if there's an open sell (exit already placed)
                    has_exit = pair in open_sells
                    
                    logger.info(f"[RECONCILE] Found untracked holding: {pair} "
                               f"qty={holding['qty']:.4f} @ ${holding['price']:.4f} "
                               f"(${holding['value']:.2f})"
                               f"{' — exit order exists' if has_exit else ''}")
                    
                    # Create position record (use current price as entry since we don't know real entry)
                    self.active_positions[pair] = {
                        'pair': pair,
                        'tool': 'reconciled',
                        'direction': 'long',
                        'leverage': 1,
                        'entry_price': holding['price'],  # Best guess — current price
                        'entry_bar': self.current_bar,
                        'entry_time': datetime.now(timezone.utc).timestamp(),
                        'position_size': holding['value'],
                        'qty': holding['qty'],
                        'sl_pct': 0.08,  # Default 8% SL
                        'hold': 48,       # Default 48h hold
                        'score': 0,
                        'total_margin_cost': 0,
                        '_reconciled': True
                    }
                    
                    # Capital reservation handled by recalculation at end of reconcile
                    
                    # If there's an open sell order, track as pending exit
                    if has_exit:
                        sell = open_sells[pair]
                        self.pending_exit_orders[pair] = {
                            "order_id": sell['txid'],
                            "placed_bar": self.current_bar,
                            "exit_price": sell['price'],
                            "reason": "reconciled_exit",
                            "pnl_pct": 0,  # Unknown
                            "pnl_dollar": 0,
                            "hours_held": 0,
                            "leverage": 1,
                            "total_margin_cost_pct": 0,
                            "side": "sell",
                            "qty": sell['qty']
                        }
                        self.active_positions[pair]['_pending_exit'] = True
                        logger.info(f"[RECONCILE] Tracking sell order {sell['txid']} for {pair} @ ${sell['price']:.4f}")
            
            # 4. Reconcile: open buy orders not tracked → add as pending entries
            for pair, buy in open_buys.items():
                if pair not in self.pending_limit_orders and pair not in self.active_positions:
                    logger.info(f"[RECONCILE] Found untracked buy order: {pair} "
                               f"qty={buy['qty']:.4f} @ ${buy['price']:.4f} ({buy['descr']})")
                    
                    position_size = buy['qty'] * buy['price']
                    
                    # Create pending limit order
                    self.pending_limit_orders[pair] = {
                        "direction": "long",
                        "qty": buy['qty'],
                        "price": buy['price'],
                        "original_price": buy['price'],
                        "original_score": 0,
                        "placed_bar": self.current_bar,
                        "order_id": buy['txid'],
                        "tool": "reconciled",
                        "retries": 0
                    }
                    
                    # Create matching active position (capital reserved)
                    self.active_positions[pair] = {
                        'pair': pair,
                        'tool': 'reconciled',
                        'direction': 'long',
                        'leverage': 1,
                        'entry_price': buy['price'],
                        'entry_bar': self.current_bar,
                        'entry_time': datetime.now(timezone.utc).timestamp(),
                        'position_size': position_size,
                        'qty': buy['qty'],
                        'sl_pct': 0.08,
                        'hold': 48,
                        'score': 0,
                        'total_margin_cost': 0,
                        '_reconciled': True
                    }
                    # Capital reservation handled by recalculation at end of reconcile
                    logger.info(f"[RECONCILE] Tracking buy order: {pair} ${position_size:.2f}")
            
            # 5. Clean up: positions we think we have but don't hold and no pending entry
            for pair in list(self.active_positions.keys()):
                if pair not in held_assets and pair not in self.pending_limit_orders:
                    pos = self.active_positions[pair]
                    if not pos.get('_reconciled'):  # Don't clean up stuff we just added
                        logger.warning(f"[RECONCILE] Phantom position {pair} — not held on Kraken, removing")
                        self.active_balance += pos['position_size']
                        del self.active_positions[pair]
                        if pair in self.pending_exit_orders:
                            del self.pending_exit_orders[pair]
            
            # 6. Recalculate active_balance based on reality
            # active_balance = total available for trading minus what's deployed
            deployed_capital = sum(
                pos['position_size'] for pos in self.active_positions.values()
            )
            # active_balance should be: allocation minus deployed
            grid_pct, active_pct = self.get_capital_allocation()
            total_active_allocation = self.total_balance * active_pct
            self.active_balance = max(0, total_active_allocation - deployed_capital)
            
            logger.info(f"[RECONCILE] Done — {len(self.active_positions)} positions, "
                       f"{len(self.pending_limit_orders)} pending buys, "
                       f"{len(self.pending_exit_orders)} pending exits, "
                       f"deployed: ${deployed_capital:.2f}, "
                       f"active balance: ${self.active_balance:.2f}")
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _compute_mtf_multiplier(self, tool, direction, htf_context):
        """Recompute the MTF multiplier for journal accuracy (mirrors apply_mtf_confirmation)."""
        if not ENABLE_MTF or not htf_context or not htf_context.get("htf_available", False):
            return 1.0
        crash_signals = {
            'volatile_oversold', 'crash_buy', 'mega_crash', 'crash_neg_ac',
            'blood_in_streets', 'quick_crash', 'crash_mean_revert', 'mega_pump_sell_t1'
        }
        if tool in crash_signals:
            return 1.0
        trend_4h = htf_context.get("trend_4h", "neutral")
        rsi_4h = htf_context.get("rsi_4h", 50.0)
        if direction == "long":
            if trend_4h == "bullish":
                return MTF_BOOST
            elif trend_4h == "bearish" and rsi_4h > 60:
                return MTF_PENALTY
            elif trend_4h == "bearish":
                return 0.8
        elif direction == "short":
            if trend_4h == "bearish":
                return MTF_BOOST
            elif trend_4h == "bullish" and rsi_4h > 50:
                return MTF_PENALTY
            elif trend_4h == "bullish":
                return 0.7
        return 1.0
    
    def _init_trade_journal(self):
        """Initialize trade journal CSV with headers if it doesn't exist."""
        if not self.trade_journal_path.exists():
            with open(self.trade_journal_path, 'w') as f:
                f.write("timestamp,event,pair,tool,direction,price,score,base_score,"
                        "mtf_penalty_applied,trend_4h,rsi_4h,bullish_4h_pct,fng,fng_regime,"
                        "leverage,position_size,sl_pct,hold_bars,reason,"
                        "pnl_pct,pnl_dollar,bars_held,close_reason,"
                        "tool_streak,balance,active_balance\n")
    
    def _journal_open(self, pair, tool, direction, price, score, base_score,
                      mtf_multiplier, htf_context, leverage, position_size, sl_pct, hold_bars, reason):
        """Log a trade open to the journal."""
        try:
            trend_4h = htf_context.get("trend_4h", "unknown") if htf_context else "unavailable"
            rsi_4h = htf_context.get("rsi_4h", 0) if htf_context else 0
            bullish_pct = getattr(self, '_bullish_4h_pct', -1)
            fng = getattr(self, 'current_fng', -1)
            fng_regime = ("Extreme Fear" if fng < 20 else "Fear" if fng < 30 else
                         "Neutral" if fng <= 70 else "Greed" if fng <= 80 else "Extreme Greed")
            streak_info = ""
            if tool in self.tool_streaks:
                s = self.tool_streaks[tool]
                streak_info = f"{s['type']}{s['streak']}" if s['type'] else "0"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            with open(self.trade_journal_path, 'a') as f:
                f.write(f"{ts},OPEN,{pair},{tool},{direction},{price:.6f},{score:.2f},{base_score:.2f},"
                        f"{mtf_multiplier:.2f},{trend_4h},{rsi_4h:.1f},{bullish_pct:.0f},{fng},{fng_regime},"
                        f"{leverage},{position_size:.2f},{sl_pct:.3f},{hold_bars},{reason.replace(',', ';')},"
                        f",,,,{streak_info},{self.total_balance:.2f},{self.active_balance:.2f}\n")
        except Exception as e:
            logger.debug(f"Journal write error (open): {e}")
    
    def _journal_close(self, pair, tool, direction, exit_price, pnl_pct, pnl_dollar,
                       bars_held, close_reason, entry_price):
        """Log a trade close to the journal."""
        try:
            bullish_pct = getattr(self, '_bullish_4h_pct', -1)
            fng = getattr(self, 'current_fng', -1)
            fng_regime = ("Extreme Fear" if fng < 20 else "Fear" if fng < 30 else
                         "Neutral" if fng <= 70 else "Greed" if fng <= 80 else "Extreme Greed")
            streak_info = ""
            if tool in self.tool_streaks:
                s = self.tool_streaks[tool]
                streak_info = f"{s['type']}{s['streak']}" if s['type'] else "0"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            with open(self.trade_journal_path, 'a') as f:
                f.write(f"{ts},CLOSE,{pair},{tool},{direction},{exit_price:.6f},,,,"
                        f",,{bullish_pct:.0f},{fng},{fng_regime},"
                        f",,,,,"
                        f"{pnl_pct:.4f},{pnl_dollar:.4f},{bars_held},{close_reason.replace(',', ';')},"
                        f"{streak_info},{self.total_balance:.2f},{self.active_balance:.2f}\n")
        except Exception as e:
            logger.debug(f"Journal write error (close): {e}")
    
    def _log_balance_snapshot(self):
        """Append balance snapshot to CSV."""
        try:
            path = LOGS_DIR / "balance_history.csv"
            write_header = not path.exists()
            with open(path, 'a') as f:
                if write_header:
                    f.write("timestamp,cycle,total_balance,active_balance,grid_balance,"
                            "margin_in_use,fng,active_positions,grid_positions,"
                            "active_profit,grid_profit\n")
                margin_in_use = sum(
                    pos.get('position_size', 0) for pos in self.active_positions.values()
                )
                grid_pos_count = sum(len(p) for p in self.grid_positions.values())
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts},{self.current_bar},{self.total_balance:.2f},"
                        f"{self.active_balance:.2f},{self.grid_balance:.2f},"
                        f"{margin_in_use:.2f},{getattr(self, 'current_fng', -1)},"
                        f"{len(self.active_positions)},{grid_pos_count},"
                        f"{self.active_profit:.2f},{self.grid_profit:.2f}\n")
        except Exception as e:
            logger.debug(f"Balance snapshot error: {e}")
    
    def _log_rejection(self, pair, tool, direction, score, reason):
        """Log a rejected signal to CSV."""
        try:
            path = LOGS_DIR / "rejected_signals.csv"
            write_header = not path.exists()
            with open(path, 'a') as f:
                if write_header:
                    f.write("timestamp,pair,tool,direction,score,reason\n")
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts},{pair},{tool},{direction},{score:.2f},{reason}\n")
        except Exception as e:
            logger.debug(f"Rejection log error: {e}")
    
    def _check_daily_rollover(self):
        """Check if date changed; if so, write daily summary and reset counters."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._current_date:
            return
        
        # Write summary for previous day
        try:
            stats = self._daily_stats
            path = LOGS_DIR / "daily_summary.csv"
            write_header = not path.exists()
            with open(path, 'a') as f:
                if write_header:
                    f.write("date,start_balance,end_balance,pnl,trades_opened,trades_closed,"
                            "wins,losses,win_rate,best_tool,worst_tool\n")
                closed = stats["trades_closed"]
                wins = stats["wins"]
                wr = (wins / closed * 100) if closed > 0 else 0
                # Best/worst tool by PnL
                tool_pnl = stats["tool_pnl"]
                best = max(tool_pnl, key=tool_pnl.get) if tool_pnl else ""
                worst = min(tool_pnl, key=tool_pnl.get) if tool_pnl else ""
                f.write(f"{self._current_date},{stats['start_balance']:.2f},"
                        f"{self.total_balance:.2f},{stats['pnl']:.2f},"
                        f"{stats['trades_opened']},{closed},{wins},{closed - wins},"
                        f"{wr:.1f},{best},{worst}\n")
        except Exception as e:
            logger.debug(f"Daily summary error: {e}")
        
        # Reset for new day
        self._current_date = today
        self._daily_stats = {
            "trades_opened": 0, "trades_closed": 0,
            "wins": 0, "losses": 0, "pnl": 0.0,
            "start_balance": self.total_balance,
            "tool_pnl": {}
        }
        logger.info(f"📅 New day: {today} — daily stats reset")
    
    # UPGRADE 4: Dynamic capital allocation based on Fear & Greed
    def get_capital_allocation(self) -> Tuple[float, float]:
        """100% active trading — grid disabled at low balance."""
        return 0.0, 1.0
    
    def rebalance_capital(self):
        """Rebalance capital allocation. Grid disabled — 100% active."""
        grid_pct, active_pct = self.get_capital_allocation()
        self.grid_balance = self.total_balance * grid_pct
        self.active_balance = self.total_balance * active_pct
    
    def _initialize_tool_stats(self):
        """Initialize tool performance stats with validated results."""
        # Initialize stats for all validated tools based on OOS testing
        validated_stats = {
            # CRASH/BEAR TOOLS - from VALIDATED_TOOLS.md
            "volatile_oversold": {"trades": 100, "wins": 74, "pnl": 207.0, "score_adj": 1.0},
            "crash_buy": {"trades": 120, "wins": 78, "pnl": 190.0, "score_adj": 1.0},
            "mega_crash": {"trades": 40, "wins": 21, "pnl": 135.0, "score_adj": 1.0},
            "crash_neg_ac": {"trades": 80, "wins": 50, "pnl": 125.0, "score_adj": 1.0},
            "blood_in_streets": {"trades": 60, "wins": 37, "pnl": 110.0, "score_adj": 1.0},
            "quick_crash": {"trades": 90, "wins": 53, "pnl": 98.0, "score_adj": 1.0},
            "crash_mean_revert": {"trades": 80, "wins": 49, "pnl": 98.0, "score_adj": 1.0},
            "vpin_dip": {"trades": 70, "wins": 41, "pnl": 73.0, "score_adj": 1.0},
            "market_panic_70": {"trades": 50, "wins": 30, "pnl": 75.0, "score_adj": 1.0},
            "flash_crash": {"trades": 60, "wins": 33, "pnl": 51.0, "score_adj": 1.0},
            "deep_dip_8h": {"trades": 70, "wins": 38, "pnl": 22.0, "score_adj": 1.0},
            "entropy_dip": {"trades": 60, "wins": 32, "pnl": 45.0, "score_adj": 1.0},
            "vpin_toxic": {"trades": 65, "wins": 35, "pnl": 45.0, "score_adj": 1.0},
            "btc_alt_spread": {"trades": 55, "wins": 30, "pnl": 45.0, "score_adj": 1.0},
            "quick_dip": {"trades": 90, "wins": 50, "pnl": 13.0, "score_adj": 1.0},
            
            # BULL/GREED TOOLS
            "mega_pump_sell_t1": {"trades": 70, "wins": 41, "pnl": 61.0, "score_adj": 1.0},
            "rsi_pump_8h": {"trades": 80, "wins": 48, "pnl": 56.0, "score_adj": 1.0},
            "falling_wedge_short": {"trades": 60, "wins": 34, "pnl": 60.0, "score_adj": 1.0},
            "greed_short_t2": {"trades": 75, "wins": 44, "pnl": 31.0, "score_adj": 1.0},
            "thursday_short": {"trades": 85, "wins": 49, "pnl": 28.0, "score_adj": 1.0},
            "mega_pump_sell_t2": {"trades": 60, "wins": 33, "pnl": 19.0, "score_adj": 1.0},
            "distribution_short": {"trades": 70, "wins": 37, "pnl": 18.0, "score_adj": 1.0},
            "late_us_short": {"trades": 75, "wins": 40, "pnl": 16.0, "score_adj": 1.0},
            "rsi_pump_12h": {"trades": 80, "wins": 44, "pnl": 13.0, "score_adj": 1.0},
            "ema_cross_short": {"trades": 90, "wins": 48, "pnl": 13.0, "score_adj": 1.0},
            "rsi_pump_fat_tail": {"trades": 45, "wins": 27, "pnl": 7.0, "score_adj": 1.0},
            "entropy_short": {"trades": 65, "wins": 36, "pnl": 2.0, "score_adj": 1.0},
            "alt_btc_revert_t3": {"trades": 70, "wins": 39, "pnl": 1.0, "score_adj": 1.0},
            
            # NEUTRAL/TRANSITION TOOLS
            "month_start_long": {"trades": 50, "wins": 27, "pnl": 72.0, "score_adj": 1.0},
            "dip_buy_5pct": {"trades": 80, "wins": 42, "pnl": 11.0, "score_adj": 1.0},
        }
        
        # Initialize all validated tools
        for tool in VALIDATED_TOOLS:
            if tool not in self.tool_stats:
                self.tool_stats[tool] = validated_stats.get(tool, 
                    {"trades": 0, "wins": 0, "pnl": 0.0, "score_adj": 1.0})
            
            # UPGRADE 7: Initialize streak tracking
            if tool not in self.tool_streaks:
                self.tool_streaks[tool] = {"streak": 0, "type": None}  # type: 'W' or 'L'
    
    # UPGRADE 7: Update consecutive win/loss tracking
    def update_tool_streak(self, tool: str, won: bool):
        """Update consecutive win/loss streak for a tool."""
        if tool not in self.tool_streaks:
            self.tool_streaks[tool] = {"streak": 0, "type": None}
        
        streak_info = self.tool_streaks[tool]
        current_type = 'W' if won else 'L'
        
        if streak_info["type"] == current_type:
            # Continue streak
            streak_info["streak"] += 1
        else:
            # Reset streak
            streak_info["streak"] = 1
            streak_info["type"] = current_type
        
        # Apply score adjustments
        stats = self.tool_stats[tool]
        if current_type == 'L' and streak_info["streak"] >= 5:
            # 5 consecutive losses: reduce score by 50%
            stats["score_adj"] = 0.5
            logger.warning(f"{tool} hit 5 consecutive losses - score reduced by 50%")
        elif current_type == 'W' and streak_info["streak"] >= 3:
            # 3 consecutive wins: boost score by 25%
            stats["score_adj"] = 1.25
            logger.info(f"{tool} hit 3 consecutive wins 🔥 - score boosted by 25%")
        else:
            # Reset to normal
            stats["score_adj"] = 1.0
    
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
            "total_balance": self.total_balance,
            "grid_positions": self.grid_positions,
            "grid_profit": self.grid_profit,
            "grid_round_trips": self.grid_round_trips,
            "grid_anchors": self.grid_anchors,
            "active_positions": self.active_positions,
            "active_profit": self.active_profit,
            "tool_stats": self.tool_stats,
            "tool_streaks": self.tool_streaks,
            "pending_limit_orders": self.pending_limit_orders,
            "pending_exit_orders": self.pending_exit_orders,
            "trade_history": self.trade_history[-500:],  # Keep last 500
            "current_bar": self.current_bar,
            "last_update": datetime.now().isoformat()
        }
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get_market_data(self) -> dict:
        """Fetch 1h and 4h market data for all pairs with multi-timeframe support."""
        market_data = {}
        
        for pair in self.get_active_pairs():
            try:
                # Get 1h klines (last 200 bars for indicators)
                klines = self.client.get_klines(pair, interval="1h", limit=200)
                if not klines:
                    continue
                    
                df = pd.DataFrame(klines)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                if len(df) < 50:  # Need enough data for indicators
                    continue
                
                # MTF: Fetch 4h candles for higher timeframe confirmation
                df_4h = None
                if ENABLE_MTF:
                    try:
                        klines_4h = self.client.get_klines(pair, interval="4h", limit=100)  # 100 bars = ~16 days
                        if klines_4h:
                            df_4h = pd.DataFrame(klines_4h)
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                if col in df_4h.columns:
                                    df_4h[col] = pd.to_numeric(df_4h[col], errors='coerce')
                            if len(df_4h) < 15:  # Need minimum data for HTF indicators
                                df_4h = None
                    except Exception as e:
                        logger.debug(f"Failed to fetch 4h data for {pair}: {e}")
                        # Graceful degradation - continue without MTF for this pair
                    
                # Get current ticker
                ticker = self.client.get_ticker(pair)
                current_price = float(ticker["price"]) if ticker else float(df['close'].iloc[-1])
                
                # Get bid/ask for limit orders (UPGRADE 1)
                orderbook = self.client.get_orderbook(pair, depth=1) if hasattr(self.client, 'get_orderbook') else None
                bid_price = float(orderbook['bids'][0][0]) if orderbook and orderbook['bids'] else current_price * 0.9995
                ask_price = float(orderbook['asks'][0][0]) if orderbook and orderbook['asks'] else current_price * 1.0005
                
                # Calculate ATR for dynamic trailing stops
                _high = df['high'].values.astype(float)
                _low = df['low'].values.astype(float)
                _close = df['close'].values.astype(float)
                if len(_high) >= 15:
                    _trs = []
                    for _i in range(-14, 0):
                        _tr = max(_high[_i] - _low[_i], abs(_high[_i] - _close[_i-1]), abs(_low[_i] - _close[_i-1]))
                        _trs.append(_tr)
                    _atr = np.mean(_trs)
                else:
                    _atr = 0
                
                market_data[pair] = {
                    "price": current_price,
                    "bid": bid_price,
                    "ask": ask_price,
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "atr": float(_atr),
                    "df": df,
                    "df_4h": df_4h  # MTF: 4h data for higher timeframe analysis
                }
                # Cache close prices for cross-pair signals
                self._price_cache[pair] = df['close'].values.astype(float)
                
            except Exception as e:
                logger.error(f"Failed to get market data for {pair}: {e}")
                continue
                
        return market_data
    
    def _refresh_volatile_pairs(self):
        """Scan all Kraken USD pairs and pick the most volatile ones.
        
        Runs every VOLATILITY_REFRESH_INTERVAL seconds. Uses 24h price range
        as volatility proxy. Keeps a minimum set of blue-chips always included.
        """
        now = datetime.now(timezone.utc).timestamp()
        if hasattr(self, '_vol_pairs_cache_ts') and now - self._vol_pairs_cache_ts < VOLATILITY_REFRESH_INTERVAL:
            return  # Use cached list
        
        try:
            # Always include blue chips
            ALWAYS_INCLUDE = {'XBTUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 'ADAUSD', 'AVAXUSD', 
                            'LINKUSD', 'DOTUSD', 'BNBUSD', 'LTCUSD'}
            
            # Get all asset pairs
            all_pairs_info = self.client._request('public/AssetPairs', {})
            if not all_pairs_info:
                return
            
            # Filter to USD pairs only
            usd_pairs = []
            for pair_name, info in all_pairs_info.items():
                # Skip non-USD, stablecoins, and leveraged tokens
                if not pair_name.endswith('USD'):
                    continue
                base = info.get('base', '')
                if base in ('ZUSD', 'USDT', 'USDC', 'DAI', 'USDG', 'PYUSD', 'TUSD'):
                    continue
                wsname = info.get('wsname', pair_name)
                altname = info.get('altname', pair_name)
                usd_pairs.append(altname)
            
            logger.info(f"[VOLATILITY] Scanning {len(usd_pairs)} USD pairs for volatility...")
            
            # Fetch 24h tickers in batches
            pair_volatility = {}
            batch_size = 50
            for i in range(0, len(usd_pairs), batch_size):
                batch = usd_pairs[i:i+batch_size]
                pair_str = ','.join(batch)
                try:
                    tickers = self.client._request('public/Ticker', {'pair': pair_str})
                    if not tickers:
                        continue
                    for pair_key, data in tickers.items():
                        # Normalize pair name back to our format
                        # Try to find matching altname
                        normalized = pair_key
                        for p in usd_pairs:
                            if p in pair_key or pair_key in p:
                                normalized = p
                                break
                        
                        try:
                            high_24h = float(data['h'][1])  # 24h high
                            low_24h = float(data['l'][1])   # 24h low
                            last = float(data['c'][0])       # Last price
                            vol_usd = float(data['v'][1]) * last  # 24h volume in USD
                            
                            if (low_24h > 0 and 
                                vol_usd >= MIN_PAIR_VOLUME_USD and
                                last >= MIN_PAIR_PRICE_USD and
                                normalized not in GEO_BLOCKED_PAIRS):
                                volatility = (high_24h - low_24h) / low_24h
                                pair_volatility[normalized] = {
                                    'volatility': volatility,
                                    'volume_usd': vol_usd,
                                    'price': last,
                                    'max_position_usd': vol_usd * MAX_POSITION_PCT_OF_VOLUME
                                }
                        except (KeyError, ValueError, ZeroDivisionError):
                            pass
                except Exception as e:
                    logger.debug(f"Ticker batch error: {e}")
                    continue
            
            if not pair_volatility:
                return
            
            # Sort by volatility, pick top N
            sorted_pairs = sorted(pair_volatility.items(), key=lambda x: x[1]['volatility'], reverse=True)
            
            # Build final list: always-include + top volatile
            selected = set(ALWAYS_INCLUDE)
            for pair, info in sorted_pairs:
                if len(selected) >= MAX_TRADING_PAIRS:
                    break
                selected.add(pair)
            
            self._dynamic_pairs = list(selected)
            self._vol_pairs_cache_ts = now
            self._pair_volatility = pair_volatility
            
            # Log what changed
            top5 = sorted_pairs[:5]
            top5_str = ", ".join([f"{p} ({v['volatility']:.1%})" for p, v in top5])
            logger.info(f"[VOLATILITY] Selected {len(self._dynamic_pairs)} pairs "
                       f"(from {len(pair_volatility)} liquid). Top 5: {top5_str}")
            
        except Exception as e:
            logger.error(f"Volatility scan failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_active_pairs(self) -> list:
        """Return current trading pairs — dynamic if available, fallback to static."""
        if hasattr(self, '_dynamic_pairs') and self._dynamic_pairs:
            return self._dynamic_pairs
        return PAIRS
    
    # ===== INDICATOR CALCULATIONS - EXACT COPY =====
    
    def calc_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI - exact same as master bot."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
            
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(delta))
        avg_loss = np.zeros(len(delta))
        
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        for i in range(period, len(delta)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50.0], rsi])
    
    def calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
            
        sma = np.full(len(prices), np.nan)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            
        return sma
    
    def calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        if len(high) < 2:
            return np.full(len(high), 0.0)
            
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First bar uses high-low
        
        atr = np.full(len(tr), 0.0)
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        return atr
    
    def calc_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), prices[0] if len(prices) > 0 else 0)
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        k = 2.0 / (period + 1)
        for i in range(period, len(prices)):
            ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
        return ema
    
    def calc_autocorrelation(self, prices: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag."""
        if len(prices) < lag + 5:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < lag + 2:
            return 0.0
        return np.corrcoef(returns[:-lag], returns[lag:])[0, 1] if not np.isnan(np.corrcoef(returns[:-lag], returns[lag:])[0, 1]) else 0.0
    
    def calc_hurst(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent."""
        if len(prices) < max_lag + 10:
            return 0.5
        log_prices = np.log(prices)
        n = len(log_prices)
        rs = []
        lags = range(2, min(max_lag, n//2))
        for lag in lags:
            ts = log_prices[:n//lag*lag].reshape(-1, lag)
            mean_ts = np.mean(ts, axis=1, keepdims=True)
            ts_centered = ts - mean_ts
            rs_values = []
            for i in range(ts.shape[0]):
                cumsum = np.cumsum(ts_centered[i])
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(ts[i])
                if s > 0:
                    rs_values.append(r / s)
            if rs_values:
                rs.append(np.mean(rs_values))
        if len(rs) < 2:
            return 0.5
        rs = np.array(rs)
        lags = np.array(list(lags))
        try:
            hurst = np.polyfit(np.log(lags), np.log(rs), 1)[0]
            return max(0.0, min(1.0, hurst))
        except:
            return 0.5
    
    def calc_entropy(self, prices: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy of returns."""
        if len(prices) < 10:
            return 3.0  # Default high entropy
        returns = np.diff(prices) / prices[:-1]
        try:
            hist, _ = np.histogram(returns, bins=bins)
            hist = hist[hist > 0]  # Remove zero bins
            p = hist / np.sum(hist)
            entropy = -np.sum(p * np.log2(p))
            return entropy
        except:
            return 3.0
    
    def calc_vpin(self, df: pd.DataFrame, window: int = 50) -> float:
        """Calculate Volume-Synchronized Probability of Informed Trading (VPIN)."""
        if len(df) < window:
            return 0.5
        try:
            # Approximate VPIN using volume and price changes
            volume = df['volume'].values.astype(float)
            close = df['close'].values.astype(float)
            
            if len(volume) < 2 or len(close) < 2:
                return 0.5
                
            # Calculate buy/sell volume approximation
            price_change = np.diff(close)
            buy_vol = np.where(price_change > 0, volume[1:], 0)
            sell_vol = np.where(price_change < 0, volume[1:], 0)
            
            if len(buy_vol) < window:
                return 0.5
                
            # Rolling VPIN calculation
            vpin_values = []
            for i in range(window, len(buy_vol)):
                window_buy = np.sum(buy_vol[i-window:i])
                window_sell = np.sum(sell_vol[i-window:i])
                total_vol = window_buy + window_sell
                if total_vol > 0:
                    vpin = abs(window_buy - window_sell) / total_vol
                    vpin_values.append(vpin)
            
            return np.mean(vpin_values[-10:]) if vpin_values else 0.5
        except:
            return 0.5
    
    # ===== MULTI-TIMEFRAME ANALYSIS =====
    
    def get_htf_context(self, data: dict) -> dict:
        """Compute higher timeframe (4h) context for multi-timeframe confirmation."""
        df_4h = data.get("df_4h")
        if df_4h is None or len(df_4h) < 15:
            # No 4h data available, return neutral context
            return {
                "trend_4h": "neutral",
                "rsi_4h": 50.0,
                "momentum_4h": 0.0,
                "atr_pct_4h": 0.0,
                "above_sma50_4h": False,
                "htf_available": False
            }
        
        try:
            # Extract 4h OHLC data
            close_4h = df_4h['close'].values.astype(float)
            high_4h = df_4h['high'].values.astype(float)
            low_4h = df_4h['low'].values.astype(float)
            
            current_price = data["price"]
            
            # 1. Trend: EMA5 vs EMA13 on 4h
            ema5_4h = self.calc_ema(close_4h, 5)
            ema13_4h = self.calc_ema(close_4h, 13)
            
            if not np.isnan(ema5_4h[-1]) and not np.isnan(ema13_4h[-1]):
                if ema5_4h[-1] > ema13_4h[-1]:
                    trend_4h = "bullish"
                elif ema5_4h[-1] < ema13_4h[-1]:
                    trend_4h = "bearish"
                else:
                    trend_4h = "neutral"
            else:
                trend_4h = "neutral"
            
            # 2. RSI(7) on 4h
            rsi7_4h = self.calc_rsi(close_4h, 7)
            rsi_4h = rsi7_4h[-1] if not np.isnan(rsi7_4h[-1]) else 50.0
            
            # 3. Momentum: 4h return over last 3 candles (12 hours)
            if len(close_4h) >= 4:
                momentum_4h = (close_4h[-1] - close_4h[-4]) / close_4h[-4] * 100
            else:
                momentum_4h = 0.0
            
            # 4. ATR as % of price on 4h
            atr14_4h = self.calc_atr(high_4h, low_4h, close_4h, 14)
            if current_price > 0 and not np.isnan(atr14_4h[-1]):
                atr_pct_4h = atr14_4h[-1] / current_price * 100
            else:
                atr_pct_4h = 0.0
            
            # 5. Price vs SMA50 on 4h
            sma50_4h = self.calc_sma(close_4h, 50)
            above_sma50_4h = (not np.isnan(sma50_4h[-1])) and (current_price > sma50_4h[-1])
            
            return {
                "trend_4h": trend_4h,
                "rsi_4h": rsi_4h,
                "momentum_4h": momentum_4h,
                "atr_pct_4h": atr_pct_4h,
                "above_sma50_4h": above_sma50_4h,
                "htf_available": True
            }
            
        except Exception as e:
            logger.debug(f"Error computing HTF context: {e}")
            return {
                "trend_4h": "neutral",
                "rsi_4h": 50.0,
                "momentum_4h": 0.0,
                "atr_pct_4h": 0.0,
                "above_sma50_4h": False,
                "htf_available": False
            }
    
    # ===== SIGNAL SCANNING - SAME 30 VALIDATED TOOLS WITH SCORE ADJUSTMENT =====
    
    def scan_signals(self, pair: str, data: dict) -> List[Tuple[dict, float]]:
        """Scan validated tools for signals on this pair."""
        signals = []
        df = data['df']
        price = data['price']
        
        if len(df) < 50:
            return signals
            
        # Compute base indicators once
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        rsi7 = self.calc_rsi(close, 7)
        sma50 = self.calc_sma(close, 50)
        atr14 = self.calc_atr(high, low, close, 14)
        ema5 = self.calc_ema(close, 5)
        ema13 = self.calc_ema(close, 13)
        
        # Current values
        cur_rsi = rsi7[-1] if not np.isnan(rsi7[-1]) else 50
        cur_atr_pct = (atr14[-1] / price * 100) if price > 0 and not np.isnan(atr14[-1]) else 0
        cur_vs_sma50 = ((price - sma50[-1]) / sma50[-1] * 100) if not np.isnan(sma50[-1]) and sma50[-1] > 0 else 0
        
        # Returns
        ret_4h = (price - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
        ret_8h = (price - close[-9]) / close[-9] * 100 if len(close) >= 9 else 0
        ret_12h = (price - close[-13]) / close[-13] * 100 if len(close) >= 13 else 0
        ret_24h = (price - close[-25]) / close[-25] * 100 if len(close) >= 25 else 0
        
        # MTF: Get higher timeframe context for confirmation
        htf_context = self.get_htf_context(data) if ENABLE_MTF else {"htf_available": False}
        
        # Helper function to apply score adjustment from UPGRADE 7
        def adjust_score(tool: str, base_score: float) -> float:
            if tool in self.tool_stats:
                return base_score * self.tool_stats[tool].get("score_adj", 1.0)
            return base_score
        
        # MTF: Helper function to apply multi-timeframe confirmation
        def apply_mtf_confirmation(tool: str, direction: str, base_score: float) -> float:
            if not ENABLE_MTF or not htf_context.get("htf_available", False):
                return base_score  # No HTF data, use original score
            
            # Crash signals bypass MTF (they're counter-trend by nature)
            crash_signals = {
                'volatile_oversold', 'crash_buy', 'mega_crash', 'crash_neg_ac', 
                'blood_in_streets', 'quick_crash', 'crash_mean_revert', 'mega_pump_sell_t1'
            }
            if tool in crash_signals:
                return base_score
            
            trend_4h = htf_context["trend_4h"]
            rsi_4h = htf_context["rsi_4h"]
            multiplier = 1.0
            
            if direction == "long":
                if trend_4h == "bullish":
                    multiplier = MTF_BOOST
                elif trend_4h == "bearish" and rsi_4h > 60:
                    multiplier = MTF_PENALTY
                elif trend_4h == "bearish":
                    multiplier = 0.8
            
            elif direction == "short":
                if trend_4h == "bearish":
                    multiplier = MTF_BOOST
                elif trend_4h == "bullish" and rsi_4h > 50:
                    multiplier = MTF_PENALTY
                elif trend_4h == "bullish":
                    multiplier = 0.7
            
            return base_score * multiplier
        
        # ===== CRASH/BEAR SIGNALS (LONG) - 15 tools =====
        
        # 1. volatile_oversold: atr_pct>3 AND rsi7<25 → LONG | WR_8h=73.8%, Ret_8h=+2.07%
        if cur_atr_pct > 3 and cur_rsi < 25:
            base_score = cur_atr_pct * (25 - cur_rsi) * 0.5  # 30-50 range
            score = adjust_score('volatile_oversold', base_score)
            score = apply_mtf_confirmation('volatile_oversold', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'volatile_oversold', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.08,
                'reason': f"VOLATILE OVERSOLD: ATR={cur_atr_pct:.1f}%, RSI={cur_rsi:.1f}"
            }, score))
        
        # 2. crash_buy: ret_24h<-10 AND rsi7<20 → LONG | WR_8h=65.1%, Ret_24h=+1.90%
        if ret_24h < -10 and cur_rsi < 20:
            base_score = abs(ret_24h) * (20 - cur_rsi) * 0.3  # 25-40 range
            score = adjust_score('crash_buy', base_score)
            score = apply_mtf_confirmation('crash_buy', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'crash_buy', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"CRASH BUY: {ret_24h:.1f}% drop 24h, RSI={cur_rsi:.1f}"
            }, score))
        
        # 3. mega_crash: ret_24h<-15 → LONG | WR_24h=52.5%, Ret_24h=+1.35%
        if ret_24h < -15:
            base_score = abs(ret_24h) * 2  # 30-50 range
            score = adjust_score('mega_crash', base_score)
            score = apply_mtf_confirmation('mega_crash', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'mega_crash', 'direction': 'long',
                'hold': 24, 'sl_pct': 0.08,
                'reason': f"MEGA CRASH: {ret_24h:.1f}% drop 24h"
            }, score))
        
        # 4. crash_neg_ac: ret_24h<-10 AND autocorr<-0.05 → LONG | WR_8h=62.1%, Ret_8h=+1.25%
        if ret_24h < -10:
            autocorr = self.calc_autocorrelation(close[-30:]) if len(close) >= 30 else 0
            if autocorr < -0.05:
                base_score = abs(ret_24h) * abs(autocorr) * 50  # 30-50 range
                score = adjust_score('crash_neg_ac', base_score)
                score = apply_mtf_confirmation('crash_neg_ac', 'long', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'crash_neg_ac', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.05,
                    'reason': f"CRASH NEG AC: {ret_24h:.1f}% drop, autocorr={autocorr:.3f}"
                }, score))
        
        # 5. blood_in_streets: 70%+ coins down >2% AND rsi7<20 → LONG | WR_24h=61.7%, Ret_24h=+1.10%
        if len(self._price_cache) >= 5 and cur_rsi < 20:
            dropping = sum(1 for cached in self._price_cache.values()
                         if len(cached) >= 5 and (cached[-1]-cached[-5])/cached[-5]*100 < -2)
            panic_pct = dropping / len(self._price_cache) * 100
            if panic_pct >= 70:
                base_score = (20 - cur_rsi) * 2  # High priority
                score = adjust_score('blood_in_streets', base_score)
                score = apply_mtf_confirmation('blood_in_streets', 'long', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'blood_in_streets', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.06,
                    'reason': f"BLOOD IN STREETS: {panic_pct:.0f}% panic + RSI={cur_rsi:.1f}"
                }, score))
        
        # 6. quick_crash: ret_8h<-10 → LONG (8h hold only) | WR_8h=59.1%, Ret_8h=+0.98%
        if ret_8h < -10:
            base_score = abs(ret_8h) * 2  # 20-30 range
            score = adjust_score('quick_crash', base_score)
            score = apply_mtf_confirmation('quick_crash', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'quick_crash', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.07,
                'reason': f"QUICK CRASH: {ret_8h:.1f}% drop 8h"
            }, score))
        
        # 7. crash_mean_revert: ret_24h<-8 AND Hurst<0.45 → LONG | WR_8h=61.3%, Ret_8h=+0.98%
        if ret_24h < -8:
            hurst = self.calc_hurst(close[-50:]) if len(close) >= 50 else 0.5
            if hurst < 0.45:
                base_score = abs(ret_24h) * (0.45 - hurst) * 10  # 20-30 range
                score = adjust_score('crash_mean_revert', base_score)
                score = apply_mtf_confirmation('crash_mean_revert', 'long', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'crash_mean_revert', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.05,
                    'reason': f"CRASH MEAN REVERT: {ret_24h:.1f}% drop, Hurst={hurst:.3f}"
                }, score))
        
        # 8. vpin_dip: ret_8h<-5 AND VPIN>0.5 → LONG | WR_8h=58.8%, Ret_8h=+0.73%
        if ret_8h < -5:
            vpin = self.calc_vpin(df)
            if vpin > 0.5:
                base_score = abs(ret_8h) * vpin * 2  # 15-25 range
                score = adjust_score('vpin_dip', base_score)
                score = apply_mtf_confirmation('vpin_dip', 'long', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'vpin_dip', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.05,
                    'reason': f"VPIN DIP: {ret_8h:.1f}% drop 8h, VPIN={vpin:.2f}"
                }, score))
        
        # 9. market_panic_70: 70%+ coins down >3% in 4h → LONG | WR_24h=59.0%, Ret_24h=+0.75%
        if len(self._price_cache) >= 5:
            dropping = sum(1 for cached in self._price_cache.values()
                         if len(cached) >= 5 and (cached[-1]-cached[-5])/cached[-5]*100 < -3)
            panic_pct = dropping / len(self._price_cache) * 100
            if panic_pct >= 70:
                base_score = panic_pct * 0.3  # 21-30 range

                score = adjust_score('market_panic_70', base_score)

                score = apply_mtf_confirmation('market_panic_70', 'long', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'market_panic_70', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.04,
                    'reason': f"MARKET PANIC 70: {panic_pct:.0f}% coins down >3%"
                }, score))
        
        # 10. flash_crash: ret_12h<-10 → LONG | WR_8h=55.8%, Ret_8h=+0.51%
        if ret_12h < -10:
            base_score = abs(ret_12h) * 1.5  # 15-25 range
            score = adjust_score('flash_crash', base_score)
            score = apply_mtf_confirmation('flash_crash', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'flash_crash', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.07,
                'reason': f"FLASH CRASH: {ret_12h:.1f}% drop 12h"
            }, score))
        
        # 11. deep_dip_8h: -10<ret_8h<-8 → LONG | WR_8h=54.8%, Ret_8h=+0.22%
        if -10 < ret_8h < -8:
            base_score = abs(ret_8h) * 1.5  # 12-15 range
            score = adjust_score('deep_dip_8h', base_score)
            score = apply_mtf_confirmation('deep_dip_8h', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'deep_dip_8h', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.05,
                'reason': f"DEEP DIP 8h: {ret_8h:.1f}% drop"
            }, score))
        
        # 12. entropy_dip: entropy<2.5 AND ret_4h<-2 → LONG | WR_8h=52.8%, Ret_8h=+0.45%
        if ret_4h < -2:
            entropy = self.calc_entropy(close[-30:]) if len(close) >= 30 else 3.0
            if entropy < 2.5:
                score = adjust_score('entropy_dip', (2.5 - entropy) * abs(ret_4h) * 2)  # 10-20 range
                signals.append(({
                    'pair': pair, 'tool': 'entropy_dip', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.04,
                    'reason': f"ENTROPY DIP: entropy={entropy:.2f}, {ret_4h:.1f}% drop 4h"
                }, score))
        
        # 13. vpin_toxic: VPIN>0.7 AND red candle → LONG | WR_8h=53.8%, Ret_8h=+0.45%
        if len(df) >= 2:
            vpin = self.calc_vpin(df)
            is_red = close[-1] < df['open'].iloc[-1]
            if vpin > 0.7 and is_red:
                base_score = vpin * 20  # 14-20 range

                score = adjust_score('vpin_toxic', base_score)

                score = apply_mtf_confirmation('vpin_toxic', 'long', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'vpin_toxic', 'direction': 'long',
                    'hold': 8, 'sl_pct': 0.04,
                    'reason': f"VPIN TOXIC: VPIN={vpin:.2f}, red candle"
                }, score))
        
        # 14. btc_alt_spread: alt lagging BTC 3%+ AND rsi7<35 → LONG | WR_24h=55.2%, Ret_24h=+0.45%
        if pair != "XBTUSD" and "XBTUSD" in self._price_cache and cur_rsi < 35:
            btc_prices = self._price_cache["XBTUSD"]
            if len(btc_prices) >= 25 and len(close) >= 25:
                btc_ret24 = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                if btc_ret24 - ret_24h > 3:  # BTC outperforming by 3%+
                    score = adjust_score('btc_alt_spread', (btc_ret24 - ret_24h) * (35 - cur_rsi) * 0.1)  # 10-15 range
                    signals.append(({
                        'pair': pair, 'tool': 'btc_alt_spread', 'direction': 'long',
                        'hold': 24, 'sl_pct': 0.05,
                        'reason': f"BTC ALT SPREAD: BTC {btc_ret24:+.1f}% vs {pair} {ret_24h:+.1f}%, RSI={cur_rsi:.1f}"
                    }, score))
        
        # 15. quick_dip: ret_4h<-5 → LONG | WR_8h=55.5%, Ret_8h=+0.13%
        if ret_4h < -5:
            base_score = abs(ret_4h) * 1.2  # 6-12 range
            score = adjust_score('quick_dip', base_score)
            score = apply_mtf_confirmation('quick_dip', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'quick_dip', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.05,
                'reason': f"QUICK DIP: {ret_4h:.1f}% drop 4h"
            }, score))
        
        # ===== BULL/GREED SIGNALS (SHORT) - 13 tools =====
        
        # 16. mega_pump_sell_t1: rsi7>80 AND ret_12h>=10 → SHORT | WR_24h=58.7%, Ret_24h=+0.61%
        if cur_rsi > 80 and ret_12h >= 10:
            base_score = 25 + (cur_rsi - 80) * 0.5 + (ret_12h - 10) * 0.3  # 25-35 range
            score = adjust_score('mega_pump_sell_t1', base_score)
            score = apply_mtf_confirmation('mega_pump_sell_t1', 'short', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'mega_pump_sell_t1', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"MEGA PUMP SELL T1: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h"
            }, score))
        
        # 17. rsi_pump_8h: rsi7>80 AND ret_8h>=10 → SHORT | WR_24h=60.3%, Ret_24h=+0.56%
        if cur_rsi > 80 and ret_8h >= 10:
            base_score = 25 + (cur_rsi - 80) * 0.4 + (ret_8h - 10) * 0.4  # 25-35 range
            score = adjust_score('rsi_pump_8h', base_score)
            score = apply_mtf_confirmation('rsi_pump_8h', 'short', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'rsi_pump_8h', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"RSI PUMP 8h: RSI={cur_rsi:.1f}, +{ret_8h:.1f}% 8h"
            }, score))
        
        # 18. falling_wedge_short: lower highs+lows converging + price>SMA50 → SHORT | WR_24h=56.7%, Ret_24h=+0.60%
        if len(high) >= 20 and len(low) >= 20 and not np.isnan(sma50[-1]) and price > sma50[-1]:
            # Simple falling wedge detection: recent highs declining, lows declining but converging
            recent_highs = high[-10:]
            recent_lows = low[-10:]
            if len(recent_highs) >= 5 and len(recent_lows) >= 5:
                high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
                if high_trend < 0 and low_trend < 0 and high_trend > low_trend:  # Converging
                    score = adjust_score('falling_wedge_short', abs(high_trend - low_trend) * 1000)  # 20-30 range
                    score = apply_mtf_confirmation('falling_wedge_short', 'short', score)  # MTF confirmation
                    signals.append(({
                        'pair': pair, 'tool': 'falling_wedge_short', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.04,
                        'reason': f"FALLING WEDGE SHORT: converging pattern, price>${sma50[-1]:.2f} SMA50"
                    }, score))
        
        # 19. greed_short_t2: rsi7>75 AND ret_8h>5 AND price>SMA50 → SHORT | WR_24h=58.5%, Ret_24h=+0.31%
        # This uses Fear & Greed index for enhanced scoring
        if cur_rsi > 75 and ret_8h > 5 and not np.isnan(sma50[-1]) and price > sma50[-1]:
            base_score = 15 + (cur_rsi - 75) * 0.3 + (ret_8h - 5) * 0.2
            # Boost score if Fear & Greed is high
            if self.current_fng > 75:
                base_score *= 1.5
            base_score = base_score  # 15-25 range

            score = adjust_score('greed_short_t2', base_score)

            score = apply_mtf_confirmation('greed_short_t2', 'short', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'greed_short_t2', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.04,
                'reason': f"GREED SHORT T2: RSI={cur_rsi:.1f}, +{ret_8h:.1f}% 8h, F&G={self.current_fng}"
            }, score))
        
        # 20. thursday_short: Thursday AND price>SMA50 → SHORT | WR_24h=57.9%, Ret_24h=+0.28%
        try:
            dow = datetime.now(timezone.utc).weekday()
            if dow == 3 and not np.isnan(sma50[-1]) and price > sma50[-1]:  # Thursday
                base_score = 12  # Fixed score

                score = adjust_score('thursday_short', base_score)

                score = apply_mtf_confirmation('thursday_short', 'short', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'thursday_short', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.03,
                    'reason': f"THURSDAY SHORT: uptrend (price>${sma50[-1]:.2f} SMA50)"
                }, score))
        except:
            pass
        
        # 21. mega_pump_sell_t2: rsi7>80 AND ret_12h>=8 → SHORT | WR_24h=55.0%, Ret_24h=+0.19%
        if cur_rsi > 80 and ret_12h >= 8:
            score = adjust_score('mega_pump_sell_t2', 18 + (cur_rsi - 80) * 0.3 + (ret_12h - 8) * 0.2)  # 18-25 range
            score = apply_mtf_confirmation('mega_pump_sell_t2', 'short', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'mega_pump_sell_t2', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.05,
                'reason': f"MEGA PUMP SELL T2: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h"
            }, score))
        
        # 22. distribution_short: lower highs + vol decline + RSI fall + price>SMA50 → SHORT | WR_24h=53.4%, Ret_24h=+0.18%
        if len(high) >= 20 and len(volume) >= 20 and not np.isnan(sma50[-1]) and price > sma50[-1]:
            # Detect distribution pattern
            recent_highs = high[-10:]
            recent_vol = volume[-10:]
            recent_rsi = rsi7[-10:]
            
            if len(recent_highs) >= 5 and len(recent_vol) >= 5 and len(recent_rsi) >= 5:
                high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                vol_trend = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
                rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
                
                if high_trend < 0 and vol_trend < 0 and rsi_trend < 0:  # All declining
                    score = adjust_score('distribution_short', abs(high_trend) * 100)  # 15-20 range
                    score = apply_mtf_confirmation('distribution_short', 'short', score)  # MTF confirmation
                    signals.append(({
                        'pair': pair, 'tool': 'distribution_short', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.04,
                        'reason': f"DISTRIBUTION SHORT: lower highs, vol decline, RSI fall"
                    }, score))
        
        # 23. late_us_short: hour==21 UTC AND price>SMA50 → SHORT | WR_24h=52.9%, Ret_24h=+0.16%
        try:
            current_hour = datetime.now(timezone.utc).hour
            if current_hour == 21 and not np.isnan(sma50[-1]) and price > sma50[-1]:
                base_score = 10  # Fixed score

                score = adjust_score('late_us_short', base_score)

                score = apply_mtf_confirmation('late_us_short', 'short', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'late_us_short', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.03,
                    'reason': f"LATE US SHORT: 21:00 UTC, price>${sma50[-1]:.2f} SMA50"
                }, score))
        except:
            pass
        
        # 24. rsi_pump_12h: rsi7>80 AND ret_12h>=8 → SHORT | WR_24h=54.9%, Ret_24h=+0.13%
        if cur_rsi > 80 and ret_12h >= 8:
            score = adjust_score('rsi_pump_12h', 15 + (cur_rsi - 80) * 0.2 + (ret_12h - 8) * 0.1)  # 15-20 range
            score = apply_mtf_confirmation('rsi_pump_12h', 'short', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'rsi_pump_12h', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.04,
                'reason': f"RSI PUMP 12h: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h"
            }, score))
        
        # 25. ema_cross_short: ema5>ema13 AND price>SMA50 → SHORT | WR_24h=53.2%, Ret_24h=+0.13%
        if (not np.isnan(ema5[-1]) and not np.isnan(ema13[-1]) and not np.isnan(sma50[-1]) and
            ema5[-1] > ema13[-1] and price > sma50[-1]):
            score = adjust_score('ema_cross_short', 10 + (ema5[-1] - ema13[-1]) / price * 1000)  # 10-15 range
            score = apply_mtf_confirmation('ema_cross_short', 'short', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'ema_cross_short', 'direction': 'short',
                'hold': 24, 'sl_pct': 0.03,
                'reason': f"EMA CROSS SHORT: EMA5>${ema5[-1]:.2f} > EMA13${ema13[-1]:.2f}, price>SMA50"
            }, score))
        
        # 26. rsi_pump_fat_tail: rsi7>80 AND ret_12h>10 AND kurtosis>5 → SHORT | WR_24h=58.9%, Ret_24h=+0.07%
        if cur_rsi > 80 and ret_12h > 10:
            # Calculate kurtosis of recent returns
            if len(close) >= 30:
                recent_returns = np.diff(close[-30:]) / close[-31:-1]
                try:
                    from scipy.stats import kurtosis
                    kurt = kurtosis(recent_returns)
                except ImportError:
                    # Simple kurtosis approximation
                    mean_ret = np.mean(recent_returns)
                    std_ret = np.std(recent_returns)
                    if std_ret > 0:
                        fourth_moment = np.mean(((recent_returns - mean_ret) / std_ret) ** 4)
                        kurt = fourth_moment - 3  # Excess kurtosis
                    else:
                        kurt = 0
                
                if kurt > 5:
                    base_score = 15 + kurt * 0.5  # 15-20 range

                    score = adjust_score('rsi_pump_fat_tail', base_score)

                    score = apply_mtf_confirmation('rsi_pump_fat_tail', 'short', score)  # MTF confirmation
                    signals.append(({
                        'pair': pair, 'tool': 'rsi_pump_fat_tail', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.04,
                        'reason': f"RSI PUMP FAT TAIL: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h, kurtosis={kurt:.1f}"
                    }, score))
        
        # 27. entropy_short: entropy<2.5 AND price>SMA50 → SHORT | WR_24h=54.8%, Ret_24h=+0.02%
        if not np.isnan(sma50[-1]) and price > sma50[-1]:
            entropy = self.calc_entropy(close[-30:]) if len(close) >= 30 else 3.0
            if entropy < 2.5:
                score = adjust_score('entropy_short', (2.5 - entropy) * 4)  # 8-12 range
                score = apply_mtf_confirmation('entropy_short', 'short', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'entropy_short', 'direction': 'short',
                    'hold': 24, 'sl_pct': 0.03,
                    'reason': f"ENTROPY SHORT: entropy={entropy:.2f}, price>SMA50"
                }, score))
        
        # 28. alt_btc_revert_t3: alt outperforms BTC 3-5% 24h → SHORT | WR_24h=56.4%, Ret_24h=+0.01%
        if pair != "XBTUSD" and "XBTUSD" in self._price_cache:
            btc_prices = self._price_cache["XBTUSD"]
            if len(btc_prices) >= 25 and len(close) >= 25:
                btc_ret24 = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                outperformance = ret_24h - btc_ret24
                if 3 <= outperformance <= 5:  # Alt outperforming BTC by 3-5%
                    base_score = outperformance * 2  # 6-10 range

                    score = adjust_score('alt_btc_revert_t3', base_score)

                    score = apply_mtf_confirmation('alt_btc_revert_t3', 'short', score)  # MTF confirmation
                    signals.append(({
                        'pair': pair, 'tool': 'alt_btc_revert_t3', 'direction': 'short',
                        'hold': 24, 'sl_pct': 0.03,
                        'reason': f"ALT BTC REVERT T3: {pair} {ret_24h:+.1f}% vs BTC {btc_ret24:+.1f}% (+{outperformance:.1f}%)"
                    }, score))
        
        # ===== NEUTRAL/TRANSITION SIGNALS - 2 tools =====
        
        # 29. month_start_long: day 1-3 → LONG | WR_24h=53.9%, Ret_24h=+0.72%
        try:
            day_of_month = datetime.now(timezone.utc).day
            if 1 <= day_of_month <= 3:
                base_score = 15  # Fixed score

                score = adjust_score('month_start_long', base_score)

                score = apply_mtf_confirmation('month_start_long', 'long', score)  # MTF confirmation
                signals.append(({
                    'pair': pair, 'tool': 'month_start_long', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.04,
                    'reason': f"MONTH START LONG: day {day_of_month} of month"
                }, score))
        except:
            pass
        
        # 30. dip_buy_5pct: ret_4h<-5 → LONG | WR_8h=52.7%, Ret_8h=+0.11%
        if ret_4h < -5:
            base_score = abs(ret_4h) * 1.0  # 5-10 range
            score = adjust_score('dip_buy_5pct', base_score)
            score = apply_mtf_confirmation('dip_buy_5pct', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'dip_buy_5pct', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.04,
                'reason': f"DIP BUY 5PCT: {ret_4h:.1f}% drop 4h"
            }, score))
        
        # ===== VALIDATED BULL MARKET TOOLS (regime-aware, OOS validated on 3yr data) =====
        
        # 31. accumulation_breakout: 2-4wk consolidation → volume breakout
        # OOS: 64 signals, 50% WR, +3.19% avg, PF=2.24 (bull regime only)
        if len(close) >= 500 and len(high) >= 336 and len(volume) >= 336:
            ab_range_high = np.max(high[-336:-48])
            ab_range_low = np.min(low[-336:-48])
            ab_range_pct = (ab_range_high - ab_range_low) / ab_range_low * 100 if ab_range_low > 0 else 100
            
            if 3 <= ab_range_pct <= 20 and price > ab_range_high * 1.01:
                ab_vol_short = np.mean(volume[-24:])
                ab_vol_long = np.mean(volume[-336:-48])
                ab_vol_ratio = ab_vol_short / ab_vol_long if ab_vol_long > 0 else 1
                
                if ab_vol_ratio >= 1.5 and cur_rsi < 72:
                    ab_breakout_pct = (price - ab_range_high) / ab_range_high * 100
                    base_score = ab_vol_ratio * 15 + ab_breakout_pct * 10 + (20 - ab_range_pct) * 2
                    score = adjust_score('accumulation_breakout', base_score)
                    score = apply_mtf_confirmation('accumulation_breakout', 'long', score)
                    signals.append(({
                        'pair': pair, 'tool': 'accumulation_breakout', 'direction': 'long',
                        'hold': 336, 'sl_pct': 0.12,
                        'reason': f"ACCUM BREAKOUT: {ab_range_pct:.1f}% range, vol {ab_vol_ratio:.1f}x, break +{ab_breakout_pct:.1f}%"
                    }, score))
        
        # 32. hurst_trend_long: Chaos theory — Hurst > 0.6 = trending regime + momentum
        # OOS: 140 signals, 45% WR, +2.37% avg, PF=1.76 (bull regime only)
        if len(close) >= 500:
            hurst_returns = np.diff(np.log(close[-168:]))
            # Simplified R/S Hurst exponent
            hurst_lags = range(10, min(100, len(hurst_returns) // 2), 5)
            hurst_rs, hurst_lv = [], []
            for hlag in hurst_lags:
                hsub = hurst_returns[-hlag:]
                hmean = np.mean(hsub)
                hdev = np.cumsum(hsub - hmean)
                hR = np.max(hdev) - np.min(hdev)
                hS = np.std(hsub)
                if hS > 0 and hR > 0:
                    hurst_rs.append(np.log(hR / hS))
                    hurst_lv.append(np.log(hlag))
            
            H = 0.5
            if len(hurst_rs) >= 3:
                from scipy import stats as sp_stats
                H = np.clip(sp_stats.linregress(hurst_lv, hurst_rs)[0], 0, 1)
            
            if H > 0.6:
                hurst_sma50 = np.mean(close[-50:])
                hurst_ret24h = (price - close[-24]) / close[-24] if len(close) >= 25 else 0
                hurst_vol_short = np.mean(volume[-48:]) if len(volume) >= 168 else 0
                hurst_vol_long = np.mean(volume[-168:]) if len(volume) >= 168 else 1
                hurst_vr = hurst_vol_short / hurst_vol_long if hurst_vol_long > 0 else 1
                
                if price > hurst_sma50 and hurst_ret24h > 0.01 and hurst_vr >= 1.2 and cur_rsi <= 70:
                    base_score = (H - 0.5) * 100 + hurst_ret24h * 50 + hurst_vr * 5
                    score = adjust_score('hurst_trend_long', base_score)
                    score = apply_mtf_confirmation('hurst_trend_long', 'long', score)
                    signals.append(({
                        'pair': pair, 'tool': 'hurst_trend_long', 'direction': 'long',
                        'hold': 168, 'sl_pct': 0.12,
                        'reason': f"HURST TREND: H={H:.2f}, ret24h={hurst_ret24h*100:.1f}%, vol={hurst_vr:.1f}x"
                    }, score))
        
        # ===== BULL SWING TOOLS — simple trend-following, hold for weeks =====
        # These use 15% trailing stop, 18% hard stop, 6-week max hold
        # Validated on 3yr OOS data, bull regime only
        
        # 33. simple_buy_uptrend: price > 50 SMA > 200 SMA, positive weekly momentum
        # OOS: 347 signals, 41% WR, +3.55%/trade, PF=1.52
        if len(close) >= 200:
            swing_sma50 = np.mean(close[-50:])
            swing_sma200 = np.mean(close[-200:])
            swing_ret1w = (price - close[-168]) / close[-168] if len(close) >= 168 else 0
            
            if (price > swing_sma50 and price > swing_sma200 and swing_sma50 > swing_sma200
                    and cur_rsi <= 70 and cur_rsi >= 35 and swing_ret1w > 0.02):
                base_score = swing_ret1w * 100
                score = adjust_score('simple_buy_uptrend', base_score)
                score = apply_mtf_confirmation('simple_buy_uptrend', 'long', score)
                signals.append(({
                    'pair': pair, 'tool': 'simple_buy_uptrend', 'direction': 'long',
                    'hold': 1008, 'sl_pct': 0.18,
                    'reason': f"UPTREND BUY: 50>200 SMA, ret1w={swing_ret1w*100:.1f}%"
                }, score))
        
        # 34. buy_weekly_green: 5%+ green week with above-avg volume
        # OOS: 257 signals, 44% WR, +5.92%/trade, PF=1.96
        if len(close) >= 336 and len(volume) >= 336:
            bwg_ret1w = (price - close[-168]) / close[-168]
            bwg_vol_ratio = np.mean(volume[-168:]) / np.mean(volume[-336:-168])
            bwg_sma200 = np.mean(close[-200:]) if len(close) >= 200 else price
            
            if bwg_ret1w >= 0.05 and bwg_vol_ratio >= 1.1 and cur_rsi <= 72 and price > bwg_sma200:
                base_score = bwg_ret1w * 100 + bwg_vol_ratio * 10
                score = adjust_score('buy_weekly_green', base_score)
                score = apply_mtf_confirmation('buy_weekly_green', 'long', score)
                signals.append(({
                    'pair': pair, 'tool': 'buy_weekly_green', 'direction': 'long',
                    'hold': 1008, 'sl_pct': 0.18,
                    'reason': f"WEEKLY GREEN: +{bwg_ret1w*100:.1f}%, vol {bwg_vol_ratio:.1f}x"
                }, score))
        
        # 35. buy_breakout_simple: new 30-day high with volume surge
        # OOS: 123 signals, 44% WR, +6.54%/trade, PF=2.20
        if len(close) >= 720 and len(volume) >= 720:
            bbs_high30d = np.max(close[-720:-24])
            bbs_vol_now = np.mean(volume[-24:])
            bbs_vol_avg = np.mean(volume[-720:-24])
            bbs_vr = bbs_vol_now / bbs_vol_avg if bbs_vol_avg > 0 else 1
            
            if price > bbs_high30d * 1.005 and bbs_vr >= 1.3 and cur_rsi <= 78:
                breakout_pct = (price / bbs_high30d - 1) * 100
                base_score = bbs_vr * 20 + breakout_pct * 50
                score = adjust_score('buy_breakout_simple', base_score)
                score = apply_mtf_confirmation('buy_breakout_simple', 'long', score)
                signals.append(({
                    'pair': pair, 'tool': 'buy_breakout_simple', 'direction': 'long',
                    'hold': 1008, 'sl_pct': 0.18,
                    'reason': f"BREAKOUT: new 30d high +{breakout_pct:.1f}%, vol {bbs_vr:.1f}x"
                }, score))
        
        # 36. buy_btc_leading: BTC pumping, alt lagging, rotation play
        # OOS: 103 signals, 31% WR, +4.64%/trade, PF=1.52, avg win +43.4%
        if pair != "XBTUSD" and "XBTUSD" in self._price_cache and len(close) >= 200:
            btc_prices = self._price_cache.get("XBTUSD", [])
            if len(btc_prices) >= 168:
                btl_btc1w = (btc_prices[-1] - btc_prices[-168]) / btc_prices[-168]
                btl_alt1w = (price - close[-168]) / close[-168] if len(close) >= 168 else 0
                btl_lag = btl_btc1w - btl_alt1w
                btl_alt48h = (price - close[-48]) / close[-48] if len(close) >= 48 else 0
                btl_sma200 = np.mean(close[-200:])
                
                if (btl_btc1w >= 0.05 and btl_lag >= 0.03 and btl_alt48h > 0
                        and cur_rsi <= 65 and price > btl_sma200 * 0.85):
                    base_score = btl_lag * 100 + btl_alt48h * 50
                    score = adjust_score('buy_btc_leading', base_score)
                    score = apply_mtf_confirmation('buy_btc_leading', 'long', score)
                    signals.append(({
                        'pair': pair, 'tool': 'buy_btc_leading', 'direction': 'long',
                        'hold': 1008, 'sl_pct': 0.18,
                        'reason': f"BTC LEADING: BTC +{btl_btc1w*100:.1f}%, alt lag {btl_lag*100:.1f}%"
                    }, score))
        
        # Enrich all signals with HTF context for journal logging
        for sig, sc in signals:
            sig['_htf_context'] = dict(htf_context) if htf_context else {}
        
        return signals
    
    # ===== UPGRADE 6: SMARTER GRID ENGINE =====
    
    def get_grid_levels_for_volatility(self, pair: str, current_price: float, data: dict) -> int:
        """UPGRADE 6: Use more levels for volatile pairs."""
        df = data.get('df')
        if df is not None and len(df) >= 15:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)
            atr14 = self.calc_atr(high, low, close, 14)
            if not np.isnan(atr14[-1]) and current_price > 0:
                atr_pct = atr14[-1] / current_price * 100
                # If ATR% > 5%, use 5 grid levels instead of 3
                if atr_pct > 5:
                    return 5
        return 3
    
    def get_grid_take_profit(self, pair: str, current_price: float, data: dict) -> float:
        """UPGRADE 6: Tighter TP in high-vol environments."""
        df = data.get('df')
        base_tp = 0.015  # 1.5% default
        
        if df is not None and len(df) >= 15:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)
            atr14 = self.calc_atr(high, low, close, 14)
            if not np.isnan(atr14[-1]) and current_price > 0:
                atr_pct = atr14[-1] / current_price * 100
                # If ATR% > 4%, use 2% TP instead of 1.5%
                if atr_pct > 4:
                    return 0.02
        
        return base_tp
    
    def get_grid_anchor(self, pair: str, current_price: float, data: dict) -> dict:
        """Get or create grid anchor for a pair."""
        if pair not in self.grid_anchors:
            # Create new anchor
            return self.create_grid_anchor(pair, current_price, data)
        
        anchor = self.grid_anchors[pair]
        
        # Check if reanchoring is needed (price drifted >10% from center)
        drift = abs(current_price - anchor["center"]) / anchor["center"]
        if drift > GRID_REANCHOR_PCT:
            return self.create_grid_anchor(pair, current_price, data)
        
        return anchor
    
    def create_grid_anchor(self, pair: str, current_price: float, data: dict) -> dict:
        """Create new grid anchor at current price."""
        # Get grid spacing for this pair
        grid_spacing = GRID_CONFIGS.get(pair, 0.012)  # Default 1.2% for new pairs
        
        # ATR-based spacing adjustment
        df = data.get('df')
        if df is not None and len(df) >= 15:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)
            atr14 = self.calc_atr(high, low, close, 14)
            if not np.isnan(atr14[-1]) and current_price > 0:
                atr_pct = atr14[-1] / current_price
                # Use ATR if it's reasonable (1-5%), otherwise fall back to config
                if 0.01 <= atr_pct <= 0.05:
                    grid_spacing = atr_pct
        
        # Adjust spacing based on SMA50 regime (downtrend = wider spacing)
        sma50 = self.calc_sma(data['df']['close'].values.astype(float), 50)
        if not np.isnan(sma50[-1]) and current_price < sma50[-1]:
            grid_spacing *= 1.5  # Wider grids in downtrends
        
        # UPGRADE 6: Get number of levels based on volatility
        num_levels = self.get_grid_levels_for_volatility(pair, current_price, data)
        
        # Log reanchoring
        if pair in self.grid_anchors:
            anchor = self.grid_anchors[pair]
            drift = (current_price - anchor["center"]) / anchor["center"]
            logger.info(f"[GRID] {pair} reanchoring: price ${current_price:.4f} drifted {drift:.1%} from center ${anchor['center']:.4f}")
        
        # Create new anchor with levels below current price
        levels = []
        for level in range(1, num_levels + 1):
            levels.append(current_price * (1 - grid_spacing * level))
        
        anchor = {"center": current_price, "levels": levels, "num_levels": num_levels}
        self.grid_anchors[pair] = anchor
        logger.info(f"[GRID] {pair} anchored @ ${current_price:.4f}, {num_levels} levels: {[f'${l:.4f}' for l in levels]}")
        return anchor
    
    def update_grids(self, market_data: dict):
        """UPGRADE 5 & 6: Smart grid with scaling balance."""
        # Grid disabled — all capital to active trading
        return
        # UPGRADE 5: Scale grid balance with current total balance
        grid_pct, _ = self.get_capital_allocation()
        grid_balance_for_allocation = self.total_balance * grid_pct
        grid_balance_per_pair = grid_balance_for_allocation / len(PAIRS)
        
        for pair in PAIRS:
            if pair not in market_data or pair not in GRID_CONFIGS:
                continue
                
            data = market_data[pair]
            current_price = data["price"]
            current_high = data.get("high", current_price)
            
            # Get or create grid anchor
            anchor = self.get_grid_anchor(pair, current_price, data)
            levels = anchor["levels"]
            num_levels = anchor.get("num_levels", 3)
            
            # Initialize positions list
            if pair not in self.grid_positions:
                self.grid_positions[pair] = []
            positions = self.grid_positions[pair]
            
            # UPGRADE 5: Calculate position size scaling with balance
            position_value = grid_balance_per_pair / num_levels  # Divided by actual levels
            qty = position_value / current_price
            
            # UPGRADE 6: Get take profit based on volatility
            grid_take_profit = self.get_grid_take_profit(pair, current_price, data)
            
            # TP multiplier based on SMA50 regime (2.0x in downtrends)
            sma50 = self.calc_sma(data['df']['close'].values.astype(float), 50)
            tp_multiplier = 2.0 if (not np.isnan(sma50[-1]) and current_price < sma50[-1]) else 1.0
            effective_tp = grid_take_profit * tp_multiplier
            
            # Check for buy fills at each level
            for i, buy_level in enumerate(levels):
                # Check if we already have a position at this level
                level_filled = any(abs(pos["buy_price"] - buy_level) / buy_level < 0.005 for pos in positions)
                
                if not level_filled and current_price <= buy_level:
                    # Buy filled at this level
                    position = {
                        "buy_price": buy_level,
                        "qty": qty,
                        "bar": self.current_bar,
                        "level": i + 1
                    }
                    positions.append(position)
                    
                    # Execute buy order (LIMIT order for maker fees)
                    if ENABLE_LIVE_TRADING:
                        try:
                            self.client.place_order(pair, "buy", "limit", qty, buy_level)
                        except Exception as e:
                            logger.error(f"Failed to place grid buy order for {pair}: {e}")
                    else:
                        logger.info(f"[DRY RUN] {pair} grid LIMIT buy: {qty:.6f} @ ${buy_level:.4f}")
                    
                    logger.info(f"[GRID] {pair} buy filled @ ${buy_level:.4f} (level {i+1}), qty: {qty:.6f}")
            
            # Check for sell fills (take profit)
            remaining_positions = []
            for pos in positions:
                sell_target = pos["buy_price"] * (1 + effective_tp)
                
                if current_high >= sell_target:
                    # Sell filled - book profit with improved fees
                    pnl = (sell_target - pos["buy_price"]) * pos["qty"] * GRID_FEE_MULTIPLIER
                    self.grid_profit += pnl
                    self.grid_round_trips += 1
                    
                    # Execute sell order (LIMIT order for maker fees)
                    if ENABLE_LIVE_TRADING:
                        try:
                            self.client.place_order(pair, "sell", "limit", pos["qty"], sell_target)
                        except Exception as e:
                            logger.error(f"Failed to place grid sell order for {pair}: {e}")
                    else:
                        logger.info(f"[DRY RUN] {pair} grid LIMIT sell: {pos['qty']:.6f} @ ${sell_target:.4f}")
                    
                    logger.info(f"[GRID] {pair} sell @ ${sell_target:.4f}, PnL: ${pnl:.2f}, total grid profit: ${self.grid_profit:.2f}")
                    # Position closed, don't add to remaining
                else:
                    remaining_positions.append(pos)
            
            self.grid_positions[pair] = remaining_positions
    
    # ===== POSITION MANAGEMENT WITH UPGRADES =====
    
    def _get_exit_params(self, tool: str, price: float, market_data_entry: dict) -> tuple:
        """Get exit parameters for validated tools only."""
        # Mean reversion strategies — TP at 8-10%
        MEAN_REVERSION = {
            'volatile_oversold', 'crash_neg_ac', 'crash_mean_revert', 
            'blood_in_streets', 'vpin_dip', 'vpin_toxic', 'entropy_dip',
            'btc_alt_spread'
        }
        
        # Crash buy strategies — TP at 10-12%
        CRASH_BUY = {
            'crash_buy', 'mega_crash', 'quick_crash', 'flash_crash',
            'market_panic_70', 'deep_dip_8h'
        }
        
        # Short strategies — TP at 6%
        SHORT_TOOLS = {
            'mega_pump_sell_t1', 'rsi_pump_8h', 'falling_wedge_short', 'greed_short_t2',
            'thursday_short', 'mega_pump_sell_t2', 'distribution_short', 'late_us_short',
            'rsi_pump_12h', 'ema_cross_short', 'rsi_pump_fat_tail', 'entropy_short',
            'alt_btc_revert_t3'
        }
        
        # Dip buy — TP at 6%
        DIP_BUY = {'quick_dip', 'dip_buy_5pct'}
        
        # Neutral tools — TP at 6%
        NEUTRAL = {'month_start_long'}
        
        # Bull swing tools — 15% trailing stop, no fixed TP (let winners run)
        BULL_SWING = {
            'buy_weekly_green', 'buy_breakout_simple',
            'simple_buy_uptrend', 'buy_btc_leading'
        }
        
        # Bull momentum tools — 8% trailing stop
        BULL_MOMENTUM = {'accumulation_breakout', 'hurst_trend_long'}
        
        if tool in BULL_SWING:
            return ('trailing', None, 0.15, None)  # 15% trailing stop
        elif tool in BULL_MOMENTUM:
            return ('trailing', None, 0.08, None)  # 8% trailing stop
        elif tool in MEAN_REVERSION:
            return ('fixed_tp', 0.08, None, None)  # 8% TP
        elif tool in CRASH_BUY:
            return ('fixed_tp', 0.10, None, None)  # 10% TP  
        elif tool in SHORT_TOOLS:
            return ('fixed_tp', 0.06, None, None)  # 6% TP
        elif tool in DIP_BUY:
            return ('fixed_tp', 0.06, None, None)  # 6% TP
        elif tool in NEUTRAL:
            return ('fixed_tp', 0.06, None, None)  # 6% TP
        else:
            # Default for any unclassified tool
            return ('default', None, None, None)
    
    # ===== REAL-TIME CRASH HANDLER =====
    
    def _on_crash_detected(self, pair, crash_type, drop_pct, current_price):
        """Called by websocket thread when a crash is detected. Queue for main loop."""
        self.pending_crash_signals.append({
            'pair': pair,
            'crash_type': crash_type,
            'drop_pct': drop_pct,
            'price': current_price,
            'timestamp': time.time()
        })
        logger.info(f"🚨 CRASH QUEUED: {pair} {crash_type} {drop_pct*100:.1f}% @ ${current_price:.2f}")
    
    def _on_pump_detected(self, pair, pump_type, change_pct, current_price):
        """Pump detected — tighten trailing stops on open positions."""
        self.pending_pump_signals.append({
            'pair': pair, 'pump_type': pump_type,
            'change_pct': change_pct, 'price': current_price
        })
    
    def _on_volume_spike(self, pair, volume_mult, current_price):
        """Volume spike — potential exit signal for profitable positions."""
        self.pending_volume_spikes.append({
            'pair': pair, 'volume_mult': volume_mult, 'price': current_price
        })
    
    def process_realtime_signals(self):
        """Process all queued real-time signals: crashes, pumps, volume spikes."""
        # Process crashes (buy the blood)
        self.process_crash_signals()
        
        # Process pumps — tighten trailing stops on open longs
        if self.pending_pump_signals:
            pumps = self.pending_pump_signals.copy()
            self.pending_pump_signals.clear()
            for pump in pumps:
                pair = pump['pair']
                if pair in self.active_positions:
                    pos = self.active_positions[pair]
                    if pos['direction'] == 'long':
                        # Tighten stop to 5% on pump (protect gains in rapid move)
                        pnl = (pump['price'] - pos['entry_price']) / pos['entry_price']
                        if pnl > 0.02:  # Only if profitable
                            pos['_pump_tighten'] = True
                            logger.info(f"🚀 PUMP TIGHTEN: {pair} +{pump['change_pct']*100:.1f}% surge, "
                                       f"tightening trail to 5% (pnl {pnl*100:+.1f}%)")
        
        # Process volume spikes — exit profitable positions at potential tops
        if self.pending_volume_spikes:
            spikes = self.pending_volume_spikes.copy()
            self.pending_volume_spikes.clear()
            for spike in spikes:
                pair = spike['pair']
                if pair in self.active_positions:
                    pos = self.active_positions[pair]
                    price = spike['price']
                    if pos['direction'] == 'long':
                        pnl = (price - pos['entry_price']) / pos['entry_price']
                    else:
                        pnl = (pos['entry_price'] - price) / pos['entry_price']
                    
                    if pnl > 0.03:  # 3%+ profit + volume spike = exit
                        logger.info(f"💰 VOLUME EXIT: {pair} {spike['volume_mult']:.1f}x volume, pnl {pnl*100:+.1f}%")
                        self.close_position(pair, price, 
                            f"RT volume spike {spike['volume_mult']:.1f}x, pnl {pnl*100:+.1f}%")
    
    def process_crash_signals(self):
        """Process any queued crash signals between regular cycles. Buy the blood."""
        if not self.pending_crash_signals:
            return
        
        signals = self.pending_crash_signals.copy()
        self.pending_crash_signals.clear()
        
        for crash in signals:
            pair = crash['pair']
            price = crash['price']
            drop = crash['drop_pct']
            
            # Skip if already have position in this pair
            if pair in self.active_positions:
                logger.info(f"Crash signal for {pair} but already have position, skipping")
                continue
            
            # Skip if at max positions
            if len(self.active_positions) >= MAX_ACTIVE_POSITIONS:
                logger.info(f"Crash signal for {pair} but at max positions, skipping")
                continue
            
            # Determine signal strength based on crash magnitude
            if abs(drop) >= 0.05:
                tool = 'ws_mega_crash'
                score = abs(drop) * 500
                sl_pct = 0.08
            elif abs(drop) >= 0.03:
                tool = 'ws_flash_crash'
                score = abs(drop) * 300
                sl_pct = 0.05
            else:
                tool = 'ws_flash_dip'
                score = abs(drop) * 200
                sl_pct = 0.04
            
            logger.info(f"🚨 EXECUTING CRASH BUY: {pair} @ ${price:.2f} ({drop*100:.1f}%) tool={tool}")
            
            signal = {
                'pair': pair, 'tool': tool, 'direction': 'long',
                'hold': 24, 'sl_pct': sl_pct,
                'reason': f"WS CRASH: {crash['crash_type']} {drop*100:.1f}% detected real-time"
            }
            self.execute_signal(signal, score)
    
    # ===== SMART EXIT HELPERS =====
    
    def _check_volume_spike_exit(self, pos, data, current_price):
        """Exit profitable positions on 2x+ volume spike (local top signal)."""
        df = data.get('df')
        if df is None or len(df) < 20:
            return None
        if pos['direction'] == 'long':
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
        if pnl_pct <= 0.01:  # Need at least 1% profit
            return None
        volumes = df['volume'].values.astype(float)[-20:]
        cur_vol = volumes[-1]
        avg_vol = np.mean(volumes[:-1])
        spike = cur_vol / avg_vol if avg_vol > 0 else 0
        if spike >= 2.0:
            return f"Volume spike exit: {spike:.1f}x avg, pnl {pnl_pct*100:+.1f}%"
        return None
    
    def _check_regime_exit(self, pos):
        """Exit bull swing positions when F&G drops to fear."""
        bull_swing_tools = {'buy_weekly_green', 'buy_breakout_simple', 
                           'simple_buy_uptrend', 'buy_btc_leading',
                           'accumulation_breakout', 'hurst_trend_long'}
        if pos['direction'] != 'long' or pos.get('tool') not in bull_swing_tools:
            return None
        fng = self.get_fng()
        if fng < 30:
            return f"Regime exit: F&G={fng} < 30, bull tool in fear"
        return None
    
    def _smart_trailing_adjustment(self, pos, data, base_trail):
        """Tighten trail on RSI overbought or momentum exhaustion."""
        df = data.get('df')
        if df is None or len(df) < 20:
            return base_trail
        
        close = df['close'].values.astype(float)
        current_price = data['price']
        adjusted = base_trail
        
        # RSI tightening: profitable long + RSI > 80 → tighten to 5%
        rsi_vals = self.calc_rsi(close, 14)
        cur_rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
        
        is_profitable = (current_price > pos['entry_price'] if pos['direction'] == 'long' 
                        else current_price < pos['entry_price'])
        
        if is_profitable:
            if pos['direction'] == 'long' and cur_rsi >= 80:
                adjusted = min(adjusted, 0.05)
            elif pos['direction'] == 'short' and cur_rsi <= 20:
                adjusted = min(adjusted, 0.05)
        
        # Momentum exhaustion: 7d momentum < 3d momentum → tighten to 7%
        if len(close) >= 168:
            mom_7d = (close[-1] - close[-168]) / close[-168]
            mom_3d = (close[-1] - close[-72]) / close[-72] if len(close) >= 72 else mom_7d
            if mom_7d < mom_3d and mom_3d > 0 and is_profitable:
                adjusted = min(adjusted, 0.07)
        
        return adjusted
    
    def _check_dca_opportunity(self, pair: str, pos: dict, current_price: float, market_data: dict):
        """Check if we should DCA (average down) into an existing crash-buy position.
        
        Only DCA if:
        1. Tool is a crash/dip buy tool
        2. Price dropped >3% further since entry
        3. Haven't already DCA'd on this position
        4. Have enough active balance
        """
        DCA_TOOLS = {'crash_buy', 'mega_crash', 'blood_in_streets', 'volatile_oversold',
                    'crash_neg_ac', 'crash_mean_revert', 'quick_crash', 'flash_crash',
                    'market_panic_70', 'deep_dip_8h', 'entropy_dip', 'vpin_dip'}
        
        if pos['tool'] not in DCA_TOOLS:
            return
        if pos.get('_dca_done'):
            return
        if pos['direction'] != 'long':
            return
        
        # Check if price dropped >3% from entry
        drop_from_entry = (pos['entry_price'] - current_price) / pos['entry_price']
        if drop_from_entry < 0.03:
            return
        
        # Check if we have enough balance for DCA (use 50% of original position size)
        dca_size = pos['position_size'] * 0.5
        if self.active_balance < dca_size:
            return
        
        # Execute DCA buy
        dca_qty = dca_size / current_price
        
        if ENABLE_LIVE_TRADING:
            try:
                bid = market_data.get(pair, {}).get("bid", current_price) if isinstance(market_data, dict) else current_price
                order_result = self.client.place_order(pair, "buy", "limit", dca_qty, bid)
                if not order_result:
                    return
                logger.info(f"[DCA] {pair} adding ${dca_size:.2f} @ ${bid:.4f} (down {drop_from_entry:.1%} from entry)")
            except Exception as e:
                logger.error(f"DCA order failed for {pair}: {e}")
                return
        else:
            logger.info(f"[DRY RUN DCA] {pair} adding ${dca_size:.2f} @ ${current_price:.4f} (down {drop_from_entry:.1%})")
        
        # Update position with new average
        old_cost = pos['entry_price'] * pos['qty']
        new_cost = current_price * dca_qty
        total_qty = pos['qty'] + dca_qty
        new_avg_price = (old_cost + new_cost) / total_qty
        
        pos['entry_price'] = new_avg_price
        pos['qty'] = total_qty
        pos['position_size'] += dca_size
        pos['_dca_done'] = True
        pos['_dca_price'] = current_price
        pos['_dca_size'] = dca_size
        
        # Reserve additional capital
        self.active_balance -= dca_size
        
        logger.info(f"[DCA COMPLETE] {pair} new avg: ${new_avg_price:.4f} | "
                   f"Total size: ${pos['position_size']:.2f} | "
                   f"New qty: {total_qty:.4f}")

    def manage_positions(self, market_data: dict):
        """Check all active positions for exits with margin cost tracking."""
        for pair in list(self.active_positions.keys()):
            if pair not in market_data:
                continue
            
            # Skip positions with pending exit orders (waiting for fill confirmation)
            if pair in self.pending_exit_orders or self.active_positions[pair].get('_pending_exit'):
                continue
                
            pos = self.active_positions[pair]
            data = market_data[pair]
            current_price = data["price"]
            
            # Check DCA opportunity for crash positions
            self._check_dca_opportunity(pair, pos, current_price, market_data)
            
            # Calculate bars held
            bars_held = self.current_bar - pos.get("entry_bar", self.current_bar)
            
            # UPGRADE 3: Add margin costs for Tier 1 tools with 2x leverage
            if pos.get("leverage", 1) == 2:
                # Add margin cost per bar (0.02% every 4 hours)
                margin_cost = pos["position_size"] * MARGIN_COST_PER_BAR * bars_held
                pos["total_margin_cost"] = pos.get("total_margin_cost", 0) + MARGIN_COST_PER_BAR * pos["position_size"]
            
            # Get exit parameters for this tool
            exit_mode, take_profit_pct, trailing_stop_pct, _ = self._get_exit_params(
                pos['tool'], current_price, data)
            
            # UPGRADE 3: Adjust stop loss for 2x leverage (tighter SL)
            effective_sl_pct = pos['sl_pct']
            if pos.get("leverage", 1) == 2:
                effective_sl_pct = pos['sl_pct'] / 2  # Tighter SL for 2x leverage
            
            # Check stop loss
            if pos['direction'] == 'long':
                sl_price = pos['entry_price'] * (1 - effective_sl_pct)
                if current_price <= sl_price:
                    self.close_position(pair, current_price, f"Stop loss @ ${sl_price:.4f}")
                    continue
            else:  # short
                sl_price = pos['entry_price'] * (1 + effective_sl_pct)
                if current_price >= sl_price:
                    self.close_position(pair, current_price, f"Stop loss @ ${sl_price:.4f}")
                    continue
            
            # Smart trailing stop — volume spike, regime, RSI, momentum, ATR-adaptive
            if trailing_stop_pct:
                # Track highest/lowest price since entry
                if 'best_price' not in pos:
                    pos['best_price'] = pos['entry_price']
                
                # SMART EXIT #1: Volume spike (instant exit on profitable + 2x vol)
                vol_exit = self._check_volume_spike_exit(pos, data, current_price)
                if vol_exit:
                    self.close_position(pair, current_price, vol_exit)
                    continue
                
                # SMART EXIT #2: Regime change (bull tools exit on F&G fear)
                regime_exit = self._check_regime_exit(pos)
                if regime_exit:
                    self.close_position(pair, current_price, regime_exit)
                    continue
                
                # SMART EXIT #3: RSI + momentum tightening + real-time pump tighten
                smart_trail = self._smart_trailing_adjustment(pos, data, trailing_stop_pct)
                
                # Real-time pump detection override — tighten to 5% if flagged
                if pos.get('_pump_tighten'):
                    smart_trail = min(smart_trail, 0.05)
                
                # Dynamic trail: use 3x ATR, bounded by smart-adjusted trail %
                atr_pct = (data.get('atr', 0) / current_price) if current_price > 0 else 0
                dynamic_trail = max(
                    smart_trail * 0.5,               # Floor: half the adjusted trail
                    min(atr_pct * 3,                  # 3x ATR
                        smart_trail * 1.5)            # Cap: 1.5x the adjusted trail
                ) if atr_pct > 0 else smart_trail
                
                if pos['direction'] == 'long':
                    if current_price > pos['best_price']:
                        pos['best_price'] = current_price
                    trail_dd = (pos['best_price'] - current_price) / pos['best_price']
                    if trail_dd >= dynamic_trail:
                        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                        self.close_position(pair, current_price, 
                            f"Trailing stop {dynamic_trail:.1%} (peak ${pos['best_price']:.2f}, pnl {pnl_pct:+.1f}%)")
                        continue
                else:  # short
                    if current_price < pos['best_price']:
                        pos['best_price'] = current_price
                    trail_up = (current_price - pos['best_price']) / pos['best_price']
                    if trail_up >= dynamic_trail:
                        pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
                        self.close_position(pair, current_price,
                            f"Trailing stop {dynamic_trail:.1%} (trough ${pos['best_price']:.2f}, pnl {pnl_pct:+.1f}%)")
                        continue
            
            # Check take profit (for fixed-TP tools)
            if take_profit_pct:
                if pos['direction'] == 'long':
                    tp_price = pos['entry_price'] * (1 + take_profit_pct)
                    if current_price >= tp_price:
                        if not pos.get('_partial_closed'):
                            self._partial_close(pair, current_price, 0.5, f"TP hit @ ${tp_price:.4f}")
                        else:
                            self.close_position(pair, current_price, f"Take profit (remaining) @ ${tp_price:.4f}")
                        continue
                else:  # short
                    tp_price = pos['entry_price'] * (1 - take_profit_pct)
                    if current_price <= tp_price:
                        if not pos.get('_partial_closed'):
                            self._partial_close(pair, current_price, 0.5, f"TP hit @ ${tp_price:.4f}")
                        else:
                            self.close_position(pair, current_price, f"Take profit (remaining) @ ${tp_price:.4f}")
                        continue
            
            # Check hold timeout (based on real elapsed hours, not bar count)
            entry_time = pos.get('entry_time', None)
            if entry_time:
                elapsed_hours = (datetime.now(timezone.utc).timestamp() - entry_time) / 3600
            else:
                # Fallback for positions opened before this fix
                elapsed_hours = bars_held * (CHECK_INTERVAL / 3600)
            if elapsed_hours >= pos['hold']:
                self.close_position(pair, current_price, f"Hold timeout ({elapsed_hours:.1f}h)")
                continue
    
    # UPGRADE 1: Handle pending limit orders
    def manage_pending_limit_orders(self, market_data: dict):
        """Simple order management: check Kraken reality, not heuristics.
        
        For each pending order:
        1. Order still open on Kraken → leave it alone
        2. Order gone + we hold the asset → it filled, confirm position
        3. Order gone + we don't hold it → cancelled/expired, free capital
        """
        if not self.pending_limit_orders:
            return
        
        # Get open orders and balances from Kraken
        open_txids = set()
        balances = {}
        if ENABLE_LIVE_TRADING:
            try:
                open_orders = self.client.get_open_orders()
                if isinstance(open_orders, dict) and 'open' in open_orders:
                    open_txids = set(open_orders['open'].keys())
                elif isinstance(open_orders, dict):
                    open_txids = set(open_orders.keys())
            except Exception as e:
                logger.debug(f"Error checking open orders: {e}")
                return  # Can't check — don't make decisions
            
            try:
                balances = self.client.get_account_balance() or {}
            except Exception as e:
                logger.debug(f"Error checking balances: {e}")
                return
        
        for pair, order_info in list(self.pending_limit_orders.items()):
            order_id = order_info.get("order_id")
            tool = order_info.get("tool", "unknown")
            direction = order_info.get("direction", "long")
            
            # Extract txid
            if isinstance(order_id, dict) and 'txid' in order_id:
                txid = order_id['txid'][0] if isinstance(order_id['txid'], list) else order_id['txid']
            elif isinstance(order_id, str):
                txid = order_id
            else:
                txid = None
            
            # 1. Order still open on Kraken — check if we should cancel it
            if txid and txid in open_txids:
                should_cancel = False
                cancel_reason = ""
                
                # Price drifted too far from entry
                original_price = order_info.get("original_price", order_info.get("price", 0))
                if pair in market_data and original_price > 0:
                    current_price = market_data[pair]["price"]
                    drift = abs(current_price - original_price) / original_price
                    if drift > PRICE_DRIFT_ABANDON:
                        should_cancel = True
                        cancel_reason = f"price drift {drift:.1%} (was ${original_price:.4f}, now ${current_price:.4f})"
                
                # Better executable signal available (ignore blocked shorts)
                if not should_cancel and hasattr(self, '_current_cycle_signals'):
                    original_score = order_info.get("original_score", 0)
                    for sig, sig_score in self._current_cycle_signals:
                        if sig['direction'] == 'short' and ENABLE_LIVE_TRADING:
                            continue
                        if sig['pair'] != pair and sig['pair'] not in self.active_positions and sig_score > original_score * 1.5:
                            should_cancel = True
                            cancel_reason = f"stronger signal: {sig['tool']} {sig['pair']} (score {sig_score:.1f} vs {original_score:.1f})"
                            break
                
                if should_cancel:
                    # Cancel on Kraken — next cycle the reality check handles the rest
                    logger.info(f"[CANCEL] {pair} — {cancel_reason}")
                    try:
                        self.client.cancel_order(txid)
                    except Exception as e:
                        logger.warning(f"Cancel failed for {pair}: {e}")
                
                continue
            
            # Order is gone from Kraken. Check if we hold the asset.
            base_asset = pair.replace('USD', '')
            held_qty = 0
            for asset, amount in balances.items():
                if asset == base_asset or asset == 'X' + base_asset:
                    held_qty = float(amount)
                    break
            
            expected_qty = order_info.get("qty", 0)
            
            if held_qty > expected_qty * 0.5:
                # 2. We hold it → order filled
                logger.info(f"[FILLED] {pair} {direction} — holding {held_qty:.4f} on Kraken")
                
                if pair not in self.active_positions:
                    entry_price = order_info.get("price", 0)
                    self.active_positions[pair] = {
                        'pair': pair,
                        'tool': tool,
                        'direction': direction,
                        'leverage': 1,
                        'entry_price': entry_price,
                        'entry_bar': order_info.get("placed_bar", self.current_bar),
                        'entry_time': datetime.now(timezone.utc).timestamp(),
                        'position_size': held_qty * entry_price,
                        'qty': held_qty,
                        'sl_pct': 0.04,
                        'hold': 24,
                        'score': order_info.get("original_score", 0),
                        'total_margin_cost': 0
                    }
                    
                    # Place TP order immediately
                    exit_mode, take_profit_pct, _, _ = self._get_exit_params(tool, entry_price, {})
                    if take_profit_pct and ENABLE_LIVE_TRADING:
                        try:
                            tp_price = entry_price * (1 + take_profit_pct) if direction == 'long' else entry_price * (1 - take_profit_pct)
                            tp_side = "sell" if direction == 'long' else "buy"
                            tp_qty = held_qty * 0.5
                            tp_result = self.client.place_order(pair, tp_side, "limit", tp_qty, tp_price)
                            if tp_result:
                                self.active_positions[pair]['_tp_order_id'] = tp_result.get('txid', [None])[0] if isinstance(tp_result, dict) else tp_result
                                self.active_positions[pair]['_tp_price'] = tp_price
                                logger.info(f"[TP PLACED] {pair} {tp_side} {tp_qty:.4f} @ ${tp_price:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to place TP for {pair}: {e}")
                
                del self.pending_limit_orders[pair]
            
            else:
                # 3. We don't hold it → order expired/cancelled, free capital
                logger.info(f"[EXPIRED] {pair} {direction} — order gone, not holding asset. Freeing capital.")
                
                if pair in self.active_positions:
                    pos = self.active_positions[pair]
                    self.active_balance += pos['position_size']
                    del self.active_positions[pair]
                    logger.info(f"[CAPITAL FREED] ${pos['position_size']:.2f} returned to active balance")
                
                self._log_rejection(pair, tool, direction, order_info.get("original_score", 0), "order_expired")
                del self.pending_limit_orders[pair]
    
    def _partial_close(self, pair: str, price: float, close_pct: float, reason: str):
        """Close a percentage of a position and keep the rest running.
        
        Args:
            pair: Trading pair
            price: Current price
            close_pct: Fraction to close (0.5 = 50%)
            reason: Reason for partial close
        """
        if pair not in self.active_positions:
            return
        
        pos = self.active_positions[pair]
        close_qty = pos['qty'] * close_pct
        keep_qty = pos['qty'] * (1 - close_pct)
        close_size = pos['position_size'] * close_pct
        
        # Calculate PnL on the closed portion
        if pos['direction'] == 'long':
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - price) / pos['entry_price']
        
        leverage = pos.get('leverage', 1)
        if leverage == 2:
            pnl_pct *= 2
        pnl_pct -= ROUND_TRIP_FEE
        pnl_dollar = pnl_pct * close_size
        
        # Execute partial sell
        if ENABLE_LIVE_TRADING:
            try:
                side = "sell" if pos['direction'] == 'long' else "buy"
                self.client.place_order(pair, side, "limit", close_qty, price, post_only=True)
                logger.info(f"[PARTIAL EXIT] {pair} {close_pct:.0%} @ ${price:.4f} (maker fee)")
            except Exception as e:
                try:
                    self.client.place_order(pair, side, "market", close_qty, price)
                    logger.info(f"[PARTIAL EXIT] {pair} {close_pct:.0%} @ ${price:.4f} (market fallback)")
                except Exception as e2:
                    logger.error(f"Failed partial close for {pair}: {e2}")
                    return
        else:
            logger.info(f"[DRY RUN] Partial close {close_pct:.0%} of {pair} @ ${price:.4f}")
        
        # Update position to reflect remaining
        pos['qty'] = keep_qty
        pos['position_size'] = pos['position_size'] * (1 - close_pct)
        pos['_partial_closed'] = True
        pos['_partial_close_price'] = price
        
        # Tighten stop loss for remaining portion (trail from current price)
        if pos['direction'] == 'long':
            # Set trailing stop at 3% below current (tighter than original)
            pos['sl_pct'] = 0.03
            pos['_trail_from'] = price
        else:
            pos['sl_pct'] = 0.03
            pos['_trail_from'] = price
        
        # Update balances for closed portion
        self.active_balance += close_size + pnl_dollar
        self.active_profit += pnl_dollar
        self.total_balance += pnl_dollar
        
        # Update daily stats
        self._daily_stats["pnl"] += pnl_dollar
        if pnl_dollar > 0:
            self._daily_stats["wins"] += 1
        
        # Journal
        self._journal_close(
            pair=pair, tool=pos['tool'], direction=pos['direction'],
            exit_price=price, pnl_pct=pnl_pct, pnl_dollar=pnl_dollar,
            bars_held=0, close_reason=f"partial_{reason}",
            entry_price=pos['entry_price']
        )
        
        logger.info(f"[PARTIAL {close_pct:.0%}] {pair} @ ${price:.4f} | "
                   f"PnL: ${pnl_dollar:+.2f} ({pnl_pct:+.2%}) | "
                   f"Remaining: {keep_qty:.4f} with 3% trailing stop")

    def close_position(self, pair: str, price: float, reason: str):
        """Close an active position with improved fees and margin cost tracking."""
        if pair not in self.active_positions:
            return
            
        pos = self.active_positions[pair]
        
        # Calculate PnL
        if pos['direction'] == 'long':
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
        else:  # short
            pnl_pct = (pos['entry_price'] - price) / pos['entry_price']
        
        # UPGRADE 3: Apply leverage multiplier for returns
        leverage = pos.get("leverage", 1)
        if leverage == 2:
            pnl_pct *= 2  # 2x the returns with 2x leverage
        
        # UPGRADE 1: Apply correct fees (mixed round trip: 0.25% entry + 0.40% exit = 0.65%)
        pnl_pct -= ROUND_TRIP_FEE
        
        # UPGRADE 3: Subtract margin costs for leveraged positions
        total_margin_cost_pct = 0
        if leverage == 2:
            # Use real elapsed time for margin cost (MARGIN_COST_PER_BAR is per 4h)
            entry_time_m = pos.get('entry_time', None)
            if entry_time_m:
                hours_elapsed = (datetime.now(timezone.utc).timestamp() - entry_time_m) / 3600
                four_hour_periods = hours_elapsed / 4.0
            else:
                four_hour_periods = (self.current_bar - pos.get("entry_bar", self.current_bar)) * (CHECK_INTERVAL / 14400)
            total_margin_cost_pct = MARGIN_COST_OPEN + (MARGIN_COST_PER_BAR * four_hour_periods)
            pnl_pct -= total_margin_cost_pct
        
        pnl_dollar = pnl_pct * pos['position_size']
        
        # NOTE: Balance updates, tool stats, journal, and position removal
        # are handled in _finalize_exit() AFTER the exit order is confirmed filled.
        
        # Execute close order — use post-only LIMIT for maker fees (0.25% vs 0.40% taker)
        # Position stays in active_positions until exit order CONFIRMED filled
        exit_order_id = None
        if ENABLE_LIVE_TRADING:
            try:
                side = "sell" if pos['direction'] == 'long' else "buy"
                qty = pos['qty']
                if leverage == 2 and hasattr(self.client, 'close_leveraged_position'):
                    self.client.close_leveraged_position(pair, side, qty, price)
                    exit_order_id = "leveraged_close"
                else:
                    # Post-only limit at current price — guarantees maker fee
                    result = self.client.place_order(pair, side, "limit", qty, price, post_only=True)
                    if isinstance(result, dict) and 'txid' in result:
                        exit_order_id = result['txid'][0] if isinstance(result['txid'], list) else result['txid']
                    elif isinstance(result, str):
                        exit_order_id = result
                    logger.info(f"[EXIT PENDING] {pair} post-only limit @ ${price:.4f} (maker fee) — waiting for fill")
            except Exception as e:
                # Fallback to market if limit fails
                logger.warning(f"Post-only limit failed for {pair}: {e}, falling back to market")
                try:
                    result = self.client.place_order(pair, side, "market", qty, price)
                    exit_order_id = "market_fallback"
                    logger.info(f"[EXIT MARKET] {pair} market fallback @ ${price:.4f} (taker fee)")
                except Exception as e2:
                    logger.error(f"Failed to close position for {pair}: {e2}")
                    return  # Don't remove position if we couldn't place ANY exit order
        else:
            leverage_str = f" (2x margin)" if leverage == 2 else ""
            logger.info(f"[DRY RUN] Close {pos['direction']} {pair} @ ${price:.4f}{leverage_str}")
            exit_order_id = "dry_run"
        
        # Store pre-computed PnL and exit info for when fill is confirmed
        entry_time = pos.get('entry_time', None)
        if entry_time:
            hours_held = (datetime.now(timezone.utc).timestamp() - entry_time) / 3600
        else:
            hours_held = (self.current_bar - pos['entry_bar']) * (CHECK_INTERVAL / 3600)
        
        # Track pending exit — position stays in active_positions until confirmed
        self.pending_exit_orders[pair] = {
            "order_id": exit_order_id,
            "placed_bar": self.current_bar,
            "exit_price": price,
            "reason": reason,
            "pnl_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
            "hours_held": hours_held,
            "leverage": leverage,
            "total_margin_cost_pct": total_margin_cost_pct if leverage == 2 else 0,
            "side": "sell" if pos['direction'] == 'long' else "buy",
            "qty": pos['qty']
        }
        
        # Mark position as pending exit (so bot doesn't try to close again)
        self.active_positions[pair]['_pending_exit'] = True
        
        leverage_str = f" (2x leverage)" if leverage == 2 else ""
        margin_cost_str = f", margin: -{total_margin_cost_pct:.2%}" if leverage == 2 else ""
        logger.info(f"[CLOSE PENDING] {pair} {pos['direction']} @ ${price:.4f}{leverage_str} | "
                   f"{reason} | PnL: {pnl_pct:.2%} (${pnl_dollar:.2f}){margin_cost_str} | "
                   f"Tool: {tool} | Held: {hours_held:.1f}h")
        
        # For dry run or market orders, finalize immediately
        if exit_order_id in ("dry_run", "market_fallback", "leveraged_close"):
            self._finalize_exit(pair, price, reason)
    
    def _finalize_exit(self, pair: str, exit_price: float, reason: str):
        """Finalize a confirmed exit — update balances, stats, journal, remove position."""
        if pair not in self.active_positions:
            return
        if pair not in self.pending_exit_orders:
            return
        
        pos = self.active_positions[pair]
        exit_info = self.pending_exit_orders[pair]
        pnl_pct = exit_info["pnl_pct"]
        pnl_dollar = exit_info["pnl_dollar"]
        hours_held = exit_info["hours_held"]
        leverage = exit_info["leverage"]
        total_margin_cost_pct = exit_info["total_margin_cost_pct"]
        tool = pos['tool']
        
        # Update balances
        self.active_balance += pos['position_size'] + pnl_dollar
        self.total_balance += pnl_dollar
        self.active_profit += pnl_dollar
        
        # Update tool stats and streaks
        if tool in self.tool_stats:
            self.tool_stats[tool]['trades'] += 1
            won = pnl_pct > 0
            if won:
                self.tool_stats[tool]['wins'] += 1
                prev_avg = self.tool_stats[tool].get('avg_win_pct', pnl_pct / 100)
                n_wins = self.tool_stats[tool]['wins']
                self.tool_stats[tool]['avg_win_pct'] = prev_avg + (pnl_pct / 100 - prev_avg) / n_wins
            else:
                n_losses = self.tool_stats[tool]['trades'] - self.tool_stats[tool]['wins']
                prev_avg = self.tool_stats[tool].get('avg_loss_pct', pnl_pct / 100)
                self.tool_stats[tool]['avg_loss_pct'] = prev_avg + (pnl_pct / 100 - prev_avg) / max(n_losses, 1)
            self.tool_stats[tool]['pnl'] += pnl_dollar
            self.update_tool_streak(tool, won)
        
        # Update daily stats
        self._daily_stats["trades_closed"] += 1
        self._daily_stats["pnl"] += pnl_dollar
        if pnl_dollar > 0:
            self._daily_stats["wins"] += 1
        else:
            self._daily_stats["losses"] += 1
        self._daily_stats["tool_pnl"][tool] = self._daily_stats["tool_pnl"].get(tool, 0) + pnl_dollar
        
        # Record trade
        trade = {
            'pair': pair,
            'tool': tool,
            'direction': pos['direction'],
            'leverage': leverage,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'entry_bar': pos['entry_bar'],
            'exit_bar': self.current_bar,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'margin_cost_pct': total_margin_cost_pct,
            'reason': reason
        }
        self.trade_history.append(trade)
        
        # Journal
        self._journal_close(
            pair=pair, tool=tool, direction=pos['direction'],
            exit_price=exit_price, pnl_pct=pnl_pct, pnl_dollar=pnl_dollar,
            bars_held=round(hours_held, 1), close_reason=reason,
            entry_price=pos['entry_price']
        )
        
        leverage_str = f" (2x)" if leverage == 2 else ""
        logger.info(f"[EXIT CONFIRMED] {pair} {pos['direction']}{leverage_str} | "
                   f"PnL: ${pnl_dollar:+.2f} ({pnl_pct:+.2%}) | Tool: {tool}")
        
        # Remove position and pending exit
        del self.active_positions[pair]
        del self.pending_exit_orders[pair]
    
    def manage_pending_exit_orders(self, market_data: dict):
        """Check if pending exit orders filled. Escalate to market if stale."""
        if not self.pending_exit_orders:
            return
        
        # Get open orders from Kraken
        open_txids = set()
        if ENABLE_LIVE_TRADING:
            try:
                open_orders = self.client.get_open_orders()
                if isinstance(open_orders, dict) and 'open' in open_orders:
                    open_txids = set(open_orders['open'].keys())
                elif isinstance(open_orders, dict):
                    open_txids = set(open_orders.keys())
            except Exception as e:
                logger.debug(f"Error checking exit orders: {e}")
                return
        
        for pair in list(self.pending_exit_orders.keys()):
            exit_info = self.pending_exit_orders[pair]
            order_id = exit_info.get("order_id")
            
            # Skip non-limit exits (already finalized)
            if order_id in ("dry_run", "market_fallback", "leveraged_close"):
                continue
            
            # Check if order filled (not in open orders anymore)
            if order_id and order_id not in open_txids:
                logger.info(f"[EXIT FILLED] {pair} exit order {order_id} confirmed filled!")
                self._finalize_exit(pair, exit_info["exit_price"], exit_info["reason"])
                continue
            
            # Check if stale — but ONLY escalate exits that should fill near current price
            # Take-profit orders ABOVE current price are supposed to wait — don't nuke them
            bars_since = self.current_bar - exit_info.get("placed_bar", self.current_bar)
            if bars_since >= LIMIT_ORDER_TIMEOUT:
                exit_price = exit_info["exit_price"]
                current_price = exit_price
                if pair in market_data:
                    current_price = market_data[pair]["price"]
                
                # Is this a TP order waiting above/below current price?
                side = exit_info["side"]
                if side == "sell" and exit_price > current_price * 1.005:
                    # Sell limit ABOVE current price = take-profit, let it sit
                    logger.debug(f"[EXIT WAITING] {pair} TP sell @ ${exit_price:.4f} (current ${current_price:.4f}) — not stale, waiting for target")
                    continue
                elif side == "buy" and exit_price < current_price * 0.995:
                    # Buy limit BELOW current price = TP for a short, let it sit
                    logger.debug(f"[EXIT WAITING] {pair} TP buy @ ${exit_price:.4f} (current ${current_price:.4f}) — not stale, waiting for target")
                    continue
                
                # Exit is near current price but didn't fill — escalate to market
                logger.warning(f"[EXIT STALE] {pair} exit limit @ ${exit_price:.4f} unfilled after {bars_since} cycles — escalating to MARKET")
                
                if ENABLE_LIVE_TRADING:
                    # Cancel the stale limit
                    try:
                        if order_id:
                            self.client.cancel_order(order_id)
                    except Exception as e:
                        logger.error(f"Failed to cancel stale exit for {pair}: {e}")
                    
                    # Place market order
                    try:
                        qty = exit_info["qty"]
                        self.client.place_order(pair, side, "market", qty, 0)
                        logger.info(f"[EXIT MARKET] {pair} forced market exit")
                    except Exception as e:
                        logger.error(f"Failed to market-exit {pair}: {e}")
                        continue  # Don't finalize if market order also failed
                
                self._finalize_exit(pair, current_price, exit_info["reason"] + " (market fallback)")
    
    def execute_signal(self, signal: dict, score: float):
        """Execute a signal with UPGRADE 1 (limit orders) and UPGRADE 3 (2x margin)."""
        pair = signal['pair']
        direction = signal['direction']
        tool = signal['tool']
        
        # US retail accounts (Non-ECP) cannot open margin positions on Kraken
        # "Reduce only" restriction — shorts require margin which is blocked by SEC/CFTC rules
        if direction == 'short' and ENABLE_LIVE_TRADING:
            logger.debug(f"Skipping {tool} ({pair}) — short blocked (US Non-ECP margin restriction)")
            self._log_rejection(pair, tool, direction, score, "short_blocked_us_margin")
            return
        
        # Skip if we already have a position in this pair
        if pair in self.active_positions:
            return
        
        # UPGRADE 7: Skip if tool has consecutive losses
        if tool in self.tool_streaks:
            streak = self.tool_streaks[tool]
            if streak["type"] == "L":
                # Weaker tools get benched faster
                weak_tools = {'ema_cross_short', 'falling_wedge_short', 'distribution_short', 'entropy_short'}
                threshold = 3 if tool in weak_tools else 5
                if streak["streak"] >= threshold:
                    logger.warning(f"Skipping {tool} - {streak['streak']} consecutive losses")
                    return
        
        # Market regime filter: block weak shorts in strong bull market
        if direction == 'short' and tool not in {'mega_pump_sell_t1', 'mega_pump_sell_t2', 'rsi_pump_8h', 'rsi_pump_fat_tail'}:
            bullish_pct = getattr(self, '_bullish_4h_pct', 50)
            if bullish_pct >= 75 and score < 20:
                logger.warning(f"Skipping {tool} ({pair}) - weak short (score={score:.1f}) in strong bull regime ({bullish_pct:.0f}% bullish 4h)")
                return
        
        # Regime filter: bull tools only fire in bull/greed regimes
        bull_tools = {'accumulation_breakout', 'hurst_trend_long',
                      'buy_weekly_green', 'buy_breakout_simple',
                      'simple_buy_uptrend', 'buy_btc_leading'}
        if tool in bull_tools:
            fng = self.get_fng()
            bullish_pct = getattr(self, '_bullish_4h_pct', 50)
            if fng < 45 or bullish_pct < 50:
                logger.info(f"Skipping {tool} ({pair}) - bull tool blocked in fear/bear (F&G={fng}, {bullish_pct:.0f}% bullish)")
                return
        
        # UPGRADE 3: Determine leverage
        # Tier 1 tools get 2x, ALL shorts need leverage=2 (margin required to sell short)
        # US Non-ECP: margin restricted on Kraken. No leverage on any trades.
        leverage = 1
        
        # UPGRADE 9: Kelly Criterion position sizing
        # Uses historical win rate and avg win/loss ratio per tool to optimize bet size
        # Kelly fraction = WR - (1-WR)/payoff_ratio, capped at 2x base risk
        base_risk = RISK_PER_TRADE  # 5%
        if tool in self.tool_stats and self.tool_stats[tool].get('total', 0) >= 5:
            ts = self.tool_stats[tool]
            total = ts['total']
            win_rate = ts['wins'] / total if total > 0 else 0.5
            avg_win = ts.get('avg_win_pct', 0.05)
            avg_loss = abs(ts.get('avg_loss_pct', 0.03))
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
            
            kelly = win_rate - (1 - win_rate) / payoff_ratio if payoff_ratio > 0 else 0
            # Half-Kelly for safety, bounded between 0.5x and 2x base risk
            kelly_fraction = max(0.5, min(2.0, 1.0 + kelly))
            risk_pct = base_risk * kelly_fraction
            logger.debug(f"Kelly sizing for {tool}: WR={win_rate:.0%}, payoff={payoff_ratio:.1f}, kelly_f={kelly_fraction:.2f}")
        else:
            risk_pct = base_risk
        
        risk_amount = self.active_balance * risk_pct
        stop_loss_pct = signal['sl_pct']
        
        if direction == 'long':
            position_size = risk_amount / stop_loss_pct
        else:  # short
            position_size = risk_amount / stop_loss_pct
        
        # EXTREME FEAR BOOST: Double down on crash buys when F&G < 15
        fear_boost_tools = {'crash_buy', 'mega_crash', 'blood_in_streets', 'volatile_oversold',
                           'crash_neg_ac', 'crash_mean_revert', 'quick_crash', 'flash_crash',
                           'market_panic_70', 'deep_dip_8h'}
        if tool in fear_boost_tools and getattr(self, 'current_fng', 50) < 15:
            position_size *= 1.5  # 50% bigger in extreme fear
            logger.info(f"[FEAR BOOST] {tool} {pair} — 1.5x size in extreme fear (F&G={self.current_fng})")
        
        # Apply leverage to position sizing (controls 2x notional with same risk)
        if leverage == 2:
            # Position size stays the same but controls 2x the notional
            pass
        
        # Don't risk more than available balance
        position_size = min(position_size, self.active_balance * 0.8)
        
        # Cap position at 1% of pair's 24h volume (liquidity guard)
        if hasattr(self, '_pair_volatility') and pair in self._pair_volatility:
            max_pos = self._pair_volatility[pair].get('max_position_usd', float('inf'))
            if position_size > max_pos:
                logger.info(f"[LIQUIDITY CAP] {pair} capped ${position_size:.2f} → ${max_pos:.2f} (1% of 24h vol)")
                position_size = max_pos
        
        # Check actual USD on Kraken before placing orders
        if ENABLE_LIVE_TRADING and position_size > 1.0:
            try:
                bal = self.client.get_account_balance()
                actual_usd = bal.get('USD', 0) + bal.get('ZUSD', 0)
                if position_size > actual_usd * 0.9:
                    position_size = actual_usd * 0.8
                    if position_size < 5.0:  # Below Kraken minimums
                        logger.warning(f"Insufficient USD (${actual_usd:.2f}) for {pair}, skipping")
                        self._log_rejection(pair, tool, direction, score, "insufficient_usd")
                        return
            except Exception:
                pass
        
        # Get current price and bid/ask
        market_data = self.get_market_data()
        if pair not in market_data:
            logger.warning(f"No market data for {pair}, skipping signal")
            self._log_rejection(pair, tool, direction, score, "no_market_data")
            return
            
        data = market_data[pair]
        current_price = data["price"]
        
        # Hard price floor — never trade sub-penny coins regardless of scanner
        if current_price < MIN_PAIR_PRICE_USD:
            logger.warning(f"[PRICE GUARD] {pair} @ ${current_price:.6f} below ${MIN_PAIR_PRICE_USD} minimum, skipping")
            self._log_rejection(pair, tool, direction, score, f"price_too_low_{current_price:.6f}")
            return
        
        # UPGRADE 1: Use LIMIT orders for entry (maker fees)
        if direction == 'long':
            entry_price = data.get("bid", current_price)  # Buy at bid for better entry
        else:
            entry_price = data.get("ask", current_price)  # Sell at ask for better entry
        
        qty = position_size / entry_price
        
        # UPGRADE 3: Add margin opening cost for leveraged positions
        margin_opening_cost = 0
        if leverage == 2:
            margin_opening_cost = position_size * MARGIN_COST_OPEN
            if self.active_balance < margin_opening_cost:
                logger.warning(f"Insufficient balance for margin opening cost: {pair}")
                self._log_rejection(pair, tool, direction, score, "insufficient_margin_balance")
                return
            self.active_balance -= margin_opening_cost
        
        # Execute LIMIT order for entry
        if ENABLE_LIVE_TRADING:
            try:
                side = "buy" if direction == 'long' else "sell"
                # Pass leverage to Kraken (2x for shorts and Tier 1 longs)
                if leverage >= 2:
                    order_id = self.client.place_order(pair, side, "limit", qty, entry_price, leverage=leverage)
                else:
                    order_id = self.client.place_order(pair, side, "limit", qty, entry_price)
                
                # Check if order actually placed
                if order_id is None:
                    logger.warning(f"Order failed for {pair} — no order_id returned, aborting position")
                    if leverage == 2:
                        self.active_balance += margin_opening_cost
                    return
                
                # Track pending limit order
                self.pending_limit_orders[pair] = {
                    "direction": direction,
                    "qty": qty,
                    "price": entry_price,
                    "original_price": entry_price,
                    "original_score": score,
                    "placed_bar": self.current_bar,
                    "order_id": order_id,
                    "tool": tool,
                    "retries": 0
                }
                
            except Exception as e:
                logger.error(f"Failed to execute {direction} limit order for {pair}: {e}")
                if leverage == 2:
                    self.active_balance += margin_opening_cost
                return
        else:
            leverage_str = " (2x margin)" if leverage == 2 else ""
            logger.info(f"[DRY RUN] {direction.upper()} {pair} LIMIT @ ${entry_price:.4f}{leverage_str}")
        
        # Create position record
        position = {
            'pair': pair,
            'tool': tool,
            'direction': direction,
            'leverage': leverage,
            'entry_price': entry_price,
            'entry_bar': self.current_bar,
            'entry_time': datetime.now(timezone.utc).timestamp(),
            'position_size': position_size,
            'qty': qty,
            'sl_pct': stop_loss_pct,
            'hold': signal['hold'],
            'score': score,
            'total_margin_cost': margin_opening_cost
        }
        
        self.active_positions[pair] = position
        self.active_balance -= position_size  # Reserve capital
        
        # Place TP order on Kraken immediately so it's waiting on the book
        exit_mode, take_profit_pct, trailing_stop_pct, _ = self._get_exit_params(tool, entry_price, {})
        if take_profit_pct and ENABLE_LIVE_TRADING:
            try:
                if direction == 'long':
                    tp_price = entry_price * (1 + take_profit_pct)
                    tp_side = "sell"
                else:
                    tp_price = entry_price * (1 - take_profit_pct)
                    tp_side = "buy"
                # Place TP for 50% of position (partial TP)
                tp_qty = qty * 0.5
                tp_result = self.client.place_order(pair, tp_side, "limit", tp_qty, tp_price)
                if tp_result:
                    position['_tp_order_id'] = tp_result.get('txid', [None])[0] if isinstance(tp_result, dict) else tp_result
                    position['_tp_price'] = tp_price
                    position['_tp_qty'] = tp_qty
                    logger.info(f"[TP PLACED] {pair} {tp_side} {tp_qty:.4f} @ ${tp_price:.4f} ({take_profit_pct:.0%} TP)")
            except Exception as e:
                logger.warning(f"Failed to place TP order for {pair}: {e}")
        
        # Update daily stats
        self._daily_stats["trades_opened"] += 1
        
        leverage_str = " (2x margin)" if leverage == 2 else ""
        margin_str = f", margin cost: ${margin_opening_cost:.2f}" if leverage == 2 else ""
        
        logger.info(f"[OPEN] {pair} {direction} LIMIT @ ${entry_price:.4f}{leverage_str} | "
                   f"Tool: {tool} | Size: ${position_size:.2f}{margin_str} | "
                   f"Score: {score:.1f} | SL: {stop_loss_pct:.1%}")
        
        # Journal: log open with full context
        htf_ctx = signal.get('_htf_context', {})
        mtf_m = self._compute_mtf_multiplier(tool, direction, htf_ctx)
        base_score = score / mtf_m if mtf_m != 0 else score
        self._journal_open(
            pair=pair, tool=tool, direction=direction, price=entry_price,
            score=score, base_score=base_score,
            mtf_multiplier=mtf_m,
            htf_context=htf_ctx, leverage=leverage,
            position_size=position_size, sl_pct=stop_loss_pct,
            hold_bars=signal['hold'], reason=signal.get('reason', '')
        )
    
    def get_fear_greed(self) -> int:
        """Get crypto Fear & Greed Index. Cached for 1 hour."""
        now = datetime.now(timezone.utc).timestamp()
        if hasattr(self, '_fng_cache') and now - self._fng_cache_ts < 3600:
            return self._fng_cache
        try:
            r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            val = int(r.json()['data'][0]['value'])
            self._fng_cache = val
            self._fng_cache_ts = now
            return val
        except:
            return 50  # Neutral on error
    
    def get_fng(self) -> int:
        """Alias for get_fear_greed()."""
        return self.get_fear_greed()
    
    def _manage_idle_staking(self):
        """Stake idle USD into yield-bearing assets when no positions are open.
        
        Strategy: When active_balance > 80% of total (mostly idle), stake into
        instant-unlock assets. When a signal fires, unstake to free capital.
        
        NOTE: Requires Earn permission on API key. Set ENABLE_STAKING=true when ready.
        """
        if not ENABLE_LIVE_TRADING or not os.getenv('ENABLE_STAKING', 'false').lower() == 'true':
            return
        
        idle_pct = self.active_balance / self.total_balance if self.total_balance > 0 else 0
        active_count = len([p for p in self.active_positions if not self.active_positions[p].get('_pending_exit')])
        
        # Stake when mostly idle (>80% available and <2 positions)
        if idle_pct > 0.8 and active_count < 2:
            stake_amount = self.active_balance * 0.5  # Stake 50% of idle
            if stake_amount < 10:
                return
            
            # Prefer USDG (stablecoin yield) for no price risk
            try:
                # USDG instant strategy
                result = self.client._request('private/Earn/Allocate', {
                    'strategy_id': 'ESDE4BG-5NNGK-5WTK6X',  # USDG instant
                    'amount': str(stake_amount)
                }, private=True)
                if result:
                    logger.info(f"[STAKE] Allocated ${stake_amount:.2f} to USDG earn")
                    self._staked_amount = getattr(self, '_staked_amount', 0) + stake_amount
            except Exception as e:
                logger.debug(f"Staking failed (need Earn API permission): {e}")
        
        # Unstake when we need capital (positions filling up or new signals)
        elif active_count >= 3 and getattr(self, '_staked_amount', 0) > 0:
            try:
                result = self.client._request('private/Earn/Deallocate', {
                    'strategy_id': 'ESDE4BG-5NNGK-5WTK6X',
                    'amount': str(self._staked_amount)
                }, private=True)
                if result:
                    logger.info(f"[UNSTAKE] Deallocated ${self._staked_amount:.2f} from USDG earn")
                    self._staked_amount = 0
            except Exception as e:
                logger.debug(f"Unstaking failed: {e}")
    
    def run_cycle(self):
        """Run one complete trading cycle with all upgrades."""
        try:
            # Process any real-time signals from websocket (crashes, pumps, volume)
            self.process_realtime_signals()
            
            # UPGRADE 8: Enhanced status logging
            logger.info("═" * 80)
            self.current_bar += 1
            
            # Check for daily rollover and log previous day's summary
            self._check_daily_rollover()
            
            # 1. Get market data
            market_data = self.get_market_data()
            if not market_data:
                logger.warning("No market data, skipping cycle")
                return
            
            # 2. Update Fear & Greed index
            fng = self.get_fear_greed()
            self.current_fng = fng
            regime_label = "Extreme Fear" if fng < 20 else "Fear" if fng < 30 else "Neutral" if fng <= 70 else "Greed" if fng <= 80 else "Extreme Greed"
            
            # 3. UPGRADE 4: Rebalance capital allocation
            self.rebalance_capital()
            grid_pct, active_pct = self.get_capital_allocation()
            
            # 4. UPGRADE 5: Update total balance from Kraken (fall back to internal)
            kraken_balance = self._sync_kraken_balance()
            if kraken_balance is not None:
                self.total_balance = kraken_balance
            else:
                self.total_balance = self.starting_balance + self.grid_profit + self.active_profit
            
            # 5. UPGRADE 6: Update smart grid engine
            self.update_grids(market_data)
            
            # Refresh dynamic pair selection (hourly)
            self._refresh_volatile_pairs()
            
            # 6. Scan for signals FIRST (needed for opportunity cost checks)
            all_signals = []
            for pair, data in market_data.items():
                try:
                    signals = self.scan_signals(pair, data)
                    all_signals.extend(signals)
                except Exception as e:
                    logger.debug(f"Signal scan failed for {pair}: {e}")
                    continue
            self._current_cycle_signals = all_signals
            
            # 7. UPGRADE 1: Manage pending limit orders (can now compare vs fresh signals)
            self.manage_pending_limit_orders(market_data)
            
            # 7b. Check pending EXIT orders (confirm fills or escalate to market)
            self.manage_pending_exit_orders(market_data)
            
            # 8. Manage existing active positions
            self.manage_positions(market_data)
            
            # 8b. SIGNAL STACKING: boost score when multiple tools agree on same pair+direction
            stacked = {}  # (pair, direction) -> [list of (signal, score)]
            for signal, score in all_signals:
                key = (signal['pair'], signal['direction'])
                if key not in stacked:
                    stacked[key] = []
                stacked[key].append((signal, score))
            
            # Rebuild all_signals with stacking boost
            stacked_signals = []
            for (pair, direction), entries in stacked.items():
                # Use the highest-scoring signal as the base
                entries.sort(key=lambda x: x[1], reverse=True)
                best_signal, best_score = entries[0]
                tool_count = len(entries)
                
                if tool_count >= 3:
                    stack_mult = 1.6
                elif tool_count >= 2:
                    stack_mult = 1.3
                else:
                    stack_mult = 1.0
                
                boosted_score = best_score * stack_mult
                
                if tool_count > 1:
                    tools_str = "+".join([e[0]['tool'] for e in entries])
                    best_signal['_stacked_tools'] = tools_str
                    best_signal['_stack_count'] = tool_count
                    logger.info(f"[STACK] {pair} {direction}: {tool_count} tools ({tools_str}) → "
                               f"score {best_score:.1f} × {stack_mult} = {boosted_score:.1f}")
                
                stacked_signals.append((best_signal, boosted_score))
            
            all_signals = stacked_signals
            
            # 9. Filter and score signals with correlation-aware limits
            # Correlated asset groups — max 2 positions per group to limit concentrated risk
            CORRELATION_GROUPS = {
                'large_cap': {'XBTUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD'},
                'alt_l1': {'ADAUSD', 'AVAXUSD', 'DOTUSD', 'NEARUSD', 'ATOMUSD', 'APTUSD', 'SUIUSD', 'ICPUSD'},
                'defi': {'LINKUSD', 'UNIUSD', 'AAVEUSD', 'LDOUSD', 'JUPUSD'},
                'meme': {'DOGEUSD', 'SHIBUSD', 'PEPEUSD', 'FLOKIUSD'},
                'mid_cap': {'XRPUSD', 'LTCUSD', 'BCHUSD', 'FILUSD', 'XLMUSD', 'TRXUSD', 'STXUSD',
                           'HBARUSD', 'ARBUSD', 'OPUSD', 'TIAUSD', 'ONDOUSD', 'RENDERUSD',
                           'ENAUSD', 'HYPEUSD', 'TAOUSD', 'INJUSD', 'KAVAUSD'},
            }
            MAX_PER_GROUP = 3  # Max same-direction positions in a correlated group
            
            if all_signals:
                all_signals.sort(key=lambda x: x[1], reverse=True)
                
                open_positions = len(self.active_positions)
                for signal, score in all_signals:
                    if open_positions >= MAX_ACTIVE_POSITIONS:
                        # Log all remaining signals as rejected
                        for rem_signal, rem_score in all_signals[all_signals.index((signal, score)):]:
                            self._log_rejection(rem_signal['pair'], rem_signal['tool'], 
                                              rem_signal['direction'], rem_score, "max_positions_reached")
                        break
                    
                    pair = signal['pair']
                    if pair in self.active_positions:
                        self._log_rejection(pair, signal['tool'], signal['direction'], score, "pair_already_open")
                        continue
                    
                    # Check correlation group limits
                    direction = signal['direction']
                    group_count = 0
                    for group_name, group_pairs in CORRELATION_GROUPS.items():
                        if pair in group_pairs:
                            for open_pair, open_pos in self.active_positions.items():
                                if open_pair in group_pairs and open_pos['direction'] == direction:
                                    group_count += 1
                            break
                    
                    if group_count >= MAX_PER_GROUP:
                        logger.debug(f"Skipping {pair} ({direction}) — correlation group limit ({group_count}/{MAX_PER_GROUP})")
                        self._log_rejection(pair, signal['tool'], signal['direction'], score, "correlation_group_limit")
                        continue
                    
                    self.execute_signal(signal, score)
                    open_positions += 1
            
            # 10. UPGRADE 8: Enhanced status report
            grid_positions = sum(len(positions) for positions in self.grid_positions.values())
            active_count = len(self.active_positions)
            growth_pct = (self.total_balance / self.starting_balance - 1) * 100
            
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CYCLE #{self.current_bar} | "
                       f"F&G: {fng} ({regime_label}) | Allocation: Grid {grid_pct:.0%} / Active {active_pct:.0%}")
            
            logger.info(f"Balance: ${self.total_balance:.2f} (start: ${self.starting_balance:.2f}, {growth_pct:+.1f}%) | "
                       f"Grid: ${self.grid_balance:.2f} | Active: ${self.active_balance:.2f}")
            
            logger.info(f"Grid: {grid_positions} positions across {len(PAIRS)} pairs | "
                       f"{self.grid_round_trips} round trips | ${self.grid_profit:.2f} profit")
            
            logger.info(f"Active: {active_count}/{MAX_ACTIVE_POSITIONS} positions open")
            
            if self.active_positions:
                for pair, pos in self.active_positions.items():
                    current_price = market_data[pair]["price"]
                    if pos['direction'] == 'long':
                        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                    else:
                        pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                    
                    # Apply leverage multiplier for display
                    if pos.get("leverage", 1) == 2:
                        pnl_pct *= 2
                    
                    entry_time = pos.get('entry_time', None)
                    if entry_time:
                        hours_held = (datetime.now(timezone.utc).timestamp() - entry_time) / 3600
                    else:
                        hours_held = (self.current_bar - pos['entry_bar']) * (CHECK_INTERVAL / 3600)
                    leverage_str = ", 2x margin" if pos.get("leverage", 1) == 2 else ""
                    logger.info(f"  → {pair} {pos['direction']} {pnl_pct:+.1%} "
                               f"({pos['tool']}, {hours_held:.1f}h held{leverage_str})")
            
            if all_signals:
                top_signals = all_signals[:3]  # Show top 3
                signal_str = ", ".join([f"{s[0]['tool']} {s[0]['pair']} (score {s[1]:.1f})" 
                                      for s in top_signals])
                logger.info(f"Signals this cycle: {signal_str}")
            
            # MTF: Show multi-timeframe regime across all pairs
            if ENABLE_MTF:
                htf_regimes = []
                htf_rsi_values = []
                
                for pair, data in market_data.items():
                    htf_context = self.get_htf_context(data)
                    if htf_context.get("htf_available", False):
                        htf_regimes.append(htf_context["trend_4h"])
                        htf_rsi_values.append(htf_context["rsi_4h"])
                
                if htf_regimes:
                    regime_counts = {}
                    for regime in htf_regimes:
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1
                    
                    total_pairs = len(htf_regimes)
                    regime_pcts = []
                    for regime in ["bearish", "neutral", "bullish"]:
                        count = regime_counts.get(regime, 0)
                        pct = count / total_pairs * 100
                        if pct > 0:
                            regime_pcts.append(f"{pct:.0f}% {regime} 4h")
                    
                    avg_rsi_4h = sum(htf_rsi_values) / len(htf_rsi_values)
                    mtf_str = ", ".join(regime_pcts) + f" | Avg 4h RSI: {avg_rsi_4h:.1f}"
                    logger.info(f"MTF: {mtf_str}")
                    # Track bullish percentage for regime filter
                    self._bullish_4h_pct = regime_counts.get("bullish", 0) / total_pairs * 100
            
            # UPGRADE 7: Show tool streaks
            hot_tools = []
            cold_tools = []
            for tool, streak in self.tool_streaks.items():
                if streak["type"] == "W" and streak["streak"] >= 3:
                    hot_tools.append(f"{tool} W{streak['streak']} 🔥")
                elif streak["type"] == "L" and streak["streak"] >= 3:
                    cold_tools.append(f"{tool} L{streak['streak']}")
            
            if hot_tools or cold_tools:
                streak_info = " | ".join(hot_tools + cold_tools)
                logger.info(f"Tool streaks: {streak_info}")
            
            # 11. Manage idle capital staking
            self._manage_idle_staking()
            
            # 12. Save state
            self._log_balance_snapshot()
            self.save_state()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 FINAL TRADING BOT STARTING - THE ULTIMATE VERSION...")
        logger.info(f"Live trading: {ENABLE_LIVE_TRADING}")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        logger.info(f"Trading pairs: {len(PAIRS)} (expanded from 16)")
        logger.info(f"Tier 1 tools with 2x margin: {len(TIER1_TOOLS)}")
        
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
                    logger.info("═" * 80)
                    # Sleep in 10s chunks, checking for crash signals between
                    remaining = sleep_time
                    while remaining > 0 and self.running:
                        time.sleep(min(10, remaining))
                        remaining -= 10
                        # Process any real-time signals during sleep
                        has_signals = (self.pending_crash_signals or 
                                      self.pending_pump_signals or 
                                      self.pending_volume_spikes)
                        if has_signals:
                            logger.info(f"🔌 Real-time event during sleep! Processing...")
                            self.process_realtime_signals()
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
            logger.info("🚀 FINAL TRADING BOT STOPPED")


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Trading Bot - Ultimate Version')
    parser.add_argument('--dry-run', action='store_true', help='Force dry run mode')
    parser.add_argument('--check', action='store_true', help='Quick sanity check and exit')
    args = parser.parse_args()
    
    if args.dry_run:
        os.environ['ENABLE_LIVE_TRADING'] = 'false'
        logger.info("Forced dry run mode")
    
    bot = FinalTradingBot()
    
    if args.check:
        logger.info("✅ FINAL Bot initialized successfully")
        logger.info(f"✅ Found {len(VALIDATED_TOOLS)} validated tools")
        logger.info(f"✅ Trading {len(PAIRS)} pairs (up from 16)")
        logger.info(f"✅ Tier 1 tools with 2x margin: {len(TIER1_TOOLS)}")
        logger.info(f"✅ Dynamic allocation based on F&G: {bot.current_fng}")
        logger.info(f"✅ Kraken Pro fees: {ENTRY_FEE:.3%} entry, {EXIT_FEE:.3%} exit")
        logger.info("✅ All 8 UPGRADES implemented")
        return
    
    bot.run()


if __name__ == "__main__":
    main()