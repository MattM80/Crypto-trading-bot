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
import csv
import json
import html
import re
import time
import signal
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Add src directory for kraken_client
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from kraken_client import KrakenClient
    from ws_monitor import CrashMonitor
    from meta_signal_model import SignalMetaModel
except ImportError as e:
    logger.error(f"Failed to import KrakenClient: {e}")
    sys.exit(1)

try:
    from asset_context import AssetContextGuard
    _ASSET_CONTEXT_AVAILABLE = True
except ImportError:
    AssetContextGuard = None  # type: ignore
    _ASSET_CONTEXT_AVAILABLE = False

# Kraken Futures (Phase 1 scaffolding — disabled for live orders by default).
# Importing is optional so the bot still runs if the futures module isn't set up.
try:
    from kraken_futures_client import KrakenFuturesClient, to_futures_symbol  # type: ignore
    _FUTURES_CLIENT_AVAILABLE = True
except ImportError:
    KrakenFuturesClient = None  # type: ignore
    to_futures_symbol = None    # type: ignore
    _FUTURES_CLIENT_AVAILABLE = False

# NinjaTrader signal export (append-only JSONL writer). Safe if missing.
try:
    from nt_signal_export import export_signal as _nt_export_signal  # type: ignore
    _NT_EXPORT_AVAILABLE = True
except ImportError:
    _nt_export_signal = None  # type: ignore
    _NT_EXPORT_AVAILABLE = False

# Configuration
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
# Kraken spot-only mode is the production default. Short detectors can still
# inform risk, but they cannot route to futures or outside execution unless the
# user explicitly disables spot-only mode and opts into an export path.
SPOT_ONLY_MODE = os.getenv("SPOT_ONLY_MODE", "true").lower() == "true"
# Futures trading for SHORTS. Kept as dormant scaffolding, but hard-disabled in
# spot-only mode because this account cannot trade futures.
ENABLE_FUTURES_TRADING = (
    os.getenv("ENABLE_FUTURES_TRADING", "false").lower() == "true" and
    not SPOT_ONLY_MODE
)
ENABLE_EXTERNAL_SIGNAL_EXPORT = (
    os.getenv("ENABLE_EXTERNAL_SIGNAL_EXPORT", "false").lower() == "true" and
    not SPOT_ONLY_MODE
)
FUTURES_SHORT_MAX_NOTIONAL_USD = float(os.getenv("FUTURES_SHORT_MAX_NOTIONAL_USD", "25"))
FUTURES_SHORT_MAX_LEVERAGE = float(os.getenv("FUTURES_SHORT_MAX_LEVERAGE", "2"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes
FNG_CACHE_TTL_SEC = int(os.getenv("FNG_CACHE_TTL_SEC", os.getenv("FNG_CACHE_SECONDS", "300")))
FNG_PROVIDER = os.getenv("FNG_PROVIDER", "coinmarketcap").strip().lower()
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "").strip()
CMC_FNG_API_URL = os.getenv("CMC_FNG_API_URL", "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical")
CMC_FNG_PAGE_URL = os.getenv("CMC_FNG_PAGE_URL", "https://coinmarketcap.com/charts/fear-and-greed-index/")
FNG_MAX_SOURCE_AGE_HOURS = float(os.getenv("FNG_MAX_SOURCE_AGE_HOURS", "36"))
FNG_LAST_GOOD_MAX_AGE_HOURS = float(os.getenv("FNG_LAST_GOOD_MAX_AGE_HOURS", "6"))
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
MIN_PAIR_VOLUME_USD = 5_000_000  # Min $5M 24h volume — real liquidity only
MIN_PAIR_PRICE_USD = 0.50       # No sub-dollar coins — spread/slippage/microcap kills you
MAX_POSITION_PCT_OF_VOLUME = 0.005  # Never be more than 0.5% of daily volume

# Coins restricted for US:FL, known rugs, naming collisions, or proven losers (microcap/illiquid)
# Pair-specific quarantines live in PAIR_POLICIES below.
GEO_BLOCKED_PAIRS = {'BLUAIUSD', 'B3USD', 'GUSD', 'DRIFTUSD', 'NIGHTUSD', 'PTBUSD', 'GHSTUSD'}

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

# Kraken pair name normalization (API key → standard name)
KRAKEN_PAIR_MAP = {
    'XETHZUSD': 'ETHUSD', 'XXBTZUSD': 'XBTUSD', 'XXLMZUSD': 'XLMUSD',
    'XXRPZUSD': 'XRPUSD', 'XLTCZUSD': 'LTCUSD', 'XXMRZUSD': 'XMRUSD',
    'XZECZUSD': 'ZECUSD', 'XETCZUSD': 'ETCUSD', 'XMLNZUSD': 'MLNUSD',
    'XREPZUSD': 'REPUSD', 'USDTZUSD': 'USDTUSD',
}
KRAKEN_PAIR_MAP_REVERSE = {v: k for k, v in KRAKEN_PAIR_MAP.items()}

def normalize_pair(pair: str) -> str:
    """Normalize Kraken pair names to standard format."""
    return KRAKEN_PAIR_MAP.get(pair, pair)

# Kraken asset → pair mapping for balance reconciliation
KRAKEN_ASSET_TO_PAIR = {
    'XETH': 'ETHUSD', 'XXBT': 'XBTUSD', 'XXLM': 'XLMUSD',
    'XXRP': 'XRPUSD', 'XLTC': 'LTCUSD', 'XXMR': 'XMRUSD',
    'XZEC': 'ZECUSD', 'XETC': 'ETCUSD', 'XREP': 'REPUSD',
}

# Constants
# Validation mode is the default for the small-account phase. It keeps the bot
# from spreading a $300 account across too many low-conviction positions while
# we collect clean forward stats. Set VALIDATION_ACCOUNT_MODE=false after the
# system proves positive EV and the account is ready to scale.
VALIDATION_ACCOUNT_MODE = os.getenv("VALIDATION_ACCOUNT_MODE", "true").lower() == "true"
MAX_ACTIVE_POSITIONS = int(os.getenv("MAX_ACTIVE_POSITIONS", "4" if VALIDATION_ACCOUNT_MODE else "8"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.04" if VALIDATION_ACCOUNT_MODE else "0.08"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10" if VALIDATION_ACCOUNT_MODE else "0.12"))
STOP_LOSS_COOLDOWN_SEC = 6 * 3600  # 6 hour cooldown on a pair after stop loss
STOP_LOSS_COOLDOWN_HARD_MIN = 3 * 3600  # Minimum 3 hour hard cooldown (no early lift)
MAX_STOP_LOSSES_PER_PAIR_PER_DAY = 1  # Max stop-outs on same pair per day before 24h blacklist
DAILY_MAX_LOSS_PCT = 0.03   # Stop trading after 3% daily drawdown (tightened 2026-04-23)
MAX_TOTAL_DRAWDOWN_PCT = 0.25  # Circuit breaker: stop ALL trading if account drops 25% from starting balance
MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.55" if VALIDATION_ACCOUNT_MODE else "0.80"))
GRID_REANCHOR_PCT = 0.10    # Reanchor grid when price moves >10% from center
LIMIT_ORDER_TIMEOUT = 2     # Cancel and re-place limit orders after 2 cycles (10 min)
MAX_LIMIT_RETRIES = 3       # Max re-places before giving up (3 retries = ~15 min total)
PRICE_DRIFT_ABANDON = 0.02  # Abandon pending order if price drifted >2% from original entry
MAX_TRADES_PER_PAIR_PER_DAY = 3   # Max entries on same pair per day (prevents RAVE-type over-concentration)
VOLATILE_RUNNER_MIN_VOL = 0.20  # Convert the second half into a runner on very volatile names
VOLATILE_RUNNER_TIGHT_TRAIL = 0.06
VOLATILE_RUNNER_WIDE_TRAIL = 0.08
VOL_QUALITY_RANGE_FLOOR = 0.35  # Avoid long-only scans on names still pinned near 24h lows
VOL_QUALITY_COLLAPSE_ATR_PCT = 12.0
VOL_QUALITY_REBOUND_1H_PCT = 0.5
VOL_QUALITY_CRASH_24H_PCT = -18.0
VOL_QUALITY_CRASH_8H_PCT = -9.0
SHORT_PRESSURE_TIGHTEN_SCORE = 14.0
SHORT_PRESSURE_CANCEL_SCORE = 16.0
SHORT_PRESSURE_ENTRY_BLOCK_SCORE = 18.0
SHORT_PRESSURE_EXIT_SCORE = 22.0
CONVICTION_DECAY_MIN_HOLD_FRACTION = 0.35
CONVICTION_DECAY_MIN_HOURS = 3.0
CONVICTION_DECAY_CHECK_BARS = 6
CONVICTION_DECAY_ENTRY_SCORE_KEEP_FRACTION = 0.75
CONVICTION_DECAY_SAFE_PNL_PCT = 0.01
CONVICTION_DECAY_EXIT_PNL_PCT = -0.01
FORWARD_TOOL_WINDOW_TRADES = 6
FORWARD_TOOL_MIN_TRADES = 3
FORWARD_TOOL_QUARANTINE_MIN_TRADES = 4
FORWARD_TOOL_BAD_WIN_RATE = 0.34
FORWARD_TOOL_SOFT_MULTIPLIER = 0.65
FORWARD_TOOL_SOFT_AVG_PNL_PCT = -0.005
FORWARD_TOOL_QUARANTINE_AVG_PNL_PCT = -0.01
FORWARD_TOOL_STRICT_VALIDATION = os.getenv("FORWARD_TOOL_STRICT_VALIDATION", "true" if VALIDATION_ACCOUNT_MODE else "false").lower() == "true"
FORWARD_TOOL_VALIDATION_BLOCK_AVG_PNL_PCT = float(os.getenv("FORWARD_TOOL_VALIDATION_BLOCK_AVG_PNL_PCT", "-0.005"))
CONTEXT_PRIOR_STRENGTH = 6.0
CONTEXT_EV_MULTIPLIER = 6.0
CONTEXT_MIN_MULT = 0.65
CONTEXT_MAX_MULT = 1.35
MAX_TREND_LEADER_RESERVED_SLOTS = 2
REPLACEMENT_SCORE_EDGE = 1.35
TREND_REPLACEMENT_SCORE_EDGE = 1.15
REPLACEMENT_PROTECT_PNL_PCT = 0.02
REPLACEMENT_MIN_SCORE = 12.0
MARKET_BEAR_CAUTION_PAIRS = 3
MARKET_BEAR_DEFENSIVE_PAIRS = 5
MARKET_BEAR_RISK_OFF_PAIRS = 7
MARKET_BEAR_CAUTION_SCORE = 12.0
MARKET_BEAR_DEFENSIVE_SCORE = 15.0
MARKET_BEAR_RISK_OFF_SCORE = 18.0
MARKET_BEAR_CAUTION_DOMINANCE = 0.85
MARKET_BEAR_DEFENSIVE_DOMINANCE = 1.10
MARKET_BEAR_RISK_OFF_DOMINANCE = 1.35
MARKET_BEAR_CAUTION_ACTIVE_PCT = 0.85
MARKET_BEAR_DEFENSIVE_ACTIVE_PCT = 0.65
MARKET_BEAR_RISK_OFF_ACTIVE_PCT = 0.45
BULL_OFFENSE_MIN_FNG = 55
BULL_OFFENSE_MIN_BULLISH_PCT = 65
BULL_OFFENSE_MAX_SHORT_DOMINANCE = 0.85
BULL_OFFENSE_MAX_SHORT_PAIRS = 2
BULL_OFFENSE_MIN_SCORE = 14.0
BULL_OFFENSE_SIZE_MULT = float(os.getenv("BULL_OFFENSE_SIZE_MULT", "1.05" if VALIDATION_ACCOUNT_MODE else "1.15"))
BULL_OFFENSE_TOTAL_EXPOSURE_PCT = float(os.getenv("BULL_OFFENSE_TOTAL_EXPOSURE_PCT", "0.65" if VALIDATION_ACCOUNT_MODE else "0.90"))
BULL_OFFENSE_POSITION_PCT = float(os.getenv("BULL_OFFENSE_POSITION_PCT", "0.12" if VALIDATION_ACCOUNT_MODE else "0.25"))
MAJOR_BREAKOUT_PAIRS = {'XBTUSD', 'ETHUSD', 'XRPUSD'}
MAJOR_BREAKOUT_MIN_FNG = 32
MAJOR_BREAKOUT_MIN_BULLISH_PCT = 58
MAJOR_BREAKOUT_MIN_VOLUME_RATIO = 1.8
MAJOR_BREAKOUT_MIN_BREAKOUT_PCT = 0.004
MAJOR_BREAKOUT_MIN_SCORE = 12.0
MAJOR_BREAKOUT_MIN_BREADTH = 0.60
MAJOR_BREAKOUT_MAX_SHORT_DOMINANCE = 1.15
PANIC_REVERSAL_DROP_24H = -8.0
PANIC_REVERSAL_MIN_VOLUME_RATIO = 1.5
PANIC_REVERSAL_MIN_LOWER_WICK_RATIO = 0.45
PANIC_REVERSAL_MAX_RSI = 30.0
PANIC_REVERSAL_BTC_CRASH_FLOOR = -8.0
MARKET_BREADTH_RECOVERY_ENABLED = os.getenv("MARKET_BREADTH_RECOVERY_ENABLED", "true").lower() == "true"
MARKET_BREADTH_RECOVERY_MIN_BULLISH_PCT = float(os.getenv("MARKET_BREADTH_RECOVERY_MIN_BULLISH_PCT", "55"))
MARKET_BREADTH_RECOVERY_MAX_SHORT_DOMINANCE = float(os.getenv("MARKET_BREADTH_RECOVERY_MAX_SHORT_DOMINANCE", "1.15"))
MARKET_BREADTH_RECOVERY_MIN_RET_4H = float(os.getenv("MARKET_BREADTH_RECOVERY_MIN_RET_4H", "0.6"))
MARKET_BREADTH_RECOVERY_MIN_RET_24H = float(os.getenv("MARKET_BREADTH_RECOVERY_MIN_RET_24H", "1.2"))
MARKET_BREADTH_RECOVERY_MIN_VOLUME_RATIO = float(os.getenv("MARKET_BREADTH_RECOVERY_MIN_VOLUME_RATIO", "1.05"))
MARKET_BREADTH_RECOVERY_MAX_RSI = float(os.getenv("MARKET_BREADTH_RECOVERY_MAX_RSI", "72"))
MARKET_BREADTH_RECOVERY_MIN_RANGE_POS = float(os.getenv("MARKET_BREADTH_RECOVERY_MIN_RANGE_POS", "0.55"))
MARKET_BREADTH_RECOVERY_BTC_MIN_RET_24H = float(os.getenv("MARKET_BREADTH_RECOVERY_BTC_MIN_RET_24H", "0.4"))
EVIDENCE_MODE = os.getenv("EVIDENCE_MODE", "true").lower() == "true"
EVIDENCE_MIN_NET_EDGE_PCT = float(os.getenv("EVIDENCE_MIN_NET_EDGE_PCT", "0.0025"))
EVIDENCE_LIVE_SOFT_MIN_TRADES = int(os.getenv("EVIDENCE_LIVE_SOFT_MIN_TRADES", "3"))
EVIDENCE_LIVE_HARD_MIN_TRADES = int(os.getenv("EVIDENCE_LIVE_HARD_MIN_TRADES", "8"))
EVIDENCE_LIVE_KILL_DOLLAR_LOSS = float(os.getenv("EVIDENCE_LIVE_KILL_DOLLAR_LOSS", "-8.0"))
AUTONOMOUS_PROOF_LADDER_ENABLED = os.getenv("AUTONOMOUS_PROOF_LADDER_ENABLED", "true").lower() == "true"
PROOF_VALIDATED_MIN_TRADES = int(os.getenv("PROOF_VALIDATED_MIN_TRADES", "8"))
PROOF_TRUSTED_MIN_TRADES = int(os.getenv("PROOF_TRUSTED_MIN_TRADES", "15"))
PROOF_VALIDATED_MIN_WIN_RATE = float(os.getenv("PROOF_VALIDATED_MIN_WIN_RATE", "0.53"))
PROOF_TRUSTED_MIN_WIN_RATE = float(os.getenv("PROOF_TRUSTED_MIN_WIN_RATE", "0.56"))
PROOF_VALIDATED_MIN_AVG_PNL_PCT = float(os.getenv("PROOF_VALIDATED_MIN_AVG_PNL_PCT", "0.004"))
PROOF_TRUSTED_MIN_AVG_PNL_PCT = float(os.getenv("PROOF_TRUSTED_MIN_AVG_PNL_PCT", "0.006"))
OPPORTUNITY_SCOUT_ENABLED = os.getenv("OPPORTUNITY_SCOUT_ENABLED", "true").lower() == "true"
OPPORTUNITY_SCOUT_MIN_SCORE = float(os.getenv("OPPORTUNITY_SCOUT_MIN_SCORE", "12.0"))
OPPORTUNITY_SCOUT_HORIZON_BARS = int(os.getenv("OPPORTUNITY_SCOUT_HORIZON_BARS", "24"))
OPPORTUNITY_SCOUT_MAX_PENDING = int(os.getenv("OPPORTUNITY_SCOUT_MAX_PENDING", "300"))
OPPORTUNITY_SCOUT_PROOF_MIN_SAMPLES = int(os.getenv("OPPORTUNITY_SCOUT_PROOF_MIN_SAMPLES", "16"))
OPPORTUNITY_SCOUT_PROOF_MIN_WIN_RATE = float(os.getenv("OPPORTUNITY_SCOUT_PROOF_MIN_WIN_RATE", "0.58"))
OPPORTUNITY_SCOUT_PROOF_MIN_AVG_PNL_PCT = float(os.getenv("OPPORTUNITY_SCOUT_PROOF_MIN_AVG_PNL_PCT", "0.005"))

# Tool evidence profiles convert the thesis into capital rules: detectors are
# allowed to compete, but fee-negative or weakly proven tools must earn entries
# through higher scores, stacking, or contextual evidence.
TOOL_EVIDENCE_PROFILES = {
    'panic_reversal_absorption': {
        'tier': 'primary_walk_forward_edge', 'min_score': 18.0, 'risk_mult': 1.10,
        'score_mult': 1.08, 'max_position_pct': 0.10, 'prior_pf': 1.31,
        'prior_return_pct': 8.39,
    },
    'major_pair_breakout': {
        'tier': 'major_continuation_edge', 'min_score': 20.0, 'risk_mult': 1.00,
        'score_mult': 1.03, 'max_position_pct': 0.10, 'prior_pf': 1.62,
    },
    'crash_neg_ac': {'tier': 'crash_reversal', 'min_score': 12.0, 'risk_mult': 1.00, 'prior_pf': 1.30},
    'mega_crash': {'tier': 'crash_reversal', 'min_score': 40.0, 'risk_mult': 0.90, 'prior_pf': 1.41},
    'crash_buy': {'tier': 'crash_reversal', 'min_score': 25.0, 'risk_mult': 0.85, 'prior_pf': 1.10},
    'blood_in_streets': {'tier': 'crash_reversal', 'min_score': 20.0, 'risk_mult': 0.75, 'prior_pf': 1.00},
    'market_panic_70': {'tier': 'crash_reversal', 'min_score': 20.0, 'risk_mult': 0.75, 'prior_pf': 1.00},
    'crash_mean_revert': {'tier': 'crash_reversal', 'min_score': 12.0, 'risk_mult': 0.85, 'prior_pf': 1.05},
    'vpin_dip': {'tier': 'probation_rebound', 'min_score': 10.0, 'risk_mult': 0.65, 'require_stack_or_context': True, 'prior_pf': 0.71},
    'vpin_toxic': {'tier': 'probation_rebound', 'min_score': 12.0, 'risk_mult': 0.65, 'require_stack_or_context': True, 'prior_pf': 1.00},
    'deep_dip_8h': {'tier': 'probation_rebound', 'min_score': 18.0, 'risk_mult': 0.45, 'require_stack_or_context': True, 'prior_pf': 0.90},
    'flash_crash': {'tier': 'probation_rebound', 'min_score': 25.0, 'risk_mult': 0.45, 'require_stack_or_context': True, 'prior_pf': 1.08},
    'btc_alt_spread': {'tier': 'thin_edge_scalper', 'min_score': 6.5, 'risk_mult': 0.80, 'require_stack_or_context': True, 'prior_pf': 0.70, 'max_position_pct': 0.08},
    'dip_buy_5pct': {'tier': 'probation_rebound', 'min_score': 10.0, 'risk_mult': 0.70, 'require_stack_or_context': True, 'prior_pf': 0.70, 'max_position_pct': 0.08},
    'month_start_long': {'tier': 'calendar_edge', 'min_score': 14.0, 'risk_mult': 0.65, 'require_stack_or_context': True, 'max_position_pct': 0.06},
    'market_breadth_recovery': {'tier': 'breadth_recovery', 'min_score': 16.0, 'risk_mult': 0.65, 'max_position_pct': 0.06, 'prior_pf': 1.00},
    'simple_buy_uptrend': {'tier': 'slow_swing_probation', 'min_score': 12.0, 'risk_mult': 0.55, 'require_bull_offense': True, 'max_position_pct': 0.06, 'prior_pf': 0.86},
    'buy_btc_leading': {'tier': 'slow_swing_probation', 'min_score': 12.0, 'risk_mult': 0.50, 'require_bull_offense': True, 'max_position_pct': 0.06, 'prior_pf': 0.73},
    'buy_weekly_green': {'tier': 'bull_swing', 'min_score': 18.0, 'risk_mult': 0.75, 'require_bull_offense': True, 'max_position_pct': 0.08},
    'buy_breakout_simple': {'tier': 'bull_swing', 'min_score': 20.0, 'risk_mult': 0.80, 'require_bull_offense': True, 'max_position_pct': 0.08},
    'accumulation_breakout': {'tier': 'bull_momentum', 'min_score': 20.0, 'risk_mult': 0.80, 'require_bull_offense': True, 'max_position_pct': 0.08},
    'hurst_trend_long': {'tier': 'bull_momentum', 'min_score': 18.0, 'risk_mult': 0.75, 'require_bull_offense': True, 'max_position_pct': 0.08},
    'scout_volume_continuation': {'tier': 'opportunity_scout', 'min_score': 16.0, 'risk_mult': 0.35, 'require_scout_watch': True, 'require_normal_market': True, 'max_position_pct': 0.035},
    'scout_trend_pullback': {'tier': 'opportunity_scout', 'min_score': 14.0, 'risk_mult': 0.30, 'require_scout_watch': True, 'require_normal_market': True, 'max_position_pct': 0.030},
    'scout_reversal_followthrough': {'tier': 'opportunity_scout', 'min_score': 16.0, 'risk_mult': 0.30, 'require_scout_watch': True, 'require_normal_market': True, 'max_position_pct': 0.030},
}

# Minimum score floors for the $300 validation phase. These are intentionally
# conservative: the live journal was near breakeven only after blocked names
# were removed, while historical tests showed low-score churn was fee-negative.
VALIDATION_HISTORICAL_PAIRS = set(ORIGINAL_PAIRS)
VALIDATION_UNKNOWN_PAIR_SCORE_FLOOR = float(os.getenv("VALIDATION_UNKNOWN_PAIR_SCORE_FLOOR", "8.0"))
VALIDATION_LONG_SCORE_FLOORS = {
    'btc_alt_spread': 6.5,
    'dip_buy_5pct': 10.0,
    'vpin_dip': 10.0,
    'vpin_toxic': 12.0,
    'deep_dip_8h': 18.0,
    'crash_mean_revert': 12.0,
    'crash_neg_ac': 10.0,
    'blood_in_streets': 20.0,
    'market_panic_70': 20.0,
    'flash_crash': 25.0,
    'crash_buy': 20.0,
    'mega_crash': 40.0,
    'month_start_long': 14.0,
    'market_breadth_recovery': 16.0,
    'simple_buy_uptrend': 12.0,
    'buy_btc_leading': 12.0,
    'buy_weekly_green': 18.0,
    'buy_breakout_simple': 20.0,
    'major_pair_breakout': 20.0,
    'panic_reversal_absorption': 18.0,
    'accumulation_breakout': 20.0,
    'hurst_trend_long': 15.0,
    'scout_volume_continuation': 16.0,
    'scout_trend_pullback': 14.0,
    'scout_reversal_followthrough': 16.0,
}

# Pair policies let the bot quarantine unstable names without hard-disabling them.
PAIR_POLICIES = {
    'RAVEUSD': {
        'state': 'blocked',
        'allowed_tools': {'crash_neg_ac', 'mega_crash', 'dip_buy_5pct'},
        'probation_tools': {'quick_dip', 'vpin_toxic', 'flash_crash'},
        'blocked_tools': {'deep_dip_8h', 'volatile_oversold', 'quick_crash'},
        'risk_multiplier': 0.50,
        'probation_risk_multiplier': 0.70,
        'max_daily_trades': 2,
        'max_volume_pct': 0.0015,
        'allow_dca': False,
        'allow_runner': False,
        'hold_multiplier': 0.75,
        'min_hold_hours': 4.0,
        'fixed_tp_multiplier': 0.75,
        'pending_price_drift_abandon': 0.01,
        'pending_cancel_pressure_score': 14.0,
        'entry_pressure_block_score': 14.0,
        'bearish_overlay_fight_multiplier': 1.35,
        'bearish_overlay_pressure_cap': 14.0,
        'collapse_requires_rebound_tools': {'dip_buy_5pct', 'quick_dip', 'vpin_toxic'},
        'collapse_short_pressure_cap': 14.0,
        'collapse_min_score': 25.0,
        'conviction_decay_safe_pnl_pct': 0.005,
        'conviction_decay_min_hold_hours': 2.0,
        'conviction_decay_min_hold_fraction': 0.25,
        'conviction_decay_check_bars': 4,
        'conviction_decay_entry_score_keep_fraction': 0.85,
        'conviction_decay_exit_pnl_pct': 0.0,
    },
}

# Validation-mode quality universe. Dynamic discovery can still find Kraken
# opportunities, but new live longs must be in a known, externally checkable
# universe unless ENABLE_QUALITY_UNIVERSE=false is set deliberately.
QUALITY_PAIR_UNIVERSE = {
    'XBTUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 'ADAUSD', 'AVAXUSD', 'LINKUSD',
    'DOTUSD', 'BNBUSD', 'LTCUSD', 'ATOMUSD', 'DOGEUSD', 'FILUSD', 'UNIUSD',
    'AAVEUSD', 'NEARUSD', 'SUIUSD', 'APTUSD', 'RENDERUSD', 'BCHUSD', 'ICPUSD',
    'TRXUSD', 'HBARUSD', 'INJUSD', 'ONDOUSD', 'XLMUSD', 'XMRUSD', 'ZECUSD',
    'JITOSOLUSD', 'TAOUSD', 'HYPEUSD',
}
ENABLE_QUALITY_UNIVERSE = os.getenv("ENABLE_QUALITY_UNIVERSE", "true" if VALIDATION_ACCOUNT_MODE else "false").lower() == "true"
ENABLE_ASSET_CONTEXT_GUARD = os.getenv("ENABLE_ASSET_CONTEXT_GUARD", "true" if VALIDATION_ACCOUNT_MODE else "false").lower() == "true"
ASSET_CONTEXT_MAX_MARKET_CAP_RANK = int(os.getenv("ASSET_CONTEXT_MAX_MARKET_CAP_RANK", "150"))
ASSET_CONTEXT_MIN_24H_CHANGE_PCT = float(os.getenv("ASSET_CONTEXT_MIN_24H_CHANGE_PCT", "-18"))
ASSET_CONTEXT_MIN_7D_CHANGE_PCT = float(os.getenv("ASSET_CONTEXT_MIN_7D_CHANGE_PCT", "-35"))
ASSET_CONTEXT_CACHE_TTL_SEC = int(os.getenv("ASSET_CONTEXT_CACHE_TTL_SEC", str(6 * 3600)))
ASSET_CONTEXT_BLOCK_UNMAPPED = os.getenv("ASSET_CONTEXT_BLOCK_UNMAPPED", "false").lower() == "true"


def get_pair_policy_config(pair: str) -> dict:
    return PAIR_POLICIES.get(normalize_pair(pair), {})


def is_pair_globally_blocked(pair: str) -> bool:
    normalized = normalize_pair(pair)
    if normalized in GEO_BLOCKED_PAIRS:
        return True
    return get_pair_policy_config(normalized).get('state') == 'blocked'

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
    'crash_buy', 'crash_neg_ac', 'blood_in_streets',
    'crash_mean_revert', 'mega_pump_sell_t1', 'rsi_pump_8h',
    'mega_crash', 'vpin_dip'
    # quick_crash removed: 33% WR, -$8.84 live — knife catcher
    # volatile_oversold removed: 0% WR live (1 trade, -$10.23) — ATR entries too early in crashes
}

# Margin costs for 2x leverage
MARGIN_COST_OPEN = 0.0002   # 0.02% to open margin position
MARGIN_COST_PER_BAR = 0.0002  # 0.02% every 4 hours (per bar)

# VALIDATED TOOLS - exact same 30 from production bot
# CRASH/BEAR TOOLS (LONG) - 15 tools
CRASH_BEAR_TOOLS = [
    "crash_buy", "mega_crash", "crash_neg_ac", "blood_in_streets",
    "crash_mean_revert", "vpin_dip", "market_panic_70", "flash_crash",
    "deep_dip_8h", "vpin_toxic", "btc_alt_spread", "panic_reversal_absorption"
    # KILLED: quick_dip (-$54 PnL), entropy_dip (0% WR, -$4.33), quick_crash (33% WR, -$8.84 — falling knives)
    # KILLED: volatile_oversold (0% WR, -$10.23 — ATR entries too early in crashes)
]

# BULL/GREED TOOLS (SHORT) - 13 tools  
BULL_GREED_TOOLS = [
    "mega_pump_sell_t1", "rsi_pump_8h", "falling_wedge_short", "greed_short_t2", "thursday_short",
    "mega_pump_sell_t2", "distribution_short", "late_us_short", "rsi_pump_12h", "ema_cross_short",
    "rsi_pump_fat_tail", "entropy_short", "alt_btc_revert_t3"
]

# NEUTRAL/TRANSITION TOOLS - 2 tools
NEUTRAL_TOOLS = ["month_start_long", "dip_buy_5pct", "market_breadth_recovery"]

# BULL MOMENTUM TOOLS (LONG) - validated on 3yr OOS, bull regime only
BULL_MOMENTUM_TOOLS = ["accumulation_breakout", "hurst_trend_long"]

# BULL SWING TOOLS (LONG) - simple trend-following, 15% trail, hold weeks
# Validated: 3yr OOS, bull regime only, 0.65% fees
BULL_SWING_TOOLS = [
    "buy_weekly_green",       # 257 sig, 44% WR, +5.92%/trade, PF=1.96
    "buy_breakout_simple",    # 123 sig, 44% WR, +6.54%/trade, PF=2.20
    "simple_buy_uptrend",     # 347 sig, 41% WR, +3.55%/trade, PF=1.52
    "buy_btc_leading",        # 103 sig, 31% WR, +4.64%/trade, PF=1.52
    "major_pair_breakout",    # Live-only major continuation when breadth and volume confirm
]

# All validated tools combined
VALIDATED_TOOLS = CRASH_BEAR_TOOLS + BULL_GREED_TOOLS + NEUTRAL_TOOLS + BULL_MOMENTUM_TOOLS + BULL_SWING_TOOLS
TREND_LEADER_TOOLS = set(BULL_MOMENTUM_TOOLS + BULL_SWING_TOOLS)
TRADE_JOURNAL_FIELDS = [
    "timestamp", "event", "pair", "tool", "direction", "price", "score", "base_score",
    "mtf_penalty_applied", "trend_4h", "rsi_4h", "bullish_4h_pct", "fng", "fng_regime",
    "leverage", "position_size", "sl_pct", "hold_bars", "reason",
    "range_pos_24h", "atr_pct", "short_pressure_score", "liquidity_cap_usage",
    "correlation_group", "collapse_gate",
    "range_pct_24h", "btc_trend_4h", "btc_ret_24h",
    "evidence_tier", "evidence_risk_mult", "evidence_live_trades", "evidence_live_pnl",
    "pnl_pct", "pnl_dollar", "bars_held", "close_reason",
    "tool_streak", "balance", "active_balance",
]


class FinalTradingBot:
    """Final trading bot with all 8 upgrades."""
    
    def __init__(self):
        self.client = KrakenClient()
        self.asset_context_guard = None
        if ENABLE_ASSET_CONTEXT_GUARD and _ASSET_CONTEXT_AVAILABLE and AssetContextGuard is not None:
            self.asset_context_guard = AssetContextGuard(
                cache_path=DATA_DIR / "sentiment_cache" / "asset_context_cache.json",
                ttl_seconds=ASSET_CONTEXT_CACHE_TTL_SEC,
                max_market_cap_rank=ASSET_CONTEXT_MAX_MARKET_CAP_RANK,
                min_24h_change_pct=ASSET_CONTEXT_MIN_24H_CHANGE_PCT,
                min_7d_change_pct=ASSET_CONTEXT_MIN_7D_CHANGE_PCT,
                block_unmapped=ASSET_CONTEXT_BLOCK_UNMAPPED,
            )
            logger.info(
                f"[ASSET CONTEXT] Guard enabled: rank<=#{ASSET_CONTEXT_MAX_MARKET_CAP_RANK}, "
                f"24h>{ASSET_CONTEXT_MIN_24H_CHANGE_PCT:.0f}%, 7d>{ASSET_CONTEXT_MIN_7D_CHANGE_PCT:.0f}%"
            )
        elif ENABLE_ASSET_CONTEXT_GUARD and not _ASSET_CONTEXT_AVAILABLE:
            logger.warning("[ASSET CONTEXT] Requested but src/asset_context.py is unavailable")
        # Kraken Futures client (optional, for shorts). Auto-verifies auth when
        # ENABLE_FUTURES_TRADING is on, otherwise stays None.
        self.futures_client = None
        self._futures_ready = False
        if ENABLE_FUTURES_TRADING and _FUTURES_CLIENT_AVAILABLE:
            try:
                self.futures_client = KrakenFuturesClient()
                if self.futures_client.ping():
                    self._futures_ready = True
                    logger.info("[FUTURES] Client ready — shorts may route to Kraken Futures")
                else:
                    logger.warning("[FUTURES] Auth ping failed — futures shorts DISABLED this session")
                    self.futures_client = None
            except Exception as e:
                logger.warning(f"[FUTURES] Init failed: {e} — futures shorts DISABLED")
                self.futures_client = None
        elif ENABLE_FUTURES_TRADING and not _FUTURES_CLIENT_AVAILABLE:
            logger.warning("[FUTURES] ENABLE_FUTURES_TRADING=true but src/kraken_futures_client.py missing")
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
        self._fng_meta = {
            'source': 'initial_neutral',
            'classification': 'Neutral',
            'raw_value': None,
            'effective_value': 50,
            'effective_reason': 'initial_neutral',
            'source_provider': 'initial_neutral',
        }
        
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
        for pos in self.active_positions.values():
            pos.setdefault('initial_position_size', pos.get('position_size', 0.0))
            pos.setdefault('regime_bucket', self._get_regime_bucket())
            pos.setdefault('_realized_partial_pnl', 0.0)
            pos.setdefault('_entry_features', {})
            pos.setdefault('_ml_features', {})
            pos.setdefault('_evidence_snapshot', {})
            pos.setdefault('_evidence_risk_multiplier', 1.0)
        self.active_profit = self.state.get("active_profit", 0.0)
        
        # Tool performance tracking with consecutive wins/losses
        self.tool_stats = self.state.get("tool_stats", {})
        self.tool_streaks = self.state.get("tool_streaks", {})  # UPGRADE 7: track streaks
        # Live-only EV tracking (separate from pretrain seeds in tool_stats).
        # Populated from trade_journal.csv on startup and incremented on every real close.
        self.live_tool_stats = self.state.get("live_tool_stats", {})
        self.opportunity_scout_pending = self.state.get("opportunity_scout_pending", [])
        self.opportunity_scout_stats = self.state.get("opportunity_scout_stats", {})
        self._initialize_tool_stats()
        self._contextual_tool_stats = {}
        self._forward_tool_outcomes = {}
        self._forward_tool_stats = {}
        
        # Price cache for cross-pair signals
        self._price_cache = {}
        self._short_pressure_by_pair = {}
        self._market_short_pressure = {
            'mode': 'normal',
            'label': 'normal',
            'active_pct': 1.0,
            'min_long_score': 0.0,
            'short_signals': 0,
            'short_pairs': 0,
            'top3_avg': 0.0,
            'dominance': 0.0,
        }
        
        # Market regime: default to 75% bullish (conservative — don't open weak shorts on cold start)
        self._bullish_4h_pct = 75
        self._avg_rsi_4h = 55.0
        
        # Trade history and current bar
        self.trade_history = self.state.get("trade_history", [])
        self.current_bar = self.state.get("current_bar", 0)
        
        # UPGRADE 1: Pending limit orders tracking
        self.pending_limit_orders = self.state.get("pending_limit_orders", {})  # pair -> order_info
        
        # Stop-loss cooldown tracker: {pair -> cooldown info}
        self._pair_cooldowns = self.state.get("pair_cooldowns", {})
        # Daily stop-loss counter: {pair -> {'count': N, 'date': 'YYYY-MM-DD'}}
        self._pair_daily_stops = self.state.get("pair_daily_stops", {})
        
        # Pending EXIT orders — don't delete positions until exit fills
        self.pending_exit_orders = self.state.get("pending_exit_orders", {})  # pair -> exit_info
        
        # Trade journal — detailed CSV for post-analysis
        self.trade_journal_path = LOGS_DIR / "trade_journal.csv"
        self._init_trade_journal()
        self._rebuild_contextual_stats_from_journal()
        self._rebuild_forward_tool_stats_from_journal()
        # Live-EV: seed from journal if empty, then re-apply adjustments so
        # pretrain-seeded score_adj values get overridden by real expectancy.
        self._bootstrap_live_tool_stats_from_journal()
        self._recompute_all_live_score_adjustments()
        self.meta_model = SignalMetaModel(DATA_DIR / "ml_models" / "signal_meta_model.json")
        self._bootstrap_meta_model_from_journal()
        
        # Daily tracking
        self._current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_stats = {
            "trades_opened": 0, "trades_closed": 0, 
            "wins": 0, "losses": 0, "pnl": 0.0,
            "start_balance": self.total_balance,
            "tool_pnl": {},
            "rejection_reasons": {},
            "close_reasons": {},
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
                    # Normalize asset name to pair format using mapping
                    if asset in KRAKEN_ASSET_TO_PAIR:
                        pair = KRAKEN_ASSET_TO_PAIR[asset]
                    else:
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
            open_stops = {}  # pair -> order_info (stop-loss sells)
            for txid, order in kraken_orders.items():
                descr = order.get('descr', {})
                pair_raw = descr.get('pair', '')
                side = descr.get('type', '')
                price = float(descr.get('price', 0))
                ordertype = str(descr.get('ordertype', '') or '').lower()

                # Normalize pair name
                pair = pair_raw.upper()
                # Try common mappings (Kraken uses weird alt names sometimes)
                if not pair.endswith('USD'):
                    for p in PAIRS:
                        if pair_raw.upper().replace('/', '').endswith(p[-3:]) and pair_raw.upper().replace('/', '').startswith(p[:3]):
                            pair = p
                            break

                # Track ALL open orders, not just ones in PAIRS — prevents orphaned sell orders
                if pair.endswith('USD'):
                    info = {
                        'txid': txid,
                        'pair': pair,
                        'side': side,
                        'price': price,
                        'qty': float(order.get('vol', 0)),
                        'descr': descr.get('order', ''),
                        'ordertype': ordertype,
                    }
                    if side == 'buy':
                        open_buys[pair] = info
                    elif side == 'sell':
                        if 'stop' in ordertype:
                            open_stops[pair] = info
                        else:
                            open_sells[pair] = info

            # Absorb any existing stop-loss sell orders onto their positions so
            # we don't try to double-place them on restart.
            for _p, _stop in open_stops.items():
                _pos = self.active_positions.get(_p)
                if _pos and not _pos.get('_sl_order_id'):
                    _pos['_sl_order_id'] = _stop['txid']
                    _pos['_sl_price'] = _stop.get('price') or _pos.get('_sl_price')
                    _pos['_sl_qty'] = _stop.get('qty') or _pos.get('_sl_qty')
                    logger.info(
                        f"[SL ABSORB] {_p} existing stop order {_stop['txid']} "
                        f"qty={_stop.get('qty', 0):.4f} @ ${_stop.get('price', 0):.4f}"
                    )
            
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
                    
                    # Place TP order for reconciled positions (10% trailing handles exit,
                    # but also place a limit sell at 8% as a backstop)
                    if ENABLE_LIVE_TRADING and not has_exit:
                        try:
                            tp_pct = 0.08
                            tp_price = holding['price'] * (1 + tp_pct)
                            tp_qty = holding['qty'] * 0.5
                            tp_result = self.client.place_order(pair, "sell", "limit", tp_qty, tp_price, post_only=True)
                            if tp_result:
                                self.active_positions[pair]['_tp_order_id'] = tp_result.get('txid', [None])[0] if isinstance(tp_result, dict) else tp_result
                                self.active_positions[pair]['_tp_price'] = tp_price
                                self.active_positions[pair]['_tp_qty'] = tp_qty
                                logger.info(f"[RECONCILE TP] {pair} sell {tp_qty:.4f} @ ${tp_price:.4f} (8% TP)")
                        except Exception as e:
                            logger.warning(f"[RECONCILE] Failed to place TP for {pair}: {e}")

                    # Place Kraken-native SL on reconciled holdings (uses holding['price'] as entry proxy)
                    if ENABLE_LIVE_TRADING and not has_exit:
                        self._place_native_stop_loss(pair, self.active_positions[pair])

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
                        "regime_bucket": self._get_regime_bucket(),
                        "entry_features": {},
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
                        'initial_position_size': position_size,
                        'qty': buy['qty'],
                        'sl_pct': 0.08,
                        'hold': 48,
                        'score': 0,
                        'total_margin_cost': 0,
                        'regime_bucket': self._get_regime_bucket(),
                        '_realized_partial_pnl': 0.0,
                        '_entry_features': {},
                        '_ml_features': {},
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

            # Backfill: any tracked long without a native stop-loss gets one now.
            # Runs once per reconcile; _place_native_stop_loss is a no-op if already set.
            if ENABLE_LIVE_TRADING:
                for _pair, _pos in list(self.active_positions.items()):
                    if _pos.get('direction', 'long') == 'long' and not _pos.get('_sl_order_id'):
                        self._place_native_stop_loss(_pair, _pos)
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _compute_mtf_multiplier(self, tool, direction, htf_context):
        """Recompute the MTF multiplier for journal accuracy (mirrors apply_mtf_confirmation)."""
        if not ENABLE_MTF or not htf_context or not htf_context.get("htf_available", False):
            return 1.0
        crash_signals = {
            'crash_buy', 'mega_crash', 'crash_neg_ac',
            'blood_in_streets', 'crash_mean_revert', 'panic_reversal_absorption',
            'mega_pump_sell_t1'
            # REMOVED: volatile_oversold (0% WR), quick_crash (33% WR) — killed tools
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
            with open(self.trade_journal_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=TRADE_JOURNAL_FIELDS).writeheader()
            return

        try:
            with open(self.trade_journal_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames or []
                missing_fields = [field for field in TRADE_JOURNAL_FIELDS if field not in existing_fields]
                if not missing_fields:
                    return
                existing_rows = list(reader)

            with open(self.trade_journal_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_JOURNAL_FIELDS)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({field: row.get(field, '') for field in TRADE_JOURNAL_FIELDS})

            logger.info(f"[JOURNAL] Migrated trade journal schema with {len(missing_fields)} new fields")
        except Exception as e:
            logger.warning(f"[JOURNAL] Failed to validate trade journal schema: {e}")

    def _build_entry_feature_snapshot(self, signal: dict, position_size: float,
                                      liquidity_cap_limit: Optional[float]) -> dict:
        """Capture entry context for later analysis and close attribution.

        Falls back to live market state (_pair_volatility, _price_cache) when the
        signal dict did not carry enrichment — this guarantees every entry has
        the context needed for post-hoc filter design.
        """
        pair = signal.get('pair', '')
        short_pressure = getattr(self, '_short_pressure_by_pair', {}).get(pair, {})
        short_pressure_score = short_pressure.get('effective_score', short_pressure.get('score', 0.0))
        liquidity_cap_usage = ''
        if liquidity_cap_limit and np.isfinite(liquidity_cap_limit) and liquidity_cap_limit > 0:
            liquidity_cap_usage = round(position_size / liquidity_cap_limit, 4)

        # range_pos_24h: prefer signal, fall back to _pair_volatility.range_position
        range_pos = self._safe_float(signal.get('_range_pos_24h'), 0.0)
        if range_pos <= 0.0:
            pvol = getattr(self, '_pair_volatility', {}).get(pair, {})
            range_pos = self._safe_float(pvol.get('range_position'), 0.0)

        # atr_pct: prefer signal, fall back to compute from _price_cache (ATR14 / price * 100)
        atr_pct = self._safe_float(signal.get('_atr_pct'), 0.0)
        if atr_pct <= 0.0:
            try:
                prices = getattr(self, '_price_cache', {}).get(pair, [])
                if prices is not None and len(prices) >= 20:
                    closes = np.asarray(prices[-60:], dtype=float)
                    # simple ATR proxy: mean abs hourly return * price / price (pct) over 14 bars
                    diffs = np.abs(np.diff(closes[-15:]))
                    if len(diffs) and closes[-1] > 0:
                        atr_pct = round(float(np.mean(diffs) / closes[-1] * 100), 4)
            except Exception:
                pass

        # 24h range_pct (high-low)/low from _pair_volatility for token-event diagnostics
        pvol = getattr(self, '_pair_volatility', {}).get(pair, {})
        range_pct_24h = self._safe_float(pvol.get('volatility'), 0.0)

        # BTC regime context: 24h return + simple trend_4h classification from SMA
        btc_trend_4h = ''
        btc_ret_24h = 0.0
        try:
            btc_prices = getattr(self, '_price_cache', {}).get('XBTUSD', [])
            if btc_prices is not None and len(btc_prices) >= 200:
                arr = np.asarray(btc_prices, dtype=float)
                btc_ret_24h = round(float((arr[-1] - arr[-24]) / arr[-24]), 4)
                sma50 = float(np.mean(arr[-50:]))
                sma200 = float(np.mean(arr[-200:]))
                last = float(arr[-1])
                if last > sma50 > sma200:
                    btc_trend_4h = 'up'
                elif last < sma50 < sma200:
                    btc_trend_4h = 'down'
                else:
                    btc_trend_4h = 'mixed'
        except Exception:
            pass

        snapshot = {
            'range_pos_24h': round(range_pos, 4),
            'atr_pct': round(atr_pct, 4),
            'short_pressure_score': round(self._safe_float(short_pressure_score, 0.0), 2),
            'liquidity_cap_usage': liquidity_cap_usage,
            'correlation_group': signal.get('_correlation_group', 'other'),
            'collapse_gate': signal.get('_collapse_gate', 'normal'),
            'range_pct_24h': round(range_pct_24h, 4),
            'btc_trend_4h': btc_trend_4h,
            'btc_ret_24h': btc_ret_24h,
        }
        # Log when we had to backfill — visibility into whether scan enrichment is firing
        if self._safe_float(signal.get('_range_pos_24h'), 0.0) <= 0.0 and range_pos > 0.0:
            logger.debug(f"[FEATURE BACKFILL] {pair} range_pos={range_pos:.3f} atr={atr_pct:.2f} range24h={range_pct_24h:.1%}")
        return snapshot

    def _get_position_feature_snapshot(self, pair: str) -> dict:
        """Get persisted entry context for journal close rows."""
        pos = self.active_positions.get(pair, {})
        features = pos.get('_entry_features', {})
        return {
            'range_pos_24h': features.get('range_pos_24h', ''),
            'atr_pct': features.get('atr_pct', ''),
            'short_pressure_score': features.get('short_pressure_score', ''),
            'liquidity_cap_usage': features.get('liquidity_cap_usage', ''),
            'correlation_group': features.get('correlation_group', ''),
            'collapse_gate': features.get('collapse_gate', ''),
            'range_pct_24h': features.get('range_pct_24h', ''),
            'btc_trend_4h': features.get('btc_trend_4h', ''),
            'btc_ret_24h': features.get('btc_ret_24h', ''),
        }

    def _build_meta_features(self, signal: dict, score: float, base_score: float,
                             mtf_multiplier: float, entry_features: Optional[dict] = None) -> dict:
        """Build the pooled meta-model feature vector from live signal context."""
        entry_features = entry_features or {}
        now = datetime.now(timezone.utc)
        correlation_group = entry_features.get('correlation_group') or signal.get('_correlation_group', 'other')
        collapse_gate = entry_features.get('collapse_gate') or signal.get('_collapse_gate', 'normal')

        short_pressure_score = self._safe_float(entry_features.get('short_pressure_score'), 0.0)
        if short_pressure_score == 0.0:
            short_pressure = getattr(self, '_short_pressure_by_pair', {}).get(signal['pair'], {})
            short_pressure_score = self._safe_float(
                short_pressure.get('effective_score', short_pressure.get('score', 0.0)),
                0.0,
            )

        liquidity_cap_usage = self._safe_float(entry_features.get('liquidity_cap_usage'), 0.0)

        return {
            'score': score,
            'base_score': base_score,
            'mtf_multiplier': mtf_multiplier,
            'rsi_4h': self._safe_float(signal.get('_htf_context', {}).get('rsi_4h'), 50.0),
            'bullish_4h_pct': self._safe_float(getattr(self, '_bullish_4h_pct', 50), 50.0),
            'fng': self._safe_float(getattr(self, 'current_fng', 50), 50.0),
            'range_pos_24h': self._safe_float(entry_features.get('range_pos_24h', signal.get('_range_pos_24h', 0.5)), 0.5),
            'atr_pct': self._safe_float(entry_features.get('atr_pct', signal.get('_atr_pct', 0.0)), 0.0),
            'short_pressure_score': short_pressure_score,
            'liquidity_cap_usage': liquidity_cap_usage,
            'corr_large_cap': 1.0 if correlation_group == 'large_cap' else 0.0,
            'corr_alt_l1': 1.0 if correlation_group == 'alt_l1' else 0.0,
            'corr_defi': 1.0 if correlation_group == 'defi' else 0.0,
            'corr_meme': 1.0 if correlation_group == 'meme' else 0.0,
            'corr_mid_cap': 1.0 if correlation_group == 'mid_cap' else 0.0,
            'collapse_gate': (
                1.0 if collapse_gate == 'rebound_confirmed' else
                -1.0 if collapse_gate == 'collapse' else
                0.0
            ),
            'hour_sin': np.sin(2 * np.pi * now.hour / 24.0),
            'hour_cos': np.cos(2 * np.pi * now.hour / 24.0),
            'dow_sin': np.sin(2 * np.pi * now.weekday() / 7.0),
            'dow_cos': np.cos(2 * np.pi * now.weekday() / 7.0),
        }

    def _meta_features_from_journal_row(self, row: dict) -> dict:
        """Rebuild pooled meta-model features from an OPEN journal row."""
        try:
            ts = datetime.strptime(row.get('timestamp', ''), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        correlation_group = row.get('correlation_group') or 'other'
        collapse_gate = row.get('collapse_gate') or 'normal'
        return {
            'score': self._safe_float(row.get('score'), 0.0),
            'base_score': self._safe_float(row.get('base_score'), 0.0),
            'mtf_multiplier': self._safe_float(row.get('mtf_penalty_applied'), 1.0),
            'rsi_4h': self._safe_float(row.get('rsi_4h'), 50.0),
            'bullish_4h_pct': self._safe_float(row.get('bullish_4h_pct'), 50.0),
            'fng': self._safe_float(row.get('fng'), 50.0),
            'range_pos_24h': self._safe_float(row.get('range_pos_24h'), 0.5),
            'atr_pct': self._safe_float(row.get('atr_pct'), 0.0),
            'short_pressure_score': self._safe_float(row.get('short_pressure_score'), 0.0),
            'liquidity_cap_usage': self._safe_float(row.get('liquidity_cap_usage'), 0.0),
            'corr_large_cap': 1.0 if correlation_group == 'large_cap' else 0.0,
            'corr_alt_l1': 1.0 if correlation_group == 'alt_l1' else 0.0,
            'corr_defi': 1.0 if correlation_group == 'defi' else 0.0,
            'corr_meme': 1.0 if correlation_group == 'meme' else 0.0,
            'corr_mid_cap': 1.0 if correlation_group == 'mid_cap' else 0.0,
            'collapse_gate': (
                1.0 if collapse_gate == 'rebound_confirmed' else
                -1.0 if collapse_gate == 'collapse' else
                0.0
            ),
            'hour_sin': np.sin(2 * np.pi * ts.hour / 24.0),
            'hour_cos': np.cos(2 * np.pi * ts.hour / 24.0),
            'dow_sin': np.sin(2 * np.pi * ts.weekday() / 7.0),
            'dow_cos': np.cos(2 * np.pi * ts.weekday() / 7.0),
        }

    def _bootstrap_meta_model_from_journal(self):
        """Train the pooled meta-model from completed journal trades."""
        if not self.trade_journal_path.exists():
            return

        samples = []
        open_trades = {}
        try:
            with open(self.trade_journal_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    event = row.get('event', '')
                    pair = row.get('pair', '')
                    if not pair:
                        continue

                    if event == 'OPEN':
                        open_trades[pair] = {
                            'features': self._meta_features_from_journal_row(row),
                            'pnl_dollar': 0.0,
                        }
                    elif event == 'CLOSE' and pair in open_trades:
                        ctx = open_trades[pair]
                        ctx['pnl_dollar'] += self._safe_float(row.get('pnl_dollar'), 0.0)
                        close_reason = (row.get('close_reason') or '').lower()
                        if close_reason.startswith('partial_'):
                            continue

                        samples.append((ctx['features'], ctx['pnl_dollar'] > 0))
                        del open_trades[pair]

            self.meta_model.fit_samples(samples)
            stats = self.meta_model.get_stats()
            logger.info(
                f"[META] Bootstrapped pooled meta-model from {stats['n_samples']} closed trades "
                f"({'active' if stats['active'] else 'warming'})"
            )
        except Exception as e:
            logger.warning(f"[META] Failed to bootstrap pooled meta-model: {e}")
    
    def _journal_open(self, pair, tool, direction, price, score, base_score,
                      mtf_multiplier, htf_context, leverage, position_size, sl_pct, hold_bars,
                      reason, feature_snapshot: Optional[dict] = None,
                      evidence_snapshot: Optional[dict] = None):
        """Log a trade open to the journal."""
        try:
            trend_4h = htf_context.get("trend_4h", "unknown") if htf_context else "unavailable"
            rsi_4h = htf_context.get("rsi_4h", 0) if htf_context else 0
            bullish_pct = getattr(self, '_bullish_4h_pct', -1)
            fng = getattr(self, 'current_fng', -1)
            fng_regime = self._fng_label_for_value(fng)
            streak_info = ""
            if tool in self.tool_streaks:
                s = self.tool_streaks[tool]
                streak_info = f"{s['type']}{s['streak']}" if s['type'] else "0"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            feature_snapshot = feature_snapshot or {}
            evidence_snapshot = evidence_snapshot or {}
            row = {
                'timestamp': ts,
                'event': 'OPEN',
                'pair': pair,
                'tool': tool,
                'direction': direction,
                'price': f"{price:.6f}",
                'score': f"{score:.2f}",
                'base_score': f"{base_score:.2f}",
                'mtf_penalty_applied': f"{mtf_multiplier:.2f}",
                'trend_4h': trend_4h,
                'rsi_4h': f"{rsi_4h:.1f}",
                'bullish_4h_pct': f"{bullish_pct:.0f}",
                'fng': fng,
                'fng_regime': fng_regime,
                'leverage': leverage,
                'position_size': f"{position_size:.2f}",
                'sl_pct': f"{sl_pct:.3f}",
                'hold_bars': hold_bars,
                'reason': reason,
                'range_pos_24h': feature_snapshot.get('range_pos_24h', ''),
                'atr_pct': feature_snapshot.get('atr_pct', ''),
                'short_pressure_score': feature_snapshot.get('short_pressure_score', ''),
                'liquidity_cap_usage': feature_snapshot.get('liquidity_cap_usage', ''),
                'correlation_group': feature_snapshot.get('correlation_group', ''),
                'collapse_gate': feature_snapshot.get('collapse_gate', ''),
                'range_pct_24h': feature_snapshot.get('range_pct_24h', ''),
                'btc_trend_4h': feature_snapshot.get('btc_trend_4h', ''),
                'btc_ret_24h': feature_snapshot.get('btc_ret_24h', ''),
                'evidence_tier': evidence_snapshot.get('tier', ''),
                'evidence_risk_mult': evidence_snapshot.get('risk_mult', ''),
                'evidence_live_trades': evidence_snapshot.get('live_trades', ''),
                'evidence_live_pnl': evidence_snapshot.get('live_pnl_dollar', ''),
                'pnl_pct': '',
                'pnl_dollar': '',
                'bars_held': '',
                'close_reason': '',
                'tool_streak': streak_info,
                'balance': f"{self.total_balance:.2f}",
                'active_balance': f"{self.active_balance:.2f}",
            }
            with open(self.trade_journal_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=TRADE_JOURNAL_FIELDS).writerow(row)
        except Exception as e:
            logger.debug(f"Journal write error (open): {e}")
    
    def _journal_close(self, pair, tool, direction, exit_price, pnl_pct, pnl_dollar,
                       bars_held, close_reason, entry_price):
        """Log a trade close to the journal."""
        try:
            bullish_pct = getattr(self, '_bullish_4h_pct', -1)
            fng = getattr(self, 'current_fng', -1)
            fng_regime = self._fng_label_for_value(fng)
            streak_info = ""
            if tool in self.tool_streaks:
                s = self.tool_streaks[tool]
                streak_info = f"{s['type']}{s['streak']}" if s['type'] else "0"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            feature_snapshot = self._get_position_feature_snapshot(pair)
            position = self.active_positions.get(pair, {})
            evidence_snapshot = position.get('_evidence_snapshot', {}) or {}
            row = {
                'timestamp': ts,
                'event': 'CLOSE',
                'pair': pair,
                'tool': tool,
                'direction': direction,
                'price': f"{exit_price:.6f}",
                'score': '',
                'base_score': '',
                'mtf_penalty_applied': '',
                'trend_4h': '',
                'rsi_4h': '',
                'bullish_4h_pct': f"{bullish_pct:.0f}",
                'fng': fng,
                'fng_regime': fng_regime,
                'leverage': '',
                'position_size': '',
                'sl_pct': '',
                'hold_bars': '',
                'reason': '',
                'range_pos_24h': feature_snapshot.get('range_pos_24h', ''),
                'atr_pct': feature_snapshot.get('atr_pct', ''),
                'short_pressure_score': feature_snapshot.get('short_pressure_score', ''),
                'liquidity_cap_usage': feature_snapshot.get('liquidity_cap_usage', ''),
                'correlation_group': feature_snapshot.get('correlation_group', ''),
                'collapse_gate': feature_snapshot.get('collapse_gate', ''),
                'range_pct_24h': feature_snapshot.get('range_pct_24h', ''),
                'btc_trend_4h': feature_snapshot.get('btc_trend_4h', ''),
                'btc_ret_24h': feature_snapshot.get('btc_ret_24h', ''),
                'evidence_tier': evidence_snapshot.get('tier', ''),
                'evidence_risk_mult': evidence_snapshot.get('risk_mult', ''),
                'evidence_live_trades': evidence_snapshot.get('live_trades', ''),
                'evidence_live_pnl': evidence_snapshot.get('live_pnl_dollar', ''),
                'pnl_pct': f"{pnl_pct:.4f}",
                'pnl_dollar': f"{pnl_dollar:.4f}",
                'bars_held': bars_held,
                'close_reason': close_reason,
                'tool_streak': streak_info,
                'balance': f"{self.total_balance:.2f}",
                'active_balance': f"{self.active_balance:.2f}",
            }
            with open(self.trade_journal_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=TRADE_JOURNAL_FIELDS).writerow(row)
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
            if hasattr(self, '_daily_stats'):
                category = self._categorize_rejection_reason(reason)
                reasons = self._daily_stats.setdefault("rejection_reasons", {})
                reasons[category] = reasons.get(category, 0) + 1
        except Exception as e:
            logger.debug(f"Rejection log error: {e}")

    def _scout_csv_path(self) -> Path:
        return LOGS_DIR / "opportunity_scout.csv"

    def _write_opportunity_scout_row(self, row: dict):
        try:
            fields = [
                'timestamp', 'event', 'candidate_id', 'cycle', 'pair', 'tool', 'direction',
                'stage', 'score', 'reason', 'entry_price', 'exit_price', 'horizon_bars',
                'net_pnl_pct', 'win', 'fng', 'fng_regime', 'bullish_4h_pct', 'market_mode',
                'stack_count', 'proof_tier', 'range_pos_24h', 'atr_pct', 'short_pressure_score',
                'volume_ratio', 'breakout_pct', 'lower_wick_ratio',
            ]
            path = self._scout_csv_path()
            write_header = not path.exists()
            with open(path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow({field: row.get(field, '') for field in fields})
        except Exception as e:
            logger.debug(f"Opportunity scout write error: {e}")

    def _record_opportunity_scout_candidate(self, signal: dict, score: float,
                                            reason: str, stage: str,
                                            market_data: Optional[dict] = None):
        """Track blocked long candidates as forward paper observations.

        This does not place trades. It lets the bot discover whether a rejected
        setup repeatedly would have beaten fees, then feeds only cautious proof
        into the gate ladder.
        """
        if not OPPORTUNITY_SCOUT_ENABLED:
            return
        if signal.get('direction') != 'long' or score < OPPORTUNITY_SCOUT_MIN_SCORE:
            return

        pair = signal.get('pair', '')
        tool = signal.get('tool', '')
        if not pair or not tool:
            return
        market_data = market_data or getattr(self, '_latest_market_data', {}) or {}
        data = market_data.get(pair, {}) or {}
        entry_price = float(data.get('price') or signal.get('price') or 0.0)
        if entry_price <= 0:
            return

        candidate_id = f"{self.current_bar}:{pair}:{tool}:{stage}"
        pending = getattr(self, 'opportunity_scout_pending', [])
        if any(item.get('candidate_id') == candidate_id for item in pending):
            return
        if any(
            item.get('pair') == pair and item.get('tool') == tool and item.get('stage') == stage
            for item in pending
        ):
            return

        fng_meta = getattr(self, '_fng_meta', {}) or {}
        proof = self._get_tool_proof_status(tool)
        candidate = {
            'candidate_id': candidate_id,
            'cycle': self.current_bar,
            'pair': pair,
            'tool': tool,
            'direction': 'long',
            'stage': stage,
            'score': float(score),
            'reason': reason or '',
            'entry_price': entry_price,
            'horizon_bars': OPPORTUNITY_SCOUT_HORIZON_BARS,
            'fng': getattr(self, 'current_fng', ''),
            'fng_regime': fng_meta.get('classification') or self._fng_label_for_value(getattr(self, 'current_fng', 50)),
            'bullish_4h_pct': getattr(self, '_bullish_4h_pct', ''),
            'market_mode': getattr(self, '_market_short_pressure', {}).get('mode', 'normal'),
            'stack_count': int(signal.get('_stack_count', 1) or 1),
            'proof_tier': proof.get('tier', 'strict'),
            'range_pos_24h': signal.get('_range_pos_24h', ''),
            'atr_pct': signal.get('_atr_pct', ''),
            'short_pressure_score': getattr(self, '_short_pressure_by_pair', {}).get(pair, {}).get('effective_score', ''),
            'volume_ratio': signal.get('_volume_ratio', ''),
            'breakout_pct': signal.get('_breakout_pct', ''),
            'lower_wick_ratio': signal.get('_lower_wick_ratio', ''),
        }
        pending.append(candidate)
        self.opportunity_scout_pending = pending[-OPPORTUNITY_SCOUT_MAX_PENDING:]
        row = dict(candidate)
        row['timestamp'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row['event'] = 'CANDIDATE'
        self._write_opportunity_scout_row(row)

    def _update_opportunity_scout_outcomes(self, market_data: dict):
        if not OPPORTUNITY_SCOUT_ENABLED:
            return
        pending = getattr(self, 'opportunity_scout_pending', [])
        if not pending:
            return

        remaining = []
        for candidate in pending:
            try:
                horizon = int(candidate.get('horizon_bars', OPPORTUNITY_SCOUT_HORIZON_BARS) or OPPORTUNITY_SCOUT_HORIZON_BARS)
                if self.current_bar - int(candidate.get('cycle', 0) or 0) < horizon:
                    remaining.append(candidate)
                    continue
                pair = candidate.get('pair', '')
                data = market_data.get(pair, {}) or {}
                exit_price = float(data.get('price') or 0.0)
                entry_price = float(candidate.get('entry_price') or 0.0)
                if exit_price <= 0 or entry_price <= 0:
                    remaining.append(candidate)
                    continue

                net_pnl_pct = (exit_price - entry_price) / entry_price - ROUND_TRIP_FEE
                won = net_pnl_pct > 0
                tool = candidate.get('tool', '')
                stats = self.opportunity_scout_stats.setdefault(
                    tool, {'samples': 0, 'wins': 0, 'sum_pnl_pct': 0.0}
                )
                stats['samples'] += 1
                if won:
                    stats['wins'] += 1
                stats['sum_pnl_pct'] += net_pnl_pct

                row = dict(candidate)
                row.update({
                    'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    'event': 'OUTCOME',
                    'exit_price': exit_price,
                    'net_pnl_pct': f"{net_pnl_pct:.5f}",
                    'win': int(won),
                })
                self._write_opportunity_scout_row(row)
                if stats['samples'] in {OPPORTUNITY_SCOUT_PROOF_MIN_SAMPLES, OPPORTUNITY_SCOUT_PROOF_MIN_SAMPLES * 2}:
                    scout = self._get_tool_scout_evidence(tool)
                    logger.info(
                        f"[SCOUT] {tool} samples={scout['samples']} wr={scout['win_rate']:.0%} "
                        f"avg_net={scout['avg_pnl_pct']*100:+.2f}%"
                    )
            except Exception as e:
                logger.debug(f"Opportunity scout outcome error: {e}")
                remaining.append(candidate)

        self.opportunity_scout_pending = remaining[-OPPORTUNITY_SCOUT_MAX_PENDING:]

    def _categorize_rejection_reason(self, reason: str) -> str:
        """Bucket rejection reasons for lightweight diagnostics."""
        reason = (reason or '').lower()
        if reason.startswith('evidence_'):
            return 'evidence_gate'
        if reason.startswith('forward_tool_quarantine'):
            return 'forward_tool_quarantine'
        if reason.startswith('meta_model_veto'):
            return 'meta_model_veto'
        if reason.startswith('market_bear_'):
            return 'market_bear'
        if 'bearish_overlay' in reason or 'short_pressure' in reason:
            return 'bearish_overlay'
        if 'correlation_group_limit' in reason:
            return 'correlation_group_limit'
        if 'max_positions_reached' in reason:
            return 'max_positions_reached'
        return reason or 'other'

    def _categorize_close_reason(self, reason: str) -> str:
        """Bucket close reasons for diagnostics snapshots."""
        reason = (reason or '').lower()
        if 'conviction decay' in reason:
            return 'conviction_decay'
        if 'hold timeout' in reason and 'no signal conviction' in reason:
            return 'hold_timeout_no_conviction'
        if 'hold timeout' in reason:
            return 'hold_timeout_other'
        if 'stop loss' in reason:
            return 'stop_loss'
        if 'bearish overlay' in reason:
            return 'bearish_overlay_exit'
        if 'trailing stop' in reason:
            return 'trailing_stop'
        if 'take profit' in reason or reason.startswith('partial_tp') or reason.startswith('partial_tp'):
            return 'take_profit'
        return reason or 'other'

    def _log_forward_diagnostics_snapshot(self):
        """Write one structured diagnostics row for recent forward-tool and exit behavior."""
        try:
            path = LOGS_DIR / "forward_diagnostics.csv"
            write_header = not path.exists()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            forward_stats = getattr(self, '_forward_tool_stats', {}) or {}
            tracked_tools = len(forward_stats)
            forward_samples = sum(stats.get('trades', 0) for stats in forward_stats.values())
            quarantined_tools = sorted(
                tool for tool, stats in forward_stats.items()
                if stats.get('blocked')
            )
            softened_tools = sorted(
                tool for tool, stats in forward_stats.items()
                if 0 < float(stats.get('multiplier', 1.0) or 1.0) < 1.0
            )
            daily_stats = getattr(self, '_daily_stats', {}) or {}
            rejection_reasons = daily_stats.get('rejection_reasons', {}) or {}
            close_reasons = daily_stats.get('close_reasons', {}) or {}
            market_mode = getattr(self, '_market_short_pressure', {}).get('mode', 'normal')

            with open(path, 'a') as f:
                if write_header:
                    f.write(
                        "timestamp,cycle,total_balance,active_balance,market_mode,"
                        "forward_samples,tracked_tools,quarantined_tools,softened_tools,"
                        "forward_tool_quarantine_rejections,meta_model_veto_rejections,"
                        "evidence_gate_rejections,bearish_overlay_rejections,conviction_decay_exits,"
                        "hold_timeout_no_conviction_exits\n"
                    )
                f.write(
                    f"{ts},{self.current_bar},{self.total_balance:.2f},{self.active_balance:.2f},"
                    f"{market_mode},{forward_samples},{tracked_tools},"
                    f"{'|'.join(quarantined_tools)},{'|'.join(softened_tools)},"
                    f"{rejection_reasons.get('forward_tool_quarantine', 0)},"
                    f"{rejection_reasons.get('meta_model_veto', 0)},"
                    f"{rejection_reasons.get('evidence_gate', 0)},"
                    f"{rejection_reasons.get('bearish_overlay', 0)},"
                    f"{close_reasons.get('conviction_decay', 0)},"
                    f"{close_reasons.get('hold_timeout_no_conviction', 0)}\n"
                )
        except Exception as e:
            logger.debug(f"Forward diagnostics snapshot error: {e}")
    
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

            diag_path = LOGS_DIR / "forward_daily_summary.csv"
            diag_header = not diag_path.exists()
            rejection_reasons = stats.get("rejection_reasons", {})
            close_reasons = stats.get("close_reasons", {})
            with open(diag_path, 'a') as f:
                if diag_header:
                    f.write(
                        "date,forward_tool_quarantine_rejections,meta_model_veto_rejections,"
                        "evidence_gate_rejections,bearish_overlay_rejections,conviction_decay_exits,"
                        "hold_timeout_no_conviction_exits\n"
                    )
                f.write(
                    f"{self._current_date},"
                    f"{rejection_reasons.get('forward_tool_quarantine', 0)},"
                    f"{rejection_reasons.get('meta_model_veto', 0)},"
                    f"{rejection_reasons.get('evidence_gate', 0)},"
                    f"{rejection_reasons.get('bearish_overlay', 0)},"
                    f"{close_reasons.get('conviction_decay', 0)},"
                    f"{close_reasons.get('hold_timeout_no_conviction', 0)}\n"
                )
        except Exception as e:
            logger.debug(f"Daily summary error: {e}")
        
        # Reset for new day
        self._current_date = today
        self._daily_stats = {
            "trades_opened": 0, "trades_closed": 0,
            "wins": 0, "losses": 0, "pnl": 0.0,
            "start_balance": self.total_balance,
            "tool_pnl": {},
            "rejection_reasons": {},
            "close_reasons": {},
        }
        logger.info(f"📅 New day: {today} — daily stats reset")
    
    # UPGRADE 4: Dynamic capital allocation based on Fear & Greed
    def get_capital_allocation(self) -> Tuple[float, float]:
        """Grid stays disabled; active allocation shrinks when market-wide bearish pressure dominates."""
        market_pressure = getattr(self, '_market_short_pressure', None) or {}
        active_pct = market_pressure.get('active_pct', 1.0)
        return 0.0, max(0.0, min(1.0, active_pct))

    def _is_bull_offense_mode(self) -> bool:
        """Press validated long edge only when the broad tape is clearly supportive."""
        market_pressure = getattr(self, '_market_short_pressure', None) or {}
        if market_pressure.get('mode') != 'normal':
            return False

        bullish_pct = getattr(self, '_bullish_4h_pct', 50)
        fng = getattr(self, 'current_fng', 50)
        dominance = market_pressure.get('dominance', 0.0)
        short_pairs = market_pressure.get('short_pairs', 0)

        return (
            bullish_pct >= BULL_OFFENSE_MIN_BULLISH_PCT and
            fng >= BULL_OFFENSE_MIN_FNG and
            dominance <= BULL_OFFENSE_MAX_SHORT_DOMINANCE and
            short_pairs <= BULL_OFFENSE_MAX_SHORT_PAIRS
        )

    def _get_total_exposure_cap(self) -> float:
        """Allow modestly higher deployment only in strong long-friendly regimes."""
        if self._is_bull_offense_mode():
            return BULL_OFFENSE_TOTAL_EXPOSURE_PCT
        return MAX_TOTAL_EXPOSURE_PCT

    def _get_position_cap_pct(self, tool: str, direction: str, score: float) -> float:
        """Lift single-position cap only for strong trend-leader longs in bull offense mode."""
        profile_cap = self._get_tool_evidence_profile(tool).get('max_position_pct')
        base_cap = MAX_POSITION_PCT
        if (
            direction == 'long' and
            self._is_trend_leader_tool(tool) and
            score >= BULL_OFFENSE_MIN_SCORE and
            self._is_bull_offense_mode()
        ):
            base_cap = BULL_OFFENSE_POSITION_PCT
        if profile_cap is not None:
            base_cap = min(base_cap, float(profile_cap))
        return base_cap

    def _get_pair_policy(self, pair: str) -> dict:
        return get_pair_policy_config(pair)

    def _pair_is_globally_blocked(self, pair: str) -> bool:
        return is_pair_globally_blocked(pair)

    def _get_quality_universe_rejection(self, pair: str, direction: str) -> Optional[str]:
        if direction != 'long' or not ENABLE_QUALITY_UNIVERSE:
            return None
        normalized = normalize_pair(pair)
        if normalized not in QUALITY_PAIR_UNIVERSE:
            return f"quality_universe_unapproved_{normalized}"
        return None

    def _evaluate_asset_context(self, pair: str, direction: str) -> dict:
        if direction != 'long' or not self.asset_context_guard:
            return {'ok': True, 'reason': 'asset_context_disabled'}
        try:
            return self.asset_context_guard.evaluate(pair)
        except Exception as exc:
            logger.debug(f"[ASSET CONTEXT] {pair} check failed: {exc}")
            return {'ok': True, 'reason': 'asset_context_exception'}

    def _get_pair_daily_trade_limit(self, pair: str) -> int:
        policy = self._get_pair_policy(pair)
        return max(1, int(policy.get('max_daily_trades', MAX_TRADES_PER_PAIR_PER_DAY)))

    def _get_pair_risk_multiplier(self, pair: str, tool: str) -> float:
        policy = self._get_pair_policy(pair)
        multiplier = float(policy.get('risk_multiplier', 1.0))
        if tool in policy.get('probation_tools', set()):
            multiplier *= float(policy.get('probation_risk_multiplier', 1.0))
        return max(0.0, multiplier)

    def _get_pair_hold_hours(self, pair: str, planned_hold_hours: float) -> float:
        policy = self._get_pair_policy(pair)
        hold_multiplier = float(policy.get('hold_multiplier', 1.0))
        min_hold_hours = float(policy.get('min_hold_hours', CHECK_INTERVAL / 3600))
        return max(min_hold_hours, float(planned_hold_hours or 0.0) * hold_multiplier)

    def _get_pair_liquidity_cap_limit(self, pair: str, default_limit: Optional[float] = None) -> Optional[float]:
        policy = self._get_pair_policy(pair)
        limit = default_limit if default_limit is not None else float('inf')
        max_volume_pct = policy.get('max_volume_pct')
        if max_volume_pct and hasattr(self, '_pair_volatility'):
            pair_info = self._pair_volatility.get(pair, {})
            volume_usd = float(pair_info.get('volume_usd', 0.0) or 0.0)
            if volume_usd > 0:
                limit = min(limit, volume_usd * float(max_volume_pct))
        return None if limit == float('inf') else limit

    def _get_pair_policy_rejection(self, signal: dict, score: float) -> Optional[str]:
        pair = signal.get('pair', '')
        direction = signal.get('direction', '')
        tool = signal.get('tool', '')

        if self._pair_is_globally_blocked(pair):
            return 'pair_blocked'
        if direction != 'long':
            return None

        policy = self._get_pair_policy(pair)
        if not policy:
            return None

        blocked_tools = policy.get('blocked_tools', set())
        if tool in blocked_tools:
            return f'pair_policy_blocked_tool_{tool}'

        allowed_tools = policy.get('allowed_tools', set())
        probation_tools = policy.get('probation_tools', set())
        if (allowed_tools or probation_tools) and tool not in allowed_tools and tool not in probation_tools:
            return f'pair_policy_unapproved_tool_{tool}'

        collapse_gate = signal.get('_collapse_gate', 'normal')
        if collapse_gate == 'collapse':
            if tool in probation_tools:
                return f'pair_policy_probation_requires_rebound_{tool}'
            if tool in policy.get('collapse_requires_rebound_tools', set()):
                return f'pair_policy_requires_rebound_{tool}'

            pressure = getattr(self, '_short_pressure_by_pair', {}).get(pair)
            effective_pressure = 0.0
            if pressure:
                effective_pressure = float(pressure.get('effective_score', pressure.get('score', 0.0)) or 0.0)

            collapse_pressure_cap = policy.get('collapse_short_pressure_cap')
            if collapse_pressure_cap is not None and effective_pressure >= float(collapse_pressure_cap):
                return f'pair_policy_collapse_pressure_{effective_pressure:.1f}'

            collapse_min_score = float(policy.get('collapse_min_score', 0.0) or 0.0)
            if collapse_min_score > 0 and score < collapse_min_score:
                return f'pair_policy_collapse_score_{score:.1f}'

        return None

    def _get_validation_score_floor(self, signal: dict) -> float:
        """Return the minimum score required for new long entries in validation mode."""
        if not VALIDATION_ACCOUNT_MODE:
            return 0.0
        if signal.get('direction') != 'long':
            return 0.0

        pair = normalize_pair(signal.get('pair', ''))
        tool = signal.get('tool', '')
        floor = float(VALIDATION_LONG_SCORE_FLOORS.get(tool, 0.0))
        proof = self._get_tool_proof_status(tool)
        floor *= float(proof.get('floor_mult', 1.0) or 1.0)

        # Dynamic Kraken discoveries are useful, but during $300 validation they
        # need to prove stronger live conviction because we do not have local
        # historical coverage for most of them.
        if pair and pair not in VALIDATION_HISTORICAL_PAIRS:
            floor = max(floor, VALIDATION_UNKNOWN_PAIR_SCORE_FLOOR)

        return floor

    def _get_validation_score_rejection(self, signal: dict, score: float) -> Optional[str]:
        floor = self._get_validation_score_floor(signal)
        if floor > 0 and score < floor:
            return f"validation_score_floor_{floor:.1f}"
        return None

    def _get_tool_evidence_profile(self, tool: str) -> dict:
        return TOOL_EVIDENCE_PROFILES.get(tool, {})

    def _get_tool_live_evidence(self, tool: str) -> dict:
        stats = self.live_tool_stats.get(tool, {}) or {}
        trades = int(stats.get('n', 0) or 0)
        wins = int(stats.get('wins', 0) or 0)
        pnl_dollar = float(stats.get('sum_pnl_dollar', 0.0) or 0.0)
        pnl_pct_sum = float(stats.get('sum_pnl_pct', 0.0) or 0.0)
        return {
            'trades': trades,
            'wins': wins,
            'win_rate': wins / trades if trades > 0 else 0.0,
            'pnl_dollar': pnl_dollar,
            'avg_pnl_pct': pnl_pct_sum / trades if trades > 0 else 0.0,
            'avg_pnl_dollar': pnl_dollar / trades if trades > 0 else 0.0,
        }

    def _get_tool_scout_evidence(self, tool: str) -> dict:
        stats = self.opportunity_scout_stats.get(tool, {}) or {}
        samples = int(stats.get('samples', 0) or 0)
        wins = int(stats.get('wins', 0) or 0)
        pnl_pct_sum = float(stats.get('sum_pnl_pct', 0.0) or 0.0)
        return {
            'samples': samples,
            'wins': wins,
            'win_rate': wins / samples if samples > 0 else 0.0,
            'avg_pnl_pct': pnl_pct_sum / samples if samples > 0 else 0.0,
            'sum_pnl_pct': pnl_pct_sum,
        }

    def _get_tool_proof_status(self, tool: str) -> dict:
        """Return autonomous gate-relaxation status earned from clean forward evidence.

        Scout-only proof can make a tool easier to observe, but only real live
        closes can fully relax stack/bull gates or increase sizing.
        """
        live = self._get_tool_live_evidence(tool)
        forward = self._forward_tool_stats.get(tool, {}) or {}
        scout = self._get_tool_scout_evidence(tool)
        status = {
            'tier': 'strict',
            'floor_mult': 1.0,
            'risk_mult': 1.0,
            'score_mult': 1.0,
            'relax_stack_gate': False,
            'relax_bull_gate': False,
            'live_trades': live['trades'],
            'live_win_rate': live['win_rate'],
            'live_avg_pnl_pct': live['avg_pnl_pct'],
            'live_pnl_dollar': live['pnl_dollar'],
            'scout_samples': scout['samples'],
            'scout_win_rate': scout['win_rate'],
            'scout_avg_pnl_pct': scout['avg_pnl_pct'],
        }
        if not AUTONOMOUS_PROOF_LADDER_ENABLED:
            return status

        scout_watch = (
            scout['samples'] >= OPPORTUNITY_SCOUT_PROOF_MIN_SAMPLES and
            scout['sum_pnl_pct'] > 0 and
            scout['win_rate'] >= OPPORTUNITY_SCOUT_PROOF_MIN_WIN_RATE and
            scout['avg_pnl_pct'] >= OPPORTUNITY_SCOUT_PROOF_MIN_AVG_PNL_PCT
        )
        if scout_watch:
            status.update({
                'tier': 'scout_watch',
                'floor_mult': 0.92,
                'score_mult': 1.02,
            })

        recent_ok = not forward.get('blocked') and float(forward.get('multiplier', 1.0) or 1.0) > 0.0
        validated = (
            live['trades'] >= PROOF_VALIDATED_MIN_TRADES and
            live['pnl_dollar'] > 0 and
            live['win_rate'] >= PROOF_VALIDATED_MIN_WIN_RATE and
            live['avg_pnl_pct'] >= PROOF_VALIDATED_MIN_AVG_PNL_PCT and
            recent_ok
        )
        if validated:
            status.update({
                'tier': 'validated',
                'floor_mult': 0.85,
                'risk_mult': 1.05,
                'score_mult': 1.03,
                'relax_stack_gate': True,
            })

        trusted = (
            live['trades'] >= PROOF_TRUSTED_MIN_TRADES and
            live['pnl_dollar'] > 0 and
            live['win_rate'] >= PROOF_TRUSTED_MIN_WIN_RATE and
            live['avg_pnl_pct'] >= PROOF_TRUSTED_MIN_AVG_PNL_PCT and
            recent_ok
        )
        if trusted:
            status.update({
                'tier': 'trusted',
                'floor_mult': 0.75,
                'risk_mult': 1.10,
                'score_mult': 1.05,
                'relax_stack_gate': True,
                'relax_bull_gate': True,
            })
        return status

    def _evaluate_tool_evidence(self, signal: dict, score: float) -> Tuple[bool, float, float, Optional[str], dict]:
        """Gate and scale long entries using evidence after fees.

        Returns: allowed, adjusted_score, risk_multiplier, rejection_reason, snapshot.
        Short signals are still useful as defensive overlays, but spot-only Kraken
        mode never routes them into capital allocation.
        """
        tool = signal.get('tool', '')
        direction = signal.get('direction', '')
        pair = normalize_pair(signal.get('pair', ''))
        profile = self._get_tool_evidence_profile(tool)
        live = self._get_tool_live_evidence(tool)
        proof = self._get_tool_proof_status(tool)
        regime_bucket = self._get_regime_bucket()
        context_mult = self._get_contextual_score_multiplier(pair, tool, regime_bucket)
        stack_count = int(signal.get('_stack_count', 1) or 1)
        snapshot = {
            'tier': profile.get('tier', 'unprofiled'),
            'proof_tier': proof.get('tier', 'strict'),
            'proof_floor_mult': proof.get('floor_mult', 1.0),
            'live_trades': live['trades'],
            'live_win_rate': live['win_rate'],
            'live_pnl_dollar': live['pnl_dollar'],
            'live_avg_pnl_pct': live['avg_pnl_pct'],
            'scout_samples': proof.get('scout_samples', 0),
            'scout_win_rate': proof.get('scout_win_rate', 0.0),
            'scout_avg_pnl_pct': proof.get('scout_avg_pnl_pct', 0.0),
            'context_mult': context_mult,
            'stack_count': stack_count,
            'risk_mult': 1.0,
            'score_mult': 1.0,
        }

        if not EVIDENCE_MODE or direction != 'long':
            return True, score, 1.0, None, snapshot

        if profile.get('require_scout_watch') and proof.get('tier') == 'strict':
            return False, score, 0.0, 'opportunity_scout_waiting_for_proof', snapshot
        if profile.get('require_normal_market'):
            market_pressure = getattr(self, '_market_short_pressure', {}) or {}
            if market_pressure.get('mode', 'normal') != 'normal':
                return False, score, 0.0, 'evidence_requires_normal_market', snapshot

        min_score = float(profile.get('min_score', 0.0) or 0.0) * float(proof.get('floor_mult', 1.0) or 1.0)
        if min_score > 0 and score < min_score:
            return False, score, 0.0, f"evidence_score_floor_{min_score:.1f}", snapshot

        if VALIDATION_ACCOUNT_MODE and tool == 'month_start_long':
            market_pressure = getattr(self, '_market_short_pressure', {}) or {}
            bullish_pct = getattr(self, '_bullish_4h_pct', 50)
            fng = getattr(self, 'current_fng', 50)
            supportive_regime = (
                market_pressure.get('mode', 'normal') == 'normal' and
                bullish_pct >= 55 and
                fng >= 45
            )
            proof_relaxed_regime = (
                proof.get('relax_bull_gate') and
                market_pressure.get('mode', 'normal') == 'normal' and
                bullish_pct >= 55 and
                fng >= 40
            )
            if not supportive_regime and not proof_relaxed_regime:
                return False, score, 0.0, 'evidence_month_start_requires_supportive_regime', snapshot

        risk_mult = float(profile.get('risk_mult', 1.0) or 1.0)
        score_mult = float(profile.get('score_mult', 1.0) or 1.0)
        has_stack = stack_count >= 2
        has_positive_context = context_mult >= 1.08
        has_live_edge = (
            live['trades'] >= EVIDENCE_LIVE_SOFT_MIN_TRADES and
            live['pnl_dollar'] > 0 and
            live['avg_pnl_pct'] >= EVIDENCE_MIN_NET_EDGE_PCT
        )

        if profile.get('require_bull_offense'):
            major_bypass = self._allow_major_pair_bull_bypass(pair, tool, signal, score)
            proof_bypass = (
                proof.get('relax_bull_gate') and
                getattr(self, '_market_short_pressure', {}).get('mode', 'normal') == 'normal' and
                getattr(self, '_bullish_4h_pct', 50) >= 55 and
                getattr(self, 'current_fng', 50) >= 45
            )
            if not self._is_bull_offense_mode() and not major_bypass and not proof_bypass:
                return False, score, 0.0, 'evidence_requires_bull_offense', snapshot

        if profile.get('require_stack_or_context') and not (
            has_stack or has_positive_context or has_live_edge or proof.get('relax_stack_gate')
        ):
            return False, score, 0.0, 'evidence_requires_stack_or_context', snapshot

        risk_mult *= float(proof.get('risk_mult', 1.0) or 1.0)
        score_mult *= float(proof.get('score_mult', 1.0) or 1.0)

        if live['trades'] >= EVIDENCE_LIVE_SOFT_MIN_TRADES:
            if live['pnl_dollar'] <= EVIDENCE_LIVE_KILL_DOLLAR_LOSS and live['win_rate'] < 0.50:
                return False, score, 0.0, f"evidence_live_drawdown_{live['pnl_dollar']:.2f}", snapshot
            if live['pnl_dollar'] < 0:
                risk_mult *= 0.45
                score_mult *= 0.75
            elif live['avg_pnl_pct'] < EVIDENCE_MIN_NET_EDGE_PCT:
                risk_mult *= 0.75
                score_mult *= 0.92
            elif live['trades'] >= EVIDENCE_LIVE_HARD_MIN_TRADES and live['avg_pnl_pct'] >= 0.01:
                risk_mult *= 1.15
                score_mult *= 1.05

        prior_pf = profile.get('prior_pf')
        if prior_pf is not None and float(prior_pf) < 1.0 and not has_live_edge:
            risk_mult *= 0.70
            score_mult *= 0.90

        if has_positive_context:
            risk_mult *= min(1.10, context_mult)
            score_mult *= min(1.05, context_mult)

        risk_mult = max(0.20, min(1.25, risk_mult))
        score_mult = max(0.50, min(1.20, score_mult))
        adjusted_score = score * score_mult
        snapshot['risk_mult'] = risk_mult
        snapshot['score_mult'] = score_mult
        return True, adjusted_score, risk_mult, None, snapshot
    
    def rebalance_capital(self):
        """Rebalance capital allocation using current live deployment."""
        grid_pct, active_pct = self.get_capital_allocation()
        self.grid_balance = self.total_balance * grid_pct
        deployed_capital = sum(
            pos.get('position_size', 0) for pos in self.active_positions.values()
        )
        self.active_balance = max(0.0, self.total_balance * active_pct - deployed_capital)
    
    def _initialize_tool_stats(self):
        """Initialize tool performance stats with validated results."""
        # Initialize stats for all validated tools based on OOS testing
        validated_stats = {
            # CRASH/BEAR TOOLS - from VALIDATED_TOOLS.md
            "volatile_oversold": {"trades": 1, "wins": 0, "pnl": -10.23, "score_adj": 0.0},  # KILLED: 0% WR live
            "crash_buy": {"trades": 120, "wins": 78, "pnl": 190.0, "score_adj": 1.0},
            "mega_crash": {"trades": 40, "wins": 21, "pnl": 135.0, "score_adj": 1.0},
            "crash_neg_ac": {"trades": 80, "wins": 50, "pnl": 125.0, "score_adj": 1.0},
            "blood_in_streets": {"trades": 60, "wins": 37, "pnl": 110.0, "score_adj": 1.0},
            "quick_crash": {"trades": 3, "wins": 1, "pnl": -8.84, "score_adj": 0.0},  # KILLED: 33% WR live
            "crash_mean_revert": {"trades": 80, "wins": 49, "pnl": 98.0, "score_adj": 1.0},
            "vpin_dip": {"trades": 70, "wins": 41, "pnl": 73.0, "score_adj": 1.0},
            "market_panic_70": {"trades": 50, "wins": 30, "pnl": 75.0, "score_adj": 1.0},
            "flash_crash": {"trades": 60, "wins": 33, "pnl": 51.0, "score_adj": 1.0},
            "deep_dip_8h": {"trades": 70, "wins": 38, "pnl": 22.0, "score_adj": 1.0},
            "entropy_dip": {"trades": 4, "wins": 0, "pnl": -4.33, "score_adj": 0.0},  # KILLED: 0% WR live
            "vpin_toxic": {"trades": 65, "wins": 35, "pnl": 45.0, "score_adj": 1.0},
            "btc_alt_spread": {"trades": 55, "wins": 30, "pnl": 45.0, "score_adj": 1.0},
            "panic_reversal_absorption": {"trades": 128, "wins": 66, "pnl": 25.17, "score_adj": 1.0},
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
            "market_breadth_recovery": {"trades": 0, "wins": 0, "pnl": 0.0, "score_adj": 1.0},

            # MAJOR BREAKOUT TOOL
            "major_pair_breakout": {"trades": 0, "wins": 0, "pnl": 0.0, "score_adj": 1.0},
        }
        
        # Initialize all validated tools
        for tool in VALIDATED_TOOLS:
            if tool not in self.tool_stats:
                self.tool_stats[tool] = validated_stats.get(tool, 
                    {"trades": 0, "wins": 0, "pnl": 0.0, "score_adj": 1.0})
            
            # UPGRADE 7: Initialize streak tracking
            if tool not in self.tool_streaks:
                self.tool_streaks[tool] = {"streak": 0, "type": None}  # streak marker: W or L
    
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

        # Live-EV override (added 2026-04-23): once we have enough real post-pretrain
        # samples, score_adj is driven by realized expectancy, not just streaks.
        self._apply_live_ev_score_adj(tool)

    # ---- Live-EV score adjustment (2026-04-23) ------------------------------
    # Problem: tool_stats score_adj was effectively frozen at 1.0 because the
    # only adjustments were streak-based and pretrain seeds inflated the sample
    # floor. This meant losing tools (e.g. quick_dip -$60 across 12 trades)
    # kept full size. This module tracks LIVE-ONLY outcomes and drives score_adj
    # from realized expectancy once a tool has enough live samples.
    LIVE_EV_MIN_TRADES = 8           # need at least this many live trades before overriding
    LIVE_EV_MIN_WR_FLOOR = 0.35      # WR below this caps adj at 0.5 regardless of avg pnl

    def _bootstrap_live_tool_stats_from_journal(self):
        """Seed live_tool_stats from trade_journal.csv CLOSE rows. Runs once at startup
        if the dict is empty. Counts each CLOSE as one sample; tolerant to missing fields."""
        if self.live_tool_stats:
            return
        try:
            if not self.trade_journal_path.exists():
                return
            with open(self.trade_journal_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('event') != 'CLOSE':
                        continue
                    close_reason = (row.get('close_reason') or row.get('reason') or '').lower()
                    if 'pre_clean_slate' in close_reason:
                        continue
                    tool = row.get('tool')
                    if not tool:
                        continue
                    try:
                        pnl_pct = float(row.get('pnl_pct') or 0.0)
                        pnl_dollar = float(row.get('pnl_dollar') or 0.0)
                    except (TypeError, ValueError):
                        continue
                    lts = self.live_tool_stats.setdefault(
                        tool, {'n': 0, 'wins': 0, 'sum_pnl_pct': 0.0, 'sum_pnl_dollar': 0.0}
                    )
                    lts['n'] += 1
                    if pnl_pct > 0:
                        lts['wins'] += 1
                    lts['sum_pnl_pct'] += pnl_pct
                    lts['sum_pnl_dollar'] += pnl_dollar
            logger.info(f"[LIVE-EV] Bootstrapped live_tool_stats from journal: {len(self.live_tool_stats)} tools")
        except Exception as e:
            logger.warning(f"[LIVE-EV] Bootstrap failed: {e}")

    def _record_live_tool_outcome(self, tool: str, pnl_pct: float, pnl_dollar: float):
        """Increment live-only stats on real close. Called from close_position."""
        lts = self.live_tool_stats.setdefault(
            tool, {'n': 0, 'wins': 0, 'sum_pnl_pct': 0.0, 'sum_pnl_dollar': 0.0}
        )
        lts['n'] += 1
        if pnl_pct > 0:
            lts['wins'] += 1
        lts['sum_pnl_pct'] += pnl_pct
        lts['sum_pnl_dollar'] += pnl_dollar

    def _apply_live_ev_score_adj(self, tool: str):
        """Override tool_stats[tool]['score_adj'] from live expectancy when n>=LIVE_EV_MIN_TRADES.
        Leaves streak-driven adj alone below threshold.
        Rule: a tool must be net-positive in realized DOLLARS before it can be boosted.
        This prevents the arithmetic-mean pnl_pct trap where many small wins hide a
        few large losses (e.g. quick_dip: avg +0.7%/trade but sum -$60)."""
        lts = self.live_tool_stats.get(tool)
        if not lts or lts['n'] < self.LIVE_EV_MIN_TRADES:
            return
        n = lts['n']
        wr = lts['wins'] / n
        avg_pnl_pct = lts['sum_pnl_pct'] / n
        sum_dollar = lts['sum_pnl_dollar']
        avg_dollar = sum_dollar / n
        # Dollar-first bucketing — a losing tool is a losing tool no matter what the
        # arithmetic-mean percent says.
        if sum_dollar <= -20.0 or avg_dollar <= -1.5:
            adj = 0.25       # strongly losing — effectively killed
        elif sum_dollar < 0 or avg_dollar < 0:
            adj = 0.50
        elif avg_pnl_pct <= 0.0:
            adj = 0.80
        elif avg_pnl_pct < 0.005:
            adj = 1.00
        elif avg_pnl_pct < 0.01:
            adj = 1.15
        else:
            adj = min(1.5, 1.0 + avg_pnl_pct * 20.0)
        # WR sanity cap
        if wr < self.LIVE_EV_MIN_WR_FLOOR:
            adj = min(adj, 0.5)
        stats = self.tool_stats.setdefault(tool, {"trades": 0, "wins": 0, "pnl": 0.0, "score_adj": 1.0})
        prev = stats.get('score_adj', 1.0)
        stats['score_adj'] = adj
        if abs(prev - adj) >= 0.1:
            logger.info(
                f"[LIVE-EV] {tool} n={n} wr={wr*100:.0f}% avg_pnl={avg_pnl_pct*100:+.2f}% "
                f"sum=${sum_dollar:+.2f} → score_adj {prev:.2f}→{adj:.2f}"
            )

    def _recompute_all_live_score_adjustments(self):
        """Sweep all tools once at startup after bootstrap."""
        for tool in list(self.live_tool_stats.keys()):
            self._apply_live_ev_score_adj(tool)

    def _is_major_pair_breakout_context(self, pair: str, tool: str, signal: Optional[dict], score: float) -> bool:
        normalized = normalize_pair(pair)
        if normalized not in MAJOR_BREAKOUT_PAIRS:
            return False
        if tool not in {'major_pair_breakout', 'simple_buy_uptrend', 'buy_btc_leading'}:
            return False
        if signal is None:
            return False

        volume_ratio = self._safe_float(signal.get('_volume_ratio'))
        breakout_pct = self._safe_float(signal.get('_breakout_pct'))
        breadth = getattr(self, '_bullish_4h_pct', 50) / 100.0
        short_dominance = self._safe_float(getattr(self, '_market_short_pressure', {}).get('dominance', 0.0))

        return (
            score >= MAJOR_BREAKOUT_MIN_SCORE and
            volume_ratio >= MAJOR_BREAKOUT_MIN_VOLUME_RATIO and
            breakout_pct >= MAJOR_BREAKOUT_MIN_BREAKOUT_PCT and
            breadth >= MAJOR_BREAKOUT_MIN_BREADTH and
            short_dominance <= MAJOR_BREAKOUT_MAX_SHORT_DOMINANCE
        )

    def _allow_major_pair_bull_bypass(self, pair: str, tool: str, signal: Optional[dict], score: float) -> bool:
        normalized = normalize_pair(pair)
        if normalized not in MAJOR_BREAKOUT_PAIRS:
            return False

        if tool == 'major_pair_breakout':
            fng_floor = MAJOR_BREAKOUT_MIN_FNG
            bullish_floor = MAJOR_BREAKOUT_MIN_BULLISH_PCT
        elif tool in {'simple_buy_uptrend', 'buy_btc_leading'}:
            fng_floor = 35
            bullish_floor = 55
        else:
            return False

        fng = self.get_fng()
        bullish_pct = getattr(self, '_bullish_4h_pct', 50)
        return (
            fng >= fng_floor and
            bullish_pct >= bullish_floor and
            self._is_major_pair_breakout_context(normalized, tool, signal, score)
        )

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Best-effort float conversion for journal/state values."""
        try:
            if value in ('', None):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_regime_bucket(self, fng: Optional[float] = None, bullish_pct: Optional[float] = None) -> str:
        """Bucket the market regime for contextual expectancy."""
        if fng is None:
            fng = getattr(self, 'current_fng', 50)
        if bullish_pct is None:
            bullish_pct = getattr(self, '_bullish_4h_pct', 50)

        if fng < 25 or bullish_pct < 35:
            return 'bear_crash'
        if fng > 60 and bullish_pct > 65:
            return 'trend_bull'
        if fng < 45 or bullish_pct < 55:
            return 'cautious'
        return 'neutral'

    def _get_tool_trade_count(self, tool: str) -> int:
        """Get historical trade count for a tool, supporting legacy state keys."""
        ts = self.tool_stats.get(tool, {})
        return int(ts.get('trades', ts.get('total', 0)) or 0)

    def _record_contextual_outcome(self, pair: str, tool: str, regime_bucket: str,
                                   pnl_pct: float, pnl_dollar: float):
        """Record one full-trade outcome for pair-tool-regime expectancy."""
        for bucket in (regime_bucket, 'all'):
            key = (pair, tool, bucket)
            stats = self._contextual_tool_stats.setdefault(key, {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'win_sum_pct': 0.0,
                'loss_sum_pct': 0.0,
            })
            stats['trades'] += 1
            stats['pnl'] += pnl_dollar
            if pnl_pct > 0:
                stats['wins'] += 1
                stats['win_sum_pct'] += pnl_pct
            else:
                stats['losses'] += 1
                stats['loss_sum_pct'] += abs(pnl_pct)

    def _has_forward_feature_snapshot(self, feature_source: Optional[dict]) -> bool:
        """Detect whether a trade carries the newer post-upgrade feature snapshot."""
        if not feature_source:
            return False

        forward_fields = (
            'range_pos_24h',
            'atr_pct',
            'short_pressure_score',
            'liquidity_cap_usage',
            'correlation_group',
            'collapse_gate',
        )
        return any(feature_source.get(field) not in ('', None) for field in forward_fields)

    def _recompute_forward_tool_stats(self, tool: str):
        """Summarize the recent forward-only outcome window for one tool."""
        outcomes = self._forward_tool_outcomes.get(tool, [])
        if not outcomes:
            self._forward_tool_stats.pop(tool, None)
            return

        trades = len(outcomes)
        wins = sum(1 for outcome in outcomes if outcome['pnl_dollar'] > 0)
        pnl_dollar = sum(outcome['pnl_dollar'] for outcome in outcomes)
        pnl_pct_sum = sum(outcome['pnl_pct'] for outcome in outcomes)
        avg_pnl_pct = pnl_pct_sum / trades if trades > 0 else 0.0
        win_rate = wins / trades if trades > 0 else 0.0

        multiplier = 1.0
        blocked = False
        if (
            FORWARD_TOOL_STRICT_VALIDATION and
            VALIDATION_ACCOUNT_MODE and
            trades >= FORWARD_TOOL_MIN_TRADES and
            pnl_dollar < 0 and
            avg_pnl_pct <= FORWARD_TOOL_VALIDATION_BLOCK_AVG_PNL_PCT
        ):
            multiplier = 0.0
            blocked = True
        elif (
            trades >= FORWARD_TOOL_QUARANTINE_MIN_TRADES and
            wins == 0 and
            pnl_dollar < 0 and
            avg_pnl_pct <= FORWARD_TOOL_QUARANTINE_AVG_PNL_PCT
        ):
            multiplier = 0.0
            blocked = True
        elif (
            trades >= FORWARD_TOOL_MIN_TRADES and
            win_rate < FORWARD_TOOL_BAD_WIN_RATE and
            pnl_dollar < 0 and
            avg_pnl_pct <= FORWARD_TOOL_SOFT_AVG_PNL_PCT
        ):
            multiplier = FORWARD_TOOL_SOFT_MULTIPLIER

        self._forward_tool_stats[tool] = {
            'trades': trades,
            'wins': wins,
            'losses': trades - wins,
            'win_rate': win_rate,
            'pnl': pnl_dollar,
            'avg_pnl_pct': avg_pnl_pct,
            'multiplier': multiplier,
            'blocked': blocked,
        }

    def _append_forward_tool_outcome(self, tool: str, pnl_pct: float, pnl_dollar: float):
        """Append one completed post-upgrade long trade outcome for tool quarantine logic."""
        if not tool:
            return

        outcomes = self._forward_tool_outcomes.setdefault(tool, [])
        outcomes.append({
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
        })
        if len(outcomes) > FORWARD_TOOL_WINDOW_TRADES:
            del outcomes[:-FORWARD_TOOL_WINDOW_TRADES]
        self._recompute_forward_tool_stats(tool)

    def _rebuild_forward_tool_stats_from_journal(self):
        """Rebuild recent forward-only long tool stats from enriched journal rows."""
        self._forward_tool_outcomes = {}
        self._forward_tool_stats = {}
        if not self.trade_journal_path.exists():
            return

        open_trades = {}
        try:
            with open(self.trade_journal_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    event = row.get('event', '')
                    pair = row.get('pair', '')
                    if not pair:
                        continue

                    if event == 'OPEN':
                        if row.get('direction') != 'long':
                            continue
                        open_trades[pair] = {
                            'tool': row.get('tool', ''),
                            'position_size': self._safe_float(row.get('position_size'), 0.0),
                            'eligible': self._has_forward_feature_snapshot(row),
                            'pnl_dollar': 0.0,
                        }
                    elif event == 'CLOSE' and pair in open_trades:
                        ctx = open_trades[pair]
                        ctx['pnl_dollar'] += self._safe_float(row.get('pnl_dollar'), 0.0)
                        close_reason = (row.get('close_reason') or '').lower()
                        if close_reason.startswith('partial_'):
                            continue

                        if ctx['eligible'] and ctx['tool']:
                            position_size = ctx['position_size']
                            total_pnl_pct = (
                                ctx['pnl_dollar'] / position_size
                                if position_size > 0 else
                                self._safe_float(row.get('pnl_pct'), 0.0)
                            )
                            self._append_forward_tool_outcome(
                                ctx['tool'], total_pnl_pct, ctx['pnl_dollar']
                            )
                        del open_trades[pair]
        except Exception as e:
            logger.warning(f"[FORWARD TOOL] Failed to rebuild recent forward tool stats: {e}")
            self._forward_tool_outcomes = {}
            self._forward_tool_stats = {}
            return

        total_samples = sum(stats['trades'] for stats in self._forward_tool_stats.values())
        if total_samples > 0:
            quarantined = sorted(
                tool for tool, stats in self._forward_tool_stats.items()
                if stats.get('blocked')
            )
            summary = (
                f"[FORWARD TOOL] Loaded {total_samples} recent forward long outcomes across "
                f"{len(self._forward_tool_stats)} tools"
            )
            if quarantined:
                summary += f"; quarantined: {', '.join(quarantined)}"
            logger.info(summary)
        else:
            logger.info("[FORWARD TOOL] No completed enriched-feature long outcomes yet")

    def _get_forward_tool_score_adjustment(self, tool: str, direction: str) -> Tuple[float, Optional[str], dict]:
        """Return a recent forward-only multiplier for long tools and a reason if it blocks."""
        if direction != 'long':
            return 1.0, None, {}

        stats = self._forward_tool_stats.get(tool, {})
        if not stats or stats.get('trades', 0) < FORWARD_TOOL_MIN_TRADES:
            return 1.0, None, stats

        multiplier = float(stats.get('multiplier', 1.0) or 1.0)
        if multiplier <= 0.0:
            reason = (
                f"forward_tool_quarantine_{stats.get('trades', 0)}_"
                f"wr_{stats.get('win_rate', 0.0):.2f}_pnl_{stats.get('pnl', 0.0):.2f}"
            )
            return 0.0, reason, stats
        return multiplier, None, stats

    def _rebuild_contextual_stats_from_journal(self):
        """Rebuild pair-tool-regime expectancy stats from the journal."""
        self._contextual_tool_stats = {}
        if not self.trade_journal_path.exists():
            return

        open_trades = {}
        try:
            with open(self.trade_journal_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    event = row.get('event', '')
                    pair = row.get('pair', '')
                    tool = row.get('tool', '')
                    if not pair or not tool:
                        continue

                    if event == 'OPEN':
                        open_trades[pair] = {
                            'tool': tool,
                            'position_size': self._safe_float(row.get('position_size'), 0.0),
                            'regime_bucket': self._get_regime_bucket(
                                self._safe_float(row.get('fng'), 50),
                                self._safe_float(row.get('bullish_4h_pct'), 50),
                            ),
                            'pnl_dollar': 0.0,
                        }
                    elif event == 'CLOSE' and pair in open_trades:
                        ctx = open_trades[pair]
                        ctx['pnl_dollar'] += self._safe_float(row.get('pnl_dollar'), 0.0)
                        close_reason = (row.get('close_reason') or '').lower()
                        if close_reason.startswith('partial_'):
                            continue

                        position_size = ctx['position_size']
                        total_pnl_pct = (
                            ctx['pnl_dollar'] / position_size
                            if position_size > 0 else
                            self._safe_float(row.get('pnl_pct'), 0.0)
                        )
                        self._record_contextual_outcome(
                            pair, ctx['tool'], ctx['regime_bucket'],
                            total_pnl_pct, ctx['pnl_dollar']
                        )
                        del open_trades[pair]
        except Exception as e:
            logger.warning(f"[CONTEXT] Failed to rebuild contextual stats: {e}")
            return

        total_samples = sum(
            stats['trades'] for key, stats in self._contextual_tool_stats.items()
            if key[2] != 'all'
        )
        logger.info(
            f"[CONTEXT] Loaded {total_samples} contextual outcomes across "
            f"{len(self._contextual_tool_stats)} buckets"
        )

    def _get_contextual_score_multiplier(self, pair: str, tool: str,
                                         regime_bucket: Optional[str] = None) -> float:
        """Adjust score using pair-tool-regime expectancy with Bayesian shrinkage."""
        regime_bucket = regime_bucket or self._get_regime_bucket()
        context = (
            self._contextual_tool_stats.get((pair, tool, regime_bucket)) or
            self._contextual_tool_stats.get((pair, tool, 'all'))
        )
        if not context:
            return 1.0

        ts = self.tool_stats.get(tool, {})
        prior_trades = max(1, self._get_tool_trade_count(tool))
        prior_wr = ts.get('wins', 0) / prior_trades if prior_trades > 0 else 0.5
        prior_avg_win = abs(ts.get('avg_win_pct', 0.03)) or 0.03
        prior_avg_loss = abs(ts.get('avg_loss_pct', 0.03)) or 0.03
        prior_exp = prior_wr * prior_avg_win - (1 - prior_wr) * prior_avg_loss

        trades = context.get('trades', 0)
        wins = context.get('wins', 0)
        losses = context.get('losses', 0)
        post_wr = (wins + CONTEXT_PRIOR_STRENGTH * prior_wr) / (trades + CONTEXT_PRIOR_STRENGTH)
        post_avg_win = (
            context.get('win_sum_pct', 0.0) + CONTEXT_PRIOR_STRENGTH * prior_avg_win
        ) / (wins + CONTEXT_PRIOR_STRENGTH)
        post_avg_loss = (
            context.get('loss_sum_pct', 0.0) + CONTEXT_PRIOR_STRENGTH * prior_avg_loss
        ) / (losses + CONTEXT_PRIOR_STRENGTH)
        post_exp = post_wr * post_avg_win - (1 - post_wr) * post_avg_loss

        multiplier = 1.0 + (post_exp - prior_exp) * CONTEXT_EV_MULTIPLIER
        if trades >= 3 and wins == 0 and context.get('pnl', 0.0) < 0:
            multiplier = min(multiplier, 0.75)
        elif trades >= 3 and wins == trades and context.get('pnl', 0.0) > 0:
            multiplier = max(multiplier, 1.15)

        return max(CONTEXT_MIN_MULT, min(CONTEXT_MAX_MULT, multiplier))
    
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
            "live_tool_stats": self.live_tool_stats,
            "opportunity_scout_pending": getattr(self, 'opportunity_scout_pending', [])[-OPPORTUNITY_SCOUT_MAX_PENDING:],
            "opportunity_scout_stats": getattr(self, 'opportunity_scout_stats', {}),
            "pair_cooldowns": getattr(self, '_pair_cooldowns', {}),
            "pair_daily_stops": getattr(self, '_pair_daily_stops', {}),
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
                            range_span = high_24h - low_24h
                            range_position = ((last - low_24h) / range_span) if range_span > 0 else 0.5
                            normalized_pair = normalize_pair(normalized)
                            quality_ok = (not ENABLE_QUALITY_UNIVERSE or normalized_pair in QUALITY_PAIR_UNIVERSE)
                            
                            if (low_24h > 0 and 
                                vol_usd >= MIN_PAIR_VOLUME_USD and
                                last >= MIN_PAIR_PRICE_USD and
                                quality_ok and
                                not is_pair_globally_blocked(normalized)):
                                volatility = (high_24h - low_24h) / low_24h
                                atr_pct = (range_span / last) if last > 0 else volatility
                                harvest_score = volatility * (0.35 + range_position)
                                if volatility >= VOLATILE_RUNNER_MIN_VOL and range_position < 0.25:
                                    harvest_score *= 0.35
                                pair_volatility[normalized_pair] = {
                                    'volatility': volatility,
                                    'harvest_score': harvest_score,
                                    'range_position': range_position,
                                    'volume_usd': vol_usd,
                                    'price': last,
                                    'high_24h': high_24h,
                                    'low_24h': low_24h,
                                    'atr_pct': atr_pct,
                                    'max_position_usd': vol_usd * MAX_POSITION_PCT_OF_VOLUME
                                }
                        except (KeyError, ValueError, ZeroDivisionError):
                            pass
                except Exception as e:
                    logger.debug(f"Ticker batch error: {e}")
                    continue
            
            if not pair_volatility:
                return
            
            # Sort by long-harvest quality, not raw range alone.
            sorted_pairs = sorted(pair_volatility.items(), key=lambda x: x[1]['harvest_score'], reverse=True)
            
            # Build final list: always-include + top volatile
            selected = set(ALWAYS_INCLUDE)
            for pair, info in sorted_pairs:
                if len(selected) >= MAX_TRADING_PAIRS:
                    break
                selected.add(pair)
            
            # Normalize all pair names
            self._dynamic_pairs = [normalize_pair(p) for p in selected]
            self._vol_pairs_cache_ts = now
            # Also normalize volatility data keys
            self._pair_volatility = {normalize_pair(k): v for k, v in pair_volatility.items()}
            
            # Log what changed
            top5 = sorted_pairs[:5]
            top5_str = ", ".join([
                f"{p} (score={v['harvest_score']:.2f}, vol={v['volatility']:.1%}, pos={v['range_position']:.0%})"
                for p, v in top5
            ])
            logger.info(f"[VOLATILITY] Selected {len(self._dynamic_pairs)} pairs "
                       f"(from {len(pair_volatility)} liquid). Top 5: {top5_str}")
            
        except Exception as e:
            logger.error(f"Volatility scan failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_active_pairs(self) -> list:
        """Return current trading pairs — dynamic if available, fallback to static.
        Always includes pairs we currently hold positions in."""
        if hasattr(self, '_dynamic_pairs') and self._dynamic_pairs:
            pairs = set(self._dynamic_pairs)
        else:
            pairs = set(PAIRS)
        
        # Always include pairs with active positions or pending orders
        for pair in self.active_positions:
            pairs.add(pair)
        for pair in self.pending_limit_orders:
            pairs.add(pair)
        for pair in self.pending_exit_orders:
            pairs.add(pair)
        
        return list(pairs)
    
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        ret_72h = (price - close[-73]) / close[-73] * 100 if len(close) >= 73 else 0
        regime_bucket = self._get_regime_bucket()
        high_24h = np.max(high[-24:]) if len(high) >= 24 else np.max(high)
        low_24h = np.min(low[-24:]) if len(low) >= 24 else np.min(low)
        range_span_24h = high_24h - low_24h
        range_pos_24h = ((price - low_24h) / range_span_24h) if range_span_24h > 0 else 0.5
        breakout_ref_high_24h = np.max(high[-25:-1]) if len(high) >= 25 else high_24h
        breakout_pct_24h = max(0.0, (price / breakout_ref_high_24h) - 1) if breakout_ref_high_24h > 0 else 0.0
        breakout_volume_avg_24h = np.mean(volume[-25:-1]) if len(volume) >= 25 else (np.mean(volume[:-1]) if len(volume) > 1 else volume[-1])
        breakout_volume_ratio_24h = (volume[-1] / breakout_volume_avg_24h) if breakout_volume_avg_24h > 0 else 0.0
        breadth_4h = getattr(self, '_bullish_4h_pct', 50) / 100.0
        short_dominance = self._safe_float(getattr(self, '_market_short_pressure', {}).get('dominance', 0.0))
        rebound_1h = (price - close[-2]) / close[-2] * 100 if len(close) >= 2 else 0
        ema_bearish = not np.isnan(ema5[-1]) and not np.isnan(ema13[-1]) and ema5[-1] < ema13[-1]
        collapse_regime = (
            cur_atr_pct >= VOL_QUALITY_COLLAPSE_ATR_PCT and
            range_pos_24h < VOL_QUALITY_RANGE_FLOOR and
            ema_bearish and
            (
                ret_24h <= VOL_QUALITY_CRASH_24H_PCT or
                (ret_8h <= VOL_QUALITY_CRASH_8H_PCT and ret_4h <= -5)
            )
        )
        rebound_confirmed = (
            close[-1] > df['open'].iloc[-1] and
            rebound_1h > VOL_QUALITY_REBOUND_1H_PCT and
            range_pos_24h > 0.25
        )
        
        # MTF: Get higher timeframe context for confirmation
        htf_context = self.get_htf_context(data) if ENABLE_MTF else {"htf_available": False}
        
        # Helper function to apply score adjustment from UPGRADE 7
        def adjust_score(tool: str, base_score: float) -> float:
            score = base_score
            if tool in self.tool_stats:
                score *= self.tool_stats[tool].get("score_adj", 1.0)
            score *= self._get_contextual_score_multiplier(pair, tool, regime_bucket)
            forward_direction = 'short' if tool in BULL_GREED_TOOLS else 'long'
            forward_mult, _, _ = self._get_forward_tool_score_adjustment(tool, forward_direction)
            score *= forward_mult
            return score
        
        # MTF: Helper function to apply multi-timeframe confirmation
        def apply_mtf_confirmation(tool: str, direction: str, base_score: float) -> float:
            if not ENABLE_MTF or not htf_context.get("htf_available", False):
                return base_score  # No HTF data, use original score
            
            # Crash signals bypass MTF (they're counter-trend by nature)
            crash_signals = {
                'crash_buy', 'mega_crash', 'crash_neg_ac', 
                'blood_in_streets', 'crash_mean_revert', 'panic_reversal_absorption',
                'mega_pump_sell_t1'
                # REMOVED: volatile_oversold (0% WR), quick_crash (33% WR) — killed tools
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

        def allow_weak_long(tool: str) -> bool:
            weak_long_tools = {
                'vpin_dip', 'vpin_toxic', 'btc_alt_spread',
                'deep_dip_8h', 'dip_buy_5pct'
            }
            if tool not in weak_long_tools:
                return True
            return not collapse_regime or rebound_confirmed
        
        # ===== CRASH/BEAR SIGNALS (LONG) - 15 tools =====
        
        # KILLED: 0% WR live (1 trade, -$10.23). ATR-based entries too early in crashes.
        # 1. volatile_oversold: atr_pct>3 AND rsi7<25 → LONG | WR_8h=73.8%, Ret_8h=+2.07%
        # if cur_atr_pct > 3 and cur_rsi < 25:
        #     base_score = cur_atr_pct * (25 - cur_rsi) * 0.5  # 30-50 range
        #     score = adjust_score('volatile_oversold', base_score)
        #     score = apply_mtf_confirmation('volatile_oversold', 'long', score)  # MTF confirmation
        #     signals.append(({
        #         'pair': pair, 'tool': 'volatile_oversold', 'direction': 'long',
        #         'hold': 8, 'sl_pct': 0.08,
        #         'reason': f"VOLATILE OVERSOLD: ATR={cur_atr_pct:.1f}%, RSI={cur_rsi:.1f}"
        #     }, score))
        
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
        
        # KILLED: 33% WR live, -$8.84. Knife catcher.
        # 6. quick_crash: ret_8h<-10 → LONG (8h hold only) | WR_8h=59.1%, Ret_8h=+0.98%
        # if ret_8h < -10:
        #     base_score = abs(ret_8h) * 2  # 20-30 range
        #     score = adjust_score('quick_crash', base_score)
        #     score = apply_mtf_confirmation('quick_crash', 'long', score)  # MTF confirmation
        #     signals.append(({
        #         'pair': pair, 'tool': 'quick_crash', 'direction': 'long',
        #         'hold': 8, 'sl_pct': 0.07,
        #         'reason': f"QUICK CRASH: {ret_8h:.1f}% drop 8h"
        #     }, score))
        
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

        # 7b. panic_reversal_absorption: strict forced-seller reversal setup.
        # Walk-forward lab 2026-04-28: full +8.39%, PF 1.31, test PF 1.19, test DD 3.93%.
        if len(close) >= 25 and ret_24h <= PANIC_REVERSAL_DROP_24H and cur_rsi <= PANIC_REVERSAL_MAX_RSI:
            candle_open = float(df['open'].iloc[-1])
            candle_range = high[-1] - low[-1]
            lower_wick_ratio = 0.0
            if candle_range > 0:
                lower_wick_ratio = (min(candle_open, close[-1]) - low[-1]) / candle_range

            btc_context_ok = pair == "XBTUSD"
            btc_ret24_context = ret_24h if pair == "XBTUSD" else 0.0
            if not btc_context_ok and "XBTUSD" in self._price_cache:
                btc_prices = self._price_cache["XBTUSD"]
                if len(btc_prices) >= 25 and btc_prices[-25] > 0:
                    btc_ret24_context = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                    btc_context_ok = btc_ret24_context >= PANIC_REVERSAL_BTC_CRASH_FLOOR

            if (
                btc_context_ok and
                close[-1] > candle_open and
                breakout_volume_ratio_24h >= PANIC_REVERSAL_MIN_VOLUME_RATIO and
                lower_wick_ratio >= PANIC_REVERSAL_MIN_LOWER_WICK_RATIO
            ):
                base_score = (
                    abs(ret_24h) +
                    (PANIC_REVERSAL_MAX_RSI - cur_rsi) +
                    breakout_volume_ratio_24h * 4.0 +
                    lower_wick_ratio * 10.0
                )
                score = adjust_score('panic_reversal_absorption', base_score)
                score = apply_mtf_confirmation('panic_reversal_absorption', 'long', score)
                signals.append(({
                    'pair': pair, 'tool': 'panic_reversal_absorption', 'direction': 'long',
                    'hold': 24, 'sl_pct': 0.04,
                    'reason': (
                        f"PANIC REVERSAL: {ret_24h:.1f}% 24h drop, RSI={cur_rsi:.1f}, "
                        f"vol={breakout_volume_ratio_24h:.1f}x, wick={lower_wick_ratio:.0%}, "
                        f"BTC24={btc_ret24_context:.1f}%"
                    ),
                    '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                    '_lower_wick_ratio': round(lower_wick_ratio, 4),
                }, score))
        
        # 8. vpin_dip: ret_8h<-5 AND VPIN>0.5 → LONG | WR_8h=58.8%, Ret_8h=+0.73%
        if ret_8h < -5 and allow_weak_long('vpin_dip'):
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
        if -10 < ret_8h < -8 and allow_weak_long('deep_dip_8h'):
            base_score = abs(ret_8h) * 1.5  # 12-15 range
            score = adjust_score('deep_dip_8h', base_score)
            score = apply_mtf_confirmation('deep_dip_8h', 'long', score)  # MTF confirmation
            signals.append(({
                'pair': pair, 'tool': 'deep_dip_8h', 'direction': 'long',
                'hold': 8, 'sl_pct': 0.04,  # Tightened from 0.05 — wins avg +2-3%, losses were hitting -8%
                'reason': f"DEEP DIP 8h: {ret_8h:.1f}% drop"
            }, score))
        
        # KILLED: 0% WR live (4 trades, -$4.33). Entropy too noisy for real markets.
        # 12. entropy_dip: entropy<2.5 AND ret_4h<-2 → LONG | WR_8h=52.8%, Ret_8h=+0.45%
        # if ret_4h < -2:
        #     entropy = self.calc_entropy(close[-30:]) if len(close) >= 30 else 3.0
        #     if entropy < 2.5:
        #         score = adjust_score('entropy_dip', (2.5 - entropy) * abs(ret_4h) * 2)  # 10-20 range
        #         signals.append(({
        #             'pair': pair, 'tool': 'entropy_dip', 'direction': 'long',
        #             'hold': 8, 'sl_pct': 0.04,
        #             'reason': f"ENTROPY DIP: entropy={entropy:.2f}, {ret_4h:.1f}% drop 4h"
        #         }, score))
        
        # 13. vpin_toxic: VPIN>0.7 AND red candle → LONG | WR_8h=53.8%, Ret_8h=+0.45%
        if len(df) >= 2 and allow_weak_long('vpin_toxic'):
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
        if pair != "XBTUSD" and "XBTUSD" in self._price_cache and cur_rsi < 35 and allow_weak_long('btc_alt_spread'):
            btc_prices = self._price_cache["XBTUSD"]
            if len(btc_prices) >= 25 and len(close) >= 25:
                btc_ret24 = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                if btc_ret24 - ret_24h > 3:  # BTC outperforming by 3%+
                    score = adjust_score('btc_alt_spread', (btc_ret24 - ret_24h) * (35 - cur_rsi) * 0.1)  # 10-15 range
                    signals.append(({
                        'pair': pair, 'tool': 'btc_alt_spread', 'direction': 'long',
                        'hold': 24, 'sl_pct': 0.04,  # Tightened from 0.05 — wins avg +2-3%, losses were hitting -8%
                        'reason': f"BTC ALT SPREAD: BTC {btc_ret24:+.1f}% vs {pair} {ret_24h:+.1f}%, RSI={cur_rsi:.1f}"
                    }, score))
        
        # 15. quick_dip: KILLED — negative lifetime PnL (-$54). Catches falling knives.
        # if ret_4h < -5:
        #     ...disabled...
        
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
        except Exception:
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
        except Exception:
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
                recent_returns = np.diff(close[-30:]) / close[-30:-1]
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
        except Exception:
            pass
        
        # 30. dip_buy_5pct: ret_4h<-5 → LONG | WR_8h=52.7%, Ret_8h=+0.11%
        if ret_4h < -5 and allow_weak_long('dip_buy_5pct'):
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
                H = np.clip(np.polyfit(hurst_lv, hurst_rs, 1)[0], 0, 1)
            
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
                    'reason': f"UPTREND BUY: 50>200 SMA, ret1w={swing_ret1w*100:.1f}%",
                    '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                    '_breakout_pct': round(breakout_pct_24h, 4),
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

        # 35b. major_pair_breakout: majors can follow 24h breakouts when breadth and volume confirm.
        if pair in MAJOR_BREAKOUT_PAIRS and len(close) >= 72:
            if (
                price > sma50[-1] and
                cur_rsi >= 48 and cur_rsi <= 78 and
                ret_24h > 1.0 and
                breakout_pct_24h >= MAJOR_BREAKOUT_MIN_BREAKOUT_PCT and
                breakout_volume_ratio_24h >= MAJOR_BREAKOUT_MIN_VOLUME_RATIO and
                breadth_4h >= MAJOR_BREAKOUT_MIN_BULLISH_PCT / 100.0 and
                short_dominance <= MAJOR_BREAKOUT_MAX_SHORT_DOMINANCE
            ):
                base_score = (
                    breakout_pct_24h * 4000 +
                    breakout_volume_ratio_24h * 6 +
                    max(ret_24h, 0) * 0.8 +
                    breadth_4h * 12
                )
                score = adjust_score('major_pair_breakout', base_score)
                score = apply_mtf_confirmation('major_pair_breakout', 'long', score)
                signals.append(({
                    'pair': pair, 'tool': 'major_pair_breakout', 'direction': 'long',
                    'hold': 336, 'sl_pct': 0.08,
                    'reason': (
                        f"MAJOR BREAKOUT: +{breakout_pct_24h*100:.1f}% above 24h high, "
                        f"vol {breakout_volume_ratio_24h:.1f}x, breadth {breadth_4h*100:.0f}%"
                    ),
                    '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                    '_breakout_pct': round(breakout_pct_24h, 4),
                }, score))

        # 35c. market_breadth_recovery: broad green-day / post-red-day recovery participation.
        # This is intentionally smaller and faster than the bull swing tools. It lets the bot
        # participate when F&G is lagging in fear but breadth, BTC context, and the asset's own
        # tape are all improving.
        if MARKET_BREADTH_RECOVERY_ENABLED and len(close) >= 73:
            btc_context_ok = pair == "XBTUSD"
            btc_ret24_context = ret_24h if pair == "XBTUSD" else 0.0
            btc_ret4_context = ret_4h if pair == "XBTUSD" else 0.0
            if not btc_context_ok and "XBTUSD" in self._price_cache:
                btc_prices = self._price_cache["XBTUSD"]
                if len(btc_prices) >= 25 and btc_prices[-25] > 0:
                    btc_ret24_context = (btc_prices[-1] - btc_prices[-25]) / btc_prices[-25] * 100
                    btc_ret4_context = (btc_prices[-1] - btc_prices[-5]) / btc_prices[-5] * 100 if len(btc_prices) >= 5 and btc_prices[-5] > 0 else 0.0
                    btc_context_ok = btc_ret24_context >= MARKET_BREADTH_RECOVERY_BTC_MIN_RET_24H and btc_ret4_context >= -0.25

            broad_followthrough = ret_24h >= max(2.0, MARKET_BREADTH_RECOVERY_MIN_RET_24H)
            post_red_rebound = ret_72h <= -2.0 and ret_4h >= 1.0
            htf_not_bearish = not htf_context.get("htf_available") or htf_context.get("trend_4h") != "bearish" or htf_context.get("momentum_4h", 0.0) > 0

            if (
                btc_context_ok and
                breadth_4h >= MARKET_BREADTH_RECOVERY_MIN_BULLISH_PCT / 100.0 and
                short_dominance <= MARKET_BREADTH_RECOVERY_MAX_SHORT_DOMINANCE and
                ret_4h >= MARKET_BREADTH_RECOVERY_MIN_RET_4H and
                ret_24h >= MARKET_BREADTH_RECOVERY_MIN_RET_24H and
                breakout_volume_ratio_24h >= MARKET_BREADTH_RECOVERY_MIN_VOLUME_RATIO and
                range_pos_24h >= MARKET_BREADTH_RECOVERY_MIN_RANGE_POS and
                cur_rsi >= 42 and cur_rsi <= MARKET_BREADTH_RECOVERY_MAX_RSI and
                not np.isnan(sma50[-1]) and
                price > sma50[-1] * 0.97 and
                htf_not_bearish and
                (broad_followthrough or post_red_rebound)
            ):
                base_score = (
                    ret_24h * 2.0 +
                    ret_4h * 3.0 +
                    max(abs(ret_72h), 0.0) * (0.35 if post_red_rebound else 0.0) +
                    breadth_4h * 10.0 +
                    breakout_volume_ratio_24h * 4.0 +
                    range_pos_24h * 6.0 +
                    max(btc_ret24_context, 0.0)
                )
                score = adjust_score('market_breadth_recovery', base_score)
                score = apply_mtf_confirmation('market_breadth_recovery', 'long', score)
                signals.append(({
                    'pair': pair, 'tool': 'market_breadth_recovery', 'direction': 'long',
                    'hold': 72, 'sl_pct': 0.06,
                    'reason': (
                        f"MARKET RECOVERY: ret24={ret_24h:.1f}%, ret4={ret_4h:.1f}%, "
                        f"ret72={ret_72h:.1f}%, breadth={breadth_4h*100:.0f}%, "
                        f"BTC24={btc_ret24_context:.1f}%, vol={breakout_volume_ratio_24h:.1f}x"
                    ),
                    '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                    '_breakout_pct': round(breakout_pct_24h, 4),
                    '_market_breadth': round(breadth_4h, 4),
                    '_btc_ret_24h': round(btc_ret24_context, 4),
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
                        'reason': f"BTC LEADING: BTC +{btl_btc1w*100:.1f}%, alt lag {btl_lag*100:.1f}%",
                        '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                        '_breakout_pct': round(breakout_pct_24h, 4),
                    }, score))

            # 37. scout-only opportunity families. These create real signal objects,
            # but the evidence gate blocks them until paper forward proof graduates
            # them to scout_watch. After that they can only pilot at tiny size.
            if OPPORTUNITY_SCOUT_ENABLED and len(close) >= 73:
                htf_trend = htf_context.get("trend_4h", "neutral") if htf_context else "neutral"
                htf_momentum = float(htf_context.get("momentum_4h", 0.0) or 0.0) if htf_context else 0.0
                current_open = float(df['open'].iloc[-1]) if 'open' in df.columns else price
                candle_span = float(high[-1] - low[-1]) if high[-1] > low[-1] else 0.0
                lower_wick_ratio = (
                    (min(current_open, close[-1]) - low[-1]) / candle_span
                    if candle_span > 0 else 0.0
                )

                if (
                    ret_4h >= 0.8 and ret_24h >= 1.5 and
                    breakout_volume_ratio_24h >= 1.20 and
                    range_pos_24h >= 0.58 and
                    cur_rsi >= 48 and cur_rsi <= 72 and
                    not np.isnan(sma50[-1]) and price > sma50[-1] * 0.98
                ):
                    base_score = (
                        ret_4h * 3.0 + ret_24h * 1.4 +
                        breakout_volume_ratio_24h * 4.0 + range_pos_24h * 6.0 +
                        max(breadth_4h - 0.45, 0.0) * 10.0
                    )
                    score = adjust_score('scout_volume_continuation', base_score)
                    score = apply_mtf_confirmation('scout_volume_continuation', 'long', score)
                    signals.append(({
                        'pair': pair, 'tool': 'scout_volume_continuation', 'direction': 'long',
                        'hold': 72, 'sl_pct': 0.055,
                        'reason': (
                            f"SCOUT VOLUME CONTINUATION: ret4={ret_4h:.1f}%, ret24={ret_24h:.1f}%, "
                            f"vol={breakout_volume_ratio_24h:.1f}x, range={range_pos_24h:.0%}"
                        ),
                        '_scout_candidate': True,
                        '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                        '_breakout_pct': round(breakout_pct_24h, 4),
                    }, score))

                if (
                    (htf_trend == "bullish" or htf_momentum > 0.4) and
                    ret_72h >= 1.0 and ret_24h >= -1.5 and rebound_1h >= 0.15 and
                    range_pos_24h >= 0.32 and range_pos_24h <= 0.72 and
                    cur_rsi >= 38 and cur_rsi <= 56 and
                    breakout_volume_ratio_24h >= 0.80 and
                    not np.isnan(sma50[-1]) and price > sma50[-1] * 0.95
                ):
                    base_score = (
                        max(ret_72h, 0.0) * 0.8 + rebound_1h * 5.0 +
                        (56 - cur_rsi) * 0.35 + range_pos_24h * 5.0 +
                        max(htf_momentum, 0.0) * 1.2
                    )
                    score = adjust_score('scout_trend_pullback', base_score)
                    score = apply_mtf_confirmation('scout_trend_pullback', 'long', score)
                    signals.append(({
                        'pair': pair, 'tool': 'scout_trend_pullback', 'direction': 'long',
                        'hold': 96, 'sl_pct': 0.06,
                        'reason': (
                            f"SCOUT TREND PULLBACK: htf={htf_trend}, mom4h={htf_momentum:.1f}%, "
                            f"rsi={cur_rsi:.0f}, rebound={rebound_1h:.1f}%"
                        ),
                        '_scout_candidate': True,
                        '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                        '_breakout_pct': round(breakout_pct_24h, 4),
                    }, score))

                if (
                    ret_72h <= -4.0 and (ret_4h >= 0.8 or ret_24h >= 0.5) and
                    lower_wick_ratio >= 0.35 and
                    range_pos_24h >= 0.28 and range_pos_24h <= 0.82 and
                    cur_rsi >= 34 and cur_rsi <= 62 and
                    breakout_volume_ratio_24h >= 1.00
                ):
                    base_score = (
                        abs(ret_72h) * 0.65 + max(ret_4h, 0.0) * 3.0 +
                        max(ret_24h, 0.0) * 1.7 + lower_wick_ratio * 7.0 +
                        breakout_volume_ratio_24h * 3.0
                    )
                    score = adjust_score('scout_reversal_followthrough', base_score)
                    score = apply_mtf_confirmation('scout_reversal_followthrough', 'long', score)
                    signals.append(({
                        'pair': pair, 'tool': 'scout_reversal_followthrough', 'direction': 'long',
                        'hold': 72, 'sl_pct': 0.065,
                        'reason': (
                            f"SCOUT REVERSAL FOLLOWTHROUGH: ret72={ret_72h:.1f}%, ret4={ret_4h:.1f}%, "
                            f"wick={lower_wick_ratio:.0%}, vol={breakout_volume_ratio_24h:.1f}x"
                        ),
                        '_scout_candidate': True,
                        '_volume_ratio': round(breakout_volume_ratio_24h, 4),
                        '_breakout_pct': round(breakout_pct_24h, 4),
                        '_lower_wick_ratio': round(lower_wick_ratio, 4),
                    }, score))
        
        # Enrich all signals with HTF context for journal logging
        for sig, sc in signals:
            sig['_htf_context'] = dict(htf_context) if htf_context else {}
            sig['_range_pos_24h'] = round(range_pos_24h, 4)
            sig['_atr_pct'] = round(cur_atr_pct, 4)
            sig['_collapse_gate'] = (
                'rebound_confirmed'
                if collapse_regime and rebound_confirmed else
                'collapse'
                if collapse_regime else
                'normal'
            )
        
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
    
    def _get_exit_params(self, pair: str, tool: str, price: float, market_data_entry: dict) -> tuple:
        """Get exit parameters for validated tools only."""
        # Mean reversion strategies — TP at 8-10%
        MEAN_REVERSION = {
            'crash_neg_ac', 'crash_mean_revert', 
            'blood_in_streets', 'vpin_dip', 'vpin_toxic',
            'btc_alt_spread'
            # REMOVED: volatile_oversold (killed), entropy_dip (killed)
        }
        
        # Crash buy strategies — TP at 10-12%
        CRASH_BUY = {
            'crash_buy', 'mega_crash', 'flash_crash',
            'market_panic_70', 'deep_dip_8h'
            # REMOVED: quick_crash (killed)
        }
        
        # Short strategies — TP at 6%
        SHORT_TOOLS = {
            'mega_pump_sell_t1', 'rsi_pump_8h', 'falling_wedge_short', 'greed_short_t2',
            'thursday_short', 'mega_pump_sell_t2', 'distribution_short', 'late_us_short',
            'rsi_pump_12h', 'ema_cross_short', 'rsi_pump_fat_tail', 'entropy_short',
            'alt_btc_revert_t3'
        }
        
        # Dip buy — TP at 6%
        DIP_BUY = {'dip_buy_5pct'}  # quick_dip removed
        
        # Neutral tools — TP at 6%
        NEUTRAL = {'month_start_long', 'market_breadth_recovery'}

        # Walk-forward panic reversal absorption — TP at 7%, per 2026-04-28 lab.
        PANIC_REVERSAL = {'panic_reversal_absorption'}
        
        # Bull swing tools — 15% trailing stop, no fixed TP (let winners run)
        BULL_SWING = {
            'buy_weekly_green', 'buy_breakout_simple',
            'simple_buy_uptrend', 'buy_btc_leading', 'major_pair_breakout'
        }
        
        # Bull momentum tools — 8% trailing stop
        BULL_MOMENTUM = {'accumulation_breakout', 'hurst_trend_long'}
        
        if tool in BULL_SWING:
            exit_params = ('trailing', None, 0.15, None)  # 15% trailing stop
        elif tool in BULL_MOMENTUM:
            exit_params = ('trailing', None, 0.08, None)  # 8% trailing stop
        elif tool in MEAN_REVERSION:
            exit_params = ('fixed_tp', 0.08, None, None)  # 8% TP
        elif tool in CRASH_BUY:
            exit_params = ('fixed_tp', 0.10, None, None)  # 10% TP
        elif tool in SHORT_TOOLS:
            exit_params = ('fixed_tp', 0.06, None, None)  # 6% TP
        elif tool in DIP_BUY:
            exit_params = ('fixed_tp', 0.06, None, None)  # 6% TP
        elif tool in NEUTRAL:
            exit_params = ('fixed_tp', 0.06, None, None)  # 6% TP
        elif tool in PANIC_REVERSAL:
            exit_params = ('fixed_tp', 0.07, None, None)  # 7% TP
        else:
            exit_params = ('trailing', None, 0.10, None)  # Default trailing stop

        exit_mode, take_profit_pct, trailing_stop_pct, extra = exit_params
        policy = self._get_pair_policy(pair)
        if take_profit_pct:
            take_profit_pct *= float(policy.get('fixed_tp_multiplier', 1.0))
        if trailing_stop_pct:
            trailing_stop_pct *= float(policy.get('trailing_stop_multiplier', 1.0))
        return (exit_mode, take_profit_pct, trailing_stop_pct, extra)
    
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
                           'simple_buy_uptrend', 'buy_btc_leading', 'major_pair_breakout',
                           'accumulation_breakout', 'hurst_trend_long'}
        if pos['direction'] != 'long' or pos.get('tool') not in bull_swing_tools:
            return None
        fng = self.get_fng()
        if fng < 30:
            return f"Regime exit: F&G={fng} < 30, bull tool in fear"
        return None

    def _is_trend_leader_tool(self, tool: str) -> bool:
        """Trend leaders are the longer-horizon bull momentum and swing tools."""
        return tool in TREND_LEADER_TOOLS

    def _get_trend_leader_reserve(self) -> int:
        """Reserve more capacity for trend leaders only when the broader regime supports them."""
        bullish_pct = getattr(self, '_bullish_4h_pct', 50)
        fng = getattr(self, 'current_fng', 50)
        if bullish_pct >= 65 and fng >= 50:
            return MAX_TREND_LEADER_RESERVED_SLOTS
        if bullish_pct >= 50 and fng >= 35:
            return 1
        return 0

    def _count_open_trend_leaders(self) -> int:
        """Count currently open trend-leader positions."""
        return sum(
            1 for pos in self.active_positions.values()
            if self._is_trend_leader_tool(pos.get('tool', ''))
        )

    def _find_replacement_candidate(self, incoming_signal: dict, incoming_score: float,
                                    market_data: dict) -> Optional[dict]:
        """Find the weakest open position worth rotating out for a materially stronger signal."""
        incoming_direction = incoming_signal.get('direction')
        incoming_is_trend = self._is_trend_leader_tool(incoming_signal.get('tool', ''))
        candidates = []

        for pair, pos in self.active_positions.items():
            if pair in self.pending_exit_orders or pos.get('_pending_exit'):
                continue
            if pos.get('entry_bar', self.current_bar) >= self.current_bar:
                continue
            if pos.get('direction') != incoming_direction:
                continue
            if pos.get('_runner_mode') and pos.get('_partial_closed'):
                continue

            current_price = market_data.get(pair, {}).get('price')
            entry_price = pos.get('entry_price', 0)
            if not current_price or entry_price <= 0:
                continue

            pnl_pct = (
                (current_price - entry_price) / entry_price
                if pos.get('direction') == 'long' else
                (entry_price - current_price) / entry_price
            )
            if pnl_pct > REPLACEMENT_PROTECT_PNL_PCT:
                continue

            pos_is_trend = self._is_trend_leader_tool(pos.get('tool', ''))
            if pos_is_trend and not incoming_is_trend:
                continue

            edge_needed = (
                TREND_REPLACEMENT_SCORE_EDGE
                if incoming_is_trend and not pos_is_trend else
                REPLACEMENT_SCORE_EDGE
            )
            score_floor = max(REPLACEMENT_MIN_SCORE, pos.get('score', 0) * edge_needed)
            if incoming_score < score_floor:
                continue

            bars_held = self.current_bar - pos.get('entry_bar', self.current_bar)
            rank = (
                1 if pos_is_trend else 0,
                1 if pnl_pct > 0 else 0,
                pnl_pct,
                pos.get('score', 0),
                -bars_held,
            )
            candidates.append((rank, pair, current_price, pnl_pct, score_floor))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, pair, current_price, pnl_pct, score_floor = candidates[0]
        return {
            'pair': pair,
            'exit_price': current_price,
            'pnl_pct': pnl_pct,
            'score_floor': score_floor,
        }

    def _build_market_short_pressure(self, signals: List[Tuple[dict, float]]) -> dict:
        """Summarize market-wide bearish pressure from blocked short setups."""
        summary = {
            'mode': 'normal',
            'label': 'normal',
            'active_pct': 1.0,
            'min_long_score': 0.0,
            'short_signals': 0,
            'short_pairs': 0,
            'top3_avg': 0.0,
            'dominance': 0.0,
        }

        short_scores = []
        short_pairs = set()
        long_scores = []
        for signal, score in signals:
            if signal.get('direction') == 'short':
                short_scores.append(score)
                short_pairs.add(signal['pair'])
            elif signal.get('direction') == 'long':
                long_scores.append(score)

        if not short_scores:
            return summary

        short_scores.sort(reverse=True)
        short_pairs_count = len(short_pairs)
        top3_avg = sum(short_scores[:3]) / min(3, len(short_scores))
        short_sum = sum(short_scores)
        long_sum = sum(long_scores)
        dominance = short_sum / max(long_sum, 1.0)

        summary.update({
            'short_signals': len(short_scores),
            'short_pairs': short_pairs_count,
            'top3_avg': top3_avg,
            'dominance': dominance,
        })

        if (short_pairs_count >= MARKET_BEAR_RISK_OFF_PAIRS and
            top3_avg >= MARKET_BEAR_RISK_OFF_SCORE and
            dominance >= MARKET_BEAR_RISK_OFF_DOMINANCE):
            summary.update({
                'mode': 'risk_off',
                'label': 'risk-off cash mode',
                'active_pct': MARKET_BEAR_RISK_OFF_ACTIVE_PCT,
                'min_long_score': MARKET_BEAR_RISK_OFF_SCORE,
            })
        elif (short_pairs_count >= MARKET_BEAR_DEFENSIVE_PAIRS and
              top3_avg >= MARKET_BEAR_DEFENSIVE_SCORE and
              dominance >= MARKET_BEAR_DEFENSIVE_DOMINANCE):
            summary.update({
                'mode': 'defensive',
                'label': 'defensive cash mode',
                'active_pct': MARKET_BEAR_DEFENSIVE_ACTIVE_PCT,
                'min_long_score': MARKET_BEAR_DEFENSIVE_SCORE,
            })
        elif (short_pairs_count >= MARKET_BEAR_CAUTION_PAIRS and
              top3_avg >= MARKET_BEAR_CAUTION_SCORE and
              dominance >= MARKET_BEAR_CAUTION_DOMINANCE):
            summary.update({
                'mode': 'caution',
                'label': 'cautious cash mode',
                'active_pct': MARKET_BEAR_CAUTION_ACTIVE_PCT,
                'min_long_score': MARKET_BEAR_CAUTION_SCORE,
            })

        return summary

    def _build_short_pressure_map(self, signals: List[Tuple[dict, float]]) -> dict:
        """Aggregate blocked short setups into same-pair bearish pressure."""
        pressure = {}
        for signal, score in signals:
            if signal.get('direction') != 'short':
                continue
            pair = signal['pair']
            info = pressure.setdefault(pair, {
                'score': 0.0,
                'count': 0,
                'tool': signal['tool'],
            })
            info['count'] += 1
            if score > info['score']:
                info['score'] = score
                info['tool'] = signal['tool']

        for info in pressure.values():
            stack_mult = 1.0 + min(0.6, 0.2 * max(0, info['count'] - 1))
            info['effective_score'] = info['score'] * stack_mult

        return pressure

    def _check_short_pressure_exit(self, pair: str, pos: dict, current_price: float) -> Optional[str]:
        """Use blocked short setups as defensive overlays for long-only trading."""
        if pos.get('direction') != 'long':
            return None

        pressure = getattr(self, '_short_pressure_by_pair', {}).get(pair)
        if not pressure:
            return None

        effective_score = pressure.get('effective_score', pressure.get('score', 0.0))
        if effective_score < SHORT_PRESSURE_TIGHTEN_SCORE:
            return None

        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] if pos['entry_price'] > 0 else 0.0

        if effective_score >= SHORT_PRESSURE_EXIT_SCORE and pnl_pct <= 0.01:
            return f"Bearish overlay: {pressure['tool']} score {effective_score:.1f}"

        if pnl_pct > 0:
            prior_anchor = pos.get('_trail_from', pos['entry_price'])
            pos['_trail_from'] = max(prior_anchor, current_price)
            pos['sl_pct'] = min(pos.get('sl_pct', 0.05), 0.02 if effective_score >= SHORT_PRESSURE_EXIT_SCORE else 0.03)
            pos['_pump_tighten'] = True

        return None

    def _check_conviction_decay_exit(self, pair: str, pos: dict, data: dict, current_price: float) -> Optional[str]:
        """Exit weak longs early when conviction disappears before the full hold window expires."""
        if pos.get('direction') != 'long':
            return None
        if pos.get('_partial_closed') or pos.get('_runner_mode'):
            return None

        entry_price = pos.get('entry_price', 0.0)
        if entry_price <= 0:
            return None

        pnl_pct = (current_price - entry_price) / entry_price
        policy = self._get_pair_policy(pair)
        safe_pnl_pct = float(policy.get('conviction_decay_safe_pnl_pct', CONVICTION_DECAY_SAFE_PNL_PCT))
        if pnl_pct > safe_pnl_pct:
            return None

        entry_time = pos.get('entry_time', None)
        if entry_time:
            elapsed_hours = (datetime.now(timezone.utc).timestamp() - entry_time) / 3600
        else:
            bars_held = self.current_bar - pos.get('entry_bar', self.current_bar)
            elapsed_hours = bars_held * (CHECK_INTERVAL / 3600)

        planned_hold_hours = max(float(pos.get('hold', 0.0) or 0.0), CHECK_INTERVAL / 3600)
        min_hold_hours = max(
            float(policy.get('conviction_decay_min_hold_hours', CONVICTION_DECAY_MIN_HOURS)),
            planned_hold_hours * float(policy.get('conviction_decay_min_hold_fraction', CONVICTION_DECAY_MIN_HOLD_FRACTION)),
        )
        if elapsed_hours < min_hold_hours:
            return None

        last_check_bar = pos.get('_conviction_decay_check_bar')
        check_bars = int(policy.get('conviction_decay_check_bars', CONVICTION_DECAY_CHECK_BARS))
        if last_check_bar is not None and self.current_bar - last_check_bar < check_bars:
            return None
        pos['_conviction_decay_check_bar'] = self.current_bar

        rescan = self.scan_signals(pair, data)
        same_direction = [
            (signal, signal_score)
            for signal, signal_score in rescan
            if signal['direction'] == pos['direction'] and self._get_pair_policy_rejection(signal, signal_score) is None
        ]
        pressure = getattr(self, '_short_pressure_by_pair', {}).get(pair)
        effective_pressure = pressure.get('effective_score', pressure.get('score', 0.0)) if pressure else 0.0
        exit_pnl_pct = float(policy.get('conviction_decay_exit_pnl_pct', CONVICTION_DECAY_EXIT_PNL_PCT))

        if not same_direction:
            if effective_pressure >= SHORT_PRESSURE_TIGHTEN_SCORE and pnl_pct <= 0.005:
                return f"Conviction decay + bearish overlay ({effective_pressure:.1f})"
            if pnl_pct <= exit_pnl_pct:
                return f"Conviction decay — no long support after {elapsed_hours:.1f}h"
            return None

        entry_score = float(pos.get('score', 0.0) or 0.0)
        keep_threshold = max(
            REPLACEMENT_MIN_SCORE,
            entry_score * float(policy.get('conviction_decay_entry_score_keep_fraction', CONVICTION_DECAY_ENTRY_SCORE_KEEP_FRACTION)),
        )
        same_tool_scores = [score for signal, score in same_direction if signal['tool'] == pos.get('tool')]
        if any(score >= keep_threshold for score in same_tool_scores):
            return None

        best_same_direction_score = max(score for _, score in same_direction)
        if best_same_direction_score >= keep_threshold and pnl_pct > exit_pnl_pct:
            return None

        if effective_pressure >= SHORT_PRESSURE_TIGHTEN_SCORE and pnl_pct <= 0.005:
            return f"Conviction decay + bearish overlay ({effective_pressure:.1f})"
        if pnl_pct <= exit_pnl_pct:
            return (
                f"Conviction decay — {pos.get('tool', 'unknown')} support faded "
                f"to {best_same_direction_score:.1f} after {elapsed_hours:.1f}h"
            )

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
        5. NOT in pair cooldown/blacklist
        6. NOT past daily drawdown limit
        7. NOT past total account drawdown limit
        """
        DCA_TOOLS = {'crash_buy', 'mega_crash', 'blood_in_streets',
                    'crash_neg_ac', 'crash_mean_revert', 'flash_crash',
                    'market_panic_70', 'deep_dip_8h', 'vpin_dip'}
        # REMOVED from DCA: volatile_oversold, quick_crash, entropy_dip — killed tools
        
        if pos['tool'] not in DCA_TOOLS:
            return
        if pos.get('_dca_done'):
            return
        if pos['direction'] != 'long':
            return
        if self._pair_is_globally_blocked(pair):
            logger.warning(f"[DCA BLOCKED] {pair} disabled for new buy-side adds")
            return

        policy = self._get_pair_policy(pair)
        if not policy.get('allow_dca', True):
            logger.warning(f"[DCA BLOCKED] {pair} disabled for new buy-side adds")
            return
        
        # === SAFETY CHECKS (fixes bypass bug) ===
        
        # Check pair cooldown/blacklist
        if hasattr(self, '_pair_cooldowns') and pair in self._pair_cooldowns:
            cd = self._pair_cooldowns[pair]
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts < cd['expires']:
                logger.info(f"[DCA BLOCKED] {pair} in cooldown/blacklist — no DCA")
                return
        
        # Check daily drawdown limit
        daily_pnl_pct = self._daily_stats["pnl"] / max(self._daily_stats.get("start_balance", self.total_balance), 1)
        if daily_pnl_pct < -DAILY_MAX_LOSS_PCT:
            logger.info(f"[DCA BLOCKED] Daily loss {daily_pnl_pct:.2%} exceeds limit — no DCA")
            return
        
        # Check total account drawdown circuit breaker
        total_drawdown = (self.total_balance - self.starting_balance) / self.starting_balance
        if total_drawdown < -MAX_TOTAL_DRAWDOWN_PCT:
            logger.warning(f"[DCA BLOCKED] Total account drawdown {total_drawdown:.2%} exceeds -{MAX_TOTAL_DRAWDOWN_PCT:.0%} circuit breaker — no DCA")
            return
        
        # === END SAFETY CHECKS ===
        
        # Check if price dropped >3% from entry
        drop_from_entry = (pos['entry_price'] - current_price) / pos['entry_price']
        if drop_from_entry < 0.03:
            return
        
        # Check if we have enough balance for DCA (use 50% of original position size)
        dca_size = pos['position_size'] * 0.5
        if self.active_balance < dca_size:
            return
        
        # Check total exposure cap (after dca_size is calculated)
        total_deployed = sum(p.get('position_size', 0) for p in self.active_positions.values())
        exposure_cap = self._get_total_exposure_cap()
        if (total_deployed + dca_size) / self.total_balance > exposure_cap:
            logger.info(f"[DCA BLOCKED] Total exposure would exceed {exposure_cap:.0%} cap — ${total_deployed+dca_size:.2f}/${self.total_balance:.2f}")
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
        
        # Recalculate stop-loss based on new average price (fix #5: SL wasn't updated after DCA)
        sl_pct = pos.get('sl_pct', 0.05)
        old_sl = pos.get('stop_loss', 0)
        pos['stop_loss'] = new_avg_price * (1 - sl_pct)
        
        # Reserve additional capital
        self.active_balance -= dca_size
        
        logger.info(f"[DCA COMPLETE] {pair} new avg: ${new_avg_price:.4f} | "
                   f"Total size: ${pos['position_size']:.2f} | "
                   f"New qty: {total_qty:.4f} | "
                   f"SL: ${old_sl:.4f} → ${pos['stop_loss']:.4f}")

    def manage_positions(self, market_data: dict):
        """Check all active positions for exits with margin cost tracking."""
        for pair in list(self.active_positions.keys()):
            if pair not in market_data:
                continue
            
            # Skip positions with pending exit orders (waiting for fill confirmation)
            if pair in self.pending_exit_orders or self.active_positions[pair].get('_pending_exit'):
                continue

            # Pending entries reserve capital but are not real holdings yet. Do
            # not run stop/TP/timeout logic until Kraken confirms the fill.
            if self.active_positions[pair].get('_pending_entry'):
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
                pair, pos['tool'], current_price, data)
            
            # UPGRADE 3: Adjust stop loss for 2x leverage (tighter SL)
            effective_sl_pct = pos['sl_pct']
            if pos.get("leverage", 1) == 2:
                effective_sl_pct = pos['sl_pct'] / 2  # Tighter SL for 2x leverage
            
            # Check stop loss
            sl_anchor = pos.get('_trail_from', pos['entry_price'])
            if pos['direction'] == 'long':
                sl_price = sl_anchor * (1 - effective_sl_pct)
                if current_price <= sl_price:
                    self.close_position(pair, current_price, f"Stop loss @ ${sl_price:.4f}")
                    continue
            else:  # short
                sl_price = sl_anchor * (1 + effective_sl_pct)
                if current_price >= sl_price:
                    self.close_position(pair, current_price, f"Stop loss @ ${sl_price:.4f}")
                    continue

            bearish_exit = self._check_short_pressure_exit(pair, pos, current_price)
            if bearish_exit:
                self.close_position(pair, current_price, bearish_exit)
                continue

            conviction_decay_exit = self._check_conviction_decay_exit(pair, pos, data, current_price)
            if conviction_decay_exit:
                self.close_position(pair, current_price, conviction_decay_exit)
                continue
            
            # Smart trailing stop — volume spike, regime, RSI, momentum, ATR-adaptive
            effective_trailing_stop_pct = trailing_stop_pct
            if not effective_trailing_stop_pct and pos.get('_runner_mode') and pos.get('_partial_closed'):
                effective_trailing_stop_pct = pos.get('_runner_trail_pct', VOLATILE_RUNNER_TIGHT_TRAIL)

            if effective_trailing_stop_pct:
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
                smart_trail = self._smart_trailing_adjustment(pos, data, effective_trailing_stop_pct)
                
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
            # Bug 3 fix: once partial TP has fired, disable the static TP band entirely.
            # The remaining half should exit via trailing stop / SL / new signal — NOT at
            # the same tp_price (which previously collapsed "partial + runner" into two
            # exits at the same price, killing fat tails on winners).
            if take_profit_pct and not pos.get('_partial_closed'):
                if pos['direction'] == 'long':
                    tp_price = pos['entry_price'] * (1 + take_profit_pct)
                    if current_price >= tp_price:
                        self._partial_close(pair, current_price, 0.5, f"TP hit @ ${tp_price:.4f}")
                        continue
                else:  # short
                    tp_price = pos['entry_price'] * (1 - take_profit_pct)
                    if current_price <= tp_price:
                        self._partial_close(pair, current_price, 0.5, f"TP hit @ ${tp_price:.4f}")
                        continue
            
            # Check hold timeout with conviction re-check
            entry_time = pos.get('entry_time', None)
            if entry_time:
                elapsed_hours = (datetime.now(timezone.utc).timestamp() - entry_time) / 3600
            else:
                # Fallback for positions opened before this fix
                elapsed_hours = bars_held * (CHECK_INTERVAL / 3600)
            if elapsed_hours >= pos['hold']:
                # Save original hold period on first timeout
                if '_original_hold' not in pos:
                    pos['_original_hold'] = pos['hold']
                
                original_hold = pos['_original_hold']
                renewals = pos.get('_conviction_renewals', 0)
                MAX_SAME_TOOL_RENEWALS = 3    # Max 4x original hold (e.g. 24h → 96h)
                MAX_CROSS_TOOL_RENEWALS = 2   # Max 2 extensions from different tools
                
                # Conviction check: re-scan signals for this pair
                # If the same tool (or any tool in same direction) still fires, hold
                rescan = self.scan_signals(pair, data)
                same_direction = [
                    s for s, s_score in rescan
                    if s['direction'] == pos['direction'] and self._get_pair_policy_rejection(s, s_score) is None
                ]
                same_tool = [s for s in same_direction if s['tool'] == pos['tool']]
                
                if same_tool and renewals < MAX_SAME_TOOL_RENEWALS:
                    # Same tool still has conviction — reset the clock
                    pos['hold'] = elapsed_hours + original_hold
                    pos['_conviction_renewals'] = renewals + 1
                    logger.info(f"[CONVICTION HOLD] {pair} — {pos['tool']} still firing after {elapsed_hours:.1f}h, "
                               f"extending hold by {original_hold}h (renewal {renewals + 1}/{MAX_SAME_TOOL_RENEWALS})")
                    continue
                elif same_direction and renewals < MAX_CROSS_TOOL_RENEWALS:
                    # Different tool, same direction — partial conviction
                    new_tool = same_direction[0]['tool']
                    pos['hold'] = elapsed_hours + original_hold
                    pos['_conviction_renewals'] = renewals + 1
                    logger.info(f"[CONVICTION HOLD] {pair} — {new_tool} supports direction after {elapsed_hours:.1f}h, "
                               f"extending hold by {original_hold}h (renewal {renewals + 1}/{MAX_CROSS_TOOL_RENEWALS})")
                    continue
                
                # No conviction or max renewals hit — exit
                reason = f"Hold timeout ({elapsed_hours:.1f}h)"
                if renewals >= MAX_SAME_TOOL_RENEWALS:
                    reason += f" — max renewals ({renewals}) reached"
                else:
                    reason += " — no signal conviction"
                self.close_position(pair, current_price, reason)
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
            pair_blocked = self._pair_is_globally_blocked(pair)
            pair_policy = self._get_pair_policy(pair)
            
            # Extract txid
            if isinstance(order_id, dict) and 'txid' in order_id:
                txid = order_id['txid'][0] if isinstance(order_id['txid'], list) else order_id['txid']
            elif isinstance(order_id, str):
                txid = order_id
            else:
                txid = None
            
            # 1. Order still open on Kraken — check if we should cancel it
            if txid and txid in open_txids:
                should_cancel = pair_blocked and direction == 'long'
                cancel_reason = "pair blocked for new entries" if should_cancel else ""
                if not should_cancel and direction == 'long':
                    policy_signal = {
                        'pair': pair,
                        'tool': tool,
                        'direction': direction,
                        '_collapse_gate': order_info.get('entry_features', {}).get('collapse_gate', 'normal'),
                    }
                    policy_reason = self._get_pair_policy_rejection(policy_signal, order_info.get('original_score', 0))
                    if policy_reason:
                        should_cancel = True
                        cancel_reason = policy_reason.replace('_', ' ')
                crash_bypass_tools = {
                    'crash_buy', 'mega_crash', 'crash_neg_ac', 'blood_in_streets',
                    'crash_mean_revert', 'flash_crash', 'market_panic_70'
                }
                
                # Price drifted too far from entry
                original_price = order_info.get("original_price", order_info.get("price", 0))
                if pair in market_data and original_price > 0:
                    current_price = market_data[pair]["price"]
                    drift = abs(current_price - original_price) / original_price
                    drift_limit = float(pair_policy.get('pending_price_drift_abandon', PRICE_DRIFT_ABANDON))
                    if drift > drift_limit:
                        should_cancel = True
                        cancel_reason = f"price drift {drift:.1%} (was ${original_price:.4f}, now ${current_price:.4f})"

                # Same-pair bearish overlay now dominates the pending long.
                short_pressure = getattr(self, '_short_pressure_by_pair', {}).get(pair)
                cancel_pressure_score = float(pair_policy.get('pending_cancel_pressure_score', SHORT_PRESSURE_CANCEL_SCORE))
                if (not should_cancel and direction == 'long' and short_pressure and
                    short_pressure.get('effective_score', 0) >= cancel_pressure_score):
                    original_score = order_info.get("original_score", 0)
                    if tool not in crash_bypass_tools or original_score < short_pressure['effective_score'] * 1.1:
                        should_cancel = True
                        cancel_reason = (
                            f"bearish overlay: {short_pressure['tool']} "
                            f"score {short_pressure['effective_score']:.1f}"
                        )

                if not should_cancel:
                    validation_signal = {
                        'pair': pair,
                        'tool': tool,
                        'direction': direction,
                    }
                    validation_reason = self._get_validation_score_rejection(
                        validation_signal,
                        float(order_info.get("original_score", 0) or 0),
                    )
                    if validation_reason:
                        should_cancel = True
                        cancel_reason = validation_reason.replace('_', ' ')

                market_pressure = getattr(self, '_market_short_pressure', {})
                weak_long_tools = {
                    'vpin_dip', 'vpin_toxic', 'btc_alt_spread',
                    'deep_dip_8h', 'dip_buy_5pct', 'month_start_long'
                }
                if not should_cancel and direction == 'long':
                    market_mode = market_pressure.get('mode', 'normal')
                    min_long_score = market_pressure.get('min_long_score', 0.0)
                    original_score = order_info.get("original_score", 0)
                    trend_exception = (
                        self._is_trend_leader_tool(tool) and
                        original_score >= min_long_score * 1.1
                    )
                    if market_mode == 'risk_off' and tool not in crash_bypass_tools and not trend_exception:
                        should_cancel = True
                        cancel_reason = market_pressure.get('label', 'market bear pressure')
                    elif (market_mode in {'defensive', 'caution'} and
                          tool in weak_long_tools and
                          original_score < min_long_score):
                        should_cancel = True
                        cancel_reason = (
                            f"{market_pressure.get('label', 'market bear pressure')} "
                            f"score floor {min_long_score:.1f}"
                        )
                
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
            # Also check Kraken's prefixed versions (XXBT, XETH, etc.)
            possible_assets = {base_asset, 'X' + base_asset, 'XX' + base_asset}
            # Add reverse mapping (e.g., ETHUSD → XETH)
            for kraken_asset, mapped_pair in KRAKEN_ASSET_TO_PAIR.items():
                if mapped_pair == pair:
                    possible_assets.add(kraken_asset)
            
            held_qty = 0
            for asset, amount in balances.items():
                if asset in possible_assets:
                    held_qty = float(amount)
                    break
            
            expected_qty = order_info.get("qty", 0)
            
            if held_qty > expected_qty * 0.5:
                # 2. We hold it → order filled
                logger.info(f"[FILLED] {pair} {direction} — holding {held_qty:.4f} on Kraken")

                entry_price = float(order_info.get("price", 0) or 0)
                if pair not in self.active_positions:
                    self.active_positions[pair] = {
                        'pair': pair,
                        'tool': tool,
                        'direction': direction,
                        'leverage': 1,
                        'entry_price': entry_price,
                        'entry_bar': order_info.get("placed_bar", self.current_bar),
                        'entry_time': datetime.now(timezone.utc).timestamp(),
                        'position_size': held_qty * entry_price,
                        'initial_position_size': held_qty * entry_price,
                        'qty': held_qty,
                        'sl_pct': 0.04,
                        'hold': 24,
                        'score': order_info.get("original_score", 0),
                        'total_margin_cost': 0,
                        'regime_bucket': order_info.get('regime_bucket', self._get_regime_bucket()),
                        '_realized_partial_pnl': 0.0,
                        '_entry_features': dict(order_info.get('entry_features', {})),
                        '_ml_features': {},
                    }
                else:
                    pos = self.active_positions[pair]
                    if entry_price > 0:
                        pos['entry_price'] = entry_price
                    pos['qty'] = held_qty
                    pos['position_size'] = held_qty * (entry_price or pos.get('entry_price', 0))
                    pos.setdefault('initial_position_size', pos['position_size'])
                    pos.setdefault('entry_time', datetime.now(timezone.utc).timestamp())
                    pos.setdefault('regime_bucket', order_info.get('regime_bucket', self._get_regime_bucket()))
                    pos.setdefault('_entry_features', dict(order_info.get('entry_features', {})))
                    pos.setdefault('_ml_features', {})

                pos = self.active_positions[pair]
                pos.pop('_pending_entry', None)

                # Place TP and native stop-loss only after the buy is confirmed;
                # Kraken spot sell orders reserve held asset balance.
                exit_mode, take_profit_pct, _, _ = self._get_exit_params(pair, tool, pos.get('entry_price', entry_price), {})
                if take_profit_pct and ENABLE_LIVE_TRADING and not pos.get('_tp_order_id'):
                    try:
                        tp_price = pos['entry_price'] * (1 + take_profit_pct) if direction == 'long' else pos['entry_price'] * (1 - take_profit_pct)
                        tp_side = "sell" if direction == 'long' else "buy"
                        tp_qty = held_qty * 0.5
                        tp_result = self.client.place_order(pair, tp_side, "limit", tp_qty, tp_price)
                        if tp_result:
                            pos['_tp_order_id'] = tp_result.get('txid', [None])[0] if isinstance(tp_result, dict) else tp_result
                            pos['_tp_price'] = tp_price
                            pos['_tp_qty'] = tp_qty
                            logger.info(f"[TP PLACED] {pair} {tp_side} {tp_qty:.4f} @ ${tp_price:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to place TP for {pair}: {e}")

                self._place_native_stop_loss(pair, pos)

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
    
    def _place_native_stop_loss(self, pair: str, pos: dict) -> None:
        """Place a Kraken-native stop-loss order at entry*(1-sl_pct) on a long.

        Stores '_sl_order_id', '_sl_price', '_sl_qty' on the position on success.
        No-op if live trading disabled, position is a short, stop already placed,
        or required inputs missing. Failures are logged — local watchdog remains
        the backup safety net.

        Kraken spot reserves asset balance per open sell order, so if a TP
        order already reserves part of the qty we only stop the *uncovered*
        portion. On partial_close the SL is re-placed on keep_qty.
        """
        if not ENABLE_LIVE_TRADING:
            return
        if pos.get('_sl_order_id'):
            return
        if pos.get('direction', 'long') != 'long':
            return
        sl_pct = float(pos.get('sl_pct', 0.04) or 0.04)
        entry = float(pos.get('entry_price', 0) or 0)
        qty = float(pos.get('qty', 0) or 0)
        # Subtract any qty already reserved by an active TP sell order
        tp_qty_reserved = 0.0
        if pos.get('_tp_order_id') and pos['_tp_order_id'] not in ('dry_run', 'market_fallback', 'market_escalation', 'market_retry'):
            tp_qty_reserved = float(pos.get('_tp_qty', 0) or 0)
        sl_qty = max(0.0, qty - tp_qty_reserved)
        if sl_qty > 0:
            sl_qty *= 0.999
        if entry <= 0 or sl_qty <= 0 or sl_pct <= 0:
            return
        stop_price = entry * (1 - sl_pct)
        try:
            txid = self.client.place_stop_loss(
                symbol=pair, side="sell", quantity=sl_qty, stop_price=stop_price,
            )
            if txid:
                pos['_sl_order_id'] = txid
                pos['_sl_price'] = stop_price
                pos['_sl_qty'] = sl_qty
                covered = "full qty" if tp_qty_reserved <= 0 else f"{sl_qty:.4f}/{qty:.4f} (TP reserves {tp_qty_reserved:.4f})"
                logger.info(
                    f"[SL PLACED] {pair} sell {sl_qty:.4f} stop @ ${stop_price:.4f} "
                    f"({sl_pct:.1%} below entry ${entry:.4f}) — {covered}"
                )
            else:
                logger.warning(f"[SL FAILED] {pair} — no txid; local watchdog still active")
        except Exception as e:
            logger.warning(f"[SL FAILED] {pair}: {e} — local watchdog still active")

    def _cancel_native_stop_loss(self, pair: str, pos: dict, reason: str = "") -> None:
        """Cancel a previously-placed native stop-loss and clear tracking."""
        txid = pos.get('_sl_order_id')
        if not txid or txid in ('dry_run',):
            pos.pop('_sl_order_id', None)
            pos.pop('_sl_price', None)
            pos.pop('_sl_qty', None)
            return
        try:
            self.client.cancel_order(txid)
            logger.info(
                f"[SL CANCEL] {pair} cancelled {txid}"
                f"{(' — ' + reason) if reason else ''}"
            )
        except Exception as e:
            logger.warning(f"[SL CANCEL] {pair} failed to cancel {txid}: {e}")
        pos.pop('_sl_order_id', None)
        pos.pop('_sl_price', None)
        pos.pop('_sl_qty', None)

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
            # Cancel existing native SL — we'll replace with a tighter one sized to the remainder
            self._cancel_native_stop_loss(pair, pos, reason="partial_close")
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
        pos['_realized_partial_pnl'] = pos.get('_realized_partial_pnl', 0.0) + pnl_dollar
        
        # Tighten stop loss for remaining portion (anchor from current price)
        pos['sl_pct'] = 0.03
        pos['_trail_from'] = price

        # Re-place native stop-loss on the remaining qty, anchored at current price
        if ENABLE_LIVE_TRADING and pos.get('direction', 'long') == 'long' and keep_qty > 0:
            try:
                new_stop = price * (1 - 0.03)
                new_txid = self.client.place_stop_loss(
                    symbol=pair, side="sell", quantity=keep_qty, stop_price=new_stop,
                )
                if new_txid:
                    pos['_sl_order_id'] = new_txid
                    pos['_sl_price'] = new_stop
                    pos['_sl_qty'] = keep_qty
                    logger.info(f"[SL RE-PLACED] {pair} sell {keep_qty:.4f} stop @ ${new_stop:.4f} (3% below partial exit)")
            except Exception as e:
                logger.warning(f"[SL RE-PLACE] {pair} failed: {e} — local watchdog still active")

        # On high-volatility long trades, let the second half run with a trail.
        pair_volatility = 0.0
        if hasattr(self, '_pair_volatility'):
            pair_volatility = self._pair_volatility.get(pair, {}).get('volatility', 0.0)

        runner_tools = {
            'crash_neg_ac', 'crash_mean_revert', 'blood_in_streets',
            'vpin_dip', 'vpin_toxic', 'btc_alt_spread',
            'crash_buy', 'mega_crash', 'flash_crash',
            'market_panic_70', 'deep_dip_8h', 'dip_buy_5pct'
        }
        if (pos['direction'] == 'long' and
            pos.get('tool') in runner_tools and
            self._get_pair_policy(pair).get('allow_runner', True) and
            pair_volatility >= VOLATILE_RUNNER_MIN_VOL):
            pos['_runner_mode'] = True
            pos['_runner_trail_pct'] = (
                VOLATILE_RUNNER_WIDE_TRAIL
                if pair_volatility >= 0.35 else
                VOLATILE_RUNNER_TIGHT_TRAIL
            )
            pos['best_price'] = max(pos.get('best_price', pos['entry_price']), price)
        
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
                # Cancel existing TP order BEFORE placing exit order to free up funds
                existing_tp = pos.get('_tp_order_id')
                if existing_tp and existing_tp not in ('dry_run', 'market_fallback', 'market_escalation', 'market_retry'):
                    try:
                        self.client.cancel_order(existing_tp)
                        logger.info(f"[EXIT PREP] {pair} cancelled existing TP order {existing_tp} before placing exit")
                    except Exception as cancel_e:
                        logger.warning(f"[EXIT PREP] {pair} failed to cancel TP {existing_tp}: {cancel_e}")
                # Cancel native stop-loss too so it doesn't double-fire while exit is pending
                self._cancel_native_stop_loss(pair, pos, reason="close_position")
                if leverage == 2 and hasattr(self.client, 'close_leveraged_position'):
                    self.client.close_leveraged_position(pair, side, qty, price)
                    exit_order_id = "leveraged_close"
                else:
                    # Post-only limit — guarantees maker fee
                    # For stop-loss exits, offset price 0.5% to ensure fill during fast moves
                    exit_price = price
                    is_stop_loss = "stop loss" in reason.lower() or "Stop loss" in reason
                    if is_stop_loss:
                        if side == "sell":
                            exit_price = price * 0.995  # 0.5% below current for sells
                        else:
                            exit_price = price * 1.005  # 0.5% above current for buys
                        logger.info(f"[SL OFFSET] {pair} stop-loss limit {side} offset: ${price:.4f} → ${exit_price:.4f} (0.5% buffer)")
                    result = self.client.place_order(pair, side, "limit", qty, exit_price, post_only=True)
                    if isinstance(result, dict) and 'txid' in result:
                        exit_order_id = result['txid'][0] if isinstance(result['txid'], list) else result['txid']
                    elif isinstance(result, str):
                        exit_order_id = result
                    logger.info(f"[EXIT PENDING] {pair} post-only limit @ ${exit_price:.4f} (maker fee) — waiting for fill")
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
                   f"Tool: {pos.get('tool', 'unknown')} | Held: {hours_held:.1f}h")
        
        # For dry run, finalize immediately. For market orders, verify fill first.
        if exit_order_id == "dry_run":
            self._finalize_exit(pair, price, reason)
        elif exit_order_id in ("market_fallback", "leveraged_close"):
            # Market orders should fill instantly, but verify on next cycle via
            # manage_pending_exit_orders. Don't blindly finalize — if the order
            # failed silently, we'd lose track of the coins.
            # Set a flag so manage_pending_exit_orders checks balance instead of open orders
            self.pending_exit_orders[pair]["_verify_balance"] = True
            logger.info(f"[EXIT MARKET] {pair} market order placed — will verify fill on next cycle")
    
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
        exclude_clean_slate_stats = bool(pos.get('_exclude_from_clean_slate_stats'))
        total_trade_pnl_dollar = pos.get('_realized_partial_pnl', 0.0) + pnl_dollar
        initial_position_size = pos.get('initial_position_size', pos.get('position_size', 0.0))
        total_trade_pnl_pct = (
            total_trade_pnl_dollar / initial_position_size
            if initial_position_size > 0 else pnl_pct
        )
        
        # Update balances
        self.active_balance += pos['position_size'] + pnl_dollar
        self.total_balance += pnl_dollar
        self.active_profit += pnl_dollar
        
        # Update tool stats and streaks. Positions carried across a deliberate
        # clean-slate reset are still managed and journaled, but their outcome
        # must not seed the fresh learner because the entry happened under old
        # regime inputs.
        if not exclude_clean_slate_stats:
            if tool in self.tool_stats:
                self.tool_stats[tool]['trades'] += 1
                won = total_trade_pnl_pct > 0
                if won:
                    self.tool_stats[tool]['wins'] += 1
                    prev_avg = self.tool_stats[tool].get('avg_win_pct', total_trade_pnl_pct)
                    n_wins = self.tool_stats[tool]['wins']
                    self.tool_stats[tool]['avg_win_pct'] = prev_avg + (total_trade_pnl_pct - prev_avg) / n_wins
                else:
                    n_losses = self.tool_stats[tool]['trades'] - self.tool_stats[tool]['wins']
                    prev_avg = self.tool_stats[tool].get('avg_loss_pct', total_trade_pnl_pct)
                    self.tool_stats[tool]['avg_loss_pct'] = prev_avg + (total_trade_pnl_pct - prev_avg) / max(n_losses, 1)
                self.tool_stats[tool]['pnl'] += total_trade_pnl_dollar
                self.update_tool_streak(tool, won)
                # Record live-only EV (used by _apply_live_ev_score_adj to drive score_adj
                # once LIVE_EV_MIN_TRADES is reached).
                self._record_live_tool_outcome(tool, total_trade_pnl_pct, total_trade_pnl_dollar)
                self._apply_live_ev_score_adj(tool)

            self._record_contextual_outcome(
                pair,
                tool,
                pos.get('regime_bucket', self._get_regime_bucket()),
                total_trade_pnl_pct,
                total_trade_pnl_dollar,
            )

            if pos.get('direction') == 'long' and self._has_forward_feature_snapshot(pos.get('_entry_features', {})):
                self._append_forward_tool_outcome(tool, total_trade_pnl_pct, total_trade_pnl_dollar)

            ml_features = pos.get('_ml_features', {})
            if ml_features:
                self.meta_model.record_trade(ml_features, total_trade_pnl_dollar > 0)
        else:
            logger.info(f"[CLEAN SLATE] {pair} close excluded from fresh learner stats")
        
        # Update daily stats
        self._daily_stats["trades_closed"] += 1
        self._daily_stats["pnl"] += pnl_dollar
        if pnl_dollar > 0:
            self._daily_stats["wins"] += 1
        else:
            self._daily_stats["losses"] += 1
        self._daily_stats["tool_pnl"][tool] = self._daily_stats["tool_pnl"].get(tool, 0) + pnl_dollar
        close_category = self._categorize_close_reason(reason)
        close_reasons = self._daily_stats.setdefault("close_reasons", {})
        close_reasons[close_category] = close_reasons.get(close_category, 0) + 1
        
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
        close_reason_for_journal = reason
        if exclude_clean_slate_stats:
            close_reason_for_journal = f"{reason} [pre_clean_slate_position]"

        self._journal_close(
            pair=pair, tool=tool, direction=pos['direction'],
            exit_price=exit_price, pnl_pct=pnl_pct, pnl_dollar=pnl_dollar,
            bars_held=round(hours_held, 1), close_reason=close_reason_for_journal,
            entry_price=pos['entry_price']
        )
        
        leverage_str = f" (2x)" if leverage == 2 else ""
        logger.info(f"[EXIT CONFIRMED] {pair} {pos['direction']}{leverage_str} | "
                   f"PnL: ${pnl_dollar:+.2f} ({pnl_pct:+.2%}) | Tool: {tool}")
        
        # Stop loss cooldown: block re-entry while price is still falling
        # No cooldown on take profits — those trades worked, coin isn't broken
        if not hasattr(self, '_pair_cooldowns'):
            self._pair_cooldowns = {}
        if not hasattr(self, '_pair_daily_stops'):
            self._pair_daily_stops = {}
        
        is_stop = "stop loss" in reason.lower() or "Stop loss" in reason
        
        if is_stop:
            now_ts = datetime.now(timezone.utc).timestamp()
            today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # Track daily stop-loss count per pair
            pair_stops = self._pair_daily_stops.get(pair, {'count': 0, 'date': today_str})
            if pair_stops['date'] != today_str:
                pair_stops = {'count': 0, 'date': today_str}  # Reset for new day
            pair_stops['count'] += 1
            self._pair_daily_stops[pair] = pair_stops
            
            # If hit max daily stops, blacklist for 24h (hard, no early lift)
            if pair_stops['count'] >= MAX_STOP_LOSSES_PER_PAIR_PER_DAY:
                cooldown_until = now_ts + 24 * 3600
                self._pair_cooldowns[pair] = {
                    'expires': cooldown_until,
                    'stop_price': exit_price,
                    'hard_until': cooldown_until,  # No early lift for 24h blacklist
                }
                logger.warning(f"[BLACKLIST] {pair} hit {pair_stops['count']} stop losses today — blocked for 24h")
            else:
                cooldown_until = now_ts + STOP_LOSS_COOLDOWN_SEC
                hard_until = now_ts + STOP_LOSS_COOLDOWN_HARD_MIN  # Min 1h hard cooldown
                self._pair_cooldowns[pair] = {
                    'expires': cooldown_until,
                    'stop_price': exit_price,
                    'hard_until': hard_until,  # Can't lift before this even if price recovers
                }
                logger.info(f"[COOLDOWN] {pair} blocked for {STOP_LOSS_COOLDOWN_HARD_MIN//3600}h minimum, up to {STOP_LOSS_COOLDOWN_SEC//3600}h if still falling (stop #{pair_stops['count']} today)")
        
        # Remove position and pending exit
        del self.active_positions[pair]
        del self.pending_exit_orders[pair]
    
    def manage_pending_exit_orders(self, market_data: dict):
        """Check if pending exit orders filled. Escalate to market if stale.
        
        Key safety: market orders verify via balance check, not blind finalization.
        TP orders that vanish without the asset being sold get re-placed.
        """
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
        
        # Get actual balances for verification (cached per cycle)
        held_balances = {}
        if ENABLE_LIVE_TRADING:
            try:
                balances = self.client.get_account_balance()
                if balances:
                    usd_assets = {'USD', 'ZUSD', 'USDT', 'USDC', 'DAI', 'USDG'}
                    for asset, amount in balances.items():
                        if asset not in usd_assets and amount > 0.0001:
                            if asset in KRAKEN_ASSET_TO_PAIR:
                                pair = KRAKEN_ASSET_TO_PAIR[asset]
                            else:
                                pair = f"{asset}USD"
                            held_balances[pair] = amount
            except Exception as e:
                logger.debug(f"Error getting balances for exit verification: {e}")
        
        for pair in list(self.pending_exit_orders.keys()):
            exit_info = self.pending_exit_orders[pair]
            order_id = exit_info.get("order_id")
            
            # Skip dry run exits (already finalized)
            if order_id == "dry_run":
                continue
            
            # Market order verification: check if we still hold the asset
            if exit_info.get("_verify_balance"):
                held_qty = held_balances.get(pair, 0)
                if pair in market_data:
                    held_value = held_qty * market_data[pair]["price"]
                else:
                    held_value = held_qty * exit_info.get("exit_price", 0)
                
                if held_value < 1.0:  # Asset sold (only dust remains)
                    logger.info(f"[EXIT VERIFIED] {pair} market exit confirmed — no longer held on Kraken")
                    self._finalize_exit(pair, exit_info["exit_price"], exit_info["reason"])
                else:
                    # Market order FAILED — we still hold the coins!
                    logger.warning(f"[EXIT FAILED] {pair} market exit did NOT fill — still hold {held_qty:.4f} (${held_value:.2f})")
                    # Remove the failed pending exit and let reconciliation re-track it
                    del self.pending_exit_orders[pair]
                    if pair in self.active_positions:
                        self.active_positions[pair].pop('_pending_exit', None)
                    # Try market sell again — but first cancel ALL open orders for this pair
                    # (stale TP limit orders lock the tokens, causing "Insufficient funds")
                    # Bug 2 fix: cap retries at 1. Each retry on a thin book converts a
                    # -4% loss into a -10-15% loss because the book moves against us.
                    prior_retries = int(exit_info.get("_retry_count", 0) or 0)
                    if prior_retries >= 1:
                        logger.error(
                            f"[EXIT RETRY CEILING] {pair} already retried {prior_retries}x — "
                            f"finalizing at last known price ${exit_info['exit_price']:.4f} "
                            f"to stop slippage cascade. Reconciliation will sweep any dust."
                        )
                        self._finalize_exit(
                            pair, exit_info["exit_price"],
                            exit_info["reason"] + " (retry ceiling)"
                        )
                        continue
                    if ENABLE_LIVE_TRADING and pair in self.active_positions:
                        try:
                            # Cancel any existing open orders for this pair first
                            try:
                                open_orders = self.client.get_open_orders()
                                if isinstance(open_orders, dict) and 'open' in open_orders:
                                    orders_dict = open_orders['open']
                                elif isinstance(open_orders, dict):
                                    orders_dict = open_orders
                                else:
                                    orders_dict = {}
                                for txid, order_info in orders_dict.items():
                                    descr = order_info.get('descr', {})
                                    order_pair = descr.get('pair', '')
                                    if pair.replace('/', '') in order_pair or order_pair in pair:
                                        logger.info(f"[EXIT CLEANUP] Cancelling stale order {txid} for {pair}: {descr.get('order', '')}")
                                        self.client.cancel_order(txid)
                            except Exception as ce:
                                logger.warning(f"[EXIT CLEANUP] Failed to cancel open orders for {pair}: {ce}")
                            
                            pos = self.active_positions[pair]
                            side = "sell" if pos['direction'] == 'long' else "buy"
                            result = self.client.place_order(pair, side, "market", held_qty, 0)
                            logger.info(f"[EXIT RETRY] {pair} re-attempting market sell (after cancelling stale orders): {result}")
                            # Don't finalize — will verify on next cycle
                            self.pending_exit_orders[pair] = {
                                "order_id": "market_retry",
                                "placed_bar": self.current_bar,
                                "exit_price": exit_info["exit_price"],
                                "reason": exit_info["reason"] + " (retry)",
                                "pnl_pct": exit_info["pnl_pct"],
                                "pnl_dollar": exit_info["pnl_dollar"],
                                "hours_held": exit_info["hours_held"],
                                "leverage": exit_info.get("leverage", 1),
                                "total_margin_cost_pct": exit_info.get("total_margin_cost_pct", 0),
                                "side": side,
                                "qty": held_qty,
                                "_verify_balance": True,
                                "_retry_count": prior_retries + 1,
                            }
                            self.active_positions[pair]['_pending_exit'] = True
                        except Exception as e:
                            logger.error(f"[EXIT RETRY FAILED] {pair}: {e} — will retry next cycle via reconciliation")
                continue
            
            # Limit order: check if filled (not in open orders anymore)
            if order_id and order_id not in open_txids:
                # Order is gone from open orders — but did it FILL or get CANCELLED?
                # Check if we still hold the asset
                held_qty = held_balances.get(pair, 0)
                if pair in market_data:
                    held_value = held_qty * market_data[pair]["price"]
                else:
                    held_value = held_qty * exit_info.get("exit_price", 0)
                
                if held_value < 1.0:  # Truly sold
                    logger.info(f"[EXIT FILLED] {pair} exit order {order_id} confirmed — asset no longer held")
                    self._finalize_exit(pair, exit_info["exit_price"], exit_info["reason"])
                else:
                    # Order vanished but we still hold! TP got cancelled/expired
                    logger.warning(f"[EXIT VANISHED] {pair} exit order {order_id} gone but still hold {held_qty:.4f} (${held_value:.2f}) — re-placing TP")
                    # Clear pending exit, let position stay active, place new TP
                    del self.pending_exit_orders[pair]
                    if pair in self.active_positions:
                        self.active_positions[pair].pop('_pending_exit', None)
                        pos = self.active_positions[pair]
                        # Re-place TP order
                        if ENABLE_LIVE_TRADING:
                            try:
                                exit_mode, tp_pct, trail_pct, _ = self._get_exit_params(
                                    pair, pos.get('tool', 'reconciled'), pos['entry_price'], {})
                                if tp_pct:
                                    tp_price = pos['entry_price'] * (1 + tp_pct) if pos['direction'] == 'long' else pos['entry_price'] * (1 - tp_pct)
                                    tp_side = "sell" if pos['direction'] == 'long' else "buy"
                                    result = self.client.place_order(pair, tp_side, "limit", held_qty, tp_price, post_only=True)
                                    if result:
                                        new_txid = result.get('txid', [None])[0] if isinstance(result, dict) else result
                                        pos['_tp_order_id'] = new_txid
                                        pos['_tp_price'] = tp_price
                                        logger.info(f"[TP REPLACED] {pair} new TP {tp_side} @ ${tp_price:.4f}")
                                else:
                                    logger.info(f"[TP SKIP] {pair} uses trailing stop, no fixed TP to place")
                            except Exception as e:
                                logger.warning(f"[TP REPLACE FAILED] {pair}: {e}")
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
                    # Cancel ALL open orders for this pair (not just the tracked one)
                    try:
                        if order_id:
                            self.client.cancel_order(order_id)
                        # Also cancel any other stale orders for this pair
                        try:
                            oo = self.client.get_open_orders()
                            if isinstance(oo, dict) and 'open' in oo:
                                oo = oo['open']
                            elif not isinstance(oo, dict):
                                oo = {}
                            for txid, oi in oo.items():
                                op = oi.get('descr', {}).get('pair', '')
                                if txid != order_id and (pair.replace('/', '') in op or op in pair):
                                    logger.info(f"[EXIT CLEANUP] Cancelling extra order {txid} for {pair}")
                                    self.client.cancel_order(txid)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error(f"Failed to cancel stale exit for {pair}: {e}")
                    
                    # Place market order — but DON'T finalize immediately
                    try:
                        qty = exit_info["qty"]
                        self.client.place_order(pair, side, "market", qty, 0)
                        logger.info(f"[EXIT MARKET] {pair} forced market exit — will verify on next cycle")
                        exit_info["_verify_balance"] = True
                        exit_info["order_id"] = "market_escalation"
                    except Exception as e:
                        logger.error(f"Failed to market-exit {pair}: {e}")
                        continue  # Don't finalize if market order also failed
                else:
                    self._finalize_exit(pair, current_price, exit_info["reason"] + " (market fallback)")
    
    def execute_signal(self, signal: dict, score: float):
        """Execute a signal with UPGRADE 1 (limit orders) and UPGRADE 3 (2x margin)."""
        pair = signal['pair']
        direction = signal['direction']
        tool = signal['tool']

        policy_reason = self._get_pair_policy_rejection(signal, score)
        if policy_reason:
            label = "PAIR BLOCKED" if policy_reason == 'pair_blocked' else 'PAIR POLICY'
            logger.warning(f"[{label}] {pair} disabled for new entries — skipping {tool}")
            self._log_rejection(pair, tool, direction, score, policy_reason)
            return

        quality_reason = self._get_quality_universe_rejection(pair, direction)
        if quality_reason:
            logger.info(f"[QUALITY UNIVERSE] {tool} {pair} blocked — {quality_reason}")
            self._log_rejection(pair, tool, direction, score, quality_reason)
            return

        validation_score = float(signal.get('_pre_evidence_score', score) or score)
        validation_reason = self._get_validation_score_rejection(signal, validation_score)
        if validation_reason:
            floor = self._get_validation_score_floor(signal)
            logger.info(
                f"[VALIDATION FLOOR] {tool} {pair} score {validation_score:.1f} < {floor:.1f} — "
                "skipping during $300 validation"
            )
            self._record_opportunity_scout_candidate(
                signal, validation_score, validation_reason, 'validation_floor',
                getattr(self, '_latest_market_data', {})
            )
            self._log_rejection(pair, tool, direction, validation_score, validation_reason)
            return

        if direction == 'long' and not signal.get('_evidence_checked'):
            allowed, adjusted_score, risk_mult, evidence_reason, evidence_snapshot = self._evaluate_tool_evidence(signal, score)
            if not allowed:
                logger.info(f"[EVIDENCE] {tool} {pair} blocked — {evidence_reason}")
                self._record_opportunity_scout_candidate(
                    signal, score, evidence_reason or 'evidence_gate', 'execute_evidence_gate',
                    getattr(self, '_latest_market_data', {})
                )
                self._log_rejection(pair, tool, direction, score, evidence_reason or 'evidence_gate')
                return
            signal['_pre_evidence_score'] = score
            signal['_evidence_checked'] = True
            signal['_evidence_risk_multiplier'] = risk_mult
            signal['_evidence_snapshot'] = evidence_snapshot
            score = adjusted_score
        evidence_risk_multiplier = float(signal.get('_evidence_risk_multiplier', 1.0) or 1.0)

        asset_context = self._evaluate_asset_context(pair, direction)
        signal['_asset_context'] = asset_context
        if not asset_context.get('ok', True):
            reason = asset_context.get('reason', 'asset_context_veto')
            logger.info(f"[ASSET CONTEXT] {tool} {pair} blocked — {reason}")
            self._log_rejection(pair, tool, direction, score, reason)
            return
        
        # STOP LOSS COOLDOWN: Don't re-enter a pair that's still falling after stopping us out
        if hasattr(self, '_pair_cooldowns') and pair in self._pair_cooldowns:
            cd = self._pair_cooldowns[pair]
            now_ts = datetime.now(timezone.utc).timestamp()
            hard_until = cd.get('hard_until', 0)
            
            if now_ts < cd['expires']:
                # HARD COOLDOWN: No early lift during minimum period (or 24h blacklist)
                if now_ts < hard_until:
                    remaining = (cd['expires'] - now_ts) / 3600
                    hard_remaining = (hard_until - now_ts) / 3600
                    is_blacklist = hard_until == cd['expires']  # 24h blacklist has hard == expires
                    label = "BLACKLIST" if is_blacklist else "HARD COOLDOWN"
                    logger.info(f"[{label}] {pair} in mandatory cooldown ({hard_remaining:.1f}h hard, {remaining:.1f}h total) — skipping {tool}")
                    self._log_rejection(pair, tool, direction, score, f"stop_loss_hard_cooldown_{hard_remaining:.1f}h")
                    return
                
                # SOFT COOLDOWN: After hard minimum, can lift if price recovered significantly
                current_price = signal.get('price', 0) or 0
                if not current_price:
                    try:
                        ticker = self.client.get_ticker(pair)
                        current_price = float(ticker['price']) if ticker else 0
                    except Exception:
                        pass
                
                # Require 5% above stop price to lift (dead cat bounces don't count)
                recovery_threshold = cd['stop_price'] * 1.05
                if current_price and current_price > recovery_threshold:
                    logger.info(f"[COOLDOWN LIFTED] {pair} price ${current_price:.4f} > ${recovery_threshold:.4f} (5% above stop ${cd['stop_price']:.4f}) — allowing re-entry")
                    del self._pair_cooldowns[pair]
                else:
                    remaining = (cd['expires'] - now_ts) / 3600
                    logger.info(f"[COOLDOWN] {pair} hasn't recovered 5% above stop (${current_price:.4f} vs ${recovery_threshold:.4f}, {remaining:.1f}h left) — skipping {tool}")
                    self._log_rejection(pair, tool, direction, score, f"stop_loss_cooldown_{remaining:.1f}h")
                    return
            else:
                del self._pair_cooldowns[pair]  # Cooldown expired

        # Bug 5 fix: TOKEN-EVENT GUARD.
        # Block entries on pairs that look like they've undergone a discontinuous
        # reprice (token split/redenomination/listing event/manipulation). Our
        # mean-reversion tools are defenseless against these — historically the
        # RAVEUSD ~20x overnight reprice and subsequent chop cost ~$35 before
        # the pair was hard-blocked. Generic guard prevents recurrence on any pair.
        try:
            vol_info = getattr(self, '_pair_volatility', {}).get(pair, {}) if hasattr(self, '_pair_volatility') else {}
            high_24h = float(vol_info.get('high_24h', 0) or 0)
            low_24h = float(vol_info.get('low_24h', 0) or 0)
            atr_pct_24h = float(vol_info.get('atr_pct', 0) or 0)
            # Range ratio: high/low on a normal volatile day is ~1.15-1.30. A ratio
            # above 2.0 in 24h almost always indicates a token event, not price action.
            range_ratio = (high_24h / low_24h) if (high_24h > 0 and low_24h > 0) else 1.0
            if range_ratio >= 2.0 or atr_pct_24h >= 0.50:
                logger.warning(
                    f"[TOKEN EVENT GUARD] {pair} 24h range ratio={range_ratio:.2f} "
                    f"atr={atr_pct_24h:.1%} — possible reprice/split/manipulation, skipping {tool}"
                )
                self._log_rejection(
                    pair, tool, direction, score,
                    f"token_event_guard_range{range_ratio:.2f}_atr{atr_pct_24h:.2f}"
                )
                return
        except Exception as _te:
            logger.debug(f"[TOKEN EVENT GUARD] {pair} check failed ({_te}) — allowing through")
        
        # PER-PAIR DAILY LIMIT: Prevent over-concentration on one pair
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if not hasattr(self, '_pair_daily_trades'):
            self._pair_daily_trades = {}
        # Reset if new day
        if getattr(self, '_pair_daily_trades_date', '') != today_str:
            self._pair_daily_trades = {}
            self._pair_daily_trades_date = today_str
        pair_today = self._pair_daily_trades.get(pair, 0)
        pair_trade_limit = self._get_pair_daily_trade_limit(pair)
        if pair_today >= pair_trade_limit:
            logger.info(f"[PAIR LIMIT] {pair} hit {pair_trade_limit} trades today — skipping {tool}")
            self._log_rejection(pair, tool, direction, score, f"pair_daily_limit_{pair_today}")
            return
        
        # REGIME GATE: Reduce activity in strong bear markets
        # Only high-conviction crash tools bypass this
        crash_bypass_tools = {
            'crash_buy', 'mega_crash', 'crash_neg_ac', 'blood_in_streets',
            'crash_mean_revert', 'flash_crash', 'market_panic_70'
        }
        if direction == 'long' and tool not in crash_bypass_tools:
            bullish_pct = getattr(self, '_bullish_4h_pct', 50)
            avg_rsi_4h = getattr(self, '_avg_rsi_4h', 50)
            fng = self.get_fng()
            
            # Strong bear: 100% bearish on 4h AND RSI < 35 AND F&G < 30
            if bullish_pct <= 10 and avg_rsi_4h < 35 and fng < 30:
                # In strong bear, need score > 15 to enter (filters out weak signals)
                if score < 15:
                    logger.info(f"[REGIME GATE] {tool} {pair} blocked — strong bear (F&G={fng}, {bullish_pct:.0f}% bull, RSI4h={avg_rsi_4h:.0f}) and weak score {score:.1f} < 15")
                    self._log_rejection(pair, tool, direction, score, f"regime_gate_strong_bear_score_{score:.1f}")
                    return
                # Even high-score signals get 50% size reduction in strong bear
                # (this will be applied later — set a flag)
                self._bear_regime_size_cut = True
            else:
                self._bear_regime_size_cut = False

        # Long-only defense: if the same pair is flashing a strong short setup,
        # require an even stronger crash-reversal signal to fight it.
        if direction == 'long':
            short_pressure = getattr(self, '_short_pressure_by_pair', {}).get(pair)
            if short_pressure:
                pair_policy = self._get_pair_policy(pair)
                bearish_score = short_pressure.get('effective_score', short_pressure.get('score', 0.0))
                overlay_block_score = float(pair_policy.get('entry_pressure_block_score', SHORT_PRESSURE_ENTRY_BLOCK_SCORE))
                fight_multiplier = float(pair_policy.get('bearish_overlay_fight_multiplier', 1.1))
                pressure_cap = float(pair_policy.get('bearish_overlay_pressure_cap', float('inf')))
                can_fight_bearish = (
                    tool in crash_bypass_tools and
                    bearish_score < pressure_cap and
                    score >= bearish_score * fight_multiplier
                )
                if bearish_score >= overlay_block_score and not can_fight_bearish:
                    logger.info(
                        f"[BEARISH OVERLAY] {tool} {pair} blocked — "
                        f"{short_pressure['tool']} short score {bearish_score:.1f} active"
                    )
                    self._log_rejection(
                        pair, tool, direction, score,
                        f"bearish_overlay_{short_pressure['tool']}_{bearish_score:.1f}"
                    )
                    return

        market_pressure = getattr(self, '_market_short_pressure', {})
        if direction == 'long':
            market_mode = market_pressure.get('mode', 'normal')
            min_long_score = market_pressure.get('min_long_score', 0.0)
            weak_long_tools = {
                'vpin_dip', 'vpin_toxic', 'btc_alt_spread',
                'deep_dip_8h', 'dip_buy_5pct', 'month_start_long'
            }
            trend_exception = (
                self._is_trend_leader_tool(tool) and score >= min_long_score * 1.1
            )
            if market_mode == 'risk_off' and tool not in crash_bypass_tools and not trend_exception:
                logger.info(
                    f"[MARKET BEAR] {tool} {pair} blocked — "
                    f"{market_pressure.get('label', 'risk-off')} with score floor {min_long_score:.1f}"
                )
                self._log_rejection(
                    pair, tool, direction, score,
                    f"market_bear_{market_mode}_{min_long_score:.1f}"
                )
                return
            if (market_mode in {'defensive', 'caution'} and
                tool in weak_long_tools and
                score < min_long_score):
                logger.info(
                    f"[MARKET BEAR] {tool} {pair} blocked — "
                    f"{market_pressure.get('label', 'market bear pressure')} score {score:.1f} < {min_long_score:.1f}"
                )
                self._log_rejection(
                    pair, tool, direction, score,
                    f"market_bear_{market_mode}_{min_long_score:.1f}"
                )
                return
        
        # DAILY DRAWDOWN LIMIT: Stop trading if we've lost too much today
        daily_pnl_pct = self._daily_stats["pnl"] / max(self._daily_stats.get("start_balance", self.total_balance), 1)
        if daily_pnl_pct < -DAILY_MAX_LOSS_PCT:
            logger.warning(f"[DAILY LIMIT] Daily loss {daily_pnl_pct:.2%} exceeds -{DAILY_MAX_LOSS_PCT:.0%} limit — no new trades")
            self._log_rejection(pair, tool, direction, score, f"daily_drawdown_{daily_pnl_pct:.2%}")
            return
        
        # TOTAL ACCOUNT CIRCUIT BREAKER: Stop all trading if cumulative losses too deep
        total_drawdown = (self.total_balance - self.starting_balance) / self.starting_balance
        if total_drawdown < -MAX_TOTAL_DRAWDOWN_PCT:
            logger.critical(f"[CIRCUIT BREAKER] Total account drawdown {total_drawdown:.2%} exceeds -{MAX_TOTAL_DRAWDOWN_PCT:.0%} limit — ALL TRADING HALTED")
            self._log_rejection(pair, tool, direction, score, f"circuit_breaker_{total_drawdown:.2%}")
            return
        
        # TOTAL EXPOSURE CAP: Don't over-deploy capital
        total_deployed = sum(p.get('position_size', 0) for p in self.active_positions.values())
        exposure_cap = self._get_total_exposure_cap()
        if total_deployed / max(self.total_balance, 1) > exposure_cap:
            logger.info(f"[EXPOSURE CAP] {total_deployed/self.total_balance:.0%} deployed exceeds {exposure_cap:.0%} cap — skipping {tool}")
            self._log_rejection(pair, tool, direction, score, f"exposure_cap_{total_deployed/self.total_balance:.0%}")
            return
        
        # US retail accounts (Non-ECP) cannot open margin positions on Kraken SPOT,
        # and this system is Kraken spot-only. Short tools remain valuable as
        # defensive overlays, but they do not become live orders.
        # Note: rejection is NOT logged to CSV — it's a structural routing check,
        # not a signal-quality issue. Logging it created ~20k rows of noise.
        if direction == 'short' and ENABLE_LIVE_TRADING:
            if ENABLE_EXTERNAL_SIGNAL_EXPORT and _NT_EXPORT_AVAILABLE and _nt_export_signal is not None:
                try:
                    exit_mode_, tp_pct_, _, _ = self._get_exit_params(pair, tool, signal.get('price', 0) or 0, {})
                except Exception:
                    tp_pct_ = 0.0
                # Resolve a usable entry price: signal.price > cached close > live ticker
                ref_price = float(signal.get('price', 0) or 0)
                if ref_price <= 0:
                    try:
                        cached = self._price_cache.get(pair)
                        if cached is not None and len(cached) > 0:
                            ref_price = float(cached[-1])
                    except Exception:
                        pass
                if ref_price <= 0:
                    try:
                        t = self.client.get_ticker(pair)
                        ref_price = float((t or {}).get('last') or (t or {}).get('c', [0])[0] or 0)
                    except Exception:
                        ref_price = 0.0
                # Approximate NT-side notional from the bot's would-be spot risk budget
                est_notional = float(self.total_balance) * float(MAX_POSITION_PCT)
                try:
                    _nt_export_signal(
                        spot_pair=pair,
                        side="short",
                        score=float(score),
                        tool=str(tool),
                        entry_price=ref_price,
                        stop_pct=float(signal.get('sl_pct', 0) or 0),
                        target_pct=float(tp_pct_ or 0),
                        notional_usd=est_notional,
                        regime=str(self._get_regime_bucket()) if hasattr(self, '_get_regime_bucket') else "",
                        notes="short_routed_to_nt",
                    )
                except Exception as _nt_e:
                    logger.debug(f"[NT-EXPORT] short signal write failed: {_nt_e}")
            if self._futures_ready and not SPOT_ONLY_MODE:
                logger.info(f"[SHORT DISABLED] {tool} {pair} score={score:.1f} — futures routing disabled")
                return
            logger.debug(f"Skipping {tool} ({pair}) — short is defensive-only in Kraken spot mode")
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
        # Tier: validated bull momentum tools can fire at F&G >= 20 (they were OOS-tested in bull regimes)
        # Other bull tools need F&G >= 40 (less validated)
        validated_bull = {'accumulation_breakout', 'hurst_trend_long'}
        other_bull = {'buy_weekly_green', 'buy_breakout_simple',
                      'simple_buy_uptrend', 'buy_btc_leading', 'major_pair_breakout'}
        if tool in validated_bull:
            fng = self.get_fng()
            bullish_pct = getattr(self, '_bullish_4h_pct', 50)
            if fng < 20 or bullish_pct < 40:
                logger.info(f"Skipping {tool} ({pair}) - validated bull blocked in extreme fear (F&G={fng}, {bullish_pct:.0f}% bullish)")
                return
        elif tool in other_bull:
            fng = self.get_fng()
            bullish_pct = getattr(self, '_bullish_4h_pct', 50)
            if fng < 40 or bullish_pct < 50:
                if self._allow_major_pair_bull_bypass(pair, tool, signal, score):
                    logger.info(
                        f"[MAJOR BYPASS] {tool} {pair} bypassed bull gate "
                        f"(F&G={fng}, {bullish_pct:.0f}% bullish, "
                        f"vol={self._safe_float(signal.get('_volume_ratio')):.1f}x)"
                    )
                else:
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
        if tool in self.tool_stats and self._get_tool_trade_count(tool) >= 5:
            ts = self.tool_stats[tool]
            total = ts.get('trades', ts.get('total', 0))
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

        regime_bucket = self._get_regime_bucket()
        context_mult = self._get_contextual_score_multiplier(pair, tool, regime_bucket)
        risk_pct *= max(0.75, min(1.25, context_mult))
        
        risk_amount = self.active_balance * risk_pct
        
        # Apply bear regime size reduction
        if getattr(self, '_bear_regime_size_cut', False):
            risk_amount *= 0.5
            logger.info(f"[BEAR CUT] {tool} {pair} — 50% size reduction in strong bear regime")

        if (
            direction == 'long' and
            self._is_trend_leader_tool(tool) and
            score >= BULL_OFFENSE_MIN_SCORE and
            self._is_bull_offense_mode()
        ):
            risk_amount *= BULL_OFFENSE_SIZE_MULT
            logger.info(
                f"[BULL OFFENSE] {tool} {pair} — size x{BULL_OFFENSE_SIZE_MULT:.2f} "
                f"in supportive bull regime"
            )

        pair_risk_multiplier = self._get_pair_risk_multiplier(pair, tool)
        if pair_risk_multiplier != 1.0:
            risk_amount *= pair_risk_multiplier
            logger.info(f"[PAIR RISK] {tool} {pair} — size x{pair_risk_multiplier:.2f} from pair policy")

        if direction == 'long' and evidence_risk_multiplier != 1.0:
            risk_amount *= evidence_risk_multiplier
            evidence_snapshot = signal.get('_evidence_snapshot', {}) or {}
            logger.info(
                f"[EVIDENCE SIZE] {tool} {pair} — size x{evidence_risk_multiplier:.2f} "
                f"({evidence_snapshot.get('tier', 'evidence')})"
            )
        
        stop_loss_pct = signal['sl_pct']
        
        if direction == 'long':
            position_size = risk_amount / stop_loss_pct
        else:  # short
            position_size = risk_amount / stop_loss_pct
        
        # EXTREME FEAR BOOST: Slightly bigger on crash buys when F&G < 15
        fear_boost_tools = {'crash_buy', 'mega_crash', 'blood_in_streets',
                           'crash_neg_ac', 'crash_mean_revert', 'flash_crash',
                           'market_panic_70', 'deep_dip_8h'}
        # REMOVED from fear_boost: volatile_oversold (0% WR), quick_crash (33% WR) — killed
        if tool in fear_boost_tools and getattr(self, 'current_fng', 50) < 15:
            position_size *= 1.25  # 25% bigger in extreme fear (was 1.5x — too aggressive)
            logger.info(f"[FEAR BOOST] {tool} {pair} — 1.25x size in extreme fear (F&G={self.current_fng})")
        
        # Apply leverage to position sizing (controls 2x notional with same risk)
        if leverage == 2:
            # Position size stays the same but controls 2x the notional
            pass
        
        # HARD CAP: Never put more than 20% of total balance in one position
        position_cap_pct = self._get_position_cap_pct(tool, direction, score)
        max_position = self.total_balance * position_cap_pct
        if position_size > max_position:
            logger.info(
                f"[SIZE CAP] {pair} capped ${position_size:.2f} → ${max_position:.2f} "
                f"({position_cap_pct:.0%} of ${self.total_balance:.2f})"
            )
            position_size = max_position
        
        # Also don't exceed available balance
        position_size = min(position_size, self.active_balance * 0.8)
        
        # Cap position at 1% of pair's 24h volume (liquidity guard)
        liquidity_cap_limit = self._get_pair_liquidity_cap_limit(pair)
        if hasattr(self, '_pair_volatility') and pair in self._pair_volatility:
            max_pos = self._pair_volatility[pair].get('max_position_usd', float('inf'))
            liquidity_cap_limit = self._get_pair_liquidity_cap_limit(pair, max_pos)
        if liquidity_cap_limit and position_size > liquidity_cap_limit:
            logger.info(f"[LIQUIDITY CAP] {pair} capped ${position_size:.2f} → ${liquidity_cap_limit:.2f}")
            position_size = liquidity_cap_limit
        
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

        htf_ctx = signal.get('_htf_context', {})
        mtf_m = self._compute_mtf_multiplier(tool, direction, htf_ctx)
        base_score = score / mtf_m if mtf_m != 0 else score
        entry_features = self._build_entry_feature_snapshot(signal, position_size, liquidity_cap_limit)
        ml_features = self._build_meta_features(signal, score, base_score, mtf_m, entry_features)
        ml_probability = self.meta_model.predict_win_probability(ml_features)

        if self.meta_model.should_veto(ml_features, score):
            logger.info(
                f"[META] {tool} {pair} vetoed — pooled prob {ml_probability:.2f}, score {score:.1f}"
            )
            self._record_opportunity_scout_candidate(
                signal, score, f"meta_model_veto_{ml_probability:.2f}", 'meta_model_veto',
                getattr(self, '_latest_market_data', {})
            )
            self._log_rejection(pair, tool, direction, score, f"meta_model_veto_{ml_probability:.2f}")
            return

        ml_multiplier = self.meta_model.get_size_multiplier(ml_features)
        if ml_multiplier != 1.0:
            position_size *= ml_multiplier
            position_size = min(position_size, max_position)
            position_size = min(position_size, self.active_balance * 0.8)
            if liquidity_cap_limit and np.isfinite(liquidity_cap_limit):
                position_size = min(position_size, liquidity_cap_limit)
            entry_features = self._build_entry_feature_snapshot(signal, position_size, liquidity_cap_limit)
            ml_features = self._build_meta_features(signal, score, base_score, mtf_m, entry_features)
            logger.info(
                f"[META] {tool} {pair} pooled prob {ml_probability:.2f} → size x{ml_multiplier:.2f}"
            )

        signal['_ml_features'] = dict(ml_features)
        signal['_ml_probability'] = ml_probability
        signal['_ml_multiplier'] = ml_multiplier

        # Bug 1 fix: FINAL HARD CLAMP on position_size.
        # Belt-and-suspenders guard against any prior sizing path (Kelly, fear boost,
        # ML multiplier, min-vol bump) accidentally exceeding the account-level cap.
        # Historical bug: a DRIFTUSD entry sized 92% of the account ($296 on a $320
        # balance) because an intermediate sizing step bypassed the earlier cap.
        # This clamp MUST be the last gate before qty is computed.
        try:
            safe_balance = float(self.total_balance or 0)
            hard_cap_pct = float(self._get_position_cap_pct(tool, direction, score))
            hard_cap_usd = safe_balance * hard_cap_pct
            if hard_cap_usd > 0 and position_size > hard_cap_usd:
                logger.warning(
                    f"[HARD SIZE CAP] {pair} {tool} clamped "
                    f"${position_size:.2f} → ${hard_cap_usd:.2f} "
                    f"({hard_cap_pct:.0%} of total_balance ${safe_balance:.2f})"
                )
                position_size = hard_cap_usd
            # Also never exceed the active_balance headroom
            if position_size > self.active_balance * 0.9:
                position_size = self.active_balance * 0.9
            # Refuse to trade with a degenerate balance snapshot
            if position_size <= 0:
                logger.warning(f"[HARD SIZE CAP] {pair} computed size ${position_size:.2f} ≤ 0 — skipping")
                self._log_rejection(pair, tool, direction, score, "size_clamp_non_positive")
                return
        except Exception as _e:
            logger.error(f"[HARD SIZE CAP] {pair} clamp failed ({_e}) — aborting entry as safety")
            self._log_rejection(pair, tool, direction, score, "size_clamp_exception")
            return

        qty = position_size / entry_price
        
        # PRE-CHECK: Verify qty meets Kraken minimum order volume before placing
        if ENABLE_LIVE_TRADING:
            try:
                min_vol = self.client.get_min_order_volume(pair)
                if min_vol is not None and qty < min_vol:
                    needed_size = min_vol * entry_price * 1.05  # 5% buffer
                    if liquidity_cap_limit and np.isfinite(liquidity_cap_limit) and needed_size > liquidity_cap_limit:
                        logger.info(f"[MIN VOL] {pair} minimum order ${needed_size:.2f} exceeds pair cap ${liquidity_cap_limit:.2f}, skipping")
                        self._log_rejection(pair, tool, direction, score, f"min_vol_exceeds_pair_cap_{min_vol}")
                        return
                    if needed_size <= self.active_balance * 0.8 and needed_size <= self.total_balance * MAX_POSITION_PCT:
                        logger.info(f"[MIN VOL] {pair} qty {qty:.4f} < min {min_vol}, bumping size ${position_size:.2f} → ${needed_size:.2f}")
                        position_size = needed_size
                        qty = position_size / entry_price
                    else:
                        logger.info(f"[MIN VOL] {pair} qty {qty:.4f} < min {min_vol}, can't afford min size ${needed_size:.2f}, skipping")
                        self._log_rejection(pair, tool, direction, score, f"below_min_vol_{qty:.2f}_need_{min_vol}")
                        return
            except Exception:
                pass
        
        # UPGRADE 3: Add margin opening cost for leveraged positions
        margin_opening_cost = 0
        if leverage == 2:
            margin_opening_cost = position_size * MARGIN_COST_OPEN
            if self.active_balance < margin_opening_cost:
                logger.warning(f"Insufficient balance for margin opening cost: {pair}")
                self._log_rejection(pair, tool, direction, score, "insufficient_margin_balance")
                return
            self.active_balance -= margin_opening_cost
        
        planned_hold_hours = self._get_pair_hold_hours(pair, signal['hold'])

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
                    "regime_bucket": regime_bucket,
                    "entry_features": dict(entry_features),
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
            'initial_position_size': position_size,
            'qty': qty,
            'sl_pct': stop_loss_pct,
            'hold': planned_hold_hours,
            'score': score,
            'total_margin_cost': margin_opening_cost,
            'regime_bucket': regime_bucket,
            '_realized_partial_pnl': 0.0,
            '_entry_features': dict(entry_features),
            '_ml_features': dict(ml_features),
            '_evidence_snapshot': dict(signal.get('_evidence_snapshot', {}) or {}),
            '_evidence_risk_multiplier': evidence_risk_multiplier,
        }
        if ENABLE_LIVE_TRADING:
            position['_pending_entry'] = True
        
        self.active_positions[pair] = position
        self.active_balance -= position_size  # Reserve capital
        
        # Increment per-pair daily trade counter
        if hasattr(self, '_pair_daily_trades'):
            self._pair_daily_trades[pair] = self._pair_daily_trades.get(pair, 0) + 1
        
        exit_mode, take_profit_pct, trailing_stop_pct, _ = self._get_exit_params(pair, tool, entry_price, {})
        if ENABLE_LIVE_TRADING:
            logger.debug(f"[ENTRY PENDING] {pair} TP/SL deferred until Kraken confirms the entry fill")
        
        # Update daily stats
        self._daily_stats["trades_opened"] += 1
        
        leverage_str = " (2x margin)" if leverage == 2 else ""
        margin_str = f", margin cost: ${margin_opening_cost:.2f}" if leverage == 2 else ""
        
        logger.info(f"[OPEN] {pair} {direction} LIMIT @ ${entry_price:.4f}{leverage_str} | "
                   f"Tool: {tool} | Size: ${position_size:.2f}{margin_str} | "
                   f"Score: {score:.1f} | SL: {stop_loss_pct:.1%}")
        
        # Journal: log open with full context
        self._journal_open(
            pair=pair, tool=tool, direction=direction, price=entry_price,
            score=score, base_score=base_score,
            mtf_multiplier=mtf_m,
            htf_context=htf_ctx, leverage=leverage,
            position_size=position_size, sl_pct=stop_loss_pct,
            hold_bars=planned_hold_hours, reason=signal.get('reason', ''),
            feature_snapshot=entry_features,
            evidence_snapshot=signal.get('_evidence_snapshot', {}) or {},
        )

        # Export to NinjaTrader feed (parallel paper/real channel for CME micros)
        if ENABLE_EXTERNAL_SIGNAL_EXPORT and _NT_EXPORT_AVAILABLE and _nt_export_signal is not None:
            try:
                _nt_export_signal(
                    spot_pair=pair,
                    side=direction,
                    score=float(score),
                    tool=str(tool),
                    entry_price=float(entry_price),
                    stop_pct=float(stop_loss_pct or 0),
                    target_pct=float(take_profit_pct or 0),
                    notional_usd=float(position_size),
                    regime=str(regime_bucket) if regime_bucket else "",
                    notes=f"leverage={leverage}",
                )
            except Exception as _nt_e:
                logger.debug(f"[NT-EXPORT] long signal write failed: {_nt_e}")

        return True
    
    def _classify_fng_value(self, value: int) -> str:
        """Classify Fear & Greed using the standard 0-100 bucket ranges."""
        if value <= 24:
            return "Extreme Fear"
        if value <= 44:
            return "Fear"
        if value <= 54:
            return "Neutral"
        if value <= 74:
            return "Greed"
        return "Extreme Greed"

    def _fng_label_for_value(self, value) -> str:
        """Best-effort Fear & Greed label for journal/status rows."""
        try:
            value_int = int(float(value))
        except (TypeError, ValueError):
            return "unknown"
        if value_int < 0 or value_int > 100:
            return "unknown"
        return self._classify_fng_value(value_int)

    def _parse_fng_timestamp(self, source_ts) -> Tuple[Optional[str], Optional[float]]:
        """Return normalized UTC timestamp and source age in hours when available."""
        if not source_ts:
            return None, None
        try:
            source_text = str(source_ts).strip()
            if source_text.replace('.', '', 1).isdigit():
                parsed_dt = datetime.fromtimestamp(int(float(source_text)), timezone.utc)
            else:
                parsed_dt = datetime.fromisoformat(source_text.replace('Z', '+00:00'))
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                parsed_dt = parsed_dt.astimezone(timezone.utc)
        except Exception:
            return str(source_ts), None

        age_hours = (datetime.now(timezone.utc) - parsed_dt).total_seconds() / 3600.0
        return parsed_dt.isoformat(), age_hours

    def _parse_fng_payload(self, payload: dict) -> dict:
        """Parse common Fear & Greed JSON shapes into one normalized dict."""
        data = payload.get('data') if isinstance(payload, dict) else None
        if isinstance(data, list) and data:
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        elif isinstance(payload, dict):
            entry = payload
        else:
            raise ValueError("Unsupported F&G payload")

        value = entry.get('value', entry.get('fearGreedValue', entry.get('score')))
        if value is None:
            raise ValueError("F&G payload missing value")

        classification = (
            entry.get('value_classification') or
            entry.get('classification') or
            entry.get('name') or
            entry.get('status')
        )
        timestamp = entry.get('timestamp') or entry.get('update_time') or entry.get('updateTime') or entry.get('date')
        return {
            'value': int(float(value)),
            'classification': classification,
            'timestamp': timestamp,
            'time_until_update': entry.get('time_until_update') or entry.get('timeUntilUpdate'),
        }

    def _find_nested_key(self, payload, key: str):
        """Find the first nested dict/list value with the requested key."""
        if isinstance(payload, dict):
            if key in payload:
                return payload[key]
            for value in payload.values():
                found = self._find_nested_key(value, key)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for value in payload:
                found = self._find_nested_key(value, key)
                if found is not None:
                    return found
        return None

    def _fetch_cmc_fng_api(self) -> Optional[dict]:
        """Fetch CMC Fear & Greed through the official API when a key is configured."""
        if not COINMARKETCAP_API_KEY:
            return None

        headers = {
            'Accept': 'application/json',
            'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
        }
        response = requests.get(CMC_FNG_API_URL, params={'limit': 1}, headers=headers, timeout=8)
        response.raise_for_status()
        parsed = self._parse_fng_payload(response.json())
        parsed['source'] = CMC_FNG_API_URL
        parsed['source_provider'] = 'coinmarketcap_api'
        return parsed

    def _extract_cmc_page_current_index(self, html_text: str) -> dict:
        """Extract current F&G from CoinMarketCap's rendered page state."""
        script_match = re.search(
            r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
            html_text,
            re.DOTALL,
        )
        if script_match:
            next_data = json.loads(html.unescape(script_match.group(1)))
            fng_data = self._find_nested_key(next_data, 'fearGreedIndexData')
            if isinstance(fng_data, dict) and isinstance(fng_data.get('currentIndex'), dict):
                return fng_data['currentIndex']

        visible_match = re.search(
            r'data-test=["\']fear-greed-index-num["\'][^>]*>(.*?)</span>',
            html_text,
            re.DOTALL,
        )
        if visible_match:
            visible_text = re.sub(r'<[^>]+>|<!--.*?-->', '', visible_match.group(1), flags=re.DOTALL)
            value_match = re.search(r'\b(\d{1,3})\b', visible_text)
            if value_match:
                return {'score': int(value_match.group(1)), 'name': None, 'updateTime': None}

        raise ValueError("CMC page did not expose fearGreedIndexData.currentIndex")

    def _fetch_cmc_fng_page(self) -> dict:
        """Fetch CMC Fear & Greed from the public CMC page state."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoTradingBot/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        response = requests.get(CMC_FNG_PAGE_URL, headers=headers, timeout=10)
        response.raise_for_status()
        parsed = self._parse_fng_payload(self._extract_cmc_page_current_index(response.text))
        parsed['source'] = CMC_FNG_PAGE_URL
        parsed['source_provider'] = 'coinmarketcap_page'
        return parsed

    def _fng_source_is_fresh(self, parsed: dict) -> bool:
        """Return True when a parsed F&G source is recent enough for live gates."""
        _, source_age_hours = self._parse_fng_timestamp(parsed.get('timestamp'))
        return source_age_hours is None or source_age_hours <= FNG_MAX_SOURCE_AGE_HOURS

    def _fetch_active_fng_source(self) -> dict:
        """Fetch the active crypto Fear & Greed value from the configured provider."""
        if FNG_PROVIDER not in {'coinmarketcap', 'cmc'}:
            raise ValueError(f"Unsupported FNG_PROVIDER={FNG_PROVIDER!r}; expected coinmarketcap")

        errors = []
        try:
            page_result = self._fetch_cmc_fng_page()
            if self._fng_source_is_fresh(page_result):
                return page_result
            _, page_age_hours = self._parse_fng_timestamp(page_result.get('timestamp'))
            errors.append(f"CMC page stale ({page_age_hours:.1f}h old)" if page_age_hours is not None else "CMC page missing timestamp")
        except Exception as exc:
            errors.append(f"CMC page: {exc}")

        try:
            api_result = self._fetch_cmc_fng_api()
            if api_result is not None:
                if self._fng_source_is_fresh(api_result):
                    return api_result
                _, api_age_hours = self._parse_fng_timestamp(api_result.get('timestamp'))
                errors.append(f"CMC API stale ({api_age_hours:.1f}h old)" if api_age_hours is not None else "CMC API missing timestamp")
        except Exception as exc:
            errors.append(f"CMC API: {exc}")

        raise RuntimeError("; ".join(errors) if errors else "No CMC F&G source was available")

    def get_fear_greed(self) -> int:
        """Get the active CoinMarketCap crypto Fear & Greed Index."""

        now = datetime.now(timezone.utc).timestamp()
        if hasattr(self, '_fng_cache') and now - self._fng_cache_ts < FNG_CACHE_TTL_SEC:
            return self._fng_cache
        try:
            parsed = self._fetch_active_fng_source()
            raw_val = parsed['value']
            raw_classification = parsed.get('classification') or self._classify_fng_value(raw_val)
            source_ts = parsed.get('timestamp')
            timestamp_utc, source_age_hours = self._parse_fng_timestamp(source_ts)

            source_stale = source_age_hours is not None and source_age_hours > FNG_MAX_SOURCE_AGE_HOURS
            if source_stale:
                raise RuntimeError(f"CMC F&G source is stale ({source_age_hours:.1f}h old)")

            classification = self._classify_fng_value(raw_val)
            self._fng_cache = raw_val
            self._fng_raw_cache = raw_val
            self._fng_cache_ts = now
            self._fng_meta = {
                'source': parsed.get('source'),
                'classification': classification,
                'raw_classification': raw_classification,
                'raw_value': raw_val,
                'effective_value': raw_val,
                'effective_reason': 'live_source',
                'source_provider': parsed.get('source_provider', 'coinmarketcap'),
                'timestamp_utc': timestamp_utc,
                'source_age_hours': source_age_hours,
                'time_until_update': parsed.get('time_until_update'),
            }
            logger.info(
                f"[FNG] provider={self._fng_meta['source_provider']} source={self._fng_meta['source']} "
                f"value={raw_val} ({classification}) "
                f"timestamp={timestamp_utc or 'unknown'}"
            )
            return raw_val
        except Exception as exc:
            if hasattr(self, '_fng_cache') and now - self._fng_cache_ts <= FNG_LAST_GOOD_MAX_AGE_HOURS * 3600:
                logger.warning(f"[FNG] Active CMC refresh failed; using recent last-good value {self._fng_cache}: {exc}")
                self._fng_meta['effective_reason'] = 'last_good_after_refresh_failure'
                self._fng_meta['error'] = str(exc)
                return self._fng_cache
            raise RuntimeError(f"[FNG] Active CoinMarketCap crypto Fear & Greed unavailable: {exc}") from exc
    
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
            self._short_pressure_by_pair = {}

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
            self._latest_market_data = market_data
            self._update_opportunity_scout_outcomes(market_data)
            
            # 2. Update Fear & Greed index
            fng = self.get_fear_greed()
            self.current_fng = fng
            fng_meta = getattr(self, '_fng_meta', {})
            regime_label = fng_meta.get('classification') or self._classify_fng_value(fng)
            raw_fng = fng_meta.get('raw_value')
            fng_reason = fng_meta.get('effective_reason', '')
            
            # 3. UPGRADE 5: Update total balance from Kraken (fall back to internal)
            kraken_balance = self._sync_kraken_balance()
            if kraken_balance is not None:
                self.total_balance = kraken_balance
            else:
                self.total_balance = self.starting_balance + self.grid_profit + self.active_profit
            
            # 4. Rebalance capital allocation using the latest balance and live positions
            self.rebalance_capital()
            grid_pct, active_pct = self.get_capital_allocation()
            
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
                    logger.opt(exception=True).debug(f"Signal scan failed for {pair}: {e}")
                    continue

            filtered_signals = []
            for signal, score in all_signals:
                tool = signal.get('tool', '')
                direction = signal.get('direction', '')
                forward_mult, quarantine_reason, forward_stats = self._get_forward_tool_score_adjustment(tool, direction)
                if direction == 'long':
                    signal['_forward_tool_mult'] = forward_mult
                if direction == 'long' and forward_mult <= 0.0:
                    logger.info(
                        f"[FORWARD TOOL] {tool} {signal.get('pair', '')} quarantined — "
                        f"recent wr={forward_stats.get('win_rate', 0.0):.2f}, "
                        f"trades={forward_stats.get('trades', 0)}, pnl=${forward_stats.get('pnl', 0.0):.2f}"
                    )
                    self._record_opportunity_scout_candidate(
                        signal, score, quarantine_reason or 'forward_tool_quarantine',
                        'forward_tool_quarantine', market_data
                    )
                    self._log_rejection(
                        signal.get('pair', ''), tool, direction, score,
                        quarantine_reason or 'forward_tool_quarantine'
                    )
                    continue
                policy_reason = self._get_pair_policy_rejection(signal, score)
                if policy_reason:
                    self._record_opportunity_scout_candidate(
                        signal, score, policy_reason, 'pair_policy', market_data
                    )
                    self._log_rejection(signal.get('pair', ''), tool, direction, score, policy_reason)
                    continue
                filtered_signals.append((signal, score))
            all_signals = filtered_signals

            self._current_cycle_signals = all_signals
            prev_market_mode = getattr(self, '_market_short_pressure', {}).get('mode', 'normal')
            self._short_pressure_by_pair = self._build_short_pressure_map(all_signals)
            self._market_short_pressure = self._build_market_short_pressure(all_signals)
            self.rebalance_capital()
            grid_pct, active_pct = self.get_capital_allocation()
            market_pressure = self._market_short_pressure
            if market_pressure.get('mode') != 'normal' or market_pressure.get('mode') != prev_market_mode:
                logger.info(
                    f"[MARKET BEAR] {market_pressure.get('label', 'normal')} — "
                    f"short_pairs={market_pressure.get('short_pairs', 0)}, "
                    f"short_signals={market_pressure.get('short_signals', 0)}, "
                    f"top3={market_pressure.get('top3_avg', 0.0):.1f}, "
                    f"dominance={market_pressure.get('dominance', 0.0):.2f}, "
                    f"active_allocation={active_pct:.0%}"
                )
            
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
            # Bull tools that get regime-gated — avoid picking these as lead if alternatives exist
            _gated_bull_tools = {'accumulation_breakout', 'hurst_trend_long',
                                 'buy_weekly_green', 'buy_breakout_simple',
                                 'simple_buy_uptrend', 'buy_btc_leading', 'major_pair_breakout',
                                 'scout_volume_continuation', 'scout_trend_pullback',
                                 'scout_reversal_followthrough'}
            for (pair, direction), entries in stacked.items():
                # Use the highest-scoring signal as the base, but prefer ungated tools as lead
                entries.sort(key=lambda x: x[1], reverse=True)
                # If lead tool is a gated bull tool and there are ungated alternatives, use those
                ungated = [(s, sc) for s, sc in entries if s['tool'] not in _gated_bull_tools]
                if ungated and entries[0][0]['tool'] in _gated_bull_tools:
                    ungated.sort(key=lambda x: x[1], reverse=True)
                    best_signal, best_score = ungated[0]
                else:
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

            # 8c. EVIDENCE GATE: after stacking, require each long detector to
            # earn capital through walk-forward, live, stacked, or contextual edge.
            evidence_filtered_signals = []
            for signal, score in all_signals:
                allowed, adjusted_score, risk_mult, evidence_reason, evidence_snapshot = self._evaluate_tool_evidence(signal, score)
                if not allowed:
                    self._record_opportunity_scout_candidate(
                        signal, score, evidence_reason or 'evidence_gate', 'evidence_gate', market_data
                    )
                    self._log_rejection(
                        signal.get('pair', ''), signal.get('tool', ''), signal.get('direction', ''),
                        score, evidence_reason or 'evidence_gate'
                    )
                    continue
                signal['_pre_evidence_score'] = score
                signal['_evidence_checked'] = True
                signal['_evidence_risk_multiplier'] = risk_mult
                signal['_evidence_snapshot'] = evidence_snapshot
                if signal.get('direction') == 'long' and (
                    abs(adjusted_score - score) >= 0.2 or abs(risk_mult - 1.0) >= 0.05
                ):
                    logger.info(
                        f"[EVIDENCE] {signal.get('tool')} {signal.get('pair')} "
                        f"tier={evidence_snapshot.get('tier')} score {score:.1f}->{adjusted_score:.1f}, "
                        f"size x{risk_mult:.2f}, live_n={evidence_snapshot.get('live_trades', 0)}, "
                        f"live_pnl=${evidence_snapshot.get('live_pnl_dollar', 0.0):+.2f}"
                    )
                evidence_filtered_signals.append((signal, adjusted_score))
            all_signals = evidence_filtered_signals
            
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
                replacement_used = False
                for signal, score in all_signals:
                    pair = signal['pair']
                    if pair in self.active_positions:
                        self._log_rejection(pair, signal['tool'], signal['direction'], score, "pair_already_open")
                        continue

                    is_trend_leader = self._is_trend_leader_tool(signal['tool'])
                    trend_reserve = self._get_trend_leader_reserve()
                    open_trend_positions = self._count_open_trend_leaders()
                    remaining_trend_reserve = max(0, trend_reserve - open_trend_positions)
                    usable_capacity = MAX_ACTIVE_POSITIONS
                    if not is_trend_leader:
                        usable_capacity -= remaining_trend_reserve

                    if len(self.active_positions) >= usable_capacity:
                        if len(self.active_positions) >= MAX_ACTIVE_POSITIONS:
                            replacement = None
                            if not replacement_used:
                                replacement = self._find_replacement_candidate(signal, score, market_data)

                            if replacement:
                                replacement_used = True
                                logger.info(
                                    f"[REPLACE] Closing {replacement['pair']} for stronger {signal['tool']} {pair} "
                                    f"(score {score:.1f} >= {replacement['score_floor']:.1f}, "
                                    f"pnl {replacement['pnl_pct']:+.1%})"
                                )
                                self.close_position(
                                    replacement['pair'],
                                    replacement['exit_price'],
                                    f"Slot replacement for {signal['tool']} {pair} score {score:.1f}"
                                )
                                self._log_rejection(
                                    pair, signal['tool'], signal['direction'], score,
                                    f"replacement_triggered_{replacement['pair']}"
                                )
                                continue

                            self._log_rejection(pair, signal['tool'], signal['direction'], score, "max_positions_reached")
                        else:
                            self._log_rejection(
                                pair, signal['tool'], signal['direction'], score,
                                f"reserved_for_trend_leaders_{remaining_trend_reserve}"
                            )
                        continue
                    
                    # Check correlation group limits
                    direction = signal['direction']
                    group_count = 0
                    correlation_group = 'other'
                    for group_name, group_pairs in CORRELATION_GROUPS.items():
                        if pair in group_pairs:
                            correlation_group = group_name
                            for open_pair, open_pos in self.active_positions.items():
                                if open_pair in group_pairs and open_pos['direction'] == direction:
                                    group_count += 1
                            break

                    signal['_correlation_group'] = correlation_group
                    
                    if group_count >= MAX_PER_GROUP:
                        logger.debug(f"Skipping {pair} ({direction}) — correlation group limit ({group_count}/{MAX_PER_GROUP})")
                        self._log_rejection(pair, signal['tool'], signal['direction'], score, "correlation_group_limit")
                        continue
                    
                    self.execute_signal(signal, score)
            
            # 10. UPGRADE 8: Enhanced status report
            grid_positions = sum(len(positions) for positions in self.grid_positions.values())
            active_count = len(self.active_positions)
            growth_pct = (self.total_balance / self.starting_balance - 1) * 100
            
            raw_note = f", raw={raw_fng}" if raw_fng is not None and raw_fng != fng else ""
            reason_note = f", {fng_reason}" if fng_reason and fng_reason not in {'source_trusted', 'live_source'} else ""
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CYCLE #{self.current_bar} | "
                       f"F&G: {fng} ({regime_label}{raw_note}{reason_note}) | "
                       f"Allocation: Grid {grid_pct:.0%} / Active {active_pct:.0%}")
            
            logger.info(f"Balance: ${self.total_balance:.2f} (start: ${self.starting_balance:.2f}, {growth_pct:+.1f}%) | "
                       f"Grid: ${self.grid_balance:.2f} | Active: ${self.active_balance:.2f}")
            
            logger.info(f"Grid: {grid_positions} positions across {len(PAIRS)} pairs | "
                       f"{self.grid_round_trips} round trips | ${self.grid_profit:.2f} profit")
            
            logger.info(f"Active: {active_count}/{MAX_ACTIVE_POSITIONS} positions open")

            self._log_forward_diagnostics_snapshot()
            
            if self.active_positions:
                for pair, pos in self.active_positions.items():
                    if pair not in market_data:
                        logger.info(f"  → {pair} {pos['direction']} (no market data this cycle)")
                        continue
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
                    # Track bullish percentage and avg RSI for regime filter
                    self._bullish_4h_pct = regime_counts.get("bullish", 0) / total_pairs * 100
                    self._avg_rsi_4h = avg_rsi_4h
            
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