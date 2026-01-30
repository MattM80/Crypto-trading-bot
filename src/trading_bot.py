"""
Main trading bot engine - orchestrates all components.
"""
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

# Resolve project root (repo root) and ensure imports work regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from the project's .env file (regardless of CWD)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from exchange_client import ExchangeClient
from strategies import create_strategy, Signal
from risk_manager import RiskManager, Position
from config.config import BotConfig, DEFAULT_CONFIG
from kraken_client import KrakenClient

class TradingBot:
    """Main bot engine"""
    
    def __init__(self, config: BotConfig = None, use_kraken: bool = False):
        """Initialize the trading bot
        
        Args:
            config: Bot configuration
            use_kraken: Use Kraken client instead of Binance
        """
        self.config = config or DEFAULT_CONFIG
        self.use_kraken = use_kraken
        self.live_trading_enabled = True
        self.max_signals_per_cycle = int(os.getenv("MAX_SIGNALS_PER_CYCLE", "2"))

        # Safety controls (env-configurable)
        self.allow_multiple_positions_per_symbol = os.getenv("ALLOW_MULTIPLE_POSITIONS_PER_SYMBOL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.kill_switch_stop_bot = os.getenv("KILL_SWITCH_STOP_BOT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.trading_paused = False

        # Trade throttles / filters
        self.max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "0") or "0")
        self.trend_filter_enabled = os.getenv("TREND_FILTER_ENABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.trend_sma_period = int(os.getenv("TREND_SMA_PERIOD", "50") or "50")
        self.trend_require_price_above_sma = os.getenv("TREND_REQUIRE_PRICE_ABOVE_SMA", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.trend_require_sma_up = os.getenv("TREND_REQUIRE_SMA_UP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        def _float_env(name: str, default: float) -> float:
            raw = os.getenv(name, "").strip()
            if raw == "":
                return float(default)
            try:
                return float(raw)
            except Exception:
                logger.warning(f"Invalid {name}={raw!r}; using default {default}")
                return float(default)

        def _int_env(name: str, default: int) -> int:
            raw = os.getenv(name, "").strip()
            if raw == "":
                return int(default)
            try:
                return int(raw)
            except Exception:
                logger.warning(f"Invalid {name}={raw!r}; using default {default}")
                return int(default)
        
        # Initialize components
        if use_kraken:
            # Live trading MUST be explicitly enabled when using Kraken (spot has no sandbox)
            self.live_trading_enabled = os.getenv("ENABLE_LIVE_TRADING", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            if not self.live_trading_enabled:
                logger.warning(
                    "Kraken live trading is DISABLED. Set ENABLE_LIVE_TRADING=true to allow real orders."
                )

            # Use Kraken client (LIVE API - real money)
            self.exchange = KrakenClient(
                api_key=os.getenv("KRAKEN_API_KEY"),
                private_key=os.getenv("KRAKEN_PRIVATE_KEY"),
                sandbox=False  # Kraken has no sandbox for spot
            )
            logger.warning("Using KRAKEN LIVE API - Real Money Trading Active")
        else:
            # Use Binance client
            self.exchange = ExchangeClient(
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                testnet=self.config.exchange.testnet
            )

        # More conservative defaults when live trading is enabled.
        if self.use_kraken and self.live_trading_enabled:
            self.max_notional_per_trade = _float_env("MAX_NOTIONAL_PER_TRADE", 10.0)
            self.max_total_exposure = _float_env("MAX_TOTAL_EXPOSURE", 20.0)
            self.daily_loss_limit = _float_env("DAILY_LOSS_LIMIT", 5.0)
        else:
            # Disabled by default unless the user sets them.
            self.max_notional_per_trade = _float_env("MAX_NOTIONAL_PER_TRADE", 0.0)
            self.max_total_exposure = _float_env("MAX_TOTAL_EXPOSURE", 0.0)
            self.daily_loss_limit = _float_env("DAILY_LOSS_LIMIT", 0.0)
        
        # Determine a starting balance for risk sizing.
        # Note: this bot does NOT reconcile fills/balances with the exchange yet.
        starting_balance_env = os.getenv("STARTING_BALANCE", "").strip()
        initial_balance = float(starting_balance_env) if starting_balance_env else 10000.0
        if use_kraken:
            try:
                balances = self.exchange.get_account_balance() or {}
                # Most Kraken USD balances appear as USD (after our client cleanup)
                usd_balance = balances.get("USD") or balances.get("USDT")
                if usd_balance is not None:
                    initial_balance = float(usd_balance)
                elif self.live_trading_enabled and not starting_balance_env:
                    raise RuntimeError(
                        "Could not determine USD/USDT balance from Kraken. "
                        "Enable 'Query Funds' permission for your API key or set STARTING_BALANCE in .env."
                    )
            except Exception as e:
                if self.live_trading_enabled and not starting_balance_env:
                    raise
                logger.warning(f"Could not fetch Kraken balance for sizing: {e}")

        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            max_position_size=self.config.risk_management.max_position_size,
            max_drawdown=self.config.risk_management.max_drawdown,
            max_open_positions=_int_env("MAX_OPEN_POSITIONS", self.config.risk_management.max_open_positions),
            allow_multiple_positions_per_symbol=self.allow_multiple_positions_per_symbol,
        )

        # Cache risk bracket percents for recomputing TP/SL on real fills.
        # These can be overridden via .env for quick tuning.
        self.stop_loss_percent = _float_env(
            "STOP_LOSS_PERCENT",
            float(self.config.risk_management.stop_loss_percent),
        )
        self.take_profit_percent = _float_env(
            "TAKE_PROFIT_PERCENT",
            float(self.config.risk_management.take_profit_percent),
        )
        
        # Create strategy
        strategy_params = {}
        if self.config.trading_strategy.strategy_type == "grid":
            strategy_params = {
                "grid_levels": self.config.trading_strategy.grid_levels,
                "range_percent": self.config.trading_strategy.grid_range_percent,
                "stop_loss_percent": self.config.risk_management.stop_loss_percent,
                "take_profit_percent": self.config.risk_management.take_profit_percent,
            }
        elif self.config.trading_strategy.strategy_type == "mean_reversion":
            strategy_params = {
                "rsi_period": self.config.trading_strategy.rsi_period,
                "rsi_overbought": self.config.trading_strategy.rsi_overbought,
                "rsi_oversold": self.config.trading_strategy.rsi_oversold,
                "bb_period": self.config.trading_strategy.bollinger_period,
                "bb_std": self.config.trading_strategy.bollinger_std,
            }
        
        self.strategy = create_strategy(
            self.config.trading_strategy.strategy_type,
            **strategy_params
        )
        
        self.is_running = False
        self.active_symbols = self.config.trading_strategy.symbols

        # Balance syncing/caching (Kraken)
        self.balance_refresh_seconds = int(os.getenv("BALANCE_REFRESH_SECONDS", "10"))
        self._cached_balances: Dict[str, float] = {}
        self._cached_balances_ts: float = 0.0

        # Live order/position syncing (Kraken)
        self.entry_order_type = os.getenv("ENTRY_ORDER_TYPE", "LIMIT").strip().upper() or "LIMIT"
        self.exit_order_type = os.getenv("EXIT_ORDER_TYPE", "MARKET").strip().upper() or "MARKET"

        # Kraken equity (mark-to-market) baseline for session-level deltas
        self._kraken_equity_baseline: Optional[float] = None
        
        logger.info(f"Bot initialized with {self.config.trading_strategy.strategy_type} strategy")

    def _compute_kraken_equity_mtm(self, balances: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Compute a simple Kraken mark-to-market equity in USD.

        Equity = USD balance + sum(base_balance * last_price) for active symbols whose quote is USD.
        This is best-effort and intended for sanity-checking performance against Kraken.
        """
        if not self.use_kraken or not balances:
            return None

        try:
            usd_balance = float(balances.get("USD", 0) or 0)
        except Exception:
            usd_balance = 0.0

        bases_seen = set()
        base_value_usd = 0.0

        for symbol in self.active_symbols:
            try:
                base, quote = self.exchange.get_pair_assets(symbol)
                if not base or not quote:
                    continue
                if quote != "USD":
                    continue
                if base in bases_seen:
                    continue
                bases_seen.add(base)

                base_bal = float(balances.get(base, 0) or 0)
                if base_bal <= 0:
                    continue
                ticker = self.exchange.get_ticker(symbol)
                if not ticker or "price" not in ticker:
                    continue
                price = float(ticker["price"])
                if price <= 0:
                    continue
                base_value_usd += base_bal * price
            except Exception:
                continue

        equity = usd_balance + base_value_usd
        return {"equity_usd": equity, "usd": usd_balance, "base_value_usd": base_value_usd}

    def _today_entry_count(self) -> int:
        """Count entries placed today from local position records."""
        today = datetime.now().date()
        count = 0
        try:
            for positions in self.risk_manager.positions.values():
                for p in positions:
                    try:
                        d = datetime.fromisoformat(p.entry_time).date()
                    except Exception:
                        continue
                    if d == today:
                        count += 1
        except Exception:
            return 0
        return count

    def _passes_trend_filter(self, df: pd.DataFrame, action: str) -> bool:
        """Simple SMA trend filter. Intended to reduce long entries in downtrends."""
        if not self.trend_filter_enabled:
            return True

        if (action or "").upper() != "BUY":
            return True

        try:
            period = max(int(self.trend_sma_period), 2)
            closes = pd.to_numeric(df["close"], errors="coerce")
            if closes.isna().all() or len(closes) < period + 1:
                return True
            sma = closes.rolling(period).mean()
            last_close = float(closes.iloc[-1])
            last_sma = float(sma.iloc[-1])
            prev_sma = float(sma.iloc[-2])

            if self.trend_require_price_above_sma and last_close < last_sma:
                return False
            if self.trend_require_sma_up and last_sma < prev_sma:
                return False

            return True
        except Exception:
            return True

    def _recalculate_brackets(self, position: Position) -> None:
        """Recompute stop_loss/take_profit around position.entry_price using configured percents."""
        try:
            entry = float(position.entry_price)
            slp = float(self.stop_loss_percent)
            tpp = float(self.take_profit_percent)
        except Exception:
            return

        if entry <= 0:
            return

        if (position.side or "").upper() == "BUY":
            position.stop_loss = entry * (1.0 - slp)
            position.take_profit = entry * (1.0 + tpp)
        else:
            position.stop_loss = entry * (1.0 + slp)
            position.take_profit = entry * (1.0 - tpp)

    def _compute_fill_price(self, order_info: Dict, fallback_price: float) -> float:
        """Best-effort fill price computation from Kraken QueryOrders fields."""
        try:
            cost = float(order_info.get("cost") or 0)
            vol_exec = float(order_info.get("vol_exec") or 0)
            if cost > 0 and vol_exec > 0:
                return cost / vol_exec
        except Exception:
            pass
        try:
            p = float(order_info.get("price") or 0)
            if p > 0:
                return p
        except Exception:
            pass
        return float(fallback_price)

    def _sync_positions_from_kraken_orders(self) -> None:
        """For Kraken live trading: update local position statuses based on real order state."""
        if not self.use_kraken:
            return

        # Collect txids we care about
        txids: List[str] = []
        for positions in self.risk_manager.positions.values():
            for p in positions:
                if p.status == "PENDING_ENTRY" and p.entry_order_id:
                    txids.append(p.entry_order_id)
                if p.status == "PENDING_EXIT" and p.exit_order_id:
                    txids.append(p.exit_order_id)

        if not txids:
            return

        orders = self.exchange.query_orders(list(dict.fromkeys(txids)))  # de-dupe, keep order
        if not orders:
            return

        for symbol, positions in self.risk_manager.positions.items():
            if not positions:
                continue

            for p in positions:
                if p.status == "PENDING_ENTRY" and p.entry_order_id:
                    info = orders.get(p.entry_order_id)
                    if info and str(info.get("status", "")).lower() == "closed":
                        try:
                            vol_exec = float(info.get("vol_exec") or 0)
                            if vol_exec > 0:
                                p.quantity = vol_exec
                        except Exception:
                            pass
                        p.entry_price = self._compute_fill_price(info, p.entry_price)
                        # Update brackets around the real fill price to avoid instant TP/SL from stale signal levels
                        self._recalculate_brackets(p)
                        p.status = "OPEN"
                        logger.info(
                            f"Entry filled on Kraken: {p.symbol} qty={p.quantity:.8f} @ {p.entry_price:.2f}"
                        )

                if p.status == "PENDING_EXIT" and p.exit_order_id:
                    info = orders.get(p.exit_order_id)
                    if info and str(info.get("status", "")).lower() == "closed":
                        exit_price = self._compute_fill_price(info, p.entry_price)
                        # Close local position only after Kraken confirms exit filled
                        self.risk_manager.close_position(symbol, exit_price, p.exit_reason or "Exit", position_id=p.id)
                        logger.info(f"Exit filled on Kraken: {p.symbol} @ {exit_price:.2f}")

    def _get_daily_realized_pnl(self) -> float:
        """Compute today's realized PnL from the local trade history."""
        try:
            today = datetime.now().date()
            total = 0.0
            for t in self.risk_manager.trade_history:
                ts = t.get("exit_time")
                if not ts:
                    continue
                try:
                    d = datetime.fromisoformat(ts).date()
                except Exception:
                    continue
                if d == today:
                    total += float(t.get("pnl") or 0)
            return total
        except Exception:
            return 0.0

    def _current_total_exposure_quote(self) -> float:
        """Approx exposure in quote currency for OPEN/PENDING_ENTRY positions (USD/USDT pairs only)."""
        exposure = 0.0
        try:
            for symbol, positions in self.risk_manager.positions.items():
                for p in positions:
                    if p.status not in {"OPEN", "PENDING_ENTRY"}:
                        continue
                    try:
                        _, quote = self.exchange.get_pair_assets(symbol)
                    except Exception:
                        quote = None
                    if quote not in {"USD", "USDT"}:
                        continue
                    exposure += float(p.entry_price) * float(p.quantity)
        except Exception:
            return exposure
        return exposure

    def _place_exit_order(self, position: Position, current_price: float, reason: str) -> bool:
        """Place the real exit order on Kraken for an OPEN position."""
        if not (self.use_kraken and self.live_trading_enabled):
            return False

        exit_side = "SELL" if position.side == "BUY" else "BUY"
        order_type = self.exit_order_type
        price = None if order_type == "MARKET" else float(current_price)

        quantity = float(position.quantity)

        # Fix for "EOrder:Insufficient funds" on sell-side due to fees reducing balance
        if exit_side == "SELL":
            try:
                base_asset, _ = self.exchange.get_pair_assets(position.symbol)
                # Use cached balances which should be fresh from the start of the loop
                balances = self._get_live_balances()
                available = float(balances.get(base_asset, 0) or 0)
                
                # If we're trying to sell more than we own, clamp it.
                if available > 0 and quantity > available:
                    logger.warning(
                        f"Adjusting exit qty for {position.symbol} from {quantity} to {available} "
                        f"to match available balance (fees likely reduced holdings)."
                    )
                    quantity = available
                    # Update local position to reflect reality
                    position.quantity = quantity
            except Exception as e:
                logger.warning(f"Could not check balance for exit adjustment: {e}")

        if quantity <= 0:
            logger.error(f"Cannot place exit order for {position.symbol}: quantity is {quantity}")
            return False

        order = self.exchange.place_order(
            symbol=position.symbol,
            side=exit_side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        if not order:
            logger.error(f"Failed to place exit order for {position.symbol}")
            return False

        txid = None
        try:
            txid = (order.get("txid") or [None])[0]
        except Exception:
            txid = None

        if not txid:
            logger.warning(f"Exit order placed but no txid returned for {position.symbol}")
            return False

        position.status = "PENDING_EXIT"
        position.exit_order_id = txid
        position.exit_reason = reason
        logger.info(f"Exit order submitted: {exit_side} {position.symbol} (txid={txid})")
        return True

    def _get_live_balances(self, force: bool = False) -> Dict[str, float]:
        """Fetch Kraken balances with a small TTL cache to avoid spamming the API."""

        if not self.use_kraken:
            return {}

        now = datetime.now().timestamp()
        ttl = max(self.balance_refresh_seconds, 0)
        if not force and ttl > 0 and (now - self._cached_balances_ts) < ttl:
            return self._cached_balances

        balances = self.exchange.get_account_balance() or {}
        self._cached_balances = balances
        self._cached_balances_ts = now
        return balances

    def _get_quote_balance_for_symbol(self, symbol: str, balances: Dict[str, float]) -> Optional[float]:
        """Return available quote balance for the given trading pair (e.g. XBTUSD -> USD balance)."""
        try:
            _, quote_asset = self.exchange.get_pair_assets(symbol)
            if not quote_asset:
                return None
            return float(balances.get(quote_asset, 0) or 0)
        except Exception:
            return None

    def _log_relevant_balances(self, balances: Dict[str, float]) -> None:
        """Log balances relevant to configured symbols (bases/quotes), if available."""
        if not balances:
            return

        relevant_assets = set()
        try:
            for symbol in self.active_symbols:
                base, quote = self.exchange.get_pair_assets(symbol)
                if base:
                    relevant_assets.add(base)
                if quote:
                    relevant_assets.add(quote)
        except Exception:
            pass

        if not relevant_assets:
            return

        parts = []
        for asset in sorted(relevant_assets):
            try:
                val = float(balances.get(asset, 0) or 0)
            except Exception:
                continue
            if val > 0:
                parts.append(f"{asset}={val:.8f}" if val < 1 else f"{asset}={val:.2f}")

        if parts:
            logger.info("Kraken balances: " + ", ".join(parts))

    def _format_pct(self, value: float) -> str:
        try:
            return f"{value:+.2f}%"
        except Exception:
            return "+0.00%"

    def _log_kraken_equity_summary(self, balances: Dict[str, float]) -> None:
        """Log a single beginner-friendly Kraken money summary.

        This is the *real* exchange view:
        - Cash = USD you can spend
        - Holdings = crypto you own
        - Equity = cash + (holdings valued at the latest price)
        """
        try:
            if not self.use_kraken or not balances:
                return

            mtm = self._compute_kraken_equity_mtm(balances)
            if not mtm or mtm.get("equity_usd") is None:
                return

            equity = float(mtm["equity_usd"])
            usd_cash = float(mtm.get("usd", 0.0) or 0.0)
            crypto_value = float(mtm.get("base_value_usd", 0.0) or 0.0)

            # Build a compact holdings list for the configured symbols.
            holdings_parts = []
            bases_seen = set()
            for symbol in self.active_symbols:
                base, quote = self.exchange.get_pair_assets(symbol)
                if not base or base in bases_seen:
                    continue
                bases_seen.add(base)
                qty = float(balances.get(base, 0) or 0)
                if qty > 0:
                    holdings_parts.append(f"{base}={qty:.8f}" if qty < 1 else f"{base}={qty:.2f}")

            holdings_str = ", ".join(holdings_parts) if holdings_parts else "(none)"

            if self._kraken_equity_baseline is None:
                self._kraken_equity_baseline = equity

            baseline = float(self._kraken_equity_baseline or equity)
            delta = equity - baseline
            pct = (delta / baseline * 100.0) if baseline else 0.0

            logger.info(
                "Kraken money (REAL): "
                f"Cash USD=${usd_cash:.2f} | Holdings: {holdings_str} | "
                f"Total Equity≈${equity:.2f} (cash ${usd_cash:.2f} + crypto ${crypto_value:.2f}) | "
                f"Change since start: ${delta:+.2f} ({self._format_pct(pct)})"
            )
        except Exception:
            return
    
    def analyze_signals(self) -> List[Signal]:
        """Analyze all symbols and generate trading signals"""
        all_signals = []
        
        for symbol in self.active_symbols:
            try:
                # Fetch market data
                klines = self.exchange.get_klines(
                    symbol=symbol,
                    interval=self.config.trading_strategy.timeframe,
                    limit=100
                )
                
                if not klines:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(klines)
                
                # Generate signals
                signals = self.strategy.generate_signals(df, symbol)
                # Optional trend filter
                filtered = []
                for s in signals:
                    try:
                        if self._passes_trend_filter(df, s.action):
                            filtered.append(s)
                    except Exception:
                        filtered.append(s)

                all_signals.extend(filtered)
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return all_signals
    
    def execute_signal(self, signal: Signal) -> bool:
        """Execute a trading signal"""

        if self.trading_paused:
            logger.warning("Trading is paused by kill-switch; skipping new entries")
            return False

        if self.max_trades_per_day > 0:
            todays = self._today_entry_count()
            if todays >= self.max_trades_per_day:
                logger.warning(
                    f"Max trades per day reached ({self.max_trades_per_day}); skipping new entry for {signal.symbol}"
                )
                return False

        if self.use_kraken and not self.live_trading_enabled:
            logger.warning(
                f"DRY RUN (Kraken): would place {signal.action} {signal.symbol} @ {signal.entry_price:.2f}"
            )
            return False

        # Check if we can open position
        can_trade, reason = self.risk_manager.can_open_position(signal.symbol)
        if not can_trade:
            logger.warning(f"Cannot trade {signal.symbol}: {reason}")
            return False
        
        # Choose an entry reference price for sizing/brackets.
        # For MARKET entries, use current ticker price to avoid using stale strategy levels.
        entry_ref_price = float(signal.entry_price)
        if self.use_kraken and self.entry_order_type == "MARKET":
            try:
                t = self.exchange.get_ticker(signal.symbol)
                if t and float(t.get("price") or 0) > 0:
                    entry_ref_price = float(t["price"])
            except Exception:
                pass

        # Calculate position size. If using Kraken, size off the LIVE quote balance (not a fixed $10k).
        balances: Dict[str, float] = {}
        live_quote_balance: Optional[float] = None
        if self.use_kraken:
            try:
                balances = self._get_live_balances()
                live_quote_balance = self._get_quote_balance_for_symbol(signal.symbol, balances)
            except Exception as e:
                logger.warning(f"Could not fetch Kraken balances for sizing: {e}")

        position_size = self.risk_manager.calculate_position_size(
            entry_price=entry_ref_price,
            stop_loss_price=(entry_ref_price * (1.0 - float(self.stop_loss_percent)))
            if (signal.action or "").upper() == "BUY"
            else (entry_ref_price * (1.0 + float(self.stop_loss_percent))),
            account_balance=live_quote_balance,
        )
        
        if position_size <= 0:
            logger.warning(f"Invalid position size for {signal.symbol}")
            return False

        # Safety cap: max notional per trade (quote currency; applies to USD/USDT pairs)
        try:
            _, quote_asset = self.exchange.get_pair_assets(signal.symbol)
        except Exception:
            quote_asset = None

        try:
            notional = float(entry_ref_price) * float(position_size)
        except Exception:
            notional = 0.0

        if self.max_notional_per_trade > 0 and quote_asset in {"USD", "USDT"} and notional > self.max_notional_per_trade:
            new_qty = float(self.max_notional_per_trade) / float(entry_ref_price)
            logger.info(
                f"Capping notional for {signal.symbol} from {notional:.2f} to {self.max_notional_per_trade:.2f} {quote_asset}"
            )
            position_size = max(new_qty, 0)

        # Safety cap: max total exposure (USD/USDT pairs only)
        if self.max_total_exposure > 0 and quote_asset in {"USD", "USDT"}:
            exposure = self._current_total_exposure_quote()
            if exposure + (float(entry_ref_price) * float(position_size)) > self.max_total_exposure:
                logger.warning(
                    f"Exposure cap hit: current={exposure:.2f}, cap={self.max_total_exposure:.2f} {quote_asset}. Skipping {signal.symbol}."
                )
                return False

        # Kraken spot does not support shorting. Treat SELL signals as "sell existing holdings" only.
        if self.use_kraken and (signal.action or "").upper() == "SELL":
            try:
                base_asset, quote_asset = self.exchange.get_pair_assets(signal.symbol)
            except Exception:
                base_asset, quote_asset = None, None

            base_free = float(balances.get(base_asset or "", 0) or 0) if balances else 0.0
            min_vol = None
            try:
                min_vol = self.exchange.get_min_order_volume(signal.symbol)
            except Exception:
                min_vol = None

            # If we don't have enough base to place a valid sell, skip.
            if base_free <= 0 or (min_vol is not None and base_free < float(min_vol)):
                return False

        # When using Kraken, cap order size to available balances.
        if self.use_kraken and balances:
            try:
                base_asset, quote_asset = self.exchange.get_pair_assets(signal.symbol)
                action = (signal.action or "").upper()
                if action == "BUY":
                    available_quote = float(balances.get(quote_asset or "", 0) or 0)
                    max_qty = (available_quote * 0.95) / float(entry_ref_price)
                else:  # SELL
                    available_base = float(balances.get(base_asset or "", 0) or 0)
                    max_qty = available_base * 0.95

                if max_qty <= 0:
                    logger.warning(
                        f"Insufficient available funds for {signal.symbol} ({base_asset}/{quote_asset}). Skipping."
                    )
                    return False

                if position_size > max_qty:
                    logger.info(
                        f"Capping {signal.symbol} size from {position_size:.8f} to {max_qty:.8f} based on live balance"
                    )
                    position_size = max_qty
            except Exception as e:
                logger.warning(f"Could not cap order size using live balances: {e}")

        # Enforce Kraken minimum order volume (avoid tiny rounding rejections like 4.99e-05 vs 5e-05)
        if self.use_kraken:
            try:
                min_vol = self.exchange.get_min_order_volume(signal.symbol)
            except Exception:
                min_vol = None

            if min_vol is not None and position_size > 0 and position_size < float(min_vol):
                # Only bump to minimum if it still respects caps and available funds.
                desired_qty = float(min_vol)
                desired_notional = float(entry_ref_price) * desired_qty

                # Notional cap check (USD/USDT pairs)
                try:
                    _, q = self.exchange.get_pair_assets(signal.symbol)
                except Exception:
                    q = None
                if self.max_notional_per_trade > 0 and q in {"USD", "USDT"} and desired_notional > self.max_notional_per_trade:
                    logger.warning(
                        f"Skipping {signal.symbol}: Kraken minimum size requires ~{desired_notional:.2f} {q}, "
                        f"but MAX_NOTIONAL_PER_TRADE={self.max_notional_per_trade:.2f}"
                    )
                    return False

                # Exposure cap check (USD/USDT pairs)
                if self.max_total_exposure > 0 and q in {"USD", "USDT"}:
                    exposure = self._current_total_exposure_quote()
                    if exposure + desired_notional > self.max_total_exposure:
                        logger.warning(
                            f"Skipping {signal.symbol}: Kraken minimum size would exceed MAX_TOTAL_EXPOSURE "
                            f"(current={exposure:.2f}, add={desired_notional:.2f}, cap={self.max_total_exposure:.2f} {q})"
                        )
                        return False

                # Live balance check for buys
                if balances and (signal.action or "").upper() == "BUY":
                    try:
                        avail_quote = float(balances.get(q or "", 0) or 0)
                        if desired_notional > (avail_quote * 0.95):
                            logger.warning(
                                f"Skipping {signal.symbol}: Kraken minimum size needs ~{desired_notional:.2f} {q} "
                                f"but available is ~{avail_quote:.2f}"
                            )
                            return False
                    except Exception:
                        pass

                logger.info(
                    f"Bumping {signal.symbol} size from {position_size:.8f} to Kraken minimum {desired_qty:.8f}"
                )
                position_size = desired_qty
        
        # Place entry order.
        # NOTE: LIMIT orders can remain open; MARKET orders should fill immediately but may slip.
        entry_order_type = self.entry_order_type
        entry_price = None if entry_order_type == "MARKET" else float(signal.entry_price)
        order = self.exchange.place_order(
            symbol=signal.symbol,
            side=signal.action,
            order_type=entry_order_type,
            quantity=position_size,
            price=entry_price,
        )
        
        if not order:
            logger.error(f"Failed to place order for {signal.symbol}")
            return False
        
        # Record position
        entry_txid = None
        try:
            entry_txid = (order.get("txid") or [None])[0]
        except Exception:
            entry_txid = None

        position = Position(
            id=(entry_txid or f"local-{datetime.now().timestamp()}"),
            symbol=signal.symbol,
            entry_price=entry_ref_price,
            quantity=position_size,
            stop_loss=entry_ref_price,
            take_profit=entry_ref_price,
            entry_time=datetime.now().isoformat(),
            side=signal.action,
            status=("PENDING_ENTRY" if (self.use_kraken and self.live_trading_enabled) else "OPEN"),
            entry_order_id=entry_txid,
        )
        # Initialize brackets around the chosen entry reference price.
        self._recalculate_brackets(position)
        self.risk_manager.record_position(position)
        
        if self.use_kraken and self.live_trading_enabled:
            logger.info(
                f"Entry order submitted (pending fill): {signal.action} {signal.symbol} @ {entry_ref_price:.2f}"
            )
        else:
            logger.info(f"Signal executed: {signal.action} {signal.symbol} @ {signal.entry_price:.2f}")
        logger.info(f"Reason: {signal.reason}")
        
        return True
    
    def monitor_positions(self) -> None:
        """Monitor open positions and manage exits"""
        
        for symbol in self.active_symbols:
            if symbol not in self.risk_manager.positions:
                continue
            
            positions = self.risk_manager.positions[symbol]
            if not positions:
                continue
            
            # Get current price once per symbol
            ticker = self.exchange.get_ticker(symbol)
            if not ticker:
                continue
            current_price = ticker["price"]

            # Manage each OPEN position independently
            for position in list(positions):
                if position.status != "OPEN":
                    continue

                # --- TIME-BASED EXIT LOGIC (30m timeout, only if not at a loss) ---
                try:
                    entry_time = datetime.fromisoformat(position.entry_time)
                except Exception:
                    entry_time = datetime.now()  # fallback, should not happen
                now = datetime.now()
                timeout = timedelta(minutes=30)
                if now - entry_time >= timeout:
                    # For BUY: only close if current_price >= entry_price
                    # For SELL: only close if current_price <= entry_price
                    if (
                        (position.side == "BUY" and current_price >= position.entry_price)
                        or (position.side == "SELL" and current_price <= position.entry_price)
                    ):
                        reason = "Time-based exit (30m, no loss)"
                        if self.use_kraken and self.live_trading_enabled:
                            self._place_exit_order(position, current_price, reason)
                        else:
                            self.risk_manager.close_position(symbol, current_price, reason, position_id=position.id)
                        logger.info(f"Time-based exit for {symbol} after 30m at favorable price.")
                        continue  # skip further checks for this position

                if position.side == "BUY":
                    if current_price <= position.stop_loss:
                        if self.use_kraken and self.live_trading_enabled:
                            self._place_exit_order(position, current_price, "Stop Loss")
                        else:
                            self.risk_manager.close_position(symbol, current_price, "Stop Loss", position_id=position.id)
                        logger.warning(f"Stop loss triggered for {symbol}")
                    elif current_price >= position.take_profit:
                        if self.use_kraken and self.live_trading_enabled:
                            self._place_exit_order(position, current_price, "Take Profit")
                        else:
                            self.risk_manager.close_position(symbol, current_price, "Take Profit", position_id=position.id)
                        logger.info(f"Take profit hit for {symbol}")
                else:  # SELL
                    if current_price >= position.stop_loss:
                        if self.use_kraken and self.live_trading_enabled:
                            self._place_exit_order(position, current_price, "Stop Loss")
                        else:
                            self.risk_manager.close_position(symbol, current_price, "Stop Loss", position_id=position.id)
                        logger.warning(f"Stop loss triggered for {symbol}")
                    elif current_price <= position.take_profit:
                        if self.use_kraken and self.live_trading_enabled:
                            self._place_exit_order(position, current_price, "Take Profit")
                        else:
                            self.risk_manager.close_position(symbol, current_price, "Take Profit", position_id=position.id)
                        logger.info(f"Take profit hit for {symbol}")
    
    async def run_trading_loop(self, interval: int = 60) -> None:
        """
        Main trading loop
        
        Args:
            interval: Time between checks in seconds
        """
        self.is_running = True
        logger.info(f"Starting trading bot (check interval: {interval}s)")
        
        try:
            while self.is_running:
                try:
                    logger.info("Analyzing signals...")

                    # Daily loss kill-switch (realized PnL from local history)
                    if self.daily_loss_limit > 0:
                        daily_pnl = self._get_daily_realized_pnl()
                        if daily_pnl <= -abs(self.daily_loss_limit):
                            self.trading_paused = True
                            logger.error(
                                f"KILL SWITCH: daily realized PnL {daily_pnl:.2f} <= -{abs(self.daily_loss_limit):.2f}. Pausing new entries."
                            )
                            if self.kill_switch_stop_bot:
                                logger.error("KILL SWITCH configured to stop bot. Exiting loop.")
                                self.is_running = False

                    # Refresh balances once per cycle (Kraken) and log the relevant ones.
                    if self.use_kraken:
                        balances = self._get_live_balances(force=True)
                        if balances:
                            self._log_relevant_balances(balances)

                            # One simple, beginner-friendly summary of *real* Kraken money.
                            self._log_kraken_equity_summary(balances)
                        else:
                            logger.warning(
                                "Could not fetch Kraken balances this cycle (check API key permissions: 'Query Funds')."
                            )

                    signals = self.analyze_signals()
                    
                    # Filter by confidence
                    high_confidence_signals = [s for s in signals if s.confidence >= 0.7]

                    # Hard safety limit: only execute a small number of signals per cycle
                    if self.max_signals_per_cycle > 0:
                        high_confidence_signals = high_confidence_signals[: self.max_signals_per_cycle]
                    else:
                        high_confidence_signals = []
                    
                    logger.info(f"Generated {len(high_confidence_signals)} high-confidence signals")
                    
                    # Execute signals
                    for signal in high_confidence_signals:
                        self.execute_signal(signal)

                    # Sync Kraken orders -> local position status (fills/exits)
                    if self.use_kraken and self.live_trading_enabled:
                        self._sync_positions_from_kraken_orders()
                    
                    # Monitor open positions
                    logger.info("Monitoring open positions...")
                    self.monitor_positions()

                    # After placing exits, sync again so fast-filling market exits are reflected immediately
                    if self.use_kraken and self.live_trading_enabled:
                        self._sync_positions_from_kraken_orders()
                    
                    # Log local strategy/risk-manager stats (these are NOT your live Kraken equity).
                    hide_local = os.getenv("HIDE_LOCAL_STATS", "").strip().lower() in {"1", "true", "yes", "y", "on"}
                    if not hide_local:
                        stats = self.risk_manager.get_portfolio_stats()
                        logger.info(
                            "Bot tracker (estimate): "
                            f"${stats['balance']:.2f} | "
                            f"P&L {stats['total_pnl_percent']:+.2f}% | "
                            f"Trades {stats['total_trades']} | "
                            f"Win rate {stats['win_rate']:.1%} "
                            "(this is NOT Kraken; use 'Kraken money (REAL)' above)"
                        )
                
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                # ...existing code...
                # Wait for next check
                await asyncio.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        
        finally:
            self.is_running = False
            self.log_final_stats()
    
    def log_final_stats(self) -> None:
        """Log final trading statistics"""
        stats = self.risk_manager.get_portfolio_stats()
        
        logger.info("=" * 60)
        logger.info("FINAL TRADING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Final (Simulated) Balance: ${stats['balance']:.2f}")
        logger.info(f"Total P&L: ${stats['total_pnl']:.2f} ({stats['total_pnl_percent']:+.2f}%)")
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(f"Winning Trades: {stats['wins']}")
        logger.info(f"Losing Trades: {stats['losses']}")
        logger.info(f"Win Rate: {stats['win_rate']:.1%}")
        logger.info(f"Max Drawdown: {stats['max_drawdown']:.1%}")
        logger.info(f"Average Win: ${stats['avg_win']:.2f}")
        logger.info(f"Average Loss: ${stats['avg_loss']:.2f}")
        logger.info("=" * 60)
    
    def stop(self) -> None:
        """Stop the trading bot"""
        logger.info("Stopping bot...")
        self.is_running = False

def main():
    """Run the trading bot"""
    # Configure logging
    logger.remove()
    logger.add("logs/trading_bot.log", rotation="500 MB", retention="10 days")
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    
    # Create and run bot
    bot = TradingBot(DEFAULT_CONFIG)
    
    try:
        asyncio.run(bot.run_trading_loop(interval=60))
    except KeyboardInterrupt:
        logger.info("Bot interrupted")
    finally:
        bot.stop()

if __name__ == "__main__":
    main()
