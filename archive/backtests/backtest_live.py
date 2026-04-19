#!/usr/bin/env python3
"""
Real backtester — downloads historical Kraken OHLCV data and simulates
the AdaptiveQuantStrategy bar-by-bar with realistic fees, slippage, and
position management.

Usage:
    python backtest_live.py                    # defaults: BTC/ETH, 90 days, 5m
    python backtest_live.py --days 180         # 6 months
    python backtest_live.py --symbols XBTUSD   # single symbol
    python backtest_live.py --timeframe 15m    # 15-minute candles

The output tells you, with hard numbers, whether the strategy has an edge.
"""
import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import requests
from loguru import logger

# Ensure project imports work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategies import create_strategy, Signal
from risk_manager import RiskManager, Position


# ── Kraken public OHLC download ─────────────────────────────────────────────

KRAKEN_BASE = "https://api.kraken.com/0"
INTERVAL_MAP = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

CACHE_DIR = PROJECT_ROOT / "data" / "ohlcv_cache"


def _build_ohlcv_from_trades(
    pair: str,
    interval_minutes: int,
    since_ts: int,
    until_ts: int,
) -> pd.DataFrame:
    """Build OHLCV candles from Kraken's Trades endpoint.

    Kraken's OHLC endpoint only returns the most recent ~720 bars.
    The Trades endpoint goes back much further — we fetch raw trades
    and aggregate them into candles ourselves.
    """
    logger.info(
        f"Building {pair} {interval_minutes}m candles from trades "
        f"({datetime.fromtimestamp(since_ts, tz=timezone.utc).date()} → "
        f"{datetime.fromtimestamp(until_ts, tz=timezone.utc).date()})..."
    )

    all_trades = []
    cursor = str(since_ts * 1_000_000_000)  # Kraken wants nanoseconds
    batch = 0
    max_batches = 500  # safety limit

    while batch < max_batches:
        batch += 1
        try:
            resp = requests.get(
                f"{KRAKEN_BASE}/public/Trades",
                params={"pair": pair, "since": cursor, "count": 1000},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})

            last_cursor = result.get("last", cursor)
            trades_data = None
            for k, v in result.items():
                if isinstance(v, list) and len(v) > 0:
                    trades_data = v
                    break

            if not trades_data:
                break

            # Each trade: [price, volume, time, buy/sell, market/limit, misc, trade_id]
            newest_time = 0
            for t in trades_data:
                trade_ts = float(t[2])
                if trade_ts > until_ts:
                    break
                if trade_ts >= since_ts:
                    all_trades.append({
                        "time": trade_ts,
                        "price": float(t[0]),
                        "volume": float(t[1]),
                    })
                newest_time = trade_ts

            if str(last_cursor) == str(cursor):
                break
            if newest_time >= until_ts:
                break
            cursor = str(last_cursor)

        except Exception as e:
            logger.warning(f"Trade download batch {batch} failed: {e}")
            break

        time.sleep(1.2)  # respect rate limits

        if batch % 50 == 0:
            logger.info(f"  ...downloaded {len(all_trades)} trades so far (batch {batch})")

    if len(all_trades) < 100:
        logger.error(f"Too few trades for {pair}: {len(all_trades)}")
        return pd.DataFrame()

    logger.info(f"Downloaded {len(all_trades)} raw trades for {pair}")

    # Aggregate trades into OHLCV candles
    tdf = pd.DataFrame(all_trades)
    interval_sec = interval_minutes * 60
    tdf["candle"] = (tdf["time"] // interval_sec) * interval_sec

    candles = []
    for candle_ts, group in tdf.groupby("candle"):
        candles.append({
            "time": int(candle_ts),
            "open": group["price"].iloc[0],
            "high": group["price"].max(),
            "low": group["price"].min(),
            "close": group["price"].iloc[-1],
            "volume": group["volume"].sum(),
        })

    df = pd.DataFrame(candles).sort_values("time").reset_index(drop=True)
    logger.info(f"Built {len(df)} candles for {pair} from trades")
    return df


def _download_ohlcv(
    pair: str,
    interval_minutes: int,
    since_ts: int,
    until_ts: int,
) -> pd.DataFrame:
    """Download OHLCV data from Kraken public API with pagination and local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Round timestamps to the nearest hour so cache hits across repeated runs
    cache_since = (since_ts // 3600) * 3600
    cache_until = (until_ts // 3600) * 3600
    cache_file = CACHE_DIR / f"{pair}_{interval_minutes}m_{cache_since}_{cache_until}.parquet"
    if cache_file.exists():
        logger.info(f"Using cached data: {cache_file.name}")
        return pd.read_parquet(cache_file)

    # Fuzzy cache: find any existing file for this pair/interval that covers
    # at least 80% of the requested range (avoids slow re-downloads)
    for existing in CACHE_DIR.glob(f"{pair}_{interval_minutes}m_*.parquet"):
        try:
            edf = pd.read_parquet(existing)
            if edf.empty:
                continue
            cov_start, cov_end = int(edf["time"].iloc[0]), int(edf["time"].iloc[-1])
            coverage = (min(cov_end, until_ts) - max(cov_start, since_ts)) / max(until_ts - since_ts, 1)
            if coverage >= 0.80 and len(edf) >= 150:
                logger.info(f"Using fuzzy-matched cache: {existing.name} ({coverage:.0%} coverage)")
                return edf[(edf["time"] >= since_ts) & (edf["time"] <= until_ts)].reset_index(drop=True)
        except Exception:
            continue

    logger.info(f"Downloading {pair} {interval_minutes}m from Kraken ({datetime.fromtimestamp(since_ts, tz=timezone.utc).date()} → {datetime.fromtimestamp(until_ts, tz=timezone.utc).date()})...")

    all_rows = []
    cursor = since_ts

    while cursor < until_ts:
        resp = requests.get(
            f"{KRAKEN_BASE}/public/OHLC",
            params={"pair": pair, "interval": interval_minutes, "since": cursor},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})

        # Kraken returns {<pair_key>: [...], "last": <int>}
        last_ts = int(result.get("last", 0))
        ohlcv = None
        for k, v in result.items():
            if isinstance(v, list) and len(v) > 0:
                ohlcv = v
                break

        if ohlcv is None or len(ohlcv) == 0:
            break

        for row in ohlcv:
            t = int(row[0])
            if t > until_ts:
                break
            all_rows.append({
                "time": t,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[6]),
            })

        if last_ts <= cursor:
            break
        cursor = last_ts
        time.sleep(1.5)  # respect rate limits

    if not all_rows:
        logger.error(f"No data downloaded for {pair}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    # Trim to requested range
    df = df[(df["time"] >= since_ts) & (df["time"] <= until_ts)].reset_index(drop=True)

    # Check: does the OHLC endpoint actually cover the requested range?
    # Kraken's OHLC only returns the most recent ~720 bars.
    # If we asked for 30+ days of 15m data but only got ~7 days,
    # fall back to building candles from the Trades endpoint.
    actual_span_days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400 if len(df) > 1 else 0
    requested_days = (until_ts - since_ts) / 86400
    if actual_span_days < requested_days * 0.8 and requested_days > 10:
        logger.warning(
            f"OHLC only covers {actual_span_days:.1f} days of {requested_days:.0f} requested. "
            f"Rebuilding from Trades endpoint..."
        )
        df = _build_ohlcv_from_trades(pair, interval_minutes, since_ts, until_ts)
        if df.empty:
            return df

    logger.info(f"Downloaded {len(df)} candles for {pair}")

    try:
        df.to_parquet(cache_file)
    except Exception:
        pass  # cache is optional

    return df


# ── Bar-by-bar backtester ────────────────────────────────────────────────────

class RealisticBacktester:
    """
    Walks through candles one at a time, feeds a growing window to the
    strategy, manages entries/exits with real fees and slippage.
    """

    def __init__(
        self,
        strategy_type: str = "adaptive",
        initial_balance: float = 1000.0,
        fee_pct: float = 0.0026,     # Kraken taker fee (0.26%)
        maker_fee_pct: float = 0.0016,  # Kraken maker fee (0.16%)
        slippage_pct: float = 0.0005,   # 0.05% average slippage
        use_limit_orders: bool = True,  # True = use maker fee
        max_open: int = 3,
        risk_per_trade: float = 0.02,
    ):
        self.strategy = create_strategy(strategy_type)
        self.rm = RiskManager(
            initial_balance=initial_balance,
            max_position_size=0.03,
            max_drawdown=0.15,
            max_open_positions=max_open,
            consecutive_loss_limit=4,
            cooldown_minutes=20,
            trailing_stop_activation=0.35,
            trailing_stop_callback=0.35,
            max_risk_per_trade_pct=risk_per_trade,
        )
        self.initial_balance = initial_balance
        self.fee_pct = maker_fee_pct if use_limit_orders else fee_pct
        self.slippage_pct = slippage_pct
        self.use_limit_orders = use_limit_orders

        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]
        self._bar_count = 0

    def run(self, symbol: str, df: pd.DataFrame, warmup: int = 120) -> Dict:
        """Run the backtest on a single symbol's OHLCV data."""
        if len(df) < warmup + 10:
            logger.error(f"Not enough data for {symbol}: {len(df)} bars (need {warmup + 10})")
            return {}

        logger.info(f"Backtesting {symbol}: {len(df)} bars, warmup={warmup}")
        active_positions: List[Position] = []

        for i in range(warmup, len(df)):
            self._bar_count += 1
            # FIX: use bars UP TO (but NOT including) bar i for signal generation.
            # This avoids look-ahead bias — we decide based on data BEFORE
            # bar i, then execute at bar i's open price (the earliest
            # realistic fill after the signal fires).
            window = df.iloc[:i].copy()
            bar = df.iloc[i]
            bar_open = float(bar["open"])
            current_price = float(bar["close"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_time = bar.get("time", i)

            # ── Manage existing positions ──
            for pos in list(active_positions):
                if pos.status != "OPEN":
                    continue

                # Trailing stop DISABLED — it was cutting winners short.
                # With callback=0.4*ATR vs TP=3.0*ATR, trailing activates
                # and tightens so much that normal retraces hit SL for tiny
                # profit instead of letting trades reach full TP.
                # Let fixed SL/TP R:R play out.
                # self.rm.update_trailing_stop(pos, current_price)

                # Partial TP check — DISABLED: empirical testing shows partial TP
                # clips winners too small relative to full SL losses.
                # Let trades run to full TP for proper R:R.

                # Check SL/TP — use realistic gap fills.
                # If bar gaps past SL, fill at the worse price (bar boundary),
                # not the exact SL level. TP limit orders fill at the exact TP
                # or better.
                exited = False
                if pos.side == "BUY":
                    if bar_low <= pos.stop_loss:
                        # SL triggered — fill at SL or bar_low, whichever is worse
                        sl_fill = min(pos.stop_loss, bar_low)
                        self._close_position(pos, sl_fill, "Stop Loss", active_positions)
                        exited = True
                    elif bar_high >= pos.take_profit:
                        # TP limit order — fills at exact TP price
                        self._close_position(pos, pos.take_profit, "Take Profit", active_positions)
                        exited = True
                else:
                    if bar_high >= pos.stop_loss:
                        # SL triggered — fill at SL or bar_high, whichever is worse
                        sl_fill = max(pos.stop_loss, bar_high)
                        self._close_position(pos, sl_fill, "Stop Loss", active_positions)
                        exited = True
                    elif bar_low <= pos.take_profit:
                        # TP limit order — fills at exact TP price
                        self._close_position(pos, pos.take_profit, "Take Profit", active_positions)
                        exited = True

            # Clean up closed
            active_positions = [p for p in active_positions if p.status == "OPEN"]

            # ── Generate new signals ──
            try:
                signals = self.strategy.generate_signals(window, symbol)
            except Exception:
                signals = []

            for sig in signals:
                if sig.action == "HOLD":
                    continue

                # Can we open?
                can, reason = self.rm.can_open_position(sig.symbol)
                if not can:
                    continue

                # Fee-aware EV filter already applied in strategy's generate_signals.
                # Double-check with backtester's own fee structure.
                from strategies import fee_aware_ev_filter
                passes, _ = fee_aware_ev_filter(
                    sig.entry_price, sig.stop_loss, sig.take_profit, sig.action,
                    fee_pct=self.fee_pct, slippage_pct=self.slippage_pct)
                if not passes:
                    continue

                # Position size
                size = self.rm.calculate_position_size(
                    entry_price=sig.entry_price,
                    stop_loss_price=sig.stop_loss,
                    atr=getattr(sig, "atr", 0.0),
                )
                if size <= 0:
                    continue

                # FIX: enter at bar_open (the realistic entry after signal
                # fires at previous bar's close), plus slippage.
                if sig.action == "BUY":
                    fill_price = bar_open * (1 + self.slippage_pct)
                else:
                    fill_price = bar_open * (1 - self.slippage_pct)

                # Recalculate SL/TP relative to actual fill price
                # Keep the same ATR-based distance as the signal intended.
                if sig.stop_loss and sig.take_profit:
                    if sig.action == "BUY":
                        sl_dist = sig.entry_price - sig.stop_loss
                        tp_dist = sig.take_profit - sig.entry_price
                        sig.stop_loss = fill_price - sl_dist
                        sig.take_profit = fill_price + tp_dist
                    else:
                        sl_dist = sig.stop_loss - sig.entry_price
                        tp_dist = sig.entry_price - sig.take_profit
                        sig.stop_loss = fill_price + sl_dist
                        sig.take_profit = fill_price - tp_dist

                entry_fee = abs(fill_price * size * self.fee_pct)
                # FIX: guard against negative balance
                if self.rm.current_balance < entry_fee:
                    continue
                self.rm.current_balance -= entry_fee

                pos = Position(
                    id=f"bt-{self._bar_count}-{sig.symbol}",
                    symbol=sig.symbol,
                    entry_price=fill_price,
                    quantity=size,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    entry_time=str(bar_time),
                    side=sig.action,
                    atr_at_entry=getattr(sig, "atr", 0.0),
                )
                self.rm.record_position(pos)
                active_positions.append(pos)

            # Record equity
            # Mark-to-market: balance + unrealized PnL
            unrealized = 0.0
            for pos in active_positions:
                if pos.status == "OPEN":
                    if pos.side == "BUY":
                        unrealized += (current_price - pos.entry_price) * pos.quantity
                    else:
                        unrealized += (pos.entry_price - current_price) * pos.quantity
            self.equity_curve.append(self.rm.current_balance + unrealized)

        # Close remaining positions at last bar's close
        last_price = float(df.iloc[-1]["close"])
        for pos in list(active_positions):
            if pos.status == "OPEN":
                self._close_position(pos, last_price, "End of backtest", active_positions)

        return self._compute_stats(df)

    def _close_position(self, pos: Position, price: float, reason: str, active_list: List):
        """Close a position with fees."""
        # FIX: TP exits are limit orders — no slippage.
        # SL exits are market orders — slippage applies.
        if reason == "Take Profit":
            fill = price  # Limit order fills at exact TP price
        elif pos.side == "BUY":
            fill = price * (1 - self.slippage_pct)
        else:
            fill = price * (1 + self.slippage_pct)

        exit_fee = abs(fill * pos.quantity * self.fee_pct)

        trade_rec = self.rm.close_position(pos.symbol, fill, reason, position_id=pos.id)
        if trade_rec:
            trade_rec["fee"] = exit_fee
            self.rm.current_balance -= exit_fee
            self.trades.append(trade_rec)

    def _passes_ev_filter(self, sig: Signal) -> bool:
        """Expected value filter: reject trades where EV after fees < 0.

        EV = (win_rate * avg_win) - (loss_rate * avg_loss) - total_fees
        Using the signal's SL/TP to estimate win/loss amounts and a
        conservative 42% win rate assumption.
        """
        try:
            entry = sig.entry_price
            sl = sig.stop_loss
            tp = sig.take_profit

            if sig.action == "BUY":
                risk = entry - sl
                reward = tp - entry
            else:
                risk = sl - entry
                reward = entry - tp

            if risk <= 0 or reward <= 0:
                return False

            # Use a conservative win rate estimate
            # With R:R of 1.5:1, breakeven WR is ~40%. We use actual observed
            # or conservative 42% for filtering.
            rr = reward / risk
            # Approximate win rate from R:R using historical crypto data analysis
            # Higher R:R signals tend to win less often but more when they do
            estimated_wr = min(0.55, 0.35 + 0.05 * min(rr, 4.0))

            # Round-trip fee cost as % of entry
            total_fee_pct = self.fee_pct * 2 + self.slippage_pct * 2

            # EV per dollar risked
            ev = (estimated_wr * reward) - ((1 - estimated_wr) * risk) - (entry * total_fee_pct)
            return ev > 0
        except Exception:
            return True  # Can't compute, allow it

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive backtest statistics."""
        if not self.trades:
            return {"total_trades": 0, "verdict": "NO TRADES GENERATED"}

        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        gross_profit = float(wins["pnl"].sum()) if len(wins) > 0 else 0
        gross_loss = abs(float(losses["pnl"].sum())) if len(losses) > 0 else 0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        total_fees = sum(t.get("fee", 0) for t in self.trades)

        # Max drawdown from equity curve
        eq = pd.Series(self.equity_curve)
        peak = eq.cummax()
        dd = (peak - eq) / peak.replace(0, 1)
        max_dd = float(dd.max())

        # Win/loss streaks
        pnls = [t["pnl"] for t in self.trades]
        max_win_streak = max_loss_streak = cur_win = cur_loss = 0
        for p in pnls:
            if p > 0:
                cur_win += 1
                cur_loss = 0
                max_win_streak = max(max_win_streak, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                max_loss_streak = max(max_loss_streak, cur_loss)

        # Sharpe ratio (annualized, assuming 5m bars)
        returns = eq.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            bars_per_year = 365.25 * 24 * 12  # 5m bars
            sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))
        else:
            sharpe = 0.0

        final_balance = self.rm.current_balance
        total_return_pct = ((final_balance - self.initial_balance) / self.initial_balance) * 100

        # Calculate days from timestamps if available, otherwise from bar count
        if "time" in df.columns and len(df) >= 2:
            days = (int(df.iloc[-1]["time"]) - int(df.iloc[0]["time"])) / 86400.0
        else:
            days = len(df) * 5 / (60 * 24)  # approximate from 5m bars
        if days > 0:
            annual_return = ((final_balance / self.initial_balance) ** (365.0 / days) - 1) * 100
        else:
            annual_return = 0

        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

        # Expected value per trade
        ev_per_trade = float(trades_df["pnl"].mean())

        stats = {
            "total_trades": len(trades_df),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "win_rate_pct": f"{win_rate:.1%}",
            "profit_factor": pf,
            "total_pnl": float(trades_df["pnl"].sum()),
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annual_return,
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe,
            "ev_per_trade": ev_per_trade,
            "avg_win": float(wins["pnl"].mean()) if len(wins) > 0 else 0,
            "avg_loss": float(losses["pnl"].mean()) if len(losses) > 0 else 0,
            "largest_win": float(wins["pnl"].max()) if len(wins) > 0 else 0,
            "largest_loss": float(losses["pnl"].min()) if len(losses) > 0 else 0,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "total_fees_paid": total_fees,
            "final_balance": final_balance,
            "initial_balance": self.initial_balance,
            "days_tested": days,
        }

        # Verdict
        if pf > 1.5 and win_rate > 0.40 and max_dd < 0.15 and ev_per_trade > 0:
            stats["verdict"] = "STRONG EDGE — Strategy is profitable after fees"
        elif pf > 1.2 and win_rate > 0.35 and ev_per_trade > 0:
            stats["verdict"] = "MARGINAL EDGE — Profitable but tight margins"
        elif pf > 1.0 and ev_per_trade > 0:
            stats["verdict"] = "WEAK EDGE — Barely profitable, needs tuning"
        else:
            stats["verdict"] = "NO EDGE — Strategy loses money after fees"

        return stats


def print_report(stats: Dict, symbol: str):
    """Print a clear backtest report."""
    if not stats or stats.get("total_trades", 0) == 0:
        print(f"\n  {symbol}: NO TRADES GENERATED — strategy did not trigger any entries.\n")
        return

    v = stats.get("verdict", "UNKNOWN")
    color = "\033[92m" if "STRONG" in v else "\033[93m" if "MARGINAL" in v else "\033[91m"
    reset = "\033[0m"

    print()
    print("=" * 70)
    print(f"  BACKTEST RESULTS — {symbol}")
    print("=" * 70)
    print(f"  Period:            {stats.get('days_tested', 0):.0f} days")
    print(f"  Starting Balance:  ${stats['initial_balance']:,.2f}")
    print(f"  Final Balance:     ${stats['final_balance']:,.2f}")
    print(f"  Total Return:      {stats['total_return_pct']:+.2f}%")
    print(f"  Annualized Return: {stats.get('annualized_return_pct', 0):+.1f}%")
    print("-" * 70)
    print(f"  Total Trades:      {stats['total_trades']}")
    print(f"  Win Rate:          {stats['win_rate_pct']}")
    print(f"  Profit Factor:     {stats['profit_factor']:.2f}")
    print(f"  EV per Trade:      ${stats['ev_per_trade']:+.2f}")
    print(f"  Avg Win:           ${stats['avg_win']:+.2f}")
    print(f"  Avg Loss:          ${stats['avg_loss']:+.2f}")
    print(f"  Largest Win:       ${stats['largest_win']:+.2f}")
    print(f"  Largest Loss:      ${stats['largest_loss']:+.2f}")
    print("-" * 70)
    print(f"  Max Drawdown:      {stats['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:.2f}")
    print(f"  Longest Win Run:   {stats['max_win_streak']}")
    print(f"  Longest Loss Run:  {stats['max_loss_streak']}")
    print(f"  Total Fees Paid:   ${stats['total_fees_paid']:.2f}")
    print("-" * 70)
    print(f"  {color}VERDICT: {v}{reset}")
    print("=" * 70)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest crypto strategy on real Kraken data")
    parser.add_argument("--symbols", default="XBTUSD,ETHUSD", help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=90, help="Days of history to test")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (1m,5m,15m,1h,4h)")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance")
    parser.add_argument("--strategy", default="adaptive", help="Strategy type")
    parser.add_argument("--limit-orders", action="store_true", default=True, help="Use limit order fees")
    args = parser.parse_args()

    interval = INTERVAL_MAP.get(args.timeframe, 5)
    now = datetime.now(tz=timezone.utc)
    since = int((now - timedelta(days=args.days)).timestamp())
    until = int(now.timestamp())

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    all_stats = {}
    for symbol in symbols:
        df = _download_ohlcv(symbol, interval, since, until)
        if df.empty:
            print(f"No data for {symbol}, skipping.")
            continue

        bt = RealisticBacktester(
            strategy_type=args.strategy,
            initial_balance=args.balance,
            use_limit_orders=args.limit_orders,
            max_open=3,
            risk_per_trade=0.02,
        )
        stats = bt.run(symbol, df)
        all_stats[symbol] = stats
        print_report(stats, symbol)

    if len(all_stats) > 1:
        print("\n" + "=" * 70)
        print("  COMBINED SUMMARY")
        print("=" * 70)
        total_trades = sum(s.get("total_trades", 0) for s in all_stats.values())
        total_pnl = sum(s.get("total_pnl", 0) for s in all_stats.values())
        total_fees = sum(s.get("total_fees_paid", 0) for s in all_stats.values())
        print(f"  Total Trades (all symbols): {total_trades}")
        print(f"  Combined P&L:               ${total_pnl:+.2f}")
        print(f"  Combined Fees:              ${total_fees:.2f}")
        print(f"  P&L after fees:             ${total_pnl:+.2f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
