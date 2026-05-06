#!/usr/bin/env python3
"""Walk-forward strategy lab for Kraken-compatible spot crypto trading.

The live bot should not be trusted with new ideas until they survive a basic
out-of-sample check. This lab tests spot-only long strategies on the local 1h
OHLCV set with Kraken-like maker fees, slippage, position caps, trailing stops,
and a $300 starting account.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h_extended"
RESEARCH_DIR = PROJECT_ROOT / "research"

STARTING_BALANCE = 300.0
FEE_PER_SIDE = 0.0019  # 0.16% maker fee + 0.03% assumed slippage
MAX_TOTAL_EXPOSURE = 0.55
MAX_POSITION_PCT = 0.10
MAX_POSITIONS = 4
WARMUP_BARS = 800

BLOCKED_PAIRS = {
    "DRIFTUSD",
    "RAVEUSD",
    "BLUAIUSD",
    "B3USD",
    "GUSD",
    "NIGHTUSD",
    "PTBUSD",
    "GHSTUSD",
}

BTC_PAIR = "XBTUSD"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    family: str
    params: Dict[str, float]
    max_positions: int = MAX_POSITIONS
    position_pct: float = MAX_POSITION_PCT
    max_total_exposure: float = MAX_TOTAL_EXPOSURE


@dataclass
class Signal:
    pair: str
    strategy: str
    family: str
    score: float
    stop_atr: float = 0.0
    stop_pct: float = 0.0
    trail_atr: float = 0.0
    trail_pct: float = 0.0
    take_profit_atr: float = 0.0
    take_profit_pct: float = 0.0
    max_hold_bars: int = 0
    reason: str = ""


@dataclass
class Position:
    pair: str
    strategy: str
    family: str
    entry_bar: int
    entry_time: str
    entry_price: float
    quantity: float
    entry_value: float
    stop_price: float
    take_profit_price: Optional[float]
    trail_atr: float
    trail_pct: float
    max_hold_bars: int
    max_price: float
    entry_fee: float
    score: float


@dataclass
class Trade:
    strategy: str
    pair: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    bars_held: int
    reason: str
    score: float


@dataclass
class BacktestResult:
    config: StrategyConfig
    segment: str
    start_time: str
    end_time: str
    final_balance: float
    return_pct: float
    max_drawdown_pct: float
    trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pct: float
    sharpe: float
    trade_log: List[Trade] = field(default_factory=list)


def normalize_pair_from_file(path: Path) -> str:
    base = path.stem.split("_")[0].replace("USDT", "")
    if base == "BTC":
        return "XBTUSD"
    return f"{base}USD"


def true_range(data_frame: pd.DataFrame) -> pd.Series:
    previous_close = data_frame["close"].shift(1)
    ranges = pd.concat(
        [
            data_frame["high"] - data_frame["low"],
            (data_frame["high"] - previous_close).abs(),
            (data_frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def add_indicators(data_frame: pd.DataFrame) -> pd.DataFrame:
    enriched = data_frame.copy()
    close = enriched["close"]
    enriched["ret_1h"] = close.pct_change(1)
    enriched["ret_4h"] = close.pct_change(4)
    enriched["ret_24h"] = close.pct_change(24)
    enriched["ret_7d"] = close.pct_change(168)
    enriched["ret_30d"] = close.pct_change(720)
    enriched["sma20"] = close.rolling(20).mean()
    enriched["sma50"] = close.rolling(50).mean()
    enriched["sma100"] = close.rolling(100).mean()
    enriched["sma200"] = close.rolling(200).mean()
    enriched["atr14"] = true_range(enriched).rolling(14).mean()
    enriched["atr_pct"] = enriched["atr14"] / close
    enriched["rsi14"] = rsi(close)
    enriched["vol_sma24"] = enriched["volume"].rolling(24).mean()
    enriched["vol_sma72"] = enriched["volume"].rolling(72).mean()
    enriched["vol_ratio24"] = enriched["volume"] / enriched["vol_sma24"].replace(0, np.nan)
    enriched["realized_vol_30d"] = close.pct_change().rolling(720).std() * np.sqrt(720)
    rolling_std = close.rolling(20).std()
    enriched["bb_width20"] = (4 * rolling_std) / close
    enriched["bb_width_rank240"] = enriched["bb_width20"].rolling(240).rank(pct=True)
    return enriched


def load_market_data(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    raw_data: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*_1h.csv")):
        pair = normalize_pair_from_file(csv_path)
        if pair in BLOCKED_PAIRS:
            continue
        data_frame = pd.read_csv(csv_path)
        data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"], utc=True)
        for column in ("open", "high", "low", "close", "volume"):
            data_frame[column] = pd.to_numeric(data_frame[column], errors="coerce")
        data_frame = data_frame.dropna(subset=["open", "high", "low", "close", "volume"])
        data_frame = data_frame.sort_values("timestamp").reset_index(drop=True)
        raw_data[pair] = data_frame

    if BTC_PAIR not in raw_data:
        raise RuntimeError("BTC/XBT data is required for regime filters")

    common_timestamps: Optional[set] = None
    for data_frame in raw_data.values():
        timestamps = set(data_frame["timestamp"])
        common_timestamps = timestamps if common_timestamps is None else common_timestamps & timestamps

    if not common_timestamps:
        raise RuntimeError("No common timestamps across market data")

    ordered_timestamps = sorted(common_timestamps)
    aligned_data: Dict[str, pd.DataFrame] = {}
    for pair, data_frame in raw_data.items():
        aligned = data_frame[data_frame["timestamp"].isin(ordered_timestamps)].copy()
        aligned = aligned.sort_values("timestamp").reset_index(drop=True)
        aligned_data[pair] = add_indicators(aligned)

    return aligned_data


def is_btc_risk_on(market_data: Dict[str, pd.DataFrame], bar_index: int, allow_neutral: bool = False) -> bool:
    btc_data = market_data[BTC_PAIR]
    row = btc_data.iloc[bar_index]
    if pd.isna(row["sma200"]):
        return False
    close = float(row["close"])
    above_sma200 = close > float(row["sma200"])
    not_crashing = float(row["ret_24h"] or 0.0) > -0.035
    if allow_neutral:
        above_sma100 = close > float(row["sma100"])
        return not_crashing and (above_sma200 or above_sma100)
    return above_sma200 and not_crashing


def pair_is_tradeable(row: pd.Series) -> bool:
    if pd.isna(row["sma200"]) or pd.isna(row["atr_pct"]):
        return False
    atr_pct = float(row["atr_pct"])
    return 0.002 <= atr_pct <= 0.12


def relative_strength_scores(
    market_data: Dict[str, pd.DataFrame],
    bar_index: int,
    lookback_short: int,
    lookback_long: int,
) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    for pair, data_frame in market_data.items():
        if pair in BLOCKED_PAIRS:
            continue
        row = data_frame.iloc[bar_index]
        if not pair_is_tradeable(row):
            continue
        if float(row["close"]) <= float(row["sma200"]):
            continue
        short_return = float(data_frame["close"].iloc[bar_index] / data_frame["close"].iloc[bar_index - lookback_short] - 1)
        long_return = float(data_frame["close"].iloc[bar_index] / data_frame["close"].iloc[bar_index - lookback_long] - 1)
        realized_vol = float(row["realized_vol_30d"] if not pd.isna(row["realized_vol_30d"]) else 0.5)
        if long_return <= 0:
            continue
        score = (0.55 * short_return + 0.45 * long_return) - (0.20 * realized_vol)
        scores.append((pair, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def generate_trend_breakout_signals(
    config: StrategyConfig,
    market_data: Dict[str, pd.DataFrame],
    bar_index: int,
) -> List[Signal]:
    if not is_btc_risk_on(market_data, bar_index):
        return []

    donchian_bars = int(config.params["donchian_bars"])
    volume_floor = float(config.params["volume_floor"])
    signals: List[Signal] = []
    for pair, data_frame in market_data.items():
        if pair in BLOCKED_PAIRS or bar_index < donchian_bars + 1:
            continue
        row = data_frame.iloc[bar_index]
        if not pair_is_tradeable(row):
            continue
        close = float(row["close"])
        prior_high = float(data_frame["high"].iloc[bar_index - donchian_bars:bar_index].max())
        volume_ratio = float(row["vol_ratio24"] if not pd.isna(row["vol_ratio24"]) else 0.0)
        if close <= prior_high:
            continue
        if close <= float(row["sma200"]) or float(row["sma50"]) <= float(row["sma200"]):
            continue
        if volume_ratio < volume_floor:
            continue
        ret_7d = float(row["ret_7d"] if not pd.isna(row["ret_7d"]) else 0.0)
        score = ((close / prior_high) - 1.0) * 500.0 + volume_ratio * 4.0 + ret_7d * 20.0
        signals.append(
            Signal(
                pair=pair,
                strategy=config.name,
                family=config.family,
                score=score,
                stop_atr=float(config.params["stop_atr"]),
                trail_atr=float(config.params["trail_atr"]),
                max_hold_bars=int(config.params["max_hold_bars"]),
                reason=f"donchian_{donchian_bars}_breakout",
            )
        )
    return signals


def generate_squeeze_breakout_signals(
    config: StrategyConfig,
    market_data: Dict[str, pd.DataFrame],
    bar_index: int,
) -> List[Signal]:
    if not is_btc_risk_on(market_data, bar_index, allow_neutral=True):
        return []

    channel_bars = int(config.params["channel_bars"])
    volume_floor = float(config.params["volume_floor"])
    squeeze_rank = float(config.params["squeeze_rank"])
    signals: List[Signal] = []
    for pair, data_frame in market_data.items():
        if pair in BLOCKED_PAIRS or bar_index < channel_bars + 240:
            continue
        row = data_frame.iloc[bar_index]
        if not pair_is_tradeable(row):
            continue
        close = float(row["close"])
        prior_high = float(data_frame["high"].iloc[bar_index - channel_bars:bar_index].max())
        volume_ratio = float(row["vol_ratio24"] if not pd.isna(row["vol_ratio24"]) else 0.0)
        width_rank = float(row["bb_width_rank240"] if not pd.isna(row["bb_width_rank240"]) else 1.0)
        if width_rank > squeeze_rank or close <= prior_high or volume_ratio < volume_floor:
            continue
        if close <= float(row["sma100"]):
            continue
        score = (squeeze_rank - width_rank) * 20.0 + volume_ratio * 5.0 + ((close / prior_high) - 1.0) * 400.0
        signals.append(
            Signal(
                pair=pair,
                strategy=config.name,
                family=config.family,
                score=score,
                stop_atr=float(config.params["stop_atr"]),
                trail_atr=float(config.params["trail_atr"]),
                take_profit_atr=float(config.params["take_profit_atr"]),
                max_hold_bars=int(config.params["max_hold_bars"]),
                reason=f"squeeze_{channel_bars}_breakout",
            )
        )
    return signals


def generate_panic_reversal_signals(
    config: StrategyConfig,
    market_data: Dict[str, pd.DataFrame],
    bar_index: int,
) -> List[Signal]:
    btc_row = market_data[BTC_PAIR].iloc[bar_index]
    if float(btc_row["ret_24h"] if not pd.isna(btc_row["ret_24h"]) else 0.0) < -0.08:
        return []

    drop_floor = float(config.params["drop_floor"])
    volume_floor = float(config.params["volume_floor"])
    wick_floor = float(config.params["wick_floor"])
    signals: List[Signal] = []
    for pair, data_frame in market_data.items():
        if pair in BLOCKED_PAIRS:
            continue
        row = data_frame.iloc[bar_index]
        if not pair_is_tradeable(row):
            continue
        ret_24h = float(row["ret_24h"] if not pd.isna(row["ret_24h"]) else 0.0)
        rsi_value = float(row["rsi14"] if not pd.isna(row["rsi14"]) else 50.0)
        volume_ratio = float(row["vol_ratio24"] if not pd.isna(row["vol_ratio24"]) else 0.0)
        candle_range = float(row["high"] - row["low"])
        if candle_range <= 0:
            continue
        lower_wick = float(min(row["open"], row["close"]) - row["low"])
        lower_wick_ratio = lower_wick / candle_range
        green_close = float(row["close"]) > float(row["open"])
        if ret_24h > -drop_floor or rsi_value > 30 or volume_ratio < volume_floor:
            continue
        if lower_wick_ratio < wick_floor or not green_close:
            continue
        score = abs(ret_24h) * 100.0 + (30.0 - rsi_value) + volume_ratio * 4.0 + lower_wick_ratio * 10.0
        signals.append(
            Signal(
                pair=pair,
                strategy=config.name,
                family=config.family,
                score=score,
                stop_pct=float(config.params["stop_pct"]),
                take_profit_pct=float(config.params["take_profit_pct"]),
                max_hold_bars=int(config.params["max_hold_bars"]),
                reason="panic_reversal_absorption",
            )
        )
    return signals


def generate_relative_strength_signals(
    config: StrategyConfig,
    market_data: Dict[str, pd.DataFrame],
    bar_index: int,
) -> List[Signal]:
    rebalance_bars = int(config.params["rebalance_bars"])
    if bar_index % rebalance_bars != 0:
        return []
    if not is_btc_risk_on(market_data, bar_index):
        return []

    lookback_short = int(config.params["lookback_short"])
    lookback_long = int(config.params["lookback_long"])
    top_n = int(config.params["top_n"])
    ranked = relative_strength_scores(market_data, bar_index, lookback_short, lookback_long)[:top_n]
    signals: List[Signal] = []
    for rank, (pair, score) in enumerate(ranked, start=1):
        if score <= 0:
            continue
        signals.append(
            Signal(
                pair=pair,
                strategy=config.name,
                family=config.family,
                score=score * 100.0 + (top_n - rank + 1),
                stop_pct=float(config.params["stop_pct"]),
                trail_pct=float(config.params["trail_pct"]),
                max_hold_bars=int(config.params["max_hold_bars"]),
                reason=f"relative_strength_rank_{rank}",
            )
        )
    return signals


def generate_signals(
    config: StrategyConfig,
    market_data: Dict[str, pd.DataFrame],
    bar_index: int,
) -> List[Signal]:
    if config.family == "trend_breakout":
        signals = generate_trend_breakout_signals(config, market_data, bar_index)
    elif config.family == "squeeze_breakout":
        signals = generate_squeeze_breakout_signals(config, market_data, bar_index)
    elif config.family == "panic_reversal":
        signals = generate_panic_reversal_signals(config, market_data, bar_index)
    elif config.family == "relative_strength":
        signals = generate_relative_strength_signals(config, market_data, bar_index)
    else:
        signals = []
    signals.sort(key=lambda signal: signal.score, reverse=True)
    return signals


class PortfolioSimulator:
    def __init__(self, market_data: Dict[str, pd.DataFrame], config: StrategyConfig, start_bar: int, end_bar: int):
        self.market_data = market_data
        self.config = config
        self.start_bar = start_bar
        self.end_bar = end_bar
        self.cash = STARTING_BALANCE
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps = list(market_data[BTC_PAIR]["timestamp"])

    def current_equity(self, bar_index: int) -> float:
        equity = self.cash
        for position in self.positions.values():
            close = float(self.market_data[position.pair]["close"].iloc[bar_index])
            equity += position.quantity * close
        return equity

    def current_exposure(self, bar_index: int) -> float:
        equity = max(self.current_equity(bar_index), 1e-9)
        exposure = 0.0
        for position in self.positions.values():
            close = float(self.market_data[position.pair]["close"].iloc[bar_index])
            exposure += position.quantity * close
        return exposure / equity

    def close_position(self, pair: str, bar_index: int, exit_price: float, reason: str) -> None:
        position = self.positions.pop(pair)
        exit_value = position.quantity * exit_price
        exit_fee = exit_value * FEE_PER_SIDE
        self.cash += exit_value - exit_fee
        pnl = (exit_value - exit_fee) - position.entry_value - position.entry_fee
        pnl_pct = pnl / max(position.entry_value + position.entry_fee, 1e-9)
        self.trades.append(
            Trade(
                strategy=position.strategy,
                pair=pair,
                entry_time=position.entry_time,
                exit_time=str(self.timestamps[bar_index]),
                entry_price=position.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                bars_held=bar_index - position.entry_bar,
                reason=reason,
                score=position.score,
            )
        )

    def should_exit_relative_strength(self, position: Position, bar_index: int) -> bool:
        rebalance_bars = int(self.config.params["rebalance_bars"])
        if bar_index % rebalance_bars != 0:
            return False
        if bar_index - position.entry_bar < rebalance_bars:
            return False
        if not is_btc_risk_on(self.market_data, bar_index):
            return True
        ranked = relative_strength_scores(
            self.market_data,
            bar_index,
            int(self.config.params["lookback_short"]),
            int(self.config.params["lookback_long"]),
        )
        keep_pairs = {pair for pair, _ in ranked[: int(self.config.params["top_n"])]}
        return position.pair not in keep_pairs

    def manage_positions(self, bar_index: int) -> None:
        for pair, position in list(self.positions.items()):
            data_frame = self.market_data[pair]
            row = data_frame.iloc[bar_index]
            open_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            if low <= position.stop_price:
                exit_price = open_price if open_price < position.stop_price else position.stop_price
                self.close_position(pair, bar_index, exit_price, "stop")
                continue

            if position.take_profit_price is not None and high >= position.take_profit_price:
                self.close_position(pair, bar_index, position.take_profit_price, "take_profit")
                continue

            position.max_price = max(position.max_price, high)
            if position.trail_atr > 0:
                atr_value = float(row["atr14"] if not pd.isna(row["atr14"]) else 0.0)
                if atr_value > 0:
                    position.stop_price = max(position.stop_price, position.max_price - position.trail_atr * atr_value)
            if position.trail_pct > 0:
                position.stop_price = max(position.stop_price, position.max_price * (1 - position.trail_pct))

            bars_held = bar_index - position.entry_bar
            if position.max_hold_bars and bars_held >= position.max_hold_bars:
                self.close_position(pair, bar_index, close, "max_hold")
                continue

            if position.family in {"trend_breakout", "squeeze_breakout"}:
                sma50 = float(row["sma50"] if not pd.isna(row["sma50"]) else close)
                if bars_held >= 24 and close < sma50:
                    self.close_position(pair, bar_index, close, "lost_sma50")
                    continue

            if position.family == "relative_strength" and self.should_exit_relative_strength(position, bar_index):
                self.close_position(pair, bar_index, close, "rotation_exit")

    def open_position(self, signal: Signal, bar_index: int) -> None:
        if signal.pair in self.positions:
            return
        if len(self.positions) >= self.config.max_positions:
            return
        if self.current_exposure(bar_index) >= self.config.max_total_exposure:
            return

        entry_price = float(self.market_data[signal.pair]["open"].iloc[bar_index])
        if entry_price <= 0:
            return
        equity = self.current_equity(bar_index)
        entry_value = min(equity * self.config.position_pct, self.cash / (1 + FEE_PER_SIDE))
        if entry_value < 5:
            return
        entry_fee = entry_value * FEE_PER_SIDE
        quantity = entry_value / entry_price
        atr_value = float(self.market_data[signal.pair]["atr14"].iloc[bar_index - 1])
        stop_candidates: List[float] = []
        if signal.stop_atr > 0 and atr_value > 0:
            stop_candidates.append(entry_price - signal.stop_atr * atr_value)
        if signal.stop_pct > 0:
            stop_candidates.append(entry_price * (1 - signal.stop_pct))
        stop_price = max(stop_candidates) if stop_candidates else entry_price * 0.92

        take_profit_price: Optional[float] = None
        if signal.take_profit_atr > 0 and atr_value > 0:
            take_profit_price = entry_price + signal.take_profit_atr * atr_value
        if signal.take_profit_pct > 0:
            take_profit_price = entry_price * (1 + signal.take_profit_pct)

        self.cash -= entry_value + entry_fee
        self.positions[signal.pair] = Position(
            pair=signal.pair,
            strategy=signal.strategy,
            family=signal.family,
            entry_bar=bar_index,
            entry_time=str(self.timestamps[bar_index]),
            entry_price=entry_price,
            quantity=quantity,
            entry_value=entry_value,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            trail_atr=signal.trail_atr,
            trail_pct=signal.trail_pct,
            max_hold_bars=signal.max_hold_bars,
            max_price=entry_price,
            entry_fee=entry_fee,
            score=signal.score,
        )

    def run(self, segment: str) -> BacktestResult:
        start_bar = max(self.start_bar, WARMUP_BARS)
        for bar_index in range(start_bar + 1, self.end_bar):
            signal_bar = bar_index - 1
            self.manage_positions(bar_index)
            signals = generate_signals(self.config, self.market_data, signal_bar)
            for signal in signals:
                self.open_position(signal, bar_index)
            self.equity_curve.append(self.current_equity(bar_index))

        final_bar = self.end_bar - 1
        for pair in list(self.positions.keys()):
            final_close = float(self.market_data[pair]["close"].iloc[final_bar])
            self.close_position(pair, final_bar, final_close, "final_close")
        self.equity_curve.append(self.cash)
        return self.build_result(segment)

    def build_result(self, segment: str) -> BacktestResult:
        final_balance = self.cash
        return_pct = (final_balance / STARTING_BALANCE - 1.0) * 100.0
        equity = np.asarray(self.equity_curve, dtype=float) if self.equity_curve else np.asarray([STARTING_BALANCE])
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / np.maximum(running_max, 1e-9)
        max_drawdown_pct = abs(float(drawdowns.min() * 100.0)) if len(drawdowns) else 0.0
        wins = [trade.pnl for trade in self.trades if trade.pnl > 0]
        losses = [trade.pnl for trade in self.trades if trade.pnl <= 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        avg_trade_pct = float(np.mean([trade.pnl_pct for trade in self.trades]) * 100.0) if self.trades else 0.0
        hourly_returns = pd.Series(equity).pct_change().dropna()
        sharpe = 0.0
        if len(hourly_returns) > 10 and float(hourly_returns.std()) > 0:
            sharpe = float(hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365))
        timestamps = self.timestamps
        return BacktestResult(
            config=self.config,
            segment=segment,
            start_time=str(timestamps[max(self.start_bar, WARMUP_BARS)]),
            end_time=str(timestamps[self.end_bar - 1]),
            final_balance=final_balance,
            return_pct=return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=len(self.trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pct=avg_trade_pct,
            sharpe=sharpe,
            trade_log=self.trades,
        )


def candidate_configs() -> List[StrategyConfig]:
    configs: List[StrategyConfig] = []
    for donchian_bars in (120, 240, 480):
        for volume_floor in (1.0, 1.3):
            configs.append(
                StrategyConfig(
                    name=f"trend_donchian_{donchian_bars}_vol_{volume_floor}",
                    family="trend_breakout",
                    params={
                        "donchian_bars": donchian_bars,
                        "volume_floor": volume_floor,
                        "stop_atr": 2.8,
                        "trail_atr": 4.0,
                        "max_hold_bars": 24 * 21,
                    },
                )
            )

    for channel_bars in (24, 48, 72):
        for volume_floor in (1.2, 1.5):
            configs.append(
                StrategyConfig(
                    name=f"squeeze_breakout_{channel_bars}_vol_{volume_floor}",
                    family="squeeze_breakout",
                    params={
                        "channel_bars": channel_bars,
                        "volume_floor": volume_floor,
                        "squeeze_rank": 0.25,
                        "stop_atr": 2.2,
                        "trail_atr": 3.2,
                        "take_profit_atr": 5.0,
                        "max_hold_bars": 24 * 7,
                    },
                )
            )

    for top_n in (2, 3, 4):
        for rebalance_bars in (24, 72):
            configs.append(
                StrategyConfig(
                    name=f"relative_strength_top_{top_n}_rebalance_{rebalance_bars}",
                    family="relative_strength",
                    params={
                        "lookback_short": 168,
                        "lookback_long": 720,
                        "top_n": top_n,
                        "rebalance_bars": rebalance_bars,
                        "stop_pct": 0.10,
                        "trail_pct": 0.08,
                        "max_hold_bars": 24 * 30,
                    },
                    max_positions=top_n,
                )
            )

    for drop_floor in (0.08, 0.12):
        for volume_floor in (1.5, 2.0):
            configs.append(
                StrategyConfig(
                    name=f"panic_reversal_drop_{int(drop_floor * 100)}_vol_{volume_floor}",
                    family="panic_reversal",
                    params={
                        "drop_floor": drop_floor,
                        "volume_floor": volume_floor,
                        "wick_floor": 0.45,
                        "stop_pct": 0.04,
                        "take_profit_pct": 0.07,
                        "max_hold_bars": 24,
                    },
                )
            )
    return configs


def run_one(
    market_data: Dict[str, pd.DataFrame],
    config: StrategyConfig,
    start_bar: int,
    end_bar: int,
    segment: str,
) -> BacktestResult:
    simulator = PortfolioSimulator(market_data, config, start_bar, end_bar)
    return simulator.run(segment)


def result_row(result: BacktestResult) -> Dict[str, object]:
    return {
        "segment": result.segment,
        "strategy": result.config.name,
        "family": result.config.family,
        "start": result.start_time,
        "end": result.end_time,
        "final_balance": round(result.final_balance, 2),
        "return_pct": round(result.return_pct, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "trades": result.trades,
        "win_rate": round(result.win_rate, 3),
        "profit_factor": round(result.profit_factor, 3),
        "avg_trade_pct": round(result.avg_trade_pct, 3),
        "sharpe": round(result.sharpe, 3),
    }


def robustness_score(train: BacktestResult, test: BacktestResult) -> float:
    if train.trades < 12 or test.trades < 5:
        return -999.0
    score = 0.0
    score += min(train.profit_factor, 3.0) * 2.0
    score += min(test.profit_factor, 3.0) * 3.0
    score += train.return_pct * 0.05 + test.return_pct * 0.10
    score += min(train.sharpe, 3.0) + min(test.sharpe, 3.0) * 1.5
    score -= max(0.0, train.max_drawdown_pct - 12.0) * 0.25
    score -= max(0.0, test.max_drawdown_pct - 12.0) * 0.50
    return score


def write_results_csv(results: Sequence[BacktestResult], path: Path) -> None:
    rows = [result_row(result) for result in results]
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    market_data: Dict[str, pd.DataFrame],
    ranked: Sequence[Tuple[float, BacktestResult, BacktestResult, BacktestResult]],
    all_results: Sequence[BacktestResult],
    path: Path,
) -> None:
    timestamps = market_data[BTC_PAIR]["timestamp"]
    lines = [
        "# Profit Strategy Lab - 2026-04-28",
        "",
        "Scope: Kraken-compatible spot-only long strategies tested on local 1h OHLCV, with blocked names excluded.",
        "",
        f"Data range: {timestamps.iloc[0]} to {timestamps.iloc[-1]}",
        f"Pairs: {', '.join(sorted(market_data.keys()))}",
        f"Assumptions: ${STARTING_BALANCE:.0f} start, {FEE_PER_SIDE*100:.2f}% cost per side, {MAX_POSITION_PCT*100:.0f}% max position, {MAX_TOTAL_EXPOSURE*100:.0f}% max total exposure.",
        "",
        "## Best Walk-Forward Candidates",
        "",
        "| Rank | Strategy | Train Ret | Train PF | Train DD | Test Ret | Test PF | Test DD | Test Trades | Full Ret | Full PF |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (score, train, test, full) in enumerate(ranked[:12], start=1):
        lines.append(
            f"| {rank} | `{train.config.name}` | {train.return_pct:.2f}% | {train.profit_factor:.2f} | "
            f"{train.max_drawdown_pct:.2f}% | {test.return_pct:.2f}% | {test.profit_factor:.2f} | "
            f"{test.max_drawdown_pct:.2f}% | {test.trades} | {full.return_pct:.2f}% | {full.profit_factor:.2f} |"
        )

    passing = [item for item in ranked if item[2].profit_factor >= 1.15 and item[2].return_pct > 0 and item[2].max_drawdown_pct <= 12]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Walk-forward candidates passing test PF >= 1.15, positive test return, and test drawdown <= 12%: {len(passing)}.",
            "A strategy should not be wired live just because it wins in-sample. The live bot should only consume candidates that survive the test segment and remain sane on full-sample behavior.",
            "",
            "## Next Live Integration Rule",
            "",
            "If a candidate passes, wire only the signal logic first and keep the existing validation-mode caps. The live bot should reject the strategy if BTC regime, liquidity, or native stop placement cannot be confirmed.",
        ]
    )
    if passing:
        best = passing[0][2]
        lines.extend(
            [
                "",
                "Recommended first candidate:",
                "",
                f"- `{best.config.name}` ({best.config.family})",
                f"- Test return: {best.return_pct:.2f}%",
                f"- Test profit factor: {best.profit_factor:.2f}",
                f"- Test max drawdown: {best.max_drawdown_pct:.2f}%",
                f"- Test trades: {best.trades}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No candidate met the live-wiring bar yet. That is a useful result: it means the next step is improving candidate logic, not forcing another weak tool into production.",
            ]
        )

    family_summary: Dict[str, List[BacktestResult]] = {}
    for result in all_results:
        if result.segment == "test":
            family_summary.setdefault(result.config.family, []).append(result)
    lines.extend(["", "## Family Summary", "", "| Family | Tested | Median Test Ret | Median Test PF | Best Test Ret |", "|---|---:|---:|---:|---:|"])
    for family, family_results in sorted(family_summary.items()):
        returns = [result.return_pct for result in family_results]
        profit_factors = [result.profit_factor for result in family_results]
        lines.append(
            f"| {family} | {len(family_results)} | {np.median(returns):.2f}% | "
            f"{np.median(profit_factors):.2f} | {max(returns):.2f}% |"
        )

    path.write_text("\n".join(lines) + "\n")


def print_top_table(ranked: Sequence[Tuple[float, BacktestResult, BacktestResult, BacktestResult]]) -> None:
    print("\nTop walk-forward candidates")
    print("strategy, train_ret, train_pf, test_ret, test_pf, test_dd, test_trades, full_ret")
    for _, train, test, full in ranked[:10]:
        print(
            f"{train.config.name}, {train.return_pct:.2f}%, {train.profit_factor:.2f}, "
            f"{test.return_pct:.2f}%, {test.profit_factor:.2f}, {test.max_drawdown_pct:.2f}%, "
            f"{test.trades}, {full.return_pct:.2f}%"
        )


def run_lab(limit: Optional[int] = None) -> Tuple[Path, Path]:
    market_data = load_market_data()
    total_bars = len(market_data[BTC_PAIR])
    split_bar = int(total_bars * 0.62)
    configs = candidate_configs()
    if limit:
        configs = configs[:limit]

    all_results: List[BacktestResult] = []
    ranked: List[Tuple[float, BacktestResult, BacktestResult, BacktestResult]] = []
    for config_index, config in enumerate(configs, start=1):
        print(f"[{config_index}/{len(configs)}] Testing {config.name}", flush=True)
        train = run_one(market_data, config, WARMUP_BARS, split_bar, "train")
        test = run_one(market_data, config, split_bar, total_bars, "test")
        full = run_one(market_data, config, WARMUP_BARS, total_bars, "full")
        all_results.extend([train, test, full])
        ranked.append((robustness_score(train, test), train, test, full))

    ranked.sort(key=lambda item: item[0], reverse=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    csv_path = RESEARCH_DIR / f"profit_strategy_lab_results_{today}.csv"
    report_path = RESEARCH_DIR / f"profit_strategy_lab_{today}.md"
    write_results_csv(all_results, csv_path)
    write_report(market_data, ranked, all_results, report_path)
    print_top_table(ranked)
    print(f"\nResults CSV: {csv_path}")
    print(f"Report: {report_path}")
    return csv_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kraken spot strategy walk-forward lab")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N configs for a quick smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_lab(limit=args.limit)