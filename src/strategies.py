"""
Advanced trading strategies with multi-indicator confirmation,
volatility-adaptive levels, market regime detection, and volume analysis.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger
from abc import ABC, abstractmethod
import inspect


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Trading signal"""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    regime: str = "unknown"  # "trending_up", "trending_down", "ranging", "high_volatility"
    atr: float = 0.0  # Current ATR value for the signal
    # Adaptive learning fields (populated by AdaptiveQuantStrategy)
    features_active: Dict[str, bool] = field(default_factory=dict)
    indicator_snapshot: Dict[str, float] = field(default_factory=dict)
    htf_trend: str = "unknown"
    btc_trend: str = "unknown"


# ---------------------------------------------------------------------------
# Technical indicator helpers (pure functions on pd.Series / pd.DataFrame)
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-smoothed)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calc_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle band (SMA), lower band."""
    mid = calc_sma(series, period)
    std = series.rolling(window=period).std()
    return mid + num_std * std, mid, mid - num_std * std


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index - measures trend strength (0-100)."""
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calc_atr(df, period)

    plus_di = 100 * calc_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * calc_ema(minus_dm, period) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = calc_ema(dx, period)
    return adx


def calc_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple Moving Average of volume."""
    vol = pd.to_numeric(df["volume"], errors="coerce")
    return vol.rolling(window=period).mean()


def calc_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling Volume Weighted Average Price.

    Uses a rolling window (default 20 bars) instead of cumulative VWAP,
    which is more meaningful for intraday crypto trading where there are
    no session boundaries.
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * vol
    rolling_tp_vol = tp_vol.rolling(window=period, min_periods=1).sum()
    rolling_vol = vol.rolling(window=period, min_periods=1).sum().replace(0, np.nan)
    return rolling_tp_vol / rolling_vol


def calc_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — a leading volume-based momentum indicator."""
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    direction = np.sign(close.diff()).fillna(0)
    return (vol * direction).cumsum()


def calc_bb_width(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger Band width — narrow = squeeze (pending breakout)."""
    close = pd.to_numeric(df["close"], errors="coerce")
    upper, mid, lower = calc_bollinger_bands(close, period, num_std)
    return ((upper - lower) / mid.replace(0, np.nan))


def calc_di(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """+DI and -DI (Directional Indicators).

    Returns (plus_di, minus_di) — both normalised 0-100.
    +DI > -DI means bulls dominate, vice-versa.
    """
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calc_atr(df, period)
    plus_di = 100 * calc_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * calc_ema(minus_dm, period) / atr.replace(0, np.nan)
    return plus_di, minus_di


def detect_rsi_divergence(
    close: pd.Series, rsi: pd.Series, lookback: int = 14
) -> str:
    """Detect bullish or bearish RSI divergence over the last *lookback* bars.

    Bullish divergence: price makes a lower low but RSI makes a higher low.
    Bearish divergence: price makes a higher high but RSI makes a lower high.

    Returns: 'bullish', 'bearish', or 'none'.
    """
    if len(close) < lookback + 2 or len(rsi) < lookback + 2:
        return "none"
    try:
        recent_close = close.iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]
        prev_close = close.iloc[-lookback * 2 : -lookback]
        prev_rsi = rsi.iloc[-lookback * 2 : -lookback]

        if len(prev_close) < 2 or len(prev_rsi) < 2:
            return "none"

        # Bullish: lower price low, higher RSI low
        if (
            float(recent_close.min()) < float(prev_close.min())
            and float(recent_rsi.min()) > float(prev_rsi.min())
        ):
            return "bullish"

        # Bearish: higher price high, lower RSI high
        if (
            float(recent_close.max()) > float(prev_close.max())
            and float(recent_rsi.max()) < float(prev_rsi.max())
        ):
            return "bearish"
    except Exception:
        pass
    return "none"


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------

def detect_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    ema_fast: int = 20,
    ema_slow: int = 50,
    atr_period: int = 14,
    volatility_lookback: int = 50,
) -> str:
    """
    Classify the current market regime:
      - "trending_up"     : ADX > 25 and fast EMA > slow EMA
      - "trending_down"   : ADX > 25 and fast EMA < slow EMA
      - "high_volatility" : ATR > 1.5x its own rolling median OR
                            price dropped > 5% in the last 10 bars
      - "ranging"         : everything else
    """
    if len(df) < max(adx_period * 2, ema_slow + 5, volatility_lookback):
        return "unknown"

    try:
        close = pd.to_numeric(df["close"], errors="coerce")
        if close.isna().sum() > len(close) * 0.3:
            return "unknown"

        adx = calc_adx(df, adx_period)
        atr = calc_atr(df, atr_period)
        ema_f = calc_ema(close, ema_fast)
        ema_s = calc_ema(close, ema_slow)

        current_adx = float(adx.iloc[-1])
        current_atr = float(atr.iloc[-1])
        median_atr = float(atr.tail(volatility_lookback).median())

        # Check for any NaN in key values
        if np.isnan(current_adx) or np.isnan(current_atr) or np.isnan(median_atr):
            return "unknown"

        # High-volatility: ATR spike
        if median_atr > 0 and current_atr > 1.5 * median_atr:
            return "high_volatility"

        # High-volatility: rapid price move (crash/melt-up detection)
        if len(close) >= 10:
            roc_10 = (float(close.iloc[-1]) - float(close.iloc[-10])) / float(close.iloc[-10])
            if abs(roc_10) > 0.05:  # >5% move in 10 bars
                return "high_volatility"

        # High-volatility: sustained drawdown (30-bar lookback catches prolonged crashes)
        if len(close) >= 30:
            roc_30 = (float(close.iloc[-1]) - float(close.iloc[-30])) / float(close.iloc[-30])
            if abs(roc_30) > 0.10:  # >10% move in 30 bars
                return "high_volatility"

        if current_adx > 25:
            if float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]):
                return "trending_up"
            else:
                return "trending_down"

        return "ranging"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Abstract strategy base
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Base strategy class"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate trading signals from OHLCV data."""
        pass


# ---------------------------------------------------------------------------
# Multi-Indicator Trend-Momentum Strategy (replaces old Grid as default)
# ---------------------------------------------------------------------------

class TrendMomentumStrategy(Strategy):
    """
    Multi-indicator trend-momentum strategy with high-conviction entries.

    Indicators used:
      1. EMA crossover (20/50) -- trend direction + fresh crossover detection
      2. MACD histogram -- momentum confirmation + decay detection
      3. RSI -- overbought/oversold guard-rails + divergence detection
      4. ADX + DI -- trend strength and directional dominance
      5. ATR -- dynamic stop-loss and take-profit levels
      6. Volume SMA -- above-average volume confirmation
      7. VWAP -- fair-value entry filter
      8. OBV -- money flow confirmation
      9. Bollinger Band width -- squeeze/breakout anticipation

    Built-in signal cooldown prevents repeated identical signals.
    Only takes LONG entries in uptrends and SHORT signals in downtrends.
    Stays flat in "high_volatility" regimes.
    """

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        adx_threshold: float = 20.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        volume_mult: float = 1.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cooldown_bars: int = 3,
        min_confidence: float = 0.45,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.adx_threshold = adx_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_mult = volume_mult
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cooldown_bars = cooldown_bars
        self.min_confidence = min_confidence
        # Signal cooldown state: {symbol: {"action": str, "bars_since": int}}
        self._last_signal: Dict[str, Dict] = {}

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        min_rows = max(self.ema_slow, 26, self.atr_period) * 2 + 10
        if len(data) < min_rows:
            return []

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().sum() > len(close) * 0.3:
            return []  # too many missing values

        # -- Core indicators --
        ema_f = calc_ema(close, self.ema_fast)
        ema_s = calc_ema(close, self.ema_slow)
        rsi = calc_rsi(close, self.rsi_period)
        atr = calc_atr(data, self.atr_period)
        adx = calc_adx(data, self.atr_period)
        macd_line, signal_line, macd_hist = calc_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        vol_sma = calc_volume_sma(data, 20)

        # -- New indicators --
        vwap = calc_vwap(data)
        obv = calc_obv(data)
        obv_sma = calc_sma(obv, 20)
        bb_width = calc_bb_width(data, 20, 2.0)
        plus_di, minus_di = calc_di(data, self.atr_period)
        divergence = detect_rsi_divergence(close, rsi, lookback=self.rsi_period)

        regime = detect_regime(data)

        # -- Snapshot current bar values --
        def _safe(s, idx=-1, default=0.0):
            try:
                v = float(s.iloc[idx])
                return default if np.isnan(v) else v
            except Exception:
                return default

        cur = {
            "price": _safe(close),
            "ema_f": _safe(ema_f),
            "ema_s": _safe(ema_s),
            "prev_ema_f": _safe(ema_f, -2),
            "prev_ema_s": _safe(ema_s, -2),
            "rsi": _safe(rsi),
            "atr": _safe(atr),
            "adx": _safe(adx),
            "plus_di": _safe(plus_di),
            "minus_di": _safe(minus_di),
            "macd": _safe(macd_line),
            "macd_signal": _safe(signal_line),
            "macd_hist": _safe(macd_hist),
            "macd_hist_prev": _safe(macd_hist, -2),
            "macd_hist_prev2": _safe(macd_hist, -3),
            "volume": _safe(pd.to_numeric(data["volume"], errors="coerce")),
            "vol_sma": _safe(vol_sma),
            "vwap": _safe(vwap),
            "obv": _safe(obv),
            "obv_sma": _safe(obv_sma),
            "bb_width": _safe(bb_width),
            "bb_width_prev": _safe(bb_width, -2),
        }

        if cur["atr"] <= 0 or cur["price"] <= 0:
            return []

        signals: List[Signal] = []

        # -- Detect crossover *events* (just happened) vs alignment (already there) --
        ema_bull_cross = cur["prev_ema_f"] <= cur["prev_ema_s"] and cur["ema_f"] > cur["ema_s"]
        ema_bear_cross = cur["prev_ema_f"] >= cur["prev_ema_s"] and cur["ema_f"] < cur["ema_s"]
        ema_bull_aligned = cur["ema_f"] > cur["ema_s"]
        ema_bear_aligned = cur["ema_f"] < cur["ema_s"]

        # -- Bollinger squeeze detection (width contracting -> breakout pending) --
        bb_squeeze = cur["bb_width"] < cur["bb_width_prev"] and cur["bb_width"] > 0

        # -- MACD momentum decay (histogram shrinking toward zero) --
        macd_bull_fading = (
            cur["macd_hist"] > 0
            and cur["macd_hist"] < cur["macd_hist_prev"]
            and cur["macd_hist_prev"] < cur["macd_hist_prev2"]
        )
        macd_bear_fading = (
            cur["macd_hist"] < 0
            and cur["macd_hist"] > cur["macd_hist_prev"]
            and cur["macd_hist_prev"] > cur["macd_hist_prev2"]
        )

        # -------------------------------------------------------------------
        # BUY conditions
        # -------------------------------------------------------------------
        buy_reasons: List[str] = []
        buy_confidence = 0.0

        # 1) EMA crossover event (fresh cross = stronger) vs alignment (weaker)
        if ema_bull_cross:
            buy_confidence += 0.20
            buy_reasons.append("EMA bullish crossover (fresh)")
        elif ema_bull_aligned:
            buy_confidence += 0.10
            buy_reasons.append("EMA bullish alignment")

        # 2) MACD bullish (histogram positive & increasing = full credit;
        #    line > signal but histogram shrinking = partial credit)
        if cur["macd"] > cur["macd_signal"] and cur["macd_hist"] > cur["macd_hist_prev"]:
            buy_confidence += 0.15
            buy_reasons.append(f"MACD bullish (hist={cur['macd_hist']:.4f})")
        elif cur["macd"] > cur["macd_signal"]:
            buy_confidence += 0.05  # MACD still bullish, but momentum fading
            buy_reasons.append(f"MACD bullish weak (hist fading)")
        elif macd_bull_fading:
            buy_confidence -= 0.05  # momentum is weakening
            buy_reasons.append("MACD momentum fading")

        # 3) RSI in healthy range
        if cur["rsi"] < self.rsi_overbought:
            buy_confidence += 0.05
            if cur["rsi"] < 40:
                buy_confidence += 0.10
                buy_reasons.append(f"RSI favorable ({cur['rsi']:.1f})")
            else:
                buy_reasons.append(f"RSI OK ({cur['rsi']:.1f})")
        else:
            buy_confidence -= 0.10  # overbought = danger for longs
            buy_reasons.append(f"RSI overbought ({cur['rsi']:.1f})")

        # 4) +DI > -DI confirms bulls dominating
        if cur["plus_di"] > cur["minus_di"]:
            buy_confidence += 0.10
            buy_reasons.append(f"+DI > -DI ({cur['plus_di']:.1f}/{cur['minus_di']:.1f})")

        # 5) ADX confirms trend strength
        if cur["adx"] > self.adx_threshold:
            buy_confidence += 0.10
            buy_reasons.append(f"ADX strong ({cur['adx']:.1f})")

        # 6) Volume above average
        if cur["vol_sma"] > 0 and cur["volume"] >= cur["vol_sma"] * self.volume_mult:
            buy_confidence += 0.05
            buy_reasons.append("Volume confirmed")

        # 7) OBV trending up (money flowing in)
        if cur["obv"] > cur["obv_sma"] and cur["obv_sma"] != 0:
            buy_confidence += 0.05
            buy_reasons.append("OBV rising")

        # 8) Price above VWAP (buying at/below fair value is better)
        if cur["vwap"] > 0:
            if cur["price"] <= cur["vwap"] * 1.002:  # at or slightly below VWAP
                buy_confidence += 0.05
                buy_reasons.append("Price near/below VWAP")
            elif cur["price"] > cur["vwap"] * 1.01:
                buy_confidence -= 0.05  # overpaying relative to VWAP
                buy_reasons.append("Price above VWAP")

        # 9) RSI bullish divergence (powerful reversal signal)
        if divergence == "bullish":
            buy_confidence += 0.15
            buy_reasons.append("RSI bullish divergence")

        # 10) Bollinger squeeze (breakout imminent, boost if trend aligns)
        if bb_squeeze and ema_bull_aligned:
            buy_confidence += 0.05
            buy_reasons.append("BB squeeze + bullish alignment")

        # -- Regime penalty/bonus --
        if regime == "high_volatility":
            buy_confidence *= 0.35
            buy_reasons.append("HIGH VOLATILITY penalty")
        elif regime == "trending_down":
            buy_confidence *= 0.45
            buy_reasons.append("Downtrend discount")
        elif regime == "trending_up":
            buy_confidence *= 1.10
            buy_reasons.append("Uptrend bonus")

        # Emit BUY if enough confirmations
        if buy_confidence >= self.min_confidence:
            sl = cur["price"] - self.atr_sl_mult * cur["atr"]
            tp = cur["price"] + self.atr_tp_mult * cur["atr"]
            risk = cur["price"] - sl
            reward = tp - cur["price"]
            if risk > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=min(buy_confidence, 1.0),
                    entry_price=cur["price"],
                    stop_loss=sl,
                    take_profit=tp,
                    reason=" | ".join(buy_reasons),
                    regime=regime,
                    atr=cur["atr"],
                ))

        # -------------------------------------------------------------------
        # SELL conditions (mirror logic)
        # -------------------------------------------------------------------
        sell_reasons: List[str] = []
        sell_confidence = 0.0

        if ema_bear_cross:
            sell_confidence += 0.20
            sell_reasons.append("EMA bearish crossover (fresh)")
        elif ema_bear_aligned:
            sell_confidence += 0.10
            sell_reasons.append("EMA bearish alignment")

        if cur["macd"] < cur["macd_signal"] and cur["macd_hist"] < cur["macd_hist_prev"]:
            sell_confidence += 0.15
            sell_reasons.append(f"MACD bearish (hist={cur['macd_hist']:.4f})")
        elif cur["macd"] < cur["macd_signal"]:
            sell_confidence += 0.05
            sell_reasons.append("MACD bearish weak (hist fading)")
        elif macd_bear_fading:
            sell_confidence -= 0.05
            sell_reasons.append("MACD bear momentum fading")

        if cur["rsi"] > self.rsi_oversold:
            sell_confidence += 0.05
            if cur["rsi"] > 60:
                sell_confidence += 0.10
                sell_reasons.append(f"RSI favorable for sell ({cur['rsi']:.1f})")
            else:
                sell_reasons.append(f"RSI OK ({cur['rsi']:.1f})")
        else:
            sell_confidence -= 0.10
            sell_reasons.append(f"RSI oversold ({cur['rsi']:.1f})")

        if cur["minus_di"] > cur["plus_di"]:
            sell_confidence += 0.10
            sell_reasons.append(f"-DI > +DI ({cur['minus_di']:.1f}/{cur['plus_di']:.1f})")

        if cur["adx"] > self.adx_threshold:
            sell_confidence += 0.10
            sell_reasons.append(f"ADX strong ({cur['adx']:.1f})")

        if cur["vol_sma"] > 0 and cur["volume"] >= cur["vol_sma"] * self.volume_mult:
            sell_confidence += 0.05
            sell_reasons.append("Volume confirmed")

        if cur["obv"] < cur["obv_sma"] and cur["obv_sma"] != 0:
            sell_confidence += 0.05
            sell_reasons.append("OBV falling")

        if cur["vwap"] > 0:
            if cur["price"] >= cur["vwap"] * 0.998:
                sell_confidence += 0.05
                sell_reasons.append("Price near/above VWAP")
            elif cur["price"] < cur["vwap"] * 0.99:
                sell_confidence -= 0.05
                sell_reasons.append("Price below VWAP")

        if divergence == "bearish":
            sell_confidence += 0.15
            sell_reasons.append("RSI bearish divergence")

        if bb_squeeze and ema_bear_aligned:
            sell_confidence += 0.05
            sell_reasons.append("BB squeeze + bearish alignment")

        if regime == "high_volatility":
            sell_confidence *= 0.40
            sell_reasons.append("HIGH VOLATILITY penalty")
        elif regime == "trending_up":
            sell_confidence *= 0.45
            sell_reasons.append("Uptrend discount")
        elif regime == "trending_down":
            sell_confidence *= 1.10
            sell_reasons.append("Downtrend bonus")

        if sell_confidence >= self.min_confidence:
            sl = cur["price"] + self.atr_sl_mult * cur["atr"]
            tp = cur["price"] - self.atr_tp_mult * cur["atr"]
            risk = sl - cur["price"]
            reward = cur["price"] - tp
            if risk > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=min(sell_confidence, 1.0),
                    entry_price=cur["price"],
                    stop_loss=sl,
                    take_profit=tp,
                    reason=" | ".join(sell_reasons),
                    regime=regime,
                    atr=cur["atr"],
                ))

        # -- Signal cooldown: suppress repeated identical signals --
        filtered_signals: List[Signal] = []
        for sig in signals:
            last_bars = self._last_signal.get(sig.symbol, {}).get(sig.action)
            if last_bars is not None and last_bars < self.cooldown_bars:
                continue  # same signal type too recently
            filtered_signals.append(sig)

        # Update cooldown tracking per symbol per action type
        if symbol not in self._last_signal:
            self._last_signal[symbol] = {}
        for sig in filtered_signals:
            self._last_signal[symbol][sig.action] = 0
        # Increment bars_since for all tracked actions that were NOT just emitted
        emitted_actions = {s.action for s in filtered_signals}
        for action in list(self._last_signal.get(symbol, {})):
            if action not in emitted_actions:
                self._last_signal[symbol][action] += 1

        return filtered_signals


# ---------------------------------------------------------------------------
# Enhanced Mean Reversion Strategy
# ---------------------------------------------------------------------------

class MeanReversionStrategy(Strategy):
    """
    Mean Reversion with multi-indicator confirmation:
      - RSI + Bollinger Bands for oversold/overbought
      - MACD histogram reversal for timing
      - Volume spike confirmation
      - ATR-based dynamic stop-loss / take-profit
      - ADX filter: only trade when ADX < 30 (ranging market)
      - Regime guard: skip entries during high_volatility / strong trends
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.5,
        adx_max: float = 30.0,
        volume_spike_mult: float = 1.2,
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.adx_max = adx_max
        self.volume_spike_mult = volume_spike_mult

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        min_rows = max(self.rsi_period, self.bb_period, 26, self.atr_period) + 10
        if len(data) < min_rows:
            return []

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().all():
            return []

        rsi = calc_rsi(close, self.rsi_period)
        upper_bb, mid_bb, lower_bb = calc_bollinger_bands(close, self.bb_period, self.bb_std)
        atr = calc_atr(data, self.atr_period)
        adx = calc_adx(data, self.atr_period)
        macd_line, signal_line, macd_hist = calc_macd(close)
        vol_sma = calc_volume_sma(data, 20)
        obv = calc_obv(data)
        obv_sma = calc_sma(obv, 20)
        divergence = detect_rsi_divergence(close, rsi, lookback=self.rsi_period)
        regime = detect_regime(data)

        cur_price = float(close.iloc[-1])
        cur_rsi = float(rsi.iloc[-1])
        cur_atr = float(atr.iloc[-1])
        cur_adx = float(adx.iloc[-1])
        cur_hist = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2])
        cur_vol = float(data["volume"].iloc[-1])
        cur_vol_sma = float(vol_sma.iloc[-1]) if not np.isnan(vol_sma.iloc[-1]) else 0
        cur_lower = float(lower_bb.iloc[-1])
        cur_upper = float(upper_bb.iloc[-1])
        cur_mid = float(mid_bb.iloc[-1])
        cur_obv = float(obv.iloc[-1])
        cur_obv_sma = float(obv_sma.iloc[-1]) if not np.isnan(obv_sma.iloc[-1]) else 0

        signals: List[Signal] = []

        # --- Guard: skip in strong trends or high volatility ---
        if regime == "high_volatility":
            return []
        if cur_adx > self.adx_max:
            return []  # mean reversion unreliable in strong trends

        # --- BUY: oversold bounce ---
        if cur_rsi < self.rsi_oversold and cur_price <= cur_lower:
            confidence = 0.50
            reasons = [f"RSI oversold ({cur_rsi:.1f})", "Price at/below lower BB"]

            # MACD histogram turning up (momentum shifting)
            if cur_hist > prev_hist:
                confidence += 0.15
                reasons.append("MACD hist turning up")

            # Volume spike
            if cur_vol_sma > 0 and cur_vol >= cur_vol_sma * self.volume_spike_mult:
                confidence += 0.10
                reasons.append("Volume spike")

            # RSI extremely oversold
            if cur_rsi < 25:
                confidence += 0.10
                reasons.append("Deeply oversold")

            # RSI bullish divergence (price lower low, RSI higher low)
            if divergence == "bullish":
                confidence += 0.15
                reasons.append("RSI bullish divergence")

            # OBV rising = smart money accumulating despite price drop
            if cur_obv > cur_obv_sma and cur_obv_sma != 0:
                confidence += 0.05
                reasons.append("OBV accumulation")

            if confidence >= 0.55 and cur_atr > 0:
                sl = cur_price - self.atr_sl_mult * cur_atr
                tp_target = cur_mid  # revert to middle band
                tp = max(tp_target, cur_price + self.atr_tp_mult * cur_atr)
                risk = cur_price - sl
                reward = tp - cur_price
                if risk > 0 and (reward / risk) >= 1.5:
                    signals.append(Signal(
                        symbol=symbol,
                        action="BUY",
                        confidence=min(confidence, 1.0),
                        entry_price=cur_price,
                        stop_loss=sl,
                        take_profit=tp,
                        reason=" | ".join(reasons),
                        regime=regime,
                        atr=cur_atr,
                    ))

        # --- SELL: overbought rejection ---
        elif cur_rsi > self.rsi_overbought and cur_price >= cur_upper:
            confidence = 0.50
            reasons = [f"RSI overbought ({cur_rsi:.1f})", "Price at/above upper BB"]

            if cur_hist < prev_hist:
                confidence += 0.15
                reasons.append("MACD hist turning down")

            if cur_vol_sma > 0 and cur_vol >= cur_vol_sma * self.volume_spike_mult:
                confidence += 0.10
                reasons.append("Volume spike")

            if cur_rsi > 75:
                confidence += 0.10
                reasons.append("Deeply overbought")

            # RSI bearish divergence (price higher high, RSI lower high)
            if divergence == "bearish":
                confidence += 0.15
                reasons.append("RSI bearish divergence")

            # OBV falling = distribution despite price rise
            if cur_obv < cur_obv_sma and cur_obv_sma != 0:
                confidence += 0.05
                reasons.append("OBV distribution")

            if confidence >= 0.55 and cur_atr > 0:
                sl = cur_price + self.atr_sl_mult * cur_atr
                tp_target = cur_mid
                tp = min(tp_target, cur_price - self.atr_tp_mult * cur_atr)
                risk = sl - cur_price
                reward = cur_price - tp
                if risk > 0 and (reward / risk) >= 1.5:
                    signals.append(Signal(
                        symbol=symbol,
                        action="SELL",
                        confidence=min(confidence, 1.0),
                        entry_price=cur_price,
                        stop_loss=sl,
                        take_profit=tp,
                        reason=" | ".join(reasons),
                        regime=regime,
                        atr=cur_atr,
                    ))

        return signals


# ---------------------------------------------------------------------------
# Enhanced Statistical Arbitrage Strategy
# ---------------------------------------------------------------------------

class StatisticalArbitrageStrategy(Strategy):
    """
    Statistical Arbitrage with improvements:
      - Z-score + RSI confirmation
      - ATR-based stops instead of fixed percent
      - Regime filter (skip high_volatility / strong trends)
    """

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        lookback_period: int = 50,
        atr_period: int = 14,
        atr_sl_mult: float = 2.0,
        rsi_period: int = 14,
    ):
        self.z_score_threshold = z_score_threshold
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.rsi_period = rsi_period

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        if len(data) < max(self.lookback_period, self.atr_period) + 5:
            return []

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().all():
            return []

        regime = detect_regime(data)
        if regime in ("high_volatility", "trending_up", "trending_down"):
            return []  # stat-arb works best in ranging markets

        mean = float(close.tail(self.lookback_period).mean())
        std = float(close.tail(self.lookback_period).std())
        if std == 0:
            return []

        cur_price = float(close.iloc[-1])
        z_score = (cur_price - mean) / std

        atr = calc_atr(data, self.atr_period)
        cur_atr = float(atr.iloc[-1])
        rsi = calc_rsi(close, self.rsi_period)
        cur_rsi = float(rsi.iloc[-1])

        signals: List[Signal] = []

        # BUY: significantly below mean + RSI confirms oversold
        if z_score < -self.z_score_threshold and cur_rsi < 40:
            confidence = min(0.5 + abs(z_score) * 0.1, 0.85)
            sl = cur_price - self.atr_sl_mult * cur_atr
            tp = mean  # revert to mean
            risk = cur_price - sl
            reward = tp - cur_price
            if risk > 0 and reward > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=confidence,
                    entry_price=cur_price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=f"Stat undervalue z={z_score:.2f}, RSI={cur_rsi:.1f}",
                    regime=regime,
                    atr=cur_atr,
                ))

        # SELL: significantly above mean + RSI confirms overbought
        elif z_score > self.z_score_threshold and cur_rsi > 60:
            confidence = min(0.5 + abs(z_score) * 0.1, 0.85)
            sl = cur_price + self.atr_sl_mult * cur_atr
            tp = mean
            risk = sl - cur_price
            reward = cur_price - tp
            if risk > 0 and reward > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=confidence,
                    entry_price=cur_price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=f"Stat overvalue z={z_score:.2f}, RSI={cur_rsi:.1f}",
                    regime=regime,
                    atr=cur_atr,
                ))

        return signals


# ---------------------------------------------------------------------------
# Scalp-Momentum Strategy (designed for small accounts $100-$1000)
# ---------------------------------------------------------------------------

class ScalpMomentumStrategy(Strategy):
    """
    Fast scalp-momentum strategy optimised for small accounts.

    Key differences from TrendMomentumStrategy:
      - Faster indicators: EMA 9/21, MACD 8/17/9, RSI 7
      - Lower confidence threshold → more trades than trend_momentum
      - Wide TP (3.0× ATR) to ensure per-trade edge > fee drag
      - SL at 1.5× ATR — not too tight (avoids noise stop-outs)
      - Works in trending AND ranging markets (only skips high_volatility)
      - 2-bar cooldown to avoid overtrading
      - Fewer required confirmations = higher signal frequency
      - Still maintains positive expectancy via:
        * R:R ≥ 1.5 minimum (most trades ~1:2)
        * Volume + OBV confirmation bonuses
        * Regime-aware penalties

    Target: Per-trade edge >1% to clear 0.52% round-trip Kraken fees.
    Expected: ~1 trade/day across 5 pairs.
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 7,
        atr_period: int = 10,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        adx_threshold: float = 15.0,
        rsi_overbought: float = 75.0,
        rsi_oversold: float = 25.0,
        macd_fast: int = 8,
        macd_slow: int = 17,
        macd_signal: int = 9,
        cooldown_bars: int = 2,
        min_confidence: float = 0.38,
        min_rr: float = 1.5,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.adx_threshold = adx_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cooldown_bars = cooldown_bars
        self.min_confidence = min_confidence
        self.min_rr = min_rr
        self._last_signal: Dict[str, Dict] = {}

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        min_rows = max(self.ema_slow, self.macd_slow, self.atr_period) * 2 + 5
        if len(data) < min_rows:
            return []

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().sum() > len(close) * 0.3:
            return []

        # --- Indicators (fast settings) ---
        ema_f = calc_ema(close, self.ema_fast)
        ema_s = calc_ema(close, self.ema_slow)
        rsi = calc_rsi(close, self.rsi_period)
        atr = calc_atr(data, self.atr_period)
        macd_line, signal_line, macd_hist = calc_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        vol_sma = calc_volume_sma(data, 10)
        obv = calc_obv(data)
        obv_sma = calc_sma(obv, 10)

        regime = detect_regime(data)

        def _s(s, idx=-1, default=0.0):
            try:
                v = float(s.iloc[idx])
                return default if np.isnan(v) else v
            except Exception:
                return default

        c = {
            "price": _s(close),
            "ema_f": _s(ema_f), "ema_s": _s(ema_s),
            "prev_ema_f": _s(ema_f, -2), "prev_ema_s": _s(ema_s, -2),
            "rsi": _s(rsi),
            "atr": _s(atr),
            "macd": _s(macd_line), "macd_sig": _s(signal_line),
            "hist": _s(macd_hist), "hist_prev": _s(macd_hist, -2),
            "vol": _s(pd.to_numeric(data["volume"], errors="coerce")),
            "vol_sma": _s(vol_sma),
            "obv": _s(obv), "obv_sma": _s(obv_sma),
        }

        if c["atr"] <= 0 or c["price"] <= 0:
            return []

        # Skip only high-volatility (crashes) — trade everything else
        if regime == "high_volatility":
            return []

        signals: List[Signal] = []

        # --- EMA events ---
        bull_cross = c["prev_ema_f"] <= c["prev_ema_s"] and c["ema_f"] > c["ema_s"]
        bear_cross = c["prev_ema_f"] >= c["prev_ema_s"] and c["ema_f"] < c["ema_s"]
        bull_aligned = c["ema_f"] > c["ema_s"]
        bear_aligned = c["ema_f"] < c["ema_s"]

        # ===================== BUY =====================
        buy_conf = 0.0
        buy_reasons = []

        if bull_cross:
            buy_conf += 0.25
            buy_reasons.append("EMA9/21 bull cross")
        elif bull_aligned:
            buy_conf += 0.10
            buy_reasons.append("EMA bullish")

        # MACD above signal
        if c["macd"] > c["macd_sig"]:
            buy_conf += 0.15
            buy_reasons.append("MACD bullish")
            if c["hist"] > c["hist_prev"]:
                buy_conf += 0.05
                buy_reasons.append("Hist rising")

        # RSI not overbought
        if c["rsi"] < self.rsi_overbought:
            buy_conf += 0.05
            if c["rsi"] < 40:
                buy_conf += 0.10
                buy_reasons.append(f"RSI low ({c['rsi']:.0f})")
        else:
            buy_conf -= 0.15
            buy_reasons.append("RSI overbought")

        # Volume above average (bonus, not required)
        if c["vol_sma"] > 0 and c["vol"] > c["vol_sma"]:
            buy_conf += 0.05
            buy_reasons.append("Vol+")

        # OBV rising
        if c["obv"] > c["obv_sma"] and c["obv_sma"] != 0:
            buy_conf += 0.05
            buy_reasons.append("OBV+")

        # Regime bonus
        if regime == "trending_up":
            buy_conf *= 1.15
        elif regime == "trending_down":
            buy_conf *= 0.6

        if buy_conf >= self.min_confidence:
            sl = c["price"] - self.atr_sl_mult * c["atr"]
            tp = c["price"] + self.atr_tp_mult * c["atr"]
            risk = c["price"] - sl
            reward = tp - c["price"]
            if risk > 0 and (reward / risk) >= self.min_rr:
                signals.append(Signal(
                    symbol=symbol, action="BUY",
                    confidence=min(buy_conf, 1.0),
                    entry_price=c["price"], stop_loss=sl, take_profit=tp,
                    reason=" | ".join(buy_reasons), regime=regime, atr=c["atr"],
                ))

        # ===================== SELL =====================
        sell_conf = 0.0
        sell_reasons = []

        if bear_cross:
            sell_conf += 0.25
            sell_reasons.append("EMA9/21 bear cross")
        elif bear_aligned:
            sell_conf += 0.10
            sell_reasons.append("EMA bearish")

        if c["macd"] < c["macd_sig"]:
            sell_conf += 0.15
            sell_reasons.append("MACD bearish")
            if c["hist"] < c["hist_prev"]:
                sell_conf += 0.05
                sell_reasons.append("Hist falling")

        if c["rsi"] > self.rsi_oversold:
            sell_conf += 0.05
            if c["rsi"] > 60:
                sell_conf += 0.10
                sell_reasons.append(f"RSI high ({c['rsi']:.0f})")
        else:
            sell_conf -= 0.15
            sell_reasons.append("RSI oversold")

        if c["vol_sma"] > 0 and c["vol"] > c["vol_sma"]:
            sell_conf += 0.05
            sell_reasons.append("Vol+")

        if c["obv"] < c["obv_sma"] and c["obv_sma"] != 0:
            sell_conf += 0.05
            sell_reasons.append("OBV-")

        if regime == "trending_down":
            sell_conf *= 1.15
        elif regime == "trending_up":
            sell_conf *= 0.6

        if sell_conf >= self.min_confidence:
            sl = c["price"] + self.atr_sl_mult * c["atr"]
            tp = c["price"] - self.atr_tp_mult * c["atr"]
            risk = sl - c["price"]
            reward = c["price"] - tp
            if risk > 0 and (reward / risk) >= self.min_rr:
                signals.append(Signal(
                    symbol=symbol, action="SELL",
                    confidence=min(sell_conf, 1.0),
                    entry_price=c["price"], stop_loss=sl, take_profit=tp,
                    reason=" | ".join(sell_reasons), regime=regime, atr=c["atr"],
                ))

        # Cooldown filter
        filtered: List[Signal] = []
        for sig in signals:
            last = self._last_signal.get(sig.symbol, {}).get(sig.action)
            if last is not None and last < self.cooldown_bars:
                continue
            filtered.append(sig)

        if symbol not in self._last_signal:
            self._last_signal[symbol] = {}
        emitted = {s.action for s in filtered}
        for sig in filtered:
            self._last_signal[symbol][sig.action] = 0
        for action in list(self._last_signal.get(symbol, {})):
            if action not in emitted:
                self._last_signal[symbol][action] += 1

        return filtered


# ---------------------------------------------------------------------------
# Legacy Grid adapter (kept for backwards compatibility but adds guards)
# ---------------------------------------------------------------------------

class GridTradingStrategy(Strategy):
    """
    Grid Trading Strategy with regime awareness.
    Only active in *ranging* markets (ADX < 25). Skips trending/high-vol.
    Uses ATR for grid spacing instead of fixed percent.
    """

    def __init__(
        self,
        grid_levels: int = 10,
        range_percent: float = 0.05,
        stop_loss_percent: float = 0.02,
        take_profit_percent: float = 0.05,
    ):
        self.grid_levels = grid_levels
        self.range_percent = range_percent
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        if len(data) < 30:
            return []

        regime = detect_regime(data)
        if regime != "ranging":
            return []  # Grid only works in ranging markets

        close = pd.to_numeric(data["close"], errors="coerce")
        cur_price = float(close.iloc[-1])
        atr = calc_atr(data, 14)
        cur_atr = float(atr.iloc[-1])
        rsi = calc_rsi(close, 14)
        cur_rsi = float(rsi.iloc[-1])

        signals: List[Signal] = []

        # Only generate a single buy or sell based on where price sits in range
        upper_bound = cur_price * (1 + self.range_percent)
        lower_bound = cur_price * (1 - self.range_percent)
        mid = (upper_bound + lower_bound) / 2

        # Buy near the bottom of range
        if cur_price < mid and cur_rsi < 45 and cur_atr > 0:
            sl = cur_price - 2 * cur_atr
            tp = cur_price + 3 * cur_atr
            risk = cur_price - sl
            reward = tp - cur_price
            if risk > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=0.60,
                    entry_price=cur_price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=f"Grid buy: ranging market, RSI={cur_rsi:.1f}",
                    regime=regime,
                    atr=cur_atr,
                ))

        # Sell near the top of range
        elif cur_price > mid and cur_rsi > 55 and cur_atr > 0:
            sl = cur_price + 2 * cur_atr
            tp = cur_price - 3 * cur_atr
            risk = sl - cur_price
            reward = cur_price - tp
            if risk > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=0.60,
                    entry_price=cur_price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=f"Grid sell: ranging market, RSI={cur_rsi:.1f}",
                    regime=regime,
                    atr=cur_atr,
                ))

        return signals


# ---------------------------------------------------------------------------
# BTC trend detector (used by AdaptiveQuantStrategy for alt filtering)
# ---------------------------------------------------------------------------

def detect_btc_trend(df: pd.DataFrame) -> str:
    """Classify BTC's current trend from OHLCV data.

    Returns:
        "crash"   – BTC dropped >3 % in 10 bars (immediate danger)
        "bearish" – EMA-20 < EMA-50 (downtrend)
        "bullish" – EMA-20 > EMA-50 (uptrend)
        "neutral" – insufficient data
    """
    if df is None or len(df) < 55:
        return "neutral"
    try:
        close = pd.to_numeric(df["close"], errors="coerce")
        if close.isna().sum() > len(close) * 0.3:
            return "neutral"
        # Crash detection: >3 % drop in last 10 bars
        if len(close) >= 10:
            roc = (float(close.iloc[-1]) - float(close.iloc[-10])) / float(close.iloc[-10])
            if roc < -0.03:
                return "crash"
        ema20 = calc_ema(close, 20)
        ema50 = calc_ema(close, 50)
        if float(ema20.iloc[-1]) > float(ema50.iloc[-1]):
            return "bullish"
        return "bearish"
    except Exception:
        return "neutral"


def detect_htf_trend(df: pd.DataFrame) -> str:
    """Classify higher-timeframe trend from OHLCV data.

    Uses the existing detect_regime() but maps to a simpler
    bullish / bearish / neutral vocabulary.
    """
    if df is None or len(df) < 55:
        return "neutral"
    regime = detect_regime(df)
    if regime == "trending_up":
        return "bullish"
    elif regime in ("trending_down", "high_volatility"):
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Adaptive Quant Strategy (the "personal quant" – learns from history)
# ---------------------------------------------------------------------------

class AdaptiveQuantStrategy(Strategy):
    """
    Adaptive multi-factor quant strategy that learns from trading history.

    Key innovations over TrendMomentumStrategy:
      1. Learnable feature weights — adjusted based on actual trade outcomes
         via the TradeJournal.  Features that predict wins get higher weights
         over time; poor features get dampened.
      2. Multi-timeframe confirmation — checks higher-TF trend before entry.
         Goes long on 5 m only when 1 h is bullish or neutral.
      3. BTC correlation filter — when BTC is crashing, skip all new alt longs.
         When BTC is bearish, reduce alt-long confidence.
      4. Dynamic confidence thresholds — per-pair, per-regime thresholds
         that adapt based on accumulated trade history.
      5. Wider take-profit (3.5 × ATR) with trailing stops to let winners run.
      6. All 12 original indicators PLUS higher-TF and BTC context.

    The strategy is fully backward-compatible: without a TradeJournal or
    multi-TF context it behaves like an improved TrendMomentumStrategy.
    """

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,      # match trend_momentum; trailing stop lets winners run
        adx_threshold: float = 20.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        volume_mult: float = 1.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cooldown_bars: int = 3,
        min_confidence: float = 0.45,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.adx_threshold = adx_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_mult = volume_mult
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cooldown_bars = cooldown_bars
        self.min_confidence = min_confidence
        self._last_signal: Dict[str, Dict] = {}

        # --- Adaptive context (set by trading bot each cycle) ---
        self._htf_data: Dict[str, pd.DataFrame] = {}   # symbol -> 1h DataFrame
        self._btc_data: Optional[pd.DataFrame] = None   # BTC primary-TF data
        self._trade_journal = None                       # TradeJournal instance
        self._feature_weights: Dict[str, float] = {}    # learned weights

    def set_context(
        self,
        htf_data: Optional[Dict[str, pd.DataFrame]] = None,
        btc_data: Optional[pd.DataFrame] = None,
        trade_journal=None,
    ) -> None:
        """Set multi-timeframe / adaptive context before signal generation."""
        if htf_data is not None:
            self._htf_data = htf_data
        if btc_data is not None:
            self._btc_data = btc_data
        if trade_journal is not None:
            self._trade_journal = trade_journal
            self._feature_weights = dict(trade_journal.feature_weights)

    def _w(self, feature: str) -> float:
        """Get learned weight for a feature (defaults to 1.0)."""
        return self._feature_weights.get(feature, 1.0)

    @property
    def _cold_start(self) -> bool:
        """True when we don't have enough trade history to trust HTF/BTC filters.
        In cold-start mode, skip multiplicative penalties since they destroy
        signal quality on insufficient data. Additive bonuses still apply."""
        if self._trade_journal is None:
            return True
        return len(self._trade_journal.completed_trades) < 30

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        min_rows = max(self.ema_slow, 26, self.atr_period) * 2 + 10
        if len(data) < min_rows:
            return []

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().sum() > len(close) * 0.3:
            return []

        # ---- Core indicators (same as TrendMomentum) ----
        ema_f = calc_ema(close, self.ema_fast)
        ema_s = calc_ema(close, self.ema_slow)
        rsi = calc_rsi(close, self.rsi_period)
        atr = calc_atr(data, self.atr_period)
        adx = calc_adx(data, self.atr_period)
        macd_line, signal_line, macd_hist = calc_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        vol_sma = calc_volume_sma(data, 20)
        vwap = calc_vwap(data)
        obv = calc_obv(data)
        obv_sma = calc_sma(obv, 20)
        bb_width = calc_bb_width(data, 20, 2.0)
        plus_di, minus_di = calc_di(data, self.atr_period)
        divergence = detect_rsi_divergence(close, rsi, lookback=self.rsi_period)
        regime = detect_regime(data)

        # ---- HTF & BTC context ----
        htf_trend = "neutral"
        if symbol in self._htf_data:
            htf_trend = detect_htf_trend(self._htf_data[symbol])

        btc_trend = "neutral"
        is_btc = symbol.upper().startswith("XBT") or symbol.upper().startswith("BTC")
        if is_btc:
            btc_trend = "bullish"  # BTC doesn't filter itself
        elif self._btc_data is not None:
            btc_trend = detect_btc_trend(self._btc_data)

        # ---- Snapshot current bar ----
        def _safe(s, idx=-1, default=0.0):
            try:
                v = float(s.iloc[idx])
                return default if np.isnan(v) else v
            except Exception:
                return default

        cur = {
            "price": _safe(close),
            "ema_f": _safe(ema_f), "ema_s": _safe(ema_s),
            "prev_ema_f": _safe(ema_f, -2), "prev_ema_s": _safe(ema_s, -2),
            "rsi": _safe(rsi),
            "atr": _safe(atr),
            "adx": _safe(adx),
            "plus_di": _safe(plus_di), "minus_di": _safe(minus_di),
            "macd": _safe(macd_line), "macd_signal": _safe(signal_line),
            "macd_hist": _safe(macd_hist),
            "macd_hist_prev": _safe(macd_hist, -2),
            "macd_hist_prev2": _safe(macd_hist, -3),
            "volume": _safe(pd.to_numeric(data["volume"], errors="coerce")),
            "vol_sma": _safe(vol_sma),
            "vwap": _safe(vwap),
            "obv": _safe(obv), "obv_sma": _safe(obv_sma),
            "bb_width": _safe(bb_width),
            "bb_width_prev": _safe(bb_width, -2),
        }

        if cur["atr"] <= 0 or cur["price"] <= 0:
            return []

        signals: List[Signal] = []

        # ---- detect crossover events ----
        ema_bull_cross = cur["prev_ema_f"] <= cur["prev_ema_s"] and cur["ema_f"] > cur["ema_s"]
        ema_bear_cross = cur["prev_ema_f"] >= cur["prev_ema_s"] and cur["ema_f"] < cur["ema_s"]
        ema_bull_aligned = cur["ema_f"] > cur["ema_s"]
        ema_bear_aligned = cur["ema_f"] < cur["ema_s"]

        bb_squeeze = cur["bb_width"] < cur["bb_width_prev"] and cur["bb_width"] > 0
        macd_bull_fading = (
            cur["macd_hist"] > 0
            and cur["macd_hist"] < cur["macd_hist_prev"]
            and cur["macd_hist_prev"] < cur["macd_hist_prev2"]
        )

        # =================================================================
        # BUY scoring  (base values match TrendMomentum; _w() adapts them)
        # =================================================================
        buy_reasons: List[str] = []
        buy_conf = 0.0
        buy_features: Dict[str, bool] = {}

        # 1) EMA crossover / alignment
        if ema_bull_cross:
            buy_conf += 0.20 * self._w("ema_cross")
            buy_reasons.append("EMA bullish crossover")
            buy_features["ema_cross"] = True
        elif ema_bull_aligned:
            buy_conf += 0.10 * self._w("ema_aligned")
            buy_reasons.append("EMA bullish aligned")
            buy_features["ema_aligned"] = True

        # 2) MACD
        if cur["macd"] > cur["macd_signal"] and cur["macd_hist"] > cur["macd_hist_prev"]:
            buy_conf += 0.15 * self._w("macd_signal")
            buy_reasons.append(f"MACD bullish (hist={cur['macd_hist']:.4f})")
            buy_features["macd_signal"] = True
            buy_features["macd_histogram"] = True
        elif cur["macd"] > cur["macd_signal"]:
            buy_conf += 0.05 * self._w("macd_signal")
            buy_reasons.append("MACD bullish (weak)")
            buy_features["macd_signal"] = True
        elif macd_bull_fading:
            buy_conf -= 0.05
            buy_reasons.append("MACD momentum fading")

        # 3) RSI
        if cur["rsi"] < self.rsi_overbought:
            buy_conf += 0.05
            if cur["rsi"] < 40:
                buy_conf += 0.10 * self._w("rsi_favorable")
                buy_reasons.append(f"RSI favorable ({cur['rsi']:.1f})")
                buy_features["rsi_favorable"] = True
            else:
                buy_reasons.append(f"RSI OK ({cur['rsi']:.1f})")
        else:
            buy_conf -= 0.10
            buy_reasons.append(f"RSI overbought ({cur['rsi']:.1f})")

        # 4) +DI > -DI
        if cur["plus_di"] > cur["minus_di"]:
            buy_conf += 0.10 * self._w("di_aligned")
            buy_reasons.append(f"+DI>{-1}DI ({cur['plus_di']:.1f}/{cur['minus_di']:.1f})")
            buy_features["di_aligned"] = True

        # 5) ADX
        if cur["adx"] > self.adx_threshold:
            buy_conf += 0.10 * self._w("adx_strong")
            buy_reasons.append(f"ADX strong ({cur['adx']:.1f})")
            buy_features["adx_strong"] = True

        # 6) Volume
        if cur["vol_sma"] > 0 and cur["volume"] >= cur["vol_sma"] * self.volume_mult:
            buy_conf += 0.05 * self._w("volume_confirmed")
            buy_reasons.append("Volume confirmed")
            buy_features["volume_confirmed"] = True

        # 7) OBV
        if cur["obv"] > cur["obv_sma"] and cur["obv_sma"] != 0:
            buy_conf += 0.05 * self._w("obv_aligned")
            buy_reasons.append("OBV rising")
            buy_features["obv_aligned"] = True

        # 8) VWAP
        if cur["vwap"] > 0:
            if cur["price"] <= cur["vwap"] * 1.002:
                buy_conf += 0.05 * self._w("vwap_favorable")
                buy_reasons.append("Price near/below VWAP")
                buy_features["vwap_favorable"] = True
            elif cur["price"] > cur["vwap"] * 1.01:
                buy_conf -= 0.05
                buy_reasons.append("Price above VWAP")

        # 9) RSI divergence
        if divergence == "bullish":
            buy_conf += 0.15 * self._w("rsi_divergence")
            buy_reasons.append("RSI bullish divergence")
            buy_features["rsi_divergence"] = True

        # 10) BB squeeze
        if bb_squeeze and ema_bull_aligned:
            buy_conf += 0.05 * self._w("bb_squeeze")
            buy_reasons.append("BB squeeze + bullish")
            buy_features["bb_squeeze"] = True

        # 11) HTF alignment — only after cold-start (need history to trust)
        if not self._cold_start:
            if htf_trend == "bullish":
                buy_conf += 0.12 * self._w("htf_aligned")
                buy_reasons.append("1h trend bullish ▲")
                buy_features["htf_aligned"] = True
            elif htf_trend == "bearish":
                buy_conf *= 0.80
                buy_reasons.append("1h trend bearish ▼ (discount)")

        # 12) BTC filter — only after cold-start
        if not self._cold_start:
            if btc_trend == "crash":
                buy_conf *= 0.25
                buy_reasons.append("BTC CRASH — heavy penalty")
            elif btc_trend == "bearish" and not is_btc:
                buy_conf *= 0.80
                buy_reasons.append("BTC bearish (alt discount)")
                buy_features["btc_favorable"] = False
            elif btc_trend == "bullish":
                buy_conf += 0.08 * self._w("btc_favorable")
                buy_reasons.append("BTC bullish ▲")
                buy_features["btc_favorable"] = True

        # 13) Regime adjustment (matches trend_momentum's proven values)
        if regime == "high_volatility":
            buy_conf *= 0.35
            buy_reasons.append("HIGH VOLATILITY penalty")
        elif regime == "trending_down":
            buy_conf *= 0.45
            buy_reasons.append("Downtrend discount")
        elif regime == "trending_up":
            buy_conf *= 1.10
            buy_reasons.append("Uptrend bonus")

        # ---- Dynamic threshold ----
        dynamic_min = self.min_confidence
        if self._trade_journal is not None:
            dynamic_min = self._trade_journal.get_dynamic_confidence_threshold(symbol, regime)

        if buy_conf >= dynamic_min:
            sl = cur["price"] - self.atr_sl_mult * cur["atr"]
            tp = cur["price"] + self.atr_tp_mult * cur["atr"]
            risk = cur["price"] - sl
            reward = tp - cur["price"]
            if risk > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=min(buy_conf, 1.0),
                    entry_price=cur["price"],
                    stop_loss=sl,
                    take_profit=tp,
                    reason=" | ".join(buy_reasons),
                    regime=regime,
                    atr=cur["atr"],
                    features_active=buy_features,
                    indicator_snapshot=cur,
                    htf_trend=htf_trend,
                    btc_trend=btc_trend,
                ))

        # =================================================================
        # SELL scoring (mirror — base values match TrendMomentum)
        # =================================================================
        sell_reasons: List[str] = []
        sell_conf = 0.0
        sell_features: Dict[str, bool] = {}

        if ema_bear_cross:
            sell_conf += 0.20 * self._w("ema_cross")
            sell_reasons.append("EMA bearish crossover")
            sell_features["ema_cross"] = True
        elif ema_bear_aligned:
            sell_conf += 0.10 * self._w("ema_aligned")
            sell_reasons.append("EMA bearish aligned")
            sell_features["ema_aligned"] = True

        if cur["macd"] < cur["macd_signal"] and cur["macd_hist"] < cur["macd_hist_prev"]:
            sell_conf += 0.15 * self._w("macd_signal")
            sell_reasons.append(f"MACD bearish (hist={cur['macd_hist']:.4f})")
            sell_features["macd_signal"] = True
            sell_features["macd_histogram"] = True
        elif cur["macd"] < cur["macd_signal"]:
            sell_conf += 0.05 * self._w("macd_signal")
            sell_reasons.append("MACD bearish (weak)")
            sell_features["macd_signal"] = True

        if cur["rsi"] > self.rsi_oversold:
            sell_conf += 0.05
            if cur["rsi"] > 60:
                sell_conf += 0.10 * self._w("rsi_favorable")
                sell_reasons.append(f"RSI high ({cur['rsi']:.1f})")
                sell_features["rsi_favorable"] = True
            else:
                sell_reasons.append(f"RSI OK ({cur['rsi']:.1f})")
        else:
            sell_conf -= 0.10
            sell_reasons.append(f"RSI oversold ({cur['rsi']:.1f})")

        if cur["minus_di"] > cur["plus_di"]:
            sell_conf += 0.10 * self._w("di_aligned")
            sell_reasons.append(f"-DI>+DI ({cur['minus_di']:.1f}/{cur['plus_di']:.1f})")
            sell_features["di_aligned"] = True

        if cur["adx"] > self.adx_threshold:
            sell_conf += 0.10 * self._w("adx_strong")
            sell_reasons.append(f"ADX strong ({cur['adx']:.1f})")
            sell_features["adx_strong"] = True

        if cur["vol_sma"] > 0 and cur["volume"] >= cur["vol_sma"] * self.volume_mult:
            sell_conf += 0.05 * self._w("volume_confirmed")
            sell_reasons.append("Volume confirmed")
            sell_features["volume_confirmed"] = True

        if cur["obv"] < cur["obv_sma"] and cur["obv_sma"] != 0:
            sell_conf += 0.05 * self._w("obv_aligned")
            sell_reasons.append("OBV falling")
            sell_features["obv_aligned"] = True

        if cur["vwap"] > 0:
            if cur["price"] >= cur["vwap"] * 0.998:
                sell_conf += 0.05 * self._w("vwap_favorable")
                sell_reasons.append("Price near/above VWAP")
                sell_features["vwap_favorable"] = True
            elif cur["price"] < cur["vwap"] * 0.99:
                sell_conf -= 0.05

        if divergence == "bearish":
            sell_conf += 0.15 * self._w("rsi_divergence")
            sell_reasons.append("RSI bearish divergence")
            sell_features["rsi_divergence"] = True

        if bb_squeeze and ema_bear_aligned:
            sell_conf += 0.05 * self._w("bb_squeeze")
            sell_reasons.append("BB squeeze + bearish")
            sell_features["bb_squeeze"] = True

        # HTF — only after cold-start
        if not self._cold_start:
            if htf_trend == "bearish":
                sell_conf += 0.12 * self._w("htf_aligned")
                sell_reasons.append("1h trend bearish ▼")
                sell_features["htf_aligned"] = True
            elif htf_trend == "bullish":
                sell_conf *= 0.80
                sell_reasons.append("1h trend bullish ▲ (discount)")

        # BTC — only after cold-start
        if not self._cold_start:
            if btc_trend == "crash":
                sell_conf += 0.10
                sell_reasons.append("BTC crashing (bonus for sell)")
                sell_features["btc_favorable"] = True
            elif btc_trend == "bullish" and not is_btc:
                sell_conf *= 0.80
                sell_reasons.append("BTC bullish (alt-sell discount)")
            elif btc_trend == "bearish":
                sell_conf += 0.08 * self._w("btc_favorable")
                sell_reasons.append("BTC bearish ▼")
                sell_features["btc_favorable"] = True

        # Regime (matches trend_momentum's proven values)
        if regime == "high_volatility":
            sell_conf *= 0.35
            sell_reasons.append("HIGH VOLATILITY penalty")
        elif regime == "trending_up":
            sell_conf *= 0.45
            sell_reasons.append("Uptrend discount")
        elif regime == "trending_down":
            sell_conf *= 1.10
            sell_reasons.append("Downtrend bonus")

        if sell_conf >= dynamic_min:
            sl = cur["price"] + self.atr_sl_mult * cur["atr"]
            tp = cur["price"] - self.atr_tp_mult * cur["atr"]
            risk = sl - cur["price"]
            reward = cur["price"] - tp
            if risk > 0 and (reward / risk) >= 1.5:
                signals.append(Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=min(sell_conf, 1.0),
                    entry_price=cur["price"],
                    stop_loss=sl,
                    take_profit=tp,
                    reason=" | ".join(sell_reasons),
                    regime=regime,
                    atr=cur["atr"],
                    features_active=sell_features,
                    indicator_snapshot=cur,
                    htf_trend=htf_trend,
                    btc_trend=btc_trend,
                ))

        # ---- Signal cooldown ----
        filtered_signals: List[Signal] = []
        for sig in signals:
            last_bars = self._last_signal.get(sig.symbol, {}).get(sig.action)
            if last_bars is not None and last_bars < self.cooldown_bars:
                continue
            filtered_signals.append(sig)

        if symbol not in self._last_signal:
            self._last_signal[symbol] = {}
        emitted_actions = {s.action for s in filtered_signals}
        for sig in filtered_signals:
            self._last_signal[symbol][sig.action] = 0
        for action in list(self._last_signal.get(symbol, {})):
            if action not in emitted_actions:
                self._last_signal[symbol][action] += 1

        return filtered_signals


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_strategy(strategy_type: str, **kwargs) -> Strategy:
    """Factory function to create strategy instances"""
    strategies = {
        "grid": GridTradingStrategy,
        "trend_momentum": TrendMomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "arbitrage": StatisticalArbitrageStrategy,
        "scalp": ScalpMomentumStrategy,
        "adaptive": AdaptiveQuantStrategy,
    }

    if strategy_type not in strategies:
        logger.warning(f"Unknown strategy: {strategy_type}, defaulting to adaptive")
        strategy_type = "adaptive"

    # Filter kwargs to only those accepted by the target class
    sig = inspect.signature(strategies[strategy_type].__init__)
    valid_params = {k for k in sig.parameters if k != "self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return strategies[strategy_type](**filtered)
