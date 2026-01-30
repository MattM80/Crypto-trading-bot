"""
Low-risk trading strategies for the crypto bot.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger
from abc import ABC, abstractmethod

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

class Strategy(ABC):
    """Base strategy class"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals"""
        pass

class GridTradingStrategy(Strategy):
    """
    Grid Trading Strategy - Low risk, high frequency strategy.
    Places buy orders below current price and sell orders above.
    """
    
    def __init__(
        self,
        grid_levels: int = 10,
        range_percent: float = 0.05,
        stop_loss_percent: float = 0.02,
        take_profit_percent: float = 0.05
    ):
        self.grid_levels = grid_levels
        self.range_percent = range_percent
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate grid trading signals"""
        if len(data) < 2:
            return []
        
        current_price = data.iloc[-1]["close"]
        signals = []
        
        # Calculate grid range
        upper_bound = current_price * (1 + self.range_percent)
        lower_bound = current_price * (1 - self.range_percent)
        
        # Create buy grid below current price
        buy_prices = np.linspace(lower_bound, current_price, self.grid_levels // 2)
        for price in buy_prices:
            if price < current_price:
                signal = Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=0.7,
                    entry_price=price,
                    stop_loss=price * (1 - self.stop_loss_percent),
                    take_profit=price * (1 + self.take_profit_percent),
                    reason=f"Grid buy level at {price:.2f}"
                )
                signals.append(signal)
        
        # Create sell grid above current price
        sell_prices = np.linspace(current_price, upper_bound, self.grid_levels // 2)
        for price in sell_prices:
            if price > current_price:
                signal = Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=0.7,
                    entry_price=price,
                    stop_loss=price * (1 + self.stop_loss_percent),
                    take_profit=price * (1 - self.take_profit_percent),
                    reason=f"Grid sell level at {price:.2f}"
                )
                signals.append(signal)
        
        return signals

class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy - Buys when price is oversold, sells when overbought.
    Uses RSI and Bollinger Bands for signal generation.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def _calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = data.rolling(window=self.bb_period).mean()
        std = data.rolling(window=self.bb_period).std()
        
        upper_band = ma + (std * self.bb_std)
        lower_band = ma - (std * self.bb_std)
        
        return upper_band, ma, lower_band
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate mean reversion signals"""
        if len(data) < max(self.rsi_period, self.bb_period):
            return []
        
        close = data["close"]
        
        # Calculate indicators
        rsi = self._calculate_rsi(close)
        upper_band, middle_band, lower_band = self._calculate_bollinger_bands(close)
        
        signals = []
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Buy signal: Oversold condition
        if current_rsi < self.rsi_oversold and current_price < lower_band.iloc[-1]:
            signal = Signal(
                symbol=symbol,
                action="BUY",
                confidence=0.8,
                entry_price=current_price,
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.05,
                reason=f"Oversold (RSI={current_rsi:.2f}, price below lower band)"
            )
            signals.append(signal)
        
        # Sell signal: Overbought condition
        elif current_rsi > self.rsi_overbought and current_price > upper_band.iloc[-1]:
            signal = Signal(
                symbol=symbol,
                action="SELL",
                confidence=0.8,
                entry_price=current_price,
                stop_loss=current_price * 1.02,
                take_profit=current_price * 0.95,
                reason=f"Overbought (RSI={current_rsi:.2f}, price above upper band)"
            )
            signals.append(signal)
        
        return signals

class StatisticalArbitrageStrategy(Strategy):
    """
    Statistical Arbitrage - Identifies when price deviates from fair value.
    More conservative, suitable for pairs trading.
    """
    
    def __init__(self, z_score_threshold: float = 2.0, lookback_period: int = 50):
        self.z_score_threshold = z_score_threshold
        self.lookback_period = lookback_period
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate statistical arbitrage signals"""
        if len(data) < self.lookback_period:
            return []
        
        close = data["close"]
        
        # Calculate mean and standard deviation
        mean = close.tail(self.lookback_period).mean()
        std = close.tail(self.lookback_period).std()
        
        if std == 0:
            return []
        
        # Calculate z-score
        current_price = close.iloc[-1]
        z_score = (current_price - mean) / std
        
        signals = []
        
        # Buy signal: Price is below mean (z-score < -threshold)
        if z_score < -self.z_score_threshold:
            signal = Signal(
                symbol=symbol,
                action="BUY",
                confidence=0.75,
                entry_price=current_price,
                stop_loss=current_price * 0.97,
                take_profit=mean,
                reason=f"Statistical undervalue (z-score={z_score:.2f})"
            )
            signals.append(signal)
        
        # Sell signal: Price is above mean (z-score > threshold)
        elif z_score > self.z_score_threshold:
            signal = Signal(
                symbol=symbol,
                action="SELL",
                confidence=0.75,
                entry_price=current_price,
                stop_loss=current_price * 1.03,
                take_profit=mean,
                reason=f"Statistical overvalue (z-score={z_score:.2f})"
            )
            signals.append(signal)
        
        return signals

def create_strategy(strategy_type: str, **kwargs) -> Strategy:
    """Factory function to create strategy instances"""
    strategies = {
        "grid": GridTradingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "arbitrage": StatisticalArbitrageStrategy,
    }
    
    if strategy_type not in strategies:
        logger.warning(f"Unknown strategy: {strategy_type}, defaulting to grid")
        strategy_type = "grid"
    
    return strategies[strategy_type](**kwargs)
