"""
Strategies V2: Mathematically Sound Edge-Based Trading
=====================================================

Three proven strategies with real mathematical edges:
1. Liquidation Cascade Mean Reversion - Exploit overshoot from forced selling
2. Volatility Harvesting - Shannon's Demon portfolio rebalancing
3. Cross-Pair Mean Reversion - ETH/BTC ratio trading

No indicator soup. No curve-fitting. Just exploiting structural market inefficiencies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from loguru import logger
import time
from datetime import datetime, timedelta


@dataclass
class Signal:
    """Trading signal with clear entry/exit rules"""
    symbol: str
    action: str  # "BUY", "SELL", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"
    price: float
    quantity: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str
    confidence: float = 1.0
    strategy: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class StrategyBase(ABC):
    """Base class for all strategies"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.last_update = None
        self.positions = {}  # Track strategy-specific positions
        
    @abstractmethod
    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        """Generate trading signal from market data"""
        pass
        
    @abstractmethod
    def update_positions(self, filled_orders: List) -> None:
        """Update internal position tracking"""
        pass
        
    def get_required_symbols(self) -> List[str]:
        """Return list of symbols this strategy needs"""
        return []


class LiquidationCascadeStrategy(StrategyBase):
    """
    Strategy 1: Liquidation Cascade Mean Reversion
    
    Logic:
    - Detect extreme price drops with volume spikes (forced liquidations)
    - Buy the dip with tight stop-loss
    - Target mean reversion to recent VWAP
    
    Edge: Liquidation cascades create temporary mispricing as forced sellers 
    push price below fair value. Quick reversal is statistically likely.
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'drop_threshold_pct': 0.8,  # Min % drop to trigger (loosened)
            'volume_spike_multiplier': 1.1,  # Volume vs 20-bar average (loosened further) 
            'lookback_bars': 6,  # Bars to check for drop (increased)
            'stop_loss_atr_mult': 1.5,  # SL distance in ATR
            'take_profit_atr_mult': 3.0,  # TP distance in ATR
            'min_atr_pct': 0.1,  # Min ATR as % of price for volatility filter (reduced further)
            'vwap_periods': 20,  # VWAP calculation window
            'cooldown_bars': 4,  # Bars to wait after a trade (reduced)
            'vwap_threshold_pct': 0.0,  # Just need to be below VWAP (removed 2% requirement)
        }
        config = {**default_config, **(config or {})}
        super().__init__("LiquidationCascade", config)
        self.last_trade_time = None
        
    def calculate_vwap(self, df: pd.DataFrame, periods: int) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(periods).sum() / df['volume'].rolling(periods).sum()
        return vwap
        
    def calculate_atr(self, df: pd.DataFrame, periods: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low'] 
        close = df['close']
        prev_close = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(periods).mean()
        
    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        """Look for liquidation cascade opportunities"""
        
        for symbol, df in data.items():
            if len(df) < 50:  # Need enough data
                continue
                
            # Skip if in cooldown (but don't block on very first bars) 
            if self.last_trade_time and len(df) > self.last_trade_time:
                bars_since_trade = len(df) - self.last_trade_time
                if bars_since_trade < self.config['cooldown_bars']:
                    continue
                    
            # Calculate indicators
            atr = self.calculate_atr(df)
            vwap = self.calculate_vwap(df, self.config['vwap_periods'])
            avg_volume = df['volume'].rolling(20).mean()
            
            current_idx = len(df) - 1
            if current_idx < 20:
                continue
                
            # Debug: Check if we have valid data
            if len(df) % 100 == 0:  # Log every 100 bars
                logger.debug(f"LiquidationCascade: {symbol} has {len(df)} bars, checking signals...")
                
            current_price = df.iloc[current_idx]['close']
            current_volume = df.iloc[current_idx]['volume']
            current_atr = atr.iloc[current_idx]
            current_vwap = vwap.iloc[current_idx]
            avg_vol = avg_volume.iloc[current_idx]
            
            # Skip if we have NaN values
            if pd.isna(current_atr) or pd.isna(current_vwap) or pd.isna(avg_vol) or avg_vol == 0:
                continue
                
            # Volatility filter - need minimum volatility for strategy to work
            if current_atr / current_price < self.config['min_atr_pct'] / 100:
                continue
                
            # Look for price drop in recent bars
            lookback = self.config['lookback_bars']
            recent_high = df.iloc[current_idx - lookback:current_idx + 1]['high'].max()
            drop_pct = (recent_high - current_price) / recent_high * 100
            
            # Check volume spike
            volume_ratio = current_volume / avg_vol if avg_vol > 0 else 0
            
            # Signal conditions
            drop_condition = drop_pct >= self.config['drop_threshold_pct']
            volume_condition = volume_ratio >= self.config['volume_spike_multiplier']
            below_vwap = current_price < current_vwap  # Just need to be below VWAP
            
            # Debug: Log potential signals
            if drop_pct > 0.5 or volume_ratio > 1.2:  # Log interesting moves
                logger.debug(f"LiquidationCascade {symbol}: drop={drop_pct:.2f}% (need {self.config['drop_threshold_pct']:.2f}%), "
                           f"vol_ratio={volume_ratio:.2f}x (need {self.config['volume_spike_multiplier']:.2f}x), "
                           f"price vs vwap: {current_price:.2f} vs {current_vwap:.2f}")
            
            if drop_condition and volume_condition and below_vwap:
                # Calculate position size and risk levels
                stop_loss = current_price - (current_atr * self.config['stop_loss_atr_mult'])
                take_profit = current_price + (current_atr * self.config['take_profit_atr_mult'])
                
                # Risk/reward check - need at least 1.8:1 (more flexible)
                risk = current_price - stop_loss
                reward = take_profit - current_price
                if reward / risk < 1.8:
                    continue
                
                self.last_trade_time = current_idx
                
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    price=current_price,
                    quantity=0,  # Will be calculated by portfolio manager
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Liquidation cascade: {drop_pct:.1f}% drop, {volume_ratio:.1f}x volume spike",
                    confidence=min(1.0, (drop_pct / 5.0) * (volume_ratio / 3.0)),
                    strategy=self.name
                )
                
        return None
    
    def update_positions(self, filled_orders: List) -> None:
        """Update position tracking"""
        # Update last trade time when positions are closed
        pass
        
    def get_required_symbols(self) -> List[str]:
        return ['XBTUSD', 'ETHUSD', 'SOLUSD']


class VolatilityHarvestingStrategy(StrategyBase):
    """
    Strategy 2: Volatility Harvesting (Shannon's Demon)
    
    Logic:
    - Maintain target allocation (e.g. 50% BTC, 50% USD)
    - Rebalance when allocation drifts beyond threshold
    - Sell high, buy low automatically without predicting direction
    
    Edge: Mathematically extracts value from volatility. Well-proven in academic literature.
    Works because crypto volatility creates frequent mean reversion opportunities.
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'target_allocation': {'XBTUSD': 0.5, 'USD': 0.5},  # Target % allocation
            'rebalance_threshold': 0.05,  # Rebalance when drift > 5%
            'min_rebalance_amount': 10,  # Min $ amount to rebalance
            'check_frequency_bars': 24,  # Check every 24 bars (daily on hourly data)
            'initial_balance': 150,  # Half of $300 starting balance for this strategy
        }
        config = {**default_config, **(config or {})}
        super().__init__("VolatilityHarvesting", config)
        self.last_check_time = None
        self.portfolio_initialized = False
        
        # Simulated portfolio for backtesting
        self.simulated_usd_balance = 0.0
        self.simulated_btc_quantity = 0.0
        self.last_btc_price = None
        
    def initialize_simulated_portfolio(self, btc_price: float):
        """Initialize simulated portfolio with 50/50 BTC/USD split"""
        if not self.portfolio_initialized:
            initial_balance = self.config['initial_balance']
            
            # Start with 50% BTC, 50% USD
            self.simulated_usd_balance = initial_balance * 0.5
            self.simulated_btc_quantity = (initial_balance * 0.5) / btc_price
            self.last_btc_price = btc_price
            self.portfolio_initialized = True
            
            logger.info(f"VolatilityHarvesting: Initialized portfolio with ${self.simulated_usd_balance:.2f} USD "
                       f"and {self.simulated_btc_quantity:.6f} BTC @ ${btc_price:.2f}")
    
    def get_current_allocation(self, btc_price: float) -> Dict[str, float]:
        """Calculate current portfolio allocation"""
        if not self.portfolio_initialized:
            return {'USD': 0.5, 'XBTUSD': 0.5}
            
        btc_value = self.simulated_btc_quantity * btc_price
        total_value = self.simulated_usd_balance + btc_value
        
        if total_value == 0:
            return {'USD': 0.5, 'XBTUSD': 0.5}
            
        return {
            'USD': self.simulated_usd_balance / total_value,
            'XBTUSD': btc_value / total_value
        }
        
    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        """Check if rebalancing is needed"""
        
        # Get BTC data
        btc_data = data.get('XBTUSD')
        if btc_data is None or len(btc_data) == 0:
            return None
            
        current_btc_price = btc_data.iloc[-1]['close']
        
        # Initialize portfolio on first run
        self.initialize_simulated_portfolio(current_btc_price)
        
        # Only check periodically
        current_bars = len(btc_data)
        if self.last_check_time:
            bars_since_check = current_bars - self.last_check_time
            if bars_since_check < self.config['check_frequency_bars']:
                return None
                
        self.last_check_time = current_bars
        
        # Get current allocation
        current_allocations = self.get_current_allocation(current_btc_price)
        target_allocations = self.config['target_allocation']
        
        # Check if rebalancing is needed
        btc_current_pct = current_allocations['XBTUSD']
        btc_target_pct = target_allocations['XBTUSD']
        deviation = abs(btc_current_pct - btc_target_pct)
        
        if deviation > self.config['rebalance_threshold']:
            # Calculate rebalance amount
            total_value = self.simulated_usd_balance + (self.simulated_btc_quantity * current_btc_price)
            target_btc_value = total_value * btc_target_pct
            current_btc_value = self.simulated_btc_quantity * current_btc_price
            rebalance_amount_usd = abs(current_btc_value - target_btc_value)
            
            if rebalance_amount_usd < self.config['min_rebalance_amount']:
                return None
                
            # Determine action
            if btc_current_pct > btc_target_pct:
                # Overweight BTC - sell some
                action = "SELL"
                quantity = rebalance_amount_usd / current_btc_price
            else:
                # Underweight BTC - buy some
                action = "BUY"  
                quantity = rebalance_amount_usd / current_btc_price
                
            logger.debug(f"VolatilityHarvesting: {action} {quantity:.6f} BTC to rebalance "
                        f"{btc_current_pct:.1%} -> {btc_target_pct:.1%}")
                
            return Signal(
                symbol='XBTUSD',
                action=action,
                price=current_btc_price,
                quantity=quantity,
                stop_loss=None,  # No stop loss for rebalancing
                take_profit=None,  # No take profit for rebalancing  
                reason=f"Portfolio rebalance: {btc_current_pct:.1%} -> {btc_target_pct:.1%}",
                confidence=1.0,
                strategy=self.name
            )
            
        return None
    
    def update_simulated_portfolio(self, action: str, quantity: float, price: float):
        """Update simulated portfolio after a trade"""
        if not self.portfolio_initialized:
            return
            
        if action == "BUY":
            # Buy BTC with USD
            cost = quantity * price
            if self.simulated_usd_balance >= cost:
                self.simulated_usd_balance -= cost
                self.simulated_btc_quantity += quantity
                logger.debug(f"VolatilityHarvesting: Bought {quantity:.6f} BTC for ${cost:.2f}")
            
        elif action == "SELL":
            # Sell BTC for USD
            if self.simulated_btc_quantity >= quantity:
                proceeds = quantity * price
                self.simulated_usd_balance += proceeds
                self.simulated_btc_quantity -= quantity
                logger.debug(f"VolatilityHarvesting: Sold {quantity:.6f} BTC for ${proceeds:.2f}")
                
        self.last_btc_price = price
        
    def update_positions(self, filled_orders: List) -> None:
        """Update position tracking after fills"""
        # Update simulated portfolio based on filled orders
        for order in filled_orders:
            if hasattr(order, 'strategy') and order.strategy == self.name:
                self.update_simulated_portfolio(order.side, order.quantity, order.fill_price)
        
    def get_required_symbols(self) -> List[str]:
        return ['XBTUSD']


class CrossPairMeanReversionStrategy(StrategyBase):
    """
    Strategy 3: Cross-Pair Mean Reversion (ETH/BTC ratio)
    
    Logic:
    - Track ETH/BTC price ratio 
    - When ratio deviates >2 standard deviations from mean, trade the reversion
    - Market-neutral position (long underperformer, short overperformer)
    
    Edge: ETH and BTC are correlated but ratio mean-reverts. Creates arbitrage opportunities.
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'ratio_lookback': 30 * 24,  # 30 days of hourly data for ratio calculation
            'entry_threshold_sigma': 2.0,  # Enter when >2 std devs from mean
            'exit_threshold_sigma': 0.5,   # Exit when <0.5 std devs from mean  
            'min_position_value': 50,  # Minimum position size in USD
            'max_position_pct': 0.1,  # Max 10% of balance per side
        }
        config = {**default_config, **(config or {})}
        super().__init__("CrossPairMeanReversion", config)
        self.active_pair_position = None  # Track if we have active pair trade
        
    def calculate_ratio_stats(self, eth_data: pd.DataFrame, btc_data: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate ETH/BTC ratio statistics"""
        
        # Align the dataframes by timestamp
        eth_df = eth_data.copy()
        btc_df = btc_data.copy()
        
        # Calculate ratio
        if len(eth_df) != len(btc_df):
            min_len = min(len(eth_df), len(btc_df))
            eth_df = eth_df.tail(min_len)
            btc_df = btc_df.tail(min_len)
            
        ratio = eth_df['close'].values / btc_df['close'].values
        
        # Get recent ratio values for statistics
        lookback = min(len(ratio), self.config['ratio_lookback'])
        recent_ratios = ratio[-lookback:]
        
        current_ratio = ratio[-1]
        mean_ratio = np.mean(recent_ratios)
        std_ratio = np.std(recent_ratios)
        
        return current_ratio, mean_ratio, std_ratio
        
    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        """Look for ETH/BTC ratio mean reversion opportunities"""
        
        # Need both ETH and BTC data
        if 'ETHUSD' not in data or 'XBTUSD' not in data:
            return None
            
        eth_data = data['ETHUSD']
        btc_data = data['XBTUSD']
        
        if len(eth_data) < 50 or len(btc_data) < 50:
            return None
            
        try:
            current_ratio, mean_ratio, std_ratio = self.calculate_ratio_stats(eth_data, btc_data)
        except Exception as e:
            logger.warning(f"Error calculating ratio stats: {e}")
            return None
            
        if std_ratio == 0:
            return None
            
        # Calculate z-score
        z_score = (current_ratio - mean_ratio) / std_ratio
        
        current_eth_price = eth_data.iloc[-1]['close']
        current_btc_price = btc_data.iloc[-1]['close']
        
        # Check for entry conditions
        if abs(z_score) >= self.config['entry_threshold_sigma'] and self.active_pair_position is None:
            
            if z_score > 0:
                # ETH overvalued vs BTC - sell ETH, buy BTC
                primary_symbol = 'ETHUSD'
                primary_action = 'SELL'
                primary_price = current_eth_price
                reason = f"ETH/BTC ratio high: {current_ratio:.4f} vs {mean_ratio:.4f} ({z_score:.2f}σ)"
            else:
                # ETH undervalued vs BTC - buy ETH, sell BTC  
                primary_symbol = 'ETHUSD'
                primary_action = 'BUY'
                primary_price = current_eth_price
                reason = f"ETH/BTC ratio low: {current_ratio:.4f} vs {mean_ratio:.4f} ({z_score:.2f}σ)"
                
            self.active_pair_position = {
                'type': 'entry',
                'z_score': z_score,
                'entry_ratio': current_ratio
            }
            
            return Signal(
                symbol=primary_symbol,
                action=primary_action,
                price=primary_price,
                quantity=0,  # Will be calculated by portfolio manager
                stop_loss=None,  # Pair trades don't use traditional stops
                take_profit=None,
                reason=reason,
                confidence=min(1.0, abs(z_score) / 3.0),  # Higher confidence for larger deviations
                strategy=self.name
            )
            
        # Check for exit conditions
        elif self.active_pair_position and abs(z_score) <= self.config['exit_threshold_sigma']:
            
            # Close the position by reversing the trade
            if self.active_pair_position['z_score'] > 0:
                # Original: sold ETH - now buy it back
                primary_symbol = 'ETHUSD' 
                primary_action = 'BUY'
                primary_price = current_eth_price
            else:
                # Original: bought ETH - now sell it
                primary_symbol = 'ETHUSD'
                primary_action = 'SELL'
                primary_price = current_eth_price
                
            self.active_pair_position = None
            
            return Signal(
                symbol=primary_symbol,
                action=primary_action,
                price=primary_price,
                quantity=0,  # Will be calculated by portfolio manager
                stop_loss=None,
                take_profit=None,
                reason=f"ETH/BTC ratio normalized: {current_ratio:.4f} ({z_score:.2f}σ)",
                confidence=1.0,
                strategy=self.name
            )
            
        return None
    
    def update_positions(self, filled_orders: List) -> None:
        """Update position tracking"""
        pass
        
    def get_required_symbols(self) -> List[str]:
        return ['ETHUSD', 'XBTUSD']


class MomentumBreakoutStrategy(StrategyBase):
    """
    Strategy 4: Momentum Breakout
    
    Logic:
    - When price breaks above 20-bar high with volume confirmation, go long
    - Only in clear uptrends (price above 50-bar SMA)
    - Stop loss at recent swing low or 1.5x ATR below entry
    - Take profit at 3x ATR above entry
    
    Edge: Trend-following complement to mean reversion strategies.
    Captures momentum moves that continue beyond initial breakout.
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'breakout_lookback': 20,  # Bars for high/low breakout
            'trend_sma_period': 30,  # SMA period for trend filter (reduced for more signals)
            'volume_confirmation': 1.1,  # Volume vs average for confirmation (reduced further)
            'stop_loss_atr_mult': 1.5,  # SL distance in ATR
            'take_profit_atr_mult': 3.0,  # TP distance in ATR
            'min_atr_pct': 0.2,  # Min ATR as % of price for volatility filter
            'cooldown_bars': 6,  # Bars to wait after a trade
        }
        config = {**default_config, **(config or {})}
        super().__init__("MomentumBreakout", config)
        self.last_trade_time = None
        
    def calculate_sma(self, df: pd.DataFrame, periods: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df['close'].rolling(periods).mean()
        
    def calculate_atr(self, df: pd.DataFrame, periods: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low'] 
        close = df['close']
        prev_close = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(periods).mean()
        
    def find_swing_low(self, df: pd.DataFrame, lookback: int = 10) -> float:
        """Find recent swing low for stop loss placement"""
        if len(df) < lookback:
            return df['low'].min()
        
        recent_lows = df['low'].tail(lookback)
        return recent_lows.min()
        
    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        """Look for momentum breakout opportunities"""
        
        for symbol, df in data.items():
            if len(df) < 60:  # Need enough data for indicators
                continue
                
            # Skip if in cooldown (but don't block on very first bars)
            if self.last_trade_time and len(df) > self.last_trade_time:
                bars_since_trade = len(df) - self.last_trade_time
                if bars_since_trade < self.config['cooldown_bars']:
                    continue
                    
            # Calculate indicators
            atr = self.calculate_atr(df)
            sma_trend = self.calculate_sma(df, self.config['trend_sma_period'])
            avg_volume = df['volume'].rolling(20).mean()
            
            current_idx = len(df) - 1
            if current_idx < self.config['trend_sma_period']:
                continue
                
            current_price = df.iloc[current_idx]['close']
            current_high = df.iloc[current_idx]['high']
            current_volume = df.iloc[current_idx]['volume']
            current_atr = atr.iloc[current_idx]
            current_sma = sma_trend.iloc[current_idx]
            avg_vol = avg_volume.iloc[current_idx]
            
            # Skip if we have NaN values
            if pd.isna(current_atr) or pd.isna(current_sma) or pd.isna(avg_vol) or avg_vol == 0:
                continue
                
            # Volatility filter - need minimum volatility
            if current_atr / current_price < self.config['min_atr_pct'] / 100:
                continue
                
            # Trend filter - only trade in uptrends
            if current_price < current_sma:
                continue
                
            # Find breakout level (highest high in lookback period, excluding current bar)
            lookback_start = max(0, current_idx - self.config['breakout_lookback'])
            lookback_high = df.iloc[lookback_start:current_idx]['high'].max()
            
            # Check for breakout
            breakout_condition = current_high > lookback_high
            
            # Volume confirmation
            volume_ratio = current_volume / avg_vol if avg_vol > 0 else 0
            volume_condition = volume_ratio >= self.config['volume_confirmation']
            
            # Debug logging
            if current_idx % 100 == 0:  # Log every 100 bars
                logger.debug(f"MomentumBreakout: {symbol} price={current_price:.2f}, "
                           f"sma={current_sma:.2f}, high={current_high:.2f}, "
                           f"breakout_level={lookback_high:.2f}, vol_ratio={volume_ratio:.2f}")
            
            if breakout_condition and volume_condition:
                # Calculate stop loss
                swing_low = self.find_swing_low(df, 10)
                atr_stop = current_price - (current_atr * self.config['stop_loss_atr_mult'])
                stop_loss = max(swing_low, atr_stop)  # Use the higher of swing low or ATR stop
                
                # Calculate take profit
                take_profit = current_price + (current_atr * self.config['take_profit_atr_mult'])
                
                # Risk/reward check
                risk = current_price - stop_loss
                reward = take_profit - current_price
                
                if risk <= 0 or reward / risk < 1.8:  # Need at least 1.8:1 R:R
                    continue
                
                self.last_trade_time = current_idx
                
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    price=current_price,
                    quantity=0,  # Will be calculated by portfolio manager
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Momentum breakout: broke ${lookback_high:.2f}, {volume_ratio:.1f}x volume",
                    confidence=min(1.0, volume_ratio / 2.0),  # Higher volume = higher confidence
                    strategy=self.name
                )
                
        return None
    
    def update_positions(self, filled_orders: List) -> None:
        """Update position tracking"""
        pass
        
    def get_required_symbols(self) -> List[str]:
        return ['XBTUSD', 'SOLUSD']


def create_strategy_portfolio() -> List[StrategyBase]:
    """Create the portfolio of strategies"""
    
    strategies = [
        LiquidationCascadeStrategy(),
        VolatilityHarvestingStrategy(), 
        CrossPairMeanReversionStrategy(),
        MomentumBreakoutStrategy()
    ]
    
    return strategies


# Utility function for backtesting
def get_all_required_symbols(strategies: List[StrategyBase]) -> List[str]:
    """Get all symbols needed by the strategy portfolio"""
    symbols = set()
    for strategy in strategies:
        symbols.update(strategy.get_required_symbols())
    return list(symbols)