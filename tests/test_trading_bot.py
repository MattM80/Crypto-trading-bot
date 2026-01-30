"""
Unit tests for trading bot components.
"""
import unittest
from datetime import datetime
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies import GridTradingStrategy, MeanReversionStrategy
from risk_manager import RiskManager, Position

class TestGridTradingStrategy(unittest.TestCase):
    """Test grid trading strategy"""
    
    def setUp(self):
        self.strategy = GridTradingStrategy(
            grid_levels=10,
            range_percent=0.05,
            stop_loss_percent=0.02,
            take_profit_percent=0.05
        )
    
    def test_signal_generation(self):
        """Test that signals are generated correctly"""
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 100.5, 101, 100.2, 100.8]
        })
        
        signals = self.strategy.generate_signals(data, 'BTCUSDT')
        
        # Should generate both buy and sell signals
        self.assertGreater(len(signals), 0)
        
        # Signals should have action "BUY" or "SELL"
        for signal in signals:
            self.assertIn(signal.action, ["BUY", "SELL"])
            self.assertGreater(signal.confidence, 0)
            self.assertLess(signal.confidence, 1)

class TestMeanReversionStrategy(unittest.TestCase):
    """Test mean reversion strategy"""
    
    def setUp(self):
        self.strategy = MeanReversionStrategy(
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            bb_period=20,
            bb_std=2.0
        )
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100, 101])
        rsi = self.strategy._calculate_rsi(prices)
        
        # RSI should be between 0 and 100
        self.assertTrue(rsi.dropna().between(0, 100).all())
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        prices = pd.Series([100 + i*0.5 for i in range(50)])
        upper, middle, lower = self.strategy._calculate_bollinger_bands(prices)
        
        # Upper band should be higher than lower band
        self.assertTrue((upper > lower).all())
        
        # Middle band should be between upper and lower
        self.assertTrue((middle >= lower).all() and (middle <= upper).all())

class TestRiskManager(unittest.TestCase):
    """Test risk manager"""
    
    def setUp(self):
        self.rm = RiskManager(
            initial_balance=10000,
            max_position_size=0.02,
            max_drawdown=0.10,
            max_open_positions=3
        )
    
    def test_position_sizing(self):
        """Test position size calculation"""
        position_size = self.rm.calculate_position_size(
            entry_price=100,
            stop_loss_price=98
        )
        
        # Position size should be positive
        self.assertGreater(position_size, 0)
    
    def test_can_open_position(self):
        """Test position opening validation"""
        can_trade, reason = self.rm.can_open_position('BTCUSDT')
        self.assertTrue(can_trade)
    
    def test_portfolio_stats(self):
        """Test portfolio statistics"""
        stats = self.rm.get_portfolio_stats()
        
        self.assertEqual(stats['total_trades'], 0)
        self.assertEqual(stats['balance'], 10000)
        self.assertEqual(stats['total_pnl'], 0)
    
    def test_trade_recording(self):
        """Test recording and closing positions"""
        position = Position(
            symbol='BTCUSDT',
            entry_price=100,
            quantity=1,
            stop_loss=98,
            take_profit=105,
            entry_time=datetime.now().isoformat(),
            side='BUY'
        )
        
        self.rm.record_position(position)
        self.assertEqual(len(self.rm.positions['BTCUSDT']), 1)
        
        # Close position
        trade = self.rm.close_position('BTCUSDT', 102, 'Test Close')
        self.assertIsNotNone(trade)
        self.assertEqual(trade['pnl'], 2)  # 102 - 100

class TestDrawdownCalculation(unittest.TestCase):
    """Test drawdown calculation"""
    
    def setUp(self):
        self.rm = RiskManager(initial_balance=10000)
    
    def test_drawdown_calculation(self):
        """Test that drawdown is calculated correctly"""
        # Initial drawdown should be 0
        self.assertEqual(self.rm.calculate_drawdown(), 0)
        
        # Simulate a loss
        self.rm.current_balance = 9000
        drawdown = self.rm.calculate_drawdown()
        self.assertGreater(drawdown, 0)

if __name__ == '__main__':
    unittest.main()
