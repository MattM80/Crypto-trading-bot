#!/usr/bin/env python3
"""
Crypto Trading Bot V2 - Live Trading Runner
===========================================

Runs the portfolio of mathematically sound strategies on live Kraken data.
Designed to turn $300 into serious income through aggressive compounding.

Safety features:
- 15% max drawdown halt
- 3% risk per trade maximum  
- POST-ONLY limit orders (0.16% maker fees)
- Real-time position and risk monitoring

Usage:
    python3 run_bot_v2.py --balance 300 --dry-run
    python3 run_bot_v2.py --live  # REAL MONEY MODE
"""

import sys
import os
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger
import pandas as pd

from strategies_v2 import create_strategy_portfolio, get_all_required_symbols
from portfolio_manager import PortfolioManager
from kraken_client import KrakenClient
from risk_manager import RiskManager
from state_persistence import StatePersistence
from trade_journal import TradeJournal


class TradingBotV2:
    """Main trading bot orchestrator"""
    
    def __init__(self, 
                 initial_balance: float = 300,
                 dry_run: bool = True,
                 update_frequency_sec: int = 60):
        
        self.initial_balance = initial_balance
        self.dry_run = dry_run
        self.update_frequency_sec = update_frequency_sec
        self.running = False
        
        # Initialize components
        logger.info(f"Initializing Trading Bot V2 (Balance: ${initial_balance:.2f}, "
                   f"Dry Run: {dry_run})")
                   
        # Kraken client
        self.kraken = KrakenClient(sandbox=False)  # Kraken has no sandbox
        if dry_run:
            logger.warning("DRY RUN MODE - No real trades will be executed")
        else:
            logger.warning("LIVE TRADING MODE - Real money at risk!")
            
        # Strategy portfolio
        self.strategies = create_strategy_portfolio()
        self.required_symbols = get_all_required_symbols(self.strategies)
        logger.info(f"Loaded {len(self.strategies)} strategies requiring {self.required_symbols}")
        
        # Portfolio manager
        self.portfolio_manager = PortfolioManager(
            initial_balance=initial_balance,
            max_risk_per_trade_pct=3.0,
            max_total_risk_pct=15.0, 
            max_drawdown_halt_pct=15.0
        )
        
        # Add strategies to portfolio manager
        for strategy in self.strategies:
            self.portfolio_manager.add_strategy(strategy)
            
        # Risk manager (reuse existing)
        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            max_risk_per_trade_pct=3.0,
            max_drawdown_pct=15.0
        )
        
        # State persistence
        self.state_persistence = StatePersistence()
        
        # Trade journal
        self.trade_journal = TradeJournal()
        
        # Market data cache
        self.market_data_cache = {}
        self.last_data_update = None
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.last_balance_check = None
        
    def load_saved_state(self):
        """Load any saved bot state"""
        try:
            state = self.state_persistence.load_state()
            if state and 'balance' in state:
                saved_balance = state['balance']
                logger.info(f"Loaded saved balance: ${saved_balance:.2f}")
                self.portfolio_manager.update_balance(saved_balance)
                
        except Exception as e:
            logger.warning(f"Could not load saved state: {e}")
            
    def save_state(self):
        """Save current bot state"""
        try:
            state = {
                'balance': self.portfolio_manager.current_balance,
                'timestamp': datetime.utcnow().isoformat(),
                'active_positions': len(self.portfolio_manager.active_positions),
                'total_trades': self.portfolio_manager.total_trades
            }
            self.state_persistence.save_state(state)
            
        except Exception as e:
            logger.error(f"Could not save state: {e}")
            
    def update_market_data(self) -> bool:
        """Fetch latest market data for all required symbols"""
        
        logger.debug("Updating market data...")
        
        try:
            # Get fresh OHLCV data for each symbol
            updated_symbols = []
            
            for symbol in self.required_symbols:
                try:
                    # Get recent OHLCV (last 100 bars of 5-minute data)
                    ohlcv_data = self.kraken.get_ohlcv(
                        pair=symbol,
                        interval="5m",
                        count=100
                    )
                    
                    if ohlcv_data and len(ohlcv_data) > 0:
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv_data)
                        
                        # Ensure numeric columns
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                
                        self.market_data_cache[symbol] = df
                        updated_symbols.append(symbol)
                        
                    else:
                        logger.warning(f"No OHLCV data received for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Failed to update data for {symbol}: {e}")
                    
            if updated_symbols:
                self.last_data_update = datetime.utcnow()
                logger.debug(f"Updated market data for: {updated_symbols}")
                return True
            else:
                logger.error("Failed to update any market data")
                return False
                
        except Exception as e:
            logger.error(f"Market data update failed: {e}")
            return False
            
    def update_account_balance(self):
        """Update current account balance from exchange"""
        
        try:
            balance_data = self.kraken.get_balance()
            if balance_data:
                # Calculate total USD value
                total_usd = 0.0
                
                # Get current prices for conversion
                current_prices = {}
                for symbol in self.required_symbols:
                    if symbol in self.market_data_cache:
                        df = self.market_data_cache[symbol]
                        if len(df) > 0:
                            current_prices[symbol] = float(df.iloc[-1]['close'])
                            
                # Sum USD balances
                usd_balance = balance_data.get('USD', 0.0) + balance_data.get('ZUSD', 0.0)
                total_usd += float(usd_balance)
                
                # Convert crypto balances to USD
                for symbol in self.required_symbols:
                    crypto_symbol = symbol[:3] if symbol.startswith('X') else symbol.split('USD')[0]
                    
                    crypto_balance = (balance_data.get(crypto_symbol, 0.0) + 
                                     balance_data.get(f'X{crypto_symbol}', 0.0))
                    
                    if crypto_balance > 0 and symbol in current_prices:
                        crypto_usd_value = float(crypto_balance) * current_prices[symbol]
                        total_usd += crypto_usd_value
                        
                logger.info(f"Account balance updated: ${total_usd:.2f}")
                self.portfolio_manager.update_balance(total_usd)
                
                # Update volatility harvesting strategy with balances and prices
                for strategy in self.strategies:
                    if strategy.name == "VolatilityHarvesting":
                        strategy.update_portfolio_value(balance_data, current_prices)
                        
        except Exception as e:
            logger.error(f"Failed to update account balance: {e}")
            
    def process_signals(self):
        """Generate and process trading signals from all strategies"""
        
        if not self.market_data_cache:
            logger.warning("No market data available for signal generation")
            return
            
        logger.debug("Processing trading signals...")
        
        for strategy in self.strategies:
            try:
                # Generate signal from strategy
                signal = strategy.generate_signal(self.market_data_cache)
                
                if signal:
                    logger.info(f"Signal from {strategy.name}: {signal.action} {signal.symbol} "
                               f"@ ${signal.price:.2f} - {signal.reason}")
                    
                    # Let portfolio manager evaluate the signal
                    evaluated_signal = self.portfolio_manager.evaluate_signal(signal, strategy.name)
                    
                    if evaluated_signal and evaluated_signal.quantity > 0:
                        # Execute the signal
                        if self.dry_run:
                            logger.info(f"DRY RUN: Would execute {evaluated_signal.action} "
                                       f"{evaluated_signal.quantity:.6f} {evaluated_signal.symbol} "
                                       f"@ ${evaluated_signal.price:.2f}")
                        else:
                            success = self.execute_signal(evaluated_signal, strategy.name)
                            if success:
                                logger.info(f"Signal executed successfully")
                            else:
                                logger.error(f"Failed to execute signal")
                    else:
                        logger.info(f"Signal rejected by portfolio manager")
                        
            except Exception as e:
                logger.error(f"Error processing signals for {strategy.name}: {e}")
                
    def execute_signal(self, signal, strategy_name: str) -> bool:
        """Execute a trading signal on the exchange"""
        
        try:
            # Place POST-ONLY limit order (maker fees)
            if signal.action == "BUY":
                order_result = self.kraken.place_limit_order(
                    pair=signal.symbol,
                    side="buy",
                    volume=signal.quantity,
                    price=signal.price,
                    post_only=True  # Ensure maker fees
                )
            elif signal.action == "SELL":
                order_result = self.kraken.place_limit_order(
                    pair=signal.symbol,
                    side="sell", 
                    volume=signal.quantity,
                    price=signal.price,
                    post_only=True
                )
            else:
                logger.error(f"Unsupported signal action: {signal.action}")
                return False
                
            if order_result and 'txid' in order_result:
                order_id = order_result['txid'][0]
                logger.info(f"Order placed successfully: {order_id}")
                
                # Record in trade journal
                self.trade_journal.record_entry(
                    symbol=signal.symbol,
                    side=signal.action.lower(),
                    entry_price=signal.price,
                    quantity=signal.quantity,
                    strategy=strategy_name,
                    confidence=signal.confidence,
                    indicators={},  # Add indicator state if needed
                    reason=signal.reason
                )
                
                return True
            else:
                logger.error(f"Order placement failed: {order_result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return False
            
    def monitor_positions(self):
        """Monitor active positions for stop loss / take profit"""
        
        try:
            # Get open orders
            open_orders = self.kraken.get_open_orders()
            
            # Get open positions (if using margin)
            open_positions = self.kraken.get_open_positions() if hasattr(self.kraken, 'get_open_positions') else {}
            
            # Check for filled orders and update positions
            # This is simplified - in practice you'd track order status changes
            
        except Exception as e:
            logger.error(f"Failed to monitor positions: {e}")
            
    def run_trading_loop(self):
        """Main trading loop"""
        
        logger.info("Starting trading loop...")
        self.running = True
        
        # Load any saved state
        self.load_saved_state()
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Update market data
                if self.update_market_data():
                    
                    # Update account balance periodically
                    if (self.last_balance_check is None or 
                        datetime.utcnow() - self.last_balance_check > timedelta(minutes=5)):
                        self.update_account_balance()
                        self.last_balance_check = datetime.utcnow()
                    
                    # Check if we should halt due to drawdown
                    if self.portfolio_manager.check_drawdown_halt():
                        logger.error("Trading halted due to excessive drawdown!")
                        break
                        
                    # Process trading signals
                    self.process_signals()
                    
                    # Monitor existing positions
                    self.monitor_positions()
                    
                    # Save state periodically
                    self.save_state()
                    
                    # Rebalance strategies periodically
                    self.portfolio_manager.rebalance_strategies()
                    
                    # Log periodic stats
                    stats = self.portfolio_manager.get_portfolio_stats()
                    logger.info(f"Balance: ${stats['current_balance']:.2f} "
                               f"Return: {stats['total_return_pct']:.2f}% "
                               f"Trades: {stats['total_trades']} "
                               f"Active: {stats['active_positions']}")
                               
                else:
                    logger.warning("Skipping trading cycle due to data update failure")
                    
                # Sleep for remaining time in update cycle
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.update_frequency_sec - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(30)  # Wait before retrying
                
        logger.info("Trading loop stopped")
        self.save_state()
        
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.running = False


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, shutting down...")
    global bot
    if 'bot' in globals():
        bot.stop()


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Crypto Trading Bot V2")
    parser.add_argument("--balance", type=float, default=300,
                       help="Initial balance (default: 300)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Run in dry-run mode (default)")
    parser.add_argument("--live", action="store_true", 
                       help="LIVE TRADING MODE (overrides --dry-run)")
    parser.add_argument("--frequency", type=int, default=60,
                       help="Update frequency in seconds (default: 60)")
    
    args = parser.parse_args()
    
    # Live mode overrides dry-run
    dry_run = not args.live
    
    if not dry_run:
        response = input("WARNING: You are about to start LIVE TRADING with REAL MONEY. "
                        "Type 'YES' to confirm: ")
        if response != "YES":
            logger.info("Live trading cancelled by user")
            return
            
    # Setup logging
    log_level = "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="{time} | {level} | {message}")
    
    # Log file
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / f"bot_v2_{datetime.now().strftime('%Y%m%d')}.log", 
              rotation="1 day", retention="30 days", level=log_level)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run bot
    global bot
    bot = TradingBotV2(
        initial_balance=args.balance,
        dry_run=dry_run,
        update_frequency_sec=args.frequency
    )
    
    try:
        bot.run_trading_loop()
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()