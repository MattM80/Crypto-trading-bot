#!/usr/bin/env python3
"""
Backtest All Strategies V2
==========================

Downloads real Kraken OHLCV data and backtests all V2 strategies with brutal honesty.
Tests on multiple symbols and timeframes to find what actually works.

No curve-fitting. No optimistic assumptions. Just facts.

Usage:
    python3 backtest_all.py                    # Default: 90 days, all symbols
    python3 backtest_all.py --days 180         # 6 months of data
    python3 backtest_all.py --timeframe 1h     # 1-hour bars
    python3 backtest_all.py --symbols XBTUSD   # Single symbol only
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategies_v2 import create_strategy_portfolio, get_all_required_symbols
from backtester_v2 import BacktesterV2


KRAKEN_BASE = "https://api.kraken.com/0"


def download_kraken_ohlc(pair: str, interval_minutes: int, days: int) -> pd.DataFrame:
    """Download OHLCV data from Kraken."""
    import time as _time
    from datetime import datetime, timezone, timedelta
    
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - (days * 86400)
    
    all_candles = []
    since = start_ts
    
    logger.info(f"Downloading {pair} data: {days} days, {interval_minutes}m intervals")
    
    while since < end_ts:
        resp = requests.get(
            f"{KRAKEN_BASE}/public/OHLC",
            params={"pair": pair, "interval": interval_minutes, "since": since},
            timeout=30
        )
        
        if resp.status_code != 200:
            logger.error(f"HTTP error {resp.status_code} for {pair}")
            break
            
        try:
            result = resp.json().get("result", {})
        except Exception as e:
            logger.error(f"JSON parse error for {pair}: {e}")
            break
        
        candles = None
        for key, val in result.items():
            if isinstance(val, list) and key != 'last':
                candles = val
                break
        
        if not candles:
            logger.warning(f"No candles data for {pair}, breaking")
            break
            
        for c in candles:
            ts = int(c[0])
            if ts > since and ts <= end_ts:
                all_candles.append({
                    'timestamp': datetime.fromtimestamp(ts, tz=timezone.utc),
                    'open': float(c[1]),
                    'high': float(c[2]),
                    'low': float(c[3]),
                    'close': float(c[4]),
                    'volume': float(c[6]),
                    'time': ts
                })
        
        if candles:
            last_ts = int(candles[-1][0])
            if last_ts <= since:
                logger.warning(f"No progress in pagination for {pair}, breaking")
                break
            since = last_ts
        else:
            break
            
        _time.sleep(1.5)  # Rate limit
        logger.debug(f"Downloaded {len(all_candles)} bars so far for {pair}")
    
    df = pd.DataFrame(all_candles)
    if not df.empty:
        df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
        
    logger.info(f"✓ {pair}: Downloaded {len(df)} bars")
    return df


def download_all_data(symbols: List[str], timeframe: str = "1h", days: int = 90) -> Dict[str, pd.DataFrame]:
    """Download historical data for all required symbols"""
    
    # Map timeframes to Kraken intervals (in minutes)
    interval_map = {"1h": 60, "15m": 15, "5m": 5, "1m": 1, "4h": 240, "1d": 1440}
    
    if timeframe not in interval_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
        
    interval_minutes = interval_map[timeframe]
    
    logger.info(f"Downloading {len(symbols)} symbols for {days} days ({timeframe} bars)")
    
    historical_data = {}
    
    for symbol in symbols:
        logger.info(f"Downloading {symbol}...")
        
        try:
            df = download_kraken_ohlc(symbol, interval_minutes, days)
            
            if len(df) > 0:
                historical_data[symbol] = df
                logger.info(f"✓ {symbol}: {len(df)} bars ready")
            else:
                logger.error(f"✗ {symbol}: No data downloaded")
                
        except Exception as e:
            logger.error(f"✗ {symbol}: Download failed - {e}")
            
        # Rate limiting - be nice to Kraken
        time.sleep(2.0)
        
    logger.info(f"Data download complete: {len(historical_data)} symbols ready")
    return historical_data


def run_individual_strategy_backtests(historical_data: Dict[str, pd.DataFrame], 
                                    initial_balance: float = 300) -> Dict:
    """Run backtests for each strategy individually"""
    
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL STRATEGY BACKTESTS")
    logger.info("="*60)
    
    strategies = create_strategy_portfolio()
    individual_results = {}
    
    for strategy in strategies:
        logger.info(f"\nTesting {strategy.name} individually...")
        
        # Create backtester
        backtester = BacktesterV2(
            initial_balance=initial_balance,
            maker_fee_pct=0.16,
            slippage_pct=0.05
        )
        
        try:
            # Filter data to only symbols this strategy needs
            strategy_symbols = strategy.get_required_symbols()
            strategy_data = {symbol: df for symbol, df in historical_data.items() 
                           if symbol in strategy_symbols}
            
            if not strategy_data:
                logger.warning(f"No data available for {strategy.name} symbols: {strategy_symbols}")
                individual_results[strategy.name] = None
                continue
                
            logger.info(f"Running {strategy.name} on {list(strategy_data.keys())}")
                
            # Run backtest with just this strategy
            result = backtester.run_backtest(
                historical_data=strategy_data,
                strategies=[strategy]
            )
            
            individual_results[strategy.name] = result
            
            # Log key metrics
            logger.info(f"\n{strategy.name} Results:")
            logger.info(f"  Return: {result.total_return_pct:.2f}%") 
            logger.info(f"  Trades: {result.total_trades}")
            logger.info(f"  Win Rate: {result.win_rate:.1%}")
            logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
            logger.info(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
            logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            
            # Show recent trades for debugging
            if result.trade_history:
                logger.info(f"  Recent trades:")
                for trade in result.trade_history[-3:]:  # Last 3 trades
                    logger.info(f"    {trade['side']} {trade['symbol']} @ ${trade['entry_price']:.2f} "
                               f"-> ${trade['exit_price']:.2f} PnL: ${trade['pnl']:.2f}")
            
            # Verdict
            if result.total_return_pct > 5 and result.win_rate > 0.5 and result.profit_factor > 1.5:
                verdict = "✓ PROFITABLE"
            elif result.total_return_pct > 0:
                verdict = "⚠ MARGINAL"  
            else:
                verdict = "✗ LOSING"
                
            logger.info(f"  Verdict: {verdict}")
            
        except Exception as e:
            logger.error(f"Failed to backtest {strategy.name}: {e}")
            import traceback
            traceback.print_exc()
            individual_results[strategy.name] = None
            
    return individual_results


def run_combined_portfolio_backtest(historical_data: Dict[str, pd.DataFrame],
                                  initial_balance: float = 300):
    """Run backtest with all strategies combined"""
    
    logger.info("\n" + "="*60)
    logger.info("COMBINED PORTFOLIO BACKTEST")
    logger.info("="*60)
    
    strategies = create_strategy_portfolio()
    
    # Create backtester
    backtester = BacktesterV2(
        initial_balance=initial_balance,
        maker_fee_pct=0.16,
        slippage_pct=0.05
    )
    
    try:
        # Run combined backtest
        result = backtester.run_backtest(
            historical_data=historical_data,
            strategies=strategies
        )
        
        logger.info(f"\nCombined Portfolio Results:")
        logger.info(f"  Initial: ${result.initial_balance:.2f}")
        logger.info(f"  Final: ${result.final_balance:.2f}")
        logger.info(f"  Return: {result.total_return_pct:.2f}%")
        logger.info(f"  Trades: {result.total_trades}")
        logger.info(f"  Win Rate: {result.win_rate:.1%}")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        # Strategy breakdown
        if result.strategy_stats:
            logger.info(f"\nStrategy Breakdown:")
            for strategy_name, stats in result.strategy_stats.items():
                logger.info(f"  {strategy_name}:")
                logger.info(f"    Allocation: {stats['allocation_pct']:.1%}")
                logger.info(f"    PnL: ${stats['total_pnl']:.2f}")
                logger.info(f"    Trades: {stats['trade_count']}")
                logger.info(f"    Win Rate: {stats['win_rate']:.1%}")
            
        return result
        
    except Exception as e:
        logger.error(f"Combined backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_specific_backtests():
    """Run the specific backtests requested in the task"""
    
    logger.info("\n" + "="*80)
    logger.info("SPECIFIC BACKTEST REQUESTS")
    logger.info("="*80)
    
    # Download data for all required symbols
    symbols = ['XBTUSD', 'ETHUSD', 'SOLUSD']
    historical_data = download_all_data(symbols, timeframe="1h", days=90)
    
    if not historical_data:
        logger.error("No data available for backtests")
        return
        
    results = {}
    
    # Individual strategy tests
    test_configs = [
        ("LiquidationCascade", "XBTUSD"),
        ("LiquidationCascade", "ETHUSD"), 
        ("LiquidationCascade", "SOLUSD"),
        ("MomentumBreakout", "XBTUSD"),
        ("MomentumBreakout", "SOLUSD"),
        ("VolatilityHarvesting", "XBTUSD"),
        ("CrossPairMeanReversion", "ETH/BTC")
    ]
    
    for strategy_name, symbol in test_configs:
        logger.info(f"\n--- {strategy_name} on {symbol} ---")
        
        try:
            # Create the specific strategy
            strategies = create_strategy_portfolio()
            target_strategy = None
            for s in strategies:
                if s.name == strategy_name:
                    target_strategy = s
                    break
                    
            if not target_strategy:
                logger.error(f"Strategy {strategy_name} not found")
                continue
                
            # Prepare data
            if strategy_name == "CrossPairMeanReversion":
                # Cross-pair needs both ETH and BTC
                strategy_data = {k: v for k, v in historical_data.items() if k in ['ETHUSD', 'XBTUSD']}
            elif symbol in historical_data:
                strategy_data = {symbol: historical_data[symbol]}
            else:
                logger.error(f"No data for {symbol}")
                continue
                
            # Run backtest
            backtester = BacktesterV2(initial_balance=300, maker_fee_pct=0.16, slippage_pct=0.05)
            result = backtester.run_backtest(strategy_data, [target_strategy])
            
            results[f"{strategy_name}_{symbol}"] = result
            
            # Log results
            logger.info(f"Return: {result.total_return_pct:.2f}%")
            logger.info(f"Trades: {result.total_trades}")
            logger.info(f"Win Rate: {result.win_rate:.1%}")
            logger.info(f"Final Balance: ${result.final_balance:.2f}")
            
        except Exception as e:
            logger.error(f"Failed {strategy_name} on {symbol}: {e}")
            
    # Combined test
    logger.info(f"\n--- ALL STRATEGIES COMBINED ---")
    try:
        strategies = create_strategy_portfolio()
        backtester = BacktesterV2(initial_balance=300, maker_fee_pct=0.16, slippage_pct=0.05)
        result = backtester.run_backtest(historical_data, strategies)
        
        results["ALL_COMBINED"] = result
        
        logger.info(f"Return: {result.total_return_pct:.2f}%")
        logger.info(f"Trades: {result.total_trades}")
        logger.info(f"Win Rate: {result.win_rate:.1%}")
        logger.info(f"Final Balance: ${result.final_balance:.2f}")
        
    except Exception as e:
        logger.error(f"Failed combined backtest: {e}")
        
    # Summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY TABLE")
    logger.info("="*80)
    
    print(f"{'Test':<30} {'Return %':<10} {'Trades':<8} {'Win Rate':<10} {'Final $':<10}")
    print("-" * 80)
    
    for test_name, result in results.items():
        if result:
            print(f"{test_name:<30} {result.total_return_pct:>8.2f}% {result.total_trades:>6d} "
                  f"{result.win_rate:>8.1%} ${result.final_balance:>8.2f}")
        else:
            print(f"{test_name:<30} {'FAILED':<10}")


def analyze_results(individual_results: Dict, combined_result) -> None:
    """Analyze and summarize all backtest results"""
    
    logger.info("\n" + "="*60)
    logger.info("RESULTS ANALYSIS")
    logger.info("="*60)
    
    # Individual strategy analysis
    profitable_strategies = []
    marginal_strategies = []
    losing_strategies = []
    
    for name, result in individual_results.items():
        if result is None:
            losing_strategies.append(name)
        elif result.total_return_pct > 5 and result.win_rate > 0.5 and result.profit_factor > 1.5:
            profitable_strategies.append((name, result.total_return_pct))
        elif result.total_return_pct > 0:
            marginal_strategies.append((name, result.total_return_pct))
        else:
            losing_strategies.append(name)
            
    logger.info(f"\nStrategy Classification:")
    logger.info(f"  Profitable: {len(profitable_strategies)} - {[s[0] for s in profitable_strategies]}")
    logger.info(f"  Marginal: {len(marginal_strategies)} - {[s[0] for s in marginal_strategies]}")
    logger.info(f"  Losing: {len(losing_strategies)} - {losing_strategies}")
    
    # Best performing strategy
    if profitable_strategies:
        best_strategy = max(profitable_strategies, key=lambda x: x[1])
        logger.info(f"  Best Performer: {best_strategy[0]} ({best_strategy[1]:.2f}%)")
    else:
        logger.info(f"  No profitable strategies found!")
        
    # Combined vs individual
    if combined_result:
        valid_results = [r for r in individual_results.values() if r is not None]
        if valid_results:
            avg_individual_return = sum(r.total_return_pct for r in valid_results) / len(valid_results)
            
            logger.info(f"\nPortfolio Effect:")
            logger.info(f"  Average Individual Return: {avg_individual_return:.2f}%") 
            logger.info(f"  Combined Portfolio Return: {combined_result.total_return_pct:.2f}%")
            
            if combined_result.total_return_pct > avg_individual_return:
                logger.info(f"  ✓ Portfolio diversification helped (+{combined_result.total_return_pct - avg_individual_return:.2f}%)")
            else:
                logger.info(f"  ⚠ Portfolio diversification hurt ({combined_result.total_return_pct - avg_individual_return:.2f}%)")
            
    # Final verdict
    logger.info(f"\n" + "="*60)
    logger.info("FINAL VERDICT")
    logger.info("="*60)
    
    if len(profitable_strategies) >= 2 and (combined_result and combined_result.total_return_pct > 10):
        logger.info("✓ RECOMMENDATION: Deploy the bot - multiple strategies show edge")
    elif len(profitable_strategies) == 1:
        logger.info("⚠ CAUTION: Only one strategy profitable - consider focusing on it")
    elif len(marginal_strategies) > 0 and len(losing_strategies) == 0:
        logger.info("⚠ MARGINAL: Strategies break even - need optimization or different approach")  
    else:
        logger.info("✗ DO NOT DEPLOY: Strategies lose money - back to drawing board")
        
    # Risk assessment
    if combined_result:
        if combined_result.max_drawdown_pct > 20:
            logger.warning(f"⚠ HIGH RISK: Max drawdown {combined_result.max_drawdown_pct:.1f}% exceeds 20%")
        elif combined_result.max_drawdown_pct > 15:
            logger.warning(f"⚠ MODERATE RISK: Max drawdown {combined_result.max_drawdown_pct:.1f}% near 15% limit")
        else:
            logger.info(f"✓ ACCEPTABLE RISK: Max drawdown {combined_result.max_drawdown_pct:.1f}% within limits")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Backtest All Strategies V2")
    parser.add_argument("--days", type=int, default=90,
                       help="Number of days of data to download (default: 90)")
    parser.add_argument("--timeframe", type=str, default="1h", 
                       choices=["1h", "15m", "5m", "4h", "1d"],
                       help="Timeframe for backtesting (default: 1h)")
    parser.add_argument("--symbols", type=str, nargs="*",
                       help="Symbols to test (default: all required symbols)")
    parser.add_argument("--balance", type=float, default=300,
                       help="Initial balance (default: 300)")
    parser.add_argument("--individual-only", action="store_true",
                       help="Only run individual strategy tests")
    parser.add_argument("--combined-only", action="store_true", 
                       help="Only run combined portfolio test")
    parser.add_argument("--specific", action="store_true",
                       help="Run the specific backtests requested in task")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    try:
        if args.specific:
            run_specific_backtests()
            return
            
        # Determine symbols to test
        if args.symbols:
            symbols = args.symbols
        else:
            strategies = create_strategy_portfolio()
            symbols = get_all_required_symbols(strategies)
            
        logger.info(f"Starting backtest: {args.days} days, {args.timeframe}, {symbols}")
        
        # Download historical data
        historical_data = download_all_data(
            symbols=symbols,
            timeframe=args.timeframe,
            days=args.days
        )
        
        if not historical_data:
            logger.error("No historical data downloaded - aborting backtest")
            return
            
        # Check data quality
        min_bars = min(len(df) for df in historical_data.values())
        if min_bars < 100:
            logger.warning(f"Limited data: only {min_bars} bars per symbol")
            
        # Run backtests
        individual_results = {}
        combined_result = None
        
        if not args.combined_only:
            individual_results = run_individual_strategy_backtests(
                historical_data, args.balance
            )
            
        if not args.individual_only:
            combined_result = run_combined_portfolio_backtest(
                historical_data, args.balance  
            )
            
        # Analyze results
        analyze_results(individual_results, combined_result)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()