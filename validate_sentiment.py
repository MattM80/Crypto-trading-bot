#!/usr/bin/env python3
"""
NEWS SENTIMENT VALIDATION SCRIPT
Tests the sentiment engine against historical Reddit data and price movements.
Validates that bullish sentiment actually correlates with price increases.
"""

import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from news_sentiment import NewsSentimentEngine

# Configuration
VALIDATION_DAYS = 30
MIN_UPVOTES = 100
PRICE_WINDOW_HOURS = 24
REDDIT_API_BASE = "https://www.reddit.com/r/{}/search.json"

# Price data sources (free)
BINANCE_API = "https://api.binance.com/api/v3/klines"

# Coin mapping for price data
BINANCE_SYMBOLS = {
    'XBTUSD': 'BTCUSDT',
    'ETHUSD': 'ETHUSDT', 
    'SOLUSD': 'SOLUSDT',
    'ADAUSD': 'ADAUSDT',
    'XRPUSD': 'XRPUSDT',
    'DOGEUSD': 'DOGEUSDT',
    'AVAXUSD': 'AVAXUSDT',
    'LINKUSD': 'LINKUSDT',
    'UNIUSD': 'UNIUSDT',
    'LTCUSD': 'LTCUSDT',
    'BCHUSD': 'BCHUSDT',
    'DOTUSD': 'DOTUSDT',
    'ATOMUSD': 'ATOMUSDT',
    'FILUSD': 'FILUSDT',
    'AAVEUSD': 'AAVEUSDT',
    'NEARUSD': 'NEARUSDT',
}


class SentimentValidator:
    """Validates sentiment engine against historical data and price movements."""
    
    def __init__(self):
        self.engine = NewsSentimentEngine()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SentimentValidator/1.0 (Educational Research)'
        })
        
        # Results storage
        self.validation_results = {
            'total_posts': 0,
            'scored_posts': 0,
            'price_data_available': 0,
            'bullish_signals': 0,
            'bearish_signals': 0,
            'bullish_correct': 0,
            'bearish_correct': 0,
            'signals_by_score': {},
            'coin_performance': {},
        }
        
    def fetch_reddit_history(self, subreddit: str, days: int = 30) -> List[Dict]:
        """Fetch historical Reddit posts using search API."""
        posts = []
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Search parameters
        params = {
            'sort': 'top',
            'restrict_sr': 'true',
            't': 'month',  # Last month
            'limit': 100,
        }
        
        try:
            url = REDDIT_API_BASE.format(subreddit)
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            for post in data['data']['children']:
                post_data = post['data']
                created = datetime.fromtimestamp(post_data.get('created_utc', 0), timezone.utc)
                
                # Filter by date and upvotes
                if (created >= start_date and 
                    created <= end_date and 
                    post_data.get('ups', 0) >= MIN_UPVOTES):
                    
                    posts.append({
                        'title': post_data.get('title', ''),
                        'selftext': post_data.get('selftext', ''),
                        'score': post_data.get('score', 0),
                        'upvotes': post_data.get('ups', 0),
                        'created_utc': post_data.get('created_utc', 0),
                        'created': created,
                        'subreddit': subreddit,
                        'permalink': post_data.get('permalink', ''),
                    })
                    
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit} with >{MIN_UPVOTES} upvotes")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to fetch r/{subreddit} history: {e}")
            return []
            
    def get_price_data(self, symbol: str, start_time: int, hours: int = 24) -> Tuple[float, float]:
        """
        Get price at start_time and price after hours.
        Returns: (start_price, end_price) or (None, None) if data unavailable.
        """
        binance_symbol = BINANCE_SYMBOLS.get(symbol)
        if not binance_symbol:
            return None, None
            
        try:
            # Get 1-hour klines for the period
            end_time = start_time + (hours * 3600 * 1000)  # Convert to milliseconds
            
            params = {
                'symbol': binance_symbol,
                'interval': '1h',
                'startTime': start_time * 1000,  # Binance uses milliseconds
                'endTime': end_time,
                'limit': hours + 5,  # Buffer
            }
            
            response = self.session.get(BINANCE_API, params=params, timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            
            if len(klines) < 2:
                return None, None
                
            # First candle close price (start)
            start_price = float(klines[0][4])  # Close price
            
            # Last candle close price (end)
            end_price = float(klines[-1][4])
            
            return start_price, end_price
            
        except Exception as e:
            logger.warning(f"Failed to get price data for {symbol}: {e}")
            return None, None
            
    def validate_signal(self, post: Dict) -> Dict:
        """Validate a single post's sentiment against price movement."""
        
        # Score the headline
        title = post['title']
        description = post.get('selftext', '')[:500]  # Limit description
        
        metadata = {
            'upvotes': post['upvotes'],
            'source': f"reddit_{post['subreddit']}",
        }
        
        score, coins = self.engine.score_headline(title, description, metadata)
        
        result = {
            'post': post,
            'sentiment_score': score,
            'coins_mentioned': coins,
            'validations': [],
        }
        
        # Skip if no sentiment or coins
        if score == 0 or not coins:
            return result
            
        # Validate against price movements for each mentioned coin
        post_time = int(post['created_utc'])
        
        for coin in coins:
            start_price, end_price = self.get_price_data(coin, post_time, PRICE_WINDOW_HOURS)
            
            if start_price is None or end_price is None:
                continue
                
            # Calculate price change
            price_change = (end_price - start_price) / start_price
            price_change_pct = price_change * 100
            
            # Determine if sentiment was correct
            bullish_signal = score > 0
            bearish_signal = score < 0
            price_went_up = price_change > 0.01  # >1% threshold
            price_went_down = price_change < -0.01  # <-1% threshold
            
            correct = False
            if bullish_signal and price_went_up:
                correct = True
            elif bearish_signal and price_went_down:
                correct = True
            elif abs(price_change) <= 0.01:  # Neutral price movement
                correct = None  # Don't count neutral movements
                
            validation = {
                'coin': coin,
                'start_price': start_price,
                'end_price': end_price,
                'price_change_pct': price_change_pct,
                'sentiment_score': score,
                'bullish_signal': bullish_signal,
                'bearish_signal': bearish_signal,
                'price_went_up': price_went_up,
                'price_went_down': price_went_down,
                'correct': correct,
            }
            
            result['validations'].append(validation)
            
        return result
        
    def run_validation(self) -> Dict:
        """Run complete validation against historical data."""
        
        print("🔍 Starting News Sentiment Validation")
        print("=" * 60)
        
        # Fetch historical Reddit data
        subreddits = ['cryptocurrency', 'bitcoin', 'ethereum']
        all_posts = []
        
        for subreddit in subreddits:
            print(f"Fetching r/{subreddit} posts...")
            posts = self.fetch_reddit_history(subreddit, VALIDATION_DAYS)
            all_posts.extend(posts)
            time.sleep(2)  # Rate limiting
            
        print(f"Total historical posts: {len(all_posts)}")
        self.validation_results['total_posts'] = len(all_posts)
        
        # Validate each post
        validated_signals = []
        
        for i, post in enumerate(all_posts):
            if i % 10 == 0:
                print(f"Processing post {i+1}/{len(all_posts)}...")
                
            validation = self.validate_signal(post)
            
            if validation['sentiment_score'] != 0 and validation['validations']:
                validated_signals.append(validation)
                self.validation_results['scored_posts'] += 1
                
            time.sleep(0.1)  # Rate limiting
            
        # Analyze results
        return self._analyze_results(validated_signals)
        
    def _analyze_results(self, validated_signals: List[Dict]) -> Dict:
        """Analyze validation results and calculate accuracy metrics."""
        
        all_validations = []
        for signal in validated_signals:
            all_validations.extend(signal['validations'])
            
        # Filter out validations with None (neutral price movements)
        valid_validations = [v for v in all_validations if v['correct'] is not None]
        
        if not valid_validations:
            print("❌ No valid price data available for analysis")
            return self.validation_results
            
        self.validation_results['price_data_available'] = len(valid_validations)
        
        # Calculate accuracy by signal type
        bullish_validations = [v for v in valid_validations if v['bullish_signal']]
        bearish_validations = [v for v in valid_validations if v['bearish_signal']]
        
        self.validation_results['bullish_signals'] = len(bullish_validations)
        self.validation_results['bearish_signals'] = len(bearish_validations)
        
        if bullish_validations:
            bullish_correct = sum(1 for v in bullish_validations if v['correct'])
            self.validation_results['bullish_correct'] = bullish_correct
            self.validation_results['bullish_accuracy'] = bullish_correct / len(bullish_validations)
        else:
            self.validation_results['bullish_accuracy'] = 0
            
        if bearish_validations:
            bearish_correct = sum(1 for v in bearish_validations if v['correct'])
            self.validation_results['bearish_correct'] = bearish_correct 
            self.validation_results['bearish_accuracy'] = bearish_correct / len(bearish_validations)
        else:
            self.validation_results['bearish_accuracy'] = 0
            
        # Overall accuracy
        total_correct = sum(1 for v in valid_validations if v['correct'])
        self.validation_results['overall_accuracy'] = total_correct / len(valid_validations)
        
        # Analyze by sentiment score strength
        score_buckets = {
            'very_bullish': [v for v in valid_validations if v['sentiment_score'] >= 3],
            'bullish': [v for v in valid_validations if 1 <= v['sentiment_score'] < 3],
            'bearish': [v for v in valid_validations if -3 < v['sentiment_score'] <= -1],
            'very_bearish': [v for v in valid_validations if v['sentiment_score'] <= -3],
        }
        
        for bucket_name, validations in score_buckets.items():
            if validations:
                correct = sum(1 for v in validations if v['correct'])
                accuracy = correct / len(validations)
                avg_return = np.mean([v['price_change_pct'] for v in validations])
                
                self.validation_results['signals_by_score'][bucket_name] = {
                    'count': len(validations),
                    'correct': correct,
                    'accuracy': accuracy,
                    'avg_return_pct': avg_return,
                }
                
        # Analyze by coin
        coin_stats = {}
        for validation in valid_validations:
            coin = validation['coin']
            if coin not in coin_stats:
                coin_stats[coin] = []
            coin_stats[coin].append(validation)
            
        for coin, validations in coin_stats.items():
            if len(validations) >= 3:  # Minimum sample size
                correct = sum(1 for v in validations if v['correct'])
                accuracy = correct / len(validations)
                avg_return = np.mean([v['price_change_pct'] for v in validations])
                
                self.validation_results['coin_performance'][coin] = {
                    'count': len(validations),
                    'correct': correct,
                    'accuracy': accuracy,
                    'avg_return_pct': avg_return,
                }
                
        return self.validation_results
        
    def print_results(self):
        """Print validation results in a formatted report."""
        
        results = self.validation_results
        
        print("\n📊 SENTIMENT VALIDATION RESULTS")
        print("=" * 60)
        
        print(f"📈 Data Coverage:")
        print(f"  • Total Reddit posts analyzed: {results['total_posts']:,}")
        print(f"  • Posts with sentiment signals: {results['scored_posts']:,}")
        print(f"  • Signals with price data: {results['price_data_available']:,}")
        
        if results['price_data_available'] == 0:
            print("\n❌ No price data available - cannot validate accuracy")
            return
            
        print(f"\n🎯 Overall Accuracy:")
        print(f"  • Combined accuracy: {results['overall_accuracy']:.1%}")
        
        print(f"\n📊 Signal Breakdown:")
        print(f"  • Bullish signals: {results['bullish_signals']:,} ({results['bullish_accuracy']:.1%} accurate)")
        print(f"  • Bearish signals: {results['bearish_signals']:,} ({results['bearish_accuracy']:.1%} accurate)")
        
        print(f"\n🔥 Performance by Signal Strength:")
        for score_type, stats in results['signals_by_score'].items():
            print(f"  • {score_type.replace('_', ' ').title()}: "
                  f"{stats['count']} signals, {stats['accuracy']:.1%} accurate, "
                  f"{stats['avg_return_pct']:+.1f}% avg return")
                  
        if results['coin_performance']:
            print(f"\n🪙 Top Coin Performance:")
            # Sort by accuracy, then by count
            sorted_coins = sorted(
                results['coin_performance'].items(),
                key=lambda x: (x[1]['accuracy'], x[1]['count']),
                reverse=True
            )
            
            for coin, stats in sorted_coins[:8]:  # Top 8
                print(f"  • {coin}: {stats['count']} signals, "
                      f"{stats['accuracy']:.1%} accurate, {stats['avg_return_pct']:+.1f}% avg return")
                      
        # Conclusion
        overall_acc = results['overall_accuracy']
        if overall_acc > 0.6:
            conclusion = "🎉 EXCELLENT - Sentiment signals show strong predictive power!"
        elif overall_acc > 0.55:
            conclusion = "✅ GOOD - Sentiment signals beat random (50%) by significant margin"
        elif overall_acc > 0.5:
            conclusion = "⚠️ MARGINAL - Slight edge over random, may need refinement"
        else:
            conclusion = "❌ POOR - Sentiment signals underperform, needs major improvements"
            
        print(f"\n🏁 Conclusion: {conclusion}")
        
        # Save detailed results
        output_file = Path(__file__).parent / "data" / "sentiment_validation.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    print("🚀 News Sentiment Validation Starting...")
    print(f"📅 Analyzing last {VALIDATION_DAYS} days")
    print(f"🔍 Minimum {MIN_UPVOTES} upvotes required")
    print(f"⏰ Price window: {PRICE_WINDOW_HOURS} hours")
    
    validator = SentimentValidator()
    
    try:
        validator.run_validation()
        validator.print_results()
        
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"❌ Validation failed: {e}")
        
    print("\n✅ Sentiment validation complete!")