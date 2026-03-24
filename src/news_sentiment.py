#!/usr/bin/env python3
"""
NEWS SENTIMENT ENGINE
Scans free news/social sources, classifies events using keyword matching,
and generates trading signals for crypto futures bot.

NO paid APIs - everything free!
NO LLM calls - keyword matching only!
Rate limited, cached, non-blocking.
"""

import time
import json
import hashlib
import requests
import feedparser
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import xml.etree.ElementTree as ET
from loguru import logger

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "sentiment_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Rate limiting: max 1 request per source per 5 minutes
REQUEST_CACHE_DURATION = 300  # 5 minutes
HEADLINE_CACHE_DURATION = 300  # 5 minutes
MAX_HEADLINES_MEMORY = 100    # Keep last 100 headlines for trend detection

# Sentiment keyword dictionaries
VERY_BULLISH = {
    'etf approved', 'etf approval', 'sec approves', 'bitcoin reserve',
    'strategic reserve', 'legal tender', 'institutional adoption',
    'trillion', 'all-time high', 'ath', 'bull run', 'moon',
    'halving', 'supply shock', 'blackrock', 'fidelity', 'nation adopts',
    'bank adoption', 'payment integration', 'amazon accepts', 'apple accepts',
    'rate cut', 'fed pivot', 'dovish', 'quantitative easing',
    'bitcoin standard', 'hyperbitcoinization', 'orange pill', 'stack sats',
    'laser eyes', 'diamond hands', 'hodl', 'to the moon', 'wen lambo',
    'number go up', 'this is gentlemen', 'we are still early',
    'mainstream adoption', 'mass adoption', 'network effect',
    'store of value', 'digital gold', 'sound money', 'hard money',
    'monetary policy', 'currency debasement', 'inflation hedge',
}

BULLISH = {
    'partnership', 'upgrade', 'launch', 'mainnet', 'rally', 'surge',
    'pump', 'breakout', 'accumulation', 'whale buy', 'inflow',
    'adoption', 'bullish', 'green', 'recovery', 'rebounds',
    'staking', 'yield', 'tvl increase', 'new high', 'breakthrough',
    'innovation', 'development', 'progress', 'milestone', 'achievement',
    'integration', 'collaboration', 'endorsement', 'support',
    'investment', 'funding', 'capital', 'expansion', 'growth',
    'optimism', 'confidence', 'strength', 'momentum', 'catalyst',
    'buying opportunity', 'undervalued', 'oversold', 'bounce',
    'reversal', 'bottom', 'support level', 'demand zone',
}

VERY_BEARISH = {
    'hack', 'hacked', 'exploit', 'drained', 'stolen', 'rug pull',
    'sec sues', 'lawsuit', 'ban', 'banned', 'crackdown', 'fraud',
    'ponzi', 'collapse', 'bankrupt', 'insolvency', 'death spiral',
    'tether depegs', 'usdt depeg', 'usdc depeg', 'bank run',
    'rate hike', 'hawkish', 'quantitative tightening',
    'exchange shutdown', 'withdrawal frozen', 'frozen withdrawals',
    'terror', 'war', 'sanction', 'emergency', 'crisis',
    'panic selling', 'capitulation', 'apocalypse', 'armageddon',
    'bubble burst', 'market crash', 'black swan', 'extinction',
    'death cross', 'bear market', 'crypto winter', 'rekt',
    'liquidation cascade', 'margin call', 'forced selling',
    'contagion', 'systemic risk', 'existential threat',
}

BEARISH = {
    'dump', 'crash', 'sell-off', 'selloff', 'plunge', 'drop',
    'bearish', 'fear', 'panic', 'liquidation', 'outflow',
    'whale sell', 'decline', 'correction', 'regulation', 'restrict',
    'vulnerability', 'bug', 'delay', 'postpone', 'concerns',
    'uncertainty', 'doubt', 'skepticism', 'pessimism', 'weakness',
    'resistance', 'rejection', 'failure', 'disappointment',
    'selling pressure', 'profit taking', 'distribution',
    'overbought', 'overvalued', 'bubble', 'speculation',
    'risk-off', 'flight to safety', 'risk aversion',
}

# Coin-specific keywords mapping to trading pairs
COIN_KEYWORDS = {
    # Original 16 pairs
    'NEARUSD': ['near protocol', 'near', 'aurora', 'nightshade'],
    'UNIUSD': ['uniswap', 'uni', 'defi', 'dex', 'amm', 'liquidity pool'],
    'AVAXUSD': ['avalanche', 'avax', 'subnet', 'ava labs'],
    'LINKUSD': ['chainlink', 'link', 'oracle', 'sergey nazarov'],
    'AAVEUSD': ['aave', 'lending', 'borrowing', 'defi protocol'],
    'SOLUSD': ['solana', 'sol', 'phantom', 'magic eden', 'anatoly'],
    'ETHUSD': ['ethereum', 'eth', 'vitalik', 'erc-20', 'layer 2', 'l2', 'gas fees', 'merge'],
    'XBTUSD': ['bitcoin', 'btc', 'satoshi', 'lightning network', 'taproot', 'segwit'],
    'DOTUSD': ['polkadot', 'dot', 'parachain', 'gavin wood', 'kusama'],
    'XLMUSD': ['stellar', 'xlm', 'stellar development foundation'],
    'XRPUSD': ['ripple', 'xrp', 'sec vs ripple', 'garlinghouse', 'cross border'],
    'ADAUSD': ['cardano', 'ada', 'charles hoskinson', 'plutus', 'catalyst'],
    'ATOMUSD': ['cosmos', 'atom', 'ibc', 'tendermint', 'interchain'],
    'DOGEUSD': ['dogecoin', 'doge', 'elon', 'musk', 'shibes', 'much wow'],
    'FILUSD': ['filecoin', 'fil', 'ipfs', 'decentralized storage'],
    'LTCUSD': ['litecoin', 'ltc', 'silver to bitcoin gold', 'charlie lee'],
    
    # New 24 pairs
    'SUIUSD': ['sui', 'move language', 'mysten labs'],
    'PEPEUSD': ['pepe', 'meme coin', 'frog'],
    'SHIBUSD': ['shiba inu', 'shib', 'shiba', 'bone', 'leash'],
    'BNBUSD': ['binance coin', 'bnb', 'binance', 'bsc', 'binance smart chain'],
    'TRXUSD': ['tron', 'trx', 'justin sun', 'tron network'],
    'HBARUSD': ['hedera', 'hbar', 'hashgraph', 'hedera hashgraph'],
    'HYPEUSD': ['hyperliquid', 'hype', 'perp dex'],
    'TAOUSD': ['bittensor', 'tao', 'ai', 'artificial intelligence'],
    'OKBUSD': ['okb', 'okx', 'ok group'],
    'INJUSD': ['injective', 'inj', 'cosmos ecosystem'],
    'ARBUSD': ['arbitrum', 'arb', 'layer 2', 'optimistic rollup'],
    'OPUSD': ['optimism', 'op', 'layer 2', 'optimistic rollup'],
    'APTUSD': ['aptos', 'apt', 'move language', 'meta'],
    'TIAUSD': ['celestia', 'tia', 'modular blockchain', 'data availability'],
    'ONDOUSD': ['ondo', 'real world assets', 'rwa', 'tokenization'],
    'RENDERUSD': ['render', 'rndr', 'gpu rendering', 'otoy'],
    'JUPUSD': ['jupiter', 'jup', 'solana dex', 'aggregator'],
    'ICPUSD': ['internet computer', 'icp', 'dfinity', 'canister'],
    'LDOUSD': ['lido', 'ldo', 'liquid staking', 'steth'],
    'BCHUSD': ['bitcoin cash', 'bch', 'big blocks', 'roger ver'],
    'STXUSD': ['stacks', 'stx', 'bitcoin layer 2', 'smart contracts'],
    'KAVAUSD': ['kava', 'cosmos ecosystem', 'defi'],
    'ENAUSD': ['ena', 'ethena', 'synthetic dollar'],
    'FLOKIUSD': ['floki', 'viking', 'meme coin'],
}

# Sources configuration
REDDIT_SOURCES = [
    'https://www.reddit.com/r/cryptocurrency/hot.json?limit=25',
    'https://www.reddit.com/r/bitcoin/hot.json?limit=25',
    'https://www.reddit.com/r/ethereum/hot.json?limit=25',
    'https://www.reddit.com/r/altcoin/hot.json?limit=25',
]

RSS_SOURCES = {
    'cointelegraph': 'https://cointelegraph.com/rss',
    'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'bitcoinmagazine': 'https://bitcoinmagazine.com/feed',
}

API_SOURCES = {
    'coingecko_trending': 'https://api.coingecko.com/api/v3/search/trending',
}


class NewsSentimentEngine:
    """
    News sentiment analysis engine using keyword matching.
    Scans free sources, classifies headlines, generates trading signals.
    """
    
    def __init__(self):
        self.request_cache = {}  # Source URL -> last request time
        self.headline_cache = {}  # Headline hash -> (score, coins, timestamp)
        self.recent_headlines = []  # Last 100 headlines for trend analysis
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoBot/1.0 (Educational Research)',
            'Accept': 'application/json, application/xml, text/html',
            'Accept-Encoding': 'gzip, deflate',
        })
        
        # Load cached state
        self._load_cache()
        
    def _load_cache(self):
        """Load cached headlines and request history."""
        cache_file = CACHE_DIR / "sentiment_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.headline_cache = data.get('headline_cache', {})
                    self.recent_headlines = data.get('recent_headlines', [])
                    # Clean old cache entries
                    self._clean_old_cache()
            except Exception as e:
                logger.warning(f"Failed to load sentiment cache: {e}")
                
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = CACHE_DIR / "sentiment_cache.json"
        try:
            data = {
                'headline_cache': self.headline_cache,
                'recent_headlines': self.recent_headlines[-MAX_HEADLINES_MEMORY:],
                'updated': time.time(),
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save sentiment cache: {e}")
            
    def _clean_old_cache(self):
        """Remove expired cache entries."""
        now = time.time()
        # Clean headline cache
        expired_keys = [
            k for k, v in self.headline_cache.items()
            if now - v.get('timestamp', 0) > HEADLINE_CACHE_DURATION
        ]
        for key in expired_keys:
            del self.headline_cache[key]
            
        # Clean recent headlines
        self.recent_headlines = [
            h for h in self.recent_headlines
            if now - h.get('timestamp', 0) < HEADLINE_CACHE_DURATION * 2
        ]
        
    def _can_request(self, source: str) -> bool:
        """Check if we can make a request to this source (rate limiting)."""
        now = time.time()
        last_request = self.request_cache.get(source, 0)
        return now - last_request >= REQUEST_CACHE_DURATION
        
    def _mark_request(self, source: str):
        """Mark that we made a request to this source."""
        self.request_cache[source] = time.time()
        
    def _hash_headline(self, title: str, description: str = '') -> str:
        """Create hash for headline deduplication."""
        text = f"{title}|{description}".lower().strip()
        return hashlib.md5(text.encode()).hexdigest()
        
    def _fetch_reddit(self, url: str) -> List[Dict]:
        """Fetch Reddit posts from JSON API."""
        if not self._can_request(url):
            return []
            
        try:
            self._mark_request(url)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            posts = []
            
            for post in data['data']['children']:
                post_data = post['data']
                posts.append({
                    'title': post_data.get('title', ''),
                    'description': post_data.get('selftext', '')[:500],  # First 500 chars
                    'score': post_data.get('score', 0),
                    'upvotes': post_data.get('ups', 0),
                    'comments': post_data.get('num_comments', 0),
                    'created': post_data.get('created_utc', 0),
                    'url': post_data.get('url', ''),
                    'subreddit': post_data.get('subreddit', ''),
                    'source': f"reddit_{post_data.get('subreddit', 'unknown')}",
                })
                
            logger.info(f"Fetched {len(posts)} Reddit posts from {url}")
            return posts
            
        except Exception as e:
            logger.warning(f"Failed to fetch Reddit {url}: {e}")
            return []
            
    def _fetch_rss(self, name: str, url: str) -> List[Dict]:
        """Fetch RSS feed."""
        if not self._can_request(url):
            return []
            
        try:
            self._mark_request(url)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            posts = []
            
            for entry in feed.entries[:25]:  # Limit to 25 items
                posts.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', entry.get('description', ''))[:500],
                    'published': entry.get('published_parsed', None),
                    'link': entry.get('link', ''),
                    'source': name,
                })
                
            logger.info(f"Fetched {len(posts)} RSS items from {name}")
            return posts
            
        except Exception as e:
            logger.warning(f"Failed to fetch RSS {name} ({url}): {e}")
            return []
            
    def _fetch_coingecko_trending(self) -> List[Dict]:
        """Fetch CoinGecko trending coins."""
        url = API_SOURCES['coingecko_trending']
        if not self._can_request(url):
            return []
            
        try:
            self._mark_request(url)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            posts = []
            
            # Trending coins generate positive sentiment
            for coin in data.get('coins', [])[:10]:  # Top 10 trending
                coin_data = coin.get('item', {})
                name = coin_data.get('name', '')
                symbol = coin_data.get('symbol', '')
                
                posts.append({
                    'title': f"{name} ({symbol}) is trending on CoinGecko",
                    'description': f"Trending rank #{coin_data.get('market_cap_rank', 'N/A')}",
                    'source': 'coingecko_trending',
                    'trending_rank': coin.get('score', 0),
                    'symbol': symbol.lower(),
                    'name': name.lower(),
                })
                
            logger.info(f"Fetched {len(posts)} trending coins from CoinGecko")
            return posts
            
        except Exception as e:
            logger.warning(f"Failed to fetch CoinGecko trending: {e}")
            return []
            
    def score_headline(self, title: str, description: str = '', metadata: Dict = None) -> Tuple[float, List[str]]:
        """
        Score a headline using keyword matching.
        Returns: (sentiment_score, [coins_mentioned])
        """
        text = (title + ' ' + description).lower().strip()
        if not text:
            return 0.0, []
            
        # Check cache first
        text_hash = self._hash_headline(title, description)
        if text_hash in self.headline_cache:
            cached = self.headline_cache[text_hash]
            return cached['score'], cached['coins']
            
        score = 0
        coins_mentioned = []
        
        # Sentiment scoring
        for phrase in VERY_BULLISH:
            if phrase in text:
                score += 3
                
        for phrase in BULLISH:
            if phrase in text:
                score += 1
                
        for phrase in VERY_BEARISH:
            if phrase in text:
                score -= 3
                
        for phrase in BEARISH:
            if phrase in text:
                score -= 1
                
        # Detect mentioned coins
        for pair, keywords in COIN_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                coins_mentioned.append(pair)
                
        # Apply metadata boosts
        if metadata:
            # Reddit: boost by upvotes
            upvotes = metadata.get('upvotes', 0)
            if upvotes > 500:
                score *= 2.0
            elif upvotes > 100:
                score *= 1.5
                
            # CoinGecko trending: automatic bullish sentiment
            if metadata.get('source') == 'coingecko_trending':
                score += 1  # Trending = mild bullish
                symbol = metadata.get('symbol', '')
                name = metadata.get('name', '')
                
                # Find matching pairs for trending coins
                for pair, keywords in COIN_KEYWORDS.items():
                    if symbol in keywords or name in keywords:
                        if pair not in coins_mentioned:
                            coins_mentioned.append(pair)
                            
        # Cache the result
        self.headline_cache[text_hash] = {
            'score': score,
            'coins': coins_mentioned,
            'timestamp': time.time(),
        }
        
        return score, coins_mentioned
        
    def fetch_all_sources(self) -> List[Dict]:
        """Fetch from all news sources."""
        all_posts = []
        
        # Reddit sources
        for reddit_url in REDDIT_SOURCES:
            posts = self._fetch_reddit(reddit_url)
            all_posts.extend(posts)
            time.sleep(0.5)  # Rate limiting
            
        # RSS sources  
        for name, rss_url in RSS_SOURCES.items():
            posts = self._fetch_rss(name, rss_url)
            all_posts.extend(posts)
            time.sleep(0.5)  # Rate limiting
            
        # CoinGecko trending
        trending = self._fetch_coingecko_trending()
        all_posts.extend(trending)
        
        logger.info(f"Fetched {len(all_posts)} total posts from all sources")
        return all_posts
        
    def get_sentiment_signals(self) -> Dict:
        """
        Get current market sentiment signals.
        Returns comprehensive sentiment data for trading decisions.
        """
        try:
            # Fetch fresh data
            posts = self.fetch_all_sources()
            
            # Process all headlines
            scored_headlines = []
            market_scores = []
            coin_scores = {}
            breaking_events = []
            
            for post in posts:
                title = post.get('title', '')
                description = post.get('description', '')
                
                if not title:
                    continue
                    
                score, coins = self.score_headline(title, description, post)
                
                if score == 0 and not coins:
                    continue  # Skip neutral, irrelevant posts
                    
                headline_data = {
                    'headline': title,
                    'score': score,
                    'coins': coins,
                    'source': post.get('source', 'unknown'),
                    'timestamp': time.time(),
                }
                
                scored_headlines.append(headline_data)
                market_scores.append(score)
                
                # Track coin-specific sentiment
                for coin in coins:
                    if coin not in coin_scores:
                        coin_scores[coin] = []
                    coin_scores[coin].append(score)
                    
                # Identify breaking events (very strong sentiment)
                if abs(score) >= 3:
                    breaking_events.append(headline_data)
                    
            # Update recent headlines for trend analysis
            self.recent_headlines.extend(scored_headlines)
            self.recent_headlines = self.recent_headlines[-MAX_HEADLINES_MEMORY:]
            
            # Calculate aggregated sentiment
            market_sentiment = sum(market_scores) / len(market_scores) if market_scores else 0
            
            # Normalize market sentiment to -10 to +10 range
            market_sentiment = max(-10, min(10, market_sentiment))
            
            # Calculate per-coin sentiment
            coin_sentiment = {}
            for coin, scores in coin_scores.items():
                avg_score = sum(scores) / len(scores)
                coin_sentiment[coin] = max(-10, min(10, avg_score))
                
            # Get trending coins from CoinGecko
            trending_coins = []
            for post in posts:
                if post.get('source') == 'coingecko_trending':
                    # Find matching pairs
                    symbol = post.get('symbol', '').lower()
                    name = post.get('name', '').lower()
                    
                    for pair, keywords in COIN_KEYWORDS.items():
                        if symbol in keywords or name in keywords:
                            trending_coins.append(pair)
                            break
                            
            # Sort breaking events by absolute score
            breaking_events.sort(key=lambda x: abs(x['score']), reverse=True)
            
            result = {
                'market_sentiment': market_sentiment,
                'coin_sentiment': coin_sentiment,
                'breaking_events': breaking_events[:10],  # Top 10
                'trending_coins': list(set(trending_coins))[:10],  # Dedupe and limit
                'total_headlines': len(scored_headlines),
                'sources_active': len(set(post.get('source', 'unknown') for post in posts)),
                'last_updated': time.time(),
            }
            
            # Save cache
            self._save_cache()
            
            logger.info(
                f"Sentiment analysis complete: "
                f"Market={market_sentiment:.1f}, "
                f"Coins={len(coin_sentiment)}, "
                f"Breaking={len(breaking_events)}, "
                f"Headlines={len(scored_headlines)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'market_sentiment': 0.0,
                'coin_sentiment': {},
                'breaking_events': [],
                'trending_coins': [],
                'total_headlines': 0,
                'sources_active': 0,
                'last_updated': time.time(),
                'error': str(e),
            }


if __name__ == "__main__":
    # Test the sentiment engine
    engine = NewsSentimentEngine()
    
    print("Testing News Sentiment Engine...")
    print("=" * 50)
    
    # Test individual headlines
    test_headlines = [
        ("Bitcoin ETF Approved by SEC", "Major milestone for crypto adoption"),
        ("Crypto exchange hacked, $100M stolen", "Another security breach hits the industry"),
        ("Ethereum 2.0 upgrade launches successfully", "Network ready for mass adoption"),
        ("Fed raises rates by 0.75%", "Market uncertainty as policy tightens"),
        ("Elon Musk tweets about Dogecoin", "Crypto community reacts to latest endorsement"),
    ]
    
    print("Individual headline scoring:")
    for title, desc in test_headlines:
        score, coins = engine.score_headline(title, desc)
        print(f"  '{title}' -> Score: {score:+.1f}, Coins: {coins}")
        
    print("\nFetching live sentiment data...")
    signals = engine.get_sentiment_signals()
    
    print(f"\nResults:")
    print(f"  Market Sentiment: {signals['market_sentiment']:+.1f}")
    print(f"  Total Headlines: {signals['total_headlines']}")
    print(f"  Active Sources: {signals['sources_active']}")
    print(f"  Breaking Events: {len(signals['breaking_events'])}")
    print(f"  Trending Coins: {len(signals['trending_coins'])}")
    
    if signals['breaking_events']:
        print("\nTop Breaking Events:")
        for event in signals['breaking_events'][:3]:
            print(f"  {event['score']:+.1f}: {event['headline'][:60]}...")
            print(f"    Coins: {event['coins']}, Source: {event['source']}")
            
    if signals['coin_sentiment']:
        print("\nCoin-Specific Sentiment:")
        for coin, sentiment in list(signals['coin_sentiment'].items())[:5]:
            print(f"  {coin}: {sentiment:+.1f}")
            
    print("\nSentiment engine test complete!")