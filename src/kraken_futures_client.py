"""
Kraken Futures API client for the ultimate futures trading bot.
Handles PF_ (perpetual) and PI_ (index) symbols with proper authentication.
"""
import os
import json
import time
import hashlib
import hmac
import base64
import threading
import requests
from pathlib import Path
import urllib.parse
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from project root (.env), regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class KrakenFuturesClient:
    """Kraken Futures exchange client with proper authentication and symbol handling."""
    
    def __init__(self, api_key: str = None, private_key: str = None, dry_run: bool = True):
        """
        Initialize Kraken Futures client.
        
        Args:
            api_key: Kraken Futures API key
            private_key: Kraken Futures private key  
            dry_run: If True, return mock data (safe for testing)
        """
        self.api_key = api_key or os.getenv("KRAKEN_FUTURES_API_KEY", "")
        self.private_key = private_key or os.getenv("KRAKEN_FUTURES_PRIVATE_KEY", "")
        self.dry_run = dry_run or not bool(self.api_key and self.private_key)
        
        # Kraken Futures API endpoints
        self.base_url = "https://futures.kraken.com"
        
        if self.dry_run:
            logger.warning("Running in DRY RUN mode - using mock data")
        else:
            logger.info("Connected to Kraken Futures API (LIVE TRADING)")
        
        self.session = requests.Session()
        
        # Rate limiting: Futures API allows more calls than spot
        self._rate_lock = threading.Lock()
        self._rate_counter: float = 0.0
        self._rate_last_time: float = time.monotonic()
        self._rate_max: float = 25.0  # Higher limit for futures
        self._rate_decay_per_sec: float = 0.5
        
        # Cache for instrument metadata
        self._instruments_cache: Optional[Dict] = None
        self._tickers_cache: Dict = {}
        self._tickers_cache_time = 0
        
        # Symbol mappings: spot -> futures
        self.symbol_map = {
            "XBTUSD": "PF_XBTUSD", "ETHUSD": "PF_ETHUSD", "SOLUSD": "PF_SOLUSD",
            "ADAUSD": "PF_ADAUSD", "AVAXUSD": "PF_AVAXUSD", "ATOMUSD": "PF_ATOMUSD",
            "LINKUSD": "PF_LINKUSD", "DOTUSD": "PF_DOTUSD", "NEARUSD": "PF_NEARUSD",
            "UNIUSD": "PF_UNIUSD", "AAVEUSD": "PF_AAVEUSD", "XLMUSD": "PF_XLMUSD",
            "XRPUSD": "PF_XRPUSD", "DOGEUSD": "PF_DOGEUSD", "FILUSD": "PF_FILUSD",
            "LTCUSD": "PF_LTCUSD", "SUIUSD": "PF_SUIUSD", "PEPEUSD": "PF_PEPEUSD",
            "SHIBUSD": "PF_SHIBUSD", "BNBUSD": "PF_BNBUSD", "TRXUSD": "PF_TRXUSD",
            "HBARUSD": "PF_HBARUSD", "HYPEUSD": "PF_HYPEUSD", "TAOUSD": "PF_TAOUSD",
            "OKBUSD": "PF_OKBUSD", "INJUSD": "PF_INJUSD", "ARBUSD": "PF_ARBUSD",
            "OPUSD": "PF_OPUSD", "APTUSD": "PF_APTUSD", "TIAUSD": "PF_TIAUSD",
            "ONDOUSD": "PF_ONDOUSD", "RENDERUSD": "PF_RENDERUSD", "JUPUSD": "PF_JUPUSD",
            "ICPUSD": "PF_ICPUSD", "LDOUSD": "PF_LDOUSD", "BCHUSD": "PF_BCHUSD",
            "STXUSD": "PF_STXUSD", "KAVAUSD": "PF_KAVAUSD", "ENAUSD": "PF_ENAUSD",
            "FLOKIUSD": "PF_FLOKIUSD"
        }

    def _convert_symbol(self, symbol: str) -> str:
        """Convert spot symbol to futures symbol."""
        return self.symbol_map.get(symbol, f"PF_{symbol}")
    
    def _get_auth_headers(self, endpoint_path: str, post_data: str = "") -> Dict[str, str]:
        """Generate authentication headers for Kraken Futures API."""
        if self.dry_run:
            return {}
            
        nonce = str(int(time.time() * 1000))
        
        # Kraken Futures auth: HMAC-SHA512 of (postData + nonce + endpointPath)
        message = post_data + nonce + endpoint_path
        sha256_hash = hashlib.sha256(message.encode('utf-8')).digest()
        hmac_digest = hmac.new(base64.b64decode(self.private_key), sha256_hash, hashlib.sha512).digest()
        authent = base64.b64encode(hmac_digest).decode('utf-8')
        
        return {
            "APIKey": self.api_key,
            "Nonce": nonce,
            "Authent": authent
        }
    
    def _rate_limit_wait(self, cost: float = 1.0) -> None:
        """Rate limiting using leaky bucket model."""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._rate_last_time
            self._rate_counter = max(0.0, self._rate_counter - elapsed * self._rate_decay_per_sec)
            self._rate_last_time = now
            
            if self._rate_counter + cost > self._rate_max:
                wait = (self._rate_counter + cost - self._rate_max) / self._rate_decay_per_sec
                time.sleep(wait)
                now2 = time.monotonic()
                elapsed2 = now2 - self._rate_last_time
                self._rate_counter = max(0.0, self._rate_counter - elapsed2 * self._rate_decay_per_sec)
                self._rate_last_time = now2
            
            self._rate_counter += cost
    
    def _request(self, endpoint: str, params: dict = None, private: bool = False) -> Optional[Dict]:
        """Make API request to Kraken Futures."""
        if self.dry_run and private:
            return self._get_mock_response(endpoint, params)
        
        if private:
            self._rate_limit_wait()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            
            if private:
                post_data = urllib.parse.urlencode(params or {})
                headers = self._get_auth_headers(f"/{endpoint}", post_data)
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                response = self.session.post(url, headers=headers, data=post_data, timeout=30)
            else:
                response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("result") == "success":
                    return data
                else:
                    logger.error(f"Kraken Futures API error: {data.get('error', 'Unknown error')}")
                    return None
            else:
                logger.error(f"Kraken Futures HTTP {response.status_code}: {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Kraken Futures request failed: {e}")
            return None
    
    def _get_mock_response(self, endpoint: str, params: dict = None) -> Dict:
        """Return mock responses for dry run mode."""
        if "tickers" in endpoint:
            # Mock ticker data
            return {
                "result": "success",
                "tickers": [
                    {
                        "symbol": "PF_XBTUSD",
                        "bid": 50000.0,
                        "ask": 50001.0,
                        "last": 50000.5,
                        "vol24h": 1000000,
                        "fundingRate": -0.00005,
                        "fundingRatePrediction": -0.00003
                    }
                ]
            }
        elif "accounts" in endpoint:
            return {
                "result": "success",
                "accounts": {
                    "cash": {"balances": {"USD": 1000.0}},
                    "flex": {"initialMargin": 50.0, "maintenanceMargin": 25.0}
                }
            }
        elif "openpositions" in endpoint:
            return {"result": "success", "openPositions": []}
        elif "instruments" in endpoint:
            return {
                "result": "success",
                "instruments": [
                    {
                        "symbol": "PF_XBTUSD",
                        "underlying": "XBT",
                        "lastTradingTime": "2030-12-31T23:59:59.000Z",
                        "tickSize": 0.5,
                        "contractSize": 1.0
                    }
                ]
            }
        else:
            return {"result": "success"}
    
    def get_tickers(self) -> Dict[str, Dict]:
        """Get ticker data for all futures symbols."""
        # Cache tickers for 10 seconds to avoid spam
        if time.time() - self._tickers_cache_time < 10 and self._tickers_cache:
            return self._tickers_cache
        
        try:
            result = self._request("derivatives/api/v3/tickers")
            if not result:
                return {}
            
            tickers = {}
            for ticker in result.get("tickers", []):
                symbol = ticker.get("symbol", "")
                if symbol.startswith("PF_"):  # Only perpetual futures
                    tickers[symbol] = {
                        "bid": float(ticker.get("bid", 0)),
                        "ask": float(ticker.get("ask", 0)),
                        "last": float(ticker.get("last", 0)),
                        "vol24h": float(ticker.get("vol24h", 0)),
                        "fundingRate": float(ticker.get("fundingRate", 0)),
                        "fundingRatePrediction": float(ticker.get("fundingRatePrediction", 0))
                    }
            
            self._tickers_cache = tickers
            self._tickers_cache_time = time.time()
            return tickers
        
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            return {}
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get orderbook for a symbol."""
        futures_symbol = self._convert_symbol(symbol)
        
        try:
            result = self._request("derivatives/api/v3/orderbook", {"symbol": futures_symbol})
            if not result:
                return None
            
            orderbook = result.get("orderBook", {})
            return {
                "bids": [[float(bid[0]), float(bid[1])] for bid in orderbook.get("bids", [])],
                "asks": [[float(ask[0]), float(ask[1])] for ask in orderbook.get("asks", [])]
            }
        
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return None
    
    def get_ohlc(self, symbol: str, interval: str = "1h", since: Optional[int] = None) -> pd.DataFrame:
        """
        Get OHLCV data. Try futures candles first, fall back to spot if needed.
        
        Args:
            symbol: Trading pair (spot format like SOLUSD)
            interval: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            since: Unix timestamp to start from
        """
        futures_symbol = self._convert_symbol(symbol)
        
        try:
            # Try Kraken Futures candles endpoint first
            params = {"symbol": futures_symbol, "interval": interval}
            if since:
                params["from"] = since
            
            result = self._request("derivatives/api/v3/candles", params)
            
            if result and "candles" in result:
                candles = result["candles"]
                df = pd.DataFrame([
                    {
                        "timestamp": pd.to_datetime(candle["time"]),
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle["volume"])
                    }
                    for candle in candles
                ])
                return df.set_index("timestamp")
            
            else:
                # Fallback to spot Kraken API for OHLC (prices are nearly identical)
                logger.info(f"Falling back to spot OHLC for {symbol}")
                from kraken_client import KrakenClient
                spot_client = KrakenClient()
                
                # Convert interval to seconds for spot API
                interval_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, 
                               "1h": 3600, "4h": 14400, "1d": 86400}
                interval_seconds = interval_map.get(interval, 3600)
                
                klines = spot_client.get_klines(symbol, interval_seconds, 500)
                if klines:
                    df = pd.DataFrame([
                        {
                            "timestamp": k["timestamp"],
                            "open": k["open"],
                            "high": k["high"],
                            "low": k["low"],
                            "close": k["close"],
                            "volume": k["volume"]
                        }
                        for k in klines
                    ])
                    return df.set_index("timestamp")
        
        except Exception as e:
            logger.error(f"Error getting OHLC for {symbol}: {e}")
        
        # Return empty DataFrame if all fails
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        if self.dry_run:
            return {"USD": 1000.0, "available_margin": 950.0}
        
        try:
            result = self._request("derivatives/api/v3/accounts", private=True)
            if not result:
                return {}
            
            accounts = result.get("accounts", {})
            cash_account = accounts.get("cash", {})
            balances = cash_account.get("balances", {})
            
            flex_account = accounts.get("flex", {})
            available_margin = cash_account.get("auxiliaryBalances", {}).get("usd", 0)
            
            return {
                "USD": float(balances.get("USD", 0)),
                "available_margin": float(available_margin)
            }
        
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        if self.dry_run:
            return []
        
        try:
            result = self._request("derivatives/api/v3/openpositions", private=True)
            if not result:
                return []
            
            positions = []
            for pos in result.get("openPositions", []):
                positions.append({
                    "symbol": pos.get("symbol", ""),
                    "side": "long" if float(pos.get("size", 0)) > 0 else "short",
                    "size": abs(float(pos.get("size", 0))),
                    "price": float(pos.get("price", 0)),
                    "unrealizedPnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": pos.get("maxLeverage", 1)
                })
            
            return positions
        
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "lmt",
        price: Optional[float] = None,
        leverage: Optional[int] = None,
        stop_price: Optional[float] = None
    ) -> Optional[str]:
        """
        Place an order.
        
        Args:
            symbol: Trading pair (spot format)
            side: "buy" or "sell"
            size: Order size
            order_type: "lmt", "mkt", "stp", "take_profit"
            price: Limit price
            leverage: Leverage (1-50)
            stop_price: Stop price for stop orders
        """
        if self.dry_run:
            order_id = f"mock_order_{int(time.time())}"
            logger.info(f"DRY RUN: {side} {size} {symbol} @ {price} (leverage: {leverage}x) -> {order_id}")
            return order_id
        
        futures_symbol = self._convert_symbol(symbol)
        
        try:
            params = {
                "symbol": futures_symbol,
                "side": side.lower(),
                "size": size,
                "orderType": order_type
            }
            
            if price and order_type in ["lmt", "stp", "take_profit"]:
                params["limitPrice"] = price
            
            if stop_price:
                params["stopPrice"] = stop_price
            
            if leverage and leverage > 1:
                params["maxLeverage"] = leverage
            
            result = self._request("derivatives/api/v3/sendorder", params, private=True)
            
            if result and result.get("result") == "success":
                order_id = result.get("sendStatus", {}).get("order_id")
                logger.info(f"Order placed: {order_id}")
                return order_id
            
            return None
        
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.dry_run:
            logger.info(f"DRY RUN: Cancel order {order_id}")
            return True
        
        try:
            params = {"order_id": order_id}
            result = self._request("derivatives/api/v3/cancelorder", params, private=True)
            
            if result and result.get("result") == "success":
                logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a symbol."""
        futures_symbol = self._convert_symbol(symbol)
        tickers = self.get_tickers()
        
        ticker = tickers.get(futures_symbol, {})
        return ticker.get("fundingRate", 0.0)
    
    def get_historical_funding(self, symbol: str, since: Optional[int] = None) -> List[Dict]:
        """Get historical funding rates."""
        futures_symbol = self._convert_symbol(symbol)
        
        try:
            params = {"symbol": futures_symbol}
            if since:
                params["since"] = since
            
            result = self._request("derivatives/api/v3/historicalfundingrates", params)
            
            if not result:
                return []
            
            funding_rates = []
            for rate in result.get("rates", []):
                funding_rates.append({
                    "timestamp": pd.to_datetime(rate.get("timestamp")),
                    "rate": float(rate.get("fundingRate", 0))
                })
            
            return funding_rates
        
        except Exception as e:
            logger.error(f"Error getting historical funding for {symbol}: {e}")
            return []