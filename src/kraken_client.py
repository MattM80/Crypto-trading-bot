"""
Kraken Sandbox API client for the trading bot.
Works globally - no geo-restrictions!
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
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from project root (.env), regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

class KrakenClient:
    """Kraken Sandbox exchange client (global, no restrictions)"""
    
    def __init__(self, api_key: str = None, private_key: str = None, sandbox: bool = True):
        """
        Initialize Kraken client.
        
        Args:
            api_key: Kraken API key
            private_key: Kraken private key
            sandbox: Use sandbox (recommended for testing)
        """
        self.api_key = api_key or os.getenv("KRAKEN_API_KEY", "")
        self.private_key = private_key or os.getenv("KRAKEN_PRIVATE_KEY", "")
        self.sandbox = False  # Kraken spot market has NO sandbox
        
        # Endpoints - Kraken SPOT only has live API
        self.base_url = "https://api.kraken.com"
        logger.warning("Connected to Kraken Live API (REAL MONEY - Use small amounts!)")
        
        self.api_version = "0"
        self.session = requests.Session()

        # Rate limiter: Kraken allows ~15 private API calls per rolling window.
        # We use a simple counter that decays over time.
        self._rate_lock = threading.Lock()
        self._rate_counter: float = 0.0  # weighted call counter
        self._rate_last_time: float = time.monotonic()
        self._rate_max: float = 15.0     # max counter before we sleep
        self._rate_decay_per_sec: float = 0.33  # counter decays ~1 per 3 sec

        # Cache for pair metadata (precision, min order, etc.)
        self._asset_pairs_cache: Optional[Dict] = None
        
        if not self.api_key or not self.private_key:
            logger.warning("No API credentials found. Running in read-only mode.")

    def _get_asset_pair_info(self, symbol: str) -> Optional[Dict]:
        """Fetch and cache AssetPairs metadata; return info for a single pair."""
        try:
            if not symbol:
                return None

            if self._asset_pairs_cache is None:
                self._asset_pairs_cache = self._request("public/AssetPairs", params={}) or {}

            # Kraken returns a dict keyed by pair name(s). Try direct match first.
            if symbol in self._asset_pairs_cache:
                return self._asset_pairs_cache.get(symbol)

            # Fallback: try case-insensitive lookup
            sym_upper = symbol.upper()
            for key, value in self._asset_pairs_cache.items():
                if key.upper() == sym_upper:
                    return value

            # Fallback: match by "altname" (common user-facing symbols like XBTUSD/ETHUSD)
            for value in self._asset_pairs_cache.values():
                try:
                    if str(value.get("altname", "")).upper() == sym_upper:
                        return value
                except Exception:
                    continue

            return None
        except Exception as e:
            logger.error(f"Failed to load AssetPairs metadata: {e}")
            return None

    def _quantize_decimal(self, value: float, decimals: int) -> str:
        """Format numeric values with a maximum number of decimals (truncate)."""
        try:
            d = Decimal(str(value))
            q = Decimal("1") if decimals <= 0 else Decimal("1." + ("0" * decimals))
            return format(d.quantize(q, rounding=ROUND_DOWN), "f")
        except Exception:
            # Fallback to Python formatting
            if decimals <= 0:
                return str(int(value))
            return f"{value:.{decimals}f}"

    def _format_price(self, symbol: str, price: float) -> str:
        info = self._get_asset_pair_info(symbol) or {}
        decimals = int(info.get("pair_decimals", 2))
        return self._quantize_decimal(price, decimals)

    def _format_volume(self, symbol: str, volume: float) -> str:
        info = self._get_asset_pair_info(symbol) or {}
        decimals = int(info.get("lot_decimals", 8))
        return self._quantize_decimal(volume, decimals)

    def get_pair_assets(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (base, quote) asset codes for a pair (e.g. XBTUSD -> (XBT, USD))."""
        info = self._get_asset_pair_info(symbol) or {}

        return self._clean_asset_code(info.get("base")), self._clean_asset_code(info.get("quote"))

    def _clean_asset_code(self, asset: Optional[str]) -> Optional[str]:
        """Normalize Kraken asset codes.

        Examples:
        - XXBT -> XBT
        - XETH -> ETH
        - ZUSD -> USD

        Important: Do NOT strip the leading X from XBT.
        """
        if not asset:
            return asset
        a = str(asset).strip()
        # Strip a single leading X/Z only when Kraken uses a 4+ char prefixed code.
        if len(a) > 3 and a[0] in {"X", "Z"}:
            a = a[1:]
        return a

    def get_min_order_volume(self, symbol: str) -> Optional[float]:
        """Return Kraken's minimum order volume for the pair, if available."""
        try:
            info = self._get_asset_pair_info(symbol) or {}
            ordermin = info.get("ordermin")
            if ordermin is None:
                return None
            return float(ordermin)
        except Exception:
            return None
    
    def _get_kraken_signature(self, urlpath: str, data: dict, nonce: str) -> str:
        """Generate Kraken API signature"""
        postdata = data.copy()
        postdata["nonce"] = nonce

        # Kraken expects SHA256(nonce + urlencoded(postdata))
        postdata_encoded = urllib.parse.urlencode(postdata)
        encoded = (nonce + postdata_encoded).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        return base64.b64encode(
            hmac.new(
                base64.b64decode(self.private_key),
                message,
                hashlib.sha512,
            ).digest()
        ).decode()

    def _normalize_interval_minutes(self, interval) -> int:
        """Normalize interval to Kraken's expected integer minutes."""
        if interval is None:
            return 5

        # Common timeframe strings
        if isinstance(interval, str):
            s = interval.strip().lower()
            mapping = {
                "1m": 1,
                "5m": 5,
                "15m": 15,
                "30m": 30,
                "1h": 60,
                "4h": 240,
                "1d": 1440,
            }
            if s in mapping:
                return mapping[s]
            # Also allow raw numbers as strings
            if s.isdigit():
                return int(s)

        # Integers: treat >= 60 as seconds if divisible by 60
        if isinstance(interval, (int, float)):
            iv = int(interval)
            if iv >= 60 and iv % 60 == 0:
                return iv // 60
            return iv

        # Fallback
        return 5

    def _normalize_order_side(self, side: str) -> str:
        if not side:
            return "buy"
        s = side.strip().lower()
        if s in {"buy", "sell"}:
            return s
        if s in {"b", "long", "buy"}:
            return "buy"
        if s in {"s", "short", "sell"}:
            return "sell"
        return "buy" if s.startswith("b") else "sell"

    def _normalize_order_type(self, order_type: str) -> str:
        if not order_type:
            return "market"
        t = order_type.strip().lower()
        if t in {"market", "limit"}:
            return t
        if t == "m":
            return "market"
        if t == "l":
            return "limit"
        return "limit" if "limit" in t else "market"

    def _rate_limit_wait(self, cost: float = 1.0) -> None:
        """Block until we have enough rate-limit headroom for a private API call.

        Uses a leaky-bucket model: the counter increments by ``cost`` per call
        and decays by ``_rate_decay_per_sec`` every second.  If the counter
        would exceed ``_rate_max``, we sleep until it decays enough.
        """
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._rate_last_time
            self._rate_counter = max(0.0, self._rate_counter - elapsed * self._rate_decay_per_sec)
            self._rate_last_time = now

            if self._rate_counter + cost > self._rate_max:
                wait = (self._rate_counter + cost - self._rate_max) / self._rate_decay_per_sec
                logger.debug(f"Kraken rate limit: sleeping {wait:.1f}s")
                time.sleep(wait)
                # Recalculate after sleep
                now2 = time.monotonic()
                elapsed2 = now2 - self._rate_last_time
                self._rate_counter = max(0.0, self._rate_counter - elapsed2 * self._rate_decay_per_sec)
                self._rate_last_time = now2

            self._rate_counter += cost
    
    def _request(
        self,
        endpoint: str,
        params: dict = None,
        private: bool = False
    ) -> Optional[Dict]:
        """Make API request to Kraken"""
        try:
            if params is None:
                params = {}

            # Rate-limit private calls
            if private:
                self._rate_limit_wait()

            url = f"{self.base_url}/0/{endpoint}"
            headers = {}
            
            if private:
                nonce = str(int(time.time() * 1000))
                signed_params = params.copy()
                signed_params["nonce"] = nonce
                headers = {
                    "API-Sign": self._get_kraken_signature(f"/0/{endpoint}", signed_params, nonce),
                    "API-Key": self.api_key
                }
                response = self.session.post(url, headers=headers, data=signed_params, timeout=30)
            else:
                # Public endpoints should use GET with query params
                response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("error"):
                    if "EGeneral:Permission denied" in result.get("error", []):
                        logger.error(
                            "Kraken API permission denied. Your API key likely lacks required permissions "
                            f"for this endpoint ({endpoint}). Enable the relevant permissions in Kraken API settings "
                            "(e.g., 'Query Funds' for Balance, 'Trade' for placing orders)."
                        )
                    logger.error(f"Kraken API error: {result['error']}")
                    return None
                return result.get("result")
            else:
                logger.error(f"API error: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balances"""
        try:
            result = self._request("private/Balance", private=True)
            
            if not result:
                return {}
            
            balances = {}
            for asset, balance_str in result.items():
                balance = float(balance_str)
                if balance > 0:
                    # Kraken uses X/Z prefixes (e.g. XXBT, ZUSD). Normalize safely.
                    clean_asset = self._clean_asset_code(asset)
                    if clean_asset not in balances:
                        balances[clean_asset] = 0
                    balances[clean_asset] += balance
            
            return balances
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get ticker information.
        
        Args:
            symbol: Kraken pair (e.g., 'XBTUSD', 'ETHUSD')
        """
        try:
            result = self._request("public/Ticker", {"pair": symbol})
            
            if not result:
                return None
            
            # Get the first result
            ticker_data = list(result.values())[0]
            
            return {
                "symbol": symbol,
                "price": float(ticker_data["c"][0]),  # Last closed price
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    def get_klines(
        self,
        symbol: str,
        interval = 300,  # Accepts "5m" (preferred) or seconds/minutes
        limit: int = 100
    ) -> List[Dict]:
        """
        Get candlestick data.
        
        Args:
            symbol: Trading pair (e.g., 'XBTUSD')
            interval: Candle interval in seconds (60=1m, 300=5m, 900=15m, 3600=1h)
            limit: Number of candles
        """
        try:
            interval_minutes = self._normalize_interval_minutes(interval)
            result = self._request(
                "public/OHLC",
                {
                    "pair": symbol,
                    "interval": interval_minutes,
                    "since": 0
                }
            )
            
            if not result:
                return []
            
            # Remove the last entry (it's a count, not data)
            candles = []
            for pair, ohlc_data in result.items():
                if isinstance(ohlc_data, list):
                    for ohlc in ohlc_data[-limit:]:
                        candles.append({
                            "timestamp": datetime.fromtimestamp(int(ohlc[0])),
                            "open": float(ohlc[1]),
                            "high": float(ohlc[2]),
                            "low": float(ohlc[3]),
                            "close": float(ohlc[4]),
                            "volume": float(ohlc[6]),
                        })
            
            return candles
        except Exception as e:
            logger.error(f"Error getting klines: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        order_type: str,  # "market" or "limit"
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Kraken pair (e.g., 'XBTUSD')
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            quantity: Amount to trade
            price: Price for limit orders
        """
        try:
            if quantity <= 0:
                logger.error("Order quantity must be positive")
                return None
            
            side_norm = self._normalize_order_side(side)
            order_type_norm = self._normalize_order_type(order_type)

            info = self._get_asset_pair_info(symbol) or {}
            ordermin = info.get("ordermin")
            if ordermin is not None:
                try:
                    min_vol = float(ordermin)
                    if quantity < min_vol:
                        logger.error(
                            f"Order volume {quantity} is below Kraken minimum {min_vol} for {symbol}"
                        )
                        return None
                except Exception:
                    pass

            if order_type_norm == "limit" and price is None:
                logger.error("Price required for limit orders")
                return None
            
            params = {
                "pair": symbol,
                "type": side_norm,
                "ordertype": order_type_norm,
                "volume": self._format_volume(symbol, quantity),
            }
            
            if price:
                params["price"] = self._format_price(symbol, float(price))
            
            logger.info(
                f"Placing {side_norm} {order_type_norm} order: {params['volume']} {symbol} @ {params.get('price', price)}"
            )
            
            result = self._request("private/AddOrder", params, private=True)
            
            if result:
                logger.info(f"Order placed: {result}")
                return result
            
            return None
        
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            result = self._request("private/OpenOrders", private=True)
            return result if result else []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            result = self._request(
                "private/CancelOrder",
                {"txid": order_id},
                private=True
            )
            
            if result:
                logger.info(f"Order {order_id} cancelled")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def query_orders(self, txids: List[str]) -> Dict[str, Dict]:
        """Query one or more orders by txid.

        Returns a dict keyed by txid, or an empty dict on failure.
        """
        try:
            if not txids:
                return {}
            result = self._request(
                "private/QueryOrders",
                {"txid": ",".join(txids)},
                private=True,
            )
            return result or {}
        except Exception as e:
            logger.error(f"Error querying orders: {e}")
            return {}

    def get_trades_history(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ofs: int = 0,
        trade_type: str = "all",
    ) -> Optional[Dict]:
        """Fetch account trade history.

        Returns Kraken's raw result dict, typically containing keys like:
        - trades: { <txid>: {pair, type, price, vol, cost, fee, time, ...}, ... }
        - count: int

        Notes:
        - Requires API key permission: 'Query Trades'.
        - Times are unix timestamps (seconds).
        """
        try:
            params: Dict[str, object] = {
                "type": (trade_type or "all"),
                "ofs": int(ofs or 0),
            }
            if start is not None:
                params["start"] = int(start)
            if end is not None:
                params["end"] = int(end)

            result = self._request("private/TradesHistory", params, private=True)
            if not result:
                return None
            return result
        except Exception as e:
            logger.error(f"Error getting trades history: {e}")
            return None

# Kraken Sandbox trading pairs (convert from common names)
KRAKEN_PAIRS = {
    "BTCUSD": "XBTUSD",
    "ETHUSD": "ETHUSD",
    "BNBUSD": "KSMBUSD",
    "XRPUSD": "XRPUSD",
}

def convert_to_kraken_pair(symbol: str) -> str:
    """Convert standard symbol to Kraken format"""
    return KRAKEN_PAIRS.get(symbol, symbol)
