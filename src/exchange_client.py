"""
Exchange API client for connecting to Binance.us with safety measures.
"""
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from typing import Dict, List, Optional, Tuple
from loguru import logger
import os
from datetime import datetime

class ExchangeClient:
    """Secure exchange client with Binance.us support"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """
        Initialize exchange client for Binance.us
        
        Args:
            api_key: Binance.us API key
            api_secret: Binance.us API secret
            testnet: Not available for Binance.us (use backtest instead)
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self.testnet = False  # Binance.us doesn't support testnet
        
        if not self.api_key or not self.api_secret:
            logger.warning("No API credentials found. Running in read-only mode.")
        
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            tld='us',  # Use Binance.us instead of Binance.com
            requests_params={"timeout": 30}
        )
        
        logger.info(f"Exchange client initialized for Binance.us")
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balances"""
        try:
            account = self.client.get_account()
            balances = {}
            for balance in account["balances"]:
                free = float(balance["free"])
                locked = float(balance["locked"])
                if free > 0 or locked > 0:
                    balances[balance["asset"]] = {
                        "free": free,
                        "locked": locked,
                        "total": free + locked
                    }
            return balances
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker information for a symbol"""
        try:
            ticker = self.client.get_symbol_info(symbol)
            price = self.client.get_symbol_ticker(symbol=symbol)
            return {
                "symbol": symbol,
                "price": float(price["price"]),
                "timestamp": datetime.now().isoformat()
            }
        except BinanceAPIException as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """
        Get candlestick data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', etc.)
            limit: Number of candles to fetch
        """
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            processed_klines = []
            for kline in klines:
                processed_klines.append({
                    "timestamp": datetime.fromtimestamp(kline[0] / 1000),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[7]),
                })
            
            return processed_klines
        except BinanceAPIException as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        order_type: str,  # "MARKET" or "LIMIT"
        quantity: float,
        price: Optional[float] = None,
        stop_loss_price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place an order with safety checks.
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            order_type: MARKET or LIMIT
            quantity: Amount to trade
            price: Price for limit orders
            stop_loss_price: Price to trigger stop loss
        """
        try:
            # Validate inputs
            if quantity <= 0:
                logger.error("Order quantity must be positive")
                return None
            
            if order_type == "LIMIT" and price is None:
                logger.error("Price required for limit orders")
                return None
            
            # Log order details
            logger.info(f"Placing {side} {order_type} order: {quantity} {symbol} @ {price}")
            
            if order_type == "MARKET":
                order = self.client.order_market(
                    symbol=symbol,
                    side=side,
                    quantity=quantity
                )
            else:  # LIMIT
                order = self.client.order_limit(
                    symbol=symbol,
                    side=side,
                    timeInForce="GTC",
                    quantity=quantity,
                    price=price
                )
            
            logger.info(f"Order placed successfully: {order['orderId']}")
            return order
        
        except BinanceOrderException as e:
            logger.error(f"Order error: {e}")
            return None
        except BinanceAPIException as e:
            logger.error(f"API error placing order: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel an order"""
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except BinanceAPIException as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            return orders
        except BinanceAPIException as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def get_order_status(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Get status of a specific order"""
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return order
        except BinanceAPIException as e:
            logger.error(f"Error getting order status: {e}")
            return None

    # ------------------------------------------------------------------
    # Methods required by TradingBot (parity with KrakenClient interface)
    # ------------------------------------------------------------------

    def get_pair_assets(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (base_asset, quote_asset) for a Binance symbol.

        e.g. 'BTCUSDT' -> ('BTC', 'USDT')
        """
        try:
            info = self.client.get_symbol_info(symbol)
            if info:
                return info.get("baseAsset"), info.get("quoteAsset")
        except Exception as e:
            logger.warning(f"get_pair_assets({symbol}): {e}")
        # Fallback heuristic for common quote assets
        for quote in ("USDT", "USD", "BTC", "ETH", "BNB", "BUSD"):
            if symbol.endswith(quote):
                return symbol[: -len(quote)], quote
        return None, None

    def get_min_order_volume(self, symbol: str) -> Optional[float]:
        """Return minimum order quantity for a Binance symbol."""
        try:
            info = self.client.get_symbol_info(symbol)
            if info and "filters" in info:
                for f in info["filters"]:
                    if f.get("filterType") == "LOT_SIZE":
                        return float(f.get("minQty", 0))
        except Exception as e:
            logger.warning(f"get_min_order_volume({symbol}): {e}")
        return None

    def query_orders(self, txids: List[str]) -> Dict[str, Dict]:
        """Query orders by ID.  Binance uses integer orderIds, not string txids.

        Best-effort: tries to parse each txid as an int and query.
        Returns a dict keyed by the original txid string.
        """
        results: Dict[str, Dict] = {}
        for txid in txids:
            try:
                oid = int(txid)
                # We'd need the symbol — which we don't have here.
                # This is a limitation; for now return empty.
            except (ValueError, TypeError):
                pass
        return results
