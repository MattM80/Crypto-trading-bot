"""
Kraken Futures API client.

Separate from src/kraken_client.py (spot). Kraken Futures uses a different
endpoint, different authentication scheme, different symbol convention, and has
its own margin/liquidation/funding logic. Do NOT mix this with the spot client.

Feature status: Phase 1 — read-only + auth verification. Order placement is
implemented but gated by ENABLE_FUTURES_TRADING so it stays disabled until we
explicitly go live with small-size shorts.

Docs: https://docs.futures.kraken.com/
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Kraken Futures symbol convention:
#   PF_<base><quote>   = perpetual inverse/linear (most common for USD-quoted perps)
#   PI_<base><quote>   = perpetual inverse
#   FI_<base><quote>_<YYMMDD> = dated inverse
# We stick to PF_ perps for USD pairs. This map converts our internal spot
# symbol (e.g. "XBTUSD") to the futures symbol.
SPOT_TO_FUTURES = {
    "XBTUSD": "PF_XBTUSD",
    "ETHUSD": "PF_ETHUSD",
    "XRPUSD": "PF_XRPUSD",
    "SOLUSD": "PF_SOLUSD",
    "ADAUSD": "PF_ADAUSD",
    "DOGEUSD": "PF_DOGEUSD",
    "LINKUSD": "PF_LINKUSD",
    "AVAXUSD": "PF_AVAXUSD",
    "DOTUSD": "PF_DOTUSD",
    "LTCUSD": "PF_LTCUSD",
    "BCHUSD": "PF_BCHUSD",
    "UNIUSD": "PF_UNIUSD",
    "ATOMUSD": "PF_ATOMUSD",
    "NEARUSD": "PF_NEARUSD",
    "FILUSD": "PF_FILUSD",
    "AAVEUSD": "PF_AAVEUSD",
    "XLMUSD": "PF_XLMUSD",
}

FUTURES_BASE_URL = "https://futures.kraken.com/derivatives/api/v3"


def to_futures_symbol(spot_pair: str) -> Optional[str]:
    """Convert internal spot symbol (e.g. 'XBTUSD') to Kraken Futures symbol."""
    return SPOT_TO_FUTURES.get(spot_pair)


class KrakenFuturesClient:
    """Minimal Kraken Futures client. Phase 1: read + signed-request auth."""

    def __init__(self, api_key: Optional[str] = None, private_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("KRAKEN_FUTURES_API_KEY", "")
        self.private_key = private_key or os.getenv("KRAKEN_FUTURES_PRIVATE_KEY", "")
        self.base_url = FUTURES_BASE_URL
        self.session = requests.Session()
        if not self.api_key or not self.private_key:
            logger.warning("[FUTURES] No credentials set — read-only public endpoints only.")
        else:
            logger.info("[FUTURES] Credentials loaded — live futures API reachable.")

    # ------------- Signing -------------
    def _sign(self, endpoint_path: str, nonce: str, post_data: str = "") -> str:
        """Kraken Futures signature:
            sha256(postData + nonce + endpointPath) -> hmac_sha512(decoded_secret) -> base64."""
        # endpoint_path must NOT include the /derivatives/api/v3 prefix
        message = (post_data + nonce + endpoint_path).encode("utf-8")
        sha256 = hashlib.sha256(message).digest()
        secret_decoded = base64.b64decode(self.private_key)
        mac = hmac.new(secret_decoded, sha256, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _private_request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        """Authenticated request. `path` is the path after /derivatives/api/v3."""
        if not self.api_key or not self.private_key:
            raise RuntimeError("Futures credentials not configured")
        params = params or {}
        post_data = urllib.parse.urlencode(params)
        nonce = str(int(time.time() * 1000))
        # Kraken expects signature computed against the path without the api_v3 prefix
        # but with leading slash included per their example:
        endpoint_path_for_sig = f"/api/v3{path}" if not path.startswith("/api/") else path
        signature = self._sign(endpoint_path_for_sig, nonce, post_data)
        headers = {
            "APIKey": self.api_key,
            "Nonce": nonce,
            "Authent": signature,
        }
        url = f"{self.base_url}{path}"
        try:
            if method.upper() == "GET":
                resp = self.session.get(url, headers=headers, params=params, timeout=15)
            else:
                resp = self.session.post(url, headers=headers, data=post_data, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"[FUTURES] Request failed {method} {path}: {e}")
            return {"result": "error", "error": str(e)}

    def _public_request(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"[FUTURES] Public request failed {path}: {e}")
            return {"result": "error", "error": str(e)}

    # ------------- Public data -------------
    def get_tickers(self) -> List[Dict]:
        """All active futures tickers. Each has symbol, markPrice, bid, ask, last24h stats."""
        resp = self._public_request("/tickers")
        return resp.get("tickers", []) if isinstance(resp, dict) else []

    def get_ticker(self, futures_symbol: str) -> Optional[Dict]:
        tickers = self.get_tickers()
        for t in tickers:
            if t.get("symbol") == futures_symbol:
                return t
        return None

    def get_instruments(self) -> List[Dict]:
        resp = self._public_request("/instruments")
        return resp.get("instruments", []) if isinstance(resp, dict) else []

    # ------------- Private: account state -------------
    def get_accounts(self) -> Dict:
        """Returns margin accounts: cash balances, open position PnL, margin usage."""
        return self._private_request("GET", "/accounts")

    def get_open_positions(self) -> List[Dict]:
        resp = self._private_request("GET", "/openpositions")
        return resp.get("openPositions", []) if isinstance(resp, dict) else []

    def get_open_orders(self) -> List[Dict]:
        resp = self._private_request("GET", "/openorders")
        return resp.get("openOrders", []) if isinstance(resp, dict) else []

    # ------------- Private: orders (gated) -------------
    def send_order(
        self,
        futures_symbol: str,
        side: str,               # "buy" | "sell"
        size: float,              # contracts (futures size, not notional USD)
        order_type: str = "mkt",  # "mkt" | "lmt" | "stp" | "take_profit"
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict:
        """Send an order. Caller MUST verify ENABLE_FUTURES_TRADING before calling."""
        params: Dict[str, str] = {
            "orderType": order_type,
            "symbol": futures_symbol,
            "side": side,
            "size": str(size),
        }
        if limit_price is not None:
            params["limitPrice"] = str(limit_price)
        if stop_price is not None:
            params["stopPrice"] = str(stop_price)
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._private_request("POST", "/sendorder", params)

    def cancel_order(self, order_id: str) -> Dict:
        return self._private_request("POST", "/cancelorder", {"order_id": order_id})

    # ------------- Health check -------------
    def ping(self) -> bool:
        """Returns True if auth is working. Used at bot startup when futures is enabled."""
        resp = self.get_accounts()
        if isinstance(resp, dict) and resp.get("result") == "success":
            return True
        logger.warning(f"[FUTURES] Auth ping failed: {resp}")
        return False
