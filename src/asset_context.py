#!/usr/bin/env python3
"""External asset-quality context for live trade vetoes.

This module is intentionally veto-only. It never creates a buy signal; it only
blocks new long entries when public market metadata says the asset is too small
or currently in a severe drawdown.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import requests


PAIR_TO_COINGECKO_ID: Dict[str, str] = {
    "XBTUSD": "bitcoin",
    "BTCUSD": "bitcoin",
    "ETHUSD": "ethereum",
    "SOLUSD": "solana",
    "ADAUSD": "cardano",
    "XRPUSD": "ripple",
    "DOTUSD": "polkadot",
    "LINKUSD": "chainlink",
    "AVAXUSD": "avalanche-2",
    "AAVEUSD": "aave",
    "UNIUSD": "uniswap",
    "LTCUSD": "litecoin",
    "ATOMUSD": "cosmos",
    "DOGEUSD": "dogecoin",
    "FILUSD": "filecoin",
    "NEARUSD": "near",
    "SUIUSD": "sui",
    "BNBUSD": "binancecoin",
    "TRXUSD": "tron",
    "HBARUSD": "hedera-hashgraph",
    "HYPEUSD": "hyperliquid",
    "TAOUSD": "bittensor",
    "OKBUSD": "okb",
    "INJUSD": "injective-protocol",
    "ARBUSD": "arbitrum",
    "OPUSD": "optimism",
    "APTUSD": "aptos",
    "TIAUSD": "celestia",
    "ONDOUSD": "ondo-finance",
    "RENDERUSD": "render-token",
    "JUPUSD": "jupiter-exchange-solana",
    "ICPUSD": "internet-computer",
    "LDOUSD": "lido-dao",
    "BCHUSD": "bitcoin-cash",
    "STXUSD": "blockstack",
    "KAVAUSD": "kava",
    "ENAUSD": "ethena",
    "FLOKIUSD": "floki",
    "XLMUSD": "stellar",
    "XMRUSD": "monero",
    "ZECUSD": "zcash",
    "JITOSOLUSD": "jito-staked-sol",
}

KRAKEN_PAIR_ALIASES = {
    "XXBTZUSD": "XBTUSD",
    "XETHZUSD": "ETHUSD",
    "XXLMZUSD": "XLMUSD",
    "XXRPZUSD": "XRPUSD",
    "XLTCZUSD": "LTCUSD",
    "XXMRZUSD": "XMRUSD",
    "XZECZUSD": "ZECUSD",
}


def normalize_pair(pair: str) -> str:
    normalized = (pair or "").upper().replace("/", "")
    return KRAKEN_PAIR_ALIASES.get(normalized, normalized)


class AssetContextGuard:
    """CoinGecko-backed asset quality veto with local caching."""

    def __init__(
        self,
        cache_path: Path,
        ttl_seconds: int = 6 * 3600,
        timeout_seconds: int = 6,
        max_market_cap_rank: int = 150,
        min_24h_change_pct: float = -18.0,
        min_7d_change_pct: float = -35.0,
        block_unmapped: bool = False,
    ):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.max_market_cap_rank = max_market_cap_rank
        self.min_24h_change_pct = min_24h_change_pct
        self.min_7d_change_pct = min_7d_change_pct
        self.block_unmapped = block_unmapped
        self.session = requests.Session()

    def evaluate(self, pair: str) -> dict:
        normalized = normalize_pair(pair)
        coin_id = PAIR_TO_COINGECKO_ID.get(normalized)
        if not coin_id:
            if self.block_unmapped:
                return {"ok": False, "reason": f"asset_context_unmapped_{normalized}"}
            return {"ok": True, "reason": f"asset_context_unmapped_{normalized}", "source": "unmapped"}

        snapshot = self._get_snapshot(coin_id)
        if not snapshot:
            return {"ok": True, "reason": "asset_context_unavailable", "source": "unavailable"}

        rank = self._num(snapshot.get("market_cap_rank"))
        change_24h = self._num(snapshot.get("price_change_percentage_24h"))
        change_7d = self._num(
            snapshot.get("price_change_percentage_7d_in_currency", snapshot.get("price_change_percentage_7d"))
        )

        if rank and rank > self.max_market_cap_rank:
            return {
                "ok": False,
                "reason": f"asset_context_rank_{int(rank)}_gt_{self.max_market_cap_rank}",
                "rank": rank,
                "source": snapshot.get("_source", "api"),
            }

        if change_24h is not None and change_24h <= self.min_24h_change_pct:
            return {
                "ok": False,
                "reason": f"asset_context_24h_drop_{change_24h:.1f}",
                "rank": rank,
                "change_24h": change_24h,
                "source": snapshot.get("_source", "api"),
            }

        if change_7d is not None and change_7d <= self.min_7d_change_pct:
            return {
                "ok": False,
                "reason": f"asset_context_7d_drop_{change_7d:.1f}",
                "rank": rank,
                "change_7d": change_7d,
                "source": snapshot.get("_source", "api"),
            }

        return {
            "ok": True,
            "reason": "asset_context_ok",
            "rank": rank,
            "change_24h": change_24h,
            "change_7d": change_7d,
            "source": snapshot.get("_source", "api"),
        }

    def _get_snapshot(self, coin_id: str) -> Optional[dict]:
        cache = self._load_cache()
        cached = cache.get(coin_id)
        now = time.time()
        if cached and now - float(cached.get("ts", 0) or 0) < self.ttl_seconds:
            data = dict(cached.get("data", {}) or {})
            data["_source"] = "cache"
            return data

        try:
            response = self.session.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": coin_id,
                    "price_change_percentage": "24h,7d",
                    "per_page": 1,
                    "page": 1,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                return None
            data = payload[0]
            cache[coin_id] = {"ts": now, "data": data}
            self._save_cache(cache)
            data = dict(data)
            data["_source"] = "api"
            return data
        except Exception:
            return None

    def _load_cache(self) -> Dict[str, dict]:
        try:
            if self.cache_path.exists():
                payload = json.loads(self.cache_path.read_text())
                if isinstance(payload, dict):
                    return payload
        except Exception:
            pass
        return {}

    def _save_cache(self, cache: Dict[str, dict]) -> None:
        try:
            self.cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        except Exception:
            pass

    def _num(self, value):
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None