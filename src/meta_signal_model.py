#!/usr/bin/env python3
"""Lightweight pooled meta-model for trade filtering and size scaling."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from loguru import logger


FEATURE_NAMES = [
    'score',
    'base_score',
    'mtf_multiplier',
    'rsi_4h',
    'bullish_4h_pct',
    'fng',
    'range_pos_24h',
    'atr_pct',
    'short_pressure_score',
    'liquidity_cap_usage',
    'corr_large_cap',
    'corr_alt_l1',
    'corr_defi',
    'corr_meme',
    'corr_mid_cap',
    'collapse_gate',
    'hour_sin',
    'hour_cos',
    'dow_sin',
    'dow_cos',
]


class SignalMetaModel:
    """A conservative pooled logistic model trained from the trade journal."""

    def __init__(self, model_path: Union[Path, str], min_samples: int = 40, learning_rate: float = 0.03):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_names = FEATURE_NAMES
        self.n_features = len(self.feature_names)
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.reset()
        self._load()

    def reset(self):
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = 0.0
        self.n_samples = 0

    def _sigmoid(self, value: float) -> float:
        value = float(np.clip(value, -500, 500))
        return 1.0 / (1.0 + np.exp(-value))

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        x = np.zeros(self.n_features, dtype=np.float64)
        for idx, name in enumerate(self.feature_names):
            value = float(features.get(name, 0.0) or 0.0)
            if name in {'score', 'base_score'}:
                x[idx] = np.clip(value / 20.0 - 1.0, -1.0, 2.0)
            elif name == 'mtf_multiplier':
                x[idx] = np.clip((value - 1.0) / 0.5, -1.0, 1.0)
            elif name in {'rsi_4h', 'bullish_4h_pct', 'fng'}:
                x[idx] = np.clip((value - 50.0) / 50.0, -1.0, 1.0)
            elif name == 'range_pos_24h':
                x[idx] = np.clip((value - 0.5) / 0.5, -1.0, 1.0)
            elif name == 'atr_pct':
                x[idx] = np.clip(value / 10.0, 0.0, 2.0) - 0.5
            elif name == 'short_pressure_score':
                x[idx] = np.clip(value / 20.0, 0.0, 2.0) - 0.5
            elif name == 'liquidity_cap_usage':
                x[idx] = np.clip(value, 0.0, 1.5) - 0.5
            elif name.startswith('corr_'):
                x[idx] = 1.0 if value > 0 else 0.0
            elif name == 'collapse_gate':
                x[idx] = np.clip(value, -1.0, 1.0)
            elif name in {'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'}:
                x[idx] = np.clip(value, -1.0, 1.0)
            else:
                x[idx] = value
        return x

    def fit_samples(self, samples: Iterable[Tuple[Dict[str, float], bool]]):
        self.reset()
        sample_count = 0
        for features, profitable in samples:
            self._update(features, profitable)
            sample_count += 1
        self._save()
        logger.info(
            f"[META] Trained pooled meta-model on {sample_count} samples "
            f"({'active' if self.is_active() else 'warming'})"
        )

    def _update(self, features: Dict[str, float], profitable: bool):
        x = self._features_to_array(features)
        y = 1.0 if profitable else 0.0
        pred = self._sigmoid(np.dot(self.weights, x) + self.bias)
        error = pred - y
        lr = self.learning_rate / (1.0 + self.n_samples * 0.002)
        self.weights -= lr * error * x
        self.bias -= lr * error
        self.n_samples += 1

    def record_trade(self, features: Dict[str, float], profitable: bool):
        self._update(features, profitable)
        self._save()

    def predict_win_probability(self, features: Dict[str, float]) -> float:
        if not self.is_active():
            return 0.5
        x = self._features_to_array(features)
        prob = self._sigmoid(np.dot(self.weights, x) + self.bias)
        return float(np.clip(prob, 0.05, 0.95))

    def get_size_multiplier(self, features: Dict[str, float]) -> float:
        prob = self.predict_win_probability(features)
        return round(0.85 + prob * 0.30, 2)

    def should_veto(self, features: Dict[str, float], score: float) -> bool:
        if not self.is_active():
            return False
        prob = self.predict_win_probability(features)
        return prob < 0.38 and score < 16.0

    def is_active(self) -> bool:
        return self.n_samples >= self.min_samples

    def get_stats(self) -> Dict[str, float]:
        return {
            'n_samples': int(self.n_samples),
            'active': self.is_active(),
            'min_samples': int(self.min_samples),
            'bias': float(self.bias),
        }

    def _save(self):
        payload = {
            'feature_names': self.feature_names,
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'n_samples': int(self.n_samples),
            'min_samples': int(self.min_samples),
        }
        with open(self.model_path, 'w') as f:
            json.dump(payload, f, indent=2)

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, 'r') as f:
                payload = json.load(f)
            if payload.get('feature_names') != self.feature_names:
                logger.info("[META] Existing model feature set mismatch, rebuilding from journal")
                return
            self.weights = np.array(payload.get('weights', []), dtype=np.float64)
            if self.weights.shape != (self.n_features,):
                self.reset()
                return
            self.bias = float(payload.get('bias', 0.0))
            self.n_samples = int(payload.get('n_samples', 0))
        except Exception as exc:
            logger.warning(f"[META] Failed to load saved pooled meta-model: {exc}")
            self.reset()