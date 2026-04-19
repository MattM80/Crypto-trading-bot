#!/usr/bin/env python3
"""
ML SIGNAL WEIGHTING ENGINE - Dynamic Learning Signal Scorer
Replaces static signal scores with learned dynamic weights based on market conditions.

Each of the 42 tools gets its own logistic regression model that learns:
"Given current market features, will this tool's signal be profitable?"

Uses pure numpy (no sklearn/tensorflow) for lightweight, fast online learning.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


# Features used to predict signal success
FEATURES = [
    'rsi_7',           # Current RSI (0-100)
    'atr_pct',         # ATR as % of price (volatility)
    'ret_4h',          # 4h return
    'ret_24h',         # 24h return
    'vs_sma50',        # % distance from SMA50
    'volume_ratio',    # Current vol / 20-bar avg
    'fng',             # Fear & Greed index (0-100)
    'hour_sin',        # Sin of hour (captures time-of-day pattern)
    'hour_cos',        # Cos of hour
    'dow_sin',         # Sin of day-of-week
    'dow_cos',         # Cos of day-of-week
    'btc_ret_24h',     # BTC's 24h return (market direction)
    'stablecoin_signal', # On-chain stablecoin flow (-5 to +5)
    'news_sentiment',  # News sentiment (-10 to +10)
    'ob_imbalance',    # Orderbook imbalance (0.1-10)
    'funding_rate',    # Futures funding rate
]


class MLSignalWeighter:
    """Online learning model for dynamic signal weighting."""
    
    def __init__(self, model_dir: str = 'data/ml_models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}  # tool_name → model weights
        self.feature_names = FEATURES
        self.n_features = len(FEATURES)
        self.learning_rate = 0.01
        self.min_samples = 20  # Need 20+ trades before trusting the model
        self.trade_history = {}  # tool → list of (features, outcome)
        
        # Load existing models
        self._load_models()
        
        logger.info(f"🧠 ML Signal Weighter initialized with {len(self.models)} pre-trained models")
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_win_probability(self, tool: str, features: dict) -> float:
        """Predict probability that this signal will be profitable."""
        if tool not in self.models or self.models[tool]['n_samples'] < self.min_samples:
            return 0.5  # No opinion yet — default to neutral
        
        x = self._features_to_array(features)
        w = self.models[tool]['weights']
        b = self.models[tool]['bias']
        z = np.dot(w, x) + b
        prob = float(self._sigmoid(z))
        
        return np.clip(prob, 0.01, 0.99)  # Avoid extreme values
    
    def get_score_multiplier(self, tool: str, features: dict) -> float:
        """
        Returns a multiplier (0.3 to 2.0) for the tool's signal score.
        - 0.3 = model says this will probably lose, reduce score by 70%
        - 1.0 = model is neutral (not enough data or 50/50)
        - 2.0 = model says this will probably win, double the score
        """
        prob = self.predict_win_probability(tool, features)
        
        # Map probability to multiplier
        # p=0.5 → mult=1.0 (neutral)
        # p=0.7 → mult=1.4
        # p=0.3 → mult=0.6
        # p=0.8 → mult=1.8 (near max boost)
        # p=0.2 → mult=0.4 (near max penalty)
        multiplier = 0.3 + (prob * 1.7)  # Range: 0.3 to 2.0
        return round(multiplier, 2)
    
    def record_trade(self, tool: str, features: dict, profitable: bool):
        """Record a completed trade for online learning."""
        if tool not in self.trade_history:
            self.trade_history[tool] = []
        
        self.trade_history[tool].append({
            'features': features.copy(),
            'outcome': 1 if profitable else 0,
            'timestamp': time.time()
        })
        
        # Online update: adjust weights after each trade
        self._update_model(tool, features, 1 if profitable else 0)
        
        # Save periodically (every 5 trades per tool)
        if len(self.trade_history[tool]) % 5 == 0:
            self._save_models()
            logger.debug(f"🧠 Updated {tool} model: {self.models[tool]['n_samples']} samples")
    
    def _update_model(self, tool: str, features: dict, y: int):
        """Online gradient descent update for logistic regression."""
        if tool not in self.models:
            self.models[tool] = {
                'weights': np.zeros(self.n_features, dtype=np.float64),
                'bias': 0.0,
                'n_samples': 0
            }
        
        model = self.models[tool]
        x = self._features_to_array(features)
        
        # Forward pass
        z = np.dot(model['weights'], x) + model['bias']
        pred = float(self._sigmoid(z))
        
        # Gradient (derivative of log-loss)
        error = pred - y
        
        # Learning rate decay (newer models learn faster)
        lr = self.learning_rate / (1 + model['n_samples'] * 0.001)
        
        # Update weights and bias
        model['weights'] -= lr * error * x
        model['bias'] -= lr * error
        model['n_samples'] += 1
    
    def _features_to_array(self, features: dict) -> np.ndarray:
        """Convert feature dict to normalized numpy array."""
        x = np.zeros(self.n_features, dtype=np.float64)
        
        for i, name in enumerate(self.feature_names):
            val = features.get(name, 0)
            
            # Normalize each feature to roughly -1 to +1 range
            if name == 'rsi_7':
                x[i] = (val - 50) / 50  # 0-100 → -1 to +1
            elif name == 'atr_pct':
                x[i] = min(val / 5, 1)  # Cap at 5% volatility
            elif name in ('ret_4h', 'ret_24h'):
                x[i] = np.clip(val / 10, -1, 1)  # ±10% → ±1
            elif name == 'vs_sma50':
                x[i] = np.clip(val / 20, -1, 1)  # ±20% from SMA → ±1
            elif name == 'volume_ratio':
                x[i] = min(val / 5, 1) - 0.4  # 0-5x vol → -0.4 to +0.6
            elif name == 'fng':
                x[i] = (val - 50) / 50  # 0-100 → -1 to +1
            elif name in ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'):
                x[i] = val  # Already -1 to +1
            elif name == 'btc_ret_24h':
                x[i] = np.clip(val / 10, -1, 1)  # ±10% → ±1
            elif name == 'stablecoin_signal':
                x[i] = np.clip(val / 5, -1, 1)  # ±5 → ±1
            elif name == 'news_sentiment':
                x[i] = np.clip(val / 10, -1, 1)  # ±10 → ±1
            elif name == 'ob_imbalance':
                x[i] = np.clip((val - 1) / 2, -1, 1)  # 0.1-3 → roughly -0.5 to +1
            elif name == 'funding_rate':
                x[i] = np.clip(val * 10000, -1, 1)  # Funding rates are tiny
            else:
                x[i] = val  # Unknown feature, use as-is
        
        return x
    
    def _save_models(self):
        """Save all models to disk as JSON."""
        try:
            save_data = {
                'models': {},
                'trade_history': self.trade_history,
                'saved_at': time.time(),
                'feature_names': self.feature_names
            }
            
            # Convert numpy arrays to lists for JSON serialization
            for tool, model in self.models.items():
                save_data['models'][tool] = {
                    'weights': model['weights'].tolist(),
                    'bias': float(model['bias']),
                    'n_samples': int(model['n_samples'])
                }
            
            model_file = self.model_dir / 'ml_models.json'
            with open(model_file, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            logger.debug(f"💾 Saved {len(self.models)} ML models to {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save ML models: {e}")
    
    def _load_models(self):
        """Load models from disk."""
        try:
            model_file = self.model_dir / 'ml_models.json'
            if not model_file.exists():
                logger.info("No existing ML models found, starting fresh")
                return
            
            with open(model_file, 'r') as f:
                save_data = json.load(f)
            
            # Restore models
            for tool, model_data in save_data.get('models', {}).items():
                self.models[tool] = {
                    'weights': np.array(model_data['weights'], dtype=np.float64),
                    'bias': float(model_data['bias']),
                    'n_samples': int(model_data['n_samples'])
                }
            
            # Restore trade history
            self.trade_history = save_data.get('trade_history', {})
            
            logger.info(f"📚 Loaded {len(self.models)} pre-trained ML models")
            
            # Log model stats
            total_samples = sum(m['n_samples'] for m in self.models.values())
            active_models = sum(1 for m in self.models.values() if m['n_samples'] >= self.min_samples)
            logger.info(f"🧠 ML Stats: {total_samples} total samples, {active_models} active models")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.models = {}
            self.trade_history = {}
    
    def get_feature_importance(self, tool: str) -> dict:
        """Show which features matter most for this tool (by absolute weight)."""
        if tool not in self.models:
            return {}
        
        weights = self.models[tool]['weights']
        importance = {name: abs(float(w)) for name, w in zip(self.feature_names, weights)}
        return dict(sorted(importance.items(), key=lambda x: -x[1]))
    
    def get_model_stats(self) -> dict:
        """Return stats for all tool models."""
        stats = {}
        for tool, model in self.models.items():
            n = model['n_samples']
            if n > 0:
                history = self.trade_history.get(tool, [])
                wins = sum(1 for t in history if t['outcome'] == 1)
                win_rate = wins / len(history) if history else 0
                
                stats[tool] = {
                    'samples': n,
                    'win_rate': win_rate,
                    'is_active': n >= self.min_samples,
                    'top_features': list(self.get_feature_importance(tool).keys())[:3]
                }
        
        return stats
    
    def get_status_summary(self) -> str:
        """Get a concise status summary for logging."""
        if not self.models:
            return "ML Models: None loaded"
        
        total_models = len(self.models)
        active_models = sum(1 for m in self.models.values() if m['n_samples'] >= self.min_samples)
        total_samples = sum(m['n_samples'] for m in self.models.values())
        
        # Show top 3 performing models by sample count
        top_models = sorted(
            [(tool, model['n_samples']) for tool, model in self.models.items()],
            key=lambda x: -x[1]
        )[:3]
        
        top_str = ", ".join([f"{tool}: {samples}" for tool, samples in top_models])
        
        return f"ML Models: {active_models}/{total_models} active | Total samples: {total_samples} | Top: [{top_str}]"
    
    def cleanup_old_history(self, max_age_days: int = 30):
        """Remove trade history older than max_age_days to prevent memory bloat."""
        cutoff = time.time() - (max_age_days * 24 * 3600)
        removed_count = 0
        
        for tool in self.trade_history:
            old_len = len(self.trade_history[tool])
            self.trade_history[tool] = [
                trade for trade in self.trade_history[tool]
                if trade['timestamp'] > cutoff
            ]
            removed_count += old_len - len(self.trade_history[tool])
        
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} old trade records")
            self._save_models()  # Save the cleaned data