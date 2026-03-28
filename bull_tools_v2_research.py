#!/usr/bin/env python3
"""
BULL TOOLS V2 RESEARCH - UNCONVENTIONAL APPROACHES
Mission: Find REAL edge in bull/chop markets using advanced techniques NO retail trader uses.

Round 1 found only 1 good tool (BTC strength rotation). We need DIFFERENT approaches:
- Chaos Theory / Nonlinear Dynamics  
- Information Theory
- Advanced Statistical Methods
- Microstructure / Volume Analysis
- Cross-Asset / Macro Signals
- Choppy Market Specific Tools

Validation requirements: Same brutal standard as crash tools.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add scipy imports for advanced methods
from scipy import stats
from scipy.stats import entropy
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h"

# Test pairs (same 16 as crash tools for comparison)
PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT", 
    "AVAXUSDT", "ATOMUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "UNIUSDT", 
    "FILUSDT", "NEARUSDT", "AAVEUSDT", "XLMUSDT"
]

# Validation settings - same as crash tools
FEES = 0.0065  # 0.65% round-trip
MIN_SIGNALS = 15
MIN_WIN_RATE = 55
MIN_PROFIT_FACTOR = 1.5

class UnconventionalTools:
    """Advanced trading tools using unconventional approaches."""
    
    def __init__(self):
        self.data = {}
        self.btc_data = None  # For cross-asset analysis
        
    def load_data(self, pair: str) -> pd.DataFrame:
        """Load and cache 1h data for a pair."""
        if pair not in self.data:
            file_path = DATA_DIR / f"{pair}_1h.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            self.data[pair] = df
            
            # Cache BTC for cross-asset analysis
            if pair == "BTCUSDT":
                self.btc_data = df
                
        return self.data[pair]
    
    # ==================== CHAOS THEORY / NONLINEAR DYNAMICS ====================
    
    def calc_lyapunov_exponent(self, price_series: np.array, m: int = 3, lag: int = 1, 
                              window: int = 100) -> float:
        """
        Calculate largest Lyapunov exponent to measure predictability.
        
        Negative Lyapunov = temporarily predictable system = tradeable window
        Positive Lyapunov = chaotic, unpredictable
        
        This is a simplified version using the Rosenstein method.
        """
        if len(price_series) < window + 50:
            return np.nan
            
        # Use log returns for better properties
        series = np.log(price_series[-window:])
        
        # Embed the series in m-dimensional space
        N = len(series)
        embedded = np.zeros((N - (m-1)*lag, m))
        
        for i in range(m):
            embedded[:, i] = series[i*lag:N-(m-1-i)*lag]
        
        # Find nearest neighbors
        distances = pdist(embedded, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Calculate divergence rates
        lyap_estimates = []
        
        for i in range(len(embedded) - 50):  # Need future data to measure divergence
            # Find nearest neighbor (excluding self)
            distance_matrix[i, i] = np.inf
            nearest_idx = np.argmin(distance_matrix[i, :])
            
            initial_dist = distance_matrix[i, nearest_idx]
            if initial_dist < 1e-8:  # Too close, skip
                continue
                
            # Track divergence over time
            divergences = []
            for step in range(1, min(50, len(embedded) - max(i, nearest_idx))):
                if i + step < len(embedded) and nearest_idx + step < len(embedded):
                    future_dist = np.linalg.norm(embedded[i + step] - embedded[nearest_idx + step])
                    if future_dist > 0:
                        divergences.append(np.log(future_dist / initial_dist) / step)
            
            if divergences:
                lyap_estimates.append(np.mean(divergences))
        
        return np.mean(lyap_estimates) if lyap_estimates else np.nan
    
    def calc_hurst_exponent(self, price_series: np.array, max_lag: int = 50) -> float:
        """
        Calculate Hurst exponent for regime detection.
        
        H > 0.5 = trending (ride it)
        H < 0.5 = mean-reverting (fade it) 
        H ≈ 0.5 = random walk (stay out)
        """
        if len(price_series) < max_lag * 2:
            return np.nan
            
        # Use log returns
        returns = np.diff(np.log(price_series))
        
        # Calculate R/S statistic for different lags
        lags = range(2, min(max_lag, len(returns) // 4))
        rs_values = []
        
        for lag in lags:
            # Split into periods of length lag
            periods = len(returns) // lag
            if periods < 2:
                continue
                
            rs_period = []
            for i in range(periods):
                period_returns = returns[i*lag:(i+1)*lag]
                
                # Calculate cumulative deviations from mean
                mean_return = np.mean(period_returns)
                cumdev = np.cumsum(period_returns - mean_return)
                
                # Range (max - min of cumulative deviations)
                R = np.max(cumdev) - np.min(cumdev)
                
                # Standard deviation
                S = np.std(period_returns)
                
                if S > 0:
                    rs_period.append(R / S)
            
            if rs_period:
                rs_values.append((lag, np.mean(rs_period)))
        
        if len(rs_values) < 3:
            return np.nan
            
        # Fit log(R/S) = H * log(lag) + constant
        lags_log = np.log([x[0] for x in rs_values])
        rs_log = np.log([x[1] for x in rs_values if x[1] > 0])
        
        if len(lags_log) != len(rs_log) or len(rs_log) < 3:
            return np.nan
            
        # Linear regression slope = Hurst exponent
        slope, _, _, _, _ = stats.linregress(lags_log[:len(rs_log)], rs_log)
        return slope
    
    def calc_fractal_dimension(self, price_series: np.array) -> float:
        """
        Calculate fractal dimension using box-counting method.
        Lower dimension = more structure = more predictable
        """
        if len(price_series) < 100:
            return np.nan
            
        # Normalize to [0,1]
        series = (price_series - np.min(price_series)) / (np.max(price_series) - np.min(price_series))
        
        # Box sizes (powers of 2)
        box_sizes = [2**i for i in range(1, min(8, int(np.log2(len(series)))))]
        box_counts = []
        
        for box_size in box_sizes:
            count = 0
            # Count boxes that contain part of the curve
            for i in range(0, len(series) - box_size, box_size):
                segment = series[i:i + box_size]
                if len(segment) > 1:
                    y_min, y_max = np.min(segment), np.max(segment)
                    boxes_needed = max(1, int((y_max - y_min) * len(series) / box_size) + 1)
                    count += boxes_needed
            
            box_counts.append(count)
        
        if len(box_counts) < 3:
            return np.nan
            
        # Fit log(count) = -D * log(box_size) + constant
        log_box_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        slope, _, _, _, _ = stats.linregress(log_box_sizes, log_counts)
        return -slope  # Fractal dimension
    
    # ==================== INFORMATION THEORY ====================
    
    def calc_transfer_entropy(self, source: np.array, target: np.array, lag: int = 1, 
                             bins: int = 10) -> float:
        """
        Calculate transfer entropy from source to target.
        Measures directional information flow between assets.
        
        When BTC entropy flows INTO an alt, that alt is about to move.
        """
        if len(source) < 100 or len(target) < 100:
            return np.nan
            
        # Ensure same length
        min_len = min(len(source), len(target))
        source = source[-min_len:]
        target = target[-min_len:]
        
        # Convert to discrete values
        source_disc = pd.cut(source, bins=bins, labels=False)
        target_disc = pd.cut(target, bins=bins, labels=False)
        
        # Remove NaN values
        valid_idx = ~(np.isnan(source_disc) | np.isnan(target_disc))
        source_disc = source_disc[valid_idx]
        target_disc = target_disc[valid_idx]
        
        if len(source_disc) < lag + 20:
            return np.nan
        
        # Create time series: Y(t), Y(t-1), X(t-lag)
        y_t = target_disc[lag:]
        y_t_lag = target_disc[:-lag]
        x_t_lag = source_disc[:-lag]
        
        # Calculate probabilities
        def calc_entropy_3d(x, y, z):
            # Joint probability P(x,y,z)
            xyz = np.column_stack([x, y, z])
            unique_xyz, counts_xyz = np.unique(xyz, axis=0, return_counts=True)
            p_xyz = counts_xyz / len(xyz)
            
            # Marginal probabilities
            xy = np.column_stack([x, y])
            unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
            p_xy = counts_xy / len(xy)
            
            yz = np.column_stack([y, z])
            unique_yz, counts_yz = np.unique(yz, axis=0, return_counts=True)
            p_yz = counts_yz / len(yz)
            
            unique_y, counts_y = np.unique(y, return_counts=True)
            p_y = counts_y / len(y)
            
            # Transfer entropy calculation
            te = 0
            for i, xyz_val in enumerate(unique_xyz):
                p_xyz_val = p_xyz[i]
                
                # Find corresponding marginals
                xy_val = xyz_val[:2]
                yz_val = xyz_val[1:]
                y_val = xyz_val[1]
                
                xy_idx = np.where((unique_xy == xy_val).all(axis=1))[0]
                yz_idx = np.where((unique_yz == yz_val).all(axis=1))[0]
                y_idx = np.where(unique_y == y_val)[0]
                
                if len(xy_idx) > 0 and len(yz_idx) > 0 and len(y_idx) > 0:
                    p_xy_val = p_xy[xy_idx[0]]
                    p_yz_val = p_yz[yz_idx[0]]
                    p_y_val = p_y[y_idx[0]]
                    
                    if p_xy_val > 0 and p_yz_val > 0 and p_y_val > 0:
                        te += p_xyz_val * np.log2((p_xyz_val * p_y_val) / (p_xy_val * p_yz_val))
            
            return te
        
        try:
            te = calc_entropy_3d(x_t_lag, y_t_lag, y_t)
            return te
        except:
            return np.nan
    
    def calc_sample_entropy(self, data: np.array, m: int = 2, r_factor: float = 0.2) -> float:
        """
        Calculate sample entropy.
        Lower entropy = more predictable = tradeable
        """
        if len(data) < 100:
            return np.nan
            
        N = len(data)
        r = r_factor * np.std(data)  # Tolerance level
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j]) <= r:
                        C[i] += 1.0
                        
            phi = np.mean([np.log(c / (N - m + 1.0)) for c in C if c > 0])
            return phi
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            return phi_m - phi_m1
        except:
            return np.nan
    
    def calc_mutual_information(self, volume: np.array, future_returns: np.array, 
                               bins: int = 10) -> float:
        """
        Calculate mutual information between volume and future returns.
        Find when volume actually predicts direction.
        """
        if len(volume) != len(future_returns) or len(volume) < 50:
            return np.nan
            
        # Discretize variables
        volume_disc = pd.cut(volume, bins=bins, labels=False)
        returns_disc = pd.cut(future_returns, bins=bins, labels=False)
        
        # Remove NaN values
        valid_idx = ~(np.isnan(volume_disc) | np.isnan(returns_disc))
        volume_disc = volume_disc[valid_idx]
        returns_disc = returns_disc[valid_idx]
        
        if len(volume_disc) < 20:
            return np.nan
        
        # Calculate mutual information
        try:
            # Joint distribution
            joint_counts = np.histogram2d(volume_disc, returns_disc, bins=bins)[0]
            joint_probs = joint_counts / np.sum(joint_counts)
            
            # Marginal distributions  
            vol_probs = np.sum(joint_probs, axis=1)
            ret_probs = np.sum(joint_probs, axis=0)
            
            # Mutual information
            mi = 0
            for i in range(bins):
                for j in range(bins):
                    if joint_probs[i, j] > 0 and vol_probs[i] > 0 and ret_probs[j] > 0:
                        mi += joint_probs[i, j] * np.log2(joint_probs[i, j] / (vol_probs[i] * ret_probs[j]))
            
            return mi
        except:
            return np.nan
    
    # ==================== ADVANCED STATISTICAL METHODS ====================
    
    def calc_vpin(self, high: np.array, low: np.array, close: np.array, volume: np.array,
                  window: int = 50) -> float:
        """
        Volume-synchronized Probability of Informed Trading.
        Detect smart money before moves.
        """
        if len(close) < window:
            return np.nan
            
        # Calculate volume imbalance
        price_change = np.diff(close)
        volume_buy = volume[1:] * (price_change > 0).astype(float)
        volume_sell = volume[1:] * (price_change < 0).astype(float)
        
        # Volume imbalance
        imbalance = np.abs(volume_buy - volume_sell)
        total_volume = volume_buy + volume_sell
        
        # Avoid division by zero
        vpin_values = np.where(total_volume > 0, imbalance / total_volume, 0)
        
        # Rolling average
        if len(vpin_values) >= window:
            return np.mean(vpin_values[-window:])
        else:
            return np.mean(vpin_values)
    
    def calc_amihud_illiquidity(self, returns: np.array, volume: np.array, 
                               window: int = 20) -> float:
        """
        Amihud illiquidity ratio.
        Sudden liquidity changes precede moves.
        """
        if len(returns) < window or len(volume) < window:
            return np.nan
            
        # Illiquidity = |return| / dollar_volume
        abs_returns = np.abs(returns[-window:])
        dollar_volume = volume[-window:]  # Assuming volume is already in dollar terms
        
        # Avoid division by zero
        illiquidity = np.where(dollar_volume > 0, abs_returns / dollar_volume, 0)
        
        return np.mean(illiquidity)
    
    def fit_ornstein_uhlenbeck(self, price_series: np.array) -> Dict[str, float]:
        """
        Fit Ornstein-Uhlenbeck process for mean reversion trading.
        Returns parameters: theta (mean reversion speed), mu (long-term mean), sigma (volatility)
        """
        if len(price_series) < 100:
            return {"theta": np.nan, "mu": np.nan, "sigma": np.nan, "half_life": np.nan}
            
        # Use log prices for OU process
        log_prices = np.log(price_series)
        
        # Calculate parameters using discrete approximation
        # dX = theta * (mu - X) * dt + sigma * dW
        # X(t+1) = X(t) + theta * (mu - X(t)) * dt + sigma * sqrt(dt) * epsilon
        
        x = log_prices[:-1]
        y = log_prices[1:]
        
        # Linear regression: y = a + b*x
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Convert to OU parameters (assuming dt = 1)
        theta = -np.log(slope)
        mu = intercept / (1 - slope) if slope != 1 else np.mean(log_prices)
        
        # Residual volatility
        residuals = y - (intercept + slope * x)
        sigma = np.std(residuals)
        
        # Half-life of mean reversion
        half_life = np.log(2) / theta if theta > 0 else np.nan
        
        return {
            "theta": theta,
            "mu": mu, 
            "sigma": sigma,
            "half_life": half_life,
            "r_squared": r_value**2
        }
    
    # ==================== MICROSTRUCTURE ANALYSIS ====================
    
    def resample_by_volume(self, df: pd.DataFrame, volume_threshold: int = 1000) -> pd.DataFrame:
        """
        Resample by volume instead of time to reveal hidden patterns.
        """
        if len(df) < 50:
            return df
            
        volume_bars = []
        current_volume = 0
        current_bar = {
            'timestamp_start': df.iloc[0]['timestamp'],
            'timestamp_end': df.iloc[0]['timestamp'],
            'open': df.iloc[0]['open'],
            'high': df.iloc[0]['high'],
            'low': df.iloc[0]['low'],
            'close': df.iloc[0]['close'],
            'volume': 0
        }
        
        for _, row in df.iterrows():
            current_bar['high'] = max(current_bar['high'], row['high'])
            current_bar['low'] = min(current_bar['low'], row['low'])
            current_bar['close'] = row['close']
            current_bar['timestamp_end'] = row['timestamp']
            current_volume += row['volume']
            
            if current_volume >= volume_threshold:
                current_bar['volume'] = current_volume
                volume_bars.append(current_bar.copy())
                
                # Start new bar
                current_volume = 0
                if len(volume_bars) > 0:
                    current_bar = {
                        'timestamp_start': row['timestamp'],
                        'timestamp_end': row['timestamp'],
                        'open': row['close'],  # Next bar opens at current close
                        'high': row['close'],
                        'low': row['close'],
                        'close': row['close'],
                        'volume': 0
                    }
        
        return pd.DataFrame(volume_bars) if volume_bars else df
    
    def calc_kyle_lambda(self, price_impact: np.array, volume: np.array) -> float:
        """
        Kyle's lambda - measure market impact.
        Low lambda = liquid = safe to trade
        """
        if len(price_impact) != len(volume) or len(price_impact) < 20:
            return np.nan
            
        # Remove zero volumes
        non_zero_idx = volume > 0
        price_impact = price_impact[non_zero_idx]
        volume = volume[non_zero_idx]
        
        if len(price_impact) < 10:
            return np.nan
            
        # Lambda = price_impact / signed_volume
        # Simplified: use absolute values
        lambda_values = np.abs(price_impact) / volume
        
        return np.mean(lambda_values)
    
    # ==================== TOOL IMPLEMENTATIONS ====================
    
    def chaos_regime_detector(self, df: pd.DataFrame) -> Dict:
        """
        Chaos theory tool: Trade when system becomes temporarily predictable.
        """
        if len(df) < 200:
            return {"signal": False, "reason": "Insufficient data"}
            
        close = df['close'].values
        
        # Calculate chaos indicators
        lyap = self.calc_lyapunov_exponent(close)
        hurst = self.calc_hurst_exponent(close)
        fractal_dim = self.calc_fractal_dimension(close)
        
        if np.isnan(lyap) or np.isnan(hurst):
            return {"signal": False, "reason": "Invalid chaos metrics"}
        
        # Signal conditions
        predictable_window = lyap < -0.01  # Negative Lyapunov = predictable
        trending_regime = hurst > 0.55     # Persistent trends
        low_complexity = fractal_dim < 1.4  # Lower dimensional = more structure
        
        if predictable_window and trending_regime and low_complexity:
            # Direction based on recent momentum
            recent_return = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
            direction = "long" if recent_return > 1 else "short" if recent_return < -1 else None
            
            if direction:
                score = abs(lyap) * 50 + (hurst - 0.5) * 100 + (1.5 - fractal_dim) * 30
                return {
                    "signal": True,
                    "direction": direction,
                    "score": score,
                    "reason": f"CHAOS PREDICTABLE: Lyap={lyap:.4f}, Hurst={hurst:.3f}, FD={fractal_dim:.3f}"
                }
        
        return {"signal": False, "reason": f"Chaotic: Lyap={lyap:.4f}, Hurst={hurst:.3f}"}
    
    def information_flow_detector(self, pair: str, df: pd.DataFrame) -> Dict:
        """
        Information theory tool: Detect BTC → alt information flow.
        """
        if pair == "BTCUSDT" or self.btc_data is None or len(df) < 100:
            return {"signal": False, "reason": "BTC pair or insufficient data"}
            
        # Align data lengths
        min_len = min(len(df), len(self.btc_data))
        alt_returns = df['returns'].values[-min_len:]
        btc_returns = self.btc_data['returns'].values[-min_len:]
        
        # Calculate transfer entropy BTC -> ALT
        te_btc_to_alt = self.calc_transfer_entropy(btc_returns, alt_returns, lag=1)
        te_alt_to_btc = self.calc_transfer_entropy(alt_returns, btc_returns, lag=1)
        
        # Calculate sample entropy of alt (predictability)
        alt_entropy = self.calc_sample_entropy(alt_returns[-100:])
        
        if np.isnan(te_btc_to_alt) or np.isnan(alt_entropy):
            return {"signal": False, "reason": "Invalid entropy calculations"}
        
        # Signal: High BTC->alt flow + Low alt entropy = alt about to move
        high_btc_flow = te_btc_to_alt > 0.1
        predictable_alt = alt_entropy < 0.5  # Low entropy = predictable
        btc_dominant = te_btc_to_alt > te_alt_to_btc * 1.5  # BTC leads
        
        # Recent BTC momentum
        btc_momentum = np.mean(btc_returns[-24:]) if len(btc_returns) >= 24 else 0
        
        if high_btc_flow and predictable_alt and btc_dominant and abs(btc_momentum) > 0.001:
            direction = "long" if btc_momentum > 0 else "short"
            score = te_btc_to_alt * 100 + (0.6 - alt_entropy) * 50 + abs(btc_momentum) * 1000
            
            return {
                "signal": True,
                "direction": direction,
                "score": score,
                "reason": f"INFO FLOW: BTC→ALT TE={te_btc_to_alt:.3f}, entropy={alt_entropy:.3f}"
            }
        
        return {"signal": False, "reason": f"No flow: TE={te_btc_to_alt:.3f}"}
    
    def mean_reversion_ou(self, df: pd.DataFrame) -> Dict:
        """
        Ornstein-Uhlenbeck mean reversion for choppy markets.
        """
        if len(df) < 150:
            return {"signal": False, "reason": "Insufficient data"}
            
        close = df['close'].values
        
        # Fit OU process to recent data
        ou_params = self.fit_ornstein_uhlenbeck(close[-100:])
        
        if np.isnan(ou_params["theta"]) or ou_params["theta"] <= 0:
            return {"signal": False, "reason": "Invalid OU parameters"}
        
        # Current price vs equilibrium
        current_price = close[-1]
        equilibrium_price = np.exp(ou_params["mu"])
        deviation = (current_price - equilibrium_price) / equilibrium_price
        
        # Signal when price deviates beyond statistical bounds
        threshold = 2 * ou_params["sigma"]  # 2-sigma threshold
        mean_reverting = ou_params["r_squared"] > 0.3  # Good fit to OU process
        fast_reversion = ou_params["half_life"] < 50  # Reasonable reversion speed
        
        if mean_reverting and fast_reversion and abs(deviation) > threshold:
            direction = "short" if deviation > 0 else "long"  # Fade extremes
            score = abs(deviation) / threshold * 50 + ou_params["r_squared"] * 30
            
            return {
                "signal": True,
                "direction": direction,
                "score": score,
                "reason": f"OU REVERSION: dev={deviation:.3f}, σ={ou_params['sigma']:.3f}, HL={ou_params['half_life']:.1f}"
            }
        
        return {"signal": False, "reason": f"No reversion: dev={deviation:.3f}"}
    
    def smart_money_vpin(self, df: pd.DataFrame) -> Dict:
        """
        VPIN-based smart money detection tool.
        """
        if len(df) < 100:
            return {"signal": False, "reason": "Insufficient data"}
            
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values  
        volume = df['volume'].values
        
        # Calculate VPIN
        vpin = self.calc_vpin(high, low, close, volume)
        
        if np.isnan(vpin):
            return {"signal": False, "reason": "Invalid VPIN"}
        
        # Also calculate recent Amihud illiquidity
        returns = df['returns'].values
        illiquidity = self.calc_amihud_illiquidity(returns, volume)
        
        # Signal: Low VPIN (informed trading) + improving liquidity + momentum
        informed_trading = vpin > 0.6  # High VPIN = informed traders active
        recent_momentum = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
        
        if informed_trading and abs(recent_momentum) > 2:
            direction = "long" if recent_momentum > 0 else "short"
            score = vpin * 50 + abs(recent_momentum) * 2
            
            return {
                "signal": True,
                "direction": direction,
                "score": score,
                "reason": f"SMART MONEY: VPIN={vpin:.3f}, momentum={recent_momentum:.1f}%"
            }
        
        return {"signal": False, "reason": f"No smart money: VPIN={vpin:.3f}"}
    
    def volume_clock_pattern(self, df: pd.DataFrame) -> Dict:
        """
        Volume clock analysis - resample by volume to find hidden patterns.
        """
        if len(df) < 100:
            return {"signal": False, "reason": "Insufficient data"}
            
        # Resample by volume
        median_volume = df['volume'].median()
        volume_threshold = median_volume * 5  # 5x median volume per bar
        
        volume_df = self.resample_by_volume(df, volume_threshold)
        
        if len(volume_df) < 20:
            return {"signal": False, "reason": "Insufficient volume bars"}
        
        # Analyze volume-based patterns
        vol_close = volume_df['close'].values
        vol_returns = np.diff(vol_close) / vol_close[:-1]
        
        # Look for momentum in volume-time
        if len(vol_returns) >= 5:
            recent_vol_momentum = np.mean(vol_returns[-5:])
            vol_volatility = np.std(vol_returns[-10:]) if len(vol_returns) >= 10 else np.nan
            
            # Signal: Strong volume-based momentum with low volatility
            if not np.isnan(vol_volatility) and abs(recent_vol_momentum) > 0.02 and vol_volatility < 0.05:
                direction = "long" if recent_vol_momentum > 0 else "short"
                score = abs(recent_vol_momentum) * 1000 + (0.1 - vol_volatility) * 100
                
                return {
                    "signal": True,
                    "direction": direction,
                    "score": score,
                    "reason": f"VOLUME CLOCK: vol_momentum={recent_vol_momentum:.4f}, vol_vol={vol_volatility:.4f}"
                }
        
        return {"signal": False, "reason": "No volume pattern"}
    
    def mutual_info_volume_predictor(self, df: pd.DataFrame) -> Dict:
        """
        Mutual information between volume and future returns.
        """
        if len(df) < 100:
            return {"signal": False, "reason": "Insufficient data"}
            
        volume = df['volume'].values[:-24]  # All but last 24 hours
        future_returns = df['returns'].values[24:]  # Future 24h returns
        
        # Calculate 4h future returns  
        if len(df) >= 28:
            future_4h = (df['close'].shift(-4) / df['close'] - 1).values[:-4]
            volume_4h = volume[:-20] if len(volume) > 20 else volume
            
            # Align lengths
            min_len = min(len(volume_4h), len(future_4h))
            volume_4h = volume_4h[-min_len:]
            future_4h = future_4h[-min_len:]
            
            mi = self.calc_mutual_information(volume_4h, future_4h)
            
            if not np.isnan(mi) and mi > 0.1:  # Significant mutual information
                # Current volume pattern
                current_volume_z = (df['volume'].iloc[-1] - df['volume'].mean()) / df['volume'].std()
                recent_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0
                
                # Signal when high MI + unusual volume + momentum
                if abs(current_volume_z) > 1.5 and abs(recent_momentum) > 1:
                    direction = "long" if recent_momentum > 0 else "short"
                    score = mi * 100 + abs(current_volume_z) * 10 + abs(recent_momentum)
                    
                    return {
                        "signal": True,
                        "direction": direction,
                        "score": score,
                        "reason": f"MUTUAL INFO: MI={mi:.3f}, vol_z={current_volume_z:.2f}"
                    }
        
        return {"signal": False, "reason": "Low mutual information"}


def backtest_tool(tool_func, pair: str, data: pd.DataFrame, 
                  hold_hours: int = 8, stop_loss: float = 0.05) -> Dict:
    """Backtest a single tool on one pair."""
    
    # OOS period: second half of data (bars 4380-8760)
    split_point = len(data) // 2
    oos_data = data.iloc[split_point:].copy().reset_index(drop=True)
    
    trades = []
    tools = UnconventionalTools()
    
    # Load BTC data for cross-asset analysis
    if pair != "BTCUSDT":
        try:
            tools.load_data("BTCUSDT")
        except:
            pass  # BTC data not needed for some tools
    
    for i in range(100, len(oos_data) - hold_hours - 1):  # Need future data for exits
        # Get signal from tool
        window_data = oos_data.iloc[:i+1].copy()
        signal = tool_func(tools, window_data, pair)
        
        if signal.get("signal", False):
            entry_price = oos_data.iloc[i]['close']
            direction = signal["direction"]
            score = signal.get("score", 0)
            
            # Calculate exit
            if direction == "long":
                # Stop loss
                stop_price = entry_price * (1 - stop_loss)
                
                # Check for stop loss hit
                exit_idx = None
                for j in range(i + 1, min(i + hold_hours + 1, len(oos_data))):
                    if oos_data.iloc[j]['low'] <= stop_price:
                        exit_idx = j
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                        break
                
                # If no stop loss, exit at hold time
                if exit_idx is None:
                    exit_idx = min(i + hold_hours, len(oos_data) - 1)
                    exit_price = oos_data.iloc[exit_idx]['close']
                    exit_reason = "time_exit"
                
                # Calculate return
                gross_return = (exit_price - entry_price) / entry_price
                net_return = gross_return - FEES  # Subtract fees
                
            else:  # short
                # Stop loss
                stop_price = entry_price * (1 + stop_loss)
                
                # Check for stop loss hit
                exit_idx = None
                for j in range(i + 1, min(i + hold_hours + 1, len(oos_data))):
                    if oos_data.iloc[j]['high'] >= stop_price:
                        exit_idx = j
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                        break
                
                # If no stop loss, exit at hold time
                if exit_idx is None:
                    exit_idx = min(i + hold_hours, len(oos_data) - 1)
                    exit_price = oos_data.iloc[exit_idx]['close']
                    exit_reason = "time_exit"
                
                # Calculate return (short)
                gross_return = (entry_price - exit_price) / entry_price  
                net_return = gross_return - FEES  # Subtract fees
            
            trades.append({
                'entry_idx': i,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'gross_return': gross_return,
                'net_return': net_return,
                'score': score,
                'reason': signal.get("reason", ""),
                'exit_reason': exit_reason
            })
    
    # Calculate metrics
    if not trades:
        return {
            'pair': pair,
            'total_signals': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'max_dd': 0,
            'profit_factor': 0,
            'sharpe': 0,
            'trades': []
        }
    
    returns = [t['net_return'] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    
    win_rate = len(wins) / len(returns) * 100
    avg_return = np.mean(returns) * 100
    total_return = np.sum(returns) * 100
    
    # Max drawdown
    cumulative = np.cumsum(returns)
    max_dd = 0
    peak = cumulative[0]
    for ret in cumulative:
        peak = max(peak, ret)
        dd = peak - ret
        max_dd = max(max_dd, dd)
    max_dd *= 100
    
    # Profit factor
    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 1e-6
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    # Sharpe ratio (annualized, assuming 8760 hours per year)
    if len(returns) > 1:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    return {
        'pair': pair,
        'total_signals': len(trades),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return, 
        'max_dd': max_dd,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'trades': trades
    }


def validate_all_tools():
    """Run OOS validation on all unconventional tools."""
    
    print("=" * 80)
    print("BULL TOOLS V2 - UNCONVENTIONAL APPROACHES VALIDATION")
    print("=" * 80)
    print()
    
    tools = UnconventionalTools()
    
    # Tool configurations: (function, hold_hours, stop_loss)
    tool_configs = {
        'chaos_regime_detector': (
            lambda tools, df, pair: tools.chaos_regime_detector(df),
            8, 0.06
        ),
        'information_flow_detector': (
            lambda tools, df, pair: tools.information_flow_detector(pair, df),
            12, 0.05
        ),
        'mean_reversion_ou': (
            lambda tools, df, pair: tools.mean_reversion_ou(df),
            6, 0.04
        ),
        'smart_money_vpin': (
            lambda tools, df, pair: tools.smart_money_vpin(df),
            8, 0.05
        ),
        'volume_clock_pattern': (
            lambda tools, df, pair: tools.volume_clock_pattern(df),
            8, 0.05
        ),
        'mutual_info_volume': (
            lambda tools, df, pair: tools.mutual_info_volume_predictor(df),
            6, 0.04
        )
    }
    
    results = {}
    
    for tool_name, (tool_func, hold_hours, stop_loss) in tool_configs.items():
        print(f"\n🔍 Testing {tool_name}...")
        tool_results = []
        
        for pair in PAIRS:
            try:
                data = tools.load_data(pair)
                result = backtest_tool(tool_func, pair, data, hold_hours, stop_loss)
                tool_results.append(result)
                
                if result['total_signals'] > 0:
                    print(f"   {pair}: {result['total_signals']} signals, {result['win_rate']:.1f}% WR, {result['avg_return']:+.2f}% avg")
                else:
                    print(f"   {pair}: No signals")
                    
            except Exception as e:
                print(f"   {pair}: Error - {e}")
                continue
        
        results[tool_name] = tool_results
    
    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze validation results and determine which tools pass."""
    
    print("\n" + "=" * 80)
    print("VALIDATION ANALYSIS")
    print("=" * 80)
    
    passed_tools = {}
    failed_tools = {}
    
    for tool_name, tool_results in results.items():
        print(f"\n📊 {tool_name.upper()}:")
        
        # Aggregate stats across all pairs
        total_signals = sum(r['total_signals'] for r in tool_results)
        
        if total_signals == 0:
            print(f"   ❌ FAILED: No signals generated")
            failed_tools[tool_name] = {"reason": "No signals", "signals": 0}
            continue
        
        if total_signals < MIN_SIGNALS:
            print(f"   ❌ FAILED: Only {total_signals} signals (need {MIN_SIGNALS}+)")
            failed_tools[tool_name] = {"reason": f"Insufficient signals ({total_signals})", "signals": total_signals}
            continue
        
        # Calculate weighted averages
        weighted_returns = []
        winning_pairs = 0
        
        for result in tool_results:
            if result['total_signals'] > 0:
                # Weight by number of signals
                for _ in range(result['total_signals']):
                    weighted_returns.append(result['avg_return'])
                
                # Check if pair passes individual criteria
                if result['win_rate'] >= MIN_WIN_RATE or result['profit_factor'] >= MIN_PROFIT_FACTOR:
                    winning_pairs += 1
        
        if not weighted_returns:
            failed_tools[tool_name] = {"reason": "No valid results", "signals": total_signals}
            continue
            
        avg_win_rate = np.mean([r['win_rate'] for r in tool_results if r['total_signals'] > 0])
        avg_return = np.mean(weighted_returns)
        avg_profit_factor = np.mean([r['profit_factor'] for r in tool_results if r['total_signals'] > 0])
        
        print(f"   📈 Total signals: {total_signals}")
        print(f"   📈 Average win rate: {avg_win_rate:.1f}%")
        print(f"   📈 Average return: {avg_return:+.2f}%")
        print(f"   📈 Average profit factor: {avg_profit_factor:.2f}")
        print(f"   📈 Passing pairs: {winning_pairs}/{len([r for r in tool_results if r['total_signals'] > 0])}")
        
        # Check validation criteria
        passes_wr = avg_win_rate >= MIN_WIN_RATE
        passes_pf = avg_profit_factor >= MIN_PROFIT_FACTOR
        has_enough_signals = total_signals >= MIN_SIGNALS
        
        if (passes_wr or passes_pf) and has_enough_signals and winning_pairs >= 1:
            tier = "T2" if avg_win_rate >= 65 and total_signals >= 30 else "T3"
            print(f"   ✅ PASSED ({tier}): Meets validation criteria")
            
            passed_tools[tool_name] = {
                "tier": tier,
                "total_signals": total_signals,
                "avg_win_rate": avg_win_rate,
                "avg_return": avg_return,
                "avg_profit_factor": avg_profit_factor,
                "winning_pairs": winning_pairs,
                "results": tool_results
            }
        else:
            reasons = []
            if not passes_wr and not passes_pf:
                reasons.append(f"WR={avg_win_rate:.1f}% (need {MIN_WIN_RATE}%) and PF={avg_profit_factor:.2f} (need {MIN_PROFIT_FACTOR})")
            if not has_enough_signals:
                reasons.append(f"Only {total_signals} signals")
            if winning_pairs == 0:
                reasons.append("No pairs pass individual criteria")
                
            print(f"   ❌ FAILED: {', '.join(reasons)}")
            failed_tools[tool_name] = {
                "reason": ', '.join(reasons),
                "signals": total_signals,
                "win_rate": avg_win_rate,
                "profit_factor": avg_profit_factor
            }
    
    return {"passed": passed_tools, "failed": failed_tools}


if __name__ == "__main__":
    # Run validation
    print("🚀 Starting unconventional bull tools validation...")
    print(f"📊 Validation criteria: {MIN_SIGNALS}+ signals, {MIN_WIN_RATE}%+ WR OR {MIN_PROFIT_FACTOR}+ PF")
    print(f"💰 Fees: {FEES*100:.2f}% round-trip")
    print()
    
    results = validate_all_tools()
    analysis = analyze_results(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ PASSED TOOLS: {len(analysis['passed'])}")
    for tool, data in analysis['passed'].items():
        print(f"   🏆 {tool} ({data['tier']}): {data['total_signals']} signals, {data['avg_win_rate']:.1f}% WR")
    
    print(f"\n❌ FAILED TOOLS: {len(analysis['failed'])}")  
    for tool, data in analysis['failed'].items():
        print(f"   💀 {tool}: {data['reason']}")
    
    # Save detailed results
    import json
    with open("/Users/lucasaust/code/Crypto-trading-bot/unconventional_tools_results.json", "w") as f:
        # Convert numpy types to native Python for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = {}
        for tool, tool_results in results.items():
            json_results[tool] = []
            for result in tool_results:
                json_result = {}
                for key, value in result.items():
                    if key == 'trades':
                        json_result[key] = [
                            {k: convert_numpy(v) for k, v in trade.items()}
                            for trade in value
                        ]
                    else:
                        json_result[key] = convert_numpy(value)
                json_results[tool].append(json_result)
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: unconventional_tools_results.json")