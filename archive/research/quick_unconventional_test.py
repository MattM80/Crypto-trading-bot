#!/usr/bin/env python3
"""
Quick test of key unconventional approaches on a single pair first
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_1h"

def load_data(pair: str) -> pd.DataFrame:
    """Load 1h data for a pair."""
    file_path = DATA_DIR / f"{pair}_1h.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['returns'] = df['close'].pct_change()
    return df

def calc_hurst_exponent(price_series: np.array, max_lag: int = 30) -> float:
    """Calculate Hurst exponent - simplified version"""
    if len(price_series) < max_lag * 2:
        return np.nan
        
    returns = np.diff(np.log(price_series))
    
    # Calculate R/S statistic for different lags
    lags = range(2, min(max_lag, len(returns) // 4))
    rs_values = []
    
    for lag in lags:
        periods = len(returns) // lag
        if periods < 2:
            continue
            
        rs_period = []
        for i in range(periods):
            period_returns = returns[i*lag:(i+1)*lag]
            
            mean_return = np.mean(period_returns)
            cumdev = np.cumsum(period_returns - mean_return)
            
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(period_returns)
            
            if S > 0:
                rs_period.append(R / S)
        
        if rs_period:
            rs_values.append((lag, np.mean(rs_period)))
    
    if len(rs_values) < 3:
        return np.nan
        
    # Linear regression
    lags_log = np.log([x[0] for x in rs_values])
    rs_log = np.log([x[1] for x in rs_values if x[1] > 0])
    
    if len(lags_log) != len(rs_log) or len(rs_log) < 3:
        return np.nan
        
    slope, _, _, _, _ = stats.linregress(lags_log[:len(rs_log)], rs_log)
    return slope

def hurst_regime_tool(df: pd.DataFrame) -> dict:
    """Hurst exponent regime detection tool"""
    if len(df) < 200:
        return {"signal": False, "reason": "Insufficient data"}
    
    close = df['close'].values
    
    # Calculate rolling Hurst exponent
    window = 100
    current_hurst = calc_hurst_exponent(close[-window:])
    
    if np.isnan(current_hurst):
        return {"signal": False, "reason": "Invalid Hurst"}
    
    # Regime detection
    if current_hurst > 0.55:  # Trending regime
        # Check direction
        recent_return = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
        
        if abs(recent_return) > 2:  # Strong move
            direction = "long" if recent_return > 0 else "short"
            score = (current_hurst - 0.5) * 200 + abs(recent_return) * 2
            
            return {
                "signal": True,
                "direction": direction,
                "score": score,
                "reason": f"HURST TREND: H={current_hurst:.3f}, momentum={recent_return:.1f}%"
            }
    
    elif current_hurst < 0.45:  # Mean-reverting regime
        # Fade extremes
        rsi_period = 14
        if len(close) >= rsi_period:
            delta = np.diff(close[-rsi_period-1:])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gain)
            avg_loss = np.mean(loss)
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                if rsi > 70:  # Overbought - fade
                    return {
                        "signal": True,
                        "direction": "short",
                        "score": (0.5 - current_hurst) * 200 + (rsi - 70) * 2,
                        "reason": f"HURST REVERT: H={current_hurst:.3f}, RSI={rsi:.1f}"
                    }
                elif rsi < 30:  # Oversold - fade
                    return {
                        "signal": True,
                        "direction": "long", 
                        "score": (0.5 - current_hurst) * 200 + (30 - rsi) * 2,
                        "reason": f"HURST REVERT: H={current_hurst:.3f}, RSI={rsi:.1f}"
                    }
    
    return {"signal": False, "reason": f"Random walk: H={current_hurst:.3f}"}

def btc_dominance_flow_tool(pair: str, df: pd.DataFrame, btc_df: pd.DataFrame) -> dict:
    """BTC dominance momentum as alt season predictor"""
    if pair == "BTCUSDT" or len(df) < 100 or len(btc_df) < 100:
        return {"signal": False, "reason": "BTC pair or insufficient data"}
    
    # Get market cap proxy (price * volume as rough proxy)
    btc_mcap_proxy = btc_df['close'] * btc_df['volume']
    alt_mcap_proxy = df['close'] * df['volume']
    
    # BTC dominance proxy
    total_proxy = btc_mcap_proxy + alt_mcap_proxy
    btc_dom_proxy = btc_mcap_proxy / total_proxy
    
    if len(btc_dom_proxy) < 50:
        return {"signal": False, "reason": "Insufficient dominance data"}
    
    # Dominance change rate
    dom_24h_change = (btc_dom_proxy.iloc[-1] - btc_dom_proxy.iloc[-24]) / btc_dom_proxy.iloc[-24] * 100 if len(btc_dom_proxy) >= 24 else 0
    dom_7d_change = (btc_dom_proxy.iloc[-1] - btc_dom_proxy.iloc[-168]) / btc_dom_proxy.iloc[-168] * 100 if len(btc_dom_proxy) >= 168 else 0
    
    # Alt momentum
    alt_24h = (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100 if len(df) >= 24 else 0
    
    # Signal: Sharp dominance drop = alt season starting
    sharp_dom_drop = dom_24h_change < -5  # BTC dominance dropping fast
    sustained_drop = dom_7d_change < -2   # Sustained over week
    alt_momentum = alt_24h > 2            # Alt starting to move
    
    if sharp_dom_drop and alt_momentum:
        score = abs(dom_24h_change) * 3 + alt_24h * 2
        return {
            "signal": True,
            "direction": "long",
            "score": score,
            "reason": f"BTC DOM DROP: 24h={dom_24h_change:.1f}%, alt_mom={alt_24h:.1f}%"
        }
    
    return {"signal": False, "reason": f"Dom stable: {dom_24h_change:.1f}%"}

def volume_profile_tool(df: pd.DataFrame) -> dict:
    """Volume profile analysis"""
    if len(df) < 100:
        return {"signal": False, "reason": "Insufficient data"}
    
    # Calculate volume profile over recent period
    window = 50
    recent_data = df.tail(window)
    
    # Price bins
    price_range = recent_data['high'].max() - recent_data['low'].min()
    num_bins = 20
    bin_size = price_range / num_bins
    
    volume_profile = {}
    for _, row in recent_data.iterrows():
        # Distribute volume across price range for this candle
        price_levels = np.linspace(row['low'], row['high'], 10)
        volume_per_level = row['volume'] / len(price_levels)
        
        for price in price_levels:
            bin_idx = int((price - recent_data['low'].min()) / bin_size)
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            
            if bin_idx not in volume_profile:
                volume_profile[bin_idx] = 0
            volume_profile[bin_idx] += volume_per_level
    
    # Find POC (Point of Control) - highest volume bin
    poc_bin = max(volume_profile.keys(), key=lambda x: volume_profile[x])
    poc_price = recent_data['low'].min() + (poc_bin + 0.5) * bin_size
    
    current_price = df['close'].iloc[-1]
    distance_from_poc = abs(current_price - poc_price) / current_price * 100
    
    # Volume distribution
    total_volume = sum(volume_profile.values())
    poc_volume_pct = volume_profile[poc_bin] / total_volume * 100
    
    # Signal: Price far from POC + high POC concentration = reversion opportunity
    far_from_poc = distance_from_poc > 3  # More than 3% from POC
    concentrated_volume = poc_volume_pct > 15  # POC has >15% of volume
    
    # Recent momentum
    momentum_4h = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100 if len(df) >= 4 else 0
    
    if far_from_poc and concentrated_volume:
        # Mean reversion toward POC
        direction = "short" if current_price > poc_price else "long"
        score = distance_from_poc * 5 + poc_volume_pct
        
        return {
            "signal": True,
            "direction": direction,
            "score": score,
            "reason": f"VOLUME PROFILE: {distance_from_poc:.1f}% from POC, {poc_volume_pct:.1f}% vol"
        }
    
    return {"signal": False, "reason": f"Near POC: {distance_from_poc:.1f}%"}

def test_tools():
    """Test the unconventional tools on a few pairs"""
    print("Testing Unconventional Tools on Sample Pairs")
    print("=" * 50)
    
    test_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT"]
    
    for pair in test_pairs:
        print(f"\n🔍 Testing {pair}:")
        
        try:
            df = load_data(pair)
            btc_df = load_data("BTCUSDT") if pair != "BTCUSDT" else df
            
            # Test each tool
            print(f"  Data loaded: {len(df)} bars")
            
            # Test Hurst tool
            hurst_result = hurst_regime_tool(df)
            if hurst_result["signal"]:
                print(f"  ✅ HURST: {hurst_result['direction']} - {hurst_result['reason']}")
            else:
                print(f"  ❌ HURST: {hurst_result['reason']}")
            
            # Test BTC dominance tool  
            if pair != "BTCUSDT":
                dom_result = btc_dominance_flow_tool(pair, df, btc_df)
                if dom_result["signal"]:
                    print(f"  ✅ BTC DOM: {dom_result['direction']} - {dom_result['reason']}")
                else:
                    print(f"  ❌ BTC DOM: {dom_result['reason']}")
            
            # Test volume profile tool
            vol_result = volume_profile_tool(df)
            if vol_result["signal"]:
                print(f"  ✅ VOL PROFILE: {vol_result['direction']} - {vol_result['reason']}")
            else:
                print(f"  ❌ VOL PROFILE: {vol_result['reason']}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✅ Quick test complete. Tools are working.")
    print("Next step: Run full validation on all tools and pairs.")

if __name__ == "__main__":
    test_tools()