#!/usr/bin/env python3
"""
Enhanced Trading Tools with Additional Filters
Based on OOS validation results, these tools add multiple confirmation layers
to improve win rates from ~30-40% to target 50%+.
"""

def scan_enhanced_signals(self, pair: str, data: dict) -> List[Tuple[dict, float]]:
    """Enhanced version of scan_signals with improved tools based on OOS validation."""
    signals = []
    df = data['df']
    price = data['price']
    
    if len(df) < 50:
        return signals
        
    # Compute features once (same as original)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    rsi7 = self.calc_rsi(close, 7)
    rsi14 = self.calc_rsi(close, 14)
    sma50 = self.calc_sma(close, 50)
    sma20 = self.calc_sma(close, 20)
    ema5 = self.calc_ema(close, 5)
    ema13 = self.calc_ema(close, 13)
    bb_mid, bb_upper, bb_lower, bb_bandwidth = self.calc_bollinger(close, 20, 2.0)
    atr14 = self.calc_atr(high, low, close, 14)
    
    # Current values
    cur_rsi = rsi7[-1]
    cur_rsi14 = rsi14[-1]
    cur_vs_sma50 = (price - sma50[-1]) / sma50[-1] * 100 if not np.isnan(sma50[-1]) and sma50[-1] > 0 else 0
    cur_vs_sma20 = (price - sma20[-1]) / sma20[-1] * 100 if not np.isnan(sma20[-1]) and sma20[-1] > 0 else 0
    cur_atr_pct = atr14[-1] / price * 100 if price > 0 and not np.isnan(atr14[-1]) else 0
    
    # Returns
    ret_4h = (price - close[-2]) / close[-2] * 100 if len(close) >= 2 else 0
    ret_8h = (price - close[-3]) / close[-3] * 100 if len(close) >= 3 else 0
    ret_12h = (price - close[-4]) / close[-4] * 100 if len(close) >= 4 else 0
    ret_24h = (price - close[-7]) / close[-7] * 100 if len(close) >= 7 else 0
    
    # Volume analysis
    avg_vol_20 = np.mean(volume[-21:-1]) if len(volume) >= 21 else np.mean(volume) if len(volume) > 0 else 1
    vol_ratio = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0
    
    # Multi-timeframe trend context
    higher_tf_bullish = False
    higher_tf_bearish = False
    if len(close) >= 50:
        sma12 = np.mean(close[-12:])
        sma12_prev = np.mean(close[-16:-4])
        higher_tf_bullish = sma12 > sma12_prev
        higher_tf_bearish = sma12 < sma12_prev
    
    # Additional context indicators
    bb_position = (price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if not np.isnan(bb_upper[-1]) and bb_upper[-1] > bb_lower[-1] else 0.5
    is_green_candle = close[-1] > df['open'].values[-1] if len(df) > 0 else True
    
    # Count consecutive red/green candles
    consecutive_red = 0
    if len(close) >= 5:
        opens = df['open'].values[-5:]
        closes = close[-5:]
        for i in range(len(closes)-1, 0, -1):
            if closes[i] < opens[i]:
                consecutive_red += 1
            else:
                break
    
    # ENHANCED TOOLS WITH MULTIPLE FILTERS
    
    # Tool 2: Enhanced Crash Buy (OOS-validated: was 29.7% WR, target 50%+)
    # Original: ret_24h < -10 and cur_rsi < 20
    # Enhanced: Add volume spike, not in strong downtrend, wait for first bounce
    if (ret_24h < -10 and cur_rsi < 20 and 
        vol_ratio > 1.5 and  # Volume confirmation
        price > sma50[-1] * 0.85 and  # Not in complete breakdown (>15% below SMA50)
        ret_4h > -5 and  # Recent bounce started (not still crashing)
        consecutive_red >= 2):  # Confirmed selling pressure before
        
        score = (20 - cur_rsi) * 2 * min(vol_ratio, 3.0)  # Volume boost
        signals.append(({
            'pair': pair, 'tool': 'crash_buy_enhanced', 'direction': 'long',
            'hold': 24, 'sl_pct': 0.05,
            'reason': f"ENHANCED CRASH BUY: {ret_24h:.1f}% drop, RSI={cur_rsi:.1f}, vol={vol_ratio:.1f}x, bounce started"
        }, score))
    
    # Tool 3: Enhanced Volatile Oversold (OOS-validated: was 31.5% WR)
    # Original: cur_atr_pct > 3 and cur_rsi < 25  
    # Enhanced: Higher volatility threshold, confirm with BB position, avoid strong downtrends
    if (cur_atr_pct > 5 and cur_rsi < 20 and
        bb_position < 0.2 and  # Near Bollinger lower band
        cur_vs_sma50 > -20 and  # Not in extreme downtrend
        vol_ratio > 2.0):  # High volume
        
        score = cur_atr_pct * (25 - cur_rsi) * min(vol_ratio, 2.0)
        signals.append(({
            'pair': pair, 'tool': 'volatile_oversold_enhanced', 'direction': 'long',
            'hold': 24, 'sl_pct': 0.08,
            'reason': f"ENHANCED VOLATILE OVERSOLD: ATR={cur_atr_pct:.1f}%, RSI={cur_rsi:.1f}, BB_pos={bb_position:.2f}"
        }, score))
    
    # Tool 4: Enhanced Relief Rally (OOS-validated: was 30.2% WR, -1.248% net)
    # Original was broken - fix by requiring stronger setup
    # Enhanced: Stronger overbought + confirmed downtrend + volume
    if (cur_rsi > 80 and cur_vs_sma50 < -5 and  # More oversold vs SMA50
        higher_tf_bearish and  # Confirmed downtrend
        vol_ratio > 1.5 and  # Volume spike
        consecutive_red <= 1):  # Not in middle of crash
        
        score = (cur_rsi - 80) * 2.0
        signals.append(({
            'pair': pair, 'tool': 'relief_rally_enhanced', 'direction': 'long',
            'hold': 12, 'sl_pct': 0.03,
            'reason': f"ENHANCED RELIEF RALLY: RSI={cur_rsi:.1f}, {cur_vs_sma50:.1f}% below SMA50, downtrend"
        }, score))
    
    # Tool 6: Enhanced Dip Buy (OOS-validated: was 40.6% WR - closest to passing)
    # Original: ret_4h < -3
    # Enhanced: Add volume confirmation, trend context, momentum
    if (ret_4h < -4 and  # Slightly deeper dip
        vol_ratio > 1.8 and  # Volume confirmation
        cur_vs_sma20 > -10 and  # Not too far from SMA20
        (higher_tf_bullish or cur_vs_sma50 > 0) and  # Uptrend context
        cur_rsi < 40):  # Not overbought
        
        score = abs(ret_4h) * 2 * min(vol_ratio, 2.0)
        signals.append(({
            'pair': pair, 'tool': 'dip_buy_enhanced', 'direction': 'long',
            'hold': 24, 'sl_pct': 0.04,
            'reason': f"ENHANCED DIP BUY: {ret_4h:.1f}% drop, vol={vol_ratio:.1f}x, uptrend context"
        }, score))
    
    # Tool 7: Enhanced RSI Pump Short (OOS-validated: 48.5% WR - PASSED but can improve)
    # Original: cur_rsi > 80 and ret_12h > 8
    # Enhanced: Add Bollinger Band break, volume confirmation
    if (cur_rsi > 80 and ret_12h > 8 and
        price > bb_upper[-1] and  # Above Bollinger upper
        vol_ratio > 2.0 and  # Volume confirmation  
        consecutive_red == 0):  # Still in uptrend (not reversing yet)
        
        score = 35 + (cur_rsi - 80) * 0.5 + min(vol_ratio, 3.0)
        signals.append(({
            'pair': pair, 'tool': 'rsi_pump_short_enhanced', 'direction': 'short',
            'hold': 8, 'sl_pct': 0.05,
            'reason': f"ENHANCED RSI PUMP SHORT: RSI={cur_rsi:.1f}, +{ret_12h:.1f}% 12h, above BB upper"
        }, score))
    
    # Tool 8: Enhanced Mega Crash (OOS-validated: 36.3% WR)
    # Original: ret_24h < -15
    # Enhanced: Extreme drop + volume spike + momentum shift
    if (ret_24h < -20 and  # More extreme threshold
        vol_ratio > 3.0 and  # High volume
        ret_4h > -3 and  # Bouncing (not still crashing)
        bb_position < 0.1):  # Extreme BB position
        
        score = abs(ret_24h) * 3 * min(vol_ratio / 3.0, 2.0)
        signals.append(({
            'pair': pair, 'tool': 'mega_crash_enhanced', 'direction': 'long',
            'hold': 24, 'sl_pct': 0.08,
            'reason': f"ENHANCED MEGA CRASH: {ret_24h:.1f}% drop, vol={vol_ratio:.1f}x, bounce started"
        }, score))
    
    # Tool 9: Enhanced Flash Crash (OOS-validated: 37.1% WR)
    # Original: ret_12h < -10
    # Enhanced: Add momentum confirmation and volume
    if (ret_12h < -15 and  # Higher threshold
        vol_ratio > 2.5 and  # Volume spike
        ret_4h > -2 and  # Momentum shifting
        cur_rsi < 30):  # Oversold
        
        score = abs(ret_12h) * 2.5 * min(vol_ratio / 2.0, 2.0)
        signals.append(({
            'pair': pair, 'tool': 'flash_crash_enhanced', 'direction': 'long',
            'hold': 24, 'sl_pct': 0.07,
            'reason': f"ENHANCED FLASH CRASH: {ret_12h:.1f}% drop 12h, vol={vol_ratio:.1f}x, RSI={cur_rsi:.1f}"
        }, score))
    
    # Tool 10: Enhanced Quick Crash (OOS-validated: 38.7% WR)
    # Original: ret_8h < -10
    # Enhanced: Volume + momentum + position sizing based on volatility
    if (ret_8h < -12 and  # Higher threshold
        vol_ratio > 2.0 and  # Volume confirmation
        ret_4h > 0 and  # Definite bounce started
        cur_atr_pct < 15):  # Not in extreme volatility (better risk/reward)
        
        score = abs(ret_8h) * 2 * min(vol_ratio, 2.0)
        signals.append(({
            'pair': pair, 'tool': 'quick_crash_enhanced', 'direction': 'long',
            'hold': 24, 'sl_pct': 0.07,
            'reason': f"ENHANCED QUICK CRASH: {ret_8h:.1f}% drop 8h, vol={vol_ratio:.1f}x, bounce confirmed"
        }, score))
    
    # NEW COMBO TOOL: Extreme Crash Combo (Multiple confirmations)
    # Combines crash detection with multiple filters for high-conviction signals
    if (ret_24h < -15 and cur_rsi < 15 and  # Extreme crash + oversold
        vol_ratio > 3.0 and  # Volume spike
        bb_position < 0.05 and  # Extreme BB position
        consecutive_red >= 3 and  # Sustained selling
        ret_4h > -1):  # Bounce starting
        
        score = 50  # Highest priority signal
        signals.append(({
            'pair': pair, 'tool': 'extreme_crash_combo', 'direction': 'long',
            'hold': 48, 'sl_pct': 0.10,
            'reason': f"EXTREME CRASH COMBO: {ret_24h:.1f}% drop, RSI={cur_rsi:.1f}, vol={vol_ratio:.1f}x, ALL confirmations"
        }, score))
    
    return signals

# UPDATED TOOL STATISTICS (Expected based on enhancements)
# These are projected improvements - would need re-validation to confirm
ENHANCED_TOOL_PROJECTIONS = {
    'crash_buy_enhanced': {'expected_wr': 52, 'improvement': '+22.3%'},
    'volatile_oversold_enhanced': {'expected_wr': 48, 'improvement': '+16.5%'}, 
    'relief_rally_enhanced': {'expected_wr': 45, 'improvement': '+14.8%'},
    'dip_buy_enhanced': {'expected_wr': 53, 'improvement': '+12.4%'},
    'rsi_pump_short_enhanced': {'expected_wr': 55, 'improvement': '+6.5%'},
    'mega_crash_enhanced': {'expected_wr': 49, 'improvement': '+12.7%'},
    'flash_crash_enhanced': {'expected_wr': 51, 'improvement': '+13.9%'},
    'quick_crash_enhanced': {'expected_wr': 52, 'improvement': '+13.3%'},
    'extreme_crash_combo': {'expected_wr': 65, 'note': 'New high-conviction signal'},
}