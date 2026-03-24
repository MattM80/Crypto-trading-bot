#!/usr/bin/env python3
"""
VOLATILITY & OPTIONS INTELLIGENCE ENGINE

Uses FREE Deribit options data to generate trading signals for futures trading.
We don't trade options directly - we use options market data as INTELLIGENCE for futures.

Key signals:
- DVOL (crypto VIX) from near-term ATM options IV
- Put/Call ratio (contrarian indicator)  
- Max pain analysis (price gravity)
- Volatility skew (fear vs greed)
- Gamma exposure (move amplification/dampening)
- Term structure (normal vs stressed markets)

All data from Deribit public API - no authentication required.
"""

import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger


class VolatilityEngine:
    """Analyzes options market data for trading intelligence."""
    
    def __init__(self):
        self.base_url = 'https://www.deribit.com/api/v2/public'
        self.cache = {}
        self.cache_ttl = 300  # 5 min cache (options data changes slower than orderbook)
        self.iv_history = []  # Track IV over time for trend detection
        self.put_call_history = []  # Track put/call ratio
        self.last_update = 0
        
        # Current state
        self.current_signals = {}
        self.last_dvol = None
        self.dvol_trend_data = []  # Store last 24 hours of DVOL readings
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make request to Deribit API with error handling."""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params or {}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Deribit API error ({endpoint}): {e}")
            return {}
    
    def _get_cached_data(self, key: str, fetch_func, *args) -> dict:
        """Get data with caching."""
        now = time.time()
        if key in self.cache and now - self.cache[key]['timestamp'] < self.cache_ttl:
            return self.cache[key]['data']
        
        data = fetch_func(*args)
        self.cache[key] = {'data': data, 'timestamp': now}
        return data
    
    def get_index_price(self, currency: str) -> float:
        """Get current index price for BTC or ETH."""
        try:
            data = self._make_request('get_index_price', {'index_name': f'{currency.lower()}_usd'})
            return data.get('result', {}).get('index_price', 0)
        except:
            return 0
    
    def get_options_chain(self, currency: str) -> List[dict]:
        """Get all active options for currency."""
        cache_key = f'options_chain_{currency}'
        return self._get_cached_data(
            cache_key, 
            self._fetch_options_chain,
            currency
        )
    
    def _fetch_options_chain(self, currency: str) -> List[dict]:
        """Fetch options instruments from Deribit."""
        data = self._make_request('get_instruments', {
            'currency': currency,
            'kind': 'option', 
            'expired': 'false'
        })
        return data.get('result', [])
    
    def get_book_summaries(self, currency: str) -> List[dict]:
        """Get option book summaries with IV, volume, OI."""
        cache_key = f'book_summaries_{currency}'
        return self._get_cached_data(
            cache_key,
            self._fetch_book_summaries, 
            currency
        )
    
    def _fetch_book_summaries(self, currency: str) -> List[dict]:
        """Fetch book summaries from Deribit."""
        data = self._make_request('get_book_summary_by_currency', {
            'currency': currency,
            'kind': 'option'
        })
        return data.get('result', [])
    
    def get_historical_volatility(self, currency: str) -> dict:
        """Get historical volatility data from Deribit."""
        cache_key = f'hist_vol_{currency}'
        return self._get_cached_data(
            cache_key,
            self._fetch_historical_volatility,
            currency
        )
    
    def _fetch_historical_volatility(self, currency: str) -> dict:
        """Fetch historical volatility from Deribit."""
        data = self._make_request('get_historical_volatility', {'currency': currency})
        return data.get('result', {})
    
    def parse_option_name(self, instrument_name: str) -> dict:
        """Parse option instrument name: BTC-25APR25-100000-C."""
        try:
            parts = instrument_name.split('-')
            if len(parts) != 4:
                return {}
            
            currency, expiry_str, strike_str, option_type = parts
            
            # Parse expiry date
            expiry_date = datetime.strptime(expiry_str, '%d%b%y').replace(tzinfo=timezone.utc)
            
            return {
                'currency': currency,
                'expiry': expiry_date,
                'strike': float(strike_str),
                'type': option_type,  # 'C' for call, 'P' for put
                'dte': (expiry_date - datetime.now(timezone.utc)).days
            }
        except:
            return {}
    
    def calc_dvol(self, book_summaries: List[dict], current_price: float) -> float:
        """
        Calculate crypto VIX equivalent from near-term ATM options IV.
        
        High IV (>80%) = market expects big move = INCREASE position sizes
        Low IV (<40%) = market expects calm = REDUCE position sizes  
        IV spike (>20% increase in 24h) = something is happening = BE ALERT
        IV crush (>20% decrease) = event passed = FADE the move
        """
        if not book_summaries or not current_price:
            return 50.0  # Default neutral IV
        
        atm_ivs = []
        
        for summary in book_summaries:
            instrument = summary.get('instrument_name', '')
            parsed = self.parse_option_name(instrument)
            
            if not parsed or parsed.get('dte', 0) < 7 or parsed.get('dte', 999) > 30:
                continue  # Only near-term options (7-30 days)
            
            strike = parsed.get('strike', 0)
            if not strike:
                continue
                
            # Check if roughly ATM (within 10% of current price)
            if abs(strike - current_price) / current_price < 0.10:
                mark_iv = summary.get('mark_iv')
                if mark_iv and mark_iv > 0:
                    # mark_iv comes from Deribit already as percentage (50.04 = 50.04%)
                    atm_ivs.append(mark_iv)
        
        if not atm_ivs:
            return 50.0
        
        dvol = np.mean(atm_ivs)  # Already in percentage form
        
        # Store for trend analysis
        now = time.time()
        self.dvol_trend_data.append({'timestamp': now, 'dvol': dvol})
        
        # Keep only last 24 hours
        cutoff = now - 24 * 3600
        self.dvol_trend_data = [d for d in self.dvol_trend_data if d['timestamp'] > cutoff]
        
        return dvol
    
    def calc_put_call_ratio(self, book_summaries: List[dict]) -> float:
        """
        Put/Call ratio by volume and open interest.
        
        High P/C (>1.2) = lots of puts = market hedging/bearish = CONTRARIAN BULLISH
        Low P/C (<0.7) = lots of calls = market greedy = CONTRARIAN BEARISH  
        Extreme P/C (>1.5) = panic hedging = BUY signal (everyone already hedged)
        Extreme low P/C (<0.5) = extreme greed = SELL signal
        """
        put_vol = 0
        call_vol = 0
        put_oi = 0
        call_oi = 0
        
        for summary in book_summaries:
            instrument = summary.get('instrument_name', '')
            volume = summary.get('volume', 0) or 0
            oi = summary.get('open_interest', 0) or 0
            
            if '-P' in instrument:  # Put option
                put_vol += volume
                put_oi += oi
            elif '-C' in instrument:  # Call option
                call_vol += volume
                call_oi += oi
        
        # Prefer volume-based ratio, fallback to OI
        if call_vol > 0:
            pcr = put_vol / call_vol
        elif call_oi > 0:
            pcr = put_oi / call_oi
        else:
            pcr = 1.0
        
        # Store for history tracking
        self.put_call_history.append({
            'timestamp': time.time(),
            'pcr': pcr,
            'put_vol': put_vol,
            'call_vol': call_vol
        })
        
        # Keep only last 100 readings
        self.put_call_history = self.put_call_history[-100:]
        
        return pcr
    
    def calc_max_pain(self, book_summaries: List[dict], current_price: float) -> Tuple[float, str]:
        """
        Max pain = price where most options expire worthless.
        Market makers push price toward max pain near expiration.
        
        If max pain is ABOVE current price → bullish bias
        If max pain is BELOW current price → bearish bias
        Strongest signal 1-3 days before expiration
        """
        if not book_summaries or not current_price:
            return current_price, 'neutral'
        
        # Get all strikes and their total open interest
        strike_oi = {}
        near_expiry_instruments = []
        
        for summary in book_summaries:
            instrument = summary.get('instrument_name', '')
            parsed = self.parse_option_name(instrument)
            oi = summary.get('open_interest', 0) or 0
            
            if not parsed or oi == 0:
                continue
            
            # Focus on options expiring within 7 days (max pain strongest near expiry)
            if parsed.get('dte', 999) <= 7:
                near_expiry_instruments.append({
                    'instrument': instrument,
                    'strike': parsed.get('strike'),
                    'type': parsed.get('type'),
                    'oi': oi
                })
        
        if not near_expiry_instruments:
            return current_price, 'neutral'
        
        # Calculate pain for each potential price level
        strikes = sorted(set(inst['strike'] for inst in near_expiry_instruments if inst['strike']))
        min_pain = float('inf')
        max_pain_price = current_price
        
        for test_price in strikes:
            total_pain = 0
            
            for inst in near_expiry_instruments:
                strike = inst['strike']
                option_type = inst['type']
                oi = inst['oi']
                
                if not strike:
                    continue
                
                # Calculate intrinsic value if price were at test_price
                if option_type == 'C':  # Call
                    intrinsic = max(0, test_price - strike)
                else:  # Put
                    intrinsic = max(0, strike - test_price)
                
                # Pain = total money lost by option holders
                total_pain += intrinsic * oi
            
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_price = test_price
        
        # Determine bias
        if max_pain_price > current_price * 1.02:  # Max pain >2% above current
            bias = 'bullish'
        elif max_pain_price < current_price * 0.98:  # Max pain >2% below current
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return max_pain_price, bias
    
    def calc_skew(self, book_summaries: List[dict], current_price: float) -> Tuple[float, str]:
        """
        IV skew: compare IV of OTM puts vs OTM calls.
        
        Put IV >> Call IV = market pricing in downside risk = BEARISH backdrop
        Call IV >> Put IV = market pricing in upside = BULLISH backdrop
        Skew flattening = fear receding = potential rally  
        Skew steepening = fear increasing = protect/short
        """
        if not book_summaries or not current_price:
            return 0, 'neutral'
        
        otm_put_ivs = []
        otm_call_ivs = []
        
        for summary in book_summaries:
            instrument = summary.get('instrument_name', '')
            parsed = self.parse_option_name(instrument)
            mark_iv = summary.get('mark_iv')
            
            if not parsed or not mark_iv or mark_iv <= 0:
                continue
            
            strike = parsed.get('strike', 0)
            option_type = parsed.get('type')
            dte = parsed.get('dte', 999)
            
            # Focus on near-term options with reasonable volume
            if dte < 7 or dte > 30:
                continue
            
            # OTM puts: strike < current price  
            if option_type == 'P' and strike < current_price * 0.90:  # 10% OTM
                otm_put_ivs.append(mark_iv)
            
            # OTM calls: strike > current price
            elif option_type == 'C' and strike > current_price * 1.10:  # 10% OTM  
                otm_call_ivs.append(mark_iv)
        
        if not otm_put_ivs or not otm_call_ivs:
            return 0, 'neutral'
        
        avg_put_iv = np.mean(otm_put_ivs)  # Already in percentage form
        avg_call_iv = np.mean(otm_call_ivs)  # Already in percentage form
        skew = avg_put_iv - avg_call_iv
        
        # Classify skew signal
        if skew > 5:  # Puts much more expensive than calls
            signal = 'fear'
        elif skew < -5:  # Calls much more expensive than puts
            signal = 'greed'
        else:
            signal = 'neutral'
        
        return skew, signal
    
    def estimate_gamma_exposure(self, book_summaries: List[dict], current_price: float) -> str:
        """
        Estimate net gamma exposure of market makers.
        
        Positive GEX = MMs are short gamma, will BUY dips and SELL rips = DAMPENS volatility
        Negative GEX = MMs are long gamma, will AMPLIFY moves = INCREASES volatility
        
        Negative GEX + crash signal = EXTRA CONVICTION (move will overshoot)  
        Positive GEX = grid environment (price oscillates)
        """
        if not book_summaries or not current_price:
            return 'neutral'
        
        # Simplified GEX estimation based on OI distribution relative to spot
        total_call_oi_above = 0  # Calls with strike > spot (likely short by MMs)
        total_put_oi_below = 0   # Puts with strike < spot (likely short by MMs)
        total_call_oi_below = 0  # Calls with strike < spot (likely long by MMs)
        total_put_oi_above = 0   # Puts with strike > spot (likely long by MMs)
        
        for summary in book_summaries:
            instrument = summary.get('instrument_name', '')
            parsed = self.parse_option_name(instrument)
            oi = summary.get('open_interest', 0) or 0
            
            if not parsed or oi == 0:
                continue
            
            strike = parsed.get('strike', 0)
            option_type = parsed.get('type')
            dte = parsed.get('dte', 999)
            
            # Focus on near-term options
            if dte > 30:
                continue
            
            if option_type == 'C':  # Calls
                if strike > current_price:
                    total_call_oi_above += oi  # MMs likely short these calls
                else:
                    total_call_oi_below += oi  # MMs likely long these calls
            else:  # Puts
                if strike < current_price:
                    total_put_oi_below += oi  # MMs likely short these puts
                else:
                    total_put_oi_above += oi  # MMs likely long these puts
        
        # Positive GEX = MMs short gamma (short OTM calls/puts)
        positive_gex = total_call_oi_above + total_put_oi_below
        
        # Negative GEX = MMs long gamma (long ITM calls/puts)  
        negative_gex = total_call_oi_below + total_put_oi_above
        
        gex_ratio = positive_gex / max(negative_gex, 1)
        
        if gex_ratio > 1.5:
            return 'positive'  # Dampening effect - MMs provide liquidity
        elif gex_ratio < 0.67:
            return 'negative'  # Amplifying effect - MMs chase moves
        else:
            return 'neutral'
    
    def calc_term_structure(self, book_summaries: List[dict]) -> str:
        """
        Compare IV across expirations.
        
        Contango (far IV > near IV) = normal, calm market
        Backwardation (near IV > far IV) = near-term event expected = VOLATILE
        """
        if not book_summaries:
            return 'contango'  # Default to normal
        
        near_term_ivs = []  # 7-20 days
        far_term_ivs = []   # 21-60 days
        
        for summary in book_summaries:
            instrument = summary.get('instrument_name', '')
            parsed = self.parse_option_name(instrument)
            mark_iv = summary.get('mark_iv')
            
            if not parsed or not mark_iv or mark_iv <= 0:
                continue
            
            dte = parsed.get('dte', 999)
            
            if 7 <= dte <= 20:
                near_term_ivs.append(mark_iv)
            elif 21 <= dte <= 60:
                far_term_ivs.append(mark_iv)
        
        if not near_term_ivs or not far_term_ivs:
            return 'contango'
        
        avg_near_iv = np.mean(near_term_ivs)
        avg_far_iv = np.mean(far_term_ivs)
        
        if avg_near_iv > avg_far_iv * 1.05:  # Near-term IV >5% higher
            return 'backwardation'  # Event risk priced in
        else:
            return 'contango'  # Normal term structure
    
    def get_volatility_signals(self) -> Dict:
        """
        Main method: returns comprehensive volatility intelligence.
        
        Returns: {
            'dvol': float,  # Current implied vol level (0-200)
            'dvol_trend': str,  # 'rising', 'falling', 'stable'
            'put_call_ratio': float,  # 0.3-3.0
            'put_call_signal': str,  # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
            'max_pain_btc': float,  # BTC max pain price
            'max_pain_eth': float,  # ETH max pain price
            'max_pain_bias': str,  # 'bullish', 'bearish', 'neutral'
            'skew': float,  # Put IV - Call IV
            'skew_signal': str,  # 'fear', 'neutral', 'greed'
            'gamma_exposure': str,  # 'positive' (dampening) or 'negative' (amplifying)
            'term_structure': str,  # 'contango' or 'backwardation'
            'regime': str,  # 'low_vol', 'normal', 'high_vol', 'extreme'
            'market_signal': float,  # -5 to +5 composite signal
            'position_size_multiplier': float,  # 0.5 to 1.5 based on vol regime
        }
        """
        try:
            logger.info("📊 Fetching volatility intelligence from Deribit...")
            
            # Get current data for BTC and ETH
            btc_price = self.get_index_price('BTC')
            eth_price = self.get_index_price('ETH')
            
            btc_summaries = self.get_book_summaries('BTC')
            eth_summaries = self.get_book_summaries('ETH')
            
            if not btc_summaries and not eth_summaries:
                logger.warning("No options data available")
                return self._default_signals()
            
            # Calculate DVOL (use BTC as primary, ETH as secondary)
            dvol_btc = self.calc_dvol(btc_summaries, btc_price) if btc_summaries else 50.0
            dvol_eth = self.calc_dvol(eth_summaries, eth_price) if eth_summaries else 50.0
            dvol = dvol_btc if btc_summaries else dvol_eth
            
            # DVOL trend analysis
            dvol_trend = self._analyze_dvol_trend()
            
            # Put/Call analysis (combine BTC + ETH for broader sample)
            all_summaries = btc_summaries + eth_summaries
            pcr = self.calc_put_call_ratio(all_summaries)
            pcr_signal = self._classify_pcr_signal(pcr)
            
            # Max pain analysis
            max_pain_btc, btc_bias = self.calc_max_pain(btc_summaries, btc_price) if btc_summaries else (btc_price, 'neutral')
            max_pain_eth, eth_bias = self.calc_max_pain(eth_summaries, eth_price) if eth_summaries else (eth_price, 'neutral')
            
            # Overall max pain bias (BTC weights more)
            if btc_bias != 'neutral':
                max_pain_bias = btc_bias
            else:
                max_pain_bias = eth_bias
            
            # Skew analysis (use BTC as primary)
            skew, skew_signal = self.calc_skew(btc_summaries, btc_price) if btc_summaries else (0, 'neutral')
            
            # Gamma exposure
            gamma_exposure = self.estimate_gamma_exposure(all_summaries, btc_price)
            
            # Term structure
            term_structure = self.calc_term_structure(all_summaries)
            
            # Volatility regime classification
            regime = self._classify_vol_regime(dvol)
            
            # Composite market signal (-5 to +5)
            market_signal = self._calculate_market_signal(dvol, pcr, skew_signal, gamma_exposure, term_structure)
            
            # Position sizing multiplier
            position_multiplier = self._calculate_position_multiplier(dvol, dvol_trend, gamma_exposure)
            
            signals = {
                'dvol': round(dvol, 1),
                'dvol_trend': dvol_trend,
                'put_call_ratio': round(pcr, 2),
                'put_call_signal': pcr_signal,
                'max_pain_btc': round(max_pain_btc, 0),
                'max_pain_eth': round(max_pain_eth, 0), 
                'max_pain_bias': max_pain_bias,
                'skew': round(skew, 1),
                'skew_signal': skew_signal,
                'gamma_exposure': gamma_exposure,
                'term_structure': term_structure,
                'regime': regime,
                'market_signal': round(market_signal, 1),
                'position_size_multiplier': round(position_multiplier, 2),
                'btc_price': btc_price,
                'eth_price': eth_price,
                'timestamp': time.time()
            }
            
            self.current_signals = signals
            self.last_update = time.time()
            
            logger.info(f"📊 Volatility signals: DVOL={dvol:.1f} ({dvol_trend}), "
                       f"P/C={pcr:.2f} ({pcr_signal}), skew={skew:.1f} ({skew_signal}), "
                       f"GEX={gamma_exposure}, regime={regime}, signal={market_signal:+.1f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating volatility signals: {e}")
            return self._default_signals()
    
    def _analyze_dvol_trend(self) -> str:
        """Analyze DVOL trend from recent data."""
        if len(self.dvol_trend_data) < 2:
            return 'stable'
        
        # Look at trend over last 6 hours
        cutoff = time.time() - 6 * 3600
        recent_data = [d for d in self.dvol_trend_data if d['timestamp'] > cutoff]
        
        if len(recent_data) < 2:
            return 'stable'
        
        # Simple linear trend
        x = [d['timestamp'] for d in recent_data]
        y = [d['dvol'] for d in recent_data]
        
        if len(x) >= 3:
            slope = np.polyfit(x, y, 1)[0]
            # Slope is per second, convert to per hour
            hourly_change = slope * 3600
            
            if hourly_change > 2:  # >2% IV increase per hour
                return 'rising'
            elif hourly_change < -2:  # >2% IV decrease per hour  
                return 'falling'
        
        return 'stable'
    
    def _classify_pcr_signal(self, pcr: float) -> str:
        """Classify put/call ratio into sentiment signal."""
        if pcr >= 1.5:
            return 'extreme_fear'  # Panic hedging - contrarian bullish
        elif pcr >= 1.2:
            return 'fear'  # High hedging - moderately bullish
        elif pcr <= 0.5:
            return 'extreme_greed'  # Call mania - contrarian bearish
        elif pcr <= 0.7:
            return 'greed'  # High speculation - moderately bearish
        else:
            return 'neutral'
    
    def _classify_vol_regime(self, dvol: float) -> str:
        """Classify current volatility regime."""
        if dvol >= 100:
            return 'extreme'  # Crisis levels
        elif dvol >= 80:
            return 'high_vol'  # Stressed market
        elif dvol <= 30:
            return 'low_vol'  # Complacent market
        else:
            return 'normal'  # Normal vol environment
    
    def _calculate_market_signal(self, dvol: float, pcr: float, skew_signal: str, 
                                gamma_exposure: str, term_structure: str) -> float:
        """Calculate composite market signal (-5 to +5)."""
        signal = 0
        
        # DVOL contribution
        if dvol > 100:
            signal += 2  # Extreme fear usually marks bottoms
        elif dvol > 80:
            signal += 1  # High vol can be buying opportunities
        elif dvol < 30:
            signal -= 1  # Low vol often precedes drops
        
        # Put/Call ratio (contrarian)
        if pcr >= 1.5:
            signal += 3  # Extreme fear - strong contrarian buy
        elif pcr >= 1.2:
            signal += 1  # Moderate fear - slight bullish
        elif pcr <= 0.5:
            signal -= 3  # Extreme greed - strong contrarian sell
        elif pcr <= 0.7:
            signal -= 1  # Moderate greed - slight bearish
        
        # Skew contribution
        if skew_signal == 'fear':
            signal += 1  # Fear in options - contrarian bullish
        elif skew_signal == 'greed':
            signal -= 1  # Greed in options - contrarian bearish
        
        # Gamma exposure  
        if gamma_exposure == 'negative':
            signal += 0.5  # Negative GEX amplifies moves - good for breakouts
        elif gamma_exposure == 'positive':
            signal -= 0.5  # Positive GEX dampens moves - range bound
        
        # Term structure
        if term_structure == 'backwardation':
            signal += 0.5  # Event risk priced in - could resolve positively
        
        return max(-5, min(5, signal))
    
    def _calculate_position_multiplier(self, dvol: float, dvol_trend: str, 
                                     gamma_exposure: str) -> float:
        """Calculate position size multiplier based on vol regime."""
        multiplier = 1.0
        
        # Base DVOL adjustment
        if dvol > 80:
            multiplier = 1.3  # High IV - big moves expected, size up
        elif dvol > 60:
            multiplier = 1.1  # Moderate high IV
        elif dvol < 30:
            multiplier = 0.7  # Low IV - small moves, size down
        elif dvol < 40:
            multiplier = 0.85  # Moderate low IV
        
        # Trend adjustment
        if dvol_trend == 'rising' and dvol > 60:
            multiplier *= 1.1  # Vol spiking - something happening
        elif dvol_trend == 'falling' and dvol < 50:
            multiplier *= 0.9  # Vol crushing - moves fading
        
        # Gamma exposure adjustment
        if gamma_exposure == 'negative':
            multiplier *= 1.1  # Negative GEX amplifies moves
        elif gamma_exposure == 'positive':
            multiplier *= 0.95  # Positive GEX dampens moves
        
        return max(0.5, min(1.5, multiplier))  # Clamp between 0.5x and 1.5x
    
    def _default_signals(self) -> Dict:
        """Return default signals when data unavailable."""
        return {
            'dvol': 50.0,
            'dvol_trend': 'stable',
            'put_call_ratio': 1.0,
            'put_call_signal': 'neutral',
            'max_pain_btc': 50000.0,
            'max_pain_eth': 3000.0,
            'max_pain_bias': 'neutral',
            'skew': 0.0,
            'skew_signal': 'neutral',
            'gamma_exposure': 'neutral',
            'term_structure': 'contango',
            'regime': 'normal',
            'market_signal': 0.0,
            'position_size_multiplier': 1.0,
            'btc_price': 0,
            'eth_price': 0,
            'timestamp': time.time()
        }


if __name__ == "__main__":
    # Quick test
    engine = VolatilityEngine()
    signals = engine.get_volatility_signals()
    print(f"Current volatility signals: {signals}")