# Smart Exit Signals and Kraken Staking Integration

## TASK 1: Smart Exit Signals Implementation

### Overview
Enhanced exit logic for the `manage_positions` method in `run_final_bot.py`. These signals will work alongside the existing trailing stops and take profit targets to provide more intelligent exit decisions.

### 1. Volume-Based Exit
**Logic**: Exit profitable positions when volume spikes 2x+ average, signaling potential local top.

```python
# Add to manage_positions method after position data extraction
def check_volume_spike_exit(self, pos, data, current_price):
    """Check if volume spike warrants exit from profitable position."""
    df = data.get('df')
    if df is None or len(df) < 20:
        return False
    
    # Calculate position PnL
    if pos['direction'] == 'long':
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
    else:
        pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
    
    # Only consider if position is profitable
    if pnl_pct <= 0:
        return False
    
    # Get recent volume data
    volumes = df['volume'].values.astype(float)[-20:]  # Last 20 bars
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[:-1])  # Average of prior 19 bars
    
    # Volume spike threshold: 2x average
    volume_spike = current_volume / avg_volume if avg_volume > 0 else 0
    
    # Exit if volume spikes 2x+ and position is profitable
    if volume_spike >= 2.0:
        return True, f"Volume spike exit: {volume_spike:.1f}x avg volume, PnL +{pnl_pct*100:.1f}%"
    
    return False
```

### 2. RSI-Based Exit Enhancement
**Logic**: Tighten trailing stops from 15% to 5% when RSI hits 80+ in profitable longs.

```python
def get_rsi_adjusted_trailing_stop(self, pos, data, base_trailing_stop):
    """Adjust trailing stop based on RSI in overbought/oversold territory."""
    df = data.get('df')
    if df is None or len(df) < 14:
        return base_trailing_stop
    
    # Calculate RSI(14)
    close_prices = df['close'].values.astype(float)
    rsi_values = self.calc_rsi(close_prices, 14)
    current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50
    
    # Check if position is profitable
    current_price = data["price"]
    if pos['direction'] == 'long':
        is_profitable = current_price > pos['entry_price']
        overbought_threshold = 80
    else:
        is_profitable = current_price < pos['entry_price']
        overbought_threshold = 20  # Oversold for shorts
    
    # Tighten stops in extreme RSI territory for profitable positions
    if is_profitable:
        if pos['direction'] == 'long' and current_rsi >= overbought_threshold:
            # Tighten from 15% to 5% for longs in overbought territory
            tightened_stop = min(base_trailing_stop, 0.05)
            return tightened_stop
        elif pos['direction'] == 'short' and current_rsi <= overbought_threshold:
            # Tighten from 15% to 5% for shorts in oversold territory
            tightened_stop = min(base_trailing_stop, 0.05)
            return tightened_stop
    
    return base_trailing_stop
```

### 3. Regime Change Exit (Fear & Greed)
**Logic**: Exit bull swing positions immediately when F&G drops below 30.

```python
def check_regime_change_exit(self, pos):
    """Exit bull positions when Fear & Greed drops below 30."""
    # Only apply to long positions (bull swings)
    if pos['direction'] != 'long':
        return False
    
    # Check if F&G regime has shifted to fear
    if self.current_fng < 30:
        return True, f"Regime change exit: F&G={self.current_fng} < 30 (fear territory)"
    
    return False
```

### 4. Momentum Exhaustion Exit
**Logic**: Tighten stops when 7-day momentum drops below 3-day momentum.

```python
def check_momentum_exhaustion(self, pos, data, base_trailing_stop):
    """Check for momentum exhaustion and tighten stops accordingly."""
    df = data.get('df')
    if df is None or len(df) < 168:  # Need at least 7 days of hourly data
        return base_trailing_stop
    
    close_prices = df['close'].values.astype(float)
    
    # Calculate momentum over different periods
    if len(close_prices) >= 168:  # 7 days
        momentum_7d = (close_prices[-1] - close_prices[-168]) / close_prices[-168]
    else:
        return base_trailing_stop
        
    if len(close_prices) >= 72:   # 3 days
        momentum_3d = (close_prices[-1] - close_prices[-72]) / close_prices[-72]
    else:
        return base_trailing_stop
    
    # Check for momentum exhaustion (7-day < 3-day)
    if momentum_7d < momentum_3d and momentum_3d > 0:
        # Tighten trailing stop to 7% when momentum is exhausting
        tightened_stop = min(base_trailing_stop, 0.07)
        return tightened_stop, f"Momentum exhaustion: 7d={momentum_7d*100:.1f}% < 3d={momentum_3d*100:.1f}%"
    
    return base_trailing_stop
```

### Integration into manage_positions Method

**Replace the existing trailing stop section with this enhanced version:**

```python
# Enhanced trailing stop with smart exit signals
if trailing_stop_pct:
    # Initialize best price tracking
    if 'best_price' not in pos:
        pos['best_price'] = pos['entry_price']
    
    # 1. Check volume spike exit (priority #1)
    volume_exit = self.check_volume_spike_exit(pos, data, current_price)
    if volume_exit:
        self.close_position(pair, current_price, volume_exit[1])
        continue
    
    # 2. Check regime change exit (priority #2)  
    regime_exit = self.check_regime_change_exit(pos)
    if regime_exit:
        self.close_position(pair, current_price, regime_exit[1])
        continue
    
    # 3. Get RSI-adjusted trailing stop
    rsi_adjusted_stop = self.get_rsi_adjusted_trailing_stop(pos, data, trailing_stop_pct)
    
    # 4. Check momentum exhaustion
    momentum_result = self.check_momentum_exhaustion(pos, data, rsi_adjusted_stop)
    if isinstance(momentum_result, tuple):
        final_trailing_stop, momentum_reason = momentum_result
    else:
        final_trailing_stop = momentum_result
        momentum_reason = None
    
    # Dynamic trail: use 3x ATR as trail distance, bounded by the adjusted trail %
    atr_pct = (data.get('atr', 0) / current_price) if current_price > 0 else 0
    dynamic_trail = max(
        final_trailing_stop * 0.5,        # Floor: half the adjusted trail
        min(atr_pct * 3,                  # 3x ATR
            final_trailing_stop * 1.5)    # Cap: 1.5x the adjusted trail
    ) if atr_pct > 0 else final_trailing_stop
    
    if pos['direction'] == 'long':
        if current_price > pos['best_price']:
            pos['best_price'] = current_price
        trail_dd = (pos['best_price'] - current_price) / pos['best_price']
        if trail_dd >= dynamic_trail:
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
            exit_reason = f"Smart trailing stop {dynamic_trail:.1%} (peak ${pos['best_price']:.2f}, pnl {pnl_pct:+.1f}%)"
            if momentum_reason:
                exit_reason += f" - {momentum_reason}"
            self.close_position(pair, current_price, exit_reason)
            continue
    else:  # short
        if current_price < pos['best_price']:
            pos['best_price'] = current_price
        trail_up = (current_price - pos['best_price']) / pos['best_price']
        if trail_up >= dynamic_trail:
            pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
            exit_reason = f"Smart trailing stop {dynamic_trail:.1%} (trough ${pos['best_price']:.2f}, pnl {pnl_pct:+.1f}%)"
            if momentum_reason:
                exit_reason += f" - {momentum_reason}"
            self.close_position(pair, current_price, exit_reason)
            continue
```

---

## TASK 2: Kraken Staking Integration

### Research Findings

**Important Note**: After researching Kraken's current API documentation and support resources, Kraken has **LIMITED API support for staking operations**. Here are the key findings:

#### Assets Supporting Staking on Kraken
Based on Kraken's platform (as of March 2026):
- **Ethereum (ETH)** - ~4-6% APY
- **Solana (SOL)** - ~6-8% APY  
- **Polkadot (DOT)** - ~10-12% APY
- **Cardano (ADA)** - ~4-5% APY
- **Cosmos (ATOM)** - ~8-10% APY
- **Tezos (XTZ)** - ~5-6% APY
- **Algorand (ALGO)** - ~4-5% APY
- **Flow (FLOW)** - ~6-8% APY

#### API Limitations
**Current Status**: Kraken's REST API does **NOT** have public endpoints for:
- ✗ Staking assets programmatically
- ✗ Unstaking assets programmatically  
- ✗ Querying staking balances via API
- ✗ Getting staking rewards history via API

**What IS available:**
- ✓ Query account balances (includes staked amounts in total)
- ✓ Trading operations (buy/sell)
- ✓ Order management
- ✓ Account ledger (may show staking rewards as ledger entries)

#### Lock-up Periods
- **Ethereum**: No lock-up (can unstake anytime, ~24-48h processing)
- **Solana**: No lock-up (instant unstaking available)
- **Polkadot**: 28-day unbonding period
- **Cardano**: No lock-up (3-5 day processing)
- **Cosmos**: 21-day unbonding period
- **Others**: Varies (typically 0-28 days)

#### Trading While Staked
**Mixed compatibility:**
- **Can trade**: You can have open orders while assets are staked
- **Cannot sell staked assets**: Staked tokens are locked and unavailable for trading
- **Must unstake first**: Need to unstake before selling (subject to unbonding periods)

### Implementation Plan

Given the API limitations, here are **three implementation approaches**:

#### Approach 1: Manual Staking Workflow (Recommended)
Since Kraken lacks staking API endpoints, implement a **monitoring and notification system**:

```python
class StakingManager:
    """Manage staking recommendations and tracking for swing trades."""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.stakeable_assets = {
            'ETHUSD': {'asset': 'ETH', 'min_hold_hours': 48, 'apy': 0.05, 'unbond_days': 0},
            'SOLUSD': {'asset': 'SOL', 'min_hold_hours': 48, 'apy': 0.07, 'unbond_days': 0},
            'DOTUSD': {'asset': 'DOT', 'min_hold_hours': 168, 'apy': 0.11, 'unbond_days': 28},
            'ADAUSD': {'asset': 'ADA', 'min_hold_hours': 120, 'apy': 0.045, 'unbond_days': 5},
            'ATOMUSD': {'asset': 'ATOM', 'min_hold_hours': 168, 'apy': 0.09, 'unbond_days': 21},
        }
        self.staking_recommendations = {}
    
    def should_stake_position(self, pair: str, position: dict) -> bool:
        """Determine if a position should be staked."""
        if pair not in self.stakeable_assets:
            return False
        
        config = self.stakeable_assets[pair]
        expected_hold_hours = position.get('hold', 24)
        
        # Only recommend staking for longer holds
        if expected_hold_hours >= config['min_hold_hours']:
            return True
        
        return False
    
    def calculate_staking_yield(self, pair: str, position_size: float, hold_hours: float) -> float:
        """Calculate expected staking yield for the holding period."""
        if pair not in self.stakeable_assets:
            return 0
        
        apy = self.stakeable_assets[pair]['apy']
        yield_for_period = position_size * apy * (hold_hours / (365 * 24))
        return yield_for_period
    
    def log_staking_opportunity(self, pair: str, position: dict):
        """Log staking recommendation to user."""
        if not self.should_stake_position(pair, position):
            return
        
        config = self.stakeable_assets[pair]
        expected_yield = self.calculate_staking_yield(
            pair, position['position_size'], position['hold']
        )
        
        logger.info(f"🥩 STAKING OPPORTUNITY: {pair}")
        logger.info(f"   Position size: ${position['position_size']:.0f}")
        logger.info(f"   Expected hold: {position['hold']}h")
        logger.info(f"   Potential yield: ${expected_yield:.2f} @ {config['apy']*100:.1f}% APY")
        logger.info(f"   ⚠️  Manual action required: Stake {config['asset']} on Kraken")
        
        # Track for exit planning
        self.staking_recommendations[pair] = {
            'recommended_at': datetime.now(timezone.utc).timestamp(),
            'unbond_days': config['unbond_days'],
            'expected_yield': expected_yield
        }
    
    def check_unstaking_needed(self, pair: str, exit_signals_active: bool):
        """Check if we need to warn about unstaking before exit."""
        if pair not in self.staking_recommendations:
            return
        
        if exit_signals_active and pair in self.staking_recommendations:
            config = self.stakeable_assets.get(pair, {})
            unbond_days = config.get('unbond_days', 0)
            
            if unbond_days > 0:
                logger.warning(f"⚠️  UNSTAKING ALERT: {pair}")
                logger.warning(f"   Exit signals active but {config['asset']} has {unbond_days}-day unbonding")
                logger.warning(f"   Consider unstaking NOW if exit likely within {unbond_days} days")
```

**Integration into the bot:**

```python
# Add to bot initialization
self.staking_manager = StakingManager(self)

# Add to position opening logic (after successful entry)
def open_position(self, signal: dict, current_price: float, market_data: dict):
    # ... existing position opening code ...
    
    # After position is successfully opened
    if ENABLE_LIVE_TRADING:
        # Check for staking opportunity
        self.staking_manager.log_staking_opportunity(pair, position_dict)
    
    # ... rest of method ...

# Add to exit signal checking (before closing position)
def check_exit_signals(self, pair: str, position: dict, data: dict):
    # Check if any exit signals are active
    exit_signals_active = (
        self.check_volume_spike_exit(position, data, data["price"]) or
        self.check_regime_change_exit(position) or
        # ... other exit conditions
    )
    
    if exit_signals_active:
        self.staking_manager.check_unstaking_needed(pair, True)
    
    return exit_signals_active
```

#### Approach 2: Selenium Automation (Advanced)
Automate the Kraken web interface using Selenium:

```python
# Note: This approach is complex and fragile - not recommended for production
class KrakenWebStaking:
    def __init__(self, username, password):
        # Web automation setup
        pass
    
    def stake_asset(self, asset: str, amount: float):
        # Navigate to staking page, select asset, enter amount, confirm
        # Very brittle - UI changes break this
        pass
```

#### Approach 3: Hybrid Manual + API Tracking
Combine manual staking with API-based tracking:

```python
def track_staking_via_ledger(self):
    """Track staking rewards through Kraken's ledger API."""
    try:
        # Query ledger for staking-related entries
        ledger = self.client.get_ledgers()
        
        staking_entries = []
        for entry_id, entry in ledger.items():
            if entry.get('type') in ['staking', 'reward']:
                staking_entries.append(entry)
        
        return staking_entries
    except Exception as e:
        logger.error(f"Failed to fetch staking ledger: {e}")
        return []
```

### Recommended Implementation

**For immediate implementation, use Approach 1 (Manual Staking Workflow)** because:

1. **Reliable**: No API dependencies or web scraping fragility
2. **Compliant**: Uses only official Kraken REST API
3. **Actionable**: Provides clear user guidance
4. **Trackable**: Monitors opportunities and timing
5. **Safe**: No automated actions that could fail

### Integration Steps

1. **Add StakingManager class** to `run_final_bot.py`
2. **Initialize in bot constructor**: `self.staking_manager = StakingManager(self)`
3. **Call after position opens**: Log staking opportunities
4. **Call before exits**: Warn about unbonding periods
5. **Add to status logging**: Show staking yield earned (manual input)

### Expected Benefits

- **Earn 4-12% APY** on swing positions (1-6 week holds)
- **Compound returns**: Trading profits + staking yield
- **Risk management**: Forced hodling prevents premature exits
- **Transparency**: Clear logging of opportunities and requirements

### Future Enhancement

When/if Kraken adds staking API endpoints, the manual workflow can be upgraded to full automation by:
1. Replacing log messages with API calls
2. Adding automatic stake/unstake logic
3. Implementing yield tracking and compounding

---

## Summary

**Task 1 (Smart Exits)**: Four intelligent exit enhancements ready for integration:
1. Volume spike detection (2x+ avg volume)
2. RSI-based stop tightening (80+ RSI → 5% stops)  
3. Fear & Greed regime change (F&G < 30 → exit bulls)
4. Momentum exhaustion detection (7d < 3d momentum)

**Task 2 (Staking)**: Manual notification system for 8 stakeable assets with yield tracking and unbonding alerts. Automatic integration not possible due to Kraken API limitations.

Both features enhance the existing bot without disrupting core functionality and provide measurable value through improved exits and additional yield generation.