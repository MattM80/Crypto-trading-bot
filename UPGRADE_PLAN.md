# Crypto Bot Upgrade Plan — April 19, 2026

**Goal:** Make the bot as profitable as possible. Stop bleeding on shitcoins, add intelligence, ride trends.

**Starting point:** $340 on $300 start (+13.4%), but peaked at $359 and bleeding. 62 closed trades. Most losses came from volatile micro-caps (DRIFT -$59, RAVE -$4 net but 24 trades and nearly killed us, RIVER -$4, NIGHT -$1). Long-only in a bear market (F&G 27, 100% bearish 4h).

**Bot file:** `run_final_bot.py` (~4071 lines)
**Trade journal:** `logs/trade_journal.csv`
**Always syntax check after edits:** `python3 -c "import ast; ast.parse(open('run_final_bot.py').read()); print('SYNTAX OK')"`
**Bot is LIVE on Kraken with real money.** Don't break it.

---

## Phase 1: Stop the Bleeding (Priority: IMMEDIATE)

### 1A. Market Cap Whitelist

**Problem:** Bot scans 657 pairs and picks the most volatile. This is how RAVE (340% volatility), DRIFT, RIVER, NIGHT end up in the portfolio. Volatility ≠ opportunity on micro-caps.

**Fix:** Hard whitelist of top coins by market cap. The volatility scanner (`_refresh_volatile_pairs`, ~line 969) should only select from this pool.

**Implementation:**

1. Add a constant near the top of the file (after the existing constants ~line 130):

```python
# WHITELIST: Only trade established coins. No micro-cap garbage.
# Top 20 by market cap + a few validated altcoins that have shown edge
PAIR_WHITELIST = {
    'XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD',
    'DOTUSD', 'LINKUSD', 'AVAXUSD', 'AAVEUSD', 'UNIUSD',
    'LTCUSD', 'ATOMUSD', 'NEARUSD', 'FILUSD', 'SUIUSD',
    'ZECUSD', 'MATICUSD', 'APTUSD', 'RENDERUSD',
    'JITOSOLUSD',  # Validated: +$3.63 live PnL
}
ENABLE_WHITELIST = True  # Set False to go back to full scan
```

2. In `_refresh_volatile_pairs` (~line 969), right after fetching the pairs list, filter:

```python
if ENABLE_WHITELIST:
    # Only consider whitelisted pairs
    pairs = {k: v for k, v in pairs.items() if normalize_pair(k) in PAIR_WHITELIST or k in PAIR_WHITELIST}
    logger.info(f"[WHITELIST] Filtered to {len(pairs)} whitelisted pairs")
```

Find where the method builds its candidate list (look for where it iterates over pairs from the API) and insert the filter there, before the volatility scoring.

3. Also add the whitelist filter in the main pair scanning loop in `run_cycle` (~line 3798). Find where it iterates over `self.trading_pairs` or similar, and skip pairs not in whitelist:

```python
if ENABLE_WHITELIST and normalize_pair(pair) not in PAIR_WHITELIST and pair not in PAIR_WHITELIST:
    continue
```

**Expected impact:** Eliminates ~$70 of the ~$80 in total losses. No more RAVE/DRIFT/RIVER/NIGHT.

### 1B. Death Spiral Detector

**Problem:** Bot buys "dips" that are actually collapses. RAVE went -77% in 24h — that's not a dip.

**Fix:** Before any entry, check 24h price change. If a coin is down >25% in 24h, hard block it regardless of signal score.

**Implementation:**

1. In `execute_signal` (~line 3321), after the pair daily limit check and before the regime gate, add:

```python
        # DEATH SPIRAL DETECTOR: Don't buy coins in freefall
        # If 24h return is worse than -25%, this isn't a dip — it's a collapse
        if direction == 'long' and pair in self._price_cache:
            cached = self._price_cache[pair]
            if len(cached) >= 25:  # Need 24h of hourly data
                ret_24h = (cached[-1] - cached[-25]) / cached[-25] * 100
                if ret_24h < -25:
                    logger.warning(f"[DEATH SPIRAL] {pair} down {ret_24h:.1f}% in 24h — blocking all longs")
                    self._log_rejection(pair, tool, direction, score, f"death_spiral_{ret_24h:.1f}pct")
                    return
```

**Expected impact:** Would have blocked every RAVE entry in the last 24h. Also catches future rug pulls.

### 1C. Restart Bot After Phase 1

```bash
# Kill existing bot
ps aux | grep run_final_bot | grep -v grep | awk '{print $2}' | xargs kill -9

# Wait and restart
sleep 3
cd /Users/lucasaust/code/Crypto-trading-bot
ENABLE_LIVE_TRADING=true nohup python3 run_final_bot.py > /dev/null 2>&1 &

# Verify it's running
sleep 15
tail -30 logs/final_bot.log
```

---

## Phase 2: Add the Brain (Priority: NEXT DAY)

### 2A. LLM Sentiment Pre-Trade Filter

**Problem:** Bot has no idea WHY a price is moving. A -16% drop from a protocol hack is a death sentence. A -16% drop because BTC sneezed is a buying opportunity.

**Fix:** Before opening any position with size > $50, make a lightweight API call to check recent news/sentiment.

**Implementation:**

1. Add a new method to the bot class:

```python
def _check_sentiment(self, pair: str) -> dict:
    """Quick sentiment check before entering a trade.
    Returns: {'ok': bool, 'reason': str, 'sentiment': str}
    """
    import openai  # or anthropic, whichever is available
    
    # Strip USD suffix to get coin name
    coin = pair.replace('USD', '').replace('XBT', 'BTC')
    
    try:
        # Use a cheap/fast model — this runs on every trade
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast
            messages=[{
                "role": "user",
                "content": f"Is there any major negative news about {coin} cryptocurrency in the last 24 hours? "
                          f"Specifically: hacks, exploits, rug pulls, delistings, SEC actions, team exits, or protocol failures. "
                          f"Reply in this exact format:\n"
                          f"SAFE: <one line reason> — if no major negative news\n"
                          f"DANGER: <one line reason> — if there IS major negative news"
            }],
            max_tokens=50,
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        
        if result.startswith('DANGER'):
            return {'ok': False, 'reason': result, 'sentiment': 'danger'}
        return {'ok': True, 'reason': result, 'sentiment': 'safe'}
    except Exception as e:
        # If API fails, don't block the trade — just log it
        logger.warning(f"[SENTIMENT] API check failed for {pair}: {e}")
        return {'ok': True, 'reason': 'API unavailable', 'sentiment': 'unknown'}
```

2. Call it in `execute_signal`, after the death spiral check:

```python
        # LLM SENTIMENT CHECK: Don't buy into bad news
        if direction == 'long' and position_size > 50:
            sentiment = self._check_sentiment(pair)
            if not sentiment['ok']:
                logger.warning(f"[SENTIMENT BLOCK] {pair} — {sentiment['reason']}")
                self._log_rejection(pair, tool, direction, score, f"sentiment_{sentiment['sentiment']}")
                return
```

3. Cache results for 1 hour to avoid redundant API calls:

```python
# In __init__:
self._sentiment_cache = {}  # {pair: {'result': dict, 'ts': float}}

# In _check_sentiment, before the API call:
now = time.time()
if pair in self._sentiment_cache:
    cached = self._sentiment_cache[pair]
    if now - cached['ts'] < 3600:  # 1 hour cache
        return cached['result']

# After getting result:
self._sentiment_cache[pair] = {'result': result_dict, 'ts': now}
```

**API key setup:** Need `OPENAI_API_KEY` in environment. Add to the startup command:
```bash
ENABLE_LIVE_TRADING=true OPENAI_API_KEY=<key> nohup python3 run_final_bot.py > /dev/null 2>&1 &
```

**Cost estimate:** GPT-4o-mini is ~$0.15/1M input tokens. At ~50 tokens per check, 20 checks/day = ~$0.0001/day. Negligible.

**Alternative if no OpenAI key:** Use CoinGecko's free API to check for unusual volume + price action patterns instead. Less intelligent but free:

```python
def _check_coin_health(self, pair: str) -> dict:
    """Check coin health via CoinGecko (free, no API key)."""
    coin_map = {
        'XBTUSD': 'bitcoin', 'ETHUSD': 'ethereum', 'SOLUSD': 'solana',
        'ADAUSD': 'cardano', 'XRPUSD': 'ripple', 'DOTUSD': 'polkadot',
        'LINKUSD': 'chainlink', 'AVAXUSD': 'avalanche-2', 'AAVEUSD': 'aave',
        'UNIUSD': 'uniswap', 'LTCUSD': 'litecoin', 'ATOMUSD': 'cosmos',
        'NEARUSD': 'near', 'FILUSD': 'filecoin', 'SUIUSD': 'sui',
        'ZECUSD': 'zcash', 'RENDERUSD': 'render-token', 'JITOSOLUSD': 'jito-governance-token',
    }
    coin_id = coin_map.get(pair)
    if not coin_id:
        return {'ok': True, 'reason': 'unmapped coin'}
    
    try:
        r = requests.get(f'https://api.coingecko.com/api/v3/coins/{coin_id}',
                        params={'localization': 'false', 'tickers': 'false', 
                                'community_data': 'false', 'developer_data': 'false'},
                        timeout=5)
        data = r.json()
        change_24h = data.get('market_data', {}).get('price_change_percentage_24h', 0)
        change_7d = data.get('market_data', {}).get('price_change_percentage_7d', 0)
        mcap_rank = data.get('market_cap_rank', 999)
        
        # Block if: >30% down in 24h, or >50% down in 7d, or market cap rank > 200
        if change_24h < -30:
            return {'ok': False, 'reason': f'Down {change_24h:.1f}% in 24h — possible collapse'}
        if change_7d < -50:
            return {'ok': False, 'reason': f'Down {change_7d:.1f}% in 7d — sustained dump'}
        if mcap_rank and mcap_rank > 200:
            return {'ok': False, 'reason': f'Market cap rank #{mcap_rank} — too small'}
        
        return {'ok': True, 'reason': f'Rank #{mcap_rank}, 24h: {change_24h:+.1f}%'}
    except Exception as e:
        return {'ok': True, 'reason': f'API error: {e}'}
```

### 2B. Adaptive Tool Weighting from Live Journal

**Problem:** Tool stats in `_initialize_tool_stats` are from backtests. Live performance differs significantly (entropy_dip: 52.8% WR backtest → 0% WR live).

**Fix:** On startup, read the trade journal CSV and compute actual live stats. Use those for Kelly sizing and tool streak logic.

**Implementation:**

1. Add a method that runs at startup (call from `__init__`, after `_initialize_tool_stats`):

```python
def _update_stats_from_journal(self):
    """Override backtest stats with actual live performance."""
    journal_path = os.path.join(self.log_dir, 'trade_journal.csv')
    if not os.path.exists(journal_path):
        return
    
    import csv
    tool_live = {}  # {tool: {'wins': 0, 'losses': 0, 'total_pnl': 0, 'win_pnls': [], 'loss_pnls': []}}
    
    with open(journal_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('event') != 'CLOSE':
                continue
            tool = row.get('tool', '')
            pnl_pct = float(row.get('pnl_pct', 0) or 0)
            pnl_dollar = float(row.get('pnl_dollar', 0) or 0)
            
            if tool not in tool_live:
                tool_live[tool] = {'wins': 0, 'losses': 0, 'total_pnl': 0, 
                                   'win_pnls': [], 'loss_pnls': []}
            
            if pnl_pct > 0:
                tool_live[tool]['wins'] += 1
                tool_live[tool]['win_pnls'].append(pnl_pct)
            else:
                tool_live[tool]['losses'] += 1
                tool_live[tool]['loss_pnls'].append(pnl_pct)
            tool_live[tool]['total_pnl'] += pnl_dollar
    
    # Override stats for tools with 3+ live trades
    for tool, live in tool_live.items():
        total = live['wins'] + live['losses']
        if total < 3:
            continue  # Not enough data
        if tool not in self.tool_stats:
            continue
        
        wr = live['wins'] / total
        avg_win = sum(live['win_pnls']) / len(live['win_pnls']) if live['win_pnls'] else 0
        avg_loss = sum(live['loss_pnls']) / len(live['loss_pnls']) if live['loss_pnls'] else 0
        
        self.tool_stats[tool].update({
            'total': total,
            'wins': live['wins'],
            'pnl': live['total_pnl'],
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
        })
        
        # Auto-disable tools with WR < 30% on 5+ trades
        if total >= 5 and wr < 0.30:
            self.tool_stats[tool]['score_adj'] = 0.0
            logger.warning(f"[ADAPTIVE] {tool} auto-disabled: {wr:.0%} WR on {total} live trades")
        # Boost tools with WR > 70% on 5+ trades
        elif total >= 5 and wr > 0.70:
            self.tool_stats[tool]['score_adj'] = min(1.5, self.tool_stats[tool].get('score_adj', 1.0) * 1.2)
            logger.info(f"[ADAPTIVE] {tool} boosted: {wr:.0%} WR on {total} live trades")
        
        logger.info(f"[LIVE STATS] {tool}: {total} trades, {wr:.0%} WR, ${live['total_pnl']:.2f} PnL")
```

2. Call it in `__init__` right after `self._initialize_tool_stats()`:
```python
self._update_stats_from_journal()
```

---

## Phase 3: Trend Following Overlay (Priority: THIS WEEK)

### 3A. Momentum/Trend Tools for Bull Regimes

**Problem:** Bot only knows how to buy crashes. When the market turns bullish, it should ride trends on liquid pairs — that's where the real money is.

**Fix:** Add 2-3 simple trend-following tools that activate when F&G > 45 and market is bullish.

**New tools to implement in the signal scanning section:**

```python
# TREND_FOLLOW_EMA: Price above 20 EMA, 20 EMA above 50 EMA, RSI 40-65 (not overbought)
# Entry: pullback to 20 EMA in an uptrend
# Exit: trailing stop, not fixed TP
if len(close) >= 50:
    ema_20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
    ema_50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
    price = close[-1]
    fng = self.get_fng()
    
    # Uptrend: EMAs stacked, price near 20 EMA (within 1.5%), not overbought
    if (ema_20 > ema_50 and 
        price > ema_50 and
        abs(price - ema_20) / ema_20 < 0.015 and  # Within 1.5% of 20 EMA
        40 < cur_rsi < 65 and
        fng > 45):
        
        score = adjust_score('trend_follow_ema', 
            (ema_20/ema_50 - 1) * 500 + (65 - cur_rsi) * 0.3)
        signals.append(({
            'pair': pair, 'tool': 'trend_follow_ema', 'direction': 'long',
            'hold': 72,  # Hold up to 72h (3 days) — trend trades need time
            'sl_pct': 0.03,  # Tight stop: 3% below entry
            'trailing_stop': True,  # NEW: use trailing stop instead of fixed TP
            'trailing_pct': 0.05,  # 5% trailing from peak
            'reason': f"TREND FOLLOW: EMA20 > EMA50, pullback to EMA20, RSI={cur_rsi:.1f}"
        }, score))
```

```python
# BREAKOUT_VOLUME: Price breaks above 24h high on 2x+ volume
# Only on top 10 coins, F&G > 40
if len(close) >= 25 and len(df) >= 2:
    high_24h = max(close[-25:-1])  # 24h high excluding current bar
    current_vol = df['volume'].iloc[-1]
    avg_vol = df['volume'].iloc[-25:-1].mean()
    fng = self.get_fng()
    
    if (close[-1] > high_24h * 1.005 and  # Breaking above 24h high
        current_vol > avg_vol * 2 and      # On 2x volume
        cur_rsi < 75 and                    # Not already overbought
        fng > 40 and
        pair in {'XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD',
                 'LINKUSD', 'AVAXUSD', 'AAVEUSD', 'DOTUSD', 'LTCUSD'}):
        
        score = adjust_score('breakout_volume',
            (close[-1]/high_24h - 1) * 200 + (current_vol/avg_vol - 1) * 5)
        signals.append(({
            'pair': pair, 'tool': 'breakout_volume', 'direction': 'long',
            'hold': 48,
            'sl_pct': 0.025,
            'trailing_stop': True,
            'trailing_pct': 0.04,
            'reason': f"BREAKOUT: Above 24h high, {current_vol/avg_vol:.1f}x volume"
        }, score))
```

### 3B. Trailing Stop Exit System

**Problem:** Fixed take-profits exit too early. The JITOSOL trade that hit +7.5% would've captured more with a trail.

**Fix:** Add a trailing stop mechanism to the exit logic.

**Implementation:**

In the position management / exit checking code (look for where stop losses and take profits are evaluated each cycle), add:

```python
# For positions with trailing_stop=True:
if pos.get('trailing_stop'):
    # Track peak price since entry
    if 'peak_price' not in pos:
        pos['peak_price'] = pos['entry_price']
    
    current_price = close[-1]  # or however current price is accessed
    
    # Update peak
    if current_price > pos['peak_price']:
        pos['peak_price'] = current_price
    
    # Check trailing stop
    trail_pct = pos.get('trailing_pct', 0.05)
    trail_price = pos['peak_price'] * (1 - trail_pct)
    
    # Only activate trailing stop after position is in profit
    if pos['peak_price'] > pos['entry_price'] * 1.01:  # At least 1% up
        if current_price <= trail_price:
            # CLOSE: trailing stop hit
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            logger.info(f"[TRAILING STOP] {pair} — peak ${pos['peak_price']:.4f}, "
                       f"trail hit @ ${current_price:.4f} ({trail_pct:.0%} from peak), "
                       f"PnL: {pnl_pct:+.1%}")
            # ... close position logic ...
```

### 3C. Add New Tools to Config

```python
# Add to VALIDATED_TOOLS or a new category:
TREND_TOOLS = ["trend_follow_ema", "breakout_volume"]

# Add to tool stats:
"trend_follow_ema": {"trades": 0, "wins": 0, "pnl": 0, "score_adj": 1.0},
"breakout_volume": {"trades": 0, "wins": 0, "pnl": 0, "score_adj": 1.0},

# Gate these tools: only fire when F&G > 45
# Already handled in the signal conditions above
```

---

## Testing & Validation

**Before going live with any phase:**

1. Syntax check: `python3 -c "import ast; ast.parse(open('run_final_bot.py').read()); print('SYNTAX OK')"`
2. Dry run for 1-2 cycles: Start without `ENABLE_LIVE_TRADING=true` and check logs
3. Check that existing positions still managed correctly after restart

**Measuring success:**
- Track these weekly: win rate, avg win $, avg loss $, net PnL, max drawdown
- Compare to pre-upgrade baseline (62 trades, -$62.71 realized, +13.4% including unrealized)
- Phase 1 goal: Eliminate shitcoin losses entirely
- Phase 2 goal: Skip at least 1 bad trade per week via sentiment
- Phase 3 goal: Capture trend moves of 5%+ without exiting at 2-3%

---

## Quick Reference: Restart Procedure

```bash
# 1. Kill
ps aux | grep run_final_bot | grep -v grep | awk '{print $2}' | xargs kill -9
sleep 3

# 2. Syntax check
cd /Users/lucasaust/code/Crypto-trading-bot
python3 -c "import ast; ast.parse(open('run_final_bot.py').read()); print('SYNTAX OK')"

# 3. Start (with optional OpenAI key for Phase 2)
ENABLE_LIVE_TRADING=true OPENAI_API_KEY=<key> nohup python3 run_final_bot.py > /dev/null 2>&1 &

# 4. Verify
sleep 15
tail -30 logs/final_bot.log
```

---

## Already Completed (Apr 19)

- [x] Killed entropy_dip (0% WR, -$4.33)
- [x] Killed volatile_oversold (0% WR, -$10.23)  
- [x] Killed quick_crash (33% WR, -$8.84)
- [x] Regime gate: blocks weak longs in strong bear (100% bearish + RSI<35 + F&G<30)
- [x] 50% size cut on longs passing regime gate in bear
- [x] Per-pair daily limit: 3 trades/pair/day max
- [x] Tighter stops: btc_alt_spread and deep_dip_8h 5%→4%
- [x] Updated tool stats with live performance
