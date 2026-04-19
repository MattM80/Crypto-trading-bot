"""Diagnose WHY trades lose money. Print every trade with full details."""
import sys, os
sys.path.insert(0, 'src')
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from backtest_live import RealisticBacktester, _download_ohlcv
from datetime import datetime, timedelta, timezone
import pandas as pd

now = datetime.now(tz=timezone.utc)
since = int((now - timedelta(days=7)).timestamp())
until = int(now.timestamp())

# Patch the backtester to print FULL trade lifecycle
original_run = RealisticBacktester.run

def verbose_run(self, symbol, df, warmup=120):
    """Override run to print signal → entry → exit details."""
    if len(df) < warmup + 10:
        return {}
    
    from risk_manager import Position
    active_positions = []
    trade_log = []
    
    for i in range(warmup, len(df)):
        self._bar_count += 1
        window = df.iloc[:i].copy()
        bar = df.iloc[i]
        bar_open = float(bar["open"])
        current_price = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        
        # Manage existing
        for pos in list(active_positions):
            if pos.status != "OPEN":
                continue
            self.rm.update_trailing_stop(pos, current_price)
            
            if pos.side == "BUY":
                if bar_low <= pos.stop_loss:
                    sl_fill = min(pos.stop_loss, bar_low)
                    pnl = (sl_fill - pos.entry_price) * pos.quantity
                    fee = abs(sl_fill * pos.quantity * self.fee_pct)
                    net = pnl - fee
                    print(f"  EXIT  {symbol:8s} {pos.side:4s} entry={pos.entry_price:.2f} SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f} | exit={sl_fill:.2f} via SL | P&L=${net:+.2f}")
                    self._close_position(pos, sl_fill, "Stop Loss", active_positions)
                elif bar_high >= pos.take_profit:
                    pnl = (pos.take_profit - pos.entry_price) * pos.quantity
                    fee = abs(pos.take_profit * pos.quantity * self.fee_pct)
                    net = pnl - fee
                    print(f"  EXIT  {symbol:8s} {pos.side:4s} entry={pos.entry_price:.2f} SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f} | exit={pos.take_profit:.2f} via TP | P&L=${net:+.2f}")
                    self._close_position(pos, pos.take_profit, "Take Profit", active_positions)
            else:
                if bar_high >= pos.stop_loss:
                    sl_fill = max(pos.stop_loss, bar_high)
                    pnl = (pos.entry_price - sl_fill) * pos.quantity
                    fee = abs(sl_fill * pos.quantity * self.fee_pct)
                    net = pnl - fee
                    print(f"  EXIT  {symbol:8s} {pos.side:4s} entry={pos.entry_price:.2f} SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f} | exit={sl_fill:.2f} via SL | P&L=${net:+.2f}")
                    self._close_position(pos, sl_fill, "Stop Loss", active_positions)
                elif bar_low <= pos.take_profit:
                    pnl = (pos.entry_price - pos.take_profit) * pos.quantity
                    fee = abs(pos.take_profit * pos.quantity * self.fee_pct)
                    net = pnl - fee
                    print(f"  EXIT  {symbol:8s} {pos.side:4s} entry={pos.entry_price:.2f} SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f} | exit={pos.take_profit:.2f} via TP | P&L=${net:+.2f}")
                    self._close_position(pos, pos.take_profit, "Take Profit", active_positions)
        
        active_positions = [p for p in active_positions if p.status == "OPEN"]
        
        # Generate signals
        try:
            signals = self.strategy.generate_signals(window, symbol)
        except Exception:
            signals = []
        
        for sig in signals:
            if sig.action == "HOLD":
                continue
            can, _ = self.rm.can_open_position(sig.symbol)
            if not can:
                continue
            
            from strategies import fee_aware_ev_filter
            passes, _ = fee_aware_ev_filter(
                sig.entry_price, sig.stop_loss, sig.take_profit, sig.action,
                fee_pct=self.fee_pct, slippage_pct=self.slippage_pct)
            if not passes:
                continue
            
            size = self.rm.calculate_position_size(
                entry_price=sig.entry_price, stop_loss_price=sig.stop_loss,
                atr=getattr(sig, "atr", 0.0))
            if size <= 0:
                continue
            
            if sig.action == "BUY":
                fill_price = bar_open * (1 + self.slippage_pct)
            else:
                fill_price = bar_open * (1 - self.slippage_pct)
            
            # KEY: what does the SL/TP adjustment look like?
            orig_sl = sig.stop_loss
            orig_tp = sig.take_profit
            orig_entry = sig.entry_price
            
            if sig.stop_loss and sig.take_profit:
                if sig.action == "BUY":
                    sl_dist = sig.entry_price - sig.stop_loss
                    tp_dist = sig.take_profit - sig.entry_price
                    new_sl = fill_price - sl_dist
                    new_tp = fill_price + tp_dist
                else:
                    sl_dist = sig.stop_loss - sig.entry_price
                    tp_dist = sig.entry_price - sig.take_profit
                    new_sl = fill_price + sl_dist
                    new_tp = fill_price - tp_dist
                
                # Check if SL is on wrong side
                if sig.action == "BUY" and new_sl > fill_price:
                    print(f"  !! BUG: BUY SL ({new_sl:.2f}) > entry ({fill_price:.2f}) | sig.entry={orig_entry:.2f} sig.sl={orig_sl:.2f} sl_dist={sl_dist:.2f} bar_open={bar_open:.2f}")
                elif sig.action == "SELL" and new_sl < fill_price:
                    print(f"  !! BUG: SELL SL ({new_sl:.2f}) < entry ({fill_price:.2f}) | sig.entry={orig_entry:.2f} sig.sl={orig_sl:.2f} sl_dist={sl_dist:.2f} bar_open={bar_open:.2f}")
                
                sig.stop_loss = new_sl
                sig.take_profit = new_tp
            
            entry_fee = abs(fill_price * size * self.fee_pct)
            if self.rm.current_balance < entry_fee:
                continue
            self.rm.current_balance -= entry_fee
            
            # Print entry
            if sig.action == "BUY":
                sl_pct = (fill_price - sig.stop_loss) / fill_price * 100
                tp_pct = (sig.take_profit - fill_price) / fill_price * 100
            else:
                sl_pct = (sig.stop_loss - fill_price) / fill_price * 100
                tp_pct = (fill_price - sig.take_profit) / fill_price * 100
            rr = tp_pct / sl_pct if sl_pct > 0 else -1
            print(f"  ENTRY {symbol:8s} {sig.action:4s} signal_price={orig_entry:.2f} fill={fill_price:.2f} gap={fill_price-orig_entry:+.2f} SL={sig.stop_loss:.2f}({sl_pct:.2f}%) TP={sig.take_profit:.2f}({tp_pct:.2f}%) R:R={rr:.1f}")
            
            pos = Position(
                id=f"bt-{self._bar_count}-{sig.symbol}",
                symbol=sig.symbol, entry_price=fill_price, quantity=size,
                stop_loss=sig.stop_loss, take_profit=sig.take_profit,
                entry_time=str(i), side=sig.action,
                atr_at_entry=getattr(sig, "atr", 0.0),
            )
            self.rm.record_position(pos)
            active_positions.append(pos)
        
        unrealized = sum(
            ((current_price - p.entry_price) if p.side == "BUY" else (p.entry_price - current_price)) * p.quantity
            for p in active_positions if p.status == "OPEN"
        )
        self.equity_curve.append(self.rm.current_balance + unrealized)
    
    # Close remaining
    if active_positions:
        last_price = float(df.iloc[-1]["close"])
        for pos in active_positions:
            if pos.status == "OPEN":
                self._close_position(pos, last_price, "End of data", active_positions)
    
    return self._compute_stats(symbol, df)

RealisticBacktester.run = verbose_run

for pair in ["SOLUSD"]:
    print(f"\n{'='*110}")
    print(f"  {pair} - 15m - 7 days")
    print(f"{'='*110}")
    
    bt = RealisticBacktester(
        strategy_type="adaptive", initial_balance=1000,
        use_limit_orders=True, max_open=3, risk_per_trade=0.02
    )
    df = _download_ohlcv(pair, 15, since, until)
    if df.empty or len(df) < 150:
        print(f"  Insufficient data: {len(df)} bars")
        continue
    stats = bt.run(pair, df)
    print(f"\n  Result: {stats.get('total_trades',0)} trades, P&L=${stats.get('total_pnl',0):+.2f}")
