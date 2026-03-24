#!/usr/bin/env python3
"""
Quick test for remaining tools validation - process just 50 bars to test logic
"""

from oos_validate_remaining import OOSRemainingValidator

class QuickTestValidator(OOSRemainingValidator):
    def validate_tools(self):
        """Run validation on just a small subset."""
        print("\nStarting quick test of remaining tools...")
        
        # Get the minimum data length across all pairs
        min_length = min(len(self.data_cache[pair]) for pair in self.pairs)
        
        # Start validation from bar 100 but only test 50 bars
        start_bar = 100
        end_bar = min(start_bar + 50, min_length - 10)  # Just 50 bars for testing
        
        print(f"Testing bars {start_bar} to {end_bar}")
        
        signals = []
        
        for bar_idx in range(start_bar, end_bar):
            if bar_idx % 10 == 0:
                print(f"Processing bar {bar_idx}/{end_bar}...")
            
            # For cross-pair tools, we need ALL pairs data at this timestamp
            all_pair_data = {}
            for pair in self.pairs:
                df = self.data_cache[pair]
                if bar_idx >= len(df):
                    continue
                all_pair_data[pair] = {
                    'df': df.iloc[:bar_idx+1].copy(),
                    'close': df['close'].iloc[:bar_idx+1].values.astype(float),
                    'open': df['open'].iloc[:bar_idx+1].values.astype(float),
                    'high': df['high'].iloc[:bar_idx+1].values.astype(float), 
                    'low': df['low'].iloc[:bar_idx+1].values.astype(float),
                    'volume': df['volume'].iloc[:bar_idx+1].values.astype(float),
                    'timestamp': df['timestamp'].iloc[bar_idx]
                }
            
            # Skip if we don't have data for all pairs
            if len(all_pair_data) != len(self.pairs):
                continue
                
            # Generate signals for this bar
            bar_signals = self.scan_remaining_tools(all_pair_data, bar_idx)
            signals.extend(bar_signals)
        
        print(f"\nGenerated {len(signals)} total signals from {end_bar - start_bar} bars")
        
        # Print signal summary
        tool_counts = {}
        for signal in signals:
            tool = signal['tool']
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        print(f"\nSignal counts by tool:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count}")
        
        # Calculate forward returns for a few signals to test
        if signals:
            print(f"\nTesting forward return calculation on first 5 signals:")
            for i, signal in enumerate(signals[:5]):
                pair = signal['pair']
                bar_idx = signal['bar_idx']
                direction = signal['direction']
                
                df = self.data_cache[pair]
                if bar_idx + 6 < len(df):
                    current_price = df['close'].iloc[bar_idx]
                    price_8h = df['close'].iloc[bar_idx + 2]
                    price_24h = df['close'].iloc[bar_idx + 6]
                    
                    if direction == 'long':
                        ret_8h = (price_8h - current_price) / current_price * 100
                        ret_24h = (price_24h - current_price) / current_price * 100
                    else:
                        ret_8h = (current_price - price_8h) / current_price * 100
                        ret_24h = (current_price - price_24h) / current_price * 100
                    
                    print(f"  {signal['tool']} ({direction}): 8h={ret_8h:+.2f}%, 24h={ret_24h:+.2f}%")

def main():
    validator = QuickTestValidator()
    validator.validate_tools()

if __name__ == "__main__":
    main()