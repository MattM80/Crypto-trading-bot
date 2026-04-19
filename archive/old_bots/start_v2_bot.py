#!/usr/bin/env python3
"""
Convenience script to start the V2 Trading Bot
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import and run the bot
from run_v2_bot import main

if __name__ == "__main__":
    sys.exit(main())