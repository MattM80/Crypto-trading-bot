"""
Tool Optimizer Integration for Crypto Bot
Provides clean API to load tier configurations from the optimizer.
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, Optional
from loguru import logger


class ToolOptimizerLoader:
    """Loads and provides tool tier configurations from the optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to tool_tiers.json in the same directory
            self.config_path = Path(__file__).parent / "tool_tiers.json"
        else:
            self.config_path = Path(config_path)
        
        self.tier_config = {}
        self.last_loaded = None
        self.load_config()
    
    def load_config(self):
        """Load tier configuration from JSON file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Tool tier config not found: {self.config_path}")
                return
            
            # Check if file was modified
            current_mtime = self.config_path.stat().st_mtime
            if self.last_loaded == current_mtime:
                return  # Already loaded latest version
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.tier_config = config.get('tiers', {})
            self.last_loaded = current_mtime
            
            logger.info(f"Loaded tool tier config with {len(self.tier_config)} tools")
            logger.info(f"Config updated at: {config.get('updated_at', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load tool tier config: {e}")
            self.tier_config = {}
    
    def get_tier1_tools(self) -> Set[str]:
        """Get set of tools that should use Tier 1 (2x margin) sizing."""
        self.load_config()  # Refresh config
        
        tier1_tools = set()
        for tool_name, config in self.tier_config.items():
            if config.get('tier') == 1 and config.get('enabled', True):
                tier1_tools.add(tool_name)
        
        logger.info(f"Tier 1 tools with 2x sizing: {len(tier1_tools)} - {sorted(tier1_tools)}")
        return tier1_tools
    
    def get_enabled_tools(self) -> Set[str]:
        """Get set of all enabled tools."""
        self.load_config()  # Refresh config
        
        enabled_tools = set()
        for tool_name, config in self.tier_config.items():
            if config.get('enabled', True):
                enabled_tools.add(tool_name)
        
        return enabled_tools
    
    def get_disabled_tools(self) -> Set[str]:
        """Get set of disabled tools that should be skipped."""
        self.load_config()  # Refresh config
        
        disabled_tools = set()
        for tool_name, config in self.tier_config.items():
            if not config.get('enabled', True):
                disabled_tools.add(tool_name)
        
        if disabled_tools:
            logger.warning(f"Disabled tools: {len(disabled_tools)} - {sorted(disabled_tools)}")
        
        return disabled_tools
    
    def get_size_multiplier(self, tool_name: str) -> float:
        """Get position size multiplier for a specific tool."""
        self.load_config()  # Refresh config
        
        if tool_name not in self.tier_config:
            return 1.0  # Default sizing
        
        config = self.tier_config[tool_name]
        if not config.get('enabled', True):
            return 0.0  # Disabled
        
        return config.get('size_mult', 1.0)
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        self.load_config()  # Refresh config
        
        if tool_name not in self.tier_config:
            return True  # Default to enabled for unknown tools
        
        return self.tier_config[tool_name].get('enabled', True)
    
    def get_tool_stats(self, tool_name: str) -> Dict:
        """Get performance stats for a tool if available."""
        self.load_config()  # Refresh config
        
        if tool_name not in self.tier_config:
            return {}
        
        config = self.tier_config[tool_name]
        return {
            'tier': config.get('tier', 0),
            'enabled': config.get('enabled', True),
            'win_rate': config.get('win_rate', 0),
            'profit_factor': config.get('profit_factor', 0),
            'trades': config.get('trades', 0),
            'size_mult': config.get('size_mult', 1.0)
        }


# Global instance for easy import
tool_optimizer = ToolOptimizerLoader()


# Convenience functions for backward compatibility
def load_tool_tiers() -> Set[str]:
    """Load and return Tier 1 tools set."""
    return tool_optimizer.get_tier1_tools()


def is_tool_disabled(tool_name: str) -> bool:
    """Check if a tool is disabled by the optimizer."""
    return not tool_optimizer.is_tool_enabled(tool_name)


def get_tool_size_multiplier(tool_name: str) -> float:
    """Get position size multiplier for a tool."""
    return tool_optimizer.get_size_multiplier(tool_name)


if __name__ == "__main__":
    # Test the loader
    loader = ToolOptimizerLoader()
    
    tier1 = loader.get_tier1_tools()
    disabled = loader.get_disabled_tools()
    
    print(f"Tier 1 tools: {tier1}")
    print(f"Disabled tools: {disabled}")
    
    # Test a specific tool
    test_tool = 'crash_buy'
    print(f"\n{test_tool} stats: {loader.get_tool_stats(test_tool)}")
    print(f"{test_tool} enabled: {loader.is_tool_enabled(test_tool)}")
    print(f"{test_tool} size multiplier: {loader.get_size_multiplier(test_tool)}")