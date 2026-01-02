"""
Configuration Loader Utility
Loads and manages configuration for the red team kit
"""

import json
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader for red team kit"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize config loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Return default configuration
        return {
            "generation": {
                "num_variants": 5,
                "default_technique": "obfuscation"
            },
            "injection": {
                "use_all_patterns": True,
                "default_pattern": None
            },
            "jailbreak": {
                "apply_all_techniques": True
            },
            "logging": {
                "level": "INFO",
                "file": None
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save(self, config: Dict[str, Any]):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.config = config

