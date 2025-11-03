"""
Logger Utility
Provides logging functionality for the red team kit
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
from pathlib import Path


class FrameworkLogger:
    """Framework logger for red team testing"""
    
    def __init__(self, name: str = "redteam_kit", log_level: int = logging.INFO, log_file: Optional[str] = None):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_level: Logging level
            log_file: Optional file path for logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_jailbreak_result(self, technique: str, prompt: str, response: str, success: bool):
        """Log jailbreak attempt result"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"JAILBREAK ATTEMPT - Technique: {technique} - Status: {status}")
        self.logger.debug(f"Prompt: {prompt[:200]}...")
        self.logger.debug(f"Response: {response[:500]}...")

