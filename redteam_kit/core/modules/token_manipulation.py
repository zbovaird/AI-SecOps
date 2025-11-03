"""
Token Manipulation Module
Implements token-level manipulation techniques for security testing
"""

from typing import List, Dict, Any
import re
import random


class TokenManipulation:
    """Token manipulation techniques for testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize token manipulation module"""
        self.config = config or {}
    
    def whitespace_manipulation(self, text: str) -> str:
        """Manipulate whitespace to evade detection"""
        # Add extra spaces
        text = re.sub(r' ', '  ', text)
        # Add newlines
        text = text.replace('.', '.\n')
        return text
    
    def case_variation(self, text: str) -> str:
        """Vary character case"""
        words = text.split()
        result = []
        for word in words:
            if random.random() < 0.3:
                # Randomly capitalize
                word = ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in word)
            result.append(word)
        return ' '.join(result)
    
    def add_unicode(self, text: str) -> str:
        """Add Unicode characters"""
        # Zero-width spaces
        text = text.replace(' ', '\u200B \u200B')
        return text
    
    def token_splitting(self, text: str) -> str:
        """Split tokens across boundaries"""
        # Insert spaces in words
        words = text.split()
        result = []
        for word in words:
            if len(word) > 4:
                mid = len(word) // 2
                word = word[:mid] + ' ' + word[mid:]
            result.append(word)
        return ' '.join(result)
    
    def apply_manipulations(self, text: str, num_techniques: int = 2) -> List[str]:
        """
        Apply multiple token manipulations
        
        Args:
            text: Original text
            num_techniques: Number of techniques to apply
        
        Returns:
            List of manipulated texts
        """
        techniques = [
            self.whitespace_manipulation,
            self.case_variation,
            self.add_unicode,
            self.token_splitting,
        ]
        
        variants = []
        selected = random.sample(techniques, min(num_techniques, len(techniques)))
        
        for technique in selected:
            variant = technique(text)
            variants.append(variant)
        
        return variants

