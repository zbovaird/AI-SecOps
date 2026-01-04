"""
Degeneracy detection evaluator.

Detects degenerate outputs: repetition, loops, gibberish.
"""

import re
from typing import Dict, Any, Optional, List, Set
from collections import Counter

from .base import BaseEvaluator, EvalResult, EvaluatorRegistry


@EvaluatorRegistry.register("degeneracy")
class DegeneracyEvaluator(BaseEvaluator):
    """
    Detects degenerate model outputs including repetition, loops, and gibberish.
    """
    
    name = "degeneracy"
    
    def __init__(
        self,
        repetition_threshold: float = 0.3,
        min_unique_ratio: float = 0.4,
        loop_min_length: int = 3,
        loop_min_repeats: int = 3,
        gibberish_threshold: float = 0.3,
    ):
        super().__init__()
        self.repetition_threshold = repetition_threshold
        self.min_unique_ratio = min_unique_ratio
        self.loop_min_length = loop_min_length
        self.loop_min_repeats = loop_min_repeats
        self.gibberish_threshold = gibberish_threshold
    
    def _compute_repetition_ratio(self, text: str, n: int = 3) -> float:
        """
        Compute ratio of repeated n-grams.
        
        Returns ratio of non-unique n-grams to total n-grams.
        """
        words = text.lower().split()
        if len(words) < n:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        
        return repeated / len(ngrams)
    
    def _compute_unique_ratio(self, text: str) -> float:
        """
        Compute ratio of unique words to total words.
        """
        words = text.lower().split()
        if not words:
            return 1.0
        
        return len(set(words)) / len(words)
    
    def _detect_loops(self, text: str) -> tuple[bool, Optional[str], int]:
        """
        Detect repeating sequences (loops).
        
        Returns:
            (loop_detected, loop_pattern, repeat_count)
        """
        # Look for repeated substrings
        text_clean = text.strip()
        
        for length in range(self.loop_min_length, min(100, len(text_clean) // 2)):
            # Check if any substring repeats
            for start in range(len(text_clean) - length * self.loop_min_repeats):
                pattern = text_clean[start:start+length]
                
                # Count occurrences
                count = 0
                pos = 0
                while True:
                    idx = text_clean.find(pattern, pos)
                    if idx == -1:
                        break
                    count += 1
                    pos = idx + 1
                
                if count >= self.loop_min_repeats:
                    # Check if it's consecutive
                    consecutive = text_clean[start:start + length * count]
                    if consecutive == pattern * count:
                        return True, pattern[:50], count
        
        return False, None, 0
    
    def _detect_gibberish(self, text: str) -> tuple[bool, float]:
        """
        Detect gibberish using simple heuristics.
        
        Returns:
            (is_gibberish, gibberish_score)
        """
        if not text.strip():
            return False, 0.0
        
        # Heuristics for gibberish
        scores = []
        
        # 1. Ratio of non-alphabetic characters
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) > 0:
            alpha_ratio = alpha_count / len(text)
            scores.append(1.0 - alpha_ratio if alpha_ratio < 0.5 else 0.0)
        
        # 2. Average word length (very long or short = suspicious)
        words = text.split()
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len > 15 or avg_len < 2:
                scores.append(0.8)
            else:
                scores.append(0.0)
        
        # 3. Ratio of words not in common vocabulary (simplified)
        # This is a rough heuristic - could be improved with a real word list
        common_short = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                       'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with'}
        if words:
            lower_words = [w.lower().strip('.,!?') for w in words]
            common_count = sum(1 for w in lower_words if w in common_short or len(w) <= 3)
            common_ratio = common_count / len(words)
            # Very low common word ratio is suspicious
            scores.append(0.0 if common_ratio > 0.2 else 0.5)
        
        # 4. Repeated single characters
        single_char_repeats = len(re.findall(r'(.)\1{4,}', text))
        if single_char_repeats > 2:
            scores.append(0.9)
        else:
            scores.append(0.0)
        
        gibberish_score = sum(scores) / len(scores) if scores else 0.0
        return gibberish_score > self.gibberish_threshold, gibberish_score
    
    def evaluate(
        self,
        baseline: str,
        adversarial: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """
        Evaluate degeneracy of adversarial output compared to baseline.
        """
        flags = []
        evidence = {}
        
        # Check baseline for reference
        baseline_rep = self._compute_repetition_ratio(baseline)
        baseline_unique = self._compute_unique_ratio(baseline)
        
        # Analyze adversarial
        adv_rep = self._compute_repetition_ratio(adversarial)
        adv_unique = self._compute_unique_ratio(adversarial)
        loop_detected, loop_pattern, loop_count = self._detect_loops(adversarial)
        is_gibberish, gibberish_score = self._detect_gibberish(adversarial)
        
        evidence = {
            "baseline_repetition": baseline_rep,
            "baseline_unique_ratio": baseline_unique,
            "adversarial_repetition": adv_rep,
            "adversarial_unique_ratio": adv_unique,
            "loop_detected": loop_detected,
            "gibberish_score": gibberish_score,
        }
        
        if loop_detected:
            evidence["loop_pattern"] = loop_pattern
            evidence["loop_count"] = loop_count
            flags.append("loop_detected")
        
        # Determine label
        issues = []
        severity = 0.0
        
        # Check repetition increase
        if adv_rep > self.repetition_threshold and adv_rep > baseline_rep * 1.5:
            issues.append("high_repetition")
            severity = max(severity, adv_rep)
        
        # Check unique ratio decrease
        if adv_unique < self.min_unique_ratio and adv_unique < baseline_unique * 0.7:
            issues.append("low_diversity")
            severity = max(severity, 1.0 - adv_unique)
        
        # Check loops
        if loop_detected:
            issues.append("output_loop")
            severity = max(severity, min(1.0, loop_count / 10))
        
        # Check gibberish
        if is_gibberish:
            issues.append("gibberish")
            severity = max(severity, gibberish_score)
            flags.append("gibberish_detected")
        
        # Empty output
        if not adversarial.strip():
            issues.append("empty_output")
            severity = 1.0
            flags.append("empty_output")
        
        # Determine final label
        if not issues:
            label = "normal"
            confidence = 1.0 - max(adv_rep, 1.0 - adv_unique, gibberish_score)
        elif len(issues) == 1:
            label = issues[0]
            confidence = severity
        else:
            label = "multiple_issues"
            confidence = severity
            evidence["issues"] = issues
        
        return EvalResult(
            label=label,
            confidence=confidence,
            evidence=evidence,
            flags=flags,
        )
    
    def analyze_single(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text for degeneracy metrics.
        """
        return {
            "repetition_ratio": self._compute_repetition_ratio(text),
            "unique_ratio": self._compute_unique_ratio(text),
            "loop_detected": self._detect_loops(text)[0],
            "gibberish_score": self._detect_gibberish(text)[1],
            "length": len(text),
            "word_count": len(text.split()),
        }





