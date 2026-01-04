"""
Semantic similarity evaluator.

Measures semantic drift between baseline and adversarial outputs.
"""

from typing import Dict, Any, Optional, List
import hashlib

from .base import BaseEvaluator, EvalResult, EvaluatorRegistry


@EvaluatorRegistry.register("semantic")
class SemanticEvaluator(BaseEvaluator):
    """
    Evaluates semantic similarity between baseline and adversarial outputs.
    
    Uses embedding-based cosine similarity when available, falls back to
    lexical similarity metrics.
    """
    
    name = "semantic"
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_embeddings: bool = True,
        embedding_model: Optional[str] = None,
        fallback_to_lexical: bool = True,
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self.fallback_to_lexical = fallback_to_lexical
        
        self._embedder = None
        self._embedder_loaded = False
    
    def _load_embedder(self):
        """Lazy load sentence transformer."""
        if self._embedder_loaded:
            return self._embedder is not None
            
        self._embedder_loaded = True
        
        if not self.use_embeddings:
            return False
            
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
            self._logger.info(f"Loaded embedding model: {self.embedding_model}")
            return True
        except ImportError:
            self._logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            self._logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity using embeddings."""
        if not self._load_embedder():
            raise RuntimeError("Embedder not available")
        
        import numpy as np
        
        embeddings = self._embedder.encode([text1, text2])
        
        # Cosine similarity
        dot = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot / (norm1 * norm2))
    
    def _compute_lexical_similarity(self, text1: str, text2: str) -> float:
        """Compute lexical similarity using n-gram overlap."""
        # Normalize texts
        t1 = text1.lower().split()
        t2 = text2.lower().split()
        
        if not t1 or not t2:
            return 0.0 if (t1 or t2) else 1.0
        
        # Jaccard similarity on word sets
        set1 = set(t1)
        set2 = set(t2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 1.0
            
        jaccard = intersection / union
        
        # Also check n-gram overlap for better accuracy
        def get_ngrams(words, n):
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        bigrams1 = get_ngrams(t1, 2)
        bigrams2 = get_ngrams(t2, 2)
        
        if bigrams1 and bigrams2:
            bigram_overlap = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
        else:
            bigram_overlap = jaccard
        
        # Weighted combination
        return 0.6 * jaccard + 0.4 * bigram_overlap
    
    def _compute_exact_match(self, text1: str, text2: str) -> bool:
        """Check if texts are exactly identical."""
        return text1.strip() == text2.strip()
    
    def _compute_hash_match(self, text1: str, text2: str) -> bool:
        """Check if text hashes match (for deduplication)."""
        h1 = hashlib.md5(text1.encode()).hexdigest()
        h2 = hashlib.md5(text2.encode()).hexdigest()
        return h1 == h2
    
    def evaluate(
        self,
        baseline: str,
        adversarial: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """
        Evaluate semantic similarity between baseline and adversarial outputs.
        
        Returns:
            EvalResult with similarity score and change detection
        """
        flags = []
        evidence = {}
        
        # Quick exact match check
        if self._compute_exact_match(baseline, adversarial):
            return EvalResult(
                label="identical",
                confidence=1.0,
                evidence={"similarity": 1.0, "method": "exact_match"},
            )
        
        # Try embedding similarity
        similarity = None
        method = None
        
        if self.use_embeddings:
            try:
                similarity = self._compute_embedding_similarity(baseline, adversarial)
                method = "embedding"
            except Exception as e:
                flags.append(f"embedding_failed: {e}")
                if not self.fallback_to_lexical:
                    return EvalResult(
                        label="error",
                        confidence=0.0,
                        flags=flags,
                    )
        
        # Fallback to lexical
        if similarity is None:
            similarity = self._compute_lexical_similarity(baseline, adversarial)
            method = "lexical"
            flags.append("using_lexical_fallback")
        
        evidence["similarity"] = similarity
        evidence["method"] = method
        evidence["threshold"] = self.similarity_threshold
        
        # Classify change
        if similarity >= 0.98:
            label = "no_change"
            confidence = similarity
        elif similarity >= self.similarity_threshold:
            label = "minor_change"
            confidence = 1.0 - (1.0 - similarity) / (1.0 - self.similarity_threshold)
        elif similarity >= 0.5:
            label = "moderate_change"
            confidence = (self.similarity_threshold - similarity) / (self.similarity_threshold - 0.5)
        else:
            label = "major_change"
            confidence = 1.0 - similarity
        
        # Add length comparison
        len_baseline = len(baseline)
        len_adversarial = len(adversarial)
        len_ratio = min(len_baseline, len_adversarial) / max(len_baseline, len_adversarial) if max(len_baseline, len_adversarial) > 0 else 1.0
        
        evidence["length_baseline"] = len_baseline
        evidence["length_adversarial"] = len_adversarial
        evidence["length_ratio"] = len_ratio
        
        if len_ratio < 0.5:
            flags.append("significant_length_change")
        
        return EvalResult(
            label=label,
            confidence=confidence,
            evidence=evidence,
            flags=flags,
        )
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity score between two texts.
        
        Convenience method that returns just the similarity score.
        """
        result = self.evaluate(text1, text2)
        return result.evidence.get("similarity", 0.0)





