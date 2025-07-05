"""
Evaluator implementations for the prompt optimization framework.

This module provides concrete implementations of evaluators for measuring
the quality of generated text against reference outputs.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from difflib import SequenceMatcher
import re

from .base import Evaluator

logger = logging.getLogger(__name__)


class AdherenceEvaluator(Evaluator):
    """
    Evaluator that measures how well generated text adheres to reference text.
    
    Uses multiple metrics including:
    - Exact match
    - Sequence similarity
    - Semantic similarity (basic keyword matching)
    - Length penalty
    """
    
    def __init__(self, 
                 exact_match_weight: float = 0.3,
                 similarity_weight: float = 0.4,
                 semantic_weight: float = 0.2,
                 length_weight: float = 0.1,
                 max_length_ratio: float = 2.0):
        """
        Initialize adherence evaluator.
        
        Args:
            exact_match_weight: Weight for exact match score.
            similarity_weight: Weight for sequence similarity score.
            semantic_weight: Weight for semantic similarity score.
            length_weight: Weight for length penalty.
            max_length_ratio: Maximum allowed length ratio (generated/reference).
        """
        self.exact_match_weight = exact_match_weight
        self.similarity_weight = similarity_weight
        self.semantic_weight = semantic_weight
        self.length_weight = length_weight
        self.max_length_ratio = max_length_ratio
        
        # Ensure weights sum to 1.0
        total_weight = (exact_match_weight + similarity_weight + 
                       semantic_weight + length_weight)
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total_weight}, not 1.0. "
                          "Consider normalizing weights.")
    
    def run(self, inputs: List[str], references: List[str], 
            generated: List[str]) -> float:
        """
        Evaluate adherence of generated text to references.
        
        Args:
            inputs: List of input prompts (not used in basic adherence).
            references: List of reference outputs.
            generated: List of generated outputs.
            
        Returns:
            Average adherence score (0.0 to 1.0, higher is better).
        """
        if len(references) != len(generated):
            raise ValueError(f"Number of references ({len(references)}) "
                           f"must match number of generated ({len(generated)})")
        
        if not references:
            return 0.0
        
        scores = []
        for ref, gen in zip(references, generated):
            score = self._evaluate_pair(ref, gen)
            scores.append(score)
        
        return float(np.mean(scores))
    
    def _evaluate_pair(self, reference: str, generated: str) -> float:
        """
        Evaluate a single reference-generated pair.
        
        Args:
            reference: Reference text.
            generated: Generated text.
            
        Returns:
            Adherence score for this pair.
        """
        # Normalize texts
        ref_norm = self._normalize_text(reference)
        gen_norm = self._normalize_text(generated)
        
        # Calculate component scores
        exact_score = self._exact_match_score(ref_norm, gen_norm)
        similarity_score = self._similarity_score(ref_norm, gen_norm)
        semantic_score = self._semantic_score(ref_norm, gen_norm)
        length_score = self._length_score(ref_norm, gen_norm)
        
        # Combine scores
        total_score = (
            self.exact_match_weight * exact_score +
            self.similarity_weight * similarity_score +
            self.semantic_weight * semantic_score +
            self.length_weight * length_score
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _exact_match_score(self, reference: str, generated: str) -> float:
        """Calculate exact match score."""
        return 1.0 if reference == generated else 0.0
    
    def _similarity_score(self, reference: str, generated: str) -> float:
        """Calculate sequence similarity score using difflib."""
        if not reference and not generated:
            return 1.0
        if not reference or not generated:
            return 0.0
        
        matcher = SequenceMatcher(None, reference, generated)
        return matcher.ratio()
    
    def _semantic_score(self, reference: str, generated: str) -> float:
        """
        Calculate semantic similarity using keyword matching.
        
        This is a basic implementation. For production use, consider
        using more sophisticated methods like sentence embeddings.
        """
        if not reference and not generated:
            return 1.0
        if not reference or not generated:
            return 0.0
        
        # Extract keywords (simple word splitting)
        ref_words = set(reference.split())
        gen_words = set(generated.split())
        
        if not ref_words and not gen_words:
            return 1.0
        if not ref_words or not gen_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(ref_words & gen_words)
        union = len(ref_words | gen_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _length_score(self, reference: str, generated: str) -> float:
        """
        Calculate length penalty score.
        
        Penalizes generated text that is too long or too short.
        """
        if not reference:
            return 1.0 if not generated else 0.0
        
        ref_len = len(reference)
        gen_len = len(generated)
        
        if ref_len == 0:
            return 1.0 if gen_len == 0 else 0.0
        
        ratio = gen_len / ref_len
        
        # Ideal ratio is 1.0, penalize deviations
        if ratio <= 1.0:
            # Short text penalty (less severe)
            return ratio
        else:
            # Long text penalty (more severe)
            if ratio > self.max_length_ratio:
                return 0.0
            else:
                return max(0.0, 1.0 - (ratio - 1.0) / (self.max_length_ratio - 1.0))


class BLEUEvaluator(Evaluator):
    """
    Simple BLEU-like evaluator for text similarity.
    
    Note: This is a simplified implementation. For production use,
    consider using libraries like sacrebleu or nltk.
    """
    
    def __init__(self, n_grams: int = 4):
        """
        Initialize BLEU evaluator.
        
        Args:
            n_grams: Maximum n-gram size to consider.
        """
        self.n_grams = n_grams
    
    def run(self, inputs: List[str], references: List[str], 
            generated: List[str]) -> float:
        """
        Evaluate using BLEU-like metric.
        
        Args:
            inputs: List of input prompts (not used).
            references: List of reference outputs.
            generated: List of generated outputs.
            
        Returns:
            Average BLEU-like score.
        """
        if len(references) != len(generated):
            raise ValueError(f"Number of references ({len(references)}) "
                           f"must match number of generated ({len(generated)})")
        
        if not references:
            return 0.0
        
        scores = []
        for ref, gen in zip(references, generated):
            score = self._bleu_score(ref, gen)
            scores.append(score)
        
        return float(np.mean(scores))
    
    def _bleu_score(self, reference: str, generated: str) -> float:
        """Calculate BLEU-like score for a single pair."""
        # Tokenize
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()
        
        if not ref_tokens and not gen_tokens:
            return 1.0
        if not ref_tokens or not gen_tokens:
            return 0.0
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, self.n_grams + 1):
            precision = self._ngram_precision(ref_tokens, gen_tokens, n)
            precisions.append(precision)
        
        # Geometric mean of precisions
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        precisions = [max(p, epsilon) for p in precisions]
        bleu = np.exp(np.mean(np.log(precisions)))
        
        # Brevity penalty
        bp = self._brevity_penalty(len(ref_tokens), len(gen_tokens))
        
        return bleu * bp
    
    def _ngram_precision(self, ref_tokens: List[str], 
                        gen_tokens: List[str], n: int) -> float:
        """Calculate n-gram precision."""
        if len(gen_tokens) < n:
            return 0.0
        
        # Generate n-grams
        ref_ngrams = [tuple(ref_tokens[i:i+n]) 
                     for i in range(len(ref_tokens) - n + 1)]
        gen_ngrams = [tuple(gen_tokens[i:i+n]) 
                     for i in range(len(gen_tokens) - n + 1)]
        
        if not gen_ngrams:
            return 0.0
        
        # Count matches
        ref_ngram_counts = {}
        for ngram in ref_ngrams:
            ref_ngram_counts[ngram] = ref_ngram_counts.get(ngram, 0) + 1
        
        matches = 0
        for ngram in gen_ngrams:
            if ngram in ref_ngram_counts and ref_ngram_counts[ngram] > 0:
                matches += 1
                ref_ngram_counts[ngram] -= 1
        
        return matches / len(gen_ngrams)
    
    def _brevity_penalty(self, ref_len: int, gen_len: int) -> float:
        """Calculate brevity penalty."""
        if gen_len >= ref_len:
            return 1.0
        elif gen_len == 0:
            return 0.0
        else:
            return np.exp(1 - ref_len / gen_len) 