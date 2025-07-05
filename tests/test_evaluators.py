"""Tests for evaluators module."""

import pytest
from promptuner.evaluators import AdherenceEvaluator, BLEUEvaluator


class TestAdherenceEvaluator:
    """Test AdherenceEvaluator class."""
    
    def test_init(self):
        """Test AdherenceEvaluator initialization."""
        evaluator = AdherenceEvaluator(
            exact_match_weight=0.25,
            similarity_weight=0.35,
            semantic_weight=0.25,
            length_weight=0.15
        )
        
        assert evaluator.exact_match_weight == 0.25
        assert evaluator.similarity_weight == 0.35
        assert evaluator.semantic_weight == 0.25
        assert evaluator.length_weight == 0.15
    
    def test_init_default_weights(self):
        """Test AdherenceEvaluator with default weights."""
        evaluator = AdherenceEvaluator()
        
        assert evaluator.exact_match_weight == 0.3
        assert evaluator.similarity_weight == 0.4
        assert evaluator.semantic_weight == 0.2
        assert evaluator.length_weight == 0.1
    
    def test_run_exact_match(self):
        """Test perfect exact match."""
        evaluator = AdherenceEvaluator()
        
        inputs = ["test input"]
        references = ["test reference"]
        generated = ["test reference"]
        
        score = evaluator.run(inputs, references, generated)
        
        # Should get high score for exact match
        assert score > 0.8
    
    def test_run_similar_texts(self):
        """Test similar but not exact texts."""
        evaluator = AdherenceEvaluator()
        
        inputs = ["test input"]
        references = ["the quick brown fox"]
        generated = ["the quick brown cat"]
        
        score = evaluator.run(inputs, references, generated)
        
        # Should get decent score for similar text
        assert 0.3 < score < 0.9
    
    def test_run_different_texts(self):
        """Test completely different texts."""
        evaluator = AdherenceEvaluator()
        
        inputs = ["test input"]
        references = ["the quick brown fox"]
        generated = ["completely different text"]
        
        score = evaluator.run(inputs, references, generated)
        
        # Should get low score for different text
        assert score < 0.5
    
    def test_run_empty_inputs(self):
        """Test with empty inputs."""
        evaluator = AdherenceEvaluator()
        
        inputs = []
        references = []
        generated = []
        
        score = evaluator.run(inputs, references, generated)
        assert score == 0.0
    
    def test_run_mismatched_lengths(self):
        """Test with mismatched input lengths."""
        evaluator = AdherenceEvaluator()
        
        inputs = ["test input"]
        references = ["test reference"]
        generated = ["test generated", "extra generated"]
        
        with pytest.raises(ValueError, match="Number of references"):
            evaluator.run(inputs, references, generated)
    
    def test_normalize_text(self):
        """Test text normalization."""
        evaluator = AdherenceEvaluator()
        
        text = "  Test   Text  With   Spaces  "
        normalized = evaluator._normalize_text(text)
        
        assert normalized == "test text with spaces"
    
    def test_exact_match_score(self):
        """Test exact match scoring."""
        evaluator = AdherenceEvaluator()
        
        # Exact match
        score = evaluator._exact_match_score("test", "test")
        assert score == 1.0
        
        # No match
        score = evaluator._exact_match_score("test", "different")
        assert score == 0.0
    
    def test_similarity_score(self):
        """Test similarity scoring."""
        evaluator = AdherenceEvaluator()
        
        # Identical strings
        score = evaluator._similarity_score("test", "test")
        assert score == 1.0
        
        # Similar strings
        score = evaluator._similarity_score("test", "tests")
        assert 0.5 < score < 1.0
        
        # Different strings
        score = evaluator._similarity_score("test", "different")
        assert score < 0.5
        
        # Empty strings
        score = evaluator._similarity_score("", "")
        assert score == 1.0
        
        # One empty string
        score = evaluator._similarity_score("test", "")
        assert score == 0.0
    
    def test_semantic_score(self):
        """Test semantic scoring."""
        evaluator = AdherenceEvaluator()
        
        # Identical words
        score = evaluator._semantic_score("test word", "test word")
        assert score == 1.0
        
        # Partial overlap
        score = evaluator._semantic_score("test word", "test different")
        assert abs(score - 1/3) < 0.001  # 1 word overlap out of 3 total words
        
        # No overlap
        score = evaluator._semantic_score("test word", "different text")
        assert score == 0.0
        
        # Empty strings
        score = evaluator._semantic_score("", "")
        assert score == 1.0
    
    def test_length_score(self):
        """Test length scoring."""
        evaluator = AdherenceEvaluator()
        
        # Same length
        score = evaluator._length_score("test", "word")
        assert score == 1.0
        
        # Generated shorter
        score = evaluator._length_score("test word", "test")
        assert abs(score - 4/9) < 0.001  # 4 chars / 9 chars ≈ 0.444
        
        # Generated longer but within limit
        score = evaluator._length_score("test", "test word")
        # For longer text, the penalty calculation should give a score between 0 and 1
        assert 0.0 <= score <= 1.0
        
        # Generated too long
        score = evaluator._length_score("test", "test word very long text")
        assert score == 0.0


class TestBLEUEvaluator:
    """Test BLEUEvaluator class."""
    
    def test_init(self):
        """Test BLEUEvaluator initialization."""
        evaluator = BLEUEvaluator(n_grams=3)
        assert evaluator.n_grams == 3
    
    def test_init_default(self):
        """Test BLEUEvaluator with default parameters."""
        evaluator = BLEUEvaluator()
        assert evaluator.n_grams == 4
    
    def test_run_exact_match(self):
        """Test BLEU with exact match."""
        evaluator = BLEUEvaluator()
        
        inputs = ["test input"]
        references = ["the quick brown fox"]
        generated = ["the quick brown fox"]
        
        score = evaluator.run(inputs, references, generated)
        
        # Should get perfect score for exact match
        assert score == 1.0
    
    def test_run_similar_texts(self):
        """Test BLEU with similar texts."""
        evaluator = BLEUEvaluator()
        
        inputs = ["test input"]
        references = ["the quick brown fox"]
        generated = ["the quick brown cat"]
        
        score = evaluator.run(inputs, references, generated)
        
        # Should get a low but positive score for similar text with epsilon
        assert 0.0 < score < 1.0
    
    def test_run_different_texts(self):
        """Test BLEU with different texts."""
        evaluator = BLEUEvaluator()
        
        inputs = ["test input"]
        references = ["the quick brown fox"]
        generated = ["completely different text"]
        
        score = evaluator.run(inputs, references, generated)
        
        # Should get low score for different text
        assert score < 0.5
    
    def test_run_empty_inputs(self):
        """Test BLEU with empty inputs."""
        evaluator = BLEUEvaluator()
        
        inputs = []
        references = []
        generated = []
        
        score = evaluator.run(inputs, references, generated)
        assert score == 0.0
    
    def test_run_mismatched_lengths(self):
        """Test BLEU with mismatched input lengths."""
        evaluator = BLEUEvaluator()
        
        inputs = ["test input"]
        references = ["test reference"]
        generated = ["test generated", "extra generated"]
        
        with pytest.raises(ValueError, match="Number of references"):
            evaluator.run(inputs, references, generated)
    
    def test_bleu_score_identical(self):
        """Test BLEU score for identical texts."""
        evaluator = BLEUEvaluator()
        
        score = evaluator._bleu_score("the quick brown fox", "the quick brown fox")
        assert score == 1.0
    
    def test_bleu_score_different(self):
        """Test BLEU score for different texts."""
        evaluator = BLEUEvaluator()
        
        score = evaluator._bleu_score("the quick brown fox", "completely different text")
        assert score < 0.5
    
    def test_bleu_score_empty(self):
        """Test BLEU score for empty texts."""
        evaluator = BLEUEvaluator()
        
        # Both empty
        score = evaluator._bleu_score("", "")
        assert score == 1.0
        
        # One empty
        score = evaluator._bleu_score("test", "")
        assert score == 0.0
        
        score = evaluator._bleu_score("", "test")
        assert score == 0.0
    
    def test_ngram_precision(self):
        """Test n-gram precision calculation."""
        evaluator = BLEUEvaluator()
        
        ref_tokens = ["the", "quick", "brown", "fox"]
        gen_tokens = ["the", "quick", "brown", "cat"]
        
        # 1-gram precision
        precision = evaluator._ngram_precision(ref_tokens, gen_tokens, 1)
        assert precision == 0.75  # 3 out of 4 words match
        
        # 2-gram precision
        precision = evaluator._ngram_precision(ref_tokens, gen_tokens, 2)
        assert precision == 2/3  # 2 out of 3 2-grams match
    
    def test_brevity_penalty(self):
        """Test brevity penalty calculation."""
        evaluator = BLEUEvaluator()
        
        # Same length
        bp = evaluator._brevity_penalty(4, 4)
        assert bp == 1.0
        
        # Generated longer
        bp = evaluator._brevity_penalty(4, 6)
        assert bp == 1.0
        
        # Generated shorter
        bp = evaluator._brevity_penalty(4, 2)
        assert 0.0 < bp < 1.0
        
        # Generated empty
        bp = evaluator._brevity_penalty(4, 0)
        assert bp == 0.0 