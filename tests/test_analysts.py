"""Tests for analysts module."""

import pytest
from unittest.mock import Mock, patch
from promptuner.analysts import LLMAnalyst, RuleBasedAnalyst
from promptuner.base import Generator


class MockGenerator:
    """Mock generator for testing."""
    
    def run(self, prompt: str, **kwargs) -> str:
        # Mock analysis response
        if "analyze these results" in prompt.lower():
            return """
SUMMARY: The prompt is performing poorly with low scores on multiple samples.
LOW_SCORE_ANALYSIS: The generated outputs are too short and lack detail.
PATTERNS:
- Short inputs tend to have lower scores
- Generated outputs are too brief
- Missing key information in outputs
IMPROVEMENT_SUGGESTIONS:
- Add instructions for more detailed responses
- Include examples in the prompt
- Add length requirements
HYPOTHESIS: The prompt lacks specificity and examples, leading to brief outputs.
"""
        return "Mock response"


class TestLLMAnalyst:
    """Test LLMAnalyst class."""
    
    def test_init(self):
        """Test LLMAnalyst initialization."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator, low_score_threshold=0.4)
        
        assert analyst.generator == generator
        assert analyst.low_score_threshold == 0.4
    
    def test_init_default_threshold(self):
        """Test LLMAnalyst with default threshold."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator)
        
        assert analyst.low_score_threshold == 0.3
    
    def test_run_with_low_scores(self):
        """Test analysis with low-scoring samples."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator, low_score_threshold=0.5)
        
        inputs = ["input1", "input2", "input3"]
        references = ["ref1", "ref2", "ref3"]
        generated = ["gen1", "gen2", "gen3"]
        scores = [0.2, 0.3, 0.6]  # Two low scores
        overall_score = 0.37
        current_prompt = "Test prompt"
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'low_score_analysis' in result
        assert 'patterns' in result
        assert 'improvement_suggestions' in result
        assert 'hypothesis' in result
        assert 'new_hypothesis' in result
        assert 'statistics' in result
        
        # Check that analysis identified issues
        assert "poorly" in result['summary'].lower()
        assert len(result['patterns']) > 0
        assert len(result['improvement_suggestions']) > 0
    
    def test_run_with_good_scores(self):
        """Test analysis with good scores."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator, low_score_threshold=0.3)
        
        inputs = ["input1", "input2"]
        references = ["ref1", "ref2"]
        generated = ["gen1", "gen2"]
        scores = [0.8, 0.9]  # Good scores
        overall_score = 0.85
        current_prompt = "Test prompt"
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert "good" in result['summary'].lower()
        assert result['low_score_analysis'] == "No low-scoring samples identified."
        assert len(result['patterns']) == 0
        assert len(result['improvement_suggestions']) == 0
    
    def test_identify_low_score_samples(self):
        """Test low-score sample identification."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator, low_score_threshold=0.5)
        
        inputs = ["input1", "input2", "input3"]
        references = ["ref1", "ref2", "ref3"]
        generated = ["gen1", "gen2", "gen3"]
        scores = [0.2, 0.6, 0.3]  # Two low scores
        
        low_samples = analyst._identify_low_score_samples(inputs, references, generated, scores)
        
        assert len(low_samples) == 2
        assert low_samples[0]['score'] == 0.2  # Lowest first
        assert low_samples[1]['score'] == 0.3
        assert low_samples[0]['input'] == "input1"
        assert low_samples[1]['input'] == "input3"
    
    def test_parse_analysis_response(self):
        """Test parsing of analysis response."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator)
        
        response = """
SUMMARY: Test summary here
LOW_SCORE_ANALYSIS: Test analysis here
PATTERNS:
- Pattern 1
- Pattern 2
IMPROVEMENT_SUGGESTIONS:
- Suggestion 1
- Suggestion 2
HYPOTHESIS: Test hypothesis here
"""
        
        result = analyst._parse_analysis_response(response)
        
        assert result['summary'] == "Test summary here"
        assert result['low_score_analysis'] == "Test analysis here"
        assert len(result['patterns']) == 2
        assert result['patterns'][0] == "Pattern 1"
        assert result['patterns'][1] == "Pattern 2"
        assert len(result['improvement_suggestions']) == 2
        assert result['improvement_suggestions'][0] == "Suggestion 1"
        assert result['improvement_suggestions'][1] == "Suggestion 2"
        assert result['hypothesis'] == "Test hypothesis here"
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator, low_score_threshold=0.5)
        
        scores = [0.1, 0.3, 0.6, 0.8, 0.9]
        stats = analyst._calculate_statistics(scores)
        
        assert stats['mean'] == 0.54
        assert stats['min'] == 0.1
        assert stats['max'] == 0.9
        assert stats['low_score_count'] == 2
        assert stats['low_score_percentage'] == 40.0
    
    def test_generator_error_handling(self):
        """Test handling of generator errors."""
        generator = Mock()
        generator.run.side_effect = Exception("Generator error")
        
        analyst = LLMAnalyst(generator, low_score_threshold=0.5)
        
        inputs = ["input1"]
        references = ["ref1"]
        generated = ["gen1"]
        scores = [0.2]
        overall_score = 0.2
        current_prompt = "Test prompt"
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'low_score_analysis' in result
        assert result['hypothesis'] == "Analysis failed - manual review needed."


class TestRuleBasedAnalyst:
    """Test RuleBasedAnalyst class."""
    
    def test_init(self):
        """Test RuleBasedAnalyst initialization."""
        analyst = RuleBasedAnalyst(low_score_threshold=0.4)
        assert analyst.low_score_threshold == 0.4
    
    def test_init_default_threshold(self):
        """Test RuleBasedAnalyst with default threshold."""
        analyst = RuleBasedAnalyst()
        assert analyst.low_score_threshold == 0.3
    
    def test_run_with_low_scores(self):
        """Test analysis with low-scoring samples."""
        analyst = RuleBasedAnalyst(low_score_threshold=0.5)
        
        inputs = ["short", "this is a much longer input text"]
        references = ["ref1", "ref2"]
        generated = ["gen1", "gen2"]
        scores = [0.2, 0.8]  # One low score
        overall_score = 0.5
        current_prompt = "Test prompt"
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'low_score_analysis' in result
        assert 'patterns' in result
        assert 'improvement_suggestions' in result
        assert 'hypothesis' in result
        assert 'statistics' in result
        
        assert "1 samples scored below" in result['low_score_analysis']
    
    def test_run_with_good_scores(self):
        """Test analysis with good scores."""
        analyst = RuleBasedAnalyst(low_score_threshold=0.3)
        
        inputs = ["input1", "input2"]
        references = ["ref1", "ref2"]
        generated = ["gen1", "gen2"]
        scores = [0.8, 0.9]  # Good scores
        overall_score = 0.85
        current_prompt = "Test prompt"
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt)
        
        assert isinstance(result, dict)
        assert "good" in result['summary'].lower()
        assert result['low_score_analysis'] == "No samples scored below the threshold."
    
    def test_analyze_patterns_long_inputs(self):
        """Test pattern analysis for long inputs."""
        analyst = RuleBasedAnalyst(low_score_threshold=0.5)
        
        inputs = ["short", "this is a much longer input text that should be detected"]
        references = ["ref1", "ref2"]
        generated = ["gen1", "gen2"]
        low_score_indices = [1]  # The long input has low score
        
        patterns = analyst._analyze_patterns(inputs, references, generated, low_score_indices)
        
        # Should detect that long inputs have lower scores
        assert any("long" in pattern.lower() for pattern in patterns)
    
    def test_analyze_patterns_short_outputs(self):
        """Test pattern analysis for short outputs."""
        analyst = RuleBasedAnalyst(low_score_threshold=0.5)
        
        inputs = ["input1", "input2"]
        references = ["ref1", "ref2"]
        generated = ["short", "this is a much longer generated output"]
        low_score_indices = [0]  # The short output has low score
        
        patterns = analyst._analyze_patterns(inputs, references, generated, low_score_indices)
        
        # Should detect that short outputs have lower scores
        assert any("short" in pattern.lower() for pattern in patterns)
    
    def test_generate_suggestions(self):
        """Test suggestion generation."""
        analyst = RuleBasedAnalyst()
        
        patterns = ["Long inputs tend to have lower scores", "Generated outputs are too short"]
        statistics = {'low_score_percentage': 40, 'std': 0.4}
        
        suggestions = analyst._generate_suggestions(patterns, statistics)
        
        assert len(suggestions) > 0
        assert any("long inputs" in suggestion.lower() for suggestion in suggestions)
        assert any("variance" in suggestion.lower() for suggestion in suggestions)
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        analyst = RuleBasedAnalyst(low_score_threshold=0.5)
        
        scores = [0.1, 0.3, 0.6, 0.8, 0.9]
        stats = analyst._calculate_statistics(scores)
        
        assert stats['mean'] == 0.54
        assert stats['min'] == 0.1
        assert stats['max'] == 0.9
        assert stats['low_score_count'] == 2
        assert stats['low_score_percentage'] == 40.0
    
    def test_empty_scores(self):
        """Test handling of empty scores."""
        analyst = RuleBasedAnalyst()
        
        inputs = []
        references = []
        generated = []
        scores = []
        overall_score = 0.0
        current_prompt = "Test prompt"
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'statistics' in result
        assert result['statistics'] == {} 


class TestHypothesisSystem:
    """Test hypothesis tracking system."""
    
    def test_llm_analyst_with_ancestor_hypotheses(self):
        """Test LLM analyst with ancestor hypotheses."""
        generator = MockGenerator()
        analyst = LLMAnalyst(generator, low_score_threshold=0.5)
        
        inputs = ["input1", "input2", "input3"]
        references = ["ref1", "ref2", "ref3"]
        generated = ["gen1", "gen2", "gen3"]
        scores = [0.2, 0.3, 0.6]  # Two low scores
        overall_score = 0.37
        current_prompt = "Test prompt"
        
        verified_hypotheses = ["Strategy A works well", "Strategy B improves performance"]
        false_hypotheses = ["Strategy C is ineffective", "Strategy D causes issues"]
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt,
                           verified_hypotheses, false_hypotheses)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'new_hypothesis' in result
        assert 'improvement_suggestions' in result
        
        # The analysis should consider ancestor hypotheses
        assert len(result['improvement_suggestions']) > 0
    
    def test_rule_based_analyst_with_ancestor_hypotheses(self):
        """Test rule-based analyst with ancestor hypotheses."""
        analyst = RuleBasedAnalyst(low_score_threshold=0.5)
        
        inputs = ["short", "this is a much longer input text"]
        references = ["ref1", "ref2"]
        generated = ["gen1", "gen2"]
        scores = [0.2, 0.8]  # One low score
        overall_score = 0.5
        current_prompt = "Test prompt"
        
        verified_hypotheses = ["Strategy A works well"]
        false_hypotheses = ["Strategy B is ineffective"]
        
        result = analyst.run(inputs, references, generated, scores, overall_score, current_prompt,
                           verified_hypotheses, false_hypotheses)
        
        assert isinstance(result, dict)
        assert 'new_hypothesis' in result
        assert result['new_hypothesis'] is not None
        
        # Should generate suggestions considering ancestor hypotheses
        assert len(result['improvement_suggestions']) > 0
    
    def test_update_parent_hypotheses_verified(self):
        """Test updating parent hypotheses - verified case."""
        from promptuner.optimizer import PromptNode
        
        generator = MockGenerator()
        analyst = LLMAnalyst(generator)
        
        # Create parent node with a hypothesis
        parent_node = PromptNode(
            prompt="Parent prompt",
            score=0.4,
            individual_scores=[0.2, 0.3, 0.5, 0.6],
            new_hypothesis="Test hypothesis to verify"
        )
        
        # Create child nodes that improve on parent's issues
        child1 = PromptNode(
            prompt="Child 1 prompt",
            score=0.6,
            individual_scores=[0.6, 0.7, 0.5, 0.6]
        )
        
        child2 = PromptNode(
            prompt="Child 2 prompt", 
            score=0.5,
            individual_scores=[0.4, 0.5, 0.6, 0.7]
        )
        
        # Update parent hypotheses
        analyst.update_parent_hypotheses(parent_node, [child1, child2])
        
        # Should verify hypothesis since child1 improved
        assert parent_node.new_hypothesis in parent_node.verified_hypothesis
        assert parent_node.new_hypothesis not in parent_node.false_hypothesis
    
    def test_update_parent_hypotheses_false(self):
        """Test updating parent hypotheses - false case."""
        from promptuner.optimizer import PromptNode
        
        generator = MockGenerator()
        analyst = LLMAnalyst(generator)
        
        # Create parent node with a hypothesis
        parent_node = PromptNode(
            prompt="Parent prompt",
            score=0.6,
            individual_scores=[0.6, 0.7, 0.5, 0.6],
            new_hypothesis="Test hypothesis to falsify"
        )
        
        # Create child nodes that don't improve on parent
        child1 = PromptNode(
            prompt="Child 1 prompt",
            score=0.4,
            individual_scores=[0.2, 0.3, 0.5, 0.6]
        )
        
        child2 = PromptNode(
            prompt="Child 2 prompt", 
            score=0.5,
            individual_scores=[0.4, 0.5, 0.4, 0.5]
        )
        
        # Update parent hypotheses
        analyst.update_parent_hypotheses(parent_node, [child1, child2])
        
        # Should falsify hypothesis since children didn't improve
        assert parent_node.new_hypothesis not in parent_node.verified_hypothesis
        assert parent_node.new_hypothesis in parent_node.false_hypothesis
    
    def test_prompt_node_get_ancestor_hypotheses(self):
        """Test getting ancestor hypotheses from prompt node."""
        from promptuner.optimizer import PromptNode
        
        # Create a tree of nodes
        root_node = PromptNode(
            prompt="Root prompt",
            verified_hypothesis=["Root verified"],
            false_hypothesis=["Root false"]
        )
        
        parent_node = PromptNode(
            prompt="Parent prompt",
            parent_id=root_node.node_id,
            verified_hypothesis=["Parent verified"],
            false_hypothesis=["Parent false"]
        )
        
        child_node = PromptNode(
            prompt="Child prompt",
            parent_id=parent_node.node_id
        )
        
        # Create tree structure
        prompt_tree = {
            root_node.node_id: root_node,
            parent_node.node_id: parent_node,
            child_node.node_id: child_node
        }
        
        # Get ancestor hypotheses
        verified, false = child_node.get_ancestor_hypotheses(prompt_tree)
        
        # Should get hypotheses from both parent and root
        assert "Parent verified" in verified
        assert "Root verified" in verified
        assert "Parent false" in false
        assert "Root false" in false
        assert len(verified) == 2
        assert len(false) == 2 