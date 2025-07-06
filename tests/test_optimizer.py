"""Tests for optimizer module."""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from promptuner.optimizer import PromptOptimizer, PromptNode
from promptuner.datasets import InMemoryDataset
from tests.test_base import MockGenerator, MockEvaluator, MockDataset


class TestPromptNode:
    """Test PromptNode class."""
    
    def test_prompt_node_creation(self):
        """Test PromptNode creation."""
        node = PromptNode(
            prompt="test prompt",
            score=0.8,
            depth=1,
            generation=1
        )
        
        assert node.prompt == "test prompt"
        assert node.score == 0.8
        assert node.depth == 1
        assert node.generation == 1
        assert node.children_ids == []
        assert node.evaluated is False
    
    def test_prompt_node_id_generation(self):
        """Test node ID generation."""
        node1 = PromptNode(prompt="test prompt 1")
        node2 = PromptNode(prompt="test prompt 2")
        node3 = PromptNode(prompt="test prompt 1")  # Same prompt
        
        assert node1.node_id != node2.node_id
        assert node1.node_id == node3.node_id  # Same prompt should have same ID
        assert node1.node_id.startswith("node_")
    
    def test_prompt_node_post_init(self):
        """Test PromptNode __post_init__ method."""
        node = PromptNode(prompt="test prompt")
        assert node.children_ids == []
        
        # Test with provided children_ids
        node2 = PromptNode(prompt="test prompt", children_ids=["child1", "child2"])
        assert node2.children_ids == ["child1", "child2"]


class TestPromptOptimizer:
    """Test PromptOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from promptuner.analysts import RuleBasedAnalyst
        self.generator = MockGenerator()
        self.evaluator = MockEvaluator()
        self.dataset = MockDataset("test.csv")
        self.analyst = RuleBasedAnalyst()
        
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimizer_init_with_initial_prompt(self):
        """Test optimizer initialization with initial prompt."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test initial prompt",
            generation_size=2,
            max_depth=3,
            max_iterations=50,
            output_dir=self.temp_dir
        )
        
        assert optimizer.root_prompt == "test initial prompt"
        assert optimizer.generation_size == 2
        assert optimizer.max_depth == 3
        assert optimizer.max_iterations == 50
        assert len(optimizer.prompt_tree) == 1
        assert optimizer.best_prompt is None
        assert optimizer.best_score == -float('inf')
        assert optimizer.baseline_score == -float('inf')
        assert optimizer.iterations == 0
        assert optimizer.failed_prompts == []
        assert optimizer.analyst == self.analyst
    
    def test_optimizer_init_with_task_description(self):
        """Test optimizer initialization with task description."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            task_description="test task description",
            output_dir=self.temp_dir
        )
        
        # Should generate initial prompt from task description
        assert optimizer.root_prompt.startswith("Generated:")
        assert "test task description" in optimizer.root_prompt
    
    def test_optimizer_init_without_prompt_or_task(self):
        """Test optimizer initialization without prompt or task description."""
        with pytest.raises(ValueError, match="Either initial_prompt or task_description must be provided"):
            PromptOptimizer(
                generator=self.generator,
                evaluator=self.evaluator,
                dataset=self.dataset,
                analyst=self.analyst,
                output_dir=self.temp_dir
            )
    
    def test_generate_initial_prompt(self):
        """Test initial prompt generation."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        task_description = "Summarize text"
        prompt = optimizer._generate_initial_prompt(task_description)
        
        assert prompt.startswith("Generated:")
        assert "Summarize text" in prompt
    
    def test_generate_candidates(self):
        """Test candidate generation."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            output_dir=self.temp_dir
        )
        
        # Create a node for testing
        node = PromptNode(prompt="current prompt")
        
        # Mock generator to return specific response
        with patch.object(optimizer.generator, 'run') as mock_run:
            mock_run.return_value = "1. First improved prompt\n2. Second improved prompt"
            
            candidates = optimizer._generate_candidates(node)
            
            assert len(candidates) == 2
            assert candidates[0] == "First improved prompt"
            assert candidates[1] == "Second improved prompt"
    
    def test_parse_candidates(self):
        """Test candidate parsing."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Test numbered format
        response = "1. First prompt\n2. Second prompt\n3. Third prompt"
        candidates = optimizer._parse_candidates(response)
        
        assert len(candidates) == 3
        assert candidates[0] == "First prompt"
        assert candidates[1] == "Second prompt"
        assert candidates[2] == "Third prompt"
        
        # Test bullet format
        response = "- First prompt\n- Second prompt"
        candidates = optimizer._parse_candidates(response)
        
        assert len(candidates) == 2
        assert candidates[0] == "First prompt"
        assert candidates[1] == "Second prompt"
    
    def test_evaluate_prompt(self):
        """Test prompt evaluation."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Get root node
        root_node = list(optimizer.prompt_tree.values())[0]
        
        # Mock evaluator to return specific score
        with patch.object(optimizer.evaluator, 'run') as mock_run:
            mock_run.return_value = 0.85
            
            score = optimizer._evaluate_prompt(root_node.node_id)
            
            assert score == 0.85
            assert root_node.score == 0.85
            assert root_node.evaluated is True
            assert optimizer.best_score == 0.85
            assert optimizer.best_prompt == "test prompt"
    
    def test_evaluate_prompt_error_handling(self):
        """Test prompt evaluation with error."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Get root node
        root_node = list(optimizer.prompt_tree.values())[0]
        
        # Mock evaluator to raise exception
        with patch.object(optimizer.evaluator, 'run') as mock_run:
            mock_run.side_effect = Exception("Evaluation error")
            
            score = optimizer._evaluate_prompt(root_node.node_id)
            
            assert score == -float('inf')
            assert root_node.score == -float('inf')
            assert root_node.evaluated is True
    
    def test_backtrack(self):
        """Test backtracking logic."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Create a tree structure for testing
        root_node = list(optimizer.prompt_tree.values())[0]
        
        # Add child nodes
        child1 = PromptNode(
            prompt="child 1",
            parent_id=root_node.node_id,
            depth=1,
            evaluated=True,
            score=0.5
        )
        child2 = PromptNode(
            prompt="child 2", 
            parent_id=root_node.node_id,
            depth=1,
            evaluated=False
        )
        
        child1_id = child1.node_id
        child2_id = child2.node_id
        
        optimizer.prompt_tree[child1_id] = child1
        optimizer.prompt_tree[child2_id] = child2
        root_node.children_ids = [child1_id, child2_id]
        
        # Set current node to child1
        optimizer.current_node_id = child1_id
        
        # Test backtrack
        next_node_id = optimizer._backtrack()
        
        # Should return a valid node ID or None
        if next_node_id is not None:
            assert next_node_id in optimizer.prompt_tree
            # Should be either child2 (unevaluated) or some other valid node
            assert optimizer.prompt_tree[next_node_id] is not None
        # If None, it means no more nodes to explore
    
    def test_backtrack_no_more_nodes(self):
        """Test backtracking when no more nodes to explore."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Get root node and mark as evaluated
        root_node = list(optimizer.prompt_tree.values())[0]
        root_node.evaluated = True
        optimizer.current_node_id = root_node.node_id
        
        # Test backtrack
        next_node_id = optimizer._backtrack()
        
        # Should return None (no more nodes)
        assert next_node_id is None
    
    def test_get_best_prompt(self):
        """Test getting best prompt."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        optimizer.best_prompt = "best prompt"
        optimizer.best_score = 0.9
        
        prompt, score = optimizer.get_best_prompt()
        
        assert prompt == "best prompt"
        assert score == 0.9
    
    def test_save_results(self):
        """Test saving results."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        optimizer.best_prompt = "best prompt"
        optimizer.best_score = 0.9
        optimizer.iterations = 5
        
        # Call save results
        optimizer._save_results()
        
        # Check that files were created
        tree_file = Path(self.temp_dir) / "prompt_tree.json"
        best_file = Path(self.temp_dir) / "best_prompt.txt"
        
        assert tree_file.exists()
        assert best_file.exists()
        
        # Check tree file content
        with open(tree_file, 'r') as f:
            tree_data = json.load(f)
        
        assert tree_data['metadata']['best_score'] == 0.9
        assert tree_data['metadata']['best_prompt'] == "best prompt"
        assert tree_data['metadata']['iterations_completed'] == 5
        assert 'tree' in tree_data
        
        # Check best prompt file content
        with open(best_file, 'r') as f:
            content = f.read()
        
        assert "Best Score: 0.9000" in content
        assert "best prompt" in content
    
    def test_load_tree(self):
        """Test loading saved tree."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Create test tree data
        tree_data = {
            'metadata': {
                'best_score': 0.9,
                'best_prompt': "loaded best prompt",
                'root_prompt': "loaded root prompt",
                'iterations_completed': 10,
                'generation_size': 3,
                'max_depth': 5,
                'max_iterations': 100
            },
            'tree': {
                'node_123456': {
                    'prompt': "loaded prompt",
                    'score': 0.9,
                    'parent_id': None,
                    'children_ids': [],
                    'depth': 0,
                    'generation': 0,
                    'evaluated': True
                }
            }
        }
        
        # Save tree data to file
        tree_file = Path(self.temp_dir) / "test_tree.json"
        with open(tree_file, 'w') as f:
            json.dump(tree_data, f)
        
        # Load tree
        optimizer.load_tree(str(tree_file))
        
        # Check loaded data
        assert optimizer.best_score == 0.9
        assert optimizer.best_prompt == "loaded best prompt"
        assert optimizer.root_prompt == "loaded root prompt"
        assert optimizer.iterations == 10
        assert len(optimizer.prompt_tree) == 1
        
        # Check loaded node
        node = optimizer.prompt_tree['node_123456']
        assert node.prompt == "loaded prompt"
        assert node.score == 0.9
        assert node.evaluated is True
    
    def test_print_tree_summary(self):
        """Test printing tree summary."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        optimizer.best_prompt = "best prompt"
        optimizer.best_score = 0.9
        optimizer.iterations = 5
        
        # Add a child node
        root_node = list(optimizer.prompt_tree.values())[0]
        child = PromptNode(
            prompt="child prompt",
            parent_id=root_node.node_id,
            depth=1,
            score=0.7,
            evaluated=True
        )
        optimizer.prompt_tree[child.node_id] = child
        root_node.children_ids = [child.node_id]
        
        # Capture output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            optimizer.print_tree_summary()
            output = captured_output.getvalue()
            
            # Check output content
            assert "PROMPT TREE SUMMARY" in output
            assert "Total nodes: 2" in output
            assert "Iterations completed: 5" in output
            assert "Best score: 0.9000" in output
            assert "best prompt" in output
            assert "Tree Structure:" in output
        finally:
            sys.stdout = sys.__stdout__
    
    @patch('promptuner.optimizer.tqdm')
    def test_run_optimization_simple(self, mock_tqdm):
        """Test running optimization with simple case."""
        # Mock tqdm to avoid progress bar in tests
        mock_tqdm.return_value = ["input1", "input2"]
        
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            max_depth=2,
            max_iterations=2,
            output_dir=self.temp_dir
        )
        
        # Mock methods for controlled behavior
        with patch.object(optimizer, '_generate_candidates') as mock_gen_candidates, \
             patch.object(optimizer, '_evaluate_prompt') as mock_eval_prompt, \
             patch.object(optimizer, '_save_results') as mock_save:
            
            # Mock candidate generation
            mock_gen_candidates.return_value = ["candidate 1", "candidate 2"]
            
            # Mock evaluation to ensure proper scoring
            def mock_eval_side_effect(node_id):
                node = optimizer.prompt_tree[node_id]
                if "test prompt" in node.prompt:
                    score = 0.5
                elif "candidate 1" in node.prompt:
                    score = 0.3
                elif "candidate 2" in node.prompt:
                    score = 0.7
                else:
                    score = 0.4
                
                # Update the node properly
                node.score = score
                node.evaluated = True
                
                # Update best if needed
                if score > optimizer.best_score:
                    optimizer.best_score = score
                    optimizer.best_prompt = node.prompt
                
                return score
            
            mock_eval_prompt.side_effect = mock_eval_side_effect
            
            # Run optimization
            best_prompt, best_score = optimizer.run()
            
            # Should find a better prompt
            assert best_score > 0.5
            
            # Should have called save_results
            mock_save.assert_called_once()
    
    def test_run_optimization_max_depth_reached(self):
        """Test optimization stopping at max depth."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=1,
            max_depth=1,  # Very shallow
            max_iterations=10,
            output_dir=self.temp_dir
        )
        
        with patch.object(optimizer, '_evaluate_prompt') as mock_eval, \
             patch.object(optimizer, '_save_results') as mock_save:
            
            # Mock evaluation with proper node update
            def mock_eval_side_effect(node_id):
                node = optimizer.prompt_tree[node_id]
                node.score = 0.5
                node.evaluated = True
                return 0.5
            
            mock_eval.side_effect = mock_eval_side_effect
            
            # Run optimization
            best_prompt, best_score = optimizer.run()
            
            # Should have stopped due to max depth
            assert optimizer.iterations <= 2  # Should stop quickly
            mock_save.assert_called_once()
    
    def test_run_optimization_no_candidates(self):
        """Test optimization when no candidates are generated."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=1,
            max_depth=3,
            max_iterations=5,
            output_dir=self.temp_dir
        )
        
        with patch.object(optimizer, '_generate_candidates') as mock_gen_candidates, \
             patch.object(optimizer, '_evaluate_prompt') as mock_eval, \
             patch.object(optimizer, '_save_results') as mock_save:
            
            # Mock no candidates generated
            mock_gen_candidates.return_value = []
            mock_eval.return_value = 0.5
            
            # Run optimization
            best_prompt, best_score = optimizer.run()
            
            # Should have stopped due to no candidates
            assert optimizer.iterations <= 2
            mock_save.assert_called_once()
    
    def test_optimizer_with_analyst(self):
        """Test optimizer with analyst integration."""
        from promptuner.analysts import RuleBasedAnalyst
        
        analyst = RuleBasedAnalyst(low_score_threshold=0.4)
        
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        assert optimizer.analyst == analyst
        
        # Test that analysis is performed during evaluation
        root_node = list(optimizer.prompt_tree.values())[0]
        
        # Mock evaluator to return detailed scores
        with patch.object(optimizer.evaluator, 'run_detailed') as mock_run_detailed:
            mock_run_detailed.return_value = (0.6, [0.3, 0.9])  # Overall score, individual scores
            
            score = optimizer._evaluate_prompt(root_node.node_id)
            
            assert score == 0.6
            assert root_node.individual_scores == [0.3, 0.9]
            assert root_node.analysis_result is not None
            assert 'summary' in root_node.analysis_result
    
    def test_generate_candidates_with_feedback(self):
        """Test candidate generation with analyst feedback."""
        from promptuner.analysts import RuleBasedAnalyst
        
        analyst = RuleBasedAnalyst(low_score_threshold=0.4)
        
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=analyst,
            task_description="Test task",
            output_dir=self.temp_dir
        )
        
        # Create a node with analysis result
        node = PromptNode(
            prompt="test prompt",
            analysis_result={
                'summary': 'Test summary',
                'patterns': ['Pattern 1', 'Pattern 2'],
                'improvement_suggestions': ['Suggestion 1', 'Suggestion 2'],
                'hypothesis': 'Test hypothesis'
            }
        )
        
        # Add failed prompts
        optimizer.failed_prompts = [
            {'prompt': 'failed prompt 1', 'score': 0.2, 'baseline_score': 0.5},
            {'prompt': 'failed prompt 2', 'score': 0.3, 'baseline_score': 0.5}
        ]
        
        # Mock generator to return specific response
        with patch.object(optimizer.generator, 'run') as mock_run:
            mock_run.return_value = "1. Improved prompt with feedback\n2. Another improved prompt"
            
            candidates = optimizer._generate_candidates(node)
            
            assert len(candidates) == 2
            assert candidates[0] == "Improved prompt with feedback"
            assert candidates[1] == "Another improved prompt"
            
            # Check that the generation prompt includes feedback
            call_args = mock_run.call_args[0][0]
            assert 'EVALUATION FEEDBACK' in call_args
            assert 'FAILED PROMPTS TO AVOID' in call_args
            assert 'Test summary' in call_args
            assert 'Pattern 1' in call_args
    
    def test_failed_prompts_tracking(self):
        """Test failed prompts tracking."""
        from promptuner.analysts import RuleBasedAnalyst
        
        analyst = RuleBasedAnalyst()
        
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Set baseline score
        optimizer.baseline_score = 0.5
        
        # Create a child node with lower score
        child_node = PromptNode(
            prompt="worse prompt",
            score=0.3,
            analysis_result={'summary': 'Poor performance'}
        )
        
        # Simulate adding failed prompt
        optimizer.failed_prompts.append({
            'prompt': child_node.prompt,
            'score': child_node.score,
            'baseline_score': optimizer.baseline_score,
            'analysis': child_node.analysis_result
        })
        
        assert len(optimizer.failed_prompts) == 1
        assert optimizer.failed_prompts[0]['prompt'] == "worse prompt"
        assert optimizer.failed_prompts[0]['score'] == 0.3
        assert optimizer.failed_prompts[0]['baseline_score'] == 0.5
    
    def test_save_results_with_analyst(self):
        """Test saving results with analyst data."""
        from promptuner.analysts import RuleBasedAnalyst
        
        analyst = RuleBasedAnalyst()
        
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=analyst,
            initial_prompt="test prompt",
            output_dir=self.temp_dir
        )
        
        # Set some data
        optimizer.best_prompt = "best prompt"
        optimizer.best_score = 0.85
        optimizer.baseline_score = 0.6
        optimizer.failed_prompts = [
            {'prompt': 'failed prompt', 'score': 0.4, 'baseline_score': 0.6}
        ]
        
        # Add analysis result to root node
        root_node = list(optimizer.prompt_tree.values())[0]
        root_node.analysis_result = {'summary': 'Test analysis'}
        root_node.individual_scores = [0.7, 0.8]
        
        # Save results
        optimizer._save_results()
        
        # Check that files were created
        tree_file = Path(optimizer.output_dir) / "prompt_tree.json"
        best_file = Path(optimizer.output_dir) / "best_prompt.txt"
        
        assert tree_file.exists()
        assert best_file.exists()
        
        # Check tree file content
        with open(tree_file, 'r') as f:
            tree_data = json.load(f)
        
        assert 'failed_prompts' in tree_data
        assert tree_data['metadata']['baseline_score'] == 0.6
        assert tree_data['metadata']['failed_prompts_count'] == 1
        
        # Check that node has analysis result
        node_data = list(tree_data['tree'].values())[0]
        assert 'analysis_result' in node_data
        assert 'individual_scores' in node_data
        
        # Check best prompt file content
        with open(best_file, 'r') as f:
            content = f.read()
        
        assert "Baseline Score: 0.6" in content
        assert "Improvement: 0.25" in content


class TestHypothesisIntegration:
    """Test hypothesis system integration with optimizer."""
    
    def setup_method(self):
        """Set up test environment."""
        import tempfile
        import os
        from tests.test_base import MockGenerator, MockEvaluator
        from promptuner.datasets import InMemoryDataset
        from promptuner.analysts import RuleBasedAnalyst
        from promptuner.optimizer import PromptOptimizer
        
        self.temp_dir = tempfile.mkdtemp()
        self.generator = MockGenerator()
        self.evaluator = MockEvaluator()
        self.dataset = InMemoryDataset(
            ["test input 1", "test input 2"],
            ["test ref 1", "test ref 2"]
        )
        self.analyst = RuleBasedAnalyst(low_score_threshold=0.4)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_optimizer_with_hypothesis_tracking(self):
        """Test optimizer with hypothesis tracking."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            max_depth=3,
            max_iterations=5,
            output_dir=self.temp_dir
        )
        
        # Get the root node
        root_node = list(optimizer.prompt_tree.values())[0]
        
        # Mock the evaluator to return detailed scores
        with patch.object(optimizer.evaluator, 'run_detailed') as mock_run_detailed:
            # Mock low scores to trigger hypothesis generation
            mock_run_detailed.return_value = (0.3, [0.2, 0.4])
            
            # Evaluate the root node
            score = optimizer._evaluate_prompt(root_node.node_id)
            
            # Should have generated a new hypothesis
            assert root_node.new_hypothesis is not None
            assert root_node.analysis_result is not None
            assert 'new_hypothesis' in root_node.analysis_result
    
    def test_hypothesis_inheritance(self):
        """Test that hypotheses are properly inherited and considered."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            max_depth=3,
            max_iterations=5,
            output_dir=self.temp_dir
        )
        
        # Setup parent node with hypotheses
        parent_node = list(optimizer.prompt_tree.values())[0]
        parent_node.verified_hypothesis = ["Good strategy A"]
        parent_node.false_hypothesis = ["Bad strategy B"]
        
        # Create child node
        child_node = PromptNode(
            prompt="Child test prompt",
            parent_id=parent_node.node_id,
            depth=1,
            generation=1
        )
        
        optimizer.prompt_tree[child_node.node_id] = child_node
        
        # Test ancestor hypothesis retrieval
        verified, false = child_node.get_ancestor_hypotheses(optimizer.prompt_tree)
        
        assert "Good strategy A" in verified
        assert "Bad strategy B" in false
    
    def test_candidate_generation_with_hypotheses(self):
        """Test candidate generation considers ancestor hypotheses."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            max_depth=3,
            max_iterations=5,
            output_dir=self.temp_dir
        )
        
        # Setup node with analysis result and hypotheses
        node = list(optimizer.prompt_tree.values())[0]
        node.analysis_result = {
            'summary': 'Test summary',
            'patterns': ['Test pattern'],
            'improvement_suggestions': ['Test suggestion'],
            'hypothesis': 'Test hypothesis',
            'new_hypothesis': 'Test new hypothesis'
        }
        
        # Mock parent with hypotheses
        parent_node = PromptNode(
            prompt="Parent prompt",
            verified_hypothesis=["Use clear instructions"],
            false_hypothesis=["Avoid complex wording"]
        )
        node.parent_id = parent_node.node_id
        optimizer.prompt_tree[parent_node.node_id] = parent_node
        
        # Mock generator to return test candidates
        with patch.object(optimizer.generator, 'run') as mock_run:
            mock_run.return_value = "1. Test candidate 1\n2. Test candidate 2"
            
            candidates = optimizer._generate_candidates(node)
            
            # Should generate candidates
            assert len(candidates) == 2
            assert "Test candidate 1" in candidates
            assert "Test candidate 2" in candidates
            
            # Check that the generation prompt included hypothesis context
            call_args = mock_run.call_args[0][0]
            assert "ANCESTOR HYPOTHESES CONTEXT" in call_args
            assert "VERIFIED STRATEGIES" in call_args
            assert "FAILED STRATEGIES" in call_args
    
    def test_hypothesis_update_integration(self):
        """Test hypothesis update integration in optimizer."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            max_depth=3,
            max_iterations=5,
            output_dir=self.temp_dir
        )
        
        # Setup parent node with hypothesis
        parent_node = list(optimizer.prompt_tree.values())[0]
        parent_node.new_hypothesis = "Test hypothesis"
        parent_node.score = 0.4
        parent_node.individual_scores = [0.2, 0.6]
        parent_node.evaluated = True
        
        # Create child node
        child_node = PromptNode(
            prompt="Child prompt",
            parent_id=parent_node.node_id,
            depth=1,
            generation=1,
            score=0.7,
            individual_scores=[0.6, 0.8],
            evaluated=True
        )
        
        optimizer.prompt_tree[child_node.node_id] = child_node
        parent_node.children_ids = [child_node.node_id]
        
        # Update parent hypotheses
        optimizer._update_parent_hypotheses(parent_node.node_id)
        
        # Should verify hypothesis since child improved
        assert parent_node.new_hypothesis in parent_node.verified_hypothesis
        assert parent_node.new_hypothesis not in parent_node.false_hypothesis
    
    def test_save_and_load_with_hypotheses(self):
        """Test saving and loading tree with hypothesis data."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            max_depth=3,
            max_iterations=5,
            output_dir=self.temp_dir
        )
        
        # Setup node with hypotheses
        node = list(optimizer.prompt_tree.values())[0]
        node.new_hypothesis = "Test new hypothesis"
        node.verified_hypothesis = ["Verified hyp 1", "Verified hyp 2"]
        node.false_hypothesis = ["False hyp 1"]
        
        # Save results
        optimizer._save_results()
        
        # Load into new optimizer
        optimizer2 = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
            analyst=self.analyst,
            initial_prompt="test prompt",
            generation_size=2,
            max_depth=3,
            max_iterations=5,
            output_dir=self.temp_dir
        )
        
        tree_file = Path(self.temp_dir) / "prompt_tree.json"
        optimizer2.load_tree(str(tree_file))
        
        # Check that hypotheses were loaded
        loaded_node = list(optimizer2.prompt_tree.values())[0]
        assert loaded_node.new_hypothesis == "Test new hypothesis"
        assert "Verified hyp 1" in loaded_node.verified_hypothesis
        assert "Verified hyp 2" in loaded_node.verified_hypothesis
        assert "False hyp 1" in loaded_node.false_hypothesis 