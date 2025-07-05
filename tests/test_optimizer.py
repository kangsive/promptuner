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
        self.generator = MockGenerator()
        self.evaluator = MockEvaluator()
        self.dataset = MockDataset("test.csv")
        
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
        assert optimizer.iterations == 0
    
    def test_optimizer_init_with_task_description(self):
        """Test optimizer initialization with task description."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
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
                output_dir=self.temp_dir
            )
    
    def test_generate_initial_prompt(self):
        """Test initial prompt generation."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
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
            initial_prompt="test prompt",
            generation_size=2,
            output_dir=self.temp_dir
        )
        
        # Mock generator to return specific response
        with patch.object(optimizer.generator, 'run') as mock_run:
            mock_run.return_value = "1. First improved prompt\n2. Second improved prompt"
            
            candidates = optimizer._generate_candidates("current prompt")
            
            assert len(candidates) == 2
            assert candidates[0] == "First improved prompt"
            assert candidates[1] == "Second improved prompt"
    
    def test_parse_candidates(self):
        """Test candidate parsing."""
        optimizer = PromptOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            dataset=self.dataset,
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
            initial_prompt="test prompt",
            generation_size=1,
            max_depth=1,  # Very shallow
            max_iterations=10,
            output_dir=self.temp_dir
        )
        
        with patch.object(optimizer, '_evaluate_prompt') as mock_eval, \
             patch.object(optimizer, '_save_results') as mock_save:
            
            # Mock evaluation
            mock_eval.return_value = 0.5
            
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