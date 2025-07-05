"""
Main prompt optimizer using DFS search on prompt family trees.

This module implements the core optimization algorithm that generates
and evaluates prompts using a tree-based search approach.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from tqdm import tqdm

from .base import Generator, Evaluator, Dataset

logger = logging.getLogger(__name__)


@dataclass
class PromptNode:
    """Node in the prompt family tree."""
    prompt: str
    score: Optional[float] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    depth: int = 0
    generation: int = 0
    evaluated: bool = False
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
    
    @property
    def node_id(self) -> str:
        """Generate unique node ID based on prompt hash."""
        return f"node_{hash(self.prompt) % 1000000:06d}"


class PromptOptimizer:
    """
    Main prompt optimizer using DFS search on prompt family trees.
    
    The optimizer maintains a tree of prompts where each node represents
    a prompt variant. It uses DFS to explore the tree, generating new
    prompts and evaluating them against a dataset.
    """
    
    def __init__(self, 
                 generator: Generator,
                 evaluator: Evaluator,
                 dataset: Dataset,
                 initial_prompt: Optional[str] = None,
                 task_description: Optional[str] = None,
                 generation_size: int = 3,
                 max_depth: int = 5,
                 max_iterations: int = 100,
                 output_dir: str = "outputs"):
        """
        Initialize the prompt optimizer.
        
        Args:
            generator: Text generator for creating new prompts.
            evaluator: Evaluator for scoring prompts.
            dataset: Dataset for evaluation.
            initial_prompt: Starting prompt (will be generated if None).
            task_description: Description of the task for prompt generation.
            generation_size: Number of prompt variants to generate per round.
            max_depth: Maximum depth of the prompt family tree.
            max_iterations: Maximum number of optimization iterations.
            output_dir: Directory to save results.
        """
        self.generator = generator
        self.evaluator = evaluator
        self.dataset = dataset
        self.task_description = task_description
        self.generation_size = generation_size
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize prompt tree
        self.prompt_tree: Dict[str, PromptNode] = {}
        self.best_prompt: Optional[str] = None
        self.best_score: float = -float('inf')
        self.iterations: int = 0
        
        # Initialize with root prompt
        if initial_prompt is None:
            if task_description is None:
                raise ValueError("Either initial_prompt or task_description must be provided")
            self.root_prompt = self._generate_initial_prompt(task_description)
        else:
            self.root_prompt = initial_prompt
        
        # Create root node
        root_node = PromptNode(
            prompt=self.root_prompt,
            depth=0,
            generation=0
        )
        self.prompt_tree[root_node.node_id] = root_node
        self.current_node_id = root_node.node_id
        
        logger.info(f"Initialized optimizer with root prompt: {self.root_prompt[:100]}...")
    
    def _generate_initial_prompt(self, task_description: str) -> str:
        """Generate initial prompt from task description."""
        generation_prompt = f"""
        Create an effective prompt for the following task:
        
        Task: {task_description}
        
        Generate a clear, specific prompt that will produce high-quality results for this task.
        Focus on being concise but comprehensive.
        
        Prompt:
        """
        
        try:
            return self.generator.run(generation_prompt.strip())
        except Exception as e:
            logger.error(f"Failed to generate initial prompt: {e}")
            raise
    
    def run(self) -> Tuple[str, float]:
        """
        Run the optimization process.
        
        Returns:
            Tuple of (best_prompt, best_score).
        """
        logger.info("Starting prompt optimization...")
        
        # Evaluate root prompt
        self._evaluate_prompt(self.current_node_id)
        
        # DFS search
        while (self.iterations < self.max_iterations and 
               self.current_node_id is not None):
            
            self.iterations += 1
            logger.info(f"Iteration {self.iterations}/{self.max_iterations}")
            
            current_node = self.prompt_tree[self.current_node_id]
            
            # Check if we've reached max depth
            if current_node.depth >= self.max_depth:
                logger.info(f"Reached max depth {self.max_depth} at node {self.current_node_id}")
                self.current_node_id = self._backtrack()
                continue
            
            # Generate new prompt candidates
            candidates = self._generate_candidates(current_node.prompt)
            
            if not candidates:
                logger.warning(f"No candidates generated for node {self.current_node_id}")
                self.current_node_id = self._backtrack()
                continue
            
            # Create child nodes
            child_nodes = []
            for candidate in candidates:
                child_node = PromptNode(
                    prompt=candidate,
                    parent_id=self.current_node_id,
                    depth=current_node.depth + 1,
                    generation=current_node.generation + 1
                )
                
                # Avoid duplicate prompts
                if child_node.node_id not in self.prompt_tree:
                    self.prompt_tree[child_node.node_id] = child_node
                    current_node.children_ids.append(child_node.node_id)
                    child_nodes.append(child_node)
            
            # Evaluate candidates and find best
            best_child = None
            best_child_score = -float('inf')
            
            for child_node in child_nodes:
                score = self._evaluate_prompt(child_node.node_id)
                
                if score > best_child_score:
                    best_child_score = score
                    best_child = child_node
            
            # Check if best child is better than current best
            if best_child and best_child_score > self.best_score:
                logger.info(f"Found better prompt with score {best_child_score:.4f}")
                self.current_node_id = best_child.node_id
                # Continue DFS from this node
            else:
                logger.info(f"No improvement found. Best child score: {best_child_score:.4f}")
                self.current_node_id = self._backtrack()
        
        logger.info("Optimization completed!")
        self._save_results()
        
        return self.best_prompt, self.best_score
    
    def _generate_candidates(self, current_prompt: str) -> List[str]:
        """Generate candidate prompts based on current prompt."""
        generation_prompt = f"""
        I have a prompt that needs to be improved. Generate {self.generation_size} better variations of this prompt.
        
        Current prompt: {current_prompt}
        
        Generate {self.generation_size} improved variations that:
        1. Are more specific and clear
        2. Use different wording or structure
        3. Add helpful context or constraints
        4. Maintain the original intent
        
        Provide only the prompts, one per line, numbered 1-{self.generation_size}:
        """
        
        try:
            response = self.generator.run(generation_prompt.strip())
            candidates = self._parse_candidates(response)
            
            logger.info(f"Generated {len(candidates)} candidate prompts")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to generate candidates: {e}")
            return []
    
    def _parse_candidates(self, response: str) -> List[str]:
        """Parse candidate prompts from generator response."""
        candidates = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.)
            if line[0].isdigit() and '.' in line:
                line = line.split('.', 1)[1].strip()
            
            # Remove leading dashes or bullets
            if line.startswith(('-', '*', '•')):
                line = line[1:].strip()
            
            if line and len(line) > 10:  # Basic validation
                candidates.append(line)
        
        return candidates[:self.generation_size]
    
    def _evaluate_prompt(self, node_id: str) -> float:
        """Evaluate a prompt and update the tree."""
        node = self.prompt_tree[node_id]
        
        if node.evaluated:
            return node.score
        
        logger.info(f"Evaluating prompt: {node.prompt[:50]}...")
        
        try:
            # Get dataset inputs and references
            inputs, references = self.dataset.get_all_data()
            
            # Generate outputs using the prompt
            generated_outputs = []
            for input_text in tqdm(inputs, desc="Generating outputs"):
                full_prompt = f"{node.prompt}\n\nInput: {input_text}\nOutput:"
                output = self.generator.run(full_prompt)
                generated_outputs.append(output)
            
            # Evaluate outputs
            score = self.evaluator.run(inputs, references, generated_outputs)
            
            # Update node
            node.score = score
            node.evaluated = True
            
            # Update best prompt if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_prompt = node.prompt
                logger.info(f"New best prompt found with score {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            node.score = -float('inf')
            node.evaluated = True
            return -float('inf')
    
    def _backtrack(self) -> Optional[str]:
        """Backtrack to find the next node to explore."""
        if self.current_node_id is None:
            return None
        
        current_node = self.prompt_tree[self.current_node_id]
        
        # Go up the tree to find a node with unexplored children
        while current_node.parent_id is not None:
            parent = self.prompt_tree[current_node.parent_id]
            
            # Find next unexplored child
            for child_id in parent.children_ids:
                child = self.prompt_tree[child_id]
                if not child.evaluated or len(child.children_ids) == 0:
                    # Found potential node to explore
                    if child.evaluated and child.depth < self.max_depth:
                        return child_id
            
            current_node = parent
        
        # No more nodes to explore
        return None
    
    def get_best_prompt(self) -> Tuple[str, float]:
        """Get the best prompt and score found so far."""
        return self.best_prompt, self.best_score
    
    def _save_results(self) -> None:
        """Save optimization results to files."""
        # Save prompt tree
        tree_data = {
            'metadata': {
                'generation_size': self.generation_size,
                'max_depth': self.max_depth,
                'max_iterations': self.max_iterations,
                'iterations_completed': self.iterations,
                'best_score': self.best_score,
                'best_prompt': self.best_prompt,
                'root_prompt': self.root_prompt
            },
            'tree': {}
        }
        
        # Convert nodes to serializable format
        for node_id, node in self.prompt_tree.items():
            tree_data['tree'][node_id] = {
                'prompt': node.prompt,
                'score': node.score,
                'parent_id': node.parent_id,
                'children_ids': node.children_ids,
                'depth': node.depth,
                'generation': node.generation,
                'evaluated': node.evaluated
            }
        
        # Save to file
        tree_file = self.output_dir / "prompt_tree.json"
        with open(tree_file, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved prompt tree to {tree_file}")
        
        # Save best prompt separately
        best_file = self.output_dir / "best_prompt.txt"
        with open(best_file, 'w', encoding='utf-8') as f:
            f.write(f"Best Score: {self.best_score:.4f}\n")
            f.write(f"Best Prompt:\n{self.best_prompt}")
        
        logger.info(f"Saved best prompt to {best_file}")
    
    def load_tree(self, tree_file: str) -> None:
        """Load a previously saved prompt tree."""
        with open(tree_file, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        # Load metadata
        metadata = tree_data['metadata']
        self.best_score = metadata['best_score']
        self.best_prompt = metadata['best_prompt']
        self.root_prompt = metadata['root_prompt']
        self.iterations = metadata['iterations_completed']
        
        # Load tree nodes
        self.prompt_tree = {}
        for node_id, node_data in tree_data['tree'].items():
            node = PromptNode(
                prompt=node_data['prompt'],
                score=node_data['score'],
                parent_id=node_data['parent_id'],
                children_ids=node_data['children_ids'],
                depth=node_data['depth'],
                generation=node_data['generation'],
                evaluated=node_data['evaluated']
            )
            self.prompt_tree[node_id] = node
        
        logger.info(f"Loaded prompt tree with {len(self.prompt_tree)} nodes")
    
    def print_tree_summary(self) -> None:
        """Print a summary of the prompt tree."""
        print(f"\n{'='*50}")
        print("PROMPT TREE SUMMARY")
        print(f"{'='*50}")
        print(f"Total nodes: {len(self.prompt_tree)}")
        print(f"Iterations completed: {self.iterations}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best prompt: {self.best_prompt[:100]}...")
        
        # Print tree structure
        print(f"\nTree Structure:")
        for node_id, node in self.prompt_tree.items():
            indent = "  " * node.depth
            score_str = f"({node.score:.4f})" if node.score is not None else "(not evaluated)"
            print(f"{indent}- {node.prompt[:50]}... {score_str}")
        
        print(f"{'='*50}\n") 