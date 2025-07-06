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

from .base import Generator, Evaluator, Dataset, Analyst

logger = logging.getLogger(__name__)


@dataclass
class PromptNode:
    """Node in the prompt family tree."""
    prompt: str
    score: Optional[float] = None
    individual_scores: Optional[List[float]] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    depth: int = 0
    generation: int = 0
    evaluated: bool = False
    analysis_result: Optional[Dict[str, Any]] = None
    new_hypothesis: Optional[str] = None
    verified_hypothesis: List[str] = None
    false_hypothesis: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.individual_scores is None:
            self.individual_scores = []
        if self.verified_hypothesis is None:
            self.verified_hypothesis = []
        if self.false_hypothesis is None:
            self.false_hypothesis = []
    
    @property
    def node_id(self) -> str:
        """Generate unique node ID based on prompt hash."""
        return f"node_{hash(self.prompt) % 1000000:06d}"
    
    def get_ancestor_hypotheses(self, prompt_tree: Dict[str, 'PromptNode']) -> Tuple[List[str], List[str]]:
        """
        Get all verified and false hypotheses from ancestor nodes.
        
        Args:
            prompt_tree: Dictionary mapping node IDs to PromptNode objects.
            
        Returns:
            Tuple of (verified_hypotheses, false_hypotheses) from all ancestors.
        """
        verified_hypotheses = []
        false_hypotheses = []
        
        # Traverse up the tree to collect ancestor hypotheses
        current_node = self
        while current_node.parent_id is not None:
            parent_node = prompt_tree.get(current_node.parent_id)
            if parent_node is None:
                break
                
            verified_hypotheses.extend(parent_node.verified_hypothesis)
            false_hypotheses.extend(parent_node.false_hypothesis)
            current_node = parent_node
        
        return verified_hypotheses, false_hypotheses


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
                 analyst: Analyst,
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
            analyst: Analyst for providing feedback on evaluation results.
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
        self.analyst = analyst
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
        self.baseline_score: float = -float('inf')  # Score of initial prompt
        self.failed_prompts: List[Dict[str, Any]] = []  # Track failed prompts
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
        self.baseline_score = self.prompt_tree[self.current_node_id].score
        
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
            
            # Generate new prompt candidates using feedback
            candidates = self._generate_candidates(current_node)
            
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
                
                # Track failed prompts (worse than baseline)
                if self.baseline_score is not None and score < self.baseline_score:
                    self.failed_prompts.append({
                        'prompt': child_node.prompt,
                        'score': score,
                        'baseline_score': self.baseline_score,
                        'analysis': child_node.analysis_result
                    })
            
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
    
    def _generate_candidates(self, current_node: PromptNode) -> List[str]:
        """Generate candidate prompts based on current prompt and feedback."""
        base_prompt = current_node.prompt
        
        # Get ancestor hypotheses
        ancestor_verified, ancestor_false = current_node.get_ancestor_hypotheses(self.prompt_tree)
        
        # Get feedback from analyst
        feedback_text = ""
        if current_node.analysis_result:
            analysis = current_node.analysis_result
            feedback_text = f"""
            
EVALUATION FEEDBACK:
- Summary: {analysis.get('summary', 'No summary available')}
- Key Issues: {'; '.join(analysis.get('patterns', []))}
- Suggestions: {'; '.join(analysis.get('improvement_suggestions', []))}
- Hypothesis: {analysis.get('hypothesis', 'No hypothesis available')}
- New Hypothesis to Test: {analysis.get('new_hypothesis', 'No new hypothesis')}
            """
        
        # Get ancestor hypotheses context
        hypotheses_text = ""
        if ancestor_verified or ancestor_false:
            hypotheses_text = f"""
            
ANCESTOR HYPOTHESES CONTEXT:
"""
            if ancestor_verified:
                hypotheses_text += f"""
VERIFIED STRATEGIES (Build on these):
{chr(10).join([f"- {hyp}" for hyp in ancestor_verified])}
"""
            
            if ancestor_false:
                hypotheses_text += f"""
FAILED STRATEGIES (Avoid these):
{chr(10).join([f"- {hyp}" for hyp in ancestor_false])}
"""
        
        # Get failed prompts context
        failed_prompts_text = ""
        if self.failed_prompts:
            recent_failures = self.failed_prompts[-3:]  # Last 3 failures
            failed_prompts_text = f"""
            
FAILED PROMPTS TO AVOID:
{chr(10).join([f"- {fp['prompt'][:100]}... (Score: {fp['score']:.3f})" for fp in recent_failures])}
            """
        
        generation_prompt = f"""
I need to improve a prompt for better performance. Here's the current situation:

CURRENT PROMPT: {base_prompt}

TASK: {self.task_description or 'Improve the given prompt'}
{feedback_text}
{hypotheses_text}
{failed_prompts_text}

Generate {self.generation_size} improved variations of the current prompt that:
1. Address the specific issues identified in the feedback
2. Build on verified strategies from ancestor hypotheses
3. Avoid failed strategies from ancestor hypotheses
4. Test the new hypothesis if provided
5. Are more specific and clear than the current prompt
6. Avoid patterns from failed prompts
7. Use different wording or structure to explore new approaches
8. Maintain the original intent while improving performance

Focus on actionable improvements based on the feedback and hypothesis context provided.

Provide only the prompts, one per line, numbered 1-{self.generation_size}:
"""
        
        try:
            response = self.generator.run(generation_prompt.strip())
            candidates = self._parse_candidates(response)
            
            logger.info(f"Generated {len(candidates)} candidate prompts with feedback and hypotheses")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to generate candidates: {e}")
            # Since analyst is required, we cannot generate candidates without feedback
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
            
            # Evaluate outputs with detailed scores
            if hasattr(self.evaluator, 'run_detailed'):
                overall_score, individual_scores = self.evaluator.run_detailed(
                    inputs, references, generated_outputs
                )
                node.individual_scores = individual_scores
            else:
                # Fallback to basic evaluation
                overall_score = self.evaluator.run(inputs, references, generated_outputs)
                node.individual_scores = [overall_score] * len(inputs)
            
            # Update node
            node.score = overall_score
            node.evaluated = True
            
            # Run analyst
            try:
                # Get ancestor hypotheses
                ancestor_verified, ancestor_false = node.get_ancestor_hypotheses(self.prompt_tree)
                
                analysis_result = self.analyst.run(
                    inputs, references, generated_outputs, 
                    node.individual_scores, overall_score, node.prompt,
                    ancestor_verified, ancestor_false
                )
                node.analysis_result = analysis_result
                
                # Store the new hypothesis generated by the analyst
                node.new_hypothesis = analysis_result.get('new_hypothesis')
                
                logger.info(f"Analysis completed: {analysis_result.get('summary', 'No summary')}")
                if node.new_hypothesis:
                    logger.info(f"New hypothesis: {node.new_hypothesis}")
            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                node.analysis_result = None
            
            # Update best prompt if this is better
            if overall_score > self.best_score:
                self.best_score = overall_score
                self.best_prompt = node.prompt
                logger.info(f"New best prompt found with score {overall_score:.4f}")
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            node.score = -float('inf')
            node.evaluated = True
            return -float('inf')
    
    def _update_parent_hypotheses(self, parent_node_id: str) -> None:
        """Update parent node's hypotheses based on children's performance."""
        parent_node = self.prompt_tree.get(parent_node_id)
        if not parent_node or not parent_node.children_ids:
            return
        
        # Get all evaluated children
        evaluated_children = []
        for child_id in parent_node.children_ids:
            child_node = self.prompt_tree.get(child_id)
            if child_node and child_node.evaluated:
                evaluated_children.append(child_node)
        
        if not evaluated_children:
            return
        
        # Update parent hypotheses using the analyst
        try:
            self.analyst.update_parent_hypotheses(parent_node, evaluated_children)
            logger.info(f"Updated hypotheses for parent {parent_node_id}: "
                       f"Verified: {len(parent_node.verified_hypothesis)}, "
                       f"False: {len(parent_node.false_hypothesis)}")
        except Exception as e:
            logger.error(f"Error updating parent hypotheses: {e}")
    
    def _backtrack(self) -> Optional[str]:
        """Backtrack to find the next node to explore."""
        if self.current_node_id is None:
            return None
        
        current_node = self.prompt_tree[self.current_node_id]
        
        # Update parent hypotheses before backtracking
        if current_node.parent_id:
            self._update_parent_hypotheses(current_node.parent_id)
        
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
                'baseline_score': self.baseline_score,
                'root_prompt': self.root_prompt,
                'failed_prompts_count': len(self.failed_prompts)
            },
            'tree': {},
            'failed_prompts': self.failed_prompts
        }
        
        # Convert nodes to serializable format
        for node_id, node in self.prompt_tree.items():
            tree_data['tree'][node_id] = {
                'prompt': node.prompt,
                'score': node.score,
                'individual_scores': node.individual_scores,
                'parent_id': node.parent_id,
                'children_ids': node.children_ids,
                'depth': node.depth,
                'generation': node.generation,
                'evaluated': node.evaluated,
                'analysis_result': node.analysis_result,
                'new_hypothesis': node.new_hypothesis,
                'verified_hypothesis': node.verified_hypothesis,
                'false_hypothesis': node.false_hypothesis
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
            f.write(f"Baseline Score: {self.baseline_score:.4f}\n")
            f.write(f"Improvement: {self.best_score - self.baseline_score:.4f}\n")
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
        self.baseline_score = metadata.get('baseline_score', -float('inf'))
        self.root_prompt = metadata['root_prompt']
        self.iterations = metadata['iterations_completed']
        
        # Load failed prompts
        self.failed_prompts = tree_data.get('failed_prompts', [])
        
        # Load tree nodes
        self.prompt_tree = {}
        for node_id, node_data in tree_data['tree'].items():
            node = PromptNode(
                prompt=node_data['prompt'],
                score=node_data['score'],
                individual_scores=node_data.get('individual_scores', []),
                parent_id=node_data['parent_id'],
                children_ids=node_data['children_ids'],
                depth=node_data['depth'],
                generation=node_data['generation'],
                evaluated=node_data['evaluated'],
                analysis_result=node_data.get('analysis_result'),
                new_hypothesis=node_data.get('new_hypothesis'),
                verified_hypothesis=node_data.get('verified_hypothesis', []),
                false_hypothesis=node_data.get('false_hypothesis', [])
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
        print(f"Baseline score: {self.baseline_score:.4f}")
        print(f"Improvement: {self.best_score - self.baseline_score:.4f}")
        print(f"Failed prompts: {len(self.failed_prompts)}")
        print(f"Best prompt: {self.best_prompt[:100]}...")
        
        # Print tree structure
        print(f"\nTree Structure:")
        for node_id, node in self.prompt_tree.items():
            indent = "  " * node.depth
            score_str = f"({node.score:.4f})" if node.score is not None else "(not evaluated)"
            analysis_str = "✓" if node.analysis_result else ""
            hypothesis_str = ""
            if node.new_hypothesis:
                hypothesis_str += " [H]"
            if node.verified_hypothesis:
                hypothesis_str += f" [V:{len(node.verified_hypothesis)}]"
            if node.false_hypothesis:
                hypothesis_str += f" [F:{len(node.false_hypothesis)}]"
            print(f"{indent}- {node.prompt[:50]}... {score_str} {analysis_str}{hypothesis_str}")
        
        print(f"{'='*50}\n") 