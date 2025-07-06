"""
Base classes for the prompt optimization framework.

This module defines the abstract base classes that all generators, evaluators,
and datasets must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Tuple, Optional
import pandas as pd


class Generator(ABC):
    """Abstract base class for text generators."""
    
    @abstractmethod
    def run(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: The input prompt to generate text from.
            **kwargs: Additional parameters for generation.
            
        Returns:
            Generated text string.
        """
        pass


class Evaluator(ABC):
    """Abstract base class for evaluators."""
    
    @abstractmethod
    def run(self, inputs: List[str], references: List[str], 
            generated: List[str]) -> float:
        """
        Evaluate the quality of generated outputs.
        
        Args:
            inputs: List of input prompts.
            references: List of reference/expected outputs.
            generated: List of generated outputs to evaluate.
            
        Returns:
            Evaluation score (higher is better).
        """
        pass
    
    def run_detailed(self, inputs: List[str], references: List[str], 
                    generated: List[str]) -> Tuple[float, List[float]]:
        """
        Evaluate the quality of generated outputs with detailed scores.
        
        Args:
            inputs: List of input prompts.
            references: List of reference/expected outputs.
            generated: List of generated outputs to evaluate.
            
        Returns:
            Tuple of (overall_score, individual_scores).
        """
        overall_score = self.run(inputs, references, generated)
        # Default implementation: return overall score for each sample
        individual_scores = [overall_score] * len(inputs)
        return overall_score, individual_scores


class Analyst(ABC):
    """Abstract base class for analyzing evaluation results."""
    
    @abstractmethod
    def run(self, inputs: List[str], references: List[str], 
            generated: List[str], scores: List[float], 
            overall_score: float, current_prompt: str,
            ancestor_verified_hypotheses: Optional[List[str]] = None,
            ancestor_false_hypotheses: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze evaluation results to provide feedback for prompt improvement.
        
        Args:
            inputs: List of input prompts.
            references: List of reference/expected outputs.
            generated: List of generated outputs.
            scores: List of individual sample scores.
            overall_score: Overall evaluation score.
            current_prompt: The prompt that generated these outputs.
            ancestor_verified_hypotheses: List of hypotheses verified by ancestors.
            ancestor_false_hypotheses: List of hypotheses falsified by ancestors.
            
        Returns:
            Analysis results containing:
            - 'summary': Overall summary of the evaluation
            - 'low_score_analysis': Analysis of low-scoring samples
            - 'patterns': Patterns identified in problematic data
            - 'improvement_suggestions': Specific suggestions for improvement
            - 'hypothesis': Hypothesis about why performance is low
            - 'new_hypothesis': New hypothesis for children to verify
        """
        pass
    
    def update_parent_hypotheses(self, parent_node: 'PromptNode', child_nodes: List['PromptNode'],
                                low_score_threshold: float = 0.3) -> None:
        """
        Update parent node's verified and false hypotheses based on children's performance.
        
        Args:
            parent_node: Parent node whose hypotheses should be updated.
            child_nodes: List of child nodes that tested the parent's hypothesis.
            low_score_threshold: Threshold for identifying low-scoring samples.
        """
        if not parent_node.new_hypothesis or not child_nodes:
            return
        
        # Check if children improved on parent's key issues (low-scoring samples)
        parent_low_score_count = sum(1 for score in parent_node.individual_scores 
                                   if score < low_score_threshold)
        
        # Find the best performing child
        best_child = max(child_nodes, key=lambda x: x.score if x.score is not None else -float('inf'))
        
        if best_child.score is None or parent_node.score is None:
            return
        
        # Count low-scoring samples for the best child
        best_child_low_score_count = sum(1 for score in best_child.individual_scores 
                                       if score < low_score_threshold)
        
        # Determine if hypothesis should be verified or falsified
        if (best_child.score > parent_node.score and 
            best_child_low_score_count < parent_low_score_count):
            # Hypothesis is verified - children improved on key issues
            if parent_node.new_hypothesis not in parent_node.verified_hypothesis:
                parent_node.verified_hypothesis.append(parent_node.new_hypothesis)
        else:
            # Hypothesis is false - children did not improve on key issues
            if parent_node.new_hypothesis not in parent_node.false_hypothesis:
                parent_node.false_hypothesis.append(parent_node.new_hypothesis)


class Dataset(ABC):
    """Abstract base class for datasets."""
    
    def __init__(self, data_path: str, batch_size: int = 32, 
                 shuffle: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._data: Optional[pd.DataFrame] = None
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load data from the specified path."""
        pass
    
    @property
    def datasize(self) -> int:
        """Get the total number of samples in the dataset."""
        if self._data is None:
            return 0
        return len(self._data)
    
    @property
    def batchsize(self) -> int:
        """Get the batch size."""
        return self.batch_size
    
    def batch_iter(self) -> Iterator[Tuple[List[str], List[str]]]:
        """
        Iterate over batches of (inputs, references).
        
        Yields:
            Tuple of (inputs, references) for each batch.
        """
        if self._data is None:
            return
        
        data = self._data.copy()
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i + self.batch_size]
            inputs = batch['input'].tolist()
            references = batch['reference'].tolist()
            yield inputs, references
    
    def get_all_data(self) -> Tuple[List[str], List[str]]:
        """
        Get all data as lists.
        
        Returns:
            Tuple of (all_inputs, all_references).
        """
        if self._data is None:
            return [], []
        return self._data['input'].tolist(), self._data['reference'].tolist() 