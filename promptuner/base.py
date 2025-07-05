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