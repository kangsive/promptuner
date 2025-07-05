"""
PromptTuner: A prompt optimization tool using DFS search on prompt family trees.

This package provides a framework for optimizing prompts through iterative
generation and evaluation using a tree-based search approach.
"""

__version__ = "0.1.0"

from .optimizer import PromptOptimizer
from .base import Generator, Evaluator, Dataset
from .generators import OpenAIGenerator, TransformerGenerator
from .evaluators import AdherenceEvaluator, BLEUEvaluator
from .datasets import CSVDataset, JSONDataset, JSONLDataset, InMemoryDataset

__all__ = [
    "PromptOptimizer",
    "Generator",
    "Evaluator", 
    "Dataset",
    "OpenAIGenerator",
    "TransformerGenerator",
    "AdherenceEvaluator",
    "BLEUEvaluator",
    "CSVDataset",
    "JSONDataset",
    "JSONLDataset",
    "InMemoryDataset",
] 