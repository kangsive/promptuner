"""Tests for base classes."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from promptuner.base import Generator, Evaluator, Dataset


class MockGenerator(Generator):
    """Mock generator for testing."""
    
    def run(self, prompt: str, **kwargs) -> str:
        return f"Generated: {prompt}"


class MockEvaluator(Evaluator):
    """Mock evaluator for testing."""
    
    def run(self, inputs, references, generated) -> float:
        return 0.8


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, data_path: str, batch_size: int = 32, shuffle: bool = False):
        self.test_data = pd.DataFrame({
            'input': ['test input 1', 'test input 2'],
            'reference': ['test ref 1', 'test ref 2']
        })
        super().__init__(data_path, batch_size, shuffle)
    
    def _load_data(self):
        self._data = self.test_data


class TestGenerator:
    """Test Generator base class."""
    
    def test_generator_interface(self):
        """Test that Generator is an abstract base class."""
        with pytest.raises(TypeError):
            Generator()
    
    def test_mock_generator(self):
        """Test mock generator implementation."""
        gen = MockGenerator()
        result = gen.run("test prompt")
        assert result == "Generated: test prompt"


class TestEvaluator:
    """Test Evaluator base class."""
    
    def test_evaluator_interface(self):
        """Test that Evaluator is an abstract base class."""
        with pytest.raises(TypeError):
            Evaluator()
    
    def test_mock_evaluator(self):
        """Test mock evaluator implementation."""
        eval = MockEvaluator()
        inputs = ["input1", "input2"]
        references = ["ref1", "ref2"]
        generated = ["gen1", "gen2"]
        
        score = eval.run(inputs, references, generated)
        assert score == 0.8


class TestDataset:
    """Test Dataset base class."""
    
    def test_dataset_interface(self):
        """Test that Dataset is an abstract base class."""
        with pytest.raises(TypeError):
            Dataset("test.csv")
    
    def test_mock_dataset(self):
        """Test mock dataset implementation."""
        dataset = MockDataset("test.csv", batch_size=1)
        
        assert dataset.datasize == 2
        assert dataset.batchsize == 1
        
        # Test batch iteration
        batches = list(dataset.batch_iter())
        assert len(batches) == 2
        assert batches[0] == (['test input 1'], ['test ref 1'])
        assert batches[1] == (['test input 2'], ['test ref 2'])
        
        # Test get_all_data
        inputs, references = dataset.get_all_data()
        assert inputs == ['test input 1', 'test input 2']
        assert references == ['test ref 1', 'test ref 2']
    
    def test_dataset_shuffle(self):
        """Test dataset shuffling."""
        dataset = MockDataset("test.csv", batch_size=2, shuffle=True)
        
        # Should still have same size
        assert dataset.datasize == 2
        
        # Get all data (order may vary due to shuffling)
        inputs, references = dataset.get_all_data()
        assert len(inputs) == 2
        assert len(references) == 2 