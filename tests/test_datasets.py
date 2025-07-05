"""Tests for datasets module."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from promptuner.datasets import CSVDataset, JSONDataset, JSONLDataset, InMemoryDataset


class TestCSVDataset:
    """Test CSVDataset class."""
    
    def test_load_csv_dataset(self):
        """Test loading CSV dataset."""
        # Create temporary CSV file
        csv_content = """input,reference
"What is the capital of France?","Paris"
"What is 2+2?","4"
"What color is the sky?","Blue"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        
        try:
            dataset = CSVDataset(csv_path, batch_size=2)
            
            assert dataset.datasize == 3
            assert dataset.batchsize == 2
            
            # Test get_all_data
            inputs, references = dataset.get_all_data()
            assert len(inputs) == 3
            assert len(references) == 3
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
            # Test batch iteration
            batches = list(dataset.batch_iter())
            assert len(batches) == 2  # 2 batches for 3 items with batch_size=2
            
            first_batch_inputs, first_batch_refs = batches[0]
            assert len(first_batch_inputs) == 2
            assert len(first_batch_refs) == 2
            
            second_batch_inputs, second_batch_refs = batches[1]
            assert len(second_batch_inputs) == 1
            assert len(second_batch_refs) == 1
            
        finally:
            os.unlink(csv_path)
    
    def test_load_csv_custom_columns(self):
        """Test loading CSV with custom column names."""
        csv_content = """question,answer
"What is the capital of France?","Paris"
"What is 2+2?","4"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        
        try:
            dataset = CSVDataset(
                csv_path, 
                input_column="question",
                reference_column="answer"
            )
            
            assert dataset.datasize == 2
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(csv_path)
    
    def test_csv_missing_columns(self):
        """Test CSV with missing required columns."""
        csv_content = """question,answer
"What is the capital of France?","Paris"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        
        try:
            # Should raise error for missing 'input' column
            with pytest.raises(ValueError, match="Input column 'input' not found"):
                CSVDataset(csv_path)
                
        finally:
            os.unlink(csv_path)
    
    def test_csv_with_nan_values(self):
        """Test CSV with NaN values."""
        csv_content = """input,reference
"What is the capital of France?","Paris"
,"4"
"What color is the sky?",
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        
        try:
            dataset = CSVDataset(csv_path)
            
            # Should filter out rows with NaN values
            assert dataset.datasize == 1
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(csv_path)


class TestJSONDataset:
    """Test JSONDataset class."""
    
    def test_load_json_array(self):
        """Test loading JSON array format."""
        json_data = [
            {"input": "What is the capital of France?", "reference": "Paris"},
            {"input": "What is 2+2?", "reference": "4"},
            {"input": "What color is the sky?", "reference": "Blue"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name
        
        try:
            dataset = JSONDataset(json_path, batch_size=2)
            
            assert dataset.datasize == 3
            assert dataset.batchsize == 2
            
            inputs, references = dataset.get_all_data()
            assert len(inputs) == 3
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(json_path)
    
    def test_load_json_single_object(self):
        """Test loading single JSON object."""
        json_data = {"input": "What is the capital of France?", "reference": "Paris"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name
        
        try:
            dataset = JSONDataset(json_path)
            
            assert dataset.datasize == 1
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(json_path)
    
    def test_load_json_nested_structure(self):
        """Test loading JSON with nested data structure."""
        json_data = {
            "data": [
                {"input": "What is the capital of France?", "reference": "Paris"},
                {"input": "What is 2+2?", "reference": "4"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name
        
        try:
            dataset = JSONDataset(json_path)
            
            assert dataset.datasize == 2
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(json_path)
    
    def test_load_json_custom_keys(self):
        """Test loading JSON with custom keys."""
        json_data = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2+2?", "answer": "4"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name
        
        try:
            dataset = JSONDataset(
                json_path,
                input_key="question",
                reference_key="answer"
            )
            
            assert dataset.datasize == 2
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(json_path)
    
    def test_json_missing_keys(self):
        """Test JSON with missing required keys."""
        json_data = [
            {"input": "What is the capital of France?", "reference": "Paris"},
            {"input": "What is 2+2?"},  # Missing reference
            {"reference": "Blue"}  # Missing input
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name
        
        try:
            dataset = JSONDataset(json_path)
            
            # Should only load the complete record
            assert dataset.datasize == 1
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(json_path)
    
    def test_json_empty_data(self):
        """Test JSON with no valid records."""
        json_data = [
            {"wrong_key": "value"},
            {"another_key": "value"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name
        
        try:
            with pytest.raises(ValueError, match="No valid records found"):
                JSONDataset(json_path)
                
        finally:
            os.unlink(json_path)


class TestJSONLDataset:
    """Test JSONLDataset class."""
    
    def test_load_jsonl_dataset(self):
        """Test loading JSONL dataset."""
        jsonl_content = '''{"input": "What is the capital of France?", "reference": "Paris"}
{"input": "What is 2+2?", "reference": "4"}
{"input": "What color is the sky?", "reference": "Blue"}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            jsonl_path = f.name
        
        try:
            dataset = JSONLDataset(jsonl_path, batch_size=2)
            
            assert dataset.datasize == 3
            assert dataset.batchsize == 2
            
            inputs, references = dataset.get_all_data()
            assert len(inputs) == 3
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(jsonl_path)
    
    def test_load_jsonl_custom_keys(self):
        """Test loading JSONL with custom keys."""
        jsonl_content = '''{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "What is 2+2?", "answer": "4"}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            jsonl_path = f.name
        
        try:
            dataset = JSONLDataset(
                jsonl_path,
                input_key="question",
                reference_key="answer"
            )
            
            assert dataset.datasize == 2
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(jsonl_path)
    
    def test_jsonl_with_empty_lines(self):
        """Test JSONL with empty lines."""
        jsonl_content = '''{"input": "What is the capital of France?", "reference": "Paris"}

{"input": "What is 2+2?", "reference": "4"}

'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            jsonl_path = f.name
        
        try:
            dataset = JSONLDataset(jsonl_path)
            
            assert dataset.datasize == 2
            inputs, references = dataset.get_all_data()
            assert len(inputs) == 2
            
        finally:
            os.unlink(jsonl_path)
    
    def test_jsonl_with_invalid_json(self):
        """Test JSONL with invalid JSON lines."""
        jsonl_content = '''{"input": "What is the capital of France?", "reference": "Paris"}
invalid json line
{"input": "What is 2+2?", "reference": "4"}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            jsonl_path = f.name
        
        try:
            dataset = JSONLDataset(jsonl_path)
            
            # Should skip invalid lines
            assert dataset.datasize == 2
            inputs, references = dataset.get_all_data()
            assert len(inputs) == 2
            
        finally:
            os.unlink(jsonl_path)
    
    def test_jsonl_missing_keys(self):
        """Test JSONL with missing required keys."""
        jsonl_content = '''{"input": "What is the capital of France?", "reference": "Paris"}
{"input": "What is 2+2?"}
{"reference": "Blue"}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            jsonl_path = f.name
        
        try:
            dataset = JSONLDataset(jsonl_path)
            
            # Should only load the complete record
            assert dataset.datasize == 1
            inputs, references = dataset.get_all_data()
            assert inputs[0] == "What is the capital of France?"
            assert references[0] == "Paris"
            
        finally:
            os.unlink(jsonl_path)


class TestInMemoryDataset:
    """Test InMemoryDataset class."""
    
    def test_in_memory_dataset(self):
        """Test InMemoryDataset functionality."""
        inputs = ["What is the capital of France?", "What is 2+2?", "What color is the sky?"]
        references = ["Paris", "4", "Blue"]
        
        dataset = InMemoryDataset(inputs, references, batch_size=2)
        
        assert dataset.datasize == 3
        assert dataset.batchsize == 2
        
        # Test get_all_data
        all_inputs, all_references = dataset.get_all_data()
        assert all_inputs == inputs
        assert all_references == references
        
        # Test batch iteration
        batches = list(dataset.batch_iter())
        assert len(batches) == 2  # 2 batches for 3 items with batch_size=2
    
    def test_in_memory_dataset_mismatched_lengths(self):
        """Test InMemoryDataset with mismatched input lengths."""
        inputs = ["input1", "input2"]
        references = ["ref1", "ref2", "ref3"]  # Extra reference
        
        with pytest.raises(ValueError, match="Number of inputs"):
            InMemoryDataset(inputs, references)
    
    def test_in_memory_dataset_empty(self):
        """Test InMemoryDataset with empty data."""
        inputs = []
        references = []
        
        dataset = InMemoryDataset(inputs, references)
        
        assert dataset.datasize == 0
        
        # Test get_all_data
        all_inputs, all_references = dataset.get_all_data()
        assert all_inputs == []
        assert all_references == []
        
        # Test batch iteration
        batches = list(dataset.batch_iter())
        assert len(batches) == 0
    
    def test_in_memory_dataset_shuffle(self):
        """Test InMemoryDataset with shuffle."""
        inputs = ["input1", "input2", "input3"]
        references = ["ref1", "ref2", "ref3"]
        
        dataset = InMemoryDataset(inputs, references, shuffle=True)
        
        assert dataset.datasize == 3
        
        # Data should still be complete after shuffling
        all_inputs, all_references = dataset.get_all_data()
        assert len(all_inputs) == 3
        assert len(all_references) == 3
        
        # All original inputs and references should be present
        assert set(all_inputs) == set(inputs)
        assert set(all_references) == set(references) 