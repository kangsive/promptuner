"""
Dataset implementations for the prompt optimization framework.

This module provides concrete implementations of datasets for loading
data from CSV, JSON, and JSONL formats.
"""

import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

from .base import Dataset

logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    """Dataset implementation for CSV files."""
    
    def __init__(self, data_path: str, 
                 input_column: str = "input",
                 reference_column: str = "reference",
                 batch_size: int = 32,
                 shuffle: bool = False):
        """
        Initialize CSV dataset.
        
        Args:
            data_path: Path to CSV file.
            input_column: Name of the input column.
            reference_column: Name of the reference column.
            batch_size: Batch size for iteration.
            shuffle: Whether to shuffle the data.
        """
        self.input_column = input_column
        self.reference_column = reference_column
        super().__init__(data_path, batch_size, shuffle)
    
    def _load_data(self) -> None:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            
            # Validate required columns
            if self.input_column not in df.columns:
                raise ValueError(f"Input column '{self.input_column}' "
                               f"not found in CSV")
            if self.reference_column not in df.columns:
                raise ValueError(f"Reference column '{self.reference_column}' "
                               f"not found in CSV")
            
            # Create standardized dataframe
            self._data = pd.DataFrame({
                'input': df[self.input_column].astype(str),
                'reference': df[self.reference_column].astype(str)
            })
            
            # Remove rows with NaN values (including "nan" strings from astype conversion)
            self._data = self._data.dropna()
            self._data = self._data[~self._data['input'].str.contains('nan', case=False, na=False)]
            self._data = self._data[~self._data['reference'].str.contains('nan', case=False, na=False)]
            
            logger.info(f"Loaded {len(self._data)} samples from CSV: {self.data_path}")
            
        except Exception as e:
            logger.error(f"Error loading CSV dataset: {e}")
            raise


class JSONDataset(Dataset):
    """Dataset implementation for JSON files."""
    
    def __init__(self, data_path: str,
                 input_key: str = "input",
                 reference_key: str = "reference",
                 batch_size: int = 32,
                 shuffle: bool = False):
        """
        Initialize JSON dataset.
        
        Args:
            data_path: Path to JSON file.
            input_key: Key for input data in JSON.
            reference_key: Key for reference data in JSON.
            batch_size: Batch size for iteration.
            shuffle: Whether to shuffle the data.
        """
        self.input_key = input_key
        self.reference_key = reference_key
        super().__init__(data_path, batch_size, shuffle)
    
    def _load_data(self) -> None:
        """Load data from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                records = data
            elif isinstance(data, dict):
                # Could be a single object or object with data array
                if self.input_key in data and self.reference_key in data:
                    # Single record
                    records = [data]
                elif 'data' in data:
                    # Nested structure
                    records = data['data']
                else:
                    raise ValueError("Unable to parse JSON structure")
            else:
                raise ValueError(f"Unsupported JSON structure: {type(data)}")
            
            # Convert to DataFrame
            inputs = []
            references = []
            
            for record in records:
                if not isinstance(record, dict):
                    continue
                    
                if self.input_key in record and self.reference_key in record:
                    inputs.append(str(record[self.input_key]))
                    references.append(str(record[self.reference_key]))
                else:
                    logger.warning(f"Skipping record missing required keys: {record}")
            
            if not inputs:
                raise ValueError("No valid records found in JSON")
            
            self._data = pd.DataFrame({
                'input': inputs,
                'reference': references
            })
            
            logger.info(f"Loaded {len(self._data)} samples from JSON: {self.data_path}")
            
        except Exception as e:
            logger.error(f"Error loading JSON dataset: {e}")
            raise


class JSONLDataset(Dataset):
    """Dataset implementation for JSONL (JSON Lines) files."""
    
    def __init__(self, data_path: str,
                 input_key: str = "input",
                 reference_key: str = "reference",
                 batch_size: int = 32,
                 shuffle: bool = False):
        """
        Initialize JSONL dataset.
        
        Args:
            data_path: Path to JSONL file.
            input_key: Key for input data in each JSON line.
            reference_key: Key for reference data in each JSON line.
            batch_size: Batch size for iteration.
            shuffle: Whether to shuffle the data.
        """
        self.input_key = input_key
        self.reference_key = reference_key
        super().__init__(data_path, batch_size, shuffle)
    
    def _load_data(self) -> None:
        """Load data from JSONL file."""
        try:
            inputs = []
            references = []
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        if not isinstance(record, dict):
                            logger.warning(f"Skipping non-object line {line_num}")
                            continue
                        
                        if self.input_key in record and self.reference_key in record:
                            inputs.append(str(record[self.input_key]))
                            references.append(str(record[self.reference_key]))
                        else:
                            logger.warning(f"Skipping line {line_num} missing required keys")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue
            
            if not inputs:
                raise ValueError("No valid records found in JSONL")
            
            self._data = pd.DataFrame({
                'input': inputs,
                'reference': references
            })
            
            logger.info(f"Loaded {len(self._data)} samples from JSONL: {self.data_path}")
            
        except Exception as e:
            logger.error(f"Error loading JSONL dataset: {e}")
            raise


class InMemoryDataset(Dataset):
    """Dataset implementation for in-memory data."""
    
    def __init__(self, inputs: list, references: list,
                 batch_size: int = 32,
                 shuffle: bool = False):
        """
        Initialize in-memory dataset.
        
        Args:
            inputs: List of input strings.
            references: List of reference strings.
            batch_size: Batch size for iteration.
            shuffle: Whether to shuffle the data.
        """
        self.inputs = inputs
        self.references = references
        
        if len(inputs) != len(references):
            raise ValueError(f"Number of inputs ({len(inputs)}) "
                           f"must match number of references ({len(references)})")
        
        # Use a dummy path for in-memory data
        super().__init__("in_memory", batch_size, shuffle)
    
    def _load_data(self) -> None:
        """Load data from memory."""
        try:
            self._data = pd.DataFrame({
                'input': [str(x) for x in self.inputs],
                'reference': [str(x) for x in self.references]
            })
            
            logger.info(f"Loaded {len(self._data)} samples from memory")
            
        except Exception as e:
            logger.error(f"Error loading in-memory dataset: {e}")
            raise 