"""
Generator implementations for the prompt optimization framework.

This module provides concrete implementations of generators using OpenAI API
and HuggingFace transformers.
"""

import asyncio
from typing import List, Optional, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import os

from openai import OpenAI, AsyncOpenAI
from transformers import pipeline, Pipeline
import torch

from .base import Generator

logger = logging.getLogger(__name__)


class OpenAIGenerator(Generator):
    """Generator using OpenAI API or compatible APIs."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.7,
                 batch_size: int = 5):
        """
        Initialize OpenAI generator.
        
        Args:
            model: Model name to use.
            api_key: OpenAI API key (defaults to env var OPENAI_API_KEY).
            base_url: Base URL for OpenAI compatible APIs.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            batch_size: Batch size for async generation.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        
        # Initialize clients
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        elif os.getenv('OPENAI_API_KEY'):
            client_kwargs['api_key'] = os.getenv('OPENAI_API_KEY')
        
        if base_url:
            client_kwargs['base_url'] = base_url
        
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
    
    def run(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                n=1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise
    
    async def run_async(self, prompt: str, **kwargs) -> str:
        """
        Generate text asynchronously.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text.
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                n=1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI async: {e}")
            raise
    
    def run_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts using async batch processing.
        
        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated texts.
        """
        async def _batch_generate():
            tasks = []
            for prompt in prompts:
                task = self.run_async(prompt, **kwargs)
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        return asyncio.run(_batch_generate())


class TransformerGenerator(Generator):
    """Generator using HuggingFace transformers."""
    
    def __init__(self, model_name: str = "gpt2",
                 device: Optional[str] = None,
                 max_tokens: int = 150,
                 temperature: float = 0.7,
                 do_sample: bool = True):
        """
        Initialize transformer generator.
        
        Args:
            model_name: HuggingFace model name.
            device: Device to run on (auto-detected if None).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
        # Auto-detect device if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize pipeline
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                return_full_text=False,
                pad_token_id=50256  # Common pad token for GPT models
            )
        except Exception as e:
            logger.error(f"Error initializing transformer pipeline: {e}")
            raise
    
    def run(self, prompt: str, **kwargs) -> str:
        """
        Generate text using transformers.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text.
        """
        try:
            # Generate text
            outputs = self.pipeline(
                prompt,
                max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                do_sample=kwargs.get('do_sample', self.do_sample),
                num_return_sequences=1,
                eos_token_id=self.pipeline.tokenizer.eos_token_id,
                pad_token_id=self.pipeline.tokenizer.pad_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating text with transformers: {e}")
            raise
    
    def run_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated texts.
        """
        try:
            # Use thread pool for batch processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.run, prompt, **kwargs) 
                          for prompt in prompts]
                results = [future.result() for future in futures]
            return results
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            raise 