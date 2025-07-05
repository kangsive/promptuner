"""Tests for generators module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from promptuner.generators import OpenAIGenerator, TransformerGenerator


class TestOpenAIGenerator:
    """Test OpenAIGenerator class."""
    
    @patch('promptuner.generators.OpenAI')
    @patch('promptuner.generators.AsyncOpenAI')
    def test_init(self, mock_async_openai, mock_openai):
        """Test OpenAIGenerator initialization."""
        generator = OpenAIGenerator(
            model="gpt-3.5-turbo",
            api_key="test_key",
            max_tokens=100,
            temperature=0.5
        )
        
        assert generator.model == "gpt-3.5-turbo"
        assert generator.max_tokens == 100
        assert generator.temperature == 0.5
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()
    
    @patch('promptuner.generators.OpenAI')
    @patch('promptuner.generators.AsyncOpenAI')
    def test_run(self, mock_async_openai, mock_openai):
        """Test OpenAIGenerator run method."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        generator = OpenAIGenerator(api_key="test_key")
        result = generator.run("test prompt")
        
        assert result == "Generated response"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-3.5-turbo"
        assert call_args[1]['messages'] == [{"role": "user", "content": "test prompt"}]
    
    @patch('promptuner.generators.OpenAI')
    @patch('promptuner.generators.AsyncOpenAI')
    def test_run_with_kwargs(self, mock_async_openai, mock_openai):
        """Test OpenAIGenerator run method with kwargs."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        generator = OpenAIGenerator(api_key="test_key")
        result = generator.run("test prompt", max_tokens=200, temperature=0.9)
        
        assert result == "Generated response"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['max_tokens'] == 200
        assert call_args[1]['temperature'] == 0.9
    
    @patch('promptuner.generators.OpenAI')
    @patch('promptuner.generators.AsyncOpenAI')
    def test_run_error_handling(self, mock_async_openai, mock_openai):
        """Test OpenAIGenerator error handling."""
        # Mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        generator = OpenAIGenerator(api_key="test_key")
        
        with pytest.raises(Exception, match="API Error"):
            generator.run("test prompt")
    
    @patch('promptuner.generators.OpenAI')
    @patch('promptuner.generators.AsyncOpenAI')
    def test_run_batch(self, mock_async_openai, mock_openai):
        """Test OpenAIGenerator batch processing."""
        # Mock async client
        mock_async_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        
        async def mock_create(*args, **kwargs):
            return mock_response
        
        mock_async_client.chat.completions.create = mock_create
        mock_async_openai.return_value = mock_async_client
        
        generator = OpenAIGenerator(api_key="test_key")
        
        # Test batch generation
        prompts = ["prompt 1", "prompt 2"]
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = ["Generated response", "Generated response"]
            results = generator.run_batch(prompts)
            
            assert len(results) == 2
            assert all(r == "Generated response" for r in results)
            mock_run.assert_called_once()


class TestTransformerGenerator:
    """Test TransformerGenerator class."""
    
    @patch('promptuner.generators.pipeline')
    @patch('promptuner.generators.torch')
    def test_init(self, mock_torch, mock_pipeline):
        """Test TransformerGenerator initialization."""
        mock_torch.cuda.is_available.return_value = False
        
        generator = TransformerGenerator(
            model_name="gpt2",
            max_tokens=100,
            temperature=0.5
        )
        
        assert generator.model_name == "gpt2"
        assert generator.max_tokens == 100
        assert generator.temperature == 0.5
        assert generator.device == "cpu"
        mock_pipeline.assert_called_once()
    
    @patch('promptuner.generators.pipeline')
    @patch('promptuner.generators.torch')
    def test_init_with_cuda(self, mock_torch, mock_pipeline):
        """Test TransformerGenerator initialization with CUDA."""
        mock_torch.cuda.is_available.return_value = True
        
        generator = TransformerGenerator(model_name="gpt2")
        
        assert generator.device == "cuda"
        mock_pipeline.assert_called_once()
    
    @patch('promptuner.generators.pipeline')
    @patch('promptuner.generators.torch')
    def test_run(self, mock_torch, mock_pipeline):
        """Test TransformerGenerator run method."""
        # Mock pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Generated response"}]
        mock_pipe.tokenizer.eos_token_id = 50256
        mock_pipe.tokenizer.pad_token_id = 50256
        mock_pipeline.return_value = mock_pipe
        
        generator = TransformerGenerator(model_name="gpt2")
        result = generator.run("test prompt")
        
        assert result == "Generated response"
        mock_pipe.assert_called_once()
    
    @patch('promptuner.generators.pipeline')
    @patch('promptuner.generators.torch')
    def test_run_with_kwargs(self, mock_torch, mock_pipeline):
        """Test TransformerGenerator run method with kwargs."""
        # Mock pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Generated response"}]
        mock_pipe.tokenizer.eos_token_id = 50256
        mock_pipe.tokenizer.pad_token_id = 50256
        mock_pipeline.return_value = mock_pipe
        
        generator = TransformerGenerator(model_name="gpt2")
        result = generator.run("test prompt", max_tokens=200, temperature=0.9)
        
        assert result == "Generated response"
        # Check that kwargs were passed to pipeline
        call_args = mock_pipe.call_args
        assert call_args[1]['max_new_tokens'] == 200
        assert call_args[1]['temperature'] == 0.9
    
    @patch('promptuner.generators.pipeline')
    @patch('promptuner.generators.torch')
    def test_run_error_handling(self, mock_torch, mock_pipeline):
        """Test TransformerGenerator error handling."""
        # Mock pipeline to raise an exception
        mock_pipe = Mock()
        mock_pipe.side_effect = Exception("Pipeline Error")
        mock_pipeline.return_value = mock_pipe
        
        generator = TransformerGenerator(model_name="gpt2")
        
        with pytest.raises(Exception, match="Pipeline Error"):
            generator.run("test prompt")
    
    @patch('promptuner.generators.pipeline')
    @patch('promptuner.generators.torch')
    def test_run_batch(self, mock_torch, mock_pipeline):
        """Test TransformerGenerator batch processing."""
        # Mock pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Generated response"}]
        mock_pipe.tokenizer.eos_token_id = 50256
        mock_pipe.tokenizer.pad_token_id = 50256
        mock_pipeline.return_value = mock_pipe
        
        generator = TransformerGenerator(model_name="gpt2")
        
        prompts = ["prompt 1", "prompt 2"]
        
        # Mock the run method directly
        with patch.object(generator, 'run', return_value="Generated response") as mock_run:
            results = generator.run_batch(prompts)
            
            assert len(results) == 2
            # Results should be generated responses
            assert all(isinstance(r, str) for r in results)
            assert all(r == "Generated response" for r in results)
            # Should have called run for each prompt
            assert mock_run.call_count == 2 