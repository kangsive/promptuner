# PromptTuner

A powerful prompt optimization tool that uses depth-first search on prompt family trees to find the best prompts for your tasks.

## Features

- **DFS-based Optimization**: Uses depth-first search to explore prompt variations systematically
- **Prompt Family Trees**: Maintains a tree structure of prompt variations with evaluation scores
- **Multiple Generators**: Support for OpenAI API, compatible APIs (like vLLM), and local transformers
- **Flexible Evaluators**: Built-in adherence and BLEU-like evaluators with extensible architecture
- **Multiple Dataset Formats**: Support for CSV, JSON, and JSONL datasets
- **Comprehensive Logging**: Detailed logging of optimization process and results
- **Result Persistence**: Save and load optimization results for later analysis

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd promptuner
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
from promptuner import PromptOptimizer, OpenAIGenerator, AdherenceEvaluator, CSVDataset

# Initialize components
generator = OpenAIGenerator(api_key="your-openai-key")
evaluator = AdherenceEvaluator()
dataset = CSVDataset("your_dataset.csv")

# Create optimizer
optimizer = PromptOptimizer(
    generator=generator,
    evaluator=evaluator,
    dataset=dataset,
    initial_prompt="Your initial prompt here",
    generation_size=3,
    max_depth=5,
    max_iterations=50
)

# Run optimization
best_prompt, best_score = optimizer.run()
print(f"Best prompt: {best_prompt}")
print(f"Best score: {best_score}")
```

### Using Task Description

```python
# Instead of providing an initial prompt, you can use a task description
optimizer = PromptOptimizer(
    generator=generator,
    evaluator=evaluator,
    dataset=dataset,
    task_description="Summarize news articles in 2-3 sentences",
    generation_size=3,
    max_depth=5,
    max_iterations=50
)

best_prompt, best_score = optimizer.run()
```

### Using Local Models

```python
from promptuner import TransformerGenerator

# Use local transformer model
generator = TransformerGenerator(model_name="gpt2", max_tokens=100)

optimizer = PromptOptimizer(
    generator=generator,
    evaluator=evaluator,
    dataset=dataset,
    initial_prompt="Your initial prompt here"
)

best_prompt, best_score = optimizer.run()
```

## Dataset Format

### CSV Format
```csv
input,reference
"What is the capital of France?","Paris"
"What is 2+2?","4"
```

### JSON Format
```json
[
  {"input": "What is the capital of France?", "reference": "Paris"},
  {"input": "What is 2+2?", "reference": "4"}
]
```

### JSONL Format
```jsonl
{"input": "What is the capital of France?", "reference": "Paris"}
{"input": "What is 2+2?", "reference": "4"}
```

## Components

### Generators

#### OpenAIGenerator
- Supports OpenAI API and compatible APIs (like vLLM)
- Async batch processing for efficiency
- Configurable model parameters

#### TransformerGenerator
- Uses HuggingFace transformers pipeline
- Supports local models
- Automatic device detection (CPU/GPU)

### Evaluators

#### AdherenceEvaluator
- Measures how well generated text adheres to reference text
- Combines multiple metrics: exact match, similarity, semantic similarity, length penalty
- Configurable weights for different metrics

#### BLEUEvaluator
- BLEU-like evaluation metric
- Configurable n-gram size
- Includes brevity penalty

### Datasets

#### CSVDataset
- Load data from CSV files
- Configurable column names
- Automatic data validation

#### JSONDataset
- Support for various JSON structures
- Handles nested data structures
- Configurable key names

#### JSONLDataset
- JSON Lines format support
- Handles malformed lines gracefully
- Memory efficient for large datasets

#### InMemoryDataset
- For programmatically created datasets
- Useful for testing and small datasets

## Configuration

### Environment Variables

```bash
# OpenAI API configuration
export OPENAI_API_KEY="your-api-key"

# For using custom endpoints (like vLLM)
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

### Optimization Parameters

- `generation_size`: Number of prompt variants to generate per round (default: 3)
- `max_depth`: Maximum depth of the prompt family tree (default: 5)
- `max_iterations`: Maximum number of optimization iterations (default: 100)
- `output_dir`: Directory to save results (default: "outputs")

## Advanced Usage

### Custom Evaluators

```python
from promptuner.base import Evaluator

class CustomEvaluator(Evaluator):
    def run(self, inputs, references, generated):
        # Implement your custom evaluation logic
        scores = []
        for ref, gen in zip(references, generated):
            score = your_custom_scoring_function(ref, gen)
            scores.append(score)
        return sum(scores) / len(scores)

# Use custom evaluator
evaluator = CustomEvaluator()
optimizer = PromptOptimizer(generator=generator, evaluator=evaluator, dataset=dataset)
```

### Saving and Loading Results

```python
# Optimization results are automatically saved
optimizer.run()

# Load previous results
optimizer.load_tree("outputs/prompt_tree.json")
best_prompt, best_score = optimizer.get_best_prompt()
```

### Analyzing Results

```python
# Print detailed tree summary
optimizer.print_tree_summary()

# Access the prompt tree directly
for node_id, node in optimizer.prompt_tree.items():
    print(f"Node {node_id}: {node.prompt[:50]}... (Score: {node.score})")
```

## Output Files

The optimizer saves results in the specified output directory:

- `prompt_tree.json`: Complete prompt family tree with metadata
- `best_prompt.txt`: Best prompt and score in human-readable format

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=promptuner

# Run specific test file
pytest tests/test_optimizer.py
```

## Examples

Check the `examples/` directory for complete usage examples:

- `basic_optimization.py`: Basic optimization workflow
- `custom_evaluator.py`: Using custom evaluators
- `batch_optimization.py`: Optimizing multiple prompts

## Performance Tips

1. **Use batch processing**: OpenAI generator supports async batch processing
2. **Adjust generation size**: Larger generation sizes explore more variations but take longer
3. **Set appropriate max_depth**: Deeper trees find better prompts but increase computation time
4. **Use local models for experimentation**: Faster iteration during development
5. **Monitor evaluation time**: Complex evaluators can become the bottleneck

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use PromptTuner in your research, please cite:

```bibtex
@software{promptuner,
  title={PromptTuner: DFS-based Prompt Optimization},
  author={PromptTuner Team},
  year={2024},
  url={https://github.com/your-org/promptuner}
}
``` 