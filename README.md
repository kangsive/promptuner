# PromptTuner

**A modular, simple, and general-purpose prompt optimization framework.**

PromptTuner provides an easy-to-use and easy-to-understand toolkit for automatically finding the best prompts for your tasks. It uses an intelligent, tree-based search strategy, enhanced by a **hypothesis tracking system**, to systematically learn from evaluation feedback and guide the optimization process.

## Features

- **Intelligent Optimization**: Uses a DFS-based search on prompt family trees, guided by a hypothesis tracking system to learn from successes and failures.
- **Modular Design**: Easily extend and customize with your own generators, evaluators, and datasets.
- **Multiple Generator Support**: Works out-of-the-box with OpenAI API, compatible endpoints (like vLLM), and local HuggingFace transformers.
- **Flexible Evaluation**: Built-in evaluators for common tasks and a simple interface for creating your own.
- **Automated Analysis**: `Analyst` classes (LLM or rule-based) automatically find patterns in failures and generate hypotheses for improvement.
- **Broad Dataset Compatibility**: Supports CSV, JSON, and JSONL files, as well as in-memory datasets.
- **Comprehensive Results**: Saves the full optimization tree, including all prompts, scores, and hypotheses, for detailed analysis.

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

PromptTuner requires an `Analyst` to enable the hypothesis tracking system. This provides intelligent and effective optimization by learning from evaluation feedback.

```python
from promptuner import (
    PromptOptimizer, 
    OpenAIGenerator, 
    AdherenceEvaluator, 
    LLMAnalyst, 
    InMemoryDataset
)
import logging

# Set up logging to see the optimization process
logging.basicConfig(level=logging.INFO)

# 1. Initialize Components
generator = OpenAIGenerator() # Assumes OPENAI_API_KEY is set
evaluator = AdherenceEvaluator()
analyst = LLMAnalyst(generator, low_score_threshold=0.4)

# 2. Create a Dataset
inputs = ["What is the capital of France?", "What is 2 + 2?"]
references = ["Paris", "4"]
dataset = InMemoryDataset(inputs, references)

# 3. Create and Run the Optimizer
optimizer = PromptOptimizer(
    generator=generator,
    evaluator=evaluator,
    dataset=dataset,
    analyst=analyst,
    task_description="Answer the following questions accurately.",
    initial_prompt=None, # Base prompt to start with, if None, optimizer will generate a base prompt according to the task description
    generation_size=3, # Number of prompt variants to generate per round
    max_depth=5, # Maximum depth of the prompt family tree
    max_iterations=5, # Maximum number of iterations to run
    output_dir="outputs"
)

# Run optimization
best_prompt, best_score = optimizer.run()

print(f"\nBest prompt found: {best_prompt}")
print(f"Best score: {best_score:.4f}")

# View optimization summary and hypothesis insights
optimizer.print_tree_summary()
```

## Hypothesis-Driven Optimization

PromptTuner's key innovation is its ability to learn from the optimization process.

1.  **Analysis**: After evaluating a prompt, the `Analyst` examines samples with low scores to identify failure patterns.
2.  **Hypothesis Generation**: Based on this analysis and the knowledge from ancestor prompts, the `Analyst` forms a `new_hypothesis`—a specific, testable idea for improvement (e.g., "Adding examples to the prompt will improve performance on complex inputs").
3.  **Hypothesis Testing**: The `Generator` creates new prompt variations designed to test this hypothesis, while also leveraging previously verified strategies and avoiding failed ones.
4.  **Hypothesis Validation**: After the new prompts are evaluated, the system determines if the hypothesis was successful. If the child prompts show improvement on the key issues, the hypothesis is marked as `verified`; otherwise, it's marked as `false`.

This cycle of generating, testing, and validating hypotheses allows the optimizer to build a cumulative understanding of what makes a good prompt for your specific task, leading to more intelligent and efficient optimization.

## Dataset Format

Your dataset should contain `input` and `reference` pairs.

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
- Supports OpenAI API and compatible APIs (like vLLM).
- Async batch processing for efficiency.
- Configurable model parameters.

#### TransformerGenerator
- Uses HuggingFace transformers pipeline.
- Supports local models.
- Automatic device detection (CPU/GPU).

### Evaluators

#### AdherenceEvaluator
- Measures how well generated text adheres to reference text.
- Combines multiple metrics: exact match, similarity, semantic similarity, length penalty.

#### BLEUEvaluator
- BLEU-like evaluation metric with a brevity penalty.

### Datasets

#### CSVDataset, JSONDataset, JSONLDataset
- Load data from respective file formats with configurable key/column names.

#### InMemoryDataset
- For programmatically created datasets, useful for testing.

### Analysts

#### LLMAnalyst
- Uses an LLM to analyze evaluation results, provide sophisticated feedback, and generate hypotheses.

#### RuleBasedAnalyst
- Uses predefined rules for deterministic analysis and hypothesis generation. A lightweight alternative to the `LLMAnalyst`.

## Configuration

### Environment Variables

```bash
# OpenAI API configuration
export OPENAI_API_KEY="your-api-key"

# For using custom endpoints (like vLLM)
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

### Optimization Parameters

- `initial_prompt`: Base prompt to start with, if None, optimizer will generate a base prompt according to the task description (default: None).
- `generation_size`: Number of prompt variants to generate per round (default: 3).
- `max_depth`: Maximum depth of the prompt family tree (default: 5).
- `max_iterations`: Maximum number of optimization iterations (default: 100).
- `output_dir`: Directory to save results (default: "outputs").

## Advanced Usage

### Using Local Models

```python
from promptuner import TransformerGenerator

# Use local transformer model
generator = TransformerGenerator(model_name="gpt2", max_tokens=100)

optimizer = PromptOptimizer(
    generator=generator,
    evaluator=evaluator,
    dataset=dataset,
    analyst=analyst,
    initial_prompt="Your initial prompt here"
)
```

### Custom Components

PromptTuner's modular design makes it easy to create your own components.

```python
from promptuner.base import Evaluator

class CustomEvaluator(Evaluator):
    def run(self, inputs, references, generated):
        # Implement your custom evaluation logic
        score = 0.0 # Your logic here
        return score

evaluator = CustomEvaluator()
optimizer = PromptOptimizer(
    generator=generator, 
    evaluator=evaluator, 
    dataset=dataset, 
    analyst=analyst
)
```

### Analyzing Results

The optimizer automatically saves results to the `output_dir`.

```python
# Load previous results to continue analysis
optimizer.load_tree("outputs/prompt_tree.json")
best_prompt, best_score = optimizer.get_best_prompt()

# Print detailed tree summary with hypothesis info
# [H] = New Hypothesis, [V:1] = 1 Verified, [F:1] = 1 False
optimizer.print_tree_summary()

# Access the prompt tree directly for custom analysis
for node_id, node in optimizer.prompt_tree.items():
    print(f"Node {node_id}: {node.prompt[:50]}... (Score: {node.score})")
    if node.verified_hypothesis:
        print(f"  Verified Hypotheses: {node.verified_hypothesis}")
```

## Output Files

- `prompt_tree.json`: The complete prompt family tree, including all prompts, scores, analyses, and hypothesis data.
- `best_prompt.txt`: The best-performing prompt and its score in a human-readable format.

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