#!/usr/bin/env python3
"""
Basic example demonstrating PromptTuner usage.

This example shows how to:
1. Create a simple dataset
2. Set up the optimizer with a mock generator
3. Run optimization
4. Analyze results
"""

import os
import logging
from promptuner import PromptOptimizer, AdherenceEvaluator
from promptuner.datasets import InMemoryDataset
from promptuner.generators import TransformerGenerator
from promptuner.analysts import RuleBasedAnalyst

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    inputs = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
    ]
    
    references = [
        "Paris",
        "4",
        "Blue",
        "William Shakespeare",
        "Jupiter",
    ]
    
    return InMemoryDataset(inputs, references, batch_size=3)


def main():
    """Main function to demonstrate PromptTuner."""
    print("🚀 PromptTuner Basic Example")
    print("=" * 50)
    
    # Create sample dataset
    print("📊 Creating sample dataset...")
    dataset = create_sample_dataset()
    print(f"   Dataset size: {dataset.datasize}")
    
    # Create evaluator
    print("🔍 Setting up evaluator...")
    evaluator = AdherenceEvaluator()
    
    # Create analyst
    print("🔬 Setting up analyst...")
    analyst = RuleBasedAnalyst(low_score_threshold=0.4)
    
    # Create generator (using a mock for this example)
    print("🤖 Setting up generator...")
    
    # For this example, we'll use a mock generator that creates simple variations
    class MockGenerator:
        def run(self, prompt, **kwargs):
            # This is a simplified mock - in real usage, use OpenAIGenerator or TransformerGenerator
            if "Create an effective prompt" in prompt:
                return "Answer the following question concisely and accurately:"
            else:
                # Generate simple variations
                variations = [
                    f"Please {prompt.lower()}",
                    f"Can you {prompt.lower()}",
                    f"I need you to {prompt.lower()}",
                ]
                return "\n".join(f"{i+1}. {var}" for i, var in enumerate(variations))
    
    generator = MockGenerator()
    
    # Create optimizer
    print("⚙️ Creating optimizer...")
    optimizer = PromptOptimizer(
        generator=generator,
        evaluator=evaluator,
        dataset=dataset,
        analyst=analyst,
        task_description="Answer questions accurately and concisely",
        generation_size=3,
        max_depth=3,
        max_iterations=10,
        output_dir="outputs"
    )
    
    print(f"   Initial prompt: {optimizer.root_prompt}")
    print(f"   Generation size: {optimizer.generation_size}")
    print(f"   Max depth: {optimizer.max_depth}")
    print(f"   Max iterations: {optimizer.max_iterations}")
    
    # Run optimization
    print("\n🔄 Running optimization...")
    print("   This may take a few minutes...")
    
    try:
        best_prompt, best_score = optimizer.run()
        
        print("\n✅ Optimization completed!")
        print("=" * 50)
        print(f"🏆 Best Score: {best_score:.4f}")
        print(f"📝 Best Prompt: {best_prompt}")
        
        # Print tree summary
        print("\n🌳 Prompt Tree Summary:")
        optimizer.print_tree_summary()
        
        # Show output files
        print("\n📁 Output Files:")
        output_dir = "outputs"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   {file} ({size} bytes)")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        logger.error(f"Optimization failed: {e}")
        return
    
    print("\n🎉 Example completed successfully!")
    print("   Check the 'outputs' directory for detailed results.")


if __name__ == "__main__":
    main() 