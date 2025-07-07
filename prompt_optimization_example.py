#!/usr/bin/env python3
"""
Example usage of the prompt optimizer with analyst feedback.
"""

import os
import logging
from promptuner import (
    PromptOptimizer,
    OpenAIGenerator,
    AdherenceEvaluator,
    LLMAnalyst,
    RuleBasedAnalyst,
    InMemoryDataset
)

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    """Main example function."""
    print("🚀 Enhanced Prompt Optimizer with Analyst Feedback Example")
    print("=" * 60)
    
    # Sample dataset for text summarization task
    sample_inputs = [
        "Write a brief summary of the following text: The quick brown fox jumps over the lazy dog. This is a common phrase used to test typing skills.",
        "Summarize: Climate change is a long-term shift in global temperatures and weather patterns. While natural factors contribute, human activities have been the main driver since the 1800s.",
        "Create a summary: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Summarize this text: The Internet has revolutionized how we communicate, work, and access information. It has connected billions of people worldwide.",
        "Write a summary: Renewable energy sources like solar and wind power are becoming increasingly important as we seek to reduce our carbon footprint."
    ]
    
    sample_references = [
        "A common typing test phrase about a fox jumping over a dog.",
        "Climate change refers to long-term shifts in global temperatures, mainly driven by human activities since the 1800s.",
        "Machine learning is an AI subset that allows computers to learn from experience without explicit programming.",
        "The Internet has transformed communication, work, and information access, connecting billions globally.",
        "Renewable energy sources like solar and wind are crucial for reducing carbon footprint."
    ]
    
    # Create dataset
    dataset = InMemoryDataset(sample_inputs, sample_references)
    
    # Initialize generator
    generator = OpenAIGenerator(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1500,
    )
    
    # Initialize evaluator
    evaluator = AdherenceEvaluator(
        exact_match_weight=0.1,
        similarity_weight=0.5,
        semantic_weight=0.3,
        length_weight=0.1
    )
    
    # Initialize analyst - try LLM analyst first, fallback to rule-based
    try:
        analyst = LLMAnalyst(
            generator=generator,
            low_score_threshold=0.4
        )
        print("✅ Using LLM Analyst for feedback generation")
    except Exception as e:
        print(f"⚠️  LLM Analyst failed, using Rule-based Analyst: {e}")
        analyst = RuleBasedAnalyst(low_score_threshold=0.4)
    
    # Initialize optimizer with analyst
    optimizer = PromptOptimizer(
        generator=generator,
        evaluator=evaluator,
        dataset=dataset,
        analyst=analyst,
        task_description="Generate concise and accurate summaries of given text",
        generation_size=3,
        max_depth=3,
        max_iterations=10,
        output_dir="outputs_with_analyst"
    )
    
    print("\n🔍 Starting optimization with analyst feedback...")
    print("The optimizer will now:")
    print("1. Evaluate prompts and get individual sample scores")
    print("2. Analyze low-scoring samples to identify patterns")
    print("3. Generate feedback for improving prompts")
    print("4. Use feedback and failed prompt history for better candidates")
    print("5. Track and avoid patterns from failed prompts")
    
    # Run optimization
    best_prompt, best_score = optimizer.run()
    
    print(f"\n🎉 Optimization completed!")
    print(f"📊 Best Score: {best_score:.4f}")
    print(f"📈 Baseline Score: {optimizer.baseline_score:.4f}")
    print(f"🚀 Improvement: {best_score - optimizer.baseline_score:.4f}")
    print(f"❌ Failed Prompts: {len(optimizer.failed_prompts)}")
    
    print(f"\n🏆 Best Prompt:")
    print("-" * 40)
    print(best_prompt)
    print("-" * 40)
    
    # Print detailed tree summary
    optimizer.print_tree_summary()
    
    # Show some analysis results
    print("\n🔬 Analysis Results Examples:")
    print("=" * 40)
    
    analyzed_nodes = [node for node in optimizer.prompt_tree.values() 
                     if node.analysis_result]
    
    for i, node in enumerate(analyzed_nodes[:2]):  # Show first 2 analyses
        print(f"\nNode {i+1} Analysis:")
        print(f"Prompt: {node.prompt[:60]}...")
        print(f"Score: {node.score:.4f}")
        
        if node.analysis_result:
            analysis = node.analysis_result
            print(f"Summary: {analysis.get('summary', 'N/A')}")
            print(f"Patterns: {analysis.get('patterns', [])}")
            print(f"Suggestions: {analysis.get('improvement_suggestions', [])}")
    
    # Show failed prompts insights
    if optimizer.failed_prompts:
        print(f"\n❌ Failed Prompts Analysis:")
        print("=" * 40)
        
        for i, failed in enumerate(optimizer.failed_prompts[:3]):  # Show first 3
            print(f"\nFailed Prompt {i+1}:")
            print(f"Prompt: {failed['prompt'][:60]}...")
            print(f"Score: {failed['score']:.4f} (Baseline: {failed['baseline_score']:.4f})")
            
            if failed.get('analysis'):
                print(f"Analysis: {failed['analysis'].get('summary', 'N/A')}")
    
    print(f"\n📁 Results saved to: {optimizer.output_dir}")
    print("   - prompt_tree.json: Complete optimization tree with analysis")
    print("   - best_prompt.txt: Best prompt and improvement metrics")
    
    print("\n🔧 Key Features Demonstrated:")
    print("✓ Analyst feedback integration")
    print("✓ Individual sample score tracking")
    print("✓ Failed prompt pattern avoidance")
    print("✓ Comprehensive evaluation analysis")
    print("✓ Feedback-driven prompt generation")


if __name__ == "__main__":
    # Check for API key if using OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using mock responses.")
        print("   Set your OpenAI API key to use real LLM generation.")
        print("   export OPENAI_API_KEY='your-api-key-here'")
    
    main() 