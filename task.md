I want to build a prompt optimizer where it takes:

- A generator as base class (should be a class with run method)
  Possible inheritant classes:
  - OpenAIGenerator
    It use OpenAI API or Any OpenAI compatible API (like vLLM deployed llms) to generate text, to optimize generation efficiency, consider using batch generation or async generation.
  - TransformerGenerator
    It use python library transformer pipeline for generation.

- A task description where we can use generator to generate initial prompt if initial prompt is not provided.

- An initial prompt:
  Baseline prompt to be optimized. Optimization will based on evaluation of generation quality, an evaluator is provided to evaluate on a dataset provided containing inputs and reference outputs.

- An evaluator class (base class) with run method returning a score calculated on dataset inputs, reference outputs and generated outputs.
  - Possible inheritant classes:
    - AdherenceEvaluator
  
- A dataset (base class) containing inputs and reference outputs (Optional).
  - Base class: should have datasize, batchsize, shuffle and batch_iter method. Support csv, json, jsonl.
  
- An analyst class (base class) to analyze the prompt and provide feedback.
  - takes evaluation results, analyze on evaluation result to undetstand why the current prompt failed or get low evaluation score, by focusing on low score samples (input, output, reference output, scores), then return analyze result including summary and analyze result of evaluation including hypothsis why we get this score, what kind of samples are lowering the score, the pattern of this kind of data and how to improve the prompt, (A hypothesis of Why it is, what cause this, how to address this). This analysis offer reference and context for the prompt optimizer to generate better prompt.
  
  - **Hypothesis Tracking**: The analysis is guided by a hypothesis system. The analyst considers `verified_hypothesis` and `false_hypothesis` from ancestor prompts to generate more accurate feedback and a `new_hypothesis`.
  - The `new_hypothesis` is a specific, testable idea for how to improve the prompt (e.g., "Adding examples will improve performance on complex inputs").
  - The analyst is also responsible for updating the parent prompt's hypotheses (`verified_hypothesis` or `false_hypothesis`) based on the performance of its children, effectively allowing the system to learn what works and what doesn't.
  - The output of the analysis (summary, patterns, suggestions, new hypothesis) provides context for the optimizer to generate better prompts.
  
And then output the best prompt with best score after optimization.

The prompt optimizer takes all above components to initialize and run. Should have a run method to run the optimization.
  - It should have prompt generation control, recording a prompt family tree with initial prompt as root. The tree nodes should store the prompt, score, analysis results, and hypothesis data (`new_hypothesis`, `verified_hypothesis`, `false_hypothesis`).
  - **Hypothesis-Driven Search**: The prompt generation process is guided by the hypothesis system. When generating new prompt candidates, it considers the `new_hypothesis` from the current node, as well as the `verified_hypothesis` and `false_hypothesis` from all ancestor nodes. This ensures that new prompts build on successful strategies and avoid failed ones.
  - It uses a DFS search. It picks the first candidate, evaluates its score, and if the score is better than its parent, it continues the search down that path. Otherwise, it backtracks.
  - When backtracking, the optimizer updates the parent node's hypotheses based on the performance of the just-evaluated children, solidifying the learned knowledge.

  - Should have method to get the best prompt (with best score) in the tree.
  - Should have generation size control, to control the number of prompts to generate in each round of optimization. If no candidate in a generation have a better score than father prompt, we should trackback to father prompt and continue in the next candidate in previous generation.
  - Should have max level of family tree control, to control the depth of the tree.
  - Should have max iteration control, to control the number of rounds of optimization.
  - When ever the max level of family tree is reached, we should stop the optimization, or when ever the max iteration is reached, we should stop the optimization, or all possible candidates in the tree are tried, we should stop the optimization.
  - When stop the optimization, it should return the best prompt with best score in the tree.
  - The family tree with prompt and score should be saved to a file in a proper format.

All implementation should be simple but effective, meaning avoiding unnecessary complexity of the code, easy to understand, but make sure it is functional well. Elegant and powerful.

Should be implemented in python.