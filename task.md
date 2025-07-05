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
  
And then output the best prompt with best score after optimization.

The prompt optimizer takes all above components to initialize and run. Should have a run method to run the optimization.
  - It should have prompt generation control, recording a prompt family tree with initial prompt as root. Every round of optimization should generate a few new prompts as new generation. (Expanding the tree), then we do a DFS search, pick the first candidate, generate outputs (on dataset inputs), evaluate to get the score, if the score is better than the current best score, we update the best score with current candidate and continue the search deep down on this node. Otherwise, we backtrack and try the next candidate.
  That means, only the first candidate in new generation which have a better score than father prompt will be selected to generate next generation of prompts and expend the prompts family tree on this track.

  - Should have method to get the best prompt (with best score) in the tree.
  - Should have generation size control, to control the number of prompts to generate in each round of optimization. If no candidate in a generation have a better score than father prompt, we should trackback to father prompt and continue in the next candidate in previous generation.
  - Should have max level of family tree control, to control the depth of the tree.
  - Should have max iteration control, to control the number of rounds of optimization.
  - When ever the max level of family tree is reached, we should stop the optimization, or when ever the max iteration is reached, we should stop the optimization, or all possible candidates in the tree are tried, we should stop the optimization.
  - When stop the optimization, it should return the best prompt with best score in the tree.
  - The family tree with prompt and score should be saved to a file in a proper format.

All implementation should be simple but effective, meaning avoiding unnecessary complexity of the code, easy to understand, but make sure it is functional well. Elegant and powerful.

Should be implemented in python.