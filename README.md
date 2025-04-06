# slim-model

A framework to transform a high-capacity language model into a hardware-optimized, inference-efficient version by pruning.
Pruning will be done using blockwise. Currently I am working on task specific pruning of attention blocks.

## Steps

1. Load the model and run it with a task specific data
2. Capture scores and metrics relevant for pruning attention block (KL divergence, Attention block entropy)
3. Based on a composite score, rank the blocks and prune the ones that have the highest scores
4. Save the pruned model
5. Finetune the pruned model

## High-Level Implementation Roadmap

- Data & Model Loading
  - Load a pretrained LLM and a representative validation dataset.
  - Instrument the model to capture activations and attention maps.
- Metrics Collection
  - Implement hooks to record metrics: activation variance, attention entropy, KL divergence.
  - Store metrics per head/layer.
- Ranking & Pruning
  - Rank heads/layers based on collected metrics.
  - Define pruning thresholds and functions to remove or bypass low-impact components.
- Evaluation & Iteration
  - Evaluate the pruned model on benchmarks (accuracy, speed, memory).
  - Iterate on thresholds and ranking criteria for balance between compression and performance.

In future, I plan to expand this to:

1. Local distillation based pruning
2. Task agnostic pruning
3. Automated NaS
4. KD based recovery
5. Support for more models

## Model Comparison with LLM Judge

The `judge.py` script allows you to compare two language models using an LLM as a judge. It runs a dataset through both models, stores the detokenized outputs, and then uses an LLM API endpoint to determine which model performed better.

### Usage

```bash
python src/judge.py \
  --original_model <hf_ckpt> \
  --pruned_model /path/to/pruned_model \
  --dataset_name cnn_dailymail \
  --dataset_config 3.0.0 \
  --dataset_split test \
  --num_samples 100 \
  --output_file results/model_comparison.json \
  --llm_api_endpoint https://api.openai.com/v1/chat/completions
```

The results are also displayed in the terminal in a formatted table.
