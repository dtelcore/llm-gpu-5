# Input Directory

Current state: generation prompts should match the repaired corpus/tokenizer assumptions. Keep newline-heavy prompts intact when you want to exercise the same boundary behavior as training.

This directory stores input files and prompts for model inference and text generation.

## Contents

- **Prompt Files**: Text files containing initial sequences for generation
- **Inference Input**: Tokenized sequences for model prediction
- **Configuration Files**: JSON/YAML files for generation settings (temperature, top-k, etc.)

## Example Input Structure

```
input/
├── prompts.txt              # Generation starting sequences (one per line)
├── inference_batch.txt      # Batch of input tokens for prediction
└── config.json              # Generation hyperparameters
```

## Using Input Files with Generation Pipeline

After model training, provide input prompts:

```python
with open('input/prompts.txt', 'r') as f:
    prompts = f.readlines()

for prompt in prompts:
    generated_text = model.generate(prompt, max_length=50)
    print(generated_text)
```

## Input Format Guidelines

- **Plain Text**: One prompt per line
- **Token IDs**: Space-separated integers (0..vocab_size-1)
- **Configuration**: JSON with keys: `temperature`, `top_k`, `top_p`, `max_length`

```json
{
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "max_length": 128
}
```
