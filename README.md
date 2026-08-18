# slm-project

Mini research project: a small decoder-only language model aimed at coding help and observable LLM cognition signals.

The project studies behavior, not hidden chain-of-thought. Each training example carries a coding task, an answer, a task type, a confidence bucket, an error category, and a source/license record. The transformer learns next-token prediction while small auxiliary heads predict the task type, error category, and confidence bucket.

## What is included

- Byte-level tokenizer with no downloaded vocabulary.
- Tiny causal transformer implemented in PyTorch.
- JSONL data schema with validation and license metadata.
- Deterministic synthetic demo data for coding tasks.
- Training, generation, evaluation, and audit CLIs.
- Tests for data and tokenizer paths, plus model tests when PyTorch is installed.
- Prompt-only pooling for cognition heads, resumable training, validation checkpoints, and calibrated evaluation metrics.

This is a mini project. The demo corpus is intentionally too small to produce a useful coding assistant. It proves the pipeline shape, not model quality.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m cognition_slm.audit --train data/demo.jsonl --eval data/eval.jsonl
.venv/bin/python -m cognition_slm.train \
  --data data/demo.jsonl \
  --eval-data data/eval.jsonl \
  --out artifacts/demo.pt \
  --steps 60 \
  --batch-size 4 \
  --eval-every 20 \
  --warmup-steps 5
.venv/bin/python -m cognition_slm.generate \
  --checkpoint artifacts/demo.pt \
  --prompt "Write a Python function that returns the factorial of n." \
  --task-type code_generation
.venv/bin/python -m cognition_slm.evaluate \
  --checkpoint artifacts/demo.pt \
  --data data/eval.jsonl
```

Resume a checkpoint by setting `--steps` to the target total step count. New checkpoints retain optimizer and scheduler state, while older model-only checkpoints load with a fresh optimizer:

```bash
.venv/bin/python -m cognition_slm.train \
  --data data/demo.jsonl \
  --eval-data data/eval.jsonl \
  --resume artifacts/demo.pt \
  --out artifacts/demo-resumed.pt \
  --steps 120
```

If PyTorch is unavailable, the audit and data tests still run. Training and model tests fail with an actionable installation message rather than silently using a fake model.

## Data boundary

`data/demo.jsonl` and `data/eval.jsonl` contain project-authored synthetic examples marked `CC0-1.0`. No external training corpus is bundled. `scripts/prepare_data.py` converts a local JSONL file into the canonical schema and requires an explicit source and license. Add only data you are allowed to use, and run the audit before training.

The project deliberately does not accept fields named `chain_of_thought`, `cot`, `hidden_reasoning`, or `private_thoughts`. A short inspectable explanation can be represented in the answer, but a verbal explanation is not evidence of a model's hidden internal process.

## Architecture

```text
JSONL records
    |
    v
schema validation + license/secret audit
    |
    v
byte tokenizer (259 tokens)
    |
    v
tiny causal transformer
    |                  \
    v                   v
next-token loss   prompt-boundary pooling
                  task / error / confidence heads
```

Default model is deliberately small: two transformer blocks, four attention heads, 128 hidden dimensions, and a 256-token context window. Change `ModelConfig` or CLI settings after the demo works.

## Research questions

1. Does a model's confidence bucket track held-out coding correctness?
2. Do error-category predictions separate syntax errors from logic errors?
3. Does adding concise inspectable feedback improve code generation without encouraging hidden-reasoning collection?
4. Do prompt-only cognition heads remain calibrated when answer tokens are withheld?

Evaluation reports scalar accuracy, macro-F1, expected calibration error, and confusion matrices for each auxiliary head. Accuracy remains in the top-level JSON fields for compatibility.

See [`docs/cognition.md`](docs/cognition.md) for definitions and limits, [`data/README.md`](data/README.md) for the data contract, and [`reports/audit.md`](reports/audit.md) for the initial audit record.

## Known limits

- A tiny synthetic corpus cannot support claims about general coding ability.
- Confidence is a label learned from annotations, not calibrated probability.
- Auxiliary heads expose behavior-level predictions, not true internal states.
- Prompt-only pooling prevents the auxiliary heads from reading answer tokens during training; it does not prove that their labels represent internal reasoning.
- Generated code is untrusted text. Execute it only in a sandbox with resource limits.
- The repository does not download internet data automatically.
