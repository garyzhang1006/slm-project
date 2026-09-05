# slm project

A decoder-only small language model for coding and language experiments, with a 2048-byte-token training context and 50M and 500M-class presets. It studies generation and observable model behavior.

The project studies behavior, not hidden chain-of-thought. Each training example carries a coding or language task, an answer, a task type, a confidence bucket, an error category, and a source/license record. The transformer learns next-token prediction while small auxiliary heads predict the task type, error category, and confidence bucket.

## What is included

- Byte-level tokenizer with no downloaded vocabulary.
- Causal transformer implemented in PyTorch, with demo, `slm-50m`, and `slm-500m` presets.
- Selectable legacy or modern transformer blocks with rotary positions, RMSNorm, SwiGLU, and tied embeddings.
- JSONL data schema with validation and license metadata.
- Project-authored synthetic data for coding and language-generation tasks.
- Training, generation, evaluation, audit, and checkpoint-benchmark CLIs.
- Tests for data and tokenizer paths, plus model tests when PyTorch is installed.
- Prompt-only pooling for cognition heads, answer-focused language loss, resumable training, validation checkpoints, and calibrated evaluation metrics.
- Static Python syntax and required-symbol checks, plus optional multi-candidate reranking without code execution.
- Optimized scaled dot-product attention, CUDA mixed precision, gradient accumulation, and optional activation checkpointing.
- Periodic atomic checkpoints with optimizer, scheduler, gradient scaler, and random state for interrupted training.

This is a mini project. The demo corpus is intentionally too small to produce a useful coding assistant. It proves the pipeline shape, not model quality.

## Local Studio

Open `launch-studio.command` in Finder, or run it from this project:

```bash
./launch-studio.command
```

Visit [SLM Studio](http://127.0.0.1:8766). The first launch creates a separate Python 3.11 environment with `uv` if needed; subsequent launches reuse it. Keep the terminal open, and press Control-C to stop. If you prefer manual setup, create `.venv-ui`, install `.[dev]`, and run `PYTHONPATH=src .venv-ui/bin/python -m cognition_slm.server`.

Studio automatically loads `artifacts/slm-50m-language-quality.pt` when that downloaded checkpoint exists, then falls back to `artifacts/slm-50m-2048-smoke.pt`. Checkpoint weights are not included in Git; use a checkpoint downloaded from your Kaggle outputs or supply `./launch-studio.command --checkpoint /path/to/model.pt`. An occupied port can be changed with `--port 8767`.

The interface includes starter prompts, task selection, temperature and response-length controls, context budgeting, response copying, and session history. Language generation is the first-run task; code starter cards select code tasks explicitly. Each prompt is independent; history is kept in page memory and clears on refresh. Prompts stay on your machine. Model text is displayed without execution.

## 2048 context and Kaggle

New training runs default to 2048 tokens. This tokenizer represents UTF-8 bytes, so 2048 includes prompt markup and special tokens and is much shorter in text than 2048 subword tokens. Existing checkpoints retain their saved context and architecture when resumed.

The `slm-50m` preset uses 12 layers and 512 hidden dimensions. The `slm-500m` preset uses 24 layers, 1,140 hidden dimensions, 10 attention heads, RoPE, RMSNorm, SwiGLU, and tied embeddings, for exactly 499,524,075 parameters with the current six task types. Use `slm-500m` only on a Kaggle GPU with activation checkpointing; it is not a local development preset. A larger model still requires a substantial licensed training corpus before its outputs become useful.

Run this inside a Kaggle GPU session with your prepared data:

```bash
PYTHONPATH=src python -m cognition_slm.train \
  --data /kaggle/input/your-data/train.jsonl \
  --eval-data /kaggle/input/your-data/eval.jsonl \
  --out /kaggle/working/slm-2048.pt \
  --preset slm-500m --device cuda --precision fp16 \
  --batch-size 1 --gradient-accumulation-steps 8 \
  --gradient-checkpointing --save-every 100 --steps 1000
```

`--steps` counts training iterations; the effective batch is microbatch size times accumulation steps. AMP may skip an optimizer update when gradients overflow. Biases and normalization vectors are excluded from weight decay in new runs. Accumulation weights language loss by supervised tokens, and auxiliary losses by examples, so variable answer lengths do not change the objective when splitting a batch.

For a reproducible private verification run, `scripts/prepare_kaggle.py` packages an explicit source allowlist, tests, and synthetic data into one script with SHA-256 checks. It excludes credentials and existing checkpoints. The runner requires Kaggle and a working CUDA device, runs the regression suite, and exercises full-length training, checkpoint resume, and generation with the larger preset:

```bash
python scripts/prepare_kaggle.py --owner YOUR_KAGGLE_USERNAME --out /tmp/slm-kaggle
kaggle kernels push -p /tmp/slm-kaggle --accelerator NvidiaTeslaT4
kaggle kernels status YOUR_KAGGLE_USERNAME/slm-2048-verification
```

The generated kernel is private and runs without internet. Its synthetic length fixture covers both coding and language task labels; it is a smoke test, not a capability benchmark. `verification.json` records hardware, source hashes, parameter count, and completion evidence. See [Kaggle verification details](docs/kaggle.md).

To build a Studio checkpoint after changing training data, select the quality runner. It combines the base conversational examples with a deterministic project-authored English/Python curriculum, trains the 50M preset on Kaggle for 1,200 steps, and probes greetings, grammar, explanations, code generation, debugging, and algorithm reasoning:

```bash
python scripts/prepare_kaggle.py \
  --owner YOUR_KAGGLE_USERNAME \
  --slug slm-50m-studio-quality \
  --runner kaggle_quality_run.py \
  --out /tmp/slm-kaggle-quality
kaggle kernels push -p /tmp/slm-kaggle-quality --accelerator NvidiaTeslaT4
```

Download `artifacts/slm-50m-language-quality.pt` from the completed kernel and start Studio with `--checkpoint` pointing to it. The quality runner remains a synthetic smoke experiment; it is intended to teach narrow English and Python patterns, not establish broad language ability. Downloaded weights stay out of Git.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ".[dev]"
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
.venv/bin/python -m cognition_slm.train \
  --data data/demo.jsonl \
  --out artifacts/legacy-demo.pt \
  --architecture legacy \
  --steps 60 \
  --batch-size 4 \
  --warmup-steps 5
.venv/bin/python -m cognition_slm.benchmark \
  --model modern=artifacts/demo.pt \
  --model legacy=artifacts/legacy-demo.pt \
  --data data/eval.jsonl
```

For sampled code generation, request several candidates. The reranker uses model likelihood and a static Python syntax bonus; it never executes generated text:

```bash
.venv/bin/python -m cognition_slm.generate \
  --checkpoint artifacts/demo.pt \
  --prompt "Write a Python function that returns the factorial of n." \
  --task-type code_generation \
  --temperature 0.8 \
  --num-candidates 4 \
  --syntax-bonus 0.5
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

## Context therapist for long histories

The repository also includes a deterministic context-care controller for another language model. It checks visible message history for token pressure, repeated turns, conflicting directives, instruction drift, unsupported success claims, and unresolved uncertainty. It returns a repair prompt plus a handoff packet that marks turns to retain or review.

Run it against the included synthetic fixture:

```bash
.venv/bin/cognition-slm-context-therapist \
  --input examples/context_history.json \
  --token-budget 512 \
  --goal "Preserve the coding task and verified evidence"
```

This layer is an observable context controller, not a consciousness probe or hidden-thought reader. `estimated_tokens` uses this project's byte tokenizer, so use a conservative budget when the downstream model uses a different tokenizer. See [`docs/context_therapy.md`](docs/context_therapy.md) for the integration contract.

## Data boundary

`data/demo.jsonl` and `data/eval.jsonl` contain project-authored synthetic examples marked `CC0-1.0`. The current snapshot has 57 training records and 22 held-out records, including `language_generation` examples for greetings, summaries, rewriting, translation, and short-form writing. No external training corpus is bundled. `scripts/prepare_data.py` converts a local JSONL file into the canonical schema and requires an explicit source and license. Add only data you are allowed to use, and run the audit before training.

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
legacy or modern causal transformer
    |                         \
    v                          v
answer-token loss       prompt-boundary pooling
                         task / error / confidence heads
    |
candidate generation -> static code checks -> selected output
```

The training CLI defaults to the `demo` preset: a modern model with two transformer blocks, four attention heads, 128 hidden dimensions, and a 2048-token context window. `ModelConfig` retains historical legacy/256 defaults for direct construction and checkpoints missing those fields. Select `--preset slm-50m` or `--preset slm-500m` for larger Kaggle runs; explicit dimension flags override a preset on new runs.

Training reports the actual maximum encoded length and number of truncated records. Truncated answers retain their last real byte rather than receiving an artificial end-of-sequence token. An oversized prompt that leaves no answer tokens is rejected. Generation rejects oversized formatted prompts instead of silently removing the answer delimiter; decoding beyond the window uses rolling context.

## Research questions

1. Does a model's confidence bucket track held-out coding correctness?
2. Do error-category predictions separate syntax errors from logic errors?
3. Does adding concise inspectable feedback improve code generation without encouraging hidden-reasoning collection?
4. Do prompt-only cognition heads remain calibrated when answer tokens are withheld?

Evaluation reports scalar accuracy, macro-F1, expected calibration error, and confusion matrices for each auxiliary head. Coding tasks also report static Python syntax validity, required-symbol recall, and a narrow static score. Accuracy remains in the top-level JSON fields for compatibility.

See [`docs/cognition.md`](docs/cognition.md) for definitions and limits, [`docs/context_therapy.md`](docs/context_therapy.md) for context care, [`data/README.md`](data/README.md) for the data contract, [`reports/audit.md`](reports/audit.md) for the initial model audit, and [`reports/context_therapy_audit.md`](reports/context_therapy_audit.md) for current context-care verification.

## Known limits

- A tiny synthetic corpus cannot support claims about general coding or language ability.
- Confidence is a label learned from annotations, not calibrated probability.
- Auxiliary heads expose behavior-level predictions, not true internal states.
- Prompt-only pooling prevents the auxiliary heads from reading answer tokens during training; it does not prove that their labels represent internal reasoning.
- Static syntax validity does not establish runtime correctness. Candidate reranking can select a valid-looking answer that is still wrong.
- Generated code is untrusted text. Execute it only in a sandbox with resource limits.
- The repository does not download internet data automatically.
