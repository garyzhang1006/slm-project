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

Visit [SLM Studio](http://127.0.0.1:8766). The first launch creates a separate Python 3.13 environment named `.venv-ui-py313` with `uv` if needed; subsequent launches reuse it. Keep the terminal open, and press Control-C to stop. If you prefer manual setup, create `.venv-ui-py313`, install `.[dev]`, and run `PYTHONPATH=src .venv-ui-py313/bin/python -m cognition_slm.server`. The launcher checks PyTorch with a 30-second deadline and reports import failures instead of waiting indefinitely. Inference memory-maps ZIP checkpoints and excludes optimizer state; training resume still loads its full state.

Studio defaults to `artifacts/slm-500m-language-quality.pt` and checks that it contains exactly 499,524,075 model parameters. Missing weights produce an error; Studio never silently falls back to a smaller checkpoint. Weights are not included in Git: download the completed Kaggle quality checkpoint into that path before launching. To deliberately use another checkpoint, supply `./launch-studio.command --checkpoint /path/to/model.pt`. An occupied port can be changed with `--port 8767`.

The interface includes starter prompts, task selection, temperature and response-length controls, context budgeting, response copying, and session history. Language generation is the first-run task; code starter cards select code tasks explicitly. Each prompt is independent; history is kept in page memory and clears on refresh. Prompts stay on your machine. Model text is displayed without execution.

## 2048 context and Kaggle

New training runs default to 2048 tokens. This tokenizer represents UTF-8 bytes, so 2048 includes prompt markup and special tokens and is much shorter in text than 2048 subword tokens. Existing checkpoints retain their saved context and architecture when resumed.

The `slm-50m` preset uses 12 layers and 512 hidden dimensions. The `slm-500m` preset uses 24 layers, 1,140 hidden dimensions, 10 attention heads, RoPE, RMSNorm, SwiGLU, and tied embeddings, for exactly 499,524,075 parameters with the current six task types. Train `slm-500m` on a Kaggle GPU with activation checkpointing. Studio can load the resulting checkpoint for inference; the existing training checkpoint is about 6 GB because it includes optimizer state, so startup requires substantial free RAM. A larger model still requires a substantial licensed training corpus before its outputs become useful.

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

For a remote 500M quality run, use the dedicated Kaggle runner. It keeps the 2,048-token context, uses activation checkpointing with an effective batch size of eight, and writes `artifacts/slm-500m-language-quality.pt` plus `quality_verification_500m.json`:

```bash
./scripts/launch_500m_kaggle.sh YOUR_KAGGLE_USERNAME
```

The launcher accepts optional kernel slug and output directory arguments. It only packages source and submits Kaggle work; it never trains locally. The equivalent underlying commands are `scripts/prepare_kaggle.py` followed by `kaggle kernels push`.

This training run is remote-only. Its report records probe outputs; it does not automatically grade their correctness. Download the checkpoint to `artifacts/slm-500m-language-quality.pt`, then run `./launch-studio.command` to load it with the default parameter-count check.

`scripts/kaggle_studio_verify.py` verifies the default server on Kaggle using an attached completed quality kernel. Package it with `--runner kaggle_studio_verify.py`, add the completed kernel's owner/slug to `kernel_sources` in the generated metadata, and submit. It runs the regression suite, launches Studio, checks its actual parameter count and context, and sends an HTTP generation request without retraining.

Historical quality reports used overlapping training/evaluation prompts and must not be interpreted as held-out performance. The curriculum generator now holds out prompt wording, but shares concepts and answers across splits; this measures narrow wording transfer. Existing weights and reports are unchanged. In the previous 500M probes, the loop explanation was incorrect and the binary-search answer was truncated.

## Quick start

The [500M code and capability audit](reports/slm-500m-code-and-capability-audit.md) verified the model size and passed 89 regression tests, but found zero correct answers on twelve unseen question probes. The current checkpoint is unsuitable for reliable English question answering. Software fixes in this revision do not retrain its weights.

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

### English training on Kaggle

The long continuation run starts from the latest QA pilot and budgets 5.5 hours for
broader English with QA replay, followed by 3.5 hours of QA. It streams the pinned
`sample-10BT` subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and collects up to roughly 60,000 complete short paragraphs. Paragraphs retain their
source URLs; the database uses ODC-BY, while underlying text rights remain with their
owners. Holdout passages are excluded, and source URLs determine English train/eval
splits. This run learns paragraph continuations, not an entire 10-billion-token corpus.

```bash
python scripts/prepare_kaggle.py --out /tmp/slm-long-kaggle \
  --owner garyzhang11111 --slug slm-500m-long-training \
  --runner kaggle_long_run.py
kaggle kernels push -p /tmp/slm-long-kaggle --accelerator NvidiaTeslaT4 --timeout 39600
```

All training and model evaluations happen on Kaggle. `--max-seconds` stops training
after a completed step and saves optimizer, scheduler, and RNG state. The runner
reserves time beyond its nine-hour training budget for data preparation, evaluation,
and checkpoint writes; actual training time can be lower. A 100,000-step ceiling is
an upper bound, not a claim that that many steps ran. Reports distinguish requested
and completed steps. The final QA checkpoint retains optimizer state for continuation.

`long_training_report.json` records actual steps, source hashes, development scores,
and generated answers. Final passage scores use one stored answer per question and
are not directly comparable to the earlier multi-reference SQuAD scores. The original
12 general questions are also rerun. Longer training does not automatically establish
conversational ability, and the runner does not install its output into Studio.

For the next question-answering run, use `--runner kaggle_qa_run.py` and
`--slug slm-500m-answer-training` with the packaging command below. This runner
attaches the completed English corpus run and verifies the latest custom question
checkpoint's hash. It uses short Dolly answers from the existing training split and
[SQuAD](https://huggingface.co/datasets/rajpurkar/squad) passage questions
(Pranav Rajpurkar and collaborators; CC-BY-SA-4.0, with Wikipedia source passages).
SQuAD passages are preserved in full; examples exceeding 1,024 byte tokens are rejected.
This teaches passage-based answering and does not establish broad factual knowledge.

The QA runner compares genuine versus shuffled prompts and records both greedy and
sampled answers. A 500-step pilot must improve development exact match by at least
two percentage points, or token F1 by five points without reducing exact match,
before a further 4,000 training steps run. This gate measures early progress, not
readiness for deployment. Each stage starts a fresh optimizer from the preceding
model weights. Final evaluation uses 128 separate passage questions and the existing
12 general-question probes. Development and final-test passages are kept separate.
See `qa_training_report.json` for the gate decision, scores, and generated answers;
no checkpoint is installed into Studio automatically.

`scripts/kaggle_english_run.py` continues training the project's own 499,524,075-parameter
model. It retains the custom architecture and existing weights; it does not load a
pretrained third-party model. The runner downloads text only on Kaggle, prepares up to
30,000 accepted short stories, then trains on instruction/answer examples with some
story replay. Oversized examples are rejected intact, and each source has 128 held-out
examples. The existing unseen question probes remain excluded from training.

Package and submit from a machine with the Kaggle CLI configured:

```bash
python scripts/prepare_kaggle.py --out /tmp/slm-english-kaggle \
  --owner garyzhang11111 --slug slm-500m-english-corpus \
  --runner kaggle_english_run.py
kaggle kernels push -p /tmp/slm-english-kaggle --accelerator NvidiaTeslaT4
```

Packaging does not execute the model. The generated private notebook selects T4 GPU and
internet access and attaches the original checkpoint kernel. Its SHA-256 must match
the recorded parent before any training starts. Training uses 4,000 English steps
and 1,000 instruction-tuning steps, with eight examples per effective batch. Each stage
resets optimizer state and preserves model weights. Checkpoints save every 250 steps;
their step counts are local to that stage. The saved context remains 2,048 byte tokens.

The runner checks actual GPU execution before running tests. Kaggle's default P100
can be detected successfully even when the installed PyTorch build cannot execute
on it; keep the explicit T4 selection when submitting. `gpu_preflight.json` records
the device, PyTorch version, and compatibility result.

The notebook runs the regression suite, records source/data hashes, and evaluates both
the original and new checkpoints. `english_training_report.json` records progress;
`baseline_evaluation.json`, `english_evaluation.json`, and `questions_evaluation.json`
contain held-out losses and generated answers. Completed training is not an English
or question-answering quality pass. Read the answers before changing Studio's checkpoint.
The runner leaves the current Studio checkpoint unchanged.

Text sources are pinned to dataset revisions in the runner:

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), by Ronen Eldan
  and Yuanzhi Li, contains synthetic English stories and uses CDLA-Sharing-1.0.
- [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
  copyright 2023 Databricks, Inc., contains human-written instructions and responses
  under CC-BY-SA-3.0. Some contexts derive from Wikipedia contributors.

Converted records retain source and license fields. Filtering and prompt formatting
are this project's modifications; dataset licenses continue to apply to redistributed
records. No downloaded corpus is committed to this repository.

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
- Normal training and Studio do not download internet data. The explicitly selected
  English Kaggle runner downloads the two licensed text datasets described above.
