# slm project

A small language model I'm building from scratch: decoder-only transformer, byte-level tokenizer, 2048-byte context, and presets from a tiny demo up to about 500M parameters. It's a learning project, so expect rough edges.

Besides predicting the next token, the model has three little side heads that guess the task type, the kind of error, and how confident it should be. That's just behavior you can observe, not a peek into hidden thoughts, and the data schema flat out refuses fields like `chain_of_thought`.

## Where it's at (honest version)

The 500M model is real (exactly 499,524,075 parameters) but it can't hold a conversation yet. It scored 30/64 on held-out template questions, only 3/24 on everyday questions, and 0/12 on unseen question probes. The problem is data, not code: it has seen well under 1% of the text a model this size needs. See [measured results](docs/elementary-results.md) and the [500M audit](reports/slm-500m-code-and-capability-audit.md) for the gory details.

The plan to fix that lives in [`compute/`](compute/README.md). Short version: either LoRA-tune a small open model (one Kaggle session) or pretrain the new `slm-160m` on real text and then fine-tune it (roughly 35 hours of Kaggle T4 time, by estimate).

## What's in the box

- A byte tokenizer, so there's no vocab file to download.
- A PyTorch transformer with a legacy block or a modern one (RoPE, RMSNorm, SwiGLU, tied embeddings).
- Presets: `demo`, `slm-50m`, `slm-160m` (new models get a GPT-2 style scaled init), and `slm-500m`.
- A JSONL data schema with validation, license tracking, and a secret scanner.
- CLIs to train, generate, evaluate, audit data, and benchmark checkpoints.
- Two training modes: instruction/answer SFT, and packed raw-text pretraining (`--pretrain-text`) that trains on every token.
- Mixed precision, gradient accumulation, activation checkpointing, and checkpoints you can resume exactly, sample order included.
- Studio, a little local web UI for poking at the model.
- A "context therapist" that checks long chat histories for drift and contradictions.
- Around 290 tests.

## Studio

```bash
./launch-studio.command
```

Then open [http://127.0.0.1:8766](http://127.0.0.1:8766). The first launch builds a Python 3.13 environment called `.venv-ui-py313` with `uv`, and later launches reuse it. Keep the terminal open and hit Control-C when you're done.

Studio has two modes:

- **Source excerpts** is the default. Paste some reference text, ask a question, and you get up to three quotes from your text with `[S1]`-style citations. No model is involved, and when nothing in your text matches, it says so instead of making something up.
- **Model response** actually runs the model. Defaults are tuned for short answers (temperature 0.3, top-p 0.9, 64 tokens), and you can also send a repetition penalty and stop sequences through the API.

Studio loads `artifacts/slm-500m-language-quality.pt` by default and checks it has exactly 499,524,075 parameters. Weights aren't in Git, so download them from Kaggle first. To use a different checkpoint, run `./launch-studio.command --checkpoint /path/to/model.pt`, and if the port is taken, add `--port 8767`.

Want just the source-excerpt mode, with no weights and no PyTorch?

```bash
./launch-studio.command --sources-only
```

A few other things to know: pasted text is capped at 12,000 bytes and questions at 2,000, and nothing gets fetched from the internet. History lives in the page until you refresh, and model output is shown as text and never executed.

## Quick start

The demo model is tiny and trains fine on a laptop CPU, but don't expect it to be smart. It's there to prove the pipeline works end to end.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m cognition_slm.audit --train data/demo.jsonl --eval data/eval.jsonl
.venv/bin/python -m cognition_slm.train \
  --data data/demo.jsonl --eval-data data/eval.jsonl \
  --out artifacts/demo.pt --steps 60 --batch-size 4 --eval-every 20 --warmup-steps 5
.venv/bin/python -m cognition_slm.generate \
  --checkpoint artifacts/demo.pt \
  --prompt "Write a Python function that returns the factorial of n." \
  --task-type code_generation
.venv/bin/python -m cognition_slm.evaluate --checkpoint artifacts/demo.pt --data data/eval.jsonl
```

`generate` defaults to `--task-type language_generation`. It also takes `--top-p`, `--repetition-penalty`, and `--stop` for cutting answers off at a string. For code, you can ask for a few candidates and let a syntax check help pick one (the generated code is never run):

```bash
.venv/bin/python -m cognition_slm.generate --checkpoint artifacts/demo.pt \
  --prompt "Write a Python function that returns the factorial of n." \
  --task-type code_generation --temperature 0.8 --num-candidates 4 --syntax-bonus 0.5
```

To compare architectures, train one with `--architecture legacy` and run `cognition_slm.benchmark --model modern=artifacts/demo.pt --model legacy=artifacts/legacy-demo.pt --data data/eval.jsonl`. Add `--device cuda` if you have a GPU.

### Resuming

Point `--resume` at a checkpoint and set `--steps` to the total you want, not how many more. Optimizer state, scheduler, RNG, and the position in the shuffled data all come back, so a resumed run matches an uninterrupted one. The learning rate comes from the checkpoint; if you pass a different `--learning-rate` you also need `--override-learning-rate`, so you can't change it by accident.

```bash
.venv/bin/python -m cognition_slm.train --data data/demo.jsonl --eval-data data/eval.jsonl \
  --resume artifacts/demo.pt --out artifacts/demo-resumed.pt --steps 120
```

## Training for real (on Kaggle)

Anything bigger than the demo trains on Kaggle's free T4 GPUs, never locally. The easiest way in is the [`compute/`](compute/README.md) folder, which has the whole "make it actually speak English" pipeline as ready-to-push Kaggle jobs: build a corpus, pretrain `slm-160m`, build chat data, fine-tune, and score it. It also has a LoRA shortcut on SmolLM2-360M-Instruct if you just want something that answers questions tonight.

A few training knobs worth knowing about:

- `--pretrain-text file.jsonl` (one `{"text": ...}` per line) packs raw text into full 2048-byte rows and puts the loss on every token. Use it for web text instead of wrapping paragraphs as fake instructions.
- `--aux-loss-weight 0` turns off the task, error, and confidence heads, which is what you want when the labels are just constants.
- `--max-seconds` stops cleanly after a step and saves everything, so a Kaggle session never gets cut off mid-write.
- fp16 overflow skips the step instead of crashing the run.

Here's a plain SFT run for a Kaggle notebook:

```bash
PYTHONPATH=src python -m cognition_slm.train \
  --data /kaggle/input/your-data/train.jsonl \
  --eval-data /kaggle/input/your-data/eval.jsonl \
  --out /kaggle/working/model.pt \
  --preset slm-160m --device cuda --precision fp16 \
  --batch-size 8 --gradient-accumulation-steps 4 \
  --gradient-checkpointing --save-every 250 --steps 2000
```

### Older Kaggle runners

`scripts/` still has the runners that produced the current 500M checkpoints. `scripts/prepare_kaggle.py` packs the source, tests, and data into one private notebook with SHA-256 checks and no credentials or weights, then you push it:

```bash
python scripts/prepare_kaggle.py --owner YOUR_KAGGLE_USERNAME --runner kaggle_run.py --out /tmp/slm-kaggle
kaggle kernels push -p /tmp/slm-kaggle --accelerator NvidiaTeslaT4
```

Always pass the explicit T4 accelerator. Kaggle's default P100 gets detected fine but the installed PyTorch can't actually run on it.

| Runner | What it did |
|---|---|
| `kaggle_run.py` | Smoke test: full-length training, resume, and generation on the big preset |
| `kaggle_quality_run.py`, `kaggle_500m_quality_run.py` | The first Studio checkpoints, trained on a small synthetic curriculum (mostly memorized it) |
| `kaggle_english_run.py` | TinyStories plus Dolly continuation of the 500M model |
| `kaggle_qa_run.py`, `kaggle_long_run.py`, `kaggle_efficient_run.py` | FineWeb-Edu paragraphs plus SQuAD question answering |
| `kaggle_short_qa_pilot.py`, `kaggle_elementary_run.py` | Short-answer and elementary-question training, now auto-scored on the 24-question holdout |
| `kaggle_studio_verify.py` | Boots Studio on Kaggle against a finished checkpoint |

`./scripts/launch_500m_kaggle.sh YOUR_KAGGLE_USERNAME` packages and submits the 500M quality run in one go. None of these runners ever swaps Studio's default checkpoint for you; read the answers in their reports first. More detail is in [docs/kaggle.md](docs/kaggle.md).

Heads up: older quality reports tested on prompts that overlapped the training data, so don't read them as held-out scores.

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
