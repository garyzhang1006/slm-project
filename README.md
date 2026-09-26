# slm project

A small language model I'm building from scratch: decoder-only transformer, byte-level tokenizer, 2048-byte context, and presets from a tiny demo up to about 500M parameters. It's a learning project, so expect rough edges.

Besides predicting the next token, the model has three little side heads that guess the task type, the kind of error, and how confident it should be. That's just behavior you can observe, not a peek into hidden thoughts, and the data schema flat out refuses fields like `chain_of_thought`.

## Where it's at (honest version)

The 500M model is real (exactly 499,524,075 parameters) but it can't hold a conversation yet. It scored 30/64 on held-out template questions, only 3/24 on everyday questions, and 0/12 on unseen question probes. The problem is data, not code: it has seen well under 1% of the text a model this size needs. See [measured results](docs/elementary-results.md) and the [500M audit](reports/slm-500m-code-and-capability-audit.md) for the gory details.

The plan to fix that lives in [`compute/`](compute/README.md). Short version: either LoRA-tune a small open model (one Kaggle session) or pretrain the new `slm-160m` on real text and then fine-tune it (roughly 55 hours of Kaggle T4 time at the speed session 1 measured).

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

## Context therapist

This is a side tool, and it doesn't need the model at all. You feed it a chat history from some other LLM, and it checks for trouble: running out of context, repeating itself, contradicting its own instructions, drifting off task, claiming "tests pass" without evidence, or leaving questions hanging. It hands back a repair prompt plus a list of which turns to keep.

```bash
.venv/bin/cognition-slm-context-therapist \
  --input examples/context_history.json \
  --token-budget 512 \
  --goal "Preserve the coding task and verified evidence"
```

It counts tokens with this project's byte tokenizer, so give it a conservative budget if your real model uses a different tokenizer. More in [docs/context_therapy.md](docs/context_therapy.md).

## Data (and whose it is)

`data/demo.jsonl` and `data/eval.jsonl` are small synthetic sets I wrote (57 training and 22 held-out records, CC0). `data/simple_questions_holdout.json` holds the 24 everyday questions the model keeps failing, and `scripts/score_holdout.py` grades answers against it automatically.

Nothing downloaded gets committed. The Kaggle jobs pull these datasets at pinned revisions, and every converted record keeps its source and license fields:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`): ODC-BY for the database, and the original text keeps its owners' rights.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li: CDLA-Sharing-1.0.
- [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), copyright 2023 Databricks, Inc.: CC-BY-SA-3.0, with some contexts from Wikipedia contributors.
- [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) by Pranav Rajpurkar and collaborators: CC-BY-SA-4.0, with Wikipedia passages.
- [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1): Apache-2.0, used by the `compute/` SFT data builder.
- The LoRA baseline starts from [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), which is Apache-2.0.

Filtering and prompt formatting are my changes, and the original licenses still apply. To bring your own data, run `scripts/prepare_data.py`, which needs an explicit source and license, and then run the audit before training. The full contract is in [data/README.md](data/README.md).

## How it fits together

```text
JSONL records or raw text
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
next-token loss          prompt pooling -> task / error / confidence heads
(answer-only for SFT,    (skipped for raw text or --aux-loss-weight 0)
 every token for raw text)
    |
generation -> optional static code checks -> output
```

The training CLI defaults to the tiny `demo` preset: 2 blocks, 4 heads, 128 dims, and a 2048-byte context. Pick `--preset slm-50m`, `slm-160m`, or `slm-500m` for bigger runs, and explicit dimension flags override the preset. Keep in mind that 2048 bytes is only about 400 English words, much shorter than 2048 subword tokens would be.

Training tells you how many records got truncated, and it rejects prompts too long to leave any room for an answer. Generation caps the answer so it fits in the window, instead of quietly sliding the prompt out of view.

## Stuff I'm curious about

1. Does the confidence head actually track whether held-out code is correct?
2. Can the error head tell syntax errors apart from logic errors?
3. Does short, readable feedback help code generation without turning into hidden-reasoning collection?
4. Do the prompt-only heads stay calibrated when they never get to see the answer?

`evaluate` reports accuracy, macro-F1, calibration error, and confusion matrices for each head, plus syntax validity and required-symbol recall for code. Definitions are in [docs/cognition.md](docs/cognition.md), and older audits are in [reports/audit.md](reports/audit.md) and [reports/context_therapy_audit.md](reports/context_therapy_audit.md).

## Known limits

- It doesn't answer general questions well yet (see "Where it's at" above).
- The confidence head predicts a label from the training data. It isn't a calibrated probability.
- The side heads predict behavior. They don't reveal anything about the model's internal state.
- Passing a syntax check doesn't mean code works, and reranking can pick a tidy-looking wrong answer.
- Treat generated code as untrusted, and only run it in a sandbox.
- Normal training and Studio never touch the internet. Only the Kaggle jobs download the datasets listed above.
