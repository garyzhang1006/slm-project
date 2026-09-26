# slm project

a small language model i'm building from scratch fr: decoder-only transformer, byte-level tokenizer, 2048-byte context, and presets from a tiny demo up to ~500m params. it's a learning project so ngl expect rough edges.

besides predicting the next token, the model has 3 lil side heads that guess the task type, the kind of error, and how confident it should be. that's just behavior u can observe, not a peek into hidden thoughts, and the data schema straight up refuses fields like `chain_of_thought`.

## where it's at (no cap version)

the 500m model is real (exactly 499,524,075 params) but it can't hold a convo yet. it scored 30/64 on held-out template qs, only 3/24 on everyday qs, and 0/12 on unseen question probes. it's giving data problem, not code problem: it's seen way under 1% of the text a model this size needs. see [measured results](docs/elementary-results.md) and the [500m audit](reports/slm-500m-code-and-capability-audit.md) for the gory details.

the plan to fix that lives in [`compute/`](compute/README.md). tl;dr: either lora-tune a small open model (1 kaggle session) or pretrain the new `slm-160m` on real text and then fine-tune it (~55 hrs of kaggle t4 time at the speed session 1 measured).

## what's in the box

- a byte tokenizer, so no vocab file to download.
- a pytorch transformer w/ a legacy block or a modern one (rope, rmsnorm, swiglu, tied embeddings).
- presets: `demo`, `slm-50m`, `slm-160m` (new models get a gpt-2 style scaled init), and `slm-500m`.
- a jsonl data schema w/ validation, license tracking, and a secret scanner.
- clis to train, generate, eval, audit data, and benchmark checkpoints.
- 2 training modes: instruction/answer sft, and packed raw-text pretraining (`--pretrain-text`) that trains on every token.
- mixed precision, grad accumulation, activation checkpointing, and checkpoints u can resume exactly, sample order included.
- studio, a lil local web ui for poking at the model.
- a "context therapist" that checks long chat histories for drift and contradictions.
- ~290 tests.

## studio

```bash
./launch-studio.command
```

then open [http://127.0.0.1:8766](http://127.0.0.1:8766). first launch builds a python 3.13 env called `.venv-ui-py313` w/ `uv`, and later launches reuse it. keep the terminal open and hit ctrl-c when ur done.

studio has 2 modes:

- **source excerpts** is the default. paste some reference text, ask a q, and u get up to 3 quotes from ur text w/ `[S1]`-style citations. no model involved, and when nothing in ur text matches it says so instead of making stuff up (no hallucination arc).
- **model response** actually runs the model. defaults are tuned for short answers (temp 0.3, top-p 0.9, 64 tokens), and u can also send a repetition penalty and stop sequences thru the api.

studio loads `artifacts/slm-500m-language-quality.pt` by default and checks it has exactly 499,524,075 params. weights aren't in git, so grab them from kaggle first. to use a diff checkpoint, run `./launch-studio.command --checkpoint /path/to/model.pt`, and if the port is taken, add `--port 8767`.

the model that actually answers qs rn is smollm2-360m-instruct w/ the lora adapter from the `slm-lora-baseline` kaggle run (18/22 exact on the holdout, lowkey goated). download its `artifacts/lora-adapter` folder, `pip install transformers peft`, and run `./launch-studio.command --lora-adapter /path/to/lora-adapter`. first launch downloads the base model (~700 mb) from hugging face. runs after the merge update also save `artifacts/lora-merged`, which has the adapter baked in, so `--lora-adapter /path/to/lora-merged` works w/ just `pip install transformers` and no extra download.

want just the source-excerpt mode, w/ no weights and no pytorch?

```bash
./launch-studio.command --sources-only
```

other stuff to know: pasted text is capped at 12,000 bytes and qs at 2,000, and nothing gets fetched from the internet. history lives in the page until u refresh, and model output is shown as text and never executed.

## quick start

the demo model is tiny and trains fine on a laptop cpu, but don't expect it to be smart lol. it's there to prove the pipeline works end to end.

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

`generate` defaults to `--task-type language_generation`. it also takes `--top-p`, `--repetition-penalty`, and `--stop` for cutting answers off at a string. for code, u can ask for a few candidates and let a syntax check help pick one (generated code is never run):

```bash
.venv/bin/python -m cognition_slm.generate --checkpoint artifacts/demo.pt \
  --prompt "Write a Python function that returns the factorial of n." \
  --task-type code_generation --temperature 0.8 --num-candidates 4 --syntax-bonus 0.5
```

to compare architectures, train one w/ `--architecture legacy` and run `cognition_slm.benchmark --model modern=artifacts/demo.pt --model legacy=artifacts/legacy-demo.pt --data data/eval.jsonl`. add `--device cuda` if u have a gpu.

### resuming

point `--resume` at a checkpoint and set `--steps` to the total u want, not how many more. optimizer state, scheduler, rng, and the position in the shuffled data all come back, so a resumed run matches an uninterrupted one. the lr comes from the checkpoint; if u pass a diff `--learning-rate` u also need `--override-learning-rate`, so u can't change it by accident.

```bash
.venv/bin/python -m cognition_slm.train --data data/demo.jsonl --eval-data data/eval.jsonl \
  --resume artifacts/demo.pt --out artifacts/demo-resumed.pt --steps 120
```

## training for real (on kaggle)

anything bigger than the demo trains on kaggle's free t4 gpus, never locally. easiest way in is the [`compute/`](compute/README.md) folder, which has the whole "make it actually speak english" pipeline as ready-to-push kaggle jobs: build a corpus, pretrain `slm-160m`, build chat data, fine-tune, and score it. it also has a lora shortcut on smollm2-360m-instruct if u just want smth that answers qs tonight.

training knobs worth knowing abt:

- `--pretrain-text file.jsonl` (one `{"text": ...}` per line) packs raw text into full 2048-byte rows and puts the loss on every token. use it for web text instead of wrapping paragraphs as fake instructions.
- `--aux-loss-weight 0` turns off the task, error, and confidence heads, which is what u want when the labels are just constants.
- `--max-seconds` stops cleanly after a step and saves everything, so a kaggle session never gets cut off mid-write.
- fp16 overflow skips the step instead of crashing the run.

here's a plain sft run for a kaggle notebook:

```bash
PYTHONPATH=src python -m cognition_slm.train \
  --data /kaggle/input/your-data/train.jsonl \
  --eval-data /kaggle/input/your-data/eval.jsonl \
  --out /kaggle/working/model.pt \
  --preset slm-160m --device cuda --precision fp16 \
  --batch-size 8 --gradient-accumulation-steps 4 \
  --gradient-checkpointing --save-every 250 --steps 2000
```

### older kaggle runners

`scripts/` still has the runners that made the current 500m checkpoints. `scripts/prepare_kaggle.py` packs the source, tests, and data into one private notebook w/ sha-256 checks and no creds or weights, then u push it:

```bash
python scripts/prepare_kaggle.py --owner YOUR_KAGGLE_USERNAME --runner kaggle_run.py --out /tmp/slm-kaggle
kaggle kernels push -p /tmp/slm-kaggle --accelerator NvidiaTeslaT4
```

always pass the explicit t4 accelerator. kaggle's default p100 gets detected fine but the installed pytorch can't actually run on it (big l).

| runner | what it did |
|---|---|
| `kaggle_run.py` | smoke test: full-length training, resume, and generation on the big preset |
| `kaggle_quality_run.py`, `kaggle_500m_quality_run.py` | the first studio checkpoints, trained on a small synthetic curriculum (mostly memorized it) |
| `kaggle_english_run.py` | tinystories + dolly continuation of the 500m model |
| `kaggle_qa_run.py`, `kaggle_long_run.py`, `kaggle_efficient_run.py` | fineweb-edu paragraphs + squad question answering |
| `kaggle_short_qa_pilot.py`, `kaggle_elementary_run.py` | short-answer and elementary-q training, now auto-scored on the 24-q holdout |
| `kaggle_studio_verify.py` | boots studio on kaggle against a finished checkpoint |

`./scripts/launch_500m_kaggle.sh YOUR_KAGGLE_USERNAME` packages and submits the 500m quality run in one go. none of these runners ever swaps studio's default checkpoint for u; read the answers in their reports first. more detail in [docs/kaggle.md](docs/kaggle.md).

heads up: older quality reports tested on prompts that overlapped the training data, so don't read them as held-out scores (they're kinda sus).

## context therapist

side tool, doesn't need the model at all. u feed it a chat history from some other llm, and it checks for trouble: running out of context, repeating itself, contradicting its own instructions, drifting off task, claiming "tests pass" w/o receipts, or leaving qs hanging. it hands back a repair prompt plus a list of which turns to keep.

```bash
.venv/bin/cognition-slm-context-therapist \
  --input examples/context_history.json \
  --token-budget 512 \
  --goal "Preserve the coding task and verified evidence"
```

it counts tokens w/ this project's byte tokenizer, so give it a conservative budget if ur real model uses a diff tokenizer. more in [docs/context_therapy.md](docs/context_therapy.md).

## data (and whose it is)

`data/demo.jsonl` and `data/eval.jsonl` are small synthetic sets i wrote (57 training and 22 held-out records, cc0). `data/simple_questions_holdout.json` holds the 24 everyday qs the model keeps fumbling, and `scripts/score_holdout.py` grades answers against it automatically.

nothing downloaded gets committed. the kaggle jobs pull these datasets at pinned revisions, and every converted record keeps its source and license fields:

- [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`): odc-by for the database, and the original text keeps its owners' rights.
- [tinystories](https://huggingface.co/datasets/roneneldan/TinyStories) by ronen eldan and yuanzhi li: cdla-sharing-1.0.
- [dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), copyright 2023 databricks, inc.: cc-by-sa-3.0, w/ some contexts from wikipedia contributors.
- [squad](https://huggingface.co/datasets/rajpurkar/squad) by pranav rajpurkar and collaborators: cc-by-sa-4.0, w/ wikipedia passages.
- [openassistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1): apache-2.0, used by the `compute/` sft data builder.
- the lora baseline starts from [smollm2-360m-instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), which is apache-2.0.

filtering and prompt formatting are my changes, and the original licenses still apply. to bring ur own data, run `scripts/prepare_data.py`, which needs an explicit source and license, then run the audit before training. full contract is in [data/readme.md](data/README.md).

## how it fits together

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

the training cli defaults to the tiny `demo` preset: 2 blocks, 4 heads, 128 dims, and a 2048-byte context. pick `--preset slm-50m`, `slm-160m`, or `slm-500m` for bigger runs, and explicit dimension flags override the preset. lowkey important: 2048 bytes is only ~400 english words, way shorter than 2048 subword tokens would be.

training tells u how many records got truncated, and it rejects prompts too long to leave any room for an answer. generation caps the answer so it fits in the window, instead of quietly sliding the prompt out of view.

## stuff i'm curious abt

1. does the confidence head actually track whether held-out code is correct?
2. can the error head tell syntax errors apart from logic errors?
3. does short, readable feedback help code gen w/o turning into hidden-reasoning collection?
4. do the prompt-only heads stay calibrated when they never get to see the answer?

`evaluate` reports accuracy, macro-f1, calibration error, and confusion matrices for each head, plus syntax validity and required-symbol recall for code. definitions are in [docs/cognition.md](docs/cognition.md), and older audits are in [reports/audit.md](reports/audit.md) and [reports/context_therapy_audit.md](reports/context_therapy_audit.md).

## known limits

- it doesn't answer general qs well yet (see "where it's at" above).
- the confidence head predicts a label from the training data. it isn't a calibrated probability.
- the side heads predict behavior. they don't reveal anything abt the model's internal state.
- passing a syntax check doesn't mean code works, and reranking can pick a tidy-looking wrong answer.
- treat generated code as untrusted, and only run it in a sandbox.
- normal training and studio never touch the internet, except `--lora-adapter` mode downloads the pinned smollm2 base model once. only the kaggle jobs download the datasets listed above.
