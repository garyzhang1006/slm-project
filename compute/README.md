# Compute pipeline for an English-answering SLM

The current 500M checkpoint answers 3 of 24 everyday questions in `data/simple_questions_holdout.json` correctly, mostly because it saw far too little plain English before it was taught to answer. This folder holds everything needed to fix that on Kaggle: a 1.5 GB English pretraining corpus, pretraining of the 160M preset across chained 11-hour GPU sessions, a short-answer fine-tune, and an evaluation against the same 24 questions. A LoRA fine-tune of an existing open model is included as a fast baseline to compare against.

Nothing in this folder trains or downloads on your own machine. Every stage runs as a private Kaggle kernel, and `compute/package.py` only bundles source files into a kernel directory.

## Stages

| Stage | Runner | Kernel slug | Hardware | Internet | Reads from |
|---|---|---|---|---|---|
| corpus | `stage1_corpus.py` | `slm-160m-corpus` | CPU | yes | nothing |
| pretrain | `stage2_pretrain.py` | `slm-160m-pretrain-<k>` | T4 | no | corpus, and pretrain session k-1 when k > 1 |
| sft_data | `stage3_sft_data.py` | `slm-sft-data` | CPU | yes | nothing |
| sft | `stage4_sft.py` | `slm-160m-sft` | T4 | no | last pretrain session, sft_data |
| eval | `stage5_evaluate.py` | `slm-160m-eval` | T4 | no | sft, corpus |
| lora | `lora_baseline.py` | `slm-lora-baseline` | T4 | yes | sft_data |

- **corpus** streams FineWeb-Edu (`sample-10BT`, ODC-By) and TinyStories (CDLA-Sharing-1.0) at pinned revisions, keeps English text only, drops duplicates and anything matching the secret patterns in `cognition_slm.audit`, and holds out about 0.5% for evaluation. It writes `corpus/pretrain_train.jsonl`, `corpus/pretrain_eval.jsonl` and `corpus/corpus_manifest.json` with counts, byte totals, hashes and licenses.
- **pretrain** trains the `slm-160m` preset (160,721,679 parameters, 2,048-byte context) with next-byte loss on the corpus. Each session stops after 11 hours, saves `artifacts/slm-160m-pretrain.pt`, and records its measured seconds per step, the step it reached and held-out bits per byte in `pretrain_session_<k>.json`. Session k resumes from session k-1.
- **sft_data** builds short question and answer records: Dolly-15k (CC BY-SA 3.0) short answers, plus project-written facts, arithmetic, word pairs and "I don't know" refusals from `compute/short_facts.py`. Any row that overlaps a question in `data/simple_questions_holdout.json` or `data/everyday_eval.json` is dropped, so both scores stay honest. It writes `sft/sft_train.jsonl`, `sft/sft_eval.jsonl` and `sft/sft_manifest.json`.
- **sft** fine-tunes the last pretrain checkpoint on those records at learning rate 1e-4 and writes `artifacts/slm-160m-sft.pt` with `sft_report.json`.
- **eval** answers the 24 holdout questions and a fixed set of English probes greedily (temperature 0, at most 64 new tokens, stopping at a newline), scores them with `scripts/score_holdout.py`, measures held-out bits per byte, and writes `eval_report.json`. It also answers the 252 questions in `data/everyday_eval.json` and reports them under `everyday_eval_scores`, per category. The pass gate is holdout accuracy above the old 3/24.
- **lora** fine-tunes `HuggingFaceTB/SmolLM2-360M-Instruct` (Apache-2.0, revision `a10cc1512eabd3dde888204e902eca88bddb4951`) with LoRA on the same `sft_train.jsonl`, scores it on the same holdout, and writes `lora_report.json` plus the adapter.

## How much GPU time

These are estimates, and the pretrain session 1 report replaces them with a measurement.

The model has N = 160,721,679 parameters and the corpus target is D = 1.5 billion bytes, which is 1.5 billion tokens because the tokenizer reads one byte per token. Training costs about 6 × N × D FLOPs:

```
6 × 160,721,679 × 1.5e9 ≈ 1.45e18 FLOPs
```

Gradient checkpointing repeats the forward pass, so the real cost is closer to 8 × N × D ≈ 1.93e18 FLOPs. The earlier 500M runs reached about 16 TFLOP/s on a Kaggle T4 at fp16, which gives 1.93e18 / 16e12 ≈ 120,000 seconds, or about 33 GPU-hours.

The step-based view agrees. One optimizer step is batch 8 × accumulation 4 × 2,048 bytes = 65,536 tokens, so one pass over the corpus is 22,889 steps (`PRETRAIN_TOTAL_STEPS` in `stages.py`). At the estimated 5.5 seconds per step (`SECONDS_PER_STEP_ESTIMATE`), that is about 35 GPU-hours, split into 4 sessions of about 7,090 steps each. Attention over 2,048 positions adds FLOPs that 6 × N × D leaves out, and it costs relatively more on a 160M model than on the 500M one, so read 35 hours as a lower bound until session 1 reports its real step time.

| Stage | Estimated T4 hours |
|---|---|
| corpus, sft_data | 0 (CPU) |
| pretrain | about 35, in 4 sessions |
| sft | 1 to 2 |
| eval | under 1 |
| lora | 1 to 2 |

Kaggle gives a weekly GPU quota (check the current number on your account page), so pretraining will likely span more than one week.

## Run order

Every stage uses the same two commands: package, then push. Replace the owner if you are not `garyzhang11111`. Commands run from the repository root.

```bash
# 1. English corpus (CPU)
python3 compute/package.py --stage corpus --out /tmp/slm-corpus
kaggle kernels push -p /tmp/slm-corpus

# 2. Short-answer data (CPU); can run at the same time as step 1
python3 compute/package.py --stage sft_data --out /tmp/slm-sft-data
kaggle kernels push -p /tmp/slm-sft-data

# 3. Pretraining session 1, after the corpus kernel has finished
python3 compute/package.py --stage pretrain --session 1 --out /tmp/slm-pretrain-1
kaggle kernels push -p /tmp/slm-pretrain-1 --accelerator NvidiaTeslaT4 --timeout 43200
```

### Chaining pretrain sessions

Kaggle stops a kernel at 12 hours, so pretraining is split into sessions. Wait for session k-1 to finish, then package and push session k. The packager attaches the corpus and `slm-160m-pretrain-<k-1>` automatically, and the runner resumes from that session's checkpoint.

```bash
kaggle kernels status garyzhang11111/slm-160m-pretrain-1
python3 compute/package.py --stage pretrain --session 2 --out /tmp/slm-pretrain-2
kaggle kernels push -p /tmp/slm-pretrain-2 --accelerator NvidiaTeslaT4 --timeout 43200
```

After session 1, read `seconds_per_step` from its `pretrain_session_1.json`. If it differs from 5.5, update `SECONDS_PER_STEP_ESTIMATE` in `stages.py` so `pretrain_sessions()` gives the right session count. Stop when the session report shows the step reached equals `PRETRAIN_TOTAL_STEPS`.

### Fine-tune and evaluate

```bash
# --pretrain-session names the last finished pretrain session (default: the estimated count, 4)
python3 compute/package.py --stage sft --pretrain-session 4 --out /tmp/slm-sft
kaggle kernels push -p /tmp/slm-sft --accelerator NvidiaTeslaT4

python3 compute/package.py --stage eval --out /tmp/slm-eval
kaggle kernels push -p /tmp/slm-eval --accelerator NvidiaTeslaT4
```

### LoRA fast path

The LoRA baseline needs only the sft_data kernel, so it can run on day one while pretraining is still going. It tells you what an existing 360M instruct model reaches on the same 24 questions with the same data, which sets the bar the 160M model has to clear.

```bash
python3 compute/package.py --stage lora --out /tmp/slm-lora
kaggle kernels push -p /tmp/slm-lora --accelerator NvidiaTeslaT4
```

## Where outputs land

Each kernel writes under `/kaggle/working`, which Kaggle keeps as the kernel output. Later stages read those outputs from `/kaggle/input` when the packager lists the kernel in `kernel_sources`. To copy an output to your machine:

```bash
kaggle kernels output garyzhang11111/slm-160m-eval -p ./kaggle-output/eval
kaggle kernels output garyzhang11111/slm-160m-sft -p ./kaggle-output/sft
```

## Loading the result in Studio

Studio's default checkpoint is the 500M model, and it rejects any other size unless you name the file. Point it at the fine-tuned 160M checkpoint:

```bash
./launch-studio.command --checkpoint ./kaggle-output/sft/slm-project/artifacts/slm-160m-sft.pt
```

Check the actual path after downloading, since Kaggle keeps the directory layout under `/kaggle/working`.

## Files

- `stages.py` holds the stage table, the step budget constants and `planned_steps()`.
- `package.py` builds one kernel directory: `run.py` (a base64 zip of `pyproject.toml`, `src/cognition_slm`, `scripts/score_holdout.py`, `data/*.json*` and `compute/*.py`), `source-manifest.json` with SHA-256 hashes, and `kernel-metadata.json`.
- `run.py` unpacks to `/kaggle/working/slm-project`, puts `src` on the path, and runs the stage runner with `--session k` when one was given.
