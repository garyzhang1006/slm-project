# Compute pipeline for an English-answering SLM

The current 500M checkpoint answers 3 of 24 everyday questions in `data/simple_questions_holdout.json` correctly, mostly because it saw far too little plain English before it was taught to answer. This folder holds everything needed to fix that on Kaggle: a 1.5 GB English pretraining corpus, pretraining of the 160M preset across chained 11-hour GPU sessions, a short-answer fine-tune, and an evaluation against the same 24 questions. A LoRA fine-tune of an existing open model is included as a fast baseline to compare against.

Nothing in this folder trains or downloads on your own machine. Every stage runs as a private Kaggle kernel, and `compute/package.py` only bundles source files into a kernel directory.

## Stages

| Stage | Runner | Kernel slug | Hardware | Internet | Reads from |
|---|---|---|---|---|---|
| corpus | `stage1_corpus.py` | `slm-160m-corpus` | CPU | yes | nothing |
| pretrain | `stage2_pretrain.py` | `slm-160m-pretrain-<k>` | T4 | no | corpus, and pretrain session k-1 when k > 1 |
| sft_data | `stage3_sft_data.py` | `slm-sft-data` | CPU | yes | nothing |
| sft | `stage4_sft.py` | `slm-160m-sft` | T4 | no | last pretrain session, sft_data, distill_data |
| eval | `stage5_evaluate.py` | `slm-160m-eval` | T4 | no | sft, corpus |
| lora | `lora_baseline.py` | `slm-lora-baseline` | T4 | yes | sft_data |
| distill_data | `distill_data.py` | `slm-distill-data` | T4 | yes | lora |
| lora_eval | `lora_eval.py` | `slm-lora-eval` | T4 | yes | lora |
| lora_1b7 | `lora_baseline.py --model 1.7b` | `slm-lora-1b7` | T4 | yes | sft_data |
| lora_1b7_eval | `lora_eval.py` | `slm-lora-1b7-eval` | T4 | yes | lora_1b7 |

- **corpus** streams FineWeb-Edu (`sample-10BT`, ODC-By) and TinyStories (CDLA-Sharing-1.0) at pinned revisions, keeps English text only, drops duplicates, anything matching the secret patterns in `cognition_slm.audit` and any document containing a question sentence from `data/simple_questions_holdout.json` or `data/everyday_eval.json`, and holds out about 0.5% for evaluation. It writes `corpus/pretrain_train.jsonl`, `corpus/pretrain_eval.jsonl` and `corpus/corpus_manifest.json` with counts, byte totals, hashes and licenses.
- **pretrain** trains the `slm-160m` preset (160,721,679 parameters, 2,048-byte context) with next-byte loss on the corpus. Each session stops after 11 hours, saves `artifacts/slm-160m-pretrain.pt`, and records its measured seconds per step, the step it reached and held-out bits per byte in `pretrain_session_<k>.json`. Session k resumes from session k-1.
- **sft_data** builds short question and answer records: Dolly-15k (CC BY-SA 3.0) short answers, English first-turn pairs from OpenAssistant oasst1 (Apache-2.0) with the best-ranked reply, plus project-written rows from `compute/short_facts.py`: facts, arithmetic, word pairs, yes or no questions, short reading passages (some that leave the answer out, taught to say so), science, animal and color facts, unit and calendar counts, and "I don't know" refusals. A bare question gets the short answer, and a full sentence only when the prompt asks for one. Dolly questions answered with one short sentence get the same "Answer in a full sentence." cue appended, so no bare question is trained to a sentence. Any row that overlaps a question in `data/simple_questions_holdout.json` or `data/everyday_eval.json` is dropped, so both scores stay honest, and the project rows also leave out the facts those files ask. It writes `sft/sft_train.jsonl`, `sft/sft_eval.jsonl` and `sft/sft_manifest.json`.
- **sft** fine-tunes the last pretrain checkpoint on those records, plus the short answers from distill_data when that kernel is attached, at learning rate 1e-4 and writes `artifacts/slm-160m-sft.pt` with `sft_report.json`.
- **eval** answers the 24 holdout questions and a fixed set of English probes greedily (temperature 0, at most 64 new tokens, stopping at a newline), scores them with `scripts/score_holdout.py`, measures held-out bits per byte, and writes `eval_report.json`. It also answers the 252 questions in `data/everyday_eval.json` and reports them under `everyday_eval_scores`, per category. The pass gate is holdout accuracy above the old 3/24.
- **lora** fine-tunes `HuggingFaceTB/SmolLM2-360M-Instruct` (Apache-2.0, revision `a10cc1512eabd3dde888204e902eca88bddb4951`) with LoRA on the same `sft_train.jsonl`, drops any row that copies a question from either eval file again, scores it on the same holdout, and writes `lora_report.json` plus the adapter. It trains two epochs, about 1.3 hours at the measured 1.7 seconds a step, and keeps the checkpoint with the lowest held-out loss. A five epoch run did best near epoch 2 and only got worse after it.
- **distill_data** has the lora adapter write short answers to Dolly questions whose human answers were too long for sft_data, screens them like stage 3 does, and writes `distill_train.jsonl` with `distill_manifest.json`, which records the adapter it used.
- **lora_eval** scores the base SmolLM2 and the lora adapter on the 252 everyday questions and the holdout, and writes `lora_eval_report.json` with the `adapter_sha256` it scored.
- **lora_1b7** runs the same runner on `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0, revision `31b70e2e869a7173562077fd711b654946d38674`). The fp32 weights take about 6.8 GB of the T4's 16 GB, so it trains one epoch on micro-batches of 4 with gradient checkpointing, about 3.2 hours at the measured 9 seconds a step. A three epoch run did best near the end of epoch 1. Every adapter folder now holds `base_model.json`, which is how `lora_eval`, `distill_data` and Studio know which base model to load.
- **lora_1b7_eval** scores the 1.7B base model and its adapter on the everyday questions and the holdout, like `lora_eval` does for the 360M model.

## How much GPU time

These are estimates, and the pretrain session 1 report replaces them with a measurement.

The model has N = 160,721,679 parameters and the corpus target is D = 1.5 billion bytes, which is 1.5 billion tokens because the tokenizer reads one byte per token. Training costs about 6 × N × D FLOPs:

```
6 × 160,721,679 × 1.5e9 ≈ 1.45e18 FLOPs
```

Gradient checkpointing repeats the forward pass, so the real cost is closer to 8 × N × D ≈ 1.93e18 FLOPs. The earlier 500M runs reached about 16 TFLOP/s on a Kaggle T4 at fp16, which gives 1.93e18 / 16e12 ≈ 120,000 seconds, or about 33 GPU-hours.

The step-based view agrees. One optimizer step is batch 8 × accumulation 4 × 2,048 bytes = 65,536 tokens, so one pass over the corpus is 22,889 steps (`PRETRAIN_TOTAL_STEPS` in `stages.py`). Session 1 measured 8.72 seconds per step (`SECONDS_PER_STEP_ESTIMATE`), well above the 5.5 the FLOP count suggested, because attention over 2,048 positions adds FLOPs that 6 × N × D leaves out and costs relatively more on a 160M model. At 8.72 seconds a full pass is about 55 GPU-hours. `pretrain_sessions()` plans 6 sessions of about 4,470 steps, and sessions 1 and 2 each ran 4,570 (`compute/RESULTS.md`), so at that pace five sessions reach 22,850 steps and a short sixth session runs the last 39.

| Stage | Estimated T4 hours |
|---|---|
| corpus, sft_data | 0 (CPU) |
| pretrain | about 55, in 6 sessions (measured speed) |
| sft | 1 to 2 |
| eval | under 1 |
| lora | about 1.3 of training (measured), plus setup and scoring |
| distill_data | about 1 |
| lora_eval | about 1 |
| lora_1b7 | about 3.2 of training (measured), plus setup and scoring |
| lora_1b7_eval | about 1 |

Kaggle gives a weekly GPU quota (check the current number on your account page), so pretraining will likely span more than one week.

## Run order

Once the corpus exists, `python3 compute/run_pipeline.py --owner YOUR_KAGGLE_USERNAME --watch` does the rest of the main chain for you: it checks every 30 minutes, pushes the next pretrain session when the last one finishes, then distill_data, sft and eval, and waits instead of pushing when the weekly GPU quota cannot cover the next stage. It also runs the 1.7B chain (lora_1b7, then lora_1b7_eval), which only gets GPU quota the other chains leave over. It drives the LoRA chain in the same loop: sft_data, then the `lora` adapter, then `lora_eval` and `distill_data`. Each follow-up records the `adapter_sha256` it used, so when the adapter is retrained both rerun, and sft waits for distilled answers from the current adapter. Both adapters record the sha256 of the sft_data train file they used, so pushing sft_data again retrains them. A push on one chain counts against the quota the other chain sees in that round. Add `--dry-run` to see the decision without pushing anything. `python3 compute/collect_results.py --owner YOUR_KAGGLE_USERNAME` downloads only the JSON reports and rewrites [RESULTS.md](RESULTS.md): bits per byte for each pretrain session, the LoRA adapter against the base model per category (re-scored with the current answer keys in `data/`), distillation counts, and the SFT and eval numbers. The manual commands below still work.

Every stage uses the same two commands: package, then push. `package.py` names every kernel and its attachments after `--owner`, which defaults to `garyzhang11111`, so on another account add `--owner YOUR_KAGGLE_USERNAME` to each `package.py` command below and use that username in `kaggle kernels status`. Commands run from the repository root.

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

After each session, read `seconds_per_step` from its `pretrain_session_k.json`. If it drifts from 8.72, update `SECONDS_PER_STEP_ESTIMATE` in `stages.py` so `pretrain_sessions()` gives the right session count. Stop when the session report shows the step reached equals `PRETRAIN_TOTAL_STEPS`.

### Fine-tune and evaluate

Before sft, `distill_data` has the LoRA-tuned SmolLM2 (from the `lora` stage) answer up to 6,000 Dolly questions whose human answers were too long for stage 3, and stage 4 appends those short answers to `sft_train`. It needs about an hour of T4 time and a finished `slm-lora-baseline` kernel.

```bash
python3 compute/package.py --stage distill_data --out /tmp/slm-distill
kaggle kernels push -p /tmp/slm-distill
# --pretrain-session K names the last pretrain session whose report says status complete (default: the estimated count, 6)
python3 compute/package.py --stage sft --pretrain-session K --out /tmp/slm-sft
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
