# Results

Collected from Kaggle kernel reports on 2026-09-26 15:35 by `compute/collect_results.py`. Regenerate it rather than editing by hand.

## slm-160m pretraining

| session | steps reached | s/step | eval bits/byte | status |
|---|---|---|---|---|
| 1 | 4570 | 8.72 | 1.289 | session_complete_resume_next |

## LoRA adapter (SmolLM2-360M-Instruct)

No LoRA report yet.

## LoRA vs base, re-scored with the current answer keys

- everyday_eval: base 7/252, LoRA 87/252 exact
- simple_questions: base 5/22, LoRA 18/22 exact

| category | base | LoRA | scored |
|---|---|---|---|
| abstain | 0 | 0 | 20 |
| arithmetic | 0 | 21 | 28 |
| colors_animals | 0 | 2 | 26 |
| counting_time | 0 | 7 | 26 |
| geography | 1 | 14 | 26 |
| opposites | 3 | 18 | 24 |
| plurals | 0 | 10 | 22 |
| reading | 0 | 5 | 30 |
| science | 0 | 2 | 25 |
| yes_no | 3 | 8 | 25 |

## Distilled answers

No distill_data manifest yet.

## slm-160m SFT

No SFT report yet.

## slm-160m eval

No eval report yet.
