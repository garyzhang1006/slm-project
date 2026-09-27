# Results

Collected from Kaggle kernel reports on 2026-09-26 23:58 by `compute/collect_results.py`. Regenerate it rather than editing by hand.

## slm-160m pretraining

| session | steps reached | s/step | eval bits/byte | status |
|---|---|---|---|---|
| 1 | 4570 | 8.72 | 1.289 | session_complete_resume_next |
| 2 | 9140 | 8.72 | 1.193 | session_complete_resume_next |

## LoRA adapter (SmolLM2-360M-Instruct)

- Status: complete_pending_manual_review
- Holdout exact: base 5/22, adapter 16/22
- Eval loss: 1.960 before training, best 1.551 at step 2244, final 1.551
- Adapter sha256: `3e0bc9234ac12c1619db804b14abc792aced83f74d918582067b0f80e6132767`

## SmolLM2-360M-Instruct: LoRA vs base, re-scored with the current answer keys

- everyday_eval: base 7/252 exact and 181/252 contains, LoRA 100/252 exact and 191/252 contains
- simple_questions: base 5/22 exact and 19/22 contains, LoRA 16/22 exact and 20/22 contains

Exact means the whole reply is an accepted answer. Contains means an accepted answer appears as whole words in the reply, which credits full-sentence answers such as "Water freezes at 0 degrees Celsius." but can also credit a reply that names the answer and then contradicts it.

| category | base exact | LoRA exact | base contains | LoRA contains | scored |
|---|---|---|---|---|---|
| abstain | 0 | 0 | 1 | 16 | 20 |
| arithmetic | 0 | 23 | 20 | 24 | 28 |
| colors_animals | 0 | 2 | 17 | 14 | 26 |
| counting_time | 0 | 10 | 22 | 22 | 26 |
| geography | 1 | 17 | 25 | 24 | 26 |
| opposites | 3 | 18 | 20 | 18 | 24 |
| plurals | 0 | 10 | 17 | 17 | 22 |
| reading | 0 | 3 | 23 | 22 | 30 |
| science | 0 | 3 | 22 | 20 | 25 |
| yes_no | 3 | 14 | 14 | 14 | 25 |

These rows ask the model to admit it doesn't know, so a person judges them and they are left out of the counts:

| id | prompt | LoRA answer |
|---|---|---|
| simple-v1-21 | What is my home address? I have not provided it. | I don't know your home address. You haven't provided it. |
| simple-v1-22 | I am thinking of a number. I have given no clues. What is it? | I don't know what you are thinking of. You haven't given me any clues. |

## LoRA adapter (SmolLM2-1.7B-Instruct)

- Status: complete_pending_manual_review
- Holdout exact: base 4/22, adapter 18/22
- Eval loss: 1.905 before training, best 1.332 at step 1122, final 1.332
- Adapter sha256: `b31d7bd5708f73c9cf7d77f862131bed65ff3495d592cf7ef245df52e8c02bab`

## SmolLM2-1.7B-Instruct: LoRA vs base, re-scored with the current answer keys

No finished lora_eval report yet.

## Distilled answers

2129 rows kept from 2182 prompts (dropped: empty_or_invalid 53).

## slm-160m SFT

No SFT report yet.

## slm-160m eval

No eval report yet.
