# Results

Collected from Kaggle kernel reports on 2026-09-30 15:32 by `compute/collect_results.py`. Regenerate it rather than editing by hand.

## slm-160m pretraining

| session | steps reached | s/step | eval bits/byte | status |
|---|---|---|---|---|
| 1 | 4570 | 8.72 | 1.289 | session_complete_resume_next |
| 2 | 9140 | 8.72 | 1.193 | session_complete_resume_next |

## LoRA adapter (SmolLM2-360M-Instruct)

- Status: complete_pending_manual_review
- Holdout exact: base 5/22, adapter 18/22
- Eval loss: 1.991 before training, best 1.540 at step 2500, final 1.644
- Adapter sha256: `137f58c6a048495ec296fbb7845b7b910ba6af40c2fe751707130bb80c126cb9`

## SmolLM2-360M-Instruct: LoRA vs base, re-scored with the current answer keys

These predictions came from a different adapter than the LoRA report above.

- everyday_eval: base 7/252 exact and 181/252 contains, LoRA 102/252 exact and 191/252 contains
- simple_questions: base 5/22 exact and 19/22 contains, LoRA 16/22 exact and 20/22 contains

Exact means the whole reply is an accepted answer, or sentences that each are one. Contains means an accepted answer appears as whole words in the reply, which credits full-sentence answers such as "Water freezes at 0 degrees Celsius." but can also credit a reply that names the answer and then contradicts it.

| category | base exact | LoRA exact | base contains | LoRA contains | scored |
|---|---|---|---|---|---|
| abstain | 0 | 2 | 1 | 16 | 20 |
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
- Holdout exact: base 4/22, adapter 21/22
- Eval loss: 1.953 before training, best 1.311 at step 1250, final 1.365
- Adapter sha256: `27cee7a602f26631e9883d6f23913bf9bb26e7d78e3eb14bd67bf20619058b09`

## SmolLM2-1.7B-Instruct: LoRA vs base, re-scored with the current answer keys

These predictions came from a different adapter than the LoRA report above.

- everyday_eval: base 8/252 exact and 213/252 contains, LoRA 149/252 exact and 234/252 contains
- simple_questions: base 4/22 exact and 20/22 contains, LoRA 18/22 exact and 21/22 contains

Exact means the whole reply is an accepted answer, or sentences that each are one. Contains means an accepted answer appears as whole words in the reply, which credits full-sentence answers such as "Water freezes at 0 degrees Celsius." but can also credit a reply that names the answer and then contradicts it.

| category | base exact | LoRA exact | base contains | LoRA contains | scored |
|---|---|---|---|---|---|
| abstain | 0 | 8 | 1 | 17 | 20 |
| arithmetic | 0 | 27 | 27 | 28 | 28 |
| colors_animals | 0 | 5 | 26 | 25 | 26 |
| counting_time | 0 | 12 | 24 | 25 | 26 |
| geography | 0 | 19 | 26 | 26 | 26 |
| opposites | 0 | 19 | 21 | 19 | 24 |
| plurals | 0 | 17 | 18 | 17 | 22 |
| reading | 0 | 14 | 27 | 29 | 30 |
| science | 0 | 4 | 23 | 24 | 25 |
| yes_no | 8 | 24 | 20 | 24 | 25 |

These rows ask the model to admit it doesn't know, so a person judges them and they are left out of the counts:

| id | prompt | LoRA answer |
|---|---|---|
| simple-v1-21 | What is my home address? I have not provided it. | I don't know your home address. You haven't provided it. |
| simple-v1-22 | I am thinking of a number. I have given no clues. What is it? | I don't know. You haven't told me. |

## Distilled answers

2129 rows kept from 2182 prompts (dropped: empty_or_invalid 53).

## slm-160m SFT

No SFT report yet.

## slm-160m eval

No eval report yet.
