# Results

Collected from Kaggle kernel reports on 2026-10-05 18:40 by `compute/collect_results.py`. Regenerate it rather than editing by hand.

## slm-160m pretraining

| session | steps reached | s/step | eval bits/byte | status |
|---|---|---|---|---|
| 1 | 4570 | 8.72 | 1.289 | session_complete_resume_next |
| 2 | 9140 | 8.72 | 1.193 | session_complete_resume_next |
| 3 | 13319 | 9.54 | 1.139 | session_complete_resume_next |
| 4 | 17662 | 9.18 | 1.103 | session_complete_resume_next |

## LoRA adapter (SmolLM2-360M-Instruct)

- Status: complete_pending_manual_review
- Holdout exact: base 5/22, adapter 19/22
- Eval loss: 2.004 before training, best 1.518 at step 2652, final 1.518
- Adapter sha256: `9eda9e89e953ccee04ddb96dee87de8fff28ff1525f621434cdc9f310d27a10d`

## SmolLM2-360M-Instruct: LoRA vs base, re-scored with the current answer keys

- everyday_eval: base 7/252 exact and 181/252 contains, LoRA 177/252 exact and 198/252 contains
- simple_questions: base 5/22 exact and 19/22 contains, LoRA 19/22 exact and 19/22 contains

Exact means the whole reply is an accepted answer, or sentences that each are one. Contains means an accepted answer appears as whole words in the reply, which credits full-sentence answers such as "Water freezes at 0 degrees Celsius." but can also credit a reply that names the answer and then contradicts it.

| category | base exact | LoRA exact | base contains | LoRA contains | scored |
|---|---|---|---|---|---|
| abstain | 0 | 6 | 1 | 19 | 20 |
| arithmetic | 0 | 23 | 20 | 23 | 28 |
| colors_animals | 0 | 13 | 17 | 14 | 26 |
| counting_time | 0 | 21 | 22 | 23 | 26 |
| geography | 1 | 22 | 25 | 24 | 26 |
| opposites | 3 | 19 | 20 | 19 | 24 |
| plurals | 0 | 20 | 17 | 20 | 22 |
| reading | 0 | 20 | 23 | 20 | 30 |
| science | 0 | 15 | 22 | 18 | 25 |
| yes_no | 3 | 18 | 14 | 18 | 25 |

These rows ask the model to admit it doesn't know, so a person judges them and they are left out of the counts:

| id | prompt | LoRA answer |
|---|---|---|
| simple-v1-21 | What is my home address? I have not provided it. | I don't know your home address. You haven't told me. |
| simple-v1-22 | I am thinking of a number. I have given no clues. What is it? | I don't know. You haven't told me. |

## LoRA adapter (SmolLM2-1.7B-Instruct)

- Status: complete_pending_manual_review
- Holdout exact: base 4/22, adapter 21/22
- Eval loss: 1.975 before training, best 1.296 at step 1326, final 1.296
- Adapter sha256: `445e381edec482033e9b46cc8fac05edb392e2711456906bd3071175b1e8f13d`

## SmolLM2-1.7B-Instruct: LoRA vs base, re-scored with the current answer keys

- everyday_eval: base 8/252 exact and 213/252 contains, LoRA 208/252 exact and 237/252 contains
- simple_questions: base 4/22 exact and 20/22 contains, LoRA 21/22 exact and 21/22 contains

Exact means the whole reply is an accepted answer, or sentences that each are one. Contains means an accepted answer appears as whole words in the reply, which credits full-sentence answers such as "Water freezes at 0 degrees Celsius." but can also credit a reply that names the answer and then contradicts it.

| category | base exact | LoRA exact | base contains | LoRA contains | scored |
|---|---|---|---|---|---|
| abstain | 0 | 3 | 1 | 19 | 20 |
| arithmetic | 0 | 28 | 27 | 28 | 28 |
| colors_animals | 0 | 24 | 26 | 25 | 26 |
| counting_time | 0 | 23 | 24 | 24 | 26 |
| geography | 0 | 23 | 26 | 26 | 26 |
| opposites | 0 | 17 | 21 | 17 | 24 |
| plurals | 0 | 20 | 18 | 20 | 22 |
| reading | 0 | 29 | 27 | 30 | 30 |
| science | 0 | 17 | 23 | 24 | 25 |
| yes_no | 8 | 24 | 20 | 24 | 25 |

These rows ask the model to admit it doesn't know, so a person judges them and they are left out of the counts:

| id | prompt | LoRA answer |
|---|---|---|
| simple-v1-21 | What is my home address? I have not provided it. | I don't know your home address. You haven't provided it. |
| simple-v1-22 | I am thinking of a number. I have given no clues. What is it? | I don't know. You haven't told me. |

## Distilled answers

1736 rows kept from 2182 prompts (dropped: cut_off 222, empty_or_invalid 112, repetitive 112).

## slm-160m SFT

No SFT report yet.

## slm-160m eval

No eval report yet.
