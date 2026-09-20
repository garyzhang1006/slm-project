# Elementary checkpoint review

Reviewed September 20, 2026, from the completed Kaggle run
`garyzhang11111/slm-500m-elementary-qa`. Reviewing its saved report started no
new training or inference. These results describe that checkpoint, not Studio's
older default weights.

The report records 499,524,075 parameters and a 2,048-byte-token context.
Training completed 2,000 iterations in 4,003.97 seconds, with 1,997 optimizer
updates and three skipped updates. The run processed 288,025 supervised tokens
on updates that were applied.

## Observed answers

On 64 held-out combinations of the training templates, greedy exact match rose
from 0/64 to 30/64:

| Task | Correct / evaluated |
| --- | --- |
| Arithmetic | 3/37 |
| Attributes | 6/6 |
| Locations | 5/5 |
| Giving and receiving | 10/10 |
| Alphabetical sorting | 2/2 |
| Copying | 3/3 |
| Grammar | 1/1 |

These small samples test combinations within familiar templates. They do not
establish broad English understanding. Across the separate 24-question audit,
manual review found three fully correct answers: identifying Eva as the pencil
recipient, correcting “She have a bicycle.”, and copying “soft yellow scarf”.
The grammar completion about hungry children included “are” but added unrelated
text, so it was not counted as fully correct.

Failures remained substantial: the model answered “14” for days in a week,
“15” for months in a year, and “44” for Python's `2 * 6`. It also failed to
clearly acknowledge missing information in the unknown-answer questions.
No reduction in general hallucination rate or reliable coding ability has been
demonstrated. The separate audit has been inspected during earlier development;
it is a recurring diagnostic, not a fresh blind benchmark.

## Checkpoint identity

- Output: `artifacts/slm-500m-elementary.pt`
- SHA-256: `2ad7dec394b666452707c0cd231c3465dbd60c9f1df793cfda50372fa1991795`
- Parent SHA-256: `ae2a1db39c1cd4722950b844d85a6cfd4e03add4a6c234f2488dc51abbbcc983`
- Source report: `slm-project/elementary_report.json` in the completed run outputs.

Studio's default checkpoint remains unchanged. This model learned useful narrow
patterns, but the reviewed answers do not justify presenting it as a working
general question-answering assistant.
