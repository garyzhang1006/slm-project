# Data and model audit

Generated from the v0.3.0 source tree on 2026-08-18.

## Data audit

Status: **PASS**

- `data/demo.jsonl`: 45 records, project-authored synthetic data, `CC0-1.0`.
- `data/eval.jsonl`: 16 held-out records, project-authored synthetic data, `CC0-1.0`.
- Train/eval ID overlap: none.
- Secret-pattern errors: none.
- Hidden-reasoning fields: none.
- Prompt-injection-like warnings: none.
- Dataset hashes: recorded in `data/MANIFEST.json`.
- Context fit: all encoded records retain answer tokens; maximum encoded length is 256.

## Verification

- `PYTHONPATH=src .venv/bin/pytest -q`: 21 passed.
- `PYTHONPATH=src .venv/bin/python -m compileall -q src tests scripts`: passed.
- `PYTHONPATH=src .venv/bin/python -m cognition_slm.audit --train data/demo.jsonl --eval data/eval.jsonl`: passed.
- Modern dry run: 45 records, maximum 256 tokens, 256-token block, status `dry-run`.
- Matched CPU training: legacy and modern models, 60 steps, seed 7, batch size 4, same optimizer settings.
- Multi-candidate generation: modern checkpoint, three candidates, completed without executing output.

## Matched smoke benchmark

Command:

```bash
PYTHONPATH=src .venv/bin/python -m cognition_slm.benchmark \
  --model legacy=artifacts/benchmark-legacy-v2.pt \
  --model modern=artifacts/benchmark-modern-v2.pt \
  --data data/eval.jsonl \
  --max-new-tokens 48
```

| Metric | Legacy | Modern | Modern minus legacy |
| --- | ---: | ---: | ---: |
| Parameters | 497,678 | 560,910 | 63,232 |
| Task accuracy | 0.3750 | 0.5625 | +0.1875 |
| Error accuracy | 0.5625 | 0.7500 | +0.1875 |
| Confidence-bucket accuracy | 0.8750 | 0.8750 | 0.0000 |
| Exact-match accuracy | 0.0000 | 0.0000 | 0.0000 |
| Python syntax validity | 0.8750 | 1.0000 | +0.1250 |
| Required-symbol recall | 0.0000 | 0.0000 | 0.0000 |
| Static code score | 0.4375 | 0.5000 | +0.0625 |

Training validation loss at step 60 was `4.5671` for legacy and `4.5790` for modern. These are smoke results on 45 synthetic training records and 16 held-out records. They do not establish general coding competence. Exact match and required-symbol recall remained zero.

## Scope limits

- The model studies observable outputs and auxiliary labels, not hidden chain-of-thought or consciousness.
- Static parsing and compilation do not prove runtime correctness.
- Generated code is untrusted text and must not be executed outside a sandbox with resource limits.
- No external training corpus is bundled or downloaded by the repository.
