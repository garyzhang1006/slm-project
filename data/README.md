# Data contract

Each non-empty JSONL line must contain:

```json
{
  "id": "stable-unique-id",
  "prompt": "coding task",
  "answer": "inspectable answer or code",
  "task_type": "code_generation",
  "confidence": 0.8,
  "error_category": "none",
  "source": "who or what produced record",
  "license": "license or permission basis"
}
```

Allowed task types and error categories are defined in `src/cognition_slm/config.py`. The loader rejects duplicate IDs, unknown labels, missing provenance, out-of-range confidence, and fields intended to hold hidden reasoning.

The committed demo records are synthetic and project-authored. They are marked `CC0-1.0` for this mini project. For real training, keep source and license metadata per record, preserve an isolated evaluation split, and run:

```bash
python -m cognition_slm.audit --train data/demo.jsonl --eval data/eval.jsonl --report reports/audit.md
```

Current snapshot:

- `data/demo.jsonl`: 31 training records across code generation, debugging, explanation, algorithm reasoning, and metacognitive review.
- `data/eval.jsonl`: 8 held-out records covering the same task families with different prompts and IDs.
- `data/MANIFEST.json`: record counts, provenance, licenses, and SHA-256 hashes for both files.

Do not pipe private prompts, API keys, repository secrets, or unlicensed scraped code into the dataset.
