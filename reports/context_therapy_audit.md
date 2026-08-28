# Context-care audit

Date: 2026-08-28

Scope: `cognition_slm.context_therapy`, its public package exports, the JSON fixture, and source-level CLI behavior.

Status: **PASS for context-care source path; PARTIAL for full SLM environment verification.**

## Observed passes

- `PYTHONPATH=src .venv/bin/python tests/test_context_therapy.py`: 13 tests passed.
- `PYTHONPATH=src .venv/bin/python tests/test_code_eval.py`: 5 tests passed.
- `PYTHONPATH=src .venv/bin/python tests/test_data_and_audit.py`: 4 tests passed.
- `PYTHONPATH=src .venv/bin/python tests/test_tokenizer.py`: 2 tests passed.
- Context CLI fixture contract: returned `conflicted`, detected contradictory directives, instruction drift, and unsupported claims, and represented all 6 input indices in the handoff.
- Module CLI with `-W error`: passed without import warnings after lazy package exports were added.
- `PYTHONPATH=src .venv/bin/python -m compileall -q src tests scripts`: passed.
- `git diff --check`: passed after each code slice.
- Secret-like text was absent from serialized focus and handoff excerpts in the redaction test.

## Model-path limitation

The model-dependent test files were not re-run in this local environment after the context-care changes. Importing `torch` reaches NumPy and fails with:

```text
RecursionError: maximum recursion depth exceeded
```

The traceback points into `numpy.__getattr__` while importing `numpy.exceptions`. Local `pip install ".[dev]"`, `pip show`, and `pip check` also stalled without output, so the installed console script was not claimed as locally verified. Source-module execution passed. A clean CI environment must verify package installation, model tests, and the generated console script.

## Interpretation

The context controller is a deterministic text-level sidecar. It does not prove coding ability, consciousness, private chain-of-thought, or recovery of every fact after compression. The existing tiny-model benchmark and its limits remain documented separately in [`reports/audit.md`](audit.md).
