# Context-care audit

Date: 2026-08-28

Scope: `cognition_slm.context_therapy`, its public package exports, the JSON fixture, and source-level CLI behavior.

Status: **PASS in clean Python 3.11 environment; default local Python 3.14 environment remains incompatible.**

## Observed passes

- `PYTHONPATH=src .venv/bin/python tests/test_context_therapy.py`: 15 context tests passed.
- Clean Python 3.11 `pytest -q`: 36 tests passed in 0.91 seconds, including model, generation, evaluation, benchmark, and training tests.
- `PYTHONPATH=src .venv/bin/python tests/test_code_eval.py`: 5 tests passed.
- `PYTHONPATH=src .venv/bin/python tests/test_data_and_audit.py`: 4 tests passed.
- `PYTHONPATH=src .venv/bin/python tests/test_tokenizer.py`: 2 tests passed.
- `uv pip install --python <clean-python-3.11> -e ".[dev]"`: built and installed `cognition-slm==0.4.0` with 17 compatible packages.
- `uv pip check --python <clean-python-3.11>`: all installed packages compatible.
- Installed `cognition-slm-context-therapist` console script: returned the expected conflicted fixture report.
- Context CLI fixture contract: returned `conflicted`, detected contradictory directives, instruction drift, and unsupported claims, and represented all 6 input indices in the handoff.
- Module CLI with `-W error`: passed without import warnings after lazy package exports were added.
- `PYTHONPATH=src .venv/bin/python -m compileall -q src tests scripts`: passed.
- `git diff --check`: passed after each code slice.
- Secret-like text was absent from serialized focus and handoff excerpts in the redaction test.
- Existing checkpoint generation smoke: exited 0 on CPU; output quality remains limited by the tiny demo model.

## Model-path limitation

The repository's default `.venv` uses Python 3.14. Importing `torch` there reaches NumPy and fails with:

```text
RecursionError: maximum recursion depth exceeded
```

The traceback points into `numpy.__getattr__` while importing `numpy.exceptions`. `pip install ".[dev]"`, `pip show`, and `pip check` also stalled in that environment. Verification recovered in a temporary Python 3.11 environment, which built the package, installed the console script, passed all 36 tests, and passed dependency checks.

## Interpretation

The context controller is a deterministic text-level sidecar. It does not prove coding ability, consciousness, private chain-of-thought, or recovery of every fact after compression. The existing tiny-model benchmark and its limits remain documented separately in [`reports/audit.md`](audit.md).
