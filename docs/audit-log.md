# audit log

| date | area | finding | where | status | commit or reason |
|---|---|---|---|---|---|
| 2026-09-28 | studio server and search | lora answers kept the stop text and reported `length` when a stop sequence ended partway through the last token, since only an exact suffix was stripped | `src/cognition_slm/lora_runtime.py:105` | fixed | 2e0eafb |
| 2026-09-28 | studio server and search | a last token that starts with the stop string and runs past it, like a newline plus spaces for stop `\n`, left that text behind; the verifier deferred how often smollm2 emits such a token | `src/cognition_slm/lora_runtime.py:105` | fixed | 2e0eafb, same cause as the row above; how often it happens needs the smollm2 tokenizer |
| 2026-09-28 | studio server and search | the readme said `pip install transformers peft`, which misses the `.venv-ui-py313` environment the launcher runs, so `--lora-adapter` failed with `ModuleNotFoundError` | `README.md:51` | fixed | 777da53 |
| 2026-09-28 | studio server and search | search treated `us` and `them` as topics that stem like `use` and `theme`, so "can you tell us?" returned a passage instead of abstaining | `src/cognition_slm/grounding.py:13` | fixed | 80c77b7 |
| 2026-09-28 | studio server and search | deeply nested json raises `RecursionError` on python 3.10 to 3.13, which the post handler did not catch, so the connection dropped with no response instead of a 400 | `src/cognition_slm/server.py:279` | fixed | 2f87d26 |
| 2026-09-28 | studio server and search | search ranked on casefolded words but highlighted ascii-only runs of the raw text, so `Straße` and `café` counted as matches without being highlighted whole | `src/cognition_slm/grounding.py:53` | fixed | 35a68fb |
| 2026-09-28 | studio server and search | casefolding `İstanbul` adds a combining dot that splits the word, so a question about istanbul abstains on a passage that names it | `src/cognition_slm/grounding.py:45` | deferred | confirmed by probe; over this round's five-fix limit |
| 2026-09-28 | studio server and search | a comment says the lora greedy cutoff matches generate.py, but lora uses 1e-3 and generate.py uses 1e-5 | `src/cognition_slm/lora_runtime.py:18` | deferred | confirmed; over this round's five-fix limit |
| 2026-09-28 | studio server and search | no http test sends a runtime `ValueError`, such as a context window overflow, through the 400 branch, so dropping that branch would still pass | `src/cognition_slm/server.py:292` | deferred | confirmed; over this round's five-fix limit |
