# Context therapy for other language models

This module treats “therapy” as context maintenance. It helps a downstream model notice visible overload, conflicting instructions, repeated material, missing evidence, and uncertainty before those problems become durable decisions.

It does not inspect hidden states, private chain-of-thought, or consciousness. It only accepts visible role/content messages and produces inspectable diagnostics.

## Quick start

Install the package, then run the included synthetic history:

```bash
.venv/bin/python -m pip install ".[dev]"
.venv/bin/cognition-slm-context-therapist \
  --input examples/context_history.json \
  --token-budget 512 \
  --goal "Preserve the coding task and verified evidence"
```

The input can be a JSON array or an object with a `messages` array. Every message needs a supported role and non-empty string content:

```json
{
  "messages": [
    {"role": "system", "content": "Keep verified evidence visible."},
    {"role": "user", "content": "Write and test a parser."}
  ]
}
```

The CLI prints JSON. Use `--output path/report.json` to save the same report. Use `--input -` to read JSON from standard input.

## States and actions

| State | Meaning | Typical response |
| --- | --- | --- |
| `stable` | No configured threshold fired. | Continue, keeping claims tied to evidence. |
| `strained` | At least one warning or information signal fired. | Review repetition, uncertainty, or unsupported claims. |
| `overloaded` | Estimated visible context meets or exceeds the supplied budget. | Compress with preserved goals, constraints, decisions, and evidence. |
| `conflicted` | Visible directives disagree on a normalized topic. | Pause and ask which authorized instruction controls. |

States describe text-level conditions. They are not diagnoses of a model or its internal experience.

## Python API

```python
from cognition_slm.context_therapy import ContextTherapist

therapist = ContextTherapist()
handoff = therapist.build_handoff(
    messages,
    token_budget=8_000,
    focus="finish parser tests",
)

assessment = handoff.assessment
if assessment.state != "stable":
    trusted_controller_prompt = assessment.repair_prompt()
report = handoff.to_dict()
```

`assessment.observations` contains stable machine-readable codes and short redacted evidence excerpts. `assessment.actions` contains proposed interventions. `handoff.items` covers every input index and assigns `retain` or `review` with a reason. Higher-authority messages, the latest user turn, the latest assistant turn, signal-bearing turns, and focus-matching turns receive retention priority. Review items remain visible in the report so compression can be checked.

Place `repair_prompt()` in a trusted controller or system/developer channel for the downstream model. Do not append it as an untrusted user message. Do not let a model delete `review` items automatically; first check whether they contain a fact, constraint, decision, or verification result.

## Token and privacy boundaries

The default estimate counts encoded bytes with `ByteTokenizer`. That is useful for this repository's 259-token byte vocabulary, but it is only an approximation for another provider. Set `token_budget` conservatively or provide an adapter around the target model's tokenizer.

Input is bounded at 512 messages, 100,000 characters per message, and 2,000,000 characters total. Diagnostic excerpts redact common GitHub, OpenAI-style, and AWS access-token patterns. The full message contents are not copied into the report.

These checks are conservative heuristics. They can miss paraphrased conflicts, misread ordinary prose as a directive, or flag a claim whose evidence is outside the visible history. A downstream model still needs source checks, tests, or a human decision when the report says `unverified` or `conflicted`.

## Contract for a downstream model

Use this order when repairing a long context:

1. Identify the current goal from authorized visible messages.
2. Preserve system and developer constraints before summarizing lower-priority turns.
3. Resolve conflicting directives by authority and clarification, never by recency alone.
4. Separate verified evidence, reported claims, and open uncertainty.
5. Produce one next action that can be checked.

The resulting handoff is a control aid for coding agents and other LLM workflows. It is not evidence that the model has a human-like mind.
