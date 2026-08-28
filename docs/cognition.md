# Observable cognition, not hidden thought

This project uses “cognition” as shorthand for measurable behavior around coding tasks:

- task type: generation, debugging, explanation, algorithm reasoning, or review;
- error category: none, syntax, logic, hallucination, incomplete, or unsafe;
- confidence bucket: low, medium, or high based on an annotation in `[0, 1]`;
- held-out answer behavior: exact-match generation in the tiny demo.

The language-model head predicts the next byte. Three auxiliary heads predict the labels above from the prompt-boundary contextual representation. This gives a compact way to ask whether the model's stated confidence or error tag tracks held-out behavior.

The training path pools auxiliary heads at the end of the encoded prompt, before answer tokens begin. Evaluation uses the same prompt-only position. The language-model receives prompt plus answer context, but its loss is computed only on answer-token targets. Generation training and behavior-label measurement therefore use separate information boundaries.

For Python generation and debugging tasks, evaluation also parses and compiles generated text without executing it. It reports syntax validity, required function-name recall, and a narrow static score. These metrics measure output form, not runtime behavior.

## Context care for long conversations

`cognition_slm.context_therapy` is a separate, deterministic controller for visible message history. It is intended to sit beside another LLM when a conversation becomes long or internally inconsistent. It can identify:

- approximate token pressure using this repository's byte tokenizer;
- repeated visible turns that can waste context budget;
- contradictory directives after role-aware normalization;
- text that tries to discard earlier instructions;
- assistant success claims without nearby evidence markers;
- explicit uncertainty that should remain visible during compression.

The controller never receives or emits private chain-of-thought. Its repair prompt tells a downstream model to return a compact handoff with current goal, hard constraints, decisions, open questions, evidence status, and next action. The handoff packet lists every input index, preserves higher-authority and current turns, and labels lower-priority turns for human or model review. Review is a suggestion, not an instruction to delete context.

Example integration:

```python
from cognition_slm.context_therapy import ContextTherapist

handoff = ContextTherapist().build_handoff(
    messages,
    token_budget=8_000,
    focus="finish parser tests",
)
if handoff.assessment.state != "stable":
    controller_prompt = handoff.assessment.repair_prompt()
    # Send controller_prompt through a trusted system/developer channel.
report = handoff.to_dict()
```

The byte-token estimate is exact for this project's tokenizer, not for an arbitrary provider. A production adapter should measure the target model's own tokenization before deciding that context is safe. The controller also does not prove that a model is conscious, has a private monologue, or has recovered every fact from a compressed history.

## Measurement protocol

Run the audit first, then train with a separate evaluation file:

```bash
python -m cognition_slm.audit --train data/demo.jsonl --eval data/eval.jsonl
python -m cognition_slm.train \
  --data data/demo.jsonl \
  --eval-data data/eval.jsonl \
  --out artifacts/demo.pt \
  --steps 60 \
  --eval-every 20
python -m cognition_slm.evaluate \
  --checkpoint artifacts/demo.pt \
  --data data/eval.jsonl
python -m cognition_slm.benchmark \
  --model legacy=artifacts/legacy.pt \
  --model modern=artifacts/modern.pt \
  --data data/eval.jsonl
```

Generation can sample multiple candidates with `--num-candidates`. The selector uses length-normalized model log probability and an optional Python syntax bonus. It does not run generated code.

The evaluator reports accuracy, macro-F1, confusion matrices, and expected calibration error for task, error, and confidence-bucket heads. Expected calibration error groups predictions by maximum class probability and compares confidence with correctness. These are output-behavior measurements, not evidence of private reasoning.

## Claims this project can support

After a real experiment with a held-out set, report measured label accuracy, macro-F1, exact-match rate, static code metrics, confusion matrices, and calibration error. Compare against simple baselines such as majority labels and a prompt-only classifier. Record seed, model configuration, optimizer settings, checkpoint step, and data hashes.

## Claims it cannot support

The model does not expose a private internal monologue, consciousness, or a ground-truth reasoning trace. A generated explanation is an output behavior. It may be useful, wrong, post-hoc, or incomplete. The audit therefore rejects fields intended to collect hidden chain-of-thought.

## Suggested next experiment

Create a larger, permissioned dataset with paired correct and intentionally flawed coding answers. Keep prompt families separated across train and evaluation, attach isolated tests for runtime behavior, and measure whether confidence predicts correctness. Store only concise, inspectable explanations when they are needed for the research question.
