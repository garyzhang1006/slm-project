# Observable cognition, not hidden thought

This project uses “cognition” as shorthand for measurable behavior around coding tasks:

- task type: generation, debugging, explanation, algorithm reasoning, or review;
- error category: none, syntax, logic, hallucination, incomplete, or unsafe;
- confidence bucket: low, medium, or high based on an annotation in `[0, 1]`;
- held-out answer behavior: exact-match generation in the tiny demo.

The language-model head predicts the next byte. Three auxiliary heads predict the labels above from the prompt-boundary contextual representation. This gives a compact way to ask whether the model's stated confidence or error tag tracks held-out behavior.

The training path now pools auxiliary heads at the end of the encoded prompt, before answer tokens begin. Evaluation uses the same prompt-only position. The language-model loss still sees prompt plus answer tokens, so generation training and behavior-label measurement use separate information boundaries.

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
```

The evaluator reports accuracy, macro-F1, confusion matrices, and expected calibration error for task, error, and confidence-bucket heads. Expected calibration error groups predictions by maximum class probability and compares confidence with correctness. These are output-behavior measurements, not evidence of private reasoning.

## Claims this project can support

After a real experiment with a held-out set, report measured label accuracy, macro-F1, exact-match rate, confusion matrices, and calibration error. Compare against simple baselines such as majority labels and a prompt-only classifier. Record seed, model configuration, optimizer settings, checkpoint step, and data hashes.

## Claims it cannot support

The model does not expose a private internal monologue, consciousness, or a ground-truth reasoning trace. A generated explanation is an output behavior. It may be useful, wrong, post-hoc, or incomplete. The audit therefore rejects fields intended to collect hidden chain-of-thought.

## Suggested next experiment

Create a larger, permissioned dataset with paired correct and intentionally flawed coding answers. Keep prompt families separated across train and evaluation, label the actual failure mode with a test result, and measure whether confidence predicts correctness. Store only concise, inspectable explanations when they are needed for the research question.
