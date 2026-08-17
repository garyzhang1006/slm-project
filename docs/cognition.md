# Observable cognition, not hidden thought

This project uses “cognition” as shorthand for measurable behavior around coding tasks:

- task type: generation, debugging, explanation, algorithm reasoning, or review;
- error category: none, syntax, logic, hallucination, incomplete, or unsafe;
- confidence bucket: low, medium, or high based on an annotation in `[0, 1]`;
- held-out answer behavior: exact-match generation in the tiny demo.

The language-model head predicts the next byte. Three auxiliary heads predict the labels above from the final contextual representation. This gives a compact way to ask whether the model's stated confidence or error tag tracks held-out behavior.

## Claims this project can support

After a real experiment with a held-out set, report measured label accuracy, exact-match rate, and calibration error. Compare against simple baselines such as majority labels and a prompt-only classifier.

## Claims it cannot support

The model does not expose a private internal monologue, consciousness, or a ground-truth reasoning trace. A generated explanation is an output behavior. It may be useful, wrong, post-hoc, or incomplete. The audit therefore rejects fields intended to collect hidden chain-of-thought.

## Suggested next experiment

Create a larger, permissioned dataset with paired correct and intentionally flawed coding answers. Keep prompt families separated across train and evaluation, label the actual failure mode with a test result, and measure whether confidence predicts correctness. Store only concise, inspectable explanations when they are needed for the research question.
