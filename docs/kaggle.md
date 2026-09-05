# Kaggle verification and training

Use `scripts/prepare_kaggle.py` to bundle the current working source. No GitHub push is needed. The generated private kernel contains the package, tests, committed JSON data, and a source hash manifest. `scripts/kaggle_run.py` refuses to train outside Kaggle or without working CUDA.

The verification run checks the regression suite and existing data audit before running ten synthetic training iterations, including resume after iteration eight, with the `slm-500m` preset at actual sequence length 2048. It uses FP16, microbatch size one, accumulation two, and activation checkpointing. Alternating length fixtures exercise coding and `language_generation` task labels. The checkpoint and `verification.json` become Kaggle outputs. These iterations establish execution and recovery, not trained coding or language ability.

The latest completed run is recorded in [`reports/slm-500m-2048-language-verification.json`](../reports/slm-500m-2048-language-verification.json). It reached 499,524,075 parameters on a Tesla T4 with ten optimizer steps and zero skipped updates.

The quality runner result is recorded in [`reports/slm-50m-language-quality-verification.json`](../reports/slm-50m-language-quality-verification.json). It combines the base conversational examples with a deterministic project-authored English/Python curriculum, trains 1,200 steps, and probes greetings, grammar, explanations, code generation, debugging, and algorithm reasoning. Treat the result as a narrow Studio smoke checkpoint, not a broad capability benchmark.

The source package uses the runtime's installed PyTorch and NumPy; it does not install dependencies or need internet. PyTorch 2.3 or later is required. See the [PyTorch mixed-precision API](https://docs.pytorch.org/docs/2.3/amp.html) and [official Kaggle kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

For real training, replace the synthetic fixture with licensed canonical JSONL records and a disjoint evaluation split. Run the existing audit first. Include examples that exercise the intended context length; increasing `block_size` alone cannot teach long-range dependencies. Inspect `truncated_records` and `max_tokens` in the training report.

Resume preserves the checkpoint's model dimensions and context. Old modern or legacy weights remain loadable; `--preset` and dimension flags do not resize them. Model-only checkpoints start a fresh optimizer. Old checkpoints with one optimizer parameter group retain that grouping. New checkpoints save random state for interruption recovery on the same hardware configuration. Changing seed, batch layout, precision, warmup, or total steps changes the continuation; extending total steps replans cosine decay. Weights trained at 256 are not presented as trained 2048 weights.

Periodic checkpoints atomically replace the output path, keeping the last complete checkpoint if serialization fails. Download the output before deleting a Kaggle session. A failed kernel has no successful `verification.json`; inspect its log and resolve the first error before rerunning.
