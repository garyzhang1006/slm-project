"""Stage table and step-budget math for the slm-160m English compute pipeline.

Pure Python with no third-party imports, so tests and compute/package.py can load it anywhere.
"""

from __future__ import annotations

import math

OWNER = "garyzhang11111"

STAGES = {
    "corpus": {"runner": "compute/stage1_corpus.py", "slug": "slm-160m-corpus",
               "internet": True, "gpu": False, "attaches": []},
    # Session k also attaches pretrain session k-1; see stage_attaches.
    "pretrain": {"runner": "compute/stage2_pretrain.py", "slug": "slm-160m-pretrain",
                 "internet": False, "gpu": True, "attaches": ["slm-160m-corpus"]},
    "sft_data": {"runner": "compute/stage3_sft_data.py", "slug": "slm-sft-data",
                 "internet": True, "gpu": False, "attaches": []},
    # The last pretrain session is added by stage_attaches, since its number is only known at run time.
    "sft": {"runner": "compute/stage4_sft.py", "slug": "slm-160m-sft",
            "internet": False, "gpu": True, "attaches": ["slm-sft-data", "slm-distill-data"]},
    # The corpus supplies pretrain_eval.jsonl for held-out bits-per-byte.
    "eval": {"runner": "compute/stage5_evaluate.py", "slug": "slm-160m-eval",
             "internet": False, "gpu": True, "attaches": ["slm-160m-sft", "slm-160m-corpus"]},
    # Internet stays on so the runner can pip install peft and download the base model. Each adapter trains the
    # runner's default epochs: the 5 and 3 epoch runs of 2026-09-29 had their lowest held-out loss near epoch 2
    # for 360M (step 2500 of 6485) and epoch 1 for 1.7B (step 1250 of 3891), and only got worse after that.
    "lora": {"runner": "compute/lora_baseline.py", "slug": "slm-lora-baseline",
             "internet": True, "gpu": True, "attaches": ["slm-sft-data"]},
    # The LoRA teacher answers Dolly questions whose human answers were too long for stage 3.
    "distill_data": {"runner": "compute/distill_data.py", "slug": "slm-distill-data",
                     "internet": True, "gpu": True, "attaches": ["slm-lora-baseline"]},
    # Scores base SmolLM2 and the trained adapter on the 252 everyday questions; internet for the base model.
    "lora_eval": {"runner": "compute/lora_eval.py", "slug": "slm-lora-eval",
                  "internet": True, "gpu": True, "attaches": ["slm-lora-baseline"]},
    # The same runners on SmolLM2-1.7B-Instruct; "args" is appended to the runner's command line. Its one epoch
    # measured 9 s a step, about 3.2 hours, so the runner's 3 hour default would cut it short.
    "lora_1b7": {"runner": "compute/lora_baseline.py", "slug": "slm-lora-1b7",
                 "args": ["--model", "1.7b", "--max-seconds", str(4 * 3600)],
                 "internet": True, "gpu": True, "attaches": ["slm-sft-data"]},
    "lora_1b7_eval": {"runner": "compute/lora_eval.py", "slug": "slm-lora-1b7-eval",
                      "internet": True, "gpu": True, "attaches": ["slm-lora-1b7"]},
}

# Byte tokenizer: one token per byte, so the corpus byte target is also the token count.
PRETRAIN_TARGET_BYTES = 1_500_000_000
BLOCK_SIZE = 2048
PRETRAIN_BATCH_SIZE = 8
PRETRAIN_GRADIENT_ACCUMULATION = 4
TOKENS_PER_STEP = PRETRAIN_BATCH_SIZE * PRETRAIN_GRADIENT_ACCUMULATION * BLOCK_SIZE
# One pass over the train split: ceil(1.5e9 / 65,536) = 22,889 optimizer steps.
PRETRAIN_TOTAL_STEPS = math.ceil(PRETRAIN_TARGET_BYTES / TOKENS_PER_STEP)
# Kaggle stops a session at 12 h; 11 h leaves an hour for setup, the final save and the report.
PRETRAIN_SESSION_SECONDS = 11 * 3600
# Measured, not estimated: pretrain session 1 (Kaggle T4, fp16, 2026-09-26) reported
# seconds_per_step 8.72 over 4,570 steps. The FLOP-based guess of 5.5 left out attention over 2,048
# positions. At 8.72 s a pass is about 55 GPU-hours in 6 planned sessions of about 4,470 steps; sessions 1
# and 2 each ran 4,570, so at that pace the sixth session runs only the last 39 steps.
SECONDS_PER_STEP_ESTIMATE = 8.72
SESSION_RESERVE_SECONDS = 600
# Raise this whenever distill_data changes which answers it keeps or how it trims them: an older build on the same
# adapter otherwise looks fresh, so run_pipeline would never rebuild it and sft would learn the rows it now rejects.
DISTILL_FILTERS_VERSION = 3


def planned_steps(budget_seconds: float, seconds_per_step: float, reserve_seconds: float) -> int:
    """Optimizer steps that fit in budget_seconds after holding back reserve_seconds."""
    if seconds_per_step <= 0:
        raise ValueError(f"seconds_per_step must be positive, got {seconds_per_step}")
    if budget_seconds < 0 or reserve_seconds < 0:
        raise ValueError(f"budget and reserve must be non-negative, got {budget_seconds} and {reserve_seconds}")
    return max(0, int((budget_seconds - reserve_seconds) // seconds_per_step))


def pretrain_sessions(total_steps: int = PRETRAIN_TOTAL_STEPS,
                      seconds_per_step: float = SECONDS_PER_STEP_ESTIMATE,
                      session_seconds: float = PRETRAIN_SESSION_SECONDS,
                      reserve_seconds: float = SESSION_RESERVE_SECONDS) -> int:
    """Chained Kaggle sessions needed to reach total_steps."""
    per_session = planned_steps(session_seconds, seconds_per_step, reserve_seconds)
    if per_session < 1:
        raise ValueError("a session cannot complete one step; raise session_seconds or lower the reserve")
    return math.ceil(total_steps / per_session)


def stage_slug(stage: str, session: int | None = None) -> str:
    """Kernel slug; pretrain needs a session number because each session is its own kernel."""
    base = _stage(stage)["slug"]
    if stage != "pretrain":
        if session is not None:
            raise ValueError(f"--session only applies to pretrain, not {stage}")
        return base
    if session is None or session < 1:
        raise ValueError(f"pretrain needs --session k with k >= 1, got {session}")
    return f"{base}-{session}"


def stage_attaches(stage: str, session: int | None = None, pretrain_session: int | None = None) -> list[str]:
    """Kernel slugs (without owner) whose outputs the stage reads from /kaggle/input."""
    attaches = list(_stage(stage)["attaches"])
    if stage == "pretrain":
        stage_slug(stage, session)
        if session > 1:
            attaches.append(stage_slug("pretrain", session - 1))
    elif stage == "sft":
        last = pretrain_sessions() if pretrain_session is None else pretrain_session
        attaches.insert(0, stage_slug("pretrain", last))
    return attaches


def _stage(stage: str) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose one of {', '.join(STAGES)}")
    return STAGES[stage]
