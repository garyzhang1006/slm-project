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
    # The LoRA teacher answers Dolly questions whose human answers were too long for stage 3. It is the 1.7B
    # adapter, which scores 215 of 252 everyday questions exact against 188 for the 360M one.
    "distill_data": {"runner": "compute/distill_data.py", "slug": "slm-distill-data",
                     "internet": True, "gpu": True, "attaches": ["slm-lora-1b7"]},
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
    # Rewrites compute/RESULTS.md and reports/ from the attached outputs, so the owner's machine never has to.
    # stage_attaches adds the pretrain sessions; sft and eval come only from package.py --also, so the push never
    # lists a kernel that may not exist yet.
    "results": {"runner": "compute/collect_results.py", "slug": "slm-results",
                "args": ["--input-dir", "/kaggle/input"],
                "internet": False, "gpu": False,
                "attaches": ["slm-lora-baseline", "slm-lora-eval", "slm-lora-1b7", "slm-lora-1b7-eval",
                             "slm-distill-data"]},
}

# Byte tokenizer: one token per byte, so the corpus byte target is also the token count. This is the train text
# stage 1 builds (its DEFAULT_TARGET_BYTES); the step total below no longer follows from it.
PRETRAIN_TARGET_BYTES = 1_500_000_000
BLOCK_SIZE = 2048
PRETRAIN_BATCH_SIZE = 8
PRETRAIN_GRADIENT_ACCUMULATION = 4
TOKENS_PER_STEP = PRETRAIN_BATCH_SIZE * PRETRAIN_GRADIENT_ACCUMULATION * BLOCK_SIZE
# Sessions 1 to 4 ran and reached this step (compute/RESULTS.md).
PRETRAIN_SESSIONS_RUN = 4
PRETRAIN_STEP_REACHED = 17_662
# The cosine schedule ends where session 5 stops instead of after one full pass of 22,889 steps: at the slower
# pace of sessions 3 and 4, session 5 would stop near step 21,900 and a sixth session would train the last 1,000
# or so steps at under 0.6% of the peak rate. Session 5 gets train.py --max-seconds 39,600 (stage2_pretrain caps
# it at PRETRAIN_SESSION_SECONDS, and setup has stayed inside SETUP_AND_SAVE_RESERVE_SECONDS in every session so
# far), so even at 10.0 s a step planned_steps(39,600, 10.0, SESSION_RESERVE_SECONDS) = 3,900 steps fit:
# 17,662 + 3,900 = 21,562. That trains 1.41e9 byte tokens, about 94% of the 1.5 GB corpus, and session 5 rejoins
# the shorter cosine at about 8% of the peak rate where the old one had it at 13%.
PRETRAIN_TOTAL_STEPS = 21_562
# Kaggle stops a session at 12 h; 11 h leaves an hour for setup, the final save and the report.
PRETRAIN_SESSION_SECONDS = 11 * 3600
# Measured, not estimated: sessions 1 and 2 (Kaggle T4, fp16) reported 8.72 s a step and sessions 3 and 4 reported
# 9.54 and 9.18, all including periodic eval and saves. The slowest keeps a session count from coming up short.
# The FLOP-based guess of 5.5 left out attention over 2,048 positions.
SECONDS_PER_STEP_ESTIMATE = 9.54
SESSION_RESERVE_SECONDS = 600
# Raise this whenever distill_data changes which answers it keeps or how it trims them: an older build on the same
# adapter otherwise looks fresh, so run_pipeline would never rebuild it and sft would learn the rows it now rejects.
DISTILL_FILTERS_VERSION = 6


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


def last_pretrain_session() -> int:
    """The session expected to finish pretraining, counted on from the sessions already run."""
    # Counting from step 0 at the slowest measured pace would round up to a sixth session that never runs.
    return PRETRAIN_SESSIONS_RUN + pretrain_sessions(PRETRAIN_TOTAL_STEPS - PRETRAIN_STEP_REACHED)


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
        # stage_slug would reject it too, but its message names --session, which sft doesn't take.
        if pretrain_session is not None and pretrain_session < 1:
            raise ValueError(f"--pretrain-session needs k >= 1, got {pretrain_session}")
        last = last_pretrain_session() if pretrain_session is None else pretrain_session
        attaches.insert(0, stage_slug("pretrain", last))
    elif stage == "results":
        if pretrain_session is not None and pretrain_session < 1:
            raise ValueError(f"--pretrain-session needs k >= 1, got {pretrain_session}")
        # Only sessions known to have run, unless the caller names a later one that has.
        last = PRETRAIN_SESSIONS_RUN if pretrain_session is None else pretrain_session
        attaches += [stage_slug("pretrain", number) for number in range(1, last + 1)]
    return attaches


def _stage(stage: str) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose one of {', '.join(STAGES)}")
    return STAGES[stage]
