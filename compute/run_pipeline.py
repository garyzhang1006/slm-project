"""Drive the slm-160m chain on Kaggle: pretrain sessions until one pass is done, then distill_data, sft, eval.

Beside it run two LoRA chains: sft_data, the 360M adapter, then lora_eval and distill_data on that adapter;
and the 1.7B adapter (lora_1b7) with its own eval, which gets whatever GPU quota the other chains leave.
Each round reads kernel statuses and reports, decides one action per chain, and (unless --dry-run) pushes the
next kernels. It never runs a model locally; every push goes to a Kaggle GPU. It checks the weekly GPU quota before
each push, so a session that would be killed half way for lack of quota waits for the reset instead.
Run from the repo root, logged in with the Kaggle CLI, e.g.
    python3 compute/run_pipeline.py --owner YOUR_KAGGLE_USERNAME --watch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.stages import PRETRAIN_SESSION_SECONDS, stage_slug  # noqa: E402

# GPU hours a push must have left in the weekly quota: the stage's own time cap plus setup and save.
STAGE_HOURS = {"pretrain": PRETRAIN_SESSION_SECONDS / 3600 + 1.0, "distill_data": 1.5, "sft": 9.5,
               "eval": 1.0, "lora": 4.0, "lora_eval": 1.0, "sft_data": 0.0,
               "lora_1b7": 5.5, "lora_1b7_eval": 1.5}
# Reports that record which adapter a LoRA follow-up used, as a path of keys to its adapter_sha256.
ADAPTER_USERS = {"lora_eval": ("lora_eval_report.json", ("adapter_sha256",)),
                 "distill_data": ("distill_manifest.json", ("teacher", "adapter_sha256")),
                 "lora_1b7_eval": ("lora_eval_report.json", ("adapter_sha256",))}
LORA_FOLLOW_UPS = ("lora_eval", "distill_data")
TERMINAL = {"done", "stop"}
# Kaggle numbers pretrain sessions from 1; a chain this long means something is looping.
MAX_PRETRAIN_SESSIONS = 20
WAITING = {"queued", "running", "new_script", "pending"}
# Consecutive failed rounds (network or CLI errors) before --watch gives up: 3 hours at the default interval.
MAX_CONSECUTIVE_FAILURES = 6


def parse_status(output: str) -> str:
    """Map `kaggle kernels status` output to missing, queued, running, complete, error, ..."""
    match = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', output)
    if match:
        return match.group(1).lower()
    if "Cannot access kernel" in output or "404" in output:
        return "missing"
    raise RuntimeError(f"Unrecognized kaggle kernels status output: {output.strip()[:300]}")


def parse_gpu_quota(output: str) -> float:
    """Remaining GPU hours from the `kaggle quota` table."""
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "GPU" and len(fields) >= 3:
            return float(fields[2].rstrip("h"))
    raise RuntimeError(f"No GPU row in kaggle quota output: {output.strip()[:300]}")


def action(kind: str, reason: str, **fields) -> dict:
    return {"kind": kind, "reason": reason, **fields}


def push_if_quota(stage: str, quota_hours: float, reason: str, **fields) -> dict:
    """fields may carry hours, the GPU hours this push needs when that is less than STAGE_HOURS[stage]."""
    needed = fields.get("hours", STAGE_HOURS[stage])
    if quota_hours < needed:
        return action("wait", f"{reason}, but only {quota_hours:.2f} GPU hours remain and {stage} needs "
                              f"{needed:.1f}; waiting for the weekly quota reset", stage=stage, **fields)
    return action("push", reason, stage=stage, **fields)


def pretrain_hours(session_report: dict) -> float:
    """GPU hours the session after session_report needs: the full cap, or less when few steps remain,
    so a last session of about 39 steps does not wait for 12 hours of quota."""
    values = [session_report.get(key) for key in ("total_steps", "step_reached", "seconds_per_step")]
    if not all(type(value) in (int, float) and value > 0 for value in values):
        return STAGE_HOURS["pretrain"]
    total, reached, seconds_per_step = values
    # A quarter extra in case the next GPU runs slower than the measured one, plus an hour for setup and the save.
    return min(STAGE_HOURS["pretrain"], max(0, total - reached) * seconds_per_step * 1.25 / 3600 + 1.0)


def next_action(status, report, quota_hours: float) -> dict:
    """One decision. status(slug) -> str and report(slug, filename) -> dict | None are injected."""
    corpus = status(stage_slug("corpus"))
    if corpus != "complete":
        return action("wait" if corpus in WAITING else "stop", f"corpus kernel is {corpus}")

    last = 0
    while last < MAX_PRETRAIN_SESSIONS and status(stage_slug("pretrain", last + 1)) != "missing":
        last += 1
    if last == 0:
        return push_if_quota("pretrain", quota_hours, "no pretrain session yet", session=1)
    current = status(stage_slug("pretrain", last))
    if current in WAITING:
        return action("wait", f"pretrain session {last} is {current}")
    if current != "complete":
        return action("stop", f"pretrain session {last} ended as {current}; read its log before retrying")
    session_report = report(stage_slug("pretrain", last), f"pretrain_session_{last}.json")
    if session_report is None:
        return unreadable(f"pretrain_session_{last}.json", f"pretrain session {last}")
    if session_report.get("status") == "session_complete_resume_next":
        if last >= MAX_PRETRAIN_SESSIONS:
            return action("stop", f"{last} pretrain sessions and still not done; check the step budget")
        return push_if_quota("pretrain", quota_hours, f"pretrain session {last} reached step "
                             f"{session_report.get('step_reached')}", session=last + 1,
                             hours=round(pretrain_hours(session_report), 2))
    if session_report.get("status") != "complete":
        return action("stop", f"pretrain session {last} report status is {session_report.get('status')!r}")

    # sft attaches the distilled answers; lora_action builds them from the current adapter.
    adapter = current_adapter(status, report)
    if not follow_up_fresh(status, report, "distill_data", adapter):
        return action("wait", "sft waits for distill_data built from the current LoRA adapter", needs="lora")

    sft = status(stage_slug("sft"))
    if sft in WAITING:
        return action("wait", f"sft is {sft}")
    if sft == "missing":
        return push_if_quota("sft", quota_hours, f"pretraining finished in session {last}", pretrain_session=last)
    if sft != "complete":
        return action("stop", f"sft ended as {sft}; read its log before retrying")
    sft_report = report(stage_slug("sft"), "sft_report.json")
    if sft_report is None:
        return unreadable("sft_report.json", "sft")
    if sft_report.get("pretrain_session") != last:
        return push_if_quota("sft", quota_hours, f"sft was built on pretrain session "
                             f"{sft_report.get('pretrain_session')}, not {last}", pretrain_session=last)
    if dig(sft_report, ("distill", "teacher_adapter_sha256")) != adapter:
        return push_if_quota("sft", quota_hours, "sft trained on answers distilled from an older LoRA adapter",
                             pretrain_session=last)

    evaluation = status(stage_slug("eval"))
    if evaluation in WAITING:
        return action("wait", f"eval is {evaluation}")
    if evaluation == "missing":
        return push_if_quota("eval", quota_hours, "sft finished")
    if evaluation != "complete":
        return action("stop", f"eval ended as {evaluation}; read its log before retrying")
    eval_report = report(stage_slug("eval"), "eval_report.json")
    if eval_report is None:
        return unreadable("eval_report.json", "eval")
    # A re-pushed sft replaces the checkpoint, so an eval that finished earlier scored the old one.
    if eval_report.get("sha256") != sft_report.get("sha256"):
        return push_if_quota("eval", quota_hours, "eval scored an older sft checkpoint")
    return action("done", "eval finished; read eval_report.json from the eval kernel output")


def unreadable(filename: str, kernel: str) -> dict:
    # Kaggle.report returns None when the download fails, which is usually a transient API error, so a
    # finished stage waits a round instead of being pushed again on a missing report.
    return action("wait", f"could not read {filename} from {kernel}; retrying next round")


def dig(value, keys: tuple[str, ...]):
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def current_adapter(status, report) -> str | None:
    """adapter_sha256 of the finished LoRA kernel, or None while it is missing, running, from an older runner,
    or trained on an older sft_data build that lora_action is about to replace."""
    if status(stage_slug("lora")) != "complete":
        return None
    lora_report = report(stage_slug("lora"), "lora_report.json")
    adapter = dig(lora_report, ("adapter_sha256",))
    if not adapter:
        return None
    stale, blocked = stale_training_data(report, lora_report)
    return None if stale or blocked else adapter


def follow_up_fresh(status, report, stage: str, adapter: str | None) -> bool | None:
    """Whether stage finished on adapter, or None when its finished report could not be read."""
    filename, keys = ADAPTER_USERS[stage]
    if adapter is None or status(stage_slug(stage)) != "complete":
        return False
    found = report(stage_slug(stage), filename)
    return None if found is None else dig(found, keys) == adapter


def stale_training_data(report, lora_report: dict) -> tuple[bool, dict | None]:
    """Whether an adapter trained on an older sft_data build, or the decision to make when that is unknown."""
    manifest = report(stage_slug("sft_data"), "sft_manifest.json")
    if manifest is None:
        return False, unreadable("sft_manifest.json", "sft_data")
    train = dig(manifest, ("files", "train", "sha256"))
    if not train:
        # Retraining could never match a manifest without the hash, so it would loop.
        return False, action("stop", "sft_manifest.json records no train sha256; push sft_data again")
    return dig(lora_report, ("data", "train", "sha256")) != train, None


def lora_action(status, report, quota_hours: float) -> dict:
    """One decision for the LoRA chain: sft_data, lora, then lora_eval and distill_data on the same adapter."""
    data = status(stage_slug("sft_data"))
    if data in WAITING:
        return action("wait", f"sft_data is {data}")
    if data == "missing":
        return action("push", "the LoRA adapter trains on sft_data", stage="sft_data")
    if data != "complete":
        return action("stop", f"sft_data ended as {data}; read its log before retrying")

    teacher = status(stage_slug("lora"))
    if teacher in WAITING:
        return action("wait", f"lora is {teacher}")
    if teacher == "missing":
        return push_if_quota("lora", quota_hours, "sft_data is ready")
    if teacher != "complete":
        return action("stop", f"lora ended as {teacher}; read its log before retrying")
    lora_report = report(stage_slug("lora"), "lora_report.json")
    if lora_report is None:
        return unreadable("lora_report.json", "the lora kernel")
    if lora_report.get("status") != "complete_pending_manual_review":
        return action("stop", f"lora report status is {lora_report.get('status')!r}")
    adapter = lora_report.get("adapter_sha256")
    followers = {stage: status(stage_slug(stage)) for stage in LORA_FOLLOW_UPS}
    stale = not adapter
    if adapter:
        stale, blocked = stale_training_data(report, lora_report)
        if blocked:
            return blocked
    if stale:
        # A new lora version replaces the output that queued follow-ups would mount, so let them finish first.
        busy = [stage for stage, state in followers.items() if state in WAITING]
        if busy:
            return action("wait", f"the adapter needs retraining once {', '.join(busy)} finishes")
        return push_if_quota("lora", quota_hours, "the adapter trained on an older sft_data build" if adapter else
                             "the adapter predates best-step selection and the merged export, so it has no "
                             "adapter_sha256")

    waiting, stopped = [], []
    for stage, state in followers.items():
        fresh = state == "complete" and follow_up_fresh(status, report, stage, adapter)
        if state in WAITING:
            waiting.append(f"{stage} is {state}")
        elif fresh is None:
            waiting.append(unreadable(ADAPTER_USERS[stage][0], stage)["reason"])
        elif state == "missing" or (state == "complete" and not fresh):
            return push_if_quota(stage, quota_hours, f"{stage} has not run on adapter {adapter[:12]}")
        elif state != "complete":
            stopped.append(f"{stage} ended as {state}; read its log before retrying")
    if stopped:
        return action("stop", "; ".join(stopped))
    if waiting:
        return action("wait", "; ".join(waiting))
    return action("done", f"lora_eval and distill_data both used adapter {adapter[:12]}")


def large_lora_action(status, report, quota_hours: float) -> dict:
    """One decision for the 1.7B chain: lora_1b7 on the shared sft_data, then lora_1b7_eval on its adapter."""
    data = status(stage_slug("sft_data"))
    if data in WAITING or data == "missing":
        # lora_action pushes sft_data; pushing it here too would start two versions in one round.
        return action("wait", f"lora_1b7 needs sft_data, which is {data}")
    if data != "complete":
        return action("stop", f"sft_data ended as {data}")
    teacher = status(stage_slug("lora_1b7"))
    if teacher in WAITING:
        return action("wait", f"lora_1b7 is {teacher}")
    if teacher == "missing":
        return push_if_quota("lora_1b7", quota_hours, "sft_data is ready")
    if teacher != "complete":
        return action("stop", f"lora_1b7 ended as {teacher}; read its log before retrying")
    lora_report = report(stage_slug("lora_1b7"), "lora_report.json")
    if lora_report is None:
        return unreadable("lora_report.json", "lora_1b7")
    adapter = lora_report.get("adapter_sha256")
    if lora_report.get("status") != "complete_pending_manual_review" or not adapter:
        return action("stop", f"lora_1b7 report status is {lora_report.get('status')!r}")
    stale, blocked = stale_training_data(report, lora_report)
    if blocked:
        return blocked
    evaluation = status(stage_slug("lora_1b7_eval"))
    if evaluation in WAITING:
        return action("wait", f"lora_1b7_eval is {evaluation}")
    if stale:
        return push_if_quota("lora_1b7", quota_hours, "lora_1b7 trained on an older sft_data build")
    fresh = evaluation == "complete" and follow_up_fresh(status, report, "lora_1b7_eval", adapter)
    if fresh is None:
        return unreadable("lora_eval_report.json", "lora_1b7_eval")
    if evaluation == "missing" or (evaluation == "complete" and not fresh):
        return push_if_quota("lora_1b7_eval", quota_hours, f"lora_1b7_eval has not run on adapter {adapter[:12]}")
    if evaluation != "complete":
        return action("stop", f"lora_1b7_eval ended as {evaluation}; read its log before retrying")
    return action("done", f"lora_1b7_eval used adapter {adapter[:12]}")


def decide_round(status, report, quota_hours: float) -> list[dict]:
    """[main chain, 360M LoRA chain, 1.7B LoRA chain] decisions, in priority order: each push leaves less
    quota for the chains after it."""
    decisions = []
    for chooser in (next_action, lora_action, large_lora_action):
        decision = chooser(status, report, quota_hours)
        if decision["kind"] == "push":
            quota_hours -= decision.get("hours", STAGE_HOURS[decision["stage"]])
        decisions.append(decision)
    main, side = decisions[0], decisions[1]
    if main.get("needs") == "lora" and side["kind"] == "stop":
        decisions[0] = action("stop", f"{main['reason']}, but the LoRA chain stopped")
    return decisions


class Kaggle:
    """Thin wrapper over the kaggle CLI, so next_action stays testable without network access."""

    def __init__(self, owner: str, command: str = "kaggle") -> None:
        self.owner, self.command = owner, command

    def run(self, *args: str, check: bool = True) -> str:
        result = subprocess.run([self.command, *args], capture_output=True, text=True)
        output = result.stdout + result.stderr
        if check and result.returncode:
            raise RuntimeError(f"kaggle {' '.join(args)} failed ({result.returncode}): {output.strip()[:500]}")
        return output

    def status(self, slug: str) -> str:
        return parse_status(self.run("kernels", "status", f"{self.owner}/{slug}", check=False))

    def quota_hours(self) -> float:
        return parse_gpu_quota(self.run("quota"))

    def report(self, slug: str, filename: str) -> dict | None:
        with tempfile.TemporaryDirectory() as directory:
            self.run("kernels", "output", f"{self.owner}/{slug}", "-p", directory,
                     "--file-pattern", re.escape(filename) + "$", check=False)
            matches = list(Path(directory).rglob(filename))
            if len(matches) != 1:
                return None
            # A truncated download or a non-object reads as missing, so the watcher waits a round for it.
            try:
                value = json.loads(matches[0].read_text())
            except ValueError:
                return None
            return value if isinstance(value, dict) else None

    def push(self, decision: dict) -> str:
        from compute.package import prepare

        with tempfile.TemporaryDirectory() as directory:
            prepare(decision["stage"], Path(directory), self.owner, decision.get("session"),
                    decision.get("pretrain_session"))
            return self.run("kernels", "push", "-p", directory).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Kaggle username that owns the kernels")
    parser.add_argument("--watch", action="store_true", help="Keep deciding every --interval seconds until done")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="Print the decision without pushing anything")
    args = parser.parse_args(argv)
    if args.interval < 60:
        parser.error("--interval must be at least 60 seconds; Kaggle status changes slowly")
    kaggle = Kaggle(args.owner)
    failures = 0
    while True:
        try:
            decisions = decide_round(kaggle.status, kaggle.report, kaggle.quota_hours())
            for decision in decisions:
                print(time.strftime("%Y-%m-%d %H:%M"), json.dumps(decision), flush=True)
                if decision["kind"] == "push" and not args.dry_run:
                    print(kaggle.push(decision), flush=True)
            failures = 0
        except (RuntimeError, OSError) as error:
            # A DNS blip or API hiccup killed the first watch run; retry next round instead of dying.
            failures += 1
            if not args.watch or failures >= MAX_CONSECUTIVE_FAILURES:
                raise
            print(time.strftime("%Y-%m-%d %H:%M"), json.dumps({"kind": "retry", "failures": failures,
                                                               "error": str(error)[:300]}), flush=True)
            time.sleep(args.interval)
            continue
        kinds = {decision["kind"] for decision in decisions}
        if kinds <= TERMINAL:
            return 0 if kinds == {"done"} else 1
        if not args.watch or args.dry_run:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
