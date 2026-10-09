from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import run_pipeline  # noqa: E402
from compute.stages import DISTILL_FILTERS_VERSION, PRETRAIN_TOTAL_STEPS, stage_attaches  # noqa: E402

RESUME = {"status": "session_complete_resume_next", "step_reached": 4570}
DONE = {"status": "complete", "step_reached": PRETRAIN_TOTAL_STEPS}


ADAPTER = "ab" * 32
LARGE_ADAPTER = "ef" * 32
TRAIN = "cd" * 32
LORA_DONE = {"status": "complete_pending_manual_review", "adapter_sha256": ADAPTER,
             "data": {"train": {"sha256": TRAIN}}}
LARGE_LORA_DONE = {**LORA_DONE, "adapter_sha256": LARGE_ADAPTER}
SFT_DATA = {("slm-sft-data", "sft_manifest.json"): {"files": {"train": {"sha256": TRAIN}}}}
# The 1.7B adapter is the distill_data teacher.
TEACHER_READY = {"slm-lora-1b7": "complete", "slm-lora-1b7-eval": "complete"}
TEACHER = {("slm-lora-1b7", "lora_report.json"): LARGE_LORA_DONE}
FRESH_FOLLOW_UPS = {("slm-lora-eval", "lora_eval_report.json"): {"adapter_sha256": ADAPTER},
                    ("slm-lora-1b7-eval", "lora_eval_report.json"): {"adapter_sha256": LARGE_ADAPTER},
                    ("slm-distill-data", "distill_manifest.json"): {"teacher": {"adapter_sha256": LARGE_ADAPTER},
                                                                    "filters_version": DISTILL_FILTERS_VERSION}}
# The distill stats sft_report.json records for rows distilled from LARGE_ADAPTER with the current filters.
DISTILLED = {"teacher_adapter_sha256": LARGE_ADAPTER, "filters_version": DISTILL_FILTERS_VERSION}


def decide(statuses, reports=None, quota=30.0, chooser=None):
    reports = reports or {}
    return (chooser or run_pipeline.next_action)(lambda slug: statuses.get(slug, "missing"),
                                                 lambda slug, filename: reports.get((slug, filename)), quota)


def decide_lora(statuses, reports=None, quota=30.0):
    return decide(statuses, reports, quota, run_pipeline.lora_action)


def decide_large(statuses, reports=None, quota=30.0):
    return decide(statuses, reports, quota, run_pipeline.large_lora_action)


class ParseTests(unittest.TestCase):
    def test_status_lines(self):
        self.assertEqual(run_pipeline.parse_status(
            'joshkerr1111/slm-160m-pretrain-2 has status "KernelWorkerStatus.RUNNING"'), "running")
        self.assertEqual(run_pipeline.parse_status(
            "Cannot access kernel 'x/y' (Permission 'kernels.get' was denied)."), "missing")
        with self.assertRaises(RuntimeError):
            run_pipeline.parse_status("Authentication required to call the Kaggle API.")

    def test_quota_table(self):
        table = ("resource  used    remaining  total   refreshAt\n"
                 "--------  ------  ---------  ------  -------------------\n"
                 "GPU       14.00h  16.00h     30.00h  2026-10-03T00:00:00\n"
                 "TPU       0.00h   20.00h     20.00h  2026-10-03T00:00:00\n")
        self.assertEqual(run_pipeline.parse_gpu_quota(table), 16.0)
        with self.assertRaises(RuntimeError):
            run_pipeline.parse_gpu_quota("Authentication required")


class DecisionTests(unittest.TestCase):
    def test_waits_for_corpus_and_stops_on_corpus_error(self):
        self.assertEqual(decide({"slm-160m-corpus": "running"})["kind"], "wait")
        self.assertEqual(decide({"slm-160m-corpus": "error"})["kind"], "stop")

    def test_first_session_is_pushed(self):
        result = decide({"slm-160m-corpus": "complete"})
        self.assertEqual((result["kind"], result["stage"], result["session"]), ("push", "pretrain", 1))

    def test_running_session_waits_and_failed_session_stops(self):
        base = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete"}
        self.assertEqual(decide({**base, "slm-160m-pretrain-2": "queued"})["kind"], "wait")
        self.assertEqual(decide({**base, "slm-160m-pretrain-2": "error"})["kind"], "stop")

    def test_next_session_needs_quota(self):
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete",
                    "slm-160m-pretrain-2": "complete"}
        reports = {("slm-160m-pretrain-2", "pretrain_session_2.json"): RESUME}
        result = decide(statuses, reports)
        self.assertEqual((result["kind"], result["session"]), ("push", 3))
        starved = decide(statuses, reports, quota=6.0)
        self.assertEqual((starved["kind"], starved["stage"]), ("wait", "pretrain"))
        self.assertIn("quota", starved["reason"])

    def test_last_pretrain_session_needs_only_the_quota_its_steps_take(self):
        statuses = {"slm-160m-corpus": "complete", "slm-sft-data": "complete",
                    **{f"slm-160m-pretrain-{session}": "complete" for session in range(1, 6)}}
        last = {**RESUME, "step_reached": 21_400, "total_steps": 21_562, "seconds_per_step": 9.54}
        reports = {("slm-160m-pretrain-5", "pretrain_session_5.json"): last}
        result = decide(statuses, reports, quota=2.0)
        self.assertEqual((result["kind"], result["session"], result["hours"]), ("push", 6, 1.54))
        # The round charges those hours, not the 12 hour cap, so the LoRA chain can still push.
        main, side, _ = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                  lambda slug, filename: reports.get((slug, filename)), 6.0)
        self.assertEqual((main["kind"], side["kind"], side["stage"]), ("push", "push", "lora"))
        # A report without the step fields still asks for the full cap.
        self.assertEqual(run_pipeline.pretrain_hours(RESUME), run_pipeline.STAGE_HOURS["pretrain"])

    def test_unreadable_reports_wait_instead_of_repushing(self):
        # Kaggle.report returns None when a download fails, so a finished stage must not be pushed again.
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete"}
        self.assertEqual(decide(statuses)["kind"], "wait")
        statuses.update({**TEACHER_READY, "slm-distill-data": "complete", "slm-160m-sft": "complete"})
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): DONE, **TEACHER, **FRESH_FOLLOW_UPS,
                   **SFT_DATA}
        self.assertEqual(decide(statuses, reports)["kind"], "wait")
        lora = {**LoraChainTests.READY, "slm-lora-eval": "complete"}
        self.assertEqual(decide_lora(lora, {("slm-lora-baseline", "lora_report.json"): LORA_DONE, **SFT_DATA})["kind"],
                         "wait")
        large = {"slm-sft-data": "complete", **TEACHER_READY, "slm-distill-data": "complete"}
        self.assertEqual(decide_large(large, {**TEACHER, **SFT_DATA})["kind"], "wait")

    def test_sft_then_eval_then_done(self):
        statuses = {"slm-160m-corpus": "complete", **{f"slm-160m-pretrain-{k}": "complete" for k in (1, 2, 3)}}
        reports = {("slm-160m-pretrain-3", "pretrain_session_3.json"): DONE, **SFT_DATA}
        waiting = decide(statuses, reports)  # no adapter or distilled answers yet
        self.assertEqual((waiting["kind"], waiting["needs"]), ("wait", "lora_1b7"))
        statuses.update({"slm-lora-1b7": "complete", "slm-distill-data": "complete"})
        reports.update(TEACHER)
        reports[("slm-distill-data", "distill_manifest.json")] = {"teacher": {"adapter_sha256": "old"}}
        self.assertEqual(decide(statuses, reports)["kind"], "wait")  # distilled from an older adapter
        reports.update(FRESH_FOLLOW_UPS)
        result = decide(statuses, reports)
        self.assertEqual((result["kind"], result["stage"], result["pretrain_session"]), ("push", "sft", 3))
        statuses["slm-160m-sft"] = "complete"
        stale = decide(statuses, {**reports, ("slm-160m-sft", "sft_report.json"): {"pretrain_session": 2}})
        self.assertEqual((stale["kind"], stale["stage"]), ("push", "sft"))
        reports[("slm-160m-sft", "sft_report.json")] = {"pretrain_session": 3, "sha256": "sft3",
                                                        "distill": DISTILLED}
        self.assertEqual(decide(statuses, reports)["stage"], "eval")
        statuses["slm-160m-eval"] = "running"
        self.assertEqual(decide(statuses, reports)["kind"], "wait")
        statuses["slm-160m-eval"] = "complete"
        reports[("slm-160m-eval", "eval_report.json")] = {"sha256": "sft3"}
        self.assertEqual(decide(statuses, reports)["kind"], "done")

    def test_a_complete_fifth_session_goes_to_sft_without_a_sixth(self):
        # The step total is set so session 5 ends the cosine; the estimated session count must not pick the kernel.
        statuses = {"slm-160m-corpus": "complete", "slm-sft-data": "complete", "slm-lora-baseline": "complete",
                    "slm-lora-eval": "complete", **TEACHER_READY, "slm-distill-data": "complete",
                    **{f"slm-160m-pretrain-{session}": "complete" for session in range(1, 6)}}
        finished = {"status": "complete", "step_reached": PRETRAIN_TOTAL_STEPS, "total_steps": PRETRAIN_TOTAL_STEPS,
                    "seconds_per_step": 9.6, "next_session": None}
        reports = {("slm-160m-pretrain-5", "pretrain_session_5.json"): finished,
                   ("slm-lora-baseline", "lora_report.json"): LORA_DONE, **TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA}
        main, side, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                      lambda slug, filename: reports.get((slug, filename)), 30.0)
        self.assertEqual((main["kind"], main["stage"], main["pretrain_session"]), ("push", "sft", 5))
        self.assertNotIn("session", main)
        self.assertEqual((side["kind"], large["kind"]), ("done", "done"))
        self.assertEqual(stage_attaches("sft", pretrain_session=main["pretrain_session"])[0], "slm-160m-pretrain-5")
        kaggle = run_pipeline.Kaggle("someone")
        kaggle.run = lambda *args, check=True: "Kernel version 1 successfully pushed."
        with mock.patch("compute.package.prepare") as prepare:
            kaggle.push(main)
        stage, _, owner, session, pretrain_session = prepare.call_args.args
        self.assertEqual((stage, owner, session, pretrain_session), ("sft", "someone", None, 5))

    def test_sft_reruns_when_distill_data_came_from_a_newer_adapter(self):
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete", "slm-lora-1b7": "complete",
                    "slm-distill-data": "complete", "slm-160m-sft": "complete"}
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): DONE, **TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA,
                   ("slm-160m-sft", "sft_report.json"): {"pretrain_session": 1,
                                                         "distill": {**DISTILLED, "teacher_adapter_sha256": "old"}}}
        result = decide(statuses, reports)
        self.assertEqual((result["kind"], result["stage"], result["pretrain_session"]), ("push", "sft", 1))
        reports[("slm-160m-sft", "sft_report.json")]["distill"]["teacher_adapter_sha256"] = LARGE_ADAPTER
        self.assertEqual(decide(statuses, reports)["stage"], "eval")

    def test_distill_data_and_sft_rerun_when_the_answer_filters_change(self):
        # The same adapter marked answers built by older filters fresh, so sft would have trained on looping lists.
        old = {"teacher": {"adapter_sha256": LARGE_ADAPTER}}
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete", "slm-sft-data": "complete",
                    **TEACHER_READY, "slm-distill-data": "complete"}
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): DONE, **TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA,
                   ("slm-distill-data", "distill_manifest.json"): old}
        self.assertEqual(decide(statuses, reports)["kind"], "wait")
        rebuild = decide_large(statuses, reports)
        self.assertEqual((rebuild["kind"], rebuild["stage"]), ("push", "distill_data"))
        self.assertIn("answer filters", rebuild["reason"])
        reports.update(FRESH_FOLLOW_UPS)
        statuses["slm-160m-sft"] = "complete"
        reports[("slm-160m-sft", "sft_report.json")] = {"pretrain_session": 1,
                                                        "distill": {"teacher_adapter_sha256": LARGE_ADAPTER}}
        result = decide(statuses, reports)
        self.assertEqual((result["kind"], result["stage"]), ("push", "sft"))
        reports[("slm-160m-sft", "sft_report.json")]["distill"] = DISTILLED
        self.assertEqual(decide(statuses, reports)["stage"], "eval")

    def test_eval_reruns_when_sft_was_pushed_again(self):
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete", "slm-lora-1b7": "complete",
                    "slm-distill-data": "complete", "slm-160m-sft": "complete", "slm-160m-eval": "complete"}
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): DONE, **TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA,
                   ("slm-160m-sft", "sft_report.json"): {"pretrain_session": 1, "sha256": "new",
                                                         "distill": DISTILLED},
                   ("slm-160m-eval", "eval_report.json"): {"sha256": "old"}}
        result = decide(statuses, reports)
        self.assertEqual((result["kind"], result["stage"]), ("push", "eval"))
        del reports[("slm-160m-eval", "eval_report.json")]
        self.assertEqual(decide(statuses, reports)["kind"], "wait")
        reports[("slm-160m-eval", "eval_report.json")] = {"sha256": "new"}
        self.assertEqual(decide(statuses, reports)["kind"], "done")


class LoraChainTests(unittest.TestCase):
    READY = {"slm-sft-data": "complete", "slm-lora-baseline": "complete"}

    def test_sft_data_then_lora(self):
        self.assertEqual(decide_lora({})["stage"], "sft_data")
        self.assertEqual(decide_lora({"slm-sft-data": "running"})["kind"], "wait")
        self.assertEqual(decide_lora({"slm-sft-data": "error"})["kind"], "stop")
        result = decide_lora({"slm-sft-data": "complete"})
        self.assertEqual((result["kind"], result["stage"]), ("push", "lora"))
        self.assertEqual(decide_lora({"slm-sft-data": "complete"}, quota=2.0)["kind"], "wait")
        self.assertEqual(decide_lora({**self.READY, "slm-lora-baseline": "error"})["kind"], "stop")

    def test_old_adapter_is_retrained_after_follow_ups_finish(self):
        old = {("slm-lora-baseline", "lora_report.json"): {"status": "complete_pending_manual_review"}}
        busy = decide_lora({**self.READY, "slm-lora-eval": "queued"}, old)
        self.assertEqual(busy["kind"], "wait")
        self.assertIn("lora_eval", busy["reason"])
        result = decide_lora({**self.READY, "slm-lora-eval": "complete"}, old)
        self.assertEqual((result["kind"], result["stage"]), ("push", "lora"))
        self.assertEqual(decide_lora(self.READY)["kind"], "wait")  # report download failed
        failed = {("slm-lora-baseline", "lora_report.json"): {"status": "failed_nonfinite_loss"}}
        self.assertEqual(decide_lora(self.READY, failed)["kind"], "stop")

    def test_follow_ups_rerun_on_a_new_adapter(self):
        reports = {("slm-lora-baseline", "lora_report.json"): LORA_DONE, **SFT_DATA}
        self.assertEqual(decide_lora(self.READY, reports)["stage"], "lora_eval")
        statuses = {**self.READY, "slm-lora-eval": "running"}
        # distill_data follows the 1.7B adapter now, so the 360M chain leaves it alone even when it is missing.
        self.assertEqual(decide_lora(statuses, reports)["kind"], "wait")
        statuses["slm-lora-eval"] = "complete"
        reports[("slm-lora-eval", "lora_eval_report.json")] = {"adapter_sha256": "old"}
        self.assertEqual(decide_lora(statuses, reports)["stage"], "lora_eval")
        reports.update(FRESH_FOLLOW_UPS)
        self.assertEqual(decide_lora(statuses, reports)["kind"], "done")
        self.assertEqual(decide_lora({**statuses, "slm-lora-eval": "error"}, reports)["kind"], "stop")

    def test_adapters_retrain_when_sft_data_is_rebuilt(self):
        # A re-pushed sft_data (a changed split or holdout screen) leaves both adapters trained on the old rows.
        statuses = {**self.READY, "slm-lora-eval": "complete"}
        reports = {("slm-lora-baseline", "lora_report.json"): LORA_DONE, **FRESH_FOLLOW_UPS, **SFT_DATA}
        self.assertEqual(decide_lora(statuses, reports)["kind"], "done")
        rebuilt = {**reports, ("slm-sft-data", "sft_manifest.json"): {"files": {"train": {"sha256": "new"}}}}
        result = decide_lora(statuses, rebuilt)
        self.assertEqual((result["kind"], result["stage"]), ("push", "lora"))
        busy = decide_lora({**statuses, "slm-lora-eval": "running"}, rebuilt)
        self.assertEqual(busy["kind"], "wait")
        self.assertIn("lora_eval", busy["reason"])
        # A running distill_data mounts the 1.7B adapter, so the 360M one retrains without waiting for it.
        result = decide_lora({**statuses, "slm-distill-data": "running"}, rebuilt)
        self.assertEqual((result["kind"], result["stage"]), ("push", "lora"))
        unread = {key: value for key, value in reports.items() if key[0] != "slm-sft-data"}
        self.assertEqual(decide_lora(statuses, unread)["kind"], "wait")
        no_hash = {**reports, ("slm-sft-data", "sft_manifest.json"): {"files": {}}}
        self.assertEqual(decide_lora(statuses, no_hash)["kind"], "stop")
        large = {"slm-sft-data": "complete", **TEACHER_READY, "slm-distill-data": "complete"}
        large_reports = {**TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA}
        self.assertEqual(decide_large(large, large_reports)["kind"], "done")
        large_reports.update(rebuilt)
        result = decide_large(large, large_reports)
        self.assertEqual((result["kind"], result["stage"]), ("push", "lora_1b7"))
        for slug, stage in (("slm-lora-1b7-eval", "lora_1b7_eval"), ("slm-distill-data", "distill_data")):
            with self.subTest(running=stage):
                busy = decide_large({**large, slug: "running"}, large_reports)
                self.assertEqual(busy["kind"], "wait")
                self.assertIn(stage, busy["reason"])

    def test_round_shares_quota_and_stops_when_main_needs_a_stopped_lora_chain(self):
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete", "slm-sft-data": "complete"}
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): RESUME}
        main, side, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                      lambda slug, filename: reports.get((slug, filename)), 14.0)
        self.assertEqual((main["kind"], main["stage"]), ("push", "pretrain"))
        self.assertEqual((side["kind"], side["stage"]), ("wait", "lora"))  # 14 - 12 hours left
        self.assertEqual((large["kind"], large["stage"]), ("wait", "lora_1b7"))
        # sft waits on the 1.7B chain, so it takes the quota first and the 360M adapter waits for the reset.
        reports[("slm-160m-pretrain-1", "pretrain_session_1.json")] = DONE
        main, side, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                      lambda slug, filename: reports.get((slug, filename)), 9.0)
        self.assertEqual((main["kind"], main["needs"]), ("wait", "lora_1b7"))
        self.assertEqual((large["kind"], large["stage"]), ("push", "lora_1b7"))
        self.assertEqual((side["kind"], side["stage"]), ("wait", "lora"))  # 9 - 5.5 hours left
        # sft needs only the 1.7B chain, so a stopped 360M chain leaves the main chain waiting for it.
        statuses.update({"slm-160m-pretrain-1": "complete", "slm-lora-baseline": "error"})
        main, side, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                      lambda slug, filename: reports.get((slug, filename)), 30.0)
        self.assertEqual((main["kind"], main["needs"], side["kind"], large["stage"]),
                         ("wait", "lora_1b7", "stop", "lora_1b7"))
        statuses["slm-lora-1b7"] = "error"
        main, side, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                      lambda slug, filename: reports.get((slug, filename)), 30.0)
        self.assertEqual((main["kind"], large["kind"]), ("stop", "stop"))
        self.assertIn("1.7B LoRA chain stopped", main["reason"])


class LargeLoraChainTests(unittest.TestCase):
    READY = {"slm-sft-data": "complete", **TEACHER_READY}

    def test_waits_for_shared_sft_data_then_trains_evaluates_and_distills(self):
        self.assertEqual(decide_large({})["kind"], "wait")
        self.assertEqual(decide_large({"slm-sft-data": "error"})["kind"], "stop")
        statuses = {"slm-sft-data": "complete"}
        pushed = decide_large(statuses)
        self.assertEqual((pushed["kind"], pushed["stage"]), ("push", "lora_1b7"))
        self.assertEqual(decide_large(statuses, quota=4.0)["kind"], "wait")
        statuses["slm-lora-1b7"] = "complete"
        self.assertEqual(decide_large(statuses)["kind"], "wait")  # report not readable yet
        reports = {**TEACHER, **SFT_DATA}
        self.assertEqual(decide_large(statuses, reports)["stage"], "lora_1b7_eval")
        statuses["slm-lora-1b7-eval"] = "complete"
        reports[("slm-lora-1b7-eval", "lora_eval_report.json")] = {"adapter_sha256": "old"}
        self.assertEqual(decide_large(statuses, reports)["stage"], "lora_1b7_eval")
        reports[("slm-lora-1b7-eval", "lora_eval_report.json")] = {"adapter_sha256": LARGE_ADAPTER}
        distill = decide_large(statuses, reports)
        self.assertEqual((distill["kind"], distill["stage"]), ("push", "distill_data"))
        self.assertEqual(decide_large(statuses, reports, quota=1.0)["kind"], "wait")
        statuses["slm-distill-data"] = "complete"
        reports.update(FRESH_FOLLOW_UPS)
        done = decide_large(statuses, reports)
        self.assertEqual(done["kind"], "done")
        self.assertIn("lora_1b7_eval and distill_data", done["reason"])
        statuses["slm-lora-1b7"] = "error"
        self.assertEqual(decide_large(statuses, reports)["kind"], "stop")

    def test_distill_data_built_on_the_360m_adapter_is_rebuilt_on_the_1b7_one(self):
        # The filters_version 5 build came from the 360M adapter (177 of 252 exact); the 1.7B one scores 208.
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete", **self.READY,
                    "slm-lora-baseline": "complete", "slm-lora-eval": "complete", "slm-distill-data": "complete"}
        on_360m = {"teacher": {"adapter_sha256": ADAPTER}, "filters_version": DISTILL_FILTERS_VERSION}
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): DONE,
                   ("slm-lora-baseline", "lora_report.json"): LORA_DONE, **TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA,
                   ("slm-distill-data", "distill_manifest.json"): on_360m}
        main, side, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                      lambda slug, filename: reports.get((slug, filename)), 30.0)
        self.assertEqual((main["kind"], main["needs"]), ("wait", "lora_1b7"))
        self.assertEqual(side["kind"], "done")
        self.assertEqual((large["kind"], large["stage"]), ("push", "distill_data"))
        self.assertIn(LARGE_ADAPTER[:12], large["reason"])
        # Once it is rebuilt, an sft trained on the 360M answers runs again.
        reports.update(FRESH_FOLLOW_UPS)
        statuses["slm-160m-sft"] = "complete"
        reports[("slm-160m-sft", "sft_report.json")] = {"pretrain_session": 1, "sha256": "sft1",
                                                        "distill": {**DISTILLED, "teacher_adapter_sha256": ADAPTER}}
        result = decide(statuses, reports)
        self.assertEqual((result["kind"], result["stage"]), ("push", "sft"))
        self.assertIn("another LoRA adapter", result["reason"])
        reports[("slm-160m-sft", "sft_report.json")]["distill"] = DISTILLED
        self.assertEqual(decide(statuses, reports)["stage"], "eval")

    def test_sft_waits_for_a_teacher_trained_on_the_current_sft_data(self):
        # Pushing sft on answers distilled from the stale adapter would spend 9.5 GPU hours, then repeat.
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete", **self.READY,
                    "slm-distill-data": "complete"}
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): DONE, **TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA}
        self.assertEqual(decide(statuses, reports)["stage"], "sft")
        reports[("slm-sft-data", "sft_manifest.json")] = {"files": {"train": {"sha256": "new"}}}
        main, _, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                   lambda slug, filename: reports.get((slug, filename)), 30.0)
        self.assertEqual((main["kind"], main["needs"]), ("wait", "lora_1b7"))
        self.assertEqual((large["kind"], large["stage"]), ("push", "lora_1b7"))

    def test_a_failed_lora_1b7_eval_waits_for_the_distill_data_that_sft_needs(self):
        # Stopping at once would end --watch while distill_data, all that sft needs, was still running.
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete", **self.READY,
                    "slm-lora-1b7-eval": "error", "slm-distill-data": "running"}
        reports = {("slm-160m-pretrain-1", "pretrain_session_1.json"): DONE, **TEACHER, **FRESH_FOLLOW_UPS, **SFT_DATA}
        main, _, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                   lambda slug, filename: reports.get((slug, filename)), 30.0)
        self.assertEqual((main["kind"], large["kind"]), ("wait", "wait"))
        self.assertIn("lora_1b7_eval ended as error", large["reason"])
        statuses["slm-distill-data"] = "complete"
        main, _, large = run_pipeline.decide_round(lambda slug: statuses.get(slug, "missing"),
                                                   lambda slug, filename: reports.get((slug, filename)), 30.0)
        self.assertEqual((main["kind"], main["stage"], large["kind"]), ("push", "sft", "stop"))


class KaggleReportTests(unittest.TestCase):
    def report(self, text):
        kaggle = run_pipeline.Kaggle("someone")

        def run(*args, check=True):
            # Stands in for `kaggle kernels output -p <dir>`, which downloads the file into that folder.
            (Path(args[args.index("-p") + 1]) / "lora_report.json").write_text(text)
            return ""

        kaggle.run = run
        return kaggle.report("slm-lora", "lora_report.json")

    def test_report_reads_a_downloaded_object(self):
        self.assertEqual(self.report('{"status": "complete"}'), {"status": "complete"})

    def test_truncated_or_non_object_report_reads_as_missing(self):
        # The watcher retries only RuntimeError and OSError, so a JSONDecodeError here would end the watch.
        for text in ("", '{"status": "comp', "[1, 2]"):
            with self.subTest(text=text):
                self.assertIsNone(self.report(text))

    def test_push_packages_the_session_the_decision_names(self):
        # Dropping session or pretrain_session would package session 1 again, or sft from the wrong checkpoint.
        kaggle, pushed = run_pipeline.Kaggle("someone"), []
        accepted = "Kernel version 3 successfully pushed.  Please check progress at https://www.kaggle.com/code/x"
        kaggle.run = lambda *args, check=True: pushed.append(args) or accepted
        for decision, expected in (({"stage": "pretrain", "session": 3}, ("pretrain", "someone", 3, None)),
                                   ({"stage": "sft", "pretrain_session": 6}, ("sft", "someone", None, 6))):
            with self.subTest(stage=decision["stage"]), mock.patch("compute.package.prepare") as prepare:
                self.assertEqual(kaggle.push(decision), accepted)
                stage, directory, owner, session, pretrain_session = prepare.call_args.args
                self.assertEqual((stage, owner, session, pretrain_session), expected)
                self.assertEqual(pushed[-1], ("kernels", "push", "-p", str(directory)))

    def test_a_rejected_push_raises(self):
        # kaggle 2.2.4 prints the server's error and exits 0, so the watcher must read the output.
        kaggle = run_pipeline.Kaggle("someone")
        kaggle.run = lambda *args, check=True: "Kernel push error: Maximum batch GPU session count reached"
        with mock.patch("compute.package.prepare"), self.assertRaisesRegex(RuntimeError, "was not accepted"):
            kaggle.push({"stage": "lora"})


class MainTests(unittest.TestCase):
    def test_dry_run_never_pushes(self):
        fake = mock.Mock()
        fake.quota_hours.return_value = 30.0
        fake.status.side_effect = lambda slug: "complete" if slug == "slm-160m-corpus" else "missing"
        with mock.patch.object(run_pipeline, "Kaggle", return_value=fake), mock.patch("builtins.print"):
            self.assertEqual(run_pipeline.main(["--owner", "someone", "--dry-run"]), 0)
        fake.push.assert_not_called()

    def test_push_happens_without_dry_run(self):
        fake = mock.Mock()
        fake.quota_hours.return_value = 30.0
        fake.status.side_effect = lambda slug: "complete" if slug == "slm-160m-corpus" else "missing"
        fake.push.return_value = "pushed"
        with mock.patch.object(run_pipeline, "Kaggle", return_value=fake), mock.patch("builtins.print"):
            run_pipeline.main(["--owner", "someone"])
        pushed = [call[0][0] for call in fake.push.call_args_list]
        self.assertEqual([(decision["stage"], decision.get("session")) for decision in pushed],
                         [("pretrain", 1), ("sft_data", None)])

    def test_the_1b7_chain_pushes_before_the_360m_chain(self):
        # Kaggle caps concurrent batch GPU sessions, so the chain sft waits on must take a free slot first.
        fake = mock.Mock()
        fake.quota_hours.return_value = 30.0
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "running", "slm-sft-data": "complete"}
        fake.status.side_effect = lambda slug: statuses.get(slug, "missing")
        fake.push.return_value = "pushed"
        with mock.patch.object(run_pipeline, "Kaggle", return_value=fake), mock.patch("builtins.print"):
            run_pipeline.main(["--owner", "someone"])
        self.assertEqual([call[0][0]["stage"] for call in fake.push.call_args_list], ["lora_1b7", "lora"])


    def test_watch_survives_transient_errors(self):
        fake = mock.Mock()
        fake.quota_hours.side_effect = [RuntimeError("Failed to resolve 'api.kaggle.com'"), 30.0]
        fake.status.side_effect = lambda slug: "error"
        with mock.patch.object(run_pipeline, "Kaggle", return_value=fake), mock.patch("builtins.print"), \
                mock.patch.object(run_pipeline.time, "sleep") as sleep:
            self.assertEqual(run_pipeline.main(["--owner", "someone", "--watch"]), 1)
        sleep.assert_called_once()

    def test_watch_gives_up_after_repeated_errors(self):
        fake = mock.Mock()
        # A finite list, so a broken cap ends in StopIteration: with sleep mocked, an endless side effect once
        # looped forever while the mocks recorded every call, and the test run grew to 32 GB of memory.
        fake.quota_hours.side_effect = [RuntimeError("down")] * (run_pipeline.MAX_CONSECUTIVE_FAILURES + 1)
        with mock.patch.object(run_pipeline, "Kaggle", return_value=fake), mock.patch("builtins.print"), \
                mock.patch.object(run_pipeline.time, "sleep"), self.assertRaises(RuntimeError):
            run_pipeline.main(["--owner", "someone", "--watch"])
        self.assertEqual(fake.quota_hours.call_count, run_pipeline.MAX_CONSECUTIVE_FAILURES)


if __name__ == "__main__":
    unittest.main()
