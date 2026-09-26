from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import run_pipeline  # noqa: E402

RESUME = {"status": "session_complete_resume_next", "step_reached": 4570}
DONE = {"status": "complete", "step_reached": 22889}


def decide(statuses, reports=None, quota=30.0):
    reports = reports or {}
    return run_pipeline.next_action(lambda slug: statuses.get(slug, "missing"),
                                    lambda slug, filename: reports.get((slug, filename)), quota)


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

    def test_missing_report_stops(self):
        statuses = {"slm-160m-corpus": "complete", "slm-160m-pretrain-1": "complete"}
        self.assertEqual(decide(statuses)["kind"], "stop")

    def test_sft_then_eval_then_done(self):
        statuses = {"slm-160m-corpus": "complete", **{f"slm-160m-pretrain-{k}": "complete" for k in (1, 2, 3)}}
        reports = {("slm-160m-pretrain-3", "pretrain_session_3.json"): DONE}
        self.assertEqual(decide(statuses, reports)["kind"], "stop")  # no LoRA teacher yet
        statuses["slm-lora-baseline"] = "running"
        self.assertEqual(decide(statuses, reports)["kind"], "wait")
        statuses["slm-lora-baseline"] = "complete"
        self.assertEqual(decide(statuses, reports)["stage"], "distill_data")
        statuses["slm-distill-data"] = "complete"
        result = decide(statuses, reports)
        self.assertEqual((result["kind"], result["stage"], result["pretrain_session"]), ("push", "sft", 3))
        statuses["slm-160m-sft"] = "complete"
        stale = decide(statuses, {**reports, ("slm-160m-sft", "sft_report.json"): {"pretrain_session": 2}})
        self.assertEqual((stale["kind"], stale["stage"]), ("push", "sft"))
        reports[("slm-160m-sft", "sft_report.json")] = {"pretrain_session": 3}
        self.assertEqual(decide(statuses, reports)["stage"], "eval")
        statuses["slm-160m-eval"] = "running"
        self.assertEqual(decide(statuses, reports)["kind"], "wait")
        statuses["slm-160m-eval"] = "complete"
        self.assertEqual(decide(statuses, reports)["kind"], "done")


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
        self.assertEqual(fake.push.call_args[0][0]["session"], 1)


    def test_watch_survives_transient_errors(self):
        fake = mock.Mock()
        fake.quota_hours.side_effect = [RuntimeError("Failed to resolve 'api.kaggle.com'"), 30.0]
        fake.status.side_effect = lambda slug: "error" if slug == "slm-160m-corpus" else "missing"
        with mock.patch.object(run_pipeline, "Kaggle", return_value=fake), mock.patch("builtins.print"), \
                mock.patch.object(run_pipeline.time, "sleep") as sleep:
            self.assertEqual(run_pipeline.main(["--owner", "someone", "--watch"]), 1)
        sleep.assert_called_once()

    def test_watch_gives_up_after_repeated_errors(self):
        fake = mock.Mock()
        fake.quota_hours.side_effect = RuntimeError("down")
        with mock.patch.object(run_pipeline, "Kaggle", return_value=fake), mock.patch("builtins.print"), \
                mock.patch.object(run_pipeline.time, "sleep"), self.assertRaises(RuntimeError):
            run_pipeline.main(["--owner", "someone", "--watch"])
        self.assertEqual(fake.quota_hours.call_count, run_pipeline.MAX_CONSECUTIVE_FAILURES)


if __name__ == "__main__":
    unittest.main()
