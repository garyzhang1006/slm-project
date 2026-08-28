import unittest

from cognition_slm.context_therapy import ContextMessage, ContextTherapist, estimate_tokens, parse_messages


class ContextTherapyTests(unittest.TestCase):
    def test_stable_context_returns_continue_action(self):
        messages = [{"role": "user", "content": "Write a function and run a focused test."}]
        assessment = ContextTherapist().assess(messages, token_budget=1000)
        self.assertEqual(assessment.state, "stable")
        self.assertEqual(assessment.actions[0].code, "continue")
        self.assertGreater(assessment.estimated_tokens, 0)

    def test_budget_pressure_is_explicit_and_actionable(self):
        messages = [ContextMessage("user", "x" * 100)]
        estimated = estimate_tokens(messages)
        assessment = ContextTherapist().assess(messages, token_budget=estimated - 1)
        self.assertEqual(assessment.state, "overloaded")
        self.assertEqual(assessment.observations[0].code, "context_over_budget")
        self.assertEqual(assessment.actions[0].code, "compress_context")

    def test_repeated_turns_trigger_deduplication(self):
        text = "Preserve the current goal and run the focused test before claiming success."
        assessment = ContextTherapist().assess(
            [{"role": "user", "content": text}, {"role": "assistant", "content": text}]
        )
        self.assertEqual(assessment.state, "strained")
        self.assertIn("repeated_context", {item.code for item in assessment.observations})
        self.assertIn("deduplicate_context", {item.code for item in assessment.actions})

    def test_message_boundary_rejects_unknown_role_and_empty_context(self):
        with self.assertRaises(ValueError):
            parse_messages([{"role": "unknown", "content": "text"}])
        with self.assertRaises(ValueError):
            parse_messages([])
        self.assertEqual(parse_messages([ContextMessage(" USER ", "text")])[0].role, "user")

    def test_serialized_report_contains_no_hidden_reasoning_field(self):
        report = ContextTherapist().assess(
            [{"role": "user", "content": "Keep visible evidence only."}]
        ).to_dict()
        self.assertNotIn("chain_of_thought", report)
        self.assertNotIn("private_thoughts", report)


if __name__ == "__main__":
    unittest.main()
