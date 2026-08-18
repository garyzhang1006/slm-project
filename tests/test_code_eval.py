import unittest

from cognition_slm.code_eval import assess_code, assess_python, extract_python, python_syntax_valid


class CodeEvaluationTests(unittest.TestCase):
    def test_extract_python_prefers_largest_fenced_block(self):
        text = "note\n```python\ndef short():\n    return 1\n```\n```python\ndef long():\n    return 2\n\n# detail\n```"
        self.assertIn("def long", extract_python(text))
        self.assertNotIn("def short", extract_python(text))

    def test_assess_python_scores_syntax_and_required_function(self):
        result = assess_python("```python\ndef add(a, b):\n    return a + b\n```", "def add(a, b): return a + b")
        self.assertTrue(result.syntax_valid)
        self.assertEqual(result.required_symbol_recall, 1.0)
        self.assertEqual(result.static_score, 1.0)

    def test_assess_python_rejects_invalid_syntax_without_execution(self):
        result = assess_python("def add(a, b)\n    return a + b", "def add(a, b): return a + b")
        self.assertFalse(result.syntax_valid)
        self.assertEqual(result.static_score, 0.0)
        self.assertTrue(result.error.startswith("SyntaxError:"))

    def test_prose_tasks_are_not_treated_as_code(self):
        self.assertIsNone(assess_code("explanation", "explanation", "code_explanation"))

    def test_python_syntax_check_ignores_code_fences(self):
        self.assertTrue(python_syntax_valid("```python\nreturn_value = 3\n```"))
        self.assertFalse(python_syntax_valid("```python\nreturn\n```"))


if __name__ == "__main__":
    unittest.main()
