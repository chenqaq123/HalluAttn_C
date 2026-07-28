from __future__ import annotations

import unittest

from mitigation.scripts.audit_throne_aqa_suite_parity import compare


class ThroneAQASuiteParityTests(unittest.TestCase):
    def test_exact_1095_answers_pass(self):
        answers = ["yes", *(["no"] * 1094)]
        result = compare({"7": answers}, {"7": list(answers)})
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["answers_compared"], 1095)

    def test_one_answer_difference_is_reported(self):
        first = ["no"] * 1095
        second = list(first)
        second[81] = "yes"
        result = compare({"7": first}, {"7": second})
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["answer_mismatches"], 1)

    def test_wrong_answer_count_fails(self):
        with self.assertRaisesRegex(ValueError, "1,095"):
            compare({"7": ["no"]}, {"7": ["no"]})


if __name__ == "__main__":
    unittest.main()
