from __future__ import annotations

import unittest

from mitigation.scripts.run_registered_throne_aqa import (
    PROMPT_FORMAT,
    throne_questions,
)


class RegisteredThroneAQATests(unittest.TestCase):
    def test_question_order_and_special_names_match_release(self) -> None:
        categories = [
            {"id": 3, "name": "Orange"},
            {"id": 1, "name": "tie"},
            {"id": 2, "name": "skis"},
        ]
        questions = throne_questions(categories)
        self.assertEqual(len(questions), 9)
        self.assertEqual(
            questions[0],
            "Please answer yes or no. Is there a necktie in this image?",
        )
        self.assertEqual(
            questions[1],
            "Please answer yes or no. Are there skis in this image?",
        )
        self.assertEqual(
            questions[2],
            "Please answer yes or no. Is there a Orange in this image?",
        )
        self.assertEqual(
            questions[3],
            "Is there a necktie in this image?",
        )
        self.assertEqual(
            questions[6],
            "Please answer yes or no. Does the text imply a necktie "
            "is in the image?",
        )

    def test_prompt_is_exact_released_layout(self) -> None:
        prompt = PROMPT_FORMAT.format(
            context="A cat sits.",
            question="Is there a cat in this image?",
        )
        self.assertEqual(
            prompt,
            "Text: A cat sits.\n\n"
            "Read the text and answer the question.\n\n"
            "Is there a cat in this image?",
        )


if __name__ == "__main__":
    unittest.main()
