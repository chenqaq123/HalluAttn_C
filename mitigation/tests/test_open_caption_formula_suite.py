from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mitigation.scripts.run_open_caption_formula_suite import validate_jobs


def args(root: Path, jobs: list[list[str]], **overrides):
    values = {
        "kind": "qwen25",
        "job": jobs,
        "allpath_heads": None,
        "pai_gate": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FormulaSuiteTests(unittest.TestCase):
    def test_rejects_duplicate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                ["vanilla", "a.jsonl", str(root / "same.jsonl")],
                ["diml", "b.jsonl", str(root / "same.jsonl")],
            ]
            with self.assertRaisesRegex(ValueError, "distinct output"):
                validate_jobs(args(root, jobs))

    def test_pai_requires_matching_active_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "kind": "qwen25",
                        "samples": 20,
                        "summary": {
                            "all_attention_ports_called": True,
                            "changed_outputs": 19,
                        },
                    }
                )
            )
            validate_jobs(
                args(
                    root,
                    [["pai", "a.jsonl", str(root / "pai.jsonl")]],
                    pai_gate=str(gate),
                )
            )

    def test_pai_rejects_zero_effect_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "kind": "qwen25",
                        "samples": 20,
                        "summary": {
                            "all_attention_ports_called": True,
                            "changed_outputs": 0,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "did not change"):
                validate_jobs(
                    args(
                        root,
                        [["pai", "a.jsonl", str(root / "pai.jsonl")]],
                        pai_gate=str(gate),
                    )
                )


if __name__ == "__main__":
    unittest.main()
