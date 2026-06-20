#!/usr/bin/env python3
"""Build a concise route summary for TDEV caption-side method decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(rel_path: str) -> Any:
    return json.loads((PROJECT_ROOT / rel_path).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def f2(value: float) -> str:
    return f"{value:.2f}"


def f4(value: float) -> str:
    return f"{value:.4f}"


def summary_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for row in payload["summaries"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


def build() -> str:
    feasibility = read_json("detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json")
    prefilter = read_json("detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_metrics.json")
    repair = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_repair/sentence_repair_metrics.json"
    )
    acceptance = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
    )

    all_mentions = summary_by_name(feasibility, "all_mentions")
    hallucinated = summary_by_name(feasibility, "hallucinated_mentions")
    selected = summary_by_name(feasibility, "tdev_selected_top_frac")
    selected_hall = summary_by_name(feasibility, "tdev_selected_hallucinated")

    lines: list[str] = []
    lines.extend(
        [
            "# Caption Method Route Summary",
            "",
            "This note is generated from saved caption-side TDEV prototype artifacts. It keeps the method decision aligned with the latest evidence: hard token suppression is technically feasible but behaviorally incomplete; the next publishable route is verifier-guided claim acceptance with constrained repair or regeneration.",
            "",
            "## Feasibility Evidence",
            "",
            "| Check | Value | Reading |",
            "|---|---:|---|",
            f"| CHAIR object mentions | {feasibility['num_mentions']} | cached mention-level scope |",
            f"| Near-position tokenizer match, all mentions | {pct(all_mentions['near_gen_pos_match_rate'])} | object claims can usually be found near generation position |",
            f"| Near-position tokenizer match, hallucinated mentions | {pct(hallucinated['near_gen_pos_match_rate'])} | hallucinated claims are also gateable |",
            f"| Near-position match, TDEV top-{pct(feasibility['top_frac'])} mentions | {pct(selected['near_gen_pos_match_rate'])} | high-risk claims remain token-locatable |",
            f"| Near-position match, selected hallucinated mentions | {pct(selected_hall['near_gen_pos_match_rate'])} | coverage is not the main bottleneck |",
            f"| Single-token COCO word fraction | {pct(feasibility['vocab_single_token_word_fraction'])} | multi-token phrase tracking is required |",
            "",
            "## Multi-Image Prefilter Evidence",
            "",
            "| Check | Value | Reading |",
            "|---|---:|---|",
            f"| Images selected | {prefilter['num_chosen_images']} | bounded GPU evaluation target |",
            f"| Selected image-word pairs | {prefilter['num_unique_image_word_pairs']} | small deny/verify workload |",
            f"| CHAIR hallucination precision | {pct(prefilter['selected_precision_hallucinated'])} | TDEV-selected claims are mostly true hallucinations |",
            f"| Hallucinated-mention coverage on selected images | {pct(prefilter['chosen_image_hallucination_coverage'])} | useful but incomplete coverage |",
            f"| Mean denied words per image | {f2(prefilter['unique_denied_words_per_image']['mean'])} | deny lists are narrow |",
            f"| P95 token sequences per image | {f2(prefilter['token_sequences_per_image']['p95'])} | token workload is bounded |",
            "",
            "## Generated-Caption Smoke Evidence",
            "",
            "| Prototype | Images | CHAIRi | Hallucinated mentions | Mean words | Removed words | Interpretation |",
            "|---|---:|---:|---:|---:|---:|---|",
            f"| t96 hard gate | {acceptance['num_examples']} | {f4(acceptance['chair']['gated']['overall']['CHAIRi'])} | {acceptance['chair']['gated']['total_hallucinated_mentions']} | {f2(acceptance['mean_gated_words'])} | 0.00 | hard gating creates substitute/escape claims |",
            f"| sentence repair | {repair['num_examples']} | {f4(repair['chair']['repaired']['overall']['CHAIRi'])} | {repair['chair']['repaired']['total_hallucinated_mentions']} | {f2(repair['mean_repaired_words'])} | {f2(repair['mean_removed_words_by_repair'])} | fixes incomplete tails but misses complete substitute claims |",
            f"| sentence acceptance | {acceptance['num_examples']} | {f4(acceptance['chair']['accepted']['overall']['CHAIRi'])} | {acceptance['chair']['accepted']['total_hallucinated_mentions']} | {f2(acceptance['mean_accepted_words'])} | {f2(acceptance['mean_removed_words_by_acceptance'])} | best hallucination reduction, but over-deletes content |",
            "",
            "## Method Decision",
            "",
            "The practical method should now be framed as **TDEV-guided claim acceptance**, not as a pure token-ban decoder. The saved runs show that object claims are usually token-locatable and deny lists are narrow, but hard token suppression alone routes the model into new unsupported claims or incomplete fragments. Sentence acceptance catches those unsupported substitutes, which is exactly the target-vs-neighbor criterion we want, but it removes too much text because it has no replacement generator.",
            "",
            "The next implementation target is therefore:",
            "",
            "1. Generate or keep a candidate sentence/span.",
            "2. Extract object-like claims, including open-vocabulary route forms.",
            "3. Map each claim to a canonical target when possible and score target-vs-neighbor evidence.",
            "4. Accept supported claims, reject unsupported claims, and ask for a constrained local repair only when rejection would delete useful content.",
            "5. Report both CHAIR and content-retention metrics; lower CHAIR alone is insufficient if mean words collapse.",
            "",
            "This preserves the paper's motivation: the method does not merely make object claims less frequent, and it does not rely on visual routing as proof. It explicitly tests whether the candidate claim is target-discriminative under related evidence.",
            "",
            "## Paper-Safe Scope",
            "",
            "Current evidence supports a diagnostic-plus-verification paper with a bounded caption-side prototype. It does not yet support claiming a complete end-to-end caption mitigation method. For ICML, the strongest practical path is a small generated-caption experiment that compares hard gate, sentence repair, sentence acceptance, and constrained repair on the same high-risk image set.",
            "",
            "## Source Artifacts",
            "",
            "- `detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_repair/sentence_repair_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/caption_method_route_summary.md")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
