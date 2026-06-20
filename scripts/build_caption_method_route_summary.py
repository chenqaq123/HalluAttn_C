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
    claim_repair = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_claim_repair/claim_repair_metrics.json"
    )
    concise = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
    )["summary"]
    claim_preservation = claim_repair["preservation"]["summary"]

    all_mentions = summary_by_name(feasibility, "all_mentions")
    hallucinated = summary_by_name(feasibility, "hallucinated_mentions")
    selected = summary_by_name(feasibility, "tdev_selected_top_frac")
    selected_hall = summary_by_name(feasibility, "tdev_selected_hallucinated")

    lines: list[str] = []
    lines.extend(
        [
            "# Caption Method Route Summary",
            "",
            "This note is generated from saved caption-side TDEV prototype artifacts. It keeps the method decision aligned with the latest evidence: hard token suppression is technically feasible but behaviorally incomplete; the next publishable route is verifier-guided claim acceptance with faithful concise captioning and constrained repair or regeneration.",
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
            f"| sentence acceptance | {acceptance['num_examples']} | {f4(acceptance['chair']['accepted']['overall']['CHAIRi'])} | {acceptance['chair']['accepted']['total_hallucinated_mentions']} | {f2(acceptance['mean_accepted_words'])} | {f2(acceptance['mean_removed_words_by_acceptance'])} | strong hallucination reduction, but drops complete mixed sentences |",
            f"| claim-local repair | {claim_repair['num_examples']} | {f4(claim_repair['chair']['repaired']['overall']['CHAIRi'])} | {claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {f2(claim_repair['mean_repaired_words'])} | {f2(claim_repair['mean_removed_words_by_repair'])} | best smoke tradeoff: preserves safe sentence prefixes before unsupported clauses |",
            "",
            "## Concise-Faithfulness Audit",
            "",
            "| Metric | Value | Reading |",
            "|---|---:|---|",
            f"| retained vanilla grounded mentions | {pct(concise['accepted_retained_vanilla_grounded_rate'])} | accepted captions keep most supported object mentions |",
            f"| hallucination reduction vs gated | {pct(concise['accepted_hallucination_reduction_vs_gated'])} | accepted captions remove most gated hallucinated mentions |",
            f"| hallucination reduction vs vanilla | {pct(concise['accepted_hallucination_reduction_vs_vanilla'])} | accepted captions improve over the original generated captions |",
            f"| object mention retention vs gated | {pct(concise['accepted_object_mention_retention_vs_gated'])} | shorter but not object-empty |",
            f"| generic/empty accepted captions | {concise['generic_or_empty_accepted']} | no accepted caption is empty/generic under the audit threshold |",
            "",
            "## Claim-Local Repair Audit",
            "",
            "| Metric | Value | Reading |",
            "|---|---:|---|",
            f"| local repair actions | {claim_repair['num_local_repair_actions']} | complete sentence prefixes saved before unsupported clauses |",
            f"| retained vanilla grounded mentions | {pct(claim_preservation['repaired_retained_vanilla_grounded_rate'])} | improves content retention over sentence acceptance |",
            f"| hallucination reduction vs gated | {pct(claim_preservation['repaired_hallucination_reduction_vs_gated'])} | keeps the same hallucination reduction as sentence acceptance |",
            f"| hallucination reduction vs vanilla | {pct(claim_preservation['repaired_hallucination_reduction_vs_vanilla'])} | improves over original generated captions |",
            f"| object mention retention vs gated | {pct(claim_preservation['repaired_object_mention_retention_vs_gated'])} | less destructive than sentence acceptance |",
            f"| generic/empty repaired captions | {claim_preservation['generic_or_empty_repaired']} | no repaired caption is empty/generic under the audit threshold |",
            "",
            "## Method Decision",
            "",
            "The practical method should now be framed as **TDEV-guided claim acceptance for faithful concise captioning with constrained local repair**, not as a pure token-ban decoder. The saved runs show that object claims are usually token-locatable and deny lists are narrow, but hard token suppression alone routes the model into new unsupported claims or incomplete fragments. Sentence acceptance catches those unsupported substitutes, which is exactly the target-vs-neighbor criterion we want. Its length reduction is not inherently bad: concise captions are preferable to long captions that keep inventing objects. The new claim-local repair smoke keeps the same hallucination reduction while preserving more grounded content by trimming only speculative or enumerating clauses when a safe prefix remains. The remaining risk is scaling this beyond five high-risk images and replacing deterministic clause trims with a controlled repair/regeneration step when the safe prefix is not enough.",
            "",
            "The next implementation target is therefore:",
            "",
            "1. Generate or keep a candidate sentence/span.",
            "2. Extract object-like claims, including open-vocabulary route forms.",
            "3. Map each claim to a canonical target when possible and score target-vs-neighbor evidence.",
            "4. Accept supported claims and reject unsupported claims; allow the caption to become shorter when unsupported detail is the only thing being removed.",
            "5. Apply constrained local repair only when rejection would remove central visible content or when an unsupported claim sits in a detachable clause.",
            "6. Report CHAIR together with concise-faithfulness metrics: retained supported objects, mean words, object mentions, empty/generic-caption rate, and manual examples.",
            "",
            "This preserves the paper's motivation: the method does not merely make object claims less frequent, and it does not rely on visual routing as proof. It explicitly tests whether the candidate claim is target-discriminative under related evidence. The desired behavior is not maximum caption length; it is concise but faithful captioning that keeps supported visual content and stops before unsupported object invention.",
            "",
            "## Paper-Safe Scope",
            "",
            "Current evidence supports a diagnostic-plus-verification paper with a bounded caption-side prototype. It does not yet support claiming a complete end-to-end caption mitigation method. For ICML, the strongest practical path is to scale the same generated-caption experiment beyond the 5-image smoke set, comparing hard gate, sentence repair, sentence acceptance, and claim-local repair on the same high-risk image set, judged by hallucination reduction and whether concise captions still preserve the main supported scene content.",
            "",
            "## Source Artifacts",
            "",
            "- `detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_repair/sentence_repair_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_claim_repair/claim_repair_metrics.json`",
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
