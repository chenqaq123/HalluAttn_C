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


def atomic_variant(
    label: str,
    reparse_dir: str,
    select_dir: str,
    preservation_dir: str,
    content_dir: str,
    gain_dir: str,
) -> dict[str, Any]:
    reparse = read_json(f"{reparse_dir}/atomic_detail_reparse_metrics.json")
    select = read_json(f"{select_dir}/verified_atomic_detail_selection_metrics.json")
    preservation = read_json(f"{preservation_dir}/caption_variant_preservation_metrics.json")["summary"]
    content = read_json(f"{content_dir}/caption_content_light_metrics.json")["summary"]
    gain = read_json(f"{gain_dir}/atomic_detail_gain_metrics.json")
    return {
        "label": label,
        "reparse": reparse,
        "select": select,
        "preservation": preservation,
        "content": content,
        "raw_gain": gain["raw_augmented_vs_repaired"]["summary"],
        "verified_gain": gain["verified_selected_vs_repaired"]["summary"],
        "gain_counts": gain["selection_counts"],
    }


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
    scaled_acceptance = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
    )
    scaled_concise = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
    )["summary"]
    scaled_claim_repair = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_claim_repair/claim_repair_metrics.json"
    )
    scaled_claim_preservation = scaled_claim_repair["preservation"]["summary"]
    scaled_acceptance_content = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json"
    )["summary"]
    scaled_claim_content = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json"
    )["summary"]
    expanded_acceptance = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
    )
    expanded_concise = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
    )["summary"]
    expanded_claim_repair = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_claim_repair/claim_repair_metrics.json"
    )
    expanded_claim_preservation = expanded_claim_repair["preservation"]["summary"]
    expanded_acceptance_content = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json"
    )["summary"]
    expanded_claim_content = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json"
    )["summary"]
    broad_acceptance = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
    )
    broad_concise = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
    )["summary"]
    broad_claim_repair = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair/claim_repair_metrics.json"
    )
    broad_claim_preservation = broad_claim_repair["preservation"]["summary"]
    broad_acceptance_content = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json"
    )["summary"]
    broad_claim_content = read_json(
        "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json"
    )["summary"]
    regen_concise = read_json(
        "detection/baselines/results/tdev_caption_controlled_regen_100_t80/controlled_regeneration_metrics.json"
    )
    regen_concise_preservation = read_json(
        "detection/baselines/results/tdev_caption_controlled_regen_100_t80_preservation/caption_variant_preservation_metrics.json"
    )["summary"]
    regen_concise_content = read_json(
        "detection/baselines/results/tdev_caption_controlled_regen_100_t80_content_light/caption_content_light_metrics.json"
    )["summary"]
    regen_detail = read_json(
        "detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_metrics.json"
    )
    regen_detail_preservation = read_json(
        "detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_preservation/caption_variant_preservation_metrics.json"
    )["summary"]
    regen_detail_content = read_json(
        "detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_content_light/caption_content_light_metrics.json"
    )["summary"]
    oracle = read_json(
        "detection/baselines/results/tdev_caption_candidate_pool_oracle_100/candidate_pool_oracle_metrics.json"
    )["summaries"]
    oracle_min = oracle["min_hallucination"]
    oracle_no_worse = oracle["no_worse_than_repair"]
    oracle_min_content = read_json(
        "detection/baselines/results/tdev_caption_candidate_pool_oracle_100_min_hallucination_content_light/caption_content_light_metrics.json"
    )["summary"]
    oracle_no_worse_content = read_json(
        "detection/baselines/results/tdev_caption_candidate_pool_oracle_100_no_worse_content_light/caption_content_light_metrics.json"
    )["summary"]
    verified_select = read_json(
        "detection/baselines/results/tdev_caption_verified_candidate_select_100/verified_candidate_selection_metrics.json"
    )
    verified_select_preservation = read_json(
        "detection/baselines/results/tdev_caption_verified_candidate_select_100_preservation/caption_variant_preservation_metrics.json"
    )["summary"]
    verified_select_content = read_json(
        "detection/baselines/results/tdev_caption_verified_candidate_select_100_content_light/caption_content_light_metrics.json"
    )["summary"]
    verified_select_r10 = read_json(
        "detection/baselines/results/tdev_caption_verified_candidate_select_100_r10/verified_candidate_selection_metrics.json"
    )
    verified_select_r10_preservation = read_json(
        "detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_preservation/caption_variant_preservation_metrics.json"
    )["summary"]
    verified_select_r10_content = read_json(
        "detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_content_light/caption_content_light_metrics.json"
    )["summary"]
    local_add = read_json(
        "detection/baselines/results/tdev_caption_verified_local_additions_100/verified_local_addition_metrics.json"
    )
    local_add_preservation = read_json(
        "detection/baselines/results/tdev_caption_verified_local_additions_100_preservation/caption_variant_preservation_metrics.json"
    )["summary"]
    local_add_content = read_json(
        "detection/baselines/results/tdev_caption_verified_local_additions_100_content_light/caption_content_light_metrics.json"
    )["summary"]
    local_add_o04 = read_json(
        "detection/baselines/results/tdev_caption_verified_local_additions_100_o04/verified_local_addition_metrics.json"
    )
    atomic_gen = read_json(
        "detection/baselines/results/tdev_caption_atomic_detail_gen_100/atomic_detail_generation_metrics.json"
    )
    atomic_select = read_json(
        "detection/baselines/results/tdev_caption_atomic_detail_select_100/verified_atomic_detail_selection_metrics.json"
    )
    atomic_select_preservation = read_json(
        "detection/baselines/results/tdev_caption_atomic_detail_select_100_preservation/caption_variant_preservation_metrics.json"
    )["summary"]
    atomic_select_content = read_json(
        "detection/baselines/results/tdev_caption_atomic_detail_select_100_content_light/caption_content_light_metrics.json"
    )["summary"]
    atomic_gain = read_json(
        "detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100/atomic_detail_gain_metrics.json"
    )
    atomic_raw_gain = atomic_gain["raw_augmented_vs_repaired"]["summary"]
    atomic_verified_gain = atomic_gain["verified_selected_vs_repaired"]["summary"]
    atomic_gain_counts = atomic_gain["selection_counts"]
    atomic_variants = [
        atomic_variant(
            "o0.50",
            "detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o050",
            "detection/baselines/results/tdev_caption_atomic_detail_select_100_o050",
            "detection/baselines/results/tdev_caption_atomic_detail_select_100_o050_preservation",
            "detection/baselines/results/tdev_caption_atomic_detail_select_100_o050_content_light",
            "detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100_o050",
        ),
        {
            "label": "o0.65",
            "reparse": {
                "num_examples": atomic_gen["num_examples"],
                "images_with_additions": atomic_gen["images_with_additions"],
                "total_additions": atomic_gen["total_additions"],
                "mean_words": atomic_gen["mean_words"],
            },
            "select": atomic_select,
            "preservation": atomic_select_preservation,
            "content": atomic_select_content,
            "raw_gain": atomic_raw_gain,
            "verified_gain": atomic_verified_gain,
            "gain_counts": atomic_gain_counts,
        },
        atomic_variant(
            "o0.85",
            "detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o085",
            "detection/baselines/results/tdev_caption_atomic_detail_select_100_o085",
            "detection/baselines/results/tdev_caption_atomic_detail_select_100_o085_preservation",
            "detection/baselines/results/tdev_caption_atomic_detail_select_100_o085_content_light",
            "detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100_o085",
        ),
    ]

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
            "## Scaled Smoke Checks",
            "",
            "| Scale | Prototype | Images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs gated | COCO-objectless / content-light | Reading |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| 20-image | t96 hard gate | {scaled_acceptance['num_examples']} | {f4(scaled_acceptance['chair']['gated']['overall']['CHAIRi'])} | {scaled_acceptance['chair']['gated']['total_hallucinated_mentions']} | {f2(scaled_acceptance['mean_gated_words'])} | -- | -- | -- | high-risk generated baseline |",
            f"| 20-image | sentence acceptance | {scaled_acceptance['num_examples']} | {f4(scaled_acceptance['chair']['accepted']['overall']['CHAIRi'])} | {scaled_acceptance['chair']['accepted']['total_hallucinated_mentions']} | {f2(scaled_acceptance['mean_accepted_words'])} | {pct(scaled_concise['accepted_retained_vanilla_grounded_rate'])} | {pct(scaled_concise['accepted_object_mention_retention_vs_gated'])} | {scaled_acceptance_content['chair_objectless']} / {scaled_acceptance_content['content_light']} | reduces hallucination but can delete too much |",
            f"| 20-image | claim-local repair | {scaled_claim_repair['num_examples']} | {f4(scaled_claim_repair['chair']['repaired']['overall']['CHAIRi'])} | {scaled_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {f2(scaled_claim_repair['mean_repaired_words'])} | {pct(scaled_claim_preservation['repaired_retained_vanilla_grounded_rate'])} | {pct(scaled_claim_preservation['repaired_object_mention_retention_vs_gated'])} | {scaled_claim_content['chair_objectless']} / {scaled_claim_content['content_light']} | preserves more supported content with the same hallucinated mention count |",
            f"| 40-image | t96 hard gate | {expanded_acceptance['num_examples']} | {f4(expanded_acceptance['chair']['gated']['overall']['CHAIRi'])} | {expanded_acceptance['chair']['gated']['total_hallucinated_mentions']} | {f2(expanded_acceptance['mean_gated_words'])} | -- | -- | -- | maximum available iter2-prefilter set |",
            f"| 40-image | sentence acceptance | {expanded_acceptance['num_examples']} | {f4(expanded_acceptance['chair']['accepted']['overall']['CHAIRi'])} | {expanded_acceptance['chair']['accepted']['total_hallucinated_mentions']} | {f2(expanded_acceptance['mean_accepted_words'])} | {pct(expanded_concise['accepted_retained_vanilla_grounded_rate'])} | {pct(expanded_concise['accepted_object_mention_retention_vs_gated'])} | {expanded_acceptance_content['chair_objectless']} / {expanded_acceptance_content['content_light']} | same hallucination count as repair but less content retention |",
            f"| 40-image | claim-local repair | {expanded_claim_repair['num_examples']} | {f4(expanded_claim_repair['chair']['repaired']['overall']['CHAIRi'])} | {expanded_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {f2(expanded_claim_repair['mean_repaired_words'])} | {pct(expanded_claim_preservation['repaired_retained_vanilla_grounded_rate'])} | {pct(expanded_claim_preservation['repaired_object_mention_retention_vs_gated'])} | {expanded_claim_content['chair_objectless']} / {expanded_claim_content['content_light']} | best larger-scale prototype tradeoff |",
            f"| 100-image | t96 hard gate | {broad_acceptance['num_examples']} | {f4(broad_acceptance['chair']['gated']['overall']['CHAIRi'])} | {broad_acceptance['chair']['gated']['total_hallucinated_mentions']} | {f2(broad_acceptance['mean_gated_words'])} | -- | -- | -- | broader high-risk generated baseline |",
            f"| 100-image | sentence acceptance | {broad_acceptance['num_examples']} | {f4(broad_acceptance['chair']['accepted']['overall']['CHAIRi'])} | {broad_acceptance['chair']['accepted']['total_hallucinated_mentions']} | {f2(broad_acceptance['mean_accepted_words'])} | {pct(broad_concise['accepted_retained_vanilla_grounded_rate'])} | {pct(broad_concise['accepted_object_mention_retention_vs_gated'])} | {broad_acceptance_content['chair_objectless']} / {broad_acceptance_content['content_light']} | strong hallucination reduction but visible-content loss remains |",
            f"| 100-image | claim-local repair | {broad_claim_repair['num_examples']} | {f4(broad_claim_repair['chair']['repaired']['overall']['CHAIRi'])} | {broad_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {f2(broad_claim_repair['mean_repaired_words'])} | {pct(broad_claim_preservation['repaired_retained_vanilla_grounded_rate'])} | {pct(broad_claim_preservation['repaired_object_mention_retention_vs_gated'])} | {broad_claim_content['chair_objectless']} / {broad_claim_content['content_light']} | slight CHAIR/content gain over acceptance; content-light cases are removed |",
            "",
            "The scaled checks support the direction but narrow the claim. On the 100-image high-risk set, sentence acceptance and claim-local repair both reduce hard-gated hallucinated mentions from 73 to 27. Claim-local repair is still slightly better than sentence acceptance: CHAIRi 0.0600 vs. 0.0622, mean words 50.74 vs. 49.14, retained vanilla grounded mentions 73.47% vs. 70.68%, and content-light cases 0 vs. 1. The old CHAIR-objectless counts, 3 vs. 4, mostly reflect non-COCO but descriptive objects such as roads, signs, poles, or watercraft. However, the repair gain is now modest and object retention vs. vanilla remains only 64.01%. This is useful caption-side prototype evidence, not a complete caption mitigation result.",
            "",
            "## Controlled Regeneration Probe",
            "",
            "| Prototype | Images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| claim-local repair | {broad_claim_repair['num_examples']} | {f4(broad_claim_repair['chair']['repaired']['overall']['CHAIRi'])} | {broad_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {f2(broad_claim_repair['mean_repaired_words'])} | {pct(broad_claim_preservation['repaired_retained_vanilla_grounded_rate'])} | {pct(broad_claim_preservation['repaired_object_mention_retention_vs_vanilla'])} | {broad_claim_content['content_light']} | current best deterministic fallback |",
            f"| controlled regen concise | {regen_concise['num_examples']} | {f4(regen_concise['chair']['regenerated']['overall']['CHAIRi'])} | {regen_concise['chair']['regenerated']['total_hallucinated_mentions']} | {f2(regen_concise['mean_words']['regenerated'])} | {pct(regen_concise_preservation['variant_retained_vanilla_grounded_rate'])} | {pct(regen_concise_preservation['variant_object_mention_retention_vs_vanilla'])} | {regen_concise_content['content_light']} | lowers CHAIR by over-compressing content |",
            f"| controlled regen detail | {regen_detail['num_examples']} | {f4(regen_detail['chair']['regenerated']['overall']['CHAIRi'])} | {regen_detail['chair']['regenerated']['total_hallucinated_mentions']} | {f2(regen_detail['mean_words']['regenerated'])} | {pct(regen_detail_preservation['variant_retained_vanilla_grounded_rate'])} | {pct(regen_detail_preservation['variant_object_mention_retention_vs_vanilla'])} | {regen_detail_content['content_light']} | recovers length but not the repair tradeoff |",
            "",
            "Prompt-only regeneration is therefore not the solution. The concise prompt reduces CHAIRi by collapsing to short safe captions, while the detail-preserving prompt restores some length but has worse CHAIRi and lower grounded-content retention than claim-local repair. The next method should be verification-in-loop regeneration or candidate selection: regenerate only missing visible details, verify each new claim against target-vs-neighbor evidence, and keep the deterministic repaired caption as a fallback.",
            "",
            "## Candidate Pool Oracle",
            "",
            "This is an upper-bound analysis, not a deployable method: it uses CHAIR labels to choose among the already generated claim-local repair, concise regeneration, and detail regeneration candidates. It asks whether the current candidate pool contains useful alternatives that a future verifier could select without hallucination growth.",
            "",
            "| Selector | Images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Selected candidates | Reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            f"| oracle min hallucination | {oracle_min['num_images']} | {f4(oracle_min['chairi'])} | {oracle_min['total_hallucinated_mentions']} | {f2(oracle_min['mean_words'])} | {pct(oracle_min['retained_vanilla_grounded_rate'])} | {pct(oracle_min['object_mention_retention_vs_vanilla'])} | {oracle_min_content['content_light']} | repair {oracle_min['selected_candidate_counts'].get('claim-local repair', 0)}, concise {oracle_min['selected_candidate_counts'].get('controlled regen concise', 0)}, detail {oracle_min['selected_candidate_counts'].get('controlled regen detail', 0)} | best possible hallucination control still costs visible content |",
            f"| oracle no-worse-than-repair | {oracle_no_worse['num_images']} | {f4(oracle_no_worse['chairi'])} | {oracle_no_worse['total_hallucinated_mentions']} | {f2(oracle_no_worse['mean_words'])} | {pct(oracle_no_worse['retained_vanilla_grounded_rate'])} | {pct(oracle_no_worse['object_mention_retention_vs_vanilla'])} | {oracle_no_worse_content['content_light']} | repair {oracle_no_worse['selected_candidate_counts'].get('claim-local repair', 0)}, concise {oracle_no_worse['selected_candidate_counts'].get('controlled regen concise', 0)}, detail {oracle_no_worse['selected_candidate_counts'].get('controlled regen detail', 0)} | small upper-bound gain: detail candidates help 20 images but do not change the conclusion |",
            "",
            "The no-worse-than-repair oracle is the relevant upper bound for a future verifier. It keeps hallucinated mentions at 27, improves CHAIRi only from 0.0600 to 0.0586, and raises retained vanilla grounded mentions from 73.47% to 75.22%. This means the existing regeneration candidates contain some recoverable detail, but the gain is too small to justify a selector-only paper claim. The next method needs better candidate generation plus target-vs-neighbor claim verification.",
            "",
            "## Verified Candidate Selection Probe",
            "",
            "Unlike the oracle above, this probe does not use CHAIR labels for selection. It accepts detail-regenerated captions only when the repaired-to-candidate closed-loop audit finds no unsupported introduced target/open-vocabulary claim, with a repaired-caption fallback.",
            "",
            "| Selector | Images | Accepted detail | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| verified selector r0.75 | {verified_select['num_examples']} | {verified_select['selected_detail_regen']} | {f4(verified_select['chair']['selected']['overall']['CHAIRi'])} | {verified_select['chair']['selected']['total_hallucinated_mentions']} | {f2(verified_select['mean_words']['selected'])} | {pct(verified_select_preservation['variant_retained_vanilla_grounded_rate'])} | {pct(verified_select_preservation['variant_object_mention_retention_vs_vanilla'])} | {verified_select_content['content_light']} | verifier admits too many compressed candidates; worse than repair |",
            f"| verified selector r1.00 | {verified_select_r10['num_examples']} | {verified_select_r10['selected_detail_regen']} | {f4(verified_select_r10['chair']['selected']['overall']['CHAIRi'])} | {verified_select_r10['chair']['selected']['total_hallucinated_mentions']} | {f2(verified_select_r10['mean_words']['selected'])} | {pct(verified_select_r10_preservation['variant_retained_vanilla_grounded_rate'])} | {pct(verified_select_r10_preservation['variant_object_mention_retention_vs_vanilla'])} | {verified_select_r10_content['content_light']} | stricter guard collapses to repair-level behavior |",
            "",
            "The deployable-style selector confirms the oracle warning. A loose word-ratio guard accepts 40 detail candidates and worsens CHAIR/content retention. A strict r1.00 guard accepts only 10 detail candidates and is statistically indistinguishable from claim-local repair: CHAIRi 0.0599 vs. 0.0600 and retained vanilla grounded 73.65% vs. 73.47%. The bottleneck is not just selection; the candidate generator must produce claim-local additions that preserve detail without rewriting away supported content.",
            "",
            "## Verified Local-Addition Probe",
            "",
            "This probe keeps the repaired caption and only appends low-overlap sentences from a detail-regenerated candidate whose introduced claims pass the closed-loop target-vs-neighbor audit. It tests whether whole-caption regeneration can serve as a source of local detail spans.",
            "",
            "| Selector | Images | Images with additions | Added sentences | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| local additions o0.55 | {local_add['num_examples']} | {local_add['images_with_additions']} | {local_add['total_added_sentences']} | {f4(local_add['chair']['selected']['overall']['CHAIRi'])} | {local_add['chair']['selected']['total_hallucinated_mentions']} | {f2(local_add['mean_words']['selected'])} | {pct(local_add_preservation['variant_retained_vanilla_grounded_rate'])} | {pct(local_add_preservation['variant_object_mention_retention_vs_vanilla'])} | {local_add_content['content_light']} | appends mostly paraphrases; adds one hallucinated mention |",
            f"| local additions o0.40 | {local_add_o04['num_examples']} | {local_add_o04['images_with_additions']} | {local_add_o04['total_added_sentences']} | {f4(local_add_o04['chair']['selected']['overall']['CHAIRi'])} | {local_add_o04['chair']['selected']['total_hallucinated_mentions']} | {f2(local_add_o04['mean_words']['selected'])} | {pct(broad_claim_preservation['repaired_retained_vanilla_grounded_rate'])} | {pct(broad_claim_preservation['repaired_object_mention_retention_vs_vanilla'])} | {broad_claim_content['content_light']} | stricter overlap rejects all additions and returns to repair |",
            "",
            "The local-addition probe narrows the fix. Whole-caption regeneration does not yield clean missing-detail spans: a loose overlap threshold appends only 6 mostly redundant sentences, increases hallucinated mentions from 27 to 28, and only nudges retained vanilla grounded from 73.47% to 73.82%; a stricter threshold appends nothing. The next generator must be explicitly trained or prompted to propose atomic missing-detail claims/spans, not full paraphrased captions.",
            "",
            "## Atomic Detail Generation Probe",
            "",
            "This is the first positive caption-side prototype after the negative controls. LLaVA is prompted to produce at most two short new visible-detail sentences from the image and repaired draft. The closed-loop verifier then keeps augmented captions only when introduced claims pass target-vs-neighbor verification.",
            "",
            "| Prototype | Images | Accepted additions | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| raw atomic augmentation | {atomic_gen['num_examples']} | {atomic_gen['images_with_additions']} images / {atomic_gen['total_additions']} spans | {f4(atomic_gen['chair']['augmented']['overall']['CHAIRi'])} | {atomic_gen['chair']['augmented']['total_hallucinated_mentions']} | {f2(atomic_gen['mean_words']['augmented'])} | -- | -- | -- | adds detail but needs verifier; hallucinated mentions rise |",
            f"| verified atomic selection | {atomic_select['num_examples']} | {atomic_select['selected_atomic_additions']} images | {f4(atomic_select['chair']['selected']['overall']['CHAIRi'])} | {atomic_select['chair']['selected']['total_hallucinated_mentions']} | {f2(atomic_select['mean_words']['selected'])} | {pct(atomic_select_preservation['variant_retained_vanilla_grounded_rate'])} | {pct(atomic_select_preservation['variant_object_mention_retention_vs_vanilla'])} | {atomic_select_content['content_light']} | improves repair without increasing hallucinated mentions |",
            "",
            "Verified atomic selection changes the caption-side conclusion. Relative to claim-local repair, it keeps hallucinated mentions fixed at 27, improves CHAIRi from 0.0600 to 0.0558, raises mean words from 50.74 to 53.33, and raises retained vanilla grounded mentions from 73.47% to 76.79%. This is still a bounded 100-image high-risk prototype, but it is aligned with the paper motivation: looking is not enough, so generated details are only accepted when their object claims pass target-vs-neighbor verification.",
            "",
            "### Atomic Overlap Sensitivity",
            "",
            "This ablation re-parses the same raw atomic generations with different overlap thresholds before verifier selection. It checks whether the positive result depends on one hand-tuned duplicate filter.",
            "",
            "| Variant | Raw candidates | Accepted images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Delta grounded | Delta hallucinated | Content-light | Reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for variant in atomic_variants:
        select = variant["select"]
        reparse = variant["reparse"]
        preservation = variant["preservation"]
        content = variant["content"]
        verified_gain = variant["verified_gain"]
        if variant["label"] == "o0.50":
            reading = "conservative filter"
        elif variant["label"] == "o0.65":
            reading = "default filter"
        else:
            reading = "best current detail/faithfulness tradeoff"
        lines.append(
            f"| verified atomic {variant['label']} | {reparse['images_with_additions']} images / {reparse['total_additions']} spans | {select['selected_atomic_additions']} | {f4(select['chair']['selected']['overall']['CHAIRi'])} | {select['chair']['selected']['total_hallucinated_mentions']} | {f2(select['mean_words']['selected'])} | {pct(preservation['variant_retained_vanilla_grounded_rate'])} | {verified_gain['delta_grounded_mentions']} | {verified_gain['delta_hallucinated_mentions']} | {content['content_light']} | {reading} |"
        )
    lines.extend(
        [
            "",
            "The overlap sweep strengthens the claim without changing its scope. Stricter filtering still improves repair without adding hallucinated mentions, and the looser o0.85 variant accepts 46 images, adds 49 grounded mentions over repair, adds 0 hallucinated mentions, and raises retained vanilla grounded mentions to 77.49%. The result is still a high-risk 100-image prototype, not a full caption benchmark result.",
            "",
            "### Atomic Gain Decomposition",
            "",
            "| Gain audit | Raw augmented vs repair | Verified selected vs repair |",
            "|---|---:|---:|",
            f"| accepted / fallback images | -- | {atomic_gain_counts['accepted_atomic_addition_images']} / {atomic_gain_counts['repair_fallback_images']} |",
            f"| delta object mentions | {atomic_raw_gain['delta_object_mentions']} | {atomic_verified_gain['delta_object_mentions']} |",
            f"| delta grounded mentions | {atomic_raw_gain['delta_grounded_mentions']} | {atomic_verified_gain['delta_grounded_mentions']} |",
            f"| delta hallucinated mentions | {atomic_raw_gain['delta_hallucinated_mentions']} | {atomic_verified_gain['delta_hallucinated_mentions']} |",
            f"| images with more grounded mentions | {atomic_raw_gain['images_with_more_grounded']} | {atomic_verified_gain['images_with_more_grounded']} |",
            "",
            "The gain decomposition shows the verifier's role directly. Raw atomic spans add 59 grounded mentions but also 4 hallucinated mentions. Verified selection keeps 34 new grounded/object mentions across 25 images while adding 0 hallucinated mentions. The improvement is therefore not an empty-caption or repetition artifact; it is the verifier selecting useful atomic additions and rejecting risky ones.",
            "",
            "## Method Decision",
            "",
            "The practical method should now be framed as **TDEV-guided claim acceptance for faithful concise captioning with constrained local repair/regeneration**, not as a pure token-ban decoder. The saved runs show that object claims are usually token-locatable and deny lists are narrow, but hard token suppression alone routes the model into new unsupported claims or incomplete fragments. Sentence acceptance catches those unsupported substitutes, which is exactly the target-vs-neighbor criterion we want. Its length reduction is not inherently bad: concise captions are preferable to long captions that keep inventing objects; the risk is only when shortening removes supported visible content or truly collapses into content-light captions. The content-light audit shows that many CHAIR-objectless captions are still descriptive. The controlled-regeneration probe, candidate-pool oracle, deployable-style verified selector, and local-addition probe show why prompt-only rewriting plus selection is insufficient. The atomic-detail probe gives the current method direction: propose short missing-detail spans, verify each introduced claim against target-vs-neighbor evidence, and fall back to the deterministic repair when no safe new detail is found.",
            "",
            "The next implementation target is therefore:",
            "",
            "1. Generate or keep a candidate sentence/span.",
            "2. Extract object-like claims, including open-vocabulary route forms.",
            "3. Map each claim to a canonical target when possible and score target-vs-neighbor evidence.",
            "4. Accept supported claims and reject unsupported claims; allow the caption to become shorter when unsupported detail is the only thing being removed.",
            "5. Apply constrained local repair first; use regeneration only as a verified candidate source for missing visible details, and fall back to the repaired caption when regenerated claims fail target-vs-neighbor checks.",
            "6. Report CHAIR together with concise-faithfulness metrics: retained supported objects, mean words, object mentions, empty/generic-caption rate, and manual examples.",
            "",
            "This preserves the paper's motivation: the method does not merely make object claims less frequent, and it does not rely on visual routing as proof. It explicitly tests whether the candidate claim is target-discriminative under related evidence. The desired behavior is not maximum caption length; it is concise but faithful captioning that keeps supported visual content and stops before unsupported object invention.",
            "",
            "## Paper-Safe Scope",
            "",
            "Current evidence supports a diagnostic-plus-verification paper with a bounded caption-side prototype. It does not yet support claiming a complete end-to-end caption mitigation method. For ICML, the strongest practical path is verification-in-loop atomic detail generation, evaluated beyond the current 100-image high-risk set with CHAIR, content-light, and content-preservation audits. The prompt-only regeneration probe, candidate-pool oracle, verified selector, and local-addition probe should be reported as negative controls.",
            "",
            "## Source Artifacts",
            "",
            "- `detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_repair/sentence_repair_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_claim_repair/claim_repair_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96/gated_generation_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_audit_ov96/closed_loop_example_audit.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_claim_repair/claim_repair_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96/gated_generation_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_audit_ov96/closed_loop_example_audit.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_claim_repair/claim_repair_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter_100_examples/multi_image_prefilter_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_prefilter/iterative_prefilter_summary.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96/gated_generation_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_audit_ov96/closed_loop_example_audit.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair/claim_repair_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_controlled_regen_100_t80/controlled_regeneration_metrics.json`",
            "- `detection/baselines/results/tdev_caption_controlled_regen_100_t80_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_controlled_regen_100_t80_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_metrics.json`",
            "- `detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_candidate_pool_oracle_100/candidate_pool_oracle_metrics.json`",
            "- `detection/baselines/results/tdev_caption_candidate_pool_oracle_100_min_hallucination_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_candidate_pool_oracle_100_no_worse_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_expansion_100_audit/closed_loop_example_audit.json`",
            "- `detection/baselines/results/tdev_caption_verified_candidate_select_100/verified_candidate_selection_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_candidate_select_100_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_candidate_select_100_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_candidate_select_100_r10/verified_candidate_selection_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_local_additions_100/verified_local_addition_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_local_additions_100_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_local_additions_100_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_verified_local_additions_100_o04/verified_local_addition_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_gen_100/atomic_detail_generation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_gen_100_audit/closed_loop_example_audit.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100/verified_atomic_detail_selection_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100/atomic_detail_gain_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o050/atomic_detail_reparse_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o050_audit/closed_loop_example_audit.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o050/verified_atomic_detail_selection_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o050_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o050_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100_o050/atomic_detail_gain_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o085/atomic_detail_reparse_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o085_audit/closed_loop_example_audit.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o085/verified_atomic_detail_selection_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o085_preservation/caption_variant_preservation_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o085_content_light/caption_content_light_metrics.json`",
            "- `detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100_o085/atomic_detail_gain_metrics.json`",
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
