#!/usr/bin/env python3
"""Positive control for the region-token mapping of the paired experiment.

For a fixed set of frozen samples this runs one extra query per sample that
names the related object R, which IS present in the image. If the 24x24
token mapping is correct, attention from the R query row should concentrate
on the mapped region M_r far more than on its row-column transpose, a
size-matched random region, or the region under the unrelated absent-object
query from the main extraction. This validates mapping orientation and gives
a descriptive present-object reference; it is not part of the main result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run_paired_absent_attention_extract_20261113 as runner


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--image-root', default=None,
                        help='Directory holding the val2017 jpgs; overrides frozen paths')
    parser.add_argument('--extraction', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--per-type', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--min-free-gb', type=float, default=16.0)
    parser.add_argument('--wait-minutes', type=float, default=0.0)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration, \
        LlavaProcessor as LlavaProcessorClass

    frozen = read(args.samples)
    extraction = Path(args.extraction)
    state = read(extraction / 'extract_state.json')
    if not state.get('complete'):
        raise ValueError('Extraction state is not complete; run the main extraction first')
    masses = np.load(extraction / 'extract_masses.npz')
    order = {int(v): i for i, v in enumerate(masses['sample_ids'])}

    rng = np.random.default_rng(args.seed)
    grid = runner.GRID
    chosen = []
    for type_name in ('contextual_co_occurrence', 'category_similarity'):
        pool = [s for s in frozen['samples'] if s['type'] == type_name]
        chosen.extend(pool[::max(1, len(pool) // args.per_type)][:args.per_type])

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    deadline = time.monotonic() + args.wait_minutes * 60
    while free_bytes < int(args.min_free_gb * 1e9):
        if time.monotonic() >= deadline:
            raise ValueError('GPU memory gate not satisfied for validation run')
        time.sleep(60)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16,
        attn_implementation='sdpa', low_cpu_mem_usage=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                              local_files_only=True, use_fast=True)
    image_processor = CLIPImageProcessor.from_pretrained(args.model, local_files_only=True)
    processor = LlavaProcessorClass(tokenizer=tokenizer, image_processor=image_processor)
    probe = runner.RowMassProbe(model)
    layers = len(probe.layers)

    def aggregate(region_indices):
        """Primary-statistic fraction with an ad-hoc region (mean over 32x32)."""
        probe.region = torch.tensor([visual[i] for i in region_indices], dtype=torch.long)
        rows = [probe.record[l][0].mean() for l in range(layers)]
        region_rows = [probe.record[l][1].mean() for l in range(layers)]
        return float(np.mean(region_rows) / np.mean(rows))

    records = []
    for sample in chosen:
        image_path = (Path(args.image_root) / Path(sample['image_file']).name
                      if args.image_root else Path(sample['image_file']))
        with Image.open(image_path) as image:
            prompt = runner.QUESTION_TEMPLATE.format(
                name=sample['R'], article=runner.article(sample['R']))
            raw = processor(text=prompt, images=image.convert('RGB'), return_tensors='pt')
        inputs = {key: value.to(device) for key, value in raw.items()}
        sequence = inputs['input_ids'][0].tolist()
        image_positions = [i for i, t in enumerate(sequence) if t == runner.IMAGE_TOKEN_ID]
        if len(image_positions) != 1:
            raise ValueError('Expected exactly one <image> token')
        visual = list(range(image_positions[0], image_positions[0] + runner.VISUAL_TOKENS))
        _, expanded_row = runner.locate_noun_row(tokenizer, inputs['input_ids'],
                                                 sample['R'], image_positions[0])
        probe.visual = torch.tensor(visual, dtype=torch.long)
        probe.expected_length = len(sequence) + runner.EXPANSION
        probe.rows = torch.tensor([expanded_row], dtype=torch.long, device=device)
        probe.record = {}
        model(**inputs, use_cache=False)
        indices = sample['region']['token_indices']
        transposed = sorted({(runner.GRID - 1 - i % runner.GRID) * runner.GRID
                             + i // runner.GRID for i in indices})
        size = len(indices)
        pool = [i for i in range(grid * grid) if i not in set(indices)]
        random_region = sorted(rng.choice(pool, size=size, replace=False).tolist())
        fraction_related_query = aggregate(indices)
        fraction_transposed = aggregate(transposed)
        fraction_random = aggregate(random_region)

        index = order[sample['sample_id']]
        rel_total = masses['rel_total'][index].mean()
        rel_region = masses['rel_region'][index].mean()
        unrel_total = masses['unrel_total'][index].mean()
        unrel_region = masses['unrel_region'][index].mean()
        records.append(dict(
            sample_id=sample['sample_id'], type=sample['type'], T=sample['T'],
            R=sample['R'], U=sample['U'], region_size=size,
            A_present_R=100.0 * fraction_related_query,
            A_present_R_transposed=100.0 * fraction_transposed,
            A_present_R_random=100.0 * fraction_random,
            A_absent_T_from_extraction=100.0 * rel_region / rel_total,
            A_absent_U_from_extraction=100.0 * unrel_region / unrel_total))
    probe.restore()

    summary = {}
    for type_name in ('contextual_co_occurrence', 'category_similarity'):
        subset = [r for r in records if r['type'] == type_name]
        if not subset:
            continue
        summary[type_name] = dict(
            n=len(subset),
            median_A_present_R=float(np.median([r["A_present_R"] for r in subset])),
            median_A_present_R_transposed=float(
                np.median([r['A_present_R_transposed'] for r in subset])),
            median_A_present_R_random=float(
                np.median([r['A_present_R_random'] for r in subset])),
            median_A_absent_T=float(np.median([r['A_absent_T_from_extraction'] for r in subset])),
            median_A_absent_U=float(np.median([r['A_absent_U_from_extraction'] for r in subset])),
            present_beats_transposed=int(sum(r['A_present_R'] > r['A_present_R_transposed']
                                             for r in subset)),
            present_beats_random=int(sum(r['A_present_R'] > r['A_present_R_random']
                                         for r in subset)))
    output = dict(schema='paired-absent-attention-region-validation-v1', seed=args.seed,
                  per_type=args.per_type, summary=summary, records=records)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(output, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
