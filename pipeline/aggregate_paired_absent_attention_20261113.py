#!/usr/bin/env python3
"""Aggregate paired absent-object attention masses into the rebuttal table.

Reads the frozen sample list and the extracted per-(layer, head) masses,
computes the pre-registered paired statistic

    A(q, M_r) = sum_{v in M_r} a_q(v) / sum_{v in V} a_q(v)
    delta A   = A(q_related, M_r) - A(q_unrelated, M_r)

where a_q(v) averages the 32x32 layer-head attention map, then reports the
image-clustered bootstrap confidence interval per association type, a
per-pair breakdown, a per-layer breakdown, and the secondary
mean-of-per-head-fractions variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TYPE_LABELS = {
    'contextual_co_occurrence': 'Contextual Co-occurrence',
    'category_similarity': 'Category Similarity',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fractions(masses):
    """Per-sample region fraction of total visual mass, primary and variant."""
    total = masses['total']
    region = masses['region']
    primary = region.sum(axis=(1, 2)) / total.sum(axis=(1, 2))  # fraction of averaged map
    variant = (region / total).mean(axis=(1, 2))  # mean of per-(layer, head) fractions
    per_layer = region.sum(axis=2) / total.sum(axis=2)  # [N, layers]
    return primary, variant, per_layer


def paired_bootstrap(delta, groups, draws, seed):
    """Percentile CI for the mean of delta under image-clustered resampling."""
    rng = np.random.default_rng(seed)
    unique = {image: index for index, image in enumerate(groups)}
    cluster_values = {}
    for image, index in unique.items():
        members = [delta[i] for i, g in enumerate(groups) if g == image]
        cluster_values[image] = members
    images = sorted(cluster_values)
    means = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(len(images), size=len(images), replace=True)
        values = [v for p in picked for v in cluster_values[images[p]]]
        means[draw] = np.mean(values)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_test(delta, draws, seed):
    """Two-sided paired permutation (sign-flip) p-value; numpy only."""
    rng = np.random.default_rng(seed)
    observed = abs(np.mean(delta))
    values = np.asarray(delta)
    count = 0
    for _ in range(draws):
        signs = rng.choice(np.array([-1.0, 1.0]), size=values.shape[0])
        if abs(np.mean(values * signs)) >= observed:
            count += 1
    return (count + 1) / (draws + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--extraction', required=True,
                        help='Directory holding extract_masses.npz / extract_rows.jsonl / state')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--output-tex', required=True)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--test-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261113)
    args = parser.parse_args()

    frozen = read(args.samples)
    extraction = Path(args.extraction)
    state = read(extraction / 'extract_state.json')
    if not state.get('complete'):
        raise ValueError('Extraction state is not complete')
    if sha(args.samples) != state['signature']['samples_sha256']:
        raise ValueError('Sample list changed after extraction')
    masses = np.load(extraction / 'extract_masses.npz')
    rows = {int(json.loads(line)['sample_id']): json.loads(line)
            for line in (extraction / 'extract_rows.jsonl').read_text().splitlines() if line.strip()}

    samples = frozen['samples']
    if sorted(masses['sample_ids'].tolist()) != sorted(s['sample_id'] for s in samples):
        raise ValueError('Extracted sample ids do not cover the frozen sample list')
    order = {int(sample_id): index for index, sample_id in enumerate(masses['sample_ids'])}

    records = []
    for sample in samples:
        index = order[sample['sample_id']]
        entry = {key: masses[key][index] for key in
                 ('rel_region', 'rel_total', 'unrel_region', 'unrel_total')}
        rel_primary, rel_variant, rel_layer = fractions(
            dict(total=entry['rel_total'], region=entry['rel_region']))
        unrel_primary, unrel_variant, unrel_layer = fractions(
            dict(total=entry['unrel_total'], region=entry['unrel_region']))
        records.append(dict(
            sample_id=sample['sample_id'], pair_id=sample['pair_id'], type=sample['type'],
            image_id=sample['image_id'], T=sample['T'], R=sample['R'], U=sample['U'],
            region_size=sample['region']['size'],
            rows=rows[sample['sample_id']],
            A_rel=100.0 * float(rel_primary[0]), A_unrel=100.0 * float(unrel_primary[0]),
            delta_pp=100.0 * (float(rel_primary[0]) - float(unrel_primary[0])),
            variant_delta_pp=100.0 * (float(rel_variant[0]) - float(unrel_variant[0])),
            total_mass_rel=float(entry['rel_total'].mean()),
            total_mass_unrel=float(entry['unrel_total'].mean()),
            per_layer_delta_pp=100.0 * (rel_layer[0] - unrel_layer[0])))

    results = dict(schema='paired-absent-attention-results-v1',
                   inputs=dict(samples_sha256=sha(args.samples),
                               extraction=str(extraction), state=state),
                   model=frozen['model'], totals=frozen['totals'], types={})
    for type_name, label in TYPE_LABELS.items():
        subset = [r for r in records if r['type'] == type_name]
        if not subset:
            continue
        delta = np.array([r['delta_pp'] for r in subset])
        variant = np.array([r['variant_delta_pp'] for r in subset])
        images = [r['image_id'] for r in subset]
        low, high = paired_bootstrap(delta, images, args.bootstrap_draws, args.seed)
        low_v, high_v = paired_bootstrap(variant, images, args.bootstrap_draws, args.seed + 1)
        pairs = {}
        for record in subset:
            pairs.setdefault(record['pair_id'], []).append(record['delta_pp'])
        results['types'][type_name] = dict(
            label=label,
            n_samples=len(subset), n_images=len(set(images)),
            A_rel_mean_pp=float(np.mean([r['A_rel'] for r in subset])),
            A_unrel_mean_pp=float(np.mean([r['A_unrel'] for r in subset])),
            delta_mean_pp=float(delta.mean()),
            delta_ci95_pp=[low, high],
            variant_delta_mean_pp=float(variant.mean()),
            variant_delta_ci95_pp=[low_v, high_v],
            delta_positive_share=float((delta > 0).mean()),
            sign_flip_p=sign_flip_test(delta, args.test_draws, args.seed + 2),
            total_mass_rel=float(np.mean([r['total_mass_rel'] for r in subset])),
            total_mass_unrel=float(np.mean([r['total_mass_unrel'] for r in subset])),
            pairs=[dict(pair_id=pair_id, T=subset[0]['T'], R=subset[0]['R'], U=subset[0]['U'],
                        n=len(values), delta_mean_pp=float(np.mean(values)))
                   for pair_id, values in sorted(pairs.items())],
            per_layer_mean_delta_pp=[float(v) for v in
                                     np.mean([r['per_layer_delta_pp'] for r in subset], axis=0)])

    write_json(args.output_json, results)

    lines = [f"# Paired absent-object attention results ({results['model']['path']})",
             '',
             f"Samples: {results['totals']['samples']} from {results['totals']['images']} images;"
             f" bootstrap {args.bootstrap_draws} image-clustered draws (seed {args.seed}).",
             '',
             '| Association Type | # Samples | # Images | Related Query Attention (%) | '
             'Unrelated Query Attention (%) | Δ Attention (pp) [95% CI] | sign-flip p |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for type_name in ('contextual_co_occurrence', 'category_similarity'):
        t = results['types'].get(type_name)
        if not t:
            continue
        lines.append(
            f"| {t['label']} | {t['n_samples']} | {t['n_images']} | "
            f"{t['A_rel_mean_pp']:.2f} | {t['A_unrel_mean_pp']:.2f} | "
            f"{t['delta_mean_pp']:+.2f} [{t['delta_ci95_pp'][0]:+.2f}, {t['delta_ci95_pp'][1]:+.2f}] | "
            f"{t['sign_flip_p']:.4f} |")
    lines += ['', '## Per-pair breakdown', '',
              '| Pair (type) | T absent / R present / U control | n | mean ΔA (pp) |', '|---|---|---:|---:|']
    for type_name in ('contextual_co_occurrence', 'category_similarity'):
        for pair in results['types'][type_name]['pairs']:
            lines.append(f"| {pair['pair_id']} | {pair['T']} / {pair['R']} / {pair['U']} "
                         f"| {pair['n']} | {pair['delta_mean_pp']:+.2f} |")
    lines += ['', '## Per-layer mean ΔA (pp)', '',
              '| Association Type | ' + ' | '.join(f'L{l}' for l in range(32)) + ' |',
              '|---|' + '---:|' * 32]
    for type_name in ('contextual_co_occurrence', 'category_similarity'):
        t = results['types'][type_name]
        lines.append(f"| {t['label']} | " + ' | '.join(f'{v:+.2f}' for v in
                                                       t['per_layer_mean_delta_pp']) + ' |')
    lines += ['', f"Mean total visual attention (related / unrelated query): "
                 + '; '.join(f"{results['types'][k]['label']} "
                             f"{results['types'][k]['total_mass_rel']:.4f} / "
                             f"{results['types'][k]['total_mass_unrel']:.4f}"
                             for k in ('contextual_co_occurrence', 'category_similarity')
                             if k in results['types'])]
    lines += ['', 'Secondary mean-of-per-head-fractions variant: '
              + '; '.join(f"{t['label']} {t['variant_delta_mean_pp']:+.2f} "
                          f"[{t['variant_delta_ci95_pp'][0]:+.2f}, {t['variant_delta_ci95_pp'][1]:+.2f}]"
                          for t in results['types'].values())]
    Path(args.output_md).write_text('\n'.join(lines) + '\n')

    tex = ['% Auto-generated by aggregate_paired_absent_attention_20261113.py',
           '% Rebuttal table: quantitative analysis of the object association prior.',
           '% A(q,M_r) is the attention share of the related-object region M_r under an',
           '% absent-object query; Delta A pairs the related with the unrelated query on',
           '% the same image and region (LLaVA-1.5-7B, 32x32 layer-head mean attention).']
    for type_name in ('contextual_co_occurrence', 'category_similarity'):
        t = results['types'][type_name]
        tex.append(f"% {t['label']}: n={t['n_samples']}, related {t['A_rel_mean_pp']:.2f}\\%, "
                   f"unrelated {t['A_unrel_mean_pp']:.2f}\\%, "
                   f"Delta {t['delta_mean_pp']:+.2f} pp "
                   f"[{t['delta_ci95_pp'][0]:+.2f}, {t['delta_ci95_pp'][1]:+.2f}]")
    Path(args.output_tex).write_text('\n'.join(tex) + '\n')
    print(json.dumps({name: {key: value for key, value in t.items()
                             if key not in ('pairs', 'per_layer_mean_delta_pp')}
                      for name, t in results['types'].items()}, indent=2))


if __name__ == '__main__':
    main()
