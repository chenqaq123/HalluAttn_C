#!/usr/bin/env python3
"""Freeze the paired absent-object attention sample list, attention-blind.

Mechanically derives association pairs from the frozen Figure 4 related
graph, selects COCO val2017 images under the pre-registered protocol
(pipeline/paired_absent_attention_protocol_20261113.md), and writes a
SHA-anchored sample list. No attention, logit, or model output is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from bisect import bisect_right
from pathlib import Path

SEED = 20261113
GRID = 24
PATCH = 14
CROP = 336
MIN_REGION_TOKENS = 3
PER_PAIR_CAP = 40
PER_IMAGE_CAP = 4
COOC_TOP_N = 16
SIMILARITY_STRATUM_CAP = 2
SIMILARITY_TOP_N = 16
COOC_MIN_SUPPORT = 40
COOC_MIN_CONDITIONAL = 0.15
COOC_MIN_LIFT = 2.0
ANIMAL_DOMAIN = 'animal'
TYPE_ORDER = {'contextual_co_occurrence': 0, 'category_similarity': 1}


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


def article(name):
    return 'an' if name[0].lower() in 'aeiou' else 'a'


def surface_forms(name):
    """Word variants used for caption matching; order-insensitive set."""
    forms = {name, name + 's'}
    if name.endswith('s') and len(name) > 1:
        forms.add(name[:-1])
    return forms


def mentioned(text_lower, forms):
    return any(re.search(r'(?<!\w)' + re.escape(form) + r'(?!\w)', text_lower)
               for form in forms)


# --- pair derivation (protocol section 3) ----------------------------------


def cooc_records(entry):
    return [r for r in entry if isinstance(r, dict)]


def derive_pairs(figure4, supercategory):
    """Two pre-registered ordered pair lists with the quantified rule trace."""
    reasons = figure4['relation_reasons']

    def in_coco(key):
        t, r = key.split('|', 1)
        return t in supercategory and r in supercategory

    cooc = []
    for key, entry in reasons.items():
        records = cooc_records(entry)
        if not records or not in_coco(key):
            continue
        best = max(records, key=lambda r: r['support'] * r['lift'])
        if (best['support'] >= COOC_MIN_SUPPORT and best['conditional'] >= COOC_MIN_CONDITIONAL
                and best['lift'] >= COOC_MIN_LIFT):
            t, r = key.split('|', 1)
            cooc.append(dict(T=t, R=r, support=best['support'],
                             conditional=best['conditional'], lift=best['lift'],
                             rank=best['support'] * best['lift']))
    cooc.sort(key=lambda p: (-p['rank'], p['T'], p['R']))
    selected_cooc, seen_unordered = [], set()
    for pair in cooc:
        unordered = tuple(sorted((pair['T'], pair['R'])))
        if unordered in seen_unordered:
            continue
        seen_unordered.add(unordered)
        selected_cooc.append(pair)
        if len(selected_cooc) == COOC_TOP_N:
            break
    if not any(p['T'] == 'traffic light' and p['R'] == 'car' for p in selected_cooc):
        best = max(cooc_records(reasons['traffic light|car']),
                   key=lambda r: r['support'] * r['lift'])
        selected_cooc.append(dict(T='traffic light', R='car', support=best['support'],
                                  conditional=best['conditional'], lift=best['lift'],
                                  rank=best['support'] * best['lift']))

    similar = []
    for key, entry in reasons.items():
        if not entry or cooc_records(entry) or not in_coco(key):
            continue
        t, r = key.split('|', 1)
        if t == r or supercategory[t] != supercategory[r]:
            continue
        similar.append(dict(T=t, R=r))
    rng = random.Random(SEED)
    strata = {}
    for pair in similar:
        strata.setdefault(supercategory[pair['T']], []).append(pair)
    selected_sim = []
    for name in sorted(strata):
        pool = strata[name][:]
        rng.shuffle(pool)
        selected_sim.extend(pool[:SIMILARITY_STRATUM_CAP])
    selected_sim = selected_sim[:SIMILARITY_TOP_N]
    if not any(p['T'] == 'bicycle' and p['R'] == 'motorcycle' for p in selected_sim):
        selected_sim.append(dict(T='bicycle', R='motorcycle'))
    return selected_cooc, selected_sim


def control_target(pair, supercategory):
    if supercategory.get(pair['T']) == ANIMAL_DOMAIN:
        return 'traffic light'
    return 'teddy bear'


# --- geometry (protocol section 5) -----------------------------------------


def resize_output_size(width, height):
    """Replicate CLIPImageProcessor shortest-edge resize (int truncation)."""
    if width <= height:
        return (CROP, int(CROP * height / width))
    return (int(CROP * width / height), CROP)


def crop_offsets(width_r, height_r):
    return (height_r - CROP) // 2, (width_r - CROP) // 2


def patch_center_original(index, width, height, width_r, height_r, top, left):
    """Token index (row-major 24x24) to its patch-center point in original image."""
    row, col = divmod(index, GRID)
    x = col * PATCH + PATCH / 2
    y = row * PATCH + PATCH / 2
    return ((x + left) * width / width_r, (y + top) * height / height_r)


def instance_mask(annotation, width, height):
    """Rasterize one iscrowd=0 instance; polygon preferred, bbox fallback."""
    import numpy as np
    from pycocotools import mask as mask_utils
    segmentation = annotation.get('segmentation')
    used_fallback = False
    if isinstance(segmentation, list) and segmentation:
        if isinstance(segmentation[0], list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rles = mask_utils.frPyObjects([segmentation], height, width)
        mask = mask_utils.decode(mask_utils.merge(rles))
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get('counts'), list):
            mask = mask_utils.decode(mask_utils.frPyObjects(segmentation, height, width))
        else:
            mask = mask_utils.decode(segmentation)
    else:
        used_fallback = True
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = annotation['bbox']
        mask[max(int(y), 0):min(int(y + h) + 1, height),
             max(int(x), 0):min(int(x + w) + 1, width)] = 1
    return mask.astype(bool), used_fallback


def region_tokens(masks, width, height, width_r, height_r, top, left):
    """Indices of the 24x24 tokens whose patch center falls inside any mask."""
    import numpy as np
    centers = [patch_center_original(i, width, height, width_r, height_r, top, left)
               for i in range(GRID * GRID)]
    columns = np.clip([round(x) for x, _ in centers], 0, width - 1)
    rows = np.clip([round(y) for _, y in centers], 0, height - 1)
    hits = np.zeros(GRID * GRID, dtype=bool)
    for mask in masks:
        hits |= mask[rows, columns].astype(bool)
    return [i for i in range(GRID * GRID) if hits[i]]


# --- first subtoken --------------------------------------------------------


def first_subtoken(tokenizer, question, name):
    """(token_id, index within question tokens) of the noun's first subtoken."""
    encoded = tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
    match = re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', question)
    if match is None:
        raise ValueError(f'Name not found in question: {name}')
    token_index = bisect_right([a for a, _ in encoded['offset_mapping']], match.start()) - 1
    if token_index < 0 or not encoded['offset_mapping'][token_index][0] <= match.start() \
            < encoded['offset_mapping'][token_index][1]:
        raise ValueError(f'Offset mismatch for {name}')
    return dict(token_id=int(encoded['input_ids'][token_index]),
                index_in_question=int(token_index))


# --- sample selection (protocol section 4) ---------------------------------


def build_samples(args):
    from transformers import AutoTokenizer

    figure4 = read(args.figure4_protocol)
    coco = read(args.instances)
    captions_raw = read(args.captions)
    images_meta = {image['id']: image for image in coco['images']}
    supercategory = {c['name']: c['supercategory'] for c in coco['categories']}
    category_id = {c['name']: c['id'] for c in coco['categories']}

    annotations_by_image = {}
    for annotation in coco['annotations']:
        if annotation.get('iscrowd', 0) == 1:
            continue
        annotations_by_image.setdefault(annotation['image_id'], []).append(annotation)
    texts_by_image = {}
    for annotation in captions_raw['annotations']:
        texts_by_image.setdefault(annotation['image_id'], []).append(annotation['caption'].lower())

    category_forms = {c['name']: surface_forms(c['name']) for c in coco['categories']}
    image_state = {}
    for image_id, meta in images_meta.items():
        anns = annotations_by_image.get(image_id, [])
        text_lower = ' \n '.join(texts_by_image.get(image_id, []))
        instance_names = {name for name, cid in category_id.items()
                          if any(a['category_id'] == cid for a in anns)}
        mentioned_names = {name for name, forms in category_forms.items()
                           if mentioned(text_lower, forms)}
        image_state[image_id] = dict(anns=anns, text_lower=text_lower, width=meta['width'],
                                     height=meta['height'], file_name=meta['file_name'],
                                     scene_names=instance_names | mentioned_names,
                                     instance_names=instance_names)

    selected_cooc, selected_sim = derive_pairs(figure4, supercategory)
    pairs = [dict(type='contextual_co_occurrence', T=p['T'], R=p['R'],
                  U=control_target(p, supercategory),
                  rule=dict(support=p['support'], conditional=p['conditional'], lift=p['lift']))
             for p in selected_cooc]
    pairs += [dict(type='category_similarity', T=p['T'], R=p['R'],
                   U=control_target(p, supercategory), rule=dict(family_only=True))
              for p in selected_sim]
    pairs.sort(key=lambda p: (TYPE_ORDER[p['type']], p['T'], p['R']))

    graph = figure4['related_graph']
    reverse_graph = {}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            reverse_graph.setdefault(neighbor, set()).add(node)
    reasons = figure4['relation_reasons']

    def cooc_between(a, b):
        return bool(cooc_records(reasons.get(f'{a}|{b}', []))) or \
            bool(cooc_records(reasons.get(f'{b}|{a}', [])))

    def absent(state, name):
        forms = surface_forms(name)
        return name not in state['instance_names'] and not mentioned(state['text_lower'], forms)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    questions, samples, counts = {}, [], []
    per_image_used = {}
    for pair in pairs:
        pair_id = f"{pair['type']}:{pair['T']}|{pair['R']}"
        for role, name in (('T', pair['T']), ('U', pair['U'])):
            question = f'Is there {article(name)} {name} in the image?'
            questions[(pair_id, role)] = first_subtoken(tokenizer, question, name)
        trace = dict(scanned=0, target_present=0, control_fail=0, related_absent=0,
                     image_cap=0, region_small=0, accepted=0)
        for image_id in sorted(images_meta):
            if trace['accepted'] >= PER_PAIR_CAP:
                break
            state = image_state[image_id]
            trace['scanned'] += 1
            if not absent(state, pair['T']):
                trace['target_present'] += 1
                continue
            control = pair['U']
            control_neighborhood = set(graph.get(control, [])) | set(reverse_graph.get(control, set()))
            if (not absent(state, control)
                    or control_neighborhood & state['scene_names']
                    or cooc_between(control, pair['R'])
                    or (control in supercategory and pair['R'] in supercategory
                        and supercategory[control] == supercategory[pair['R']])):
                trace['control_fail'] += 1
                continue
            related = [a for a in state['anns'] if a['category_id'] == category_id[pair['R']]]
            if not related:
                trace['related_absent'] += 1
                continue
            if per_image_used.get(image_id, 0) >= PER_IMAGE_CAP:
                trace['image_cap'] += 1
                continue
            width, height = state['width'], state['height']
            width_r, height_r = resize_output_size(width, height)
            top, left = crop_offsets(width_r, height_r)
            masks, fallback_flags = [], []
            for annotation in related:
                mask, fallback = instance_mask(annotation, width, height)
                masks.append(mask)
                fallback_flags.append(fallback)
            indices = region_tokens(masks, width, height, width_r, height_r, top, left)
            if len(indices) < MIN_REGION_TOKENS:
                trace['region_small'] += 1
                continue
            image_path = Path(args.images) / state['file_name']
            per_image_used[image_id] = per_image_used.get(image_id, 0) + 1
            samples.append(dict(
                sample_id=len(samples), pair_id=pair_id, image_id=int(image_id),
                image_file=str(image_path), image_sha256=sha(image_path),
                type=pair['type'], T=pair['T'], R=pair['R'], U=pair['U'],
                query_related=questions[(pair_id, 'T')],
                query_unrelated=questions[(pair_id, 'U')],
                region=dict(token_indices=indices, size=len(indices),
                            ann_ids=[int(a['id']) for a in related],
                            used_bbox_fallback=any(fallback_flags),
                            transform=dict(width=width, height=height, width_r=width_r,
                                           height_r=height_r, top=top, left=left))))
            trace['accepted'] += 1
        counts.append(dict(pair_id=pair_id, type=pair['type'], T=pair['T'], R=pair['R'],
                           U=pair['U'], **trace))

    output = dict(
        schema='paired-absent-attention-samples-v1', seed=SEED,
        model=dict(path=args.tokenizer, dtype='float16', visual_tokens=GRID * GRID),
        inputs=dict(instances_sha256=sha(args.instances), captions_sha256=sha(args.captions),
                    figure4_sha256=sha(args.figure4_protocol), image_dir=str(Path(args.images)),
                    tokenizer_source=args.tokenizer_source,
                    tokenizer_source_tokenizer_model_sha256=sha(
                        str(Path(args.tokenizer_source) / 'tokenizer.model'))
                    if args.tokenizer_source else None),
        template='USER: <image>\\nIs there {article} {name} in the image? ASSISTANT:',
        pairs=counts, samples=samples,
        totals=dict(pairs=len(counts), samples=len(samples),
                    by_type={t: sum(1 for s in samples if s['type'] == t)
                             for t in TYPE_ORDER},
                    images=len({s['image_id'] for s in samples})))
    write_json(args.output, output)
    print(json.dumps(output['totals'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure4-protocol', required=True)
    parser.add_argument('--instances', required=True)
    parser.add_argument('--captions', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--tokenizer-source', default=None,
                        help='Original snapshot the converted tokenizer was rebuilt from')
    parser.add_argument('--output', required=True)
    build_samples(parser.parse_args())


if __name__ == '__main__':
    main()
