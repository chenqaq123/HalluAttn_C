#!/usr/bin/env python3
"""Extract paired absent-object attention rows on frozen samples.

For every frozen sample this runs the related and the unrelated absent-object
question through LLaVA-1.5-7B once each. The native SDPA forward is kept
unchanged; per decoder layer an FP32 causal-softmax probe recomputes the
selected first-subtoken rows and stores per-(layer, head) visual mass over
the related-object region and over all visual tokens. The first samples are
audited against a dense FP32 recomputation of all rows.

transformers 4.37 keeps a single <image> token in input_ids and expands it to
576 consecutive embeddings inside LlavaForConditionalGeneration.forward, so
every probe position is expressed in the expanded key space: visual keys are
the 576 slots starting at the input index of the <image> token, text rows
after the image shift by num_patches - 1, and the expanded length is
len(input_ids) + 575.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time

from pathlib import Path

import numpy as np

IMAGE_TOKEN_ID = 32000
VISUAL_TOKENS = 576
EXPANSION = VISUAL_TOKENS - 1
QUESTION_TEMPLATE = 'USER: <image>\nIs there {article} {name} in the image? ASSISTANT:'


def article(name):
    return 'an' if name[0].lower() in 'aeiou' else 'a'


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


class RowMassProbe:
    """FP32 selected-row attention masses alongside the untouched SDPA forward."""

    def __init__(self, model):
        import torch
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        self.torch = torch
        self.apply_rotary_pos_emb = apply_rotary_pos_emb
        self.layers = model.language_model.model.layers
        self.num_heads = model.config.text_config.num_attention_heads
        self.num_key_value_heads = model.config.text_config.num_key_value_heads
        self.head_dim = model.config.text_config.hidden_size // self.num_heads
        self.rows = None
        self.visual = None
        self.region = None
        self.expected_length = None
        self.record = {}
        self.originals = []
        for module in self.layers:
            attention = module.self_attn
            self.originals.append(attention.forward)
            attention.forward = self._wrap(attention, attention.forward)

    def _wrap(self, module, original):
        probe = self

        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            if probe.rows is not None and getattr(module, 'layer_idx', None) is not None:
                probe._probe(module, args, kwargs)
            return result

        return forward

    def _probe(self, module, args, kwargs):
        torch = self.torch
        hidden_states = kwargs.get('hidden_states', args[0] if args else None)
        position_ids = kwargs.get('position_ids')
        if hidden_states is None:
            raise ValueError('Probe could not read hidden states')
        batch, length, _ = hidden_states.shape
        if self.expected_length is not None and length != self.expected_length:
            raise ValueError(f'Expanded sequence length {length} differs from expected '
                             f'{self.expected_length}')
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        query = module.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = module.k_proj(hidden_states).view(
            batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # transformers 4.37 rotary embedding takes the integer sequence length.
        cos, sin = module.rotary_emb(key, seq_len=length)
        query, key = self.apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        if self.num_key_value_heads != self.num_heads:
            key = key.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        rows = self.rows.to(hidden_states.device)
        # transformers 4.37 SdpaAttention relies on F.scaled_dot_product_attention's
        # default scaling, i.e. 1/sqrt(head_dim); mirror it for the FP32 probe.
        selected = torch.matmul(query.index_select(2, rows).float(),
                                key.transpose(2, 3).float()) * (self.head_dim ** -0.5)
        positions = torch.arange(length, device=hidden_states.device)
        allowed = positions[None, :] <= rows[:, None]
        selected = selected.masked_fill(~allowed[None, None], float('-inf'))
        weights = torch.softmax(selected, dim=-1, dtype=torch.float32)
        mass_total = weights.index_select(-1, self.visual.to(hidden_states.device)).sum(-1)
        mass_region = weights.index_select(
            -1, self.region.to(hidden_states.device)).sum(-1)
        # Store as [n_rows, heads] so row slicing matches the dense audit.
        self.record[int(module.layer_idx)] = (
            mass_total[0].transpose(-2, -1).contiguous().float().cpu(),
            mass_region[0].transpose(-2, -1).contiguous().float().cpu())

    def restore(self):
        for module, original in zip(self.layers, self.originals):
            module.self_attn.forward = original


def locate_noun_row(tokenizer, input_ids, name, image_input_index):
    """(input_row, expanded_row) of the noun's first subtoken in the prompt.

    The noun is located by cumulative decoding: token j covers exactly the
    characters between the decoded lengths of the prefixes before and after
    it, so no re-tokenization or offset round-trip is assumed (the special
    <image> token makes re-tokenization differ from the processor chunks).
    """
    tokens = input_ids[0].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    match = re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text)
    if match is None:
        raise ValueError(f'Noun {name!r} not found in decoded prompt')
    special = set(tokenizer.all_special_ids)
    non_special = [i for i, token in enumerate(tokens) if token not in special]
    decoded = tokenizer.decode(tokens[:non_special[0] + 1], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
    start, boundaries = 0, []
    for i in non_special:
        decoded = tokenizer.decode(tokens[:i + 1], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        boundaries.append((start, len(decoded)))
        start = len(decoded)
    input_row = None
    for (low, high), i in zip(boundaries, non_special):
        if low <= match.start() < high:
            input_row = i
            break
    if input_row is None:
        raise ValueError(f'Noun {name!r} does not align with a token boundary')
    expanded_row = input_row + (EXPANSION if input_row > image_input_index else 0)
    return input_row, expanded_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--image-root', default=None,
                        help='Directory holding the val2017 jpgs; overrides the frozen '
                             'absolute image paths (file names must match)')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None,
                        help='Tokenizer directory; defaults to the model directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--min-free-gb', type=float, default=16.0,
                        help='Wait until this much GPU memory is free before loading')
    parser.add_argument('--wait-minutes', type=float, default=0.0,
                        help='Poll for free GPU memory up to this many minutes')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--audit-count', type=int, default=8)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration, \
        LlavaProcessor

    frozen = read(args.samples)
    if frozen['schema'] != 'paired-absent-attention-samples-v1':
        raise ValueError('Unknown sample list schema')
    samples = frozen['samples'] if args.limit is None else frozen['samples'][:args.limit]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer or args.model
    signature = dict(samples_sha256=sha(args.samples), model=args.model,
                     tokenizer=tokenizer_path, template=QUESTION_TEMPLATE,
                     image_root=args.image_root, dtype='float16')
    state_path = output / 'extract_state.json'
    if state_path.exists():
        state = read(state_path)
        if state['signature'] != signature:
            raise ValueError('Extraction resume signature differs')
    else:
        state = dict(signature=signature, processed=0, audits=[], elapsed_seconds=0.0,
                     complete=False)

    done = state['processed']
    if state['complete']:
        print(json.dumps(dict(stage='already_complete', samples=done)), flush=True)
        return

    keys = ('rel_region', 'rel_total', 'unrel_region', 'unrel_total')
    arrays = {key: [] for key in keys}
    processed_ids = []
    if done > 0:
        masses_path = output / 'extract_masses.npz'
        if not masses_path.exists():
            raise ValueError('Resume state exists but extraction masses are missing')
        masses = np.load(masses_path)
        if sorted(masses['sample_ids'].tolist()) != sorted(
                s['sample_id'] for s in samples[:done]):
            raise ValueError('Saved masses do not match the resumed prefix')
        for key in keys:
            arrays[key] = [masses[key][index] for index in range(done)]
        processed_ids = [int(v) for v in masses['sample_ids'].tolist()]
        metadata_lines = (output / 'extract_rows.jsonl').read_text().splitlines()
        if len([x for x in metadata_lines if x.strip()]) != done:
            raise ValueError('Metadata line count differs from resumed state')

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    required_bytes = int(args.min_free_gb * 1e9)
    deadline = time.monotonic() + args.wait_minutes * 60
    waited = False
    while free_bytes < required_bytes:
        if time.monotonic() >= deadline:
            raise ValueError(f'GPU memory gate: need {args.min_free_gb} GB free, '
                             f'found {free_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB')
        if not waited:
            print(json.dumps(dict(stage='waiting_for_gpu_memory',
                                  free_gb=round(free_bytes / 1e9, 2),
                                  required_gb=args.min_free_gb)), flush=True)
            waited = True
        time.sleep(60)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16,
        attn_implementation='sdpa', low_cpu_mem_usage=True).to(device).eval()
    # The snapshot's tokenizer.json predates this environment's tokenizers
    # library, so the processor is assembled from the rebuilt fast tokenizer
    # plus the snapshot's CLIPImageProcessor instead of from_pretrained.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    image_processor = CLIPImageProcessor.from_pretrained(args.model, local_files_only=True)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    probe = RowMassProbe(model)
    layers = len(probe.layers)

    metadata_path = output / 'extract_rows.jsonl'
    started = time.monotonic()
    previous_elapsed = state['elapsed_seconds']

    def checkpoint(complete=False):
        state['processed'] = len(processed_ids)
        state['elapsed_seconds'] = previous_elapsed + time.monotonic() - started
        state['complete'] = complete
        if processed_ids:
            np.savez(output / 'extract_masses.npz',
                     sample_ids=np.array(processed_ids, dtype=np.int64),
                     **{key: np.stack(values).astype(np.float32)
                        for key, values in arrays.items()})
        write_json(state_path, state)

    def run_query(sample, name, subtoken, image_path, dense=False):
        with Image.open(image_path) as image:
            prompt = QUESTION_TEMPLATE.format(name=name, article=article(name))
            raw = processor(text=prompt, images=image.convert('RGB'), return_tensors='pt')
        inputs = {key: value.to(device) for key, value in raw.items()}
        sequence = inputs['input_ids'][0].tolist()
        image_positions = [i for i, t in enumerate(sequence) if t == IMAGE_TOKEN_ID]
        if len(image_positions) != 1:
            raise ValueError(f'Expected one <image> token, found {len(image_positions)}')
        image_input_index = image_positions[0]
        visual = list(range(image_input_index, image_input_index + VISUAL_TOKENS))
        input_row, expanded_row = locate_noun_row(tokenizer, inputs['input_ids'], name,
                                                  image_input_index)
        if int(sequence[input_row]) != int(subtoken['token_id']):
            raise ValueError(f'First-subtoken id mismatch for {name!r} on image '
                             f'{sample["image_id"]}')
        probe.visual = torch.tensor(visual, dtype=torch.long)
        probe.region = torch.tensor([visual[i] for i in sample['region']['token_indices']],
                                    dtype=torch.long)
        probe.expected_length = len(sequence) + EXPANSION
        if dense:
            probe.rows = torch.arange(probe.expected_length, device=device)
            probe.record = {}
            model(**inputs, use_cache=False)
            reference = {layer: value for layer, value in probe.record.items()}
            probe.rows = torch.tensor([expanded_row], dtype=torch.long, device=device)
            probe.record = {}
            model(**inputs, use_cache=False)
            if set(probe.record) != set(range(layers)):
                raise ValueError('Missing decoder layers in probe record')
            for layer, (mass_total, mass_region) in probe.record.items():
                ref_total, ref_region = reference[layer]
                delta_total = float(np.max(np.abs(mass_total.numpy()
                                                  - ref_total[expanded_row].numpy())))
                delta_region = float(np.max(np.abs(mass_region.numpy()
                                                   - ref_region[expanded_row].numpy())))
                if not np.allclose(mass_total.numpy(), ref_total[expanded_row].numpy(),
                                   rtol=1e-4, atol=1e-6) or \
                   not np.allclose(mass_region.numpy(), ref_region[expanded_row].numpy(),
                                   rtol=1e-4, atol=1e-6):
                    raise ValueError(f'Dense audit failed at layer {layer}: max|Δtotal|='
                                     f'{delta_total:.3e}, max|Δregion|={delta_region:.3e}')
            return
        probe.rows = torch.tensor([expanded_row], dtype=torch.long, device=device)
        probe.record = {}
        model(**inputs, use_cache=False)
        if set(probe.record) != set(range(layers)):
            raise ValueError('Missing decoder layers in probe record')
        mass_total = np.stack([probe.record[layer][0].numpy() for layer in range(layers)])
        mass_region = np.stack([probe.record[layer][1].numpy() for layer in range(layers)])
        if not np.isfinite(mass_total).all() or not np.isfinite(mass_region).all():
            raise ValueError('Non-finite attention mass')
        if np.any(mass_total < 0) or np.any(mass_total > 1.00001):
            raise ValueError('Invalid visual attention mass')
        return mass_total, mass_region, input_row

    for sample in samples[done:]:
        image_path = (Path(args.image_root) / Path(sample['image_file']).name
                      if args.image_root else Path(sample['image_file']))
        if sha(image_path) != sample['image_sha256']:
            raise ValueError(f'Image changed since freeze: {image_path}')
        rel_total, rel_region, rel_row = run_query(sample, sample['T'],
                                                   sample['query_related'], image_path)
        unrel_total, unrel_region, unrel_row = run_query(sample, sample['U'],
                                                         sample['query_unrelated'], image_path)
        if len(processed_ids) < args.audit_count:
            run_query(sample, sample['T'], sample['query_related'], image_path, dense=True)
            run_query(sample, sample['U'], sample['query_unrelated'], image_path, dense=True)
            state['audits'].append(dict(sample_id=sample['sample_id'],
                                        comparison='selected versus dense FP32 rows on '
                                                   'identical SDPA hidden states'))
        arrays['rel_total'].append(rel_total)
        arrays['rel_region'].append(rel_region)
        arrays['unrel_total'].append(unrel_total)
        arrays['unrel_region'].append(unrel_region)
        processed_ids.append(int(sample['sample_id']))
        with metadata_path.open('a') as stream:
            stream.write(json.dumps(dict(
                sample_id=int(sample['sample_id']), image_id=sample['image_id'],
                pair_id=sample['pair_id'], type=sample['type'], T=sample['T'],
                R=sample['R'], U=sample['U'], region_size=sample['region']['size'],
                related_input_row=rel_row, unrelated_input_row=unrel_row)) + '\n')
        if len(processed_ids) % args.checkpoint_every == 0:
            checkpoint()
            print(json.dumps(dict(stage='extracting', samples=len(processed_ids),
                                  total=len(samples),
                                  elapsed_seconds=state['elapsed_seconds'])), flush=True)
    checkpoint(complete=True)
    probe.restore()
    print(json.dumps(dict(stage='complete', samples=len(processed_ids),
                          audits=len(state['audits']),
                          elapsed_seconds=state['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
