"""Command-line entry points for scoring, extraction, training, and captioning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .detector import StructuralDetector, RASEDetector


def load_runtime(args):
    from .utils import load_model_and_processor, torch_dtype
    dtype = torch_dtype(args.dtype)
    model, processor = load_model_and_processor(args.model, torch.device(args.device),
        attn_implementation='sdpa', dtype=dtype, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model.requires_grad_(False)
    return model, processor, dtype


def prepared_inputs(model, processor, image, instruction, device, dtype):
    from .model_adapter import model_type, processor_inputs
    # Native chat templates construct image markers and assistant prefixes.
    prompt = processor.apply_chat_template([{'role':'user', 'content':[
        {'type':'image'}, {'type':'text','text':instruction}]}], tokenize=False, add_generation_prompt=True)
    batch = processor_inputs(processor, model_type(model), image, prompt)
    batch = {k:v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
             if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
    return batch, prompt


def caption(args):
    import spacy
    from PIL import Image
    from .pipeline import OnlineRASE
    from .video import OnlineVideoRASE, video_inputs, video_prompt
    model, processor, dtype = load_runtime(args)
    detector = RASEDetector(args.checkpoint, args.semantic_checkpoint, args.device) if args.semantic_checkpoint else StructuralDetector(args.checkpoint, args.device)
    detector.validate_backbone(model.config)
    threshold = args.threshold if args.threshold is not None else getattr(detector, 'threshold', None)
    if threshold is None:
        raise ValueError('Provide --threshold for S-only captioning')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must lie in [0, 1]')
    nlp = spacy.load(args.spacy_model)
    runner_type = OnlineRASE
    if args.video_record:
        row = json.loads(Path(args.video_record).read_text())
        prompt = video_prompt(processor, args.prompt)
        inputs, _ = video_inputs(processor, row, prompt, min_pixels=args.min_pixels or 3136,
            max_pixels=args.max_pixels or 200704, device=args.device, dtype=dtype)
        runner_type = OnlineVideoRASE
    else:
        inputs, prompt = prepared_inputs(model, processor, Image.open(args.image).convert('RGB'),
                                        args.prompt, args.device, dtype)
    runner = runner_type(model, processor, nlp, args.vocabulary, detector,
        threshold=threshold, device=args.device, dtype=dtype,
        max_new_tokens=args.max_new_tokens, prompt=prompt)
    try:
        result = runner.generate_online(inputs, benchmark=args.vocabulary_key)
        print(json.dumps({'caption':result['caption'], 'rollbacks':result['total_rollbacks']}, ensure_ascii=False))
    finally:
        runner.evidence.close()


@torch.inference_mode()
def extract(args):
    """Replay caption tokens; mentions specify zero-based first-subtoken indices."""
    from PIL import Image
    from .pipeline import SGEvidence
    from .s_relations import attention_to_s_edges
    from .model_adapter import visual_layout, set_attention_implementation
    model, processor, dtype = load_runtime(args)
    evidence = SGEvidence(model)
    ss, gs, labels, images = [], [], [], []
    records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
    try:
        for row in records:
            if args.video:
                from .video import video_inputs, video_prompt, video_visual_positions
                prompt = video_prompt(processor, row.get('prompt', args.prompt))
                inputs, _ = video_inputs(processor, row, prompt, min_pixels=args.min_pixels or 3136,
                    max_pixels=args.max_pixels or 200704, device=args.device, dtype=dtype)
                visual = video_visual_positions(model, inputs).to(args.device)
            else:
                inputs, _ = prepared_inputs(model, processor, Image.open(row['image']).convert('RGB'),
                    row.get('prompt', args.prompt), args.device, dtype)
                visual = visual_layout(model, inputs, inputs['input_ids'], scope='all', llava_base_tokens=576).positions.to(args.device)
            evidence.before_prefill(visual)
            set_attention_implementation(model, 'sdpa')
            output = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
            past = output.past_key_values
            evidence.after_prefill()
            del output
            targets = {int(m['token_index']):int(m['label']) for m in row['mentions']}
            tokens = row['generated_ids']
            if len(targets) != len(row['mentions']) or any(i < 0 or i >= len(tokens) for i in targets):
                raise ValueError('Mention token indices must be unique and within generated_ids')
            prompt_length = inputs['input_ids'].shape[-1]
            set_attention_implementation(model, 'pure_s_canonical_capture')
            for index, token in enumerate(tokens[:max(targets, default=-1)+1]):
                evidence.before_token(token, want=index in targets)
                position = prompt_length + index
                output = model(input_ids=torch.tensor([[token]],device=args.device), past_key_values=past,
                    attention_mask=torch.ones((1, position+1),device=args.device,dtype=torch.long),
                    cache_position=torch.tensor([position],device=args.device),
                    use_cache=True, output_attentions=True, logits_to_keep=1, return_dict=True)
                past = output.past_key_values
                if index in targets:
                    signal, _ = evidence.signal()
                    ss.append(attention_to_s_edges(signal.conditional.flatten(0,1)).half().cpu().numpy())
                    gs.append(evidence.g(signal).cpu().numpy())
                    labels.append(targets[index]); images.append(row['source_video_id'] if args.video else row['image_id'])
                del output
            evidence.clear()
            del past
    finally:
        set_attention_implementation(model, 'sdpa')
        evidence.close()
    if not ss:
        raise ValueError('Records must contain at least one labeled object mention')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output, s=np.stack(ss), g=np.stack(gs), labels=np.asarray(labels), image_ids=np.asarray(images))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify-checkpoints')
    verify.add_argument('--manifest', default='checkpoints/manifest.json')
    score = sub.add_parser('score-s', help='Score a tensor of selected S features')
    score.add_argument('--checkpoint', required=True)
    score.add_argument('--features', required=True, help='.npy array [mentions, selected_edges]')
    score.add_argument('--device', default='cpu')
    training = sub.add_parser('train')
    training.add_argument('--features', required=True, help='.npz with s, g, labels, image_ids')
    training.add_argument('--output', required=True)
    training.add_argument('--device', default='cpu')
    training.add_argument('--backbone-model-type', required=True, choices=['llava','llava_next','qwen2_5_vl','qwen3_vl'])
    training.add_argument('--seed', type=int, default=42)
    for name, default in [('s',7),('g',7),('fusion',30)]:
        training.add_argument(f'--{name}-epochs',type=int,default=default)
    for command in ['caption','extract']:
        p = sub.add_parser(command)
        p.add_argument('--model', required=True, help='Local backbone checkpoint directory')
        p.add_argument('--device', default='cuda:0')
        p.add_argument('--dtype', choices=['float16','bfloat16','float32'], default='bfloat16')
        p.add_argument('--min-pixels', type=int)
        p.add_argument('--max-pixels', type=int)
        p.add_argument('--prompt', default='Describe this image in detail.')
        if command == 'caption':
            media = p.add_mutually_exclusive_group(required=True)
            media.add_argument('--image')
            media.add_argument('--video-record')
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--semantic-checkpoint')
            p.add_argument('--threshold', type=float)
            p.add_argument('--vocabulary', default='configs/objects.json')
            p.add_argument('--vocabulary-key', default='objects')
            p.add_argument('--spacy-model', default='en_core_web_sm')
            p.add_argument('--max-new-tokens', type=int, default=512)
        else:
            p.add_argument('--video', action='store_true')
            p.add_argument('--records', required=True)
            p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'verify-checkpoints':
        manifest = Path(args.manifest)
        for name, item in json.loads(manifest.read_text()).items():
            path = manifest.parent.parent/item['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Checkpoint hash differs: {name}')
            StructuralDetector(path)
            print(f'{name}: verified')
    elif args.command == 'score-s':
        result = StructuralDetector(args.checkpoint,args.device).score_features(torch.from_numpy(np.load(args.features,allow_pickle=False)))
        print(json.dumps(result.cpu().tolist()))
    elif args.command == 'train':
        from .training import train
        with np.load(args.features, allow_pickle=False) as data:
            result = train(data['s'],data['g'],data['labels'],data['image_ids'],args.output,
                device=args.device,seed=args.seed,s_epochs=args.s_epochs,g_epochs=args.g_epochs,
                fusion_epochs=args.fusion_epochs,backbone_model_type=args.backbone_model_type)
        Path(args.output,'training.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    elif args.command == 'caption':
        caption(args)
    else:
        extract(args)

if __name__ == '__main__':
    main()
