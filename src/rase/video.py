"""Native Qwen2.5-VL video sampling and visual-token adapter."""
from __future__ import annotations
import hashlib
import math
import numpy as np
import torch
from .pipeline import OnlineRASE


def frame_indices(start, end, count):
    """Uniform, distinct frame IDs in the half-open clip interval."""
    if start < 0 or end - start < count or count < 2 or count % 2:
        raise ValueError('Video clips require enough frames and an even sample count')
    return np.linspace(start, end - 1, count).round().astype(np.int64).tolist()


def _decode_count_verified(row):
    """Resolve annotated frame indices using decoded frames.

    Return the requested RGB frames in their annotated order.
    """
    import cv2
    expected=int(row['source_frame_count'])
    indices=list(map(int,row['sampled_frame_indices']))
    if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=expected:
        raise ValueError('Invalid annotation frame ordinals: '+row['clip_id'])
    wanted=set(indices);selected={};decoded=0
    cap=cv2.VideoCapture(str(row['video_path']))
    try:
        if not cap.isOpened():raise ValueError('Cannot decode video: '+row['video_path'])
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if decoded in wanted:selected[decoded]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            decoded+=1
            if decoded>expected:break
    finally:cap.release()
    if decoded!=expected:
        raise ValueError(f'Actual decoded/annotation frame-count mismatch: {row["clip_id"]}: decoded={decoded}, annotation={expected}')
    if set(selected)!=wanted:raise ValueError('Missing requested annotation frames: '+row['clip_id'])
    return np.stack([selected[i] for i in indices])


def decode_frames(row):
    import cv2
    cap = cv2.VideoCapture(str(row['video_path']))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot decode video: ' + row['video_path'])
        count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Missing video timebase')
        if abs(fps - row['source_fps']) > max(.05, .01 * row['source_fps']):
            raise ValueError('Decoder/annotation FPS mismatch: ' + row['clip_id'])
        if abs(count - row['source_frame_count']) > 1:
            cap.release()
            return _decode_count_verified(row)
        frames = []
        for index in row['sampled_frame_indices']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > .5:
                raise ValueError(f'Failed exact frame read: {row["clip_id"]} @ {index}')
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return np.stack(frames)
    finally:
        cap.release()


def video_prompt(processor, instruction):
    return processor.apply_chat_template([dict(role='user', content=[
        dict(type='video'), dict(type='text', text=instruction)])],
        tokenize=False, add_generation_prompt=True)


def video_inputs(processor, row, prompt, *, min_pixels, max_pixels, device, dtype):
    frames = decode_frames(row)
    indices = row['sampled_frame_indices']
    # Preserve the actual sampling timebase, not the source FPS or default 2 FPS.
    sampled_fps = (len(indices) - 1) * row['source_fps'] / (indices[-1] - indices[0])
    frame_hash = hashlib.sha256(frames.tobytes()).hexdigest()
    inputs = processor(text=[prompt], videos=[torch.from_numpy(frames).permute(0, 3, 1, 2)],
        return_tensors='pt', videos_kwargs=dict(do_sample_frames=False, fps=sampled_fps,
            do_resize=True, size=dict(shortest_edge=min_pixels,longest_edge=max_pixels)))
    # This Transformers version keeps video_processor.size independent of the
    # legacy min_pixels/max_pixels attributes. Validate actual processed grids.
    grid=inputs['video_grid_thw'][0]
    area=int(grid[1])*int(grid[2])*int(processor.video_processor.patch_size)**2
    if not min_pixels <= area <= max_pixels:
        raise ValueError('Video processor did not apply the registered pixel budget')
    values = {k: v.to(device, dtype=dtype if v.is_floating_point() and k != 'second_per_grid_ts' else v.dtype)
              if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    metadata = dict(decoded_frames_sha256=frame_hash, sampled_fps=sampled_fps,
        sampled_frame_indices=indices, video_grid_thw=values['video_grid_thw'].tolist(),
        second_per_grid_ts=[float(x) for x in values['second_per_grid_ts']])
    return values, metadata


def video_visual_positions(model, inputs):
    if model.config.model_type != 'qwen2_5_vl':
        raise ValueError('This video adapter is verified only for Qwen2.5-VL')
    if 'image_grid_thw' in inputs:
        raise ValueError('Mixed image/video inputs are not this video protocol')
    grid = inputs.get('video_grid_thw')
    if not isinstance(grid, torch.Tensor) or grid.shape != (1, 3):
        raise ValueError('Exactly one video_grid_thw row required')
    t, h, w = [int(x) for x in grid[0].tolist()]
    merge = int(model.config.vision_config.spatial_merge_size)
    if t < 1 or h < 1 or w < 1 or h % merge or w % merge:
        raise ValueError('Invalid native video grid')
    positions = torch.where(inputs['input_ids'][0] == int(model.config.video_token_id))[0]
    if positions.numel() != t * h * w // merge ** 2:
        raise ValueError('Video placeholder count and native post-merge grid disagree')
    timing = inputs.get('second_per_grid_ts')
    if timing is None or len(timing) != 1 or not math.isfinite(float(timing[0])) or float(timing[0]) <= 0:
        raise ValueError('Missing/invalid native video temporal scale')
    return positions


class OnlineVideoRASE(OnlineRASE):
    """The exact image rollback policy, changing only visual-key resolution."""
    def _online_visual_positions(self, inputs):
        return video_visual_positions(self.model, inputs).to(self.device)

    def prepare_inputs(self, *args, **kwargs):
        raise TypeError('Use video_inputs with explicit sampling metadata')
