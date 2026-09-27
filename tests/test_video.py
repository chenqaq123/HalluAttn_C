from types import SimpleNamespace as NS
import pytest
import torch
from rase.video import frame_indices, video_visual_positions

def test_uniform_frames_half_open_distinct_even():
    ids = frame_indices(17, 100, 16)
    assert len(ids) == len(set(ids)) == 16
    assert ids[0] == 17 and ids[-1] == 99
    with pytest.raises(ValueError): frame_indices(0, 12, 16)
    with pytest.raises(ValueError): frame_indices(0, 50, 15)

def test_all_native_video_tokens_includes_temporal_axis():
    model = NS(config=NS(model_type='qwen2_5_vl', video_token_id=7,
                       vision_config=NS(spatial_merge_size=2)))
    inputs = dict(input_ids=torch.tensor([[1]+[7]*24+[2]]),
                  video_grid_thw=torch.tensor([[3,4,8]]), second_per_grid_ts=[.75])
    assert video_visual_positions(model, inputs).tolist() == list(range(1,25))
    inputs['input_ids'] = inputs['input_ids'][:, :20]
    with pytest.raises(ValueError): video_visual_positions(model, inputs)

def test_video_metadata_is_json_and_temporal_scale_stays_float32(monkeypatch):
    import json
    import numpy as np
    import rase.video as adapter
    monkeypatch.setattr(adapter,'decode_frames',lambda row:np.zeros((2,28,28,3),np.uint8))
    def processor(**kwargs):
        assert kwargs['videos_kwargs']['do_sample_frames'] is False
        assert kwargs['videos_kwargs']['fps']==3.0
        assert kwargs['videos_kwargs']['size']==dict(shortest_edge=784,longest_edge=784)
        return dict(input_ids=torch.tensor([[7]]),video_grid_thw=torch.tensor([[1,2,2]]),
            pixel_values_videos=torch.ones(1,3),second_per_grid_ts=torch.tensor([2/3]))
    processor.video_processor=NS(patch_size=14)
    values,meta=adapter.video_inputs(processor,dict(sampled_frame_indices=[0,10],source_fps=30.),
        'prompt',min_pixels=784,max_pixels=784,device='cpu',dtype=torch.bfloat16)
    json.dumps(meta)
    assert values['second_per_grid_ts'].dtype==torch.float32
    assert values['pixel_values_videos'].dtype==torch.bfloat16