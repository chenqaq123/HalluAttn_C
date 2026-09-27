from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
import torch
from rase.features import semantic_features
from rase.detector import StructuralDetector, RASEDetector, RiskFusion
from rase.pipeline import selected_values, dimensions
from rase.s_relations import canonical_edge_heads, attention_to_s_edges
from rase.training import select_edges, train
from rase.learned_s import image_folds

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('layers,heads,k', [(32,32,5238),(28,28,3070),(36,32,6630)])
def test_paper_dimensions(layers, heads, k):
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=layers,num_attention_heads=heads))
    assert dimensions(config)['selected_edges'] == k


def test_semantics_use_cosine_selected_target_probability():
    # The attention winner and cosine winner differ, and probability differs
    # from a logit margin. This distinguishes all three paper definitions.
    query = torch.tensor([[1.,0.]])
    patches = torch.tensor([[[1.,0.],[0.,2.],[-1.,0.],[0.,-1.]]])
    attention = torch.tensor([[[.1,.7,.1,.1],[.1,.7,.1,.1]]])
    result = semantic_features(query, patches, attention, attention, 0, torch.nn.Identity(), torch.nn.Identity())
    expected = torch.tensor([1., 0., torch.softmax(torch.tensor([1.,0.]),0)[0]])
    torch.testing.assert_close(result, expected)


def test_top_fraction_rounds_up_and_averages_probabilities():
    query = torch.tensor([[1.,0.]])
    patches = torch.zeros(1,21,2); patches[0,:,1]=1
    patches[0,0]=torch.tensor([2.,0.]); patches[0,1]=torch.tensor([1.,1.])
    attention=torch.ones(1,2,21)
    actual=semantic_features(query,patches,attention,attention/21,0,torch.nn.Identity(),torch.nn.Identity())
    expected=torch.softmax(patches[0,:2],-1)[:,0].mean()
    torch.testing.assert_close(actual[2],expected)


def test_s_pair_order_and_mass_invariance():
    torch.manual_seed(3)
    attention=torch.rand(6,20)
    indices=[0,3,7,12]
    pairs=torch.tensor(canonical_edge_heads(6)[indices])
    expected=attention_to_s_edges(attention)[indices].half()
    actual=selected_values(attention,pairs,chunk=2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-3)
    torch.testing.assert_close(attention_to_s_edges(attention*torch.arange(1,7)[:,None]),attention_to_s_edges(attention))


def test_all_four_published_checkpoints():
    manifest=json.loads((ROOT/'checkpoints/manifest.json').read_text())
    assert len(manifest)==4
    assert len(list((ROOT/'checkpoints').rglob('*.pt')))==4
    for entry in manifest.values():
        path=ROOT/entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        detector=StructuralDetector(path)
        score=detector.score_features(torch.full((2,entry['selected_edges']),.5))
        assert score.shape==(2,) and torch.isfinite(score).all()
        assert ((score>=0)&(score<=1)).all()


def test_grounded_selection_uses_only_fitting_images():
    s=np.array([[.2,.9,.1],[.4,.7,.2],[.1,.2,.9]])
    labels=np.array([0,0,1]); fitting=np.array([0,1])
    before=select_edges(s,labels,fitting)
    s[2]=[1e5,1e5,-1e5]
    np.testing.assert_array_equal(select_edges(s,labels,fitting),before)


def test_repeated_image_stays_in_one_fold():
    ids=np.repeat(np.arange(20),3)
    folds=image_folds(ids,5,42)
    assert all(len(set(folds[ids==i]))==1 for i in range(20))


def test_training_and_fusion_roundtrip(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(2)
    s=rng.random((40,6)).astype(np.float16)
    g=rng.random((40,3)).astype(np.float32)
    labels=np.tile([0,1],20)
    report=train(s,g,labels,np.repeat(np.arange(20),2),tmp_path,s_epochs=1,g_epochs=1,fusion_epochs=1)
    detector=RASEDetector(tmp_path/'s_mlp.pt',tmp_path/'semantic_fusion.pt')
    assert detector.fusion.linear.in_features==2
    assert 0<=report['threshold']<=1
    signal=SimpleNamespace(conditional=torch.ones(1,4,3))
    evidence=SimpleNamespace(g=lambda signal:torch.ones(3))
    risk,s_risk=detector.score(signal,evidence)
    assert 0<=risk<=1 and 0<=s_risk<=1
    with np.load(tmp_path/'oof_scores.npz') as data:
        assert len(data['scores'])==40 and np.isfinite(data['scores']).all()


def test_outer_fold_labels_do_not_affect_its_predictions(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(4)
    s=rng.random((30,6)).astype(np.float16)
    g=rng.random((30,3)).astype(np.float32)
    labels=np.tile([0,1],15); ids=np.repeat(np.arange(15),2)
    heldout=image_folds(ids,5,42)==0
    changed=labels.copy(); changed[heldout]=1-changed[heldout]
    for name,y in [('first',labels),('changed',changed)]:
        train(s,g,y,ids,tmp_path/name,s_epochs=1,g_epochs=1,fusion_epochs=1)
    with np.load(tmp_path/'first/oof_scores.npz') as a, np.load(tmp_path/'changed/oof_scores.npz') as b:
        np.testing.assert_array_equal(a['scores'][heldout],b['scores'][heldout])


def test_source_video_string_ids_stay_in_one_fold():
    ids=np.repeat([f'video_{i}' for i in range(10)],3)
    folds=image_folds(ids,5,42)
    assert all(len(set(folds[ids==i]))==1 for i in np.unique(ids))


@pytest.mark.parametrize('family', ['llava', 'llava_next', 'qwen2_5_vl', 'qwen3_vl'])
def test_visual_layout_preserves_all_native_tokens(family):
    from rase.model_adapter import visual_layout
    config = SimpleNamespace(model_type=family, image_token_index=99,
        image_token_id=99, vision_config=SimpleNamespace(spatial_merge_size=2))
    model = SimpleNamespace(config=config)
    ids = torch.tensor([[1] + [99] * 640 + [2]])
    inputs = {'image_grid_thw': torch.tensor([[1,32,80]])}
    layout = visual_layout(model, inputs, ids)
    assert layout.positions.tolist() == list(range(1,641))
    assert layout.expanded_count == 640
