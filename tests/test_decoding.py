import types

import pytest
import torch

from rase.candidate_vocab import ObjectCandidateVocabulary
from rase.decoding import OnlineRollback, PrefixObjectMatcher
from test_candidate_vocab import _NLP


def vocabulary(words):
    return ObjectCandidateVocabulary(dict(schema=ObjectCandidateVocabulary.SCHEMA,
        benchmarks={'test':dict(entries=[dict(identity=w, surface=w, origin='chair') for w in words])}),_NLP())


def test_streaming_boundaries_no_future_and_no_premature_subword_match():
    m=PrefixObjectMatcher(vocabulary(['dog','traffic','traffic light']),_NLP(),'test')
    assert m.update('dog')==[]
    assert m.update('doghouse ')==[]
    assert m.update('doghouse traffic ')==[]
    got=m.update('doghouse traffic light ')
    assert [x['word'] for x in got]==['traffic light']
    assert got[0]['char_start']==9


def test_first_token_repeated_identity_and_eos_flush():
    m=PrefixObjectMatcher(vocabulary(['car']),_NLP(),'test')
    assert m.update('car ' )[0]['char_start']==0
    assert m.update('car car ' )[0]['char_start']==4
    assert m.update('car car car',terminal=True)[0]['char_start']==8
    m.reset('car ')
    assert m.update('car car ')[0]['char_start']==4


class Cache:
    def __init__(self,tokens): self.tokens=list(tokens)
    def get_seq_length(self): return len(self.tokens)
    def crop(self,n): self.tokens=self.tokens[:n]


class Model:
    generation_config=types.SimpleNamespace(eos_token_id=0)
    def __init__(self, policy):
        self.policy=policy;self.backend='sdpa';self.last_probe_token=None;self.calls=[]
    def __call__(self,input_ids,past_key_values=None,**kw):
        ids=input_ids[0].tolist()
        if past_key_values is None:
            cache=Cache(ids)
        else:
            cache=past_key_values
            assert kw['cache_position'].item()==len(cache.tokens)
            assert len(ids)==1
            cache.tokens.extend(ids)
        if self.backend=='rase_attention_capture': self.last_probe_token=ids[-1]
        self.calls.append((self.backend,list(cache.tokens),id(cache)))
        logits=torch.full((1,1,40),-1000.)
        for rank,token in enumerate(self.policy(cache.tokens[1:])):
            logits[0,0,token]=100-rank
        return types.SimpleNamespace(past_key_values=cache,logits=logits)
    def generate(self,*a,**kw): raise AssertionError('Full caption generate must not be called')


def runner(monkeypatch,pieces,words,policy,risks):
    import rase.decoding as module
    r=object.__new__(OnlineRollback)
    r.model=Model(policy);r.threshold=.49;r.max_new_tokens=512;r.device='cpu'
    r.capture={};r.candidate_vocabulary=vocabulary(words);r.nlp=_NLP()
    r.tokenizer=types.SimpleNamespace(decode=lambda ids,**kw: ''.join(pieces[i] for i in ids if i!=0))
    r._set_attention=lambda backend:setattr(r.model,'backend',backend)
    r._online_signal=lambda visual:(torch.tensor([risks.get(r.model.last_probe_token,.1)]),False)
    r._pending_score=lambda signal:float(signal.item())
    monkeypatch.setattr(module,'visual_layout',lambda *a,**kw:types.SimpleNamespace(positions=torch.tensor([0])))
    return r


def test_repeated_rejection_restores_kv_and_accumulates_bans(monkeypatch):
    pieces={i:'obj'+str(i-10) for i in range(10,23)}
    r=runner(monkeypatch,pieces,list(pieces.values()),
        lambda ids:list(range(10,23)) if not ids else [0],{i:.9 for i in range(10,22)})
    out=r.generate_online({'input_ids':torch.tensor([[99]])},benchmark='test',audit_cache=True)
    assert out['caption']=='obj12'
    assert out['total_rollbacks']==out['max_rollbacks_at_one_prefix']==12
    assert out['counters']['kv_restores']==12
    assert out['counters']['generation_prefills']==1
    assert [x['prefix_attempt'] for x in out['actions']]==list(range(1,13))
    assert [x['banned_first_ids'] for x in out['actions']]==[list(range(10,11+i)) for i in range(12)]
    assert all(x['detected_at_generated_tokens']==2 for x in out['actions'])
    assert all(x['argmax_equal'] for x in out['cache_checks'])
    assert len(out['mentions'])==1 and out['mentions'][0]['query_pos']==1
    assert out['mentions'][0]['risk']<.49


def test_split_object_uses_original_first_subtoken_signal(monkeypatch):
    pieces={1:'A',2:' do',3:'g',4:'.'}
    r=runner(monkeypatch,pieces,['dog'],lambda ids:[0],{2:.8,3:.1})
    out=r.generate_online({'input_ids':torch.tensor([[99]])},benchmark='test',
        intervene=False,forced_ids=[1,2,3,4,0],audit_query_positions=[1])
    assert len(out['mentions'])==1
    assert out['mentions'][0]['first_id']==2
    assert out['mentions'][0]['risk']==pytest.approx(.8)
    assert out['audit_scores']['0']==pytest.approx(.8)
    assert out['mentions'][0]['recognition_delay_tokens']==2


def test_teacher_forcing_cannot_be_used_for_intervention(monkeypatch):
    r=runner(monkeypatch,{1:'dog'},['dog'],lambda ids:[0],{})
    with pytest.raises(ValueError,match='requires intervene=False'):
        r.generate_online({'input_ids':torch.tensor([[99]])},benchmark='test',forced_ids=[1])


def test_length_boundary_does_not_force_accept_high_risk_token(monkeypatch):
    pieces={i:'obj'+str(i-10) for i in range(10,23)}
    r=runner(monkeypatch,pieces,list(pieces.values()),
        lambda ids:list(range(10,23)) if not ids else [0],{i:.9 for i in range(10,22)})
    r.max_new_tokens=1
    out=r.generate_online({'input_ids':torch.tensor([[99]])},benchmark='test')
    assert out['total_rollbacks']==12
    assert out['caption']=='obj12'
    assert out['stopping_reason']=='length_limit_after_candidate_checks'
    assert out['mentions'][0]['risk']<r.threshold
    gen=[x[2] for x in r.model.calls if x[0]=='sdpa'][1:]
    probe=[x[2] for x in r.model.calls if x[0]=='rase_attention_capture']
    assert gen and probe and all(g!=p for g,p in zip(gen,probe))
