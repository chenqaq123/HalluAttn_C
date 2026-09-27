"""Prefix-specific rollback with generation and evidence cache restoration."""
from bisect import bisect_right
from collections import Counter
import copy
import hashlib
import json
import re

import torch

from .runtime import RuntimeBase
from .model_adapter import visual_layout


class PrefixObjectMatcher:
    """Commit lexical decisions at word boundaries using left context only.

    A complete shorter phrase waits when it could extend into a longer allowed
    phrase. Decisions already committed are not revised by later sentence text.
    Every occurrence is checked, including a noun starting at generated token 0.
    """
    def __init__(self, vocabulary, nlp, benchmark):
        self.phrases, self.max_length = vocabulary.compiled[benchmark]
        self.nlp = nlp
        self.lemma = vocabulary._lemma
        self.extensions = {key[:n] for key in self.phrases for n in range(1, len(key))}
        self.cursor = 0
        self.last_stable_text = None

    def reset(self, prefix_text):
        self.cursor = len(prefix_text)
        self.last_stable_text = None

    def update(self, text, *, terminal=False):
        stable_end = len(text)
        if not terminal:
            unfinished = re.search(r"[\w\uFFFD]+$", text)
            if unfinished:
                stable_end = unfinished.start()
        stable = text[:stable_end]
        if stable == self.last_stable_text and not terminal:
            return []
        self.last_stable_text = stable
        tokens = list(self.nlp(stable))
        index = next((i for i, t in enumerate(tokens) if t.idx >= self.cursor), len(tokens))
        result = []
        while index < len(tokens):
            first = tokens[index]
            if first.is_space or first.is_punct:
                self.cursor = first.idx + len(first.text); index += 1
                continue
            available = []
            for token in tokens[index:index+self.max_length]:
                if token.is_space or token.is_punct:
                    break
                available.append(token)
            key = tuple(self.lemma(t) for t in available)
            # At this boundary the last complete word may still be the start of
            # a compound. Wait for only the next lexical unit, not a full answer.
            reaches_end = index + len(available) == len(tokens)
            if not terminal and reaches_end and key in self.extensions:
                break
            matched = None
            for length in range(len(available), 0, -1):
                entry = self.phrases.get(key[:length])
                if entry and available[length-1].pos_ in {'NOUN', 'PROPN'}:
                    matched = (entry, length); break
            if matched:
                entry, length = matched
                end = available[length-1].idx + len(available[length-1].text)
                result.append(dict(word=entry.identity, surface=text[first.idx:end],
                    char_start=first.idx, char_end=end, matched_term=entry.surface,
                    vocabulary_origin=entry.origin))
                self.cursor = end; index += length
            else:
                self.cursor = first.idx + len(first.text); index += 1
        return result


def prefix_fingerprint(prompt, generated):
    return hashlib.sha256(json.dumps([*prompt, *generated], separators=(',', ':')).encode()).hexdigest()


class OnlineRollback(RuntimeBase):
    """Incremental generation with prefix-specific object rejection."""

    def _before_online_prefill(self, visual):
        pass

    def _after_online_prefill(self):
        pass

    def _before_online_probe(self, token):
        pass

    def _online_signal(self, visual):
        raise NotImplementedError("The evidence adapter supplies the current token signal")

    def _pending_score(self, signal):
        raise NotImplementedError("The detector adapter supplies the object risk")

    def _finish_online(self):
        pass

    def _online_visual_positions(self, inputs):
        # Video adapters override visual-token resolution.
        return visual_layout(self.model, inputs, inputs['input_ids']).positions.to(self.device)

    @torch.inference_mode()
    def generate_online(self, inputs, *, benchmark, intervene=True, forced_ids=None,
                        audit_query_positions=(), pulse=None, audit_cache=False):
        if forced_ids is not None and intervene:
            raise ValueError('Teacher forcing requires intervene=False')
        pulse = pulse or (lambda **kw: None)
        prompt_ids = inputs['input_ids']; n = int(prompt_ids.shape[-1])
        prompt = prompt_ids[0].tolist()
        matcher = PrefixObjectMatcher(self.candidate_vocabulary, self.nlp, benchmark)
        visual = self._online_visual_positions(inputs)
        self._before_online_prefill(visual)
        self._set_attention('sdpa')
        output = self.model(**inputs, use_cache=True, output_attentions=False,
                            logits_to_keep=1, return_dict=True)
        gen_cache = output.past_key_values
        self._after_online_prefill()
        # Independent canonical S state: sharing a mutable cache would corrupt
        # both the generation distribution and the detector query position.
        probe_cache = copy.deepcopy(gen_cache)
        logits = output.logits[:, -1, :].float().clone()
        del output
        if not hasattr(gen_cache, 'crop') or not hasattr(probe_cache, 'crop'):
            raise ValueError('Online rollback requires croppable KV caches')
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        ids, starts, before_logits = [], [], []
        signals, mentions, actions = {}, [], []
        rejected = {}; attempts = Counter(); audited_scores = {}
        audit_positions = set(audit_query_positions)
        cache_checks = []
        text = ''; pending_action = None
        counters = dict(generation_prefills=1, probe_prefill_copies=1,
            generation_token_forwards=0, probe_token_forwards=0, kv_restores=0,
            peak_pending_attention_rows=0)

        def risk_at(index):
            signal, fallback = signals[index]
            if not isinstance(signal, (int, float)):
                signal = self._pending_score(signal)
                signals[index] = (signal, fallback)
            return signal, fallback

        try:
            while len(ids) < self.max_new_tokens:
                key = tuple(ids)
                banned = rejected.get(key, set()) if intervene else set()
                selection = logits.clone()
                if banned:
                    selection[:, sorted(banned)] = -torch.inf
                if not torch.isfinite(selection).any():
                    raise RuntimeError('All token alternatives exhausted')
                if forced_ids is not None:
                    if len(ids) >= len(forced_ids):
                        break
                    token = int(forced_ids[len(ids)])
                else:
                    token = int(selection.argmax(-1).item())
                if token in banned:
                    raise AssertionError('Rejected token was selected again at the same prefix')
                if pending_action is not None:
                    pending_action['replacement_first_id'] = token
                    pending_action = None
                index = len(ids)
                starts.append(len(text)); before_logits.append(logits.clone()); ids.append(token)
                terminal = token in eos or len(ids) == self.max_new_tokens
                if forced_ids is not None and len(ids) == len(forced_ids):
                    terminal = True
                if token not in eos:
                    token_ids = torch.tensor([[token]], device=self.device)
                    common = dict(input_ids=token_ids, attention_mask=torch.ones((1,n+index+1),
                        device=self.device, dtype=torch.long),
                        cache_position=torch.tensor([n+index], device=self.device),
                        use_cache=True, logits_to_keep=1, return_dict=True)
                    if int(gen_cache.get_seq_length()) != n+index or int(probe_cache.get_seq_length()) != n+index:
                        raise AssertionError('KV cache and token prefix lengths diverged')
                    self._set_attention('sdpa')
                    out = self.model(**common, past_key_values=gen_cache, output_attentions=False)
                    gen_cache = out.past_key_values; logits = out.logits[:, -1, :].float().clone()
                    del out; counters['generation_token_forwards'] += 1
                    self._set_attention('rase_attention_capture')
                    self.capture.update(raw={}, attention={})
                    self._before_online_probe(token)
                    out = self.model(**common, past_key_values=probe_cache, output_attentions=True)
                    probe_cache = out.past_key_values
                    distribution, fallback = self._online_signal(visual)
                    signals[index] = (distribution, fallback)
                    del out, distribution
                    self.capture.update(raw={}, attention={})
                    counters['probe_token_forwards'] += 1
                    if index in audit_positions:
                        audited_scores[str(index-1)] = risk_at(index)[0]
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                fresh = matcher.update(text, terminal=terminal)
                rolled_back = False
                for candidate in fresh:
                    first = bisect_right(starts, candidate['char_start'])-1
                    if first not in signals:
                        raise AssertionError('Missing original first-subtoken signal for a streaming candidate')
                    risk, fallback = risk_at(first)
                    mention = dict(**candidate, token_pos=n+first-1, query_pos=n+first,
                        gen_pos=first-1, first_id=ids[first], risk=risk,
                        observed_through_gen_pos=len(ids)-1,
                        recognition_delay_tokens=len(ids)-1-first,
                        underflow_fallback=fallback)
                    if intervene and risk >= self.threshold:
                        prefix = tuple(ids[:first]); forbidden = rejected.setdefault(prefix, set())
                        if ids[first] in forbidden:
                            raise AssertionError('A banned first token returned under the same prefix')
                        forbidden.add(ids[first]); attempts[prefix] += 1
                        action = dict(slot=n+first-1, word=mention['word'], surface=mention['surface'],
                            risk=risk, matched_term=mention['matched_term'],
                            vocabulary_origin=mention['vocabulary_origin'], banned_first_id=ids[first],
                            banned_first_ids=sorted(forbidden), prefix_attempt=attempts[prefix],
                            retry_key_sha256=prefix_fingerprint(prompt,prefix),
                            detected_at_generated_tokens=len(ids),
                            discarded_generated_tokens=len(ids)-first,
                            detected_prefix=text)
                        actions.append(action); pending_action=action
                        restore_length = n+first
                        gen_cache.crop(restore_length); probe_cache.crop(restore_length)
                        logits = before_logits[first].clone()
                        if audit_cache:
                            self._set_attention('sdpa')
                            short = torch.tensor([[*prompt, *prefix]], device=self.device)
                            check = self.model(input_ids=short, attention_mask=torch.ones_like(short),
                                **self._vision_kwargs(inputs), use_cache=False, logits_to_keep=1, return_dict=True)
                            error = float((check.logits[:, -1, :].float()-logits).abs().max())
                            # Full-sequence SDPA need not be bit-identical; the
                            # restored original next-token distribution must agree.
                            cache_checks.append(dict(prefix_tokens=restore_length, max_logit_difference=error,
                                argmax_equal=int(check.logits[:, -1, :].argmax()) == int(logits.argmax())))
                            del check
                            if not cache_checks[-1]['argmax_equal']:
                                raise ValueError('Restored generation logits fail fresh-prefix argmax audit')
                        ids = ids[:first]; starts = starts[:first]; before_logits = before_logits[:first]
                        signals = {i:v for i,v in signals.items() if i<first}
                        mentions = [m for m in mentions if m['query_pos'] < n+first]
                        text = self.tokenizer.decode(ids, skip_special_tokens=True)
                        matcher.reset(text)
                        counters['kv_restores'] += 1
                        pulse(stage='rollback', actions=len(actions), generated_tokens=len(ids),
                            word=action['word'], risk=risk, prefix_attempt=action['prefix_attempt'])
                        rolled_back = True
                        break
                    mentions.append(mention)
                if rolled_back:
                    continue
                # Compact signals at lexical boundaries; retain at most eight
                # unresolved attention rows and score older rows eagerly.
                frontier = max(0, bisect_right(starts, matcher.cursor)-1)
                signals = {i:v for i,v in signals.items() if i>=frontier}
                raw_indices = [i for i,v in signals.items() if not isinstance(v[0], (int, float))]
                for old_index in raw_indices[:-8]:
                    risk_at(old_index)
                counters['peak_pending_attention_rows'] = max(counters['peak_pending_attention_rows'],min(8,len(raw_indices)))
                pulse(stage='token', actions=len(actions), generated_tokens=len(ids))
                if terminal:
                    break
            if intervene and any(m['risk'] >= self.threshold for m in mentions):
                raise AssertionError('Accepted mention exceeds the rejection threshold')
            counts = [dict(prefix_key_sha256=prefix_fingerprint(prompt,k),
                generated_prefix_length=len(k), rollbacks=v) for k,v in attempts.items()]
            return dict(caption=text.strip(), full_ids=torch.tensor([[*prompt,*ids]], device=self.device),
                mentions=mentions, actions=actions, rollback_counts_by_prefix=counts,
                rollback_counts_by_absolute_slot=dict(Counter(a['slot'] for a in actions)),
                total_rollbacks=len(actions), max_rollbacks_at_one_prefix=max(attempts.values(), default=0),
                stopping_reason='eos_after_candidate_checks' if ids and ids[-1] in eos else 'length_limit_after_candidate_checks',
                counters=counters, cache_checks=cache_checks, audit_scores=audited_scores,
                underflow_fallback_mentions=sum(m['underflow_fallback'] for m in mentions))
        finally:
            self.capture.update(raw={}, attention={})
            self._finish_online()
            self._set_attention('sdpa')
