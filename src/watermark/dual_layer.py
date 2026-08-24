'''Dual-layer watermarking: public topic boost + private KGW boost in one pass.

Layer 1 (public):  the prompt's topic is inferred by cosine similarity in OPT's
own embedding space; that topic's static uniform greenlist
(data/greenlist/<topic>.csv) is boosted by delta_public at every step.
Layer 2 (private): a keyed rolling greenlist derived per position via
key_manager.derive_set(key, prev_tokens) and boosted by delta_private.

One LogitsProcessor applies both; w1/w2 switch layers off for ablations
(layer1-only = w2=0, layer2-only = w1=0, dual = w1=w2).

Everything runs in the tokenizer id-space (len(tokenizer)), shared with the
greenlist CSVs and both detectors.
'''
import os

import pandas as pd
import torch
from transformers import LogitsProcessor

from src.utils.key_manager import derive_set, generate_key
from src.watermark.first_layer import build_topic_matrix, load_uniform_greenlists

DEFAULT_GREENLIST_DIR = "data/greenlist"


class DualLayerWatermarkProcessor(LogitsProcessor):
    """
    Applies both watermark boosts to the next-token logits at every step.
    boosted_logit = base_logit + w1 * delta_public + w2 * delta_private
    """

    def __init__(self, topic_green_ids, key, vocab_size, green_fraction=0.5,
                 delta_public=2.0, delta_private=0.7, w1=1.0, w2=1.0,
                 prev_token_size=5):
        self.topic_ids = torch.tensor(topic_green_ids, dtype=torch.long)
        self.key = key
        self.vocab_size = vocab_size
        self.green_fraction = green_fraction
        self.delta_public = delta_public
        self.delta_private = delta_private
        self.w1 = w1
        self.w2 = w2
        self.prev_token_size = prev_token_size

    def __call__(self, input_ids, scores):
        if self.w1 != 0.0:                       # layer 1: fixed topic greenlist
            scores[..., self.topic_ids.to(scores.device)] += self.w1 * self.delta_public

        if self.w2 != 0.0:                       # layer 2: keyed rolling greenlist
            for b in range(input_ids.shape[0]):
                preferred = derive_set(
                    vocab_size=self.vocab_size,
                    green_fraction=self.green_fraction,
                    key=self.key,
                    prev_tokens_size=self.prev_token_size,
                    prev_tokens=input_ids[b].tolist(),
                )
                idx = torch.tensor(list(preferred), dtype=torch.long,
                                   device=scores.device)
                scores[b, idx] += self.w2 * self.delta_private
        return scores


class DualWaterMarking:
    """End-to-end dual watermarking over the uniform static greenlists.

    watermark_text(prompt)          -> str   (dual-watermarked continuation only)
    generate(prompt)                -> dict  {"topic", "topic_score",
                                              "plain_output",
                                              "dual_watermarked_output"}
    extract_topic(prompt)           -> (topic, ranked [(topic, cos), ...])
    watermark(prompts, ...)         -> pd.DataFrame batch (+ single-layer
                                       ablations when include_single_layers)
    """

    def __init__(self, model, tokenizer, key=None,
                 greenlist_dir=DEFAULT_GREENLIST_DIR, split="all",
                 greenlists=None, normed_embeddings=None, topic_matrix=None,
                 green_fraction=0.5, prev_token_size=5,
                 delta_public=2.0, delta_private=0.7, w1=1.0, w2=1.0,
                 max_new_tokens=200, temperature=1.0, top_p=0.9, seed=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.split = split
        self.vocab_size = len(tokenizer)      # id-space shared by lists + detector
        self.green_fraction = green_fraction
        self.prev_token_size = prev_token_size
        self.delta_public = delta_public
        self.delta_private = delta_private
        self.w1, self.w2 = w1, w2
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if seed is not None:
            torch.manual_seed(seed)

        if key is None:                       # reuse .env secret or mint a new one
            key = os.getenv("WATERMARK_SECRET_KEY") or generate_key()
        self.key = key

        if greenlists is None:
            greenlists = load_uniform_greenlists(greenlist_dir)
        self.greenlists = greenlists
        self.topics = sorted(greenlists)

        if normed_embeddings is None:
            emb = self.model.get_input_embeddings().weight.detach().float()
            normed_embeddings = emb / emb.norm(dim=1, keepdim=True)
        self.normed_embeddings = normed_embeddings
        self.topic_matrix = (topic_matrix if topic_matrix is not None
                             else build_topic_matrix(self.model, self.tokenizer, self.topics))

    @torch.no_grad()
    def extract_topic(self, prompt):
        """Mean-pool prompt embeddings, cosine vs each topic vector."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        vec = self.normed_embeddings[ids].mean(dim=0)
        vec = vec / vec.norm()
        scores = (self.topic_matrix @ vec).tolist()
        ranked = sorted(zip(self.topics, scores), key=lambda x: -x[1])
        return ranked[0][0], ranked

    def _make_processor(self, topic, w1, w2):
        """Both layers in one processor; set a weight to 0 to drop that layer."""
        return DualLayerWatermarkProcessor(
            topic_green_ids=self.greenlists[topic][self.split],
            key=self.key,
            vocab_size=self.vocab_size,
            green_fraction=self.green_fraction,
            delta_public=self.delta_public,
            delta_private=self.delta_private,
            w1=w1, w2=w2,
            prev_token_size=self.prev_token_size,
        )

    def _generate(self, inputs, processor=None):
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=True,
                      temperature=self.temperature, top_p=self.top_p)
        if processor is not None:
            kwargs["logits_processor"] = [processor]
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _inputs(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    def watermark_text(self, prompt, topic=None):
        """Single-shot dual watermark: returns ONLY the watermarked text.

        topic=None routes via extract_topic; pass an explicit topic to force it.
        """
        if topic is None:
            topic, _ = self.extract_topic(prompt)
        processor = self._make_processor(topic, self.w1, self.w2)
        return self._generate(self._inputs(prompt), processor)

    def generate(self, prompt, topic=None):
        """Plain vs dual-watermarked continuation of `prompt`.

        Returns {"topic", "topic_score", "plain_output",
                 "dual_watermarked_output"}.
        """
        if topic is None:
            topic, ranked = self.extract_topic(prompt)
            topic_score = round(ranked[0][1], 4)
        else:
            topic_score = None

        inputs = self._inputs(prompt)
        plain = self._generate(inputs)
        dual = self._generate(
            inputs, self._make_processor(topic, self.w1, self.w2))
        return {
            "topic": topic,
            "topic_score": topic_score,
            "plain_output": plain,
            "dual_watermarked_output": dual,
        }

    def watermark(self, prompts, include_single_layers=True):
        """Batch: one row per prompt with all four variants.

        include_single_layers adds layer1_only_output (w2=0) and
        layer2_only_output (w1=0) for ablation/evaluation.
        """
        rows = []
        for i, prompt in enumerate(prompts):
            topic, ranked = self.extract_topic(prompt)
            inputs = self._inputs(prompt)

            row = {
                "id": i,
                "prompt": prompt,
                "topic": topic,
                "topic_score": round(ranked[0][1], 4),
                "plain_output": self._generate(inputs),
                "dual_watermarked_output": self._generate(
                    inputs, self._make_processor(topic, self.w1, self.w2)),
            }
            if include_single_layers:
                row["layer1_only_output"] = self._generate(
                    inputs, self._make_processor(topic, 1.0, 0.0))
                row["layer2_only_output"] = self._generate(
                    inputs, self._make_processor(topic, 0.0, 1.0))
            rows.append(row)
        return pd.DataFrame(rows)
