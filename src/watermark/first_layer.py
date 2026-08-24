'''First layer: public, topic-based watermarking over STATIC uniform greenlists.

Each topic has a fixed greenlist CSV under data/greenlist/<topic>.csv
(columns: token_id, token, similarity, type in {content, connector, residual});
connectors and below-threshold residuals were round-robined across topics, so
every list covers the whole vocabulary.

The prompt topic is inferred by cosine similarity in OPT's own embedding space
(the same geometry that built the lists); the assigned topic's greenlist is then
boosted during sampling. Structure mirrors second_layer.py.
'''
import glob
import os

import pandas as pd
import torch
from transformers import LogitsProcessor


class TopicBoostProcessor(LogitsProcessor):
    """Adds `delta` to one topic's greenlist logits at every generation step."""

    def __init__(self, green_token_ids, delta):
        self.ids = torch.tensor(green_token_ids, dtype=torch.long)
        self.delta = delta

    def __call__(self, input_ids, scores):
        scores[..., self.ids.to(scores.device)] += self.delta
        return scores


def load_uniform_greenlists(greenlist_dir):
    """One CSV per topic -> {topic: {"all": [ids], "content": [ids]}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(greenlist_dir, "*.csv"))):
        topic = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        assert df["token_id"].is_unique, f"duplicate ids in {path}"
        out[topic] = {
            "all": df["token_id"].astype(int).tolist(),
            "content": df.loc[df["type"] == "content",
                              "token_id"].astype(int).tolist(),
        }
    return out


def load_topic_greenlists(csv_path):
    """Legacy single-CSV loader (kept for dual_layer.py compatibility)."""
    df = pd.read_csv(csv_path)
    return {t: g["token_id"].astype(int).tolist()
            for t, g in df.groupby("topic")}


def build_topic_matrix(model, tokenizer, topics):
    """(K, d) unit-norm rows: OPT input embedding of each topic's first subtoken."""
    emb = model.get_input_embeddings().weight.detach().float()
    emb = emb / emb.norm(dim=1, keepdim=True)
    ids = [tokenizer.encode(" " + t, add_special_tokens=False)[0] for t in topics]
    return emb[ids]


class TopicWiseWatermarking:
    """Public topic watermark: route the prompt, boost its static greenlist.

    extract_topic(prompt)   -> (best_topic, ranked [(topic, cos_sim), ...])
    generate(prompt)        -> {"topic", "topic_score",
                                "plain_output", "watermarked_output"}
    generate_batch(prompts) -> pd.DataFrame of generate() dicts (+ id column)
    """

    def __init__(self, model, tokenizer, greenlist_dir="data/greenlist",
                 delta=4.0, split="all", max_new_tokens=200,
                 temperature=1.0, top_p=0.9, seed=0):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.delta = delta
        self.split = split
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if seed is not None:
            torch.manual_seed(seed)

        self.greenlists = load_uniform_greenlists(greenlist_dir)
        self.topics = sorted(self.greenlists)

        emb = self.model.get_input_embeddings().weight.detach().float()
        self.normed_embeddings = emb / emb.norm(dim=1, keepdim=True)
        self.topic_matrix = self.normed_embeddings[
            [self.tokenizer.encode(" " + t, add_special_tokens=False)[0]
             for t in self.topics]
        ]

    @torch.no_grad()
    def extract_topic(self, prompt):
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_vec = self.normed_embeddings[ids].mean(dim=0)
        prompt_vec = prompt_vec / prompt_vec.norm()
        scores = self.topic_matrix @ prompt_vec
        ranked = sorted(zip(self.topics, scores.tolist()), key=lambda x: -x[1])
        return ranked[0][0], ranked

    def _generate(self, inputs, processor=None):
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=True,
                      temperature=self.temperature, top_p=self.top_p)
        if processor is not None:
            kwargs["logits_processor"] = [processor]
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def generate(self, prompt, topic=None):
        """Plain vs watermarked continuation of `prompt`.

        topic=None infers it via extract_topic; pass an explicit topic to
        force routing (e.g. for controlled experiments).
        """
        if topic is None:
            topic, ranked = self.extract_topic(prompt)
            topic_score = round(ranked[0][1], 4)
        else:
            topic_score = None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        processor = TopicBoostProcessor(self.greenlists[topic][self.split],
                                        self.delta)
        return {
            "topic": topic,
            "topic_score": topic_score,
            "plain_output": self._generate(inputs),
            "watermarked_output": self._generate(inputs, processor),
        }

    def generate_batch(self, prompts):
        rows = [{"id": i, "prompt": p, **self.generate(p)}
                for i, p in enumerate(prompts)]
        return pd.DataFrame(rows)
