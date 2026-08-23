'''First layer: public, topic-based watermarking.

Extracts each prompt's topic (cosine similarity in OPT's own embedding space,
against predetermined topic vectors) and boosts the logits of that topic's
greenlist tokens during generation.
'''
import torch
import pandas as pd

from transformers import LogitsProcessor


def load_topic_greenlists(csv_path):
    """CSV(topic, token_id, ...) -> {topic: [token_id, ...]}."""
    df = pd.read_csv(csv_path)
    return {t: g["token_id"].astype(int).tolist() for t, g in df.groupby("topic")}


def build_topic_matrix(model, tokenizer, topics):
    """(K, d) unit-norm rows: OPT input embedding of each topic's first subtoken."""
    emb = model.get_input_embeddings().weight.detach()
    emb = emb / emb.norm(dim=1, keepdim=True)
    ids = [tokenizer.encode(" " + t, add_special_tokens=False)[0] for t in topics]
    return emb[ids]


class TopicBoostProcessor(LogitsProcessor):
    """Adds `delta` to the logits of one topic's greenlist tokens at every step."""

    def __init__(self, green_token_ids, delta=2.0):
        self.green_token_ids = torch.tensor(green_token_ids, dtype=torch.long)
        self.delta = delta

    def __call__(self, input_ids, scores):
        # scores: (batch, vocab) next-token logits
        scores[..., self.green_token_ids.to(scores.device)] += self.delta
        return scores


class TopicWiseWatermarking:
    """Extract topic -> assign -> boost its greenlist tokens while generating."""

    def __init__(self, model, tokenizer, greenlist_csv="topic_greenlists.csv",
                 delta=2.0, temperature=1.0, top_p=0.9, max_new_tokens=50):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.delta = delta
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.greenlists = load_topic_greenlists(greenlist_csv)
        self.topics = sorted(self.greenlists)
        self.topic_matrix = build_topic_matrix(model, tokenizer, self.topics)

    @torch.no_grad()
    def extract_topic(self, prompt):
        """Mean-pool prompt token embeddings, cosine vs each topic vector."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        vec = self.model.get_input_embeddings().weight[ids].mean(dim=0)
        vec = vec / vec.norm()
        scores = self.topic_matrix @ vec
        ranked = sorted(zip(self.topics, scores.tolist()), key=lambda x: -x[1])
        return ranked[0][0], ranked

    def _generate(self, inputs, processors=None):
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                logits_processor=processors,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def watermark(self, prompts):
        """Plain vs topic-boosted generation for each prompt. Returns DataFrame."""
        rows = []
        for i, prompt in enumerate(prompts):
            topic, ranked = self.extract_topic(prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            plain = self._generate(inputs)
            watermarked = self._generate(
                inputs,
                processors=[TopicBoostProcessor(self.greenlists[topic], self.delta)],
            )
            rows.append({
                "id": i,
                "prompt": prompt,
                "topic": topic,
                "topic_score": round(ranked[0][1], 4),
                "plain_output": plain,
                "watermarked_output": watermarked,
            })
        return pd.DataFrame(rows)
