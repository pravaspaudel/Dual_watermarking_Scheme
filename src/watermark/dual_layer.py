import torch
import pandas as pd

from transformers import LogitsProcessor
from src.utils.key_manager import derive_set
from src.watermark.first_layer import load_topic_greenlists, build_topic_matrix

class DualLayerWatermarkProcessor(LogitsProcessor):
    """
    Applies both watermark boosts to the next-token logits at every step.
    boosted_logit = base_logit + w1 * delta_public + w2 * delta_private
    """
    def __init__(self, topic_green_ids, key, vocab_size, green_fraction=0.5,
                 delta_public=2.0, delta_private=0.7, w1=1.0, w2=1.0, prev_token_size=5):
        self.topic_green_ids = torch.tensor(topic_green_ids, dtype=torch.long)
        self.key = key
        self.vocab_size = vocab_size
        self.green_fraction = green_fraction
        self.delta_public = delta_public
        self.delta_priv = delta_private
        self.w1 = w1
        self.w2 = w2
        self.prev_token_size = prev_token_size

    def __call__(self, input_ids, scores):
        if self.w1 != 0.0:
            topic_ids = self.topic_green_ids.to(scores.device)
            scores[..., topic_ids] += self.w1 * self.delta_public
        if self.w2 != 0.0:
            batch_size = input_ids.shape[0]
            for b in range(batch_size):
                history = input_ids[b].tolist()
                preferred_set = derive_set(
                    vocab_size=self.vocab_size,
                    green_fraction=self.green_fraction,
                    key=self.key,
                    prev_tokens_size=self.prev_token_size,
                    prev_tokens=history,
                )
                preferred = torch.tensor(
                    list(preferred_set), dtype=torch.long, device=scores.device
                )
                scores[b, preferred] += self.w2 * self.delta_priv
        return scores


class DualWaterMarking:
    """
    This is class for end to end dual watermarking . It will do the following:-
    1. First, it will extract topic from prompt (same as layer 1).
    2. build a DualLayerWatermarkProcessor for layer 1 + layer 2 functionaility. 
    3. generate plain vs dual watermarked text for comparison
    """

    def __init__(self,model,tokenizer,key,greenlist_csv="topic_greenlists.csv",green_fraction=0.5,delta_public=2.0,
                 delta_private=0.7,w1 = 1.0,w2=1.0,prev_token_size=5,temperature=1.0,top_p=0.9,max_new_tokens=50):

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.key = key
        self.vocab_size = model.get_input_embeddings().weight.shape[0]
 
        self.green_fraction = green_fraction
        self.delta_public = delta_public
        self.delta_priv = delta_private
        self.w1 = w1
        self.w2 = w2
        self.prev_token_size = prev_token_size
 
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
 
        self.greenlists = load_topic_greenlists(greenlist_csv)
        self.topics = sorted(self.greenlists)
        self.topic_matrix = build_topic_matrix(model, tokenizer, self.topics)

    @torch.no_grad()
    def extract_topic(self,prompt):
        """Mean-pool prompt token embeddings, cosine vs each topic vector."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        vec = self.model.get_input_embeddings().weight[ids].mean(dim=0)
        vec = vec / vec.norm()
        scores = self.topic_matrix @ vec
        ranked = sorted(zip(self.topics, scores.tolist()), key=lambda x: -x[1])
        return ranked[0][0], ranked

    def _make_processor(self,topic,w1,w2):

        return DualLayerWatermarkProcessor(
            topic_green_ids=self.greenlists[topic],
            key=self.key,
            vocab_size=self.vocab_size,
            green_fraction=self.green_fraction,
            delta_public=self.delta_public,
            delta_private=self.delta_priv,
            w1=w1,
            w2=w2,
            prev_token_size=self.prev_token_size,
        )

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

    def watermark(self, prompts, include_single_layers=True):
        """
        Generate plain vs dual-watermarked output for each prompt.
        If include_single_layers=True, also generates Layer-1-only (w2=0) and
        Layer-2-only (w1=0) outputs - useful for Phase 7 ablation/evaluation.
        Returns a DataFrame.
        """
        rows = []
        for i, prompt in enumerate(prompts):
            topic, ranked = self.extract_topic(prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
 
            plain = self._generate(inputs)
            dual_output = self._generate(
                inputs, processors=[self._make_processor(topic, self.w1, self.w2)]
            )
 
            row = {
                "id": i,
                "prompt": prompt,
                "topic": topic,
                "topic_score": round(ranked[0][1], 4),
                "plain_output": plain,
                "dual_watermarked_output": dual_output,
            }
 
            if include_single_layers:
                row["layer1_only_output"] = self._generate(
                    inputs, processors=[self._make_processor(topic, 1.0, 0.0)]
                )
                row["layer2_only_output"] = self._generate(
                    inputs, processors=[self._make_processor(topic, 0.0, 1.0)]
                )
 
            rows.append(row)
 
        return pd.DataFrame(rows)