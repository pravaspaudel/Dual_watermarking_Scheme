import torch
import pandas as pd

from transformers import LogitsProcessor
from src.utils.key_manager import derive_set

class PrivateWatermarkProcessor(LogitsProcessor):
    def __init__(self,key,vocab_size,green_fraction=0.5,delta_private=0.7,prev_token_size=5):
        self.key = key
        self.vocab_size=vocab_size
        self.green_fraction = green_fraction
        self.delta_private = delta_private
        self.prev_token_size = prev_token_size

    def __call__(self,input_ids,scores):
        """"
        internally this receives input_ids -> tokens generated so far
        scores -> model's next-token logits
        """

        batch_size = input_ids.shape[0]

        for b in range(batch_size):

            history = input_ids[b].tolist()

            preferred_set = derive_set(vocab_size=self.vocab_size,
                                       green_fraction=self.green_fraction,
                                       key=self.key,
                                       prev_tokens_size=self.prev_token_size,
                                       prev_tokens=history
                                       )

            preferred = torch.tensor(list(preferred_set),
                                     dtype=torch.long,
                                     device=scores.device)

            scores[b,preferred] += self.delta_private

        return scores
    
def generation_pipeline(
    prompts,
    model,
    tokenizer,
    processors=None,
    max_new_tokens=50,
    do_sample=True,
    temperature=1.0,
    top_p=0.5,
):
    """
    Generate plain and watermarked outputs for a list of prompts.

    Returns:
        pd.DataFrame
    """
    model.eval()
    results = []

    if processors is None:
        processors = []

    for i, prompt in enumerate(prompts):

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            plain_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

            watermarked_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                logits_processor=processors,
            )

        plain_txt = tokenizer.decode(
            plain_outputs[0],
            skip_special_tokens=True
        )

        watermarked_text = tokenizer.decode(
            watermarked_outputs[0],
            skip_special_tokens=True
        )

        results.append({
            "id": i,
            "prompt": prompt,
            "plain_output": plain_txt,
            "watermarked_output": watermarked_text,
        })

    return pd.DataFrame(results)