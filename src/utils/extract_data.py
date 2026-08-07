from datasets import load_dataset
from itertools import islice
import json

def load_c4_dataset(configuration="realnewslike",
                    split="train",
                    streaming=True
    ):

    dataset = load_dataset("allenai/c4",configuration,split=split,streaming=streaming)
    return dataset


def process_c4(dataset,tokenizer,num_docs=1000,prompt_length=50,ref_length=200):
    samples = []
    seen_prompts = set()

    for i,item in enumerate(dataset):
        text = item["text"]
        token = tokenizer(text,add_special_token=False)["input_ids"]

        if len(token) < prompt_length + ref_length:
            continue

        prompt_tokens = token[:prompt_length]
        ref_tokens = token[prompt_length:prompt_length+ref_length]

        prompt_txt = tokenizer.decode(prompt_tokens,skip_special_tokens=True)
        ref_txt = tokenizer.decode(ref_tokens,skip_special_tokens=True)

        if prompt_txt in seen_prompts:
            continue

        seen_prompts.add(prompt_txt)

        data = {
            "prompt_text":prompt_txt,
            "prompt_tokens":prompt_tokens,
            "reference_text": ref_txt,
            "reference_tokens": ref_tokens,
            "full_text":text,
            "domain": "realnews"
        }
        samples.append(data)

        if len(samples) >= num_docs:
            break

    return samples



