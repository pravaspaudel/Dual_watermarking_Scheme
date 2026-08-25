import math
from collections import Counter

import numpy as np
import torch
import pandas as pd
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from datasets import load_dataset

from src.utils.key_manager import derive_set

_JUDGE_CACHE = {}


def load_judge_model(judge_model_name="gpt2-large", device="cuda",
                     torch_dtype=None, load_in_4bit=False):
    """
    Loads (and caches) an independent judge model for external perplexity
    scoring. Default: gpt2-large. Swap to e.g. "EleutherAI/pythia-1.4b" if
    preferred. NEVER pass an OPT checkpoint here - that would bias the
    eval toward the watermark's own generator.

    torch_dtype:   e.g. torch.float16 / torch.bfloat16; None keeps the
                   checkpoint's native fp32 (31 GB for a 7B judge -> will
                   OOM everywhere).
    load_in_4bit:  bitsandbytes 4-bit quantization. REQUIRED for >=7B judges
                   on a 16 GB T4 (fp16 weights alone are ~15.4 GB). When True
                   the model is placed via device_map and .to(device) skipped.
    """
    if judge_model_name in _JUDGE_CACHE:
        return _JUDGE_CACHE[judge_model_name]

    tokenizer = AutoTokenizer.from_pretrained(judge_model_name)

    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if load_in_4bit:
        # Newer transformers removed the `load_in_4bit=` shortcut from
        # from_pretrained; the BitsAndBytesConfig route works on both old
        # and new versions.
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(judge_model_name, **model_kwargs)

    if not load_in_4bit:
        model = model.to(device)

    model.eval()

    _JUDGE_CACHE[judge_model_name] = (model, tokenizer)
    return model, tokenizer


def split_continuation(prompt, full_text):
    """
    Your generation pipeline decodes: prompt + generated continuation.
    This extracts only the generated continuation, so metrics aren't
    diluted by the (identical, uninteresting) prompt text.
    Falls back to the full text if the prefix doesn't match exactly
    (can happen due to tokenizer detokenization quirks).
    """
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()


@torch.no_grad()
def _perplexity_core(prompt, continuation, model, tokenizer, device="cuda"):
    """
    Shared implementation: perplexity of `continuation` under `model`,
    conditioned on `prompt` as context. Only continuation tokens
    contribute to the loss (prompt tokens are masked with -100).
    PPL = exp(average negative log likelihood).
    """
    if not continuation.strip():
        return float("nan")

    prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    full_enc = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False)

    prompt_ids = prompt_enc.input_ids
    full_ids = full_enc.input_ids
    prompt_len = prompt_ids.shape[1]

    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None:
        max_len = getattr(model.config, "n_positions", 1024)

    if full_ids.shape[1] > max_len:
        full_ids = full_ids[:, :max_len]

    prompt_len = min(prompt_len, full_ids.shape[1])
    full_ids = full_ids.to(device)

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens out of the loss

    if (labels != -100).sum().item() == 0:
        return float("nan")  # prompt ate the whole context window

    outputs = model(input_ids=full_ids, labels=labels)
    return torch.exp(outputs.loss).item()


def compute_self_perplexity(prompt, continuation, model, tokenizer, device="cuda"):
    """
    Perplexity scored by the SAME model that generated the text
    (e.g. OPT-2.7B generates -> OPT-2.7B evaluates). Useful as a sanity
    check, but NOT the required quality metric - use compute_external_perplexity
    for the reported numbers, since a model scoring its own text is biased
    toward looking fluent.
    """
    return _perplexity_core(prompt, continuation, model, tokenizer, device)


def compute_external_perplexity(prompt, continuation, judge_model, judge_tokenizer, device="cuda"):
    """
    Perplexity scored by an INDEPENDENT judge model
    (e.g. OPT-2.7B generates -> GPT-2-large evaluates). This is the
    metric required by the spec.
    """
    return _perplexity_core(prompt, continuation, judge_model, judge_tokenizer, device)


def compute_self_perplexity_batch(prompts, continuations, model, tokenizer, device="cuda"):
    return [
        compute_self_perplexity(p, c, model, tokenizer, device)
        for p, c in zip(prompts, continuations)
    ]


def compute_external_perplexity_batch(prompts, continuations, judge_model, judge_tokenizer, device="cuda"):
    return [
        compute_external_perplexity(p, c, judge_model, judge_tokenizer, device)
        for p, c in zip(prompts, continuations)
    ]
