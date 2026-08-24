'''First-layer (public, topic-based) watermark detection.

Counterpart of kgw_detection.py for the topic-wise scheme: re-infers the topic
from the text via cosine similarity in OPT's embedding space, looks up that
topic's fixed greenlist (static uniform CSVs under data/greenlist/<topic>.csv),
counts green-token hits and tests them against the H0 rate gamma = |G|/vocab_size
with a z-score plus an exact binomial upper-tail p-value.
'''
import math

import pandas as pd
import torch
from scipy.stats import binomtest

Z_THRESHOLD = 4.0
P_VALUE_THRESHOLD = 0.05

from src.watermark.first_layer import build_topic_matrix, load_uniform_greenlists


def prepare(
    model,
    tokenizer,
    greenlist_dir="data/greenlist",
    split="all",
):
    """
        builds the shared detection state once: per-topic green sets for the given
        split plus the unit-norm embedding table and topic matrix used for routing.
        MUST use the same model/tokenizer/dir/split as generation so routing and
        gamma match. returns dict: topics, green_sets, normed_embeddings, topic_matrix
    """

    greenlists = load_uniform_greenlists(greenlist_dir)
    topics = sorted(greenlists)
    green_sets = {t: frozenset(v[split]) for t, v in greenlists.items()}

    emb = model.get_input_embeddings().weight.detach().float()
    normed_embeddings = emb / emb.norm(dim=1, keepdim=True)

    return {
        "topics": topics,
        "green_sets": green_sets,
        "normed_embeddings": normed_embeddings,
        "topic_matrix": build_topic_matrix(model, tokenizer, topics),
    }


def _extract_topic(
    text,
    tokenizer,
    state,
):
    """
        re-infers the topic of `text`; mirrors first_layer.TopicWiseWatermarking.
        extract_topic so detection routes identically to generation.
        returns (best_topic, best_cosine), or (None, 0.0) for empty text.
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )
    if not token_ids:
        return None, 0.0

    vec = state["normed_embeddings"][token_ids].mean(dim=0)
    vec = vec / vec.norm()

    scores = (state["topic_matrix"] @ vec).tolist()
    ranked = sorted(zip(state["topics"], scores), key=lambda x: -x[1])

    return ranked[0][0], ranked[0][1]


def detect_topic_watermark(
    text,
    tokenizer,
    state,
    vocab_size,
    z_threshold=Z_THRESHOLD,
    p_value_threshold=P_VALUE_THRESHOLD,
):
    """
        returns dictionary containing: topic, topic_score, ownership_score,
        match_count, num_positions, gamma, z_score, p_value, confirmed
    """

    topic, topic_score = _extract_topic(text, tokenizer, state)

    if topic is None:
        return {"topic": None, "topic_score": None, "ownership_score": 0.0,
                "match_count": 0, "num_positions": 0, "gamma": 0.0,
                "z_score": 0.0, "p_value": 1.0, "confirmed": False}

    green_set = state["green_sets"][topic]
    gamma = len(green_set) / vocab_size

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    matches = sum(1 for tid in token_ids if tid in green_set)
    total_positions = len(token_ids)

    ownership_score = (
        matches / total_positions
        if total_positions > 0
        else 0.0
    )

    if total_positions > 0 and gamma > 0:

        z_score = (
            (matches - total_positions * gamma)
            / math.sqrt(total_positions * gamma * (1 - gamma))
        )

        p_value = binomtest(
            k=matches,
            n=total_positions,
            p=gamma,
            alternative="greater",
        ).pvalue

    else:
        z_score = 0.0
        p_value = 1.0

    confirmed = (
        z_score >= z_threshold
        and p_value < p_value_threshold
    )

    return {
        "topic": topic,
        "topic_score": round(topic_score, 4),
        "ownership_score": round(ownership_score, 4),
        "match_count": matches,
        "num_positions": total_positions,
        "gamma": round(gamma, 5),
        "z_score": round(z_score, 4),
        "p_value": p_value,
        "confirmed": bool(confirmed),
    }


def detect_dataframe(
    df,
    tokenizer,
    state,
    text_column="text",
    vocab_size=None,
    z_threshold=Z_THRESHOLD,
    p_value_threshold=P_VALUE_THRESHOLD,
):
    """
        this function will return the df containing the original data plus detection results.
    """

    if vocab_size is None:
        vocab_size = len(tokenizer)

    detection_results = []

    for _, row in df.iterrows():

        result = detect_topic_watermark(
            text=row[text_column],
            tokenizer=tokenizer,
            state=state,
            vocab_size=vocab_size,
            z_threshold=z_threshold,
            p_value_threshold=p_value_threshold,
        )

        detection_results.append(result)

    detection_df = pd.DataFrame(detection_results)

    return pd.concat(
        [df.reset_index(drop=True),
         detection_df.reset_index(drop=True)],
        axis=1,
    )
