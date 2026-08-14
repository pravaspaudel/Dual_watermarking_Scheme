import pandas as pd
from scipy.stats import binomtest

GREEN_FRACTION = 0.5
PREV_TOKEN_SIZE = 5
DETECTION_THRESHOLD = 0.60
P_VALUE_THRESHOLD = 0.05

from src.utils.key_manager import derive_set

def detect_private_watermark(
    text,
    tokenizer,
    key,
    vocab_size,
    green_fraction,
    prev_token_size,
    threshold,
    p_value_threshold,
):
    """
        returns dictionary containing: ownership_score,match_count,num_positions,p_value,confirmed
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    matches = 0
    total_positions = 0

    for position in range(prev_token_size, len(token_ids)):

        history = token_ids[:position]

        preferred_set = derive_set(
            vocab_size=vocab_size,
            green_fraction=green_fraction,
            key=key,
            prev_tokens_size=prev_token_size,
            prev_tokens=history,
        )

        if token_ids[position] in preferred_set:
            matches += 1

        total_positions += 1

    ownership_score = (
        matches / total_positions
        if total_positions > 0
        else 0.0
    )

    p_value = binomtest(
        k=matches,
        n=total_positions,
        p=green_fraction,
        alternative="greater",
    ).pvalue if total_positions > 0 else 1.0

    confirmed = (
        ownership_score > threshold
        and p_value < p_value_threshold
    )

    return {
        "ownership_score": ownership_score,
        "match_count": matches,
        "num_positions": total_positions,
        "p_value": p_value,
        "confirmed": confirmed,
    }


def detect_dataframe(
    df,
    tokenizer,
    key,
    text_column="wm_output",
    vocab_size=None,
    green_fraction=GREEN_FRACTION,
    prev_token_size=PREV_TOKEN_SIZE,
    threshold=DETECTION_THRESHOLD,
    p_value_threshold=P_VALUE_THRESHOLD,
):
    """
        this function will return the df containing the original data plus detection results.
    """

    if vocab_size is None:
        vocab_size = len(tokenizer)

    detection_results = []

    for _, row in df.iterrows():

        result = detect_private_watermark(
            text=row[text_column],
            tokenizer=tokenizer,
            key=key,
            vocab_size=vocab_size,
            green_fraction=green_fraction,
            prev_token_size=prev_token_size,
            threshold=threshold,
            p_value_threshold=p_value_threshold,
        )

        detection_results.append(result)

    detection_df = pd.DataFrame(detection_results)

    return pd.concat(
        [df.reset_index(drop=True),
         detection_df.reset_index(drop=True)],
        axis=1,
    )