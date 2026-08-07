import os
import hmac
import hashlib
import numpy as np

# from dotenv import load_dotenv
def generate_key(size:int = 32): 
    """
    generate key and store in .env
    """
    random_val = os.urandom(32)
    key = random_val.hex()

    with open(".env","w") as f:
        f.write(f"WATERMARK_SECRET_KEY={key}")

    return key

def derive_seed(secret_key: bytes, previous_token: str) -> int:
    """
    Derive a deterministic random seed from a secret key
    and the previous token.

    Args:
        secret_key: bytes loaded from .env
        previous_token: previous generated token

    """
    digest = hmac.new(
        secret_key,
        previous_token.encode("utf-8"),
        hashlib.sha256
    ).digest()

    # Use the first 8 bytes as a 64-bit seed
    return int.from_bytes(digest[:8], byteorder="big")


def derive_set(vocab_size,key,prev_tokens,green_fraction=0.5,prev_tokens_size=1):

    seed = derive_seed(key,prev_tokens,prev_tokens_size)

    state = np.random.RandomState(seed)

    permutation = state.permutation(vocab_size)

    green_size = int(green_fraction* vocab_size)

    return set(permutation[:green_size])