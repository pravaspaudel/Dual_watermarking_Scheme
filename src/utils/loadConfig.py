import json
from pathlib import Path

def load_config(config_name: str):
    """
    returns the dictionary of config file of the layer 
    """

    if not config_name.endswith(".json"):
        config_name += ".json"

    config_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / config_name
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        return json.load(f)