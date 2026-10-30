import re
import json
import yaml
import numpy as np

def extract_seed_from_roll(roll_number: str) -> int:
    digits = re.findall(r'\d', str(roll_number))
    if not digits:
        raise ValueError("No numeric characters found in roll number.")
    return int("".join(digits))

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config.get("seed") is None:
        config["seed"] = extract_seed_from_roll(config["roll_number"])
    return config

def save_json(data, file_path):
    def default_converter(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, default=default_converter)