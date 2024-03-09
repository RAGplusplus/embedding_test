import os
import json

def save_json(content: dict, filename: str = "out", out_dir: str = "results"):
    """Save a Python dictionary to a json file"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{filename}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=4)