import os, json, time, pathlib
from typing import Any, Dict

def run_dir(name: str) -> str:
    d = os.path.join("runs", name)
    os.makedirs(d, exist_ok=True)
    return d

def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")
