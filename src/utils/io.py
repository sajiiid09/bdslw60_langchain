import os
import json
import yaml
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def path_with_root(root: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(root, rel))
