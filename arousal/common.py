"""Small, deterministic artifact and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_hash(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "arousal").glob("*.py")) + [root / "run.py"]
    return stable_hash({p.relative_to(root).as_posix(): file_hash(p) for p in paths if p.exists()})


def public_config(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def checked_child(root, relative):
    root = Path(root).resolve()
    child = (root / relative).resolve()
    if not child.is_relative_to(root):
        raise ValueError(f"Artifact path escapes output directory: {relative}")
    return child
