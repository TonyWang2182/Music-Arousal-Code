"""Split-independent raw feature extraction with verified per-song resumption."""
from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

import numpy as np

from .common import code_hash, file_hash, read_json, stable_hash, write_json
from .encoders import MERTEncoder, MERT_MODEL, MERT_SHA, handcrafted, load_audio


def _versions() -> dict:
    result = {}
    for name in ("numpy", "scipy", "soundfile", "torch", "transformers", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _valid_features(arrays, n: int, layers: list[int], include_mert: bool) -> bool:
    if arrays["mir"].shape != (n, 32) or not np.isfinite(arrays["mir"]).all():
        return False
    return not include_mert or all(arrays[f"mert_{layer}"].shape == (n, 768) and
                                  np.isfinite(arrays[f"mert_{layer}"]).all() for layer in layers)


def extract_features(data_root: Path, out_dir: Path, cfg: dict) -> dict:
    """Write features/raw_features.npz and features/feature_manifest.json.

    Raw features are independent of split and label values. Per-song caches key
    audio bytes and extractor provenance, while the assembled manifest binds to
    the current data manifest, preventing stale labels/order entering training.
    No fake encoder fallback exists. Missing MERT dependencies raise an error.
    """
    data_root, out_dir = Path(data_root).resolve(), Path(out_dir)
    manifest = read_json(out_dir / "data_manifest.json")
    audit = read_json(out_dir / "data_audit.json")
    if manifest.get("status") != "complete" or audit.get("status") != "complete" or audit.get("data_content_hash") != manifest["data_content_hash"]:
        raise ValueError("A current, complete real-data audit is required before extracting features")
    rows = manifest["rows"]
    song_ids = np.asarray([row["song_id"] for row in rows], dtype=np.int64)
    if len(set(song_ids.tolist())) != len(song_ids):
        raise ValueError("Repeated song IDs in data manifest")
    for row in rows:
        path = (data_root / row["audio"]).resolve()
        if not path.is_relative_to(data_root) or file_hash(path) != row["audio_sha256"]:
            raise ValueError(f"Audio changed since audit or escapes data_root: {row['song_id']}")
    layers = sorted({int(layer) for layer in cfg.get("layers", [4, 8, 12])})
    if any(layer not in (4, 8, 12) for layer in layers):
        raise ValueError("This frozen protocol supports only MERT layers 4, 8 and 12")
    include_mert = any(rep in {"mert", "fusion"} for rep in cfg.get("representations", ["mir", "mert", "fusion"]))
    revision = cfg.get("mert_revision", MERT_SHA)
    if include_mert and revision != MERT_SHA:
        raise ValueError("Checkpoint differs from the frozen MERT revision; start a separately documented protocol")
    extraction = {"version": 2, "sample_rate": 24000, "gain": "original; checkpoint processor retained",
                  "mert_model": MERT_MODEL if include_mert else None, "mert_revision": revision if include_mert else None,
                  "mert_layers": layers if include_mert else [], "chunk_seconds": 5.0, "device": cfg.get("device", "cpu"),
                  "pooling": "actual_output_frame_weighted", "mir_dim": 32,
                  "implementation": {name: file_hash(Path(__file__).with_name(name)) for name in ("features.py", "encoders.py")},
                  "dependencies": _versions()}
    config_hash = stable_hash(extraction)
    identity = {"data_content_hash": manifest["data_content_hash"], "config_hash": config_hash}
    feature_hash = stable_hash(identity)
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    target, metadata_path = feature_dir / "raw_features.npz", feature_dir / "feature_manifest.json"
    if cfg.get("_resume") and target.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("feature_hash") == feature_hash and metadata.get("status") == "complete" and metadata.get("npz_sha256") == file_hash(target):
            with np.load(target, allow_pickle=False) as cache:
                if np.array_equal(cache["song_ids"], song_ids) and _valid_features(cache, len(rows), layers, include_mert):
                    return metadata
    write_json(metadata_path, dict(identity, feature_hash=feature_hash, status="running", extraction=extraction))
    encoder = None
    arrays = {"mir": []}
    if include_mert:
        arrays.update({f"mert_{layer}": [] for layer in layers})
    computed, reused, start_time = 0, 0, time.monotonic()
    try:
        for position, row in enumerate(rows):
            item_key = stable_hash({"audio_sha256": row["audio_sha256"], "pcm_sha256": row["pcm_sha256"], "config_hash": config_hash})
            item_path = feature_dir / "cache" / f"{row['song_id']}_{item_key}.npz"
            item_meta = item_path.with_suffix(".json")
            values = None
            if cfg.get("_resume") and item_path.exists() and item_meta.exists():
                saved = read_json(item_meta)
                if saved.get("key") == item_key and saved.get("sha256") == file_hash(item_path):
                    with np.load(item_path, allow_pickle=False) as cache:
                        if _valid_features(cache, 1, layers, include_mert):
                            values = {key: cache[key][0] for key in arrays}
            if values is None:
                wave = load_audio(data_root / row["audio"], sample_rate=24000)
                values = {"mir": handcrafted(wave, 24000)}
                if include_mert:
                    if encoder is None:
                        encoder = MERTEncoder(revision=revision, device=cfg.get("device", "cpu"), chunk_seconds=5.0)
                    all_layers = encoder.encode(wave, 24000)
                    values.update({f"mert_{layer}": all_layers[layer] for layer in layers})
                _save_npz(item_path, **{key: value[None, :] for key, value in values.items()})
                write_json(item_meta, {"key": item_key, "sha256": file_hash(item_path), "config_hash": config_hash,
                                       "encoder": encoder.metadata if encoder is not None else None})
                computed += 1
            else:
                reused += 1
            for key in arrays:
                arrays[key].append(values[key])
            if position == 0 or (position + 1) % 25 == 0 or position + 1 == len(rows):
                print(f"Features {position + 1}/{len(rows)}; computed={computed}, reused={reused}", flush=True)
        arrays = {key: np.stack(value) for key, value in arrays.items()}
        if not _valid_features(arrays, len(rows), layers, include_mert):
            raise ValueError("Incomplete/nonfinite feature arrays")
        _save_npz(target, song_ids=song_ids, **arrays)
        # With all per-song caches reused, recover the actual checkpoint processor
        # metadata rather than claiming a new model load occurred.
        if encoder is not None:
            encoder_metadata = encoder.metadata
        elif include_mert:
            encoder_metadata = read_json(item_meta).get("encoder")
        else:
            encoder_metadata = None
        metadata = dict(identity, feature_hash=feature_hash, status="complete", extraction=extraction,
                        code_hash=code_hash(), npz_sha256=file_hash(target), encoder=encoder_metadata,
                        n_clips=len(rows), dimensions={key: values.shape[1] for key, values in arrays.items()},
                        newly_computed=computed, reused=reused, elapsed_seconds=time.monotonic() - start_time,
                        smoke_only=bool(manifest.get("smoke_only")), normalization_fitted=False)
        write_json(metadata_path, metadata)
        return metadata
    except Exception as exc:
        write_json(metadata_path, dict(identity, feature_hash=feature_hash, status="failed", extraction=extraction,
                                      error=f"{type(exc).__name__}: {exc}"))
        raise
