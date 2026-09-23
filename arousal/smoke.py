"""Synthetic plumbing check. Never a DEAM experiment or paper result."""
from pathlib import Path
import numpy as np
from .common import file_hash, read_json, stable_hash, write_json


def prepare_smoke(out_dir, cfg):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.get("smoke_only"):
        raise ValueError("Synthetic data cannot enter a scientific run")
    path = out_dir / "data_manifest.json"
    feature_path = out_dir / "features" / "raw_features.npz"
    if cfg.get("_resume") and path.exists() and feature_path.exists():
        info = read_json(out_dir / "features" / "feature_manifest.json")
        if not read_json(path).get("smoke_only") or info["npz_sha256"] != file_hash(feature_path):
            raise ValueError("Smoke artifact mismatch")
        return read_json(path)
    rng = np.random.default_rng(20260919)
    n = 300
    latent = rng.normal(size=(n, 12))
    mir = np.hstack([latent, rng.normal(size=(n, 20))])
    mert = latent @ rng.normal(size=(12, 768)) / 3 + rng.normal(scale=0.2, size=(n, 768))
    labels = np.clip(0.5 + 0.3 * np.tanh(latent[:, 0] + 0.3 * latent[:, 1]) + rng.normal(scale=0.025, size=n), 0, 1)
    rows = [{"song_id": i + 1, "artist_group": f"synthetic_artist_{i//3:03d}",
             "group_id": f"synthetic_artist_{i//3:03d}", "arousal": float(labels[i]),
             "arousal_std": None, "audio": None, "audio_sha256": None, "pcm_sha256": None}
            for i in range(n)]
    manifest = {"schema_version": 2, "dataset": "SYNTHETIC_SMOKE_NOT_DEAM", "smoke_only": True,
                "data_content_hash": stable_hash(rows), "rows": rows}
    write_json(path, manifest)
    for seed in cfg["split_seeds"]:
        groups = np.random.default_rng(seed).permutation(sorted(set(r["group_id"] for r in rows)))
        membership = {g: ("train" if j < 70 else "val" if j < 85 else "test") for j, g in enumerate(groups)}
        assignments = {str(r["song_id"]): membership[r["group_id"]] for r in rows}
        write_json(out_dir / "splits" / f"split_{seed}.json",
                   {"split_seed": seed, "assignments": assignments, "split_hash": stable_hash(assignments),
                    "data_content_hash": manifest["data_content_hash"], "smoke_only": True})
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"song_ids": np.arange(1, n + 1), "mir": mir}
    arrays.update({f"mert_{layer}": mert + 0.01 * layer for layer in cfg["layers"]})
    np.savez_compressed(feature_path, **arrays)
    write_json(feature_path.parent / "feature_manifest.json",
               {"smoke_only": True, "status": "complete", "feature_hash": file_hash(feature_path),
                "npz_sha256": file_hash(feature_path), "data_content_hash": manifest["data_content_hash"],
                "mert_revision": "SYNTHETIC_NO_ENCODER", "layers": cfg["layers"]})
    write_json(out_dir / "data_audit.json", {"status": "synthetic_only", "n_rows": n,
               "warning": "No audio, annotation or pretrained encoder validity is established by this smoke run."})
    return manifest
