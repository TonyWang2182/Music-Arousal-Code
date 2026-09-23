"""Train-only preprocessing, frozen validation tuning and artist bagging.

``train_all`` consumes the audited manifest and raw (unstandardized) features.
It writes portable JSON + NumPy models and predictions without test labels.
The same preprocessing fitted on the complete training partition is used by
the full-training single model and every bootstrap member. This deliberately
keeps the representation coordinates fixed when studying model disagreement.
"""
from __future__ import annotations

import os
import platform
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import code_hash, file_hash, public_config, read_json, stable_hash, write_json


SCHEMA_VERSION = 2


class TrainingFailure(RuntimeError):
    """An experiment cannot be presented as complete."""


def artist_bootstrap_indices(groups, seed: int, member: int) -> np.ndarray:
    """Sample G groups with replacement and retain every song in each draw.

    The RNG depends on the run seed and member index, never on representation,
    head, projection dimension or the eventual ensemble size. Thus K=3 is an
    exact prefix of K=10, and all model configurations share the same draws.
    ``groups`` should be duplicate-connected artist groups from data audit.
    """
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Artist bootstrap needs at least two training groups")
    if seed < 0 or member < 0:
        raise ValueError("Seeds and member indices must be nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(member), 7103]))
    draws = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in draws])


def _member_seed(seed: int, member: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(member), 8317]).generate_state(1)[0])


def _raw_matrix(features: Mapping[str, np.ndarray] | np.ndarray,
                representation: str, layer: int | None) -> np.ndarray:
    if isinstance(features, np.ndarray):
        x = np.asarray(features, dtype=np.float64)
    elif representation == "mir":
        x = np.asarray(features["mir"], dtype=np.float64)
    elif representation == "mert":
        x = np.asarray(features[f"mert_{layer}"], dtype=np.float64)
    elif representation == "fusion":
        x = np.concatenate([features["mir"], features[f"mert_{layer}"]], axis=1)
        x = np.asarray(x, dtype=np.float64)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite two-dimensional matrix")
    return x


def _fit_preprocessor(x: np.ndarray, train_indices: np.ndarray,
                      projection_dim: int | None):
    """Fit only on train rows; project every row with the frozen transform."""
    scaler = StandardScaler().fit(x[train_indices])
    arrays = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    transformed = scaler.transform(x)
    if projection_dim is not None:
        limit = min(len(train_indices), x.shape[1])
        if projection_dim < 1 or projection_dim > limit:
            raise ValueError(f"PCA{projection_dim} requires at least that many training "
                             f"rows and features; available limit is {limit}")
        pca = PCA(n_components=int(projection_dim), whiten=False, svd_solver="full")
        pca.fit(transformed[train_indices])
        transformed = pca.transform(transformed)
        arrays.update(pca_mean=pca.mean_, pca_components=pca.components_)
    return arrays, transformed


def _transform(x: np.ndarray, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if x.shape[1] != len(arrays["scaler_mean"]):
        raise ValueError("Saved model and input feature dimensions disagree")
    x = (x - arrays["scaler_mean"]) / arrays["scaler_scale"]
    if "pca_components" in arrays:
        x = (x - arrays["pca_mean"]) @ arrays["pca_components"].T
    return x


def _fit_regressor(x, y, head: str, alpha: float, seed: int, cfg: dict):
    if head == "ridge":
        model = Ridge(alpha=float(alpha), solver="svd", fit_intercept=True)
    elif head == "mlp":
        train_cfg = cfg.get("training", {})
        model = MLPRegressor(
            hidden_layer_sizes=(int(train_cfg.get("hidden_units", 64)),),
            activation="tanh", solver="lbfgs", alpha=float(alpha),
            max_iter=int(train_cfg.get("max_iter", 3000)),
            max_fun=int(train_cfg.get("max_fun", 20000)),
            tol=float(train_cfg.get("tol", 1e-6)), random_state=int(seed),
            early_stopping=False,
        )
    else:
        raise ValueError(f"Unknown regressor: {head}")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(x, y)
    log = {
        "seed": int(seed), "alpha": float(alpha),
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in captured),
        "warnings": [{"category": w.category.__name__, "message": str(w.message)}
                     for w in captured],
    }
    if getattr(model, "n_iter_", None) is not None:
        log["n_iter"] = int(model.n_iter_)
    if hasattr(model, "loss_"):
        log["loss"] = float(model.loss_)
    return model, log


def _export_regressor(model, head: str, prefix: str) -> dict:
    if head == "ridge":
        return {f"{prefix}_coef": np.asarray(model.coef_, dtype=np.float64),
                f"{prefix}_intercept": np.asarray(model.intercept_, dtype=np.float64)}
    return {f"{prefix}_w0": np.asarray(model.coefs_[0], dtype=np.float64),
            f"{prefix}_b0": np.asarray(model.intercepts_[0], dtype=np.float64),
            f"{prefix}_w1": np.asarray(model.coefs_[1], dtype=np.float64),
            f"{prefix}_b1": np.asarray(model.intercepts_[1], dtype=np.float64)}


def _predict_regressor(x, arrays, head: str, prefix: str):
    if head == "ridge":
        pred = x @ arrays[f"{prefix}_coef"] + arrays[f"{prefix}_intercept"]
    else:
        hidden = np.tanh(x @ arrays[f"{prefix}_w0"] + arrays[f"{prefix}_b0"])
        pred = (hidden @ arrays[f"{prefix}_w1"] + arrays[f"{prefix}_b1"]).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if not np.isfinite(pred).all():
        raise TrainingFailure("Regressor produced nonfinite predictions")
    return np.clip(pred, 0.0, 1.0)


def _save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def predict_saved(model_path: Path, features: Mapping[str, np.ndarray] | np.ndarray):
    """Return ``(member_predictions[N,K], single_predictions[N])``.

    ``features`` is an aligned mapping containing ``mir`` and/or ``mert_L`` raw
    matrices, or the already-concatenated raw matrix for this representation.
    No scaler is refitted. Files contain only numeric arrays (no pickle).
    """
    model_path = Path(model_path)
    metadata = read_json(model_path)
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("status") != "complete":
        raise ValueError("Unsupported or incomplete saved model")
    weights_path = (model_path.parent / metadata["weights_file"]).resolve()
    if not weights_path.is_relative_to(model_path.parent.resolve()):
        raise ValueError("Saved weights path escapes the model directory")
    if file_hash(weights_path) != metadata["weights_sha256"]:
        raise ValueError("Saved model weights failed their content fingerprint")
    with np.load(weights_path, allow_pickle=False) as weights:
        arrays = dict(weights)
    x = _raw_matrix(features, metadata["representation"], metadata["layer"])
    x = _transform(x, arrays)
    single = _predict_regressor(x, arrays, metadata["head"], "single")
    members = np.column_stack([
        _predict_regressor(x, arrays, metadata["head"], f"member_{k:03d}")
        for k in range(metadata["members"])
    ])
    return members, single


def _cached(path: Path, fingerprint: str):
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("fingerprint") != fingerprint:
        raise TrainingFailure(f"Stale artifact at {path}. Start a new output directory "
                              "or rerun without --resume; caches cannot cross protocols.")
    return value


def _tune(directory, signature, features, train, val, labels, representation,
          head, projection_dim, layers, cfg):
    path = directory / "tuning.json"
    fingerprint = stable_hash(signature)
    if cfg.get("_resume", False):
        cached = _cached(path, fingerprint)
        if cached is not None and cached.get("status") == "complete":
            return cached
    candidates = []
    alphas = cfg.get("ridge_alphas", [0.1, 1.0, 10.0]) if head == "ridge" else cfg.get(
        "mlp_alphas", [0.001, 0.1, 10.0])
    for layer in layers:
        x = _raw_matrix(features, representation, layer)
        _, projected = _fit_preprocessor(x, train, projection_dim)
        for alpha in alphas:
            model, log = _fit_regressor(projected[train], labels[train], head,
                                        alpha, 0, cfg)
            pred = np.clip(model.predict(projected[val]).ravel(), 0.0, 1.0)
            mae = float(np.mean(np.abs(pred - labels[val])))
            if not np.isfinite(mae):
                raise TrainingFailure("Nonfinite validation MAE")
            candidates.append({"layer": layer, "alpha": float(alpha),
                               "validation_mae": mae, **log})
    if not candidates:
        raise ValueError("Tuning requires at least one layer and regularization candidate")
    # Iteration order is the deterministic tie-break; no test values are read.
    selected = min(candidates, key=lambda item: item["validation_mae"])
    result = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
              "status": "complete" if selected["converged"] else "failed",
              "selection_rule": "full-training single model, seed=0, clipped validation MAE",
              "candidates": candidates, "selected": selected,
              "projection_dim": projection_dim}
    write_json(path, result)
    if not selected["converged"]:
        raise TrainingFailure(f"Selected tuning model did not converge; see {path}")
    return result


def _train_run(out_dir, directory, signature, song_ids, features, train, labels,
               groups, split_seed, representation, head, variant, seed, layer,
               alpha, projection_dim, count, main_members, cfg):
    directory = directory / f"seed_{seed}"
    metadata_path = directory / "model.json"
    prediction_path = directory / "predictions.npz"
    status_path = directory / "status.json"
    fingerprint = stable_hash({**signature, "model_seed": seed, "layer": layer,
                               "alpha": alpha, "members": count})
    record = {
        "split_seed": int(split_seed), "representation": representation,
        "head": head, "variant": variant, "seed": int(seed), "layer": layer,
        "alpha": float(alpha), "projection_dim": projection_dim, "members": int(count),
        "main_members": int(main_members),
        "model_path": metadata_path.relative_to(out_dir).as_posix(),
        "predictions_path": prediction_path.relative_to(out_dir).as_posix(),
    }
    if cfg.get("_resume", False):
        cached = _cached(status_path, fingerprint)
        if cached is not None and cached.get("status") == "complete":
            for name, digest in cached["files"].items():
                path = directory / name
                if not path.exists() or file_hash(path) != digest:
                    raise TrainingFailure(f"Corrupt cached model artifact: {path}")
            return record
    write_json(status_path, {"status": "running", "fingerprint": fingerprint})
    try:
        x = _raw_matrix(features, representation, layer)
        arrays, projected = _fit_preprocessor(x, train, projection_dim)
        fit_log = []
        single, log = _fit_regressor(projected[train], labels[train], head, alpha, seed, cfg)
        fit_log.append({"role": "single_full_training", **log})
        write_json(directory / "convergence.json", fit_log)
        if not log["converged"]:
            raise TrainingFailure("Final full-training single model did not converge")
        arrays.update(_export_regressor(single, head, "single"))
        bootstrap_hashes = []
        for member in range(count):
            local_indices = artist_bootstrap_indices(groups[train], seed, member)
            sampled = train[local_indices]
            bootstrap_hashes.append(stable_hash(song_ids[sampled].tolist()))
            model, log = _fit_regressor(projected[sampled], labels[sampled], head,
                                        alpha, _member_seed(seed, member), cfg)
            fit_log.append({"role": "bootstrap_member", "member": member,
                            "sampled_rows": int(len(sampled)),
                            "bootstrap_sha256": bootstrap_hashes[-1], **log})
            write_json(directory / "convergence.json", fit_log)
            if not log["converged"]:
                raise TrainingFailure(f"Final bootstrap member {member} did not converge")
            arrays.update(_export_regressor(model, head, f"member_{member:03d}"))
        weights_path = directory / "weights.npz"
        _save_npz(weights_path, **arrays)
        metadata = {"schema_version": SCHEMA_VERSION, "status": "complete",
                    "fingerprint": fingerprint, "representation": representation,
                    "head": head, "variant": variant, "layer": layer, "alpha": float(alpha),
                    "projection_dim": projection_dim, "members": int(count),
                    "main_members": int(main_members), "model_seed": int(seed),
                    "split_seed": int(split_seed), "weights_file": "weights.npz",
                    "weights_sha256": file_hash(weights_path), "preprocessing":
                    "StandardScaler and optional non-whitened PCA fitted on full training rows only",
                    "bootstrap_unit": "duplicate-connected artist group",
                    "bootstrap_sha256": bootstrap_hashes,
                    "training_song_ids": song_ids[train].tolist(),
                    "provenance": signature}
        write_json(metadata_path, metadata)
        # Reload the portable model to generate canonical predictions. Training
        # and evaluation therefore exercise the same numerical inference path.
        members, single_pred = predict_saved(metadata_path, features)
        reference_single = np.clip(single.predict(projected).ravel(), 0.0, 1.0)
        if not np.allclose(reference_single, single_pred, rtol=1e-10, atol=1e-10):
            raise TrainingFailure("Portable model reload changed predictions")
        _save_npz(prediction_path, song_ids=song_ids, members=members, single=single_pred,
                  mean=members[:, :main_members].mean(axis=1))
        files = {name: file_hash(directory / name) for name in
                 ["model.json", "weights.npz", "predictions.npz", "convergence.json"]}
        write_json(status_path, {"status": "complete", "fingerprint": fingerprint,
                                 "files": files})
        return record
    except Exception as error:
        write_json(status_path, {"status": "failed", "fingerprint": fingerprint,
                                 "error": f"{type(error).__name__}: {error}"})
        raise


def _load_inputs(out_dir, cfg):
    manifest = read_json(out_dir / "data_manifest.json")
    if not cfg.get("smoke_only", False):
        audit = read_json(out_dir / "data_audit.json")
        if (manifest.get("status") != "complete" or audit.get("status") != "complete"
                or not manifest.get("data_content_hash")
                or audit.get("data_content_hash") != manifest["data_content_hash"]):
            raise TrainingFailure("Current data audit is incomplete or does not match the manifest")
    rows = manifest["rows"] if isinstance(manifest, dict) else manifest
    song_ids = np.asarray([str(row["song_id"]) for row in rows])
    if len(set(song_ids)) != len(song_ids):
        raise ValueError("Data manifest contains duplicate song IDs")
    labels = np.asarray([row["arousal"] for row in rows], dtype=np.float64)
    if not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
        raise ValueError("Manifest arousal must already be normalized to [0,1]")
    groups = np.asarray([str(row.get("group_id", row["artist_group"])) for row in rows])
    npz_path = out_dir / "features" / "raw_features.npz"
    feature_manifest_path = out_dir / "features" / "feature_manifest.json"
    if not feature_manifest_path.exists():
        raise ValueError("Missing feature_manifest.json; run feature extraction first")
    feature_manifest = read_json(feature_manifest_path)
    if feature_manifest.get("status") != "complete":
        raise TrainingFailure("Feature extraction is not complete")
    npz_hash = file_hash(npz_path)
    if feature_manifest.get("npz_sha256") != npz_hash:
        raise TrainingFailure("Raw feature content differs from its extraction manifest")
    if not cfg.get("smoke_only", False) and feature_manifest.get("data_content_hash") != manifest["data_content_hash"]:
        raise TrainingFailure("Features belong to a different audited dataset")
    with np.load(npz_path, allow_pickle=False) as raw:
        raw_ids = [str(item) for item in raw["song_ids"]]
        if len(set(raw_ids)) != len(raw_ids) or set(raw_ids) != set(song_ids):
            raise ValueError("Raw feature IDs do not match the audited manifest")
        mapping = {song_id: i for i, song_id in enumerate(raw_ids)}
        permutation = np.asarray([mapping[song_id] for song_id in song_ids])
        keys = ["mir"] + [f"mert_{layer}" for layer in cfg.get("layers", [4, 8, 12])]
        features = {}
        for key in keys:
            if key not in raw:
                raise ValueError(f"Missing raw feature matrix: {key}")
            matrix = np.asarray(raw[key], dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid feature matrix: {key}")
            if not cfg.get("smoke_only", False):
                expected = 32 if key == "mir" else 768
                if matrix.shape[1] != expected:
                    raise ValueError(f"Full protocol requires {key} to have {expected} columns")
            features[key] = matrix[permutation]
    return song_ids, labels, groups, features, {
        "raw_features_sha256": npz_hash,
        "feature_manifest_sha256": stable_hash(feature_manifest),
    }


def _split_indices(out_dir, seed, song_ids, groups):
    path = out_dir / "splits" / f"split_{seed}.json"
    split = read_json(path)
    assignments = {str(k): v for k, v in split["assignments"].items()}
    if set(assignments) != set(song_ids):
        raise ValueError("Split assignments must contain exactly the audited song IDs")
    names = np.asarray([assignments[song_id] for song_id in song_ids], dtype=str)
    names[names == "validation"] = "val"
    if not set(names).issubset({"train", "val", "test"}):
        raise ValueError("Split names must be train, val, or test")
    for group in np.unique(groups):
        if len(set(names[groups == group])) != 1:
            raise ValueError(f"Artist/duplicate group {group!r} crosses partitions")
    indices = {name: np.flatnonzero(names == name) for name in ["train", "val", "test"]}
    if any(len(value) == 0 for value in indices.values()):
        raise ValueError("Train, validation, and test partitions must all be nonempty")
    return indices, file_hash(path)


def train_all(out_dir: Path, cfg: dict) -> list[dict]:
    """Train the frozen experiment and publish an index only on full success.

    With ``cfg['_resume']`` true, complete per-run artifacts are reused only if
    feature, split, train/validation-label, configuration and code fingerprints
    match. A stale fingerprint fails explicitly. Without resume, artifacts in
    the requested configuration are regenerated; old configurations are never
    included in the final index. Test labels never enter tuning or model fits;
    whole-dataset lineage remains fingerprinted through the feature manifest.
    The ``mean`` prediction always uses the first ``members``
    configured members, including primary-split files containing K=10.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "training_status.json"
    index_path = out_dir / "model_index.json"
    # An index from a previous run cannot advertise success during a new or
    # failed run. Removing this generated index leaves recoverable models intact.
    if index_path.exists():
        index_path.unlink()
    config_hash = stable_hash(public_config(cfg))
    provenance = {"configuration_sha256": config_hash, "code_sha256": code_hash(),
                  "python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    write_json(status_path, {"status": "running", **provenance})
    records = []
    try:
        song_ids, labels, groups, features, feature_provenance = _load_inputs(out_dir, cfg)
        members = int(cfg.get("members", 3))
        large_members = int(cfg.get("large_members", 10))
        if members < 1 or large_members < members:
            raise ValueError("Need 1 <= members <= large_members")
        reps = [str(name).lower() for name in cfg.get("representations", ["mir", "mert", "fusion"])]
        heads = [str(name).lower() for name in cfg.get("heads", ["ridge", "mlp"])]
        split_seeds = [int(seed) for seed in cfg.get("split_seeds", [42, 43, 44])]
        model_seeds = [int(seed) for seed in cfg.get("model_seeds", [0, 1, 2])]
        primary = int(cfg.get("primary_split", 42))
        if not reps or not heads or not split_seeds or not model_seeds:
            raise ValueError("The experiment grid cannot be empty")
        if len(set(reps)) != len(reps) or len(set(heads)) != len(heads):
            raise ValueError("Repeated representations or heads in experiment grid")
        if len(set(split_seeds)) != len(split_seeds) or len(set(model_seeds)) != len(model_seeds):
            raise ValueError("Repeated seeds in experiment grid")
        with threadpool_limits(limits=int(cfg.get("threads", 1))):
            for split_seed in split_seeds:
                partitions, split_hash = _split_indices(out_dir, split_seed, song_ids, groups)
                train, val = partitions["train"], partitions["val"]
                train_val = np.concatenate([train, val])
                split_provenance = {
                    **provenance, **feature_provenance, "split_sha256": split_hash,
                    "train_validation_labels_sha256": stable_hash([
                        [str(song_ids[i]), float(labels[i])] for i in train_val]),
                    "training_groups_sha256": stable_hash(groups[train].tolist()),
                }
                for representation in reps:
                    for head in heads:
                        directory = out_dir / "models" / f"split_{split_seed}" / representation / head
                        layers = [None] if representation == "mir" else [int(layer) for layer in cfg.get("layers", [4, 8, 12])]
                        projection_dim = int(cfg.get("projection_dim", 32))
                        main_signature = {**split_provenance, "representation": representation,
                                          "head": head, "variant": "main", "projection_dim": projection_dim,
                                          "candidate_layers": layers}
                        tuning = _tune(directory / "main", main_signature, features, train,
                                       val, labels, representation, head, projection_dim, layers, cfg)
                        chosen_layer = tuning["selected"]["layer"]
                        variants = [("main", projection_dim, tuning, main_signature)]
                        if cfg.get("ablations", True) and split_seed == primary and representation != "mir":
                            variant = "pca128" if head == "mlp" else "no_pca"
                            dim = 128 if head == "mlp" else None
                            signature = {**split_provenance, "representation": representation,
                                         "head": head, "variant": variant, "projection_dim": dim,
                                         "candidate_layers": [chosen_layer]}
                            ablation_tuning = _tune(directory / variant, signature, features,
                                                   train, val, labels, representation, head,
                                                   dim, [chosen_layer], cfg)
                            variants.append((variant, dim, ablation_tuning, signature))
                        for variant, dim, selected, signature in variants:
                            count = large_members if (cfg.get("ablations", True)
                                                      and split_seed == primary and variant == "main") else members
                            for seed in model_seeds:
                                record = _train_run(
                                    out_dir, directory / variant, signature, song_ids, features,
                                    train, labels, groups, split_seed, representation, head, variant,
                                    seed, selected["selected"]["layer"], selected["selected"]["alpha"],
                                    dim, count, members, cfg,
                                )
                                records.append(record)
                                print(f"train split={split_seed} {representation}/{head}/{variant} "
                                      f"seed={seed} K={count}: complete", flush=True)
        write_json(index_path, records)
        write_json(status_path, {"status": "complete", "model_records": len(records),
                                 "model_index_sha256": file_hash(index_path), **provenance})
        return records
    except Exception as error:
        write_json(status_path, {"status": "failed", "completed_records": len(records),
                                 "error": f"{type(error).__name__}: {error}", **provenance})
        raise
