"""Evaluation must reject failed, stale, mixed-mode and corrupted artifacts."""
import shutil

import pytest

from arousal.common import file_hash, read_json, stable_hash, write_json
from arousal.evaluate import load_cases
from arousal.model import train_all
from arousal.smoke import prepare_smoke


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("completed_integrity_smoke")
    cfg = {"smoke_only": True, "split_seeds": [42], "primary_split": 42,
           "model_seeds": [0], "members": 3, "large_members": 10, "layers": [4],
           "representations": ["mir"], "heads": ["ridge"], "projection_dim": 32,
           "ablations": False, "ridge_alphas": [0.1], "threads": 1}
    prepare_smoke(out, cfg)
    train_all(out, cfg)
    return out, cfg


@pytest.fixture
def artifacts(completed_smoke, tmp_path):
    source, cfg = completed_smoke
    out = tmp_path / "run"
    shutil.copytree(source, out)
    return out, dict(cfg)


def test_completed_smoke_is_accepted(artifacts):
    out, cfg = artifacts
    manifest, cases = load_cases(out, cfg)
    assert manifest["smoke_only"] is True
    assert len(cases) == 1 and len(cases[0]["ids"]) == 45


@pytest.mark.parametrize("filename,status", [
    ("training_status.json", "failed"),
    ("training_status.json", "running"),
    ("data_audit.json", "failed"),
    ("features/feature_manifest.json", "failed"),
])
def test_failed_stage_cannot_be_evaluated(artifacts, filename, status):
    out, cfg = artifacts
    value = read_json(out / filename)
    value["status"] = status
    write_json(out / filename, value)
    with pytest.raises(ValueError):
        load_cases(out, cfg)


@pytest.mark.parametrize("change", ["config", "code", "split", "labels", "features", "index"])
def test_stale_inputs_are_rejected(artifacts, change):
    out, cfg = artifacts
    if change == "config":
        cfg["ridge_alphas"] = [10]
    elif change == "code":
        training = read_json(out / "training_status.json")
        training["code_sha256"] = "older-source-tree"
        write_json(out / "training_status.json", training)
    elif change == "split":
        split_path = out / "splits/split_42.json"
        split = read_json(split_path)
        split["assignments"]["1"] = "test" if split["assignments"]["1"] != "test" else "train"
        write_json(split_path, split)
    elif change == "labels":
        manifest = read_json(out / "data_manifest.json")
        manifest["rows"][0]["arousal"] = 0.99
        write_json(out / "data_manifest.json", manifest)
    elif change == "features":
        with (out / "features/raw_features.npz").open("ab") as stream:
            stream.write(b"changed-feature-bytes")
    else:
        index = read_json(out / "model_index.json")
        index[0]["seed"] = 99
        write_json(out / "model_index.json", index)
    with pytest.raises(ValueError):
        load_cases(out, cfg)


@pytest.mark.parametrize("filename", ["predictions.npz", "weights.npz", "model.json", "convergence.json"])
def test_model_artifact_tampering_is_rejected(artifacts, filename):
    out, cfg = artifacts
    index = read_json(out / "model_index.json")
    path = (out / index[0]["model_path"]).parent / filename
    if filename.endswith(".json"):
        content = read_json(path)
        if isinstance(content, list):
            content[0]["converged"] = False
        else:
            content["alpha"] = 99
        write_json(path, content)
    else:
        with path.open("ab") as stream:
            stream.write(b"changed-model-bytes")
    with pytest.raises(ValueError, match="fingerprint"):
        load_cases(out, cfg)


def test_partial_model_is_rejected_even_with_complete_global_status(artifacts):
    out, cfg = artifacts
    index = read_json(out / "model_index.json")
    status_path = (out / index[0]["model_path"]).parent / "status.json"
    status = read_json(status_path)
    status["status"] = "failed"
    write_json(status_path, status)
    with pytest.raises(ValueError, match="Per-model"):
        load_cases(out, cfg)


@pytest.mark.parametrize("change", ["config", "manifest", "features", "formal_disguise"])
def test_smoke_and_formal_modes_cannot_mix(artifacts, change):
    out, cfg = artifacts
    if change == "config":
        cfg["smoke_only"] = False
    elif change in {"manifest", "features"}:
        path = out / ("data_manifest.json" if change == "manifest" else "features/feature_manifest.json")
        value = read_json(path)
        value["smoke_only"] = False
        write_json(path, value)
    else:
        cfg["smoke_only"] = False
        manifest = read_json(out / "data_manifest.json")
        manifest.update(smoke_only=False, dataset="DEAM", status="complete")
        write_json(out / "data_manifest.json", manifest)
        feature = read_json(out / "features/feature_manifest.json")
        feature["smoke_only"] = False
        write_json(out / "features/feature_manifest.json", feature)
        audit = read_json(out / "data_audit.json")
        audit["status"] = "complete"
        write_json(out / "data_audit.json", audit)
    with pytest.raises(ValueError):
        load_cases(out, cfg)


def test_changed_split_with_its_local_hash_updated_still_fails_model_provenance(artifacts):
    out, cfg = artifacts
    split_path = out / "splits/split_42.json"
    split = read_json(split_path)
    # A harmless metadata edit retains the split's own assignment fingerprint,
    # but changes the exact split artifact to which the models were bound.
    split["unexpected_revision"] = "copied split file"
    write_json(split_path, split)
    with pytest.raises(ValueError, match="provenance"):
        load_cases(out, cfg)


def test_rehashed_corrupt_predictions_cannot_change_mean_semantics(artifacts):
    import numpy as np
    out, cfg = artifacts
    record = read_json(out / "model_index.json")[0]
    path = out / record["predictions_path"]
    with np.load(path, allow_pickle=False) as values:
        arrays = dict(values)
    arrays["mean"] = np.zeros_like(arrays["mean"])
    np.savez_compressed(path, **arrays)
    status_path = path.parent / "status.json"
    status = read_json(status_path)
    status["files"]["predictions.npz"] = file_hash(path)
    write_json(status_path, status)
    with pytest.raises(ValueError, match="prefix average"):
        load_cases(out, cfg)
