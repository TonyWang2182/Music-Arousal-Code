"""Scientific invariants: leakage boundaries, portable inference, shared draws."""
from pathlib import Path

import numpy as np
import pytest

from arousal.common import file_hash, read_json, write_json
from arousal.model import (TrainingFailure, _fit_preprocessor,
                           artist_bootstrap_indices, predict_saved, train_all)


def _experiment(out: Path, *, heads=None, representations=None, test_label_offset=0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(154)
    n = 60
    latent = rng.normal(size=n)
    mir = rng.normal(size=(n, 8))
    mert = rng.normal(size=(n, 12))
    mir[:, 0], mert[:, 0] = latent, latent
    labels = np.clip(0.5 + 0.12 * latent, 0, 1)
    labels[48:] = np.clip(labels[48:] + test_label_offset, 0, 1)
    rows = [{"song_id": str(i), "artist_group": f"artist_{i // 2}",
             "group_id": f"group_{i // 2}", "arousal": float(labels[i])} for i in range(n)]
    write_json(out / "data_manifest.json", {"rows": rows})
    write_json(out / "splits" / "split_42.json", {"assignments": {
        str(i): "train" if i < 36 else "val" if i < 48 else "test" for i in range(n)}})
    feature_dir = out / "features"
    feature_dir.mkdir(exist_ok=True)
    np.savez_compressed(feature_dir / "raw_features.npz", song_ids=np.arange(n).astype(str),
                        mir=mir, mert_4=mert)
    write_json(feature_dir / "feature_manifest.json", {
        "source": "synthetic test fixture", "status": "complete",
        "npz_sha256": file_hash(feature_dir / "raw_features.npz")})
    cfg = {"smoke_only": True, "split_seeds": [42], "primary_split": 42,
           "model_seeds": [0], "members": 3, "large_members": 5, "layers": [4],
           "representations": representations or ["mir"], "heads": heads or ["ridge"],
           "projection_dim": 4, "ablations": False, "ridge_alphas": [0.1, 1],
           "mlp_alphas": [0.1, 1], "threads": 1,
           "training": {"hidden_units": 4, "max_iter": 3000, "max_fun": 20000, "tol": 1e-6}}
    return cfg, {"mir": mir, "mert_4": mert}, labels


def test_scaler_and_pca_are_fitted_only_to_training_rows():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 7))
    train = np.arange(20)
    reference, _ = _fit_preprocessor(x, train, 4)
    x[20:] += 1e6
    arrays, transformed = _fit_preprocessor(x, train, 4)
    for key in reference:
        np.testing.assert_allclose(arrays[key], reference[key], rtol=0, atol=0)
    np.testing.assert_allclose(arrays["scaler_mean"], x[train].mean(axis=0))
    assert np.isfinite(transformed).all()


def test_group_bootstrap_keeps_groups_and_has_exact_prefix():
    groups = np.asarray(["a", "a", "b", "c", "c", "c", "d", "d"])
    small = [artist_bootstrap_indices(groups, 2, k) for k in range(3)]
    large = [artist_bootstrap_indices(groups, 2, k) for k in range(10)]
    for a, b in zip(small, large):
        np.testing.assert_array_equal(a, b)
    for draw in large:
        for group in np.unique(groups):
            original = np.flatnonzero(groups == group)
            counts = [np.count_nonzero(draw == i) for i in original]
            assert len(set(counts)) == 1


@pytest.mark.parametrize("head", ["ridge", "mlp"])
def test_portable_save_reload_and_prediction_files(head, tmp_path):
    cfg, features, _ = _experiment(tmp_path, heads=[head])
    records = train_all(tmp_path, cfg)
    assert len(records) == 1
    record = records[0]
    members, single = predict_saved(tmp_path / record["model_path"], features)
    with np.load(tmp_path / record["predictions_path"], allow_pickle=False) as predictions:
        np.testing.assert_allclose(members, predictions["members"], rtol=0, atol=0)
        np.testing.assert_allclose(single, predictions["single"], rtol=0, atol=0)
        np.testing.assert_allclose(predictions["mean"], members.mean(axis=1))
        assert set(predictions.files) == {"song_ids", "members", "single", "mean"}
    assert members.shape == (60, 3)
    assert single.shape == (60,)
    assert np.all((members >= 0) & (members <= 1))
    assert read_json(tmp_path / "training_status.json")["status"] == "complete"


def test_changing_test_labels_cannot_change_models_or_tuning(tmp_path):
    first, second = tmp_path / "original", tmp_path / "changed"
    cfg_a, _, _ = _experiment(first)
    cfg_b, _, _ = _experiment(second, test_label_offset=0.3)
    a, b = train_all(first, cfg_a)[0], train_all(second, cfg_b)[0]
    assert (a["layer"], a["alpha"]) == (b["layer"], b["alpha"])
    meta_a, meta_b = read_json(first / a["model_path"]), read_json(second / b["model_path"])
    # Input label changes in test are absent even from the training provenance.
    assert meta_a["fingerprint"] == meta_b["fingerprint"]
    with np.load(first / a["predictions_path"]) as pa, np.load(second / b["predictions_path"]) as pb:
        np.testing.assert_array_equal(pa["members"], pb["members"])
        np.testing.assert_array_equal(pa["single"], pb["single"])


def test_shared_bootstrap_across_representations_and_k_prefix(tmp_path):
    cfg, _, _ = _experiment(tmp_path, representations=["mir", "mert"])
    records = train_all(tmp_path, cfg)
    meta = [read_json(tmp_path / record["model_path"]) for record in records]
    assert meta[0]["bootstrap_sha256"] == meta[1]["bootstrap_sha256"]
    with np.load(tmp_path / records[0]["predictions_path"]) as values:
        prefix = values["members"].copy()
    # MIR has the K ablation but no projection ablation, allowing a tiny fixture.
    cfg["representations"] = ["mir"]
    cfg["ablations"] = True
    big = train_all(tmp_path, cfg)[0]
    with np.load(tmp_path / big["predictions_path"]) as values:
        np.testing.assert_array_equal(prefix, values["members"][:, :3])
        np.testing.assert_array_equal(values["mean"], prefix.mean(axis=1))
    assert big["members"] == 5 and big["main_members"] == 3


def test_resume_rejects_changed_protocol_and_corruption(tmp_path):
    cfg, _, _ = _experiment(tmp_path)
    records = train_all(tmp_path, cfg)
    cfg["_resume"] = True
    assert train_all(tmp_path, cfg) == records
    cfg["ridge_alphas"] = [2]
    with pytest.raises(TrainingFailure, match="Stale artifact"):
        train_all(tmp_path, cfg)
    assert not (tmp_path / "model_index.json").exists()
    cfg["ridge_alphas"] = [0.1, 1]
    record = train_all(tmp_path, cfg)[0]
    with (tmp_path / record["predictions_path"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(TrainingFailure, match="Corrupt cached"):
        train_all(tmp_path, cfg)
    assert read_json(tmp_path / "training_status.json")["status"] == "failed"


def test_cross_partition_artist_group_is_rejected(tmp_path):
    cfg, _, _ = _experiment(tmp_path)
    manifest = read_json(tmp_path / "data_manifest.json")
    manifest["rows"][-1]["group_id"] = manifest["rows"][0]["group_id"]
    write_json(tmp_path / "data_manifest.json", manifest)
    with pytest.raises(ValueError, match="crosses partitions"):
        train_all(tmp_path, cfg)


def test_selected_nonconvergence_prevents_success_index(tmp_path, monkeypatch):
    import arousal.model as module
    cfg, _, _ = _experiment(tmp_path)
    real_fit = module._fit_regressor

    def unconverged(*args, **kwargs):
        fitted, log = real_fit(*args, **kwargs)
        log["converged"] = False
        return fitted, log

    monkeypatch.setattr(module, "_fit_regressor", unconverged)
    with pytest.raises(TrainingFailure, match="Selected tuning model"):
        train_all(tmp_path, cfg)
    assert not (tmp_path / "model_index.json").exists()
    assert read_json(tmp_path / "training_status.json")["status"] == "failed"
