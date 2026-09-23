"""Evaluate saved predictions without retraining or reading test labels in selectors."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import numpy as np

from .common import checked_child, code_hash, file_hash, public_config, read_json, stable_hash, write_json
from .selection import evaluate_catalog, paired_artist_jackknife


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No results to write: {path}")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def prediction_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    denominator = y.var() + pred.var() + (y.mean() - pred.mean()) ** 2
    if denominator <= 1e-15:
        raise ValueError("CCC undefined for identical constant label and prediction vectors")
    ccc = 2 * np.mean((y - y.mean()) * (pred - pred.mean())) / denominator
    return {"mae": float(np.abs(y - pred).mean()), "ccc": float(ccc)}


def _summarize(rows, keys, values):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in groups.items():
        row = dict(zip(keys, key))
        row["n_seeds"] = len(group)
        for value in values:
            numeric = [r[value] for r in group if r[value] is not None]
            row[value] = float(np.mean(numeric)) if numeric else None
            row[value + "_seed_sd"] = float(np.std(numeric, ddof=1)) if len(numeric) > 1 else None
        result.append(row)
    return result


def load_cases(out_dir, cfg):
    # Evaluation is also a resumable entry point: do not trust a leftover index
    # from a failed run, or files copied from another dataset/protocol.
    from .common import code_hash, public_config, stable_hash

    out_dir = Path(out_dir)
    smoke = bool(cfg.get("smoke_only", False))
    manifest = read_json(out_dir / "data_manifest.json")
    audit = read_json(out_dir / "data_audit.json")
    feature_manifest = read_json(out_dir / "features" / "feature_manifest.json")
    if bool(manifest.get("smoke_only", False)) != smoke or bool(feature_manifest.get("smoke_only", False)) != smoke:
        raise ValueError("Smoke/real-data mode mismatch across configuration and artifacts")
    if smoke:
        if manifest.get("dataset") != "SYNTHETIC_SMOKE_NOT_DEAM" or audit.get("status") != "synthetic_only":
            raise ValueError("A smoke evaluation requires explicitly synthetic artifacts")
        dataset_hash = stable_hash(manifest["rows"])
        if audit.get("n_rows") != len(manifest["rows"]):
            raise ValueError("Synthetic audit row count differs from the manifest")
    else:
        if (manifest.get("dataset") != "DEAM" or manifest.get("status") != "complete"
                or audit.get("status") != "complete" or len(manifest["rows"]) != 1744):
            raise ValueError("Formal evaluation requires a complete, audited 1,744-clip DEAM manifest")
        identity_keys = ("protocol", "normalization", "rows", "source_checksums", "artist_aliases")
        if any(key not in manifest for key in identity_keys):
            raise ValueError("DEAM manifest lacks its audited content identity")
        dataset_hash = stable_hash({key: manifest[key] for key in identity_keys})
        if audit.get("data_content_hash") != dataset_hash or audit.get("n_clips") != 1744:
            raise ValueError("Current data audit is stale or incomplete")
    if manifest.get("data_content_hash") != dataset_hash:
        raise ValueError("Data manifest content differs from its audit fingerprint")
    if feature_manifest.get("status") != "complete" or feature_manifest.get("data_content_hash") != dataset_hash:
        raise ValueError("Feature extraction is incomplete or belongs to another dataset")
    raw_hash = file_hash(out_dir / "features" / "raw_features.npz")
    if feature_manifest.get("npz_sha256") != raw_hash:
        raise ValueError("Raw feature content failed its extraction fingerprint")
    feature_manifest_hash = stable_hash(feature_manifest)
    config_hash, current_code_hash = stable_hash(public_config(cfg)), code_hash()
    training = read_json(out_dir / "training_status.json")
    if training.get("status") != "complete":
        raise ValueError("Training is not complete; evaluation cannot use leftover model files")
    if training.get("configuration_sha256") != config_hash or training.get("code_sha256") != current_code_hash:
        raise ValueError("Training configuration or code provenance is stale")
    index_path = out_dir / "model_index.json"
    if training.get("model_index_sha256") != file_hash(index_path):
        raise ValueError("Model index content failed its training fingerprint")
    by_id = {int(row["song_id"]): row for row in manifest["rows"]}
    if len(by_id) != len(manifest["rows"]):
        raise ValueError("Duplicate manifest song IDs")
    labels = np.asarray([row["arousal"] for row in manifest["rows"]], dtype=float)
    if not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
        raise ValueError("Manifest arousal must be finite and normalized to [0,1]")
    index = read_json(index_path)
    if not isinstance(index, list) or not index:
        raise ValueError("A complete model_index.json list is required")
    if training.get("model_records") != len(index):
        raise ValueError("Training completion count differs from the model index")
    expected = set()
    for split in cfg["split_seeds"]:
        for rep in cfg["representations"]:
            for head in cfg["heads"]:
                for seed in cfg["model_seeds"]:
                    expected.add((split, rep, head, "main", seed))
    if cfg.get("ablations", True):
        for rep in ("mert", "fusion"):
            if rep not in cfg["representations"]:
                continue
            for head, variant in (("mlp", "pca128"), ("ridge", "no_pca")):
                if head in cfg["heads"]:
                    for seed in cfg["model_seeds"]:
                        expected.add((cfg["primary_split"], rep, head, variant, seed))
    observed = [(r["split_seed"], r["representation"], r["head"], r["variant"], r["seed"]) for r in index]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError(f"Model matrix incomplete or duplicated: missing={expected-set(observed)}, extra={set(observed)-expected}")
    cases = []
    splits = {}
    main_layers = {(r["split_seed"], r["representation"], r["head"], r["seed"]): r["layer"]
                   for r in index if r["variant"] == "main"}
    for record in index:
        split_seed = record["split_seed"]
        if split_seed not in splits:
            split_path = out_dir / "splits" / f"split_{split_seed}.json"
            split = read_json(split_path)
            assignment = split["assignments"]
            if set(assignment) != {str(i) for i in by_id} or set(assignment.values()) != {"train", "val", "test"}:
                raise ValueError("Split assignments must exactly partition the audited IDs")
            if split.get("data_content_hash") != dataset_hash or bool(split.get("smoke_only", False)) != smoke:
                raise ValueError("Split belongs to a different dataset or smoke mode")
            expected_split_hash = stable_hash(assignment if smoke else {"seed": split_seed, "assignments": assignment})
            if split.get("split_hash") != expected_split_hash:
                raise ValueError("Split content differs from its split fingerprint")
            ownership = defaultdict(set)
            for row in manifest["rows"]:
                owner = assignment[str(row["song_id"])]
                # These checks remain useful when evaluating an archived run
                # without access to its raw audio directory.
                for key in ("artist_group", "group_id", "audio_sha256", "pcm_sha256"):
                    if row.get(key) is not None:
                        ownership[(key, str(row[key]))].add(owner)
            if any(len(owners) > 1 for owners in ownership.values()):
                raise ValueError("Artist or duplicate recording crosses split partitions")
            train_rows = [r for r in manifest["rows"] if assignment[str(r["song_id"])] == "train"]
            val_rows = [r for r in manifest["rows"] if assignment[str(r["song_id"])] == "val"]
            splits[split_seed] = (split, {
                "configuration_sha256": config_hash, "code_sha256": current_code_hash,
                "raw_features_sha256": raw_hash, "feature_manifest_sha256": feature_manifest_hash,
                "split_sha256": file_hash(split_path),
                "train_validation_labels_sha256": stable_hash([[str(r["song_id"]), float(r["arousal"])] for r in train_rows + val_rows]),
                "training_groups_sha256": stable_hash([str(r.get("group_id", r["artist_group"])) for r in train_rows]),
            }, [str(r["song_id"]) for r in train_rows])
        split, expected_provenance, train_ids = splits[split_seed]
        assignment = split["assignments"]
        model_path = checked_child(out_dir, record["model_path"])
        path = checked_child(out_dir, record["predictions_path"])
        if model_path.name != "model.json" or path != model_path.parent / "predictions.npz":
            raise ValueError("Model index references predictions outside its model directory")
        model = read_json(model_path)
        status = read_json(model_path.parent / "status.json")
        if model.get("status") != "complete" or status.get("status") != "complete":
            raise ValueError("Per-model training did not complete")
        required_files = {"model.json", "weights.npz", "predictions.npz", "convergence.json"}
        if not required_files.issubset(status.get("files", {})):
            raise ValueError("Per-model completion record lacks required file fingerprints")
        for filename in required_files:
            if file_hash(model_path.parent / filename) != status["files"][filename]:
                raise ValueError(f"Model artifact failed its completion fingerprint: {filename}")
        if model.get("weights_file") != "weights.npz" or model.get("weights_sha256") != status["files"]["weights.npz"]:
            raise ValueError("Model weight reference differs from its completed weights")
        logs = read_json(model_path.parent / "convergence.json")
        if len(logs) != record["members"] + 1 or any(log.get("converged") is not True for log in logs):
            raise ValueError("Final model convergence is incomplete")
        for key in ("split_seed", "representation", "head", "variant", "layer", "alpha", "projection_dim", "members", "main_members"):
            if model.get(key) != record.get(key):
                raise ValueError(f"Model index and saved model disagree on {key}")
        if model.get("model_seed") != record["seed"] or model.get("training_song_ids") != train_ids:
            raise ValueError("Saved model training seed or training IDs disagree")
        provenance = model.get("provenance", {})
        if any(provenance.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError("Model provenance differs from current features, split, labels, config or code")
        for key in ("representation", "head", "variant", "projection_dim"):
            if provenance.get(key) != record[key]:
                raise ValueError(f"Model configuration provenance disagrees on {key}")
        fingerprint = stable_hash({**provenance, "model_seed": record["seed"],
                                   "layer": record["layer"], "alpha": record["alpha"], "members": record["members"]})
        if model.get("fingerprint") != fingerprint or status.get("fingerprint") != fingerprint:
            raise ValueError("Per-model provenance fingerprint is inconsistent")
        needed = cfg["large_members"] if (record["variant"] == "main" and split_seed == cfg["primary_split"] and cfg.get("ablations", True)) else cfg["members"]
        dim = cfg["projection_dim"] if record["variant"] == "main" else 128 if record["variant"] == "pca128" else None
        if record["members"] != needed or record["main_members"] != cfg["members"] or record["projection_dim"] != dim:
            raise ValueError("Model ensemble or projection differs from the requested experiment")
        if record["representation"] == "mir":
            if record["layer"] is not None:
                raise ValueError("MIR cannot have a selected MERT layer")
        elif record["layer"] not in cfg["layers"]:
            raise ValueError("Selected MERT layer is outside the frozen search")
        if record["variant"] != "main" and record["layer"] != main_layers[(split_seed, record["representation"], record["head"], record["seed"])]:
            raise ValueError("Projection ablation changed the main-model MERT layer")
        with np.load(path, allow_pickle=False) as saved:
            song_ids = np.asarray(saved["song_ids"], dtype=np.int64)
            if len(np.unique(song_ids)) != len(song_ids) or set(song_ids.tolist()) != set(by_id):
                raise ValueError("Prediction IDs must exactly match the audited dataset")
            order = np.argsort(song_ids)
            song_ids = song_ids[order]
            members = np.asarray(saved["members"], dtype=float)[order]
            single = np.asarray(saved["single"], dtype=float)[order]
            stored_mean = np.asarray(saved["mean"], dtype=float)[order]
        if members.ndim != 2 or members.shape[0] != len(song_ids) or single.shape != (len(song_ids),):
            raise ValueError("Invalid prediction dimensions")
        if members.shape[1] != needed:
            raise ValueError(f"Incorrect ensemble size for {record}")
        if any(not np.isfinite(values).all() or np.any((values < 0) | (values > 1)) for values in (members, single, stored_mean)):
            raise ValueError("Predictions must be finite and clipped to [0,1]")
        if stored_mean.shape != (len(song_ids),) or not np.allclose(stored_mean, members[:, :cfg["members"]].mean(1), rtol=1e-12, atol=1e-12):
            raise ValueError("Saved mean is not the declared main-member prefix average")
        mask = np.array([assignment[str(i)] == "test" for i in song_ids])
        ids = song_ids[mask]
        case = {**record, "ids": ids, "y": np.array([by_id[int(i)]["arousal"] for i in ids]),
                "groups": np.array([by_id[int(i)].get("group_id", by_id[int(i)]["artist_group"]) for i in ids]),
                "member_predictions": members[mask], "single_predictions": single[mask],
                "predictions_sha256": file_hash(path)}
        cases.append(case)
    return manifest, cases


def _contrast_case(cases, targets, budget, selector, beta, mask):
    values = []
    for case in cases:
        for target in targets:
            values.append(evaluate_catalog(case["y"][mask], case["member_predictions"][mask, :3],
                          case["single_predictions"][mask], case["ids"][mask], target, budget,
                          selector, beta)["error"])
    return float(np.mean(values))


def _paired_intervals(cases, cfg):
    # Only the predeclared main-model, B=32 contrasts receive pointwise intervals.
    groups = defaultdict(list)
    for case in cases:
        if case["variant"] == "main":
            groups[(case["split_seed"], case["representation"], case["head"])].append(case)
    records = []
    budget = 32
    if budget not in cfg["budgets"]:
        return records
    for (split, rep, head), runs in groups.items():
        for lhs, rhs in (("mean_prediction", "single"), ("mean_distance", "mean_prediction"), ("penalty", "mean_distance")):
            def statistic(mask, left=lhs, right=rhs, rs=runs):
                return (_contrast_case(rs, cfg["targets"], budget, left, 1.0 if left == "penalty" else 0, mask)
                        - _contrast_case(rs, cfg["targets"], budget, right, 0, mask))
            estimate = paired_artist_jackknife(statistic, runs[0]["groups"], minimum_catalog=budget)
            records.append({"split_seed": split, "head": head, "representation": rep,
                            "metric": "selection_error", "budget": budget, "target": "macro",
                            "lhs": lhs, "rhs": rhs, **estimate})
    for split in cfg["split_seeds"]:
        for head in cfg["heads"]:
            for left, right in (("mert", "mir"), ("fusion", "mert")):
                if (split, left, head) not in groups or (split, right, head) not in groups:
                    continue
                lrun, rrun = groups[(split, left, head)], groups[(split, right, head)]
                if not np.array_equal(lrun[0]["ids"], rrun[0]["ids"]):
                    raise ValueError("Paired comparisons require identical catalogs")
                def statistic(mask, lr=lrun, rr=rrun):
                    return (_contrast_case(lr, cfg["targets"], budget, "mean_prediction", 0, mask)
                            - _contrast_case(rr, cfg["targets"], budget, "mean_prediction", 0, mask))
                estimate = paired_artist_jackknife(statistic, lrun[0]["groups"], minimum_catalog=budget)
                records.append({"split_seed": split, "head": head, "representation": "paired",
                                "metric": "selection_error", "budget": budget, "target": "macro",
                                "lhs": left, "rhs": right, **estimate})
                def pred_statistic(mask, lr=lrun, rr=rrun):
                    def mae(rs):
                        return np.mean([np.abs(c["y"][mask] - c["member_predictions"][mask, :3].mean(1)).mean() for c in rs])
                    return float(mae(lr) - mae(rr))
                estimate = paired_artist_jackknife(pred_statistic, lrun[0]["groups"])
                records.append({"split_seed": split, "head": head, "representation": "paired",
                                "metric": "prediction_mae", "budget": "", "target": "",
                                "lhs": left, "rhs": right, **estimate})
    return records


def evaluate_all(out_dir, cfg):
    out_dir = Path(out_dir)
    manifest, cases = load_cases(out_dir, cfg)
    prediction_rows, selection_rows = [], []
    for case in cases:
        ks = [cfg["members"]]
        if case["variant"] == "main" and case["split_seed"] == cfg["primary_split"] and cfg.get("ablations", True):
            ks.append(cfg["large_members"])
        for k in ks:
            fields = {name: case[name] for name in ("split_seed", "representation", "head", "variant", "seed")}
            fields["k"] = k
            members = case["member_predictions"][:, :k]
            metrics = prediction_metrics(case["y"], members.mean(axis=1))
            prediction_rows.append({**fields, **metrics})
            policies = [("random", 0.0), ("single", 0.0), ("mean_prediction", 0.0),
                        ("mean_distance", 0.0), ("oracle", 0.0)] + [("penalty", b) for b in cfg["betas"]]
            for selector, beta in policies:
                for budget in cfg["budgets"]:
                    targets = []
                    for target in cfg["targets"]:
                        metrics = evaluate_catalog(case["y"], members, case["single_predictions"],
                                                   case["ids"], target, budget, selector, beta)
                        row = {**fields, "selector": selector, "beta": beta, "budget": budget,
                               "target": target, **metrics}
                        selection_rows.append(row)
                        targets.append(row)
                    macro = {**fields, "selector": selector, "beta": beta, "budget": budget, "target": "macro"}
                    for name in ("error", "regret", "optimism_gap", "oracle_error"):
                        values = [r[name] for r in targets if r[name] is not None]
                        macro[name] = float(np.mean(values)) if values else None
                    selection_rows.append(macro)
    keys = ["split_seed", "representation", "head", "variant", "k"]
    prediction_summary = _summarize(prediction_rows, keys, ["mae", "ccc"])
    selection_summary = _summarize(selection_rows, keys + ["selector", "beta", "budget", "target"],
                                   ["error", "regret", "optimism_gap", "oracle_error"])
    tables = out_dir / "tables"
    write_csv(tables / "prediction_per_seed.csv", prediction_rows)
    write_csv(tables / "prediction_summary.csv", prediction_summary)
    write_csv(tables / "selection_per_seed.csv", selection_rows)
    write_csv(tables / "selection_summary.csv", selection_summary)
    intervals = _paired_intervals(cases, cfg)
    if intervals:
        write_csv(tables / "paired_differences.csv", intervals)
    coverage, baselines = [], []
    for split_seed in cfg["split_seeds"]:
        assignments = read_json(out_dir / "splits" / f"split_{split_seed}.json")["assignments"]
        train_y = [r["arousal"] for r in manifest["rows"] if assignments[str(r["song_id"])] == "train"]
        for part in ("train", "val", "test"):
            rows = [r for r in manifest["rows"] if assignments[str(r["song_id"])] == part]
            y = np.array([r["arousal"] for r in rows])
            for target in cfg["targets"]:
                coverage.append({"split_seed": split_seed, "partition": part, "target": target,
                                 "n_songs": len(y), "n_groups": len(set(r.get("group_id", r["artist_group"]) for r in rows)),
                                 "n_within_005": int(np.sum(np.abs(y - target) <= 0.05)),
                                 "label_mean": float(y.mean()), "label_min": float(y.min()), "label_max": float(y.max())})
            if part == "test":
                baselines.append({"split_seed": split_seed, "baseline": "training_mean",
                                  **prediction_metrics(y, np.full(len(y), np.mean(train_y)))})
    write_csv(tables / "data_coverage.csv", coverage)
    write_csv(tables / "prediction_baseline.csv", baselines)
    report = {"status": "smoke_evaluated" if cfg.get("smoke_only") else "evaluated",
              "smoke_only": bool(cfg.get("smoke_only")), "model_records": len(cases),
              "prediction_rows": len(prediction_rows), "selection_rows": len(selection_rows),
              "paired_contrasts": len(intervals), "evaluation": "exact_uniform_without_replacement",
              "interval_scope": "exploratory pointwise approximate; fixed models; artist/duplicate groups",
              "table_sha256": {p.name: file_hash(p) for p in sorted(tables.glob("*.csv"))},
              "prediction_sha256": {c["predictions_path"]: c["predictions_sha256"] for c in cases},
              "model_index_sha256": file_hash(out_dir / "model_index.json"),
              "data_manifest_sha256": file_hash(out_dir / "data_manifest.json"),
              "split_sha256": {str(s): file_hash(out_dir / "splits" / f"split_{s}.json") for s in cfg["split_seeds"]},
              "config_hash": stable_hash(public_config(cfg)), "code_hash": code_hash()}
    write_json(out_dir / "evaluation_status.json", report)
    return report
