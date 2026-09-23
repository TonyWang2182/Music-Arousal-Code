"""Numerical integrity checks and a compact, non-editorial run report."""
from pathlib import Path
import numpy as np
from .common import checked_child, code_hash, file_hash, public_config, read_json, stable_hash, write_json
from .evaluate import load_cases


def verify_and_report(out_dir, cfg):
    out_dir = Path(out_dir)
    manifest, cases = load_cases(out_dir, cfg)
    evaluation = read_json(out_dir / "evaluation_status.json")
    smoke = bool(cfg.get("smoke_only"))
    if bool(manifest.get("smoke_only")) != smoke or bool(evaluation.get("smoke_only")) != smoke:
        raise ValueError("Smoke and real experiment artifacts cannot be mixed")
    expected_status = "smoke_evaluated" if smoke else "evaluated"
    expected_lineage = {"model_index_sha256": file_hash(out_dir / "model_index.json"),
                       "data_manifest_sha256": file_hash(out_dir / "data_manifest.json"),
                       "split_sha256": {str(s): file_hash(out_dir / "splits" / f"split_{s}.json") for s in cfg["split_seeds"]},
                       "config_hash": stable_hash(public_config(cfg)), "code_hash": code_hash(),
                       "prediction_sha256": {c["predictions_path"]: c["predictions_sha256"] for c in cases}}
    if evaluation.get("status") != expected_status or any(evaluation.get(k) != v for k,v in expected_lineage.items()):
        raise ValueError("Evaluation is incomplete or belongs to different inputs; rerun evaluate")
    for name, expected in evaluation["table_sha256"].items():
        if file_hash(out_dir / "tables" / name) != expected:
            raise ValueError(f"Table changed since evaluation: {name}")
    from .model import predict_saved
    maximum = 0.0
    with np.load(out_dir / "features" / "raw_features.npz", allow_pickle=False) as features:
        for case in cases:
            rep = case["representation"]
            x = (features["mir"] if rep == "mir" else features[f"mert_{case['layer']}"] if rep == "mert" else
                 np.concatenate([features["mir"], features[f"mert_{case['layer']}"]], axis=1))
            with np.load(checked_child(out_dir, case["predictions_path"]), allow_pickle=False) as predictions:
                feature_ids = [str(v) for v in features["song_ids"]]
                saved_ids = [str(v) for v in predictions["song_ids"]]
                if len(set(feature_ids)) != len(feature_ids) or set(feature_ids) != set(saved_ids):
                    raise ValueError("Feature and saved prediction IDs mismatch")
                by_id = {v:i for i,v in enumerate(feature_ids)}
                members, single = predict_saved(checked_child(out_dir, case["model_path"]), x[[by_id[v] for v in saved_ids]])
                error = max(float(np.max(np.abs(members - predictions["members"]))),
                            float(np.max(np.abs(single - predictions["single"]))),
                            float(np.max(np.abs(members[:, :cfg['members']].mean(1) - predictions["mean"]))))
                maximum = max(maximum, error)
    if maximum > 1e-10:
        raise ValueError(f"Saved-model prediction roundtrip mismatch: {maximum}")
    verification = {"status": "passed", "smoke_only": smoke, "model_records": len(cases),
                    "max_reload_prediction_difference": maximum,
                    "checks": ["complete_model_matrix", "unique_prediction_ids", "expected_member_counts",
                               "table_hashes", "model_reload", "main_prediction_uses_first_three_members"],
                    "limitations": ["GPU encoder and real-data validity are not established by synthetic testing"] if smoke else []}
    write_json(out_dir / "verification.json", verification)
    title = "SYNTHETIC SMOKE RUN — NOT RESEARCH RESULTS" if smoke else "DEAM experimental outputs"
    text = f"# {title}\n\n"
    text += f"Verified model records: {len(cases)}. Maximum prediction reload difference: {maximum:.3g}.\n\n"
    text += "All numerical results are in tables/. Prediction and selection summaries average metrics after each seed's decisions.\n\n"
    text += "- prediction_summary.csv: regression performance per split, configuration and ensemble size.\n"
    text += "- selection_summary.csv: exact expected target error and regret, including each target and macro average.\n"
    text += "- paired_differences.csv: predeclared contrasts with pointwise approximate artist jackknife intervals.\n"
    text += "- data_coverage.csv: candidate availability near every target.\n"
    text += "- prediction_baseline.csv: training-mean predictor evaluated on held-out labels.\n\n"
    text += "Intervals condition on fitted models and the study's catalog sampling assumptions. They do not measure cross-dataset generalization. Split variability and training-seed variability are descriptive, separate quantities.\n"
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return verification
