"""One entry point: preflight, prepare, features, train, evaluate, report, all."""
from __future__ import annotations
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from arousal.common import code_hash, public_config, read_json, stable_hash, write_json


def validate_config(cfg):
    if not cfg.get("smoke_only"):
        fixed = {"split_seeds": [42,43,44], "model_seeds": [0,1,2], "primary_split": 42,
                 "representations": ["mir","mert","fusion"], "heads": ["ridge","mlp"],
                 "members": 3, "large_members": 10, "projection_dim": 32, "layers": [4,8,12],
                 "targets": [0.25,0.5,0.75], "budgets": [4,8,16,32], "betas": [0,0.5,1,2],
                 "ablations": True, "ridge_alphas": [0.1,1,10], "mlp_alphas": [0.001,0.1,10],
                 "mert_revision": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"}
        for key, expected in fixed.items():
            if cfg.get(key) != expected:
                raise ValueError(f"Full protocol requires {key}={expected!r}; use a separate exploratory protocol instead")
        if cfg["training"] != {"max_iter":3000,"max_fun":20000,"tol":1e-6,"hidden_units":64}:
            raise ValueError("Full training protocol changed")
    if cfg["members"] != 3 or cfg["large_members"] != 10:
        raise ValueError("This implementation's declared main/ablation member sizes are 3/10")


def environment():
    info = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for package in ("numpy","scipy","scikit-learn","soundfile","torch","transformers","huggingface-hub","threadpoolctl","einops","joblib","cloudpickle"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_capabilities"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["cuda_available"] = False
    return info


def preflight(cfg, out, info):
    if cfg.get("smoke_only"):
        result = {"status": "smoke_dependencies_only", "smoke_only": True}
    else:
        if sys.version_info[:2] != (3,12):
            raise RuntimeError("The supported runtime is Python 3.12")
        expected = {"numpy":"2.3.5", "scipy":"1.17.0", "scikit-learn":"1.8.0",
                    "soundfile":"0.14.0", "torch":"2.7.1", "transformers":"4.44.2",
                    "huggingface-hub":"0.36.2", "threadpoolctl":"3.6.0",
                    "einops":"0.8.2", "joblib":"1.6.0", "cloudpickle":"3.1.2"}
        bad = {p:info["packages"].get(p) for p,v in expected.items()
               if (info["packages"].get(p) or "").split("+")[0] != v}
        if bad:
            raise RuntimeError(f"Pinned runtime mismatch: {bad}. Follow setup_cloud.sh.")
        if cfg["device"] == "cuda" and not info["cuda_available"]:
            raise RuntimeError("CUDA requested but no CUDA device is available")
        if cfg["device"] == "cuda" and info.get("torch_cuda") != "12.8":
            raise RuntimeError("This GPU release requires the pinned PyTorch CUDA 12.8 wheel; follow setup_cloud.sh")
        import numpy as np
        from arousal.encoders import MERTEncoder
        encoder = MERTEncoder(revision=cfg["mert_revision"], device=cfg["device"])
        wave = (0.05 * np.sin(2*np.pi*220*np.arange(24000)/24000)).astype(np.float32)
        embedded = encoder.encode(wave)
        result = {"status":"passed", "smoke_only":False, "short_inference_shape":list(embedded.shape),
                  "encoder":encoder.metadata, "purpose":"runtime validation only; no research result"}
        del encoder
        import gc
        gc.collect()
        if cfg["device"] == "cuda":
            import torch
            torch.cuda.empty_cache()
    write_json(out / "preflight.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/full.json")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/full"))
    parser.add_argument("--stage", choices=["preflight","prepare","features","train","evaluate","report","all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu","cuda"])
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    if args.device:
        cfg["device"] = args.device
    validate_config(cfg)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(cfg.get("threads", 4))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = stable_hash(public_config(cfg))
    source_hash = code_hash()
    state_path = out / "run_status.json"
    old = read_json(state_path) if state_path.exists() else None
    if old and (old["config_hash"] != fingerprint or old["code_hash"] != source_hash):
        raise ValueError("Output belongs to a different configuration/code version. Choose a fresh --out.")
    if old and not args.resume:
        raise ValueError("Output exists. Use --resume with unchanged code/config, or choose a fresh --out.")
    cfg["_resume"] = args.resume
    write_json(out / "config.json", public_config(cfg))
    info = environment()
    write_json(out / "environment.json", info)
    state = {"status":"running", "smoke_only":bool(cfg.get("smoke_only")), "config_hash":fingerprint,
             "code_hash":source_hash, "requested_stage":args.stage, "completed_stages":old.get("completed_stages",[]) if old else [],
             "started_at_unix":time.time()}
    write_json(state_path, state)
    try:
        if args.stage == "preflight":
            preflight(cfg, out, info)
            print(info, flush=True)
            state["status"] = "preflight_passed"
        else:
            stages = ["preflight","prepare","features","train","evaluate","report"] if args.stage == "all" else [args.stage]
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=cfg.get("threads",4)):
                for stage in stages:
                    print(f"[{stage}] {'SYNTHETIC SMOKE' if cfg.get('smoke_only') else 'DEAM'}", flush=True)
                    state["active_stage"] = stage
                    write_json(state_path, state)
                    if stage == "preflight":
                        preflight(cfg, out, info)
                    elif stage in ("prepare","features") and cfg.get("smoke_only"):
                        from arousal.smoke import prepare_smoke
                        # Same process's second synthetic stage reuses its generated features.
                        prepare_smoke(out, {**cfg, "_resume": stage == "features" or args.resume})
                    elif stage == "prepare":
                        from arousal.data import prepare_data
                        prepare_data(args.data.resolve(), out, cfg)
                    elif stage == "features":
                        from arousal.features import extract_features
                        extract_features(args.data.resolve(), out, cfg)
                    elif stage == "train":
                        from arousal.model import train_all
                        train_all(out, cfg)
                    elif stage == "evaluate":
                        from arousal.evaluate import evaluate_all
                        evaluate_all(out, cfg)
                    elif stage == "report":
                        from arousal.report import verify_and_report
                        verification = verify_and_report(out, cfg)
                        if bool(verification["smoke_only"]) != bool(cfg.get("smoke_only")):
                            raise ValueError("Verification provenance differs from requested run mode")
                    if stage not in state["completed_stages"]:
                        state["completed_stages"].append(stage)
                    write_json(state_path, state)
            if args.stage in ("all","report"):
                # Report verifies the full model and evaluation matrix even when run alone.
                state["status"] = "smoke_complete" if cfg.get("smoke_only") else "complete"
            else:
                state["status"] = "stage_complete"
        state["finished_at_unix"] = time.time()
        write_json(state_path, state)
        print(f"{state['status']}: {out}", flush=True)
        return 0
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), finished_at_unix=time.time())
        write_json(state_path, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
