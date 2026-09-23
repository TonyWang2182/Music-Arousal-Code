"""Check local DEAM ZIPs and cache pinned MERT before starting a research run.

Only data/raw, the Hugging Face cache and setup_checks are written. No runs/
directory, experiment configuration or historical result is changed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from arousal.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parent
# Official pinned revision; weight hash comes from the Hub LFS metadata.
MERT_FILES = {
    "config.json": "ea2627c4c7825cd66f3c944b6b966331604c35928174e0100cd4a82829424e32",
    "preprocessor_config.json": "cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f",
    "configuration_MERT.py": "ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03",
    "modeling_MERT.py": "6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c",
    "pytorch_model.bin": "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd",
}


def check_data(data_root: Path, reuse_raw: Path | None = None):
    from arousal.data import FILES
    raw = data_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    checked = []
    missing = []
    for name, expected in FILES.items():
        path = raw / name
        if not path.exists() and reuse_raw is not None:
            source = reuse_raw / name
            if source.is_file():
                if file_hash(source) != expected:
                    raise ValueError(f"Old archive checksum mismatch; not copied: {source}")
                print(f"Reusing original archive: {source}", flush=True)
                # Write a temporary file first; interruption cannot leave a
                # partial ZIP under the final name. Linking commits it without
                # overwriting a file another process may have created meanwhile.
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=raw, prefix=f".{name}.", suffix=".copying", delete=False) as dst:
                        temporary = Path(dst.name)
                        with source.open("rb") as src:
                            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    if file_hash(temporary) != expected:
                        raise ValueError(f"Copied archive failed checksum: {source}")
                    try:
                        os.link(temporary, path)
                    except FileExistsError:
                        pass  # The common checks below validate the existing file.
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        if not path.exists():
            missing.append(str(path))
            continue
        if not path.is_file():
            raise ValueError(f"Expected a ZIP FILE, but this path is a directory: {path}")
        if file_hash(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}. Keep the file for inspection; do not bypass verification.")
        print(f"SHA256 OK: {name}; checking ZIP CRC...", flush=True)
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt ZIP member: {name}: {bad}")
        checked.append({"name": name, "sha256": expected, "bytes": path.stat().st_size})
    if missing:
        raise FileNotFoundError("Upload/extract the supplied DEAM data bundle first. Missing:\n" + "\n".join(missing))
    print("DATA_READY: all three original ZIPs passed. No DEAM network download is needed.", flush=True)
    return {"status": "passed", "data_root": str(data_root.resolve()), "archives": checked,
            "scope": "Archive checks only; real audio audit runs in the experiment's prepare stage."}


def cache_model(device: str, checks: Path):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from arousal.encoders import MERT_MODEL, MERT_SHA
    paths = {}
    for name in MERT_FILES:
        try:
            path = hf_hub_download(MERT_MODEL, name, revision=MERT_SHA, local_files_only=True)
            print(f"Cached: {name}", flush=True)
        except LocalEntryNotFoundError:
            print(f"Downloading pinned official file: {name}", flush=True)
            path = hf_hub_download(MERT_MODEL, name, revision=MERT_SHA)
        digest = file_hash(path)
        if digest != MERT_FILES[name]:
            raise ValueError(f"Pinned MERT file checksum mismatch: {path}. Keep the cache for inspection; do not bypass verification.")
        paths[name] = {"path": path, "bytes": Path(path).stat().st_size, "sha256": digest}
    print("Testing real MERT loading and inference with networking disabled...", flush=True)
    offline_env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # A child process sees offline variables before importing HF/transformers.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--offline-check",
                    "--device", device, "--checks", str(checks)],
                   cwd=ROOT, env=offline_env, check=True)
    return {"status": "passed", "model": MERT_MODEL, "revision": MERT_SHA,
            "files": paths, "offline_inference_passed": True, "device": device}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["data", "model", "all"], default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--reuse-raw", type=Path, help="Optionally copy missing verified ZIPs from an older data/raw folder")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--checks", type=Path, default=ROOT / "setup_checks")
    parser.add_argument("--offline-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    checks = args.checks.resolve()
    checks.mkdir(parents=True, exist_ok=True)
    if args.offline_check:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("Offline check requires both offline flags before Python starts")
        from run import environment, preflight
        cfg = read_json(ROOT / "configs/full.json")
        cfg["device"] = args.device
        info = environment()
        write_json(checks / "environment.json", info)
        result = preflight(cfg, checks, info)
        if result["short_inference_shape"] != [13, 768]:
            raise ValueError(f"Unexpected MERT output: {result['short_inference_shape']}")
        print("MERT_OFFLINE_READY: real checkpoint inference passed; shape=(13, 768).", flush=True)
        return 0
    # These defaults match cloud_env.sh and cloud_run.sh; callers may select a
    # persistent cache explicitly before launching Python.
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    if args.only in ("data", "all"):
        write_json(checks / "data_assets.json", check_data(args.data.resolve(), args.reuse_raw))
    if args.only in ("model", "all"):
        write_json(checks / "model_assets.json", cache_model(args.device, checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
