# Music Arousal Prediction Gains Do Not Always Translate into Better Target-Conditioned Selection

Code accompanying the paper of the same title. This repository compares arousal prediction with target-conditioned music selection on DEAM using six configurations: **MIR / frozen MERT / Fusion × Ridge / MLP**. It selects original recordings and evaluates them against held-out human ratings. It does not generate or edit audio, train an agent, or claim personalized emotion regulation.

## Protocol

- Use DEAM's 1,744 short excerpts and static whole-excerpt arousal means, normalized as `(rating - 1) / 8`; exclude 58 full songs. Retain annotation SD as metadata.
- Audit audio, IDs, labels, artist names and exact duplicates. Grouped splits use seeds 42/43/44 and approximately 70%/15%/15% train/validation/test groups. Artist aliases and near-duplicates cannot be exhaustively ruled out automatically.
- Decode complete excerpts at 24 kHz, mono, without additional gain normalization. Features are MIR32, MERT768 and Fusion800. MERT uses five-second chunks and output-frame-weighted pooling, with its official input processor.
- Fit scaling and non-whitened PCA32 on training data only. Validation MAE selects layer 4/8/12 and regularization. Models use three training seeds and three shared artist-bootstrap draws per seed; the full-training single-model comparator is separate.
- Compare single-model, mean-prediction, mean-distance and disagreement-penalized selection; report random and label-oracle references. Targets are 0.25/0.50/0.75, pool sizes 4/8/16/32 and penalties 0/0.5/1/2.
- Integrate the exact expectation over uniform pools of distinct song IDs. Report target error, oracle regret and per-target coverage. Artist-group jackknife intervals condition on fitted models; training-seed SD is separate. Overlapping splits are descriptive stability checks.
- On split 42, additionally compare PCA32 with PCA128 (MLP) or no PCA (Ridge), and K=3 with K=10 using identical first three members. The complete run contains 66 model records. Disagreement is a selection heuristic, not calibrated uncertainty.

## Install

Use **Linux, Python 3.12** and a compatible NVIDIA driver. This release pins **PyTorch 2.7.1+cu128** for the RTX 6000D (Blackwell); the installer checks real GPU matrix multiplication and convolution. PyTorch is installed separately from the other pinned requirements.

From the repository root:

```bash
sha256sum -c RELEASE_SHA256SUMS.txt
bash setup_cloud.sh
```

PyTorch comes from its [official CUDA 12.8 index](https://download.pytorch.org/whl/cu128). If ordinary Python dependencies download slowly, an optional [Tsinghua PyPI mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) can be selected for this installation:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5 bash setup_cloud.sh
```

The explicit PyTorch index in the script still takes precedence. Do not substitute a generic `torch==2.7.1` package with a different CUDA build.

## Prepare local data and MERT

Obtain the original archives from the [DEAM maintainers](https://cvml.unige.ch/databases/DEAM/) or your existing verified local copy. Upload the three original files into:

```text
data/raw/DEAM_audio.zip
data/raw/DEAM_Annotations.zip
data/raw/metadata.zip
```

Keep these inner archives zipped: the pipeline extracts what it needs. `configs/full.json` disables dataset downloading, so a missing archive fails locally instead of waiting for the data host. Original SHA256 values are fixed in `arousal/data.py`; the preparation command checks both SHA256 and ZIP CRC. A differently packaged archive is not interchangeable merely because its filename matches.

```bash
.venv/bin/python prepare_assets.py --only data
# Choose a persistent cache before downloading. On AutoDL, for example:
export HF_HOME=/root/autodl-tmp/hf-cache
source cloud_env.sh
.venv/bin/python prepare_assets.py --only model
```

`cloud_env.sh` uses **https://huggingface.co**, and optionally sources AutoDL's `/etc/network_turbo` when present. It must be sourced in the current terminal. That accelerator targets supported model/code hosts; it does not replace the PyTorch or PyPI download sources. On another server, choose a writable persistent `HF_HOME` instead.

The model is [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M), revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`. Preparation downloads only five required files, verifies their hashes, then runs real inference in an offline child process. It writes checks to `setup_checks/`, leaving experiment outputs untouched.

For an offline server, transfer an already prepared Hugging Face cache to the same `HF_HOME` structure, preserving its `hub/` blobs and snapshot symlinks. Then check it without network access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python prepare_assets.py --only model
```

Missing or modified model files cause an error. Reuse the same cache environment when resuming.

## Quick check, then full experiment

```bash
.venv/bin/python -u run.py --config configs/quick.json --out runs/quick --resume
bash cloud_run.sh start
bash cloud_run.sh status
```

**Quick is a synthetic pipeline test, not a DEAM result or proof of GPU/MERT readiness.** Its completion status is `smoke_complete`. The separate asset checks above test real data archives and real MERT inference.

`cloud_run.sh` uses a file lock and `nohup`, resumes `runs/full_rtx6000d_20260920`, and appends to `logs/full_rtx6000d_20260920.log`. It requires `flock` from util-linux and passed data/model preparation checks. The full run uses the cached model offline. Closing SSH is safe after launch; powering off the machine stops computation.

For a different output path, use the common entry point directly, with only one process writing to that path:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u run.py --config configs/full.json --data data \
  --out runs/full_custom --resume
```

Stages are `preflight`, `prepare`, `features`, `train`, `evaluate`, `report`, and `all` (default). `--resume` checks source/configuration fingerprints before reusing outputs. If either changes, use a fresh output directory; never edit the recorded hashes to bypass a mismatch.

After completing a run, regenerate numerical tables and the report from its saved predictions:

```bash
.venv/bin/python run.py --config configs/full.json --out runs/full_rtx6000d_20260920 --stage evaluate --resume
.venv/bin/python run.py --config configs/full.json --out runs/full_rtx6000d_20260920 --stage report --resume
```

These stages do not repeat GPU feature extraction. Keep the exact source and configuration used for the run.

For CPU use, install with `bash setup_cloud.sh cpu`, prepare the model with `--device cpu`, and add `--device cpu` to manual `run.py` commands. The background launcher uses the default GPU configuration.

## Outputs and verification

Each run saves configuration/environment, data audit and split manifests, raw feature caches, model states, individual member predictions, convergence records, CSV tables and a report. Formal results require `run_status.json: status=complete, smoke_only=false` and `verification.json: status=passed`, with all 66 records verified. Nonconverged selected/final models block completion.

Back up the complete run, exact source, dependency environment and reusable data/model caches privately. This repository intentionally contains no recordings, annotations, checkpoints, run outputs, paper files or personal cloud logs.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests cover grouping and label isolation, exact selection versus enumeration, beta-zero equivalence, shared member prefixes, save/reload agreement and artifact integrity. The release checksum list verifies source files before execution; run fingerprints additionally bind scientific outputs to their inputs.

See [THIRD_PARTY.md](THIRD_PARTY.md) for resource terms. Original project software is MIT licensed; DEAM and MERT retain their own terms. Downstream artist separation does not establish absence of pretrained-data overlap.
