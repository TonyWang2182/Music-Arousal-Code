#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Select a Python 3.12 cloud image or environment first"'
if [ -x .venv/bin/python ]; then
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Existing .venv is not Python 3.12; use a fresh project directory"'
fi
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "${1:-cuda}" = "cpu" ]; then
  .venv/bin/python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/python -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python - "${1:-cuda}" <<'PY'
import sys
import torch
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
if sys.argv[1] != "cpu":
    assert torch.cuda.is_available(), "CUDA unavailable: inspect nvidia-smi and the installed wheel"
    assert torch.version.cuda == "12.8", "Expected the CUDA 12.8 wheel"
    print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
    a = torch.ones((64, 64), device="cuda")
    assert torch.all(a @ a == 64).item()
    y = torch.nn.functional.conv1d(torch.ones((1, 8, 1024), device="cuda"), torch.ones((16, 8, 5), device="cuda"))
    assert torch.isfinite(y).all().item() and torch.all(y == 40).item()
    torch.cuda.synchronize()
    print("GPU_OPERATIONS_READY")
PY
mkdir -p setup_checks
.venv/bin/python -m pip freeze > setup_checks/pip-freeze.txt
printf '%s\n' 'Environment installed. Follow README.md: verify local DEAM, cache MERT, pass offline inference, then start the experiment.'
