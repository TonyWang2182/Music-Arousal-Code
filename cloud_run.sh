#!/usr/bin/env bash
# Restart this exact release in its own directory; leave older projects alone.
set -euo pipefail
cd "$(dirname "$0")"
export HF_ENDPOINT=https://huggingface.co
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
OUT=runs/full_rtx6000d_20260920
LOG=logs/full_rtx6000d_20260920.log
mkdir -p logs
case "${1:-start}" in
  start)
    command -v flock >/dev/null || { printf '%s\n' 'Missing flock (util-linux); install it before starting.' >&2; exit 1; }
    test -x .venv/bin/python || { printf '%s\n' 'Run bash setup_cloud.sh first.' >&2; exit 1; }
    # Take the lock before changing any output. A second click cannot create
    # two simultaneous writers; the background process inherits this lock.
    exec 9>logs/full_rtx6000d_20260920.lock
    flock -n 9 || { printf '%s\n' 'This experiment is already running. Use: bash cloud_run.sh status'; exit 1; }
    .venv/bin/python - <<'PY'
from pathlib import Path
from arousal.common import read_json
for name in ('data_assets.json', 'model_assets.json'):
    path = Path('setup_checks') / name
    if not path.is_file() or read_json(path).get('status') != 'passed':
        raise SystemExit('Finish prepare_assets.py data/model checks first; see README.md')
PY
    nohup .venv/bin/python -u run.py --config configs/full.json --data data --out "$OUT" --resume >> "$LOG" 2>&1 < /dev/null &
    printf '%s\n' "$!" > logs/full_rtx6000d_20260920.pid
    printf 'Launched PID %s. Log: %s\n' "$!" "$LOG"
    printf '%s\n' 'Use bash cloud_run.sh status to check progress. Close the browser if needed, but keep the cloud machine running.'
    ;;
  status)
    if command -v flock >/dev/null && ! flock -n logs/full_rtx6000d_20260920.lock -c true; then
      printf '%s\n' 'Process lock: HELD (a process is using this experiment output)'
    else
      printf '%s\n' 'Process lock: FREE (read run_status below for completion or failure)'
    fi
    if [ -f "$OUT/run_status.json" ]; then cat "$OUT/run_status.json"; fi
    if [ -f "$LOG" ]; then tail -n 25 "$LOG"; fi
    ;;
  *) printf '%s\n' 'Usage: bash cloud_run.sh start|status' >&2; exit 2 ;;
esac
