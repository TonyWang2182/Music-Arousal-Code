#!/usr/bin/env bash
# Source this after dependency installation, before downloading MERT.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf '%s\n' 'Use: source cloud_env.sh (not bash cloud_env.sh)' >&2
  exit 1
fi
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
else
  printf '%s\n' 'AutoDL acceleration script is absent; using your normal network.'
fi
export HF_ENDPOINT=https://huggingface.co
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
# Official HTTP fallback avoids an additional Xet/CAS transport failure mode.
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_ENABLE_HF_TRANSFER
printf 'Official model endpoint: %s\nModel cache: %s\n' "$HF_ENDPOINT" "$HF_HUB_CACHE"
