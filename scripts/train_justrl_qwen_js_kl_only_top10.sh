#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_js_kl_only_top10}
export TOKEN_SELECTION_SCORE=divergence
export DIVERGENCE_ESTIMATOR=JSD
export TOKEN_MASK_SCOPE=kl_only
exec bash "${SCRIPT_DIR}/train_justrl_qwen_js_top10.sh"
