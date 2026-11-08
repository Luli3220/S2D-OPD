#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_reward_magnitude_top10}
export TOKEN_SELECTION_SCORE=reward_magnitude
export TOKEN_MASK_SCOPE=both
exec bash "${SCRIPT_DIR}/train_justrl_qwen_js_top10.sh"
