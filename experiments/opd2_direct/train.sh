#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export OPD2_ENABLED=True
export OPD2_CENTER_TOP_K=1024
export LOG_PROB_TOP_K=16
export JSD_SCORE_TOP_K=null
export TOP_K_STRATEGY=only_stu
export TOKEN_SELECTION_SCORE=divergence
export TOKEN_MASK_SCOPE=both
export JS_TOKEN_FILTER_ENABLED=False
export USE_KL_LOSS=True
export ADAPTIVE_KL_LOSS_COEF=True
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_opd2_gateonly_direct_k16_center1024}

exec bash "${REPO_ROOT}/scripts/train_s2d_opd.sh"
