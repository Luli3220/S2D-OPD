#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash experiments/jsd_support/train.sh SCORE_K (e.g. 8 or 64)" >&2
  exit 2
fi
export JSD_SCORE_TOP_K="$1"
export LOG_PROB_TOP_K=16
export TOKEN_SELECTION_SCORE=divergence
export DIVERGENCE_ESTIMATOR=JSD
export TOP_K_STRATEGY=only_stu
export JS_TOKEN_FILTER_ENABLED=True
export JS_TOP_FRACTION=0.10
export JS_TOKEN_SELECTION_MODE=relative
export JS_TOKEN_SELECTION_AGGREGATION=response-agg
export TOKEN_MASK_SCOPE=both
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_js_scorek${JSD_SCORE_TOP_K}_rewardk16_top10}

exec bash "${REPO_ROOT}/scripts/train_justrl_qwen_js_top10.sh"
