#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Exact TA selection on the full rollout batch, with the Direct-OPD objective.
# Model/data/GPU overrides are inherited from the existing JustRL launcher.
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export JS_TOP_FRACTION=${JS_TOP_FRACTION:-0.10}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_ta_rho${JS_TOP_FRACTION}_k${LOG_PROB_TOP_K}}
export TOKEN_SELECTION_SCORE=teachability
export TOKEN_MASK_SCOPE=both
export TOP_K_STRATEGY=only_stu
export JS_TOKEN_FILTER_ENABLED=True
export JS_TOKEN_SELECTION_MODE=relative
export JS_TOKEN_SELECTION_AGGREGATION=rollout-agg

exec bash "${SCRIPT_DIR}/train_justrl_qwen_js_top10.sh"
