#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}

# Default diagnostic: the initial HF student checkpoint, without any training.
STAGE=${STAGE:-initial}
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"${REPO_ROOT}/models/Qwen3-1.7B"}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-"${REPO_ROOT}/models/JustRL-DeepSeek-1.5B"}
TEACHER_REF_MODEL_PATH=${TEACHER_REF_MODEL_PATH:-"${REPO_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B"}
TRAIN_DATASET=${TRAIN_DATASET:-"${REPO_ROOT}/datasets/train/skywork-or1-math-dapo-original.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs/${STAGE}"}
SAMPLE_SEED=${SAMPLE_SEED:-0}
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-'{}'}
SCORE_GPUS=${SCORE_GPUS:-$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')}
if ! [[ "${SCORE_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "No scoring GPUs found; set SCORE_GPUS to the number of available visible GPUs." >&2
  exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/audit.py" generate \
  --student "${STUDENT_MODEL_PATH}" --dataset "${TRAIN_DATASET}" --output "${OUTPUT_DIR}" \
  --prompts "${NUM_PROMPTS:-128}" --responses "${N_RESPONSES:-4}" --seed "${SAMPLE_SEED}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-1024}" --max-response-length "${MAX_RESP_LENGTH:-2048}" \
  --temperature "${TEMPERATURE:-1.0}" --tensor-parallel-size "${ROLLOUT_TP:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
  --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"

# Independent replicas partition frozen responses; no Ray or shared NCCL process group.
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${SCORE_GPUS}" \
  "${SCRIPT_DIR}/audit.py" score \
  --student "${STUDENT_MODEL_PATH}" --teacher "${REWARD_MODEL_PATH}" \
  --reference "${TEACHER_REF_MODEL_PATH}" --bank "${OUTPUT_DIR}/bank.jsonl" \
  --output "${OUTPUT_DIR}/scores" --supports 8 16 32 64 128 \
  --dtype "${SCORE_DTYPE:-float32}" --attention "${SCORE_ATTENTION:-sdpa}" \
  --teacher-temperature "${TEACHER_TEMPERATURE:-1.0}" --chunk-positions "${CHUNK_POSITIONS:-64}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/audit.py" summarize \
  --scores "${OUTPUT_DIR}/scores" --output "${OUTPUT_DIR}/analysis" \
  --retention "${JS_TOP_FRACTION:-0.10}" --bootstrap "${BOOTSTRAP_REPEATS:-2000}" --seed "${SAMPLE_SEED}"
