# Setup

This repository includes the **<u>S</u>elective <u>S</u>upervision for <u>D</u>irect-OPD** ($S^2$D-OPD) training code and processed evaluation datasets. Model weights and the main training parquet must be downloaded separately.

## Environment

```bash
conda create -n s2d-opd python=3.12 -y
conda activate s2d-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
cd ..
```

The launch scripts use `/usr/bin/python3.12` by default. Override it when the environment uses another path:

```bash
export PYTHON_BIN="$(which python)"
```

## Required files

The default paths are:

```text
models/
  Qwen3-1.7B/
  JustRL-DeepSeek-1.5B/
  DeepSeek-R1-Distill-Qwen-1.5B/

datasets/
  train/skywork-or1-math-dapo-original.parquet

eval/
  aime24.parquet
  aime25.parquet
  aime26.parquet
  hmmt_nov_2025.parquet
  hmmt_feb_2026.parquet
```

Download the model weights from [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [JustRL-DeepSeek-1.5B](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B), and [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). The processed evaluation files are already included.

### Training data

Download the [Skywork-OR1 math parquet](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/blob/main/data/math-00000-of-00001.parquet), then convert it from the repository root:

```bash
mkdir -p datasets/raw datasets/train

curl -L --fail \
  https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/resolve/main/data/math-00000-of-00001.parquet \
  -o datasets/raw/skywork-or1-math.parquet

python scripts/prepare_skywork_math.py \
  --input datasets/raw/skywork-or1-math.parquet \
  --output datasets/train/skywork-or1-math-dapo-original.parquet
```

The converter applies the DAPO-style prompt used by Direct-OPD and validates the generated parquet.

## Training

Launch the default selective training configuration:

```bash
bash scripts/train_s2d_opd.sh
```

The repository also provides launchers for the top-10%, bottom-10%, random-token, and percentile-interval experiments under `scripts/`.

To redirect models, data, checkpoints, and logs:

```bash
MODEL_ROOT=/path/to/models \
DATA_ROOT=/path/to/datasets \
OUTPUT_ROOT=/path/to/checkpoints \
LOG_ROOT=/path/to/logs \
bash scripts/train_s2d_opd.sh
```

Common overrides:

```bash
ACTOR_MODEL_PATH=/path/to/student \
REWARD_MODEL_PATH=/path/to/post_rl_teacher \
TEACHER_REF_MODEL_PATH=/path/to/pre_rl_teacher \
TRAIN_DATASET=/path/to/train.parquet \
TOTAL_TRAINING_STEPS=300 \
GPUS_PER_NODE=8 \
NUM_NODES=1 \
bash scripts/train_s2d_opd.sh
```

AIME24 and AIME25 are used as validation sets during training. Checkpoints are saved under `${OUTPUT_ROOT}/${EXPERIMENT_NAME}`, and validation generations are written under `${CHECKPOINT_DIR}/outputs/validation_log/`.

## Selection and masking ablations

The following launchers use Qwen3-1.7B with the JustRL teacher/reference pair,
the existing top-10% per-response selection, and separate experiment directories:

| Experiment | Command | Reward positions | Anchor-KL positions |
| --- | --- | --- | --- |
| Student entropy | `bash scripts/train_justrl_qwen_entropy_top10.sh` | Selected | Selected |
| Reward magnitude | `bash scripts/train_justrl_qwen_reward_magnitude_top10.sh` | Selected | Selected |
| JSD reward-only mask | `bash scripts/train_justrl_qwen_js_reward_only_top10.sh` | Selected | All valid |
| JSD KL-only mask | `bash scripts/train_justrl_qwen_js_kl_only_top10.sh` | All valid | Selected |

The main launcher exposes two independent settings:

- `TOKEN_SELECTION_SCORE=divergence|entropy|reward_magnitude` (default: `divergence`).
  Divergence continues to use `DIVERGENCE_ESTIMATOR=JSD|FKL|RKL`.
- `TOKEN_MASK_SCOPE=both|reward_only|kl_only` (default: `both`).
  These options control **where masking applies**, not whether a loss term exists.
  Both one-term masking ablations require `USE_KL_LOSS=True`.

For example, entropy selection with a dense KL anchor:

```bash
TOKEN_SELECTION_SCORE=entropy TOKEN_MASK_SCOPE=reward_only \
JS_TOP_FRACTION=0.10 JS_TOKEN_SELECTION_MODE=relative \
bash scripts/train_s2d_opd.sh
```

Entropy is `-sum_v p_student(v) log p_student(v)` over the **full vocabulary**,
at the training temperature, obtained from the existing student log-probability
forward. It is not the entropy of renormalized top-16 candidates. Reward magnitude
is `sum_v p_bar_student(v) * abs(log p_teacher(v) - log p_reference(v))`
on the existing Direct-OPD candidate support, with normalized student weights.
Absolute values are taken before summing, so positive and negative shifts do not
cancel. Neither selector needs an extra model forward. Scores and masks are
detached and fixed for the actor updates on that rollout batch.

All ranking scopes, percentile intervals and absolute cutoffs apply to the chosen
score. Thresholds therefore have score-specific units; the legacy `js_*` option
names are retained for compatibility. Random selection still ignores scores.
`actor/js_*` score statistics remain teacher/reference divergence diagnostics;
`actor/selection_score_*` reports the actual ranking signal. The reward and KL
keep ratios are logged separately as `actor/reward_token_keep_ratio` and
`actor/kl_token_keep_ratio` (the latter describes the mask when KL is enabled).

With the default `LOSS_AGG_MODE=token-mean`, each term is divided by its own mask's
valid-position count **within each micro-batch**, then uses the existing gradient
accumulation scaling. This preserves the previous reduction behavior; it is not
a new global token-count reduction across micro-batches or distributed ranks.
The adaptive KL controller still uses the all-token reward statistic. The default
entropy regularization coefficient is zero; if explicitly enabled, entropy
regularization retains the selected-position mask regardless of mask scope.
`JS_TOKEN_FILTER_ENABLED=False` makes reward and KL both dense.

## Held-out evaluation

Evaluate a Hugging Face-format model on AIME26 and the two HMMT sets:

```bash
bash scripts/eval.sh /path/to/model
```

The evaluation launcher uses the same rollout and validation path as training with `VAL_ONLY=True`. Use `TEST_DATASET` to select a subset or `VAL_N` to change the number of sampled responses per problem.

## Notes

- `LOGGER="['console']"` disables W&B logging.
- `MANAGE_RAY=True` starts and stops a local Ray head inside the launcher.
- `PYTHON_BIN`, `CUDA_VISIBLE_DEVICES`, and the batch-size variables can be overridden for the local environment.
