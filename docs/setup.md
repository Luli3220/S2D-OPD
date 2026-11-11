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

- `TOKEN_SELECTION_SCORE=divergence|entropy|reward_magnitude|teachability` (default: `divergence`).
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

For divergence, entropy and reward magnitude, all ranking scopes, percentile
intervals and absolute cutoffs apply to the chosen score. Thresholds therefore
have score-specific units; the legacy `js_*` option
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

## TA selection + Direct-OPD

```bash
bash scripts/train_justrl_qwen_ta_top10.sh
```

The launcher keeps the existing JustRL teacher/reference pair, Direct-OPD reward,
policy loss, anchor-KL term, adaptive KL controller and loss reduction. It replaces
the position selector with the teachability score from
[TA-OPD, Section 4](https://arxiv.org/abs/2605.26844). The default is top-16 candidate
support and a 10% position budget; both are configurable:

```bash
LOG_PROB_TOP_K=32 JS_TOP_FRACTION=0.05 \
bash scripts/train_justrl_qwen_ta_top10.sh
```

For each response position, the selector computes:

- `D = KL(p_teacher_bar || p_student_bar)` on the **deduplicated union** of
  student and RL-teacher top-k IDs, with both distributions renormalized on that union.
- `C = sum(p_teacher(v) for v in student_topk)`, using full-vocabulary teacher
  probabilities, without renormalizing the coverage mass.
- `score = Norm(D) * Norm(C)`, where `Norm(z) = clip((z - Q05) / (Q95 - Q05 + 1e-8), 0, 1)`.

Quantiles and the top-`ceil(rho * N)` mask are computed once over all valid
positions in the **complete gathered rollout batch**. Padding and, when present,
format-masked responses do not participate. Ties use the earlier flattened
position; empty batches select nothing. This is `rollout-agg`, distinct from the
existing per-response and PPO-mini-batch ranking scopes. Scores and masks stay
detached and fixed throughout the actor updates for this rollout.

This implementation uses exact cross-probabilities on the union, rather than
filling missing top-k probabilities with epsilon. It adds one student scoring
forward on RL-teacher top-k IDs. Teacher scoring is reused. The original student
top-k IDs remain the reward/loss candidates, and the reference teacher continues
to define the Direct-OPD log-ratio reward. The launcher applies the TA mask to
both the reward-driven policy loss and anchor KL. Loss normalization is still
the existing per-micro-batch reduction described above, not TA-OPD's full objective.

The supported configuration is `reward_mode=delta_opd`, `TOP_K_STRATEGY=only_stu`,
FSDP/FSDP2, and actor `ulysses_sequence_parallel_size=1` (the current default).
The existing specified-ID scoring path does not shard target IDs for Ulysses;
TA configuration rejects that combination explicitly. Direct use of the main
launcher requires `TOKEN_SELECTION_SCORE=teachability`,
`JS_TOKEN_SELECTION_AGGREGATION=rollout-agg`, and
`JS_TOKEN_SELECTION_MODE=relative` (or `top_js`).

`ta/disagreement_*`, `ta/compatibility_*`, `ta/score_*`, and
`ta/selected_fraction` report rollout-wide selector statistics. Existing
teacher/reference divergence diagnostics retain their original meaning.
The experiment name includes the position budget and K by default. For matched
selector comparisons, also match ranking scope/budget allocation: the existing
JSD top-10% launcher selects per response, whereas TA selects over the rollout.

## Independent JSD scoring support

For support-size training checks with **fixed top-16 reward/update candidates**,
see [the support-audit guide](../experiments/jsd_support/README.md).
Run `bash experiments/jsd_support/train.sh 8` or `64`. These set the independent
`JSD_SCORE_TOP_K`; changing `LOG_PROB_TOP_K` alone would also change the reward.

## OPD2-style gating + Direct-OPD

Run `bash experiments/opd2_direct/train.sh` to transfer the centered-sign-agreement
gate to our top-16 Direct-OPD update. Centering determines the gate only; surviving
reward values remain exactly the original raw Direct-OPD rewards.
See [the experiment notes](../experiments/opd2_direct/README.md)
for exact centering, variable candidate retention, dense anchor KL and differences
from the paper's sampled-token objective. This mode must not be combined with
JSD/entropy/TA position selection.

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
