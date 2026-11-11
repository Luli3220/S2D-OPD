# OPD2-style gate-only comparison + Direct-OPD

This experiment transfers the centered joint-condition **gate** from
[On-Policy Delta Distillation, Section 2.3](https://arxiv.org/abs/2607.15161)
to the existing Direct-OPD candidate update. Report it as **OPD2-style gating +
Direct-OPD (gate only)**, not as a full reproduction of sampled-token OPD2.
Centering is used only to construct the boolean gate. It does not change reward
magnitudes: each surviving candidate keeps its exact original Direct-OPD reward.

## Gate definition

At each student-generated state and candidate v, define

```
delta(v) = log p_teacher(v) - log p_teacher_reference(v)
opd(v)   = log p_teacher(v) - log p_current_student(v)
```

Compute each expectation under the **current student**, independently at each
state. The centering support is its top-1024 token IDs, force-including the
sampled token by replacing the last candidate if needed. Weights are the original
full-vocabulary student probabilities, **without top-k renormalization**:

```
b_delta = sum_centering_support p_student(v) * delta(v)
b_opd   = sum_centering_support p_student(v) * opd(v)
A_delta(v) = delta(v) - b_delta
A_opd(v)   = opd(v) - b_opd
gate(v)    = (A_delta(v) * A_opd(v) > 0)
```

Both positive/positive and negative/negative updates survive. A zero product is
masked, following the strict condition in the paper's Eq. 8. The current official
trainer instead zeros negative products only; that implementation keeps the
zero-product boundary case. We use the paper's strict condition, which turns off
the delta update when the student equals the teacher.

The Direct-OPD reward/update support remains the original student top-16:

```
reward_weight(v) = softmax(student_logp_on_top16)[v]
update_signal(v) = reward_weight(v) * delta(v) * gate(v)
```

In code, we mask the existing reward tensor directly, preserving its original
values and dtype at retained entries; we do not recompute its weights. The
paper's centered A_delta magnitude is intentionally not substituted into the
update, so this comparison isolates gating rather than also changing rewards.

The centering support is used only for expectations. It does not replace the
top-16 reward candidates. No top-10% position selector is applied and there is
no fixed retention budget. Gating is **per candidate**, not one response-position
mask broadcast to every candidate. Loss still averages over all valid response
positions, rather than dividing by the number of retained candidates.

## What is retained / adapted

- Retain the JustRL-DeepSeek-1.5B / DeepSeek-R1-Distill-Qwen-1.5B teacher pair,
  Qwen3-1.7B student, data, rollout settings, PPO update and all other baseline
  training settings. The paper's teacher-base is mapped to our pre-RL teacher
  reference; this comparison targets the gating mechanism in our transfer setup.
- Retain the original initial-student anchor KL at **all valid positions**, and
  the adaptive-KL rule. Its `delta_opd/weighted_reward_mean` input is deliberately
  computed from the **ungated original top-16 reward**, so changing the gate does
  not also change the controller's input definition.
- Retain our raw top-16 probability-weighted delta update and reward scale. The paper
  applies centered A_delta magnitudes to the loss; here they only decide the gate.
  The paper also
  updates only sampled tokens, disables anchor KL, uses temperature 0.7, and
  scales rewards by 0.1; these settings are **not** copied into this controlled
  mechanism-transfer experiment. Current-student probabilities, not initial
  student anchor probabilities, define the OPD agreement signal.
- Follow the official top-1024 expectation convention and sampled-token
  inclusion, but accumulate means in fp32 in bounded position chunks instead of
  the official trainer's native bf16 reductions. Model forwards preserve the
  existing training precision.

Implementation reference:
[official OPD2 trainer](https://github.com/naver-ai/opd2/blob/dca428a122219e7641e2981dd9bac71d4a7a6eb4/src/open_r1/trainers/opd2_trainer.py),
in particular `_forward_and_get_logp_topk` and `_compute_opd_rewards`.

## Launch one experiment

Set existing model/data path variables to their actual server locations, matching
the previous JustRL/Qwen3-1.7B runs. Keep the same seed configuration.

```bash
export PYTHON_BIN="$(command -v python)"
export ACTOR_MODEL_PATH="/path/to/Qwen3-1.7B"
export REWARD_MODEL_PATH="/path/to/JustRL-DeepSeek-1.5B"
export TEACHER_REF_MODEL_PATH="/path/to/DeepSeek-R1-Distill-Qwen-1.5B"
export VAL_BEFORE_TRAIN=False
bash experiments/opd2_direct/train.sh
```

The default launcher uses 8 GPUs and a separate experiment name. No seed override
is invented by this launcher; reuse the matched baseline's actual seed setup.
`OPD2_ENABLED=False` is the main script's default and leaves existing experiments
unchanged. Simultaneous JSD/top-10%/entropy/TA selection is rejected.

The centering path adds one no-grad student, teacher, and teacher-reference
forward per rollout. It scores specified IDs without constructing a 1024x1024
teacher/student overlap matrix, reduces expectations in position chunks, and
frees the large tensors before updating the actor. Top-1024 ID/probability banks
still require more memory than the top-16 baseline.

Check `opd2/center_top_k=1024`, finite `opd2/center_support_mass` and
`opd2/gate_keep_ratio`, and `opd2/weighted_reward_mean`. Gate retention is measured
over valid candidate entries and varies during training; it is not expected to
equal 10%. Actor reward/KL position keep ratios remain 1.0 because positions and
the anchor KL remain dense. All masks, means and probability weights are detached.
