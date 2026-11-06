<h1 align="center">Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD</h1>

<p align="center">
  <b>Distill the RL-induced policy shifts that matter most.</b>
</p>

<p align="center">
  Yibo Zhao* · Zixuan Yang* · Yunshi Lan · Xiang Li†
</p>

<p align="center">
  School of Data Science and Engineering<br>
  East China Normal University
</p>

<p align="center">
  * Equal contribution · † Corresponding author
</p>

<p align="center">
  <a href="https://hub.docker.com/r/yibozhao624/selective-direct-opd">
    <img src="https://img.shields.io/badge/Docker-Environment-2496ED?style=for-the-badge&amp;logo=docker&amp;logoColor=white" alt="Docker environment">
  </a>
  <a href="https://arxiv.org/abs/2609.29142">
    <img src="https://img.shields.io/badge/arXiv-2609.29142-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="arXiv paper">
  </a>
  <a href="https://huggingface.co/SelectiveDOPD">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face models">
  </a>
</p>

This repository contains the official code release for **<u>S</u>elective <u>S</u>upervision for <u>D</u>irect-OPD** (S2D-OPD).

## Quick Start

Use the [prebuilt Docker environment](https://hub.docker.com/r/yibozhao624/selective-direct-opd) with the published image tag:

```bash
docker pull yibozhao624/selective-direct-opd:cu128-py312-np226-wandb-latest
docker run --gpus all --rm -it \
  -v "$PWD":/workspace -w /workspace \
  yibozhao624/selective-direct-opd:cu128-py312-np226-wandb-latest bash
```

The image provides the software environment; model weights and the training data must still be prepared as described below. Alternatively, install dependencies with Conda:

```bash
conda create -n s2d-opd python=3.12 -y
conda activate s2d-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install math-verify pyarrow transformers
cd ..

export PYTHON_BIN="$(command -v python)"
```

See the [setup guide](docs/setup.md) for data preparation, paths, and launch options.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── docs/setup.md                         # Detailed setup instructions
├── eval/                                 # Validation and held-out datasets
├── scripts/
│   ├── train_s2d_opd.sh                    # Configurable training entry point
│   ├── eval.sh                             # Validation-only evaluation entry point
│   ├── prepare_skywork_math.py             # Training-data converter
│   ├── train_ladder.sh                     # Select a divergence percentile interval
│   ├── train_justrl_qwen_js_top10.sh        # Highest-divergence 10% of tokens
│   ├── train_justrl_qwen_js_bottom10.sh     # Lowest-divergence 10% of tokens
│   ├── train_justrl_qwen_random_top10.sh    # Random 10% of tokens
│   └── convert_fsdp_to_hf.sh               # Export FSDP checkpoints to HF format
└── verl/
    ├── scripts/                           # Dependency installation and utilities
    ├── tests/                             # Bundled tests
    ├── requirements*.txt                  # Dependency specifications
    ├── pyproject.toml                     # Package/build configuration
    ├── setup.py
    ├── LICENSE                            # Upstream license
    └── verl/
        ├── trainer/                       # Training entry point, PPO loop and configs
        ├── workers/                       # Actor, rollout and reward-model workers
        ├── models/                        # Model implementations and integrations
        ├── utils/                         # Data loading, reward scoring and utilities
        └── model_merger/                  # Checkpoint export implementation
```

The token-selection implementation is in [`verl/verl/workers/actor/dp_actor.py`](verl/verl/workers/actor/dp_actor.py). Experiment settings and environment-variable overrides are defined in [`scripts/train_s2d_opd.sh`](scripts/train_s2d_opd.sh).

Student-entropy and reward-magnitude selectors, plus separate reward-only and
KL-only masking ablations, are available through dedicated launchers. See
[selection and masking ablations](docs/setup.md#selection-and-masking-ablations)
for commands, signal definitions, and normalization details.

## Models and data

The datasets under `eval/` have two distinct roles: AIME24 and AIME25 are used for validation during training, while AIME26, HMMT November 2025, and HMMT February 2026 are reserved for held-out evaluation. Download the following model weights separately and place them under `models/`:

- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (student)
- [JustRL-DeepSeek-1.5B](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B) (post-RL teacher)
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) (pre-RL teacher reference)

The default paths are:

```text
models/
├── Qwen3-1.7B/                            # Student
├── JustRL-DeepSeek-1.5B/                   # Post-RL teacher
└── DeepSeek-R1-Distill-Qwen-1.5B/           # Pre-RL teacher reference
datasets/
└── train/
    └── skywork-or1-math-dapo-original.parquet
eval/
├── aime24.parquet
├── aime25.parquet
├── aime26.parquet
├── hmmt_nov_2025.parquet
└── hmmt_feb_2026.parquet
```

The training data is not included. Follow the [setup guide](docs/setup.md#training-data) to prepare the Skywork-OR1 training parquet.

To use files stored elsewhere, set the following variables before training:

```bash
export ACTOR_MODEL_PATH="/path/to/Qwen3-1.7B"
export REWARD_MODEL_PATH="/path/to/JustRL-DeepSeek-1.5B"
export TEACHER_REF_MODEL_PATH="/path/to/DeepSeek-R1-Distill-Qwen-1.5B"
export TRAIN_DATASET="/path/to/train.parquet"
export TEST_DATASET="['/path/to/aime24.parquet','/path/to/aime25.parquet']"
```

## Running experiments

Run commands from the repository root after activating the environment and preparing the required files. For local console logging without a W&B account:

```bash
export PYTHON_BIN="$(command -v python)"
export LOGGER="['console']"

# Highest-divergence 10% of tokens
bash scripts/train_justrl_qwen_js_top10.sh

# Lowest-divergence 10% of tokens
bash scripts/train_justrl_qwen_js_bottom10.sh

# Random 10% of tokens (default selection seed: 42)
bash scripts/train_justrl_qwen_random_top10.sh

# A specified divergence percentile interval, e.g. the 20th–30th percentiles
bash scripts/train_ladder.sh 20_30
```

Choose one experiment per run. For a custom relative-selection experiment:

```bash
JS_TOKEN_SELECTION_MODE=relative \
JS_TOP_FRACTION=0.10 \
EXPERIMENT_NAME=s2d_opd_relative_top10 \
bash scripts/train_s2d_opd.sh
```

The scripts otherwise default to console and W&B logging. The console-only override above keeps experiment logs local. By default, `MANAGE_RAY=True` stops existing local Ray processes and starts a local Ray head; use a dedicated training machine or set `MANAGE_RAY=False` when using an already configured Ray runtime.

Checkpoints are saved under `${OUTPUT_ROOT}/${EXPERIMENT_NAME}`, validation generations under its `outputs/validation_log/` directory, and training logs under `${LOG_ROOT}/${EXPERIMENT_NAME}.log`.

## Evaluation

Evaluate a Hugging Face-format model on the three held-out datasets:

```bash
bash scripts/eval.sh /path/to/model
```

By default, the script evaluates AIME26, HMMT November 2025, and HMMT February 2026. It uses the same rollout and validation path as training with `VAL_ONLY=True`. Set `TEST_DATASET` to evaluate a subset and `VAL_N` to change the number of sampled responses per problem.

## Acknowledgments

This implementation builds on [Direct-OPD](https://github.com/BytedTsinghua-SIA/Direct-OPD) and [verl](https://github.com/volcengine/verl). We thank their contributors for releasing the code and infrastructure.

## Reference

```bibtex
@misc{zhao2026tokenworthdistillingselective,
      title={Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD},
      author={Yibo Zhao and Zixuan Yang and Yunshi Lan and Xiang Li},
      year={2026},
      eprint={2609.29142},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.29142},
}
```
