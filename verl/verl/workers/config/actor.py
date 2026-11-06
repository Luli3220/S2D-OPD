# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = ["PolicyLossConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        entropy_coeff (float): Entropy coefficient for regularization.
        js_token_filter_enabled (bool): Whether to filter response-token positions in actor loss.
        js_top_fraction (float): Fraction of valid token positions retained within the configured ranking scope.
        js_token_selection_mode (str): Token selection mode. Options: 'relative', 'absolute', 'ladder',
            'percentile_areas', 'random', plus compatible aliases 'top_js' and 'bottom_js'.
        js_token_selection_aggregation (str): Ranking scope. 'response-agg' ranks within each response;
            'batch-agg' pools all valid states in each PPO mini-batch.
        js_divergence_threshold (float): Absolute divergence cutoff for retaining valid response tokens.
        js_ladder_start_percent (float): Lower divergence-rank percentile for ladder selection.
        js_ladder_end_percent (float): Upper divergence-rank percentile for ladder selection.
        divergence_percentile_areas (list[list[float]]): Divergence-rank percentile areas to union.
        js_token_selection_seed (int): Base seed used by random token selection.
        divergence_estimator (str): Ranking divergence. Options: 'JSD', 'FKL', or 'RKL'.
        token_selection_score (str): 'divergence', 'entropy', or 'reward_magnitude'.
        token_mask_scope (str): Apply selection to 'both', 'reward_only', or 'kl_only' loss terms.
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0
    js_token_filter_enabled: bool = False
    js_top_fraction: float = 0.05
    js_token_selection_mode: str = "relative"
    js_token_selection_aggregation: str = "response-agg"
    js_divergence_threshold: float = 0.0
    js_ladder_start_percent: float = 90.0
    js_ladder_end_percent: float = 100.0
    divergence_percentile_areas: list[list[float]] = field(default_factory=list)
    js_token_selection_seed: int = 42
    divergence_estimator: str = "JSD"
    token_selection_score: str = "divergence"
    token_mask_scope: str = "both"
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 2.5
    adaptive_kl_loss_coef: bool = False
    adaptive_kl_loss_reward_key: str = "delta_opd/weighted_reward_mean"
    adaptive_kl_loss_eps: float = 0.02
    adaptive_kl_loss_min_coef: float = 0.5
    adaptive_kl_loss_max_coef: float = 2.5
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    data_loader_seed = 1
    rollout_n: int = 1  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

        if not 0.0 < self.js_top_fraction <= 1.0:
            raise ValueError(f"js_top_fraction must be in (0, 1], got {self.js_top_fraction}")
        valid_js_token_selection_modes = {
            "relative",
            "absolute",
            "top_js",
            "bottom_js",
            "ladder",
            "percentile_areas",
            "random",
        }
        if self.js_token_selection_mode not in valid_js_token_selection_modes:
            raise ValueError(
                "js_token_selection_mode must be one of "
                f"{sorted(valid_js_token_selection_modes)}, got {self.js_token_selection_mode!r}"
            )
        valid_js_token_selection_aggregations = {"batch-agg", "response-agg"}
        if self.js_token_selection_aggregation not in valid_js_token_selection_aggregations:
            raise ValueError(
                "js_token_selection_aggregation must be one of "
                f"{sorted(valid_js_token_selection_aggregations)}, "
                f"got {self.js_token_selection_aggregation!r}"
            )
        if not math.isfinite(self.js_divergence_threshold) or self.js_divergence_threshold < 0.0:
            raise ValueError(
                "js_divergence_threshold must be finite and non-negative, "
                f"got {self.js_divergence_threshold}"
            )
        if not 0.0 <= self.js_ladder_start_percent < self.js_ladder_end_percent <= 100.0:
            raise ValueError(
                "JS ladder bounds must satisfy "
                "0 <= js_ladder_start_percent < js_ladder_end_percent <= 100, "
                f"got start={self.js_ladder_start_percent}, end={self.js_ladder_end_percent}"
            )
        if self.js_token_selection_mode == "percentile_areas":
            if not self.divergence_percentile_areas:
                raise ValueError(
                    "divergence_percentile_areas must contain at least one area when "
                    "js_token_selection_mode='percentile_areas'"
                )
            for area in self.divergence_percentile_areas:
                if not isinstance(area, Sequence) or isinstance(area, (str, bytes)) or len(area) != 2:
                    raise ValueError(
                        "Each divergence_percentile_areas entry must be a [start, end] pair, "
                        f"got {area!r}"
                    )
                start_percent, end_percent = area
                try:
                    valid_area = (
                        math.isfinite(start_percent)
                        and math.isfinite(end_percent)
                        and 0.0 <= start_percent < end_percent <= 100.0
                    )
                except TypeError as exc:
                    raise TypeError(
                        "divergence_percentile_areas bounds must be finite numbers, "
                        f"got {area!r}"
                    ) from exc
                if not valid_area:
                    raise ValueError(
                        "divergence_percentile_areas must satisfy 0 <= start < end <= 100 "
                        f"with finite bounds, got {area!r}"
                    )
        if self.js_token_selection_seed < 0:
            raise ValueError(f"js_token_selection_seed must be non-negative, got {self.js_token_selection_seed}")
        valid_divergence_estimators = {"JSD", "FKL", "RKL"}
        if self.divergence_estimator not in valid_divergence_estimators:
            raise ValueError(
                "divergence_estimator must be one of "
                f"{sorted(valid_divergence_estimators)}, got {self.divergence_estimator!r}"
            )
        if self.token_selection_score not in {"divergence", "entropy", "reward_magnitude"}:
            raise ValueError(f"Invalid token_selection_score: {self.token_selection_score!r}")
        if self.token_mask_scope not in {"both", "reward_only", "kl_only"}:
            raise ValueError(f"Invalid token_mask_scope: {self.token_mask_scope!r}")
        if self.js_token_filter_enabled and self.token_mask_scope != "both" and not self.use_kl_loss:
            raise ValueError("Reward-only/KL-only masking ablations require use_kl_loss=True")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        data_loader_seed (Optional[int]): Seed for data loader. If None, uses global seed.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
