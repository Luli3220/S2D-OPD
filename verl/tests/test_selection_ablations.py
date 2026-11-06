# Copyright 2026 S2D-OPD contributors
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

"""CPU checks of selector signals and the real actor loss/gradient routing."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.actor import dp_actor
from verl.workers.actor.dp_actor import (
    DataParallelPPOActor,
    _build_js_ladder_token_mask,
    _compute_delta_opd_rm_scores,
    _compute_token_selection_scores,
    _get_token_selection_scores,
    _split_token_loss_masks,
)
from verl.workers.config.actor import ActorConfig


def actor_config(**kwargs):
    return ActorConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=2, ppo_mini_batch_size=2, **kwargs)


def test_reward_magnitude_does_not_cancel_and_uses_student_weights():
    # Teacher and reference swap candidates: equal/opposite rewards cancel in
    # the signed expectation, but must still produce a positive selection score.
    student = torch.tensor([[[0.5, 0.5], [0.8, 0.2]]], requires_grad=True)
    teacher = torch.tensor([[[0.4, 0.1], [0.4, 0.1]]]).log()
    reference = torch.tensor([[[0.1, 0.4], [0.1, 0.1]]]).log()
    rewards, _ = _compute_delta_opd_rm_scores(student.log(), teacher, reference)
    scores = _compute_token_selection_scores("reward_magnitude", torch.zeros(1, 2), rewards, torch.ones(1, 2))
    expected = torch.log(torch.tensor(4.0)) * torch.tensor([[1.0, 0.8]])
    torch.testing.assert_close(scores, expected)
    assert rewards[0, 0].sum() == 0
    assert not scores.requires_grad


def test_entropy_uses_full_vocabulary_tail_information_and_ignores_padding():
    # Same top-2 probabilities, different tail entropy; top-k-only entropy
    # cannot distinguish these states. The second state must rank higher.
    probabilities = torch.tensor([[[0.4, 0.3, 0.29, 0.01], [0.4, 0.3, 0.15, 0.15]]])
    entropy = -(probabilities * probabilities.log()).sum(-1)
    entropy = torch.cat((entropy, torch.tensor([[float("nan")]])), dim=1).requires_grad_()
    mask = torch.tensor([[1, 1, 0]])
    scores = _compute_token_selection_scores("entropy", torch.zeros(1, 3), torch.zeros(1, 3, 2), mask, entropy)
    selected = _build_js_ladder_token_mask(scores, mask, 90, 100)
    assert selected.tolist() == [[False, True, False]]
    assert scores[0, 2] == 0
    assert not scores.requires_grad
    with pytest.raises(ValueError, match="full-vocabulary"):
        _compute_token_selection_scores("entropy", scores, torch.zeros(1, 3, 2), mask)


@pytest.mark.parametrize("score_type", ["divergence", "entropy", "reward_magnitude"])
def test_scores_reject_nonfinite_valid_positions(score_type):
    bad = torch.tensor([[float("nan")]])
    with pytest.raises(ValueError, match="Non-finite"):
        _compute_token_selection_scores(score_type, bad, bad.unsqueeze(-1), torch.ones(1, 1), bad)


def test_non_divergence_selector_never_silently_uses_jsd():
    batch = {"js_divergence": torch.ones(1, 2)}
    assert _get_token_selection_scores(batch, "divergence") is batch["js_divergence"]
    for score_type in ["entropy", "reward_magnitude"]:
        with pytest.raises(ValueError, match="Missing token_selection_scores"):
            _get_token_selection_scores(batch, score_type)


@pytest.mark.parametrize("scope", ["both", "reward_only", "kl_only"])
def test_masks_never_select_padding_and_disabled_filter_is_dense(scope):
    valid = torch.tensor([[1, 1, 0]])
    selected = torch.tensor([[False, True, True]])
    reward, kl = _split_token_loss_masks(valid, selected, scope)
    assert reward[0, 2] == kl[0, 2] == 0
    reward, kl = _split_token_loss_masks(valid, None, scope)
    torch.testing.assert_close(reward, valid)
    torch.testing.assert_close(kl, valid)


@pytest.mark.parametrize("score_type", ["divergence", "entropy", "reward_magnitude"])
def test_reward_worker_exports_selection_scores_before_entropy_cleanup(monkeypatch, score_type):
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    actor = object.__new__(DataParallelPPOActor)
    actor.config = actor_config(token_selection_score=score_type)
    data = DataProto.from_dict(
        tensors={
            "student_top_k_log_probs": torch.tensor([[[0.8, 0.2], [0.6, 0.4]]]).log(),
            "teacher_on_student_log_probs": torch.tensor([[[0.4, 0.2], [0.3, 0.2]]]).log(),
            "teacher_ref_on_student_log_probs": torch.tensor([[[0.2, 0.2], [0.3, 0.1]]]).log(),
            "response_mask": torch.ones(1, 2),
            "entropys": torch.tensor([[1.3, 1.8]]),
        }
    )
    output = actor.compute_delta_opd_reward(data)
    if score_type == "entropy":
        expected = data.batch["entropys"].clone()
    elif score_type == "reward_magnitude":
        expected = output.batch["rm_scores"].abs().sum(-1)
    else:
        expected = output.batch["js_divergence"]
    data.batch.pop("entropys")  # The trainer later drops its diagnostic entropy tensor.
    torch.testing.assert_close(output.batch["token_selection_scores"], expected)


@pytest.mark.parametrize("scope", ["both", "reward_only", "kl_only"])
@pytest.mark.parametrize("filter_enabled", [True, False])
def test_real_actor_update_routes_and_normalizes_each_gradient(monkeypatch, scope, filter_enabled):
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    actor = object.__new__(DataParallelPPOActor)
    actor.config = actor_config(
        use_kl_loss=True,
        kl_loss_coef=0.7,
        js_token_filter_enabled=filter_enabled,
        token_mask_scope=scope,
        token_selection_score="entropy",
        js_top_fraction=0.1,
    )
    actor.actor_module = torch.nn.Module()
    reward_logits = torch.nn.Parameter(torch.full((2, 3, 2), -1.0))
    sampled_logp = torch.nn.Parameter(torch.full((2, 3), -0.5))
    actor.actor_optimizer = torch.optim.SGD([reward_logits, sampled_logp], lr=0.01)
    captured = {}

    def forward(*args, **kwargs):
        return None, sampled_logp, None, reward_logits

    def capture_step():
        captured["reward"] = reward_logits.grad.clone()
        captured["kl"] = sampled_logp.grad.clone()
        return torch.tensor(0.0)

    actor._forward_micro_batch = forward
    actor._optimizer_step = capture_step
    valid = torch.tensor([[1, 1, 1], [1, 1, 0]])
    selection = torch.tensor([[0, 1, 0], [1, 0, 0]])
    data = DataProto.from_dict(
        tensors={
            "responses": torch.zeros(2, 3, dtype=torch.long),
            "response_mask": valid,
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
            "position_ids": torch.zeros(2, 4, dtype=torch.long),
            "old_log_probs": torch.full((2, 3), -0.5),
            "advantages": torch.ones(2, 3, 2),
            "ref_log_prob": torch.full((2, 3), -0.9),
            # Deliberately opposite rankings: update must use entropy, not JSD.
            "js_divergence": torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.7, 0.0]]),
            "token_selection_scores": torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.2, 0.0]]),
        },
        meta_info={"temperature": 1.0},
    )
    metrics = actor.update_policy(data)
    if not filter_enabled:
        selection = valid
    reward_mask = valid if scope == "kl_only" else selection
    kl_mask = valid if scope == "reward_only" else selection
    expected_reward = -reward_mask.float().unsqueeze(-1).expand(2, 3, 2) / reward_mask.sum()
    # low_var_kl = exp(ref - logp) - (ref - logp) - 1
    expected_kl = 0.7 * (1 - torch.exp(torch.tensor(-0.4))) * kl_mask / kl_mask.sum()
    torch.testing.assert_close(captured["reward"], expected_reward)
    torch.testing.assert_close(captured["kl"], expected_kl)
    assert metrics["actor/reward_token_keep_ratio"][0] == pytest.approx(reward_mask.sum().item() / 5)
    assert metrics["actor/kl_token_keep_ratio"][0] == pytest.approx(kl_mask.sum().item() / 5)


@pytest.mark.parametrize("field,value", [("token_selection_score", "bad"), ("token_mask_scope", "bad")])
def test_config_rejects_unknown_options(field, value):
    with pytest.raises(ValueError, match=field):
        actor_config(**{field: value})


def test_one_term_masking_requires_an_anchor():
    with pytest.raises(ValueError, match="use_kl_loss"):
        actor_config(js_token_filter_enabled=True, token_mask_scope="reward_only")


def test_hydra_defaults_match_dataclass_defaults():
    root = Path(__file__).resolve().parents[1] / "verl/trainer/config"
    configs = [OmegaConf.load(root / "actor/actor.yaml")]
    for filename in ["_generated_ppo_trainer.yaml", "_generated_ppo_megatron_trainer.yaml"]:
        configs.append(OmegaConf.load(root / filename).actor_rollout_ref.actor)
    defaults = actor_config()
    for config in configs:
        assert config.token_selection_score == defaults.token_selection_score == "divergence"
        assert config.token_mask_scope == defaults.token_mask_scope == "both"
