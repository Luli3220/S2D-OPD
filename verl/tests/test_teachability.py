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

import math

import pytest
import torch

from verl.trainer.ppo.teachability import build_teachability_selection, compute_teachability_statistics


@pytest.mark.parametrize("k", [16, 32])
def test_union_statistics_match_explicit_vocabulary_restriction(k):
    generator = torch.Generator().manual_seed(17)
    student = torch.randn(2, 3, 80, generator=generator).log_softmax(-1).requires_grad_()
    teacher = torch.randn(2, 3, 80, generator=generator).log_softmax(-1).requires_grad_()
    student_logp, student_ids = student.topk(k, dim=-1)
    teacher_logp, teacher_ids = teacher.topk(k, dim=-1)
    teacher_in_student = (teacher_ids.unsqueeze(-1) == student_ids.unsqueeze(-2)).any(-1)
    valid = torch.tensor([[1, 1, 1], [1, 1, 0]])
    disagreement, compatibility = compute_teachability_statistics(
        student_logp,
        teacher.gather(-1, student_ids),
        teacher_logp,
        student.gather(-1, teacher_ids),
        teacher_in_student,
        valid,
    )
    for b, t in torch.nonzero(valid).tolist():
        union = torch.unique(torch.cat((student_ids[b, t], teacher_ids[b, t])))
        p = teacher[b, t, union].softmax(-1)
        q = student[b, t, union].softmax(-1)
        torch.testing.assert_close(disagreement[b, t], (p * (p.log() - q.log())).sum())
        torch.testing.assert_close(compatibility[b, t], teacher[b, t, student_ids[b, t]].exp().sum())
    assert disagreement[1, 2] == compatibility[1, 2] == 0
    assert not disagreement.requires_grad and not compatibility.requires_grad


def test_coverage_uses_exact_teacher_mass_not_intersection_or_union_normalization():
    # Student top-2 = {0,1}; teacher top-2 = {1,2}. Token 1 must occur once.
    student = torch.tensor([[[0.55, 0.25, 0.15, 0.05]]]).log()
    teacher = torch.tensor([[[0.10, 0.40, 0.35, 0.15]]]).log()
    disagreement, compatibility = compute_teachability_statistics(
        student[..., [0, 1]], teacher[..., [0, 1]],
        teacher[..., [1, 2]], student[..., [1, 2]],
        torch.tensor([[[True, False]]]), torch.ones(1, 1),
    )
    p = torch.tensor([0.10, 0.40, 0.35]) / 0.85
    q = torch.tensor([0.55, 0.25, 0.15]) / 0.95
    assert compatibility.item() == pytest.approx(0.50)  # intersection-only gives 0.40
    assert disagreement.item() == pytest.approx((p * (p.log() - q.log())).sum().item())


def test_statistics_ignore_nonfinite_padding_but_reject_nonfinite_valid_positions():
    logp = torch.tensor([[[-0.5, -1.5], [float("nan"), -float("inf")]]])
    overlap = torch.ones_like(logp, dtype=torch.bool)
    disagreement, compatibility = compute_teachability_statistics(
        logp, logp, logp, logp, overlap, torch.tensor([[1, 0]])
    )
    assert torch.isfinite(disagreement).all() and torch.isfinite(compatibility).all()
    assert disagreement[0, 1] == compatibility[0, 1] == 0
    with pytest.raises(ValueError, match="Non-finite"):
        compute_teachability_statistics(logp, logp, logp, logp, overlap, torch.ones(1, 2))


def test_global_budget_can_select_no_tokens_from_a_response_and_ignores_padding():
    d = torch.tensor([[1.0, 2.0, float("nan")], [3.0, 4.0, 5.0]], requires_grad=True)
    c = torch.tensor([[0.1, 0.2, float("inf")], [0.3, 0.4, 0.5]], requires_grad=True)
    valid = torch.tensor([[1, 1, 0], [1, 1, 1]])
    scores, selected = build_teachability_selection(d, c, valid, 0.4)
    assert selected.tolist() == [[False, False, False], [False, True, True]]
    assert selected.sum().item() == math.ceil(5 * 0.4)
    assert scores[0, 2] == 0 and torch.isfinite(scores).all()
    # Q05(D)=1.2, Q95(D)=4.8; Q05(C)=0.12, Q95(C)=0.48.
    assert scores[1, 0].item() == pytest.approx(0.25)
    assert not scores.requires_grad


def test_compatibility_can_reject_the_largest_disagreement():
    d = torch.tensor([[0.0, 10.0, 9.0, 1.0]])
    c = torch.tensor([[0.0, 0.0, 1.0, 0.1]])
    _, selected = build_teachability_selection(d, c, torch.ones_like(d), 0.25)
    assert selected.tolist() == [[False, False, True, False]]


def test_constant_statistics_ties_empty_and_full_budgets():
    valid = torch.tensor([[0, 1, 1], [1, 0, 1]])
    values = torch.ones_like(valid, dtype=torch.float32)
    scores, selected = build_teachability_selection(values, values, valid, 0.25)
    assert scores.eq(0).all()
    assert selected.tolist() == [[False, True, False], [False, False, False]]
    _, full = build_teachability_selection(values, values, valid, 1.0)
    assert torch.equal(full, valid.bool())
    scores, selected = build_teachability_selection(values, values, torch.zeros_like(valid), 0.1)
    assert scores.eq(0).all() and not selected.any()


def test_noncontiguous_statistics_preserve_position_alignment():
    d = torch.arange(6, dtype=torch.float32).reshape(3, 2).T
    c = d / 10
    valid = torch.ones(3, 2, dtype=torch.bool).T
    _, selected = build_teachability_selection(d, c, valid, 0.2)
    assert selected.tolist() == [[False, False, True], [False, False, True]]


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan")])
def test_invalid_budget_is_rejected(fraction):
    with pytest.raises(ValueError, match="top_fraction"):
        build_teachability_selection(torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2), fraction)
