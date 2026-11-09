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

"""TA selection (arXiv:2605.26844, Eqs. 9-13) for the Direct-OPD objective.

Compute raw statistics on workers, then normalize and select on the complete
rollout batch in the trainer. The selector never changes reward candidates.
"""

import math

import torch


@torch.no_grad()
def compute_teachability_statistics(
    student_logp: torch.Tensor,
    teacher_on_student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    student_on_teacher_logp: torch.Tensor,
    teacher_in_student_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return forward KL on the deduplicated union and exact teacher coverage.

    All log probabilities are normalized over the full vocabulary. The first
    pair is on student top-k IDs, the second on RL-teacher top-k IDs. Only KL
    renormalizes on the union; coverage retains the original teacher mass.
    """
    if student_logp.ndim != 3 or student_logp.shape[-1] == 0:
        raise ValueError("TA selection requires nonempty [B, T, K] student log probabilities")
    if response_mask.shape != student_logp.shape[:2]:
        raise ValueError("TA response_mask must match the [B, T] log-probability dimensions")
    if teacher_on_student_logp.shape != student_logp.shape:
        raise ValueError("Teacher-on-student probabilities must match student top-k probabilities")
    if (
        teacher_logp.ndim != 3
        or teacher_logp.shape[:2] != student_logp.shape[:2]
        or teacher_logp.shape[-1] == 0
        or student_on_teacher_logp.shape != teacher_logp.shape
        or teacher_in_student_mask.shape != teacher_logp.shape
    ):
        raise ValueError("TA teacher support, cross-scores and overlap mask must have matching [B, T, K] shapes")

    valid = response_mask.to(device=student_logp.device, dtype=torch.bool)
    logps = []
    for values in (student_logp, teacher_on_student_logp, teacher_logp, student_on_teacher_logp):
        values = values.detach().to(device=student_logp.device, dtype=torch.float32)
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Non-finite TA log probabilities at valid response positions")
        # Padding can contain arbitrary placeholders, including NaN/-inf.
        logps.append(values.masked_fill(~valid.unsqueeze(-1), 0.0))
    student_logp, teacher_on_student_logp, teacher_logp, student_on_teacher_logp = logps

    union_valid = torch.cat(
        (
            torch.ones_like(student_logp, dtype=torch.bool),
            ~teacher_in_student_mask.to(device=student_logp.device, dtype=torch.bool),
        ),
        dim=-1,
    )
    student_union = torch.cat((student_logp, student_on_teacher_logp), dim=-1)
    teacher_union = torch.cat((teacher_on_student_logp, teacher_logp), dim=-1)
    student_union = student_union.masked_fill(~union_valid, -torch.inf)
    teacher_union = teacher_union.masked_fill(~union_valid, -torch.inf)
    student_union = (student_union - torch.logsumexp(student_union, dim=-1, keepdim=True)).masked_fill(
        ~union_valid, 0.0
    )
    teacher_union = (teacher_union - torch.logsumexp(teacher_union, dim=-1, keepdim=True)).masked_fill(
        ~union_valid, 0.0
    )
    teacher_weights = teacher_union.exp().masked_fill(~union_valid, 0.0)
    disagreement = (teacher_weights * (teacher_union - student_union)).sum(dim=-1).clamp_min(0.0)
    compatibility = teacher_on_student_logp.exp().sum(dim=-1).clamp(0.0, 1.0)
    return disagreement.masked_fill(~valid, 0.0), compatibility.masked_fill(~valid, 0.0)


@torch.no_grad()
def build_teachability_selection(
    disagreement: torch.Tensor,
    compatibility: torch.Tensor,
    response_mask: torch.Tensor,
    top_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize over all valid rollout positions and retain exactly ceil(rho*N).

    This must run once on the gathered rollout batch, before actor dispatch.
    Ties keep the earlier flattened position, making degenerate scores stable.
    Empty batches select nothing; constant statistics normalize to zero.
    """
    if (
        disagreement.ndim != 2
        or disagreement.shape != compatibility.shape
        or disagreement.shape != response_mask.shape
    ):
        raise ValueError("TA statistics and response_mask must have the same [B, T] shape")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("TA top_fraction must be in (0, 1]")
    valid = response_mask.to(device=disagreement.device, dtype=torch.bool)
    disagreement = disagreement.detach().float()
    compatibility = compatibility.detach().to(device=disagreement.device, dtype=torch.float32)
    scores = torch.zeros_like(disagreement)
    selected = torch.zeros_like(valid)
    if not valid.any():
        return scores, selected

    def normalize(values: torch.Tensor) -> torch.Tensor:
        values = values[valid]
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite TA statistics at valid response positions")
        quantiles = torch.quantile(values, values.new_tensor([0.05, 0.95]))
        return ((values - quantiles[0]) / (quantiles[1] - quantiles[0] + 1e-8)).clamp(0.0, 1.0)

    valid_scores = normalize(disagreement) * normalize(compatibility)
    scores[valid] = valid_scores
    valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(-1)
    keep_count = math.ceil(top_fraction * valid_indices.numel())
    ranked = torch.argsort(valid_scores, descending=True, stable=True)
    flat_selected = torch.zeros(valid.numel(), device=valid.device, dtype=torch.bool)
    flat_selected[valid_indices[ranked[:keep_count]]] = True
    return scores, flat_selected.reshape(valid.shape)
