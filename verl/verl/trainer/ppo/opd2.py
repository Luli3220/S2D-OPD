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

"""OPD2-style centered agreement gating adapted to Direct-OPD candidates.

See arXiv:2607.15161, Eqs. 6-9. This is not sampled-token OPD2 reproduction:
the existing top-k update and student-anchor KL are retained. Runtime imports
are lazy so the configuration/data-routing functions can be checked on CPU.
"""


def validate_opd2_gating(actor_config, reward_k: int, reward_mode: str, strategy: str) -> bool:
    if not actor_config.get("opd2_enabled", False):
        return False
    if reward_mode != "delta_opd" or strategy != "only_stu" or reward_k <= 0:
        raise ValueError("OPD2-style gating requires delta_opd, only_stu, and positive reward top-k")
    if actor_config.get("strategy", "fsdp") not in {"fsdp", "fsdp2"}:
        raise ValueError("OPD2-style gating currently requires FSDP/FSDP2")
    if actor_config.get("js_token_filter_enabled", False):
        raise ValueError("OPD2-style gating must not be combined with position filtering/top-10%")
    if actor_config.get("jsd_score_top_k", None) is not None:
        raise ValueError("OPD2-style gating must not be combined with independent JSD scoring")
    if actor_config.get("token_selection_score", "divergence") != "divergence":
        raise ValueError("OPD2-style gating must not be combined with other selector modes")
    center_k = actor_config.get("opd2_center_top_k", 1024)
    if type(center_k) is not int or center_k < reward_k:
        raise ValueError("opd2_center_top_k must be an integer at least as large as reward top-k")
    return True


def force_include_sampled_token(ids, log_probs, responses, sampled_log_probs):
    """Official centering support convention: replace the last entry on a miss."""
    import torch

    missing = ~(ids == responses.unsqueeze(-1)).any(dim=-1)
    ids, log_probs = ids.clone(), log_probs.clone()
    ids[..., -1] = torch.where(missing, responses, ids[..., -1])
    log_probs[..., -1] = torch.where(missing, sampled_log_probs, log_probs[..., -1])
    return ids, log_probs


def student_weighted_expectation(student_logp, target_logp, response_mask, chunk_positions: int = 32):
    """Truncated E_student[target] with TRUE student probabilities, not top-k softmax.

    Accumulate in fp32 in bounded position chunks. Never upcast the complete
    [B,T,1024] bank at once. Padding does not participate in any expectation.
    """
    import torch

    if (
        student_logp.ndim != 3
        or (target_logp is not None and student_logp.shape != target_logp.shape)
        or student_logp.shape[:2] != response_mask.shape
    ):
        raise ValueError("Centering tensors must align as [B,T,K] and response_mask as [B,T]")
    output = torch.zeros(student_logp.shape[:2], device=student_logp.device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, student_logp.shape[1], chunk_positions):
            end = start + chunk_positions
            valid = response_mask[:, start:end].to(student_logp.device).bool().unsqueeze(-1)
            s = student_logp[:, start:end].detach().float().masked_fill(~valid, 0)
            target = (target_logp[:, start:end].detach().to(s.device).float().masked_fill(~valid, 0)
                      if target_logp is not None else 1.0)
            if not torch.isfinite(s).all() or (target_logp is not None and not torch.isfinite(target).all()):
                raise ValueError("Non-finite centering log probabilities at valid positions")
            probabilities = s.exp().masked_fill(~valid, 0)
            output[:, start:end] = (probabilities * target).sum(-1)
    return output


def collect_opd2_centering(batch, actor_worker, teacher_worker, reference_worker, center_k: int) -> dict:
    """Score an independent top-1024 support; keep all Direct-OPD tensors intact."""
    scoring = batch.select(
        batch_keys=["responses", "response_mask", "input_ids", "attention_mask", "position_ids"],
        meta_info_keys=[],
    )
    scoring.meta_info = dict(batch.meta_info)
    scoring.meta_info["opd2_center_top_k_override"] = center_k
    student_output = actor_worker.compute_log_prob(scoring)
    scoring = scoring.union(student_output)
    scoring.meta_info.pop("opd2_center_top_k_override", None)
    # Existing reward-worker K=0 + supplied candidate IDs gathers cross log-probs
    # directly, without allocating an unnecessary 1024x1024 top-k overlap matrix.
    scoring.meta_info["log_prob_top_k"] = 0
    scoring.meta_info["top_k_strategy"] = "only_stu"
    if "entropys" in scoring.batch.keys():
        scoring.batch.pop("entropys")
    del student_output
    student = scoring.batch["student_top_k_log_probs"]
    valid = scoring.batch["response_mask"]
    mean_student = student_weighted_expectation(student, student, valid)
    teacher_output = teacher_worker.compute_rm_score(scoring)
    mean_teacher = student_weighted_expectation(student, teacher_output.batch["teacher_on_student_log_probs"], valid)
    del teacher_output
    reference_output = reference_worker.compute_rm_score(scoring)
    mean_reference = student_weighted_expectation(
        student, reference_output.batch["teacher_on_student_log_probs"], valid
    )
    # E[1] documents the actual captured probability mass, including bf16 roundoff.
    mass = student_weighted_expectation(student, None, valid)
    return {"opd2_delta_baseline": mean_teacher - mean_reference,
            "opd2_opd_baseline": mean_teacher - mean_student,
            "opd2_center_mass": mass}


def compute_opd2_candidate_gate(student_logp, teacher_logp, reference_logp,
                               delta_baseline, opd_baseline, response_mask):
    """Use centered signals only to decide the gate; do not construct rewards.

    Uses the paper's strict product > 0 condition; a zero OPD advantage stops the
    delta update. The caller masks its original Direct-OPD reward tensor, keeping
    every retained value/dtype unchanged. No ranking or retained-count reduction.
    """
    import torch

    if student_logp.ndim != 3 or student_logp.shape != teacher_logp.shape or student_logp.shape != reference_logp.shape:
        raise ValueError("OPD2 reward tensors must have matching [B,T,K] shapes")
    if (
        delta_baseline.shape != response_mask.shape or opd_baseline.shape != response_mask.shape
        or student_logp.shape[:2] != response_mask.shape
    ):
        raise ValueError("OPD2 baselines must align with valid response positions")
    with torch.no_grad():
        valid = response_mask.to(student_logp.device).bool().unsqueeze(-1)
        student, teacher, reference = [x.detach().to(student_logp.device).float().masked_fill(~valid, 0)
                                       for x in (student_logp, teacher_logp, reference_logp)]
        delta_mean = delta_baseline.detach().to(student.device).float().masked_fill(~valid.squeeze(-1), 0)
        opd_mean = opd_baseline.detach().to(student.device).float().masked_fill(~valid.squeeze(-1), 0)
        if not all(torch.isfinite(x).all() for x in (student, teacher, reference, delta_mean, opd_mean)):
            raise ValueError("Non-finite OPD2 signals at valid response positions")
        delta = (teacher - reference) - delta_mean.unsqueeze(-1)
        opd = (teacher - student) - opd_mean.unsqueeze(-1)
        gate = ((delta * opd) > 0) & valid
    return gate.detach()
