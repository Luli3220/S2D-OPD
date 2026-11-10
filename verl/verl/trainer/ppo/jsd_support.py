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

"""Independent JSD scoring support; never replace reward/update candidate tensors.

This module deliberately has no model/runtime imports. The larger-support path
uses a separate scoring batch and the existing worker interfaces, preserving the
original top-k choice even when low-precision ties change topk(K) prefixes.
"""


def resolve_jsd_score_top_k(actor_config, reward_top_k: int, reward_mode: str, strategy: str) -> int:
    requested = actor_config.get("jsd_score_top_k", None)
    if requested is None:
        return reward_top_k
    if type(requested) is not int or requested <= 0:
        raise ValueError("jsd_score_top_k must be a positive integer or null")
    if not actor_config.get("js_token_filter_enabled", False):
        raise ValueError("Independent JSD scoring support requires token filtering")
    if actor_config.get("token_selection_score", "divergence") != "divergence":
        raise ValueError("Independent JSD scoring support requires token_selection_score='divergence'")
    if actor_config.get("divergence_estimator", "JSD") != "JSD":
        raise ValueError("Independent JSD support experiments require divergence_estimator='JSD'")
    if reward_mode != "delta_opd" or strategy != "only_stu" or reward_top_k <= 0:
        raise ValueError("Independent JSD scoring requires delta_opd, only_stu, and positive reward top-k")
    if actor_config.get("strategy", "fsdp") not in {"fsdp", "fsdp2"}:
        raise ValueError("Independent JSD scoring currently requires an FSDP/FSDP2 actor")
    return requested


def collect_jsd_score_log_probs(batch, actor_worker, teacher_worker, reference_worker, score_top_k: int) -> dict:
    """Three extra no-grad scoring forwards, only when scoring K > reward K.

    Keep the original batch and every original reward tensor unchanged. All state
    alignment, temperature handling and dynamic microbatch restoration are delegated
    to the same worker methods used by the ordinary Direct-OPD path.
    """
    scoring = batch.select(
        batch_keys=["responses", "response_mask", "input_ids", "attention_mask", "position_ids"],
        meta_info_keys=[],
    )
    scoring.meta_info = dict(batch.meta_info)
    scoring.meta_info["jsd_score_top_k_override"] = score_top_k
    student_output = actor_worker.compute_log_prob(scoring)
    scoring = scoring.union(student_output)
    scoring.meta_info.pop("jsd_score_top_k_override", None)
    scoring.meta_info["log_prob_top_k"] = score_top_k
    scoring.meta_info["top_k_strategy"] = "only_stu"
    # Reward workers read old_log_probs even when they only return top-k scores.
    if "entropys" in scoring.batch.keys():
        scoring.batch.pop("entropys")
    del student_output
    teacher_output = teacher_worker.compute_rm_score(scoring)
    teacher_logp = teacher_output.batch["teacher_on_student_log_probs"]
    del teacher_output
    reference_output = reference_worker.compute_rm_score(scoring)
    reference_logp = reference_output.batch["teacher_on_student_log_probs"]
    if teacher_logp.shape != reference_logp.shape or teacher_logp.shape[-1] != score_top_k:
        raise ValueError("Independent teacher/reference scoring did not return the requested support")
    return {"jsd_score_teacher_log_probs": teacher_logp, "jsd_score_reference_log_probs": reference_logp}


def select_jsd_log_probs(batch, score_top_k: int | None):
    """Return only the selector's log-probs, with reward tensors left untouched."""
    teacher = batch["teacher_on_student_log_probs"]
    reference = batch["teacher_ref_on_student_log_probs"]
    if score_top_k is None:
        return teacher, reference
    if score_top_k <= teacher.shape[-1]:
        return teacher[..., :score_top_k], reference[..., :score_top_k]
    keys = ("jsd_score_teacher_log_probs", "jsd_score_reference_log_probs")
    if any(key not in batch.keys() for key in keys):
        raise ValueError("Larger JSD scoring support is missing its independent scoring tensors")
    scored_teacher, scored_reference = (batch[key] for key in keys)
    if (
        scored_teacher.shape != scored_reference.shape
        or scored_teacher.shape[:2] != teacher.shape[:2]
        or scored_teacher.shape[-1] != score_top_k
    ):
        raise ValueError("Independent JSD log-prob shapes do not match the requested support and states")
    return scored_teacher, scored_reference
