# Copyright 2026 S2D-OPD contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""Small, dependency-light reference calculations for the fixed-rollout audit."""

from __future__ import annotations

import math

import numpy as np


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in nats, over the last axis (including zeros)."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or np.any(p < 0) or np.any(q < 0):
        raise ValueError("Expected matching nonnegative probability distributions")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite probabilities")
    if not np.allclose(p.sum(-1), 1) or not np.allclose(q.sum(-1), 1):
        raise ValueError("Probabilities must sum to one")
    mixture = (p + q) / 2
    tiny = np.finfo(np.float64).tiny
    terms = p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(mixture, tiny)))
    terms += q * (np.log(np.maximum(q, tiny)) - np.log(np.maximum(mixture, tiny)))
    return np.maximum(terms.sum(-1) / 2, 0)


def coarsened_jsd(p: np.ndarray, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Keep the specified candidates and collapse everything else into one tail."""
    p_head = np.take_along_axis(p, ids, axis=-1)
    q_head = np.take_along_axis(q, ids, axis=-1)
    p_dist = np.concatenate([p_head, np.maximum(1 - p_head.sum(-1, keepdims=True), 0)], axis=-1)
    q_dist = np.concatenate([q_head, np.maximum(1 - q_head.sum(-1, keepdims=True), 0)], axis=-1)
    return jsd(p_dist, q_dist)


def top_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Match the actor's per-response 90_100 ladder selection, including ties."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 < fraction <= 1:
        raise ValueError("Expected a finite score vector and fraction in (0, 1]")
    mask = np.zeros(len(scores), dtype=bool)
    if len(scores):
        keep = max(1, math.ceil(len(scores) * fraction))
        mask[np.argsort(scores, kind="stable")[-keep:]] = True
    return mask


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a, b = average_ranks(a), average_ranks(b)
    a, b = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else float("nan")


def response_metrics(scores: np.ndarray, fraction: float) -> tuple[list[dict], np.ndarray]:
    """The last column must be full-vocabulary JSD. No padded positions enter."""
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected nonempty finite [response_tokens, resolutions] scores")
    masks = np.stack([top_mask(column, fraction) for column in scores.T], axis=1)
    full, gold = scores[:, -1], masks[:, -1]
    gold_mass = full[gold].sum()
    rows = []
    for column, mask in zip(scores.T, masks.T):
        intersection = int((mask & gold).sum())
        threshold = column[mask].min()
        rows.append({
            "overlap": intersection / int(gold.sum()),
            "jaccard": intersection / int((mask | gold).sum()),
            "random_overlap_expected": int(gold.sum()) / len(gold),
            "retained_full_jsd_ratio": float(full[mask].sum() / gold_mass) if gold_mass > 0 else float("nan"),
            "spearman": spearman(column, full),
            "mean_jsd": float(column.mean()),
            "mean_full_jsd": float(full.mean()),
            "mean_abs_error": float(np.abs(full - column).mean()),
            "boundary_tie_count": int((column == threshold).sum()),
        })
    return rows, masks


def prompt_bootstrap(values: list[float], repeats: int, seed: int) -> tuple[float, float, float, int]:
    """Input contains one mean per prompt, not independent token measurements."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(values.mean())
    if len(values) < 2 or repeats < 1:
        return mean, float("nan"), float("nan"), len(values)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high), len(values)
