#!/usr/bin/env python3
# Copyright 2026 S2D-OPD contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""Generate a frozen token bank, score JSD resolutions, then summarize masks.

This directory is self-contained: no verl imports, training, or Ray lifecycle.
Heavy dependencies are imported only by the subcommands that need them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import prompt_bootstrap, response_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path: str) -> dict:
    root = Path(path).resolve()
    if not (root / "config.json").is_file():
        raise ValueError(f"Expected a local HF model directory: {root}. Export FSDP checkpoints first.")
    weights = sorted(list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin")))
    if not weights:
        raise ValueError(f"No HF model weights found: {root}")
    configs = {
        p.name: sha256(p) for p in sorted(root.iterdir())
        if p.is_file() and (p.suffix == ".json" or p.name.endswith(".jinja"))
    }
    return {
        "path": str(root), "config_sha256": configs,
        "weight_files": [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                         for p in weights],
    }


def provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "python": platform.python_version(),
            "script_sha256": sha256(Path(__file__)),
            "metrics_sha256": sha256(Path(__file__).with_name("metrics.py"))}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_bank(path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["bank_sha256"] != sha256(path):
        raise ValueError("Frozen rollout bank checksum differs from its manifest")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != manifest["responses"] or len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Missing or duplicate frozen rollout records")
    if any(not r["prompt_token_ids"] or not r["response_token_ids"] for r in records):
        raise ValueError("Empty prompt or response in bank")
    return records, manifest


def generate(args) -> None:
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    import vllm
    from vllm import LLM, SamplingParams

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank = output / "bank.jsonl"
    settings = {
        "student": model_identity(args.student), "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset), "seed": args.seed, "prompts": args.prompts,
        "responses_per_prompt": args.responses, "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length, "temperature": args.temperature,
        "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
        "chat_template_kwargs": json.loads(args.chat_template_kwargs),
        "tensor_parallel_size": args.tensor_parallel_size, "rollout_dtype": "bfloat16",
        "trust_remote_code": args.trust_remote_code,
    }
    if bank.exists() or bank.with_suffix(".manifest.json").exists():
        _, previous = load_bank(bank)
        if previous["settings"] != settings:
            raise ValueError("Existing bank settings differ. Use a new OUTPUT_DIR; bank will not be overwritten.")
        print(f"Reusing verified frozen bank: {bank}", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=args.trust_remote_code)
    table = pq.read_table(args.dataset, columns=["prompt"])
    rng = np.random.default_rng(args.seed)
    selected = []
    for row_index in rng.permutation(table.num_rows):
        messages = table.column("prompt")[int(row_index)].as_py()
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **settings["chat_template_kwargs"]
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if 0 < len(ids) <= args.max_prompt_length:
            selected.append((int(row_index), ids))
        if len(selected) == args.prompts:
            break
    if len(selected) != args.prompts:
        raise ValueError(f"Only {len(selected)} valid prompts available; requested {args.prompts}")
    llm = LLM(
        model=args.student, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend="mp", seed=args.seed, trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_response_length,
        enable_prefix_caching=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for _, ids in selected],
        SamplingParams(n=args.responses, temperature=args.temperature, top_p=1.0, top_k=-1,
                       max_tokens=args.max_response_length, repetition_penalty=1.0, seed=args.seed),
    )
    if len(outputs) != len(selected):
        raise ValueError("Generation returned an unexpected number of prompts")
    temporary = bank.with_suffix(".jsonl.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for (row_index, ids), result in zip(selected, outputs):
            if list(result.prompt_token_ids) != ids or len(result.outputs) != args.responses:
                raise ValueError("Generation changed prompt IDs or returned an unexpected sample count")
            for sample in result.outputs:
                response_ids = list(sample.token_ids)
                if not response_ids:
                    raise ValueError("Empty generated response")
                record = {
                    "record_id": f"row{row_index}/sample{sample.index}", "prompt_id": row_index,
                    "response_index": sample.index, "prompt_token_ids": ids, "response_token_ids": response_ids,
                    "finish_reason": sample.finish_reason, "stop_reason": sample.stop_reason,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    temporary.replace(bank)
    write_json(bank.with_suffix(".manifest.json"), {
        "schema_version": 1, "settings": settings, "responses": count, "bank_sha256": sha256(bank),
        "vllm_version": vllm.__version__, "provenance": provenance(),
        "sampled_dataset_rows": [row for row, _ in selected],
    })
    print(f"Saved frozen bank: {bank} ({count} responses)", flush=True)


def torch_scores(student_logits, teacher_logits, reference_logits, supports, teacher_temperature):
    """All resolutions share the same fp32-normalized teacher/reference distributions."""
    import torch

    teacher = (teacher_logits.float() / teacher_temperature).log_softmax(-1)
    reference = (reference_logits.float() / teacher_temperature).log_softmax(-1)
    mixture = torch.logaddexp(teacher, reference) - np.log(2.0)
    full = 0.5 * (teacher.exp() * (teacher - mixture) + reference.exp() * (reference - mixture)).sum(-1)
    ids = student_logits.topk(max(supports), dim=-1).indices
    p = teacher.gather(-1, ids).exp()
    q = reference.gather(-1, ids).exp()
    columns = []
    for k in supports:
        p_head, q_head = p[:, :k], q[:, :k]
        # Match the actor's guard against numerical selected-mass overshoot.
        p_head = p_head / p_head.sum(-1, keepdim=True).clamp_min(1)
        q_head = q_head / q_head.sum(-1, keepdim=True).clamp_min(1)
        p_dist = torch.cat([p_head, (1 - p_head.sum(-1, keepdim=True)).clamp_min(0)], -1)
        q_dist = torch.cat([q_head, (1 - q_head.sum(-1, keepdim=True)).clamp_min(0)], -1)
        mix = (p_dist + q_dist) / 2
        tiny = torch.finfo(torch.float32).tiny
        terms = p_dist * (p_dist.clamp_min(tiny).log() - mix.clamp_min(tiny).log())
        terms += q_dist * (q_dist.clamp_min(tiny).log() - mix.clamp_min(tiny).log())
        columns.append((terms.sum(-1) / 2).clamp(0, np.log(2.0)))
    columns.append(full.clamp(0, np.log(2.0)))
    values = torch.stack(columns, dim=-1)
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite JSD at valid response positions")
    return values


def response_hidden_states(model, input_ids, prompt_length: int, response_length: int):
    """Return causal states predicting response tokens, including the first one."""
    if prompt_length < 1 or response_length < 1 or input_ids.shape[1] != prompt_length + response_length:
        raise ValueError("Prompt/response lengths do not match the token sequence")
    start = prompt_length - 1
    return model.model(
        input_ids=input_ids, attention_mask=input_ids.new_ones(input_ids.shape),
        use_cache=False, return_dict=True,
    ).last_hidden_state[0, start:start + response_length]


def score(args) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records, bank_manifest = load_bank(args.bank)
    if model_identity(args.student) != bank_manifest["settings"]["student"]:
        raise ValueError("Scoring student must be exactly the frozen bank's rollout checkpoint")
    supports = sorted(set(args.supports))
    if not supports or min(supports) < 1:
        raise ValueError("Support sizes must be positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise ValueError("Scoring requires a CUDA GPU; use summarize for CPU-only analysis")
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", local_rank)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "student": model_identity(args.student), "teacher": model_identity(args.teacher),
        "reference": model_identity(args.reference), "supports": supports, "world_size": world,
        "bank_sha256": bank_manifest["bank_sha256"], "dtype": args.dtype,
        "teacher_temperature": args.teacher_temperature, "attention": args.attention,
        "chunk_positions": args.chunk_positions, "trust_remote_code": args.trust_remote_code,
        "scorer_sha256": sha256(Path(__file__)),
    }
    meta_path = output / f"scores.rank{rank:03d}.json"
    arrays_path = output / f"scores.rank{rank:03d}.npz"
    if meta_path.exists() or arrays_path.exists():
        if not meta_path.exists() or not arrays_path.exists():
            raise ValueError("Incomplete score shard. Use a fresh output directory.")
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings or previous["arrays_sha256"] != sha256(arrays_path):
            raise ValueError("Existing scoring shard differs. Use a fresh output directory.")
        print(f"Reusing verified score shard {rank}", flush=True)
        return
    tokenizers = [AutoTokenizer.from_pretrained(p, trust_remote_code=args.trust_remote_code)
                  for p in [args.student, args.teacher, args.reference]]
    if tokenizers[1].get_vocab() != tokenizers[2].get_vocab():
        raise ValueError("Teacher/reference token-to-ID mappings differ: full-vocabulary JSD is undefined")
    # The training pipeline also passes student token IDs directly to both teachers.
    teacher_vocab = tokenizers[1].get_vocab()
    student_vocab = tokenizers[0].get_vocab()
    mismatches = [(token, idx) for token, idx in student_vocab.items()
                  if token in teacher_vocab and teacher_vocab[token] != idx]
    # Match Direct-OPD's raw student-ID scoring, including its special-token
    # convention. Remapping IDs or re-tokenizing would change the fixed states.
    token_id_audit = {"student_teacher_shared_token_id_mismatches": len(mismatches),
                      "examples": mismatches[:10], "input_convention": "raw_student_token_ids"}
    dtype = getattr(torch, args.dtype)
    models = []
    for path in [args.student, args.teacher, args.reference]:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation=args.attention, trust_remote_code=args.trust_remote_code,
        ).to(device).eval()
        if model.config.model_type not in {"qwen2", "qwen3"} or not hasattr(model, "model"):
            raise ValueError("Chunked scoring is implemented for Qwen2/Qwen3 causal language models")
        models.append(model)
    vocab_sizes = [m.get_output_embeddings().weight.shape[0] for m in models]
    if len(set(vocab_sizes)) != 1 or max(supports) > vocab_sizes[0]:
        raise ValueError(f"Incompatible model output vocabularies/support sizes: {vocab_sizes}")
    local_records = records[rank::world]
    scores, offsets = [], [0]
    with torch.inference_mode():
        for index, record in enumerate(local_records):
            ids = record["prompt_token_ids"] + record["response_token_ids"]
            if min(ids) < 0 or max(ids) >= min(vocab_sizes):
                raise ValueError("Bank contains out-of-vocabulary token IDs")
            length = len(record["response_token_ids"])
            input_ids = torch.tensor([ids], device=device, dtype=torch.long)
            # Logits at prompt_length-1 predict response token 0; include the sampled EOS if present.
            hidden = [response_hidden_states(m, input_ids, len(record["prompt_token_ids"]), length) for m in models]
            chunks = []
            for position in range(0, length, args.chunk_positions):
                logits = [m.get_output_embeddings()(h[position:position + args.chunk_positions])
                          for m, h in zip(models, hidden)]
                chunks.append(torch_scores(*logits, supports, args.teacher_temperature).cpu().numpy())
                del logits
            scores.append(np.concatenate(chunks))
            offsets.append(offsets[-1] + length)
            del hidden, input_ids, chunks
            print(f"rank {rank}: scored {index + 1}/{len(local_records)} ({record['record_id']})", flush=True)
    values = np.concatenate(scores) if scores else np.empty((0, len(supports) + 1), dtype=np.float32)
    temporary = arrays_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, scores=values, offsets=np.asarray(offsets, dtype=np.int64))
    temporary.replace(arrays_path)
    write_json(meta_path, {
        "settings": settings, "records": [{"record_id": r["record_id"], "prompt_id": r["prompt_id"]}
                                            for r in local_records],
        "arrays_sha256": sha256(arrays_path), "rank": rank,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(device), "provenance": provenance(),
        "bank": str(args.bank.resolve()), "token_id_audit": token_id_audit,
    })


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args) -> None:
    metadata_paths = sorted(args.scores.glob("scores.rank*.json"))
    if not metadata_paths:
        raise ValueError("No score shards found")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    settings = shards[0]["settings"]
    world = settings["world_size"]
    if len(shards) != world or {s["rank"] for s in shards} != set(range(world)):
        raise ValueError("Missing or duplicate scoring ranks; wait for torchrun to finish")
    bank_path = getattr(args, "bank", None) or Path(shards[0]["bank"])
    bank, manifest = load_bank(bank_path)
    if manifest["bank_sha256"] != settings["bank_sha256"]:
        raise ValueError("Score shards do not match the bank")
    by_id = {}
    labels = [str(k) for k in settings["supports"]] + ["full"]
    for meta_path, shard in zip(metadata_paths, shards):
        path = meta_path.with_suffix(".npz")
        if shard["settings"] != settings or sha256(path) != shard["arrays_sha256"]:
            raise ValueError("Mixed scoring settings or corrupted score shard")
        with np.load(path, allow_pickle=False) as arrays:
            values, offsets = arrays["scores"], arrays["offsets"]
        if len(offsets) != len(shard["records"]) + 1 or offsets[0] != 0 or offsets[-1] != len(values):
            raise ValueError("Score-shard offsets do not match its records")
        if len(shard["records"]) and np.any(np.diff(offsets) <= 0):
            raise ValueError("Response offsets must be strictly increasing")
        if values.ndim != 2 or values.shape[1] != len(labels) or not np.isfinite(values).all():
            raise ValueError("Invalid JSD score matrix")
        for index, record in enumerate(shard["records"]):
            if record["record_id"] in by_id:
                raise ValueError("Duplicate scored response")
            by_id[record["record_id"]] = (record["prompt_id"], values[offsets[index]:offsets[index + 1]])
    if set(by_id) != {r["record_id"] for r in bank}:
        raise ValueError("Scoring did not cover exactly the frozen rollout bank")
    rows, all_scores, all_masks, offsets = [], [], [], [0]
    intersections = np.zeros((len(labels), len(labels)), dtype=np.int64)
    total_kept = 0
    for record in bank:
        prompt_id, scores = by_id[record["record_id"]]
        if prompt_id != record["prompt_id"] or len(scores) != len(record["response_token_ids"]):
            raise ValueError("Scored positions do not match the frozen response")
        # Nested coarse-grainings must approach the full JSD monotonically, up to fp32 rounding.
        if np.any(np.diff(scores, axis=1) < -2e-6):
            raise ValueError("JSD violates nested-support monotonicity; inspect scoring numerics")
        metrics, masks = response_metrics(scores, args.retention)
        kept = int(masks[:, -1].sum())
        for label, metric in zip(labels, metrics):
            rows.append({"record_id": record["record_id"], "prompt_id": prompt_id, "k": label,
                         "tokens": len(scores), "kept": kept, **metric})
        intersections += masks.astype(np.int64).T @ masks.astype(np.int64)
        total_kept += kept
        all_scores.append(scores)
        all_masks.append(masks)
        offsets.append(offsets[-1] + len(scores))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_response.csv", rows)
    summary = []
    for label in labels:
        selected_rows = [r for r in rows if r["k"] == label]
        result = {"k": label, "prompts": len({r['prompt_id'] for r in selected_rows}),
                  "responses": len(selected_rows), "tokens": sum(r["tokens"] for r in selected_rows)}
        for name in ["overlap", "jaccard", "random_overlap_expected", "retained_full_jsd_ratio", "spearman",
                     "mean_jsd", "mean_full_jsd", "mean_abs_error"]:
            clusters = defaultdict(list)
            for row in selected_rows:
                if np.isfinite(row[name]):
                    clusters[row["prompt_id"]].append(row[name])
            prompt_means = [float(np.mean(values)) for values in clusters.values()]
            mean, low, high, count = prompt_bootstrap(prompt_means, args.bootstrap, args.seed)
            result.update({name: mean, name + "_ci_low": low, name + "_ci_high": high,
                           name + "_defined_prompts": count})
        summary.append(result)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "pairwise_mask_overlap.csv", [
        {"k": label, **{other: float(intersections[i, j] / total_kept) for j, other in enumerate(labels)}}
        for i, label in enumerate(labels)
    ])
    with (args.output / "token_scores_and_masks.npz").open("wb") as handle:
        np.savez_compressed(handle, scores=np.concatenate(all_scores), masks=np.concatenate(all_masks),
                            offsets=np.asarray(offsets), labels=np.asarray(labels),
                            record_ids=np.asarray([r["record_id"] for r in bank]),
                            prompt_ids=np.asarray([r["prompt_id"] for r in bank]))
    write_json(args.output / "analysis_manifest.json", {
        "settings": settings, "retention": args.retention, "bootstrap_repeats": args.bootstrap,
        "bootstrap_seed": args.seed, "aggregation": "mean over responses within prompt, then mean over prompts",
        "undefined_metrics": "CSV nan; constant-score Spearman and zero-full-JSD mass ratios are undefined",
        "provenance": provenance(),
    })
    for result in summary:
        print(f"K={result['k']:>4}: overlap={result['overlap']:.4f}, "
              f"full-JSD retained={result['retained_full_jsd_ratio']:.4f}, Spearman={result['spearman']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Sample training prompts and freeze student rollouts once")
    gen.add_argument("--student", required=True)
    gen.add_argument("--dataset", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--prompts", type=int, default=128)
    gen.add_argument("--responses", type=int, default=4)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--max-prompt-length", type=int, default=1024)
    gen.add_argument("--max-response-length", type=int, default=2048)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--tensor-parallel-size", type=int, default=1)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    gen.add_argument("--chat-template-kwargs", default="{}")
    gen.add_argument("--trust-remote-code", action="store_true")
    gen.set_defaults(func=generate)
    scoring = commands.add_parser("score", help="Score the same token bank; supports torchrun data parallelism")
    for name in ["student", "teacher", "reference"]:
        scoring.add_argument("--" + name, required=True)
    scoring.add_argument("--bank", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--supports", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    scoring.add_argument("--teacher-temperature", type=float, default=1.0)
    scoring.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    scoring.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    scoring.add_argument("--chunk-positions", type=int, default=64)
    scoring.add_argument("--trust-remote-code", action="store_true")
    scoring.set_defaults(func=score)
    summary = commands.add_parser("summarize", help="CPU-only mask comparison and prompt-cluster bootstrap")
    summary.add_argument("--scores", type=Path, required=True)
    summary.add_argument("--bank", type=Path, help="Override bank location after copying results to another machine")
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--retention", type=float, default=0.10)
    summary.add_argument("--bootstrap", type=int, default=2000)
    summary.add_argument("--seed", type=int, default=0)
    summary.set_defaults(func=summarize)
    args = parser.parse_args()
    for name in ["prompts", "responses", "max_prompt_length", "max_response_length", "tensor_parallel_size",
                 "chunk_positions", "temperature", "teacher_temperature"]:
        if hasattr(args, name) and not getattr(args, name) > 0:
            parser.error(f"{name} must be positive")
    if getattr(args, "dtype", None) == "float32" and getattr(args, "attention", None) == "flash_attention_2":
        parser.error("flash_attention_2 requires --dtype bfloat16 or float16")
    args.func(args)


if __name__ == "__main__":
    main()
