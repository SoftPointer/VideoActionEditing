#!/usr/bin/env python3
"""Materialize detached block-22 ``Phi_v1`` action representations on SP4.

Generated media are authenticated authoring evidence only.  This executable
reopens the producer-time clean latent and Gaussian, runs a frozen Bernini
prompt/no-op pair at exact40 index 29, observes block 22 without replacing its
output, reconstructs the native global target order, and saves only detached
FP32 ``[21,32]`` codes.  It has no optimizer and never emits an editor target.

Official sidecars require ten-branch independent reviews sealed before this
run.  ``--allow-unreviewed-technical-only`` may be used to debug engineering,
but every resulting receipt is permanently marked non-admissible and must be
re-extracted after reviews; the manifest builder rejects it.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import generic_action_manifest_v1 as manifests  # noqa: E402


PLAN_SCHEMA = "bernini-phi-v1-sidecar-sp4-plan-v1"
GAP_SCHEMA = "bernini-phi-v1-sidecar-gap-receipt-v1"
RUN_SCHEMA = "bernini-phi-v1-sidecar-sp4-run-receipt-v1"
SELECTED_BRANCHES = ("action", "noop", "reverse", "incomplete")
FORWARD_BRANCHES = ("action", "reverse", "incomplete", "camera_only", "appearance_only")
ALL_BRANCHES = (
    "action", "noop", "incomplete", "reverse", "shuffle", "wrong_actor",
    "wrong_object", "camera_only", "appearance_only", "generic_wrong_motion",
)
SCHEDULE_INDEX = 29
PHI_BLOCK_INDEX = 22
SIGMA_COORDINATES = None  # authenticated from the frozen native runtime


class PhiV1MaterializationError(RuntimeError):
    """A plan, review, artifact, distributed layout, or code failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PhiV1MaterializationError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhiV1MaterializationError(message)


def _plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PhiV1MaterializationError(f"{label} unavailable: {path}") from error
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{label} must be a plain file")
    return path.resolve(strict=True)


def _plain_dir(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PhiV1MaterializationError(f"{label} unavailable: {path}") from error
    _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{label} must be a plain directory")
    return path.resolve(strict=True)


def _read_json(path: str | Path, label: str, expected_sha256: Optional[str] = None) -> tuple[dict[str, Any], Path, str]:
    source = _plain_file(path, label)
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require(observed == expected_sha256, f"{label} SHA-256 differs")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise PhiV1MaterializationError(f"cannot decode {label}") from error
    _require(type(value) is dict, f"{label} root must be an object")
    return value, source, observed


def _verify_object_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    declared = value.get(field)
    _require(type(declared) is str and len(declared) == 64, f"{label} digest differs")
    unsigned = dict(value)
    del unsigned[field]
    _require(object_sha256(unsigned) == declared, f"{label} digest differs")


def _write_create_only(path: Path, value: Mapping[str, Any], mode: int = 0o400) -> str:
    _require(path.is_absolute() and path.parent.is_dir(), f"output parent unavailable: {path}")
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    os.chmod(path, mode)
    return hashlib.sha256(raw).hexdigest()


def _candidate_receipt(path: Path) -> dict[str, Any]:
    value, _, observed = _read_json(path, "candidate generation receipt")
    _verify_object_seal(value, "receipt_digest", "candidate generation receipt")
    _require(value.get("schema_version") == "pair-v5-frozen-bernini-t2v-calibration-receipt-v1", "candidate receipt schema differs")
    candidate = value.get("candidate", {})
    artifacts = value.get("artifacts", {})
    _require(candidate.get("semantic_branch") in ALL_BRANCHES, "candidate branch differs")
    _require(candidate.get("full_t2v_caption_utf8_sha256") == hashlib.sha256(candidate.get("full_t2v_caption", "").encode("utf-8")).hexdigest(), "candidate caption SHA differs")
    _require(artifacts.get("mp4", {}).get("frame_count") == 81 and artifacts.get("mp4", {}).get("fps") == 25, "candidate media is not exact81/25fps")
    for field in ("pseudo_target", "student_initial_noise", "student_input", "training_donor"):
        _require(value.get("artifact_use_contract", {}).get(field) is False, f"candidate {field} authority differs")
    for artifact_name in ("mp4", "predecode_clean_latent", "official_initial_gaussian"):
        artifact = artifacts[artifact_name]
        artifact_path = _plain_file(artifact["path"], f"candidate {artifact_name}")
        _require(file_sha256(artifact_path) == artifact["sha256"], f"candidate {artifact_name} SHA differs")
    return {**value, "_file_path": str(path), "_file_sha256": observed}


def _expected_cells(authoring: Mapping[str, Any], population: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    profiles = {row["profile_id"]: row for row in population["inherited_bank_profiles"]}
    inherited: dict[str, Mapping[str, Any]] = {}
    for family in population["action_families"]:
        for row in family["inherited_identity_scenes"]:
            inherited[row["source_iid"]] = row
    authoring_by_iid = {row["iid"]: row for row in authoring["cells"]}
    cells: list[dict[str, Any]] = []
    for iid in sorted(iid for iid, row in authoring_by_iid.items() if row["analysis_split"] == split):
        row = inherited[iid]
        profile = profiles[row["source_bank_profile"]]
        for index, seed in enumerate(row["seeds"]):
            prefix = profile["seed1_candidate_prefix" if index == 0 else "seed2_candidate_prefix"]
            cells.append(
                {
                    "source_iid": iid,
                    "analysis_split": split,
                    "seed": seed,
                    "candidate_ids": {branch: f"{prefix}{iid}-{branch}" for branch in ALL_BRANCHES},
                }
            )
    return cells


def build_plan(
    *, authoring_path: str | Path, population_path: str | Path, split: str,
    generation_roots: Sequence[str | Path], review_root: Optional[str | Path],
    output: str | Path, gap_output: str | Path, allow_unreviewed_technical_only: bool,
) -> dict[str, Any]:
    authoring, _, _ = _read_json(authoring_path, "authoring registry", manifests.AUTHORING_SHA256)
    population, _, _ = _read_json(population_path, "population registry", manifests.POPULATION_SHA256)
    roots = [_plain_dir(path, "generation root") for path in generation_roots]
    generation: dict[str, Path] = {}
    for root in roots:
        for receipt in root.rglob("pair-v5-t2v-calibration-receipt.json"):
            candidate_id = receipt.parent.name
            if candidate_id in generation and generation[candidate_id] != receipt:
                raise PhiV1MaterializationError(f"duplicate candidate receipt: {candidate_id}")
            generation[candidate_id] = receipt.resolve(strict=True)
    reviews: dict[str, tuple[Path, str, Mapping[str, Any]]] = {}
    if review_root is not None:
        for path in _plain_dir(review_root, "review root").rglob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if value.get("schema_version") == manifests.REVIEW_SCHEMA:
                candidate_id = value.get("candidate_id")
                _require(candidate_id not in reviews, f"duplicate review receipt: {candidate_id}")
                review_sha = file_sha256(path)
                validated = manifests.validate_review_receipt(path.resolve(strict=True), review_sha)
                reviews[candidate_id] = (path.resolve(strict=True), review_sha, validated)
    cells = _expected_cells(authoring, population, split)
    missing_media: list[str] = []
    missing_reviews: list[str] = []
    plan_rows: list[dict[str, Any]] = []
    for cell in cells:
        candidate_ids = cell["candidate_ids"]
        media_refs: dict[str, dict[str, str]] = {}
        review_refs: dict[str, Optional[dict[str, str]]] = {}
        for branch in ALL_BRANCHES:
            candidate_id = candidate_ids[branch]
            receipt = generation.get(candidate_id)
            if receipt is None:
                missing_media.append(candidate_id)
            else:
                media_refs[branch] = {"path": str(receipt), "file_sha256": file_sha256(receipt)}
            review = reviews.get(candidate_id)
            if review is None:
                missing_reviews.append(candidate_id)
                review_refs[branch] = None
            else:
                _require(review[2]["candidate_id"] == candidate_id and review[2]["branch"] == branch, "review candidate/branch differs")
                review_refs[branch] = {"path": str(review[0]), "file_sha256": review[1]}
        if not any(candidate_ids[branch] in missing_media for branch in set(FORWARD_BRANCHES) | {"noop"}):
            plan_rows.append({**cell, "generation_receipts": media_refs, "review_receipts": review_refs})
    gap = {
        "schema_version": GAP_SCHEMA,
        "split": split,
        "expected_seed_cells": 8,
        "complete_six_branch_seed_cells": len(plan_rows),
        "missing_generation_candidate_ids": sorted(missing_media),
        "missing_review_candidate_ids": sorted(missing_reviews),
        "block22_sidecar_count_before_run": 0,
        "generated_media_is_optimizer_input_or_target": False,
        "optimizer_authorized": False,
    }
    gap = {**gap, "receipt_digest": object_sha256(gap)}
    _write_create_only(Path(gap_output), gap)
    _require(not missing_media and len(plan_rows) == 8, "generation media closure is incomplete; gap receipt written")
    official_review = not missing_reviews
    _require(official_review or allow_unreviewed_technical_only, "reviews are incomplete; only explicit technical-only extraction is allowed")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": f"generic-action-phi-v1-{split}-r1",
        "analysis_split": split,
        "mode": "OFFICIAL_REVIEWED" if official_review else "UNREVIEWED_TECHNICAL_ONLY",
        "expected_seed_cells": 8,
        "model_forwards": 82,
        "forward_accounting": "8_cells_x_5_prompt_pairs_x_2_plus_2_hook_parity",
        "phi_v1": {
            "block_index": PHI_BLOCK_INDEX,
            "teacher_exact40_index": SCHEDULE_INDEX,
            "p32_seed": manifests.P32_SEED,
            "nuisance_order": ["camera_only", "appearance_only_gram_schmidt_off_camera"],
        },
        "generated_media_is_optimizer_input_or_target": False,
        "rows": plan_rows,
    }
    plan = {**plan, "plan_digest": object_sha256(plan)}
    _write_create_only(Path(output), plan)
    return plan


class Block22SP4PairCapture:
    """Observe two target-only forwards and all-gather global phase-major rows."""

    def __init__(self, transformer: Any, *, latent_shape: Sequence[int], rank: int):
        import torch

        shape = tuple(int(item) for item in latent_shape)
        _require(len(shape) == 5 and shape[:3] == (1, 16, 21), "latent shape differs")
        _require(shape[3] % 2 == shape[4] % 2 == 0, "latent patch geometry differs")
        self.patch_height = shape[3] // 2
        self.patch_width = shape[4] // 2
        self.patch_positions = self.patch_height * self.patch_width
        self.global_tokens = 21 * self.patch_positions
        self.local_tokens = math.ceil(self.global_tokens / 4)
        self.rank = rank
        blocks = getattr(transformer, "blocks", None)
        _require(blocks is not None and len(blocks) == 30, "transformer block closure differs")
        _require(callable(getattr(blocks[PHI_BLOCK_INDEX], "register_forward_hook", None)), "block22 is not hookable")
        self._captures: list[Any] = []
        self._active = False
        self._handle = blocks[PHI_BLOCK_INDEX].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        import torch

        if not self._active:
            return None
        candidate = output[0] if isinstance(output, tuple) else output
        _require(isinstance(candidate, torch.Tensor), "block22 output is not tensor")
        _require(tuple(candidate.shape) == (1, self.local_tokens, manifests.HIDDEN_WIDTH), "block22 local target geometry differs")
        _require(not candidate.requires_grad and bool(torch.isfinite(candidate).all().item()), "block22 output is not detached finite")
        _require(len(self._captures) < 2, "block22 fired too many times")
        self._captures.append(candidate.detach().float().contiguous())
        return None

    def begin(self) -> None:
        _require(not self._active and not self._captures, "capture state differs")
        self._active = True

    def finish(self, dist: Any) -> tuple[Any, Any, list[dict[str, int]]]:
        import torch

        _require(self._active and len(self._captures) == 2, "capture pair closure differs")
        global_values = []
        for local in self._captures:
            gathered = [torch.empty_like(local) for _ in range(4)]
            dist.all_gather(gathered, local)
            merged = torch.cat(gathered, dim=1)
            _require(int(merged.shape[1]) == 4 * self.local_tokens, "SP4 gather length differs")
            global_values.append(merged[:, : self.global_tokens, :].contiguous())
        layouts = [
            {
                "rank": rank,
                "start": rank * self.local_tokens,
                "stop": min((rank + 1) * self.local_tokens, self.global_tokens),
                "selected": max(min(self.local_tokens, self.global_tokens - rank * self.local_tokens), 0),
                "append_padding_removed": max((rank + 1) * self.local_tokens - self.global_tokens, 0) if rank == 3 else 0,
            }
            for rank in range(4)
        ]
        _require(sum(row["selected"] for row in layouts) == self.global_tokens, "SP4 target coverage differs")
        self._captures = []
        self._active = False
        return global_values[0], global_values[1], layouts

    def close(self) -> None:
        _require(not self._active, "cannot close active capture")
        self._handle.remove()


def _raw_tensor_sha(value: Any) -> str:
    import torch

    cpu = value.detach().float().cpu().contiguous()
    _require(isinstance(cpu, torch.Tensor) and bool(torch.isfinite(cpu).all().item()), "tensor digest input differs")
    raw = struct.pack(f"<{cpu.numel()}f", *cpu.reshape(-1).tolist())
    return hashlib.sha256(raw).hexdigest()


def _save_f32le(path: Path, value: Any) -> str:
    import torch

    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    cpu = value.detach().float().cpu().contiguous()
    _require(isinstance(cpu, torch.Tensor) and bool(torch.isfinite(cpu).all().item()), "save tensor differs")
    raw = struct.pack(f"<{cpu.numel()}f", *cpu.reshape(-1).tolist())
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def _gram_schmidt_project(raw: Any, camera: Any, appearance: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    vector = raw.float().reshape(-1)
    camera_vector = camera.float().reshape(-1)
    appearance_vector = appearance.float().reshape(-1)
    camera_norm = camera_vector.norm()
    _require(bool(torch.isfinite(camera_norm).item()) and float(camera_norm.item()) > 1.0e-8, "camera nuisance is degenerate")
    camera_unit = camera_vector / camera_norm
    appearance_orthogonal = appearance_vector - (appearance_vector @ camera_unit) * camera_unit
    appearance_norm = appearance_orthogonal.norm()
    _require(bool(torch.isfinite(appearance_norm).item()) and float(appearance_norm.item()) > 1.0e-8, "appearance nuisance degenerates after Gram-Schmidt")
    appearance_unit = appearance_orthogonal / appearance_norm
    projected = vector - (vector @ camera_unit) * camera_unit
    projected = projected - (projected @ appearance_unit) * appearance_unit
    pre_norm = vector.norm()
    post_norm = projected.norm()
    _require(bool(torch.isfinite(projected).all().item()) and float(pre_norm.item()) > 1.0e-8 and float(post_norm.item()) > 1.0e-8, "action quotient degenerates under nuisance projection")
    normalized = projected.reshape_as(raw).float().contiguous()
    normalized[:, 0, :] = 0.0
    temporal_mean = normalized[:, 1:, :].mean(dim=1)
    _require(float(temporal_mean.abs().max().item()) <= 2.0e-5, "nuisance projection left the registered temporal-DC subspace")
    final_norm = normalized.reshape(-1).norm()
    _require(float(final_norm.item()) > 1.0e-8, "post-DC quotient degenerates")
    normalized = (normalized / final_norm).float().contiguous()
    normalized[:, 0, :] = 0.0
    survival = float((vector @ projected / (pre_norm * post_norm)).item())
    _require(math.isfinite(survival), "nuisance survival cosine is non-finite")
    return normalized, {
        "camera_raw_sha256": _raw_tensor_sha(camera),
        "appearance_raw_sha256": _raw_tensor_sha(appearance),
        "camera_norm": float(camera_norm.item()),
        "appearance_after_gs_norm": float(appearance_norm.item()),
        "pre_projection_norm": float(pre_norm.item()),
        "post_projection_norm": float(post_norm.item()),
        "survival_cosine": survival,
        "finite_non_degenerate": True,
    }


def _runtime_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-unreviewed-technical-only", action="store_true")


def run_sp4(args: argparse.Namespace) -> int:
    plan, plan_path, plan_sha = _read_json(args.plan, "Phi_v1 plan", args.expected_plan_sha256)
    _verify_object_seal(plan, "plan_digest", "Phi_v1 plan")
    _require(plan.get("schema_version") == PLAN_SCHEMA and plan.get("expected_seed_cells") == 8 and len(plan.get("rows", [])) == 8, "Phi_v1 plan closure differs")
    official = plan.get("mode") == "OFFICIAL_REVIEWED"
    _require(official or (plan.get("mode") == "UNREVIEWED_TECHNICAL_ONLY" and args.allow_unreviewed_technical_only), "unreviewed plan is not authorized even for technical extraction")

    import torch
    import torch.distributed as dist
    import temporal_counterfactual_action_scorer_v1 as frozen_pair
    import materialize_latent_temporal_event_critic_core4 as legacy_materializer

    frozen = frozen_pair._frozen_d541801_runtime()
    frozen_pair.validate_native_coordinate_runtime(frozen)
    coordinate = next((row for row in frozen_pair.contract.NATIVE_SIGMA_COORDINATES if row[0] == SCHEDULE_INDEX), None)
    _require(coordinate is not None, "exact40 index29 is unavailable")
    sigma_value = float(coordinate[1])
    legacy = frozen.native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.trainer.validate_source_trees(
            args.bernini_root, args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise PhiV1MaterializationError(str(error)) from error
    _require(transformer_config.get("num_attention_heads") == 12, "checkpoint attention heads differ")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    import generic_source_anchored_action_v1 as core

    _require(torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None, "SP4 Phi_v1 requires AUH ROCm GPUs")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "-1"))
    _require(world == 4 and 0 <= rank < 4 and 0 <= local_rank < 4, "WORLD4 rank environment differs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120), rank=rank, world_size=world)
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", local_rank)
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    _require(transformer is not None and diffusion.transformer_2 is None and not any(parameter.requires_grad for parameter in renderer.parameters()), "frozen renderer closure differs")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs())
    checkpoint_results: list[Any] = [None]
    if rank == 0:
        try:
            checkpoint_results[0] = {
                "ok": True,
                "identity": frozen.native_generation.source_audit.validate_checkpoint_content(
                    checkpoint, Path(args.checkpoint_content_manifest)
                ),
            }
        except Exception as error:
            checkpoint_results[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(checkpoint_results, src=0)
    _require(isinstance(checkpoint_results[0], Mapping) and checkpoint_results[0].get("ok") is True, f"checkpoint content audit failed: {checkpoint_results[0]}")
    output = Path(args.output_dir)
    if rank == 0:
        _require(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink(), "output must be a fresh absolute directory")
        output.mkdir()
    dist.barrier()

    p32 = core.fixed_p32().to(device=device)
    p32_digests: list[Any] = [None] * 4
    dist.all_gather_object(p32_digests, _raw_tensor_sha(p32))
    _require(len(set(p32_digests)) == 1, "P32 bytes differ across ranks")
    p32_path = output / "phi_v1_p32.f32le"
    if rank == 0:
        p32_sha = _save_f32le(p32_path, p32)
    else:
        p32_sha = p32_digests[0]
    dist.barrier()
    _require(file_sha256(p32_path) == p32_sha == p32_digests[0], "persisted P32 differs")
    core_path = Path(core.__file__).resolve(strict=True)
    forward_count = 0
    receipt_bindings: list[dict[str, Any]] = []
    hook_parity_done = False
    for cell in plan["rows"]:
        all_receipts = {branch: _candidate_receipt(Path(cell["generation_receipts"][branch]["path"])) for branch in ALL_BRANCHES}
        receipts = {branch: all_receipts[branch] for branch in set(FORWARD_BRANCHES) | {"noop"}}
        if official:
            for branch in ALL_BRANCHES:
                review_ref = cell["review_receipts"].get(branch)
                _require(isinstance(review_ref, Mapping), f"official review missing for {branch}")
                review = manifests.validate_review_receipt(review_ref["path"], review_ref["file_sha256"])
                generation_receipt = all_receipts[branch]
                _require(review["candidate_id"] == generation_receipt["candidate"]["candidate_id"] and review["branch"] == branch, "review/generation identity differs")
                _require(review["media_sha256"] == generation_receipt["artifacts"]["mp4"]["sha256"], "review/generation media SHA differs")
        for branch, receipt in receipts.items():
            ref = cell["generation_receipts"][branch]
            _require(receipt["_file_sha256"] == ref["file_sha256"], "plan/candidate receipt SHA differs")
            candidate = receipt["candidate"]
            _require(candidate["candidate_id"] == cell["candidate_ids"][branch] and candidate["semantic_branch"] == branch and candidate["analysis_split"] == cell["analysis_split"] and candidate["seed"] == cell["seed"], "candidate cell binding differs")
        gaussian_artifacts = [receipt["artifacts"]["official_initial_gaussian"] for receipt in receipts.values()]
        gaussian = frozen._load_exact81_tensor(gaussian_artifacts[0], key="official_initial_gaussian", label="official Gaussian")
        gaussian_identity = frozen.verify_native_tensor_value_identity(gaussian, gaussian_artifacts[0], label="official Gaussian")
        for artifact in gaussian_artifacts[1:]:
            other = frozen._load_exact81_tensor(artifact, key="official_initial_gaussian", label="same-cell Gaussian")
            identity = frozen.verify_native_tensor_value_identity(other, artifact, label="same-cell Gaussian")
            _require(identity == gaussian_identity and torch.equal(other, gaussian), "same-cell Gaussian values differ")
        epsilon = gaussian.to(device=device).contiguous()
        noop_candidate = receipts["noop"]["candidate"]
        noop_prompt = frozen.native_generation.build_task_prompt("t2v", noop_candidate["full_t2v_caption"], prompt_cleaner=prompt_clean)
        raw_codes: dict[str, Any] = {}
        layouts: Optional[list[dict[str, int]]] = None
        for branch in FORWARD_BRANCHES:
            receipt = receipts[branch]
            clean_cpu = frozen._load_exact81_tensor(receipt["artifacts"]["predecode_clean_latent"], key="normalized_clean_latent", label=f"{branch} clean latent")
            legacy_materializer.verify_authenticated_native_clean_tensor_identity(clean_cpu, receipt["artifacts"]["predecode_clean_latent"], label=f"{branch} clean latent", frozen=frozen)
            _require(tuple(clean_cpu.shape) == tuple(gaussian.shape), "clean/Gaussian shape differs")
            clean = clean_cpu.to(device=device).contiguous()
            sigma = torch.tensor([sigma_value], dtype=torch.float32, device=device)
            x_sigma = ((1.0 - sigma.reshape(1, 1, 1, 1, 1)) * clean + sigma.reshape(1, 1, 1, 1, 1) * epsilon).float().contiguous().detach()
            branch_candidate = receipt["candidate"]
            branch_prompt = frozen.native_generation.build_task_prompt("t2v", branch_candidate["full_t2v_caption"], prompt_cleaner=prompt_clean)
            conditions, _ = frozen_pair._encode_prompt_pair(renderer, tokenizer, action_prompt=branch_prompt, noop_prompt=noop_prompt, device=device, frozen=frozen)
            if not hook_parity_done:
                off_action, off_noop, _ = frozen_pair.forward_native_prompt_pair(
                    diffusion=diffusion, transformer=transformer, x_sigma=x_sigma,
                    native_schedule_index=SCHEDULE_INDEX,
                    action_condition=conditions["target_action"], noop_condition=conditions["noop"],
                )
                forward_count += 2
            observer = Block22SP4PairCapture(transformer, latent_shape=clean.shape, rank=rank)
            observer.begin()
            action_velocity, noop_velocity, _ = frozen_pair.forward_native_prompt_pair(
                diffusion=diffusion, transformer=transformer, x_sigma=x_sigma,
                native_schedule_index=SCHEDULE_INDEX,
                action_condition=conditions["target_action"], noop_condition=conditions["noop"],
            )
            action_hidden, noop_hidden, current_layouts = observer.finish(dist)
            observer.close()
            forward_count += 2
            if not hook_parity_done:
                _require(torch.equal(off_action, action_velocity) and torch.equal(off_noop, noop_velocity), "block22 hook changes frozen forward bytes")
                hook_parity_done = True
            layouts = current_layouts if layouts is None else layouts
            _require(layouts == current_layouts, "SP4 layout changed within cell")
            raw_code = core.phi_v1_from_global_hidden_delta(action_hidden - noop_hidden, condition_tokens=0, p32=p32)
            _require(tuple(raw_code.shape) == (1, 21, 32) and bool(torch.isfinite(raw_code).all().item()), "raw Phi_v1 code differs")
            raw_codes[branch] = raw_code.detach().float().contiguous()
        camera = raw_codes["camera_only"]
        appearance = raw_codes["appearance_only"]
        for branch in SELECTED_BRANCHES:
            row_id = f"gaav1:{cell['analysis_split']}:{cell['source_iid']}:s{cell['seed']}:{branch}"
            if branch == "noop":
                code = torch.zeros((1, 21, 32), dtype=torch.float32, device=device)
                camera_norm = float(camera.reshape(-1).norm().item())
                appearance_orth = appearance.reshape(-1) - (appearance.reshape(-1) @ (camera.reshape(-1) / camera.reshape(-1).norm())) * (camera.reshape(-1) / camera.reshape(-1).norm())
                nuisance = {
                    "camera_raw_sha256": _raw_tensor_sha(camera),
                    "appearance_raw_sha256": _raw_tensor_sha(appearance),
                    "camera_norm": camera_norm,
                    "appearance_after_gs_norm": float(appearance_orth.norm().item()),
                    "pre_projection_norm": 0.0,
                    "post_projection_norm": 0.0,
                    "survival_cosine": 1.0,
                    "finite_non_degenerate": camera_norm > 1.0e-8 and float(appearance_orth.norm().item()) > 1.0e-8,
                }
                _require(nuisance["finite_non_degenerate"], "noop nuisance coordinates degenerate")
            else:
                code, nuisance = _gram_schmidt_project(raw_codes[branch], camera, appearance)
            code_digests: list[Any] = [None] * 4
            dist.all_gather_object(code_digests, _raw_tensor_sha(code))
            _require(len(set(code_digests)) == 1, "Phi_v1 code bytes differ across ranks")
            if rank == 0:
                row_dir = output / row_id.replace(":", "__")
                row_dir.mkdir()
                tensor_path = row_dir / "quotient.f32le"
                tensor_sha = _save_f32le(tensor_path, code)
                review_status = "PASS_SEALED_BEFORE_EXTRACTION" if official else "MISSING_UNREVIEWED_TECHNICAL_ONLY"
                sidecar = {
                    "schema_version": manifests.SIDECAR_SCHEMA,
                    "row_id": row_id,
                    "candidate_id": cell["candidate_ids"][branch],
                    "source_iid": cell["source_iid"],
                    "analysis_split": cell["analysis_split"],
                    "seed": cell["seed"],
                    "branch": branch,
                    "phi_v1": {
                        "hook": "transformer_1.blocks[22].output",
                        "block_index": 22,
                        "teacher_exact40_index": 29,
                        "sp_world": 4,
                        "sp_order": "rank0_rank1_rank2_rank3_contiguous_global_target_indices",
                        "append_padding_removed": True,
                        "target_layout": "phase_major_21_then_patch_y_x",
                        "pooling": "fixed_spatial_mean",
                        "phase0": "exact_positive_zero",
                        "temporal_dc": "phases_1_20_per_channel_mean_subtracted",
                        "p32_seed": manifests.P32_SEED,
                        "p32_shape": [1536, 32],
                        "p32_raw_path": str(p32_path),
                        "p32_raw_sha256": p32_sha,
                        "p32_generator_path": str(core_path),
                        "p32_generator_source_sha256": file_sha256(core_path),
                        "nuisance_order": ["camera_only", "appearance_only_gram_schmidt_off_camera"],
                    },
                    "tensor": {
                        "path": str(tensor_path),
                        "raw_sha256": tensor_sha,
                        "dtype": "float32",
                        "byte_order": "little",
                        "shape": [21, 32],
                        "normalization": "exact_zero_not_normalized" if branch == "noop" else "global_l2_unit",
                    },
                    "nuisance_projection": nuisance,
                    "review_status": review_status,
                    "generated_media_is_optimizer_input_or_target": False,
                    "optimizer_authorized": False,
                }
                sidecar = {**sidecar, "receipt_digest": object_sha256(sidecar)}
                sidecar_path = row_dir / "phi-v1-sidecar-receipt.json"
                sidecar_sha = _write_create_only(sidecar_path, sidecar)
                receipt_bindings.append({"row_id": row_id, "path": str(sidecar_path), "file_sha256": sidecar_sha, "receipt_digest": sidecar["receipt_digest"], "review_status": review_status})
        del epsilon, gaussian
    _require(forward_count == 82, f"model forward count differs: {forward_count}")
    run_receipt: list[Any] = [None]
    if rank == 0:
        value = {
            "schema_version": RUN_SCHEMA,
            "plan_path": str(plan_path),
            "plan_file_sha256": plan_sha,
            "plan_digest": plan["plan_digest"],
            "mode": plan["mode"],
            "world_size": 4,
            "model_forwards": forward_count,
            "sidecar_count": len(receipt_bindings),
            "sidecars": receipt_bindings,
            "p32_raw_path": str(p32_path),
            "p32_raw_sha256": p32_sha,
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "training_performed": False,
            "optimizer_created": False,
            "generated_media_is_optimizer_input_or_target": False,
            "optimizer_authorized": False,
        }
        value = {**value, "receipt_digest": object_sha256(value)}
        run_path = output / "phi-v1-sidecar-run-receipt.json"
        run_sha = _write_create_only(run_path, value)
        run_receipt[0] = {"path": str(run_path), "file_sha256": run_sha, "receipt_digest": value["receipt_digest"]}
    dist.broadcast_object_list(run_receipt, src=0)
    dist.barrier()
    dist.destroy_process_group()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("build-plan")
    plan.add_argument("--authoring", required=True)
    plan.add_argument("--population", required=True)
    plan.add_argument("--split", choices=("fit", "confirmation"), required=True)
    plan.add_argument("--generation-root", action="append", required=True)
    plan.add_argument("--review-root")
    plan.add_argument("--output", required=True)
    plan.add_argument("--gap-output", required=True)
    plan.add_argument("--allow-unreviewed-technical-only", action="store_true")
    run = commands.add_parser("run-sp4")
    _runtime_parser(run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-plan":
        build_plan(
            authoring_path=args.authoring,
            population_path=args.population,
            split=args.split,
            generation_roots=args.generation_root,
            review_root=args.review_root,
            output=args.output,
            gap_output=args.gap_output,
            allow_unreviewed_technical_only=args.allow_unreviewed_technical_only,
        )
        return 0
    return run_sp4(args)


if __name__ == "__main__":
    raise SystemExit(main())
