#!/usr/bin/env python3
"""Contracts and tensor operators for the Prior-Preserving Phase-Plan Teacher.

P3T is a bridge ablation, not the dense transport method.  It tests whether a
manifest-derived phase teacher plus frozen-base replay can improve motion
without repainting the source.  Dense source transport lives in ``spt_v2``.
This module deliberately has no Bernini dependency so its contracts can be
tested without loading a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


class P3TContractError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise P3TContractError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise P3TContractError(f"{label} must be non-empty text without NUL")
    return " ".join(value.split())


@dataclass(frozen=True)
class ManifestPlan:
    iid: str
    edit_instruction: str
    generation_instruction: str
    target_plan: Mapping[str, Any]
    source_census: Mapping[str, Any]
    compiled_plan: str
    row_digest: str


class PreviewManifest:
    """Hash-bound, exact-IID plan table. Missing/duplicate/malformed rows fail."""

    def __init__(self, path: str | Path, *, expected_sha256: str):
        requested = Path(path).expanduser()
        try:
            mode = requested.lstat().st_mode
        except OSError as error:
            raise P3TContractError(f"preview manifest is unavailable: {error}") from error
        if not stat.S_ISREG(mode) or requested.is_symlink():
            raise P3TContractError("preview manifest must be a plain non-symlink file")
        self.path = requested.resolve(strict=True)
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "") is None:
            raise P3TContractError("expected preview manifest digest must be lowercase SHA-256")
        self.sha256 = file_sha256(self.path)
        if self.sha256 != expected_sha256:
            raise P3TContractError("preview manifest SHA-256 differs")
        rows: dict[str, ManifestPlan] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise P3TContractError(f"cannot read preview manifest: {error}") from error
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise P3TContractError(f"blank preview manifest row at line {line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise P3TContractError(f"invalid preview JSON at line {line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise P3TContractError(f"preview line {line_number} is not an object")
            iid = _text(raw.get("iid"), label=f"IID at line {line_number}")
            if iid in rows:
                raise P3TContractError(f"duplicate preview IID: {iid}")
            raw_edit = raw.get("edit_instruction")
            if (
                not isinstance(raw_edit, str)
                or not raw_edit.strip()
                or "\x00" in raw_edit
            ):
                raise P3TContractError(
                    f"edit_instruction for {iid} must be non-empty text without NUL"
                )
            if raw.get("edit_instruction_sha256") != hashlib.sha256(
                raw_edit.encode("utf-8")
            ).hexdigest():
                raise P3TContractError(f"edit instruction digest differs for {iid}")
            edit_instruction = _text(
                raw_edit, label=f"edit_instruction for {iid}"
            )
            raw_instruction = raw.get("generation_instruction")
            if not isinstance(raw_instruction, str) or not raw_instruction.strip() or "\x00" in raw_instruction:
                raise P3TContractError(f"generation_instruction for {iid} must be non-empty text without NUL")
            declared_instruction = raw.get("generation_instruction_sha256")
            actual_instruction = hashlib.sha256(raw_instruction.encode("utf-8")).hexdigest()
            if declared_instruction != actual_instruction:
                raise P3TContractError(f"generation instruction digest differs for {iid}")
            instruction = _text(raw_instruction, label=f"generation_instruction for {iid}")
            plan = raw.get("target_plan")
            if not isinstance(plan, dict) or plan.get("iid") != iid:
                raise P3TContractError(f"target_plan IID differs for {iid}")
            census = raw.get("source_census")
            if not isinstance(census, dict) or census.get("iid") != iid:
                raise P3TContractError(f"source_census IID differs for {iid}")
            declared_plan = raw.get("target_plan_sha256")
            if declared_plan is None and isinstance(raw.get("provenance"), dict):
                declared_plan = raw["provenance"].get("target_plan_sha256")
            if declared_plan is not None and declared_plan != object_sha256(plan):
                raise P3TContractError(f"target plan digest differs for {iid}")
            claimed_row = raw.get("row_digest")
            if claimed_row is not None:
                unsigned = dict(raw)
                unsigned.pop("row_digest", None)
                if claimed_row != object_sha256(unsigned):
                    raise P3TContractError(f"row digest differs for {iid}")
            row_digest = claimed_row or object_sha256(raw)
            rows[iid] = ManifestPlan(
                iid,
                edit_instruction,
                instruction,
                plan,
                census,
                compile_phase_plan(plan, census),
                row_digest,
            )
        if not rows:
            raise P3TContractError("preview manifest is empty")
        self._rows = rows
        self.membership_sha256 = object_sha256(sorted((iid, row.row_digest) for iid, row in rows.items()))

    def require(self, iid: str) -> ManifestPlan:
        try:
            return self._rows[iid]
        except KeyError as error:
            raise P3TContractError(f"dataset IID is absent from preview manifest: {iid}") from error

    def receipt(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "rows": len(self._rows), "membership_sha256": self.membership_sha256}


def compile_phase_plan(
    target_plan: Mapping[str, Any], source_census: Mapping[str, Any]
) -> str:
    """Compile v16 targets into stable 21-latent-phase conditioning text.

    The teacher uses stable source references plus each target motion exactly
    once.  This is shorter than the synthetic generation prompt and avoids
    repeating every action three times near the 512-token limit.
    """

    subjects = target_plan.get("dynamic_subject_targets")
    camera = target_plan.get("camera_target")
    source_subjects = source_census.get("dynamic_subjects")
    if (
        not isinstance(subjects, list)
        or not subjects
        or not isinstance(camera, Mapping)
        or not isinstance(source_subjects, list)
        or not source_subjects
    ):
        raise P3TContractError("target_plan requires non-empty dynamic_subject_targets and camera_target")
    references: dict[str, str] = {}
    for index, subject in enumerate(source_subjects):
        if not isinstance(subject, Mapping):
            raise P3TContractError("source subject must be an object")
        sid = _text(subject.get("subject_id"), label=f"source subject {index} id")
        if sid in references:
            raise P3TContractError(f"duplicate source subject id: {sid}")
        references[sid] = _text(
            subject.get("stable_reference"), label=f"source subject {sid} reference"
        )
    clauses: list[str] = []
    for index, subject in enumerate(subjects):
        if not isinstance(subject, Mapping):
            raise P3TContractError("target subject must be an object")
        sid = _text(subject.get("subject_id"), label=f"subject {index} id")
        if sid not in references:
            raise P3TContractError(f"target subject lacks source reference: {sid}")
        motion = _text(
            subject.get("target_motion"), label=f"subject {sid} target motion"
        )
        clauses.append(f"{references[sid]}: {motion}")
    camera_motion = _text(camera.get("target_motion"), label="camera target motion")
    agents = " | ".join(clauses)
    return (
        "Training-only semantic motion teacher over 21 latent phases. "
        "PREPARE phases 00-03: preserve the exact source state and initiate the specified motions. "
        "EXECUTE phases 04-15: follow the frame-timed subject motions above. "
        "SETTLE phases 16-20: complete and hold the specified final states. "
        f"SUBJECT MOTIONS: [{agents}]. "
        f"CAMERA phases 00-20: {camera_motion}. Preserve identities, appearance, and unedited scene content."
    )


def compile_generic_phase_wrapper(instruction: str) -> str:
    """Source-only deterministic inference wrapper with no target-plan oracle."""

    instruction = _text(instruction, label="raw edit instruction")
    return (
        f"{instruction} Phase plan over 21 latent phases. "
        f"PREPARE phases 00-03: preserve source identity and establish the action start. "
        f"EXECUTE phases 04-15: perform the requested action: [{instruction}]. "
        f"SETTLE phases 16-20: complete the requested action and stabilize appearance. "
        "Preserve identities, appearance, camera behavior, background, and unedited scene content."
    )


def temporal_project(field: Any, *, latent_frames: int = 21) -> Any:
    if getattr(field, "ndim", None) != 3:
        raise P3TContractError("field must have shape [B,N,D]")
    tokens = int(field.shape[1])
    if tokens <= 0 or tokens % latent_frames:
        raise P3TContractError("packed token count is not divisible by latent frames")
    grid = field.reshape(int(field.shape[0]), latent_frames, tokens // latent_frames, int(field.shape[2]))
    return (grid - grid.mean(dim=1, keepdim=True)).reshape_as(field)


def temporal_complement(field: Any, *, latent_frames: int = 21) -> Any:
    return field - temporal_project(field, latent_frames=latent_frames)


def shifted_sigmas(*, steps: int = 40, flow_shift: float = 5.0) -> tuple[float, ...]:
    if type(steps) is not int or steps <= 0 or not math.isfinite(flow_shift) or flow_shift <= 0:
        raise P3TContractError("invalid shifted sigma schedule")
    return tuple(flow_shift * (1 - i / steps) / (1 + (flow_shift - 1) * (1 - i / steps)) for i in range(steps)) + (0.0,)


def interval_weight(sigma: Any, *, steps: int = 40, flow_shift: float = 5.0) -> Any:
    import torch
    schedule = shifted_sigmas(steps=steps, flow_shift=flow_shift)
    centers = torch.as_tensor([(a + b) / 2 for a, b in zip(schedule, schedule[1:])], device=sigma.device, dtype=torch.float32)
    widths = torch.as_tensor([abs(a - b) for a, b in zip(schedule, schedule[1:])], device=sigma.device, dtype=torch.float32)
    widths = widths / widths.mean()
    nearest = (sigma.float().reshape(-1, 1) - centers.reshape(1, -1)).abs().argmin(dim=1)
    return widths[nearest].reshape(sigma.shape)


def deterministic_source_corruption(clean: Any, *, kind: str, seed: int, strength: float = 0.25) -> Any:
    """Deterministic source-condition corruption for a separate restoration arm.

    Supports [B,C,T,H,W]. It intentionally returns only the corrupted condition;
    clean remains the caller's target. No random global RNG state is touched.
    """

    import torch
    if getattr(clean, "ndim", None) != 5 or not 0 <= float(strength) <= 1:
        raise P3TContractError("restoration latent/strength contract differs")
    if kind == "speed":
        indices = torch.linspace(0, int(clean.shape[2]) - 1, int(clean.shape[2]), device=clean.device)
        offset = max(1, int(round(float(strength) * max(1, int(clean.shape[2]) - 1))))
        indices = ((indices.long() + offset).clamp(max=int(clean.shape[2]) - 1))
        return clean.index_select(2, indices)
    result = clean.clone()
    generator = torch.Generator(device=clean.device).manual_seed(int(seed))
    if kind == "tube":
        height = max(1, round(int(clean.shape[3]) * float(strength)))
        width = max(1, round(int(clean.shape[4]) * float(strength)))
        top = int(torch.randint(0, max(1, int(clean.shape[3]) - height + 1), (), generator=generator, device=clean.device).item())
        left = int(torch.randint(0, max(1, int(clean.shape[4]) - width + 1), (), generator=generator, device=clean.device).item())
        result[:, :, :, top:top + height, left:left + width] = 0
    elif kind == "cube":
        span = max(1, round(int(clean.shape[2]) * float(strength)))
        start = int(torch.randint(0, max(1, int(clean.shape[2]) - span + 1), (), generator=generator, device=clean.device).item())
        result[:, :, start:start + span] = 0
    else:
        raise P3TContractError("restoration corruption must be speed, tube, or cube")
    return result
