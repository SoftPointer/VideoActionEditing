"""Distributed fresh source census for expanding the Goku atomic-edit pool.

This stage deliberately does *not* reuse a historical Qwen verdict as a gate.
It binds an immutable mother manifest and the exhausted/old selected manifest,
excludes both old IIDs and old group IDs, and performs a fresh geometry and
actor-motion census of every remaining source video.

Workers publish one create-only terminal receipt per IID.  ``finalize`` only
accepts a complete receipt closure, merges in mother-manifest order, extracts
lossless frame-zero anchors, and emits ``selected.jsonl`` rows accepted by
``goku_action_anchor_qwen.validate_input_row``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import geometry as geometry_module
from . import goku_action_anchor_prefilter as prefilter
from . import goku_action_anchor_qwen as anchor_qwen_module
from . import motion_features as motion_features_module
from .goku_action_anchor_qwen import validate_input_row
from .rules import score_action_rule, stable_group_split


SCHEMA_VERSION = "motive-goku-atomic-source-expand-v1"
RECEIPT_SCHEMA = "motive-goku-atomic-source-expand-receipt-v1"
SUMMARY_SCHEMA = "motive-goku-atomic-source-expand-summary-v1"
DONE_SCHEMA = "motive-goku-atomic-source-expand-done-v1"
NORMALIZED_SCHEMA = "motive-goku-atomic-source-expand-input-v1"
RECEIPT_DIR = "receipts"
ANCHOR_DIR = "anchors"
EVALUATED_NAME = "evaluated.jsonl"
SELECTED_NAME = "selected.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
FINAL_ENTRIES = frozenset(
    {ANCHOR_DIR, EVALUATED_NAME, SELECTED_NAME, SUMMARY_NAME, DONE_NAME}
)


class SourceExpandError(RuntimeError):
    """Fail-closed source-expansion error."""


@dataclass(frozen=True)
class SourceExpandConfig:
    """Wan-compatible media contract plus the proven old motion thresholds."""

    analysis_frames: int = 32
    resize_width: int = 256
    active_speed_threshold: float = 0.005
    required_frames: int = 81
    required_fps: float = 25.0
    fps_tolerance: float = 1e-3
    min_short_side: int = 640
    min_pixels: int = 640 * 640
    min_duration_seconds: float = 3.15
    max_duration_seconds: float = 3.30
    min_residual_speed_p90: float = 0.005
    min_active_pixel_fraction: float = 0.010
    min_active_frame_fraction: float = 0.40
    min_actor_likeness: float = 0.25
    min_temporal_coverage: float = 0.40
    min_largest_component_share: float = 0.08
    max_spatial_energy_entropy: float = 0.94

    def validate(self) -> None:
        if self.analysis_frames < 3:
            raise ValueError("analysis_frames must be at least 3")
        if self.resize_width < 32:
            raise ValueError("resize_width must be at least 32")
        if self.required_frames < 3 or self.required_fps <= 0:
            raise ValueError("required frame geometry is invalid")
        if self.fps_tolerance < 0:
            raise ValueError("fps_tolerance must be non-negative")
        if self.min_short_side <= 0 or self.min_pixels <= 0:
            raise ValueError("resolution thresholds must be positive")
        if not 0 < self.min_duration_seconds <= self.max_duration_seconds:
            raise ValueError("duration thresholds are invalid")

    def prefilter_config(self) -> prefilter.PrefilterConfig:
        """Return the old dynamic-actor gate with the new media geometry."""

        return prefilter.PrefilterConfig(
            sample_size=1,
            workers=1,
            max_per_family=1,
            analysis_frames=self.analysis_frames,
            resize_width=self.resize_width,
            active_speed_threshold=self.active_speed_threshold,
            min_short_side=self.min_short_side,
            min_pixels=self.min_pixels,
            min_fps=self.required_fps - self.fps_tolerance,
            max_fps=self.required_fps + self.fps_tolerance,
            min_duration_seconds=self.min_duration_seconds,
            max_duration_seconds=self.max_duration_seconds,
            min_source_frames=self.required_frames,
            min_residual_speed_p90=self.min_residual_speed_p90,
            min_active_pixel_fraction=self.min_active_pixel_fraction,
            min_active_frame_fraction=self.min_active_frame_fraction,
            min_actor_likeness=self.min_actor_likeness,
            min_temporal_coverage=self.min_temporal_coverage,
            min_largest_component_share=self.min_largest_component_share,
            max_spatial_energy_entropy=self.max_spatial_energy_entropy,
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _implementation_bundle() -> dict[str, Any]:
    """Bind every local module that can affect census or output validity."""

    modules = {
        "goku_atomic_source_expand": Path(__file__).resolve(strict=True),
        "geometry": Path(str(geometry_module.__file__)).resolve(strict=True),
        "motion_features": Path(str(motion_features_module.__file__)).resolve(
            strict=True
        ),
        "goku_action_anchor_prefilter": Path(str(prefilter.__file__)).resolve(
            strict=True
        ),
        "goku_action_anchor_qwen_validator": Path(
            str(anchor_qwen_module.__file__)
        ).resolve(strict=True),
    }
    records = {
        name: {"path": str(path), "sha256": _file_sha256(path)}
        for name, path in sorted(modules.items())
    }
    return {
        "schema_version": "motive-goku-source-expand-implementation-bundle-v1",
        "modules": records,
        "bundle_digest": _object_digest(records),
    }


def _execution_config_digest(
    config: SourceExpandConfig, implementation: Mapping[str, Any]
) -> str:
    return _object_digest(
        {
            "config": asdict(config),
            "implementation_bundle_digest": implementation["bundle_digest"],
        }
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )


def _sha_field(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceExpandError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise SourceExpandError(f"invalid JSON in {context}: {error}") from error
    if not isinstance(value, dict):
        raise SourceExpandError(f"{context} is not a JSON object")
    return value


def _read_jsonl(path: Path, *, context: str) -> tuple[list[dict[str, Any]], bytes]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise SourceExpandError(f"{context} must be a plain file: {resolved}")
    raw = resolved.read_bytes()
    if not raw:
        raise SourceExpandError(f"{context} is empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise SourceExpandError(f"blank line in {context}:{line_number}")
        rows.append(_parse_object(line, context=f"{context}:{line_number}"))
    return rows, raw


def _text(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _family(row: Mapping[str, Any], *, prompt: str) -> str:
    family = prefilter._primary_family(row)
    if family != "unknown":
        return family
    decision = score_action_rule(
        prompt,
        source_caption=_text(row, ("source_caption",)),
    )
    if decision.action_families:
        return str(decision.action_families[0])
    return "unclassified_source_motion"


def _normalize_row(
    raw: Mapping[str, Any],
    *,
    mother_rank: int,
    source_ref: str,
) -> dict[str, Any]:
    iid = _text(raw, ("iid", "case_id", "id"))
    if not iid or prefilter._SAFE_IID.fullmatch(iid) is None or iid in {".", ".."}:
        raise SourceExpandError(f"unsafe or missing IID at {source_ref}: {iid!r}")
    src_video = _text(raw, ("src_video", "source_video"))
    if not src_video:
        raise SourceExpandError(f"iid={iid} has no source video")
    prompt = _text(raw, ("prompt", "instruction_en", "instruction"))
    if not prompt:
        prompt = "Create a clear atomic action edit for the visible moving subject."
    source_caption = _text(raw, ("source_caption",))
    if not source_caption:
        source_caption = "The source video shows visible subjects in motion."
    edited_caption = _text(raw, ("edited_caption", "target_caption"))
    if not edited_caption:
        edited_caption = f"The visible subject performs this action: {prompt}"
    group_id = _text(raw, ("group_id",))
    if not group_id:
        group_id, _ = stable_group_split(
            source_video=src_video,
            source_caption=source_caption,
        )
    qwen_evidence = raw.get("qwen_evidence")
    qwen_present = isinstance(qwen_evidence, Mapping)
    diagnostic: dict[str, Any] | None = None
    if qwen_present:
        gate, reasons = prefilter.qwen_source_gate(raw)
        diagnostic = {"legacy_gate_projection": gate, "diagnostic_reasons": reasons}
    raw_digest = _object_digest(raw)
    return {
        "schema_version": NORMALIZED_SCHEMA,
        "iid": iid,
        "group_id": group_id,
        "family": _family(raw, prompt=prompt),
        "prompt": prompt,
        "source_caption": source_caption,
        "edited_caption": edited_caption,
        "src_video": src_video,
        "tgt_video": _text(raw, ("tgt_video", "edited_video", "target_video")),
        "mother_rank": mother_rank,
        "mother_source_ref": source_ref,
        "mother_row_sha256": raw_digest,
        "legacy_qwen_provenance": {
            "present": qwen_present,
            "evidence_sha256": _object_digest(qwen_evidence) if qwen_present else None,
            "diagnostic": diagnostic,
            "authoritative": False,
            "used_as_gate": False,
        },
    }


def load_mother_input(path: str | Path) -> tuple[list[dict[str, Any]], str, str]:
    """Load normalized/fused JSONL or a raw ``combine_json`` directory."""

    input_path = Path(path).expanduser().resolve(strict=True)
    normalized: list[dict[str, Any]] = []
    if input_path.is_file() and not input_path.is_symlink():
        rows, raw = _read_jsonl(input_path, context="mother JSONL")
        digest = _sha256_bytes(raw)
        kind = "jsonl_bytes"
        for rank, row in enumerate(rows, start=1):
            normalized.append(
                _normalize_row(row, mother_rank=rank, source_ref=f"{input_path}:{rank}")
            )
    elif input_path.is_dir() and not input_path.is_symlink():
        files = sorted(input_path.glob("*_all.json"), key=lambda item: item.name)
        if not files:
            files = sorted(input_path.glob("*.json"), key=lambda item: item.name)
        if not files:
            raise SourceExpandError(f"no JSON files in raw input directory: {input_path}")
        binding: list[dict[str, Any]] = []
        for rank, item in enumerate(files, start=1):
            if item.is_symlink() or not item.is_file():
                raise SourceExpandError(f"raw input entry is not a plain file: {item}")
            raw_bytes = item.read_bytes()
            binding.append(
                {
                    "name": item.name,
                    "bytes": len(raw_bytes),
                    "sha256": _sha256_bytes(raw_bytes),
                }
            )
            normalized.append(
                _normalize_row(
                    _parse_object(raw_bytes, context=str(item)),
                    mother_rank=rank,
                    source_ref=str(item),
                )
            )
        digest = _object_digest(binding)
        kind = "raw_directory_file_closure_v1"
    else:
        raise SourceExpandError(f"mother input must be a plain file or directory: {input_path}")
    seen: set[str] = set()
    for row in normalized:
        iid = str(row["iid"])
        if iid in seen:
            raise SourceExpandError(f"duplicate mother IID: {iid}")
        seen.add(iid)
    return normalized, digest, kind


def _load_old_selected(
    path: str | Path,
) -> tuple[list[dict[str, Any]], str, set[str], set[str]]:
    rows, raw = _read_jsonl(Path(path), context="old selected manifest")
    old_iids: set[str] = set()
    old_groups: set[str] = set()
    for index, row in enumerate(rows, start=1):
        iid = _text(row, ("iid",))
        group = _text(row, ("group_id",))
        if not iid or not group:
            raise SourceExpandError(
                f"old selected row {index} must contain non-empty iid and group_id"
            )
        if iid in old_iids:
            raise SourceExpandError("old selected IID values must be unique")
        old_iids.add(iid)
        old_groups.add(group)
    return rows, _sha256_bytes(raw), old_iids, old_groups


def _bind_inputs(
    *,
    input_path: str | Path,
    old_selected: str | Path,
    expected_input_sha256: str,
    expected_old_selected_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str], set[str]]:
    rows, input_sha, input_kind = load_mother_input(input_path)
    old_rows, old_sha, old_iids, old_groups = _load_old_selected(old_selected)
    if input_sha != _sha_field(expected_input_sha256, name="expected input SHA"):
        raise SourceExpandError(
            f"mother input SHA differs: expected={expected_input_sha256} actual={input_sha}"
        )
    if old_sha != _sha_field(
        expected_old_selected_sha256, name="expected old selected SHA"
    ):
        raise SourceExpandError(
            "old selected SHA differs: "
            f"expected={expected_old_selected_sha256} actual={old_sha}"
        )
    implementation = _implementation_bundle()
    bindings = {
        "input_path": str(Path(input_path).expanduser().resolve(strict=True)),
        "input_sha256": input_sha,
        "input_binding_kind": input_kind,
        "input_rows": len(rows),
        "old_selected_path": str(
            Path(old_selected).expanduser().resolve(strict=True)
        ),
        "old_selected_sha256": old_sha,
        "old_selected_rows": len(old_rows),
        "implementation": implementation,
        "implementation_bundle_digest": implementation["bundle_digest"],
    }
    bindings["binding_digest"] = _object_digest(bindings)
    return rows, bindings, old_iids, old_groups


def _resolve_source_video(value: str, root: Path) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw if raw.is_absolute() else root / raw]
    # Raw Goku JSONs can retain a dataset-relative prefix while --video-root
    # points directly at subject_movement/extracted.
    basename_fallback = root / raw.name
    if basename_fallback not in candidates:
        candidates.append(basename_fallback)
    failures: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            if resolved.is_file():
                return resolved
        except (FileNotFoundError, OSError, ValueError) as error:
            failures.append(f"{candidate}:{type(error).__name__}")
    raise SourceExpandError(
        f"source video cannot be resolved inside video_root ({'; '.join(failures)})"
    )


def _receipt_path(work_dir: Path, iid: str) -> Path:
    return work_dir / RECEIPT_DIR / f"{iid}.json"


def _write_create_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary receipt exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"create-only receipt exists: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    return _object_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )


def _validate_receipt(
    path: Path,
    *,
    row: Mapping[str, Any],
    bindings: Mapping[str, Any],
    config_digest: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SourceExpandError(f"receipt is not a plain file: {path}")
    receipt = _parse_object(path.read_bytes(), context=str(path))
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "iid": row["iid"],
        "mother_rank": row["mother_rank"],
        "mother_row_sha256": row["mother_row_sha256"],
        "input_binding_digest": bindings["binding_digest"],
        "config_digest": config_digest,
        "implementation_bundle_digest": bindings[
            "implementation_bundle_digest"
        ],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise SourceExpandError(f"receipt {path} has mismatched {key}")
    if receipt.get("receipt_digest") != _receipt_digest(receipt):
        raise SourceExpandError(f"receipt digest differs: {path}")
    if not isinstance(receipt.get("row"), dict):
        raise SourceExpandError(f"receipt lacks evaluated row: {path}")
    return receipt


def _base_evaluated(
    row: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        **dict(row),
        "input_manifest_sha256": bindings["input_sha256"],
        "old_selected_sha256": bindings["old_selected_sha256"],
        "resolved_src_video": None,
        "media": None,
        "motion": None,
        "actor_motion": None,
        "score_components": None,
        "prefilter_score": None,
        "source_video_sha256": None,
        "eligible": False,
        "rejection_reasons": [],
        "selected": False,
        "selection_rank": None,
        "within_family_rank": None,
        "anchor_image": None,
        "resolved_anchor_image": None,
        "anchor_sha256": None,
    }


def _media_reasons(
    result: Mapping[str, Any], config: SourceExpandConfig
) -> list[str]:
    reasons = prefilter._media_motion_reasons(result, config.prefilter_config())
    media = result["media"]
    motion = result["motion"]
    if int(media["frame_count"]) != config.required_frames:
        reasons.append("frame_count_not_wan81")
    if not math.isclose(
        float(media["fps"]),
        config.required_fps,
        rel_tol=0.0,
        abs_tol=config.fps_tolerance,
    ):
        reasons.append("fps_not_wan25")
    container_duration = float(media["frame_count"]) / float(media["fps"])
    if not config.min_duration_seconds <= container_duration <= config.max_duration_seconds:
        reasons.append("container_duration_out_of_range")
    if float(motion["scene_cut_ratio"]) != 0.0:
        reasons.append("scene_cut_nonzero")
    return list(dict.fromkeys(reasons))


def _make_receipt(
    evaluated: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    config_digest: str,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "iid": evaluated["iid"],
        "mother_rank": evaluated["mother_rank"],
        "mother_row_sha256": evaluated["mother_row_sha256"],
        "input_binding_digest": bindings["binding_digest"],
        "config_digest": config_digest,
        "implementation_bundle_digest": bindings[
            "implementation_bundle_digest"
        ],
        "row": dict(evaluated),
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return receipt


def _commit_analysis_result(
    *,
    evaluated: dict[str, Any],
    result: Mapping[str, Any],
    work: Path,
    bindings: Mapping[str, Any],
    config: SourceExpandConfig,
    config_digest: str,
) -> tuple[int, int]:
    """Validate one result and durably publish its receipt immediately."""

    if str(result.get("iid")) != str(evaluated["iid"]):
        raise SourceExpandError("analysis result identity differs")
    eligible = 0
    error = 0
    if not result.get("ok"):
        evaluated["rejection_reasons"].append(
            f"analysis_error:{result.get('error_type', 'unknown')}"
        )
        evaluated["analysis_error"] = result.get("error")
        error = 1
    else:
        evaluated["media"] = dict(result["media"])
        evaluated["media"]["container_duration_seconds"] = (
            float(result["media"]["frame_count"])
            / float(result["media"]["fps"])
        )
        evaluated["motion"] = result["motion"]
        evaluated["actor_motion"] = result["actor_motion"]
        evaluated["rejection_reasons"].extend(_media_reasons(result, config))
        score, components = prefilter._quality_score(
            result["media"], result["motion"], result["actor_motion"]
        )
        evaluated["prefilter_score"] = score
        evaluated["score_components"] = components
        source = Path(str(evaluated["resolved_src_video"]))
        before = source.stat()
        evaluated["source_video_sha256"] = _file_sha256(source)
        after = source.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_size != int(evaluated["media"]["file_size_bytes"])
            or after.st_mtime_ns != int(evaluated["media"]["mtime_ns_at_analysis"])
        ):
            evaluated["rejection_reasons"].append("source_changed_after_analysis")
        evaluated["eligible"] = not evaluated["rejection_reasons"]
        eligible = int(evaluated["eligible"])
    receipt = _make_receipt(
        evaluated, bindings=bindings, config_digest=config_digest
    )
    _write_create_only(
        _receipt_path(work, str(evaluated["iid"])), _json_bytes(receipt)
    )
    return eligible, error


def _stream_parallel_analysis(
    pending: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    max_workers: int,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield completed analyses with at most ``2 * max_workers`` in flight."""

    iterator = iter(pending)
    in_flight: dict[Any, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(len(pending), 2 * max_workers)):
            evaluated, payload = next(iterator)
            in_flight[executor.submit(prefilter._analyze_payload, payload)] = evaluated
        while in_flight:
            completed, unused_pending = wait(
                in_flight, return_when=FIRST_COMPLETED
            )
            for future in completed:
                evaluated = in_flight.pop(future)
                # result() intentionally propagates a worker crash.  Receipts
                # yielded before it remain durable and make the rerun resumable.
                yield evaluated, future.result()
                try:
                    next_evaluated, next_payload = next(iterator)
                except StopIteration:
                    continue
                in_flight[
                    executor.submit(prefilter._analyze_payload, next_payload)
                ] = next_evaluated


def run_worker(
    *,
    input_path: str | Path,
    old_selected: str | Path,
    video_root: str | Path,
    work_dir: str | Path,
    expected_input_sha256: str,
    expected_old_selected_sha256: str,
    worker_index: int,
    num_workers: int,
    local_workers: int = 8,
    config: SourceExpandConfig | None = None,
) -> dict[str, Any]:
    """Run one deterministic shard, resuming only valid terminal receipts."""

    config = config or SourceExpandConfig()
    config.validate()
    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise SourceExpandError("worker_index must be in [0, num_workers)")
    if local_workers <= 0:
        raise SourceExpandError("local_workers must be positive")
    rows, bindings, old_iids, old_groups = _bind_inputs(
        input_path=input_path,
        old_selected=old_selected,
        expected_input_sha256=expected_input_sha256,
        expected_old_selected_sha256=expected_old_selected_sha256,
    )
    root = Path(video_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise SourceExpandError(f"video_root must be a plain directory: {root}")
    work = Path(work_dir).expanduser().resolve(strict=False)
    (work / RECEIPT_DIR).mkdir(parents=True, exist_ok=True)
    config_digest = _execution_config_digest(config, bindings["implementation"])
    assigned = [row for row in rows if (int(row["mother_rank"]) - 1) % num_workers == worker_index]
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    resumed = 0
    excluded = 0
    immediate = 0
    for row in assigned:
        receipt_path = _receipt_path(work, str(row["iid"]))
        if os.path.lexists(receipt_path):
            _validate_receipt(
                receipt_path,
                row=row,
                bindings=bindings,
                config_digest=config_digest,
            )
            resumed += 1
            continue
        evaluated = _base_evaluated(row, bindings=bindings)
        if row["iid"] in old_iids:
            evaluated["rejection_reasons"].append("excluded_old_iid")
        if row["group_id"] in old_groups:
            evaluated["rejection_reasons"].append("excluded_old_group_id")
        if evaluated["rejection_reasons"]:
            excluded += 1
            receipt = _make_receipt(
                evaluated, bindings=bindings, config_digest=config_digest
            )
            _write_create_only(receipt_path, _json_bytes(receipt))
            continue
        try:
            source = _resolve_source_video(str(row["src_video"]), root)
            evaluated["resolved_src_video"] = str(source)
        except Exception as error:
            evaluated["rejection_reasons"].append(
                f"source_video_error:{type(error).__name__}"
            )
            evaluated["source_video_error"] = str(error)
            receipt = _make_receipt(
                evaluated, bindings=bindings, config_digest=config_digest
            )
            _write_create_only(receipt_path, _json_bytes(receipt))
            immediate += 1
            continue
        pending.append(
            (evaluated, prefilter._analysis_payload(evaluated, config=config.prefilter_config()))
        )

    eligible = 0
    errors = 0
    if local_workers == 1:
        completed_results = (
            (evaluated, prefilter._analyze_payload(payload))
            for evaluated, payload in pending
        )
    else:
        completed_results = _stream_parallel_analysis(
            pending, max_workers=local_workers
        )
    for evaluated, result in completed_results:
        item_eligible, item_error = _commit_analysis_result(
            evaluated=evaluated,
            result=result,
            work=work,
            bindings=bindings,
            config=config,
            config_digest=config_digest,
        )
        eligible += item_eligible
        errors += item_error
    return {
        "worker_index": worker_index,
        "num_workers": num_workers,
        "assigned": len(assigned),
        "resumed": resumed,
        "excluded": excluded,
        "source_errors": immediate,
        "analyzed": len(pending),
        "analysis_errors": errors,
        "eligible": eligible,
        "bindings": bindings,
        "config_digest": config_digest,
    }


def _publish_directory(output_dir: Path, writer: Any) -> None:
    target = output_dir.expanduser().resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"create-only final output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        writer(stage, target)
        if {entry.name for entry in stage.iterdir()} != FINAL_ENTRIES:
            raise SourceExpandError("final staging closure differs")
        if os.path.lexists(target):
            raise FileExistsError(f"final output appeared during publication: {target}")
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def finalize(
    *,
    input_path: str | Path,
    old_selected: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    expected_input_sha256: str,
    expected_old_selected_sha256: str,
    sample_size: int,
    config: SourceExpandConfig | None = None,
) -> dict[str, Any]:
    """Verify the complete receipt closure and publish deterministic rows."""

    config = config or SourceExpandConfig()
    config.validate()
    if sample_size <= 0:
        raise SourceExpandError("sample_size must be positive")
    rows, bindings, unused_old_iids, unused_old_groups = _bind_inputs(
        input_path=input_path,
        old_selected=old_selected,
        expected_input_sha256=expected_input_sha256,
        expected_old_selected_sha256=expected_old_selected_sha256,
    )
    work = Path(work_dir).expanduser().resolve(strict=True)
    config_digest = _execution_config_digest(config, bindings["implementation"])
    receipts: list[dict[str, Any]] = []
    for row in rows:
        path = _receipt_path(work, str(row["iid"]))
        if not os.path.lexists(path):
            raise SourceExpandError(
                f"receipt closure incomplete; missing mother rank {row['mother_rank']}: {path}"
            )
        receipts.append(
            _validate_receipt(
                path, row=row, bindings=bindings, config_digest=config_digest
            )
        )
    evaluated = [dict(receipt["row"]) for receipt in receipts]
    if [int(row["mother_rank"]) for row in evaluated] != list(
        range(1, len(rows) + 1)
    ):
        raise SourceExpandError("receipt merge order differs from mother order")
    eligible = [row for row in evaluated if row["eligible"]]
    selected_prelim: list[dict[str, Any]] = []
    fresh_groups: set[str] = set()
    for row in eligible:
        group = str(row["group_id"])
        if group in fresh_groups:
            row["rejection_reasons"] = [
                *row["rejection_reasons"],
                "duplicate_fresh_group_not_selected",
            ]
            continue
        fresh_groups.add(group)
        selected_prelim.append(row)
        if len(selected_prelim) == sample_size:
            break

    target = Path(output_dir).expanduser().resolve(strict=False)

    def writer(stage: Path, final_output: Path) -> None:
        anchor_root = stage / ANCHOR_DIR
        anchor_root.mkdir()
        selected: list[dict[str, Any]] = []
        family_counts: Counter[str] = Counter()
        selected_by_iid: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(selected_prelim, start=1):
            source = Path(str(row["resolved_src_video"]))
            before = source.stat()
            source_sha = _file_sha256(source)
            anchor_bytes, width, height = prefilter._extract_anchor_png_bytes(source)
            after = source.stat()
            if source_sha != row["source_video_sha256"]:
                raise SourceExpandError(f"source SHA changed before finalize: {row['iid']}")
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or after.st_size != int(row["media"]["file_size_bytes"])
                or after.st_mtime_ns != int(row["media"]["mtime_ns_at_analysis"])
            ):
                raise SourceExpandError(f"source stat changed before finalize: {row['iid']}")
            relative = Path(ANCHOR_DIR) / f"{row['iid']}.png"
            (stage / relative).write_bytes(anchor_bytes)
            complete = dict(row)
            family_counts[str(row["family"])] += 1
            complete.update(
                {
                    "selected": True,
                    "selection_rank": rank,
                    "within_family_rank": family_counts[str(row["family"])],
                    "anchor_image": relative.as_posix(),
                    "resolved_anchor_image": str(
                        (final_output / relative).resolve(strict=False)
                    ),
                    "anchor_sha256": _sha256_bytes(anchor_bytes),
                    "media": {
                        **dict(row["media"]),
                        "anchor_width": width,
                        "anchor_height": height,
                        "anchor_frame_index": 0,
                        "anchor_encoding": "lossless_png",
                    },
                }
            )
            validate_input_row(complete)
            selected.append(complete)
            selected_by_iid[str(complete["iid"])] = complete
        final_evaluated = [
            selected_by_iid.get(str(row["iid"]), row) for row in evaluated
        ]
        evaluated_raw = _jsonl_bytes(final_evaluated)
        selected_raw = _jsonl_bytes(selected)
        (stage / EVALUATED_NAME).write_bytes(evaluated_raw)
        (stage / SELECTED_NAME).write_bytes(selected_raw)
        rejection_counts: Counter[str] = Counter()
        for row in final_evaluated:
            rejection_counts.update(row["rejection_reasons"])
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "inputs": bindings,
            "config": asdict(config),
            "config_digest": config_digest,
            "implementation": bindings["implementation"],
            "semantics": {
                "fresh_media_geometry_motion_analysis": True,
                "legacy_qwen_provenance_only": True,
                "legacy_qwen_used_as_gate": False,
                "old_iid_and_group_exclusion": True,
                "selection_order": "mother_rank_ascending_unique_group",
                "anchor": "lossless_exact_decoded_frame_zero_png",
            },
            "counts": {
                "mother": len(rows),
                "receipts": len(receipts),
                "eligible": len(eligible),
                "selected": len(selected),
                "requested": sample_size,
                "selection_shortfall": max(sample_size - len(selected), 0),
            },
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "selected_mother_ranks": [row["mother_rank"] for row in selected],
            "production_eligible": False,
        }
        summary_raw = _json_bytes(summary)
        (stage / SUMMARY_NAME).write_bytes(summary_raw)
        anchors = {row["anchor_image"]: row["anchor_sha256"] for row in selected}
        artifacts = {
            EVALUATED_NAME: _sha256_bytes(evaluated_raw),
            SELECTED_NAME: _sha256_bytes(selected_raw),
            SUMMARY_NAME: _sha256_bytes(summary_raw),
            ANCHOR_DIR: _object_digest(anchors),
        }
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "input_binding_digest": bindings["binding_digest"],
            "config_digest": config_digest,
            "implementation_bundle_digest": bindings[
                "implementation_bundle_digest"
            ],
            "counts": summary["counts"],
            "artifacts": artifacts,
            "anchor_sha256": anchors,
            "artifact_digest": _object_digest(artifacts),
        }
        (stage / DONE_NAME).write_bytes(_json_bytes(done))

    _publish_directory(target, writer)
    return json.loads((target / SUMMARY_NAME).read_text(encoding="utf-8"))


def build_input(
    *, input_raw_dir: str | Path, output_jsonl: str | Path
) -> dict[str, Any]:
    """Create a deterministic normalized JSONL from raw ``combine_json``."""

    rows, raw_digest, binding_kind = load_mother_input(input_raw_dir)
    target = Path(output_jsonl).expanduser().resolve(strict=False)
    raw = _jsonl_bytes(rows)
    _write_create_only(target, raw)
    result = {
        "schema_version": NORMALIZED_SCHEMA,
        "rows": len(rows),
        "raw_input_sha256": raw_digest,
        "raw_input_binding_kind": binding_kind,
        "output_sha256": _sha256_bytes(raw),
        "output": str(target),
    }
    _write_create_only(
        target.with_suffix(target.suffix + ".summary.json"), _json_bytes(result)
    )
    return result


def _add_bound_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--old-selected", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-old-selected-sha256", required=True)


def _add_config(parser: argparse.ArgumentParser) -> None:
    defaults = SourceExpandConfig()
    for field, argument_type in (
        ("analysis_frames", int),
        ("resize_width", int),
        ("active_speed_threshold", float),
        ("required_frames", int),
        ("required_fps", float),
        ("fps_tolerance", float),
        ("min_short_side", int),
        ("min_pixels", int),
        ("min_duration_seconds", float),
        ("max_duration_seconds", float),
        ("min_residual_speed_p90", float),
        ("min_active_pixel_fraction", float),
        ("min_active_frame_fraction", float),
        ("min_actor_likeness", float),
        ("min_temporal_coverage", float),
        ("min_largest_component_share", float),
        ("max_spatial_energy_entropy", float),
    ):
        parser.add_argument(
            "--" + field.replace("_", "-"),
            type=argument_type,
            default=getattr(defaults, field),
        )


def _config_from_args(args: argparse.Namespace) -> SourceExpandConfig:
    return SourceExpandConfig(
        **{name: getattr(args, name) for name in SourceExpandConfig.__dataclass_fields__}
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh distributed Goku source census for atomic editing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalize = subparsers.add_parser("build-input")
    normalize.add_argument("--input-raw-dir", required=True, type=Path)
    normalize.add_argument("--output-jsonl", required=True, type=Path)
    worker = subparsers.add_parser("worker")
    _add_bound_inputs(worker)
    _add_config(worker)
    worker.add_argument("--video-root", required=True, type=Path)
    worker.add_argument("--work-dir", required=True, type=Path)
    worker.add_argument("--worker-index", required=True, type=int)
    worker.add_argument("--num-workers", required=True, type=int)
    worker.add_argument("--local-workers", type=int, default=8)
    final = subparsers.add_parser("finalize")
    _add_bound_inputs(final)
    _add_config(final)
    final.add_argument("--work-dir", required=True, type=Path)
    final.add_argument("--output-dir", required=True, type=Path)
    final.add_argument("--sample-size", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-input":
        result = build_input(
            input_raw_dir=args.input_raw_dir, output_jsonl=args.output_jsonl
        )
    elif args.command == "worker":
        result = run_worker(
            input_path=args.input,
            old_selected=args.old_selected,
            video_root=args.video_root,
            work_dir=args.work_dir,
            expected_input_sha256=args.expected_input_sha256,
            expected_old_selected_sha256=args.expected_old_selected_sha256,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
            local_workers=args.local_workers,
            config=_config_from_args(args),
        )
    else:
        result = finalize(
            input_path=args.input,
            old_selected=args.old_selected,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            expected_input_sha256=args.expected_input_sha256,
            expected_old_selected_sha256=args.expected_old_selected_sha256,
            sample_size=args.sample_size,
            config=_config_from_args(args),
        )
    print(_canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
