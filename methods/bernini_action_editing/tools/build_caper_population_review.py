#!/usr/bin/env python3
"""Build a fail-visible, synchronized review page for CAPER populations.

The tool is deliberately read-only with respect to both experiment roots.  It
enumerates cells from the sealed registries, validates attempt/cell/arm
receipts, hashes the declared MP4s, and asks ffprobe to count exact81/25fps
frames.  Missing and failed attempts remain in the report; no available seed
or duplicate target is substituted for a failed registered sibling arm.

The output HTML and adjacent JSON audit are the only files written.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


FRAME_COUNT = 81
FPS = 25
DURATION_SECONDS = FRAME_COUNT / FPS
NATIVE_ARM = "native-source-video-only-v2v-endpoint"
ARM_ORDER = ("target", "noop", "incomplete", "phase-order-violation")
NATIVE_ATTEMPT_SCHEMA = "bernini-caper-native-kseed-attempt-receipt-v1"
NATIVE_CELL_SCHEMA = "bernini-caper-native-kseed-cell-receipt-v1"
NATIVE_MASTER_SCHEMA = "bernini-caper-native-kseed-population-all8-receipt-v1"
SIBLING_ATTEMPT_SCHEMA = (
    "bernini-caper-native-counterfactual-sibling-attempt-receipt-v1"
)
SIBLING_CELL_SCHEMA = (
    "bernini-caper-native-counterfactual-sibling-cell-receipt-v1"
)
SIBLING_ARM_SCHEMA = (
    "bernini-caper-native-counterfactual-sibling-arm-receipt-v1"
)
SIBLING_MASTER_SCHEMA = (
    "bernini-caper-native-counterfactual-sibling-population-receipt-v1"
)
_CELL = re.compile(r"^(?P<split>[a-z]+)-(?P<source>[0-9a-f]+)-s(?P<seed>[0-9]+)$")


class CaperReviewError(RuntimeError):
    """Raised when a registry or command boundary is unusable."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CaperReviewError(f"not a plain file: {path}")
    return path.resolve(strict=True)


def _load_object(path: Path) -> dict[str, Any]:
    resolved = _plain_file(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CaperReviewError(f"cannot decode JSON: {path}") from error
    if not isinstance(value, dict):
        raise CaperReviewError(f"JSON root is not an object: {path}")
    return value


def _load_sealed(path: Path) -> tuple[dict[str, Any], str]:
    value = _load_object(path)
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(unsigned) != declared:
        raise CaperReviewError(f"receipt digest differs: {path}")
    return value, declared


def _parse_cell(cell_id: str) -> tuple[str, str, int]:
    matched = _CELL.fullmatch(cell_id)
    if matched is None:
        raise CaperReviewError(f"invalid registered cell id: {cell_id}")
    return matched.group("split"), matched.group("source"), int(matched.group("seed"))


def _status(
    state: str,
    message: str,
    *,
    path: Path | None = None,
    receipt: Path | None = None,
    probe: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"status": state, "message": message}
    if path is not None:
        row["path"] = str(path)
    if receipt is not None:
        row["receipt"] = str(receipt)
    if probe is not None:
        row["probe"] = dict(probe)
    if details:
        row["details"] = dict(details)
    return row


def probe_exact81_video(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Count the selected video stream with ffprobe and require exact81/25fps."""

    video = _plain_file(path)
    command = (
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video),
    )
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CaperReviewError(f"cannot execute ffprobe for {video}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise CaperReviewError(f"ffprobe failed for {video}: {detail}")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        streams = payload["streams"]
        stream = streams[0]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise CaperReviewError(f"ffprobe output differs for {video}") from error
    width = stream.get("width")
    height = stream.get("height")
    frames = stream.get("nb_read_frames")
    rate = stream.get("avg_frame_rate")
    if (
        len(streams) != 1
        or type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or frames != str(FRAME_COUNT)
        or rate != f"{FPS}/1"
    ):
        raise CaperReviewError(
            f"not exact81/25fps: {video} "
            f"(frames={frames!r}, fps={rate!r}, size={width!r}x{height!r})"
        )
    return {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "width": width,
        "height": height,
        "ffprobe_count_frames": True,
    }


class _MediaValidator:
    def __init__(self, *, ffprobe: str) -> None:
        self.ffprobe = ffprobe
        self._hashes: dict[Path, str] = {}
        self._probes: dict[Path, dict[str, Any]] = {}

    def validate(
        self,
        path: Path,
        *,
        expected_sha256: object,
        receipt: Path | None,
        label: str,
    ) -> dict[str, Any]:
        try:
            video = _plain_file(path)
        except CaperReviewError as error:
            return _status("missing", f"{label}: {error}", path=path, receipt=receipt)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            return _status(
                "invalid", f"{label}: missing declared SHA-256", path=video, receipt=receipt
            )
        try:
            observed = self._hashes.setdefault(video, file_sha256(video))
        except OSError as error:
            return _status(
                "invalid", f"{label}: cannot hash video: {error}", path=video, receipt=receipt
            )
        if observed != expected_sha256:
            return _status(
                "invalid",
                f"{label}: MP4 SHA-256 differs",
                path=video,
                receipt=receipt,
                details={"expected_sha256": expected_sha256, "observed_sha256": observed},
            )
        try:
            if video not in self._probes:
                self._probes[video] = probe_exact81_video(video, ffprobe=self.ffprobe)
        except CaperReviewError as error:
            return _status("invalid", f"{label}: {error}", path=video, receipt=receipt)
        return _status(
            "valid",
            f"{label}: receipt/hash/exact81/25fps verified",
            path=video,
            receipt=receipt,
            probe=self._probes[video],
            details={"sha256": observed},
        )


def _validate_native_registry(
    path: Path, *, phase: str
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    registry = _load_object(path)
    if registry.get("schema_version") != "bernini-caper-native-kseed-population-sit-v1":
        raise CaperReviewError("native registry schema differs")
    design = registry.get("population_design", {}).get(phase)
    if not isinstance(design, dict) or not isinstance(design.get("cell_order"), list):
        raise CaperReviewError(f"native registry has no phase {phase!r}")
    cells = list(design["cell_order"])
    if (
        len(cells) != len(set(cells))
        or design.get("expected_cell_count") != len(cells)
        or design.get("cartesian_population_required") is not True
        or design.get("seed_filtering_or_best_of_k_authorized") is not False
    ):
        raise CaperReviewError("native registry population closure differs")
    source_rows = registry.get("sources")
    if not isinstance(source_rows, list):
        raise CaperReviewError("native registry sources differ")
    sources = {
        str(row.get("source_id")): row
        for row in source_rows
        if isinstance(row, dict) and row.get("split") == phase
    }
    for cell in cells:
        split, source_id, _ = _parse_cell(cell)
        if split != phase or source_id not in sources:
            raise CaperReviewError(f"native registry cell/source closure differs: {cell}")
    return registry, cells, sources


def _validate_sibling_registry(
    path: Path, *, phase: str
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    registry = _load_object(path)
    if (
        registry.get("schema_version")
        != "bernini-caper-native-counterfactual-sibling-population-sit-v1"
        or registry.get("arm_order") != list(ARM_ORDER)
    ):
        raise CaperReviewError("sibling registry schema/arm order differs")
    design = registry.get("population_design")
    if not isinstance(design, dict) or not isinstance(design.get("cell_order"), list):
        raise CaperReviewError("sibling registry population differs")
    if design.get("split") != phase:
        raise CaperReviewError(
            f"sibling registry split {design.get('split')!r} does not match phase {phase!r}"
        )
    cells = list(design["cell_order"])
    if (
        len(cells) != len(set(cells))
        or design.get("expected_cell_count") != len(cells)
        or design.get("arms_per_cell") != len(ARM_ORDER)
        or design.get("cartesian_population_required") is not True
        or design.get("seed_filtering_or_best_of_k_authorized") is not False
        or design.get("replacement_seed_authorized") is not False
    ):
        raise CaperReviewError("sibling registry population closure differs")
    source = registry.get("source")
    if not isinstance(source, dict):
        raise CaperReviewError("sibling registry source differs")
    sources = {str(source.get("source_id")): source}
    for cell in cells:
        _, source_id, _ = _parse_cell(cell)
        if source_id not in sources:
            raise CaperReviewError(f"sibling cell/source closure differs: {cell}")
    return registry, cells, sources


def _attempt(
    root: Path,
    cell_id: str,
    *,
    schema: str,
    sibling: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = root / "attempts" / f"{cell_id}.json"
    if not path.is_file() or path.is_symlink():
        return _status("missing", "registered attempt receipt is absent", receipt=path), None
    try:
        row, digest = _load_sealed(path)
        if row.get("schema_version") != schema or row.get("cell_id") != cell_id:
            raise CaperReviewError("attempt schema/cell differs")
        _, source_id, seed = _parse_cell(cell_id)
        if row.get("source_id") != source_id or row.get("seed") != seed:
            raise CaperReviewError("attempt source/seed differs")
        if sibling and (
            row.get("arm_order") != list(ARM_ORDER)
            or row.get("expected_arm_count") != len(ARM_ORDER)
            or row.get("cell_process_attempt_recorded_even_on_failure") is not True
            or row.get("all_four_arm_outcomes_closed")
            is not (row.get("attempt_success") is True)
            or row.get("unobserved_or_incomplete_arm_outcomes_possible")
            is not (row.get("attempt_success") is not True)
        ):
            raise CaperReviewError("sibling attempt arm closure differs")
        if not sibling and row.get("seed_attempt_recorded_even_on_failure") is not True:
            raise CaperReviewError("native seed attempt closure differs")
        if (
            row.get("seed_discarded") is not False
            or row.get("retry_or_replacement_seed_authorized") is not False
        ):
            raise CaperReviewError("attempt permits seed selection/replacement")
        if row.get("attempt_success") is not True:
            return (
                _status(
                    "failed",
                    "registered attempt completed with failure",
                    receipt=path,
                    details={
                        "process_exit_code": row.get("process_exit_code"),
                        "attempt_status": row.get("attempt_status"),
                        "partial_or_complete_cell_artifacts": row.get(
                            "partial_or_complete_cell_artifacts", []
                        ),
                        "receipt_digest": digest,
                    },
                ),
                row,
            )
        if row.get("process_exit_code") != 0:
            raise CaperReviewError("successful attempt has nonzero exit code")
        return (
            _status(
                "valid",
                "registered attempt receipt verified",
                receipt=path,
                details={"receipt_digest": digest},
            ),
            row,
        )
    except (CaperReviewError, OSError) as error:
        return _status("invalid", str(error), receipt=path), None


def _pointer_matches(pointer: object, expected: Path) -> bool:
    return isinstance(pointer, str) and Path(pointer).name == expected.name


def _inspect_native_target(
    root: Path,
    cell_id: str,
    *,
    validator: _MediaValidator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_status, attempt_row = _attempt(
        root, cell_id, schema=NATIVE_ATTEMPT_SCHEMA, sibling=False
    )
    if attempt_status["status"] != "valid" or attempt_row is None:
        return attempt_status, {"attempt": attempt_status}
    cell_root = root / cell_id
    receipt_path = cell_root / "receipt.json"
    try:
        receipt, digest = _load_sealed(receipt_path)
        split, source_id, seed = _parse_cell(cell_id)
        sampling = receipt.get("sampling", {})
        input_row = receipt.get("input", {})
        if (
            receipt.get("schema_version") != NATIVE_CELL_SCHEMA
            or receipt.get("cell_id") != cell_id
            or receipt.get("population_phase") != split
            or receipt.get("seed_filtering_or_best_of_k_authorized") is not False
            or receipt.get("training_performed") is not False
            or receipt.get("optimizer_created") is not False
            or receipt.get("parameter_update") is not False
            or input_row.get("source_id") != source_id
            or sampling.get("seed") != seed
            or sampling.get("frame_count") != FRAME_COUNT
            or sampling.get("fps") != FPS
        ):
            raise CaperReviewError("native cell receipt boundary differs")
        if (
            attempt_row.get("child_receipt_digest") != digest
            or attempt_row.get("child_receipt_file_sha256") != file_sha256(receipt_path)
        ):
            raise CaperReviewError("native attempt/cell receipt binding differs")
        output = receipt.get("outputs", {}).get(NATIVE_ARM)
        if not isinstance(output, dict):
            raise CaperReviewError("native target output receipt is absent")
        expected = cell_root / f"{NATIVE_ARM}.mp4"
        if (
            not _pointer_matches(output.get("path"), expected)
            or output.get("frame_count") != FRAME_COUNT
            or output.get("fps") != FPS
        ):
            raise CaperReviewError("native target output pointer differs")
        media = validator.validate(
            expected,
            expected_sha256=output.get("sha256"),
            receipt=receipt_path,
            label="native K-seed target",
        )
        return media, {
            "attempt": attempt_status,
            "cell_receipt": _status(
                "valid",
                "native cell receipt verified",
                receipt=receipt_path,
                details={"receipt_digest": digest},
            ),
            "official_gaussian_raw_sha256": receipt.get("sampling", {}).get(
                "official_gaussian_raw_sha256"
            ),
        }
    except (CaperReviewError, OSError) as error:
        invalid = _status("invalid", str(error), receipt=receipt_path)
        return invalid, {"attempt": attempt_status, "cell_receipt": invalid}


def _inspect_sibling_cell(
    root: Path,
    cell_id: str,
    *,
    validator: _MediaValidator,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    attempt_status, attempt_row = _attempt(
        root, cell_id, schema=SIBLING_ATTEMPT_SCHEMA, sibling=True
    )
    if attempt_status["status"] != "valid" or attempt_row is None:
        return ({arm: dict(attempt_status) for arm in ARM_ORDER}, {"attempt": attempt_status})
    cell_root = root / cell_id
    receipt_path = cell_root / "receipt.json"
    try:
        receipt, digest = _load_sealed(receipt_path)
        _, source_id, seed = _parse_cell(cell_id)
        if (
            receipt.get("schema_version") != SIBLING_CELL_SCHEMA
            or receipt.get("cell_id") != cell_id
            or receipt.get("source_id") != source_id
            or receipt.get("seed") != seed
            or receipt.get("arm_order") != list(ARM_ORDER)
            or receipt.get("expected_arm_count") != len(ARM_ORDER)
            or receipt.get("complete_arm_count") != len(ARM_ORDER)
            or receipt.get("all_four_sibling_arms_complete") is not True
            or receipt.get("training_performed") is not False
            or receipt.get("optimizer_created") is not False
            or receipt.get("parameter_update") is not False
            or receipt.get("preference_admission_performed") is not False
            or receipt.get("partial_population_scientific_claim_authorized") is not False
        ):
            raise CaperReviewError("sibling cell receipt boundary differs")
        if (
            attempt_row.get("cell_receipt_digest") != digest
            or attempt_row.get("cell_receipt_file_sha256") != file_sha256(receipt_path)
        ):
            raise CaperReviewError("sibling attempt/cell receipt binding differs")
        pointers = receipt.get("arm_receipts")
        if (
            not isinstance(pointers, list)
            or [row.get("arm") for row in pointers if isinstance(row, dict)]
            != list(ARM_ORDER)
        ):
            raise CaperReviewError("sibling ordered arm receipt closure differs")
    except (CaperReviewError, OSError) as error:
        invalid = _status("invalid", str(error), receipt=receipt_path)
        return (
            {arm: dict(invalid) for arm in ARM_ORDER},
            {"attempt": attempt_status, "cell_receipt": invalid},
        )

    results: dict[str, dict[str, Any]] = {}
    gaussian: dict[str, Any] = {}
    arm_audits: dict[str, Any] = {}
    for arm_index, (arm, pointer) in enumerate(zip(ARM_ORDER, pointers)):
        arm_path = cell_root / f"{arm}.receipt.json"
        try:
            if (
                not _pointer_matches(pointer.get("path"), arm_path)
                or pointer.get("file_sha256") != file_sha256(arm_path)
            ):
                raise CaperReviewError(f"{arm} arm receipt pointer differs")
            arm_receipt, arm_digest = _load_sealed(arm_path)
            if pointer.get("receipt_digest") != arm_digest:
                raise CaperReviewError(f"{arm} arm receipt digest pointer differs")
            if (
                arm_receipt.get("schema_version") != SIBLING_ARM_SCHEMA
                or arm_receipt.get("cell_id") != cell_id
                or arm_receipt.get("arm") != arm
                or arm_receipt.get("arm_index") != arm_index
                or arm_receipt.get("source_id") != source_id
                or arm_receipt.get("seed") != seed
                or arm_receipt.get("training_performed") is not False
                or arm_receipt.get("optimizer_created") is not False
                or arm_receipt.get("parameter_update") is not False
                or arm_receipt.get("preference_admission_performed") is not False
            ):
                raise CaperReviewError(f"{arm} arm receipt boundary differs")
            sampling = arm_receipt.get("sampling", {})
            if (
                sampling.get("frame_count") != FRAME_COUNT
                or sampling.get("fps") != FPS
                or sampling.get("seed") != seed
            ):
                raise CaperReviewError(f"{arm} sampling receipt differs")
            output = arm_receipt.get("output")
            if not isinstance(output, dict):
                raise CaperReviewError(f"{arm} output receipt is absent")
            video = cell_root / f"{arm}.mp4"
            if (
                not _pointer_matches(output.get("path"), video)
                or output.get("frame_count") != FRAME_COUNT
                or output.get("fps") != FPS
            ):
                raise CaperReviewError(f"{arm} output pointer differs")
            results[arm] = validator.validate(
                video,
                expected_sha256=output.get("sha256"),
                receipt=arm_path,
                label=f"same-seed sibling {arm}",
            )
            gaussian[arm] = arm_receipt.get("sampling", {}).get(
                "official_gaussian_raw_sha256"
            )
            arm_audits[arm] = _status(
                "valid",
                f"{arm} arm receipt verified",
                receipt=arm_path,
                details={"receipt_digest": arm_digest},
            )
        except (CaperReviewError, OSError) as error:
            results[arm] = _status("invalid", str(error), receipt=arm_path)
            arm_audits[arm] = dict(results[arm])
    valid_gaussians = {value for value in gaussian.values() if isinstance(value, str)}
    shared_declared = receipt.get("shared_contract", {}).get(
        "official_gaussian_raw_sha256"
    )
    gaussian_status = (
        "valid"
        if len(gaussian) == len(ARM_ORDER)
        and len(valid_gaussians) == 1
        and shared_declared in valid_gaussians
        else "invalid"
    )
    if gaussian_status == "invalid":
        for arm in ARM_ORDER:
            if results[arm]["status"] == "valid":
                results[arm] = _status(
                    "invalid",
                    "same-seed sibling official Gaussian equality differs",
                    path=Path(results[arm]["path"]),
                    receipt=Path(results[arm]["receipt"]),
                    details={"observed_by_arm": gaussian, "declared": shared_declared},
                )
    return results, {
        "attempt": attempt_status,
        "cell_receipt": _status(
            "valid",
            "sibling cell receipt verified",
            receipt=receipt_path,
            details={"receipt_digest": digest},
        ),
        "arm_receipts": arm_audits,
        "same_official_gaussian": _status(
            gaussian_status,
            "four sibling arms share one official Gaussian value"
            if gaussian_status == "valid"
            else "four sibling arms do not expose one shared official Gaussian value",
            details={"observed_by_arm": gaussian, "declared": shared_declared},
        ),
    }


def _master_receipt(
    root: Path,
    *,
    filename: str,
    schema: str,
    expected_cells: Sequence[str],
    sibling: bool,
) -> dict[str, Any]:
    path = root / filename
    if not path.is_file() or path.is_symlink():
        return _status("missing", "population master receipt is absent", receipt=path)
    try:
        row, digest = _load_sealed(path)
        cell_order = row.get("cell_order") if sibling else row.get("registered_cell_order")
        if (
            row.get("schema_version") != schema
            or cell_order != list(expected_cells)
            or row.get("seed_filtering_or_best_of_k_authorized") is not False
        ):
            raise CaperReviewError("population master schema/cell order differs")
        state = "valid" if row.get("population_complete") is True else "failed"
        return _status(
            state,
            "complete population master receipt verified"
            if state == "valid"
            else "population master explicitly records an incomplete/failed population",
            receipt=path,
            details={
                "receipt_digest": digest,
                "population_decision": row.get("population_decision"),
                "failed_attempts": row.get("failed_attempts", []),
            },
        )
    except (CaperReviewError, OSError) as error:
        return _status("invalid", str(error), receipt=path)


def _source_path(source: Mapping[str, Any], source_root: Path | None) -> Path:
    source_id = str(source.get("source_id"))
    if source_root is not None:
        candidates = (
            source_root / source_id / "source_video.mp4",
            source_root / f"{source_id}.mp4",
            source_root / "samples" / source_id / "samples" / source_id / "source_video.mp4",
        )
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        return candidates[0]
    declared = source.get("source_video")
    return Path(declared) if isinstance(declared, str) else Path("missing-source-video")


def _ledger_extras(root: Path, expected_cells: Iterable[str]) -> list[str]:
    attempts = root / "attempts"
    if not attempts.is_dir() or attempts.is_symlink():
        return []
    expected = {f"{cell}.json" for cell in expected_cells}
    return sorted(
        path.name
        for path in attempts.glob("*.json")
        if path.is_file() and path.name not in expected
    )


def build_review(
    *,
    native_registry_path: Path,
    native_root: Path,
    sibling_registry_path: Path,
    sibling_root: Path,
    phase: str,
    output_html: Path,
    audit_json: Path | None = None,
    source_root: Path | None = None,
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Validate both registered populations and write HTML plus JSON audit."""

    native_registry_path = native_registry_path.expanduser().resolve()
    sibling_registry_path = sibling_registry_path.expanduser().resolve()
    native_root = native_root.expanduser().resolve()
    sibling_root = sibling_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve() if source_root is not None else None
    native_registry, native_cells, native_sources = _validate_native_registry(
        native_registry_path, phase=phase
    )
    sibling_registry, sibling_cells, sibling_sources = _validate_sibling_registry(
        sibling_registry_path, phase=phase
    )
    all_cells = list(native_cells)
    all_cells.extend(cell for cell in sibling_cells if cell not in set(all_cells))
    sources = dict(native_sources)
    sources.update(sibling_sources)
    validator = _MediaValidator(ffprobe=ffprobe)

    master = {
        "native": _master_receipt(
            native_root,
            filename=f"{phase}-population-receipt.json",
            schema=NATIVE_MASTER_SCHEMA,
            expected_cells=native_cells,
            sibling=False,
        ),
        "sibling": _master_receipt(
            sibling_root,
            filename="population-receipt.json",
            schema=SIBLING_MASTER_SCHEMA,
            expected_cells=sibling_cells,
            sibling=True,
        ),
    }
    extra_ledgers = {
        "native": _ledger_extras(native_root, native_cells),
        "sibling": _ledger_extras(sibling_root, sibling_cells),
    }

    cells: list[dict[str, Any]] = []
    native_set = set(native_cells)
    sibling_set = set(sibling_cells)
    source_cache: dict[str, dict[str, Any]] = {}
    for cell_id in all_cells:
        split, source_id, seed = _parse_cell(cell_id)
        source = sources.get(source_id)
        if source is None:
            raise CaperReviewError(f"no registry source for {cell_id}")
        if source_id not in source_cache:
            src = _source_path(source, source_root)
            source_cache[source_id] = validator.validate(
                src,
                expected_sha256=source.get("source_video_sha256"),
                receipt=None,
                label=f"registered source {source_id}",
            )
        source_media = dict(source_cache[source_id])

        if cell_id in native_set:
            native_target, native_audit = _inspect_native_target(
                native_root, cell_id, validator=validator
            )
        else:
            native_target = _status("not_registered", "cell is not in native registry")
            native_audit = {"attempt": dict(native_target)}

        if cell_id in sibling_set:
            sibling_roles, sibling_audit = _inspect_sibling_cell(
                sibling_root, cell_id, validator=validator
            )
            roles = {arm: dict(sibling_roles[arm]) for arm in ARM_ORDER}
            target_origin = "same-seed sibling target (fixed; no fallback)"
        else:
            roles = {
                "target": dict(native_target),
                "noop": _status(
                    "not_registered", "arm is not registered for this source/seed"
                ),
                "incomplete": _status(
                    "not_registered", "arm is not registered for this source/seed"
                ),
                "phase-order-violation": _status(
                    "not_registered", "arm is not registered for this source/seed"
                ),
            }
            sibling_audit = {"attempt": _status("not_registered", "cell not registered")}
            target_origin = "native K-seed target"

        duplicate_match: dict[str, Any]
        if cell_id in native_set and cell_id in sibling_set:
            native_gaussian = native_audit.get("official_gaussian_raw_sha256")
            sibling_gaussian = sibling_audit.get("same_official_gaussian", {}).get(
                "details", {}
            ).get("declared")
            duplicate_match = _status(
                "valid" if native_gaussian == sibling_gaussian and native_gaussian else "invalid",
                "native and sibling target coordinates declare the same official Gaussian"
                if native_gaussian == sibling_gaussian and native_gaussian
                else "native/sibling target official Gaussian binding differs or is unavailable",
                details={"native": native_gaussian, "sibling": sibling_gaussian},
            )
        else:
            duplicate_match = _status("not_registered", "no overlapping duplicate target")

        cells.append(
            {
                "cell_id": cell_id,
                "split": split,
                "source_id": source_id,
                "seed": seed,
                "actor_kind": source.get("actor_kind"),
                "identity_id": source.get("identity_id"),
                "scene_id": source.get("scene_id"),
                "source": source_media,
                "roles": roles,
                "target_origin": target_origin,
                "native_target_duplicate": native_target,
                "native_sibling_target_coordinate": duplicate_match,
                "audit": {"native": native_audit, "sibling": sibling_audit},
            }
        )

    required_statuses: list[str] = []
    for cell in cells:
        required_statuses.append(cell["source"]["status"])
        if cell["cell_id"] in sibling_set:
            required_statuses.extend(cell["roles"][arm]["status"] for arm in ARM_ORDER)
        elif cell["cell_id"] in native_set:
            required_statuses.append(cell["roles"]["target"]["status"])
    required_statuses.extend(master[name]["status"] for name in ("native", "sibling"))
    review_complete = (
        all(state == "valid" for state in required_statuses)
        and not extra_ledgers["native"]
        and not extra_ledgers["sibling"]
    )
    counts: dict[str, int] = {}
    for cell in cells:
        for media in (cell["source"], *cell["roles"].values()):
            counts[media["status"]] = counts.get(media["status"], 0) + 1

    audit: dict[str, Any] = {
        "schema_version": "bernini-caper-population-html-review-audit-v1",
        "review_complete": review_complete,
        "selection_policy": {
            "cell_order": "native_registry_order_then_sibling_only_registry_order",
            "sibling_target_source": "sibling_attempt_only",
            "fallback_to_native_target_when_sibling_attempt_fails": False,
            "seed_filtering_or_best_of_k": False,
            "all_missing_failed_invalid_not_registered_cards_visible": True,
        },
        "phase": phase,
        "native_registry": {
            "path": str(native_registry_path),
            "sha256": file_sha256(_plain_file(native_registry_path)),
            "schema_version": native_registry.get("schema_version"),
            "cell_order": native_cells,
        },
        "sibling_registry": {
            "path": str(sibling_registry_path),
            "sha256": file_sha256(_plain_file(sibling_registry_path)),
            "schema_version": sibling_registry.get("schema_version"),
            "cell_order": sibling_cells,
            "arm_order": list(ARM_ORDER),
        },
        "master_receipts": master,
        "unexpected_attempt_receipts": extra_ledgers,
        "status_counts": counts,
        "cells": cells,
    }
    audit["audit_digest"] = object_sha256(audit)

    output_html = output_html.expanduser().resolve()
    audit_json = (
        audit_json.expanduser().resolve()
        if audit_json is not None
        else output_html.with_name(f"{output_html.stem}.audit.json")
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    if audit_json.parent != output_html.parent:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        _render_html(audit, output_html, audit_json), encoding="utf-8"
    )
    audit_json.write_bytes(canonical_json_bytes(audit) + b"\n")
    return audit


def _url(output_html: Path, value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    try:
        relative = os.path.relpath(path, start=output_html.parent)
    except ValueError:
        return path.as_uri() if path.is_absolute() else quote(path.as_posix())
    return quote(Path(relative).as_posix(), safe="/:._-")


def _badge(state: str) -> str:
    return f'<span class="badge {html.escape(state)}">{html.escape(state.upper())}</span>'


def _media_card(
    *,
    title: str,
    media: Mapping[str, Any],
    group: str,
    output_html: Path,
    leader: bool = False,
    extra: str = "",
) -> str:
    state = str(media.get("status", "invalid"))
    message = html.escape(str(media.get("message", "no diagnostic")))
    receipt_url = _url(output_html, media.get("receipt"))
    receipt_link = (
        f'<a href="{html.escape(receipt_url)}">receipt</a>' if receipt_url else ""
    )
    head = (
        f'<div class="head"><h3>{html.escape(title)}</h3>'
        f'<div>{_badge(state)} {receipt_link}</div><p>{message}</p>{extra}</div>'
    )
    if state == "valid" and (video_url := _url(output_html, media.get("path"))):
        leader_attr = " data-leader" if leader else ""
        body = (
            f'<video data-group="{html.escape(group)}"{leader_attr} controls muted '
            f'playsinline preload="metadata" src="{html.escape(video_url)}"></video>'
        )
    else:
        details = media.get("details")
        detail_text = ""
        if isinstance(details, dict) and details:
            detail_text = html.escape(
                json.dumps(details, sort_keys=True, ensure_ascii=False, default=str)
            )
        body = (
            f'<div class="placeholder {html.escape(state)}"><strong>{html.escape(state.upper())}</strong>'
            f'<span>{message}</span><code>{detail_text}</code></div>'
        )
    return f'<article class="card {html.escape(state)}">{head}{body}</article>'


def _render_html(
    audit: Mapping[str, Any], output_html: Path, audit_json: Path
) -> str:
    cells = audit["cells"]
    sections: list[str] = []
    for cell in cells:
        group = cell["cell_id"]
        duplicate = cell["native_target_duplicate"]
        coordinate = cell["native_sibling_target_coordinate"]
        duplicate_url = _url(output_html, duplicate.get("path"))
        duplicate_receipt = _url(output_html, duplicate.get("receipt"))
        duplicate_links = []
        if duplicate_url:
            duplicate_links.append(f'<a href="{html.escape(duplicate_url)}">native target MP4</a>')
        if duplicate_receipt:
            duplicate_links.append(
                f'<a href="{html.escape(duplicate_receipt)}">native target receipt</a>'
            )
        target_extra = (
            '<p class="duplicate">Primary: '
            + html.escape(cell["target_origin"])
            + ". Independent native duplicate: "
            + _badge(str(duplicate["status"]))
            + " ".join(duplicate_links)
            + ". Coordinate: "
            + _badge(str(coordinate["status"]))
            + "</p>"
        )
        cards = [
            _media_card(
                title="Source",
                media=cell["source"],
                group=group,
                output_html=output_html,
                leader=True,
            ),
            _media_card(
                title="Target",
                media=cell["roles"]["target"],
                group=group,
                output_html=output_html,
                extra=target_extra,
            ),
            _media_card(
                title="No-op",
                media=cell["roles"]["noop"],
                group=group,
                output_html=output_html,
            ),
            _media_card(
                title="Incomplete",
                media=cell["roles"]["incomplete"],
                group=group,
                output_html=output_html,
            ),
            _media_card(
                title="Phase-order violation",
                media=cell["roles"]["phase-order-violation"],
                group=group,
                output_html=output_html,
            ),
        ]
        sections.append(
            '<section class="sample" id="'
            + html.escape(group)
            + '"><div class="sample-title"><div><h2>'
            + html.escape(group)
            + "</h2><p>source="
            + html.escape(cell["source_id"])
            + " · seed="
            + html.escape(str(cell["seed"]))
            + " · identity="
            + html.escape(str(cell.get("identity_id")))
            + " · scene="
            + html.escape(str(cell.get("scene_id")))
            + f'</p></div><button data-play="{html.escape(group)}">播放本 cell</button></div>'
            + '<div class="grid">'
            + "".join(cards)
            + "</div></section>"
        )

    master_rows = []
    for name, row in audit["master_receipts"].items():
        receipt_url = _url(output_html, row.get("receipt"))
        link = f'<a href="{html.escape(receipt_url)}">receipt</a>' if receipt_url else ""
        master_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{_badge(row['status'])}</td>"
            f"<td>{html.escape(row['message'])} {link}</td></tr>"
        )
    counts = " ".join(
        f"{html.escape(key)}={value}" for key, value in sorted(audit["status_counts"].items())
    )
    complete = bool(audit["review_complete"])
    verdict = "COMPLETE" if complete else "INCOMPLETE / FAIL-VISIBLE"
    verdict_class = "valid" if complete else "failed"
    audit_name = _url(output_html, str(audit_json)) or f"{output_html.stem}.audit.json"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAPER native K-seed + same-seed sibling review</title>
<style>
:root{{--bg:#070b12;--panel:#101824;--card:#090e16;--line:#293851;--text:#eef5ff;--muted:#a8b4c8;--ok:#55d69e;--bad:#ff7883;--miss:#ffd06f;--blue:#7eb5ff}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#17243a 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif}}main{{max-width:2000px;margin:auto;padding:20px}}.hero,.sample,.evidence{{border:1px solid var(--line);border-radius:16px;background:rgba(16,24,36,.96);padding:17px;margin-bottom:17px}}h1,h2,h3,p{{margin-top:0}}h1{{margin-bottom:7px;font-size:clamp(25px,3vw,38px)}}h2{{font-size:18px;margin-bottom:4px}}h3{{font-size:15px;margin-bottom:7px}}a{{color:#9ac4ff}}.muted,.head p,.sample-title p{{color:var(--muted)}}.controls{{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:14px 0;padding:10px;border:1px solid var(--line);border-radius:12px;background:rgba(7,11,18,.95);backdrop-filter:blur(10px)}}button,select{{border:1px solid #3a4e6d;border-radius:8px;background:#17263b;color:var(--text);padding:7px 10px;cursor:pointer}}button.primary{{background:#215da8}}input[type=range]{{width:min(420px,48vw)}}.sample-title{{display:flex;justify-content:space-between;align-items:start;gap:12px}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:10px}}.card{{overflow:hidden;border:1px solid var(--line);border-radius:11px;background:var(--card)}}.card.invalid,.card.failed{{border-color:#793d48}}.card.missing{{border-color:#786333}}.head{{padding:10px;min-height:144px}}.head p{{font-size:12px;margin:7px 0 0}}.duplicate{{font-size:11px!important;border-top:1px solid #25344a;padding-top:7px}}video{{display:block;width:100%;aspect-ratio:31/30;object-fit:contain;background:#000}}.placeholder{{aspect-ratio:31/30;display:flex;flex-direction:column;justify-content:center;gap:8px;padding:15px;background:#0b111b;text-align:center}}.placeholder span{{color:var(--muted)}}.placeholder code{{max-height:120px;overflow:auto;white-space:pre-wrap;text-align:left;font-size:10px}}.badge{{display:inline-block;padding:2px 7px;border:1px solid var(--line);border-radius:999px;font-size:10px}}.badge.valid{{color:var(--ok);border-color:#286d53}}.badge.failed,.badge.invalid{{color:var(--bad);border-color:#7d3b47}}.badge.missing,.badge.not_registered{{color:var(--miss);border-color:#796431}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid var(--line);padding:8px;text-align:left}}@media(max-width:1500px){{.grid{{grid-template-columns:repeat(3,minmax(220px,1fr))}}}}@media(max-width:850px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:560px){{main{{padding:9px}}.grid{{grid-template-columns:1fr}}.head{{min-height:0}}}}
</style></head><body><main>
<section class="hero"><h1>CAPER population review</h1><p class="muted">Native K-seed + registered same-source/same-seed siblings · exact81 / 25 fps · fixed registry order · no best-of-K</p><p>{_badge(verdict_class)} <strong>{verdict}</strong> · {html.escape(counts)} · <a href="{html.escape(audit_name)}">machine-readable audit</a></p><p>Sibling-covered cells always use the sibling target. A failed sibling attempt is never replaced by the independently rendered native target; that duplicate remains visibly audited on the target card.</p>
<div class="controls"><button class="primary" id="play-all">播放全部有效视频</button><button id="pause-all">全部暂停</button><button id="reset-all">全部回到 0</button><label>速度 <select id="rate"><option>.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label><input id="seek" type="range" min="0" max="{DURATION_SECONDS:.2f}" step="0.01" value="0"><span id="time">0.00 / {DURATION_SECONDS:.2f} s</span></div></section>
{"".join(sections)}
<section class="evidence"><h2>Population receipts</h2><table><thead><tr><th>population</th><th>status</th><th>evidence</th></tr></thead><tbody>{"".join(master_rows)}</tbody></table><p>Unexpected native attempt receipts: {html.escape(json.dumps(audit['unexpected_attempt_receipts']['native']))}<br>Unexpected sibling attempt receipts: {html.escape(json.dumps(audit['unexpected_attempt_receipts']['sibling']))}</p></section>
</main><script>
const videos=[...document.querySelectorAll('video')],seek=document.getElementById('seek'),time=document.getElementById('time'),rate=document.getElementById('rate');let active=[];let leader=null;const limit={DURATION_SECONDS:.2f};
const clamp=x=>Math.max(0,Math.min(limit,Number(x)||0));const pauseAll=()=>{{videos.forEach(v=>v.pause());active=[];leader=null}};
const ready=v=>v.readyState>=1?Promise.resolve():new Promise(resolve=>{{const done=()=>resolve();v.addEventListener('loadedmetadata',done,{{once:true}});v.addEventListener('error',done,{{once:true}})}});
async function seekOne(v,t){{await ready(v);if(Number.isFinite(v.duration))t=Math.min(t,Math.max(0,v.duration-.001));try{{v.currentTime=t}}catch(_e){{}}}}
async function playGroup(group){{pauseAll();active=group==='all'?videos:videos.filter(v=>v.dataset.group===group);const t=clamp(seek.value);await Promise.all(active.map(v=>seekOne(v,t)));active.forEach(v=>v.playbackRate=Number(rate.value));leader=active[0]||null;await Promise.allSettled(active.map(v=>v.play()))}}
document.getElementById('play-all').onclick=()=>playGroup('all');document.querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>playGroup(b.dataset.play));document.getElementById('pause-all').onclick=pauseAll;document.getElementById('reset-all').onclick=async()=>{{pauseAll();await Promise.all(videos.map(v=>seekOne(v,0)));seek.value=0;time.textContent=`0.00 / ${{limit.toFixed(2)}} s`}};rate.onchange=()=>videos.forEach(v=>v.playbackRate=Number(rate.value));seek.oninput=async()=>{{pauseAll();const t=clamp(seek.value);await Promise.all(videos.map(v=>seekOne(v,t)));time.textContent=`${{t.toFixed(2)}} / ${{limit.toFixed(2)}} s`}};
videos.forEach(v=>v.addEventListener('timeupdate',()=>{{if(v!==leader||v.paused)return;const t=clamp(v.currentTime);seek.value=t;time.textContent=`${{t.toFixed(2)}} / ${{limit.toFixed(2)}} s`;active.forEach(other=>{{if(other!==v&&!other.seeking&&Math.abs(other.currentTime-t)>.08)other.currentTime=t}})}}));
</script></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-registry", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--sibling-registry", type=Path, required=True)
    parser.add_argument("--sibling-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("fit", "lockbox"), default="fit")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return zero after writing a fail-visible incomplete page",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = build_review(
        native_registry_path=args.native_registry,
        native_root=args.native_root,
        sibling_registry_path=args.sibling_registry,
        sibling_root=args.sibling_root,
        phase=args.phase,
        output_html=args.output,
        audit_json=args.audit_json,
        source_root=args.source_root,
        ffprobe=args.ffprobe,
    )
    print(canonical_json_bytes({
        "review_complete": audit["review_complete"],
        "status_counts": audit["status_counts"],
        "cell_count": len(audit["cells"]),
        "audit_digest": audit["audit_digest"],
    }).decode("ascii"))
    return 0 if audit["review_complete"] or args.allow_incomplete else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "CaperReviewError",
    "FRAME_COUNT",
    "FPS",
    "build_review",
    "canonical_json_bytes",
    "file_sha256",
    "main",
    "object_sha256",
    "probe_exact81_video",
]
