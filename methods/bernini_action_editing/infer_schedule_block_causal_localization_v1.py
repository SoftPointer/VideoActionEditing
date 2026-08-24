#!/usr/bin/env python3
"""One-load Stage-A schedule x block decoded prompt-swap localization.

The formal profile loads one frozen Bernini renderer, executes the fixed
six-video C0 engineering gate, and only after that gate passes continues in
the same process/model load to the preregistered 112-video grid.  Every query
is stateless: no sampler or denoising scheduler is instantiated or stepped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, NoReturn, Optional, Sequence
import weakref


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy_infer  # noqa: E402
import infer_source_noised_carrier_stage_b_v1 as stageb_infer  # noqa: E402
import inference_sigma_strata as exact40  # noqa: E402
import schedule_block_target_row_prompt_swap_v1 as swap  # noqa: E402
import source_noised_ladder_v1 as ladder  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
import train_lora as legacy_train  # noqa: E402


RECEIPT_SCHEMA = "bernini-schedule-block-causal-localization-runtime-v1"
C0_GATE_SCHEMA = "bernini-schedule-block-causal-localization-c0-engineering-gate-v1"
TOPOLOGY_SCHEMA = "bernini-schedule-block-causal-localization-topology-v1"
PROFILES = ("smoke-only", "smoke-then-full-fixed")
FORMAL_PROFILE = "smoke-then-full-fixed"
IID = "00435ad621c44fac"
SEED = 2026081401
SP_SIZE = 4
FPS = 25
EXPECTED_HW = (592, 400)
EXPECTED_LATENT_SHAPE = (1, 16, 21, 74, 50)
EXPECTED_FRAMES = 81
EXPECTED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
SOURCE_VIDEO_SHA256 = "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1"
SOURCE_DATASET_SPEC_SHA256 = "62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920"
SOURCE_DATASET_PARQUET_SHA256 = "77d89b3ec2e563f624bab62451b49b616ffa7f7890db6105c4458617aac0d106"
SOURCE_DATASET_RECEIPT_SHA256 = "6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb"
SOURCE_DATASET_RECEIPT_DIGEST = "12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738"
ORBIT_DATASET_SPEC_SHA256 = "72c0f104b123a1b7ad69f32697a0b7f7e8c2fdf766c951f3c0bed7518f0f564f"
ORBIT_DATASET_SPEC_DIGEST = "25522068a18893afbc21f54a7851dbf641bc10ea7229653cdfe0c772be1f934e"
ORBIT_REFERENCE_ENCODING_CONTRACT_DIGEST = (
    "181e93b1620cafce7de3806b334b6bfdd8e24aa633119cbd6506f3761175a269"
)
ORBIT_DATASET_PARQUET_SHA256 = "845727b8e9c461b9cf1f8bb98c0e27519599ffffcd6619bd8895250a2e075baf"
ORBIT_DATASET_RECEIPT_SHA256 = "c088eb0128c3c807941f60eb3e763d0e71f4c8dbb190c60b9c0dad6caeca0230"
ORBIT_DATASET_RECEIPT_DIGEST = "9000dd9dace16501587196ac8459b620529301508ee6c98662f266b3b29b8982"
ORBIT_REVIEW_SHA256 = "dc2d83322357196cec84418ddf4318d9fc7d1eb41269cb216739bae7c6169651"
PROMPT_AUTHORITY_SHA256 = swap.PROMPT_AUTHORITY_SHA256
PROMPT_ROW_DIGEST = "5da2592528633a4886d4f06946ba700ff69827c445748080de7de07e2b365245"
CALIBRATION_INITIAL_GAUSSIAN_RAW_SHA256 = (
    "758f13eb0606564707f741fec6da857650089ba3a9ba52867c1b690485c35f62"
)
CALIBRATION_SEED = 2026080821
SOURCE_DATASET_SPEC_DIGEST = "de2f92f314da538f8af322a8f1db23cbdf1feab4b28d2da66248d25309a25595"
PROMPT_SHA256 = MappingProxyType({
    "noop": "8b88abdd980fbb6cff3397492fe647da8fa2fdd95b75346449e29fc257d6ffdc",
    "forward": "b8b3f19e854c8c517549cdaf319af3cf5f07b719444c5468b5081dbf1507ded7",
    "reverse": "07de07b8afb40c8872cdbc38544875eda3f010622350c2333f5bc398338f15a9",
    "incomplete": "3dfba50307dba421f1eebe140008fd18e6ae80143d3caffd697bc0cc36ad6534",
    "camera_only": "8877d8196c5f121a02fdc20ea3070957fe6303fcba3ce51fc73f0ab3ce4572bf",
    "appearance_only": "10e45fefc4d1fb51cee22f2c8b740809c025429bd254b3f484af8b2fc842d50a",
})
PROMPT_UTF8_BYTES = MappingProxyType({
    "noop": 390, "forward": 379, "reverse": 379,
    "incomplete": 389, "camera_only": 393, "appearance_only": 378,
})
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RNG_WORLD4_VECTOR_SCHEMA = "stage-a-capture-rng-world4-rank-vector-v1"


class StageARuntimeError(RuntimeError):
    """Raised before incomplete or ambiguous Stage-A evidence is accepted."""


def fail(message: str) -> NoReturn:
    raise StageARuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise StageARuntimeError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class StableFileSnapshot:
    path: Path
    sha256: str
    size: int
    identity: tuple[int, int, int, int, int]
    raw: Optional[bytes] = None


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns), stat.S_IMODE(value.st_mode),
    )


def stable_file_snapshot(
    path_value: str | Path, *, label: str, retain_bytes: bool = False,
) -> StableFileSnapshot:
    """Read/hash a canonical regular file and reject any concurrent mutation."""

    path = _plain_absolute_file(path_value, label=label)
    before = path.stat()
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if _stat_identity(opened) != _stat_identity(before):
                fail(f"{label} identity changed before read")
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                if retain_bytes:
                    chunks.append(block)
            after_open = os.fstat(handle.fileno())
    except OSError as error:
        raise StageARuntimeError(f"cannot snapshot {label}: {error}") from error
    after = path.lstat()
    identity = _stat_identity(before)
    if (
        not stat.S_ISREG(after.st_mode)
        or path.is_symlink()
        or identity != _stat_identity(after_open)
        or identity != _stat_identity(after)
    ):
        fail(f"{label} changed while hashing")
    raw = b"".join(chunks) if retain_bytes else None
    if raw is not None and len(raw) != int(before.st_size):
        fail(f"{label} byte length changed while hashing")
    return StableFileSnapshot(path, digest.hexdigest(), int(before.st_size), identity, raw)


def file_sha256(path: Path) -> str:
    return stable_file_snapshot(path, label=str(path)).sha256


def _sha(value: Any, *, length: int = 64, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA digest")
    return value


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    snapshot = stable_file_snapshot(path, label=label, retain_bytes=True)
    if snapshot.raw is None:
        fail(f"{label} snapshot bytes were not retained")
    return _strict_json_bytes(snapshot.raw, label=label)


def _strict_json_bytes(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def no_constant(value: str) -> None:
        fail(f"{label} contains non-finite {value}")

    def no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"), parse_constant=no_constant,
            object_pairs_hook=no_duplicate,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageARuntimeError(f"cannot read {label}: {error}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} must be one object")
    return value


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"{label} must be a canonical plain file")
    return resolved


def _plain_absolute_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISDIR(resolved.lstat().st_mode):
        fail(f"{label} must be a canonical plain directory")
    return resolved


def _embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = _sha(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if object_sha256(unsigned) != declared:
        fail(f"{label} embedded digest differs")
    return declared


def _sealed_root_files(root_value: str | Path, *, label: str) -> tuple[Path, StableFileSnapshot, StableFileSnapshot]:
    root = _plain_absolute_directory(root_value, label=label)
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != {"dataset.parquet", "receipt.json"}:
        fail(f"{label} closure differs")
    parquet = stable_file_snapshot(entries["dataset.parquet"], label=f"{label} parquet")
    receipt = stable_file_snapshot(
        entries["receipt.json"], label=f"{label} receipt", retain_bytes=True,
    )
    return root, parquet, receipt


def _receipt_from_snapshot(snapshot: StableFileSnapshot, *, label: str) -> Mapping[str, Any]:
    if snapshot.raw is None:
        fail(f"{label} snapshot bytes were not retained")
    return _strict_json_bytes(snapshot.raw, label=label)


def _dataset_record_gate(
    record: Any, *, parquet: StableFileSnapshot, root: Path, label: str,
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        fail(f"{label} dataset record differs")
    raw_path = record.get("path")
    if type(raw_path) is not str or raw_path != str(parquet.path):
        fail(f"{label} dataset canonical path string differs")
    try:
        declared_path = Path(raw_path).resolve(strict=True)
    except OSError as error:
        raise StageARuntimeError(f"{label} dataset path is unavailable") from error
    if (
        declared_path != parquet.path or declared_path.parent != root
        or record.get("sha256") != parquet.sha256
        or type(record.get("rows")) is not int or int(record["rows"]) <= 0
        or not isinstance(record.get("iids"), list)
        or len(record["iids"]) != int(record["rows"])
        or len(set(record["iids"])) != len(record["iids"])
        or record["iids"].count(IID) != 1
    ):
        fail(f"{label} dataset path/SHA/IID closure differs")
    return record


def sealed_source_self_identity(root_value: str | Path) -> Mapping[str, Any]:
    """Validate the source-self pair as cross-authority without loading latents."""

    label = "source-self cross-authority dataset"
    root, parquet, receipt_snapshot = _sealed_root_files(root_value, label=label)
    if parquet.sha256 != SOURCE_DATASET_PARQUET_SHA256 or receipt_snapshot.sha256 != SOURCE_DATASET_RECEIPT_SHA256:
        fail(f"{label} external file SHA differs")
    receipt = _receipt_from_snapshot(receipt_snapshot, label=f"{label} receipt")
    if (
        receipt.get("schema_version") != "bernini-source-self-role-repaint-dataset-receipt-v2"
        or receipt.get("complete") is not True
        or _embedded_digest(receipt, label=label) != SOURCE_DATASET_RECEIPT_DIGEST
    ):
        fail(f"{label} receipt contract differs")
    spec = receipt.get("spec")
    if (
        not isinstance(spec, Mapping)
        or spec.get("sha256") != SOURCE_DATASET_SPEC_SHA256
        or spec.get("digest") != SOURCE_DATASET_SPEC_DIGEST
        or "file_sha256" in spec
    ):
        fail(f"{label} materialization spec differs")
    if set(spec) != {"path", "sha256", "digest"}:
        fail(f"{label} materialization spec key closure differs")
    live_spec_path = _plain_absolute_file(spec["path"], label=f"{label} live spec")
    live_spec_snapshot = stable_file_snapshot(
        live_spec_path, label=f"{label} live spec", retain_bytes=True
    )
    if live_spec_snapshot.sha256 != SOURCE_DATASET_SPEC_SHA256:
        fail(f"{label} live materialization spec SHA differs")
    if live_spec_snapshot.raw is None:
        fail(f"{label} live spec bytes absent")
    live_spec = _strict_json_bytes(live_spec_snapshot.raw, label=f"{label} live spec")
    live_unsigned = dict(live_spec)
    live_digest = live_unsigned.pop("spec_digest", None)
    if live_digest != SOURCE_DATASET_SPEC_DIGEST or object_sha256(live_unsigned) != live_digest:
        fail(f"{label} live materialization spec digest differs")
    dataset = _dataset_record_gate(receipt.get("dataset"), parquet=parquet, root=root, label=label)
    # Column projection is deliberate: source-self posterior/blob columns are
    # never read, decoded, or passed to the model in this experiment.
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            parquet.path, columns=["iid", "source_video_sha256", "row_digest"]
        )
        rows = table.to_pylist()
    except Exception as error:
        raise StageARuntimeError(f"cannot read source-self authority columns: {error}") from error
    after = stable_file_snapshot(parquet.path, label=f"{label} parquet after projection")
    if after.identity != parquet.identity or after.sha256 != parquet.sha256:
        fail(f"{label} parquet changed during authority projection")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("iid") == IID]
    if (
        len(rows) != dataset["rows"] or len(selected) != 1
        or selected[0].get("source_video_sha256") != SOURCE_VIDEO_SHA256
    ):
        fail(f"{label} selected IID/source video differs")
    return {
        "root": str(root), "parquet_sha256": SOURCE_DATASET_PARQUET_SHA256,
        "receipt_sha256": SOURCE_DATASET_RECEIPT_SHA256,
        "receipt_digest": SOURCE_DATASET_RECEIPT_DIGEST,
        "materialization_spec_sha256": SOURCE_DATASET_SPEC_SHA256,
        "materialization_spec_digest": SOURCE_DATASET_SPEC_DIGEST,
        "iid": IID, "source_video_sha256": SOURCE_VIDEO_SHA256,
        "row_digest": selected[0].get("row_digest"),
        "projected_columns": ["iid", "source_video_sha256", "row_digest"],
        "posterior_blob_columns_read": [], "latents_consumed": False,
    }


def sealed_orbit_identity(root_value: str | Path) -> Mapping[str, Any]:
    label = "orbit model-input dataset"
    root, parquet, receipt_snapshot = _sealed_root_files(root_value, label=label)
    if parquet.sha256 != ORBIT_DATASET_PARQUET_SHA256 or receipt_snapshot.sha256 != ORBIT_DATASET_RECEIPT_SHA256:
        fail(f"{label} external file SHA differs")
    receipt = _receipt_from_snapshot(receipt_snapshot, label=f"{label} receipt")
    if (
        receipt.get("schema_version")
        != "bernini-appearance-counterfactual-identity-orbit-dataset-receipt-v3"
        or receipt.get("complete") is not True
        or _embedded_digest(receipt, label=label) != ORBIT_DATASET_RECEIPT_DIGEST
    ):
        fail(f"{label} receipt contract differs")
    spec = receipt.get("spec")
    if (
        not isinstance(spec, Mapping)
        or spec.get("file_sha256") != ORBIT_DATASET_SPEC_SHA256
        or spec.get("digest") != ORBIT_DATASET_SPEC_DIGEST
        or spec.get("reference_encoding_contract_digest")
        != ORBIT_REFERENCE_ENCODING_CONTRACT_DIGEST
        or "sha256" in spec
    ):
        fail(f"{label} materialization spec differs")
    if set(spec) != {
        "path", "file_sha256", "digest", "reference_encoding_contract_digest"
    }:
        fail(f"{label} materialization spec key closure differs")
    live_spec_path = _plain_absolute_file(spec["path"], label=f"{label} live spec")
    if stable_file_snapshot(live_spec_path, label=f"{label} live spec").sha256 != ORBIT_DATASET_SPEC_SHA256:
        fail(f"{label} live materialization spec SHA differs")
    dataset = _dataset_record_gate(receipt.get("dataset"), parquet=parquet, root=root, label=label)
    pinned_vae = receipt.get("pinned_vae_identity")
    if not isinstance(pinned_vae, Mapping):
        fail(f"{label} pinned VAE identity differs")
    unsigned_vae = dict(pinned_vae)
    pinned_vae_digest = _sha(
        unsigned_vae.pop("vae_identity_digest", None),
        label=f"{label} pinned VAE identity digest",
    )
    if object_sha256(unsigned_vae) != pinned_vae_digest:
        fail(f"{label} pinned VAE identity digest differs")
    row_digests = dataset.get("row_digests")
    if not isinstance(row_digests, list) or len(row_digests) != dataset["rows"]:
        fail(f"{label} row digest closure differs")
    selected_index = dataset["iids"].index(IID)
    if row_digests[selected_index] != swap.ORBIT_ROW_DIGEST:
        fail(f"{label} IID00435 row digest differs")
    return {
        "root": str(root), "parquet_sha256": ORBIT_DATASET_PARQUET_SHA256,
        "receipt_sha256": ORBIT_DATASET_RECEIPT_SHA256,
        "receipt_digest": ORBIT_DATASET_RECEIPT_DIGEST,
        "materialization_spec_sha256": ORBIT_DATASET_SPEC_SHA256,
        "materialization_spec_digest": ORBIT_DATASET_SPEC_DIGEST,
        "reference_encoding_contract_digest": ORBIT_REFERENCE_ENCODING_CONTRACT_DIGEST,
        "iid": IID, "row_digest": swap.ORBIT_ROW_DIGEST,
        "pinned_vae_identity_digest": pinned_vae_digest,
        "all_target_and_owner_latents_from_orbit_row": True,
    }


def load_prompt_authority() -> Mapping[str, Any]:
    path = _plain_absolute_file(
        METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json",
        label="prompt authority",
    )
    if file_sha256(path) != PROMPT_AUTHORITY_SHA256:
        fail("prompt authority SHA differs")
    value = _strict_json(path, label="prompt authority")
    rows = value.get("cells")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("iid") == IID] if isinstance(rows, list) else []
    if len(selected) != 1:
        fail("IID00435 prompt authority row differs")
    row = selected[0]
    if object_sha256(row) != PROMPT_ROW_DIGEST:
        fail("IID00435 prompt row digest differs")
    descriptions = row.get("branch_descriptions")
    if not isinstance(descriptions, Mapping):
        fail("prompt branch descriptions differ")
    prompts: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for branch in swap.TEXT_BRANCHES:
        authoring_key = swap.PROMPT_AUTHORITY_MAPPING[branch]
        text = " ".join(
            (str(row["scene_caption"]).strip(), str(descriptions[authoring_key]).strip(), str(row["camera_caption"]).strip())
        )
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != PROMPT_SHA256[branch] or len(text.encode("utf-8")) != PROMPT_UTF8_BYTES[branch]:
            fail(f"{branch} prompt bytes differ")
        prompts[branch] = text
        mapping[branch] = authoring_key
    return {
        "path": str(path), "sha256": PROMPT_AUTHORITY_SHA256,
        "iid": IID, "row_digest": PROMPT_ROW_DIGEST,
        "branch_to_authoring_key": mapping, "prompts": prompts,
        "prompt_sha256": dict(PROMPT_SHA256),
        "forward_mapping": "forward<-branch_descriptions.action",
    }


def load_orbit_review_authority() -> Mapping[str, Any]:
    path = _plain_absolute_file(
        METHOD_ROOT / "assets/appearance_identity_orbit_portrait2_review_v1.json",
        label="orbit review authority",
    )
    snapshot = stable_file_snapshot(path, label="orbit review authority", retain_bytes=True)
    if snapshot.sha256 != ORBIT_REVIEW_SHA256:
        fail("orbit review authority SHA differs")
    if snapshot.raw is None:
        fail("orbit review authority snapshot bytes were not retained")
    value = _strict_json_bytes(snapshot.raw, label="orbit review authority")
    rows = value.get("rows")
    selected = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("iid") == IID
    ] if isinstance(rows, list) else []
    if len(selected) != 1 or value.get("downstream_training_results_seen") is not False:
        fail("orbit review IID/blinding contract differs")
    row = selected[0]
    source = row.get("source")
    variant = row.get("variant_a")
    gates = row.get("qualification_gates")
    if (
        not isinstance(source, Mapping)
        or source.get("video_sha256") != SOURCE_VIDEO_SHA256
        or not isinstance(variant, Mapping)
        or variant.get("video_sha256") != swap.WRONG_OWNER_VIDEO_SHA256
        or variant.get("native_arm") != "rv2v"
        or variant.get("native_receipt_file_sha256")
        != "a5633eb1a264dde233e8327d3d13337aea651f77d2cf5fe6a6b97dec298d817f"
        or variant.get("native_receipt_digest")
        != "0fc1467300baf57db990d3abf687e43059b17d0042c3c7744bbb68e84ecacf0c"
        or not isinstance(gates, Mapping)
    ):
        fail("orbit review source/variant authority differs")
    expected_gate_names = {
        "appearance_identity_changed_from_source",
        "motion_phase_and_order_preserved", "camera_path_preserved",
        "background_scene_preserved", "object_correspondence_preserved",
        "temporal_quality_passed", "spatial_quality_passed",
        "no_extra_actor_or_object",
    }
    for member in ("variant_a", "variant_b"):
        member_gates = gates.get(member)
        if (
            not isinstance(member_gates, Mapping)
            or set(member_gates) != expected_gate_names
            or any(item is not True for item in member_gates.values())
        ):
            fail(f"orbit review {member} qualification differs")
    cross = gates.get("cross_member")
    if not isinstance(cross, Mapping) or not cross or any(item is not True for item in cross.values()):
        fail("orbit review cross-member qualification differs")
    return {
        "path": str(path), "sha256": snapshot.sha256, "iid": IID,
        "source_video_sha256": SOURCE_VIDEO_SHA256,
        "wrong_owner_variant": swap.WRONG_OWNER_VARIANT,
        "wrong_owner_video_sha256": swap.WRONG_OWNER_VIDEO_SHA256,
        "wrong_owner_native_arm": "rv2v",
        "wrong_owner_native_receipt_file_sha256": variant["native_receipt_file_sha256"],
        "wrong_owner_native_receipt_digest": variant["native_receipt_digest"],
        "all_qualification_gates_true": True,
    }


def expected_artifact_names(profile: str) -> tuple[str, ...]:
    if profile not in PROFILES:
        fail("profile differs")
    names = ["c0-plan.json"]
    names.extend(f"c0/{cell.output_name}" for cell in swap.c0_smoke_cells())
    if profile == FORMAL_PROFILE:
        names.append("full-plan.json")
        names.extend(f"full/{cell.output_name}" for cell in swap.full_grid_cells())
    return tuple(names)


_COUNTERS = (
    "base_calls", "alternate_calls", "capture_calls", "plain_noop_calls",
    "noop_swap_selected_calls", "noop_swap_unselected_calls",
    "mixed_selected_calls", "mixed_unselected_calls", "no_context_calls",
    "non_target_parity_checks", "noop_full_parity_checks",
)


def expected_processor_deltas(phase: str, band_name: str) -> tuple[Mapping[str, int], ...]:
    selected = set(swap.band_blocks(band_name))
    rows: list[Mapping[str, int]] = []
    for index in range(swap.TOTAL_BLOCKS):
        row = {name: 0 for name in _COUNTERS}
        row["base_calls"] = 1
        if phase == "capture":
            row["capture_calls"] = 1
        elif phase == "plain_noop":
            row["plain_noop_calls"] = 1
        elif phase == "noop_swap":
            if index in selected:
                row.update(base_calls=2, alternate_calls=1, noop_swap_selected_calls=1, non_target_parity_checks=1, noop_full_parity_checks=1)
            else:
                row["noop_swap_unselected_calls"] = 1
        elif phase == "mixed":
            if index in selected:
                row.update(base_calls=2, alternate_calls=1, mixed_selected_calls=1, non_target_parity_checks=1)
            else:
                row["mixed_unselected_calls"] = 1
        else:
            fail("processor phase differs")
        rows.append(row)
    return tuple(rows)


def audit_processor_delta(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
    *, phase: str, band_name: str, owner_binding_digest: str, owner: str,
    schedule_index: int, branch: str, execution_id: str,
) -> Mapping[str, Any]:
    expected = expected_processor_deltas(phase, band_name)
    if len(before) != swap.TOTAL_BLOCKS or len(after) != swap.TOTAL_BLOCKS:
        fail("processor inventory is not exact all30")
    observed: list[dict[str, int]] = []
    for index, (left, right, wanted) in enumerate(zip(before, after, expected)):
        if left.get("block_index") != index or right.get("block_index") != index:
            fail("processor block order differs")
        delta = {name: int(right.get(name, -1)) - int(left.get(name, -1)) for name in _COUNTERS}
        if delta != wanted:
            fail(f"processor hook count differs at block {index}")
        digests = right.get("owner_input_binding_digests")
        if not isinstance(digests, list) or owner_binding_digest not in digests:
            fail(f"processor owner binding absent at block {index}")
        observed.append(delta)
    value = {
        "execution_id": execution_id, "owner": owner,
        "schedule_index": schedule_index, "branch": branch,
        "phase": phase, "band_name": band_name,
        "selected_blocks": list(swap.band_blocks(band_name)),
        "installed_block_indices": list(range(swap.TOTAL_BLOCKS)),
        "owner_input_binding_digest": owner_binding_digest,
        "per_block_counter_deltas": observed,
        "exact_hook_counts": True, "all30_processor_inventory": True,
    }
    return {**value, "digest": object_sha256(value)}


def build_c0_engineering_gate(
    *, noop_parity: Sequence[Mapping[str, Any]], processor_audits: Sequence[Mapping[str, Any]],
    cache_audits: Sequence[Mapping[str, Any]],
    output_records: Sequence[Mapping[str, Any]], first_forward_consensus: Mapping[str, Any],
    model_integrity: Mapping[str, Any], no_update: Mapping[str, Any],
) -> Mapping[str, Any]:
    if [row.get("owner") for row in noop_parity] != list(swap.OWNERS):
        fail("C0 noop parity owner closure differs")
    for row in noop_parity:
        if (
            row.get("plain_vs_swap_velocity_raw_bytes_equal") is not True
            or row.get("plain_vs_swap_predecode_raw_bytes_equal") is not True
            or row.get("plain_velocity_sha256") != row.get("swap_velocity_sha256")
            or row.get("plain_predecode_sha256") != row.get("swap_predecode_sha256")
            or row.get("internal_noop_swap_decoded") is not False
        ):
            fail("C0 noop-swap byte parity differs")
    if len(output_records) != 6 or any(
        row.get("frames") != EXPECTED_FRAMES or row.get("fps") != float(FPS)
        or row.get("hw") != list(EXPECTED_HW) for row in output_records
    ):
        fail("C0 six-media completion differs")
    if not processor_audits or any(
        row.get("exact_hook_counts") is not True
        or row.get("all30_processor_inventory") is not True
        for row in processor_audits
    ):
        fail("C0 processor audit differs")
    if first_forward_consensus.get("world_size") != 4 or first_forward_consensus.get("passed") is not True:
        fail("C0 first-forward WORLD4 consensus differs")
    if model_integrity.get("pre_c0_sha256") != model_integrity.get("post_c0_sha256") or model_integrity.get("bytes_unchanged") is not True:
        fail("C0 model bytes changed")
    required_no_update = {
        "gradient_enabled": False, "optimizer_present": False,
        "scheduler_present": False, "scheduler_steps": 0,
        "parameter_gradients_present": False, "parameter_updates": 0,
    }
    if any(no_update.get(key) != value for key, value in required_no_update.items()):
        fail("C0 no-update contract differs")
    value = {
        "schema_version": C0_GATE_SCHEMA, "engineering_pass": True,
        "scientific_pass_claimed": False, "visual_selection_performed": False,
        "decoded_output_count": 6, "internal_noop_parity_decoded_output_count": 0,
        "noop_parity": list(noop_parity), "processor_audits": list(processor_audits),
        "cache_audits": list(cache_audits),
        "first_forward_consensus": dict(first_forward_consensus),
        "model_integrity": dict(model_integrity), "no_update": dict(no_update),
        "decoded_output_names": [row["name"] for row in output_records],
        "decoded_output_record_digests": [object_sha256(row) for row in output_records],
        "media_complete": True,
    }
    return {**value, "digest": object_sha256(value)}


def finalize_receipt(unsigned: Mapping[str, Any]) -> Mapping[str, Any]:
    if "receipt_digest" in unsigned:
        fail("unsigned receipt already has a digest")
    return {**dict(unsigned), "receipt_digest": object_sha256(unsigned)}


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        fail(f"{label} key closure differs")
    return value


def _validate_digest_record(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be one object")
    unsigned = dict(value)
    declared = _sha(unsigned.pop("digest", None), label=f"{label} digest")
    if object_sha256(unsigned) != declared:
        fail(f"{label} digest differs")
    return value


_BINDING_KEYS = frozenset({
    "owner", "schedule_index", "timestep", "sigma_float32_be_hex",
    "orbit_row_digest", "target_source_full_blob_sha256",
    "owner_full_blob_sha256", "owner_reference_blob_sha256",
    "decoded_target_tensor_sha256", "decoded_owner_full_tensor_sha256",
    "decoded_owner_reference_tensor_sha256", "epsilon_sha256",
    "target_x_s_sha256", "prepared_visual_prefix_sha256",
    "prepared_prefix_rotary_sha256", "total_tokens", "condition_tokens",
    "source_ids", "owner_pair_switch_audited_by_single_binding",
})


def _binding_from_receipt(value: Any, *, label: str) -> swap.OwnerInputBinding:
    row = _require_exact_keys(value, set(_BINDING_KEYS), label=label)
    if row.get("owner_pair_switch_audited_by_single_binding") is not False:
        fail(f"{label} makes an unaudited pair claim")
    try:
        binding = swap.OwnerInputBinding(
            owner=row["owner"], schedule_index=row["schedule_index"],
            timestep=row["timestep"], sigma_float32_be_hex=row["sigma_float32_be_hex"],
            orbit_row_digest=row["orbit_row_digest"],
            target_source_full_blob_sha256=row["target_source_full_blob_sha256"],
            owner_full_blob_sha256=row["owner_full_blob_sha256"],
            owner_reference_blob_sha256=tuple(row["owner_reference_blob_sha256"]),
            decoded_target_tensor_sha256=row["decoded_target_tensor_sha256"],
            decoded_owner_full_tensor_sha256=row["decoded_owner_full_tensor_sha256"],
            decoded_owner_reference_tensor_sha256=tuple(
                row["decoded_owner_reference_tensor_sha256"]
            ),
            epsilon_sha256=row["epsilon_sha256"],
            target_x_s_sha256=row["target_x_s_sha256"],
            prepared_visual_prefix_sha256=row["prepared_visual_prefix_sha256"],
            prepared_prefix_rotary_sha256=row["prepared_prefix_rotary_sha256"],
            total_tokens=row["total_tokens"], condition_tokens=row["condition_tokens"],
            source_ids=tuple(row["source_ids"]),
        )
    except (KeyError, TypeError, ValueError, swap.PromptSwapError) as error:
        raise StageARuntimeError(f"{label} is invalid: {error}") from error
    if canonical_json_bytes(binding.receipt()) != canonical_json_bytes(row):
        fail(f"{label} is not canonical")
    return binding


def _validate_pair_rows(value: Any, schedules: Sequence[int], *, label: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(schedules):
        fail(f"{label} owner-pair count differs")
    result: dict[int, Mapping[str, Any]] = {}
    pair_keys = {
        "schedule_index", "correct_owner_binding", "wrong_owner_binding",
        "pair_validation", "actual_object_hashes_recomputed",
        "world4_consensus_before_forward", "digest",
        "actual_tensor_bundle", "actual_tensor_bundle_digest",
        "world4_actual_tensor_bundle_rank_digests", "pack_geometry",
        "pack_geometry_digest", "x_s_construction",
    }
    for expected_schedule, raw in zip(schedules, value):
        row = _require_exact_keys(raw, pair_keys, label=f"{label} pair")
        _validate_digest_record(row, label=f"{label} pair")
        if (
            row.get("schedule_index") != expected_schedule
            or row.get("actual_object_hashes_recomputed") is not True
            or row.get("world4_consensus_before_forward") is not True
        ):
            fail(f"{label} owner-pair execution binding differs")
        correct = _binding_from_receipt(
            row["correct_owner_binding"], label=f"{label} correct binding"
        )
        wrong = _binding_from_receipt(
            row["wrong_owner_binding"], label=f"{label} wrong binding"
        )
        expected_pair = swap.validate_owner_pair_bindings(correct, wrong)
        if canonical_json_bytes(row["pair_validation"]) != canonical_json_bytes(expected_pair):
            fail(f"{label} owner-pair validation differs")
        tensor_bundle = row.get("actual_tensor_bundle")
        geometry = row.get("pack_geometry")
        x_s = row.get("x_s_construction")
        expected_tensor_bundle = {
            "decoded_target_tensor_sha256": correct.decoded_target_tensor_sha256,
            "decoded_correct_owner_full_tensor_sha256": correct.decoded_owner_full_tensor_sha256,
            "decoded_correct_owner_reference_tensor_sha256": list(
                correct.decoded_owner_reference_tensor_sha256
            ),
            "decoded_wrong_owner_full_tensor_sha256": wrong.decoded_owner_full_tensor_sha256,
            "decoded_wrong_owner_reference_tensor_sha256": list(
                wrong.decoded_owner_reference_tensor_sha256
            ),
            "epsilon_sha256": correct.epsilon_sha256,
            "target_x_s_sha256": correct.target_x_s_sha256,
            "correct_prepared_visual_prefix_sha256": correct.prepared_visual_prefix_sha256,
            "wrong_prepared_visual_prefix_sha256": wrong.prepared_visual_prefix_sha256,
            "prepared_prefix_rotary_sha256": correct.prepared_prefix_rotary_sha256,
        }
        geometry_keys = {
            "orbit_member_order", "owner_aliases", "full_latent_shape",
            "reference_latent_shape", "reference_rgb_indices",
            "condition_components", "target_component", "source_ids",
            "concat_order", "total_tokens", "condition_tokens", "target_tokens",
            "target_is_strict_suffix", "sp4_layout_receipts",
            "append_false_then_contiguous_rank_chunks",
        }
        x_s_keys = {
            "function", "formula", "schedule_index", "sigma_float32_be_hex",
            "clean_sha256", "epsilon_sha256", "x_s_sha256", "actual_recomputed",
        }
        if (
            not isinstance(tensor_bundle, Mapping)
            or dict(tensor_bundle) != expected_tensor_bundle
            or row.get("actual_tensor_bundle_digest") != object_sha256(tensor_bundle)
            or row.get("world4_actual_tensor_bundle_rank_digests")
            != [row.get("actual_tensor_bundle_digest")] * 4
            or not isinstance(geometry, Mapping)
            or set(geometry) != geometry_keys
            or row.get("pack_geometry_digest") != object_sha256(geometry)
            or geometry.get("orbit_member_order") != ["V0", "V1", "V2"]
            or geometry.get("owner_aliases")
            != {"correct_owner": "V0/source", "wrong_owner": "V1/variant_a"}
            or geometry.get("full_latent_shape") != list(EXPECTED_LATENT_SHAPE)
            or geometry.get("reference_latent_shape") != [1, 16, 1, 74, 50]
            or geometry.get("reference_rgb_indices") != [0, 27, 53, 80]
            or geometry.get("condition_components")
            != ["owner_full_21", "owner_ref0_1", "owner_ref27_1", "owner_ref53_1", "owner_ref80_1"]
            or geometry.get("target_component") != "unchanged_source_x_s_21"
            or geometry.get("source_ids") != list(swap.NATIVE_SOURCE_IDS)
            or geometry.get("concat_order")
            != ["video", "ref0", "ref1", "ref2", "ref3", "target"]
            or geometry.get("total_tokens") != correct.total_tokens
            or geometry.get("condition_tokens") != correct.condition_tokens
            or geometry.get("target_tokens") != correct.total_tokens - correct.condition_tokens
            or geometry.get("target_is_strict_suffix") is not True
            or geometry.get("append_false_then_contiguous_rank_chunks") is not True
            or not isinstance(geometry.get("sp4_layout_receipts"), list)
            or geometry["sp4_layout_receipts"] != [
                dict(swap.NativeTargetSuffixLayout(
                    correct.total_tokens, correct.condition_tokens, rank, SP_SIZE
                ).receipt()) for rank in range(SP_SIZE)
            ]
            or not isinstance(x_s, Mapping)
            or set(x_s) != x_s_keys
            or x_s.get("function")
            != "source_noised_ladder_v1.shared_noise_source_state"
            or x_s.get("formula")
            != "(1-sigma_float32_authority)*decoded_V0+sigma_float32_authority*epsilon"
            or x_s.get("schedule_index") != expected_schedule
            or x_s.get("sigma_float32_be_hex") != correct.sigma_float32_be_hex
            or x_s.get("clean_sha256") != correct.decoded_target_tensor_sha256
            or x_s.get("epsilon_sha256") != correct.epsilon_sha256
            or x_s.get("x_s_sha256") != correct.target_x_s_sha256
            or x_s.get("actual_recomputed") is not True
        ):
            fail(f"{label} actual tensor/pack geometry closure differs")
        result[expected_schedule] = row
    return result


def build_cross_schedule_owner_closure(
    full_pairs: Mapping[int, Mapping[str, Any]], *, c0_pair: Mapping[str, Any],
) -> Mapping[str, Any]:
    schedules = tuple(swap.policy.REGISTERED_SCHEDULE_INDICES)
    if tuple(full_pairs) != schedules or canonical_json_bytes(full_pairs[29]) != canonical_json_bytes(c0_pair):
        fail("C0/full s29 owner-pair identity differs")
    bindings: dict[tuple[int, str], swap.OwnerInputBinding] = {}
    for schedule in schedules:
        for owner in swap.OWNERS:
            bindings[(schedule, owner)] = _binding_from_receipt(
                full_pairs[schedule][f"{owner}_binding"],
                label=f"cross schedule s{schedule} {owner}",
            )
    stable_fields = (
        "orbit_row_digest", "target_source_full_blob_sha256",
        "owner_full_blob_sha256", "owner_reference_blob_sha256",
        "decoded_target_tensor_sha256", "decoded_owner_full_tensor_sha256",
        "decoded_owner_reference_tensor_sha256", "epsilon_sha256",
        "prepared_visual_prefix_sha256", "prepared_prefix_rotary_sha256",
        "total_tokens", "condition_tokens", "source_ids",
    )
    for owner in swap.OWNERS:
        reference = bindings[(schedules[0], owner)]
        if any(
            getattr(bindings[(schedule, owner)], field) != getattr(reference, field)
            for schedule in schedules[1:] for field in stable_fields
        ):
            fail(f"{owner} actual decoded/noise/prefix identity changed across schedule")
    epsilon_sha = bindings[(schedules[0], "correct_owner")].epsilon_sha256
    if any(binding.epsilon_sha256 != epsilon_sha for binding in bindings.values()):
        fail("Stage-A epsilon changed across owner/schedule")
    unsigned = {
        "schedule_indices": list(schedules), "single_epsilon_sha256": epsilon_sha,
        "single_epsilon_reused_across_all_schedules_and_owners": True,
        "decoded_owner_and_reference_tensors_stable_across_schedules": True,
        "prepared_owner_prefix_and_rotary_stable_across_schedules": True,
        "c0_s29_and_full_s29_binding_raw_equal": True,
        "target_x_s_sha256_by_schedule": {
            str(schedule): bindings[(schedule, "correct_owner")].target_x_s_sha256
            for schedule in schedules
        },
        "each_x_s_recomputed_from_same_epsilon_clean_and_exact_sigma": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _capture_id(profile: str, owner: str, schedule: int, branch: str) -> str:
    return f"{profile}-capture-s{schedule}-{owner}-{branch}"


def _noop_swap_id(owner: str) -> str:
    return f"c0-internal-s{swap.C0_SCHEDULE_INDEX}-{owner}-{swap.C0_BAND}-noop-swap"


def _expected_c0_audit_descriptors() -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for owner in swap.OWNERS:
        cells = [cell for cell in swap.c0_smoke_cells() if cell.owner == owner]
        rows.extend((
            {"owner": owner, "schedule": swap.C0_SCHEDULE_INDEX, "phase": "capture",
             "band": swap.ALL30_BAND, "branch": "forward",
             "execution_id": _capture_id("c0", owner, swap.C0_SCHEDULE_INDEX, "forward")},
            {"owner": owner, "schedule": swap.C0_SCHEDULE_INDEX, "phase": "plain_noop",
             "band": swap.NONE_BAND, "branch": "noop", "execution_id": cells[0].cell_id},
            {"owner": owner, "schedule": swap.C0_SCHEDULE_INDEX, "phase": "noop_swap",
             "band": swap.C0_BAND, "branch": "noop", "execution_id": _noop_swap_id(owner)},
            {"owner": owner, "schedule": swap.C0_SCHEDULE_INDEX, "phase": "mixed",
             "band": swap.C0_BAND, "branch": "forward", "execution_id": cells[1].cell_id},
            {"owner": owner, "schedule": swap.C0_SCHEDULE_INDEX, "phase": "mixed",
             "band": swap.ALL30_BAND, "branch": "forward", "execution_id": cells[2].cell_id},
        ))
    return tuple(rows)


def _expected_full_audit_descriptors() -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for schedule in swap.policy.REGISTERED_SCHEDULE_INDICES:
        for branch in swap.NONNOOP_BRANCHES:
            rows.append({
                "owner": "correct_owner", "schedule": schedule, "phase": "capture",
                "band": swap.ALL30_BAND, "branch": branch,
                "execution_id": _capture_id("full", "correct_owner", schedule, branch),
            })
        rows.append({
            "owner": "wrong_owner", "schedule": schedule, "phase": "capture",
            "band": swap.ALL30_BAND, "branch": "forward",
            "execution_id": _capture_id("full", "wrong_owner", schedule, "forward"),
        })
        for cell in swap.full_grid_cells():
            if cell.schedule_index != schedule:
                continue
            phase = "plain_noop" if cell.role == "noop_baseline" else "mixed"
            rows.append({
                "owner": cell.owner, "schedule": schedule, "phase": phase,
                "band": cell.band_name, "branch": cell.branch,
                "execution_id": cell.cell_id,
            })
    return tuple(rows)


def _expected_cache_specs(profile: str) -> tuple[Mapping[str, Any], ...]:
    if profile == "c0":
        schedules = (swap.C0_SCHEDULE_INDEX,)
        plan = swap.c0_smoke_cells()
        branches_by_owner = {
            "correct_owner": ("forward",), "wrong_owner": ("forward",),
        }
    elif profile == "full":
        schedules = swap.policy.REGISTERED_SCHEDULE_INDICES
        plan = swap.full_grid_cells()
        branches_by_owner = {
            "correct_owner": tuple(swap.NONNOOP_BRANCHES),
            "wrong_owner": ("forward",),
        }
    else:
        fail("cache profile differs")
    rows: list[Mapping[str, Any]] = []
    ordinal = 0
    for schedule in schedules:
        for owner in swap.OWNERS:
            for branch in branches_by_owner[owner]:
                mixed = tuple(
                    cell.cell_id for cell in plan
                    if cell.schedule_index == schedule and cell.owner == owner
                    and cell.branch == branch and cell.role != "noop_baseline"
                )
                reuse_count = sum(
                    len(swap.band_blocks(cell.band_name)) for cell in plan
                    if cell.schedule_index == schedule and cell.owner == owner
                    and cell.branch == branch and cell.role != "noop_baseline"
                )
                rows.append({
                    "cache_instance_ordinal": ordinal,
                    "execution_id": _capture_id(profile, owner, schedule, branch),
                    "owner": owner, "schedule_index": schedule, "branch": branch,
                    "mixed_execution_ids": list(mixed), "expected_reuse_count": reuse_count,
                })
                ordinal += 1
    return tuple(rows)


def _validate_cache_audits(
    value: Any, *, profile: str, pairs: Mapping[int, Mapping[str, Any]],
    processor_audits: Sequence[Mapping[str, Any]],
    text_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    specs = _expected_cache_specs(profile)
    if not isinstance(value, list) or len(value) != len(specs):
        fail(f"{profile} fresh cache count differs")
    keys = {
        "cache_instance_ordinal", "execution_id", "owner", "schedule_index",
        "branch", "owner_input_binding_digest", "capture_processor_audit_digest",
        "capture_prediction_sha256", "capture_prediction_world4_consensus",
        "capture_prediction_rank_sha256",
        "capture_prompt_runtime_binding_digest", "capture_prompt_embedding_sha256",
        "capture_prediction_discarded", "capture_decode_performed",
        "capture_scheduler_steps", "rng_state_before_sha256", "rng_state_after_sha256",
        "rng_state_before_world4_rank_sha256",
        "rng_state_after_world4_rank_sha256",
        "rng_state_unchanged", "mixed_execution_ids", "expected_reuse_count",
        "cache_receipt", "terminal_cache_audit", "fresh_for_exact_owner_schedule_branch",
        "cache_object_identity_unique_within_phase",
        "digest",
    }
    processor_by_id = {
        item.get("execution_id"): item for item in processor_audits
        if isinstance(item, Mapping)
    }
    seen_capture_results: set[str] = set()
    for index, (raw, spec) in enumerate(zip(value, specs)):
        row = _require_exact_keys(raw, keys, label=f"{profile} cache[{index}]")
        _validate_digest_record(row, label=f"{profile} cache[{index}]")
        owner = str(spec["owner"])
        schedule = int(spec["schedule_index"])
        binding = _binding_from_receipt(
            pairs[schedule][f"{owner}_binding"], label=f"{profile} cache binding"
        )
        processor = processor_by_id.get(spec["execution_id"])
        if not isinstance(processor, Mapping):
            fail(f"{profile} cache capture processor audit absent")
        for sha_key in (
            "capture_prediction_sha256", "rng_state_before_sha256",
            "rng_state_after_sha256", "capture_processor_audit_digest",
            "capture_prompt_runtime_binding_digest", "capture_prompt_embedding_sha256",
        ):
            _sha(row.get(sha_key), label=f"{profile} cache[{index}] {sha_key}")
        rng_before_rank = row.get("rng_state_before_world4_rank_sha256")
        rng_after_rank = row.get("rng_state_after_world4_rank_sha256")
        rng_before_digest = _rng_world4_vector_sha256(
            rng_before_rank, label=f"{profile} cache[{index}] RNG before"
        )
        rng_after_digest = _rng_world4_vector_sha256(
            rng_after_rank, label=f"{profile} cache[{index}] RNG after"
        )
        if (
            row.get("cache_instance_ordinal") != spec["cache_instance_ordinal"]
            or row.get("execution_id") != spec["execution_id"]
            or row.get("owner") != owner or row.get("schedule_index") != schedule
            or row.get("branch") != spec["branch"]
            or row.get("owner_input_binding_digest") != binding.digest
            or row.get("capture_processor_audit_digest") != processor.get("digest")
            or row.get("capture_prediction_world4_consensus") is not True
            or row.get("capture_prediction_rank_sha256")
            != [row.get("capture_prediction_sha256")] * 4
            or row.get("capture_prompt_runtime_binding_digest")
            != text_bindings[str(spec["branch"])].get("digest")
            or row.get("capture_prompt_embedding_sha256")
            != text_bindings[str(spec["branch"])].get("embedding_sha256")
            or row.get("capture_prediction_discarded") is not True
            or row.get("capture_decode_performed") is not False
            or row.get("capture_scheduler_steps") != 0
            or rng_before_rank != rng_after_rank
            or row.get("rng_state_before_sha256") != rng_before_digest
            or row.get("rng_state_after_sha256") != rng_after_digest
            or rng_before_digest != rng_after_digest
            or row.get("rng_state_unchanged") is not True
            or row.get("mixed_execution_ids") != spec["mixed_execution_ids"]
            or row.get("expected_reuse_count") != spec["expected_reuse_count"]
            or row.get("fresh_for_exact_owner_schedule_branch") is not True
            or row.get("cache_object_identity_unique_within_phase") is not True
        ):
            fail(f"{profile} cache[{index}] provenance differs")
        cache_receipt = _validate_digest_record(
            row.get("cache_receipt"), label=f"{profile} cache receipt[{index}]"
        )
        terminal = _validate_digest_record(
            row.get("terminal_cache_audit"), label=f"{profile} terminal cache[{index}]"
        )
        _require_exact_keys(cache_receipt, {
            "branch", "expected_block_indices", "captured_block_indices",
            "captured_shapes", "captured_content_identity_by_block", "sealed",
            "capturing", "capture_aborted", "reuse_count",
            "captured_hidden_or_output_reused", "captured_text_encoder_state_only",
            "digest",
        }, label=f"{profile} cache receipt[{index}]")
        _require_exact_keys(terminal, {
            "branch", "block_identity_digest",
            "all_30_content_and_versions_unchanged", "digest",
        }, label=f"{profile} terminal cache[{index}]")
        shapes = cache_receipt.get("captured_shapes")
        identities = cache_receipt.get("captured_content_identity_by_block")
        block_keys = {str(block) for block in range(swap.TOTAL_BLOCKS)}
        if (
            not isinstance(shapes, Mapping) or set(shapes) != block_keys
            or not isinstance(identities, Mapping) or set(identities) != block_keys
        ):
            fail(f"{profile} cache[{index}] per-block closure differs")
        first_shape: Optional[list[int]] = None
        first_dtype: Optional[str] = None
        for block in range(swap.TOTAL_BLOCKS):
            key = str(block)
            shape = shapes[key]
            identity = _require_exact_keys(
                identities[key], {
                    "shape", "dtype", "raw_sha256", "tensor_version", "digest",
                }, label=f"{profile} cache[{index}] identity block {block}",
            )
            _validate_digest_record(
                identity, label=f"{profile} cache[{index}] identity block {block}"
            )
            if (
                not isinstance(shape, list) or len(shape) != 3
                or any(type(item) is not int or item <= 0 for item in shape)
                or shape[0] != 1 or identity.get("shape") != shape
                or identity.get("dtype") != "torch.bfloat16"
                or not (
                    (type(identity.get("tensor_version")) is int
                     and int(identity["tensor_version"]) >= 0)
                    or identity.get("tensor_version")
                    == "inference_tensor_no_version"
                )
            ):
                fail(f"{profile} cache[{index}] identity geometry differs at block {block}")
            _sha(
                identity.get("raw_sha256"),
                label=f"{profile} cache[{index}] block {block} raw content",
            )
            if first_shape is None:
                first_shape = list(shape)
                first_dtype = str(identity["dtype"])
            elif shape != first_shape or identity.get("dtype") != first_dtype:
                fail(f"{profile} cache[{index}] post-condition geometry changed by block")
        if (
            cache_receipt.get("branch") != spec["branch"]
            or cache_receipt.get("expected_block_indices") != list(range(swap.TOTAL_BLOCKS))
            or cache_receipt.get("captured_block_indices") != list(range(swap.TOTAL_BLOCKS))
            or cache_receipt.get("sealed") is not True
            or cache_receipt.get("capturing") is not False
            or cache_receipt.get("capture_aborted") is not False
            or cache_receipt.get("reuse_count") != spec["expected_reuse_count"]
            or cache_receipt.get("captured_hidden_or_output_reused") is not False
            or cache_receipt.get("captured_text_encoder_state_only") is not True
            or terminal.get("branch") != spec["branch"]
            or terminal.get("block_identity_digest")
            != object_sha256(identities)
            or terminal.get("all_30_content_and_versions_unchanged") is not True
        ):
            fail(f"{profile} cache[{index}] terminal evidence differs")
        # A capture result hash is not an identity nonce and may numerically
        # collide across owners; freshness is established by exact unique
        # execution IDs/ordinals and distinct cache receipts, not this set.
        seen_capture_results.add(str(row["capture_prediction_sha256"]))


def _validate_c0_noop_parity(
    value: Any, *, pairs: Mapping[int, Mapping[str, Any]],
    processor_audits: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(value, list) or len(value) != 2:
        fail("C0 noop parity owner count differs")
    keys = {
        "owner", "schedule_index", "band_name", "plain_execution_id",
        "swap_execution_id", "owner_input_binding_digest",
        "plain_processor_audit_digest", "swap_processor_audit_digest",
        "plain_vs_swap_velocity_raw_bytes_equal",
        "plain_vs_swap_predecode_raw_bytes_equal", "plain_velocity_sha256",
        "swap_velocity_sha256", "plain_predecode_sha256", "swap_predecode_sha256",
        "internal_noop_swap_decoded", "full_velocity_compared",
        "full_predecode_compared", "digest",
    }
    audits = {
        row.get("execution_id"): row for row in processor_audits
        if isinstance(row, Mapping)
    }
    c0_cells = swap.c0_smoke_cells()
    for index, (row, owner) in enumerate(zip(value, swap.OWNERS)):
        checked = _require_exact_keys(row, keys, label=f"C0 noop parity[{index}]")
        _validate_digest_record(checked, label=f"C0 noop parity[{index}]")
        plain_id = next(
            cell.cell_id for cell in c0_cells
            if cell.owner == owner and cell.role == "noop_baseline"
        )
        swap_id = _noop_swap_id(owner)
        binding = _binding_from_receipt(
            pairs[swap.C0_SCHEDULE_INDEX][f"{owner}_binding"],
            label=f"C0 parity {owner} binding",
        )
        plain_audit = audits.get(plain_id)
        swap_audit = audits.get(swap_id)
        if not isinstance(plain_audit, Mapping) or not isinstance(swap_audit, Mapping):
            fail("C0 noop parity processor audit link absent")
        for key in (
            "plain_velocity_sha256", "swap_velocity_sha256",
            "plain_predecode_sha256", "swap_predecode_sha256",
            "plain_processor_audit_digest", "swap_processor_audit_digest",
        ):
            _sha(checked.get(key), label=f"C0 noop parity[{index}] {key}")
        if (
            checked.get("owner") != owner
            or checked.get("schedule_index") != swap.C0_SCHEDULE_INDEX
            or checked.get("band_name") != swap.C0_BAND
            or checked.get("plain_execution_id") != plain_id
            or checked.get("swap_execution_id") != swap_id
            or checked.get("owner_input_binding_digest") != binding.digest
            or checked.get("plain_processor_audit_digest") != plain_audit.get("digest")
            or checked.get("swap_processor_audit_digest") != swap_audit.get("digest")
            or checked.get("plain_vs_swap_velocity_raw_bytes_equal") is not True
            or checked.get("plain_vs_swap_predecode_raw_bytes_equal") is not True
            or checked.get("plain_velocity_sha256") != checked.get("swap_velocity_sha256")
            or checked.get("plain_predecode_sha256") != checked.get("swap_predecode_sha256")
            or checked.get("internal_noop_swap_decoded") is not False
            or checked.get("full_velocity_compared") is not True
            or checked.get("full_predecode_compared") is not True
        ):
            fail(f"C0 noop parity[{index}] differs")


def _validate_processor_audit(
    row: Any, *, owner: str, schedule: int, phase: str, band: str, branch: str,
    execution_id: str, binding_digest: str, label: str,
) -> None:
    keys = {
        "execution_id", "owner", "schedule_index", "branch",
        "phase", "band_name", "selected_blocks", "installed_block_indices",
        "owner_input_binding_digest", "per_block_counter_deltas",
        "exact_hook_counts", "all30_processor_inventory", "digest",
    }
    checked = _require_exact_keys(row, keys, label=label)
    _validate_digest_record(checked, label=label)
    if (
        checked.get("execution_id") != execution_id or checked.get("owner") != owner
        or checked.get("schedule_index") != schedule or checked.get("branch") != branch
        or checked.get("phase") != phase or checked.get("band_name") != band
        or checked.get("selected_blocks") != list(swap.band_blocks(band))
        or checked.get("installed_block_indices") != list(range(swap.TOTAL_BLOCKS))
        or checked.get("owner_input_binding_digest") != binding_digest
        or checked.get("exact_hook_counts") is not True
        or checked.get("all30_processor_inventory") is not True
        or checked.get("per_block_counter_deltas")
        != [dict(item) for item in expected_processor_deltas(phase, band)]
    ):
        fail(f"{label} processor audit differs for {owner}")


_TEXT_BASE_KEYS = frozenset({
    "branch", "prompt_sha256", "input_ids_sha256", "attention_mask_sha256",
    "t5_input_lens_sha256", "text_lens", "embedding_sha256",
    "input_ids_shape", "attention_mask_shape", "t5_input_lens_shape",
    "embedding_shape", "embedding_dtype", "embedding_device_type",
    "encoded_call_ordinal",
})
_TEXT_KEYS = frozenset({
    *_TEXT_BASE_KEYS, "runtime_tensor_binding_digest",
    "world4_rank_binding_digests", "world4_consensus", "digest",
})


def _validate_text_runtime(value: Any) -> Mapping[str, Mapping[str, Any]]:
    text_runtime = _require_exact_keys(
        value, {"branches", "all_encoded_once_before_forward", "digest"},
        label="text runtime",
    )
    _validate_digest_record(text_runtime, label="text runtime")
    if text_runtime.get("all_encoded_once_before_forward") is not True:
        fail("text prompts were not encoded once before forward")
    branches = text_runtime.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != set(swap.TEXT_BRANCHES):
        fail("text runtime branch closure differs")
    checked: dict[str, Mapping[str, Any]] = {}
    for branch in swap.TEXT_BRANCHES:
        binding = _require_exact_keys(
            branches[branch], set(_TEXT_KEYS), label=f"text runtime {branch}"
        )
        _validate_digest_record(binding, label=f"text runtime {branch}")
        input_shape = binding.get("input_ids_shape")
        attention_shape = binding.get("attention_mask_shape")
        lens_shape = binding.get("t5_input_lens_shape")
        embedding_shape = binding.get("embedding_shape")
        text_lens = binding.get("text_lens")
        if (
            not isinstance(input_shape, list) or len(input_shape) != 2
            or any(type(item) is not int for item in input_shape)
            or input_shape[0] != 1 or not 0 < input_shape[1] <= 512
            or attention_shape != input_shape
            or lens_shape != [1, 1]
            or embedding_shape != [1, 512, 4096]
            or text_lens != [512]
        ):
            fail(f"text runtime {branch} tensor geometry differs")
        base = {key: binding[key] for key in _TEXT_BASE_KEYS}
        runtime_digest = object_sha256(base)
        if (
            binding.get("branch") != branch
            or binding.get("prompt_sha256") != PROMPT_SHA256[branch]
            or binding.get("encoded_call_ordinal") != swap.TEXT_BRANCHES.index(branch)
            or binding.get("embedding_dtype") != "torch.bfloat16"
            or binding.get("embedding_device_type") != "cuda"
            or binding.get("runtime_tensor_binding_digest") != runtime_digest
            or binding.get("world4_consensus") is not True
            or binding.get("world4_rank_binding_digests") != [runtime_digest] * 4
        ):
            fail(f"text runtime {branch} identity/consensus differs")
        for key in (
            "prompt_sha256", "input_ids_sha256", "attention_mask_sha256",
            "t5_input_lens_sha256", "embedding_sha256",
            "runtime_tensor_binding_digest",
        ):
            _sha(binding.get(key), label=f"text runtime {branch} {key}")
        checked[branch] = binding
    return checked


def _validate_model_integrity(value: Any) -> Mapping[str, Any]:
    keys = {
        "certificate_schema", "pre_sha256", "post_c0_sha256", "post_sha256",
        "parameter_tensors", "buffer_tensors", "bytes_unchanged",
        "all_parameters_frozen", "all_parameter_gradients_absent",
    }
    record = _require_exact_keys(value, keys, label="model integrity")
    for key in ("pre_sha256", "post_c0_sha256", "post_sha256"):
        _sha(record.get(key), label=f"model integrity {key}")
    if (
        record.get("certificate_schema")
        != "torch-module-parameters-buffers-raw-sha256-v1"
        or record.get("pre_sha256") != record.get("post_c0_sha256")
        or record.get("pre_sha256") != record.get("post_sha256")
        or type(record.get("parameter_tensors")) is not int
        or int(record["parameter_tensors"]) <= 0
        or type(record.get("buffer_tensors")) is not int
        or int(record["buffer_tensors"]) < 0
        or record.get("bytes_unchanged") is not True
        or record.get("all_parameters_frozen") is not True
        or record.get("all_parameter_gradients_absent") is not True
    ):
        fail("terminal model integrity differs")
    return record


def _validate_input_invariants(
    value: Any, *, profile: str,
    pairs: Mapping[int, Mapping[str, Any]],
    text_bindings: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    record = _validate_digest_record(value, label="actual model input invariants")
    _require_exact_keys(record, {
        "schema_version", "pre_c0_snapshot", "stage_snapshot_digests",
        "stage_world4_rank_digests", "full_phase_executed",
        "actual_objects_rehashed_at_each_phase",
        "all_actual_input_bytes_unchanged", "digest",
    }, label="actual model input invariants")
    snapshot = _validate_digest_record(
        record.get("pre_c0_snapshot"), label="pre-C0 actual input snapshot"
    )
    _require_exact_keys(snapshot, {
        "epsilon_sha256", "prompt_embedding_sha256_by_branch",
        "schedule_inputs", "digest",
    }, label="pre-C0 actual input snapshot")
    prompt_hashes = snapshot.get("prompt_embedding_sha256_by_branch")
    if (
        not isinstance(prompt_hashes, Mapping)
        or set(prompt_hashes) != set(swap.TEXT_BRANCHES)
        or any(
            prompt_hashes.get(branch) != text_bindings[branch].get("embedding_sha256")
            for branch in swap.TEXT_BRANCHES
        )
    ):
        fail("actual prompt embedding invariant differs")
    schedules = (
        tuple(swap.policy.REGISTERED_SCHEDULE_INDICES)
        if profile == FORMAL_PROFILE else (swap.C0_SCHEDULE_INDEX,)
    )
    schedule_inputs = snapshot.get("schedule_inputs")
    if not isinstance(schedule_inputs, Mapping) or set(schedule_inputs) != {
        str(schedule) for schedule in schedules
    }:
        fail("actual schedule input invariant closure differs")
    for schedule in schedules:
        schedule_row = _require_exact_keys(
            schedule_inputs[str(schedule)], {"x_s_sha256", "owners"},
            label=f"s{schedule} actual input invariant",
        )
        owners = schedule_row.get("owners")
        if not isinstance(owners, Mapping) or set(owners) != set(swap.OWNERS):
            fail(f"s{schedule} owner input invariant closure differs")
        for owner in swap.OWNERS:
            owner_row = _require_exact_keys(owners[owner], {
                "owner_input_binding_digest", "packed_latents_sha256",
                "packed_rotary_sha256", "source_ids",
                "prepared_visual_prefix_sha256",
                "prepared_prefix_rotary_sha256",
            }, label=f"s{schedule} {owner} actual input invariant")
            binding = _binding_from_receipt(
                pairs[schedule][f"{owner}_binding"],
                label=f"s{schedule} {owner} invariant binding",
            )
            if (
                owner_row.get("owner_input_binding_digest") != binding.digest
                or schedule_row.get("x_s_sha256") != binding.target_x_s_sha256
                or snapshot.get("epsilon_sha256") != binding.epsilon_sha256
                or owner_row.get("source_ids") != list(binding.source_ids)
                or owner_row.get("prepared_visual_prefix_sha256")
                != binding.prepared_visual_prefix_sha256
                or owner_row.get("prepared_prefix_rotary_sha256")
                != binding.prepared_prefix_rotary_sha256
            ):
                fail(f"s{schedule} {owner} actual input/binding invariant differs")
            for key in ("packed_latents_sha256", "packed_rotary_sha256"):
                _sha(owner_row.get(key), label=f"s{schedule} {owner} {key}")
    stage_digests = _require_exact_keys(
        record.get("stage_snapshot_digests"),
        {"pre_c0", "post_c0", "post_full", "terminal"},
        label="actual input stage digests",
    )
    rank_digests = _require_exact_keys(
        record.get("stage_world4_rank_digests"),
        {"pre_c0", "post_c0", "post_full", "terminal"},
        label="actual input stage rank digests",
    )
    expected_full = profile == FORMAL_PROFILE
    pre_digest = snapshot["digest"]
    if (
        record.get("schema_version") != "stage-a-actual-model-input-invariants-v1"
        or record.get("full_phase_executed") is not expected_full
        or record.get("actual_objects_rehashed_at_each_phase") is not True
        or record.get("all_actual_input_bytes_unchanged") is not True
        or stage_digests.get("pre_c0") != pre_digest
        or stage_digests.get("post_c0") != pre_digest
        or stage_digests.get("terminal") != pre_digest
        or stage_digests.get("post_full") != (pre_digest if expected_full else None)
        or any(
            rank_digests.get(stage)
            != ([digest] * 4 if digest is not None else None)
            for stage, digest in stage_digests.items()
        )
    ):
        fail("actual input phase/world4 invariant differs")
    return record


_STATISTICS_KEYS = frozenset({
    "block_index", *_COUNTERS, "owner_input_binding_digests",
})


def _validate_processor_patch(
    value: Any, *, processor_audits: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    patch = _validate_digest_record(value, label="processor patch")
    _require_exact_keys(patch, {
        "schema_version", "installed_block_indices", "installed_projection",
        "base_processor_reused_for_noop_and_branch",
        "same_current_hidden_states_for_both_selected_calls",
        "non_target_rows_select_noop_output", "unselected_blocks_single_official_call",
        "capture_text_condition_only", "capture_hidden_or_output_reused",
        "owner_binding_required_by_every_context",
        "installation_and_restore_transactional", "inference_no_grad_required",
        "attention_parameters_frozen_and_grad_free_required", "optimizer_present",
        "parameter_update_authorized", "restored", "statistics", "digest",
    }, label="processor patch")
    if (
        patch.get("schema_version") != swap.SCHEMA_VERSION
        or patch.get("installed_block_indices") != list(range(swap.TOTAL_BLOCKS))
        or patch.get("installed_projection") != "blocks.{0..29}.attn2.processor"
        or patch.get("restored") is not True
        or patch.get("capture_text_condition_only") is not True
        or patch.get("capture_hidden_or_output_reused") is not False
        or patch.get("base_processor_reused_for_noop_and_branch") is not True
        or patch.get("same_current_hidden_states_for_both_selected_calls") is not True
        or patch.get("non_target_rows_select_noop_output") is not True
        or patch.get("unselected_blocks_single_official_call") is not True
        or patch.get("owner_binding_required_by_every_context") is not True
        or patch.get("installation_and_restore_transactional") is not True
        or patch.get("inference_no_grad_required") is not True
        or patch.get("attention_parameters_frozen_and_grad_free_required") is not True
        or patch.get("optimizer_present") is not False
        or patch.get("parameter_update_authorized") is not False
    ):
        fail("processor patch terminal receipt differs")
    statistics = patch.get("statistics")
    if not isinstance(statistics, list) or len(statistics) != swap.TOTAL_BLOCKS:
        fail("processor patch terminal statistics count differs")
    expected_owner_digests = sorted({
        str(row["owner_input_binding_digest"]) for row in processor_audits
    })
    for block_index, raw in enumerate(statistics):
        row = _require_exact_keys(
            raw, set(_STATISTICS_KEYS),
            label=f"processor patch statistics[{block_index}]",
        )
        expected = {
            counter: sum(
                int(audit["per_block_counter_deltas"][block_index][counter])
                for audit in processor_audits
            )
            for counter in _COUNTERS
        }
        if (
            row.get("block_index") != block_index
            or any(row.get(counter) != count for counter, count in expected.items())
            or row.get("owner_input_binding_digests") != expected_owner_digests
        ):
            fail(f"processor patch cumulative statistics[{block_index}] differ")
    return patch


def _validate_calibration(value: Any) -> Mapping[str, Any]:
    calibration = _validate_digest_record(value, label="calibration")
    _require_exact_keys(calibration, {
        "family4_seed", "family4_initial_gaussian_raw_sha256", "stage_a_seed",
        "family4_noise_consumed_by_stage_a", "family4_and_stage_a_seed_distinct",
        "branch_authority", "digest",
    }, label="calibration")
    if (
        calibration.get("family4_seed") != CALIBRATION_SEED
        or calibration.get("family4_initial_gaussian_raw_sha256")
        != CALIBRATION_INITIAL_GAUSSIAN_RAW_SHA256
        or calibration.get("stage_a_seed") != SEED
        or calibration.get("family4_noise_consumed_by_stage_a") is not False
        or calibration.get("family4_and_stage_a_seed_distinct") is not True
        or calibration.get("branch_authority") != swap.branch_calibration_authority()
    ):
        fail("family4 calibration evidence/role differs")
    return calibration


_OUTPUT_KEYS = frozenset({
    "name", "phase", "cell", "cell_digest", "owner_input_binding",
    "owner_input_binding_digest", "owner_pair_validation_digest",
    "velocity_sha256", "predecode_x0_hat_sha256", "sha256", "frames",
    "fps", "hw", "decode_input_latent_sha256", "decode_input_latent_shape",
    "vae_frozen_eval", "actual_object_binding_used_for_forward",
    "global_prompt_branch", "global_noop_prompt_runtime_binding_digest",
    "global_noop_embedding_sha256", "processor_audit_digest",
    "branch_capture_cache_audit_digest", "result_rank_sha256",
    "result_world4_consensus",
    "decode_input_dtype", "decode_input_device_type",
    "decode_input_contiguous", "decode_input_finite",
})


def _fixed_output_cells(profile: str) -> tuple[tuple[str, swap.ExperimentCell], ...]:
    rows: list[tuple[str, swap.ExperimentCell]] = [
        ("c0", cell) for cell in swap.c0_smoke_cells()
    ]
    if profile == FORMAL_PROFILE:
        rows.extend(("full", cell) for cell in swap.full_grid_cells())
    return tuple(rows)


def _validate_output_rows(
    rows: Any, *, profile: str, artifacts: Mapping[str, Any],
    pair_rows_by_phase: Mapping[str, Mapping[int, Mapping[str, Any]]],
    noop_text_binding: Mapping[str, Any],
    processor_by_execution: Mapping[str, Mapping[str, Any]],
    cache_by_execution: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = _fixed_output_cells(profile)
    if not isinstance(rows, list) or len(rows) != len(expected):
        fail("runtime decoded output count differs")
    seen_names: set[str] = set()
    for index, (raw, (phase, cell)) in enumerate(zip(rows, expected)):
        row = _require_exact_keys(raw, set(_OUTPUT_KEYS), label=f"output[{index}]")
        name = f"{phase}/{cell.output_name}"
        if (
            row.get("name") != name or name in seen_names
            or row.get("phase") != phase
            or canonical_json_bytes(row.get("cell")) != canonical_json_bytes(cell.receipt())
            or row.get("cell_digest") != object_sha256(cell.receipt())
            or row.get("sha256") != artifacts.get(name)
            or row.get("frames") != EXPECTED_FRAMES or row.get("fps") != float(FPS)
            or row.get("hw") != list(EXPECTED_HW)
            or row.get("decode_input_latent_shape") != list(EXPECTED_LATENT_SHAPE)
            or row.get("decode_input_dtype") != "torch.float32"
            or row.get("decode_input_device_type") != "cuda"
            or row.get("decode_input_contiguous") is not True
            or row.get("decode_input_finite") is not True
            or row.get("decode_input_latent_sha256")
            != row.get("predecode_x0_hat_sha256")
            or row.get("vae_frozen_eval") is not True
            or row.get("actual_object_binding_used_for_forward") is not True
            or row.get("global_prompt_branch") != "noop"
            or row.get("global_noop_prompt_runtime_binding_digest")
            != noop_text_binding.get("digest")
            or row.get("global_noop_embedding_sha256")
            != noop_text_binding.get("embedding_sha256")
            or row.get("result_world4_consensus") is not True
            or row.get("result_rank_sha256") != [row.get("velocity_sha256")] * 4
        ):
            fail(f"output[{index}] fixed cell/media contract differs")
        for key in (
            "velocity_sha256", "predecode_x0_hat_sha256", "sha256",
            "owner_input_binding_digest", "owner_pair_validation_digest",
            "processor_audit_digest", "global_noop_prompt_runtime_binding_digest",
            "global_noop_embedding_sha256",
        ):
            _sha(row.get(key), label=f"output[{index}] {key}")
        binding = _binding_from_receipt(
            row.get("owner_input_binding"), label=f"output[{index}] binding"
        )
        if (
            binding.owner != cell.owner or binding.schedule_index != cell.schedule_index
            or binding.digest != row["owner_input_binding_digest"]
        ):
            fail(f"output[{index}] binding/cell differs")
        pair_row = pair_rows_by_phase[phase][cell.schedule_index]
        if (
            pair_row["pair_validation"][
                f"{cell.owner}_input_binding_digest"
            ] != binding.digest
            or pair_row["pair_validation"]["digest"]
            != row["owner_pair_validation_digest"]
        ):
            fail(f"output[{index}] pair chain differs")
        processor = processor_by_execution.get(cell.cell_id)
        if not isinstance(processor, Mapping) or row["processor_audit_digest"] != processor.get("digest"):
            fail(f"output[{index}] processor audit chain differs")
        if cell.role == "noop_baseline":
            if row.get("branch_capture_cache_audit_digest") is not None:
                fail(f"output[{index}] noop unexpectedly chains a branch cache")
        else:
            capture_id = _capture_id(phase, cell.owner, cell.schedule_index, cell.branch)
            cache = cache_by_execution.get(capture_id)
            if (
                not isinstance(cache, Mapping)
                or row.get("branch_capture_cache_audit_digest") != cache.get("digest")
            ):
                fail(f"output[{index}] branch cache chain differs")
        seen_names.add(name)


def validate_c0_before_full(
    *, gate: Mapping[str, Any], pair_receipt: Mapping[str, Any],
    text_runtime: Mapping[str, Any], output_rows: Sequence[Mapping[str, Any]],
    expected_input_bundle_digest: str,
) -> bool:
    """Deep engineering-only C0 admission performed before full-grid starts."""

    keys = {
        "schema_version", "engineering_pass", "scientific_pass_claimed",
        "visual_selection_performed", "decoded_output_count",
        "internal_noop_parity_decoded_output_count", "noop_parity",
        "processor_audits", "cache_audits", "first_forward_consensus",
        "model_integrity", "no_update", "decoded_output_names",
        "decoded_output_record_digests", "media_complete", "digest",
    }
    checked = _require_exact_keys(gate, keys, label="pre-full C0 gate")
    _validate_digest_record(checked, label="pre-full C0 gate")
    if (
        checked.get("schema_version") != C0_GATE_SCHEMA
        or checked.get("engineering_pass") is not True
        or checked.get("scientific_pass_claimed") is not False
        or checked.get("visual_selection_performed") is not False
        or checked.get("decoded_output_count") != 6
        or checked.get("internal_noop_parity_decoded_output_count") != 0
        or checked.get("media_complete") is not True
        or not isinstance(output_rows, Sequence) or len(output_rows) != 6
    ):
        fail("pre-full C0 engineering boundary differs")
    pairs = _validate_pair_rows(
        [pair_receipt], (swap.C0_SCHEDULE_INDEX,), label="pre-full C0"
    )
    text_bindings = _validate_text_runtime(text_runtime)
    audits = checked.get("processor_audits")
    descriptors = _expected_c0_audit_descriptors()
    if not isinstance(audits, list) or len(audits) != len(descriptors):
        fail("pre-full C0 processor audit count differs")
    for index, (audit, descriptor) in enumerate(zip(audits, descriptors)):
        owner = str(descriptor["owner"])
        binding = _binding_from_receipt(
            pairs[swap.C0_SCHEDULE_INDEX][f"{owner}_binding"],
            label=f"pre-full C0 {owner} binding",
        )
        _validate_processor_audit(
            audit, owner=owner, schedule=int(descriptor["schedule"]),
            phase=str(descriptor["phase"]), band=str(descriptor["band"]),
            branch=str(descriptor["branch"]),
            execution_id=str(descriptor["execution_id"]),
            binding_digest=binding.digest,
            label=f"pre-full C0 processor audit[{index}]",
        )
    _validate_c0_noop_parity(
        checked.get("noop_parity"), pairs=pairs, processor_audits=audits
    )
    _validate_cache_audits(
        checked.get("cache_audits"), profile="c0", pairs=pairs,
        processor_audits=audits, text_bindings=text_bindings,
    )
    artifacts = {
        str(row.get("name")): row.get("sha256")
        for row in output_rows if isinstance(row, Mapping)
    }
    _validate_output_rows(
        list(output_rows), profile="smoke-only", artifacts=artifacts,
        pair_rows_by_phase={"c0": pairs, "full": {}},
        noop_text_binding=text_bindings["noop"],
        processor_by_execution={row["execution_id"]: row for row in audits},
        cache_by_execution={
            row["execution_id"]: row for row in checked["cache_audits"]
        },
    )
    first = _validate_digest_record(
        checked.get("first_forward_consensus"), label="pre-full first forward"
    )
    first_cache = checked["cache_audits"][0]
    if (
        first.get("input_bundle_digest") != expected_input_bundle_digest
        or first.get("first_execution_id") != first_cache.get("execution_id")
        or first.get("first_output_sha256")
        != first_cache.get("capture_prediction_sha256")
        or first.get("first_output_rank_digests")
        != first_cache.get("capture_prediction_rank_sha256")
        or checked.get("decoded_output_names")
        != [f"c0/{cell.output_name}" for cell in swap.c0_smoke_cells()]
        or checked.get("decoded_output_record_digests")
        != [object_sha256(row) for row in output_rows]
    ):
        fail("pre-full C0 first-forward/output chain differs")
    return True


def validate_receipt(value: Mapping[str, Any], *, expected_profile: Optional[str] = None) -> Mapping[str, Any]:
    top_keys = {
        "schema_version", "complete", "profile", "model_load_count",
        "one_process_one_model_load", "same_load_c0_then_full", "seed",
        "datasets", "source_video", "prompt_authority", "text_runtime",
        "orbit_review_authority",
        "model", "distributed", "first_forward_consensus", "owner_pairs",
        "input_invariants",
        "method_source", "execution",
        "c0", "full", "outputs", "artifacts", "model_integrity", "no_update",
        "processor_patch", "calibration", "interpretation", "receipt_digest",
        "terminal_authority_audit",
    }
    if (
        not isinstance(value, Mapping) or set(value) != top_keys
        or value.get("schema_version") != RECEIPT_SCHEMA
    ):
        fail("runtime receipt schema differs")
    profile = value.get("profile")
    if profile not in PROFILES or (expected_profile is not None and profile != expected_profile):
        fail("runtime receipt profile differs")
    if _embedded_digest(value, label="runtime receipt") != value.get("receipt_digest"):
        fail("runtime receipt digest differs")
    if (
        value.get("complete") is not True or value.get("model_load_count") != 1
        or value.get("one_process_one_model_load") is not True
        or value.get("same_load_c0_then_full") is not (profile == FORMAL_PROFILE)
        or value.get("seed") != SEED
    ):
        fail("runtime completion/model-load contract differs")
    datasets = value.get("datasets")
    _require_exact_keys(
        datasets, {"source_self_cross_authority", "orbit_model_inputs"},
        label="datasets",
    )
    source_cross = datasets.get("source_self_cross_authority") if isinstance(datasets, Mapping) else None
    source_keys = {
        "root", "parquet_sha256", "receipt_sha256", "receipt_digest",
        "materialization_spec_sha256", "materialization_spec_digest", "iid",
        "source_video_sha256", "row_digest", "projected_columns",
        "posterior_blob_columns_read", "latents_consumed",
    }
    if (
        not isinstance(source_cross, Mapping) or set(source_cross) != source_keys
        or source_cross.get("parquet_sha256") != SOURCE_DATASET_PARQUET_SHA256
        or source_cross.get("receipt_sha256") != SOURCE_DATASET_RECEIPT_SHA256
        or source_cross.get("receipt_digest") != SOURCE_DATASET_RECEIPT_DIGEST
        or source_cross.get("materialization_spec_sha256") != SOURCE_DATASET_SPEC_SHA256
        or source_cross.get("materialization_spec_digest") != SOURCE_DATASET_SPEC_DIGEST
        or source_cross.get("iid") != IID
        or source_cross.get("source_video_sha256") != SOURCE_VIDEO_SHA256
        or source_cross.get("projected_columns")
        != ["iid", "source_video_sha256", "row_digest"]
        or source_cross.get("posterior_blob_columns_read") != []
        or source_cross.get("latents_consumed") is not False
    ):
        fail("source-self dataset latent role differs")
    orbit = datasets.get("orbit_model_inputs") if isinstance(datasets, Mapping) else None
    orbit_keys = {
        "root", "parquet_sha256", "receipt_sha256", "receipt_digest",
        "materialization_spec_sha256", "materialization_spec_digest",
        "reference_encoding_contract_digest", "iid", "row_digest",
        "pinned_vae_identity_digest",
        "all_target_and_owner_latents_from_orbit_row", "vae_runtime_validation",
        "orbit_tensor_broadcast",
    }
    if (
        not isinstance(orbit, Mapping) or set(orbit) != orbit_keys
        or orbit.get("parquet_sha256") != ORBIT_DATASET_PARQUET_SHA256
        or orbit.get("receipt_sha256") != ORBIT_DATASET_RECEIPT_SHA256
        or orbit.get("receipt_digest") != ORBIT_DATASET_RECEIPT_DIGEST
        or orbit.get("materialization_spec_sha256") != ORBIT_DATASET_SPEC_SHA256
        or orbit.get("materialization_spec_digest") != ORBIT_DATASET_SPEC_DIGEST
        or orbit.get("reference_encoding_contract_digest")
        != ORBIT_REFERENCE_ENCODING_CONTRACT_DIGEST
        or orbit.get("iid") != IID
        or orbit.get("all_target_and_owner_latents_from_orbit_row") is not True
        or orbit.get("row_digest") != swap.ORBIT_ROW_DIGEST
        or _SHA256.fullmatch(str(orbit.get("pinned_vae_identity_digest"))) is None
        or not isinstance(orbit.get("vae_runtime_validation"), Mapping)
        or not isinstance(orbit.get("orbit_tensor_broadcast"), Mapping)
    ):
        fail("orbit model-input authority differs")
    vae_validation = _require_exact_keys(
        orbit["vae_runtime_validation"], {
            "dataset_vae_identity_digest", "training_checkpoint_root", "vae_files",
            "all_offline_encoder_files_rehashed_before_training", "digest",
        }, label="orbit VAE runtime validation",
    )
    _validate_digest_record(vae_validation, label="orbit VAE runtime validation")
    if (
        vae_validation.get("dataset_vae_identity_digest")
        != orbit.get("pinned_vae_identity_digest")
        or
        vae_validation.get("all_offline_encoder_files_rehashed_before_training") is not True
        or not isinstance(vae_validation.get("vae_files"), Mapping)
        or not vae_validation["vae_files"]
        or any(
            type(relative) is not str or not relative.startswith("vae/")
            or ".." in Path(relative).parts
            or _SHA256.fullmatch(str(item)) is None
            for relative, item in vae_validation["vae_files"].items()
        )
    ):
        fail("orbit VAE runtime file closure differs")
    orbit_broadcast = _require_exact_keys(
        orbit["orbit_tensor_broadcast"], {
            "source_rank", "tensor_digest", "tensor_sha256_by_member",
            "world4_rank_digests", "world4_consensus",
        },
        label="orbit tensor broadcast",
    )
    expected_tensor_names = {
        f"V{member}.{suffix}"
        for member in range(3)
        for suffix in ("video", "ref0", "ref27", "ref53", "ref80")
    }
    tensor_map = orbit_broadcast.get("tensor_sha256_by_member")
    if (
        orbit_broadcast.get("source_rank") != 0
        or not isinstance(tensor_map, Mapping) or set(tensor_map) != expected_tensor_names
        or any(_SHA256.fullmatch(str(item)) is None for item in tensor_map.values())
        or orbit_broadcast.get("tensor_digest") != object_sha256(tensor_map)
        or orbit_broadcast.get("world4_rank_digests")
        != [orbit_broadcast.get("tensor_digest")] * 4
        or orbit_broadcast.get("world4_consensus") is not True
    ):
        fail("orbit SP4 broadcast source rank differs")
    _sha(orbit_broadcast.get("tensor_digest"), label="orbit tensor broadcast digest")
    source_video = _require_exact_keys(
        value.get("source_video"),
        {"path", "sha256", "cross_authority_only", "model_condition_consumed"},
        label="source video",
    )
    if (
        source_video.get("sha256") != SOURCE_VIDEO_SHA256
        or source_video.get("cross_authority_only") is not True
        or source_video.get("model_condition_consumed") is not False
    ):
        fail("source display video role differs")
    if canonical_json_bytes(value.get("prompt_authority")) != canonical_json_bytes(load_prompt_authority()):
        fail("runtime prompt authority differs")
    text_runtime = value.get("text_runtime")
    text_branches = _validate_text_runtime(text_runtime)
    if canonical_json_bytes(value.get("orbit_review_authority")) != canonical_json_bytes(load_orbit_review_authority()):
        fail("runtime orbit review authority differs")
    model = _require_exact_keys(
        value.get("model"), {
            "bernini_root", "veomni_root", "checkpoint",
            "checkpoint_content_manifest", "bernini_commit", "veomni_commit",
            "checkpoint_tree_sha256", "checkpoint_manifest_sha256",
            "checkpoint_verified_file_count", "checkpoint_content_audit",
            "renderer", "transformer_count", "transformer_block_count",
        }, label="model",
    )
    if (
        model.get("bernini_commit") != EXPECTED_BERNINI_COMMIT
        or model.get("veomni_commit") != EXPECTED_VEOMNI_COMMIT
        or model.get("checkpoint_tree_sha256") != EXPECTED_CHECKPOINT_TREE_SHA256
        or model.get("checkpoint_manifest_sha256") != EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or model.get("checkpoint_verified_file_count") != 23
        or model.get("renderer") != "Bernini-R-1.3B-transformer_1"
        or model.get("transformer_count") != 1
        or model.get("transformer_block_count") != swap.TOTAL_BLOCKS
        or not isinstance(model.get("checkpoint_content_audit"), Mapping)
    ):
        fail("runtime model closure differs")
    checkpoint_audit = _require_exact_keys(
        model["checkpoint_content_audit"], {
            "manifest_path", "manifest_sha256", "verified_file_count",
            "every_non_cache_file_sha256_verified", "verified_entries_digest",
        }, label="checkpoint content audit",
    )
    if (
        checkpoint_audit.get("manifest_path") != model.get("checkpoint_content_manifest")
        or checkpoint_audit.get("manifest_sha256") != EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or checkpoint_audit.get("verified_file_count") != 23
        or checkpoint_audit.get("every_non_cache_file_sha256_verified") is not True
        or vae_validation.get("training_checkpoint_root") != model.get("checkpoint")
    ):
        fail("checkpoint content audit differs")
    _sha(checkpoint_audit.get("verified_entries_digest"), label="checkpoint entries digest")
    integrity = _validate_model_integrity(value.get("model_integrity"))
    method_source = _require_exact_keys(
        value.get("method_source"), {"revision", "archive_sha256"}, label="method source"
    )
    _sha(method_source.get("revision"), length=40, label="method source revision")
    _sha(method_source.get("archive_sha256"), label="method source archive")
    execution = _require_exact_keys(
        value.get("execution"), {
            "distributed_invocation_count", "model_load_count", "vae_load_count",
            "c0_model_forward_count", "c0_capture_forward_count",
            "c0_internal_parity_forward_count", "c0_decoded_output_count",
            "full_model_forward_count", "full_capture_forward_count",
            "full_decoded_output_count", "total_model_forward_count",
            "scheduler_instance_count", "scheduler_step_count", "optimizer_instance_count",
        }, label="execution counts",
    )
    expected_execution = {
        "distributed_invocation_count": 1, "model_load_count": 1,
        "vae_load_count": 1, "c0_model_forward_count": 10,
        "c0_capture_forward_count": 2, "c0_internal_parity_forward_count": 2,
        "c0_decoded_output_count": 6,
        "full_model_forward_count": 136 if profile == FORMAL_PROFILE else 0,
        "full_capture_forward_count": 24 if profile == FORMAL_PROFILE else 0,
        "full_decoded_output_count": 112 if profile == FORMAL_PROFILE else 0,
        "total_model_forward_count": 146 if profile == FORMAL_PROFILE else 10,
        "scheduler_instance_count": 0, "scheduler_step_count": 0,
        "optimizer_instance_count": 0,
    }
    if dict(execution) != expected_execution:
        fail("runtime single-invocation execution counts differ")
    distributed = _validate_digest_record(value.get("distributed"), label="distributed")
    _require_exact_keys(distributed, {
        "world_size", "local_world_size", "nodes", "ranks_per_node",
        "ulysses_sp_size", "sp4_crosses_nodes", "rank_hostname_local_rank",
        "topology_admission", "topology_admitted_collectively_before_output", "digest",
    }, label="distributed")
    topology = _validate_digest_record(
        distributed.get("topology_admission"), label="topology admission"
    )
    _require_exact_keys(topology, {
        "path", "world_size", "empty_on_every_rank",
        "collective_before_output_reservation", "digest",
    }, label="topology admission")
    placement = distributed.get("rank_hostname_local_rank")
    placement_valid = isinstance(placement, list) and len(placement) == 4
    hostnames: list[str] = []
    if placement_valid:
        for rank, raw in enumerate(placement):
            if not isinstance(raw, Mapping) or set(raw) != {"rank", "local_rank", "hostname"}:
                placement_valid = False
                break
            hostname = raw.get("hostname")
            if (
                raw.get("rank") != rank
                or raw.get("local_rank") != rank % 2
                or type(hostname) is not str or not hostname or "\x00" in hostname
            ):
                placement_valid = False
                break
            hostnames.append(hostname)
    placement_valid = (
        placement_valid and len(hostnames) == 4
        and hostnames[0] == hostnames[1]
        and hostnames[2] == hostnames[3]
        and hostnames[0] != hostnames[2]
    )
    if (
        distributed.get("world_size") != 4
        or distributed.get("ulysses_sp_size") != 4
        or distributed.get("nodes") != 2
        or distributed.get("ranks_per_node") != 2
        or distributed.get("topology_admitted_collectively_before_output") is not True
        or distributed.get("local_world_size") != 2
        or distributed.get("sp4_crosses_nodes") is not True
        or not placement_valid
        or topology.get("world_size") != 4
        or topology.get("empty_on_every_rank") is not True
        or topology.get("collective_before_output_reservation") is not True
    ):
        fail("runtime WORLD4 topology differs")
    first = _validate_digest_record(value.get("first_forward_consensus"), label="first forward consensus")
    _require_exact_keys(first, {
        "world_size", "passed", "before_any_real_forward",
        "actual_model_text_input_tensor_hashes", "input_bundle_digest",
        "input_rank_digests", "first_execution_id", "first_output_sha256",
        "first_output_rank_digests", "first_output_world4_consensus", "digest",
    }, label="first forward consensus")
    if (
        first.get("world_size") != 4 or first.get("passed") is not True
        or first.get("before_any_real_forward") is not True
        or first.get("actual_model_text_input_tensor_hashes") is not True
        or first.get("first_execution_id")
        != _capture_id("c0", "correct_owner", swap.C0_SCHEDULE_INDEX, "forward")
        or first.get("first_output_world4_consensus") is not True
        or first.get("input_rank_digests") != [first.get("input_bundle_digest")] * 4
        or first.get("first_output_rank_digests") != [first.get("first_output_sha256")] * 4
    ):
        fail("runtime first-forward consensus differs")
    _sha(first.get("input_bundle_digest"), label="first forward input bundle")
    _sha(first.get("first_output_sha256"), label="first forward output")
    owner_pairs = _require_exact_keys(
        value.get("owner_pairs"), {"c0", "full", "cross_schedule_closure"}, label="owner pairs"
    )
    c0_pairs = _validate_pair_rows(
        owner_pairs["c0"], (swap.C0_SCHEDULE_INDEX,), label="c0"
    )
    if profile == FORMAL_PROFILE:
        full_pairs = _validate_pair_rows(
            owner_pairs["full"], swap.policy.REGISTERED_SCHEDULE_INDICES,
            label="full",
        )
        expected_cross = build_cross_schedule_owner_closure(
            full_pairs, c0_pair=c0_pairs[swap.C0_SCHEDULE_INDEX]
        )
        if canonical_json_bytes(owner_pairs["cross_schedule_closure"]) != canonical_json_bytes(expected_cross):
            fail("owner pair cross-schedule closure differs")
    else:
        if owner_pairs["full"] is not None or owner_pairs["cross_schedule_closure"] is not None:
            fail("smoke-only owner-pair full evidence must be absent")
        full_pairs = {}
    c0_correct_binding = _binding_from_receipt(
        c0_pairs[swap.C0_SCHEDULE_INDEX]["correct_owner_binding"],
        label="orbit broadcast correct-owner binding",
    )
    c0_wrong_binding = _binding_from_receipt(
        c0_pairs[swap.C0_SCHEDULE_INDEX]["wrong_owner_binding"],
        label="orbit broadcast wrong-owner binding",
    )
    if (
        tensor_map.get("V0.video") != c0_correct_binding.decoded_owner_full_tensor_sha256
        or [tensor_map.get(f"V0.ref{index}") for index in (0, 27, 53, 80)]
        != list(c0_correct_binding.decoded_owner_reference_tensor_sha256)
        or tensor_map.get("V1.video") != c0_wrong_binding.decoded_owner_full_tensor_sha256
        or [tensor_map.get(f"V1.ref{index}") for index in (0, 27, 53, 80)]
        != list(c0_wrong_binding.decoded_owner_reference_tensor_sha256)
    ):
        fail("orbit broadcast tensors do not chain to actual owner bindings")
    invariant_pairs = full_pairs if profile == FORMAL_PROFILE else c0_pairs
    input_invariants = _validate_input_invariants(
        value.get("input_invariants"), profile=profile,
        pairs=invariant_pairs, text_bindings=text_branches,
    )
    c0 = value.get("c0")
    c0_keys = {
        "schema_version", "engineering_pass", "scientific_pass_claimed",
        "visual_selection_performed", "decoded_output_count",
        "internal_noop_parity_decoded_output_count", "noop_parity",
        "processor_audits", "cache_audits", "first_forward_consensus",
        "model_integrity", "no_update", "decoded_output_names",
        "decoded_output_record_digests", "media_complete", "digest",
    }
    if (
        not isinstance(c0, Mapping) or set(c0) != c0_keys
        or c0.get("schema_version") != C0_GATE_SCHEMA
        or c0.get("engineering_pass") is not True
        or c0.get("decoded_output_count") != 6
        or c0.get("internal_noop_parity_decoded_output_count") != 0
        or c0.get("scientific_pass_claimed") is not False
        or c0.get("visual_selection_performed") is not False
        or c0.get("media_complete") is not True
    ):
        fail("C0 engineering gate differs")
    _validate_digest_record(c0, label="C0 gate")
    c0_binding = {
        owner: c0_pairs[swap.C0_SCHEDULE_INDEX][
            f"{owner}_binding"
        ] for owner in swap.OWNERS
    }
    audits = c0.get("processor_audits")
    descriptors = _expected_c0_audit_descriptors()
    if not isinstance(audits, list) or len(audits) != len(descriptors):
        fail("C0 processor audit count differs")
    for index, (audit, descriptor) in enumerate(zip(audits, descriptors)):
        owner = str(descriptor["owner"])
        digest = _binding_from_receipt(
            c0_binding[owner], label=f"C0 {owner} binding"
        ).digest
        _validate_processor_audit(
            audit, owner=owner, schedule=int(descriptor["schedule"]),
            phase=str(descriptor["phase"]), band=str(descriptor["band"]),
            branch=str(descriptor["branch"]),
            execution_id=str(descriptor["execution_id"]),
            binding_digest=digest, label=f"C0 processor audit[{index}]",
        )
    _validate_c0_noop_parity(c0.get("noop_parity"), pairs=c0_pairs, processor_audits=audits)
    _validate_cache_audits(
        c0.get("cache_audits"), profile="c0", pairs=c0_pairs,
        processor_audits=audits, text_bindings=text_branches,
    )
    expected_input_bundle = {
        "model_sha256": integrity["pre_sha256"],
        "text_runtime_digest": text_runtime["digest"],
        "epsilon_sha256": _binding_from_receipt(
            c0_pairs[swap.C0_SCHEDULE_INDEX]["correct_owner_binding"],
            label="C0 first input binding",
        ).epsilon_sha256,
        "orbit_row_digest": swap.ORBIT_ROW_DIGEST,
        "owner_pair_digests": (
            [full_pairs[index]["digest"] for index in swap.policy.REGISTERED_SCHEDULE_INDICES]
            if profile == FORMAL_PROFILE
            else [c0_pairs[swap.C0_SCHEDULE_INDEX]["digest"]]
        ),
        "c0_plan_digest": swap.build_plan("c0-smoke")["plan_digest"],
        "full_plan_digest": (
            swap.build_plan("full-grid")["plan_digest"]
            if profile == FORMAL_PROFILE else None
        ),
        "actual_input_pre_c0_snapshot_digest": input_invariants[
            "pre_c0_snapshot"
        ]["digest"],
    }
    first_cache = c0["cache_audits"][0]
    if (
        first.get("input_bundle_digest") != object_sha256(expected_input_bundle)
        or first.get("first_execution_id") != first_cache.get("execution_id")
        or first.get("first_output_sha256")
        != first_cache.get("capture_prediction_sha256")
        or first.get("first_output_rank_digests")
        != first_cache.get("capture_prediction_rank_sha256")
    ):
        fail("first-forward input/output evidence chain differs")
    expected_c0_names = [f"c0/{cell.output_name}" for cell in swap.c0_smoke_cells()]
    output_rows = value.get("outputs")
    if not isinstance(output_rows, list):
        fail("runtime outputs must be one list")
    if (
        c0.get("decoded_output_names") != expected_c0_names
        or c0.get("decoded_output_record_digests")
        != [object_sha256(row) for row in output_rows[:6]]
        or canonical_json_bytes(c0.get("first_forward_consensus"))
        != canonical_json_bytes(value.get("first_forward_consensus"))
    ):
        fail("C0 output/first-forward evidence chain differs")
    full = value.get("full")
    if profile == FORMAL_PROFILE:
        full_keys = {
            "started_after_c0_pass", "same_model_load", "fixed_plan_no_adaptation",
            "decoded_output_count", "completed", "processor_audits", "cache_audits",
            "plan_digest", "digest",
        }
        if (
            not isinstance(full, Mapping) or set(full) != full_keys
            or full.get("started_after_c0_pass") is not True
            or full.get("same_model_load") is not True
            or full.get("fixed_plan_no_adaptation") is not True
            or full.get("decoded_output_count") != 112
            or full.get("completed") is not True
            or full.get("plan_digest") != swap.build_plan("full-grid")["plan_digest"]
        ):
            fail("formal full-grid completion differs")
        _validate_digest_record(full, label="full grid")
        full_audits = full.get("processor_audits")
        full_descriptors = _expected_full_audit_descriptors()
        if not isinstance(full_audits, list) or len(full_audits) != len(full_descriptors):
            fail("full-grid processor audit count differs")
        for index, (audit, descriptor) in enumerate(
            zip(full_audits, full_descriptors)
        ):
            owner = str(descriptor["owner"])
            schedule = int(descriptor["schedule"])
            pair = full_pairs[schedule]
            binding = _binding_from_receipt(
                pair[f"{owner}_binding"], label=f"full {owner} binding"
            )
            _validate_processor_audit(
                audit, owner=owner, schedule=schedule,
                phase=str(descriptor["phase"]), band=str(descriptor["band"]),
                branch=str(descriptor["branch"]),
                execution_id=str(descriptor["execution_id"]),
                binding_digest=binding.digest,
                label=f"full processor audit[{index}] {descriptor['branch']}",
            )
        _validate_cache_audits(
            full.get("cache_audits"), profile="full", pairs=full_pairs,
            processor_audits=full_audits, text_bindings=text_branches,
        )
        expected_output_count = 118
    else:
        if full is not None:
            fail("smoke-only receipt may not contain full-grid evidence")
        expected_output_count = 6
    expected_names = expected_artifact_names(profile)
    artifacts = value.get("artifacts")
    if (
        not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_names)
        or any(_SHA256.fullmatch(str(item)) is None for item in artifacts.values())
    ):
        fail("runtime artifact closure differs")
    all_processor_audits = list(audits) + (
        list(full.get("processor_audits", [])) if isinstance(full, Mapping) else []
    )
    all_cache_audits = list(c0.get("cache_audits", [])) + (
        list(full.get("cache_audits", [])) if isinstance(full, Mapping) else []
    )
    _validate_output_rows(
        output_rows, profile=profile, artifacts=artifacts,
        pair_rows_by_phase={"c0": c0_pairs, "full": full_pairs},
        noop_text_binding=text_branches["noop"],
        processor_by_execution={row["execution_id"]: row for row in all_processor_audits},
        cache_by_execution={row["execution_id"]: row for row in all_cache_audits},
    )
    if len(output_rows) != expected_output_count:
        fail("runtime decoded output count differs")
    no_update = _require_exact_keys(
        value.get("no_update"), {
            "gradient_enabled", "optimizer_present", "scheduler_present",
            "scheduler_steps", "parameter_gradients_present", "parameter_updates",
            "torch_inference_mode_all_forwards",
        }, label="no-update",
    )
    if no_update != {
        "gradient_enabled": False, "optimizer_present": False,
        "scheduler_present": False, "scheduler_steps": 0,
        "parameter_gradients_present": False, "parameter_updates": 0,
        "torch_inference_mode_all_forwards": True,
    }:
        fail("runtime no-update contract differs")
    c0_model = _require_exact_keys(
        c0.get("model_integrity"), {"pre_c0_sha256", "post_c0_sha256", "bytes_unchanged"},
        label="C0 model integrity",
    )
    c0_no_update = _require_exact_keys(
        c0.get("no_update"), {
            "gradient_enabled", "optimizer_present", "scheduler_present",
            "scheduler_steps", "parameter_gradients_present", "parameter_updates",
        }, label="C0 no-update",
    )
    if (
        c0_model.get("pre_c0_sha256") != integrity.get("pre_sha256")
        or c0_model.get("post_c0_sha256") != integrity.get("post_c0_sha256")
        or c0_model.get("bytes_unchanged") is not True
        or any(c0_no_update.get(key) != no_update.get(key) for key in c0_no_update)
    ):
        fail("C0 model/no-update evidence chain differs")
    _validate_processor_patch(
        value.get("processor_patch"), processor_audits=all_processor_audits,
    )
    _validate_calibration(value.get("calibration"))
    terminal_authority = _validate_digest_record(
        value.get("terminal_authority_audit"), label="terminal authority audit"
    )
    _require_exact_keys(terminal_authority, {
        "source_self_identity_digest", "orbit_identity_digest",
        "source_video_sha256", "checkpoint_content_audit_digest",
        "all_live_authorities_reopened_and_stable", "digest",
    }, label="terminal authority audit")
    if (
        terminal_authority.get("source_self_identity_digest") != object_sha256(source_cross)
        or terminal_authority.get("orbit_identity_digest")
        != object_sha256({
            key: item for key, item in orbit.items()
            if key not in {"vae_runtime_validation", "orbit_tensor_broadcast"}
        })
        or terminal_authority.get("source_video_sha256") != SOURCE_VIDEO_SHA256
        or terminal_authority.get("checkpoint_content_audit_digest")
        != object_sha256(checkpoint_audit)
        or terminal_authority.get("all_live_authorities_reopened_and_stable") is not True
    ):
        fail("terminal authority evidence chain differs")
    interpretation = value.get("interpretation")
    required = {
        "c0_engineering_only": True,
        "noop_role": "numerical_baseline_not_semantic_negative",
        "incomplete_role": "calibration_failed_exploratory_not_scientific_veto",
        "reverse_role": "directional_negative_candidate_only",
        "camera_and_appearance_role": "exploratory_confounded_nuisance_controls",
        "negative_cluster_semantically_validated": False,
        "scientific_veto_authorized": False,
        "scientific_selection_performed": False,
        "visual_adaptive_cell_selection": False,
        "automatic_scientific_claim": False,
        "method_success_claimed": False,
    }
    if not isinstance(interpretation, Mapping) or dict(interpretation) != required:
        fail("runtime scientific interpretation boundary differs")
    return value


def verify_bundle(root_value: str | Path, *, expected_profile: Optional[str] = None, verify_media: bool = True) -> Mapping[str, Any]:
    root = _plain_absolute_directory(root_value, label="runtime bundle")
    receipt_path = _plain_absolute_file(root / "receipt.json", label="runtime receipt")
    receipt_snapshot = stable_file_snapshot(
        receipt_path, label="runtime receipt", retain_bytes=True
    )
    if receipt_snapshot.raw is None:
        fail("runtime receipt bytes absent")
    parsed_receipt = _strict_json_bytes(receipt_snapshot.raw, label="runtime receipt")
    if receipt_snapshot.raw != canonical_json_bytes(parsed_receipt) + b"\n":
        fail("runtime receipt serialization differs")
    receipt = validate_receipt(parsed_receipt, expected_profile=expected_profile)
    profile = str(receipt["profile"])
    topology_path = _plain_absolute_directory(
        receipt["distributed"]["topology_admission"]["path"],
        label="live topology admission directory",
    )
    if any(topology_path.iterdir()):
        fail("live topology admission directory is no longer empty")
    datasets = receipt["datasets"]
    live_source = sealed_source_self_identity(
        datasets["source_self_cross_authority"]["root"]
    )
    if canonical_json_bytes(live_source) != canonical_json_bytes(
        datasets["source_self_cross_authority"]
    ):
        fail("live source-self cross-authority changed")
    live_orbit = sealed_orbit_identity(datasets["orbit_model_inputs"]["root"])
    declared_orbit_base = {
        key: value for key, value in datasets["orbit_model_inputs"].items()
        if key not in {"vae_runtime_validation", "orbit_tensor_broadcast"}
    }
    if canonical_json_bytes(live_orbit) != canonical_json_bytes(declared_orbit_base):
        fail("live orbit model-input authority changed")
    source_path = _plain_absolute_file(
        receipt["source_video"]["path"], label="live source display video"
    )
    if file_sha256(source_path) != SOURCE_VIDEO_SHA256:
        fail("live source display video changed")
    model = receipt["model"]
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy_train.validate_source_trees(
                model["bernini_root"], model["veomni_root"],
                expected_bernini_commit=EXPECTED_BERNINI_COMMIT,
                expected_veomni_commit=EXPECTED_VEOMNI_COMMIT,
            )
        )
        checkpoint, _ = legacy_train.validate_checkpoint(model["checkpoint"])
        checkpoint_audit = stageb_infer.validate_checkpoint_content(
            checkpoint, model["checkpoint_content_manifest"],
            expected_manifest_sha256=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        )
    except Exception as error:
        raise StageARuntimeError(f"live model closure verification failed: {error}") from error
    if (
        str(bernini_root) != model["bernini_root"]
        or str(veomni_root) != model["veomni_root"]
        or bernini_revision != model["bernini_commit"]
        or veomni_revision != model["veomni_commit"]
        or canonical_json_bytes(checkpoint_audit)
        != canonical_json_bytes(model["checkpoint_content_audit"])
    ):
        fail("live model source/checkpoint closure changed")
    legacy_train.activate_source_trees(bernini_root, veomni_root)
    try:
        import train_source_self_identity_orbit_v4 as orbit_train

        orbit_dataset = orbit_train.load_orbit_dataset(
            datasets["orbit_model_inputs"]["root"],
            expected_receipt_sha256=ORBIT_DATASET_RECEIPT_SHA256,
            expected_spec_sha256=ORBIT_DATASET_SPEC_SHA256,
        )
        live_vae_validation = orbit_train.validate_dataset_vae_against_checkpoint(
            orbit_dataset, checkpoint
        )
        orbit_dataset.assert_unchanged()
    except Exception as error:
        raise StageARuntimeError(
            f"live orbit/VAE closure verification failed: {error}"
        ) from error
    if canonical_json_bytes(live_vae_validation) != canonical_json_bytes(
        datasets["orbit_model_inputs"]["vae_runtime_validation"]
    ):
        fail("live orbit pinned VAE identity changed")
    expected_files = set(expected_artifact_names(profile)) | {"receipt.json"}
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            fail("runtime bundle contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
        elif path.is_dir() and path.relative_to(root).as_posix() not in ({"c0", "full"} if profile == FORMAL_PROFILE else {"c0"}):
            fail("runtime bundle directory closure differs")
    if actual_files != expected_files:
        fail("runtime bundle file closure differs")
    artifacts = receipt["artifacts"]
    for name in expected_artifact_names(profile):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            fail("fixed artifact path is unsafe")
        target = root / relative
        if target.is_symlink() or file_sha256(target) != artifacts[name]:
            fail(f"runtime artifact SHA differs: {name}")
    for name, plan_profile in (("c0-plan.json", "c0-smoke"), ("full-plan.json", "full-grid")):
        if name not in expected_files:
            continue
        plan = _strict_json(root / name, label=name)
        swap.validate_plan(plan, profile=plan_profile)
        plan_snapshot = stable_file_snapshot(root / name, label=name, retain_bytes=True)
        if plan_snapshot.raw != canonical_json_bytes(plan) + b"\n":
            fail(f"{name} serialization differs")
    if verify_media:
        decode = __import__("tools.materialize_vae", fromlist=["_decode_exact_video"])._decode_exact_video
        for row, (phase, cell) in zip(receipt["outputs"], _fixed_output_cells(profile)):
            expected_name = f"{phase}/{cell.output_name}"
            if row["name"] != expected_name:
                fail("runtime output path differs from fixed plan")
            path = root / expected_name
            frames, fps, hw = decode(path)
            legacy_infer.validate_exact_video_metadata(int(frames.shape[0]), fps)
            if tuple(hw) != EXPECTED_HW:
                fail("runtime MP4 geometry differs")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--profile", required=True, choices=PROFILES)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--topology-dir", required=True)
    run.add_argument("--bernini-root", required=True)
    run.add_argument("--veomni-root", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--checkpoint-content-manifest", required=True)
    run.add_argument("--source-dataset-root", required=True)
    run.add_argument("--wrong-owner-dataset-root", required=True)
    run.add_argument("--source-video", required=True)
    run.add_argument("--method-source-revision", required=True)
    run.add_argument("--method-source-archive-sha256", required=True)
    run.add_argument("--seed", type=int, default=SEED)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", required=True)
    verify.add_argument("--profile", choices=PROFILES)
    return parser


def _validate_run_cli(args: argparse.Namespace) -> None:
    if args.seed != SEED:
        fail("Stage-A seed differs")
    _sha(args.method_source_revision, length=40, label="method source revision")
    _sha(args.method_source_archive_sha256, label="method source archive SHA")
    for name in (
        "bernini_root", "veomni_root", "checkpoint", "source_dataset_root",
        "wrong_owner_dataset_root",
    ):
        _plain_absolute_directory(getattr(args, name), label=name)
    for name in ("checkpoint_content_manifest", "source_video"):
        _plain_absolute_file(getattr(args, name), label=name)
    _plain_absolute_directory(args.topology_dir, label="topology directory")


# Heavy distributed implementation is below; model-free import and verifier do
# not import torch, diffusers, transformers, or identity-orbit modules.


def _write_json_create(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return hashlib.sha256(raw).hexdigest()


def _collective_digest(value: str, *, group: Any, label: str) -> list[str]:
    import torch.distributed as dist

    value = stageb_infer.collective_rank_call(
        lambda: _sha(value, label=label), group=group,
        label=f"local {label} digest admission",
    )
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, value, group=group)
    if gathered != [value] * 4:
        fail(f"{label} WORLD4 consensus differs")
    return [str(item) for item in gathered]


def _rng_world4_vector_sha256(values: Any, *, label: str) -> str:
    if not isinstance(values, list) or len(values) != 4:
        fail(f"{label} must contain exact WORLD4 rank order")
    checked = [
        _sha(value, label=f"{label} rank {rank}")
        for rank, value in enumerate(values)
    ]
    return object_sha256({
        "schema_version": _RNG_WORLD4_VECTOR_SCHEMA,
        "world4_rank_sha256": checked,
    })


def _world4_rng_state_evidence(rows: Any) -> Mapping[str, Any]:
    """Canonicalize rank-local RNG invariants without requiring rank equality."""

    if not isinstance(rows, list) or len(rows) != 4:
        fail("capture RNG evidence must contain exact WORLD4 rows")
    before: list[str] = []
    after: list[str] = []
    for expected_rank, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "rank", "before_sha256", "after_sha256",
        }:
            fail(f"capture RNG rank {expected_rank} key closure differs")
        if type(raw.get("rank")) is not int or raw.get("rank") != expected_rank:
            fail(f"capture RNG rank order differs at {expected_rank}")
        before_sha = _sha(
            raw.get("before_sha256"),
            label=f"capture RNG rank {expected_rank} before",
        )
        after_sha = _sha(
            raw.get("after_sha256"),
            label=f"capture RNG rank {expected_rank} after",
        )
        if before_sha != after_sha:
            fail(f"capture RNG state changed on rank {expected_rank}")
        before.append(before_sha)
        after.append(after_sha)
    before_digest = _rng_world4_vector_sha256(
        before, label="capture RNG before WORLD4 vector"
    )
    after_digest = _rng_world4_vector_sha256(
        after, label="capture RNG after WORLD4 vector"
    )
    if before_digest != after_digest:
        fail("capture RNG WORLD4 vector changed")
    return {
        "rng_state_before_sha256": before_digest,
        "rng_state_after_sha256": after_digest,
        "rng_state_before_world4_rank_sha256": before,
        "rng_state_after_world4_rank_sha256": after,
        "rng_state_unchanged": True,
    }


def _collective_rng_state_evidence(
    *, before_sha256: str, after_sha256: str, group: Any, label: str,
) -> Mapping[str, Any]:
    import torch.distributed as dist

    local = stageb_infer.collective_rank_call(
        lambda: {
            "rank": int(dist.get_rank(group=group)),
            "before_sha256": _sha(
                before_sha256, label=f"{label} local RNG before"
            ),
            "after_sha256": (
                _sha(after_sha256, label=f"{label} local RNG after")
                if before_sha256 == after_sha256
                else fail(f"{label} changed local RNG state")
            ),
        },
        group=group, label=f"{label} local RNG invariant",
    )
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local, group=group)
    return stageb_infer.collective_rank_call(
        lambda: _world4_rng_state_evidence(gathered), group=group,
        label=f"{label} WORLD4 RNG evidence",
    )


def _collective_topology_admission(path: Path, *, group: Any) -> Mapping[str, Any]:
    import torch.distributed as dist

    local: Mapping[str, Any]
    try:
        root = _plain_absolute_directory(path, label="topology directory")
        entries = sorted(item.name for item in root.iterdir())
        local = {"ok": not entries, "path": str(root), "entries": entries}
    except Exception as error:
        local = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local, group=group)
    if (
        any(not isinstance(item, Mapping) or item.get("ok") is not True for item in gathered)
        or len({item.get("path") for item in gathered}) != 1
    ):
        fail(f"topology directory admission differs across WORLD4: {gathered!r}")
    dist.barrier(group=group)
    value = {
        "path": local["path"], "world_size": 4,
        "empty_on_every_rank": True,
        "collective_before_output_reservation": True,
    }
    return {**value, "digest": object_sha256(value)}


def _module_state_certificate(module: Any) -> Mapping[str, Any]:
    import torch

    rows: list[tuple[str, str, Any]] = []
    rows.extend(("parameter", name, tensor) for name, tensor in module.named_parameters())
    rows.extend(("buffer", name, tensor) for name, tensor in module.named_buffers())
    identities = [(kind, name) for kind, name, _ in rows]
    if len(identities) != len(set(identities)):
        fail("model state contains duplicate parameter/buffer names")
    digest = hashlib.sha256()
    parameter_tensors = buffer_tensors = 0
    for kind, name, tensor in sorted(rows, key=lambda item: (item[0], item[1])):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            fail("model state contains a non-materialized tensor")
        if kind == "parameter":
            parameter_tensors += 1
            if tensor.requires_grad or tensor.grad is not None:
                fail("frozen model parameter is trainable or carries a gradient")
        else:
            buffer_tensors += 1
        metadata = canonical_json_bytes({
            "kind": kind, "name": name, "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        })
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(int(raw.numel()).to_bytes(8, "big"))
        digest.update(raw.numpy().tobytes(order="C"))
    return {
        "certificate_schema": "torch-module-parameters-buffers-raw-sha256-v1",
        "sha256": digest.hexdigest(), "parameter_tensors": parameter_tensors,
        "buffer_tensors": buffer_tensors, "all_parameters_frozen": True,
        "all_parameter_gradients_absent": True,
    }


def _rng_state_sha256() -> str:
    import torch

    value = {
        "cpu": runtime.tensor_sha256(torch.random.get_rng_state()),
        "cuda": [runtime.tensor_sha256(item) for item in torch.cuda.get_rng_state_all()],
    }
    return object_sha256(value)


def _blob_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, bytes):
        fail(f"{label} must be sealed posterior bytes")
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ActualOwnerPack:
    owner: str
    schedule_index: int
    branch: Any
    binding: swap.OwnerInputBinding
    layout: swap.NativeTargetSuffixLayout


def _row_blob_hashes(row: Any, *, member_name: str) -> tuple[str, tuple[str, ...]]:
    full = _blob_sha256(
        row.posterior_blobs[f"{member_name}_full_posterior_blob"],
        label=f"{member_name} full posterior",
    )
    refs = tuple(
        _blob_sha256(
            row.posterior_blobs[f"{member_name}_ref{index}_posterior_blob"],
            label=f"{member_name} ref{index} posterior",
        )
        for index in (0, 27, 53, 80)
    )
    return full, refs


def build_actual_owner_pair(
    *, transformer: Any, native: Any, row: Any, source_member: Any,
    wrong_member: Any, epsilon: Any, schedule_index: int, sp_rank: int,
) -> tuple[ActualOwnerPack, ActualOwnerPack, Mapping[str, Any], Any]:
    """Build bindings only from the actual row bytes and device tensors used."""

    import torch

    if (
        row.iid != IID or row.row_digest != swap.ORBIT_ROW_DIGEST
        or tuple(row.variant_native_arms) != ("rv2v", "r2v")
        or tuple(row.full_shape) != (1, 32, 21, 74, 50)
        or tuple(row.reference_shape) != (1, 32, 1, 74, 50)
        or getattr(source_member, "name", None) != "V0"
        or getattr(wrong_member, "name", None) != "V1"
    ):
        fail("actual orbit row/member alias or geometry differs")
    clean = source_member.video_latent
    if (
        tuple(int(item) for item in clean.shape) != EXPECTED_LATENT_SHAPE
        or clean.dtype != torch.float32 or epsilon.dtype != torch.float32
        or epsilon.shape != clean.shape or epsilon.device != clean.device
        or clean.requires_grad or epsilon.requires_grad
        or clean.grad_fn is not None or epsilon.grad_fn is not None
        or not clean.is_contiguous() or not epsilon.is_contiguous()
        or not bool(torch.isfinite(clean).all().item())
        or not bool(torch.isfinite(epsilon).all().item())
    ):
        fail("actual source/epsilon tensor contract differs")
    x_s = ladder.shared_noise_source_state(
        clean, epsilon, exact40.PINNED_POSITIVE_SIGMAS[schedule_index]
    )
    source_blob, source_refs = _row_blob_hashes(row, member_name="source")
    wrong_blob, wrong_refs = _row_blob_hashes(row, member_name="variant_a")
    if (
        source_blob != swap.OWNER_FULL_BLOB_SHA256["correct_owner"]
        or source_refs != swap.OWNER_REFERENCE_BLOB_SHA256["correct_owner"]
        or wrong_blob != swap.OWNER_FULL_BLOB_SHA256["wrong_owner"]
        or wrong_refs != swap.OWNER_REFERENCE_BLOB_SHA256["wrong_owner"]
    ):
        fail("actual orbit posterior blob hashes differ from owner authority")
    member_by_owner = {
        "correct_owner": source_member, "wrong_owner": wrong_member,
    }
    blob_by_owner = {
        "correct_owner": (source_blob, source_refs),
        "wrong_owner": (wrong_blob, wrong_refs),
    }
    packs: dict[str, ActualOwnerPack] = {}
    for owner in swap.OWNERS:
        member = member_by_owner[owner]
        pack = native.build_native_rv2v_pack(
            transformer, donor_video=member.video_latent,
            image_references=member.image_references, noisy_target=x_s,
        )
        branch = pack.video_image
        if tuple(branch.source_ids) != swap.NATIVE_SOURCE_IDS:
            fail("actual VI pack source IDs differ")
        if int(branch.rotary.shape[2]) != branch.total_tokens:
            fail("actual VI rotary sequence geometry differs")
        prefix = branch.latents[:, : branch.condition_tokens].detach().contiguous()
        rotary_prefix = branch.rotary.narrow(2, 0, branch.condition_tokens).detach().contiguous()
        owner_blob, owner_refs = blob_by_owner[owner]
        binding = swap.OwnerInputBinding(
            owner=owner, schedule_index=schedule_index,
            timestep=exact40.PINNED_TIMESTEPS[schedule_index],
            sigma_float32_be_hex=exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index],
            orbit_row_digest=row.row_digest,
            target_source_full_blob_sha256=source_blob,
            owner_full_blob_sha256=owner_blob,
            owner_reference_blob_sha256=tuple(owner_refs),
            decoded_target_tensor_sha256=runtime.tensor_sha256(clean),
            decoded_owner_full_tensor_sha256=runtime.tensor_sha256(member.video_latent),
            decoded_owner_reference_tensor_sha256=tuple(
                runtime.tensor_sha256(item) for item in member.image_references
            ),
            epsilon_sha256=runtime.tensor_sha256(epsilon),
            target_x_s_sha256=runtime.tensor_sha256(x_s),
            prepared_visual_prefix_sha256=runtime.tensor_sha256(prefix),
            prepared_prefix_rotary_sha256=runtime.tensor_sha256(rotary_prefix),
            total_tokens=branch.total_tokens,
            condition_tokens=branch.condition_tokens,
            source_ids=tuple(branch.source_ids),
        )
        layout = swap.NativeTargetSuffixLayout(
            total_tokens=branch.total_tokens, condition_tokens=branch.condition_tokens,
            sequence_parallel_rank=sp_rank, sequence_parallel_size=SP_SIZE,
        )
        packs[owner] = ActualOwnerPack(owner, schedule_index, branch, binding, layout)
    correct, wrong = packs["correct_owner"], packs["wrong_owner"]
    if (
        not swap.raw_tensor_bytes_equal(
            correct.branch.latents[:, correct.binding.condition_tokens :],
            wrong.branch.latents[:, wrong.binding.condition_tokens :],
        )
        or not swap.raw_tensor_bytes_equal(
            correct.branch.rotary.narrow(
                2, correct.binding.condition_tokens,
                correct.binding.total_tokens - correct.binding.condition_tokens,
            ),
            wrong.branch.rotary.narrow(
                2, wrong.binding.condition_tokens,
                wrong.binding.total_tokens - wrong.binding.condition_tokens,
            ),
        )
    ):
        fail("owner pair changed actual packed target suffix/rotary")
    pair_validation = swap.validate_owner_pair_bindings(correct.binding, wrong.binding)
    target_tokens = correct.binding.total_tokens - correct.binding.condition_tokens
    layout_receipts = [
        dict(swap.NativeTargetSuffixLayout(
            total_tokens=correct.binding.total_tokens,
            condition_tokens=correct.binding.condition_tokens,
            sequence_parallel_rank=rank, sequence_parallel_size=SP_SIZE,
        ).receipt())
        for rank in range(SP_SIZE)
    ]
    tensor_bundle = {
        "decoded_target_tensor_sha256": correct.binding.decoded_target_tensor_sha256,
        "decoded_correct_owner_full_tensor_sha256": correct.binding.decoded_owner_full_tensor_sha256,
        "decoded_correct_owner_reference_tensor_sha256": list(
            correct.binding.decoded_owner_reference_tensor_sha256
        ),
        "decoded_wrong_owner_full_tensor_sha256": wrong.binding.decoded_owner_full_tensor_sha256,
        "decoded_wrong_owner_reference_tensor_sha256": list(
            wrong.binding.decoded_owner_reference_tensor_sha256
        ),
        "epsilon_sha256": correct.binding.epsilon_sha256,
        "target_x_s_sha256": correct.binding.target_x_s_sha256,
        "correct_prepared_visual_prefix_sha256": correct.binding.prepared_visual_prefix_sha256,
        "wrong_prepared_visual_prefix_sha256": wrong.binding.prepared_visual_prefix_sha256,
        "prepared_prefix_rotary_sha256": correct.binding.prepared_prefix_rotary_sha256,
    }
    tensor_bundle_digest = object_sha256(tensor_bundle)
    # The caller first places this entire local pack construction behind a
    # symmetric rank-status gate, then performs the actual digest collective.
    tensor_bundle_rank_digests = [tensor_bundle_digest] * 4
    pack_geometry = {
        "orbit_member_order": ["V0", "V1", "V2"],
        "owner_aliases": {"correct_owner": "V0/source", "wrong_owner": "V1/variant_a"},
        "full_latent_shape": list(EXPECTED_LATENT_SHAPE),
        "reference_latent_shape": [1, 16, 1, 74, 50],
        "reference_rgb_indices": [0, 27, 53, 80],
        "condition_components": ["owner_full_21", "owner_ref0_1", "owner_ref27_1", "owner_ref53_1", "owner_ref80_1"],
        "target_component": "unchanged_source_x_s_21",
        "source_ids": list(swap.NATIVE_SOURCE_IDS),
        "concat_order": list(correct.branch.concat_order),
        "total_tokens": correct.binding.total_tokens,
        "condition_tokens": correct.binding.condition_tokens,
        "target_tokens": target_tokens,
        "target_is_strict_suffix": True,
        "sp4_layout_receipts": layout_receipts,
        "append_false_then_contiguous_rank_chunks": True,
    }
    pair_unsigned = {
        "schedule_index": schedule_index,
        "correct_owner_binding": dict(correct.binding.receipt()),
        "wrong_owner_binding": dict(wrong.binding.receipt()),
        "pair_validation": dict(pair_validation),
        "actual_object_hashes_recomputed": True,
        "world4_consensus_before_forward": True,
        "actual_tensor_bundle": tensor_bundle,
        "actual_tensor_bundle_digest": tensor_bundle_digest,
        "world4_actual_tensor_bundle_rank_digests": tensor_bundle_rank_digests,
        "pack_geometry": pack_geometry,
        "pack_geometry_digest": object_sha256(pack_geometry),
        "x_s_construction": {
            "function": "source_noised_ladder_v1.shared_noise_source_state",
            "formula": "(1-sigma_float32_authority)*decoded_V0+sigma_float32_authority*epsilon",
            "schedule_index": schedule_index,
            "sigma_float32_be_hex": exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index],
            "clean_sha256": correct.binding.decoded_target_tensor_sha256,
            "epsilon_sha256": correct.binding.epsilon_sha256,
            "x_s_sha256": correct.binding.target_x_s_sha256,
            "actual_recomputed": True,
        },
    }
    pair_receipt = {**pair_unsigned, "digest": object_sha256(pair_unsigned)}
    return correct, wrong, pair_receipt, x_s


@dataclass(frozen=True)
class QueryResult:
    velocity: Any
    x0_hat: Any
    processor_audit: Mapping[str, Any]
    velocity_rank_sha256: tuple[str, str, str, str]


def _encode_all_prompts_once(
    *, renderer: Any, tokenizer: Any, prompt_authority: Mapping[str, Any],
    device: Any, world_group: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import torch

    receipts: dict[str, Mapping[str, Any]] = {}
    embeddings: dict[str, Any] = {}
    prompts = prompt_authority.get("prompts")
    if not isinstance(prompts, Mapping):
        fail("prompt authority text map differs")
    for ordinal, branch in enumerate(swap.TEXT_BRANCHES):
        def encode_one() -> tuple[Mapping[str, Any], Any]:
            tokenized = runtime.tokenize_generic_instruction(
                tokenizer, str(prompts[branch]), device
            )
            with torch.inference_mode():
                text_lens, text_embeds = renderer.get_t5_text_embeddings(
                    tokenized["input_ids"], tokenized["attention_mask"],
                    tokenized["t5_input_lens"],
                )
            text_embeds = text_embeds.detach().contiguous()
            if (
                list(text_embeds.shape) != [1, 512, 4096]
                or text_embeds.dtype != torch.bfloat16
                or text_embeds.device != device or text_embeds.requires_grad
                or text_embeds.grad_fn is not None
                or not bool(torch.isfinite(text_embeds).all().item())
                or [int(item) for item in text_lens] != [512]
                or list(tokenized["t5_input_lens"].shape) != [1, 1]
                or tokenized["input_ids"].ndim != 2
                or list(tokenized["input_ids"].shape)
                != list(tokenized["attention_mask"].shape)
                or int(tokenized["input_ids"].shape[0]) != 1
                or not 0 < int(tokenized["input_ids"].shape[1]) <= 512
            ):
                fail(f"{branch} runtime text embedding differs")
            base = {
                "branch": branch, "prompt_sha256": PROMPT_SHA256[branch],
                "input_ids_sha256": runtime.tensor_sha256(tokenized["input_ids"]),
                "attention_mask_sha256": runtime.tensor_sha256(tokenized["attention_mask"]),
                "t5_input_lens_sha256": runtime.tensor_sha256(tokenized["t5_input_lens"]),
                "text_lens": [int(item) for item in text_lens],
                "embedding_sha256": runtime.tensor_sha256(text_embeds),
                "input_ids_shape": list(tokenized["input_ids"].shape),
                "attention_mask_shape": list(tokenized["attention_mask"].shape),
                "t5_input_lens_shape": list(tokenized["t5_input_lens"].shape),
                "embedding_shape": list(text_embeds.shape),
                "embedding_dtype": str(text_embeds.dtype),
                "embedding_device_type": text_embeds.device.type,
                "encoded_call_ordinal": ordinal,
            }
            return base, text_embeds

        base, text_embeds = stageb_infer.collective_rank_call(
            encode_one, group=world_group, label=f"local {branch} text encoding"
        )
        tensor_binding_digest = object_sha256(base)
        rank_digests = _collective_digest(
            tensor_binding_digest, group=world_group,
            label=f"{branch} text tensor binding",
        )
        def finalize_text_binding() -> Mapping[str, Any]:
            unsigned = {
                **base, "runtime_tensor_binding_digest": tensor_binding_digest,
                "world4_rank_binding_digests": rank_digests,
                "world4_consensus": True,
            }
            return {**unsigned, "digest": object_sha256(unsigned)}

        receipts[branch] = stageb_infer.collective_rank_call(
            finalize_text_binding, group=world_group,
            label=f"local {branch} text receipt construction",
        )
        embeddings[branch] = text_embeds
    def finalize_text_runtime() -> Mapping[str, Any]:
        value = {
            "branches": receipts, "all_encoded_once_before_forward": True,
        }
        return {**value, "digest": object_sha256(value)}

    return (
        stageb_infer.collective_rank_call(
            finalize_text_runtime, group=world_group,
            label="local text runtime closure",
        ),
        embeddings,
    )


def _run_query(
    *, diffusion: Any, native: Any, orbit_train: Any, handle: Any,
    owner_pack: ActualOwnerPack, prompt_embeds: Any, phase: str,
    band_name: str, branch: str, execution_id: str, cache: Optional[Any],
    x_s: Any, world_group: Any,
) -> QueryResult:
    import torch

    def preflight() -> tuple[Any, Any, Any]:
        if torch.is_grad_enabled():
            fail("Stage-A query entered with gradients enabled")
        before = handle.statistics()
        invocation = swap.PromptSwapInvocation(
            phase=phase, schedule_index=owner_pack.schedule_index,
            band_name=band_name, branch=branch, owner=owner_pack.owner,
            owner_binding=owner_pack.binding, prompt_object=prompt_embeds,
            layout=owner_pack.layout, cache=cache,
        )
        timestep = torch.tensor(
            [exact40.PINNED_TIMESTEPS[owner_pack.schedule_index]],
            dtype=torch.float32, device=owner_pack.branch.latents.device,
        )
        return before, invocation, timestep

    before, invocation, timestep = stageb_infer.collective_rank_call(
        preflight, group=world_group, label=f"{execution_id} local preforward admission"
    )
    # The native call contains SP/RCCL collectives.  Its internal asymmetric
    # failure is intentionally an unrecoverable process-group failure; every
    # rank-local operation immediately before and after it is symmetrically
    # admitted, and we do not claim that wrapping RCCL makes it recoverable.
    with torch.inference_mode(), swap.activate_prompt_swap(
        invocation, encoder_hidden_states=prompt_embeds
    ), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed_prediction = native.forward_native_target_branch(
            diffusion, owner_pack.branch, timestep=timestep,
            cond_embeds=prompt_embeds,
        )
    def postflight() -> tuple[Any, Any, Mapping[str, Any], str]:
        if packed_prediction.requires_grad or packed_prediction.grad_fn is not None:
            fail("Stage-A query produced a graph-connected prediction")
        velocity = orbit_train.unpack_native_target_tokens(
            packed_prediction, video_shape=tuple(x_s.shape)
        ).detach().contiguous()
        if (
            tuple(int(item) for item in velocity.shape) != EXPECTED_LATENT_SHAPE
            or velocity.device != x_s.device or not velocity.is_floating_point()
            or not bool(torch.isfinite(velocity).all().item())
        ):
            fail("Stage-A velocity tensor contract differs")
        sigma = torch.tensor(
            exact40.PINNED_POSITIVE_SIGMAS[owner_pack.schedule_index],
            dtype=torch.float32, device=x_s.device,
        )
        x0_hat = (x_s.float() - sigma * velocity.float()).detach().contiguous()
        if (
            x0_hat.dtype != torch.float32 or x0_hat.device != x_s.device
            or tuple(int(item) for item in x0_hat.shape) != EXPECTED_LATENT_SHAPE
            or not x0_hat.is_contiguous() or not bool(torch.isfinite(x0_hat).all().item())
        ):
            fail("Stage-A x0_hat tensor contract differs")
        after = handle.statistics()
        audit = audit_processor_delta(
            before, after, phase=phase, band_name=band_name,
            owner_binding_digest=owner_pack.binding.digest, owner=owner_pack.owner,
            schedule_index=owner_pack.schedule_index, branch=branch,
            execution_id=execution_id,
        )
        return velocity, x0_hat, audit, runtime.tensor_sha256(velocity)

    velocity, x0_hat, audit, velocity_sha = stageb_infer.collective_rank_call(
        postflight, group=world_group, label=f"{execution_id} local postforward audit"
    )
    rank_hashes = _collective_digest(
        velocity_sha, group=world_group, label=f"{execution_id} velocity"
    )
    return QueryResult(
        velocity, x0_hat, audit, tuple(rank_hashes)  # type: ignore[arg-type]
    )


def _capture_branch_cache(
    *, profile: str, cache_ordinal: int, diffusion: Any, native: Any,
    orbit_train: Any, handle: Any, owner_pack: ActualOwnerPack,
    branch: str, branch_prompt: Any, branch_text_binding: Mapping[str, Any],
    x_s: Any, world_group: Any,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    execution_id = _capture_id(
        profile, owner_pack.owner, owner_pack.schedule_index, branch
    )
    cache, rng_before = stageb_infer.collective_rank_call(
        lambda: (swap.PostConditionBranchCache(branch), _rng_state_sha256()),
        group=world_group, label=f"{execution_id} local capture initialization",
    )
    result = _run_query(
        diffusion=diffusion, native=native, orbit_train=orbit_train, handle=handle,
        owner_pack=owner_pack, prompt_embeds=branch_prompt, phase="capture",
        band_name=swap.ALL30_BAND, branch=branch, execution_id=execution_id,
        cache=cache, x_s=x_s, world_group=world_group,
    )
    rng_after = stageb_infer.collective_rank_call(
        lambda: (
            _rng_state_sha256()
            if cache.sealed
            else fail("capture forward did not seal its cache")
        ),
        group=world_group, label=f"{execution_id} local capture RNG postflight",
    )
    rng_evidence = _collective_rng_state_evidence(
        before_sha256=rng_before, after_sha256=rng_after,
        group=world_group, label=execution_id,
    )

    def capture_postflight() -> Mapping[str, Any]:
        return {
            "cache_instance_ordinal": cache_ordinal, "execution_id": execution_id,
            "owner": owner_pack.owner, "schedule_index": owner_pack.schedule_index,
            "branch": branch, "owner_input_binding_digest": owner_pack.binding.digest,
            "capture_processor_audit_digest": result.processor_audit["digest"],
            "capture_prediction_sha256": runtime.tensor_sha256(result.velocity),
            "capture_prediction_rank_sha256": list(result.velocity_rank_sha256),
            "capture_prediction_world4_consensus": True,
            "capture_prompt_runtime_binding_digest": branch_text_binding["digest"],
            "capture_prompt_embedding_sha256": branch_text_binding["embedding_sha256"],
            "capture_prediction_discarded": True, "capture_decode_performed": False,
            "capture_scheduler_steps": 0, **dict(rng_evidence),
        }

    draft = stageb_infer.collective_rank_call(
        capture_postflight, group=world_group,
        label=f"{execution_id} local capture postflight",
    )
    return cache, result.processor_audit, draft


def _finalize_cache_audit(
    cache: Any, draft: Mapping[str, Any], *, profile: str,
) -> Mapping[str, Any]:
    spec = next(
        row for row in _expected_cache_specs(profile)
        if row["execution_id"] == draft["execution_id"]
    )
    terminal = cache.assert_unchanged()
    receipt = cache.receipt()
    if cache.reuse_count != spec["expected_reuse_count"]:
        fail("branch cache reuse count differs from fixed plan")
    unsigned = {
        **dict(draft), "mixed_execution_ids": list(spec["mixed_execution_ids"]),
        "expected_reuse_count": spec["expected_reuse_count"],
        "cache_receipt": dict(receipt), "terminal_cache_audit": dict(terminal),
        "fresh_for_exact_owner_schedule_branch": True,
        "cache_object_identity_unique_within_phase": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _admit_unique_cache_object(
    cache: Any, live_references: dict[int, Any], *, label: str,
) -> bool:
    for stale_identity, reference in tuple(live_references.items()):
        if not isinstance(reference, weakref.ReferenceType):
            fail(f"{label} cache identity registry differs")
        if reference() is None:
            del live_references[stale_identity]
    identity = id(cache)
    existing = live_references.get(identity)
    if existing is not None:
        live = existing()
        if live is cache:
            fail(f"{label} cache object was reused")
        if live is not None:
            fail(f"{label} cache identity collided with another live object")
    try:
        reference = weakref.ref(cache)
    except TypeError as error:
        raise StageARuntimeError(
            f"{label} cache object does not support weak identity tracking"
        ) from error
    if reference() is not cache:
        fail(f"{label} cache weak identity differs")
    live_references[identity] = reference
    return True


def _build_noop_parity_record(
    *, owner: str, schedule: int, owner_pack: ActualOwnerPack,
    plain: QueryResult, noop_swap: QueryResult, plain_execution_id: str,
) -> Mapping[str, Any]:
    if (
        not swap.raw_tensor_bytes_equal(plain.velocity, noop_swap.velocity)
        or not swap.raw_tensor_bytes_equal(plain.x0_hat, noop_swap.x0_hat)
    ):
        fail("C0 internal noop-swap full velocity/predecode parity failed")
    unsigned = {
        "owner": owner, "schedule_index": schedule, "band_name": swap.C0_BAND,
        "plain_execution_id": plain_execution_id,
        "swap_execution_id": _noop_swap_id(owner),
        "owner_input_binding_digest": owner_pack.binding.digest,
        "plain_processor_audit_digest": plain.processor_audit["digest"],
        "swap_processor_audit_digest": noop_swap.processor_audit["digest"],
        "plain_vs_swap_velocity_raw_bytes_equal": True,
        "plain_vs_swap_predecode_raw_bytes_equal": True,
        "plain_velocity_sha256": runtime.tensor_sha256(plain.velocity),
        "swap_velocity_sha256": runtime.tensor_sha256(noop_swap.velocity),
        "plain_predecode_sha256": runtime.tensor_sha256(plain.x0_hat),
        "swap_predecode_sha256": runtime.tensor_sha256(noop_swap.x0_hat),
        "internal_noop_swap_decoded": False,
        "full_velocity_compared": True, "full_predecode_compared": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _latent_output_record(
    *, phase: str, cell: swap.ExperimentCell, owner_pack: ActualOwnerPack,
    pair_receipt: Mapping[str, Any], query: QueryResult,
    text_binding: Mapping[str, Any], cache_audit_digest: Optional[str],
) -> Mapping[str, Any]:
    if cell.owner != owner_pack.owner or cell.schedule_index != owner_pack.schedule_index:
        fail("output cell/actual owner pack differs")
    return {
        "name": f"{phase}/{cell.output_name}", "phase": phase,
        "cell": dict(cell.receipt()), "cell_digest": object_sha256(cell.receipt()),
        "owner_input_binding": dict(owner_pack.binding.receipt()),
        "owner_input_binding_digest": owner_pack.binding.digest,
        "owner_pair_validation_digest": pair_receipt["pair_validation"]["digest"],
        "velocity_sha256": runtime.tensor_sha256(query.velocity),
        "predecode_x0_hat_sha256": runtime.tensor_sha256(query.x0_hat),
        "decode_input_latent_sha256": runtime.tensor_sha256(query.x0_hat),
        "decode_input_latent_shape": list(query.x0_hat.shape),
        "decode_input_dtype": str(query.x0_hat.dtype),
        "decode_input_device_type": query.x0_hat.device.type,
        "decode_input_contiguous": bool(query.x0_hat.is_contiguous()),
        "decode_input_finite": True,
        "global_prompt_branch": "noop",
        "global_noop_prompt_runtime_binding_digest": text_binding["digest"],
        "global_noop_embedding_sha256": text_binding["embedding_sha256"],
        "processor_audit_digest": query.processor_audit["digest"],
        "branch_capture_cache_audit_digest": cache_audit_digest,
        "result_rank_sha256": list(query.velocity_rank_sha256),
        "result_world4_consensus": True,
        "actual_object_binding_used_for_forward": True,
    }


def _rank_zero_call(
    callback: Callable[[], Any], *, rank: int, group: Any, label: str,
) -> Any:
    import torch.distributed as dist

    envelope: list[Any] = [None]
    if rank == 0:
        try:
            envelope[0] = {"ok": True, "value": callback()}
        except Exception as error:
            envelope[0] = {
                "ok": False, "error_type": type(error).__name__, "error": str(error),
            }
    dist.broadcast_object_list(envelope, src=0, group=group)
    result = envelope[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"{label} failed on rank zero: {result!r}")
    return result.get("value")


def _prepare_phase_directory(
    *, stage: Path, phase: str, rank: int, group: Any,
) -> str:
    profile = "c0-smoke" if phase == "c0" else "full-grid"

    def create() -> str:
        directory = stage / phase
        os.mkdir(directory, mode=0o700)
        return _write_json_create(stage / f"{phase}-plan.json", swap.build_plan(profile))

    result = _rank_zero_call(create, rank=rank, group=group, label=f"prepare {phase}")
    import torch.distributed as dist

    dist.barrier(group=group)
    return str(result)


def _load_rank_zero_vae(
    *, checkpoint: Path, device: Any, rank: int, group: Any,
) -> Any:
    from diffusers.models import AutoencoderKLWan

    vae: Any = None
    envelope: list[Any] = [None]
    if rank == 0:
        try:
            vae = AutoencoderKLWan.from_pretrained(
                str(checkpoint), subfolder="vae", torch_dtype=__import__("torch").float32,
                local_files_only=True,
            ).to(device)
            vae.requires_grad_(False)
            vae.eval()
            envelope[0] = {"ok": True}
        except Exception as error:
            envelope[0] = {
                "ok": False, "error_type": type(error).__name__, "error": str(error),
            }
    import torch.distributed as dist

    dist.broadcast_object_list(envelope, src=0, group=group)
    if not isinstance(envelope[0], Mapping) or envelope[0].get("ok") is not True:
        fail(f"rank-zero VAE load failed: {envelope[0]!r}")
    return vae


def _move_rank_zero_module(
    module: Any, destination: Any, *, rank: int, group: Any, label: str,
) -> None:
    def move() -> bool:
        if module is None:
            fail(f"rank-zero {label} module is absent")
        module.to(destination)
        return True

    if _rank_zero_call(move, rank=rank, group=group, label=label) is not True:
        fail(f"rank-zero {label} move acknowledgement differs")


def _decode_phase_rank_zero(
    *, stage: Path, phase: str, vae: Any, latent_rows: Sequence[tuple[Mapping[str, Any], Any]],
    device: Any, rank: int, group: Any, vae_decode: Any, save_output: Any,
) -> list[Mapping[str, Any]]:
    def decode() -> list[Mapping[str, Any]]:
        if vae is None:
            fail("rank-zero VAE is absent")
        torch = __import__("torch")
        decode_exact = __import__(
            "tools.materialize_vae", fromlist=["_decode_exact_video"]
        )._decode_exact_video
        outputs: list[Mapping[str, Any]] = []
        for base, cpu_latent in latent_rows:
            latent = cpu_latent.to(device=device, dtype=torch.float32).contiguous()
            if (
                tuple(int(item) for item in latent.shape) != EXPECTED_LATENT_SHAPE
                or latent.requires_grad or latent.grad_fn is not None
                or not latent.is_contiguous()
                or not bool(torch.isfinite(latent).all().item())
                or runtime.tensor_sha256(latent) != base["decode_input_latent_sha256"]
            ):
                fail("rank-zero VAE input differs from consensus predecode latent")
            with torch.inference_mode():
                decoded = vae_decode(vae, latent)
            path = stage / str(base["name"])
            stageb_infer.save_validated_vae_decoded_clip(
                decoded, output_path=path, expected_height=EXPECTED_HW[0],
                expected_width=EXPECTED_HW[1], fps=FPS, save_output_fn=save_output,
            )
            frames, reported_fps, hw = decode_exact(path)
            legacy_infer.validate_exact_video_metadata(int(frames.shape[0]), reported_fps)
            if tuple(hw) != EXPECTED_HW:
                fail("decoded Stage-A MP4 geometry differs")
            outputs.append({
                **dict(base), "sha256": file_sha256(path),
                "frames": EXPECTED_FRAMES, "fps": float(FPS), "hw": list(hw),
                "vae_frozen_eval": True,
            })
        return outputs

    result = _rank_zero_call(
        decode, rank=rank, group=group, label=f"decode {phase} outputs"
    )
    if not isinstance(result, list):
        fail(f"decoded {phase} record broadcast differs")
    return result


def _execute_c0(
    *, diffusion: Any, native: Any, orbit_train: Any, handle: Any,
    packs: Mapping[int, Mapping[str, ActualOwnerPack]],
    pair_receipts: Mapping[int, Mapping[str, Any]], x_s_by_schedule: Mapping[int, Any],
    prompt_embeddings: Mapping[str, Any], text_bindings: Mapping[str, Mapping[str, Any]],
    world_group: Any,
) -> Mapping[str, Any]:
    schedule = swap.C0_SCHEDULE_INDEX
    processor_audits: list[Mapping[str, Any]] = []
    cache_audits: list[Mapping[str, Any]] = []
    parity_rows: list[Mapping[str, Any]] = []
    latent_rows: list[tuple[Mapping[str, Any], Any]] = []
    live_cache_identities: dict[int, Any] = {}
    first_capture: Optional[Mapping[str, Any]] = None
    cells = swap.c0_smoke_cells()
    for cache_ordinal, owner in enumerate(swap.OWNERS):
        owner_pack = packs[schedule][owner]
        cache, capture_audit, draft = _capture_branch_cache(
            profile="c0", cache_ordinal=cache_ordinal, diffusion=diffusion,
            native=native, orbit_train=orbit_train, handle=handle,
            owner_pack=owner_pack, branch="forward",
            branch_prompt=prompt_embeddings["forward"],
            branch_text_binding=text_bindings["forward"],
            x_s=x_s_by_schedule[schedule], world_group=world_group,
        )
        stageb_infer.collective_rank_call(
            lambda: _admit_unique_cache_object(
                cache, live_cache_identities, label="C0 owner axis"
            ),
            group=world_group, label=f"C0 {owner} local cache freshness",
        )
        processor_audits.append(capture_audit)
        if first_capture is None:
            first_capture = dict(draft)
        owner_cells = [cell for cell in cells if cell.owner == owner]
        plain = _run_query(
            diffusion=diffusion, native=native, orbit_train=orbit_train,
            handle=handle, owner_pack=owner_pack,
            prompt_embeds=prompt_embeddings["noop"], phase="plain_noop",
            band_name=swap.NONE_BAND, branch="noop",
            execution_id=owner_cells[0].cell_id, cache=None,
            x_s=x_s_by_schedule[schedule], world_group=world_group,
        )
        processor_audits.append(plain.processor_audit)
        noop_swap = _run_query(
            diffusion=diffusion, native=native, orbit_train=orbit_train,
            handle=handle, owner_pack=owner_pack,
            prompt_embeds=prompt_embeddings["noop"], phase="noop_swap",
            band_name=swap.C0_BAND, branch="noop",
            execution_id=_noop_swap_id(owner), cache=None,
            x_s=x_s_by_schedule[schedule], world_group=world_group,
        )
        processor_audits.append(noop_swap.processor_audit)
        parity_rows.append(stageb_infer.collective_rank_call(
            lambda: _build_noop_parity_record(
                owner=owner, schedule=schedule, owner_pack=owner_pack,
                plain=plain, noop_swap=noop_swap,
                plain_execution_id=owner_cells[0].cell_id,
            ),
            group=world_group, label=f"C0 {owner} local noop byte parity",
        ))
        mixed_results: list[QueryResult] = []
        for cell in owner_cells[1:]:
            query = _run_query(
                diffusion=diffusion, native=native, orbit_train=orbit_train,
                handle=handle, owner_pack=owner_pack,
                prompt_embeds=prompt_embeddings["noop"], phase="mixed",
                band_name=cell.band_name, branch="forward",
                execution_id=cell.cell_id, cache=cache,
                x_s=x_s_by_schedule[schedule], world_group=world_group,
            )
            processor_audits.append(query.processor_audit)
            mixed_results.append(query)
        cache_audit = stageb_infer.collective_rank_call(
            lambda: _finalize_cache_audit(cache, draft, profile="c0"),
            group=world_group, label=f"C0 {owner} terminal cache audit",
        )
        cache_audits.append(cache_audit)
        pair = pair_receipts[schedule]
        records_and_queries = [(owner_cells[0], plain), *zip(owner_cells[1:], mixed_results)]
        for cell, query in records_and_queries:
            latent_rows.append(stageb_infer.collective_rank_call(
                lambda cell=cell, query=query: _cpu_latent_row(
                    phase="c0", cell=cell, owner_pack=owner_pack,
                    pair_receipt=pair, query=query, text_binding=text_bindings["noop"],
                    cache_audit_digest=(
                        None if cell.role == "noop_baseline" else cache_audit["digest"]
                    ),
                ),
                group=world_group, label=f"{cell.cell_id} local CPU output record",
            ))
    stageb_infer.collective_rank_call(
        lambda: (
            True if first_capture is not None and len(processor_audits) == 10
            and len(cache_audits) == 2 and len(latent_rows) == 6
            else fail("C0 execution closure differs")
        ),
        group=world_group, label="C0 local execution closure",
    )
    return {
        "processor_audits": processor_audits, "cache_audits": cache_audits,
        "noop_parity": parity_rows, "latent_rows": latent_rows,
        "first_capture": first_capture,
    }


def _execute_full(
    *, diffusion: Any, native: Any, orbit_train: Any, handle: Any,
    packs: Mapping[int, Mapping[str, ActualOwnerPack]],
    pair_receipts: Mapping[int, Mapping[str, Any]], x_s_by_schedule: Mapping[int, Any],
    prompt_embeddings: Mapping[str, Any], text_bindings: Mapping[str, Mapping[str, Any]],
    world_group: Any,
) -> Mapping[str, Any]:
    processor_audits: list[Mapping[str, Any]] = []
    cache_drafts: dict[tuple[int, str, str], tuple[Any, Mapping[str, Any]]] = {}
    latent_rows: list[tuple[Mapping[str, Any], Any]] = []
    finalized_by_id: dict[str, Mapping[str, Any]] = {}
    live_cache_identities: dict[int, Any] = {}
    cache_ordinal = 0
    cells = swap.full_grid_cells()
    for schedule in swap.policy.REGISTERED_SCHEDULE_INDICES:
        for owner, branches in (
            ("correct_owner", swap.NONNOOP_BRANCHES),
            ("wrong_owner", ("forward",)),
        ):
            for branch in branches:
                cache, audit, draft = _capture_branch_cache(
                    profile="full", cache_ordinal=cache_ordinal,
                    diffusion=diffusion, native=native, orbit_train=orbit_train,
                    handle=handle, owner_pack=packs[schedule][owner], branch=branch,
                    branch_prompt=prompt_embeddings[branch],
                    branch_text_binding=text_bindings[branch],
                    x_s=x_s_by_schedule[schedule], world_group=world_group,
                )
                processor_audits.append(audit)
                stageb_infer.collective_rank_call(
                    lambda cache=cache: _admit_unique_cache_object(
                        cache, live_cache_identities,
                        label="full owner/schedule/branch",
                    ),
                    group=world_group,
                    label=f"full s{schedule} {owner} {branch} local cache freshness",
                )
                cache_drafts[(schedule, owner, branch)] = (cache, draft)
                cache_ordinal += 1
        schedule_cells = [cell for cell in cells if cell.schedule_index == schedule]
        query_by_cell: dict[str, QueryResult] = {}
        for cell in schedule_cells:
            if cell.role == "noop_baseline":
                phase_name, cache = "plain_noop", None
            else:
                phase_name = "mixed"
                cache = cache_drafts[(schedule, cell.owner, cell.branch)][0]
            query = _run_query(
                diffusion=diffusion, native=native, orbit_train=orbit_train,
                handle=handle, owner_pack=packs[schedule][cell.owner],
                prompt_embeds=prompt_embeddings["noop"], phase=phase_name,
                band_name=cell.band_name, branch=cell.branch,
                execution_id=cell.cell_id, cache=cache,
                x_s=x_s_by_schedule[schedule], world_group=world_group,
            )
            processor_audits.append(query.processor_audit)
            query_by_cell[cell.cell_id] = query
        finalized: dict[tuple[str, str], Mapping[str, Any]] = {}
        for owner, branches in (
            ("correct_owner", swap.NONNOOP_BRANCHES),
            ("wrong_owner", ("forward",)),
        ):
            for branch in branches:
                cache, draft = cache_drafts[(schedule, owner, branch)]
                finalized[(owner, branch)] = stageb_infer.collective_rank_call(
                    lambda cache=cache, draft=draft: _finalize_cache_audit(
                        cache, draft, profile="full"
                    ),
                    group=world_group,
                    label=f"full s{schedule} {owner} {branch} terminal cache audit",
                )
                finalized_by_id[
                    str(finalized[(owner, branch)]["execution_id"])
                ] = finalized[(owner, branch)]
                del cache_drafts[(schedule, owner, branch)]
        for cell in schedule_cells:
            query = query_by_cell[cell.cell_id]
            cache_digest = (
                None if cell.role == "noop_baseline"
                else finalized[(cell.owner, cell.branch)]["digest"]
            )
            latent_rows.append(stageb_infer.collective_rank_call(
                lambda cell=cell, query=query, cache_digest=cache_digest: _cpu_latent_row(
                    phase="full", cell=cell,
                    owner_pack=packs[schedule][cell.owner],
                    pair_receipt=pair_receipts[schedule], query=query,
                    text_binding=text_bindings["noop"],
                    cache_audit_digest=cache_digest,
                ),
                group=world_group, label=f"{cell.cell_id} local CPU output record",
            ))
    ordered_cache_audits = [
        finalized_by_id[str(spec["execution_id"])]
        for spec in _expected_cache_specs("full")
    ]
    stageb_infer.collective_rank_call(
        lambda: (
            True if len(processor_audits) == 136
            and len(ordered_cache_audits) == 24 and len(latent_rows) == 112
            else fail("full-grid execution closure differs")
        ),
        group=world_group, label="full-grid local execution closure",
    )
    return {
        "processor_audits": processor_audits,
        "cache_audits": ordered_cache_audits, "latent_rows": latent_rows,
    }


def _calibration_receipt() -> Mapping[str, Any]:
    value = {
        "family4_seed": CALIBRATION_SEED,
        "family4_initial_gaussian_raw_sha256": CALIBRATION_INITIAL_GAUSSIAN_RAW_SHA256,
        "stage_a_seed": SEED, "family4_noise_consumed_by_stage_a": False,
        "family4_and_stage_a_seed_distinct": True,
        "branch_authority": swap.branch_calibration_authority(),
    }
    return {**value, "digest": object_sha256(value)}


def _interpretation_receipt() -> Mapping[str, Any]:
    return {
        "c0_engineering_only": True,
        "noop_role": "numerical_baseline_not_semantic_negative",
        "incomplete_role": "calibration_failed_exploratory_not_scientific_veto",
        "reverse_role": "directional_negative_candidate_only",
        "camera_and_appearance_role": "exploratory_confounded_nuisance_controls",
        "negative_cluster_semantically_validated": False,
        "scientific_veto_authorized": False,
        "scientific_selection_performed": False,
        "visual_adaptive_cell_selection": False,
        "automatic_scientific_claim": False,
        "method_success_claimed": False,
    }


def _directory_reservation_identity(path: Path) -> tuple[int, int, int, int, int]:
    current = os.lstat(path)
    if not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode):
        fail("Stage-A staging reservation is not a plain directory")
    return (
        int(current.st_dev), int(current.st_ino), stat.S_IMODE(current.st_mode),
        int(current.st_uid), int(current.st_gid),
    )


def _publish_bundle(
    *, output: Path, stage: Path, receipt: Mapping[str, Any],
    reserved_stage_identity: Sequence[int], rank: int, group: Any,
) -> None:
    def publish() -> Mapping[str, Any]:
        expected_identity = tuple(int(item) for item in reserved_stage_identity)
        if len(expected_identity) != 5:
            fail("Stage-A staging reservation identity shape differs")
        identity = _directory_reservation_identity(stage)
        reserved = runtime._OUTPUT_STAGE_IDENTITIES.get(str(stage))
        if (
            identity != expected_identity or reserved != identity[:3]
        ):
            fail("reserved Stage-A staging directory identity differs")
        _write_json_create(stage / "receipt.json", receipt)
        if _directory_reservation_identity(stage) != expected_identity:
            fail("Stage-A staging directory changed while writing receipt")
        verify_bundle(stage, expected_profile=str(receipt["profile"]), verify_media=True)
        runtime.fsync_directory(stage)
        if _directory_reservation_identity(stage) != expected_identity:
            fail("Stage-A staging directory changed after verification")
        runtime._rename_directory_noreplace(stage, output)
        runtime.fsync_directory(output.parent)
        if _directory_reservation_identity(output) != expected_identity:
            fail("published Stage-A directory differs from reserved inode")
        verify_bundle(output, expected_profile=str(receipt["profile"]), verify_media=True)
        runtime._OUTPUT_STAGE_IDENTITIES.pop(str(stage), None)
        return {"output": str(output), "receipt_digest": receipt["receipt_digest"]}

    _rank_zero_call(publish, rank=rank, group=group, label="publish Stage-A bundle")


def _terminal_authority_audit(
    *, source_identity: Mapping[str, Any], orbit_identity: Mapping[str, Any],
    source_video: StableFileSnapshot, dataset: Any, bernini_root: Path,
    veomni_root: Path, checkpoint: Path, checkpoint_manifest: Path,
) -> Mapping[str, Any]:
    live_source = sealed_source_self_identity(source_identity["root"])
    live_orbit = sealed_orbit_identity(orbit_identity["root"])
    current_source_video = stable_file_snapshot(
        source_video.path, label="terminal source display video"
    )
    dataset.assert_unchanged()
    bernini_checked, veomni_checked, bernini_revision, veomni_revision = (
        legacy_train.validate_source_trees(
            bernini_root, veomni_root,
            expected_bernini_commit=EXPECTED_BERNINI_COMMIT,
            expected_veomni_commit=EXPECTED_VEOMNI_COMMIT,
        )
    )
    checkpoint_checked, _ = legacy_train.validate_checkpoint(checkpoint)
    checkpoint_audit = stageb_infer.validate_checkpoint_content(
        checkpoint_checked, checkpoint_manifest,
        expected_manifest_sha256=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
    )
    if (
        canonical_json_bytes(live_source) != canonical_json_bytes(source_identity)
        or canonical_json_bytes(live_orbit) != canonical_json_bytes(orbit_identity)
        or current_source_video.identity != source_video.identity
        or current_source_video.sha256 != source_video.sha256
        or bernini_checked != bernini_root or veomni_checked != veomni_root
        or bernini_revision != EXPECTED_BERNINI_COMMIT
        or veomni_revision != EXPECTED_VEOMNI_COMMIT
        or checkpoint_checked != checkpoint
    ):
        fail("terminal source/orbit/model authority changed")
    unsigned = {
        "source_self_identity_digest": object_sha256(live_source),
        "orbit_identity_digest": object_sha256(live_orbit),
        "source_video_sha256": current_source_video.sha256,
        "checkpoint_content_audit_digest": object_sha256(checkpoint_audit),
        "all_live_authorities_reopened_and_stable": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _load_local_orbit_inputs(
    *, orbit_train: Any, dataset_root: str, checkpoint: Path, device: Any,
) -> tuple[Any, Mapping[str, Any], Any, Any, Any]:
    dataset = orbit_train.load_orbit_dataset(
        dataset_root, expected_receipt_sha256=ORBIT_DATASET_RECEIPT_SHA256,
        expected_spec_sha256=ORBIT_DATASET_SPEC_SHA256,
    )
    vae_runtime_validation = orbit_train.validate_dataset_vae_against_checkpoint(
        dataset, checkpoint
    )
    selected_rows = [row for row in dataset.rows if row.iid == IID]
    if len(selected_rows) != 1:
        fail("orbit runtime dataset IID00435 uniqueness differs")
    row = selected_rows[0]
    mean, std, channels = legacy_train._vae_statistics(checkpoint)
    if channels != 16:
        fail("VAE latent channel geometry differs")
    orbit = orbit_train.build_identity_orbit_from_row(
        row, vae_mean=mean, vae_std=std, device=device
    )
    if tuple(member.name for member in orbit.members) != ("V0", "V1", "V2"):
        fail("actual orbit member order differs")
    return dataset, vae_runtime_validation, row, orbit, (mean, std)


def _post_broadcast_orbit_admission(
    *, dataset: Any, orbit: Any, broadcast: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, Any]]:
    dataset.assert_unchanged()
    members = tuple(orbit.members)
    if tuple(getattr(member, "name", None) for member in members) != ("V0", "V1", "V2"):
        fail("post-broadcast orbit member order differs")
    tensor_map: dict[str, str] = {}
    for member in members:
        if len(tuple(member.image_references)) != 4:
            fail("post-broadcast orbit reference count differs")
        tensor_map[f"{member.name}.video"] = runtime.tensor_sha256(member.video_latent)
        for index, tensor in zip((0, 27, 53, 80), member.image_references):
            tensor_map[f"{member.name}.ref{index}"] = runtime.tensor_sha256(tensor)
    digest = object_sha256(tensor_map)
    if broadcast.get("source_rank") != 0 or broadcast.get("tensor_digest") != digest:
        fail("post-broadcast orbit tensor digest differs")
    record = {
        "source_rank": 0, "tensor_digest": digest,
        "tensor_sha256_by_member": tensor_map,
        "world4_rank_digests": [digest] * 4, "world4_consensus": True,
    }
    return members[0], members[1], record


def _load_local_renderer_tokenizer(
    *, BerniniRendererConfig: Any, BerniniRendererModel: Any,
    AutoTokenizer: Any, bernini_root: Path, checkpoint: Path,
    transformer_config: Mapping[str, Any], device: Any,
) -> tuple[Any, Any, Any]:
    torch = __import__("torch")
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True, **legacy_train.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy_train.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).to(device)
    renderer.requires_grad_(False)
    renderer.eval()
    transformer = renderer.diff_dec.transformer
    if (
        transformer is None or renderer.diff_dec.transformer_2 is not None
        or len(tuple(transformer.blocks)) != swap.TOTAL_BLOCKS
    ):
        fail("Stage-A requires one exact 30-block Bernini transformer_1")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy_infer.tokenizer_load_kwargs()
    )
    return renderer, transformer, tokenizer


def _make_stage_a_epsilon(*, torch: Any, device: Any) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    value = torch.randn(
        EXPECTED_LATENT_SHAPE, generator=generator, dtype=torch.float32
    ).to(device=device).contiguous()
    if (
        value.requires_grad or value.grad_fn is not None or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        fail("Stage-A epsilon tensor contract differs")
    return value


def _assert_same_module_certificate(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, label: str,
) -> bool:
    if canonical_json_bytes(expected) != canonical_json_bytes(observed):
        fail(f"transformer bytes changed during {label}")
    return True


def _cpu_latent_row(
    *, phase: str, cell: swap.ExperimentCell, owner_pack: ActualOwnerPack,
    pair_receipt: Mapping[str, Any], query: QueryResult,
    text_binding: Mapping[str, Any], cache_audit_digest: Optional[str],
) -> tuple[Mapping[str, Any], Any]:
    import torch

    base = _latent_output_record(
        phase=phase, cell=cell, owner_pack=owner_pack,
        pair_receipt=pair_receipt, query=query, text_binding=text_binding,
        cache_audit_digest=cache_audit_digest,
    )
    latent = query.x0_hat.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        latent.device.type != "cpu" or latent.dtype != torch.float32
        or tuple(int(item) for item in latent.shape) != EXPECTED_LATENT_SHAPE
        or not latent.is_contiguous() or not bool(torch.isfinite(latent).all().item())
    ):
        fail("CPU decode latent contract differs")
    return base, latent


def snapshot_actual_model_inputs(
    *, packs: Mapping[int, Mapping[str, ActualOwnerPack]],
    x_s_by_schedule: Mapping[int, Any], epsilon: Any,
    prompt_embeddings: Mapping[str, Any],
    text_bindings: Mapping[str, Mapping[str, Any]],
    schedules: Sequence[int],
) -> Mapping[str, Any]:
    """Re-hash every long-lived tensor/object reused by Stage-A forwards."""

    epsilon_sha = runtime.tensor_sha256(epsilon)
    prompt_hashes: dict[str, str] = {}
    for branch in swap.TEXT_BRANCHES:
        if branch not in prompt_embeddings or branch not in text_bindings:
            fail("actual prompt embedding closure differs")
        prompt_sha = runtime.tensor_sha256(prompt_embeddings[branch])
        if prompt_sha != text_bindings[branch].get("embedding_sha256"):
            fail(f"actual {branch} prompt embedding changed")
        prompt_hashes[branch] = prompt_sha
    schedule_inputs: dict[str, Mapping[str, Any]] = {}
    for schedule in schedules:
        if schedule not in packs or schedule not in x_s_by_schedule:
            fail("actual schedule input closure differs")
        x_s_sha = runtime.tensor_sha256(x_s_by_schedule[schedule])
        owners: dict[str, Mapping[str, Any]] = {}
        for owner in swap.OWNERS:
            pack = packs[schedule][owner]
            binding = pack.binding
            branch = pack.branch
            prefix = branch.latents[:, : binding.condition_tokens].detach().contiguous()
            rotary_prefix = branch.rotary.narrow(
                2, 0, binding.condition_tokens
            ).detach().contiguous()
            if (
                pack.owner != owner or pack.schedule_index != schedule
                or binding.owner != owner or binding.schedule_index != schedule
                or tuple(branch.source_ids) != binding.source_ids
                or x_s_sha != binding.target_x_s_sha256
                or epsilon_sha != binding.epsilon_sha256
                or runtime.tensor_sha256(prefix)
                != binding.prepared_visual_prefix_sha256
                or runtime.tensor_sha256(rotary_prefix)
                != binding.prepared_prefix_rotary_sha256
            ):
                fail("actual owner pack no longer matches its sealed binding")
            owners[owner] = {
                "owner_input_binding_digest": binding.digest,
                "packed_latents_sha256": runtime.tensor_sha256(branch.latents),
                "packed_rotary_sha256": runtime.tensor_sha256(branch.rotary),
                "source_ids": list(branch.source_ids),
                "prepared_visual_prefix_sha256": binding.prepared_visual_prefix_sha256,
                "prepared_prefix_rotary_sha256": binding.prepared_prefix_rotary_sha256,
            }
        schedule_inputs[str(schedule)] = {
            "x_s_sha256": x_s_sha, "owners": owners,
        }
    unsigned = {
        "epsilon_sha256": epsilon_sha,
        "prompt_embedding_sha256_by_branch": prompt_hashes,
        "schedule_inputs": schedule_inputs,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def build_input_invariant_receipt(
    *, profile: str, pre_c0: Mapping[str, Any], post_c0: Mapping[str, Any],
    post_full: Optional[Mapping[str, Any]], terminal: Mapping[str, Any],
) -> Mapping[str, Any]:
    if canonical_json_bytes(pre_c0) != canonical_json_bytes(post_c0):
        fail("actual model inputs changed during C0")
    if profile == FORMAL_PROFILE:
        if post_full is None or canonical_json_bytes(pre_c0) != canonical_json_bytes(post_full):
            fail("actual model inputs changed during full grid")
    elif post_full is not None:
        fail("smoke-only input invariant unexpectedly has a full phase")
    if canonical_json_bytes(pre_c0) != canonical_json_bytes(terminal):
        fail("actual model inputs changed before terminal receipt")
    digest = str(pre_c0["digest"])
    stage_digests: dict[str, Any] = {
        "pre_c0": digest,
        "post_c0": str(post_c0["digest"]),
        "post_full": str(post_full["digest"]) if post_full is not None else None,
        "terminal": str(terminal["digest"]),
    }
    rank_digests = {
        stage: ([value] * 4 if value is not None else None)
        for stage, value in stage_digests.items()
    }
    unsigned = {
        "schema_version": "stage-a-actual-model-input-invariants-v1",
        "pre_c0_snapshot": dict(pre_c0),
        "stage_snapshot_digests": stage_digests,
        "stage_world4_rank_digests": rank_digests,
        "full_phase_executed": profile == FORMAL_PROFILE,
        "actual_objects_rehashed_at_each_phase": True,
        "all_actual_input_bytes_unchanged": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _build_first_forward_record(
    *, c0_execution: Mapping[str, Any], input_bundle_digest: str,
    input_rank_digests: Sequence[str],
) -> Mapping[str, Any]:
    first_capture = c0_execution.get("first_capture")
    if not isinstance(first_capture, Mapping):
        fail("C0 first capture evidence is absent")
    unsigned = {
        "world_size": 4, "passed": True, "before_any_real_forward": True,
        "actual_model_text_input_tensor_hashes": True,
        "input_bundle_digest": input_bundle_digest,
        "input_rank_digests": list(input_rank_digests),
        "first_execution_id": first_capture.get("execution_id"),
        "first_output_sha256": first_capture.get("capture_prediction_sha256"),
        "first_output_rank_digests": first_capture.get("capture_prediction_rank_sha256"),
        "first_output_world4_consensus": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def run_main(args: argparse.Namespace) -> int:
    _validate_run_cli(args)
    source_identity = sealed_source_self_identity(args.source_dataset_root)
    orbit_identity_base = sealed_orbit_identity(args.wrong_owner_dataset_root)
    prompt_authority = load_prompt_authority()
    orbit_review = load_orbit_review_authority()
    source_video = stable_file_snapshot(args.source_video, label="source display video")
    if source_video.sha256 != SOURCE_VIDEO_SHA256:
        fail("source display video SHA differs")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy_train.validate_source_trees(
                args.bernini_root, args.veomni_root,
                expected_bernini_commit=EXPECTED_BERNINI_COMMIT,
                expected_veomni_commit=EXPECTED_VEOMNI_COMMIT,
            )
        )
        checkpoint, transformer_config = legacy_train.validate_checkpoint(args.checkpoint)
        checkpoint_audit = stageb_infer.validate_checkpoint_content(
            checkpoint, args.checkpoint_content_manifest,
            expected_manifest_sha256=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        )
    except (legacy_train.TrainingContractError, stageb_infer.StageBInferenceError) as error:
        raise StageARuntimeError(str(error)) from error
    if legacy_train.CHECKPOINT_TREE_SHA256 != EXPECTED_CHECKPOINT_TREE_SHA256:
        fail("checkpoint tree pin differs")
    legacy_infer.validate_inference_source_files(bernini_root)
    legacy_train.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode
    from transformers import AutoTokenizer
    import source_self_native_ref_contrastive_v3 as native
    import train_source_self_identity_orbit_v4 as orbit_train

    contract = runtime.distributed_contract(topology=runtime.WORLD4_DP1_SP4)
    device = runtime.initialise_distributed(contract)
    parallel = runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    physical = stageb_infer.validate_two_node_two_rank_placement(
        contract, parallel.world_group
    )
    topology_admission = _collective_topology_admission(
        Path(args.topology_dir), group=parallel.world_group
    )
    output, stage = runtime.prepare_output_transaction(
        args.output_dir, contract.rank, parallel.world_group
    )
    reserved_stage_identity = _rank_zero_call(
        lambda: list(_directory_reservation_identity(stage)),
        rank=contract.rank, group=parallel.world_group,
        label="capture Stage-A staging reservation identity",
    )
    c0_plan_sha = _prepare_phase_directory(
        stage=stage, phase="c0", rank=contract.rank, group=parallel.world_group
    )
    stageb_infer.collective_rank_call(
        lambda: (
            True if transformer_config.get("num_attention_heads") == 12
            else fail("Bernini 1.3B attention geometry differs")
        ),
        group=parallel.world_group, label="local transformer geometry admission",
    )

    dataset, vae_runtime_validation, orbit_row, orbit, _ = (
        stageb_infer.collective_rank_call(
            lambda: _load_local_orbit_inputs(
                orbit_train=orbit_train, dataset_root=args.wrong_owner_dataset_root,
                checkpoint=checkpoint, device=device,
            ),
            group=parallel.world_group, label="local orbit dataset/decode preparation",
        )
    )
    orbit_broadcast = orbit_train.broadcast_orbit_within_sp(
        orbit, parallel=parallel
    )
    source_member, wrong_member, orbit_broadcast = stageb_infer.collective_rank_call(
        lambda: _post_broadcast_orbit_admission(
            dataset=dataset, orbit=orbit, broadcast=orbit_broadcast
        ),
        group=parallel.world_group, label="local post-broadcast orbit admission",
    )
    _collective_digest(
        str(orbit_broadcast["tensor_digest"]), group=parallel.world_group,
        label="post-broadcast orbit tensors",
    )
    epsilon = stageb_infer.collective_rank_call(
        lambda: _make_stage_a_epsilon(torch=torch, device=device),
        group=parallel.world_group, label="local Stage-A epsilon construction",
    )
    epsilon_sha = stageb_infer.collective_rank_call(
        lambda: runtime.tensor_sha256(epsilon), group=parallel.world_group,
        label="local Stage-A epsilon content identity",
    )
    _collective_digest(
        epsilon_sha, group=parallel.world_group, label="single Stage-A epsilon"
    )

    renderer, transformer, tokenizer = stageb_infer.collective_rank_call(
        lambda: _load_local_renderer_tokenizer(
            BerniniRendererConfig=BerniniRendererConfig,
            BerniniRendererModel=BerniniRendererModel, AutoTokenizer=AutoTokenizer,
            bernini_root=bernini_root, checkpoint=checkpoint,
            transformer_config=transformer_config, device=device,
        ),
        group=parallel.world_group, label="local renderer/tokenizer load",
    )
    model_pre = stageb_infer.collective_rank_call(
        lambda: _module_state_certificate(transformer),
        group=parallel.world_group, label="local loaded transformer certificate",
    )
    _collective_digest(
        str(model_pre["sha256"]), group=parallel.world_group,
        label="loaded transformer state before text/pack",
    )
    text_runtime, prompt_embeddings = _encode_all_prompts_once(
        renderer=renderer, tokenizer=tokenizer, prompt_authority=prompt_authority,
        device=device, world_group=parallel.world_group,
    )
    stageb_infer.collective_rank_call(
        lambda: (
            setattr(renderer, "t5_text_encoder", None), torch.cuda.empty_cache(), True
        )[-1],
        group=parallel.world_group, label="local text encoder release",
    )
    del tokenizer

    schedules = (
        swap.policy.REGISTERED_SCHEDULE_INDICES
        if args.profile == FORMAL_PROFILE else (swap.C0_SCHEDULE_INDEX,)
    )
    packs: dict[int, Mapping[str, ActualOwnerPack]] = {}
    pair_receipts: dict[int, Mapping[str, Any]] = {}
    x_s_by_schedule: dict[int, Any] = {}
    for schedule in schedules:
        correct, wrong, pair_receipt, x_s = stageb_infer.collective_rank_call(
            lambda schedule= schedule: build_actual_owner_pair(
                transformer=transformer, native=native, row=orbit_row,
                source_member=source_member, wrong_member=wrong_member,
                epsilon=epsilon, schedule_index=schedule, sp_rank=contract.sp_rank,
            ),
            group=parallel.world_group, label=f"s{schedule} local owner pack",
        )
        _collective_digest(
            str(pair_receipt["actual_tensor_bundle_digest"]),
            group=parallel.world_group,
            label=f"s{schedule} actual owner tensor bundle",
        )
        _collective_digest(
            str(pair_receipt["digest"]), group=parallel.world_group,
            label=f"s{schedule} owner pair receipt",
        )
        packs[schedule] = {"correct_owner": correct, "wrong_owner": wrong}
        pair_receipts[schedule] = pair_receipt
        x_s_by_schedule[schedule] = x_s
    cross_schedule = stageb_infer.collective_rank_call(
        lambda: (
            build_cross_schedule_owner_closure(
                pair_receipts, c0_pair=pair_receipts[swap.C0_SCHEDULE_INDEX]
            ) if args.profile == FORMAL_PROFILE else None
        ),
        group=parallel.world_group, label="local cross-schedule owner closure",
    )
    input_pre_c0 = stageb_infer.collective_rank_call(
        lambda: snapshot_actual_model_inputs(
            packs=packs, x_s_by_schedule=x_s_by_schedule, epsilon=epsilon,
            prompt_embeddings=prompt_embeddings,
            text_bindings=text_runtime["branches"], schedules=schedules,
        ),
        group=parallel.world_group, label="local pre-C0 actual input snapshot",
    )
    _collective_digest(
        str(input_pre_c0["digest"]), group=parallel.world_group,
        label="pre-C0 actual model inputs",
    )
    model_pre_c0 = stageb_infer.collective_rank_call(
        lambda: _module_state_certificate(transformer),
        group=parallel.world_group, label="local post-pack transformer certificate",
    )
    _collective_digest(
        str(model_pre_c0["sha256"]), group=parallel.world_group,
        label="post-pack pre-C0 transformer state",
    )
    stageb_infer.collective_rank_call(
        lambda: _assert_same_module_certificate(
            model_pre, model_pre_c0, label="text encoding and native pack preparation"
        ),
        group=parallel.world_group, label="local pre-C0 transformer equality",
    )
    input_bundle = stageb_infer.collective_rank_call(
        lambda: {
            "model_sha256": model_pre["sha256"],
            "text_runtime_digest": text_runtime["digest"],
            "epsilon_sha256": epsilon_sha,
            "orbit_row_digest": orbit_row.row_digest,
            "owner_pair_digests": [pair_receipts[index]["digest"] for index in schedules],
            "c0_plan_digest": swap.build_plan("c0-smoke")["plan_digest"],
            "full_plan_digest": (
                swap.build_plan("full-grid")["plan_digest"]
                if args.profile == FORMAL_PROFILE else None
            ),
            "actual_input_pre_c0_snapshot_digest": input_pre_c0["digest"],
        },
        group=parallel.world_group, label="local first-forward input bundle",
    )
    input_bundle_digest = stageb_infer.collective_rank_call(
        lambda: object_sha256(input_bundle), group=parallel.world_group,
        label="local first-forward input bundle digest",
    )
    input_rank_digests = _collective_digest(
        input_bundle_digest, group=parallel.world_group,
        label="first-forward model/text/input bundle",
    )

    handle = stageb_infer.collective_rank_call(
        lambda: swap.install_prompt_swap_processors(transformer),
        group=parallel.world_group, label="local prompt-swap processor installation",
    )
    try:
        with torch.inference_mode():
            c0_execution = _execute_c0(
                diffusion=renderer.diff_dec, native=native, orbit_train=orbit_train,
                handle=handle, packs=packs, pair_receipts=pair_receipts,
                x_s_by_schedule=x_s_by_schedule, prompt_embeddings=prompt_embeddings,
                text_bindings=text_runtime["branches"], world_group=parallel.world_group,
            )
        input_post_c0 = stageb_infer.collective_rank_call(
            lambda: snapshot_actual_model_inputs(
                packs=packs, x_s_by_schedule=x_s_by_schedule, epsilon=epsilon,
                prompt_embeddings=prompt_embeddings,
                text_bindings=text_runtime["branches"], schedules=schedules,
            ),
            group=parallel.world_group, label="local post-C0 actual input snapshot",
        )
        _collective_digest(
            str(input_post_c0["digest"]), group=parallel.world_group,
            label="post-C0 actual model inputs",
        )
        stageb_infer.collective_rank_call(
            lambda: (
                True if canonical_json_bytes(input_pre_c0)
                == canonical_json_bytes(input_post_c0)
                else fail("actual model inputs changed during C0")
            ),
            group=parallel.world_group,
            label="local post-C0 actual input equality",
        )
        first_forward = stageb_infer.collective_rank_call(
            lambda: _build_first_forward_record(
                c0_execution=c0_execution,
                input_bundle_digest=input_bundle_digest,
                input_rank_digests=input_rank_digests,
            ),
            group=parallel.world_group, label="local first-forward evidence",
        )
        model_post_c0 = stageb_infer.collective_rank_call(
            lambda: _module_state_certificate(transformer),
            group=parallel.world_group, label="local post-C0 transformer certificate",
        )
        _collective_digest(
            str(model_post_c0["sha256"]), group=parallel.world_group,
            label="post-C0 transformer state",
        )
        stageb_infer.collective_rank_call(
            lambda: _assert_same_module_certificate(
                model_pre, model_post_c0, label="C0"
            ),
            group=parallel.world_group, label="local post-C0 transformer equality",
        )
        vae = _load_rank_zero_vae(
            checkpoint=checkpoint, device=device, rank=contract.rank,
            group=parallel.world_group,
        )
        c0_outputs = _decode_phase_rank_zero(
            stage=stage, phase="c0", vae=vae,
            latent_rows=c0_execution["latent_rows"], device=device,
            rank=contract.rank, group=parallel.world_group,
            vae_decode=_vae_decode, save_output=save_output,
        )
        no_update_c0 = {
            "gradient_enabled": False, "optimizer_present": False,
            "scheduler_present": False, "scheduler_steps": 0,
            "parameter_gradients_present": False, "parameter_updates": 0,
        }
        c0_gate = stageb_infer.collective_rank_call(
            lambda: build_c0_engineering_gate(
                noop_parity=c0_execution["noop_parity"],
                processor_audits=c0_execution["processor_audits"],
                cache_audits=c0_execution["cache_audits"], output_records=c0_outputs,
                first_forward_consensus=first_forward,
                model_integrity={
                    "pre_c0_sha256": model_pre["sha256"],
                    "post_c0_sha256": model_post_c0["sha256"], "bytes_unchanged": True,
                }, no_update=no_update_c0,
            ),
            group=parallel.world_group, label="local C0 engineering gate",
        )
        _collective_digest(
            str(c0_gate["digest"]), group=parallel.world_group,
            label="C0 engineering gate",
        )
        stageb_infer.collective_rank_call(
            lambda: validate_c0_before_full(
                gate=c0_gate,
                pair_receipt=pair_receipts[swap.C0_SCHEDULE_INDEX],
                text_runtime=text_runtime, output_rows=c0_outputs,
                expected_input_bundle_digest=input_bundle_digest,
            ),
            group=parallel.world_group,
            label="deep C0 engineering admission before full grid",
        )

        full_execution: Optional[Mapping[str, Any]] = None
        full_outputs: list[Mapping[str, Any]] = []
        full_record: Optional[Mapping[str, Any]] = None
        full_plan_sha: Optional[str] = None
        input_post_full: Optional[Mapping[str, Any]] = None
        if args.profile == FORMAL_PROFILE:
            full_plan_sha = _prepare_phase_directory(
                stage=stage, phase="full", rank=contract.rank,
                group=parallel.world_group,
            )
            _move_rank_zero_module(
                vae, "cpu", rank=contract.rank, group=parallel.world_group,
                label="VAE to CPU before full grid",
            )
            stageb_infer.collective_rank_call(
                torch.cuda.empty_cache, group=parallel.world_group,
                label="local CUDA cache release before full grid",
            )
            with torch.inference_mode():
                full_execution = _execute_full(
                    diffusion=renderer.diff_dec, native=native, orbit_train=orbit_train,
                    handle=handle, packs=packs, pair_receipts=pair_receipts,
                    x_s_by_schedule=x_s_by_schedule, prompt_embeddings=prompt_embeddings,
                    text_bindings=text_runtime["branches"], world_group=parallel.world_group,
                )
            input_post_full = stageb_infer.collective_rank_call(
                lambda: snapshot_actual_model_inputs(
                    packs=packs, x_s_by_schedule=x_s_by_schedule, epsilon=epsilon,
                    prompt_embeddings=prompt_embeddings,
                    text_bindings=text_runtime["branches"], schedules=schedules,
                ),
                group=parallel.world_group,
                label="local post-full actual input snapshot",
            )
            _collective_digest(
                str(input_post_full["digest"]), group=parallel.world_group,
                label="post-full actual model inputs",
            )
            stageb_infer.collective_rank_call(
                lambda: (
                    True if canonical_json_bytes(input_pre_c0)
                    == canonical_json_bytes(input_post_full)
                    else fail("actual model inputs changed during full grid")
                ),
                group=parallel.world_group,
                label="local post-full actual input equality",
            )
            _move_rank_zero_module(
                vae, device, rank=contract.rank, group=parallel.world_group,
                label="VAE to device for full decode",
            )
            full_outputs = _decode_phase_rank_zero(
                stage=stage, phase="full", vae=vae,
                latent_rows=full_execution["latent_rows"], device=device,
                rank=contract.rank, group=parallel.world_group,
                vae_decode=_vae_decode, save_output=save_output,
            )
            def build_full_record() -> Mapping[str, Any]:
                if not isinstance(full_execution, Mapping):
                    fail("full execution result is absent")
                full_unsigned = {
                    "started_after_c0_pass": True, "same_model_load": True,
                    "fixed_plan_no_adaptation": True, "decoded_output_count": 112,
                    "completed": True,
                    "processor_audits": list(full_execution["processor_audits"]),
                    "cache_audits": list(full_execution["cache_audits"]),
                    "plan_digest": swap.build_plan("full-grid")["plan_digest"],
                }
                return {**full_unsigned, "digest": object_sha256(full_unsigned)}

            full_record = stageb_infer.collective_rank_call(
                build_full_record, group=parallel.world_group,
                label="local full-grid terminal record",
            )
        model_post = stageb_infer.collective_rank_call(
            lambda: _module_state_certificate(transformer),
            group=parallel.world_group, label="local terminal transformer certificate",
        )
        _collective_digest(
            str(model_post["sha256"]), group=parallel.world_group,
            label="terminal transformer state",
        )
        stageb_infer.collective_rank_call(
            lambda: _assert_same_module_certificate(
                model_pre, model_post, label="Stage-A"
            ),
            group=parallel.world_group, label="local terminal transformer equality",
        )
    finally:
        stageb_infer.collective_rank_call(
            handle.restore, group=parallel.world_group,
            label="local prompt-swap processor restoration",
        )

    patch_receipt = stageb_infer.collective_rank_call(
        handle.receipt, group=parallel.world_group,
        label="local prompt-swap terminal receipt",
    )
    input_terminal = stageb_infer.collective_rank_call(
        lambda: snapshot_actual_model_inputs(
            packs=packs, x_s_by_schedule=x_s_by_schedule, epsilon=epsilon,
            prompt_embeddings=prompt_embeddings,
            text_bindings=text_runtime["branches"], schedules=schedules,
        ),
        group=parallel.world_group, label="local terminal actual input snapshot",
    )
    _collective_digest(
        str(input_terminal["digest"]), group=parallel.world_group,
        label="terminal actual model inputs",
    )
    input_invariants = stageb_infer.collective_rank_call(
        lambda: build_input_invariant_receipt(
            profile=args.profile, pre_c0=input_pre_c0, post_c0=input_post_c0,
            post_full=input_post_full, terminal=input_terminal,
        ),
        group=parallel.world_group, label="local actual input invariant receipt",
    )
    _collective_digest(
        str(input_invariants["digest"]), group=parallel.world_group,
        label="actual model input invariant receipt",
    )
    terminal_authority = stageb_infer.collective_rank_call(
        lambda: _terminal_authority_audit(
            source_identity=source_identity, orbit_identity=orbit_identity_base,
            source_video=source_video, dataset=dataset,
            bernini_root=bernini_root, veomni_root=veomni_root,
            checkpoint=checkpoint,
            checkpoint_manifest=Path(args.checkpoint_content_manifest),
        ),
        group=parallel.world_group, label="terminal Stage-A live authority audit",
    )
    _collective_digest(
        str(terminal_authority["digest"]), group=parallel.world_group,
        label="terminal authority audit",
    )
    stageb_infer.collective_rank_call(
        lambda: renderer.to("cpu"), group=parallel.world_group,
        label="local renderer terminal CPU move",
    )
    _move_rank_zero_module(
        vae, "cpu", rank=contract.rank, group=parallel.world_group,
        label="terminal VAE CPU move",
    )
    stageb_infer.collective_rank_call(
        torch.cuda.empty_cache, group=parallel.world_group,
        label="local terminal CUDA cache release",
    )
    outputs = stageb_infer.collective_rank_call(
        lambda: list(c0_outputs) + list(full_outputs),
        group=parallel.world_group, label="local decoded output closure",
    )
    artifact_hashes = _rank_zero_call(
        lambda: {
            name: file_sha256(stage / name) for name in expected_artifact_names(args.profile)
        }, rank=contract.rank, group=parallel.world_group, label="hash output artifacts",
    )
    def build_terminal_receipt() -> Mapping[str, Any]:
        if (
            artifact_hashes.get("c0-plan.json") != c0_plan_sha
            or (
                args.profile == FORMAL_PROFILE
                and artifact_hashes.get("full-plan.json") != full_plan_sha
            )
        ):
            fail("written fixed plan SHA differs from prepared plan")
        distributed_unsigned = {
            **{key: value for key, value in physical.items() if key != "digest"},
            "topology_admission": dict(topology_admission),
            "topology_admitted_collectively_before_output": True,
        }
        distributed = {
            **distributed_unsigned, "digest": object_sha256(distributed_unsigned)
        }
        orbit_identity = {
            **dict(orbit_identity_base),
            "vae_runtime_validation": dict(vae_runtime_validation),
            "orbit_tensor_broadcast": dict(orbit_broadcast),
        }
        model_integrity = {
            "certificate_schema": model_pre["certificate_schema"],
            "pre_sha256": model_pre["sha256"],
            "post_c0_sha256": model_post_c0["sha256"],
            "post_sha256": model_post["sha256"],
            "parameter_tensors": model_pre["parameter_tensors"],
            "buffer_tensors": model_pre["buffer_tensors"], "bytes_unchanged": True,
            "all_parameters_frozen": True, "all_parameter_gradients_absent": True,
        }
        execution = {
            "distributed_invocation_count": 1, "model_load_count": 1,
            "vae_load_count": 1, "c0_model_forward_count": 10,
            "c0_capture_forward_count": 2, "c0_internal_parity_forward_count": 2,
            "c0_decoded_output_count": 6,
            "full_model_forward_count": 136 if args.profile == FORMAL_PROFILE else 0,
            "full_capture_forward_count": 24 if args.profile == FORMAL_PROFILE else 0,
            "full_decoded_output_count": 112 if args.profile == FORMAL_PROFILE else 0,
            "total_model_forward_count": 146 if args.profile == FORMAL_PROFILE else 10,
            "scheduler_instance_count": 0, "scheduler_step_count": 0,
            "optimizer_instance_count": 0,
        }
        unsigned_receipt = {
            "schema_version": RECEIPT_SCHEMA, "complete": True,
            "profile": args.profile, "model_load_count": 1,
            "one_process_one_model_load": True,
            "same_load_c0_then_full": args.profile == FORMAL_PROFILE, "seed": SEED,
            "datasets": {
                "source_self_cross_authority": dict(source_identity),
                "orbit_model_inputs": orbit_identity,
            },
            "source_video": {
                "path": str(source_video.path), "sha256": source_video.sha256,
                "cross_authority_only": True, "model_condition_consumed": False,
            },
            "prompt_authority": dict(prompt_authority),
            "text_runtime": dict(text_runtime),
            "orbit_review_authority": dict(orbit_review),
            "model": {
                "bernini_root": str(bernini_root), "veomni_root": str(veomni_root),
                "checkpoint": str(checkpoint),
                "checkpoint_content_manifest": str(Path(args.checkpoint_content_manifest)),
                "bernini_commit": bernini_revision, "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": EXPECTED_CHECKPOINT_TREE_SHA256,
                "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
                "checkpoint_verified_file_count": 23,
                "checkpoint_content_audit": dict(checkpoint_audit),
                "renderer": "Bernini-R-1.3B-transformer_1", "transformer_count": 1,
                "transformer_block_count": swap.TOTAL_BLOCKS,
            },
            "method_source": {
                "revision": args.method_source_revision,
                "archive_sha256": args.method_source_archive_sha256,
            },
            "execution": execution, "distributed": distributed,
            "first_forward_consensus": first_forward,
            "input_invariants": dict(input_invariants),
            "owner_pairs": {
                "c0": [pair_receipts[swap.C0_SCHEDULE_INDEX]],
                "full": (
                    [pair_receipts[index] for index in swap.policy.REGISTERED_SCHEDULE_INDICES]
                    if args.profile == FORMAL_PROFILE else None
                ),
                "cross_schedule_closure": cross_schedule,
            },
            "c0": c0_gate, "full": full_record, "outputs": outputs,
            "artifacts": artifact_hashes, "model_integrity": model_integrity,
            "no_update": {
                **no_update_c0, "torch_inference_mode_all_forwards": True,
            },
            "processor_patch": dict(patch_receipt),
            "calibration": _calibration_receipt(),
            "terminal_authority_audit": dict(terminal_authority),
            "interpretation": _interpretation_receipt(),
        }
        return finalize_receipt(unsigned_receipt)

    receipt = stageb_infer.collective_rank_call(
        build_terminal_receipt, group=parallel.world_group,
        label="local terminal receipt construction",
    )
    stageb_infer.collective_rank_call(
        lambda: validate_receipt(receipt, expected_profile=args.profile),
        group=parallel.world_group, label="local terminal receipt verification",
    )
    _collective_digest(
        str(receipt["receipt_digest"]), group=parallel.world_group,
        label="terminal Stage-A receipt",
    )
    _publish_bundle(
        output=output, stage=stage, receipt=receipt,
        reserved_stage_identity=reserved_stage_identity,
        rank=contract.rank, group=parallel.world_group,
    )
    dist.barrier(group=parallel.world_group)
    if contract.rank == 0:
        print(json.dumps({
            "output": str(output), "profile": args.profile,
            "receipt_digest": receipt["receipt_digest"], "decoded_outputs": len(outputs),
        }, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "verify":
        verify_bundle(args.output_dir, expected_profile=args.profile, verify_media=True)
        return 0
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
