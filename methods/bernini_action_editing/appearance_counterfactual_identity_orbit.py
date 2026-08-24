#!/usr/bin/env python3
"""Closed v3/RV2V-4 input contract for appearance identity orbits.

An orbit row contains one raw source and two appearance variants produced from
that exact source by the frozen Bernini native identity-generation canary.
Each variant explicitly binds its own native arm (``r2v`` or ``rv2v``), so a
scientifically stronger RV2V+RV2V orbit is representable without pretending
that one member came from R2V.  The variants may come from separate prompts,
seeds and native receipts, but their prompt hashes and MP4 content must differ.

This module validates provenance only; it neither trains a model nor claims
that the generated videos preserve motion.  That semantic claim must come
from an independently produced, content-addressed qualification seal.

The qualification seal is optional so failed/unreviewed engineering assets
can still be materialized for inspection.  Its absence always yields
``scientific_use_authorized == False``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Optional, Sequence

import source_self_runtime as runtime


SPEC_SCHEMA = "bernini-appearance-counterfactual-identity-orbit-spec-v3"
QUALIFICATION_SCHEMA = (
    "bernini-appearance-counterfactual-identity-orbit-external-qualification-v2"
)
NATIVE_RECEIPT_SCHEMA = "bernini-native-identity-generation-canary-v1"
NATIVE_METHOD = "frozen-bernini-native-identity-generation-canary"
FRAME_COUNT = 81
FPS = 25.0
NUM_INFERENCE_STEPS = 40
MEMBER_NAMES = ("source", "variant_a", "variant_b")
GENERATED_MEMBER_NAMES = ("variant_a", "variant_b")
ALLOWED_NATIVE_ARMS = ("r2v", "rv2v")
V4_MEMBER_ALIASES = {"V0": "source", "V1": "variant_a", "V2": "variant_b"}
REFERENCE_INDICES = (0, 27, 53, 80)
REFERENCE_COUNT = 4
FULL_VIDEO_ENCODE_CALLS_PER_ROW = 3
INDEPENDENT_RGB_REFERENCE_ENCODE_CALLS_PER_ROW = 12
INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW = 15
PINNED_REFERENCE_ENCODING_CONTRACT_DIGEST = (
    "181e93b1620cafce7de3806b334b6bfdd8e24aa633119cbd6506f3761175a269"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SPEC_KEYS = frozenset(
    {"schema_version", "reference_encoding_contract", "rows", "spec_digest"}
)
_REFERENCE_ENCODING_CONTRACT_KEYS = frozenset(
    {
        "frame_count",
        "reference_rgb_indices",
        "reference_count",
        "full_video_encode_calls_per_row",
        "independent_rgb_reference_encode_calls_per_row",
        "independent_vae_encode_calls_per_row",
        "references_from_full_video_posterior_slice",
        "native_deployment_visual_conditioning",
        "digest",
    }
)
_ROW_KEYS = frozenset(
    {"iid", "source", "variant_a", "variant_b", "qualification"}
)
_SOURCE_KEYS = frozenset({"video_path", "video_sha256"})
_QUALIFIED_GENERATED_MEMBER_KEYS = frozenset(
    {"video_path", "video_sha256", "native_arm"}
)
_GENERATED_KEYS = frozenset(
    {
        "video_path",
        "video_sha256",
        "native_arm",
        "native_receipt_path",
        "native_receipt_file_sha256",
        "native_receipt_digest",
    }
)
_QUALIFICATION_BINDING_KEYS = frozenset({"path", "file_sha256", "digest"})

_NATIVE_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "method_source_revision",
        "method_source_archive_sha256",
        "bernini_commit",
        "veomni_commit",
        "bernini_inference_files",
        "checkpoint",
        "arms",
        "input",
        "preprocessing",
        "prompt_contract",
        "conditioning",
        "sampling",
        "latent_geometry",
        "condition_identities",
        "source_condition_artifact",
        "initial_noise_artifacts",
        "generated_identities",
        "outputs",
        "freeze_certificate",
        "runtime_versions",
        "interpretation",
        "experimental_canary",
        "production_claim_forbidden",
        "scientific_claim_authorized",
        "receipt_digest",
    }
)
_NATIVE_INPUT_KEYS = frozenset(
    {
        "source_video_path",
        "source_video_sha256",
        "action_prompt_utf8_sha256",
        "action_prompt_utf8_bytes",
        "accepted_external_conditions",
        "target_video",
        "external_reference_image_or_video",
        "external_mask_flow_pose_track_trajectory",
        "external_first_frame_anchor",
    }
)
_NATIVE_OUTPUT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "frame_count",
        "fps",
        "height",
        "width",
        "normalized_clean_latent",
    }
)
_NATIVE_SAMPLING_KEYS = frozenset(
    {
        "num_frames",
        "num_inference_steps",
        "guidance_mode",
        "omega_vid",
        "omega_img",
        "omega_txt",
        "omega_scale",
        "flow_shift",
        "seed",
        "eta",
        "norm_threshold",
        "momentum",
        "target_initialization",
        "target_mixed_with_source_latent",
        "custom_sampler_or_scheduler",
        "same_seed_and_target_shape_across_arms",
        "single_expert",
        "ulysses_size",
    }
)
_NATIVE_CONDITIONING_KEYS = frozenset(
    {
        "full_source_video_count",
        "source_derived_reference_count",
        "source_frame_indices",
        "reference_encoding",
        "reference_from_temporal_video_latent_slice",
        "source_ids",
    }
)
_NATIVE_SOURCE_ID_KEYS = frozenset(
    {
        "target_source_id",
        "video_source_ids",
        "reference_source_ids",
        "conditioning_source_count",
        "max_conditioning_source_id",
        "within_pretrained_source_ids_1_through_5",
        "source_id_interpolation_required",
    }
)
_NATIVE_CHECKPOINT_KEYS = frozenset({"path", "tree_sha256", "content"})
_NATIVE_CHECKPOINT_CONTENT_KEYS = frozenset(
    {
        "manifest_path",
        "manifest_sha256_computed",
        "manifest_sha256_expected",
        "verified_entries_digest",
        "verified_file_count",
        "every_file_sha256_verified",
    }
)
_QUALIFICATION_KEYS = frozenset(
    {
        "schema_version",
        "iid",
        "members",
        "evaluation_protocol",
        "qualification_gates",
        "downstream_training_results_seen",
        "qualification_passed",
        "receipt_digest",
    }
)
_QUALIFICATION_PROTOCOL_KEYS = frozenset(
    {
        "qualifier_id",
        "protocol_sha256",
        "external_to_materializer",
        "blind_to_downstream_training_results",
        "full_video_reviewed",
        "all_81_frames_reviewed",
    }
)
_MEMBER_GATE_KEYS = frozenset(
    {
        "appearance_identity_changed_from_source",
        "motion_phase_and_order_preserved",
        "camera_path_preserved",
        "background_scene_preserved",
        "object_correspondence_preserved",
        "temporal_quality_passed",
        "spatial_quality_passed",
        "no_extra_actor_or_object",
    }
)
_CROSS_MEMBER_GATE_KEYS = frozenset(
    {
        "variant_a_and_variant_b_semantic_identities_distinct",
        "same_motion_across_all_members",
        "same_camera_across_all_members",
        "same_scene_across_all_members",
    }
)


class AppearanceOrbitError(RuntimeError):
    """Raised before ambiguous or mutable orbit evidence is accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return runtime.canonical_json_bytes(value)
    except runtime.SourceSelfRuntimeError as error:
        raise AppearanceOrbitError(str(error)) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def reference_encoding_contract() -> Mapping[str, Any]:
    """Return the exact RV2V-4 offline-posterior contract bound by every spec."""

    if (
        len(REFERENCE_INDICES) != REFERENCE_COUNT
        or REFERENCE_INDICES != (0, 27, 53, 80)
        or FULL_VIDEO_ENCODE_CALLS_PER_ROW != len(MEMBER_NAMES)
        or INDEPENDENT_RGB_REFERENCE_ENCODE_CALLS_PER_ROW
        != len(MEMBER_NAMES) * REFERENCE_COUNT
        or INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW
        != FULL_VIDEO_ENCODE_CALLS_PER_ROW
        + INDEPENDENT_RGB_REFERENCE_ENCODE_CALLS_PER_ROW
    ):
        raise AppearanceOrbitError("RV2V-4 reference encoding registry differs")
    value = {
        "frame_count": FRAME_COUNT,
        "reference_rgb_indices": list(REFERENCE_INDICES),
        "reference_count": REFERENCE_COUNT,
        "full_video_encode_calls_per_row": FULL_VIDEO_ENCODE_CALLS_PER_ROW,
        "independent_rgb_reference_encode_calls_per_row": (
            INDEPENDENT_RGB_REFERENCE_ENCODE_CALLS_PER_ROW
        ),
        "independent_vae_encode_calls_per_row": INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW,
        "references_from_full_video_posterior_slice": False,
        "native_deployment_visual_conditioning": "one_video_plus_four_rgb_refs",
    }
    digest = object_sha256(value)
    if digest != PINNED_REFERENCE_ENCODING_CONTRACT_DIGEST:
        raise AppearanceOrbitError("pinned RV2V-4 reference encoding digest differs")
    return {**value, "digest": digest}


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise AppearanceOrbitError(f"{label} must be a lowercase SHA-256")
    return value


def _require_closed_mapping(
    value: Any, expected: frozenset[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise AppearanceOrbitError(
            f"{label} field closure differs: expected={sorted(expected)} actual={actual}"
        )
    return value


def _require_exact_bool(value: Any, expected: bool, *, label: str) -> None:
    if value is not expected:
        raise AppearanceOrbitError(f"{label} must be {expected}")


def _require_exact_number(value: Any, expected: float, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AppearanceOrbitError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or abs(number - expected) > 1.0e-3:
        raise AppearanceOrbitError(f"{label} must be {expected}")


def _canonical_plain_file(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise AppearanceOrbitError(f"{label} path must be non-empty text")
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise AppearanceOrbitError(f"{label} path must be absolute")
    try:
        if requested.is_symlink():
            raise AppearanceOrbitError(f"{label} must not be a symlink")
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise AppearanceOrbitError(f"{label} is unavailable: {error}") from error
    if requested != resolved or resolved.is_symlink() or not stat.S_ISREG(mode):
        raise AppearanceOrbitError(f"{label} must be a canonical plain file")
    return resolved


def _checkpoint_content_identity(
    receipt: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    """Return path-independent checkpoint content identity from a native receipt.

    The same immutable manifest may be copied into different experiment
    directories.  Its absolute ``manifest_path`` is provenance for that one
    receipt, not part of model identity.  Comparing the entire checkpoint
    mapping would therefore reject byte-identical model trees solely because
    their audit manifests live at different paths.
    """

    checkpoint = _require_closed_mapping(
        receipt.get("checkpoint"), _NATIVE_CHECKPOINT_KEYS, label=f"{label} checkpoint"
    )
    content = _require_closed_mapping(
        checkpoint.get("content"),
        _NATIVE_CHECKPOINT_CONTENT_KEYS,
        label=f"{label} checkpoint content",
    )
    for name in ("path",):
        value = checkpoint.get(name)
        if type(value) is not str or not value.startswith("/") or "\x00" in value:
            raise AppearanceOrbitError(f"{label} checkpoint.{name} is invalid")
    manifest_path = content.get("manifest_path")
    if (
        type(manifest_path) is not str
        or not manifest_path.startswith("/")
        or "\x00" in manifest_path
    ):
        raise AppearanceOrbitError(f"{label} checkpoint manifest_path is invalid")
    tree_sha = _require_sha256(
        checkpoint.get("tree_sha256"), label=f"{label} checkpoint tree SHA"
    )
    expected = _require_sha256(
        content.get("manifest_sha256_expected"),
        label=f"{label} expected checkpoint manifest SHA",
    )
    computed = _require_sha256(
        content.get("manifest_sha256_computed"),
        label=f"{label} computed checkpoint manifest SHA",
    )
    entries = _require_sha256(
        content.get("verified_entries_digest"),
        label=f"{label} verified checkpoint entries digest",
    )
    count = content.get("verified_file_count")
    if (
        expected != computed
        or content.get("every_file_sha256_verified") is not True
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise AppearanceOrbitError(f"{label} checkpoint content is not fully verified")
    return {
        "tree_sha256": tree_sha,
        "manifest_sha256": expected,
        "verified_entries_digest": entries,
        "verified_file_count": count,
        "every_file_sha256_verified": True,
    }


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "sha256": self.sha256,
        }


def _snapshot(path: Path) -> FileSnapshot:
    try:
        before = path.stat()
        digest = runtime.file_sha256(path)
        after = path.stat()
    except (OSError, runtime.SourceSelfRuntimeError) as error:
        raise AppearanceOrbitError(f"cannot snapshot {path}: {error}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or not stat.S_ISREG(after.st_mode):
        raise AppearanceOrbitError(f"file changed while snapshotting: {path}")
    return FileSnapshot(
        str(path),
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_mode),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
        digest,
    )


class FileMutationAudit:
    """Hash/stat every input before use and again after all VAE encodes."""

    def __init__(self) -> None:
        self._pre: dict[Path, FileSnapshot] = {}
        self._roles: dict[Path, set[str]] = {}
        self._finalized = False

    def register(
        self, value: Any, *, expected_sha256: str, role: str
    ) -> Path:
        if self._finalized:
            raise AppearanceOrbitError("cannot register a file after mutation audit finalization")
        path = _canonical_plain_file(value, label=role)
        expected = _require_sha256(expected_sha256, label=f"{role} SHA-256")
        if path not in self._pre:
            snapshot = _snapshot(path)
            if snapshot.sha256 != expected:
                raise AppearanceOrbitError(
                    f"{role} SHA-256 differs: expected={expected} actual={snapshot.sha256}"
                )
            self._pre[path] = snapshot
            self._roles[path] = {role}
        else:
            if self._pre[path].sha256 != expected:
                raise AppearanceOrbitError(
                    f"one path is registered with conflicting hashes: {path}"
                )
            self._roles[path].add(role)
            self.assert_current(path)
        return path

    def assert_current(self, path: Path) -> None:
        if path not in self._pre:
            raise AppearanceOrbitError(f"unregistered mutation-audit path: {path}")
        if _snapshot(path) != self._pre[path]:
            raise AppearanceOrbitError(f"input file changed after registration: {path}")

    def finalize(self) -> tuple[Mapping[str, Any], ...]:
        if self._finalized:
            raise AppearanceOrbitError("mutation audit may be finalized only once")
        records: list[Mapping[str, Any]] = []
        for path in sorted(self._pre, key=lambda item: str(item)):
            post = _snapshot(path)
            pre = self._pre[path]
            stable = post == pre
            records.append(
                {
                    "path": str(path),
                    "roles": sorted(self._roles[path]),
                    "pre": pre.as_dict(),
                    "post": post.as_dict(),
                    "pre_post_stat_and_hash_stable": stable,
                }
            )
            if not stable:
                raise AppearanceOrbitError(f"input file mutated during materialization: {path}")
        self._finalized = True
        return tuple(records)


def _reject_constant(value: str) -> None:
    raise AppearanceOrbitError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise AppearanceOrbitError(f"duplicate JSON key is forbidden: {key!r}")
        output[key] = value
    return output


def _load_registered_json(
    path: Path,
    *,
    audit: FileMutationAudit,
    label: str,
    maximum_bytes: int = 8 * 1024 * 1024,
) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AppearanceOrbitError(f"cannot read {label}: {error}") from error
    if not raw or len(raw) > maximum_bytes:
        raise AppearanceOrbitError(f"{label} byte length is invalid")
    audit.assert_current(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AppearanceOrbitError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, dict):
        raise AppearanceOrbitError(f"{label} root must be one object")
    return value


@dataclass(frozen=True)
class BoundVideo:
    path: Path
    sha256: str


@dataclass(frozen=True)
class NativeGeneratedVideo:
    name: str
    native_arm: str
    video: BoundVideo
    receipt_path: Path
    receipt_file_sha256: str
    receipt_digest: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class Qualification:
    path: Path
    file_sha256: str
    digest: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class OrbitSpecRow:
    iid: str
    source: BoundVideo
    variant_a: NativeGeneratedVideo
    variant_b: NativeGeneratedVideo
    qualification: Optional[Qualification]

    @property
    def scientific_use_authorized(self) -> bool:
        return self.qualification is not None


@dataclass(frozen=True)
class LoadedOrbitSpec:
    path: Path
    file_sha256: str
    digest: str
    reference_encoding_contract: Mapping[str, Any]
    rows: tuple[OrbitSpecRow, ...]


def build_materialization_spec(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a closed spec object; live files remain validated at load time."""

    if isinstance(rows, (str, bytes)) or not rows:
        raise AppearanceOrbitError("materialization spec requires at least one row")
    copied = [dict(row) for row in rows]
    for index, row in enumerate(copied):
        _require_closed_mapping(row, _ROW_KEYS, label=f"spec row {index}")
        iid = row.get("iid")
        if type(iid) is not str or _IID.fullmatch(iid) is None:
            raise AppearanceOrbitError(f"spec row {index} IID is invalid")
        _require_closed_mapping(row.get("source"), _SOURCE_KEYS, label=f"{iid} source")
        for member_name in GENERATED_MEMBER_NAMES:
            member = _require_closed_mapping(
                row.get(member_name),
                _GENERATED_KEYS,
                label=f"{iid} {member_name}",
            )
            if member.get("native_arm") not in ALLOWED_NATIVE_ARMS:
                raise AppearanceOrbitError(
                    f"{iid} {member_name} native_arm must be r2v or rv2v"
                )
        qualification = row.get("qualification")
        if qualification is not None:
            _require_closed_mapping(
                qualification,
                _QUALIFICATION_BINDING_KEYS,
                label=f"{iid} qualification binding",
            )
    iids = [row["iid"] for row in copied]
    if len(set(iids)) != len(iids):
        raise AppearanceOrbitError("materialization spec IIDs must be unique")
    value: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA,
        "reference_encoding_contract": dict(reference_encoding_contract()),
        "rows": copied,
    }
    value["spec_digest"] = object_sha256(value)
    return value


def _validate_native_receipt(
    *,
    iid: str,
    arm: str,
    source: BoundVideo,
    member: BoundVideo,
    receipt_path: Path,
    receipt_file_sha256: str,
    declared_receipt_digest: str,
    audit: FileMutationAudit,
) -> Mapping[str, Any]:
    receipt = _load_registered_json(
        receipt_path, audit=audit, label=f"{iid} {arm} native receipt"
    )
    _require_closed_mapping(
        receipt, _NATIVE_TOP_LEVEL_KEYS, label=f"{iid} {arm} native receipt"
    )
    if receipt.get("schema_version") != NATIVE_RECEIPT_SCHEMA:
        raise AppearanceOrbitError(f"{iid} {arm} native receipt schema differs")
    if receipt.get("method") != NATIVE_METHOD:
        raise AppearanceOrbitError(f"{iid} {arm} native method differs")
    embedded = _require_sha256(
        receipt.get("receipt_digest"), label=f"{iid} {arm} embedded receipt digest"
    )
    if embedded != declared_receipt_digest:
        raise AppearanceOrbitError(f"{iid} {arm} declared receipt digest differs")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != embedded:
        raise AppearanceOrbitError(f"{iid} {arm} embedded receipt digest differs")
    _checkpoint_content_identity(receipt, label=f"{iid} {arm} native")

    arms = receipt.get("arms")
    if (
        not isinstance(arms, list)
        or not arms
        or len(set(arms)) != len(arms)
        or any(name not in ("t2v", "r2v", "rv2v") for name in arms)
        or arm not in arms
    ):
        raise AppearanceOrbitError(f"{iid} {arm} native arm closure differs")
    arm_set = set(arms)
    for name in (
        "prompt_contract",
        "conditioning",
        "sampling",
        "initial_noise_artifacts",
        "generated_identities",
        "outputs",
    ):
        mapping = receipt.get(name)
        if not isinstance(mapping, dict) or set(mapping) != arm_set:
            raise AppearanceOrbitError(f"{iid} {arm} native {name} arm closure differs")

    input_contract = _require_closed_mapping(
        receipt.get("input"), _NATIVE_INPUT_KEYS, label=f"{iid} {arm} native input"
    )
    if (
        input_contract.get("source_video_path") != str(source.path)
        or input_contract.get("source_video_sha256") != source.sha256
    ):
        raise AppearanceOrbitError(f"{iid} {arm} receipt input source binding differs")
    if input_contract.get("accepted_external_conditions") != [
        "source_video",
        "action_prompt",
    ]:
        raise AppearanceOrbitError(f"{iid} {arm} accepted external conditions differ")
    for key in (
        "target_video",
        "external_reference_image_or_video",
        "external_mask_flow_pose_track_trajectory",
        "external_first_frame_anchor",
    ):
        _require_exact_bool(
            input_contract.get(key), False, label=f"{iid} {arm} input.{key}"
        )

    preprocessing = receipt.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise AppearanceOrbitError(f"{iid} {arm} preprocessing is absent")
    if preprocessing.get("frame_count") != FRAME_COUNT:
        raise AppearanceOrbitError(f"{iid} {arm} preprocessing is not exact81")
    _require_exact_number(
        preprocessing.get("fps"), FPS, label=f"{iid} {arm} preprocessing fps"
    )
    if (
        preprocessing.get("temporal_policy")
        != "all_integer_frames_0_through_80_no_subsampling"
        or preprocessing.get("external_shared_i0") is not False
    ):
        raise AppearanceOrbitError(f"{iid} {arm} preprocessing contract differs")

    output = _require_closed_mapping(
        receipt["outputs"].get(arm),
        _NATIVE_OUTPUT_KEYS,
        label=f"{iid} {arm} native output",
    )
    if output.get("path") != str(member.path) or output.get("sha256") != member.sha256:
        raise AppearanceOrbitError(f"{iid} {arm} output path/hash binding differs")
    if output.get("frame_count") != FRAME_COUNT:
        raise AppearanceOrbitError(f"{iid} {arm} native output is not exact81")
    _require_exact_number(output.get("fps"), FPS, label=f"{iid} {arm} output fps")
    for name in ("height", "width"):
        if type(output.get(name)) is not int or output[name] <= 0:
            raise AppearanceOrbitError(f"{iid} {arm} output {name} is invalid")

    sampling = _require_closed_mapping(
        receipt["sampling"].get(arm),
        _NATIVE_SAMPLING_KEYS,
        label=f"{iid} {arm} native sampling",
    )
    if (
        sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or sampling.get("guidance_mode")
        != {"r2v": "r2v_apg", "rv2v": "rv2v"}[arm]
        or sampling.get("target_initialization")
        != "official_gen_wanx22_fresh_gaussian"
        or sampling.get("target_mixed_with_source_latent") is not False
        or sampling.get("custom_sampler_or_scheduler") is not False
        or sampling.get("same_seed_and_target_shape_across_arms") is not True
    ):
        raise AppearanceOrbitError(f"{iid} {arm} is not frozen native exact40 sampling")
    if isinstance(sampling.get("seed"), bool) or not isinstance(sampling.get("seed"), int):
        raise AppearanceOrbitError(f"{iid} {arm} native seed is invalid")

    expected_condition = {
        "r2v": (0, 5, [0, 20, 40, 60, 80], [], [1, 2, 3, 4, 5]),
        "rv2v": (1, 4, [0, 27, 53, 80], [1], [2, 3, 4, 5]),
    }[arm]
    conditioning = _require_closed_mapping(
        receipt["conditioning"].get(arm),
        _NATIVE_CONDITIONING_KEYS,
        label=f"{iid} {arm} native conditioning",
    )
    if (
        conditioning.get("full_source_video_count") != expected_condition[0]
        or conditioning.get("source_derived_reference_count") != expected_condition[1]
        or conditioning.get("source_frame_indices") != expected_condition[2]
        or conditioning.get("reference_encoding")
        != "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]"
        or conditioning.get("reference_from_temporal_video_latent_slice") is not False
    ):
        raise AppearanceOrbitError(f"{iid} {arm} native conditioning differs")
    source_ids = _require_closed_mapping(
        conditioning.get("source_ids"),
        _NATIVE_SOURCE_ID_KEYS,
        label=f"{iid} {arm} native source IDs",
    )
    if (
        type(source_ids.get("target_source_id")) is not int
        or source_ids.get("target_source_id") != 0
        or source_ids.get("video_source_ids") != expected_condition[3]
        or source_ids.get("reference_source_ids") != expected_condition[4]
        or any(
            type(value) is not int
            for key in ("video_source_ids", "reference_source_ids")
            for value in source_ids.get(key, ())
        )
        or type(source_ids.get("conditioning_source_count")) is not int
        or source_ids.get("conditioning_source_count") != 5
        or type(source_ids.get("max_conditioning_source_id")) is not int
        or source_ids.get("max_conditioning_source_id") != 5
        or source_ids.get("within_pretrained_source_ids_1_through_5") is not True
        or source_ids.get("source_id_interpolation_required") is not False
    ):
        raise AppearanceOrbitError(f"{iid} {arm} native source-ID contract differs")

    freeze = receipt.get("freeze_certificate")
    if not isinstance(freeze, dict) or (
        freeze.get("base_frozen") is not True
        or freeze.get("trainable_parameter_tensors") != 0
        or freeze.get("trainable_parameter_elements") != 0
        or freeze.get("lora_module_count") != 0
    ):
        raise AppearanceOrbitError(f"{iid} {arm} frozen-base certificate differs")
    interpretation = receipt.get("interpretation")
    if not isinstance(interpretation, dict) or (
        interpretation.get("training_performed") is not False
        or interpretation.get("quality_claim") is not False
        or interpretation.get("best_arm_selected") is not False
    ):
        raise AppearanceOrbitError(f"{iid} {arm} native no-training certificate differs")
    if (
        receipt.get("experimental_canary") is not True
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise AppearanceOrbitError(f"{iid} {arm} native interpretation closure differs")

    audit.assert_current(source.path)
    audit.assert_current(member.path)
    audit.assert_current(receipt_path)
    if runtime.file_sha256(receipt_path) != receipt_file_sha256:
        raise AppearanceOrbitError(f"{iid} {arm} native receipt file hash changed")
    return receipt


def _validate_qualification(
    *,
    iid: str,
    binding: Mapping[str, Any],
    members: Mapping[str, BoundVideo],
    native_arms: Mapping[str, str],
    audit: FileMutationAudit,
) -> Qualification:
    path = audit.register(
        binding.get("path"),
        expected_sha256=binding.get("file_sha256"),
        role=f"{iid}:qualification_seal",
    )
    declared_digest = _require_sha256(
        binding.get("digest"), label=f"{iid} qualification digest"
    )
    receipt = _load_registered_json(
        path, audit=audit, label=f"{iid} external qualification seal"
    )
    _require_closed_mapping(receipt, _QUALIFICATION_KEYS, label=f"{iid} qualification")
    if receipt.get("schema_version") != QUALIFICATION_SCHEMA or receipt.get("iid") != iid:
        raise AppearanceOrbitError(f"{iid} qualification identity differs")
    embedded = _require_sha256(
        receipt.get("receipt_digest"), label=f"{iid} qualification embedded digest"
    )
    if embedded != declared_digest:
        raise AppearanceOrbitError(f"{iid} qualification declared digest differs")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != embedded:
        raise AppearanceOrbitError(f"{iid} qualification embedded digest differs")

    bound_members = receipt.get("members")
    if not isinstance(bound_members, dict) or set(bound_members) != set(MEMBER_NAMES):
        raise AppearanceOrbitError(f"{iid} qualification member closure differs")
    for name in MEMBER_NAMES:
        expected_keys = (
            _SOURCE_KEYS if name == "source" else _QUALIFIED_GENERATED_MEMBER_KEYS
        )
        member = _require_closed_mapping(
            bound_members.get(name),
            expected_keys,
            label=f"{iid} qualification {name}",
        )
        if (
            member.get("video_path") != str(members[name].path)
            or member.get("video_sha256") != members[name].sha256
        ):
            raise AppearanceOrbitError(f"{iid} qualification {name} binding differs")
        if name != "source" and member.get("native_arm") != native_arms.get(name):
            raise AppearanceOrbitError(
                f"{iid} qualification {name} native arm binding differs"
            )

    protocol = _require_closed_mapping(
        receipt.get("evaluation_protocol"),
        _QUALIFICATION_PROTOCOL_KEYS,
        label=f"{iid} qualification protocol",
    )
    qualifier_id = protocol.get("qualifier_id")
    if type(qualifier_id) is not str or not qualifier_id.strip() or "\x00" in qualifier_id:
        raise AppearanceOrbitError(f"{iid} qualification qualifier_id is invalid")
    _require_sha256(
        protocol.get("protocol_sha256"), label=f"{iid} qualification protocol SHA"
    )
    for key in (
        "external_to_materializer",
        "blind_to_downstream_training_results",
        "full_video_reviewed",
        "all_81_frames_reviewed",
    ):
        _require_exact_bool(protocol.get(key), True, label=f"{iid} protocol.{key}")

    gates = receipt.get("qualification_gates")
    if not isinstance(gates, dict) or set(gates) != {
        "variant_a",
        "variant_b",
        "cross_member",
    }:
        raise AppearanceOrbitError(f"{iid} qualification gate closure differs")
    for member_name in GENERATED_MEMBER_NAMES:
        member_gates = _require_closed_mapping(
            gates.get(member_name),
            _MEMBER_GATE_KEYS,
            label=f"{iid} {member_name} qualification gates",
        )
        for key in _MEMBER_GATE_KEYS:
            _require_exact_bool(
                member_gates.get(key),
                True,
                label=f"{iid} {member_name} gate.{key}",
            )
    cross_gates = _require_closed_mapping(
        gates.get("cross_member"),
        _CROSS_MEMBER_GATE_KEYS,
        label=f"{iid} cross-member qualification gates",
    )
    for key in _CROSS_MEMBER_GATE_KEYS:
        _require_exact_bool(
            cross_gates.get(key), True, label=f"{iid} cross-member gate.{key}"
        )
    _require_exact_bool(
        receipt.get("downstream_training_results_seen"),
        False,
        label=f"{iid} downstream_training_results_seen",
    )
    _require_exact_bool(
        receipt.get("qualification_passed"), True, label=f"{iid} qualification_passed"
    )
    audit.assert_current(path)
    return Qualification(
        path=path,
        file_sha256=_require_sha256(
            binding.get("file_sha256"), label=f"{iid} qualification file SHA"
        ),
        digest=declared_digest,
        receipt=receipt,
    )


def load_materialization_spec(
    path_value: str | Path,
    *,
    expected_sha256: str,
    audit: FileMutationAudit,
) -> LoadedOrbitSpec:
    """Load a source plus two explicitly arm-bound appearance variants."""

    spec_path = audit.register(
        str(path_value),
        expected_sha256=expected_sha256,
        role="materialization_spec",
    )
    spec = _load_registered_json(spec_path, audit=audit, label="materialization spec")
    _require_closed_mapping(spec, _SPEC_KEYS, label="materialization spec")
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise AppearanceOrbitError("materialization spec schema differs")
    reference_contract = _require_closed_mapping(
        spec.get("reference_encoding_contract"),
        _REFERENCE_ENCODING_CONTRACT_KEYS,
        label="materialization spec reference encoding contract",
    )
    if dict(reference_contract) != dict(reference_encoding_contract()):
        raise AppearanceOrbitError(
            "materialization spec is not exact RV2V-4 reference encoding"
        )
    declared_spec_digest = _require_sha256(
        spec.get("spec_digest"), label="materialization spec digest"
    )
    unsigned_spec = dict(spec)
    unsigned_spec.pop("spec_digest")
    if object_sha256(unsigned_spec) != declared_spec_digest:
        raise AppearanceOrbitError("materialization spec embedded digest differs")
    raw_rows = spec.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise AppearanceOrbitError("materialization spec requires at least one row")

    output_rows: list[OrbitSpecRow] = []
    seen_iids: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _require_closed_mapping(raw_row, _ROW_KEYS, label=f"spec row {index}")
        iid = row.get("iid")
        if type(iid) is not str or _IID.fullmatch(iid) is None or iid in seen_iids:
            raise AppearanceOrbitError(f"spec row {index} IID is invalid/duplicate")
        seen_iids.add(iid)

        source_value = _require_closed_mapping(
            row.get("source"), _SOURCE_KEYS, label=f"{iid} source"
        )
        source_path = audit.register(
            source_value.get("video_path"),
            expected_sha256=source_value.get("video_sha256"),
            role=f"{iid}:source_video",
        )
        if source_path.suffix.lower() != ".mp4":
            raise AppearanceOrbitError(f"{iid} source must be an MP4")
        source = BoundVideo(
            source_path,
            _require_sha256(
                source_value.get("video_sha256"), label=f"{iid} source video SHA"
            ),
        )

        generated: dict[str, NativeGeneratedVideo] = {}
        for member_name in GENERATED_MEMBER_NAMES:
            value = _require_closed_mapping(
                row.get(member_name),
                _GENERATED_KEYS,
                label=f"{iid} {member_name}",
            )
            native_arm = value.get("native_arm")
            if native_arm not in ALLOWED_NATIVE_ARMS:
                raise AppearanceOrbitError(
                    f"{iid} {member_name} native_arm must be r2v or rv2v"
                )
            video_path = audit.register(
                value.get("video_path"),
                expected_sha256=value.get("video_sha256"),
                role=f"{iid}:{member_name}_video",
            )
            if video_path.suffix.lower() != ".mp4":
                raise AppearanceOrbitError(
                    f"{iid} {member_name} output must be an MP4"
                )
            video = BoundVideo(
                video_path,
                _require_sha256(
                    value.get("video_sha256"),
                    label=f"{iid} {member_name} video SHA",
                ),
            )
            receipt_file_sha = _require_sha256(
                value.get("native_receipt_file_sha256"),
                label=f"{iid} {member_name} native receipt file SHA",
            )
            receipt_digest = _require_sha256(
                value.get("native_receipt_digest"),
                label=f"{iid} {member_name} native receipt digest",
            )
            receipt_path = audit.register(
                value.get("native_receipt_path"),
                expected_sha256=receipt_file_sha,
                role=f"{iid}:{member_name}_native_receipt",
            )
            receipt = _validate_native_receipt(
                iid=iid,
                arm=native_arm,
                source=source,
                member=video,
                receipt_path=receipt_path,
                receipt_file_sha256=receipt_file_sha,
                declared_receipt_digest=receipt_digest,
                audit=audit,
            )
            generated[member_name] = NativeGeneratedVideo(
                member_name,
                native_arm,
                video,
                receipt_path,
                receipt_file_sha,
                receipt_digest,
                receipt,
            )

        variant_a = generated["variant_a"]
        variant_b = generated["variant_b"]
        member_paths = {source.path, variant_a.video.path, variant_b.video.path}
        member_hashes = {
            source.sha256,
            variant_a.video.sha256,
            variant_b.video.sha256,
        }
        if len(member_paths) != 3:
            raise AppearanceOrbitError(f"{iid} orbit member paths must be distinct")
        if variant_a.video.sha256 == variant_b.video.sha256:
            raise AppearanceOrbitError(
                f"{iid} variant_a and variant_b MP4 content must be distinct"
            )

        variant_a_receipt = variant_a.receipt
        variant_b_receipt = variant_b.receipt
        prompt_a = variant_a_receipt["input"]["action_prompt_utf8_sha256"]
        prompt_b = variant_b_receipt["input"]["action_prompt_utf8_sha256"]
        if prompt_a == prompt_b:
            raise AppearanceOrbitError(
                f"{iid} variant_a and variant_b prompt hashes must be distinct"
            )
        if (
            _checkpoint_content_identity(
                variant_a_receipt, label=f"{iid} variant_a native"
            )
            != _checkpoint_content_identity(
                variant_b_receipt, label=f"{iid} variant_b native"
            )
            or variant_a_receipt["bernini_commit"]
            != variant_b_receipt["bernini_commit"]
            or variant_a_receipt["veomni_commit"]
            != variant_b_receipt["veomni_commit"]
        ):
            raise AppearanceOrbitError(
                f"{iid} variants' checkpoint or frozen source revision differs"
            )

        qualification_value = row.get("qualification")
        qualification: Optional[Qualification]
        if qualification_value is None:
            qualification = None
        else:
            binding = _require_closed_mapping(
                qualification_value,
                _QUALIFICATION_BINDING_KEYS,
                label=f"{iid} qualification binding",
            )
            qualification = _validate_qualification(
                iid=iid,
                binding=binding,
                members={
                    "source": source,
                    "variant_a": variant_a.video,
                    "variant_b": variant_b.video,
                },
                native_arms={
                    "variant_a": variant_a.native_arm,
                    "variant_b": variant_b.native_arm,
                },
                audit=audit,
            )
            if len(member_hashes) != 3:
                raise AppearanceOrbitError(
                    f"{iid} qualified orbit members must be content-distinct"
                )
        output_rows.append(
            OrbitSpecRow(
                iid=iid,
                source=source,
                variant_a=variant_a,
                variant_b=variant_b,
                qualification=qualification,
            )
        )
    audit.assert_current(spec_path)
    return LoadedOrbitSpec(
        path=spec_path,
        file_sha256=_require_sha256(expected_sha256, label="materialization spec SHA"),
        digest=declared_spec_digest,
        reference_encoding_contract=dict(reference_contract),
        rows=tuple(output_rows),
    )


def qualification_seal_body(
    *,
    iid: str,
    members: Mapping[str, Mapping[str, Any]],
    qualifier_id: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Return the exact positive seal for an external reviewer to publish.

    This helper does not write a file and must not be called by the
    materializer as a substitute for external full-video review.
    """

    if type(iid) is not str or _IID.fullmatch(iid) is None:
        raise AppearanceOrbitError("qualification IID is invalid")
    if set(members) != set(MEMBER_NAMES):
        raise AppearanceOrbitError(
            "qualification members must be source/variant_a/variant_b"
        )
    copied_members: dict[str, dict[str, Any]] = {}
    for name in MEMBER_NAMES:
        expected_keys = (
            _SOURCE_KEYS if name == "source" else _QUALIFIED_GENERATED_MEMBER_KEYS
        )
        value = _require_closed_mapping(
            dict(members[name]), expected_keys, label=f"qualification {name}"
        )
        copied_members[name] = {
            "video_path": str(value["video_path"]),
            "video_sha256": _require_sha256(
                value["video_sha256"], label=f"qualification {name} SHA"
            ),
        }
        if name != "source":
            native_arm = value.get("native_arm")
            if native_arm not in ALLOWED_NATIVE_ARMS:
                raise AppearanceOrbitError(
                    f"qualification {name} native_arm must be r2v or rv2v"
                )
            copied_members[name]["native_arm"] = native_arm
    if type(qualifier_id) is not str or not qualifier_id.strip() or "\x00" in qualifier_id:
        raise AppearanceOrbitError("qualification qualifier_id is invalid")
    protocol_sha256 = _require_sha256(
        protocol_sha256, label="qualification protocol SHA"
    )
    member_gates = {key: True for key in sorted(_MEMBER_GATE_KEYS)}
    cross_gates = {key: True for key in sorted(_CROSS_MEMBER_GATE_KEYS)}
    body: dict[str, Any] = {
        "schema_version": QUALIFICATION_SCHEMA,
        "iid": iid,
        "members": copied_members,
        "evaluation_protocol": {
            "qualifier_id": qualifier_id,
            "protocol_sha256": protocol_sha256,
            "external_to_materializer": True,
            "blind_to_downstream_training_results": True,
            "full_video_reviewed": True,
            "all_81_frames_reviewed": True,
        },
        "qualification_gates": {
            "variant_a": dict(member_gates),
            "variant_b": dict(member_gates),
            "cross_member": cross_gates,
        },
        "downstream_training_results_seen": False,
        "qualification_passed": True,
    }
    body["receipt_digest"] = object_sha256(body)
    return body


__all__ = [
    "ALLOWED_NATIVE_ARMS",
    "AppearanceOrbitError",
    "BoundVideo",
    "FileMutationAudit",
    "FileSnapshot",
    "FPS",
    "FRAME_COUNT",
    "FULL_VIDEO_ENCODE_CALLS_PER_ROW",
    "GENERATED_MEMBER_NAMES",
    "INDEPENDENT_RGB_REFERENCE_ENCODE_CALLS_PER_ROW",
    "INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW",
    "LoadedOrbitSpec",
    "MEMBER_NAMES",
    "NATIVE_METHOD",
    "NATIVE_RECEIPT_SCHEMA",
    "NUM_INFERENCE_STEPS",
    "OrbitSpecRow",
    "PINNED_REFERENCE_ENCODING_CONTRACT_DIGEST",
    "QUALIFICATION_SCHEMA",
    "REFERENCE_COUNT",
    "REFERENCE_INDICES",
    "SPEC_SCHEMA",
    "V4_MEMBER_ALIASES",
    "build_materialization_spec",
    "canonical_json_bytes",
    "load_materialization_spec",
    "object_sha256",
    "qualification_seal_body",
    "reference_encoding_contract",
]
