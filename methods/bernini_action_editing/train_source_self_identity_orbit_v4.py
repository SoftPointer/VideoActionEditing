#!/usr/bin/env python3
"""Train Bernini on sealed counterfactual identity orbits (scientific v4).

The trainer consumes only the create-only parquet emitted by
``materialize_appearance_counterfactual_identity_orbit.py``.  Every row is an
exact81/25fps orbit ``(source, variant_a, variant_b)`` externally qualified to
preserve motion, camera, and scene while changing appearance.  Each generated
variant binds its actual native arm independently, so R2V+R2V, R2V+RV2V,
RV2V+R2V, and RV2V+RV2V are legal without assigning a fictional role.  It
never consumes an action-edit target, mask, flow, pose, track, box, or
trajectory.

Training is one fixed 36-microbatch A/C/B/C cycle from
``source_self_identity_orbit_v4``.  Every microbatch uses four distinct
coordinates from the pinned exact40 UniPC shift-5 schedule.  The first
experiment is frozen to rho=0, so the noisy base is the original Gaussian
byte-for-byte.  A short run is legal only as a hash-sealed prefix of the first
36-step cycle and is explicitly non-scientific.

Bernini's native RV2V field requires four transformer forwards.  Retaining
all four graphs for several cells and sigmas would defeat the explicit
no-gradient-checkpointing contract.  This trainer therefore computes the
guided predictions without gradients, differentiates the v4 objective with
respect to those prediction leaves, and then replays the native fields one at
a time with their exact linear VJP coefficients.  With the frozen/eval base
and zero-dropout Q/O adapter this is the exact gradient of the registered
four-forward formula while bounding graph memory to one transformer forward.

This is appearance/motion-role pretext training, not evidence that semantic
action editing works.  A completed run can only authorize the next held-out
role-composition experiment; it cannot authorize an action-editing claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import appearance_counterfactual_identity_orbit as appearance  # noqa: E402
import source_self_identity_orbit_v4 as orbit_v4  # noqa: E402
import source_self_native_ref_contrastive_v3 as native  # noqa: E402
import source_self_native_rv2v_guidance as guidance  # noqa: E402
import source_self_native_target_adapter as target_adapter  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
import train_lora as legacy  # noqa: E402
from tools import materialize_appearance_counterfactual_identity_orbit as materializer  # noqa: E402


METHOD_NAME = "bernini-counterfactual-identity-orbit-v4"
RUN_RECEIPT_SCHEMA = "bernini-counterfactual-identity-orbit-training-receipt-v5"
HISTORY_SCHEMA = "bernini-counterfactual-identity-orbit-step-history-v5"
ADAPTER_SCHEMA = "bernini-native-target-row-qo-lora-checkpoint-v2"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
FPS = 25.0
LATENT_PHASES = 21
REFERENCE_PHASES = 1
REFERENCE_COUNT = 4
INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW = 15
SIGMAS_PER_MICROBATCH = 4
MICROBATCH_CYCLE_STEPS = 36
LORA_RANK = 8
LORA_ALPHA = 8.0
DEFAULT_LEARNING_RATE = 1.0e-5
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260808
DIAGNOSTIC_SIGMA_INDEX = 20
MAX_CORRECT_ERROR_DEGRADATION = 1.25
MIN_WRONG_SCENE_SENSITIVITY_RETENTION = 0.50
MIN_NONZERO_SENSITIVITY = 1.0e-8
VJP_REPLAY_RTOL = 2.0e-5
VJP_REPLAY_ATOL = 2.0e-5
GENERIC_INSTRUCTION = (
    "Use the ordered donor video's temporal evolution and camera trajectory, "
    "while reconstructing the appearance identity selected by the four "
    "reference frames. Preserve all scene content not selected by the references."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class IdentityOrbitTrainingError(RuntimeError):
    """Raised before an ambiguous input, update, or publication is accepted."""


def _validate_rv2v4_registry() -> Mapping[str, Any]:
    reference_contract = appearance.reference_encoding_contract()
    native_contract = native.native_rv2v4_reference_contract()
    if (
        tuple(appearance.REFERENCE_INDICES) != (0, 27, 53, 80)
        or tuple(orbit_v4.REFERENCE_INDICES) != tuple(appearance.REFERENCE_INDICES)
        or appearance.REFERENCE_COUNT != REFERENCE_COUNT
        or native.REFERENCE_COUNT != REFERENCE_COUNT
        or appearance.INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW
        != INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW
        or reference_contract.get("native_deployment_visual_conditioning")
        != "one_video_plus_four_rgb_refs"
        or native_contract.get("total_visual_condition_count") != 5
        or native_contract.get("vi_video_source_ids") != [1.0]
        or native_contract.get("vi_image_source_ids") != [2.0, 3.0, 4.0, 5.0]
        or native_contract.get("i_image_source_ids") != [1.0, 2.0, 3.0, 4.0]
        or native_contract.get("patch_call_source_ids")
        != [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0]
        or native_contract.get("patch_call_roles")
        != [
            "video:VI",
            "ref0:VI",
            "ref0:I",
            "ref1:VI",
            "ref1:I",
            "ref2:VI",
            "ref2:I",
            "ref3:VI",
            "ref3:I",
            "target",
        ]
        or native_contract.get("branch_concat_order")
        != {
            "none": ["target"],
            "V": ["video", "target"],
            "I": ["ref0", "ref1", "ref2", "ref3", "target"],
            "VI": ["video", "ref0", "ref1", "ref2", "ref3", "target"],
        }
        or native_contract.get("latent_concat_dim") != 1
        or native_contract.get("rotary_concat_dim") != 2
        or native_contract.get("source_id_interpolation_used") is not False
        or guidance.guidance_receipt().get(
            "native_rv2v4_reference_contract_digest"
        )
        != native_contract.get("digest")
    ):
        raise IdentityOrbitTrainingError("cross-module native RV2V-4 registry differs")
    return {
        "reference_encoding": dict(reference_contract),
        "native_visual_pack": dict(native_contract),
    }


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise IdentityOrbitTrainingError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise IdentityOrbitTrainingError(
            f"{label} must be a lowercase SHA-{'1' if length == 40 else '256'}"
        )
    return value


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise IdentityOrbitTrainingError(
            f"{label} contains non-finite constant {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise IdentityOrbitTrainingError(
                    f"{label} contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IdentityOrbitTrainingError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, dict):
        raise IdentityOrbitTrainingError(f"{label} root must be one object")
    return value


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> "FileSnapshot":
        if path.is_symlink() or not path.is_file():
            raise IdentityOrbitTrainingError(f"input is not a plain file: {path}")
        before = path.stat()
        digest = runtime.file_sha256(path)
        after = path.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise IdentityOrbitTrainingError(f"input changed while hashing: {path}")
        return cls(
            path,
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            digest,
        )

    def assert_unchanged(self) -> None:
        current = FileSnapshot.capture(self.path)
        if current != self:
            raise IdentityOrbitTrainingError(f"input changed during training: {self.path}")

    def receipt(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "size": self.size,
            "sha256": self.sha256,
            "pre_post_stat_and_hash_stable": True,
        }


@dataclass(frozen=True)
class OrbitRow:
    iid: str
    posterior_blobs: Mapping[str, bytes]
    full_shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    row_digest: str
    qualification_digest: str
    variant_native_arms: tuple[str, str]
    appearance_factor_digest: str


@dataclass(frozen=True)
class OrbitDataset:
    root: Path
    parquet_snapshot: FileSnapshot
    receipt_snapshot: FileSnapshot
    receipt_digest: str
    spec_sha256: str
    pinned_vae_identity: Mapping[str, Any]
    native_arms_by_iid: Mapping[str, Mapping[str, str]]
    live_provenance_audit_digest: str
    rows: tuple[OrbitRow, ...]

    def assert_unchanged(self) -> None:
        self.parquet_snapshot.assert_unchanged()
        self.receipt_snapshot.assert_unchanged()


def _load_posterior_parameters(blob: bytes, *, phases: int, label: str) -> Any:
    import torch

    try:
        value = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except TypeError as error:
        raise IdentityOrbitTrainingError(
            f"{label} runtime lacks fail-closed weights_only loading"
        ) from error
    except Exception as error:
        raise IdentityOrbitTrainingError(f"cannot decode {label}: {error}") from error
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.requires_grad
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 32, phases)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or int(value.shape[3]) % 2
        or int(value.shape[4]) % 2
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise IdentityOrbitTrainingError(
            f"{label} must be detached contiguous FP32 [1,32,{phases},evenH,evenW]"
        )
    return value


def _parse_ascii_json_field(value: Any, *, label: str) -> Any:
    if type(value) is not str:
        raise IdentityOrbitTrainingError(f"{label} must be canonical JSON text")
    try:
        parsed = json.loads(value, parse_constant=lambda item: (_ for _ in ()).throw(
            IdentityOrbitTrainingError(f"{label} contains {item}")
        ))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IdentityOrbitTrainingError(f"cannot parse {label}: {error}") from error
    if canonical_json_bytes(parsed).decode("ascii") != value:
        raise IdentityOrbitTrainingError(f"{label} is not canonical JSON")
    return parsed


def load_orbit_dataset(
    root_value: str | Path,
    *,
    expected_receipt_sha256: str,
    expected_spec_sha256: str,
) -> OrbitDataset:
    """Load the sealed materializer output and reject every weaker contract."""

    rv2v4_registry = _validate_rv2v4_registry()
    expected_receipt = _sha(
        expected_receipt_sha256, length=64, label="dataset receipt SHA"
    )
    expected_spec = _sha(
        expected_spec_sha256, length=64, label="materialization spec SHA"
    )
    requested = Path(root_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise IdentityOrbitTrainingError(
            "dataset root must be an absolute non-symlink directory"
        )
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise IdentityOrbitTrainingError(f"dataset root is unavailable: {error}") from error
    if root != requested or not root.is_dir() or root.is_symlink():
        raise IdentityOrbitTrainingError("dataset root is not canonical/plain")
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != {"dataset.parquet", "receipt.json"}:
        raise IdentityOrbitTrainingError("dataset artifact closure differs")
    parquet_snapshot = FileSnapshot.capture(entries["dataset.parquet"])
    receipt_snapshot = FileSnapshot.capture(entries["receipt.json"])
    if receipt_snapshot.sha256 != expected_receipt:
        raise IdentityOrbitTrainingError("dataset receipt external SHA differs")
    receipt = _strict_json_bytes(entries["receipt.json"].read_bytes(), label="dataset receipt")
    unsigned = dict(receipt)
    declared_digest = unsigned.pop("receipt_digest", None)
    qualification = receipt.get("qualification")
    dataset_record = receipt.get("dataset")
    spec_record = receipt.get("spec")
    encoding = receipt.get("encoding_contract")
    native_input = receipt.get("native_input_contract")
    materialization_mutation = receipt.get("input_file_mutation_audit")
    pinned_vae_identity = receipt.get("pinned_vae_identity")
    if (
        receipt.get("schema_version") != materializer.RECEIPT_SCHEMA
        or receipt.get("method") != materializer.METHOD_NAME
        or object_sha256(unsigned) != declared_digest
        or receipt.get("complete") is not True
        or receipt.get("create_only") is not True
        or receipt.get("scientific_use_authorized") is not True
        or receipt.get("appearance_orbit_pretext_only") is not True
        or receipt.get("paired_action_target_present") is not False
        or receipt.get("synthetic_action_target_present") is not False
        or receipt.get("direct_action_edit_supervision_present") is not False
        or receipt.get("mask_flow_pose_track_box_trajectory_used") is not False
        or receipt.get("direct_action_edit_claim_authorized") is not False
        or not isinstance(qualification, Mapping)
        or qualification.get("all_rows_externally_qualified") is not True
        or qualification.get("missing_qualification_iids") != []
        or qualification.get("semantic_identity_distinction_attested") is not True
        or qualification.get("same_motion_camera_scene_attested") is not True
        or not isinstance(encoding, Mapping)
        or encoding.get("member_order") != list(appearance.MEMBER_NAMES)
        or encoding.get("v4_member_aliases") != appearance.V4_MEMBER_ALIASES
        or encoding.get("reference_count") != REFERENCE_COUNT
        or encoding.get("reference_rgb_indices") != list(appearance.REFERENCE_INDICES)
        or encoding.get("reference_encoding_contract_digest")
        != rv2v4_registry["reference_encoding"]["digest"]
        or encoding.get("full_video_encode_calls_per_row") != 3
        or encoding.get("independent_rgb_reference_encode_calls_per_row") != 12
        or encoding.get("independent_vae_encode_calls_per_row")
        != INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW
        or encoding.get("references_from_full_video_posterior_slice") is not False
        or encoding.get("native_deployment_visual_conditioning")
        != "one_video_plus_four_rgb_refs"
        or not isinstance(native_input, Mapping)
        or native_input.get("allowed_native_arms")
        != list(appearance.ALLOWED_NATIVE_ARMS)
        or not isinstance(native_input.get("member_native_arms_by_iid"), Mapping)
        or native_input.get("frame_count") != FRAME_COUNT
        or float(native_input.get("fps", -1.0)) != FPS
        or native_input.get("num_inference_steps") != 40
        or native_input.get("base_frozen") is not True
        or native_input.get("training_performed") is not False
        or native_input.get("external_target_accepted") is not False
        or native_input.get("mask_flow_pose_track_box_trajectory_used") is not False
        or native_input.get("variant_prompt_hashes_distinct_required") is not True
        or native_input.get("variant_mp4_and_rgb_content_distinct_required") is not True
        or native_input.get("variant_same_seed_required") is not False
        or not isinstance(materialization_mutation, Mapping)
        or materialization_mutation.get("all_files_pre_post_stat_and_hash_stable")
        is not True
        or type(materialization_mutation.get("file_count")) is not int
        or materialization_mutation["file_count"] <= 0
        or not isinstance(materialization_mutation.get("files"), list)
        or len(materialization_mutation["files"])
        != materialization_mutation["file_count"]
        or not isinstance(pinned_vae_identity, Mapping)
    ):
        raise IdentityOrbitTrainingError("dataset scientific/materialization contract differs")
    unsigned_materialization_mutation = dict(materialization_mutation)
    materialization_mutation_digest = unsigned_materialization_mutation.pop(
        "digest", None
    )
    if (
        object_sha256(unsigned_materialization_mutation)
        != materialization_mutation_digest
        or any(
            not isinstance(record, Mapping)
            or record.get("pre_post_stat_and_hash_stable") is not True
            for record in materialization_mutation["files"]
        )
    ):
        raise IdentityOrbitTrainingError("dataset materialization mutation audit differs")
    unsigned_vae_identity = dict(pinned_vae_identity)
    vae_identity_digest = unsigned_vae_identity.pop("vae_identity_digest", None)
    if (
        object_sha256(unsigned_vae_identity) != vae_identity_digest
        or pinned_vae_identity.get("every_vae_file_sha256_verified") is not True
        or pinned_vae_identity.get("posterior_representation")
        != "latent_dist.parameters_fp32"
        or pinned_vae_identity.get("posterior_sample_materialized") is not False
        or pinned_vae_identity.get("checkpoint_content_manifest_sha256")
        != materializer.pinned.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or pinned_vae_identity.get("vae_config_sha256")
        != materializer.pinned.EXPECTED_VAE_CONFIG_SHA256
        or not isinstance(pinned_vae_identity.get("vae_files"), Mapping)
        or not pinned_vae_identity["vae_files"]
    ):
        raise IdentityOrbitTrainingError("dataset pinned VAE identity differs")
    if (
        not isinstance(spec_record, Mapping)
        or spec_record.get("file_sha256") != expected_spec
        or _SHA256.fullmatch(str(spec_record.get("digest"))) is None
        or spec_record.get("reference_encoding_contract_digest")
        != rv2v4_registry["reference_encoding"]["digest"]
    ):
        raise IdentityOrbitTrainingError("dataset binds a different materialization spec")
    if (
        not isinstance(dataset_record, Mapping)
        or Path(str(dataset_record.get("path"))).resolve(strict=True)
        != parquet_snapshot.path
        or dataset_record.get("sha256") != parquet_snapshot.sha256
        or type(dataset_record.get("rows")) is not int
        or dataset_record["rows"] < DP_SIZE
    ):
        raise IdentityOrbitTrainingError("dataset receipt does not bind parquet bytes/rows")

    # Re-open the receipt-bound v3/RV2V-4 spec through its closed validator.  This
    # independently re-hashes the source MP4s, both variants, both native
    # receipts, and every external qualification seal.  In particular, native
    # arms are learned from the two receipt bindings; no R2V/RV2V pairing is
    # assigned by this trainer.
    provenance_audit = appearance.FileMutationAudit()
    try:
        loaded_spec = appearance.load_materialization_spec(
            str(spec_record.get("path")),
            expected_sha256=expected_spec,
            audit=provenance_audit,
        )
        provenance_records = provenance_audit.finalize()
    except appearance.AppearanceOrbitError as error:
        raise IdentityOrbitTrainingError(
            f"cannot revalidate live appearance-orbit provenance: {error}"
        ) from error
    if (
        loaded_spec.file_sha256 != expected_spec
        or loaded_spec.digest != spec_record.get("digest")
        or dict(loaded_spec.reference_encoding_contract)
        != rv2v4_registry["reference_encoding"]
        or len(loaded_spec.rows) != dataset_record["rows"]
        or any(not row.scientific_use_authorized for row in loaded_spec.rows)
    ):
        raise IdentityOrbitTrainingError("live v3 RV2V-4 materialization spec binding differs")
    live_provenance_audit = {
        "records": list(provenance_records),
        "all_files_pre_post_stat_and_hash_stable": True,
        "revalidated_by_closed_v3_rv2v4_contract": True,
    }
    live_provenance_audit_digest = object_sha256(live_provenance_audit)
    spec_rows_by_iid = {row.iid: row for row in loaded_spec.rows}

    before = parquet_snapshot.path.stat()
    try:
        import pyarrow.parquet as pq

        raw_rows = pq.read_table(parquet_snapshot.path).to_pylist()
    except Exception as error:
        raise IdentityOrbitTrainingError(f"cannot read orbit parquet: {error}") from error
    after = parquet_snapshot.path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or runtime.file_sha256(parquet_snapshot.path) != parquet_snapshot.sha256
    ):
        raise IdentityOrbitTrainingError("dataset parquet changed while reading")
    if len(raw_rows) != dataset_record["rows"]:
        raise IdentityOrbitTrainingError("dataset parquet row count differs")

    rows: list[OrbitRow] = []
    seen: set[str] = set()
    cohort_shapes: set[tuple[int, ...]] = set()
    parsed_native_arms: dict[str, Mapping[str, str]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or raw.get("schema_version") != materializer.ROW_SCHEMA:
            raise IdentityOrbitTrainingError("orbit row schema differs")
        iid = raw.get("iid")
        if type(iid) is not str or _IID.fullmatch(iid) is None or iid in seen:
            raise IdentityOrbitTrainingError("orbit IID is invalid or duplicated")
        seen.add(iid)
        spec_row = spec_rows_by_iid.get(iid)
        if spec_row is None or spec_row.qualification is None:
            raise IdentityOrbitTrainingError(
                f"{iid} is absent or unqualified in the live v3 RV2V-4 spec"
            )
        actual_posterior_fields = {
            key for key in raw if str(key).endswith("_posterior_blob")
        }
        if actual_posterior_fields != set(materializer.POSTERIOR_FIELDS):
            raise IdentityOrbitTrainingError(f"{iid} posterior field closure differs")
        if materializer._row_digest(raw) != raw.get("row_digest"):
            raise IdentityOrbitTrainingError(f"{iid} row digest differs")
        if (
            raw.get("frame_count") != FRAME_COUNT
            or float(raw.get("fps", -1.0)) != FPS
            or raw.get("reference_count") != REFERENCE_COUNT
            or raw.get("reference_encoding_contract_digest")
            != rv2v4_registry["reference_encoding"]["digest"]
            or raw.get("independent_vae_encode_calls")
            != INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW
            or raw.get("three_full_videos_independently_encoded") is not True
            or raw.get("twelve_rgb_references_independently_encoded") is not True
            or raw.get("references_from_full_video_posterior_slice") is not False
            or raw.get("native_deployment_visual_conditioning")
            != "one_video_plus_four_rgb_refs"
            or raw.get("native_receipt_source_output_exactly_bound") is not True
            or raw.get("native_exact81_25fps_40steps_frozen") is not True
            or raw.get("external_target_accepted") is not False
            or raw.get("mask_flow_pose_track_box_trajectory_used") is not False
            or raw.get("paired_action_target_present") is not False
            or raw.get("synthetic_action_target_present") is not False
            or raw.get("rgb_member_tensors_content_distinct") is not True
            or raw.get("member_role")
            != "appearance_identity_orbit_same_motion_candidate"
            or raw.get("scientific_use_authorized") is not True
            or raw.get("direct_action_edit_claim_authorized") is not False
        ):
            raise IdentityOrbitTrainingError(f"{iid} row role/scientific contract differs")
        member_order = _parse_ascii_json_field(
            raw.get("member_order_json"), label=f"{iid} member order"
        )
        v4_aliases = _parse_ascii_json_field(
            raw.get("v4_member_aliases_json"), label=f"{iid} v4 member aliases"
        )
        member_bindings = _parse_ascii_json_field(
            raw.get("member_bindings_json"), label=f"{iid} member bindings"
        )
        appearance_factors = _parse_ascii_json_field(
            raw.get("appearance_intervention_generation_factors_json"),
            label=f"{iid} appearance intervention factors",
        )
        reference_indices = _parse_ascii_json_field(
            raw.get("reference_indices_json"), label=f"{iid} reference indices"
        )
        rgb_digests = _parse_ascii_json_field(
            raw.get("rgb_tensor_sha256_json"), label=f"{iid} RGB tensor hashes"
        )
        qualification_binding = _parse_ascii_json_field(
            raw.get("qualification_binding_json"), label=f"{iid} qualification"
        )
        mutation = _parse_ascii_json_field(
            raw.get("input_file_mutation_audit_json"), label=f"{iid} mutation audit"
        )
        encode_metadata = _parse_ascii_json_field(
            raw.get("independent_vae_encode_metadata_json"),
            label=f"{iid} independent VAE encode metadata",
        )
        expected_artifact_roles = dict(materializer.POSTERIOR_ARTIFACT_ROLES)
        if (
            not isinstance(encode_metadata, Mapping)
            or set(encode_metadata) != set(materializer.POSTERIOR_FIELDS)
            or any(
                not isinstance(encode_metadata[field], Mapping)
                or type(encode_metadata[field].get("encode_call_index")) is not int
                or encode_metadata[field].get("encode_call_index") != call_index
                or encode_metadata[field].get("artifact_role")
                != expected_artifact_roles[field]
                or encode_metadata[field].get("encoded_independently") is not True
                or encode_metadata[field].get("encoded_directly_from_rgb") is not True
                or encode_metadata[field].get(
                    "reference_from_full_video_posterior_slice"
                )
                is not False
                for call_index, field in enumerate(materializer.POSTERIOR_FIELDS)
            )
        ):
            raise IdentityOrbitTrainingError(
                f"{iid} independent 15-call VAE metadata closure differs"
            )

        variant_a = spec_row.variant_a
        variant_b = spec_row.variant_b
        expected_arms = {
            "variant_a": variant_a.native_arm,
            "variant_b": variant_b.native_arm,
        }
        expected_member_bindings = {
            "source": {
                "video_path": str(spec_row.source.path),
                "video_sha256": spec_row.source.sha256,
                "native_receipt": None,
            },
            "variant_a": {
                "video_path": str(variant_a.video.path),
                "video_sha256": variant_a.video.sha256,
                "native_arm": variant_a.native_arm,
                "native_receipt": {
                    "path": str(variant_a.receipt_path),
                    "file_sha256": variant_a.receipt_file_sha256,
                    "digest": variant_a.receipt_digest,
                    "arm": variant_a.native_arm,
                },
            },
            "variant_b": {
                "video_path": str(variant_b.video.path),
                "video_sha256": variant_b.video.sha256,
                "native_arm": variant_b.native_arm,
                "native_receipt": {
                    "path": str(variant_b.receipt_path),
                    "file_sha256": variant_b.receipt_file_sha256,
                    "digest": variant_b.receipt_digest,
                    "arm": variant_b.native_arm,
                },
            },
        }
        expected_appearance_factors = {
            "variant_a": {
                "native_arm": variant_a.native_arm,
                "action_prompt_utf8_sha256": variant_a.receipt["input"][
                    "action_prompt_utf8_sha256"
                ],
                "seed": variant_a.receipt["sampling"][variant_a.native_arm]["seed"],
            },
            "variant_b": {
                "native_arm": variant_b.native_arm,
                "action_prompt_utf8_sha256": variant_b.receipt["input"][
                    "action_prompt_utf8_sha256"
                ],
                "seed": variant_b.receipt["sampling"][variant_b.native_arm]["seed"],
            },
            "distinct_action_prompt_hashes_required": True,
            "distinct_variant_mp4_and_rgb_content_required": True,
            "same_seed_required": False,
            "semantic_identity_distinction_requires_external_qualification": True,
        }
        expected_qualification_binding = {
            "present": True,
            "path": str(spec_row.qualification.path),
            "file_sha256": spec_row.qualification.file_sha256,
            "digest": spec_row.qualification.digest,
            "scientific_use_authorized": True,
        }
        mutation_files = mutation.get("files") if isinstance(mutation, Mapping) else None
        if (
            member_order != list(appearance.MEMBER_NAMES)
            or v4_aliases != appearance.V4_MEMBER_ALIASES
            or member_bindings != expected_member_bindings
            or appearance_factors != expected_appearance_factors
            or reference_indices != list(appearance.REFERENCE_INDICES)
            or qualification_binding != expected_qualification_binding
            or raw.get("source_video_path") != str(spec_row.source.path)
            or raw.get("source_video_sha256") != spec_row.source.sha256
            or raw.get("variant_a_native_arm") != variant_a.native_arm
            or raw.get("variant_a_video_path") != str(variant_a.video.path)
            or raw.get("variant_a_video_sha256") != variant_a.video.sha256
            or raw.get("variant_a_native_receipt_path")
            != str(variant_a.receipt_path)
            or raw.get("variant_a_native_receipt_file_sha256")
            != variant_a.receipt_file_sha256
            or raw.get("variant_a_native_receipt_digest")
            != variant_a.receipt_digest
            or raw.get("variant_b_native_arm") != variant_b.native_arm
            or raw.get("variant_b_video_path") != str(variant_b.video.path)
            or raw.get("variant_b_video_sha256") != variant_b.video.sha256
            or raw.get("variant_b_native_receipt_path")
            != str(variant_b.receipt_path)
            or raw.get("variant_b_native_receipt_file_sha256")
            != variant_b.receipt_file_sha256
            or raw.get("variant_b_native_receipt_digest")
            != variant_b.receipt_digest
            or raw.get("qualification_seal_path")
            != str(spec_row.qualification.path)
            or raw.get("qualification_seal_file_sha256")
            != spec_row.qualification.file_sha256
            or raw.get("qualification_seal_digest") != spec_row.qualification.digest
            or len(
                {
                    spec_row.source.sha256,
                    variant_a.video.sha256,
                    variant_b.video.sha256,
                }
            )
            != 3
            or not isinstance(rgb_digests, Mapping)
            or set(rgb_digests) != set(appearance.MEMBER_NAMES)
            or any(_SHA256.fullmatch(str(value)) is None for value in rgb_digests.values())
            or len(set(rgb_digests.values())) != 3
            or not isinstance(mutation, Mapping)
            or mutation.get("all_files_pre_post_stat_and_hash_stable") is not True
            or not isinstance(mutation_files, list)
            or not mutation_files
            or mutation.get("digest") != object_sha256(mutation_files)
            or any(
                not isinstance(record, Mapping)
                or record.get("pre_post_stat_and_hash_stable") is not True
                for record in mutation_files
            )
        ):
            raise IdentityOrbitTrainingError(f"{iid} qualification/mutation binding differs")
        parsed_native_arms[iid] = expected_arms
        if any(
            not isinstance(raw.get(field), (bytes, bytearray, memoryview))
            for field in materializer.POSTERIOR_FIELDS
        ):
            raise IdentityOrbitTrainingError(f"{iid} posterior blob closure differs")
        blobs = {
            field: bytes(raw[field]) for field in materializer.POSTERIOR_FIELDS
        }
        decoded = {
            field: _load_posterior_parameters(
                blob,
                phases=(
                    LATENT_PHASES if "_full_posterior_blob" in field else REFERENCE_PHASES
                ),
                label=f"{iid} {field}",
            )
            for field, blob in blobs.items()
        }
        full_shapes = {
            tuple(decoded[f"{member}_full_posterior_blob"].shape)
            for member in appearance.MEMBER_NAMES
        }
        ref_shapes = {
            tuple(decoded[f"{member}_ref{index}_posterior_blob"].shape)
            for member in appearance.MEMBER_NAMES
            for index in appearance.REFERENCE_INDICES
        }
        if len(full_shapes) != 1 or len(ref_shapes) != 1:
            raise IdentityOrbitTrainingError(f"{iid} posterior geometry differs within orbit")
        full_shape = next(iter(full_shapes))
        ref_shape = next(iter(ref_shapes))
        if full_shape[:2] != ref_shape[:2] or full_shape[3:] != ref_shape[3:]:
            raise IdentityOrbitTrainingError(f"{iid} full/reference spatial geometry differs")
        cohort_shapes.add(full_shape)
        rows.append(
            OrbitRow(
                iid=iid,
                posterior_blobs=blobs,
                full_shape=full_shape,
                reference_shape=ref_shape,
                row_digest=str(raw["row_digest"]),
                qualification_digest=str(qualification_binding["digest"]),
                variant_native_arms=(variant_a.native_arm, variant_b.native_arm),
                appearance_factor_digest=object_sha256(appearance_factors),
            )
        )
    # The held-out wrong-scene gate exchanges references across rows.  Native
    # packing cannot define that intervention across different latent buckets.
    if len(cohort_shapes) != 1:
        raise IdentityOrbitTrainingError(
            "v4 cohort must use one latent bucket for held-out cross-scene gates"
        )
    if dataset_record.get("iids") != [row.iid for row in rows] or dataset_record.get(
        "row_digests"
    ) != [row.row_digest for row in rows]:
        raise IdentityOrbitTrainingError("dataset receipt row membership differs")
    receipt_native_arms = native_input.get("member_native_arms_by_iid")
    if (
        receipt_native_arms != parsed_native_arms
        or set(spec_rows_by_iid) != seen
    ):
        raise IdentityOrbitTrainingError(
            "dataset variant native arms differ from row/spec receipts"
        )
    return OrbitDataset(
        root=root,
        parquet_snapshot=parquet_snapshot,
        receipt_snapshot=receipt_snapshot,
        receipt_digest=str(declared_digest),
        spec_sha256=expected_spec,
        pinned_vae_identity=dict(pinned_vae_identity),
        native_arms_by_iid={
            iid: dict(arms) for iid, arms in parsed_native_arms.items()
        },
        live_provenance_audit_digest=live_provenance_audit_digest,
        rows=tuple(rows),
    )


def validate_dataset_vae_against_checkpoint(
    dataset: OrbitDataset, checkpoint: Path
) -> Mapping[str, Any]:
    """Re-hash every receipt-bound VAE file used for offline posteriors."""

    identity = dict(dataset.pinned_vae_identity)
    try:
        bound_root = Path(str(identity["checkpoint_root"])).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise IdentityOrbitTrainingError(
            f"dataset VAE checkpoint root is unavailable: {error}"
        ) from error
    if bound_root != checkpoint:
        raise IdentityOrbitTrainingError(
            "offline orbit posteriors were encoded by a different checkpoint root"
        )
    files = identity.get("vae_files")
    assert isinstance(files, Mapping)
    observed: dict[str, str] = {}
    for relative, declared in sorted(files.items()):
        if (
            type(relative) is not str
            or not relative.startswith("vae/")
            or ".." in Path(relative).parts
        ):
            raise IdentityOrbitTrainingError("dataset VAE file path is unsafe")
        expected = _sha(declared, length=64, label=f"dataset VAE {relative}")
        path = checkpoint / relative
        if path.is_symlink() or not path.is_file():
            raise IdentityOrbitTrainingError(f"dataset VAE file is absent: {relative}")
        observed[relative] = runtime.file_sha256(path)
        if observed[relative] != expected:
            raise IdentityOrbitTrainingError(f"dataset VAE file differs: {relative}")
    value = {
        "dataset_vae_identity_digest": identity["vae_identity_digest"],
        "training_checkpoint_root": str(checkpoint),
        "vae_files": observed,
        "all_offline_encoder_files_rehashed_before_training": True,
    }
    return {**value, "digest": object_sha256(value)}


def _posterior_mode(
    blob: bytes, mean: Any, std: Any, *, phases: int, label: str
) -> Any:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    parameters = _load_posterior_parameters(blob, phases=phases, label=label)
    mode = DiagonalGaussianDistribution(parameters).mode().float()
    mode = ((mode - mean.unsqueeze(0)) / std.unsqueeze(0)).detach().contiguous()
    if (
        tuple(int(item) for item in mode.shape[:3]) != (1, 16, phases)
        or not bool(torch.isfinite(mode).all().item())
    ):
        raise IdentityOrbitTrainingError(f"{label} normalized posterior mode differs")
    return mode


def build_identity_orbit_from_row(
    row: OrbitRow, *, vae_mean: Any, vae_std: Any, device: Any
) -> orbit_v4.IdentityOrbit:
    members: list[orbit_v4.IdentityOrbitMember] = []
    for index, materialized_name in enumerate(appearance.MEMBER_NAMES):
        orbit_name = orbit_v4.ORBIT_MEMBER_NAMES[index]
        if appearance.V4_MEMBER_ALIASES.get(orbit_name) != materialized_name:
            raise IdentityOrbitTrainingError(
                "materialized member order differs from the registered v4 aliases"
            )
        video = _posterior_mode(
            row.posterior_blobs[f"{materialized_name}_full_posterior_blob"],
            vae_mean,
            vae_std,
            phases=LATENT_PHASES,
            label=f"{row.iid} {materialized_name} full",
        ).to(device=device, dtype=__import__("torch").float32)
        refs = tuple(
            _posterior_mode(
                row.posterior_blobs[
                    f"{materialized_name}_ref{rgb_index}_posterior_blob"
                ],
                vae_mean,
                vae_std,
                phases=REFERENCE_PHASES,
                label=f"{row.iid} {materialized_name} ref{rgb_index}",
            ).to(device=device, dtype=__import__("torch").float32)
            for rgb_index in appearance.REFERENCE_INDICES
        )
        members.append(
            orbit_v4.IdentityOrbitMember(orbit_name, video, refs)
        )
    return orbit_v4.IdentityOrbit(
        tuple(members),  # type: ignore[arg-type]
        same_motion_attested=True,
        same_camera_attested=True,
        same_scene_attested=True,
        appearance_only_counterfactual_attested=True,
        independently_encoded_rgb_refs_attested=True,
    )


def broadcast_orbit_within_sp(
    orbit: orbit_v4.IdentityOrbit, *, parallel: runtime.ParallelContext
) -> Mapping[str, Any]:
    import torch.distributed as dist

    source_rank = runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    tensors: list[tuple[str, Any]] = []
    for member in orbit.members:
        tensors.append((f"{member.name}.video", member.video_latent))
        tensors.extend(
            (f"{member.name}.ref{index}", value)
            for index, value in zip(appearance.REFERENCE_INDICES, member.image_references)
        )
    for _, value in tensors:
        dist.broadcast(value, src=source_rank, group=parallel.sp_group)
    digest = object_sha256(
        {name: runtime.tensor_sha256(value) for name, value in tensors}
    )
    runtime.digest_consensus(
        digest,
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label="identity orbit tensors",
    )
    return {"source_rank": source_rank, "tensor_digest": digest}


def unpack_native_target_tokens(
    packed: Any, *, video_shape: Sequence[int]
) -> Any:
    """Invert Wan's exact ``(t,h,w),(pt,ph,pw,c)`` target-token order."""

    import torch

    shape = tuple(int(item) for item in video_shape)
    if len(shape) != 5 or shape[:3] != (1, 16, LATENT_PHASES):
        raise IdentityOrbitTrainingError("target video shape must be [1,16,21,H,W]")
    batch, channels, phases, height, width = shape
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise IdentityOrbitTrainingError("target video spatial dimensions must be positive/even")
    tokens = phases * (height // 2) * (width // 2)
    if (
        not isinstance(packed, torch.Tensor)
        or packed.ndim != 3
        or tuple(int(item) for item in packed.shape) != (batch, tokens, 64)
        or not packed.is_floating_point()
    ):
        raise IdentityOrbitTrainingError(
            f"native target output must be [1,{tokens},64]"
        )
    patches = packed.reshape(
        batch, phases, height // 2, width // 2, 1, 2, 2, channels
    )
    return (
        patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )


def native_rv2v_vjp_branches(
    pack: native.NativeRV2VPack,
    *,
    cond_embeds: Any,
    uncond_embeds: Any,
) -> tuple[tuple[str, native.NativeRV2VBranch, Any, float], ...]:
    """Expanded exact coefficients of Bernini's registered RV2V formula."""

    branches = (
        ("none_uncond", pack.none, uncond_embeds, -0.25),
        ("V_uncond", pack.video, uncond_embeds, -3.25),
        ("VI_uncond", pack.video_image, uncond_embeds, 0.5),
        ("VI_cond", pack.video_image, cond_embeds, 4.0),
    )
    if tuple(item[0] for item in branches) != tuple(
        guidance.guidance_receipt()["forward_order"]
    ):
        raise IdentityOrbitTrainingError("native RV2V branch order differs")
    if not math.isclose(sum(item[3] for item in branches), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise IdentityOrbitTrainingError("expanded native RV2V coefficients differ")
    return branches


def _forward_one_native_branch(
    diffusion: Any,
    branch: native.NativeRV2VBranch,
    *,
    timestep: Any,
    text: Any,
    adapter: target_adapter.NativeTargetAdapterHandle,
    sp_rank: int,
) -> Any:
    route = target_adapter.NativeTargetRoute(
        total_tokens=branch.total_tokens,
        condition_tokens=branch.condition_tokens,
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=SP_SIZE,
        branch_name=branch.name,
    )
    with adapter.route(route):
        return native.forward_native_target_branch(
            diffusion, branch, timestep=timestep, cond_embeds=text
        )


def _guided_prediction_no_grad(
    diffusion: Any,
    pack: native.NativeRV2VPack,
    *,
    timestep: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
    adapter: target_adapter.NativeTargetAdapterHandle,
    sp_rank: int,
    video_shape: Sequence[int],
) -> Any:
    import torch

    components: dict[str, Any] = {}
    with torch.no_grad():
        for name, branch, text, _ in native_rv2v_vjp_branches(
            pack, cond_embeds=cond_embeds, uncond_embeds=uncond_embeds
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                components[name] = _forward_one_native_branch(
                    diffusion,
                    branch,
                    timestep=timestep,
                    text=text,
                    adapter=adapter,
                    sp_rank=sp_rank,
                )
    # Keep the expression identical to the pinned vendor formula.  Expanding
    # it before evaluation can change bf16 rounding even though its VJP is the
    # exact coefficient tuple above.
    none = components["none_uncond"]
    video = components["V_uncond"]
    vi_u = components["VI_uncond"]
    vi_c = components["VI_cond"]
    guided = (
        none
        + guidance.OMEGA_VIDEO * (video - none)
        + guidance.OMEGA_IMAGE * (vi_u - video)
        + guidance.OMEGA_TEXT * (vi_c - vi_u)
    )
    return unpack_native_target_tokens(guided.float(), video_shape=video_shape)


def _build_pack(
    transformer: Any,
    cell: orbit_v4.OrbitCell,
    states: orbit_v4.SourceRichStates,
    *,
    sigma_position: int,
) -> native.NativeRV2VPack:
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return orbit_v4.pack_orbit_cell_at_sigma(
            transformer, cell, states, sigma_position=sigma_position
        )


def _replay_prediction_vjp(
    diffusion: Any,
    transformer: Any,
    cell: orbit_v4.OrbitCell,
    states: orbit_v4.SourceRichStates,
    *,
    sigma_position: int,
    output_cotangent: Any,
    expected_guided: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
    adapter: target_adapter.NativeTargetAdapterHandle,
    sp_rank: int,
) -> float:
    """Replay four fields serially and apply the exact guided-output VJP."""

    import torch

    if (
        output_cotangent.shape != cell.target.shape
        or output_cotangent.requires_grad
        or not bool(torch.isfinite(output_cotangent).all().item())
    ):
        raise IdentityOrbitTrainingError("guided-output cotangent differs")
    pack = _build_pack(transformer, cell, states, sigma_position=sigma_position)
    timestep = states.timesteps[sigma_position : sigma_position + 1]
    replay_components: dict[str, Any] = {}
    for name, branch, text, coefficient in native_rv2v_vjp_branches(
        pack, cond_embeds=cond_embeds, uncond_embeds=uncond_embeds
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = _forward_one_native_branch(
                diffusion,
                branch,
                timestep=timestep,
                text=text,
                adapter=adapter,
                sp_rank=sp_rank,
            )
            video = unpack_native_target_tokens(
                packed, video_shape=tuple(cell.target.shape)
            )
        # Preserve the vendor output dtype for the replay parity expression;
        # promoting individual components before the formula would change the
        # bf16 rounding path relative to the no-grad prepass.
        replay_components[name] = video.detach()
        torch.autograd.backward(
            video,
            # Autograd through ``guided.float()`` first casts the upstream
            # cotangent back to the vendor dtype and only then applies the
            # exactly representable binary coefficient.
            grad_tensors=(
                output_cotangent.to(video.dtype) * float(coefficient)
            ),
        )
    none = replay_components["none_uncond"]
    video = replay_components["V_uncond"]
    vi_u = replay_components["VI_uncond"]
    vi_c = replay_components["VI_cond"]
    replay_guided = (
        none
        + guidance.OMEGA_VIDEO * (video - none)
        + guidance.OMEGA_IMAGE * (vi_u - video)
        + guidance.OMEGA_TEXT * (vi_c - vi_u)
    )
    difference = (replay_guided.float() - expected_guided.float()).abs()
    maximum = float(difference.max().item())
    scale = float(expected_guided.float().abs().max().item())
    if maximum > VJP_REPLAY_ATOL + VJP_REPLAY_RTOL * scale:
        raise IdentityOrbitTrainingError(
            f"native VJP replay changed guided prediction: max_abs={maximum} scale={scale}"
        )
    return maximum


def prefix_seal_body(step_count: int) -> Mapping[str, Any]:
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or not 1 <= step_count < MICROBATCH_CYCLE_STEPS
    ):
        raise IdentityOrbitTrainingError("sealed prefix length must lie in [1,35]")
    cycle = orbit_v4.registered_orbit_microbatch_cycle()
    value = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "role": "incomplete_first_cycle_prefix_seal",
        "cycle_digest": orbit_v4.orbit_microbatch_cycle_receipt()["digest"],
        "prefix_start_ordinal": 0,
        "prefix_step_count": step_count,
        "prefix_step_digests": [step.receipt()["digest"] for step in cycle[:step_count]],
        "continuation_or_scientific_claim_authorized": False,
    }
    return {**value, "digest": object_sha256(value)}


def _noise_seed(base_seed: int, step: int, row_index: int, dp_rank: int) -> int:
    material = (
        f"{base_seed}\0identity-orbit-v4\0{step}\0{row_index}\0{dp_rank}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def _diagnostic_noise_seed(base_seed: int, row_index: int, dp_rank: int) -> int:
    material = (
        f"{base_seed}\0identity-orbit-v4-heldout\0{row_index}\0{dp_rank}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def _sample_epsilon(shape: Sequence[int], *, seed: int, device: Any) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return value.to(device=device).contiguous()


def _tokenize_positive(tokenizer: Any, text: str) -> tuple[Any, Any]:
    import torch

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    if (
        input_ids.ndim != 2
        or tuple(input_ids.shape) != tuple(attention_mask.shape)
        or int(input_ids.shape[0]) != 1
        or int(input_ids.shape[1]) <= 0
    ):
        raise IdentityOrbitTrainingError("positive tokenizer output differs")
    length = int(input_ids.shape[1])
    if length >= 512:
        return input_ids[:, :512], attention_mask[:, :512]
    padding = 512 - length
    return (
        torch.cat((input_ids, input_ids.new_zeros((1, padding))), dim=1),
        torch.cat((attention_mask, attention_mask.new_zeros((1, padding))), dim=1),
    )


def _tokenize_negative(tokenizer: Any, text: str) -> tuple[Any, Any]:
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if tuple(encoded.input_ids.shape) != (1, 512) or tuple(
        encoded.attention_mask.shape
    ) != (1, 512):
        raise IdentityOrbitTrainingError("negative tokenizer output differs")
    return encoded.input_ids, encoded.attention_mask


def _frozen_text_embeddings(renderer: Any, tokenizer: Any, device: Any) -> tuple[Any, Any, str]:
    import torch

    positive_ids, positive_mask = _tokenize_positive(tokenizer, GENERIC_INSTRUCTION)
    negative_ids, negative_mask = _tokenize_negative(tokenizer, DEFAULT_NEGATIVE_PROMPT)
    with torch.inference_mode():
        conditional = renderer.encode_prompt(
            positive_ids.to(device), positive_mask.to(device)
        ).detach()
        unconditional = renderer.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    if (
        conditional.ndim != 3
        or conditional.shape != unconditional.shape
        or conditional.device != device
        or unconditional.device != device
        or conditional.requires_grad
        or unconditional.requires_grad
        or not bool(torch.isfinite(conditional).all().item())
        or not bool(torch.isfinite(unconditional).all().item())
    ):
        raise IdentityOrbitTrainingError("frozen conditional/unconditional embeddings differ")
    digest = object_sha256(
        {
            "instruction": GENERIC_INSTRUCTION,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "conditional": runtime.tensor_sha256(conditional),
            "unconditional": runtime.tensor_sha256(unconditional),
        }
    )
    return conditional, unconditional, digest


def _gradient_checkpointing_enabled(renderer: Any, transformer: Any) -> bool:
    return any(
        bool(getattr(owner, name, False))
        for owner, name in (
            (renderer, "is_gradient_checkpointing"),
            (transformer, "gradient_checkpointing"),
            (transformer, "is_gradient_checkpointing"),
        )
    )


def _disable_gradient_checkpointing(renderer: Any, transformer: Any) -> Mapping[str, Any]:
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if _gradient_checkpointing_enabled(renderer, transformer):
        raise IdentityOrbitTrainingError(
            "gradient checkpointing remains enabled after explicit disable"
        )
    return orbit_v4.validate_microbatch_runtime(
        gradient_checkpointing_enabled=False
    )


def _causal_snapshot(
    *,
    diffusion: Any,
    transformer: Any,
    orbit: orbit_v4.IdentityOrbit,
    wrong_orbit: orbit_v4.IdentityOrbit,
    epsilon: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
    adapter: target_adapter.NativeTargetAdapterHandle,
    sp_rank: int,
) -> tuple[Mapping[str, float], Any, Any, Any]:
    import torch

    cell = next(
        item
        for item in orbit_v4.build_identity_orbit_cells(orbit, transforms=("identity",))
        if item.key == orbit_v4.OrbitCellKey(0, 0, "identity")
    )
    states = orbit_v4.build_orbit_cell_states(
        cell,
        epsilon,
        indices=(DIAGNOSTIC_SIGMA_INDEX,),
        rho_schedule=orbit_v4.SourceRichRhoSchedule(max_rho=0.0),
    )
    correct_pack = _build_pack(transformer, cell, states, sigma_position=0)
    timestep = states.timesteps[:1]
    correct_velocity = _guided_prediction_no_grad(
        diffusion,
        correct_pack,
        timestep=timestep,
        cond_embeds=cond_embeds,
        uncond_embeds=uncond_embeds,
        adapter=adapter,
        sp_rank=sp_rank,
        video_shape=tuple(cell.target.shape),
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        wrong_pack = native.build_native_rv2v_pack(
            transformer,
            donor_video=cell.donor,
            image_references=wrong_orbit.members[0].image_references,
            noisy_target=states.noisy[0],
        )
    wrong_velocity = _guided_prediction_no_grad(
        diffusion,
        wrong_pack,
        timestep=timestep,
        cond_embeds=cond_embeds,
        uncond_embeds=uncond_embeds,
        adapter=adapter,
        sp_rank=sp_rank,
        video_shape=tuple(cell.target.shape),
    )
    correct_clean = (states.noise_base[0] - correct_velocity.float()).detach()
    wrong_clean = (states.noise_base[0] - wrong_velocity.float()).detach()
    target = cell.target.detach().float()
    correct_error = float((correct_clean - target).square().mean().item())
    sensitivity = float(
        (wrong_clean - correct_clean).square().mean().sqrt().item()
    )
    if (
        not math.isfinite(correct_error)
        or not math.isfinite(sensitivity)
        or sensitivity < MIN_NONZERO_SENSITIVITY
    ):
        raise IdentityOrbitTrainingError(
            "held-out pre/post causal snapshot is non-finite or reference-insensitive"
        )
    return (
        {
            "correct_target_error": correct_error,
            "wrong_scene_output_sensitivity": sensitivity,
        },
        correct_clean,
        wrong_clean,
        target,
    )


def _save_and_roundtrip_adapter(
    path: Path,
    adapter: target_adapter.NativeTargetAdapterHandle,
) -> Mapping[str, Any]:
    """Persist, reload, copy into the live adapter, and prove exact equality."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = dict(adapter.state_dict_for_save())
    named = adapter.trainable_named_parameters()
    expected_keys = [name for name, _ in named]
    if list(state) != expected_keys:
        raise IdentityOrbitTrainingError("adapter save state key order differs")
    before_digest = runtime.trainable_parameters_digest(named)
    metadata = {
        "schema_version": ADAPTER_SCHEMA,
        "adapter_contract_digest": str(adapter.receipt()["digest"]),
        "block_indices_json": canonical_json_bytes(list(adapter.block_indices)).decode("ascii"),
        "rank": str(LORA_RANK),
        "alpha_hex": float(LORA_ALPHA).hex(),
        "rho_hex": float(0.0).hex(),
        "native_guidance_digest": str(guidance.guidance_receipt()["digest"]),
        "native_schedule_digest": str(native.native_unipc40_schedule_receipt()["digest"]),
        "native_rv2v4_reference_contract_digest": str(
            native.native_rv2v4_reference_contract()["digest"]
        ),
        "reference_rgb_indices_json": canonical_json_bytes(
            list(appearance.REFERENCE_INDICES)
        ).decode("ascii"),
        "gradient_checkpointing_enabled": "false",
        "adapter_activation_schedule": "all_40_native_unipc_forward_coordinates",
        "inference_requires_same_target_route_and_rho": "true",
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".safetensors",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(state, str(temporary), metadata=metadata)
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            loaded_metadata = dict(opened.metadata() or {})
            loaded_keys = list(opened.keys())
            loaded = {name: opened.get_tensor(name).contiguous() for name in loaded_keys}
        if loaded_metadata != metadata or loaded_keys != sorted(expected_keys):
            # safetensors exposes keys lexicographically; compare membership
            # separately from the trainer's semantic parameter order.
            if loaded_metadata != metadata or set(loaded_keys) != set(expected_keys):
                raise IdentityOrbitTrainingError("adapter safetensors closure differs")
        by_name = dict(named)
        for name in expected_keys:
            restored = loaded[name]
            expected = state[name]
            if (
                restored.dtype != torch.float32
                or not torch.equal(restored, expected)
                or tuple(restored.shape) != tuple(by_name[name].shape)
            ):
                raise IdentityOrbitTrainingError(
                    f"adapter safetensors tensor differs: {name}"
                )
            # This is an actual file -> live-module strict load, not merely a
            # comparison of two CPU dictionaries.
            by_name[name].data.copy_(
                restored.to(device=by_name[name].device, dtype=by_name[name].dtype)
            )
        after_digest = runtime.trainable_parameters_digest(named)
        if after_digest != before_digest:
            raise IdentityOrbitTrainingError("live adapter changed across strict roundtrip")
        runtime.durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema_version": ADAPTER_SCHEMA,
        "file_sha256": runtime.file_sha256(path),
        "tensor_count": len(expected_keys),
        "parameter_digest_before_save": before_digest,
        "parameter_digest_after_file_load": after_digest,
        "file_tensors_equal_pre_save_parameters": True,
        "file_loaded_into_live_adapter": True,
        "strict_key_shape_dtype_value_roundtrip": True,
        "metadata": metadata,
    }


def _publish_create_only_run_bundle(stage: Path, output: Path) -> None:
    """Create the final directory exclusively and publish its receipt last.

    A POSIX directory rename can replace an empty directory created by a
    concurrent writer.  The exclusive ``mkdir`` below is the no-overwrite
    primitive.  The three payload files are hard-linked before ``receipt.json``;
    readers therefore have an unambiguous commit marker.  If publication
    crashes, the final directory remains visibly incomplete and is never
    deleted or overwritten by this transaction.
    """

    expected_stage = output.parent / f".{output.name}.staging"
    expected_files = {
        "adapter.safetensors",
        "optimizer.pt",
        "history.json",
        "receipt.json",
    }
    if stage != expected_stage or stage.parent != output.parent:
        raise IdentityOrbitTrainingError("run stage/output transaction paths differ")
    if output.exists() or output.is_symlink():
        raise IdentityOrbitTrainingError("create-only output appeared before publication")
    staged = {path.name: path for path in stage.iterdir()}
    if set(staged) != expected_files or any(
        path.is_symlink() or not path.is_file() for path in staged.values()
    ):
        raise IdentityOrbitTrainingError("run publish stage closure differs")
    try:
        output.mkdir(mode=0o750, exist_ok=False)
    except FileExistsError as error:
        raise IdentityOrbitTrainingError("create-only output already exists") from error
    runtime.fsync_directory(output.parent)
    try:
        for name in ("adapter.safetensors", "optimizer.pt", "history.json"):
            os.link(stage / name, output / name)
        # The receipt is the last link and is the only publication commit marker.
        os.link(stage / "receipt.json", output / "receipt.json")
        runtime.fsync_directory(output)
        runtime.fsync_directory(output.parent)
        published = {path.name: path for path in output.iterdir()}
        if set(published) != expected_files or any(
            path.is_symlink() or not path.is_file() for path in published.values()
        ):
            raise IdentityOrbitTrainingError("published run file closure differs")
        if any(
            runtime.file_sha256(published[name])
            != runtime.file_sha256(staged[name])
            for name in expected_files
        ):
            raise IdentityOrbitTrainingError("published run bytes differ from stage")
    except Exception:
        # Never remove a path after exclusive final-directory creation: another
        # observer may already have seen it.  Missing receipt.json is the
        # fail-closed recovery signal.
        raise
    for name in expected_files:
        (stage / name).unlink()
    stage.rmdir()
    runtime.fsync_directory(output.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expected-dataset-receipt-sha256", required=True)
    parser.add_argument("--expected-materialization-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", required=True, choices=("sealed-prefix-canary", "complete-cycle")
    )
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--expected-prefix-digest", default=None)
    parser.add_argument("--rho", type=float, choices=(0.0,), default=0.0)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-frames", type=int, choices=(FRAME_COUNT,), default=FRAME_COUNT)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--ack-pretext-not-action-editing", action="store_true")
    parser.add_argument(
        "--ack-incomplete-cycle-no-scientific-claim", action="store_true"
    )
    return parser


def validate_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        (WORLD_SIZE, SP_SIZE, DP_SIZE)
        != (runtime.WORLD_SIZE, runtime.SP_SIZE, runtime.DP_SIZE)
        or args.num_frames != FRAME_COUNT
        or args.rho != 0.0
    ):
        raise IdentityOrbitTrainingError("WORLD8 DP2xSP4/exact81/rho0 contract differs")
    if args.ack_pretext_not_action_editing is not True:
        raise IdentityOrbitTrainingError("--ack-pretext-not-action-editing is mandatory")
    if (
        isinstance(args.max_steps, bool)
        or not isinstance(args.max_steps, int)
        or args.max_steps <= 0
    ):
        raise IdentityOrbitTrainingError("max_steps must be a positive integer")
    prefix: Optional[Mapping[str, Any]] = None
    if args.mode == "sealed-prefix-canary":
        if not 1 <= args.max_steps < MICROBATCH_CYCLE_STEPS:
            raise IdentityOrbitTrainingError("prefix canary must contain 1..35 steps")
        if args.ack_incomplete_cycle_no_scientific_claim is not True:
            raise IdentityOrbitTrainingError(
                "incomplete prefix requires explicit no-scientific-claim acknowledgement"
            )
        prefix = prefix_seal_body(args.max_steps)
        if args.expected_prefix_digest != prefix["digest"]:
            raise IdentityOrbitTrainingError("sealed prefix digest differs")
    else:
        if args.max_steps % MICROBATCH_CYCLE_STEPS:
            raise IdentityOrbitTrainingError(
                "complete-cycle mode requires a positive multiple of 36 steps"
            )
        if args.expected_prefix_digest not in (None, ""):
            raise IdentityOrbitTrainingError("complete-cycle mode forbids a prefix digest")
        if args.ack_incomplete_cycle_no_scientific_claim:
            raise IdentityOrbitTrainingError("complete-cycle mode received prefix acknowledgement")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_dataset_receipt_sha256",
        "expected_materialization_spec_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise IdentityOrbitTrainingError("checkpoint tree differs from audited Bernini 1.3B")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise IdentityOrbitTrainingError("learning rate must be finite and positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise IdentityOrbitTrainingError("max grad norm must be finite and positive")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or not 0 <= args.seed < 2**63:
        raise IdentityOrbitTrainingError("seed must lie in [0,2^63)")
    return {
        "mode": args.mode,
        "max_steps": args.max_steps,
        "cycle_complete": args.max_steps % MICROBATCH_CYCLE_STEPS == 0,
        "sealed_prefix": prefix,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_contract = validate_cli(args)
    rv2v4_registry = _validate_rv2v4_registry()
    dataset = load_orbit_dataset(
        args.dataset_root,
        expected_receipt_sha256=args.expected_dataset_receipt_sha256,
        expected_spec_sha256=args.expected_materialization_spec_sha256,
    )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise IdentityOrbitTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise IdentityOrbitTrainingError("pinned Bernini attention-head count differs")
    dataset_vae_checkpoint_binding = validate_dataset_vae_against_checkpoint(
        dataset, checkpoint
    )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = runtime.distributed_contract()
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, distributed.rank, parallel.world_group
    )

    legacy.seed_same_sample(args.seed)
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.t5_text_encoder.eval()
    renderer.to(device)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise IdentityOrbitTrainingError("v4 requires only Bernini transformer_1")
    checkpointing_receipt = _disable_gradient_checkpointing(renderer, transformer)
    adapter = target_adapter.install_native_target_adapter(
        transformer,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        block_indices=target_adapter.DEFAULT_BLOCK_INDICES,
    )
    # Newly assigned wrappers default to ``training=True`` even when their
    # parent was already put in eval mode.  Re-apply eval recursively; LoRA
    # parameters remain trainable, while every replay is dropout-free.
    renderer.eval()
    if any(
        wrapper.training
        for _, wrapper in (*adapter.q_wrappers, *adapter.o_wrappers)
    ):
        raise IdentityOrbitTrainingError("native target adapter is not in eval mode")
    trainable = adapter.trainable_named_parameters()
    if not adapter.base_parameters_frozen():
        raise IdentityOrbitTrainingError("base parameters changed trainability")
    initial_digest = runtime.synchronize_initial_parameters(
        trainable, parallel.world_group
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    conditional, unconditional, text_digest = _frozen_text_embeddings(
        renderer, tokenizer, device
    )
    # Frozen T5 kernels are deterministic in intent, but establish one exact
    # authoritative embedding byte sequence before any replicated SP/DP
    # forward rather than relying on device-local low-bit equality.
    dist.broadcast(conditional, src=0, group=parallel.world_group)
    dist.broadcast(unconditional, src=0, group=parallel.world_group)
    text_digest = object_sha256(
        {
            "instruction": GENERIC_INSTRUCTION,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "conditional": runtime.tensor_sha256(conditional),
            "unconditional": runtime.tensor_sha256(unconditional),
        }
    )
    runtime.digest_consensus(
        text_digest,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="identity orbit frozen prompts",
    )
    renderer.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()
    vae_mean, vae_std, _ = legacy._vae_statistics(checkpoint)

    cycle = orbit_v4.registered_orbit_microbatch_cycle()
    cycle_receipt = orbit_v4.orbit_microbatch_cycle_receipt()
    rho_schedule = orbit_v4.SourceRichRhoSchedule(max_rho=0.0)
    history: list[Mapping[str, Any]] = []
    orbit_cache: dict[int, orbit_v4.IdentityOrbit] = {}
    broadcast_receipts: dict[int, Mapping[str, Any]] = {}

    def get_orbit(row_index: int) -> orbit_v4.IdentityOrbit:
        if row_index not in orbit_cache:
            value = build_identity_orbit_from_row(
                dataset.rows[row_index],
                vae_mean=vae_mean,
                vae_std=vae_std,
                device=device,
            )
            broadcast_receipts[row_index] = broadcast_orbit_within_sp(
                value, parallel=parallel
            )
            orbit_cache[row_index] = value
        return orbit_cache[row_index]

    initial_row_index = distributed.arm_index % len(dataset.rows)
    wrong_row_index = (initial_row_index + 1) % len(dataset.rows)
    initial_orbit = get_orbit(initial_row_index)
    wrong_orbit = get_orbit(wrong_row_index)
    diagnostic_epsilon = _sample_epsilon(
        initial_orbit.members[0].video_latent.shape,
        seed=_diagnostic_noise_seed(args.seed, initial_row_index, distributed.arm_index),
        device=device,
    )
    pre_metrics, _, _, _ = _causal_snapshot(
        diffusion=diffusion,
        transformer=transformer,
        orbit=initial_orbit,
        wrong_orbit=wrong_orbit,
        epsilon=diagnostic_epsilon,
        cond_embeds=conditional,
        uncond_embeds=unconditional,
        adapter=adapter,
        sp_rank=distributed.sp_rank,
    )

    for global_step in range(args.max_steps):
        cycle_index, ordinal = divmod(global_step, MICROBATCH_CYCLE_STEPS)
        microbatch = cycle[ordinal]
        row_index = (cycle_index * DP_SIZE + distributed.arm_index) % len(dataset.rows)
        current_orbit = get_orbit(row_index)
        cells = {
            cell.key: cell for cell in orbit_v4.build_identity_orbit_cells(current_orbit)
        }
        sigma_indices = native.schedule_indices_for_step(
            seed=args.seed,
            step=global_step,
            samples_per_step=SIGMAS_PER_MICROBATCH,
        )
        noise_seed = _noise_seed(
            args.seed, global_step, row_index, distributed.arm_index
        )
        epsilon = _sample_epsilon(
            current_orbit.members[0].video_latent.shape,
            seed=noise_seed,
            device=device,
        )
        supervision: dict[orbit_v4.OrbitCellKey, orbit_v4.SourceRichStates] = {}
        detached_predictions: dict[orbit_v4.OrbitCellKey, Any] = {}
        for key in microbatch.keys:
            cell = cells[key]
            states = orbit_v4.build_orbit_cell_states(
                cell,
                epsilon,
                indices=sigma_indices,
                rho_schedule=rho_schedule,
            )
            per_sigma: list[Any] = []
            for sigma_position in range(SIGMAS_PER_MICROBATCH):
                pack = _build_pack(
                    transformer, cell, states, sigma_position=sigma_position
                )
                per_sigma.append(
                    _guided_prediction_no_grad(
                        diffusion,
                        pack,
                        timestep=states.timesteps[
                            sigma_position : sigma_position + 1
                        ],
                        cond_embeds=conditional,
                        uncond_embeds=unconditional,
                        adapter=adapter,
                        sp_rank=distributed.sp_rank,
                        video_shape=tuple(cell.target.shape),
                    )
                )
            supervision[key] = states
            detached_predictions[key] = (
                torch.stack(per_sigma, dim=0).detach().requires_grad_(True)
            )

        optimizer.zero_grad(set_to_none=True)
        objective = orbit_v4.identity_orbit_microbatch_objective(
            microbatch,
            detached_predictions,
            supervision,
            current_orbit,
        )
        if not runtime.world_all_true(
            bool(torch.isfinite(objective.loss.detach()).item()),
            group=parallel.world_group,
        ):
            raise IdentityOrbitTrainingError("non-finite v4 objective blocked update")
        # This backward computes dL/d(guided prediction) only.  Adapter
        # parameters are absent from this small leaf graph.
        objective.loss.backward()
        replay_max_abs = 0.0
        for key in microbatch.keys:
            prediction_leaf = detached_predictions[key]
            if prediction_leaf.grad is None:
                raise IdentityOrbitTrainingError("objective leaf has no output cotangent")
            for sigma_position in range(SIGMAS_PER_MICROBATCH):
                replay_max_abs = max(
                    replay_max_abs,
                    _replay_prediction_vjp(
                        diffusion,
                        transformer,
                        cells[key],
                        supervision[key],
                        sigma_position=sigma_position,
                        output_cotangent=prediction_leaf.grad[sigma_position].detach(),
                        expected_guided=prediction_leaf[sigma_position].detach(),
                        cond_embeds=conditional,
                        uncond_embeds=unconditional,
                        adapter=adapter,
                        sp_rank=distributed.sp_rank,
                    ),
                )
        preclip_norm = runtime.synchronize_gradients(trainable, parallel)
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)):
            raise IdentityOrbitTrainingError("gradient clipping is non-finite")
        optimizer.step()
        parameter_digest = runtime.parameter_consensus(
            trainable,
            parallel.world_group,
            f"identity orbit adapter step {global_step + 1}",
        )
        local_record = {
            "step": global_step + 1,
            "cycle_index": cycle_index,
            "cycle_ordinal": ordinal,
            "microbatch_digest": microbatch.receipt()["digest"],
            "step_type": microbatch.step_type,
            "cell_keys": [orbit_v4.orbit_cell_key_id(key) for key in microbatch.keys],
            "dp_rank": distributed.arm_index,
            "sp_rank": distributed.sp_rank,
            "row_index": row_index,
            "iid": dataset.rows[row_index].iid,
            "noise_seed": noise_seed,
            "sigma_indices": list(sigma_indices),
            "timesteps": [native.NATIVE_UNIPC40_TIMESTEPS[index] for index in sigma_indices],
            "loss": float(objective.loss.detach().item()),
            "reconstruction_cycle_contribution": float(
                objective.reconstruction_cycle_contribution.detach().item()
            ),
            "factor_cycle_contribution": float(
                objective.factor_cycle_contribution.detach().item()
            ),
            "raw_factor_value": float(objective.raw_factor_value.detach().item()),
            "preclip_gradient_norm_world_average": preclip_norm,
            "vjp_replay_max_abs": replay_max_abs,
            "parameter_digest": parameter_digest,
        }
        sp_projection = {
            key: value for key, value in local_record.items() if key != "sp_rank"
        }
        runtime.digest_consensus(
            object_sha256(sp_projection),
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"identity orbit step {global_step + 1} SP record",
        )
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_record, group=parallel.world_group)
        history.append(
            {
                "step": global_step + 1,
                "dp_records": [gathered[0], gathered[4]],
            }
        )
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step + 1,
                        "cycle_ordinal": ordinal,
                        "loss_dp0": gathered[0]["loss"],
                        "loss_dp1": gathered[4]["loss"],
                        "preclip_gradient_norm": preclip_norm,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_digest = runtime.parameter_consensus(
        trainable, parallel.world_group, "identity orbit final adapter"
    )
    if final_digest == initial_digest:
        raise IdentityOrbitTrainingError(
            "optimizer run completed without changing the native target adapter"
        )
    post_metrics, post_correct, post_wrong, post_target = _causal_snapshot(
        diffusion=diffusion,
        transformer=transformer,
        orbit=initial_orbit,
        wrong_orbit=wrong_orbit,
        epsilon=diagnostic_epsilon,
        cond_embeds=conditional,
        uncond_embeds=unconditional,
        adapter=adapter,
        sp_rank=distributed.sp_rank,
    )
    maximum_correct = (
        float(pre_metrics["correct_target_error"]) * MAX_CORRECT_ERROR_DEGRADATION
        + 1.0e-12
    )
    minimum_sensitivity = max(
        float(pre_metrics["wrong_scene_output_sensitivity"])
        * MIN_WRONG_SCENE_SENSITIVITY_RETENTION,
        MIN_NONZERO_SENSITIVITY,
    )
    post_gate = orbit_v4.heldout_wrong_scene_gate(
        correct_prediction_clean=post_correct,
        wrong_scene_prediction_clean=post_wrong,
        exact_orbit_target=post_target,
        maximum_correct_error=maximum_correct,
        minimum_wrong_scene_sensitivity=minimum_sensitivity,
    )
    local_gate = {
        "dp_rank": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "iid": dataset.rows[initial_row_index].iid,
        "wrong_scene_iid": dataset.rows[wrong_row_index].iid,
        "pre": dict(pre_metrics),
        "post": dict(post_metrics),
        "post_gate": dict(post_gate),
    }
    gate_projection = {key: value for key, value in local_gate.items() if key != "sp_rank"}
    runtime.digest_consensus(
        object_sha256(gate_projection),
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label="pre/post causal gate",
    )
    gathered_gates: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_gates, local_gate, group=parallel.world_group)
    causal_gates = [gathered_gates[0], gathered_gates[4]]
    gathered_broadcasts: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_broadcasts,
        {
            "dp_rank": distributed.arm_index,
            "rows": {
                str(index): dict(value)
                for index, value in sorted(broadcast_receipts.items())
            },
        },
        group=parallel.world_group,
    )
    broadcast_receipts_by_dp = [gathered_broadcasts[0], gathered_broadcasts[4]]
    all_causal_gates_passed = all(
        item["post_gate"]["accepted"] is True for item in causal_gates
    )
    if not all_causal_gates_passed:
        raise IdentityOrbitTrainingError(
            "post-training wrong-scene causal gate failed on at least one DP arm"
        )
    dataset.assert_unchanged()

    dist.barrier(group=parallel.world_group)
    if distributed.rank == 0:
        adapter_path = stage / "adapter.safetensors"
        optimizer_path = stage / "optimizer.pt"
        history_path = stage / "history.json"
        roundtrip = _save_and_roundtrip_adapter(adapter_path, adapter)
        runtime.atomic_torch_save(
            optimizer_path,
            {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "optimizer": optimizer.state_dict(),
                "global_step": args.max_steps,
                "adapter_parameter_digest": final_digest,
                "cycle_digest": cycle_receipt["digest"],
            },
        )
        history_value = {
            "schema_version": HISTORY_SCHEMA,
            "step_count": args.max_steps,
            "fixed_cycle_digest": cycle_receipt["digest"],
            "dynamic_cell_selection": False,
            "records": history,
        }
        runtime.atomic_json(history_path, history_value)
        receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "mode": args.mode,
            "optimizer_steps": args.max_steps,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "latent_phases": LATENT_PHASES,
            "rho": 0.0,
            "rho0_original_gaussian_values": True,
            "source_rich_positive_rho_trained": False,
            "run_contract": dict(run_contract),
            "dataset": {
                "root": str(dataset.root),
                "parquet": dict(dataset.parquet_snapshot.receipt()),
                "receipt": dict(dataset.receipt_snapshot.receipt()),
                "receipt_digest": dataset.receipt_digest,
                "materialization_spec_sha256": dataset.spec_sha256,
                "reference_encoding_contract": dict(
                    rv2v4_registry["reference_encoding"]
                ),
                "rows": len(dataset.rows),
                "iids": [row.iid for row in dataset.rows],
                "row_digests": [row.row_digest for row in dataset.rows],
                "v4_member_aliases": dict(appearance.V4_MEMBER_ALIASES),
                "variant_native_arms_by_iid": {
                    iid: dict(arms)
                    for iid, arms in dataset.native_arms_by_iid.items()
                },
                "appearance_factor_digests": {
                    row.iid: row.appearance_factor_digest for row in dataset.rows
                },
                "live_provenance_audit_digest": (
                    dataset.live_provenance_audit_digest
                ),
                "live_source_variant_native_receipt_and_qualification_files_revalidated": True,
                "all_rows_externally_qualified": True,
                "post_training_input_mutation_audit_passed": True,
                "offline_vae_matches_training_checkpoint": dict(
                    dataset_vae_checkpoint_binding
                ),
                "action_edit_target_present": False,
                "mask_flow_pose_track_box_trajectory_used": False,
            },
            "objective": {
                "schema": orbit_v4.SCHEMA_VERSION,
                "fixed_microbatch_cycle": dict(cycle_receipt),
                "dynamic_cell_selection": False,
                "sigmas_per_microbatch": SIGMAS_PER_MICROBATCH,
                "exact40_schedule": dict(native.native_unipc40_schedule_receipt()),
                "adapter_activation_schedule_train_and_inference": (
                    "all_40_native_unipc_forward_coordinates"
                ),
                "same_epsilon_across_cells_within_microbatch": True,
                "native_rv2v4_visual_conditioning": dict(
                    rv2v4_registry["native_visual_pack"]
                ),
                "native_rv2v_guidance": dict(guidance.guidance_receipt()),
                "vjp_replay": {
                    "method": "no_grad_guided_leaf_then_serial_exact_linear_vjp",
                    "coefficients": {
                        "none_uncond": -0.25,
                        "V_uncond": -3.25,
                        "VI_uncond": 0.5,
                        "VI_cond": 4.0,
                    },
                    "one_transformer_graph_resident_at_a_time": True,
                    "dropout": 0.0,
                    "base_eval_mode": True,
                    "per_step_replay_max_abs": [
                        max(item["vjp_replay_max_abs"] for item in record["dp_records"])
                        for record in history
                    ],
                },
            },
            "adapter": {
                **dict(adapter.receipt()),
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "scope": "blocks0-22 attn1 target-row Q/O only",
                "initial_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
                "changed_by_training": final_digest != initial_digest,
                "strict_real_file_roundtrip": dict(roundtrip),
            },
            "gradient_checkpointing": {
                **dict(checkpointing_receipt),
                "renderer_enabled": False,
                "transformer_enabled": False,
            },
            "causal_gates": {
                "split": "heldout_wrong_scene_only_not_in_objective",
                "heldout_intervention_not_necessarily_heldout_iid": True,
                "wrong_scene_iids_may_be_other_training_cohort_rows": True,
                "cross_iid_generalization_claim_authorized": False,
                "diagnostic_sigma_index": DIAGNOSTIC_SIGMA_INDEX,
                "maximum_correct_error_degradation_factor": MAX_CORRECT_ERROR_DEGRADATION,
                "minimum_wrong_scene_sensitivity_retention_factor": MIN_WRONG_SCENE_SENSITIVITY_RETENTION,
                "minimum_nonzero_sensitivity": MIN_NONZERO_SENSITIVITY,
                "dp_records": causal_gates,
                "all_dp_gates_passed": all_causal_gates_passed,
                "used_to_change_training_cells": False,
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "data_parallel_size": DP_SIZE,
                "ulysses_sequence_parallel_size": SP_SIZE,
                "sp_groups": [list(item) for item in runtime.SP_GROUP_RANKS],
                "dp_groups": [list(item) for item in runtime.DP_GROUP_RANKS],
                "all_eight_gpus_used": True,
                "gradient_sync": [
                    "SP4_all_reduce_sum_then_divide_by_4",
                    "DP2_all_reduce_sum_then_divide_by_2",
                ],
                "orbit_tensor_broadcasts_by_dp": broadcast_receipts_by_dp,
            },
            "optimizer": {
                "type": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": 0.0,
                "max_gradient_norm": args.max_grad_norm,
            },
            "prompts": {
                "generic_instruction": GENERIC_INSTRUCTION,
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                "embedding_digest": text_digest,
                "target_caption_or_embedding_conditioning": False,
            },
            "runtime": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "single_expert": "transformer_1",
            },
            "artifacts": {
                "adapter.safetensors": runtime.file_sha256(adapter_path),
                "optimizer.pt": runtime.file_sha256(optimizer_path),
                "history.json": runtime.file_sha256(history_path),
            },
            "pretext_training_only": True,
            "appearance_motion_role_learning_claim_only": True,
            "semantic_action_learned": False,
            "action_editing_claim_authorized": False,
            "video_quality_claim_authorized": False,
            # Even a complete orbit cycle and both causal gates are only a
            # go/no-go signal for a fresh held-out composition experiment.
            # They are not an independent scientific result and cannot be
            # promoted into an action-editing claim by this trainer.
            "next_heldout_role_composition_experiment_authorized": bool(
                run_contract["cycle_complete"] and all_causal_gates_passed
            ),
            "scientific_claim_authorized": False,
            "scientific_claim_scope": "none",
            "long_training_automatically_submitted": False,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        runtime.atomic_json(stage / "receipt.json", receipt)
        runtime.verify_staged_run_bundle(stage, receipt)
        runtime.fsync_directory(stage)
        _publish_create_only_run_bundle(stage, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "optimizer_steps": args.max_steps,
                    "adapter_parameter_digest": final_digest,
                    "all_causal_gates_passed": all_causal_gates_passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    if not output.is_dir() or output.is_symlink() or stage.exists():
        raise IdentityOrbitTrainingError("atomic output publication did not complete")
    adapter.restore()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_SCHEMA",
    "DEFAULT_NEGATIVE_PROMPT",
    "GENERIC_INSTRUCTION",
    "HISTORY_SCHEMA",
    "IdentityOrbitTrainingError",
    "MICROBATCH_CYCLE_STEPS",
    "RUN_RECEIPT_SCHEMA",
    "SIGMAS_PER_MICROBATCH",
    "build_identity_orbit_from_row",
    "build_parser",
    "load_orbit_dataset",
    "main",
    "native_rv2v_vjp_branches",
    "prefix_seal_body",
    "unpack_native_target_tokens",
    "validate_cli",
]
