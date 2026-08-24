"""Hash-bound source-only manifest for the OASIS Phase-A noise bank.

The manifest closes four source cells (dog/human x fit/confirmation), two
registered seeds, three initial-noise arms, and the complete action/negative
caption registry used by the separate T2V calibration.  It carries no paired
target, proposal media, mask, flow, pose, trajectory, or optimizer authority.

The optional scalar-calibration binding is metadata-only.  Its loader accepts
only a separately sealed v4 mainline authorization and never exports T2V
media, latents, or Gaussians to the native RV2V candidate generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-oasis-phase-a-frozen-oracle-manifest-v2"
SAMPLE_SCHEMA = "bernini-oasis-phase-a-source-cell-v2"
WEAK_DECOY_SCHEMA = "bernini-oasis-weak-wrongref-diagnostic-v1"
SCALAR_CALIBRATION_BINDING_SCHEMA = (
    "bernini-oasis-t2v-scalar-calibration-binding-v1"
)
SCALAR_CALIBRATION_EVIDENCE_SCHEMA = (
    "bernini-oasis-t2v-scalar-calibration-evidence-v1"
)

FRAME_COUNT = 81
FPS = 25.0
SAMPLER_STEPS = 40
REFERENCE_INDICES = (0, 27, 53, 80)
FAMILY_ORDER = ("dog_sit_hold", "human_stand_hold")
SPLIT_ORDER = ("fit", "confirmation")
ARM_ORDER = (
    "official_gaussian",
    "source_appearance_set_rho005",
    "source_appearance_set_rho010",
)
BRANCH_ORDER = (
    "action",
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)

# The allocation is one eight-GPU node, but the runtime consists of two
# independent WORLD4/Ulysses-SP4 groups rather than one WORLD8 process group.
TOPOLOGY = {
    "allocation_gpu_count": 8,
    "allocation_gpu_type": "MI210",
    "process_group_count": 2,
    "process_group_world_size": 4,
    "ulysses_sequence_parallel_size": 4,
    "groups_run_concurrently": True,
    "family_by_process_group": {
        "0": "dog_sit_hold",
        "1": "human_stand_hold",
    },
    "physical_gpu_ordinals_by_process_group": {
        "0": [0, 1, 2, 3],
        "1": [4, 5, 6, 7],
    },
}

INFORMATION_FLOW = {
    "external_inference_inputs": ["source_video", "complete_action_caption"],
    "source_derived_internal_inputs": [
        "full_source_RV2V_condition",
        "four_independently_encoded_source_T1_frames_as_unordered_set",
        "native_seed",
    ],
    "paired_target_video_or_latent": False,
    "t2v_media_or_latent_to_rv2v": False,
    "mask_flow_pose_track_or_trajectory": False,
    "proposal_media_latent_or_motion_donor": False,
    "wrongref_proxy_used_for_authorization": False,
    "source_frame_order_consumed_by_noise_operator": False,
    "source_temporal_phase_consumed_by_noise_operator": False,
    "scalar_calibration_media_consumed_by_candidate_runtime": False,
    "confirmation_consumed_by_optimizer": False,
    "phase_a_training_performed": False,
    "endpoint_selection_performed": False,
    "optimizer_authorized": False,
    "rho_zero_exact_native_gaussian_control": True,
    "active_rho_external_initial_noise_injection": True,
    "scientific_action_editing_success_claim": False,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_tree_sha256",
        "t2v_scalar_calibration",
        "frame_count",
        "fps",
        "sampler_steps",
        "reference_indices",
        "seed_order",
        "arm_order",
        "topology",
        "sample_count",
        "samples",
        "information_flow",
        "manifest_digest",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "family",
        "analysis_split",
        "source_video_path",
        "source_video_sha256",
        "source_caption",
        "complete_action_caption",
        "actor_binding",
        "raw_caption_by_branch",
        "raw_caption_bank_sha256",
        "calibration_candidate_id",
        "calibration_event_receipt_digest",
        "weak_wrongref_diagnostic",
        "sample_digest",
    }
)
_DECOY_FIELDS = frozenset(
    {
        "schema_version",
        "available",
        "path",
        "sha256",
        "proxy_kind",
        "known_confounds",
        "identity_only_claim",
        "used_for_authorization",
    }
)
_SCALAR_BINDING_FIELDS = frozenset(
    {"schema_version", "status", "path", "file_sha256", "evidence_digest"}
)
_SCALAR_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_tree_sha256",
        "family_order",
        "formal_scalar_source",
        "family_gates",
        "scalar_calibration_only",
        "source_media_consumed_by_oasis_runtime",
        "source_latent_or_gaussian_consumed_by_oasis_runtime",
        "source_media_or_latent_used_as_teacher",
        "optimizer_authorized",
        "frozen_controller_prompt_family_qualified",
        "scientific_action_editing_success_claim",
        "evidence_digest",
    }
)
_FORMAL_SCALAR_SOURCE_FIELDS = frozenset(
    {
        "validator_id",
        "path",
        "file_sha256",
        "authorization_digest",
        "formal_validator_recomputed",
    }
)
_FAMILY_SCALAR_GATE_FIELDS = frozenset(
    {
        "family",
        "formal_action_family_id",
        "fit_candidate_ids",
        "confirmation_candidate_ids",
        "branch_order",
        "score_definition",
        "minimum_robust_action_log_ratio",
        "minimum_action_log_ratio_by_negative",
        "fit_event_and_scalar_gate_passed",
        "confirmation_event_and_scalar_gate_passed",
        "gate_digest",
    }
)
_FORMAL_MAINLINE_AUTHORIZATION_SCHEMA = (
    "bernini-pair-v5-t2v-calibration-mainline-authorization-v4"
)
_FORMAL_MAINLINE_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "source_bank_spec_sha256",
        "source_bank_receipt_digest",
        "formal_score_provenance_set_digest",
        "formal_score_schema",
        "formal_score_filename",
        "formal_score_scalar_definition",
        "formal_score_arithmetic_contract_digest",
        "preregistration_digest",
        "calibration_receipt_digest",
        "family_mapping_set_digest",
        "checkpoint_tree_sha256",
        "score_count",
        "branch_order",
        "action_family_order",
        "all_formal_scalar_provenance_recomputed",
        "formal_receipts_validated_by_active_v4_canonical_code",
        "active_repository_score_schema_consumed",
        "legacy_v3_compatibility_score_consumed",
        "initialization_ablation_teacher_or_adapter_artifact_consumed",
        "t2v_media_latent_gaussian_or_proposal_exported_to_native_scorer",
        "only_family_maps_threshold_prompts_and_scalar_digests_exported",
        "calibration_maps_authorized",
        "native_rv2v_optimizer_authorized",
        "scientific_action_editing_claim",
        "authorization_digest",
    }
)
_FORMAL_FAMILY_ID = {
    "dog_sit_hold": "dog-sit-facing-camera",
    "human_stand_hold": "human-rise-to-stand",
}


class OASISManifestError(RuntimeError):
    """The Phase-A source/evidence plan is not closed or hash-bound."""


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
        raise OASISManifestError(
            "manifest is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise OASISManifestError(
            "instruction text must be non-empty UTF-8 without NUL"
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_instruction_binding_digest(
    *, source_video_sha256: str, edit_instruction_sha256: str
) -> str:
    return object_sha256(
        {
            "source_video_sha256": _sha(
                source_video_sha256, label="source video SHA"
            ),
            "edit_instruction_sha256": _sha(
                edit_instruction_sha256, label="edit instruction SHA"
            ),
        }
    )


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OASISManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("ascii")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OASISManifestError(f"{label} contains {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OASISManifestError(f"{label} is not strict ASCII JSON") from error


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OASISManifestError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise OASISManifestError(
            f"{label} is outside the closed identifier grammar"
        )
    return value


def _plain_absolute_file(value: Any, *, label: str, verify: bool) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise OASISManifestError(f"{label} must be an absolute path")
    path = Path(value)
    if verify and (not path.is_file() or path.is_symlink()):
        raise OASISManifestError(f"{label} must be an existing plain file")
    return path.resolve(strict=verify)


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise OASISManifestError(f"{label} must be non-empty text without NUL")
    return value


@dataclass(frozen=True)
class WeakWrongRefDiagnostic:
    available: bool
    path: Optional[Path]
    sha256: Optional[str]
    proxy_kind: str
    known_confounds: tuple[str, ...]


@dataclass(frozen=True)
class OASISSourceCell:
    sample_id: str
    family: str
    analysis_split: str
    source_video: Path
    source_video_sha256: str
    source_caption: str
    complete_action_caption: str
    actor_binding: str
    raw_caption_by_branch: Mapping[str, str]
    raw_caption_bank_sha256: str
    calibration_candidate_id: str
    calibration_event_receipt_digest: str
    weak_wrongref_diagnostic: WeakWrongRefDiagnostic
    sample_digest: str

    # Compatibility with the dedicated source-set runner's minimal naming.
    # These are derived from sealed rich fields rather than separately trusted.
    @property
    def edit_instruction(self) -> str:
        return self.complete_action_caption

    @property
    def edit_instruction_sha256(self) -> str:
        return text_sha256(self.complete_action_caption)

    @property
    def source_instruction_binding_digest(self) -> str:
        return source_instruction_binding_digest(
            source_video_sha256=self.source_video_sha256,
            edit_instruction_sha256=self.edit_instruction_sha256,
        )


@dataclass(frozen=True)
class OASISPhaseAManifest:
    path: Path
    file_sha256: str
    checkpoint_tree_sha256: str
    scalar_calibration_status: str
    scalar_calibration_path: Optional[Path]
    scalar_calibration_file_sha256: Optional[str]
    scalar_calibration_evidence_digest: Optional[str]
    seed_order: tuple[int, int]
    samples: tuple[OASISSourceCell, ...]
    manifest_digest: str

    def cells_for_family(self, family: str) -> tuple[OASISSourceCell, ...]:
        if family not in FAMILY_ORDER:
            raise OASISManifestError(
                "family is outside Phase-A preregistration"
            )
        return tuple(cell for cell in self.samples if cell.family == family)

    def assert_unchanged(self) -> None:
        if file_sha256(self.path) != self.file_sha256:
            raise OASISManifestError("Phase-A manifest changed after preflight")
        if self.scalar_calibration_status == "resolved":
            if (
                self.scalar_calibration_path is None
                or self.scalar_calibration_file_sha256 is None
                or file_sha256(self.scalar_calibration_path)
                != self.scalar_calibration_file_sha256
            ):
                raise OASISManifestError(
                    "dedicated scalar-calibration evidence changed after preflight"
                )
        for cell in self.samples:
            if file_sha256(cell.source_video) != cell.source_video_sha256:
                raise OASISManifestError(f"source video changed: {cell.sample_id}")
            decoy = cell.weak_wrongref_diagnostic
            if decoy.available and decoy.path is not None and decoy.sha256 is not None:
                if file_sha256(decoy.path) != decoy.sha256:
                    raise OASISManifestError(
                        f"weak decoy changed: {cell.sample_id}"
                    )


def _validate_prompt_bank(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(BRANCH_ORDER):
        raise OASISManifestError(
            "raw caption bank must close action plus all nine negatives"
        )
    result = {
        branch: _text(value[branch], label=f"raw caption {branch}")
        for branch in BRANCH_ORDER
    }
    if len(set(result.values())) != len(result):
        raise OASISManifestError(
            "raw caption branches must be distinct strings"
        )
    return result


def _validate_weak_decoy(
    value: Any, *, verify_files: bool, label: str
) -> WeakWrongRefDiagnostic:
    if not isinstance(value, Mapping) or set(value) != set(_DECOY_FIELDS):
        raise OASISManifestError(f"{label} weak decoy field closure differs")
    if value.get("schema_version") != WEAK_DECOY_SCHEMA:
        raise OASISManifestError(f"{label} weak decoy schema differs")
    if (
        value.get("identity_only_claim") is not False
        or value.get("used_for_authorization") is not False
    ):
        raise OASISManifestError(
            f"{label} wrong-ref proxy cannot authorize identity"
        )
    available = value.get("available")
    if type(available) is not bool:
        raise OASISManifestError(
            f"{label} weak decoy availability must be bool"
        )
    proxy_kind = value.get("proxy_kind")
    allowed = {
        "none",
        "same_class_confounded_background_scale_species",
        "same_class_center_crop_geometry_proxy",
        "multi_decoy_robust_diagnostic",
    }
    if proxy_kind not in allowed:
        raise OASISManifestError(f"{label} weak decoy kind differs")
    confounds = value.get("known_confounds")
    if not isinstance(confounds, list) or any(
        not isinstance(item, str) or not item for item in confounds
    ):
        raise OASISManifestError(
            f"{label} weak decoy confounds must be a string list"
        )
    if available:
        path = _plain_absolute_file(
            value.get("path"), label=f"{label} weak decoy", verify=verify_files
        )
        digest = _sha(value.get("sha256"), label=f"{label} weak decoy SHA")
        if not confounds or proxy_kind == "none":
            raise OASISManifestError(
                f"{label} available decoy must disclose confounds"
            )
        if verify_files and file_sha256(path) != digest:
            raise OASISManifestError(f"{label} weak decoy SHA differs")
    else:
        if value.get("path") is not None or value.get("sha256") is not None:
            raise OASISManifestError(
                f"{label} unavailable decoy must have null path/hash"
            )
        if proxy_kind != "none":
            raise OASISManifestError(
                f"{label} unavailable decoy kind must be none"
            )
        path = None
        digest = None
    return WeakWrongRefDiagnostic(
        available=available,
        path=path,
        sha256=digest,
        proxy_kind=str(proxy_kind),
        known_confounds=tuple(confounds),
    )


def _validate_sample(
    value: Any, *, verify_files: bool, ordinal: int
) -> OASISSourceCell:
    label = f"sample[{ordinal}]"
    if not isinstance(value, Mapping) or set(value) != set(_SAMPLE_FIELDS):
        raise OASISManifestError(f"{label} field closure differs")
    row = dict(value)
    declared = _sha(row.pop("sample_digest"), label=f"{label} digest")
    if object_sha256(row) != declared or row.get("schema_version") != SAMPLE_SCHEMA:
        raise OASISManifestError(f"{label} digest/schema differs")
    sample_id = _safe_id(row.get("sample_id"), label=f"{label} sample ID")
    family = row.get("family")
    split = row.get("analysis_split")
    if family not in FAMILY_ORDER or split not in SPLIT_ORDER:
        raise OASISManifestError(f"{label} family/split differs")
    source = _plain_absolute_file(
        row.get("source_video_path"),
        label=f"{label} source video",
        verify=verify_files,
    )
    source_sha = _sha(row.get("source_video_sha256"), label=f"{label} source SHA")
    if verify_files and file_sha256(source) != source_sha:
        raise OASISManifestError(f"{label} source video SHA differs")
    source_caption = _text(row.get("source_caption"), label=f"{label} source caption")
    complete_action_caption = _text(
        row.get("complete_action_caption"),
        label=f"{label} complete action caption",
    )
    actor_binding = _text(
        row.get("actor_binding"), label=f"{label} actor binding"
    )
    captions = _validate_prompt_bank(row.get("raw_caption_by_branch"))
    caption_digest = _sha(
        row.get("raw_caption_bank_sha256"),
        label=f"{label} caption bank SHA",
    )
    if object_sha256(captions) != caption_digest:
        raise OASISManifestError(f"{label} caption bank digest differs")
    calibration_candidate = _safe_id(
        row.get("calibration_candidate_id"),
        label=f"{label} calibration candidate",
    )
    event_digest = _sha(
        row.get("calibration_event_receipt_digest"),
        label=f"{label} calibration event receipt",
    )
    decoy = _validate_weak_decoy(
        row.get("weak_wrongref_diagnostic"),
        verify_files=verify_files,
        label=label,
    )
    return OASISSourceCell(
        sample_id=sample_id,
        family=str(family),
        analysis_split=str(split),
        source_video=source,
        source_video_sha256=source_sha,
        source_caption=source_caption,
        complete_action_caption=complete_action_caption,
        actor_binding=actor_binding,
        raw_caption_by_branch=captions,
        raw_caption_bank_sha256=caption_digest,
        calibration_candidate_id=calibration_candidate,
        calibration_event_receipt_digest=event_digest,
        weak_wrongref_diagnostic=decoy,
        sample_digest=declared,
    )


def _validate_scalar_calibration_binding(
    value: Any, *, verify_files: bool
) -> tuple[str, Optional[Path], Optional[str], Optional[str]]:
    if not isinstance(value, Mapping) or set(value) != set(_SCALAR_BINDING_FIELDS):
        raise OASISManifestError(
            "T2V scalar-calibration binding field closure differs"
        )
    if value.get("schema_version") != SCALAR_CALIBRATION_BINDING_SCHEMA:
        raise OASISManifestError("T2V scalar-calibration binding schema differs")
    status = value.get("status")
    if status not in {"unresolved", "resolved"}:
        raise OASISManifestError(
            "scalar-calibration status must be unresolved or resolved"
        )
    if status == "unresolved":
        if any(
            value.get(name) is not None
            for name in ("path", "file_sha256", "evidence_digest")
        ):
            raise OASISManifestError(
                "unresolved scalar calibration must have null path/hash/digest"
            )
        return status, None, None, None
    path = _plain_absolute_file(
        value.get("path"),
        label="dedicated scalar-calibration evidence",
        verify=verify_files,
    )
    file_digest = _sha(
        value.get("file_sha256"), label="scalar-calibration evidence file SHA"
    )
    evidence_digest = _sha(
        value.get("evidence_digest"), label="scalar-calibration evidence digest"
    )
    if verify_files and file_sha256(path) != file_digest:
        raise OASISManifestError("scalar-calibration evidence file SHA differs")
    return status, path, file_digest, evidence_digest


def validate_manifest_mapping(
    value: Any,
    *,
    manifest_path: Path,
    manifest_file_sha256: str,
    verify_files: bool,
) -> OASISPhaseAManifest:
    if not isinstance(value, Mapping) or set(value) != set(_ROOT_FIELDS):
        raise OASISManifestError("Phase-A manifest root field closure differs")
    row = dict(value)
    declared = _sha(row.pop("manifest_digest"), label="manifest digest")
    if object_sha256(row) != declared or row.get("schema_version") != SCHEMA_VERSION:
        raise OASISManifestError("Phase-A manifest digest/schema differs")
    if (
        row.get("frame_count") != FRAME_COUNT
        or float(row.get("fps")) != FPS
        or row.get("sampler_steps") != SAMPLER_STEPS
        or row.get("reference_indices") != list(REFERENCE_INDICES)
        or row.get("arm_order") != list(ARM_ORDER)
        or row.get("topology") != TOPOLOGY
        or row.get("information_flow") != INFORMATION_FLOW
    ):
        raise OASISManifestError(
            "exact81/exact40/topology/information-flow contract differs"
        )
    checkpoint_tree = _sha(
        row.get("checkpoint_tree_sha256"), label="checkpoint tree"
    )
    seeds = row.get("seed_order")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or any(type(item) is not int or not 0 <= item < 2**63 for item in seeds)
    ):
        raise OASISManifestError(
            "Phase-A requires exactly two unique seeds in [0,2^63)"
        )
    (
        scalar_status,
        scalar_path,
        scalar_file_sha,
        scalar_evidence_digest,
    ) = _validate_scalar_calibration_binding(
        row.get("t2v_scalar_calibration"), verify_files=verify_files
    )
    samples_raw = row.get("samples")
    if (
        not isinstance(samples_raw, list)
        or row.get("sample_count") != 4
        or len(samples_raw) != 4
    ):
        raise OASISManifestError(
            "Phase-A manifest requires exactly four source cells"
        )
    samples = tuple(
        _validate_sample(item, verify_files=verify_files, ordinal=ordinal)
        for ordinal, item in enumerate(samples_raw)
    )
    expected_order = tuple(
        (family, split) for family in FAMILY_ORDER for split in SPLIT_ORDER
    )
    if tuple((item.family, item.analysis_split) for item in samples) != expected_order:
        raise OASISManifestError(
            "source cells must preserve family-major fit/confirmation order"
        )
    if (
        len({item.sample_id for item in samples}) != 4
        or len({item.source_video_sha256 for item in samples}) != 4
        or len({item.calibration_candidate_id for item in samples}) != 4
        or len({item.sample_digest for item in samples}) != 4
    ):
        raise OASISManifestError(
            "sample/source/calibration identities must be unique"
        )
    return OASISPhaseAManifest(
        path=manifest_path,
        file_sha256=manifest_file_sha256,
        checkpoint_tree_sha256=checkpoint_tree,
        scalar_calibration_status=scalar_status,
        scalar_calibration_path=scalar_path,
        scalar_calibration_file_sha256=scalar_file_sha,
        scalar_calibration_evidence_digest=scalar_evidence_digest,
        seed_order=(seeds[0], seeds[1]),
        samples=samples,
        manifest_digest=declared,
    )


def load_phase_a_manifest(
    path_value: str | Path,
    expected_file_sha256: str,
    *,
    verify_files: bool = True,
) -> OASISPhaseAManifest:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise OASISManifestError(
            "Phase-A manifest must be an absolute existing plain file"
        )
    path = path.resolve(strict=True)
    expected = _sha(expected_file_sha256, label="manifest file SHA")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise OASISManifestError("Phase-A manifest file SHA differs")
    parsed = _strict_json(payload, label="Phase-A manifest")
    return validate_manifest_mapping(
        parsed,
        manifest_path=path,
        manifest_file_sha256=expected,
        verify_files=verify_files,
    )


def _validate_formal_authorization(
    value: Any,
    *,
    phase_a: OASISPhaseAManifest,
    expected_authorization_digest: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(_FORMAL_MAINLINE_AUTHORIZATION_FIELDS)
    ):
        raise OASISManifestError(
            "formal PAIR-v5 scalar authorization field closure differs"
        )
    unsigned = dict(value)
    declared = _sha(
        unsigned.pop("authorization_digest"),
        label="formal PAIR-v5 authorization digest",
    )
    for field in (
        "source_bank_spec_sha256",
        "source_bank_receipt_digest",
        "formal_score_provenance_set_digest",
        "formal_score_arithmetic_contract_digest",
        "preregistration_digest",
        "calibration_receipt_digest",
        "family_mapping_set_digest",
        "checkpoint_tree_sha256",
    ):
        _sha(value.get(field), label=f"formal authorization {field}")
    string_fields = (
        "formal_score_schema",
        "formal_score_filename",
        "formal_score_scalar_definition",
    )
    if any(
        not isinstance(value.get(field), str) or not value.get(field)
        for field in string_fields
    ):
        raise OASISManifestError("formal PAIR-v5 authorization differs")
    if (
        declared != expected_authorization_digest
        or object_sha256(unsigned) != declared
        or value.get("schema_version") != _FORMAL_MAINLINE_AUTHORIZATION_SCHEMA
        or value.get("checkpoint_tree_sha256")
        != phase_a.checkpoint_tree_sha256
        or value.get("score_count") != 40
        or value.get("branch_order") != list(BRANCH_ORDER)
        or value.get("action_family_order")
        != [_FORMAL_FAMILY_ID[family] for family in FAMILY_ORDER]
        or value.get("all_formal_scalar_provenance_recomputed") is not True
        or value.get("formal_receipts_validated_by_active_v4_canonical_code")
        is not True
        or value.get("active_repository_score_schema_consumed") is not True
        or value.get("legacy_v3_compatibility_score_consumed") is not False
        or value.get(
            "initialization_ablation_teacher_or_adapter_artifact_consumed"
        )
        is not False
        or value.get(
            "t2v_media_latent_gaussian_or_proposal_exported_to_native_scorer"
        )
        is not False
        or value.get(
            "only_family_maps_threshold_prompts_and_scalar_digests_exported"
        )
        is not True
        or value.get("calibration_maps_authorized") is not True
        or value.get("native_rv2v_optimizer_authorized") is not False
        or value.get("scientific_action_editing_claim") is not False
    ):
        raise OASISManifestError("formal PAIR-v5 authorization differs")


def load_dedicated_scalar_calibration_evidence(
    phase_a: OASISPhaseAManifest,
) -> Mapping[str, Any]:
    """Load scalar-only calibration authority; never expose T2V media."""

    if not isinstance(phase_a, OASISPhaseAManifest):
        raise OASISManifestError(
            "scalar evidence loader requires a Phase-A manifest"
        )
    if phase_a.scalar_calibration_status != "resolved":
        raise OASISManifestError(
            "OASIS_T2V_SCALAR_CALIBRATION_UNRESOLVED: "
            "formal fit+confirmation scalar evidence is required"
        )
    path = phase_a.scalar_calibration_path
    expected_file_sha = phase_a.scalar_calibration_file_sha256
    expected_evidence_digest = phase_a.scalar_calibration_evidence_digest
    if path is None or expected_file_sha is None or expected_evidence_digest is None:
        raise OASISManifestError("resolved scalar evidence binding is incomplete")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_file_sha:
        raise OASISManifestError("dedicated scalar evidence file changed")
    root = _strict_json(payload, label="dedicated OASIS scalar evidence")
    if not isinstance(root, Mapping) or set(root) != set(_SCALAR_EVIDENCE_FIELDS):
        raise OASISManifestError(
            "dedicated scalar evidence field closure differs"
        )
    unsigned = dict(root)
    declared = _sha(unsigned.pop("evidence_digest"), label="scalar evidence digest")
    if (
        declared != expected_evidence_digest
        or object_sha256(unsigned) != declared
        or root.get("schema_version") != SCALAR_CALIBRATION_EVIDENCE_SCHEMA
        or root.get("checkpoint_tree_sha256")
        != phase_a.checkpoint_tree_sha256
        or root.get("family_order") != list(FAMILY_ORDER)
    ):
        raise OASISManifestError(
            "dedicated scalar evidence digest/schema/binding differs"
        )
    source = root.get("formal_scalar_source")
    if (
        not isinstance(source, Mapping)
        or set(source) != set(_FORMAL_SCALAR_SOURCE_FIELDS)
    ):
        raise OASISManifestError("formal scalar source binding differs")
    if source.get("validator_id") != "validate_pair_v5_t2v_calibration_mainline_v3":
        raise OASISManifestError("formal scalar validator ID differs")
    source_path = _plain_absolute_file(
        source.get("path"), label="formal PAIR-v5 scalar evidence", verify=True
    )
    source_sha = _sha(
        source.get("file_sha256"), label="formal scalar evidence SHA"
    )
    source_authorization_digest = _sha(
        source.get("authorization_digest"),
        label="formal scalar authorization digest",
    )
    if (
        source.get("formal_validator_recomputed") is not True
        or file_sha256(source_path) != source_sha
    ):
        raise OASISManifestError(
            "formal PAIR-v5 scalar evidence was not recomputed/bound"
        )
    formal = _strict_json(
        source_path.read_bytes(), label="formal PAIR-v5 scalar authorization"
    )
    _validate_formal_authorization(
        formal,
        phase_a=phase_a,
        expected_authorization_digest=source_authorization_digest,
    )

    gates = root.get("family_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(FAMILY_ORDER):
        raise OASISManifestError(
            "scalar calibration must cover both families"
        )
    for family in FAMILY_ORDER:
        gate = gates[family]
        if (
            not isinstance(gate, Mapping)
            or set(gate) != set(_FAMILY_SCALAR_GATE_FIELDS)
        ):
            raise OASISManifestError(f"scalar family gate differs: {family}")
        gate_unsigned = dict(gate)
        gate_digest = _sha(
            gate_unsigned.pop("gate_digest"), label=f"{family} gate digest"
        )
        fit_expected = [
            cell.calibration_candidate_id
            for cell in phase_a.samples
            if cell.family == family and cell.analysis_split == "fit"
        ]
        confirmation_expected = [
            cell.calibration_candidate_id
            for cell in phase_a.samples
            if cell.family == family and cell.analysis_split == "confirmation"
        ]
        if (
            object_sha256(gate_unsigned) != gate_digest
            or gate.get("family") != family
            or gate.get("formal_action_family_id") != _FORMAL_FAMILY_ID[family]
            or gate.get("fit_candidate_ids") != fit_expected
            or gate.get("confirmation_candidate_ids") != confirmation_expected
            or gate.get("branch_order") != list(BRANCH_ORDER)
            or gate.get("score_definition")
            != "known_target_velocity_global_MACE_action_vs_all_nine_stable_log_ratio"
            or gate.get("fit_event_and_scalar_gate_passed") is not True
            or gate.get("confirmation_event_and_scalar_gate_passed") is not True
        ):
            raise OASISManifestError(
                f"scalar family authorization differs: {family}"
            )
        robust_threshold = gate.get("minimum_robust_action_log_ratio")
        negative_thresholds = gate.get("minimum_action_log_ratio_by_negative")
        if (
            isinstance(robust_threshold, bool)
            or not isinstance(robust_threshold, (int, float))
            or not math.isfinite(float(robust_threshold))
            or not isinstance(negative_thresholds, Mapping)
            or set(negative_thresholds) != set(BRANCH_ORDER[1:])
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in negative_thresholds.values()
            )
        ):
            raise OASISManifestError(
                f"scalar family thresholds differ: {family}"
            )
    if (
        root.get("scalar_calibration_only") is not True
        or root.get("source_media_consumed_by_oasis_runtime") is not False
        or root.get("source_latent_or_gaussian_consumed_by_oasis_runtime")
        is not False
        or root.get("source_media_or_latent_used_as_teacher") is not False
        or root.get("optimizer_authorized") is not False
        or root.get("frozen_controller_prompt_family_qualified") is not True
        or root.get("scientific_action_editing_success_claim") is not False
    ):
        raise OASISManifestError("scalar-only information/authority closure differs")
    return dict(root)


def seal_manifest_draft(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "manifest_digest" in value:
        raise OASISManifestError("manifest draft must be an unsealed mapping")
    unsigned = dict(value)
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def seal_sample_draft(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "sample_digest" in value:
        raise OASISManifestError("sample draft must be an unsealed mapping")
    unsigned = dict(value)
    return {**unsigned, "sample_digest": object_sha256(unsigned)}


def static_contract() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "sample_schema": SAMPLE_SCHEMA,
        "weak_decoy_schema": WEAK_DECOY_SCHEMA,
        "family_order": list(FAMILY_ORDER),
        "split_order": list(SPLIT_ORDER),
        "arm_order": list(ARM_ORDER),
        "branch_order": list(BRANCH_ORDER),
        "topology": TOPOLOGY,
        "rollout_count": 4 * 2 * len(ARM_ORDER),
        "rollout_count_per_family": 2 * 2 * len(ARM_ORDER),
        "scalar_calibration_binding_schema": SCALAR_CALIBRATION_BINDING_SCHEMA,
        "scalar_calibration_evidence_schema": SCALAR_CALIBRATION_EVIDENCE_SCHEMA,
        "formal_scalar_validator": "validate_pair_v5_t2v_calibration_mainline_v3",
        "formal_scalar_authorization_schema": (
            _FORMAL_MAINLINE_AUTHORIZATION_SCHEMA
        ),
        "old_cagd_authority_accepted": False,
        "wrongref_proxy_used_for_authorization": False,
        "information_flow": INFORMATION_FLOW,
        "candidate_generation_only": True,
        "scientific_action_editing_success_claim": False,
    }
    return {**value, "receipt_digest": object_sha256(value)}


__all__ = [
    "ARM_ORDER",
    "BRANCH_ORDER",
    "FAMILY_ORDER",
    "FPS",
    "FRAME_COUNT",
    "INFORMATION_FLOW",
    "OASISManifestError",
    "OASISPhaseAManifest",
    "OASISSourceCell",
    "REFERENCE_INDICES",
    "SAMPLER_STEPS",
    "SAMPLE_SCHEMA",
    "SCALAR_CALIBRATION_BINDING_SCHEMA",
    "SCALAR_CALIBRATION_EVIDENCE_SCHEMA",
    "SCHEMA_VERSION",
    "SPLIT_ORDER",
    "TOPOLOGY",
    "WEAK_DECOY_SCHEMA",
    "WeakWrongRefDiagnostic",
    "canonical_json_bytes",
    "file_sha256",
    "load_dedicated_scalar_calibration_evidence",
    "load_phase_a_manifest",
    "object_sha256",
    "seal_manifest_draft",
    "seal_sample_draft",
    "source_instruction_binding_digest",
    "static_contract",
    "text_sha256",
    "validate_manifest_mapping",
]
