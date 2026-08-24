#!/usr/bin/env python3
"""Fail-closed v15c-r10 contract for a true four-role joint-null observer.

This is a source-only observation/consumption seam.  It neither launches a
renderer nor enters an edit route.  The immutable registry expands four
different banks of 64 source-token controls matched only on the preregistered
token count, active-mask rule, and key-pooling statistic.  Their semantic or
distributional exchangeability with a real role phrase is not established.
The historical ``joint-null`` label below therefore means an aligned negative
control bank, not samples from a proved null distribution.  Equal control
index across roles is alignment only; it is not a common randomization
transform.  A future SP4 observer must serialize two separate repeat artifacts
with the exact arrays below.  Local separation and bit equality do not prove
that two independent producer processes ran.

    real/shuffled: [selected_block, role, phase, height, width]
    joint null:    [selected_block, role, null, phase, height, width].

The strict loader replays registry, tensor, SP4 gather, repeat, and receipt
hashes before adapting the tensor to the v15c-r9 future ABI.  A common null
broadcast over the role axis, or any pair of byte-identical role-null slices,
is rejected even when a receipt claims that four-role evidence exists.

No real r10 tensor is shipped by this module.  Consequently the checked-in
plan remains NO-GO until a separately authorized source-only observer run
produces a complete artifact satisfying this contract.
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

import numpy as np

try:
    from . import source_role_authority_v15c_r9 as r9
except ImportError:  # pragma: no cover - flat sealed snapshot
    import source_role_authority_v15c_r9 as r9


SCHEMA_VERSION = "bernini-four-role-joint-null-observer-v15c-r10-local"
REGISTRY_SCHEMA_VERSION = "bernini-e00-four-role-joint-null-registry-v15c-r10"
ARTIFACT_RECEIPT_SCHEMA_VERSION = (
    "bernini-four-role-joint-null-artifact-receipt-v15c-r10-local"
)
R9_ADAPTER_SCHEMA_VERSION = "bernini-v15c-r10-joint-null-to-r9-affinity-v1"

ROLE_NAMES = ("human_agent", "old_actor", "new_actor", "recipient")
SELECTED_BLOCK_INDICES = (4, 9, 14, 19, 24)
NULL_COUNT = 64
PHASE_COUNT = 21
GRID_HEIGHT = 37
GRID_WIDTH = 25
SP_SIZE = 4
REPEAT_COUNT = 2
GLOBAL_VISUAL_TOKENS = PHASE_COUNT * GRID_HEIGHT * GRID_WIDTH
PADDED_LOCAL_TOKENS = math.ceil(GLOBAL_VISUAL_TOKENS / SP_SIZE)
COLLECTIVE_CHANNEL_COUNT = 2 * len(ROLE_NAMES) + len(ROLE_NAMES) * NULL_COUNT
COLLECTIVE_CHANNEL_LAYOUT = {
    "real": [0, len(ROLE_NAMES)],
    "shuffled": [len(ROLE_NAMES), 2 * len(ROLE_NAMES)],
    "joint_null_role_major": [2 * len(ROLE_NAMES), COLLECTIVE_CHANNEL_COUNT],
    "joint_null_role_count": len(ROLE_NAMES),
    "joint_null_count_per_role": NULL_COUNT,
}
REAL_SHAPE = (
    len(SELECTED_BLOCK_INDICES),
    len(ROLE_NAMES),
    PHASE_COUNT,
    GRID_HEIGHT,
    GRID_WIDTH,
)
NULL_SHAPE = (
    len(SELECTED_BLOCK_INDICES),
    len(ROLE_NAMES),
    NULL_COUNT,
    PHASE_COUNT,
    GRID_HEIGHT,
    GRID_WIDTH,
)
SOURCE_VIDEO_SHA256 = (
    "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
)
SOURCE_TEXT_PROVENANCE_SHA256 = (
    "42284074b9a770664ccc9f6ffb2744248a5629e22d06d56711498f37abe02c09"
)
TOKEN_INPUT_IDS_SHA256 = (
    "29c64e1005bc625c64a194d7056c6c1d9b15b78bb994c14793d46fa71d00983e"
)
TOKEN_ATTENTION_MASK_SHA256 = (
    "86c7129afa1c6cc35f5104ce2bf534dff382c6a6e3c2c063b284ed328bcd14a3"
)

# Filled from the checked-in registry after deterministic expansion.  These
# are intentionally repeated in code so replacing and re-signing the JSON is
# insufficient to change the preregistration.
REGISTRY_SHA256 = "b184cef1a7e797c24226b7615057cfb284b7ac9cf8a40657871f705cc696d79e"
ROLE_CONTROL_REGISTRY_SHA256 = (
    "4744c8165efb3710c76e25ed1a2e796fe82dc72a8aa12f7dc1ddbbb60dec85fb",
    "4a8949a10e837a7821daf4e77852924df31e908954de020d16b82a6b4f15b023",
    "1ca1bfa1e683a00cc5fc209665d92fdc544798cfa87ca00db48806b75a104c0b",
    "2fd677f9bdce917ff7ed5a5df6ec0236c77777ad6dacbea86fd68e82b0c27e75",
)
JOINT_INDEX_REGISTRY_SHA256 = (
    "a66ea28c2bdbefddb88434415eed2f3658956b4e1f06a5eec56e1d9f27ce1ba9"
)
CAPTURE_CHANNEL_REGISTRY_SHA256 = (
    "caba9f40811b61f94acbe226f07bdcc6eac239e2669476b97436bdbf4629d699"
)
# No real capture is checked in, so no channel-value root is trusted yet.
# A future collection must add its root to a fresh reviewed release before its
# tensor can reach any downstream control-rank calculation.
PINNED_CAPTURE_CHANNEL_VALUE_BINDING_SHA256: str | None = None

DEFAULT_REGISTRY_ASSET = (
    Path(__file__).resolve().parent
    / "assets"
    / "interaction_e00_four_role_joint_null_registry_v15c_r10.json"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FourRoleJointNullObserverV15CR10Error(RuntimeError):
    """The preregistered source observer or joint-null artifact differs."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FourRoleJointNullObserverV15CR10Error(
            "value is not finite canonical JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FourRoleJointNullObserverV15CR10Error("cannot hash artifact file") from error
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype != np.dtype("<f4") and array.dtype != np.dtype("float32"):
        raise FourRoleJointNullObserverV15CR10Error("array digest requires float32")
    descriptor = {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return object_sha256(descriptor)


def _exact_sha(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise FourRoleJointNullObserverV15CR10Error(f"{label} differs")
    return value


def _exact_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise FourRoleJointNullObserverV15CR10Error(f"{label} differs")
    return value


def _plain_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise FourRoleJointNullObserverV15CR10Error(
            f"{label} must be an absolute non-symlink"
        )
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise FourRoleJointNullObserverV15CR10Error(f"{label} is missing") from error
    if not stat.S_ISREG(mode):
        raise FourRoleJointNullObserverV15CR10Error(f"{label} is not a plain file")
    return path.resolve(strict=True)


@dataclass(frozen=True)
class WrongControlV15CR10:
    role: str
    null_index: int
    source_token_indices: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    control_sha256: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "role": self.role,
            "null_index": self.null_index,
            "source_token_indices": list(self.source_token_indices),
            "source_token_ids": list(self.source_token_ids),
            "token_count": len(self.source_token_indices),
            "key_statistic": "per_token_l2_normalize_then_mean_then_l2_normalize",
            "active_attention_mask_required": True,
            "wrong_control_excludes_every_locked_role_token": True,
        }

    def __post_init__(self) -> None:
        if (
            self.role not in ROLE_NAMES
            or type(self.null_index) is not int
            or not 0 <= self.null_index < NULL_COUNT
            or not self.source_token_indices
            or len(self.source_token_indices) != len(self.source_token_ids)
            or len(set(self.source_token_indices)) != len(self.source_token_indices)
            or any(type(item) is not int or item < 0 for item in self.source_token_indices)
            or any(type(item) is not int or item < 0 for item in self.source_token_ids)
            or object_sha256(self.payload()) != self.control_sha256
        ):
            raise FourRoleJointNullObserverV15CR10Error("wrong-control receipt differs")


@dataclass(frozen=True)
class RoleControlRegistryV15CR10:
    role: str
    role_index: int
    source_substring: str
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    controls: tuple[WrongControlV15CR10, ...]
    registry_sha256: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "role": self.role,
            "role_index": self.role_index,
            "source_substring": self.source_substring,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_ids": list(self.token_ids),
            "token_ids_sha256": self.token_ids_sha256,
            "matching_contract": {
                "exact_role_token_count": len(self.token_ids),
                "active_attention_mask": True,
                "same_key_statistic": (
                    "per_token_l2_normalize_then_mean_then_l2_normalize"
                ),
                "semantically_wrong": "excludes_every_locked_role_token",
            },
            "controls": [dict(item.payload(), control_sha256=item.control_sha256) for item in self.controls],
        }

    def __post_init__(self) -> None:
        if (
            self.role_index != ROLE_NAMES.index(self.role)
            or type(self.role_index) is not int
            or type(self.token_start) is not int
            or type(self.token_end) is not int
            or self.token_end - self.token_start != len(self.token_ids)
            or object_sha256(list(self.token_ids)) != self.token_ids_sha256
            or len(self.controls) != NULL_COUNT
            or tuple(item.null_index for item in self.controls) != tuple(range(NULL_COUNT))
            or any(item.role != self.role for item in self.controls)
            or len({item.source_token_indices for item in self.controls}) != NULL_COUNT
            or object_sha256(self.payload()) != self.registry_sha256
        ):
            raise FourRoleJointNullObserverV15CR10Error("role control registry differs")


@dataclass(frozen=True)
class JointNullRegistryV15CR10:
    active_token_ids: tuple[int, ...]
    locked_role_token_indices: tuple[int, ...]
    eligible_wrong_token_indices: tuple[int, ...]
    roles: tuple[RoleControlRegistryV15CR10, ...]
    joint_index_registry_sha256: str
    capture_channel_registry_sha256: str
    registry_sha256: str
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            len(self.active_token_ids) != 56
            or any(type(item) is not int for item in self.active_token_ids)
            or any(type(item) is not int for item in self.locked_role_token_indices)
            or any(type(item) is not int for item in self.eligible_wrong_token_indices)
            or object_sha256(list(self.active_token_ids)) != TOKEN_INPUT_IDS_SHA256
            or tuple(item.role for item in self.roles) != ROLE_NAMES
            or tuple(item.registry_sha256 for item in self.roles)
            != ROLE_CONTROL_REGISTRY_SHA256
            or len(set(ROLE_CONTROL_REGISTRY_SHA256)) != len(ROLE_NAMES)
            or self.joint_index_registry_sha256 != JOINT_INDEX_REGISTRY_SHA256
            or self.capture_channel_registry_sha256
            != CAPTURE_CHANNEL_REGISTRY_SHA256
            or self.registry_sha256 != REGISTRY_SHA256
        ):
            raise FourRoleJointNullObserverV15CR10Error("joint registry identity differs")


def _derive_control(
    *,
    namespace: str,
    role: str,
    null_index: int,
    token_count: int,
    eligible: Sequence[int],
    active_ids: Sequence[int],
) -> WrongControlV15CR10:
    scores = []
    for source_index in eligible:
        digest = hashlib.sha256(
            (
                f"{namespace}\x00{role}\x00{null_index:02d}\x00"
                f"{source_index:03d}"
            ).encode("utf-8")
        ).digest()
        scores.append((digest, source_index))
    chosen = tuple(item[1] for item in sorted(scores)[:token_count])
    token_ids = tuple(active_ids[index] for index in chosen)
    payload = {
        "role": role,
        "null_index": null_index,
        "source_token_indices": list(chosen),
        "source_token_ids": list(token_ids),
        "token_count": token_count,
        "key_statistic": "per_token_l2_normalize_then_mean_then_l2_normalize",
        "active_attention_mask_required": True,
        "wrong_control_excludes_every_locked_role_token": True,
    }
    return WrongControlV15CR10(
        role=role,
        null_index=null_index,
        source_token_indices=chosen,
        source_token_ids=token_ids,
        control_sha256=object_sha256(payload),
    )


def _expand_role_registry(
    row: Mapping[str, Any],
    *,
    namespace: str,
    eligible: Sequence[int],
    active_ids: Sequence[int],
) -> RoleControlRegistryV15CR10:
    required = {
        "role", "role_index", "source_substring", "token_start", "token_end",
        "token_ids", "token_ids_sha256", "control_registry_sha256",
    }
    if type(row) is not dict or set(row) != required:
        raise FourRoleJointNullObserverV15CR10Error("role registry fields differ")
    token_ids = tuple(row["token_ids"])
    controls = tuple(
        _derive_control(
            namespace=namespace,
            role=row["role"],
            null_index=index,
            token_count=len(token_ids),
            eligible=eligible,
            active_ids=active_ids,
        )
        for index in range(NULL_COUNT)
    )
    return RoleControlRegistryV15CR10(
        role=row["role"],
        role_index=row["role_index"],
        source_substring=row["source_substring"],
        token_start=row["token_start"],
        token_end=row["token_end"],
        token_ids=token_ids,
        token_ids_sha256=row["token_ids_sha256"],
        controls=controls,
        registry_sha256=row["control_registry_sha256"],
    )


def _joint_index_payload(
    roles: Sequence[RoleControlRegistryV15CR10],
) -> list[Mapping[str, Any]]:
    return [
        {
            "null_index": index,
            "role_control_sha256": {
                role.role: role.controls[index].control_sha256 for role in roles
            },
        }
        for index in range(NULL_COUNT)
    ]


def capture_channel_registry_v15c_r10(
    roles: Sequence[RoleControlRegistryV15CR10],
) -> tuple[Mapping[str, Any], ...]:
    if tuple(item.role for item in roles) != ROLE_NAMES:
        raise FourRoleJointNullObserverV15CR10Error(
            "capture channel role registry differs"
        )
    rows: list[Mapping[str, Any]] = []
    for role_index, role in enumerate(roles):
        payload = {
            "channel_index": role_index,
            "kind": "real_role",
            "role": role.role,
            "role_index": role_index,
            "null_index": None,
            "source_key_role": role.role,
            "source_key_token_ids_sha256": role.token_ids_sha256,
            "control_sha256": None,
        }
        rows.append({**payload, "channel_sha256": object_sha256(payload)})
    for role_index, role in enumerate(roles):
        wrong = roles[(role_index + 1) % len(roles)]
        channel_index = len(roles) + role_index
        payload = {
            "channel_index": channel_index,
            "kind": "cyclic_shuffled_role",
            "role": role.role,
            "role_index": role_index,
            "null_index": None,
            "source_key_role": wrong.role,
            "source_key_token_ids_sha256": wrong.token_ids_sha256,
            "control_sha256": None,
        }
        rows.append({**payload, "channel_sha256": object_sha256(payload)})
    for role_index, role in enumerate(roles):
        for null_index, control in enumerate(role.controls):
            channel_index = 2 * len(roles) + role_index * NULL_COUNT + null_index
            payload = {
                "channel_index": channel_index,
                "kind": "role_matched_wrong_null",
                "role": role.role,
                "role_index": role_index,
                "null_index": null_index,
                "source_key_role": None,
                "source_key_token_ids_sha256": object_sha256(
                    list(control.source_token_ids)
                ),
                "control_sha256": control.control_sha256,
            }
            rows.append({**payload, "channel_sha256": object_sha256(payload)})
    if (
        len(rows) != COLLECTIVE_CHANNEL_COUNT
        or tuple(row["channel_index"] for row in rows)
        != tuple(range(COLLECTIVE_CHANNEL_COUNT))
        or len({row["channel_sha256"] for row in rows})
        != COLLECTIVE_CHANNEL_COUNT
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "capture channel registry is not one-to-one"
        )
    return tuple(rows)


def load_joint_null_registry_v15c_r10(
    path: str | Path = DEFAULT_REGISTRY_ASSET,
) -> JointNullRegistryV15CR10:
    asset = _plain_file(Path(path), "joint-null registry")
    try:
        raw = json.loads(asset.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FourRoleJointNullObserverV15CR10Error("cannot parse joint registry") from error
    required = {
        "schema_version", "status", "event_id", "source_video_sha256",
        "source_text_provenance_sha256", "model_text_sha256",
        "tokenizer_tree_sha256", "token_input_ids_sha256",
        "token_attention_mask_sha256", "active_token_count", "active_token_ids",
        "role_names", "roles", "all_locked_role_token_indices",
        "eligible_wrong_token_indices", "null_count", "control_derivation",
        "joint_index_registry_sha256", "capture_channel_registry_sha256",
        "selected_block_indices", "tensor_abi",
        "runtime_contract", "consumer_contract", "registry_sha256",
    }
    if type(raw) is not dict or set(raw) != required:
        raise FourRoleJointNullObserverV15CR10Error("joint registry fields differ")
    derivation = {
        "algorithm": "sha256_rank_without_replacement_v1",
        "namespace": "e00-v15c-r10-role-matched-wrong-controls-20260821",
        "role_specific_seed": True,
        "same_control_index_across_roles_is_alignment_only": True,
        "common_randomization_transform_proven": False,
        "exact_role_token_count_matched": True,
        "active_attention_mask_matched": True,
        "per_token_key_normalize_then_mean_then_normalize_matched": True,
        "excludes_every_locked_role_token": True,
        "control_order_is_semantically_inert_but_hash_bound": True,
    }
    runtime = raw["runtime_contract"]
    consumer = raw["consumer_contract"]
    tensor_abi = raw["tensor_abi"]
    if (
        raw["schema_version"] != REGISTRY_SCHEMA_VERSION
        or raw["status"] != (
            "PREREGISTERED_SOURCE_ONLY_OBSERVER_PLAN_NO_REAL_TENSOR_"
            "NO_STATISTICAL_FWER_NO_GO"
        )
        or raw["event_id"] != "pour-liquid-into-cup"
        or raw["source_video_sha256"] != SOURCE_VIDEO_SHA256
        or raw["source_text_provenance_sha256"] != SOURCE_TEXT_PROVENANCE_SHA256
        or raw["token_input_ids_sha256"] != TOKEN_INPUT_IDS_SHA256
        or raw["token_attention_mask_sha256"] != TOKEN_ATTENTION_MASK_SHA256
        or type(raw["active_token_count"]) is not int
        or raw["active_token_count"] != 56
        or raw["role_names"] != list(ROLE_NAMES)
        or type(raw["null_count"]) is not int
        or raw["null_count"] != NULL_COUNT
        or type(raw["selected_block_indices"]) is not list
        or any(type(item) is not int for item in raw["selected_block_indices"])
        or raw["selected_block_indices"] != list(SELECTED_BLOCK_INDICES)
        or raw["control_derivation"] != derivation
        or tensor_abi != {
            "null_layout": ["selected_block", "role", "null", "phase", "height", "width"],
            "null_shape": list(NULL_SHAPE),
            "real_layout": ["selected_block", "role", "phase", "height", "width"],
            "real_shape": list(REAL_SHAPE),
            "sp_size": SP_SIZE,
            "sp_sharding": "append_pad_contiguous_visual_tokens",
            "collective_channel_order": COLLECTIVE_CHANNEL_LAYOUT,
            "append_padding_value": 0.0,
            "collective_location": "outside_attn2_after_official_output",
            "current_sp4_evidence_scope": (
                "hypothetical_global_array_to_SP4_shard_reconstruction_only"
            ),
            "actual_rank_shard_files_required_for_execution_claim": True,
            "required_separate_repeat_artifacts": REPEAT_COUNT,
            "producer_process_independence_requires_external_provenance": True,
            "repeat_artifact_filenames": {
                "repeat_0": [
                    "repeat_0_real_affinity.npy",
                    "repeat_0_shuffled_affinity.npy",
                    "repeat_0_joint_null_affinity.npy",
                ],
                "repeat_1": [
                    "repeat_1_real_affinity.npy",
                    "repeat_1_shuffled_affinity.npy",
                    "repeat_1_joint_null_affinity.npy",
                ],
            },
            "repeat_equality": "bit_exact",
        }
        or runtime != {
            "source_only": True,
            "anchor_consumed": False,
            "target_edit_instruction_consumed": False,
            "frozen_model_required": True,
            "eval_mode_required": True,
            "all_adapters_off_required": True,
            "attn2_observer_output_modified": False,
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
            "optimizer_updates": 0,
            "renderer_forward_calls_current": 0,
        }
        or consumer != {
            "r9_affinity_shape": list(NULL_SHAPE),
            "common_null_broadcast_forbidden": True,
            "pairwise_byte_identical_role_null_tensor_forbidden": True,
            "control_bank_kind": "aligned_negative_control_bank",
            "global_gate_name": (
                "global_role_proposal_control_max_rank_gate_pass"
            ),
            "finite_control_order_statistic_name": (
                "plus_one_control_exceedance_rank"
            ),
            "control_exchangeability_proven": False,
            "same_index_common_randomization_transform_proven": False,
            "statistical_error_control_available": False,
            "pre_affinity_ordered_proposal_family_binding_required": True,
            "proposal_family_binding_stage": (
                "source_only_before_any_affinity_observation"
            ),
            "proposal_family_binding_fields": [
                "ordered_proposal_ids",
                "mask_sha256_by_proposal",
                "track_sha256_by_proposal",
                "geometry_sha256_by_proposal",
                "family_sha256",
            ],
            "observer_tensor_alone_qualifies_role_assignment": False,
            "mechanical_candidate_only": True,
            "human_reject_only_review_still_required": True,
        }
    ):
        raise FourRoleJointNullObserverV15CR10Error("joint registry policy differs")
    active_ids = tuple(raw["active_token_ids"])
    locked = tuple(raw["all_locked_role_token_indices"])
    eligible = tuple(raw["eligible_wrong_token_indices"])
    if (
        any(type(item) is not int or item < 0 or item >= 55 for item in eligible)
        or len(set(eligible)) != len(eligible)
        or set(eligible) & set(locked)
        or set(eligible) | set(locked) != set(range(55))
        or 55 in eligible
    ):
        raise FourRoleJointNullObserverV15CR10Error("wrong-token pool differs")
    roles = tuple(
        _expand_role_registry(
            row,
            namespace=derivation["namespace"],
            eligible=eligible,
            active_ids=active_ids,
        )
        for row in raw["roles"]
    )
    for role in roles:
        if (
            tuple(active_ids[role.token_start : role.token_end]) != role.token_ids
            or any(
                set(control.source_token_indices) & set(locked)
                or len(control.source_token_indices) != len(role.token_ids)
                for control in role.controls
            )
        ):
            raise FourRoleJointNullObserverV15CR10Error(
                "matched-but-wrong role control differs"
            )
    for index in range(NULL_COUNT):
        controls = [role.controls[index] for role in roles]
        if len({item.source_token_indices for item in controls}) != len(ROLE_NAMES):
            raise FourRoleJointNullObserverV15CR10Error(
                "joint null index reuses a role control"
            )
    if object_sha256(_joint_index_payload(roles)) != raw["joint_index_registry_sha256"]:
        raise FourRoleJointNullObserverV15CR10Error("joint null index hash differs")
    channels = capture_channel_registry_v15c_r10(roles)
    if (
        object_sha256(list(channels)) != raw["capture_channel_registry_sha256"]
        or raw["capture_channel_registry_sha256"]
        != CAPTURE_CHANNEL_REGISTRY_SHA256
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "capture channel registry hash differs"
        )
    payload = dict(raw)
    claimed = payload.pop("registry_sha256")
    if object_sha256(payload) != claimed or claimed != REGISTRY_SHA256:
        raise FourRoleJointNullObserverV15CR10Error("joint registry hash differs")
    return JointNullRegistryV15CR10(
        active_token_ids=active_ids,
        locked_role_token_indices=locked,
        eligible_wrong_token_indices=eligible,
        roles=roles,
        joint_index_registry_sha256=raw["joint_index_registry_sha256"],
        capture_channel_registry_sha256=raw["capture_channel_registry_sha256"],
        registry_sha256=claimed,
        raw=raw,
    )


def validate_runtime_tokenization_v15c_r10(
    *,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    registry: JointNullRegistryV15CR10,
) -> Mapping[str, Any]:
    if not isinstance(registry, JointNullRegistryV15CR10):
        raise FourRoleJointNullObserverV15CR10Error("runtime lacks joint registry")
    ids = tuple(input_ids)
    mask = tuple(attention_mask)
    if (
        ids != registry.active_token_ids
        or len(mask) != len(ids)
        or any(type(item) is not int or item != 1 for item in mask)
        or object_sha256(list(ids)) != TOKEN_INPUT_IDS_SHA256
        or object_sha256(list(mask)) != TOKEN_ATTENTION_MASK_SHA256
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "runtime tokenization differs from preregistration"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": registry.registry_sha256,
        "token_input_ids_sha256": TOKEN_INPUT_IDS_SHA256,
        "token_attention_mask_sha256": TOKEN_ATTENTION_MASK_SHA256,
        "role_control_registry_sha256": list(ROLE_CONTROL_REGISTRY_SHA256),
        "joint_index_registry_sha256": JOINT_INDEX_REGISTRY_SHA256,
        "runtime_exact": True,
        "source_only": True,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    return {**payload, "receipt_sha256": object_sha256(payload)}


def role_null_slices_pairwise_distinct_v15c_r10(null_bank: np.ndarray) -> bool:
    if (
        not isinstance(null_bank, np.ndarray)
        or null_bank.shape != NULL_SHAPE
        or null_bank.dtype != np.float32
        or not null_bank.flags.c_contiguous
        or not bool(np.isfinite(null_bank).all())
    ):
        return False
    for left in range(len(ROLE_NAMES)):
        for right in range(left + 1, len(ROLE_NAMES)):
            if np.array_equal(null_bank[:, left], null_bank[:, right]):
                return False
            if array_sha256(null_bank[:, left]) == array_sha256(null_bank[:, right]):
                return False
    return True


def _ordered_global_channel_arrays(
    real: np.ndarray,
    shuffled: np.ndarray,
    null_bank: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if (
        real.shape != REAL_SHAPE
        or shuffled.shape != REAL_SHAPE
        or null_bank.shape != NULL_SHAPE
        or any(
            value.dtype != np.float32 or not value.flags.c_contiguous
            for value in (real, shuffled, null_bank)
        )
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "global channel tensor contract differs"
        )
    rows: list[np.ndarray] = []
    rows.extend(np.ascontiguousarray(real[:, role]) for role in range(len(ROLE_NAMES)))
    rows.extend(
        np.ascontiguousarray(shuffled[:, role]) for role in range(len(ROLE_NAMES))
    )
    rows.extend(
        np.ascontiguousarray(null_bank[:, role, null_index])
        for role in range(len(ROLE_NAMES))
        for null_index in range(NULL_COUNT)
    )
    if len(rows) != COLLECTIVE_CHANNEL_COUNT:
        raise FourRoleJointNullObserverV15CR10Error(
            "global channel count differs"
        )
    return tuple(rows)


def capture_channel_value_binding_payload_v15c_r10(
    *,
    registry: JointNullRegistryV15CR10,
    repeat_real: Sequence[np.ndarray],
    repeat_shuffled: Sequence[np.ndarray],
    repeat_null_bank: Sequence[np.ndarray],
) -> Mapping[str, Any]:
    if (
        not isinstance(registry, JointNullRegistryV15CR10)
        or len(repeat_real) != REPEAT_COUNT
        or len(repeat_shuffled) != REPEAT_COUNT
        or len(repeat_null_bank) != REPEAT_COUNT
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "channel-value repeat binding differs"
        )
    channel_registry = capture_channel_registry_v15c_r10(registry.roles)
    repeat_channels = [
        _ordered_global_channel_arrays(
            repeat_real[index], repeat_shuffled[index], repeat_null_bank[index]
        )
        for index in range(REPEAT_COUNT)
    ]
    rows = []
    for channel_index, channel in enumerate(channel_registry):
        row = dict(channel)
        row["repeat_array_sha256"] = [
            array_sha256(repeat_channels[repeat][channel_index])
            for repeat in range(REPEAT_COUNT)
        ]
        rows.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": registry.registry_sha256,
        "capture_channel_registry_sha256": (
            registry.capture_channel_registry_sha256
        ),
        "channel_count": COLLECTIVE_CHANNEL_COUNT,
        "channel_order": COLLECTIVE_CHANNEL_LAYOUT,
        "required_separate_repeat_artifacts": REPEAT_COUNT,
        "channels": rows,
    }


def _rank_interval(rank: int) -> tuple[int, int, int]:
    if type(rank) is not int or not 0 <= rank < SP_SIZE:
        raise FourRoleJointNullObserverV15CR10Error("SP4 rank differs")
    start = rank * PADDED_LOCAL_TOKENS
    stop = min(GLOBAL_VISUAL_TOKENS, start + PADDED_LOCAL_TOKENS)
    return start, stop, stop - start


def _expected_padded_collective_tensor(
    *,
    real_block: np.ndarray,
    shuffled_block: np.ndarray,
    null_block: np.ndarray,
    rank: int,
) -> np.ndarray:
    if (
        real_block.shape != REAL_SHAPE[1:]
        or shuffled_block.shape != REAL_SHAPE[1:]
        or null_block.shape != NULL_SHAPE[1:]
        or any(
            value.dtype != np.float32 or not value.flags.c_contiguous
            for value in (real_block, shuffled_block, null_block)
        )
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "SP4 collective source tensor differs"
        )
    start, stop, valid = _rank_interval(rank)
    result = np.zeros(
        (COLLECTIVE_CHANNEL_COUNT, PADDED_LOCAL_TOKENS), dtype=np.float32
    )
    result[0 : len(ROLE_NAMES), :valid] = real_block.reshape(
        len(ROLE_NAMES), GLOBAL_VISUAL_TOKENS
    )[:, start:stop]
    result[len(ROLE_NAMES) : 2 * len(ROLE_NAMES), :valid] = (
        shuffled_block.reshape(len(ROLE_NAMES), GLOBAL_VISUAL_TOKENS)[:, start:stop]
    )
    result[2 * len(ROLE_NAMES) :, :valid] = null_block.reshape(
        len(ROLE_NAMES) * NULL_COUNT, GLOBAL_VISUAL_TOKENS
    )[:, start:stop]
    if valid < PADDED_LOCAL_TOKENS and bool(
        np.any(result[:, valid:] != np.float32(0.0))
    ):
        raise FourRoleJointNullObserverV15CR10Error("SP4 append padding differs")
    return result


def _validate_sp4_repeat_transcript(
    value: Any,
    *,
    repeat_real: Sequence[np.ndarray],
    repeat_shuffled: Sequence[np.ndarray],
    repeat_null_bank: Sequence[np.ndarray],
) -> str:
    if type(value) is not dict or set(value) != {
        "blocks", "required_separate_repeat_artifacts", "repeat_equality",
        "implicit_collective_calls_inside_attn2", "collective_channel_layout",
        "capture_channel_registry_sha256", "append_padding_value",
        "reconstruction_scope", "actual_rank_shard_files_consumed",
        "producer_process_independence_verified",
        "transcript_sha256",
    }:
        raise FourRoleJointNullObserverV15CR10Error("SP4 transcript fields differ")
    payload = dict(value)
    transcript_sha = payload.pop("transcript_sha256")
    if (
        object_sha256(payload) != _exact_sha(transcript_sha, "SP4 transcript SHA")
        or type(value["required_separate_repeat_artifacts"]) is not int
        or value["required_separate_repeat_artifacts"] != REPEAT_COUNT
        or value["repeat_equality"] != "bit_exact"
        or type(value["implicit_collective_calls_inside_attn2"]) is not int
        or value["implicit_collective_calls_inside_attn2"] != 0
        or value["collective_channel_layout"] != COLLECTIVE_CHANNEL_LAYOUT
        or value["capture_channel_registry_sha256"]
        != CAPTURE_CHANNEL_REGISTRY_SHA256
        or value["reconstruction_scope"]
        != "hypothetical_global_array_to_SP4_shard_reconstruction_only"
        or value["actual_rank_shard_files_consumed"] is not False
        or value["producer_process_independence_verified"] is not False
        or type(value["append_padding_value"]) not in (int, float)
        or isinstance(value["append_padding_value"], bool)
        or float(value["append_padding_value"]) != 0.0
        or type(value["blocks"]) is not list
        or len(value["blocks"]) != len(SELECTED_BLOCK_INDICES)
        or len(repeat_real) != REPEAT_COUNT
        or len(repeat_shuffled) != REPEAT_COUNT
        or len(repeat_null_bank) != REPEAT_COUNT
    ):
        raise FourRoleJointNullObserverV15CR10Error("SP4 transcript hash/policy differs")
    for block_position, block in enumerate(value["blocks"]):
        if type(block) is not dict or set(block) != {
            "block_index", "repeats", "repeat_bit_exact"
        }:
            raise FourRoleJointNullObserverV15CR10Error("SP4 block transcript differs")
        if (
            type(block["block_index"]) is not int
            or block["block_index"] != SELECTED_BLOCK_INDICES[block_position]
            or block["repeat_bit_exact"] is not True
            or type(block["repeats"]) is not list
            or len(block["repeats"]) != REPEAT_COUNT
        ):
            raise FourRoleJointNullObserverV15CR10Error("SP4 block/repeat order differs")
        rank_tensor_digests: list[list[str]] = []
        metadata_digests: set[str] = set()
        for repeat_index, repeat in enumerate(block["repeats"]):
            real = repeat_real[repeat_index]
            shuffled = repeat_shuffled[repeat_index]
            null_bank = repeat_null_bank[repeat_index]
            expected_assembled = {
                "real_array_sha256": array_sha256(real[block_position]),
                "shuffled_array_sha256": array_sha256(shuffled[block_position]),
                "role_null_array_sha256": [
                    array_sha256(null_bank[block_position, role])
                    for role in range(len(ROLE_NAMES))
                ],
            }
            required = {
                "repeat_index", "capture_pass_label", "ranks",
                "assembled", "repeat_receipt_sha256",
            }
            if type(repeat) is not dict or set(repeat) != required:
                raise FourRoleJointNullObserverV15CR10Error("SP4 repeat fields differ")
            repeat_payload = dict(repeat)
            repeat_sha = repeat_payload.pop("repeat_receipt_sha256")
            if (
                object_sha256(repeat_payload)
                != _exact_sha(repeat_sha, "repeat receipt SHA")
                or type(repeat["repeat_index"]) is not int
                or repeat["repeat_index"] != repeat_index
                or repeat["capture_pass_label"]
                != f"source-observer-pass-{repeat_index}"
                or repeat["assembled"] != expected_assembled
                or type(repeat["ranks"]) is not list
                or len(repeat["ranks"]) != SP_SIZE
            ):
                raise FourRoleJointNullObserverV15CR10Error(
                    "SP4 repeat hash/assembly differs"
                )
            current_rank_digests = []
            for rank, row in enumerate(repeat["ranks"]):
                required_rank = {
                    "sp_rank", "global_start", "global_stop",
                    "padded_local_tokens", "valid_local_tokens",
                    "collective_channel_count", "collective_tensor_sha256",
                    "padded_collective_shape", "append_padding_all_zero",
                    "channel_layout_sha256", "capture_channel_registry_sha256",
                    "metadata_sha256",
                }
                if type(row) is not dict or set(row) != required_rank:
                    raise FourRoleJointNullObserverV15CR10Error("SP4 rank fields differ")
                start, stop, valid = _rank_interval(rank)
                metadata_payload = {
                    "block_index": block["block_index"],
                    "repeat_index": repeat_index,
                    "capture_pass_label": repeat["capture_pass_label"],
                    "registry_sha256": REGISTRY_SHA256,
                    **{key: row[key] for key in required_rank if key != "metadata_sha256"},
                }
                expected_collective = _expected_padded_collective_tensor(
                    real_block=np.ascontiguousarray(real[block_position]),
                    shuffled_block=np.ascontiguousarray(shuffled[block_position]),
                    null_block=np.ascontiguousarray(null_bank[block_position]),
                    rank=rank,
                )
                expected_tensor_sha = array_sha256(expected_collective)
                if (
                    type(row["sp_rank"]) is not int
                    or row["sp_rank"] != rank
                    or type(row["global_start"]) is not int
                    or row["global_start"] != start
                    or type(row["global_stop"]) is not int
                    or row["global_stop"] != stop
                    or type(row["padded_local_tokens"]) is not int
                    or row["padded_local_tokens"] != PADDED_LOCAL_TOKENS
                    or type(row["valid_local_tokens"]) is not int
                    or row["valid_local_tokens"] != valid
                    or type(row["collective_channel_count"]) is not int
                    or row["collective_channel_count"] != COLLECTIVE_CHANNEL_COUNT
                    or row["padded_collective_shape"]
                    != [COLLECTIVE_CHANNEL_COUNT, PADDED_LOCAL_TOKENS]
                    or any(
                        type(item) is not int
                        for item in row["padded_collective_shape"]
                    )
                    or row["append_padding_all_zero"] is not True
                    or row["channel_layout_sha256"]
                    != object_sha256(COLLECTIVE_CHANNEL_LAYOUT)
                    or row["capture_channel_registry_sha256"]
                    != CAPTURE_CHANNEL_REGISTRY_SHA256
                    or row["collective_tensor_sha256"] != expected_tensor_sha
                    or object_sha256(metadata_payload)
                    != _exact_sha(row["metadata_sha256"], "rank metadata SHA")
                ):
                    raise FourRoleJointNullObserverV15CR10Error(
                        "SP4 rank gather geometry differs"
                    )
                tensor_sha = _exact_sha(
                    row["collective_tensor_sha256"], "collective tensor SHA"
                )
                current_rank_digests.append(tensor_sha)
                if row["metadata_sha256"] in metadata_digests:
                    raise FourRoleJointNullObserverV15CR10Error(
                        "SP4 rank metadata was cloned"
                    )
                metadata_digests.add(row["metadata_sha256"])
            rank_tensor_digests.append(current_rank_digests)
        if rank_tensor_digests[0] != rank_tensor_digests[1]:
            raise FourRoleJointNullObserverV15CR10Error(
                "separate SP4 repeat artifacts are not bit deterministic"
            )
    return transcript_sha


@dataclass(frozen=True)
class LoadedJointNullArtifactV15CR10:
    repeat_real: tuple[np.ndarray, np.ndarray]
    repeat_shuffled: tuple[np.ndarray, np.ndarray]
    repeat_null_bank: tuple[np.ndarray, np.ndarray]
    receipt: Mapping[str, Any]
    receipt_sha256: str

    @property
    def real(self) -> np.ndarray:
        return self.repeat_real[0]

    @property
    def shuffled(self) -> np.ndarray:
        return self.repeat_shuffled[0]

    @property
    def null_bank(self) -> np.ndarray:
        return self.repeat_null_bank[0]


def _load_npy_exact(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    plain = _plain_file(path, "tensor file")
    try:
        value = np.load(str(plain), allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise FourRoleJointNullObserverV15CR10Error("cannot load tensor file") from error
    if (
        not isinstance(value, np.ndarray)
        or value.shape != expected_shape
        or value.dtype != np.float32
        or not value.flags.c_contiguous
        or not bool(np.isfinite(value).all())
    ):
        raise FourRoleJointNullObserverV15CR10Error("tensor array contract differs")
    return value


def validate_loaded_joint_null_artifact_v15c_r10(
    artifact: LoadedJointNullArtifactV15CR10,
    *,
    registry: JointNullRegistryV15CR10,
    expected_capture_channel_value_binding_sha256: str | None = (
        PINNED_CAPTURE_CHANNEL_VALUE_BINDING_SHA256
    ),
) -> Mapping[str, Any]:
    if (
        not isinstance(artifact, LoadedJointNullArtifactV15CR10)
        or not isinstance(registry, JointNullRegistryV15CR10)
    ):
        raise FourRoleJointNullObserverV15CR10Error("joint artifact type differs")
    repeat_groups = (
        artifact.repeat_real,
        artifact.repeat_shuffled,
        artifact.repeat_null_bank,
    )
    if any(type(group) is not tuple or len(group) != REPEAT_COUNT for group in repeat_groups):
        raise FourRoleJointNullObserverV15CR10Error("two repeat artifacts are required")
    for group, shape in zip(repeat_groups, (REAL_SHAPE, REAL_SHAPE, NULL_SHAPE)):
        for value in group:
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != np.float32
                or not value.flags.c_contiguous
                or not bool(np.isfinite(value).all())
            ):
                raise FourRoleJointNullObserverV15CR10Error(
                    "loaded repeat tensor contract differs"
                )
        if np.shares_memory(group[0], group[1]):
            raise FourRoleJointNullObserverV15CR10Error(
                "repeat tensors are not separate artifacts"
            )
        if not np.array_equal(group[0], group[1]) or array_sha256(group[0]) != array_sha256(group[1]):
            raise FourRoleJointNullObserverV15CR10Error(
                "repeat artifacts are not bit exact"
            )
    all_repeat_arrays = (
        artifact.repeat_real[0], artifact.repeat_shuffled[0],
        artifact.repeat_null_bank[0], artifact.repeat_real[1],
        artifact.repeat_shuffled[1], artifact.repeat_null_bank[1],
    )
    if any(
        np.shares_memory(all_repeat_arrays[left], all_repeat_arrays[right])
        for left in range(len(all_repeat_arrays))
        for right in range(left + 1, len(all_repeat_arrays))
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "repeat triplet arrays share memory"
        )
    for null_bank in artifact.repeat_null_bank:
        if not role_null_slices_pairwise_distinct_v15c_r10(null_bank):
            raise FourRoleJointNullObserverV15CR10Error(
                "common/broadcast/byte-identical role-null tensor is forbidden"
            )
    receipt = artifact.receipt
    required = {
        "schema_version", "status", "event_id", "source_video_sha256",
        "source_text_provenance_sha256", "registry_sha256",
        "role_control_registry_sha256", "joint_index_registry_sha256",
        "capture_channel_registry_sha256",
        "capture_channel_value_binding_sha256",
        "selected_block_indices", "role_names", "null_index_alignment",
        "tensor_files", "sp4_repeat_transcript", "source_only",
        "frozen_model", "eval_mode", "all_adapters_off",
        "attn2_observer_output_modified", "anchor_consumed",
        "target_edit_instruction_consumed", "four_role_joint_null_available",
        "common_null_broadcast_used", "route_authorized", "decode_authorized",
        "training_authorized", "optimizer_updates", "observer_tensor_contract_pass_only",
        "global_role_proposal_control_max_rank_gate_pass",
        "pre_affinity_ordered_proposal_family_binding_verified",
        "control_exchangeability_proven",
        "same_index_common_randomization_transform_proven",
        "statistical_error_control_available",
        "actual_sp4_rank_shard_files_present", "sp4_gather_execution_verified",
        "remote_execution_verified", "scientific_claim_authorized",
        "receipt_sha256",
    }
    if type(receipt) is not dict or set(receipt) != required:
        raise FourRoleJointNullObserverV15CR10Error("artifact receipt fields differ")
    receipt_payload = dict(receipt)
    claimed_receipt_sha = receipt_payload.pop("receipt_sha256")
    tensor_files = receipt["tensor_files"]
    if (
        object_sha256(receipt_payload)
        != _exact_sha(claimed_receipt_sha, "artifact receipt SHA")
        or claimed_receipt_sha != artifact.receipt_sha256
        or receipt["schema_version"] != ARTIFACT_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != "REAL_SOURCE_OBSERVER_TENSOR_LOCAL_CONTRACT_PASS_ONLY"
        or receipt["event_id"] != "pour-liquid-into-cup"
        or receipt["source_video_sha256"] != SOURCE_VIDEO_SHA256
        or receipt["source_text_provenance_sha256"] != SOURCE_TEXT_PROVENANCE_SHA256
        or receipt["registry_sha256"] != registry.registry_sha256
        or receipt["role_control_registry_sha256"]
        != list(ROLE_CONTROL_REGISTRY_SHA256)
        or receipt["joint_index_registry_sha256"] != JOINT_INDEX_REGISTRY_SHA256
        or receipt["capture_channel_registry_sha256"]
        != CAPTURE_CHANNEL_REGISTRY_SHA256
        or type(receipt["selected_block_indices"]) is not list
        or any(type(item) is not int for item in receipt["selected_block_indices"])
        or receipt["selected_block_indices"] != list(SELECTED_BLOCK_INDICES)
        or receipt["role_names"] != list(ROLE_NAMES)
        or receipt["null_index_alignment"]
        != (
            "same_control_index_j_is_alignment_only_across_blocks_phases_"
            "proposals_with_distinct_role_controls"
        )
        or any(receipt[name] is not expected for name, expected in {
            "source_only": True,
            "frozen_model": True,
            "eval_mode": True,
            "all_adapters_off": True,
            "attn2_observer_output_modified": False,
            "anchor_consumed": False,
            "target_edit_instruction_consumed": False,
            "four_role_joint_null_available": True,
            "common_null_broadcast_used": False,
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
            "observer_tensor_contract_pass_only": True,
            "global_role_proposal_control_max_rank_gate_pass": False,
            "pre_affinity_ordered_proposal_family_binding_verified": False,
            "control_exchangeability_proven": False,
            "same_index_common_randomization_transform_proven": False,
            "statistical_error_control_available": False,
            "actual_sp4_rank_shard_files_present": False,
            "sp4_gather_execution_verified": False,
            "remote_execution_verified": False,
            "scientific_claim_authorized": False,
        }.items())
        or type(receipt["optimizer_updates"]) is not int
        or receipt["optimizer_updates"] != 0
        or type(tensor_files) is not dict
        or set(tensor_files) != {"repeat_0", "repeat_1"}
    ):
        raise FourRoleJointNullObserverV15CR10Error("artifact receipt policy differs")
    expected_shapes = {"real": REAL_SHAPE, "shuffled": REAL_SHAPE, "joint_null": NULL_SHAPE}
    repeat_array_sha: list[dict[str, str]] = []
    for repeat_index in range(REPEAT_COUNT):
        repeat_key = f"repeat_{repeat_index}"
        repeat_rows = tensor_files[repeat_key]
        if type(repeat_rows) is not dict or set(repeat_rows) != {
            "real", "shuffled", "joint_null"
        }:
            raise FourRoleJointNullObserverV15CR10Error("repeat tensor map differs")
        arrays = {
            "real": artifact.repeat_real[repeat_index],
            "shuffled": artifact.repeat_shuffled[repeat_index],
            "joint_null": artifact.repeat_null_bank[repeat_index],
        }
        digest_row: dict[str, str] = {}
        for name, array in arrays.items():
            row = repeat_rows[name]
            required_row = {
                "filename", "shape", "dtype", "file_size", "file_sha256",
                "array_sha256", "capture_repeat_index",
            }
            if name == "joint_null":
                required_row.add("role_array_sha256")
            if type(row) is not dict or set(row) != required_row:
                raise FourRoleJointNullObserverV15CR10Error(
                    "tensor receipt fields differ"
                )
            expected_digest = array_sha256(array)
            digest_row[name] = expected_digest
            if (
                type(row["capture_repeat_index"]) is not int
                or row["capture_repeat_index"] != repeat_index
                or row["shape"] != list(expected_shapes[name])
                or any(type(item) is not int for item in row["shape"])
                or row["dtype"] != "float32"
                or row["array_sha256"] != expected_digest
                or type(row["file_size"]) is not int
                or row["file_size"] <= 0
            ):
                raise FourRoleJointNullObserverV15CR10Error("tensor receipt differs")
            _exact_sha(row["file_sha256"], "tensor file SHA")
            if name == "joint_null":
                expected_roles = [
                    array_sha256(array[:, role]) for role in range(len(ROLE_NAMES))
                ]
                if (
                    row["role_array_sha256"] != expected_roles
                    or len(set(expected_roles)) != len(ROLE_NAMES)
                ):
                    raise FourRoleJointNullObserverV15CR10Error(
                        "role-null tensor digest differs"
                    )
        repeat_array_sha.append(digest_row)
    if repeat_array_sha[0] != repeat_array_sha[1]:
        raise FourRoleJointNullObserverV15CR10Error(
            "repeat tensor digests are not bit exact"
        )
    channel_binding_payload = capture_channel_value_binding_payload_v15c_r10(
        registry=registry,
        repeat_real=artifact.repeat_real,
        repeat_shuffled=artifact.repeat_shuffled,
        repeat_null_bank=artifact.repeat_null_bank,
    )
    channel_binding_sha = object_sha256(channel_binding_payload)
    if receipt["capture_channel_value_binding_sha256"] != channel_binding_sha:
        raise FourRoleJointNullObserverV15CR10Error(
            "ordered 264-channel value binding differs"
        )
    caller_expected_matched = expected_capture_channel_value_binding_sha256 is not None
    if caller_expected_matched:
        if (
            _exact_sha(
                expected_capture_channel_value_binding_sha256,
                "expected capture channel-value binding SHA",
            )
            != channel_binding_sha
        ):
            raise FourRoleJointNullObserverV15CR10Error(
                "capture channel-value binding differs from caller expectation"
            )
    release_pinned = PINNED_CAPTURE_CHANNEL_VALUE_BINDING_SHA256 is not None
    if release_pinned and (
        _exact_sha(
            PINNED_CAPTURE_CHANNEL_VALUE_BINDING_SHA256,
            "release-pinned capture channel-value binding SHA",
        )
        != channel_binding_sha
    ):
        raise FourRoleJointNullObserverV15CR10Error(
            "capture channel-value binding differs from reviewed release pin"
        )
    transcript_sha = _validate_sp4_repeat_transcript(
        receipt["sp4_repeat_transcript"],
        repeat_real=artifact.repeat_real,
        repeat_shuffled=artifact.repeat_shuffled,
        repeat_null_bank=artifact.repeat_null_bank,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_receipt_sha256": claimed_receipt_sha,
        "registry_sha256": registry.registry_sha256,
        "joint_index_registry_sha256": registry.joint_index_registry_sha256,
        "capture_channel_registry_sha256": (
            registry.capture_channel_registry_sha256
        ),
        "pinned_capture_channel_value_binding_sha256": (
            PINNED_CAPTURE_CHANNEL_VALUE_BINDING_SHA256
        ),
        "role_control_registry_sha256": list(ROLE_CONTROL_REGISTRY_SHA256),
        "capture_channel_value_binding_sha256": channel_binding_sha,
        "repeat_array_sha256": repeat_array_sha,
        "sp4_repeat_transcript_sha256": transcript_sha,
        "four_role_joint_null_tensor_contract_verified": True,
        "common_null_broadcast_detected": False,
        "control_bank_kind": "aligned_negative_control_bank",
        "finite_control_order_statistic_name": "plus_one_control_exceedance_rank",
        "control_exchangeability_proven": False,
        "same_index_common_randomization_transform_proven": False,
        "statistical_error_control_available": False,
        "pre_affinity_ordered_proposal_family_binding_verified": False,
        "global_role_proposal_control_max_rank_gate_pass": False,
        "observer_tensor_contract_pass_only": True,
        "separate_repeat_artifacts_verified": True,
        "producer_process_independence_verified": False,
        "caller_expected_capture_channel_value_binding_matched": (
            caller_expected_matched
        ),
        "independent_capture_channel_value_binding_pinned": release_pinned,
        "official_r10_runner_present": False,
        "actual_sp4_rank_shard_files_replayed": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    return {**payload, "validation_sha256": object_sha256(payload)}


def load_joint_null_artifact_v15c_r10(
    artifact_directory: str | Path,
    receipt_path: str | Path,
    *,
    registry: JointNullRegistryV15CR10 | None = None,
) -> LoadedJointNullArtifactV15CR10:
    registry = registry or load_joint_null_registry_v15c_r10()
    root = Path(artifact_directory)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise FourRoleJointNullObserverV15CR10Error(
            "artifact directory must be absolute/non-symlink"
        )
    receipt_file = _plain_file(Path(receipt_path), "artifact receipt")
    if receipt_file.parent != root.resolve(strict=True):
        raise FourRoleJointNullObserverV15CR10Error("artifact receipt escaped directory")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FourRoleJointNullObserverV15CR10Error("cannot parse artifact receipt") from error
    if type(receipt) is not dict or type(receipt.get("tensor_files")) is not dict:
        raise FourRoleJointNullObserverV15CR10Error("artifact receipt tensor map differs")
    arrays: dict[str, list[np.ndarray]] = {
        "real": [], "shuffled": [], "joint_null": []
    }
    resolved_paths: set[Path] = set()
    file_identities: set[tuple[int, int]] = set()
    for repeat_index in range(REPEAT_COUNT):
        repeat_key = f"repeat_{repeat_index}"
        repeat_rows = receipt["tensor_files"].get(repeat_key)
        if type(repeat_rows) is not dict:
            raise FourRoleJointNullObserverV15CR10Error("repeat tensor map differs")
        expected_names = {
            "real": f"repeat_{repeat_index}_real_affinity.npy",
            "shuffled": f"repeat_{repeat_index}_shuffled_affinity.npy",
            "joint_null": f"repeat_{repeat_index}_joint_null_affinity.npy",
        }
        for name, filename in expected_names.items():
            row = repeat_rows.get(name)
            if type(row) is not dict or row.get("filename") != filename:
                raise FourRoleJointNullObserverV15CR10Error("tensor filename differs")
            path = root / filename
            plain = _plain_file(path, "tensor file")
            file_stat = plain.stat()
            identity = (int(file_stat.st_dev), int(file_stat.st_ino))
            if (
                plain.parent != root.resolve(strict=True)
                or plain in resolved_paths
                or identity in file_identities
            ):
                raise FourRoleJointNullObserverV15CR10Error(
                    "tensor file escaped directory or reused a path/inode"
                )
            resolved_paths.add(plain)
            file_identities.add(identity)
            if (
                type(row.get("file_size")) is not int
                or file_stat.st_size != row.get("file_size")
                or file_sha256(plain) != row.get("file_sha256")
            ):
                raise FourRoleJointNullObserverV15CR10Error(
                    "tensor file byte receipt differs"
                )
            shape = REAL_SHAPE if name != "joint_null" else NULL_SHAPE
            arrays[name].append(_load_npy_exact(plain, shape))
    artifact = LoadedJointNullArtifactV15CR10(
        repeat_real=(arrays["real"][0], arrays["real"][1]),
        repeat_shuffled=(arrays["shuffled"][0], arrays["shuffled"][1]),
        repeat_null_bank=(arrays["joint_null"][0], arrays["joint_null"][1]),
        receipt=receipt,
        receipt_sha256=receipt.get("receipt_sha256"),
    )
    validate_loaded_joint_null_artifact_v15c_r10(artifact, registry=registry)
    return artifact


def adapt_joint_null_artifact_to_r9_v15c_r10(
    artifact: LoadedJointNullArtifactV15CR10,
    *,
    registry: JointNullRegistryV15CR10,
    expected_capture_channel_value_binding_sha256: str | None = (
        PINNED_CAPTURE_CHANNEL_VALUE_BINDING_SHA256
    ),
) -> r9.R6AffinityInputV15CR9:
    validation = validate_loaded_joint_null_artifact_v15c_r10(
        artifact,
        registry=registry,
        expected_capture_channel_value_binding_sha256=(
            expected_capture_channel_value_binding_sha256
        ),
    )
    if validation["independent_capture_channel_value_binding_pinned"] is not True:
        raise FourRoleJointNullObserverV15CR10Error(
            "r10-to-r9 adapter NO-GO: no independent capture binding is pinned"
        )
    # Even a pinned global tensor is not an executed SP4 gather and no fresh
    # r10 runner/postflight yet binds a source-only pre-affinity proposal set.
    # Therefore this release deliberately exports no available=True r9 object.
    raise FourRoleJointNullObserverV15CR10Error(
        "r10-to-r9 adapter NO-GO: actual rank shards and fresh runner/postflight are absent"
    )


def current_no_tensor_status_v15c_r10() -> Mapping[str, Any]:
    registry = load_joint_null_registry_v15c_r10()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PREREGISTERED_PLAN_ONLY_NO_REAL_TENSOR_"
            "NO_STATISTICAL_FWER_NO_GO"
        ),
        "registry_sha256": registry.registry_sha256,
        "role_control_registry_sha256": list(ROLE_CONTROL_REGISTRY_SHA256),
        "joint_index_registry_sha256": registry.joint_index_registry_sha256,
        "expected_null_shape": list(NULL_SHAPE),
        "real_tensor_present": False,
        "four_role_joint_null_verified": False,
        "r9_future_affinity_constructed": False,
        "control_bank_kind": "aligned_negative_control_bank",
        "plus_one_control_exceedance_rank_computed": False,
        "global_role_proposal_control_max_rank_gate_pass": False,
        "pre_affinity_ordered_proposal_family_binding_verified": False,
        "control_exchangeability_proven": False,
        "same_index_common_randomization_transform_proven": False,
        "statistical_error_control_available": False,
        "mechanical_candidate_qualified": False,
        "remote_execution_verified": False,
        "separate_repeat_artifacts_present": False,
        "producer_process_independence_verified": False,
        "actual_sp4_rank_shard_files_present": False,
        "official_r10_runner_present": False,
        "official_r10_postflight_present": False,
        "renderer_forward_calls": 0,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "scientific_claim_authorized": False,
    }
    return {**payload, "status_sha256": object_sha256(payload)}


__all__ = [
    "ARTIFACT_RECEIPT_SCHEMA_VERSION", "COLLECTIVE_CHANNEL_COUNT",
    "DEFAULT_REGISTRY_ASSET", "FourRoleJointNullObserverV15CR10Error",
    "GLOBAL_VISUAL_TOKENS", "GRID_HEIGHT", "GRID_WIDTH",
    "JOINT_INDEX_REGISTRY_SHA256", "JointNullRegistryV15CR10",
    "LoadedJointNullArtifactV15CR10", "NULL_COUNT", "NULL_SHAPE",
    "PADDED_LOCAL_TOKENS", "PHASE_COUNT", "REAL_SHAPE", "REGISTRY_SHA256",
    "REPEAT_COUNT", "ROLE_CONTROL_REGISTRY_SHA256", "ROLE_NAMES",
    "SCHEMA_VERSION", "SELECTED_BLOCK_INDICES", "SOURCE_VIDEO_SHA256",
    "SP_SIZE", "adapt_joint_null_artifact_to_r9_v15c_r10", "array_sha256",
    "canonical_bytes", "current_no_tensor_status_v15c_r10", "file_sha256",
    "load_joint_null_artifact_v15c_r10", "load_joint_null_registry_v15c_r10",
    "object_sha256", "role_null_slices_pairwise_distinct_v15c_r10",
    "validate_loaded_joint_null_artifact_v15c_r10",
    "validate_runtime_tokenization_v15c_r10",
]
