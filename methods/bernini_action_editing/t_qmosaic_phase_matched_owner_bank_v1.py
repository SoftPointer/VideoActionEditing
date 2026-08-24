#!/usr/bin/env python3
"""Phase-matched pure-T2V owner quotient bank for T-Q-MOSAIC v1.

This engineering-only tensor boundary closes one narrow prerequisite: the
historical owner artifact was queried only at UniPC index 33, whereas the
trajectory intervention consumes three coordinates (20, 28, and 33).  For a
fixed dog or human cell, this module requires six independent owner queries
(``three phases x two preregistered seeds``) and builds one detached,
normalized, spatial-orderless temporal quotient for every query.

The bank does not implement or copy a sampler, does not build an editor VJP,
and does not authorize an intervention.  Three small pack/unpack helpers are
kept separately to regression-test Bernini's pinned ``1x2x2`` state layout;
that layout is not retained by the owner bank.

Only the quotient tensors and digest-only receipts survive construction.
Owner RGB, clean latent, Gaussian/noise, noisy state, full hidden tensors, and
spatial coordinate count/layout do not.  Cryptographic values supplied by a
caller remain bindings rather than file/signature authority, so a future GPU
materializer must still reuse the authenticated owner loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import struct
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import torch

import materialize_self_imagined_owner_core2_v1 as _owner_materializer
import self_imagined_motion_cotangent_v1 as _motion
import t_qmosaic_bernini_unipc_runtime_adapter_v1 as _runtime
import t_qmosaic_trajectory_intervention_v1 as _trajectory


BANK_SCHEMA_VERSION = "bernini-t-qmosaic-phase-owner-quotient-bank-v1"
STATE_BINDING_SCHEMA_VERSION = "bernini-t-qmosaic-phase-owner-state-bindings-v1"

PHASE_INDICES = _trajectory.CAPTURE_PRE_STEP_INDICES
PHASE_TIMESTEPS = _trajectory.CAPTURE_TIMESTEPS
PHASE_SIGMAS = _trajectory.CAPTURE_SIGMAS
PHASE_SIGMA_FLOAT32_HEX = tuple(
    _trajectory.PINNED_SIGMA_FLOAT32_HEX[index] for index in PHASE_INDICES
)
PROMPT_ORDER = ("action", "reverse_wrong_family", "common_scene_noop")
SP_SIZE = 4
PACK_TEMPORAL_PATCH = 1
PACK_SPATIAL_PATCH = 2
PACK_LAYOUT = (
    "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c);"
    "pt=1;ph=2;pw=2"
)
PACKED_CHANNELS = 16 * PACK_TEMPORAL_PATCH * PACK_SPATIAL_PATCH**2
MINIMUM_SPECIFICITY_MARGIN = 0.1
MINIMUM_TWO_SEED_COSINE = 0.05

PINNED_REGISTRY_FILE_SHA256 = (
    "01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96"
)
PINNED_BERNINI_REVISION = _runtime.PINNED_BERNINI_COMMIT
PINNED_VEOMNI_REVISION = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
PINNED_WAN_DIFFUSION_SHA256 = _runtime.PINNED_WAN_DIFFUSION_SHA256

# Public cell metadata intentionally excludes owner latent geometry.  The
# geometry is a private input-validation fact and never enters a receipt.
CELL_SPECS = MappingProxyType(
    {
        "dog": MappingProxyType(
            {
                "owner_generation_seed": 2026081501,
                "query_seeds": (2026081502, 2026081503),
                "source_iid": "7b88a1ca1f804f41",
                "source_video_sha256": (
                    "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
                ),
                "action_family_id": "dog-stand-to-sit-facing-camera",
                "action_caption_utf8_sha256": (
                    "5c8defcffe8413cd556e90c5e17345eb21b51bc787d62d6cde0698cd303cd736"
                ),
                "noop_caption_utf8_sha256": (
                    "aa0b2462d297fe31e47c676e21f1a5d8d99f21983493da034cf1482d79d91441"
                ),
                "reverse_wrong_family_caption_utf8_sha256": (
                    "7913c0ff365301a2b970dbc4c2098c37a76301b8bf2e153850e143633f0199a5"
                ),
            }
        ),
        "human": MappingProxyType(
            {
                "owner_generation_seed": 2026081504,
                "query_seeds": (2026081505, 2026081506),
                "source_iid": "a35b590961d24694",
                "source_video_sha256": (
                    "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed"
                ),
                "action_family_id": "human-one-knee-to-upright-stand",
                "action_caption_utf8_sha256": (
                    "472f6bf16e90c4fa8a6538a9c34f02946ef70278fe5d7916c425f76a6581826b"
                ),
                "noop_caption_utf8_sha256": (
                    "1be9aa015aba37c8df5c0e22c92ef7117258cfdaa97de6373a8e9323a0198007"
                ),
                "reverse_wrong_family_caption_utf8_sha256": (
                    "e821960d84dd4f04fa7845823e312bbcdc0276af0ba34ba08dff596ad5ace93b"
                ),
            }
        ),
    }
)
_OWNER_LATENT_SHAPES = MappingProxyType(
    {
        "dog": (1, 16, 21, 60, 62),
        "human": (1, 16, 21, 64, 58),
    }
)

_OWNER_AUTHORITY_FIELDS = (
    "registry_file_sha256",
    "owner_master_receipt_digest",
    "owner_child_receipt_digest",
    "external_full81_audit_sidecar_receipt_digest",
    "owner_clean_latent_file_sha256",
    "owner_clean_latent_tensor_sha256",
    "checkpoint_content_receipt_digest",
    "bernini_revision",
    "veomni_revision",
)
_FORWARD_PROOF_FIELDS = (
    "native_schedule_index",
    "native_timestep",
    "sigma_float32_hex",
    "query_seed",
    "prompt_order",
    "official_gaussian_tensor_sha256",
    "same_x_sigma_tensor_sha256",
    "same_x_sigma_object_for_all_three_prompts",
    "shared_tensor_bytes_unchanged",
    "hook_coordinate",
    "target_suffix_only",
    "source_condition_consumed",
    "mask_flow_pose_track_trajectory_consumed",
    "spatial_orderless_sketch",
    "full_hidden_persisted",
    "transformer_frozen",
    "adapter_loaded",
    "phase_forward_invocation_digest",
)
_STATE_ROW_FIELDS = (
    "native_schedule_index",
    "native_timestep",
    "sigma_float32_hex",
    "query_seed",
    "official_gaussian_tensor_sha256",
    "same_x_sigma_tensor_sha256",
)
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONSTRUCTION_TOKEN = object()


class TQMosaicPhaseOwnerBankError(RuntimeError):
    """The fixed phase-owner quotient-bank contract was violated."""


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
        raise TQMosaicPhaseOwnerBankError(
            "receipt is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_clone(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("ascii"))


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise TQMosaicPhaseOwnerBankError("receipt is already sealed")
    plain = _json_clone(dict(unsigned))
    return {**plain, "receipt_digest": object_sha256(plain)}


def _closed(value: Any, fields: Iterable[str], *, label: str) -> dict[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected:
        raise TQMosaicPhaseOwnerBankError(f"{label} field closure differs")
    return dict(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TQMosaicPhaseOwnerBankError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise TQMosaicPhaseOwnerBankError(f"{label} must be full lowercase SHA-1")
    return value


def _tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    try:
        return _motion.tensor_value_digest(
            value.detach().to(device="cpu").contiguous(), label=label
        )
    except _motion.SelfImaginedCotangentContractError as error:
        raise TQMosaicPhaseOwnerBankError(str(error)) from error


def _cell_spec(cell_id: Any) -> Mapping[str, Any]:
    if type(cell_id) is not str or cell_id not in CELL_SPECS:
        raise TQMosaicPhaseOwnerBankError("cell must be fixed dog or human")
    return CELL_SPECS[cell_id]


def _owner_source_prompt_binding(cell_id: str) -> dict[str, Any]:
    spec = _cell_spec(cell_id)
    return {
        "source_iid": spec["source_iid"],
        "source_video_sha256": spec["source_video_sha256"],
        "action_family_id": spec["action_family_id"],
        "owner_generation_seed": spec["owner_generation_seed"],
        "ordered_query_seeds": list(spec["query_seeds"]),
        "action_caption_utf8_sha256": spec["action_caption_utf8_sha256"],
        "noop_caption_utf8_sha256": spec["noop_caption_utf8_sha256"],
        "reverse_wrong_family_caption_utf8_sha256": spec[
            "reverse_wrong_family_caption_utf8_sha256"
        ],
        "owner_mode": "frozen_bernini_pure_t2v",
        "source_video_condition_consumed_by_owner": False,
    }


def _validate_owner_authority_binding(
    value: Any, *, clean_tensor_sha256: str
) -> Mapping[str, Any]:
    row = _closed(value, _OWNER_AUTHORITY_FIELDS, label="owner authority binding")
    for name in _OWNER_AUTHORITY_FIELDS:
        if name in ("bernini_revision", "veomni_revision"):
            _sha1(row[name], label=name)
        else:
            _sha256(row[name], label=name)
    if (
        row["registry_file_sha256"] != PINNED_REGISTRY_FILE_SHA256
        or row["owner_clean_latent_tensor_sha256"] != clean_tensor_sha256
        or row["bernini_revision"] != PINNED_BERNINI_REVISION
        or row["veomni_revision"] != PINNED_VEOMNI_REVISION
    ):
        raise TQMosaicPhaseOwnerBankError("owner authority binding differs")
    return MappingProxyType(_json_clone(row))


def _validate_clean_latent(
    value: Any, *, cell_id: str, label: str = "owner clean latent"
) -> torch.Tensor:
    _cell_spec(cell_id)
    expected = _OWNER_LATENT_SHAPES[cell_id]
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.dtype != torch.float32
        or value.device.type == "meta"
        or tuple(map(int, value.shape)) != expected
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise TQMosaicPhaseOwnerBankError(
            f"{label} must be detached finite FP32 with fixed {cell_id} geometry"
        )
    return value


def _validate_hidden(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.dtype != torch.float32
        or value.device.type == "meta"
        or value.ndim != 4
        or int(value.shape[0]) != 1
        or int(value.shape[1]) != _motion.LATENT_PHASES
        or int(value.shape[2]) <= 0
        or int(value.shape[3]) != _motion.HIDDEN_SIZE
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise TQMosaicPhaseOwnerBankError(
            f"{label} must be detached finite FP32 [1,21,K,1536] with K>0"
        )
    return value


def _storage_identity(value: torch.Tensor) -> tuple[str, int | None, int]:
    try:
        storage = value.untyped_storage()
    except AttributeError:  # pragma: no cover - Torch 1.12 compatibility
        storage = value.storage()._untyped()
    return (value.device.type, value.device.index, int(storage.data_ptr()))


def _validate_nested_axes(
    value: Any,
    *,
    seeds: Sequence[int],
    leaf_roles: Sequence[str] | None,
    label: str,
) -> None:
    if type(value) is not dict or tuple(value) != PHASE_INDICES:
        raise TQMosaicPhaseOwnerBankError(f"{label} phase order differs")
    for phase_index in PHASE_INDICES:
        phase = value[phase_index]
        if type(phase) is not dict or tuple(phase) != tuple(seeds):
            raise TQMosaicPhaseOwnerBankError(f"{label} seed order differs")
        if leaf_roles is not None:
            for query_seed in seeds:
                leaf = phase[query_seed]
                if type(leaf) is not dict or tuple(leaf) != tuple(leaf_roles):
                    raise TQMosaicPhaseOwnerBankError(
                        f"{label} prompt-role order differs"
                    )


def _float32_hex(value: float) -> str:
    return struct.pack(">f", float(value)).hex()


class FixedOwnerStateBindingsV1:
    """Digest-only six-row state bundle returned as one non-selective unit."""

    def __init__(
        self,
        *,
        token: object,
        cell_id: str,
        clean_latent_sha256: str,
        rows: tuple[Mapping[str, Any], ...],
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TQMosaicPhaseOwnerBankError("state binding constructor is private")
        spec = _cell_spec(cell_id)
        if len(rows) != len(PHASE_INDICES) * len(spec["query_seeds"]):
            raise TQMosaicPhaseOwnerBankError("state binding row count differs")
        checked_rows: list[dict[str, Any]] = []
        for position, phase_index in enumerate(PHASE_INDICES):
            for query_seed in spec["query_seeds"]:
                raw = rows[len(checked_rows)] if len(rows) > len(checked_rows) else None
                row = _closed(raw, _STATE_ROW_FIELDS, label="owner state binding row")
                _sha256(
                    row["official_gaussian_tensor_sha256"],
                    label="official Gaussian tensor digest",
                )
                _sha256(
                    row["same_x_sigma_tensor_sha256"],
                    label="x-sigma tensor digest",
                )
                if (
                    row["native_schedule_index"] != phase_index
                    or row["native_timestep"] != PHASE_TIMESTEPS[position]
                    or row["sigma_float32_hex"]
                    != PHASE_SIGMA_FLOAT32_HEX[position]
                    or row["query_seed"] != query_seed
                ):
                    raise TQMosaicPhaseOwnerBankError(
                        "owner state binding schedule bits differ"
                    )
                checked_rows.append(row)
        expected_pairs = tuple(
            (phase, seed)
            for phase in PHASE_INDICES
            for seed in spec["query_seeds"]
        )
        if tuple(
            (row["native_schedule_index"], row["query_seed"])
            for row in checked_rows
        ) != expected_pairs:
            raise TQMosaicPhaseOwnerBankError("state binding row closure differs")
        self._cell_id = cell_id
        self._clean_latent_sha256 = _sha256(
            clean_latent_sha256, label="owner clean latent tensor binding"
        )
        self._rows = tuple(
            MappingProxyType(_json_clone(row)) for row in checked_rows
        )
        self._construction_digest = self._build_receipt()["receipt_digest"]

    def rows_in_fixed_order(self) -> tuple[dict[str, Any], ...]:
        self.receipt()
        return tuple(_json_clone(dict(row)) for row in self._rows)

    def _build_receipt(self) -> dict[str, Any]:
        spec = _cell_spec(self._cell_id)
        unsigned = {
            "schema_version": STATE_BINDING_SCHEMA_VERSION,
            "evidence_tier": "ENGINEERING_ONLY",
            "cell_id": self._cell_id,
            "owner_generation_seed": spec["owner_generation_seed"],
            "ordered_query_seeds": list(spec["query_seeds"]),
            "schedule_sha256": _trajectory.PINNED_SCHEDULE_SHA256,
            "phase_indices": list(PHASE_INDICES),
            "phase_timesteps": list(PHASE_TIMESTEPS),
            "phase_sigma_float32_hex": list(PHASE_SIGMA_FLOAT32_HEX),
            "rows": [_json_clone(dict(row)) for row in self._rows],
            "owner_clean_latent_tensor_sha256_binding": self._clean_latent_sha256,
            "same_query_gaussian_digest_reused_across_all_three_phases": True,
            "all_x_sigma_digests_distinct_per_query_seed": True,
            "owner_rgb_persisted": False,
            "owner_clean_latent_tensor_persisted": False,
            "owner_gaussian_or_noise_tensor_persisted": False,
            "owner_x_sigma_tensor_persisted": False,
            "owner_patch_layout_or_spatial_coordinate_count_persisted": False,
            "seed_selection": False,
            "seed_ranking": False,
            "seed_averaging": False,
            "external_callback_consumed": False,
            "optimizer_constructed": False,
            "training_update_authorized": False,
            "parameter_update_performed": False,
        }
        return _seal(unsigned)

    def receipt(self) -> dict[str, Any]:
        sealed = self._build_receipt()
        if sealed["receipt_digest"] != self._construction_digest:
            raise TQMosaicPhaseOwnerBankError(
                "owner state bindings changed after construction"
            )
        return sealed


def derive_fixed_owner_state_bindings_v1(
    *, cell_id: str, owner_clean_latent: torch.Tensor
) -> FixedOwnerStateBindingsV1:
    """Derive all six official-Gaussian state hashes without retaining tensors."""

    clean = _validate_clean_latent(owner_clean_latent, cell_id=cell_id)
    spec = _cell_spec(cell_id)
    rows: list[Mapping[str, Any]] = []
    per_seed_state_digests: dict[int, list[str]] = {
        seed: [] for seed in spec["query_seeds"]
    }
    for query_seed in spec["query_seeds"]:
        epsilon = _owner_materializer.official_gaussian_for_query_seed(
            query_seed,
            shape=tuple(map(int, clean.shape)),
            device=clean.device,
        )
        epsilon_digest = _tensor_sha256(
            epsilon, label=f"{cell_id} query {query_seed} official Gaussian"
        )
        seed_rows: list[Mapping[str, Any]] = []
        for position, phase_index in enumerate(PHASE_INDICES):
            sigma = torch.tensor(
                PHASE_SIGMAS[position], dtype=torch.float32, device=clean.device
            )
            x_sigma = (clean + sigma * (epsilon - clean)).float().contiguous().detach()
            x_sigma_digest = _tensor_sha256(
                x_sigma,
                label=f"{cell_id} phase {phase_index} query {query_seed} x_sigma",
            )
            row = {
                "native_schedule_index": phase_index,
                "native_timestep": PHASE_TIMESTEPS[position],
                "sigma_float32_hex": PHASE_SIGMA_FLOAT32_HEX[position],
                "query_seed": query_seed,
                "official_gaussian_tensor_sha256": epsilon_digest,
                "same_x_sigma_tensor_sha256": x_sigma_digest,
            }
            seed_rows.append(row)
            per_seed_state_digests[query_seed].append(x_sigma_digest)
            del x_sigma
        # Rows are emitted phase-major below; this local list avoids retaining
        # any tensor while preserving each seed's one-Gaussian derivation.
        for row in seed_rows:
            rows.append(row)
        del epsilon

    # Reorder from seed-major construction to the fixed phase-major API.
    by_pair = {
        (row["native_schedule_index"], row["query_seed"]): row for row in rows
    }
    ordered_rows = tuple(
        by_pair[(phase, seed)]
        for phase in PHASE_INDICES
        for seed in spec["query_seeds"]
    )
    if any(len(set(values)) != len(PHASE_INDICES) for values in per_seed_state_digests.values()):
        raise TQMosaicPhaseOwnerBankError("phase owner x_sigma states alias")
    return FixedOwnerStateBindingsV1(
        token=_CONSTRUCTION_TOKEN,
        cell_id=cell_id,
        clean_latent_sha256=_tensor_sha256(
            clean, label=f"{cell_id} owner clean latent"
        ),
        rows=ordered_rows,
    )


def _validate_forward_proof(
    proof: Any, *, expected_state: Mapping[str, Any]
) -> Mapping[str, Any]:
    row = _closed(proof, _FORWARD_PROOF_FIELDS, label="owner phase forward proof")
    _sha256(
        row["phase_forward_invocation_digest"],
        label="phase forward invocation digest",
    )
    if (
        row["native_schedule_index"] != expected_state["native_schedule_index"]
        or row["native_timestep"] != expected_state["native_timestep"]
        or row["sigma_float32_hex"] != expected_state["sigma_float32_hex"]
        or row["query_seed"] != expected_state["query_seed"]
        or row["prompt_order"] != list(PROMPT_ORDER)
        or row["official_gaussian_tensor_sha256"]
        != expected_state["official_gaussian_tensor_sha256"]
        or row["same_x_sigma_tensor_sha256"]
        != expected_state["same_x_sigma_tensor_sha256"]
        or row["same_x_sigma_object_for_all_three_prompts"] is not True
        or row["shared_tensor_bytes_unchanged"] is not True
        or row["hook_coordinate"] != _motion.HOOK_COORDINATE
        or row["target_suffix_only"] is not True
        or row["source_condition_consumed"] is not False
        or row["mask_flow_pose_track_trajectory_consumed"] is not False
        or row["spatial_orderless_sketch"] is not True
        or row["full_hidden_persisted"] is not False
        or row["transformer_frozen"] is not True
        or row["adapter_loaded"] is not False
    ):
        raise TQMosaicPhaseOwnerBankError("owner phase forward proof differs")
    return MappingProxyType(_json_clone(row))


@dataclass(frozen=True)
class PhaseOwnerQuotientRowV1:
    phase_index: int
    timestep: int
    sigma_float32_hex: str
    query_seed: int
    unit_feature: torch.Tensor
    unit_feature_sha256: str
    raw_feature_norm: float
    prompt_specificity_receipt: Mapping[str, Any]
    state_binding_digest: str
    hidden_triplet_digest: str
    forward_proof_digest: str
    phase_forward_invocation_digest: str

    def portable(self) -> dict[str, Any]:
        live = _tensor_sha256(
            self.unit_feature,
            label=f"phase {self.phase_index} query {self.query_seed} live owner quotient",
        )
        if live != self.unit_feature_sha256:
            raise TQMosaicPhaseOwnerBankError("owner quotient bytes changed")
        return {
            "phase_index": self.phase_index,
            "timestep": self.timestep,
            "sigma_float32_hex": self.sigma_float32_hex,
            "query_seed": self.query_seed,
            "unit_feature_shape": list(map(int, self.unit_feature.shape)),
            "unit_feature_dtype": str(self.unit_feature.dtype),
            "unit_feature_sha256": self.unit_feature_sha256,
            "raw_feature_norm": self.raw_feature_norm,
            "prompt_specificity": _json_clone(dict(self.prompt_specificity_receipt)),
            "state_binding_digest": self.state_binding_digest,
            "hidden_triplet_digest": self.hidden_triplet_digest,
            "forward_proof_digest": self.forward_proof_digest,
            "phase_forward_invocation_digest": self.phase_forward_invocation_digest,
        }


class PhaseMatchedOwnerQuotientBankV1:
    """Mutation-checked six-row bank; there is no single-seed accessor."""

    def __init__(
        self,
        *,
        token: object,
        cell_id: str,
        rows: tuple[PhaseOwnerQuotientRowV1, ...],
        phase_seed_audits: tuple[Mapping[str, Any], ...],
        authority_binding: Mapping[str, Any],
        state_binding_receipt_digest: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TQMosaicPhaseOwnerBankError("owner bank constructor is private")
        spec = _cell_spec(cell_id)
        expected_pairs = tuple(
            (phase, seed)
            for phase in PHASE_INDICES
            for seed in spec["query_seeds"]
        )
        if (
            tuple((row.phase_index, row.query_seed) for row in rows)
            != expected_pairs
            or tuple(row["phase_index"] for row in phase_seed_audits)
            != PHASE_INDICES
        ):
            raise TQMosaicPhaseOwnerBankError("owner bank row closure differs")
        if len({row.hidden_triplet_digest for row in rows}) != len(rows):
            raise TQMosaicPhaseOwnerBankError("phase hidden triplet digests alias")
        if len({row.phase_forward_invocation_digest for row in rows}) != len(rows):
            raise TQMosaicPhaseOwnerBankError("phase forward invocations alias")
        for query_seed in spec["query_seeds"]:
            state_digests = {
                row.state_binding_digest
                for row in rows
                if row.query_seed == query_seed
            }
            if len(state_digests) != len(PHASE_INDICES):
                raise TQMosaicPhaseOwnerBankError("per-seed phase state bindings alias")
        self._cell_id = cell_id
        self._rows = rows
        self._phase_seed_audits = tuple(
            MappingProxyType(_json_clone(row)) for row in phase_seed_audits
        )
        self._authority_binding = MappingProxyType(_json_clone(authority_binding))
        self._state_binding_receipt_digest = _sha256(
            state_binding_receipt_digest, label="state binding receipt digest"
        )
        self._construction_digest = self._build_receipt()["receipt_digest"]

    @property
    def cell_id(self) -> str:
        return self._cell_id

    @property
    def ordered_query_seeds(self) -> tuple[int, int]:
        return tuple(_cell_spec(self._cell_id)["query_seeds"])

    def rows_in_fixed_order(self) -> tuple[PhaseOwnerQuotientRowV1, ...]:
        self.receipt()
        return self._rows

    def tensor_payload_in_fixed_order(self) -> dict[str, torch.Tensor]:
        self.receipt()
        return {
            f"phase_{row.phase_index}_query_seed_{row.query_seed}": (
                row.unit_feature.detach().cpu().contiguous().clone()
            )
            for row in self._rows
        }

    def _build_receipt(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": BANK_SCHEMA_VERSION,
            "evidence_tier": "ENGINEERING_ONLY",
            "cell_id": self._cell_id,
            "owner_source_seed_prompt_binding": _owner_source_prompt_binding(
                self._cell_id
            ),
            "schedule_sha256": _trajectory.PINNED_SCHEDULE_SHA256,
            "phase_indices": list(PHASE_INDICES),
            "phase_timesteps": list(PHASE_TIMESTEPS),
            "phase_sigma_float32_hex": list(PHASE_SIGMA_FLOAT32_HEX),
            "rows": [row.portable() for row in self._rows],
            "per_phase_two_seed_audits": [
                _json_clone(dict(row)) for row in self._phase_seed_audits
            ],
            "owner_authority_binding": _json_clone(dict(self._authority_binding)),
            "state_binding_receipt_digest": self._state_binding_receipt_digest,
            "quotient_config_digest": object_sha256(
                _motion.MotionQuotientConfig().receipt()
            ),
            "allowed_persistent_tensor_channel": (
                "detached_normalized_phase_matched_prompt_relative_"
                "spatial_orderless_temporal_hidden_quotient"
            ),
            "all_action_vs_noop_and_reverse_gates_passed_per_phase_seed": True,
            "all_phases_and_seeds_retained_separately": True,
            "index33_tensor_reuse_or_broadcast": False,
            "all_phase_hidden_tensor_objects_and_storages_distinct": True,
            "all_phase_hidden_tensor_value_digests_distinct": True,
            "all_phase_hidden_triplet_digests_distinct": True,
            "all_phase_forward_invocation_digests_distinct": True,
            "all_phase_state_bindings_distinct_per_seed": True,
            "owner_rgb_persisted": False,
            "owner_clean_latent_tensor_or_shape_persisted": False,
            "owner_gaussian_or_noise_tensor_persisted": False,
            "owner_x_sigma_tensor_persisted": False,
            "owner_full_hidden_tensor_persisted": False,
            "owner_patch_layout_or_spatial_coordinate_count_persisted": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "seed_selection": False,
            "seed_ranking": False,
            "seed_averaging": False,
            "phase_selection": False,
            "cross_phase_gate_compensation": False,
            "external_callback_consumed": False,
            "callback_authority": False,
            "dose_input_authorized": False,
            "sign_input_authorized": False,
            "candidate_selection_authorized": False,
            "input_file_or_signature_authority_revalidated_by_this_core": False,
            "packed_state_vjp_constructed": False,
            "trajectory_intervention_authorized": False,
            "gpu_experiment_authorized": False,
            "semantic_success_assessed": False,
            "scientific_claim_authorized": False,
            "optimizer_constructed": False,
            "optimizer_authorized": False,
            "training_update_authorized": False,
            "parameter_update_performed": False,
        }
        return _seal(unsigned)

    def receipt(self) -> dict[str, Any]:
        sealed = self._build_receipt()
        if sealed["receipt_digest"] != self._construction_digest:
            raise TQMosaicPhaseOwnerBankError("owner bank changed after construction")
        return sealed


def build_phase_matched_owner_bank_v1(
    *,
    cell_id: str,
    owner_clean_latent: torch.Tensor,
    hidden_triplets: Mapping[int, Mapping[int, Mapping[str, torch.Tensor]]],
    forward_proofs: Mapping[int, Mapping[int, Mapping[str, Any]]],
    owner_authority_binding: Mapping[str, Any],
) -> PhaseMatchedOwnerQuotientBankV1:
    """Build all six fixed owner rows; no callback or selection knob exists."""

    clean = _validate_clean_latent(owner_clean_latent, cell_id=cell_id)
    spec = _cell_spec(cell_id)
    seeds = tuple(spec["query_seeds"])
    _validate_nested_axes(
        hidden_triplets,
        seeds=seeds,
        leaf_roles=PROMPT_ORDER,
        label="owner hidden triplets",
    )
    _validate_nested_axes(
        forward_proofs,
        seeds=seeds,
        leaf_roles=None,
        label="owner forward proofs",
    )
    clean_digest = _tensor_sha256(clean, label=f"{cell_id} owner clean latent")
    authority = _validate_owner_authority_binding(
        owner_authority_binding, clean_tensor_sha256=clean_digest
    )
    states = derive_fixed_owner_state_bindings_v1(
        cell_id=cell_id, owner_clean_latent=clean
    )
    state_rows = states.rows_in_fixed_order()
    state_by_pair = {
        (row["native_schedule_index"], row["query_seed"]): row
        for row in state_rows
    }

    checked_hidden: dict[
        tuple[int, int], dict[str, torch.Tensor]
    ] = {}
    hidden_triplet_digests: dict[tuple[int, int], str] = {}
    proofs: dict[tuple[int, int], Mapping[str, Any]] = {}
    seen_object_ids: set[int] = set()
    seen_storages: set[tuple[str, int | None, int]] = set()
    seen_value_digests: set[str] = set()
    seen_invocations: set[str] = set()

    # Validate the complete six-row surface before deriving any quotient.  In
    # particular, an index-33 tensor/view/clone cannot be broadcast to 20/28.
    for phase_index in PHASE_INDICES:
        for query_seed in seeds:
            pair = (phase_index, query_seed)
            proof = _validate_forward_proof(
                forward_proofs[phase_index][query_seed],
                expected_state=state_by_pair[pair],
            )
            invocation = proof["phase_forward_invocation_digest"]
            if invocation in seen_invocations:
                raise TQMosaicPhaseOwnerBankError(
                    "phase forward invocation digest reused"
                )
            seen_invocations.add(invocation)
            proofs[pair] = proof

            checked: dict[str, torch.Tensor] = {}
            role_digests: dict[str, str] = {}
            for role in PROMPT_ORDER:
                value = _validate_hidden(
                    hidden_triplets[phase_index][query_seed][role],
                    label=f"phase {phase_index} query {query_seed} {role} hidden",
                )
                object_id = id(value)
                storage_id = _storage_identity(value)
                value_digest = _tensor_sha256(
                    value,
                    label=f"phase {phase_index} query {query_seed} {role} hidden",
                )
                if (
                    object_id in seen_object_ids
                    or storage_id in seen_storages
                    or value_digest in seen_value_digests
                ):
                    raise TQMosaicPhaseOwnerBankError(
                        "phase hidden tensor was reused, broadcast, aliased, or cloned"
                    )
                seen_object_ids.add(object_id)
                seen_storages.add(storage_id)
                seen_value_digests.add(value_digest)
                checked[role] = value
                role_digests[role] = value_digest
            if len({tuple(value.shape) for value in checked.values()}) != 1:
                raise TQMosaicPhaseOwnerBankError("owner prompt hidden shapes differ")
            checked_hidden[pair] = checked
            hidden_triplet_digests[pair] = object_sha256(
                {
                    "phase_index": phase_index,
                    "query_seed": query_seed,
                    "prompt_order": list(PROMPT_ORDER),
                    "role_tensor_sha256": role_digests,
                }
            )

    if len(set(hidden_triplet_digests.values())) != len(hidden_triplet_digests):
        raise TQMosaicPhaseOwnerBankError("phase hidden triplet digests alias")

    rows: list[PhaseOwnerQuotientRowV1] = []
    phase_audits: list[Mapping[str, Any]] = []
    for position, phase_index in enumerate(PHASE_INDICES):
        phase_templates: list[_motion.FrozenOwnerTemplate] = []
        for query_seed in seeds:
            pair = (phase_index, query_seed)
            proof = proofs[pair]
            checked = checked_hidden[pair]
            action_residual = (
                checked["action"] - checked["common_scene_noop"]
            ).float().contiguous().detach()
            reverse_residual = (
                checked["reverse_wrong_family"] - checked["common_scene_noop"]
            ).float().contiguous().detach()
            common_null = (
                checked["common_scene_noop"] - checked["common_scene_noop"]
            ).float().contiguous().detach()
            try:
                template = _motion.build_frozen_owner_template(
                    action_residual,
                    query_seed=query_seed,
                    owner_provenance={
                        "cell_id": cell_id,
                        "owner_generation_seed": spec["owner_generation_seed"],
                        "query_seed": query_seed,
                        "owner_mode": "frozen_bernini_pure_t2v",
                        "owner_exact81_action_audit_passed": True,
                        "owner_used_source_video_condition": False,
                    },
                )
                specificity = _motion.audit_prompt_specificity(
                    template,
                    action_residual=action_residual,
                    reverse_wrong_family_residual=reverse_residual,
                    common_scene_null_residual=common_null,
                    same_x_sigma_binding_digest=proof[
                        "same_x_sigma_tensor_sha256"
                    ],
                    minimum_margin=MINIMUM_SPECIFICITY_MARGIN,
                )
            except _motion.SelfImaginedCotangentContractError as error:
                raise TQMosaicPhaseOwnerBankError(str(error)) from error
            if not specificity.passed:
                raise TQMosaicPhaseOwnerBankError(
                    f"phase {phase_index} query {query_seed} specificity failed"
                )
            owned = template.unit_feature.detach().cpu().float().contiguous().clone()
            unit_digest = _tensor_sha256(
                owned, label=f"phase {phase_index} query {query_seed} owner quotient"
            )
            state_row = state_by_pair[pair]
            rows.append(
                PhaseOwnerQuotientRowV1(
                    phase_index=phase_index,
                    timestep=PHASE_TIMESTEPS[position],
                    sigma_float32_hex=PHASE_SIGMA_FLOAT32_HEX[position],
                    query_seed=query_seed,
                    unit_feature=owned,
                    unit_feature_sha256=unit_digest,
                    raw_feature_norm=template.raw_feature_norm,
                    prompt_specificity_receipt=MappingProxyType(
                        _json_clone(specificity.receipt())
                    ),
                    state_binding_digest=object_sha256(state_row),
                    hidden_triplet_digest=hidden_triplet_digests[pair],
                    forward_proof_digest=object_sha256(dict(proof)),
                    phase_forward_invocation_digest=proof[
                        "phase_forward_invocation_digest"
                    ],
                )
            )
            phase_templates.append(template)
            del action_residual, reverse_residual, common_null
        try:
            two_seed = _motion.audit_two_seed_templates(
                phase_templates, minimum_cosine=MINIMUM_TWO_SEED_COSINE
            )
        except _motion.SelfImaginedCotangentContractError as error:
            raise TQMosaicPhaseOwnerBankError(str(error)) from error
        if not two_seed.passed:
            raise TQMosaicPhaseOwnerBankError(
                f"phase {phase_index} two-seed audit failed"
            )
        phase_audits.append(
            {
                "phase_index": phase_index,
                "timestep": PHASE_TIMESTEPS[position],
                "sigma_float32_hex": PHASE_SIGMA_FLOAT32_HEX[position],
                **two_seed.receipt(),
                "noncompensating_phase_gate": True,
            }
        )
    return PhaseMatchedOwnerQuotientBankV1(
        token=_CONSTRUCTION_TOKEN,
        cell_id=cell_id,
        rows=tuple(rows),
        phase_seed_audits=tuple(phase_audits),
        authority_binding=authority,
        state_binding_receipt_digest=states.receipt()["receipt_digest"],
    )


def _validate_spatial_shape(latent_shape: Sequence[int]) -> tuple[int, ...]:
    try:
        shape = tuple(map(int, latent_shape))
    except (TypeError, ValueError) as error:
        raise TQMosaicPhaseOwnerBankError(
            "Bernini spatial latent geometry differs"
        ) from error
    if (
        len(shape) != 5
        or shape[:3] != (1, 16, 21)
        or shape[3] <= 0
        or shape[4] <= 0
        or shape[3] % PACK_SPATIAL_PATCH != 0
        or shape[4] % PACK_SPATIAL_PATCH != 0
    ):
        raise TQMosaicPhaseOwnerBankError("Bernini spatial latent geometry differs")
    return shape


def expected_packed_shape(latent_shape: Sequence[int]) -> tuple[int, int, int]:
    """Return pinned Bernini's packed shape; this is not a bank receipt field."""

    shape = _validate_spatial_shape(latent_shape)
    positions = (
        shape[2]
        * (shape[3] // PACK_SPATIAL_PATCH)
        * (shape[4] // PACK_SPATIAL_PATCH)
    )
    return (shape[0], positions, PACKED_CHANNELS)


def pack_pinned_bernini_state_v1(
    spatial: torch.Tensor, *, latent_shape: Sequence[int]
) -> torch.Tensor:
    """Exact pinned Bernini ``_to_packed`` layout helper; no sampler logic."""

    shape = _validate_spatial_shape(latent_shape)
    if (
        not isinstance(spatial, torch.Tensor)
        or spatial.layout != torch.strided
        or spatial.dtype != torch.float32
        or spatial.device.type == "meta"
        or tuple(map(int, spatial.shape)) != shape
        or not bool(torch.isfinite(spatial).all().item())
    ):
        raise TQMosaicPhaseOwnerBankError("spatial UniPC state differs")
    b, c, t, height, width = shape
    h = height // PACK_SPATIAL_PATCH
    w = width // PACK_SPATIAL_PATCH
    return (
        spatial.reshape(b, c, t, 1, h, 2, w, 2)
        .permute(0, 2, 4, 6, 3, 5, 7, 1)
        .reshape(b, t * h * w, PACKED_CHANNELS)
    )


def unpack_pinned_bernini_state_v1(
    packed: torch.Tensor, *, latent_shape: Sequence[int]
) -> torch.Tensor:
    """Exact pinned Bernini ``_to_spatial`` layout helper; preserves autograd."""

    shape = _validate_spatial_shape(latent_shape)
    expected = expected_packed_shape(shape)
    if (
        not isinstance(packed, torch.Tensor)
        or packed.layout != torch.strided
        or packed.dtype != torch.float32
        or packed.device.type == "meta"
        or tuple(map(int, packed.shape)) != expected
        or not bool(torch.isfinite(packed).all().item())
    ):
        raise TQMosaicPhaseOwnerBankError("packed UniPC state differs")
    b, c, t, height, width = shape
    h = height // PACK_SPATIAL_PATCH
    w = width // PACK_SPATIAL_PATCH
    return (
        packed.reshape(b, t, h, w, 1, 2, 2, c)
        .permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(shape)
    )


def preregistered_canary_receipt_v1() -> dict[str, Any]:
    """Return the immutable, owner-bank-only surface for a WORLD4 canary."""

    unsigned = {
        "schema_version": "bernini-t-qmosaic-phase-bank-world4-canary-plan-v1",
        "evidence_tier": "ENGINEERING_ONLY",
        "cell_order": ["dog", "human"],
        "owner_source_seed_prompt_bindings": {
            cell: _owner_source_prompt_binding(cell) for cell in CELL_SPECS
        },
        "schedule_sha256": _trajectory.PINNED_SCHEDULE_SHA256,
        "phase_indices": list(PHASE_INDICES),
        "phase_timesteps": list(PHASE_TIMESTEPS),
        "phase_sigma_float32_hex": list(PHASE_SIGMA_FLOAT32_HEX),
        "num_inference_steps": _trajectory.EXACT_SCHEDULER_CALLS,
        "frame_count": 81,
        "world_size": SP_SIZE,
        "ulysses_size": SP_SIZE,
        "guidance_mode": "t2v_apg_owner_hidden_queries",
        "owner_prompt_order": list(PROMPT_ORDER),
        "owner_hidden_forward_count_per_cell": (
            len(PHASE_INDICES) * 2 * len(PROMPT_ORDER)
        ),
        "editor_hidden_forward_count_per_cell": 0,
        "decoded_video_count": 0,
        "packed_state_vjp_materialized": False,
        "trajectory_replay_performed": False,
        "owner_rgb_persisted": False,
        "owner_clean_latent_tensor_or_shape_persisted": False,
        "owner_gaussian_or_noise_tensor_persisted": False,
        "owner_x_sigma_tensor_persisted": False,
        "owner_full_hidden_tensor_persisted": False,
        "owner_patch_layout_or_spatial_coordinate_count_persisted": False,
        "seed_selection": False,
        "seed_ranking": False,
        "seed_averaging": False,
        "dose_input_authorized": False,
        "sign_input_authorized": False,
        "external_callback_authority": False,
        "mask_flow_pose_track_trajectory_consumed": False,
        "gpu_submission_authorized": False,
        "scientific_claim_authorized": False,
        "optimizer_constructed": False,
        "training_update_authorized": False,
        "parameter_update_performed": False,
    }
    return _seal(unsigned)


if (
    PHASE_INDICES != (20, 28, 33)
    or PHASE_TIMESTEPS != (833, 682, 516)
    or tuple(_float32_hex(value) for value in PHASE_SIGMAS)
    != PHASE_SIGMA_FLOAT32_HEX
    or PACKED_CHANNELS != 64
):  # pragma: no cover - import-time cross-contract guard
    raise RuntimeError("T-Q phase-bank pinned constants differ")


__all__ = [
    "BANK_SCHEMA_VERSION",
    "CELL_SPECS",
    "FixedOwnerStateBindingsV1",
    "MINIMUM_SPECIFICITY_MARGIN",
    "MINIMUM_TWO_SEED_COSINE",
    "PACKED_CHANNELS",
    "PACK_LAYOUT",
    "PHASE_INDICES",
    "PHASE_SIGMA_FLOAT32_HEX",
    "PHASE_SIGMAS",
    "PHASE_TIMESTEPS",
    "PINNED_REGISTRY_FILE_SHA256",
    "PINNED_WAN_DIFFUSION_SHA256",
    "PROMPT_ORDER",
    "PhaseMatchedOwnerQuotientBankV1",
    "PhaseOwnerQuotientRowV1",
    "TQMosaicPhaseOwnerBankError",
    "build_phase_matched_owner_bank_v1",
    "canonical_json_bytes",
    "derive_fixed_owner_state_bindings_v1",
    "expected_packed_shape",
    "object_sha256",
    "pack_pinned_bernini_state_v1",
    "preregistered_canary_receipt_v1",
    "unpack_pinned_bernini_state_v1",
]
