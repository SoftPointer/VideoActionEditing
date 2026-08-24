"""Precomputed-latent dataset contract for PACT/OmniVideo2 training.

Qwen, UMT5, VAE encoding and actor tracking are intentionally offline.  A
training worker only loads digest-bound tensors, which keeps the online memory
budget focused on the 1.3B renderer and adapters.
"""

from __future__ import annotations

import hashlib
import io
import inspect
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import (
    canonical_json_bytes,
    file_sha256,
    load_jsonl,
    validate_atomic_row,
)


PAYLOAD_FORMAT = "pact-precomputed-latents-v2"
ENCODER_CONTRACT_FORMAT = "pact-omnivideo2-offline-encoder-contract-v1"

# These are the exact channel statistics embedded in the official Wan2.1 VAE
# implementation used by OmniVideo2-1.3B.  They are part of the latent
# coordinate system, not tunable dataset metadata.  Changing even one value
# therefore defines a different, incompatible offline encoder contract.
WAN21_VAE_CHANNEL_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN21_VAE_CHANNEL_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)
WAN21_VAE_STRIDE = (4, 8, 8)
WAN21_VAE_INPUT_PIXEL_RANGE = (-1.0, 1.0)
WAN21_VAE_POSTERIOR_MODE = "mean"
UMT5_EMBEDDING_DIM = 4096
UMT5_MAX_SEQUENCE_LENGTH = 512
UMT5_SEGMENT_ORDER = ("target_caption", "edit_instruction")
UMT5_PADDING_POLICY = "slice_to_attention_mask_length"
VLM_EMBEDDING_DIM = 2048
VLM_FEATURE_TENSOR = "vlm_last_hidden_states"
VLM_TOKEN_SELECTION = "attention_mask_then_drop_system_prefix"

# (payload field, canonical atomic-manifest field).  Keep this immutable and
# shared with the publication binder so an offline payload cannot be replayed
# against another video's latents, masks, track, or prompt contract merely by
# copying the original three semantic hashes.
PAYLOAD_PROVENANCE_BINDINGS = (
    ("parent_row_sha256", "parent_row_sha256"),
    ("source_video_sha256", "source_video_sha256"),
    (
        "global_counterfactual_target_video_sha256",
        "global_counterfactual_target_video_sha256",
    ),
    ("source_component_mask_sha256", "source_component_mask_sha256"),
    ("target_component_mask_sha256", "target_component_mask_sha256"),
    ("track_record_sha256", "track_record_sha256"),
    ("edit_instruction_sha256", "edit_instruction_sha256"),
    ("target_caption_contract_sha256", "target_caption_contract_sha256"),
)


class DatasetContractError(ValueError):
    """Raised when a precomputed training payload is incomplete or unsafe."""


def _sha256_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DatasetContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _closed_mapping(
    value: Any, *, name: str, fields: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetContractError(f"{name} must be an object")
    result = dict(value)
    if set(result) != fields:
        raise DatasetContractError(
            f"{name} fields differ: missing={sorted(fields - set(result))}, "
            f"unknown={sorted(set(result) - fields)}"
        )
    return result


def _exact_float_list(
    value: Any, *, name: str, expected: Sequence[float]
) -> list[float]:
    if type(value) is not list:
        raise DatasetContractError(f"{name} must be a JSON list of floats")
    if len(value) != len(expected):
        raise DatasetContractError(
            f"{name} must contain exactly {len(expected)} values"
        )
    for index, (actual, required) in enumerate(zip(value, expected)):
        if type(actual) is not float or actual != required:
            raise DatasetContractError(
                f"{name}[{index}] must be the exact float constant {required!r}"
            )
    return list(value)


def _exact_int_list(
    value: Any, *, name: str, expected: Sequence[int]
) -> list[int]:
    if type(value) is not list:
        raise DatasetContractError(f"{name} must be a JSON list of integers")
    if len(value) != len(expected):
        raise DatasetContractError(
            f"{name} must contain exactly {len(expected)} values"
        )
    for index, (actual, required) in enumerate(zip(value, expected)):
        if type(actual) is not int or actual != required:
            raise DatasetContractError(
                f"{name}[{index}] must be the exact integer constant {required}"
            )
    return list(value)


def _exact_str_list(
    value: Any, *, name: str, expected: Sequence[str]
) -> list[str]:
    if type(value) is not list:
        raise DatasetContractError(f"{name} must be a JSON list of strings")
    if len(value) != len(expected) or any(
        type(actual) is not str or actual != required
        for actual, required in zip(value, expected)
    ):
        raise DatasetContractError(f"{name} must equal {list(expected)!r}")
    return list(value)


def _exact_int(value: Any, *, name: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise DatasetContractError(
            f"{name} must be the exact integer constant {expected}"
        )
    return value


def validate_encoder_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed Wan2.1 offline-encoder coordinate contract.

    Checkpoint *manifest* digests bind potentially sharded UMT5 and VLM
    encoders.  The VAE is one official checkpoint file and is bound directly.
    All non-digest values are exact constants so a worker can never silently
    mix alternate latent normalization or context preprocessing conventions.
    """

    contract = _closed_mapping(
        value,
        name="encoder_contract",
        fields={"format", "vae", "umt5", "vlm"},
    )
    if (
        type(contract["format"]) is not str
        or contract["format"] != ENCODER_CONTRACT_FORMAT
    ):
        raise DatasetContractError(
            f"encoder_contract format must be {ENCODER_CONTRACT_FORMAT}"
        )

    vae = _closed_mapping(
        contract["vae"],
        name="encoder_contract.vae",
        fields={
            "checkpoint_sha256",
            "preprocessing_contract_sha256",
            "input_pixel_range",
            "posterior_mode",
            "channel_mean",
            "channel_std",
            "stride",
        },
    )
    vae_digest = _sha256_text(
        vae["checkpoint_sha256"],
        name="encoder_contract.vae.checkpoint_sha256",
    )
    vae_preprocessing_digest = _sha256_text(
        vae["preprocessing_contract_sha256"],
        name="encoder_contract.vae.preprocessing_contract_sha256",
    )
    pixel_range = _exact_float_list(
        vae["input_pixel_range"],
        name="encoder_contract.vae.input_pixel_range",
        expected=WAN21_VAE_INPUT_PIXEL_RANGE,
    )
    if (
        type(vae["posterior_mode"]) is not str
        or vae["posterior_mode"] != WAN21_VAE_POSTERIOR_MODE
    ):
        raise DatasetContractError(
            "encoder_contract.vae.posterior_mode must be the exact string 'mean'"
        )
    channel_mean = _exact_float_list(
        vae["channel_mean"],
        name="encoder_contract.vae.channel_mean",
        expected=WAN21_VAE_CHANNEL_MEAN,
    )
    channel_std = _exact_float_list(
        vae["channel_std"],
        name="encoder_contract.vae.channel_std",
        expected=WAN21_VAE_CHANNEL_STD,
    )
    stride = _exact_int_list(
        vae["stride"],
        name="encoder_contract.vae.stride",
        expected=WAN21_VAE_STRIDE,
    )

    umt5 = _closed_mapping(
        contract["umt5"],
        name="encoder_contract.umt5",
        fields={
            "checkpoint_manifest_sha256",
            "preprocessing_contract_sha256",
            "embedding_dim",
            "max_sequence_length_per_segment",
            "segment_order",
            "padding_policy",
        },
    )
    umt5_digest = _sha256_text(
        umt5["checkpoint_manifest_sha256"],
        name="encoder_contract.umt5.checkpoint_manifest_sha256",
    )
    umt5_preprocessing_digest = _sha256_text(
        umt5["preprocessing_contract_sha256"],
        name="encoder_contract.umt5.preprocessing_contract_sha256",
    )
    umt5_dim = _exact_int(
        umt5["embedding_dim"],
        name="encoder_contract.umt5.embedding_dim",
        expected=UMT5_EMBEDDING_DIM,
    )
    umt5_max_length = _exact_int(
        umt5["max_sequence_length_per_segment"],
        name="encoder_contract.umt5.max_sequence_length_per_segment",
        expected=UMT5_MAX_SEQUENCE_LENGTH,
    )
    umt5_segment_order = _exact_str_list(
        umt5["segment_order"],
        name="encoder_contract.umt5.segment_order",
        expected=UMT5_SEGMENT_ORDER,
    )
    if type(umt5["padding_policy"]) is not str or (
        umt5["padding_policy"] != UMT5_PADDING_POLICY
    ):
        raise DatasetContractError(
            f"encoder_contract.umt5.padding_policy must be {UMT5_PADDING_POLICY!r}"
        )

    vlm = _closed_mapping(
        contract["vlm"],
        name="encoder_contract.vlm",
        fields={
            "checkpoint_manifest_sha256",
            "feature_extraction_contract_sha256",
            "embedding_dim",
            "feature_tensor",
            "token_selection",
        },
    )
    vlm_digest = _sha256_text(
        vlm["checkpoint_manifest_sha256"],
        name="encoder_contract.vlm.checkpoint_manifest_sha256",
    )
    vlm_feature_contract_digest = _sha256_text(
        vlm["feature_extraction_contract_sha256"],
        name="encoder_contract.vlm.feature_extraction_contract_sha256",
    )
    vlm_dim = _exact_int(
        vlm["embedding_dim"],
        name="encoder_contract.vlm.embedding_dim",
        expected=VLM_EMBEDDING_DIM,
    )
    if type(vlm["feature_tensor"]) is not str or (
        vlm["feature_tensor"] != VLM_FEATURE_TENSOR
    ):
        raise DatasetContractError(
            f"encoder_contract.vlm.feature_tensor must be {VLM_FEATURE_TENSOR!r}"
        )
    if type(vlm["token_selection"]) is not str or (
        vlm["token_selection"] != VLM_TOKEN_SELECTION
    ):
        raise DatasetContractError(
            f"encoder_contract.vlm.token_selection must be {VLM_TOKEN_SELECTION!r}"
        )

    # Return a new JSON-native value rather than references owned by an
    # untrusted torch payload.  Its canonical bytes are consequently stable.
    return {
        "format": ENCODER_CONTRACT_FORMAT,
        "vae": {
            "checkpoint_sha256": vae_digest,
            "preprocessing_contract_sha256": vae_preprocessing_digest,
            "input_pixel_range": pixel_range,
            "posterior_mode": WAN21_VAE_POSTERIOR_MODE,
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "stride": stride,
        },
        "umt5": {
            "checkpoint_manifest_sha256": umt5_digest,
            "preprocessing_contract_sha256": umt5_preprocessing_digest,
            "embedding_dim": umt5_dim,
            "max_sequence_length_per_segment": umt5_max_length,
            "segment_order": umt5_segment_order,
            "padding_policy": UMT5_PADDING_POLICY,
        },
        "vlm": {
            "checkpoint_manifest_sha256": vlm_digest,
            "feature_extraction_contract_sha256": vlm_feature_contract_digest,
            "embedding_dim": vlm_dim,
            "feature_tensor": VLM_FEATURE_TENSOR,
            "token_selection": VLM_TOKEN_SELECTION,
        },
    }


def encoder_contract_sha256(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the validated canonical encoder contract."""

    contract = validate_encoder_contract(value)
    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()


def _finite_float_tensor(
    value: Any, *, name: str, ndim: int, leading_channels: int | None = None
) -> Tensor:
    if not isinstance(value, Tensor):
        raise DatasetContractError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise DatasetContractError(f"{name} must have {ndim} dimensions")
    if not value.is_floating_point():
        raise DatasetContractError(f"{name} must have a floating dtype")
    if min(value.shape) <= 0:
        raise DatasetContractError(f"{name} has an empty dimension")
    if leading_channels is not None and value.shape[0] != leading_channels:
        raise DatasetContractError(
            f"{name} first dimension must be {leading_channels}, got {value.shape[0]}"
        )
    if not bool(torch.isfinite(value).all()):
        raise DatasetContractError(f"{name} contains NaN or Inf")
    return value


def validate_precomputed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetContractError("payload must be an object")
    payload = dict(value)
    if payload.get("format") != PAYLOAD_FORMAT:
        raise DatasetContractError(f"payload format must be {PAYLOAD_FORMAT}")
    required = {
        "atom_id",
        "encoder_contract",
        "source_latent",
        "global_target_latent",
        "source_component_mask",
        "target_component_mask",
        "text_context",
        "vlm_context",
    } | {payload_field for payload_field, _ in PAYLOAD_PROVENANCE_BINDINGS}
    missing = sorted(required - set(payload))
    if missing:
        raise DatasetContractError(f"payload fields missing: {missing}")

    payload["encoder_contract"] = validate_encoder_contract(
        payload["encoder_contract"]
    )

    source = _finite_float_tensor(
        payload["source_latent"], name="source_latent", ndim=4
    )
    target = _finite_float_tensor(
        payload["global_target_latent"], name="global_target_latent", ndim=4
    )
    if source.shape != target.shape:
        raise DatasetContractError("source and target latent shapes differ")
    if source.shape[0] != len(WAN21_VAE_CHANNEL_MEAN):
        raise DatasetContractError(
            "Wan2.1 source and target latents must have exactly 16 channels"
        )
    if source.device.type != "cpu" or target.device.type != "cpu":
        raise DatasetContractError("offline latent payloads must be stored on CPU")

    for key in ("source_component_mask", "target_component_mask"):
        mask = _finite_float_tensor(payload[key], name=key, ndim=4, leading_channels=1)
        if mask.shape[1:] != source.shape[1:]:
            raise DatasetContractError(f"{key} spatiotemporal shape differs from latent")
        if not bool(((mask >= 0) & (mask <= 1)).all()):
            raise DatasetContractError(f"{key} values must lie in [0, 1]")
        if mask.device.type != "cpu":
            raise DatasetContractError(f"{key} must be stored on CPU")

    text = _finite_float_tensor(payload["text_context"], name="text_context", ndim=2)
    vlm = _finite_float_tensor(payload["vlm_context"], name="vlm_context", ndim=2)
    if text.shape[-1] != 4096:
        raise DatasetContractError("text_context last dimension must be 4096")
    if vlm.shape[-1] != 2048:
        raise DatasetContractError("vlm_context last dimension must be 2048")
    if text.device.type != "cpu" or vlm.device.type != "cpu":
        raise DatasetContractError("offline contexts must be stored on CPU")

    atom_id = payload["atom_id"]
    if not isinstance(atom_id, str) or not atom_id:
        raise DatasetContractError("payload atom_id must be a non-empty string")
    for payload_field, _ in PAYLOAD_PROVENANCE_BINDINGS:
        _sha256_text(payload[payload_field], name=payload_field)
    return payload


def _safe_torch_load_bytes(data: bytes, *, path: Path) -> Mapping[str, Any]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise DatasetContractError(
            "this PyTorch lacks safe weights_only loading; upgrade PyTorch"
        )
    value = torch.load(
        io.BytesIO(data), map_location="cpu", weights_only=True
    )
    if not isinstance(value, Mapping):
        raise DatasetContractError(f"payload at {path} is not a mapping")
    return value


def _resolve_payload_path(payload_root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetContractError(
            "latent_payload_path must be relative to payload_root without '..'"
        )
    candidate = payload_root / relative
    current = candidate
    while current != payload_root:
        if current.is_symlink():
            raise DatasetContractError(f"payload path contains a symlink: {current}")
        parent = current.parent
        if parent == current:
            raise DatasetContractError("payload path escaped payload_root")
        current = parent
    resolved = candidate.resolve()
    try:
        resolved.relative_to(payload_root)
    except ValueError as exc:
        raise DatasetContractError("payload path escaped payload_root") from exc
    return resolved


class AtomicLatentDataset(Dataset[dict[str, Any]]):
    """Load a training-authorized atomic manifest and tensor payloads."""

    def __init__(
        self,
        manifest_path: os.PathLike[str] | str,
        *,
        payload_root: os.PathLike[str] | str | None = None,
        require_training_authorized: bool = True,
        require_payload_digest: bool = True,
        verify_payload_digest: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.payload_root = Path(
            self.root if payload_root is None else payload_root
        ).resolve()
        if not self.payload_root.is_dir():
            raise DatasetContractError(
                f"payload_root is not a directory: {self.payload_root}"
            )
        rows = load_jsonl(self.manifest_path)
        if not rows:
            raise DatasetContractError("atomic training manifest is empty")
        self.rows = []
        atom_ids: set[str] = set()
        for raw in rows:
            row = validate_atomic_row(raw)
            if require_training_authorized and row["training_authorized"] is not True:
                raise DatasetContractError(
                    f"atom {row['atom_id']} is not authorized for training"
                )
            if row["atom_id"] in atom_ids:
                raise DatasetContractError(f"duplicate atom_id: {row['atom_id']}")
            atom_ids.add(row["atom_id"])
            raw_path = row.get("latent_payload_path")
            if not isinstance(raw_path, str) or not raw_path:
                raise DatasetContractError(
                    f"atom {row['atom_id']} lacks latent_payload_path"
                )
            payload_path = _resolve_payload_path(self.payload_root, raw_path)
            if not payload_path.is_file():
                raise DatasetContractError(f"payload does not exist: {payload_path}")
            expected_digest = row.get("latent_payload_sha256")
            if require_payload_digest and (
                not isinstance(expected_digest, str) or len(expected_digest) != 64
            ):
                raise DatasetContractError(
                    f"atom {row['atom_id']} lacks a valid latent_payload_sha256"
                )
            if expected_digest is not None:
                _sha256_text(expected_digest, name="latent_payload_sha256")
            if verify_payload_digest and expected_digest is not None:
                actual = file_sha256(payload_path)
                if actual != expected_digest:
                    raise DatasetContractError(
                        f"payload digest differs for atom {row['atom_id']}"
                    )
            row["_resolved_payload_path"] = str(payload_path)
            row["_verify_payload_digest"] = bool(verify_payload_digest)
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        payload_path = Path(row["_resolved_payload_path"])
        payload_bytes = payload_path.read_bytes()
        if row["_verify_payload_digest"]:
            actual = hashlib.sha256(payload_bytes).hexdigest()
            if actual != row["latent_payload_sha256"]:
                raise DatasetContractError(
                    f"payload digest differs for atom {row['atom_id']} at load time"
                )
        payload = validate_precomputed_payload(
            _safe_torch_load_bytes(payload_bytes, path=payload_path)
        )
        if payload["atom_id"] != row["atom_id"]:
            raise DatasetContractError(
                f"payload atom_id differs for manifest atom {row['atom_id']}"
            )
        for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS:
            if payload[payload_field] != row.get(atomic_field):
                raise DatasetContractError(
                    f"payload {payload_field} differs from atomic {atomic_field} "
                    f"for manifest atom {row['atom_id']}"
                )
        return {
            "atom_id": row["atom_id"],
            "encoder_contract": payload["encoder_contract"],
            "encoder_contract_sha256": encoder_contract_sha256(
                payload["encoder_contract"]
            ),
            "source_latent": payload["source_latent"],
            "global_target_latent": payload["global_target_latent"],
            "source_component_mask": payload["source_component_mask"],
            "target_component_mask": payload["target_component_mask"],
            "text_context": payload["text_context"],
            "vlm_context": payload["vlm_context"],
            "metadata": {
                "parent_iid": row["parent_iid"],
                "selected_subject_ids": row["selected_subject_ids"],
                "edit_instruction": row["edit_instruction"],
            },
        }


def collate_atomic_latents(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise DatasetContractError("cannot collate an empty batch")
    contracts: list[dict[str, Any]] = []
    contract_digests: list[str] = []
    for index, sample in enumerate(samples):
        if (
            "encoder_contract" not in sample
            or "encoder_contract_sha256" not in sample
        ):
            raise DatasetContractError(
                f"sample {index} lacks encoder_contract provenance"
            )
        contract = validate_encoder_contract(sample["encoder_contract"])
        digest = encoder_contract_sha256(contract)
        supplied_digest = _sha256_text(
            sample["encoder_contract_sha256"],
            name=f"samples[{index}].encoder_contract_sha256",
        )
        if supplied_digest != digest:
            raise DatasetContractError(
                f"sample {index} encoder_contract_sha256 differs from canonical contract"
            )
        contracts.append(contract)
        contract_digests.append(digest)
    if any(
        contract != contracts[0] or digest != contract_digests[0]
        for contract, digest in zip(contracts[1:], contract_digests[1:])
    ):
        raise DatasetContractError("cannot collate mixed encoder_contract values")

    stacked_keys = (
        "source_latent",
        "global_target_latent",
        "source_component_mask",
        "target_component_mask",
    )
    result: dict[str, Any] = {}
    result["encoder_contract"] = contracts[0]
    result["encoder_contract_sha256"] = contract_digests[0]
    for key in stacked_keys:
        try:
            result[key] = torch.stack([sample[key] for sample in samples], dim=0)
        except RuntimeError as exc:
            raise DatasetContractError(
                f"batch has incompatible tensor shapes for {key}"
            ) from exc
    result["text_context"] = [sample["text_context"] for sample in samples]
    result["vlm_context"] = [sample["vlm_context"] for sample in samples]
    result["atom_id"] = [sample["atom_id"] for sample in samples]
    result["metadata"] = [sample["metadata"] for sample in samples]
    return result
