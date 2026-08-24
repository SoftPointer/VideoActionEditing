#!/usr/bin/env python3
"""Train the exact-81 source-self role/repaint engineering canary.

The one authorized positive is a raw real source video independently encoded
by the same pinned VAE as its appearance-corrupted donors and three source RGB
references.  The model never receives a paired or synthetic edited target.
The legacy default is DP2 x Ulysses-SP4: ranks 0--3 and 4--7 see different
source rows, different noise seeds and different registered donor styles.  An
explicit WORLD4 DP1 x SP4 profile preserves those same two logical arms by
running their controls first, then accumulating two half-scaled backwards
before one optimizer step.

The first launch is exactly one optimizer step at sigma=1 and rho=0.  Donor
DC/reverse/style1/style2, wrong-reference, reference-off, and the donor-order x
reference-identity 2x2 grid are no-gradient causal diagnostics only.  This
pretext canary cannot establish semantic action editing or motion preservation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_self_role_repaint as role  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
import train_lora as legacy  # noqa: E402


METHOD_NAME = "bernini-source-self-role-repaint-exact81-canary-v3"
DATASET_ROW_SCHEMA = "bernini-source-self-role-repaint-row-v2"
DATASET_RECEIPT_SCHEMA = "bernini-source-self-role-repaint-dataset-receipt-v2"
RUN_RECEIPT_SCHEMA = "bernini-source-self-role-repaint-training-receipt-v3"
HISTORY_SCHEMA = "bernini-source-self-role-repaint-step-history-v3"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
LOGICAL_ARM_COUNT = 2
FRAME_COUNT = 81
LATENT_PHASES = 21
REFERENCE_PHASES = 1
TIMESTEP = 1000
SIGMA = 1.0
DEFAULT_RHO = 0.0
CANARY_STEPS = 1
LORA_RANK = 8
LORA_ALPHA = 8.0
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260808
GENERIC_INSTRUCTION = (
    "Restore the original video appearance from the clean source references "
    "while following the ordered donor's temporal evolution and camera path."
)
SP_GROUP_RANKS = ((0, 1, 2, 3), (4, 5, 6, 7))
DP_GROUP_RANKS = ((0, 4), (1, 5), (2, 6), (3, 7))
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

POSTERIOR_FIELDS = (
    "clean_target_posterior_blob",
    "style1_donor_posterior_blob",
    "style2_donor_posterior_blob",
    "ref0_posterior_blob",
    "ref40_posterior_blob",
    "ref80_posterior_blob",
)


class SourceSelfTrainingError(RuntimeError):
    """Raised before an ambiguous optimizer step or output publication."""


def canonical_json_bytes(value: Any) -> bytes:
    return role.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return role.object_sha256(value)


def file_sha256(path: Path) -> str:
    return runtime.file_sha256(path)


def tensor_sha256(value: Any) -> str:
    return runtime.tensor_sha256(value)


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise SourceSelfTrainingError(f"{label} path must be non-empty text")
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise SourceSelfTrainingError(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise SourceSelfTrainingError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not stat.S_ISREG(mode) or resolved.is_symlink():
        raise SourceSelfTrainingError(f"{label} must be a canonical plain file")
    return resolved


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise SourceSelfTrainingError(f"{label} must be a lowercase SHA-{1 if length == 40 else 256}")
    return value


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SourceSelfTrainingError(f"{label} contains non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SourceSelfTrainingError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SourceSelfTrainingError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise SourceSelfTrainingError(f"{label} root must be an object")
    return value


@dataclass(frozen=True)
class MaterializedRow:
    iid: str
    source_video_sha256: str
    clean_blob: bytes
    style1_blob: bytes
    style2_blob: bytes
    refs: Mapping[int, bytes]
    reference_order: tuple[int, int, int]
    clean_shape: tuple[int, ...]
    row_digest: str


@dataclass(frozen=True)
class MaterializedDataset:
    root: Path
    parquet: Path
    parquet_sha256: str
    receipt: Path
    receipt_sha256: str
    receipt_digest: str
    rows: tuple[MaterializedRow, ...]


def _blob_sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _row_digest(row: Mapping[str, Any]) -> str:
    candidate = {
        key: value
        for key, value in row.items()
        if key not in POSTERIOR_FIELDS and key != "row_digest"
    }
    candidate["posterior_blob_sha256"] = {
        field: _blob_sha(bytes(row[field])) for field in POSTERIOR_FIELDS
    }
    return object_sha256(candidate)


def _load_posterior_parameters(blob: bytes, *, phases: int, label: str) -> Any:
    import torch

    try:
        value = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(io.BytesIO(blob), map_location="cpu")
    except Exception as error:
        raise SourceSelfTrainingError(f"cannot decode {label}: {error}") from error
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
        raise SourceSelfTrainingError(
            f"{label} must be detached contiguous FP32 [1,32,{phases},evenH,evenW]"
        )
    return value


def _load_dataset(root_value: str | Path, expected_spec_sha256: str) -> MaterializedDataset:
    root = Path(root_value).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise SourceSelfTrainingError("dataset root must be absolute and non-symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise SourceSelfTrainingError(f"dataset root is unavailable: {error}") from error
    if not root.is_dir() or root.is_symlink():
        raise SourceSelfTrainingError("dataset root must be a plain directory")
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != {"dataset.parquet", "receipt.json"}:
        raise SourceSelfTrainingError("dataset artifact closure differs")
    if any(not path.is_file() or path.is_symlink() for path in entries.values()):
        raise SourceSelfTrainingError("dataset contains a non-plain artifact")
    receipt_path = entries["receipt.json"]
    receipt_raw = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    receipt = _strict_json_bytes(receipt_raw, label="dataset receipt")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if (
        receipt.get("schema_version") != DATASET_RECEIPT_SCHEMA
        or object_sha256(unsigned) != declared
        or receipt.get("complete") is not True
        or receipt.get("references_independently_encoded_from_rgb") is not True
        or receipt.get("references_from_video_posterior_slice") is not False
        or receipt.get("independent_vae_encode_calls_per_row") != 6
        or receipt.get("all_six_calls_share_one_pinned_vae_identity") is not True
        or receipt.get("paired_dataset_accessed") is not False
        or receipt.get("prior_posterior_accessed") is not False
        or receipt.get("target_video_path_present") is not False
        or receipt.get("target_video_accessed") is not False
        or receipt.get("edited_target_accessed") is not False
        or receipt.get("synthetic_edited_target_present") is not False
        or receipt.get("action_supervision_present") is not False
        or receipt.get("semantic_motion_preservation_claimed") is not False
    ):
        raise SourceSelfTrainingError("dataset receipt scientific/role contract differs")
    expected_spec = _sha(expected_spec_sha256, length=64, label="materialization spec SHA")
    if not isinstance(receipt.get("spec"), Mapping) or receipt["spec"].get("sha256") != expected_spec:
        raise SourceSelfTrainingError("dataset receipt binds a different materialization spec")
    parquet = entries["dataset.parquet"]
    parquet_sha = file_sha256(parquet)
    dataset_record = receipt.get("dataset")
    if (
        not isinstance(dataset_record, Mapping)
        or Path(str(dataset_record.get("path"))).resolve(strict=True) != parquet
        or dataset_record.get("sha256") != parquet_sha
    ):
        raise SourceSelfTrainingError("dataset receipt does not bind parquet bytes")
    before = parquet.stat()
    try:
        import pyarrow.parquet as pq

        raw_rows = pq.read_table(parquet).to_pylist()
    except Exception as error:
        raise SourceSelfTrainingError(f"cannot read source-self parquet: {error}") from error
    after = parquet.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or file_sha256(parquet) != parquet_sha
    ):
        raise SourceSelfTrainingError("dataset parquet changed while reading")
    if len(raw_rows) < DP_SIZE or dataset_record.get("rows") != len(raw_rows):
        raise SourceSelfTrainingError("dataset requires at least two materialized rows")
    rows: list[MaterializedRow] = []
    seen_iids: set[str] = set()
    clean_shapes: set[tuple[int, ...]] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or raw.get("schema_version") != DATASET_ROW_SCHEMA:
            raise SourceSelfTrainingError("materialized row schema differs")
        iid = raw.get("iid")
        if type(iid) is not str or _IID.fullmatch(iid) is None or iid in seen_iids:
            raise SourceSelfTrainingError("materialized IID is invalid or duplicated")
        seen_iids.add(iid)
        if _row_digest(raw) != raw.get("row_digest"):
            raise SourceSelfTrainingError(f"{iid} row digest differs")
        if (
            raw.get("frame_count") != FRAME_COUNT
            or float(raw.get("fps", -1.0)) != 25.0
            or raw.get("independent_vae_encode_calls") != 6
            or raw.get("clean_target_encoded_from_same_raw_source_rgb") is not True
            or raw.get("clean_style_refs_share_one_pinned_vae_identity") is not True
            or raw.get("paired_dataset_accessed") is not False
            or raw.get("prior_posterior_accessed") is not False
            or raw.get("reference_from_video_posterior_slice") is not False
            or raw.get("edited_target_accessed") is not False
            or raw.get("action_label_present") is not False
            or raw.get("semantic_motion_preservation_claimed") is not False
        ):
            raise SourceSelfTrainingError(f"{iid} row role/scientific contract differs")
        try:
            indices = tuple(json.loads(raw["reference_indices_json"]))
            order = tuple(json.loads(raw["reference_order_json"]))
            clean_shape = tuple(json.loads(raw["clean_posterior_shape_json"]))
            metadata = json.loads(raw["independent_encode_metadata_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise SourceSelfTrainingError(f"{iid} row metadata is invalid: {error}") from error
        if (
            indices != role.REFERENCE_RGB_INDICES
            or tuple(sorted(order)) != role.REFERENCE_RGB_INDICES
            or not isinstance(metadata, Mapping)
            or len(metadata) != 6
        ):
            raise SourceSelfTrainingError(f"{iid} independent reference contract differs")
        blobs = {field: bytes(raw[field]) for field in POSTERIOR_FIELDS}
        clean = _load_posterior_parameters(
            blobs["clean_target_posterior_blob"], phases=LATENT_PHASES, label=f"{iid} clean target"
        )
        for field in ("style1_donor_posterior_blob", "style2_donor_posterior_blob"):
            donor = _load_posterior_parameters(blobs[field], phases=LATENT_PHASES, label=f"{iid} {field}")
            if tuple(donor.shape) != tuple(clean.shape):
                raise SourceSelfTrainingError(f"{iid} donor/target posterior geometry differs")
        refs = {
            index: blobs[f"ref{index}_posterior_blob"] for index in role.REFERENCE_RGB_INDICES
        }
        for index, blob in refs.items():
            ref = _load_posterior_parameters(blob, phases=REFERENCE_PHASES, label=f"{iid} ref {index}")
            if tuple(ref.shape[:2]) != tuple(clean.shape[:2]) or tuple(ref.shape[3:]) != tuple(clean.shape[3:]):
                raise SourceSelfTrainingError(f"{iid} ref/target spatial geometry differs")
        if tuple(clean.shape) != clean_shape:
            raise SourceSelfTrainingError(f"{iid} declared clean shape differs")
        clean_shapes.add(clean_shape)
        rows.append(
            MaterializedRow(
                iid=iid,
                source_video_sha256=_sha(raw.get("source_video_sha256"), length=64, label=f"{iid} source SHA"),
                clean_blob=blobs["clean_target_posterior_blob"],
                style1_blob=blobs["style1_donor_posterior_blob"],
                style2_blob=blobs["style2_donor_posterior_blob"],
                refs=refs,
                reference_order=order,
                clean_shape=clean_shape,
                row_digest=str(raw["row_digest"]),
            )
        )
    if len(clean_shapes) != 1:
        raise SourceSelfTrainingError("canary cohort must use one spatial bucket")
    if dataset_record.get("iids") != [item.iid for item in rows] or dataset_record.get("row_digests") != [item.row_digest for item in rows]:
        raise SourceSelfTrainingError("dataset receipt row membership differs")
    return MaterializedDataset(
        root=root,
        parquet=parquet,
        parquet_sha256=parquet_sha,
        receipt=receipt_path,
        receipt_sha256=receipt_sha,
        receipt_digest=str(declared),
        rows=tuple(rows),
    )


def _posterior_mode(blob: bytes, mean: Any, std: Any, *, phases: int, label: str) -> Any:
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    parameters = _load_posterior_parameters(blob, phases=phases, label=label)
    mode = DiagonalGaussianDistribution(parameters).mode().squeeze(0).float()
    mode = ((mode - mean) / std).detach().contiguous()
    if tuple(mode.shape[:2]) != (16, phases) or not bool(__import__("torch").isfinite(mode).all().item()):
        raise SourceSelfTrainingError(f"{label} normalized posterior mode differs")
    return mode


def pack_latent_patches(latent: Any, *, phases: int) -> Any:
    import torch

    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.ndim != 4
        or tuple(int(item) for item in latent.shape[:2]) != (16, phases)
        or int(latent.shape[2]) % 2
        or int(latent.shape[3]) % 2
    ):
        raise SourceSelfTrainingError(f"latent must be FP32 [16,{phases},evenH,evenW]")
    channels, frames, height, width = (int(item) for item in latent.shape)
    return (
        latent.reshape(channels, frames, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(frames * (height // 2) * (width // 2), channels, 1, 2, 2)
        .contiguous()
    )


def packed_output_field(patches: Any) -> Any:
    return runtime.packed_output_field(patches)


@dataclass(frozen=True)
class PreparedCondition:
    input_patches: Any
    rotary: Any
    layout: role.TokenRoleLayout


@dataclass(frozen=True)
class PreparedStep:
    main: PreparedCondition
    controls: Mapping[str, PreparedCondition]
    target_velocity: Any
    epsilon_sha256: str
    conditional_base_receipt: Mapping[str, Any]
    tensor_identities: Mapping[str, str]


@dataclass(frozen=True)
class PreparedLogicalArm:
    logical_arm: int
    row_index: int
    iid: str
    wrong_ref_iid: str
    main_style: int
    noise_seed: int
    main: PreparedCondition
    target_velocity: Any
    epsilon_sha256: str
    conditional_base_receipt: Mapping[str, Any]
    tensor_identities: Mapping[str, str]
    control_metrics: Mapping[str, float]
    control_names: tuple[str, ...]
    absent_reference_layout: Mapping[str, Any]


def _condition(
    donor: Any,
    references: Sequence[Any],
    noisy_target: Any,
    *,
    rope: Any,
    device: Any,
) -> PreparedCondition:
    import torch

    if len(references) not in {0, role.REFERENCE_COUNT}:
        raise SourceSelfTrainingError(
            "image references must be truly absent or exactly three modes"
        )
    donor_patches = pack_latent_patches(donor, phases=LATENT_PHASES)
    reference_patches = [pack_latent_patches(item, phases=REFERENCE_PHASES) for item in references]
    target_patches = pack_latent_patches(noisy_target, phases=LATENT_PHASES)
    layout = role.TokenRoleLayout.contiguous(
        donor_tokens=int(donor_patches.shape[0]),
        reference_tokens=[int(item.shape[0]) for item in reference_patches],
        target_tokens=int(target_patches.shape[0]),
    )
    patches = torch.cat((donor_patches, *reference_patches, target_patches), dim=0).to(device)
    donor_rope = rope(donor.unsqueeze(0).to(device), source_id=1)
    reference_rope = [
        rope(item.unsqueeze(0).to(device), source_id=index + 2)
        for index, item in enumerate(references)
    ]
    target_rope = rope(noisy_target.unsqueeze(0).to(device), source_id=0)
    rotary = torch.cat((donor_rope, *reference_rope, target_rope), dim=2)
    rotary = rotary.squeeze(0).permute(1, 0, 2).contiguous()
    if int(patches.shape[0]) != layout.total_tokens or int(rotary.shape[0]) != layout.total_tokens:
        raise SourceSelfTrainingError("donor+refs+target pack geometry differs")
    return PreparedCondition(patches, rotary, layout)


def _temporal_dc(donor: Any) -> Any:
    return donor.mean(dim=1, keepdim=True).expand_as(donor).clone().detach().contiguous()


def prepare_step(
    *,
    clean: Any,
    style1: Any,
    style2: Any,
    correct_refs: Sequence[Any],
    wrong_refs: Sequence[Any],
    main_style: int,
    epsilon: Any,
    rho: float,
    rope: Any,
    device: Any,
) -> PreparedStep:
    if main_style not in {1, 2}:
        raise SourceSelfTrainingError("main donor style must be 1 or 2")
    if len(correct_refs) != 3 or len(wrong_refs) != 3:
        raise SourceSelfTrainingError("correct/wrong reference triplets are required")
    source_batch = clean.unsqueeze(0).contiguous()
    epsilon_batch = epsilon.unsqueeze(0).contiguous()
    e_s, base_receipt = role.source_rich_conditional_base(
        epsilon_batch, source_batch, rho=rho
    )
    noisy = e_s.squeeze(0).detach().contiguous()  # sigma=1
    velocity = (e_s - source_batch).squeeze(0).detach().contiguous()
    target_velocity = packed_output_field(pack_latent_patches(velocity, phases=LATENT_PHASES)).to(device)
    ordered = style1 if main_style == 1 else style2
    reverse = role.reverse_donor_phases(ordered.unsqueeze(0)).squeeze(0).contiguous()
    factorial = {
        "ordered_correct_refs": (ordered, correct_refs),
        "ordered_wrong_refs": (ordered, wrong_refs),
        "reverse_correct_refs": (reverse, correct_refs),
        "reverse_wrong_refs": (reverse, wrong_refs),
    }
    controls: dict[str, PreparedCondition] = {
        name: _condition(donor, refs, noisy, rope=rope, device=device)
        for name, (donor, refs) in factorial.items()
    }
    controls.update(
        {
            "donor_dc_correct_refs": _condition(
                _temporal_dc(ordered), correct_refs, noisy, rope=rope, device=device
            ),
            "style1_correct_refs": _condition(style1, correct_refs, noisy, rope=rope, device=device),
            "style2_correct_refs": _condition(style2, correct_refs, noisy, rope=rope, device=device),
            "ordered_refs_absent": _condition(
                ordered, (), noisy, rope=rope, device=device
            ),
        }
    )
    main = controls["ordered_correct_refs"]
    for name, item in controls.items():
        if name != "ordered_refs_absent" and item.layout.receipt() != main.layout.receipt():
            raise SourceSelfTrainingError("three-reference control token layouts differ")
    absent = controls["ordered_refs_absent"].layout
    if (
        absent.reference_tokens != ()
        or absent.reference_token_total != 0
        or absent.condition_tokens != absent.donor_tokens
        or absent.total_tokens >= main.layout.total_tokens
    ):
        raise SourceSelfTrainingError("absent-reference control still contains ref tokens")
    identities = {
        "clean_target": tensor_sha256(clean),
        "style1_donor": tensor_sha256(style1),
        "style2_donor": tensor_sha256(style2),
        "epsilon": tensor_sha256(epsilon),
        "eS": tensor_sha256(e_s.squeeze(0)),
        "target_velocity": tensor_sha256(velocity),
        **{
            f"correct_ref_{index}": tensor_sha256(value)
            for index, value in enumerate(correct_refs)
        },
        **{
            f"wrong_ref_{index}": tensor_sha256(value)
            for index, value in enumerate(wrong_refs)
        },
    }
    if identities["clean_target"] in {
        identities["style1_donor"], identities["style2_donor"]
    }:
        raise SourceSelfTrainingError("clean target aliases an offline style donor")
    return PreparedStep(
        main=main,
        controls=controls,
        target_velocity=target_velocity,
        epsilon_sha256=identities["epsilon"],
        conditional_base_receipt=base_receipt,
        tensor_identities=identities,
    )


def _prediction(
    renderer: Any,
    transformer: Any,
    condition: PreparedCondition,
    *,
    text_lens: Any,
    text_embs: Any,
) -> Any:
    import torch

    embedded = transformer.patch_embedding(condition.input_patches).flatten(1).unsqueeze(0)
    rotary = condition.rotary.permute(1, 0, 2).unsqueeze(0)
    value = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=embedded,
        timesteps=embedded.new_tensor([TIMESTEP], dtype=torch.int64),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[condition.layout.total_tokens],
        batch_text_seqlen=text_lens,
    )
    start = condition.layout.condition_tokens
    target = value[:, start : start + condition.layout.target_tokens, :]
    if tuple(target.shape) != (1, condition.layout.target_tokens, role.PATCH_VALUES):
        raise SourceSelfTrainingError("target-row prediction geometry differs")
    return target


def _relative_l2(value: Any, reference: Any) -> float:
    numerator = (value.float() - reference.float()).square().mean().sqrt()
    denominator = reference.float().square().mean().sqrt().clamp_min(1.0e-12)
    result = float((numerator / denominator).item())
    if not math.isfinite(result):
        raise SourceSelfTrainingError("causal control relative L2 is non-finite")
    return result


def _atomic_adapter_safetensors(
    path: Path,
    adapter: role.SourceSelfAdapterHandle,
    *,
    conditional_base_rho: float,
) -> None:
    import torch
    from safetensors.torch import save_file

    tensors = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in adapter.trainable_named_parameters()
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            tensors,
            str(temporary),
            metadata=dict(
                adapter.safetensors_metadata(
                    conditional_base_rho=conditional_base_rho
                )
            ),
        )
        runtime.durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expected-materialization-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("engineering-canary",))
    parser.add_argument(
        "--parallel-topology",
        choices=tuple(runtime.PARALLEL_TOPOLOGIES),
        default=runtime.WORLD8_DP2_SP4.profile,
        help="Explicit physical topology; WORLD8 remains the legacy default.",
    )
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--adapter-block-scope", choices=("early-mid-0-22", "all30-ablation"), default="early-mid-0-22")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument(
        "--method-source-revision-kind",
        choices=("git-commit", "content-closure-sha1"),
        default="git-commit",
    )
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--method-source-manifest-sha256")
    parser.add_argument("--ack-upstream-training-use-forbidden", action="store_true")
    parser.add_argument("--num-frames", type=int, choices=(FRAME_COUNT,), default=FRAME_COUNT)
    return parser


def validate_cli(args: argparse.Namespace) -> tuple[int, ...]:
    if (
        (WORLD_SIZE, SP_SIZE, DP_SIZE)
        != (runtime.WORLD_SIZE, runtime.SP_SIZE, runtime.DP_SIZE)
        or SP_GROUP_RANKS != runtime.SP_GROUP_RANKS
        or DP_GROUP_RANKS != runtime.DP_GROUP_RANKS
    ):
        raise SourceSelfTrainingError(
            "trainer/runtime DP2 x SP4 topology constants differ"
        )
    if args.mode != "engineering-canary" or args.num_frames != FRAME_COUNT:
        raise SourceSelfTrainingError("first source-self launch is exact81 one-step only")
    runtime.parallel_topology(args.parallel_topology)
    if args.ack_upstream_training_use_forbidden is not True:
        raise SourceSelfTrainingError("--ack-upstream-training-use-forbidden is mandatory")
    if args.rho != 0.0:
        raise SourceSelfTrainingError("engineering canary is frozen to rho=0 standard Gaussian")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SourceSelfTrainingError("learning rate must be finite and positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise SourceSelfTrainingError("max grad norm must be finite and positive")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or not 0 <= args.seed < 2**63:
        raise SourceSelfTrainingError("seed must lie in [0,2^63)")
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_materialization_spec_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if args.method_source_manifest_sha256 is not None:
        _sha(
            args.method_source_manifest_sha256,
            length=64,
            label="method_source_manifest_sha256",
        )
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise SourceSelfTrainingError("checkpoint tree differs from audited Bernini 1.3B")
    return (
        role.TRAINABLE_BLOCK_INDICES
        if args.adapter_block_scope == "early-mid-0-22"
        else tuple(range(role.TOTAL_BLOCKS_1P3B))
    )


def _noise_seed(base_seed: int, step: int, row_index: int, dp_rank: int) -> int:
    raw = f"{base_seed}\0source-self\0{step}\0{row_index}\0{dp_rank}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31)


def logical_arms_for_topology(
    topology: runtime.ParallelTopology, physical_dp_rank: int
) -> tuple[int, ...]:
    """Preserve the two-arm objective under parallel or serial placement."""

    if topology == runtime.WORLD8_DP2_SP4:
        if not 0 <= physical_dp_rank < LOGICAL_ARM_COUNT:
            raise SourceSelfTrainingError("WORLD8 physical DP rank is invalid")
        return (physical_dp_rank,)
    if topology == runtime.WORLD4_DP1_SP4:
        if physical_dp_rank != 0:
            raise SourceSelfTrainingError("WORLD4 DP1 physical rank must be zero")
        return tuple(range(LOGICAL_ARM_COUNT))
    raise SourceSelfTrainingError("unsupported physical topology")


def logical_loss_scale(topology: runtime.ParallelTopology) -> float:
    """Scale serial arms so both profiles optimize the same two-arm mean."""

    if LOGICAL_ARM_COUNT % topology.dp_size:
        raise SourceSelfTrainingError("logical arms cannot be balanced over physical DP")
    return float(topology.dp_size) / float(LOGICAL_ARM_COUNT)


def placement_receipt(contract: runtime.DistributedContract) -> dict[str, Any]:
    topology = contract.topology
    if topology.world_size % contract.local_world_size:
        raise SourceSelfTrainingError("world/local placement is not integral")
    nodes = topology.world_size // contract.local_world_size
    return {
        "nodes": nodes,
        "local_world_size": contract.local_world_size,
        "ranks_per_node": contract.local_world_size,
        "sp4_crosses_nodes": topology.sp_size > contract.local_world_size,
        "preferred_world4_placement": (
            topology == runtime.WORLD4_DP1_SP4
            and nodes == 2
            and contract.local_world_size == 2
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    block_indices = validate_cli(args)
    dataset = _load_dataset(args.dataset_root, args.expected_materialization_spec_sha256)
    # ``pyarrow`` may retain its temporary parquet allocation after the exact
    # two-row table has been converted into immutable Python row objects.  Do
    # not carry that allocator cache into four replicated renderer loads.
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass
    gc.collect()
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise SourceSelfTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise SourceSelfTrainingError("pinned Bernini attention-head count differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state

    topology = runtime.parallel_topology(args.parallel_topology)
    contract = runtime.distributed_contract(topology=topology)
    device = runtime.initialise_distributed(contract)
    parallel = runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=topology.sp_size)
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    local_logical_arms = logical_arms_for_topology(topology, contract.arm_index)
    physical_placement = placement_receipt(contract)
    runtime.digest_consensus(
        object_sha256(physical_placement),
        group=parallel.world_group,
        expected_count=topology.world_size,
        label="source-self physical placement",
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
    transformer = renderer.diff_dec.transformer
    if transformer is None or renderer.diff_dec.transformer_2 is not None:
        raise SourceSelfTrainingError("source-self requires only Bernini transformer_1")
    renderer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": role.checkpoint_route_context_fn,
        }
    )
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        raise SourceSelfTrainingError("source-self requires non-reentrant checkpointing")
    adapter = role.install_source_self_adapter(
        transformer,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        block_indices=block_indices,
    )
    trainable = adapter.trainable_named_parameters()
    if not adapter.base_parameters_frozen():
        raise SourceSelfTrainingError("Bernini base changed trainability after adapter install")
    initial_digest = runtime.synchronize_initial_parameters(
        trainable,
        parallel.world_group,
        expected_count=topology.world_size,
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
    text = runtime.tokenize_generic_instruction(
        tokenizer, GENERIC_INSTRUCTION, device
    )
    text_digest = object_sha256(
        {name: tensor_sha256(value) for name, value in sorted(text.items())}
    )
    runtime.digest_consensus(
        text_digest,
        group=parallel.world_group,
        expected_count=topology.world_size,
        label="source-self generic text",
    )
    with torch.inference_mode():
        text_lens, text_embs = renderer.get_t5_text_embeddings(
            text["input_ids"], text["attention_mask"], text["t5_input_lens"]
        )
    if getattr(text_embs, "requires_grad", False):
        raise SourceSelfTrainingError("frozen T5 embeddings unexpectedly require gradients")
    # The generic instruction is encoded once.  Moving the frozen T5 module
    # to CPU would keep one full copy resident in each rank's 56 GiB Slurm
    # cgroup; release it instead.  The detached text embeddings/lens are the
    # only values consumed by all subsequent renderer calls.
    renderer.t5_text_encoder = None
    del tokenizer, text
    gc.collect()
    torch.cuda.empty_cache()
    if renderer.t5_text_encoder is not None:
        raise SourceSelfTrainingError("frozen T5 encoder was not released")
    vae_mean, vae_std, _ = legacy._vae_statistics(checkpoint)
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    prepared_arms: list[PreparedLogicalArm] = []
    # Every control for every local logical arm precedes every graph-bearing
    # forward.  WORLD4 therefore retains the strong no-gradient control
    # contract while serializing the two logical DP arms.
    for logical_arm in local_logical_arms:
        local_index = logical_arm % len(dataset.rows)
        wrong_index = (local_index + 1) % len(dataset.rows)
        local_row = dataset.rows[local_index]
        wrong_row = dataset.rows[wrong_index]
        if (
            local_row.iid == wrong_row.iid
            or local_row.clean_shape != wrong_row.clean_shape
        ):
            raise SourceSelfTrainingError(
                "wrong-reference row must be distinct with the same bucket"
            )
        clean = _posterior_mode(
            local_row.clean_blob,
            vae_mean,
            vae_std,
            phases=LATENT_PHASES,
            label=f"{local_row.iid} clean",
        )
        style1 = _posterior_mode(
            local_row.style1_blob,
            vae_mean,
            vae_std,
            phases=LATENT_PHASES,
            label=f"{local_row.iid} style1",
        )
        style2 = _posterior_mode(
            local_row.style2_blob,
            vae_mean,
            vae_std,
            phases=LATENT_PHASES,
            label=f"{local_row.iid} style2",
        )
        if clean.data_ptr() in {style1.data_ptr(), style2.data_ptr()}:
            raise SourceSelfTrainingError(
                "clean target must be an independent tensor object"
            )
        correct_ref_map = {
            index: _posterior_mode(
                local_row.refs[index],
                vae_mean,
                vae_std,
                phases=REFERENCE_PHASES,
                label=f"{local_row.iid} ref{index}",
            )
            for index in role.REFERENCE_RGB_INDICES
        }
        wrong_ref_map = {
            index: _posterior_mode(
                wrong_row.refs[index],
                vae_mean,
                vae_std,
                phases=REFERENCE_PHASES,
                label=f"{wrong_row.iid} wrong-ref{index}",
            )
            for index in role.REFERENCE_RGB_INDICES
        }
        correct_refs = [
            correct_ref_map[index] for index in local_row.reference_order
        ]
        wrong_refs = [
            wrong_ref_map[index] for index in local_row.reference_order
        ]
        if any(
            a.data_ptr() == b.data_ptr()
            for a in correct_refs
            for b in wrong_refs
        ):
            raise SourceSelfTrainingError(
                "correct and wrong references alias storage"
            )
        seed = _noise_seed(args.seed, step, local_index, logical_arm)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        epsilon = torch.randn(
            tuple(clean.shape), generator=generator, dtype=torch.float32
        ).contiguous()
        prepared = prepare_step(
            clean=clean,
            style1=style1,
            style2=style2,
            correct_refs=correct_refs,
            wrong_refs=wrong_refs,
            main_style=logical_arm + 1,
            epsilon=epsilon,
            rho=args.rho,
            rope=rope,
            device=device,
        )
        control_predictions: dict[str, Any] = {}
        for name, condition in prepared.controls.items():
            control_invocation = role.RouteInvocation(
                condition.layout,
                sequence_parallel_rank=contract.sp_rank,
                sequence_parallel_size=topology.sp_size,
            )
            with adapter.route(control_invocation):
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    control_predictions[name] = _prediction(
                        renderer,
                        transformer,
                        condition,
                        text_lens=text_lens,
                        text_embs=text_embs,
                    ).detach()
            if any(parameter.grad is not None for _, parameter in trainable):
                raise SourceSelfTrainingError(
                    "no-gradient causal control touched adapter gradients"
                )
        baseline = control_predictions["ordered_correct_refs"]
        control_metrics = {
            name: _relative_l2(value, baseline)
            for name, value in control_predictions.items()
            if name != "ordered_correct_refs"
        }
        prepared_arms.append(
            PreparedLogicalArm(
                logical_arm=logical_arm,
                row_index=local_index,
                iid=local_row.iid,
                wrong_ref_iid=wrong_row.iid,
                main_style=logical_arm + 1,
                noise_seed=seed,
                main=prepared.main,
                target_velocity=prepared.target_velocity,
                epsilon_sha256=prepared.epsilon_sha256,
                conditional_base_receipt=prepared.conditional_base_receipt,
                tensor_identities=prepared.tensor_identities,
                control_metrics=control_metrics,
                control_names=tuple(sorted(prepared.controls)),
                absent_reference_layout={
                    **prepared.controls[
                        "ordered_refs_absent"
                    ].layout.receipt(),
                    "reference_source_ids": [],
                },
            )
        )
        del control_predictions, baseline, prepared

    if any(parameter.grad is not None for _, parameter in trainable):
        raise SourceSelfTrainingError(
            "all-arm no-gradient controls must precede clean backward state"
        )
    conditional_digests = {
        object_sha256(dict(item.conditional_base_receipt))
        for item in prepared_arms
    }
    if len(conditional_digests) != 1:
        raise SourceSelfTrainingError(
            "logical arms use different conditional-base contracts"
        )
    runtime.digest_consensus(
        next(iter(conditional_digests)),
        group=parallel.world_group,
        expected_count=topology.world_size,
        label="source-self conditional base",
    )
    optimizer.zero_grad(set_to_none=True)
    loss_scale = logical_loss_scale(topology)
    losses: dict[int, float] = {}
    # Only after all controls have completed do graph-bearing logical arms run.
    # Each route encloses its own forward/backward so checkpoint recomputation
    # sees the exact corresponding role layout.
    for logical in prepared_arms:
        main_invocation = role.RouteInvocation(
            logical.main.layout,
            sequence_parallel_rank=contract.sp_rank,
            sequence_parallel_size=topology.sp_size,
        )
        with adapter.route(main_invocation):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = _prediction(
                    renderer,
                    transformer,
                    logical.main,
                    text_lens=text_lens,
                    text_embs=text_embs,
                )
                loss = torch.nn.functional.mse_loss(
                    prediction.float(),
                    logical.target_velocity.float(),
                    reduction="mean",
                )
            if not runtime.world_all_true(
                bool(torch.isfinite(loss.detach()).item()),
                group=parallel.world_group,
            ):
                raise SourceSelfTrainingError(
                    "non-finite source-self loss blocked optimizer step"
                )
            scaled_loss = loss * loss_scale
            scaled_loss.backward()
            losses[logical.logical_arm] = float(loss.detach().item())

    preclip_norm = runtime.synchronize_gradients(trainable, parallel)
    clipped = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in trainable], args.max_grad_norm
    )
    if not math.isfinite(float(clipped)):
        raise SourceSelfTrainingError(
            "gradient clipping produced a non-finite norm"
        )
    optimizer.step()
    final_digest = runtime.parameter_consensus(
        trainable,
        parallel.world_group,
        "source-self final adapter",
        expected_count=topology.world_size,
    )

    local_records: list[dict[str, Any]] = []
    for logical in prepared_arms:
        local_record = {
            "schema_version": HISTORY_SCHEMA,
            "step": 1,
            "logical_arm": logical.logical_arm,
            "physical_dp_rank": contract.arm_index,
            "sp_rank": contract.sp_rank,
            "row_index": logical.row_index,
            "iid": logical.iid,
            "wrong_ref_iid": logical.wrong_ref_iid,
            "main_style": logical.main_style,
            "noise_seed": logical.noise_seed,
            "epsilon_sha256": logical.epsilon_sha256,
            "flow_matching_loss_unscaled": losses[logical.logical_arm],
            "logical_loss_scale": loss_scale,
            "preclip_gradient_norm_logical_two_arm_mean": preclip_norm,
            "causal_control_relative_l2_vs_ordered_correct_refs": dict(
                logical.control_metrics
            ),
            "factorial_cells": [
                cell.cell_id for cell in role.heldout_factorial_cells()
            ],
            "all_controls_no_gradient": True,
            "all_logical_controls_preceded_any_backward": True,
            "parameter_sha256_after_logical_mean_step": final_digest,
        }
        sp_projection = {
            key: value for key, value in local_record.items() if key != "sp_rank"
        }
        runtime.digest_consensus(
            object_sha256(sp_projection),
            group=parallel.sp_group,
            expected_count=topology.sp_size,
            label=f"source-self logical arm {logical.logical_arm} record",
        )
        local_records.append(local_record)

    gathered: list[Any] = [None] * topology.world_size
    dist.all_gather_object(gathered, local_records, group=parallel.world_group)
    history: list[dict[str, Any]] = []
    for sp_group in topology.sp_group_ranks:
        leader_records = gathered[sp_group[0]]
        if not isinstance(leader_records, list):
            raise SourceSelfTrainingError("SP leader history is not a list")
        history.extend(leader_records)
    history.sort(key=lambda item: item["logical_arm"])
    if (
        len(history) != LOGICAL_ARM_COUNT
        or {item["logical_arm"] for item in history}
        != set(range(LOGICAL_ARM_COUNT))
        or len({item["iid"] for item in history}) != LOGICAL_ARM_COUNT
        or len({item["noise_seed"] for item in history}) != LOGICAL_ARM_COUNT
        or {item["main_style"] for item in history} != {1, 2}
    ):
        raise SourceSelfTrainingError(
            "two logical rows/noise seeds/styles are not distinct and complete"
        )

    dist.barrier(group=parallel.world_group)
    receipt: Optional[dict[str, Any]] = None
    rank_zero_publication_error: Optional[str] = None
    if contract.rank == 0:
        try:
            receipt_arm = prepared_arms[0]
            adapter_path = stage / "adapter.safetensors"
            optimizer_path = stage / "optimizer.pt"
            history_path = stage / "history.json"
            _atomic_adapter_safetensors(
                adapter_path, adapter, conditional_base_rho=args.rho
            )
            runtime.atomic_torch_save(
                optimizer_path,
                {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "optimizer": optimizer.state_dict(),
                "global_step": CANARY_STEPS,
                "adapter_sha256": final_digest,
                },
            )
            history_value = {
                "schema_version": HISTORY_SCHEMA,
                "steps": CANARY_STEPS,
                "logical_records": history,
            }
            runtime.atomic_json(history_path, history_value)
            receipt = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "mode": "engineering-canary",
            "optimizer_steps": CANARY_STEPS,
            "frame_count": FRAME_COUNT,
            "latent_phases": LATENT_PHASES,
            "sigma": SIGMA,
            "timestep": TIMESTEP,
            "rho": args.rho,
            "noisy_target_equation": "x_sigma=(1-sigma)*z+sigma*eS; sigma=1",
            "target_velocity_equation": "v*=eS-z",
            "dataset": {
                "root": str(dataset.root),
                "parquet_sha256": dataset.parquet_sha256,
                "receipt_sha256": dataset.receipt_sha256,
                "receipt_digest": dataset.receipt_digest,
                "rows": len(dataset.rows),
                "positive_target_role": (
                    "independent_pinned_vae_encode_of_same_raw_clean_source_rgb"
                ),
                "paired_dataset_accessed": False,
                "prior_posterior_accessed": False,
                "target_video_path_present": False,
                "independent_vae_encode_calls_per_row": 6,
                "edited_target_accessed": False,
                "action_supervision_present": False,
            },
            "visual_pack": {
                **receipt_arm.main.layout.receipt(),
                "source_ids": {
                    "ordered_donor": 1,
                    "reference_in_preregistered_order": [2, 3, 4],
                    "noisy_target": 0,
                },
                "reference_rgb_indices": list(role.REFERENCE_RGB_INDICES),
                "reference_order_per_iid": True,
                "reference_from_video_posterior_slice": False,
                "clean_target_is_not_model_input": True,
            },
            "adapter": dict(adapter.receipt()),
            "trainable_scope_exact": "explicit_role_embedding+early_mid_target_row_attn1_Q_O_LoRA",
            "initial_adapter_sha256": initial_digest,
            "final_adapter_sha256": final_digest,
            "base_frozen": True,
            "key_value_frozen": True,
            "cross_attention_frozen": True,
            "late_blocks_frozen": args.adapter_block_scope == "early-mid-0-22",
            "vae_frozen_and_absent_from_training_process": True,
            "t5_frozen": True,
            "t5_released_after_one_frozen_embedding": True,
            "distributed": {
                "profile": topology.profile,
                "world_size": topology.world_size,
                "physical_data_parallel_size": topology.dp_size,
                "ulysses_sequence_parallel_size": topology.sp_size,
                "sp_groups": [list(item) for item in topology.sp_group_ranks],
                "dp_groups": [list(item) for item in topology.dp_group_ranks],
                "placement": physical_placement,
                "logical_arm_count": LOGICAL_ARM_COUNT,
                "logical_arms_per_physical_dp_rank": {
                    str(dp_rank): list(
                        logical_arms_for_topology(topology, dp_rank)
                    )
                    for dp_rank in range(topology.dp_size)
                },
                "serial_logical_arm_accumulation": topology.dp_size == 1,
                "all_logical_controls_precede_any_backward": True,
                "logical_loss_scale_per_local_backward": loss_scale,
                "logical_objective": "mean(logical_arm_0,logical_arm_1)",
                "different_logical_rows": True,
                "different_logical_noise_seeds": True,
                "different_logical_donor_styles": True,
                "gradient_sync": [
                    f"SP{topology.sp_size}_all_reduce_sum_then_divide_by_"
                    f"{topology.sp_size}",
                    *(
                        [
                            f"DP{topology.dp_size}_all_reduce_sum_then_divide_by_"
                            f"{topology.dp_size}"
                        ]
                        if topology.dp_size > 1
                        else []
                    ),
                ],
                "dp_all_reduce_skipped_for_dp1": topology.dp_size == 1,
                "adapter_gradients_logical_two_arm_mean": True,
                "parameter_consensus_across_physical_world": True,
                "one_optimizer_step_after_all_local_logical_backwards": True,
                "checkpoint_recomputation_role_context_preserved": True,
            },
            "controls": {
                "optimizer_supervision": "none",
                "cells": list(receipt_arm.control_names),
                "factorial_2x2": [cell.cell_id for cell in role.heldout_factorial_cells()],
                "same_sigma_noisy_target_within_logical_sample": True,
                "correct_reverse_donor": True,
                "correct_wrong_reference": True,
                "donor_dc": True,
                "registered_style1_style2": True,
                "references_truly_absent": True,
                "absent_reference_control": dict(
                    receipt_arm.absent_reference_layout
                ),
                "metrics_in_history": True,
            },
            "conditional_base": dict(receipt_arm.conditional_base_receipt),
            "rho0_standard_gaussian_default": True,
            "rho_gt_zero_launched": False,
            "rho_gt_zero_requires_separate_checkpoint_and_matched_inference": True,
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
            },
            "artifacts": {
                "adapter.safetensors": file_sha256(adapter_path),
                "optimizer.pt": file_sha256(optimizer_path),
                "history.json": file_sha256(history_path),
            },
            "upstream_training_use_forbidden_acknowledged": True,
            "pretext_training_only": True,
            "semantic_motion_preservation_claimed": False,
            "natural_semantic_action_learned": False,
            "action_editing_claim_authorized": False,
            "video_quality_claim_authorized": False,
            "scientific_claim_authorized": False,
            "long_training_scientific_gate_passed": False,
            "long_training_automatically_submitted": False,
            "method_source_revision": args.method_source_revision,
            "method_source_revision_kind": args.method_source_revision_kind,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "method_source_manifest_sha256": args.method_source_manifest_sha256,
        }
            receipt["receipt_digest"] = object_sha256(receipt)
            runtime.atomic_json(stage / "receipt.json", receipt)
            runtime.verify_staged_run_bundle(stage, receipt)
            runtime.fsync_directory(stage)
        except Exception as error:
            rank_zero_publication_error = f"{type(error).__name__}: {error}"
    runtime.publish_output_transaction(
        output,
        stage,
        receipt,
        contract.rank,
        parallel.world_group,
        rank_zero_error=rank_zero_publication_error,
    )
    if contract.rank == 0:
        print(json.dumps({"output": str(output), "adapter_sha256": final_digest}, sort_keys=True), flush=True)
    adapter.restore()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
