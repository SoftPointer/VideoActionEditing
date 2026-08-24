#!/usr/bin/env python3
"""Fail-closed contract for the R64 held-out preservation decode.

This contract is deliberately narrower than action-editing evaluation.  It
binds the completed generic Stage-R64 receipt/checkpoint to the eight real
``source-only-v3`` held-out rows and permits exactly two native RV2V arms per
row: a carrier-disabled frozen-base arm and the trained carrier arm.  Both
arms use the same source, no-op instruction, seed and official Gaussian.

No action representation, action prompt, target video, reward, scorer,
ranking or checkpoint selection is admitted by this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import json
import math
from pathlib import Path
import re
import sys
import types
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-generic-source-carrier-r64-heldout-contract-v1"
RECEIPT_SCHEMA = "bernini-generic-source-carrier-r64-heldout-receipt-v1"
TRAINING_RECEIPT_SCHEMA = (
    "bernini-generic-source-anchored-action-training-receipt-v1"
)
CHECKPOINT_SCHEMA = "bernini-generic-source-anchored-action-checkpoint-v1"
TRAINING_METHOD = "bernini-generic-source-anchored-action-v1"
SOURCE_MANIFEST_SHA256 = (
    "128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d"
)
RAW_PARQUET_SHA256 = (
    "706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b"
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
BERNINI_MODEL_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
ROUTE_SCHEMA = "bernini-clean-source-visual-context-decode-route-v1"
ADAPTER_SCHEMA = "bernini-clean-source-visual-context-adapter-v1"
MEMORY_SCHEMA = "bernini-clean-source-visual-memory-v1"
TRANSFORMER_CONFIG_DIGEST = (
    "3802225b715e939064da7270705031a201c4585dbe1f4d3bfa37afe0be1d475b"
)
NATIVE_BRANCH_ORDER = ("none_uncond", "V_uncond", "VI_uncond", "VI_cond")
ADAPTER_BLOCK_INDICES = (8, 12, 16, 20)
RAW_SAFE_COLUMNS = (
    "iid", "group_id", "family", "source_video_path",
    "source_video_declared_path", "source_video_sha256",
)
R64_CHECKPOINT_SHA256 = (
    "b037496df99ea01d5a7e3fa509aac4c451806a6e47ecb7a1070529abde249726"
)
R64_TRAINING_RECEIPT_SHA256 = (
    "0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"
)
R64_TRAINING_RECEIPT_DIGEST = (
    "1632e4ed9b9c4375a8abf129fd4829f49995eb3c02f1b63a07fdca6aa67d08e5"
)
SOURCE_MANIFEST_DIGEST = (
    "3d553eb18168d0e59ecf1a79c312e240434edb01c84146f8a7c2d1afa17dc665"
)
R64_CARRIER_INITIAL_SHA256 = (
    "2a0ea8870310774614ced5b43609e026fb41b1bea57165d0663955b4ce4273bb"
)
R64_CARRIER_FINAL_SHA256 = (
    "144a13deb91bba460419de419a6dd9ac5362422d2f3230947a6f96351fa0dee3"
)
ARMS = ("frozen-base", "trained-carrier-r64")
FRAME_COUNT = 81
FPS = 25
NUM_INFERENCE_STEPS = 40
WORLD_SIZE = 4
SP_SIZE = 4
HELDOUT_ROWS = 8
MEDIA_ROWS = HELDOUT_ROWS * len(ARMS)
GENERIC_NOOP_INSTRUCTION = (
    "Reconstruct the input source video without editing it. Preserve every "
    "person, animal, object, identity, appearance, scene detail, camera motion, "
    "framing, timing, and action exactly as shown in the source. Do not add, "
    "remove, replace, restyle, or change anything."
)
GENERIC_NOOP_SHA256 = hashlib.sha256(
    GENERIC_NOOP_INSTRUCTION.encode("utf-8")
).hexdigest()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IID = re.compile(r"[0-9a-f]{16}\Z")
RELEASE_PREPROCESSING_TOOL_SHA256 = {
    "tools.build_renderer_dataset": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
    ),
    "tools.materialize_vae": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
    ),
}


class R64HeldoutContractError(RuntimeError):
    """Raised before ambiguous evidence can be published."""


def fail(message: str) -> NoReturn:
    raise R64HeldoutContractError(message)


def bind_release_preprocessing_tools(method_root: Path) -> Mapping[str, str]:
    """Bind lazy source-video preprocessing imports to this exact release.

    Several audited legacy helpers intentionally import ``tools.materialize_vae``
    only when a source video is decoded.  A bare ``from tools import ...`` can
    otherwise resolve through an unrelated namespace package.  Bind one private
    package search location before any model-facing module is imported, then
    authenticate both members needed by the actual source preparation path.
    """

    try:
        root = method_root.resolve(strict=True)
        tools_root = (root / "tools").resolve(strict=True)
    except OSError as error:
        raise R64HeldoutContractError(
            "release preprocessing-tools root is unavailable"
        ) from error
    if (
        not method_root.is_absolute()
        or root != method_root
        or method_root.is_symlink()
        or not method_root.is_dir()
        or tools_root != root / "tools"
        or (root / "tools").is_symlink()
        or not tools_root.is_dir()
    ):
        fail("release preprocessing-tools root is not one canonical directory")

    expected = {
        "tools.build_renderer_dataset": tools_root / "build_renderer_dataset.py",
        "tools.materialize_vae": tools_root / "materialize_vae.py",
    }
    for label, path in expected.items():
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            fail(f"release preprocessing member differs: {label}")

    package = sys.modules.get("tools")
    if package is None:
        package = types.ModuleType("tools")
        package.__package__ = "tools"
        package.__path__ = [str(tools_root)]
        specification = importlib.machinery.ModuleSpec(
            "tools", loader=None, is_package=True
        )
        specification.submodule_search_locations = [str(tools_root)]
        package.__spec__ = specification
        sys.modules["tools"] = package
    else:
        try:
            locations = tuple(
                Path(value).resolve(strict=True)
                for value in getattr(package, "__path__", ())
            )
        except (OSError, TypeError, ValueError) as error:
            raise R64HeldoutContractError(
                "preloaded tools package search path is ambiguous"
            ) from error
        if locations != (tools_root,):
            fail("preloaded tools package is not the exact heldout release package")

    identities: dict[str, str] = {}
    for label, expected_path in expected.items():
        try:
            module = importlib.import_module(label)
            origin = Path(module.__file__).resolve(strict=True)
        except (AttributeError, ImportError, OSError, TypeError) as error:
            raise R64HeldoutContractError(
                f"cannot import exact release preprocessing member: {label}"
            ) from error
        if origin != expected_path:
            fail(f"release preprocessing import origin differs: {label}")
        identity = file_sha256(origin)
        if identity != RELEASE_PREPROCESSING_TOOL_SHA256[label]:
            fail(f"release preprocessing member SHA-256 differs: {label}")
        identities[label] = identity
    return identities


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise R64HeldoutContractError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise R64HeldoutContractError(f"{label} is unavailable") from error
    if resolved != requested or not requested.is_file() or requested.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return requested


def _absolute_declared_path(value: Any, *, label: str) -> Path:
    """Admit a sealed remote path lexically when local byte checks are disabled."""

    if not isinstance(value, str) or not value or "\x00" in value:
        fail(f"{label} must be one absolute declared path")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or any(
        part in ("", ".", "..") for part in path.parts[1:]
    ):
        fail(f"{label} must be one absolute declared path")
    return path


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R64HeldoutContractError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def heldout_seed(iid: str) -> int:
    if not isinstance(iid, str) or _IID.fullmatch(iid) is None:
        fail("held-out IID must be sixteen lowercase hex digits")
    payload = f"20260814\0generic-source-carrier-r64-heldout-v1\0{iid}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


@dataclass(frozen=True)
class R64Authority:
    receipt_path: Path
    receipt_file_sha256: str
    receipt_digest: str
    checkpoint_path: Path
    checkpoint_file_sha256: str
    source_manifest_path: Path
    source_manifest_file_sha256: str
    source_manifest_digest: str
    carrier_initial_sha256: str
    carrier_final_sha256: str
    checkpoint_tree_sha256: str
    checkpoint_content_manifest_sha256: str

    def as_receipt(self) -> Mapping[str, Any]:
        return {
            "training_receipt": str(self.receipt_path),
            "training_receipt_file_sha256": self.receipt_file_sha256,
            "training_receipt_digest": self.receipt_digest,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "source_manifest": str(self.source_manifest_path),
            "source_manifest_file_sha256": self.source_manifest_file_sha256,
            "source_manifest_digest": self.source_manifest_digest,
            "carrier_initial_sha256": self.carrier_initial_sha256,
            "carrier_final_sha256": self.carrier_final_sha256,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
            "checkpoint_content_manifest_sha256": (
                self.checkpoint_content_manifest_sha256
            ),
        }


def load_r64_authority(
    receipt_value: str | Path,
    *,
    expected_receipt_sha256: str = R64_TRAINING_RECEIPT_SHA256,
    expected_checkpoint_sha256: str = R64_CHECKPOINT_SHA256,
    expected_source_manifest_sha256: str = SOURCE_MANIFEST_SHA256,
    verify_files: bool = True,
) -> R64Authority:
    """Validate the completed R64 receipt without granting an action claim."""

    expected_receipt_sha256 = _sha(expected_receipt_sha256, label="R64 receipt SHA")
    expected_checkpoint_sha256 = _sha(
        expected_checkpoint_sha256, label="R64 checkpoint SHA"
    )
    expected_source_manifest_sha256 = _sha(
        expected_source_manifest_sha256, label="source manifest SHA"
    )
    receipt_path = _plain_file(receipt_value, label="R64 training receipt")
    if file_sha256(receipt_path) != expected_receipt_sha256:
        fail("R64 training receipt bytes differ")
    value = _strict_json(receipt_path, label="R64 training receipt")
    unsigned = dict(value)
    receipt_digest = _sha(
        unsigned.pop("receipt_digest", None), label="R64 receipt digest"
    )
    checkpoint = value.get("checkpoint")
    data = value.get("data")
    objective = value.get("objective")
    distributed = value.get("distributed")
    model = value.get("model")
    counts = value.get("parameter_counts")
    initial = value.get("initial_component_sha256")
    final = value.get("final_component_sha256")
    terminal = value.get("terminal_toctou_audit")
    pair = value.get("pair_invariants")
    if (
        object_sha256(unsigned) != receipt_digest
        or receipt_digest != R64_TRAINING_RECEIPT_DIGEST
        or value.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or value.get("method") != TRAINING_METHOD
        or value.get("complete") is not True
        or value.get("experiment") != "joint_source_anchored_v1"
        or value.get("execution_profile") != "stage-r64"
        or value.get("stage_r_complete") is not True
        or value.get("stage_r_updates") != 64
        or value.get("planner_updates") != 0
        or value.get("operator_updates") != 0
        or value.get("complete_action_result") is not False
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("file_sha256") != expected_checkpoint_sha256
        or not isinstance(data, Mapping)
        or data.get("manifest_file_sha256") != expected_source_manifest_sha256
        or data.get("manifest_digest") != SOURCE_MANIFEST_DIGEST
        or data.get("optimizer_rows_read") != 64
        or data.get("synthetic_target_index1_bytes_read") is not False
        or data.get("generated_media_read") is not False
        or data.get("action_family_used_for_routing") is not False
        or not isinstance(objective, Mapping)
        or objective.get("name") != "real_source_same_noise_noop_flow_matching"
        or objective.get("memory_input_kind")
        != "same_noise_forward_noised_source"
        or objective.get("same_epsilon_and_sigma_target_memory") is not True
        or objective.get("synthetic_target") is not False
        or objective.get("reward") is not False
        or not isinstance(distributed, Mapping)
        or distributed.get("world_size") != WORLD_SIZE
        or distributed.get("sp_size") != SP_SIZE
        or distributed.get("one_shared_model") is not True
        or distributed.get("rank_action_family_partition") is not False
        or not isinstance(model, Mapping)
        or model.get("bernini_commit") != BERNINI_COMMIT
        or model.get("veomni_commit") != VEOMNI_COMMIT
        or model.get("checkpoint_tree_sha256") != CHECKPOINT_TREE_SHA256
        or model.get("frozen_base_transformer_sha256_initial")
        != model.get("frozen_base_transformer_sha256_terminal")
        or not isinstance(counts, Mapping)
        or counts.get("carrier") != 2_036_996
        or not isinstance(initial, Mapping)
        or not isinstance(final, Mapping)
        or initial.get("P") != final.get("P")
        or initial.get("O") != final.get("O")
        or initial.get("R") != R64_CARRIER_INITIAL_SHA256
        or final.get("R") != R64_CARRIER_FINAL_SHA256
        or not isinstance(terminal, Mapping)
        or terminal.get("unchanged") is not True
        or terminal.get("checkpoint_manifest_sha256_reverified")
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
        or terminal.get("source_manifest_sha256_reverified")
        != expected_source_manifest_sha256
        or terminal.get("bernini_commit_reverified") != BERNINI_COMMIT
        or terminal.get("veomni_commit_reverified") != VEOMNI_COMMIT
        or not isinstance(pair, Mapping)
        or pair.get("checkpoint_tree_sha256") != CHECKPOINT_TREE_SHA256
        or pair.get("checkpoint_content_manifest_sha256")
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
        or pair.get("source_manifest_sha256") != expected_source_manifest_sha256
    ):
        fail("R64 receipt completion/data/model scope differs")
    if verify_files:
        checkpoint_path = _plain_file(checkpoint.get("path"), label="R64 checkpoint")
        source_manifest_path = _plain_file(
            data.get("manifest_path"), label="source-only-v3 manifest"
        )
    else:
        checkpoint_path = _absolute_declared_path(
            checkpoint.get("path"), label="R64 checkpoint"
        )
        source_manifest_path = _absolute_declared_path(
            data.get("manifest_path"), label="source-only-v3 manifest"
        )
    if verify_files and (
        file_sha256(checkpoint_path) != expected_checkpoint_sha256
        or file_sha256(source_manifest_path) != expected_source_manifest_sha256
    ):
        fail("R64 checkpoint/source manifest bytes changed")
    content_identity = model.get("checkpoint_content_identity")
    if (
        not isinstance(content_identity, Mapping)
        or set(content_identity) != {
            "checkpoint_root", "digest", "every_non_cache_file_sha256_verified",
            "manifest_path", "manifest_sha256", "tree_sha256",
            "verified_entries_digest", "verified_file_count",
        }
        or content_identity.get("manifest_sha256")
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
        or content_identity.get("tree_sha256") != CHECKPOINT_TREE_SHA256
        or content_identity.get("every_non_cache_file_sha256_verified") is not True
        or content_identity.get("verified_file_count") != 23
        or _SHA256.fullmatch(str(content_identity.get("digest"))) is None
        or _SHA256.fullmatch(
            str(content_identity.get("verified_entries_digest"))
        ) is None
    ):
        fail("R64 receipt lacks base checkpoint content identity")
    checkpoint_manifest_sha = _sha(
        content_identity.get("manifest_sha256"),
        label="base checkpoint content manifest SHA",
    )
    if checkpoint_manifest_sha != CHECKPOINT_CONTENT_MANIFEST_SHA256:
        fail("R64 receipt base checkpoint manifest authority differs")
    return R64Authority(
        receipt_path=receipt_path,
        receipt_file_sha256=expected_receipt_sha256,
        receipt_digest=receipt_digest,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=expected_checkpoint_sha256,
        source_manifest_path=source_manifest_path,
        source_manifest_file_sha256=expected_source_manifest_sha256,
        source_manifest_digest=_sha(
            data.get("manifest_digest"), label="source manifest digest"
        ),
        carrier_initial_sha256=_sha(initial.get("R"), label="initial carrier SHA"),
        carrier_final_sha256=_sha(final.get("R"), label="final carrier SHA"),
        checkpoint_tree_sha256=_sha(
            model.get("checkpoint_tree_sha256"), label="base checkpoint tree SHA"
        ),
        checkpoint_content_manifest_sha256=checkpoint_manifest_sha,
    )


def load_carrier_checkpoint_strict(
    authority: R64Authority, handle: Any
) -> Mapping[str, Any]:
    """Strictly load only the trained R component into an installed carrier."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH supplies torch
        raise R64HeldoutContractError("PyTorch is required") from error
    try:
        payload = torch.load(
            authority.checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise R64HeldoutContractError("cannot safely load R64 checkpoint") from error
    component_state = payload.get("component_state") if isinstance(payload, Mapping) else None
    pair = payload.get("pair_invariants") if isinstance(payload, Mapping) else None
    initial = payload.get("initial_component_sha256") if isinstance(payload, Mapping) else None
    final = payload.get("final_component_sha256") if isinstance(payload, Mapping) else None
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("method") != TRAINING_METHOD
        or payload.get("experiment") != "joint_source_anchored_v1"
        or payload.get("execution_profile") != "stage-r64"
        or payload.get("completed_stages") != ["R"]
        or payload.get("incomplete_stages") != []
        or payload.get("completed_stage_updates") != {"R": 64}
        or payload.get("complete_action_result") is not False
        or not isinstance(component_state, Mapping)
        or set(component_state) != {"R", "P", "O"}
        or not isinstance(pair, Mapping)
        or pair.get("source_manifest_sha256") != authority.source_manifest_file_sha256
        or payload.get("source_manifest_digest") != authority.source_manifest_digest
        or not isinstance(initial, Mapping)
        or not isinstance(final, Mapping)
        or initial.get("R") != authority.carrier_initial_sha256
        or final.get("R") != authority.carrier_final_sha256
        or initial.get("P") != final.get("P")
        or initial.get("O") != final.get("O")
    ):
        fail("R64 checkpoint stage/component closure differs")
    r_state = component_state["R"]
    named = tuple(handle.trainable_named_parameters())
    expected_keys = {f"carrier.{name}" for name, _ in named}
    if not isinstance(r_state, Mapping) or set(r_state) != expected_keys:
        fail("R64 carrier state key closure differs")
    with torch.no_grad():
        for name, parameter in named:
            value = r_state[f"carrier.{name}"]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(parameter.shape)
                or value.dtype != torch.float32
                or value.requires_grad
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
            ):
                fail(f"R64 carrier tensor differs: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    loaded_named = tuple((f"carrier.{name}", parameter) for name, parameter in named)
    digest = hashlib.sha256()
    for name, parameter in loaded_named:
        tensor = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).cpu()
        try:
            digest.update(raw.numpy().tobytes(order="C"))
        except RuntimeError as error:
            if "Numpy is not available" not in str(error):
                raise
            digest.update(bytes(raw.tolist()))
    loaded_sha = digest.hexdigest()
    if loaded_sha != authority.carrier_final_sha256:
        fail("strictly loaded R64 carrier digest differs")
    return {
        "carrier_parameter_sha256": loaded_sha,
        "carrier_parameter_count": sum(parameter.numel() for _, parameter in named),
        "checkpoint_complete_action_result": False,
        "planner_loaded": False,
        "operator_loaded": False,
    }


def validate_exact81_media(path_value: str | Path) -> Mapping[str, Any]:
    """Decode every frame with the authenticated release-local Python path.

    The R64 packet must not depend on a login-shell ``PATH`` or an unpinned
    external ``ffprobe`` executable.  Reuse the exact ``materialize_vae``
    decoder already authenticated by :func:`bind_release_preprocessing_tools`:
    it opens the MP4 through decord, asks for all indices ``0..80`` in one
    batch, and returns the decoded THWC uint8 RGB array.  This wrapper preserves
    the former exact81/25-fps/positive-geometry admission semantics while also
    proving that every declared frame is actually decodable.
    """

    path = _plain_file(path_value, label="exact81 media")
    method_root = Path(__file__).resolve(strict=True).parent
    identities = bind_release_preprocessing_tools(method_root)
    if identities != RELEASE_PREPROCESSING_TOOL_SHA256:
        fail("release media decoder identity differs")
    try:
        materializer = importlib.import_module("tools.materialize_vae")
        decoder = getattr(materializer, "_decode_exact_video")
        frames, fps, source_hw = decoder(path)
        shape = tuple(int(value) for value in frames.shape)
        dtype = str(frames.dtype)
    except Exception as error:
        raise R64HeldoutContractError(
            f"cannot fully decode exact81 media with release Python decoder: {path}"
        ) from error
    if (
        not isinstance(source_hw, tuple)
        or len(source_hw) != 2
        or any(type(value) is not int or value <= 0 for value in source_hw)
        or len(shape) != 4
        or shape[0] != FRAME_COUNT
        or shape[-1] != 3
        or shape[1:3] != source_hw
        or shape[1] <= 0
        or shape[2] <= 0
        or dtype != "uint8"
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or abs(float(fps) - FPS) > 1.0e-6
    ):
        fail(f"media is not fully decodable exact81/{FPS}fps RGB: {path}")
    return {
        "decoder": "release-tools.materialize_vae._decode_exact_video",
        "decoder_backend": "decord",
        "decoder_source_sha256": identities["tools.materialize_vae"],
        "all_frames_decoded": True,
        "frame_count": FRAME_COUNT,
        "fps": float(fps),
        "height": source_hw[0],
        "width": source_hw[1],
        "channels": 3,
        "dtype": dtype,
    }


def _ffprobe_exact81(path: Path) -> Mapping[str, Any]:
    """Compatibility name; validation is Python-only and spawns no binary."""

    return validate_exact81_media(path)


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < (0 if allow_zero else 1):
        fail(f"{label} is outside its admitted range")
    return value


def _validate_adapter_architecture(
    value: Any, *, strict_load: Mapping[str, Any]
) -> None:
    keys = {
        "schema_version", "gpu_validated", "scientific_quality_claim",
        "runtime_source_commit", "model_revision", "checkpoint_manifest_sha256",
        "transformer_config_digest", "block_indices", "block_scope_status",
        "optimizer_authorized_by_this_receipt", "insertion", "query_source",
        "key_value_source", "memory_input_kinds_supported",
        "native_self_attention_kv_replaced", "native_self_attention_kv_replayed",
        "native_text_cross_attention_changed", "native_blocks_replaced",
        "native_structure_untouched", "condition_rows_directly_written",
        "sp_empty_target_rank_graph_anchor",
        "sp_collective_backward_graph_isomorphic",
        "target_noise_read_by_memory_encoder", "source_reads_target_noise",
        "zero_initialized_output_projection", "multiplicative_gain_initial_value",
        "double_zero_dead_parameterization", "checkpoint_context_fn_required",
        "base_parameters_frozen", "memory_encoder", "trainable", "feature_reward",
        "vlm_reward", "synthetic_target_required", "digest",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        fail("adapter architecture key closure differs")
    unsigned = dict(value)
    digest = unsigned.pop("digest", None)
    memory = value.get("memory_encoder")
    trainable = value.get("trainable")
    fixed = {
        "schema_version": ADAPTER_SCHEMA,
        "gpu_validated": False,
        "scientific_quality_claim": False,
        "runtime_source_commit": BERNINI_COMMIT,
        "model_revision": BERNINI_MODEL_REVISION,
        "checkpoint_manifest_sha256": CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "block_indices": list(ADAPTER_BLOCK_INDICES),
        "block_scope_status": "structural_candidate_not_causally_admitted",
        "optimizer_authorized_by_this_receipt": False,
        "insertion": "registered_forward_hook_on_frozen_block_output",
        "query_source": "current_frozen_block_input_target_rows",
        "key_value_source": "independent_source_visual_memory_only",
        "memory_input_kinds_supported": [
            "clean_source", "same_noise_forward_noised_source"
        ],
        "native_self_attention_kv_replaced": False,
        "native_self_attention_kv_replayed": False,
        "native_text_cross_attention_changed": False,
        "native_blocks_replaced": False,
        "native_structure_untouched": True,
        "condition_rows_directly_written": False,
        "sp_empty_target_rank_graph_anchor": (
            "query_times_trainable_exact_zero_on_every_rank"
        ),
        "sp_collective_backward_graph_isomorphic": True,
        "target_noise_read_by_memory_encoder": "declared_per_memory_receipt",
        "source_reads_target_noise": "declared_per_memory_receipt",
        "zero_initialized_output_projection": True,
        "multiplicative_gain_initial_value": 1.0,
        "double_zero_dead_parameterization": False,
        "checkpoint_context_fn_required": True,
        "base_parameters_frozen": True,
        "feature_reward": False,
        "vlm_reward": False,
        "synthetic_target_required": False,
    }
    if (
        object_sha256(unsigned) != digest
        or any(value.get(key) != expected for key, expected in fixed.items())
        or value.get("transformer_config_digest") != TRANSFORMER_CONFIG_DIGEST
        or strict_load.get("adapter_architecture_digest") != digest
        or strict_load.get("loaded_block_indices") != list(ADAPTER_BLOCK_INDICES)
    ):
        fail("adapter architecture authority differs")
    memory_keys = {
        "schema_version", "input", "patchifier", "patch_size",
        "temporal_patch_stride", "temporal_pooling",
        "spatial_pooling_only_when_needed", "memory_token_cap", "encoder_width",
        "hidden_size", "pipeline", "position_representation",
        "position_parameters_trainable", "position_added_after_projection",
        "explicit_source_role_id", "target_noise_argument_present",
        "text_argument_present", "digest",
    }
    if not isinstance(memory, Mapping) or set(memory) != memory_keys:
        fail("adapter memory encoder key closure differs")
    memory_unsigned = dict(memory)
    memory_digest = memory_unsigned.pop("digest", None)
    if (
        object_sha256(memory_unsigned) != memory_digest
        or memory.get("schema_version") != MEMORY_SCHEMA
        or memory.get("input")
        != "detached_registered_source_visual_latent_[1,16,F,H,W]"
        or memory.get("patchifier") != "trainable_conv3d"
        or memory.get("patch_size") != [1, 4, 4]
        or memory.get("temporal_patch_stride") != 1
        or memory.get("temporal_pooling") is not False
        or memory.get("spatial_pooling_only_when_needed") is not True
        or memory.get("memory_token_cap") != 1024
        or memory.get("encoder_width") != 256
        or memory.get("hidden_size") != 1536
        or memory.get("pipeline")
        != (
            "patchify->spatial_budget->layer_norm->projection->layer_norm"
            "+fixed_3d_fourier_phase_y_x+source_role_id"
        )
        or memory.get("position_representation")
        != "fixed_absolute_3d_fourier_phase_y_x_v1"
        or memory.get("position_parameters_trainable") is not False
        or memory.get("position_added_after_projection") is not True
        or memory.get("explicit_source_role_id") != 1
        or memory.get("target_noise_argument_present") is not False
        or memory.get("text_argument_present") is not False
    ):
        fail("adapter memory encoder architecture differs")
    expected_shapes: dict[str, list[int]] = {
        "encoder.patchifier.weight": [256, 16, 1, 4, 4],
        "encoder.patchifier.bias": [256],
        "encoder.patch_norm.weight": [256],
        "encoder.patch_norm.bias": [256],
        "encoder.projection.weight": [1536, 256],
        "encoder.projection.bias": [1536],
        "encoder.source_role.weight": [2, 1536],
    }
    for block in ADAPTER_BLOCK_INDICES:
        prefix = f"adapters.{block}"
        expected_shapes[f"{prefix}.residual_gain"] = []
        for projection in ("query", "key", "value"):
            expected_shapes[f"{prefix}.{projection}.weight"] = [64, 1536]
        expected_shapes[f"{prefix}.output.weight"] = [1536, 64]
    observed_shapes: dict[str, list[int]] = {}
    if not isinstance(trainable, list):
        fail("adapter trainable manifest differs")
    for row in trainable:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"name", "shape", "dtype"}
            or not isinstance(row.get("name"), str)
            or row.get("name") in observed_shapes
            or row.get("dtype") != "torch.float32"
            or not isinstance(row.get("shape"), list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in row.get("shape", [])
            )
        ):
            fail("adapter trainable row differs")
        observed_shapes[str(row["name"])] = list(row["shape"])
    parameter_count = sum(
        math.prod(shape) if shape else 1 for shape in observed_shapes.values()
    )
    if (
        observed_shapes != expected_shapes
        or parameter_count != 2_036_996
        or parameter_count != strict_load.get("carrier_parameter_count")
    ):
        fail("loaded carrier does not bind the exact four-block architecture")


def _validate_freeze_certificate(value: Any) -> None:
    wrapper_keys = {"before", "after", "unchanged", "certificate_sha256"}
    certificate_keys = {
        "base_frozen", "model_eval", "adapter_modules_absent", "module_count",
        "module_topology_sha256", "parameter_tensor_count", "parameter_byte_count",
        "buffer_tensor_count", "buffer_byte_count", "state_metadata_sha256",
        "state_content_sha256", "exact_parameter_and_buffer_bytes_hashed",
        "device_and_storage_address_excluded",
    }
    if not isinstance(value, Mapping) or set(value) != wrapper_keys:
        fail("base freeze certificate wrapper differs")
    before, after = value.get("before"), value.get("after")
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or set(before) != certificate_keys
        or set(after) != certificate_keys
        or before != after
        or value.get("unchanged") is not True
        or value.get("certificate_sha256") != object_sha256(before)
    ):
        fail("base freeze certificate before/after closure differs")
    if (
        before.get("base_frozen") is not True
        or before.get("model_eval") is not True
        or before.get("adapter_modules_absent") is not True
        or before.get("exact_parameter_and_buffer_bytes_hashed") is not True
        or before.get("device_and_storage_address_excluded") is not True
    ):
        fail("base freeze certificate semantics differ")
    for key in ("module_count", "parameter_tensor_count", "parameter_byte_count"):
        _positive_int(before.get(key), label=f"base freeze {key}")
    for key in ("buffer_tensor_count", "buffer_byte_count"):
        _positive_int(before.get(key), label=f"base freeze {key}", allow_zero=True)
    for key in (
        "module_topology_sha256", "state_metadata_sha256", "state_content_sha256"
    ):
        _sha(before.get(key), label=f"base freeze {key}")


def _validate_route_trace(trace: Any, *, row: Mapping[str, Any]) -> None:
    aggregate_keys = {
        "schema_version", "world_size", "sequence_parallel_size",
        "rank_trace_digests", "semantic_projection_digest", "exact40",
        "shared_step_calls_per_rank", "trace_digest", "rank_traces",
    }
    rank_keys = {
        "schema_version", "source_control_arm", "target_source_video_sha256",
        "memory_source_video_sha256", "memory_transform", "memory_input_kind",
        "exact40", "step_count", "shared_step_call_count", "native_branch_order",
        "target_tokens", "sequence_parallel_size", "sequence_parallel_rank",
        "shared_step_only_wrapped", "native_guidance_changed", "scheduler_changed",
        "optimizer_present", "memory_build_count", "calls", "trace_digest",
    }
    call_keys = {
        "branch", "step_index", "timestep", "total_tokens", "condition_tokens",
        "target_tokens", "route_enabled", "memory_source_video_sha256",
        "memory_construction_digest",
    }
    if not isinstance(trace, Mapping) or set(trace) != aggregate_keys:
        fail("route aggregate key closure differs")
    rank_traces = trace.get("rank_traces")
    if (
        trace.get("schema_version") != ROUTE_SCHEMA
        or trace.get("world_size") != WORLD_SIZE
        or trace.get("sequence_parallel_size") != SP_SIZE
        or trace.get("exact40") is not True
        or trace.get("shared_step_calls_per_rank")
        != NUM_INFERENCE_STEPS * len(NATIVE_BRANCH_ORDER)
        or not isinstance(rank_traces, list)
        or len(rank_traces) != WORLD_SIZE
    ):
        fail("route aggregate execution differs")
    aggregate_unsigned = dict(trace)
    declared_aggregate = aggregate_unsigned.pop("trace_digest", None)
    aggregate_unsigned.pop("rank_traces", None)
    if object_sha256(aggregate_unsigned) != declared_aggregate:
        fail("route aggregate digest differs")
    expected_enabled = row.get("arm") == "trained-carrier-r64"
    expected_control = "correct" if expected_enabled else "carrier-off"
    expected_memory_source = row.get("source_video_sha256") if expected_enabled else None
    expected_transform = "identity" if expected_enabled else None
    expected_memory_kind = (
        "same_noise_forward_noised_source" if expected_enabled else None
    )
    projections: list[Mapping[str, Any]] = []
    rank_digests: list[str] = []
    for rank, rank_trace in enumerate(rank_traces):
        if not isinstance(rank_trace, Mapping) or set(rank_trace) != rank_keys:
            fail("route rank-trace key closure differs")
        unsigned = dict(rank_trace)
        declared = unsigned.pop("trace_digest", None)
        if object_sha256(unsigned) != declared:
            fail("route rank-trace digest differs")
        if (
            rank_trace.get("schema_version") != ROUTE_SCHEMA
            or rank_trace.get("source_control_arm") != expected_control
            or rank_trace.get("target_source_video_sha256")
            != row.get("source_video_sha256")
            or rank_trace.get("memory_source_video_sha256") != expected_memory_source
            or rank_trace.get("memory_transform") != expected_transform
            or rank_trace.get("memory_input_kind") != expected_memory_kind
            or rank_trace.get("exact40") is not True
            or rank_trace.get("step_count") != NUM_INFERENCE_STEPS
            or rank_trace.get("shared_step_call_count")
            != NUM_INFERENCE_STEPS * len(NATIVE_BRANCH_ORDER)
            or rank_trace.get("native_branch_order") != list(NATIVE_BRANCH_ORDER)
            or _positive_int(
                rank_trace.get("target_tokens"), label="route target_tokens"
            ) <= 0
            or rank_trace.get("sequence_parallel_size") != SP_SIZE
            or rank_trace.get("sequence_parallel_rank") != rank
            or rank_trace.get("shared_step_only_wrapped") is not True
            or rank_trace.get("native_guidance_changed") is not False
            or rank_trace.get("scheduler_changed") is not False
            or rank_trace.get("optimizer_present") is not False
            or rank_trace.get("memory_build_count")
            != (NUM_INFERENCE_STEPS if expected_enabled else 0)
        ):
            fail("route rank-trace semantics differ")
        calls = rank_trace.get("calls")
        if (
            not isinstance(calls, list)
            or len(calls) != NUM_INFERENCE_STEPS * len(NATIVE_BRANCH_ORDER)
        ):
            fail("route call count differs")
        target_tokens = int(rank_trace["target_tokens"])
        for step in range(NUM_INFERENCE_STEPS):
            group = calls[
                step * len(NATIVE_BRANCH_ORDER):(step + 1) * len(NATIVE_BRANCH_ORDER)
            ]
            if any(not isinstance(call, Mapping) or set(call) != call_keys for call in group):
                fail("route call key closure differs")
            timesteps = []
            memory_digests = []
            for branch_index, call in enumerate(group):
                timestep = call.get("timestep")
                if (
                    isinstance(timestep, bool)
                    or not isinstance(timestep, (int, float))
                    or not math.isfinite(float(timestep))
                    or call.get("branch") != NATIVE_BRANCH_ORDER[branch_index]
                    or call.get("step_index") != step
                    or call.get("target_tokens") != target_tokens
                    or isinstance(call.get("condition_tokens"), bool)
                    or not isinstance(call.get("condition_tokens"), int)
                    or call.get("condition_tokens")
                    != (0 if branch_index == 0 else call.get("condition_tokens"))
                    or (branch_index > 0 and call.get("condition_tokens", 0) <= 0)
                    or call.get("total_tokens")
                    != target_tokens + call.get("condition_tokens", -1)
                    or call.get("route_enabled") is not expected_enabled
                    or call.get("memory_source_video_sha256") != expected_memory_source
                ):
                    fail("route call semantics differ")
                memory_digest = call.get("memory_construction_digest")
                if expected_enabled:
                    _sha(memory_digest, label="route memory construction digest")
                elif memory_digest is not None:
                    fail("carrier-off route unexpectedly constructed memory")
                timesteps.append(float(timestep))
                memory_digests.append(memory_digest)
            if len(set(timesteps)) != 1 or len(set(memory_digests)) != 1:
                fail("route step branches do not share timestep/memory")
        projection = dict(unsigned)
        projection.pop("sequence_parallel_rank", None)
        projections.append(projection)
        rank_digests.append(str(declared))
    if (
        projections[1:] != [projections[0]] * (WORLD_SIZE - 1)
        or trace.get("rank_trace_digests") != rank_digests
        or trace.get("semantic_projection_digest") != object_sha256(projections[0])
        or trace.get("trace_digest") != row.get("route_trace_digest")
    ):
        fail("route rank order/digest/semantic projection binding differs")


def validate_receipt(
    value: Mapping[str, Any], *, expected_runtime_source_revision: str,
    expected_runtime_source_closure_sha256: str,
    expected_launcher_sha256: str, media_root: Optional[Path] = None,
    verify_media: bool = False,
) -> Mapping[str, Any]:
    if (
        not isinstance(expected_runtime_source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected_runtime_source_revision) is None
    ):
        fail("expected runtime source revision differs")
    _sha(
        expected_runtime_source_closure_sha256,
        label="expected runtime source closure SHA",
    )
    _sha(expected_launcher_sha256, label="expected launcher SHA")
    root_keys = {
        "schema_version", "method", "complete", "complete_action_result",
        "action_claim_forbidden", "quality_claimed", "r64_authority",
        "strict_load", "source_manifest", "sources", "rows", "execution",
        "evidence", "authority", "receipt_digest",
    }
    if set(value) != root_keys:
        fail("held-out evaluation receipt key closure differs")
    unsigned = dict(value)
    declared = _sha(unsigned.pop("receipt_digest", None), label="evaluation receipt digest")
    rows = value.get("rows")
    source_rows = value.get("sources")
    execution = value.get("execution")
    authority = value.get("authority")
    r64 = value.get("r64_authority")
    strict_load = value.get("strict_load")
    source_manifest = value.get("source_manifest")
    evidence = value.get("evidence")
    if (
        object_sha256(unsigned) != declared
        or value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("method") != "bernini-generic-source-carrier-r64-heldout-v1"
        or value.get("complete") is not True
        or value.get("complete_action_result") is not False
        or value.get("action_claim_forbidden") is not True
        or value.get("quality_claimed") is not False
        or not isinstance(rows, list)
        or len(rows) != MEDIA_ROWS
        or not isinstance(source_rows, list)
        or len(source_rows) != HELDOUT_ROWS
        or not isinstance(r64, Mapping)
        or set(r64) != {
            "training_receipt", "training_receipt_file_sha256",
            "training_receipt_digest", "checkpoint", "checkpoint_file_sha256",
            "source_manifest", "source_manifest_file_sha256",
            "source_manifest_digest", "carrier_initial_sha256",
            "carrier_final_sha256", "checkpoint_tree_sha256",
            "checkpoint_content_manifest_sha256",
        }
        or r64.get("training_receipt_file_sha256")
        != R64_TRAINING_RECEIPT_SHA256
        or r64.get("training_receipt_digest") != R64_TRAINING_RECEIPT_DIGEST
        or r64.get("checkpoint_file_sha256") != R64_CHECKPOINT_SHA256
        or r64.get("source_manifest_file_sha256") != SOURCE_MANIFEST_SHA256
        or r64.get("source_manifest_digest") != SOURCE_MANIFEST_DIGEST
        or r64.get("carrier_initial_sha256") != R64_CARRIER_INITIAL_SHA256
        or r64.get("carrier_final_sha256") != R64_CARRIER_FINAL_SHA256
        or r64.get("checkpoint_tree_sha256") != CHECKPOINT_TREE_SHA256
        or r64.get("checkpoint_content_manifest_sha256")
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
        or not isinstance(r64.get("training_receipt"), str)
        or not isinstance(r64.get("checkpoint"), str)
        or not isinstance(r64.get("source_manifest"), str)
        or any(
            _SHA256.fullmatch(str(r64.get(key))) is None
            for key in (
                "training_receipt_digest", "source_manifest_digest",
                "carrier_initial_sha256", "carrier_final_sha256",
                "checkpoint_tree_sha256", "checkpoint_content_manifest_sha256",
            )
        )
        or r64.get("carrier_initial_sha256") == r64.get("carrier_final_sha256")
        or not isinstance(strict_load, Mapping)
        or set(strict_load) != {
            "carrier_parameter_sha256", "carrier_parameter_count",
            "checkpoint_complete_action_result", "planner_loaded", "operator_loaded",
            "adapter_architecture_digest", "loaded_block_indices",
        }
        or strict_load.get("carrier_parameter_sha256")
        != r64.get("carrier_final_sha256")
        or strict_load.get("carrier_parameter_count") != 2_036_996
        or strict_load.get("checkpoint_complete_action_result") is not False
        or strict_load.get("planner_loaded") is not False
        or strict_load.get("operator_loaded") is not False
        or _SHA256.fullmatch(
            str(strict_load.get("adapter_architecture_digest"))
        ) is None
        or strict_load.get("loaded_block_indices") != list(ADAPTER_BLOCK_INDICES)
        or not isinstance(source_manifest, Mapping)
        or set(source_manifest) != {
            "path", "file_sha256", "manifest_digest", "split", "rows", "row_order"
        }
        or source_manifest.get("path") != r64.get("source_manifest")
        or source_manifest.get("file_sha256") != SOURCE_MANIFEST_SHA256
        or source_manifest.get("manifest_digest") != r64.get("source_manifest_digest")
        or source_manifest.get("split") != "heldout"
        or source_manifest.get("rows") != HELDOUT_ROWS
        or source_manifest.get("row_order") != "iid-lexicographic"
        or not isinstance(execution, Mapping)
        or set(execution) != {
            "world_size", "sequence_parallel_size", "num_inference_steps",
            "frame_count", "fps", "arms",
            "same_source_seed_prompt_gaussian_within_pair", "base_arm", "trained_arm",
        }
        or execution.get("world_size") != WORLD_SIZE
        or execution.get("sequence_parallel_size") != SP_SIZE
        or execution.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or execution.get("frame_count") != FRAME_COUNT
        or execution.get("fps") != FPS
        or execution.get("arms") != list(ARMS)
        or execution.get("same_source_seed_prompt_gaussian_within_pair") is not True
        or execution.get("base_arm")
        != "same-loaded-model authenticated carrier-off route"
        or execution.get("trained_arm")
        != "strictly-loaded R64 same-noise carrier route"
        or not isinstance(authority, Mapping)
        or set(authority) != {
            "manual_preservation_review_pending", "action_evaluation_performed",
            "reward_present", "ranking_present", "selection_present",
            "optimizer_present", "backward_performed", "parameter_update",
            "target_video_read",
        }
        or authority.get("manual_preservation_review_pending") is not True
        or authority.get("action_evaluation_performed") is not False
        or authority.get("reward_present") is not False
        or authority.get("ranking_present") is not False
        or authority.get("selection_present") is not False
        or authority.get("optimizer_present") is not False
        or authority.get("backward_performed") is not False
        or authority.get("parameter_update") is not False
        or authority.get("target_video_read") is not False
        or not isinstance(evidence, Mapping)
        or set(evidence) != {
            "runtime_source", "pinned_sources", "base_checkpoint", "raw_projection",
            "source_preprocessing", "native_prompt_sha256", "adapter_architecture",
            "independent_loaded_tensor_digest_before",
            "independent_loaded_tensor_digest_after", "base_freeze_certificate",
            "route_traces", "host_trim_after_load", "runtime_versions",
        }
    ):
        fail("held-out evaluation receipt root differs")

    runtime_source = evidence["runtime_source"]
    pinned_sources = evidence["pinned_sources"]
    base_checkpoint = evidence["base_checkpoint"]
    raw_projection = evidence["raw_projection"]
    route_traces = evidence["route_traces"]
    content_identity = base_checkpoint.get("content_identity") \
        if isinstance(base_checkpoint, Mapping) else None
    if (
        not isinstance(runtime_source, Mapping)
        or set(runtime_source) != {"revision", "closure_sha256", "launcher_sha256"}
        or runtime_source.get("revision") != expected_runtime_source_revision
        or runtime_source.get("closure_sha256")
        != expected_runtime_source_closure_sha256
        or runtime_source.get("launcher_sha256") != expected_launcher_sha256
        or not isinstance(pinned_sources, Mapping)
        or set(pinned_sources) != {
            "bernini_commit", "veomni_commit", "wan_diffusion_sha256",
            "bernini_inference_files",
        }
        or pinned_sources.get("bernini_commit") != BERNINI_COMMIT
        or pinned_sources.get("veomni_commit") != VEOMNI_COMMIT
        or _SHA256.fullmatch(str(pinned_sources.get("wan_diffusion_sha256"))) is None
        or not isinstance(pinned_sources.get("bernini_inference_files"), Mapping)
        or not isinstance(base_checkpoint, Mapping)
        or set(base_checkpoint) != {
            "path", "tree_sha256", "content_identity", "opened_read_only"
        }
        or base_checkpoint.get("tree_sha256") != r64.get("checkpoint_tree_sha256")
        or base_checkpoint.get("opened_read_only") is not True
        or not isinstance(content_identity, Mapping)
        or set(content_identity) != {
            "manifest_path", "manifest_sha256_computed", "manifest_sha256_expected",
            "verified_file_count", "every_file_sha256_verified",
            "verified_entries_digest",
        }
        or content_identity.get("manifest_sha256_computed")
        != r64.get("checkpoint_content_manifest_sha256")
        or content_identity.get("manifest_sha256_expected")
        != r64.get("checkpoint_content_manifest_sha256")
        or content_identity.get("verified_file_count") != 23
        or content_identity.get("every_file_sha256_verified") is not True
        or _SHA256.fullmatch(
            str(content_identity.get("verified_entries_digest"))
        ) is None
        or not isinstance(raw_projection, Mapping)
        or set(raw_projection) != {
            "path", "file_sha256", "safe_columns_read", "target_columns_read"
        }
        or raw_projection.get("file_sha256") != RAW_PARQUET_SHA256
        or raw_projection.get("safe_columns_read") != list(RAW_SAFE_COLUMNS)
        or raw_projection.get("target_columns_read") is not False
        or evidence.get("independent_loaded_tensor_digest_before")
        != evidence.get("independent_loaded_tensor_digest_after")
        or _SHA256.fullmatch(str(evidence.get("native_prompt_sha256"))) is None
        or not isinstance(route_traces, Mapping)
    ):
        fail("held-out evaluation evidence authority differs")

    _validate_adapter_architecture(
        evidence.get("adapter_architecture"), strict_load=strict_load
    )
    _validate_freeze_certificate(evidence.get("base_freeze_certificate"))

    source_keys = {
        "iid", "group_id", "action_family_provenance_only",
        "source_video_sha256", "seed", "relative_mp4", "mp4_sha256",
        "frame_count", "fps",
    }
    source_by_iid: dict[str, Mapping[str, Any]] = {}
    expected_media: set[str] = set()
    for source in source_rows:
        if not isinstance(source, Mapping) or set(source) != source_keys:
            fail("source evidence row key closure differs")
        iid = source.get("iid")
        expected_relative = f"media/{iid}__source.mp4"
        if (
            not isinstance(iid, str) or _IID.fullmatch(iid) is None
            or iid in source_by_iid
            or not isinstance(source.get("group_id"), str)
            or not source.get("group_id")
            or not isinstance(source.get("action_family_provenance_only"), str)
            or not source.get("action_family_provenance_only")
            or _SHA256.fullmatch(str(source.get("source_video_sha256"))) is None
            or source.get("seed") != heldout_seed(iid)
            or source.get("relative_mp4") != expected_relative
            or source.get("mp4_sha256") != source.get("source_video_sha256")
            or source.get("frame_count") != FRAME_COUNT
            or source.get("fps") != FPS
        ):
            fail("source evidence row fields differ")
        source_by_iid[iid] = source
        expected_media.add(expected_relative)

    keys: set[tuple[str, str]] = set()
    by_iid: dict[str, list[Mapping[str, Any]]] = {}
    row_keys = {
        "record_id", "iid", "group_id", "action_family_provenance_only", "arm",
        "source_video_sha256", "seed", "instruction", "instruction_sha256",
        "initial_gaussian_sha256", "route_trace_digest", "carrier_enabled",
        "latent_shape", "relative_mp4", "mp4_sha256", "frame_count", "fps",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_keys:
            fail("evaluation row key closure differs")
        iid, arm = row.get("iid"), row.get("arm")
        source = source_by_iid.get(str(iid))
        expected_relative = f"media/{iid}__{arm}.mp4"
        if (
            not isinstance(iid, str)
            or _IID.fullmatch(iid) is None
            or source is None
            or arm not in ARMS
            or row.get("record_id") != f"{iid}__{arm}"
            or row.get("group_id") != source.get("group_id")
            or row.get("action_family_provenance_only")
            != source.get("action_family_provenance_only")
            or row.get("source_video_sha256") != source.get("source_video_sha256")
            or row.get("seed") != source.get("seed")
            or row.get("instruction") != GENERIC_NOOP_INSTRUCTION
            or row.get("instruction_sha256") != GENERIC_NOOP_SHA256
            or row.get("carrier_enabled")
            is not (arm == "trained-carrier-r64")
            or _SHA256.fullmatch(str(row.get("route_trace_digest"))) is None
            or not isinstance(row.get("latent_shape"), list)
            or len(row.get("latent_shape")) != 5
            or row.get("latent_shape")[:3] != [1, 16, 21]
            or any(type(item) is not int or item <= 0 for item in row["latent_shape"])
            or row.get("frame_count") != FRAME_COUNT
            or row.get("fps") != FPS
            or row.get("relative_mp4") != expected_relative
            or _SHA256.fullmatch(str(row.get("mp4_sha256"))) is None
            or _SHA256.fullmatch(str(row.get("initial_gaussian_sha256"))) is None
        ):
            fail("evaluation row fields differ")
        key = (iid, str(arm))
        if key in keys:
            fail("duplicate held-out evaluation arm")
        keys.add(key)
        by_iid.setdefault(iid, []).append(row)
        expected_media.add(expected_relative)
    if len(by_iid) != HELDOUT_ROWS:
        fail("receipt does not contain exactly eight held-out IIDs")
    if set(evidence["source_preprocessing"]) != set(source_by_iid):
        fail("source preprocessing exact held-out IID set differs")
    for iid, pair in by_iid.items():
        if {str(row["arm"]) for row in pair} != set(ARMS):
            fail(f"{iid} lacks one base/trained arm")
        shared = {
            (
                row["source_video_sha256"], row["seed"], row["instruction_sha256"],
                row["initial_gaussian_sha256"],
                tuple(row.get("latent_shape", ())),
            )
            for row in pair
        }
        if len(shared) != 1:
            fail(f"{iid} base/trained pairing changed source/seed/prompt/Gaussian")
        for row in pair:
            trace = route_traces.get(str(row["record_id"]))
            _validate_route_trace(trace, row=row)
    if set(route_traces) != {str(row["record_id"]) for row in rows}:
        fail("route-trace exact record set differs")
    if verify_media:
        if media_root is None:
            fail("media root is required for media verification")
        media_dir = media_root / "media"
        if media_dir.is_symlink() or not media_dir.is_dir():
            fail("media directory differs")
        observed: set[str] = set()
        for path in media_dir.rglob("*"):
            if path.is_symlink() or not path.is_file():
                fail("media directory contains a non-plain artifact")
            observed.add(path.relative_to(media_root).as_posix())
        if observed != expected_media or len(observed) != HELDOUT_ROWS + MEDIA_ROWS:
            fail("media artifact exact set differs")
        hashes = {
            str(row["relative_mp4"]): str(row["mp4_sha256"])
            for row in [*source_rows, *rows]
        }
        for relative, expected_sha in hashes.items():
            path = media_root / relative
            if file_sha256(path) != expected_sha:
                fail("evaluation/source MP4 bytes differ")
            validate_exact81_media(path)
    return value


__all__ = [
    "ADAPTER_BLOCK_INDICES", "ARMS", "CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "CHECKPOINT_TREE_SHA256", "FRAME_COUNT", "FPS", "GENERIC_NOOP_INSTRUCTION",
    "GENERIC_NOOP_SHA256", "HELDOUT_ROWS", "MEDIA_ROWS",
    "NUM_INFERENCE_STEPS", "R64Authority", "R64HeldoutContractError",
    "R64_CHECKPOINT_SHA256", "R64_TRAINING_RECEIPT_SHA256", "RAW_PARQUET_SHA256",
    "RAW_SAFE_COLUMNS", "RELEASE_PREPROCESSING_TOOL_SHA256",
    "RECEIPT_SCHEMA", "SOURCE_MANIFEST_SHA256", "SP_SIZE", "WORLD_SIZE",
    "bind_release_preprocessing_tools", "canonical_json_bytes", "file_sha256",
    "heldout_seed",
    "load_carrier_checkpoint_strict", "load_r64_authority", "object_sha256",
    "validate_exact81_media", "validate_receipt",
]
