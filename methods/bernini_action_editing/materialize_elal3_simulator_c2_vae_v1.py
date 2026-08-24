#!/usr/bin/env python3
"""Materialize the exact16 simulator C2 videos with the frozen Bernini VAE.

The output is a create-only, model-bound, exact16 fp32 safetensors bundle and
a canonical provenance receipt.  This is privileged simulator diagnostic
data only.  It is not formal C2, source+instruction inference, real-video
evidence, exact160, or a scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import elal3_simulator_c2_label_v1 as labels
import train_lora as legacy
from tools import materialize_vae


SCHEMA_VERSION = "bernini-elal3-simulator-c2-exact16-latent-bundle-v1"
RECEIPT_SCHEMA = (
    "bernini-elal3-simulator-c2-exact16-latent-bundle-receipt-v1"
)
PACKET_MANIFEST_SHA256 = labels.EXPECTED_MANIFEST_SHA256
DERIVATIVE_AUTHORITY_SHA256 = labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256
EXPERIMENT_CONTRACT_SHA256 = labels.EXPECTED_EXPERIMENT_CONTRACT_SHA256
MODEL_AUTHORITY_RELATIVE_PATH = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_real_model_authority_v1.json"
)
MODEL_AUTHORITY_PATH = Path(__file__).resolve().parents[2] / MODEL_AUTHORITY_RELATIVE_PATH
MODEL_AUTHORITY_SCHEMA = "bernini-elal3-c2-real-model-authority-v1"
MODEL_AUTHORITY_SHA256 = (
    "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d"
)
MODEL_AUTHORITY_DIGEST = (
    "c2c0c9037dea2fd56aa13ac56416bf38c6167686c75b69f0b4b568c82e670c1f"
)
FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
MAX_PIXELS = 245_760
SPATIAL_STRIDE = 16
EXPECTED_BUCKET_HW = (416, 560)
EXPECTED_LATENT_SHAPE = (1, 16, 21, 52, 70)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# These are execution-source authorities, not merely documentation pins.  The
# materializer validates the actual imported module ``__file__`` for every row
# before and after the long encode.  The materializer's own SHA/size cannot be
# embedded without a circular hash, so it is an explicit required CLI pin.
RUNTIME_SOURCE_PINS = {
    "elal3_simulator_c2_label_v1": {
        "relative_path": "elal3_simulator_c2_label_v1.py",
        "sha256": "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11",
        "size": 76939,
    },
    "elal3_c0_v1": {
        "relative_path": "elal3_c0_v1.py",
        "sha256": "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862",
        "size": 31330,
    },
    "train_lora": {
        "relative_path": "train_lora.py",
        "sha256": "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
        "size": 66931,
    },
    "tools.materialize_vae": {
        "relative_path": "tools/materialize_vae.py",
        "sha256": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        "size": 32195,
    },
    "tools.build_renderer_dataset": {
        "relative_path": "tools/build_renderer_dataset.py",
        "sha256": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        "size": 31012,
    },
}


def bundle_key(row_index: int, row_id: str, variant: str) -> str:
    slug = row_id.replace("-", "_")
    return f"r{row_index:02d}__{slug}__{variant}"


TENSOR_ORDER = tuple(
    bundle_key(row_index, row_id, variant)
    for row_index, row_id in enumerate(labels.C2_ROW_IDS)
    for variant in labels.MEDIA_ORDER
)


def expected_safetensors_metadata() -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "row_ids": json.dumps(list(labels.C2_ROW_IDS), separators=(",", ":")),
        "variant_order": json.dumps(
            list(labels.MEDIA_ORDER), separators=(",", ":")
        ),
        "tensor_order": json.dumps(list(TENSOR_ORDER), separators=(",", ":")),
        "tensor_order_digest": object_sha256(list(TENSOR_ORDER)),
        "tensor_count": "16",
        "bucket_hw": "416,560",
        "latent_shape_each": "1,16,21,52,70",
        "dtype_each": "F32",
    }


class ELAL3SimulatorC2VAEError(RuntimeError):
    """Raised before publishing an ambiguous exact16 latent bundle."""


def fail(message: str) -> None:
    raise ELAL3SimulatorC2VAEError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3SimulatorC2VAEError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def stable_stream_hash_path(
    path: Path,
    *,
    label: str,
    expected_sha256: Optional[str],
    expected_size: Optional[int],
    expected_mode: int,
    allowed_root: Path,
) -> dict[str, Any]:
    """Hash twice through one held file FD and held no-follow openat parents."""

    if expected_sha256 is not None and _SHA256.fullmatch(expected_sha256) is None:
        fail(f"{label} expected SHA-256 differs")
    if expected_size is not None and (
        type(expected_size) is not int or expected_size < 0
    ):
        fail(f"{label} expected size differs")
    root = allowed_root.resolve(strict=True)
    requested = path
    try:
        relative = requested.relative_to(root)
    except ValueError as error:
        raise ELAL3SimulatorC2VAEError(f"{label} escapes root") from error
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        fail(f"{label} relative component closure differs")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = os.open(str(root), directory_flags)
    parent_fd = root_fd
    directory_fds: list[int] = []
    file_fd: Optional[int] = None
    try:
        root_before = os.fstat(root_fd)
        root_named_before = root.lstat()
        if (
            _stat_identity(root_before) != _stat_identity(root_named_before)
            or not stat.S_ISDIR(root_before.st_mode)
            or stat.S_ISLNK(root_named_before.st_mode)
        ):
            fail(f"{label} held root differs")
        directory_before = [_stat_identity(root_before)]
        for component in relative.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=parent_fd)
            directory_fds.append(child)
            parent_fd = child
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                fail(f"{label} parent component differs")
            directory_before.append(_stat_identity(info))
        name = relative.parts[-1]
        named_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(named_before.st_mode)
            or not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink != 1
            or stat.S_IMODE(named_before.st_mode) != expected_mode
            or (
                expected_size is not None
                and named_before.st_size != expected_size
            )
        ):
            fail(f"{label} file type/mode/link/size differs")
        file_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        file_before = os.fstat(file_fd)

        def hash_pass() -> str:
            digest = hashlib.sha256()
            while True:
                block = os.read(file_fd, 1 << 22)
                if not block:
                    break
                digest.update(block)
            return digest.hexdigest()

        first_sha = hash_pass()
        os.lseek(file_fd, 0, os.SEEK_SET)
        second_sha = hash_pass()
        file_after = os.fstat(file_fd)
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        directory_after = [_stat_identity(os.fstat(root_fd))] + [
            _stat_identity(os.fstat(item)) for item in directory_fds
        ]
        root_named_after = root.lstat()
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)
        os.close(root_fd)
    identity = _stat_identity(file_before)
    accepted_sha = first_sha if expected_sha256 is None else expected_sha256
    accepted_size = int(file_before.st_size) if expected_size is None else expected_size
    if (
        first_sha != accepted_sha
        or second_sha != accepted_sha
        or identity != _stat_identity(file_after)
        or identity != _stat_identity(named_before)
        or identity != _stat_identity(named_after)
        or directory_before != directory_after
        or directory_before[0] != _stat_identity(root_named_after)
    ):
        fail(f"{label} held-FD double-hash/identity replay differs")
    return {
        "path": str(requested),
        "sha256": accepted_sha,
        "size": accepted_size,
        "mode": expected_mode,
        "device": file_before.st_dev,
        "inode": file_before.st_ino,
        "nlink": file_before.st_nlink,
        "held_fd_double_hash_verified": True,
        "held_openat_parent_chain_replayed": True,
    }


def require_identical_replay(before: Any, after: Any, *, label: str) -> None:
    """Fail closed when any byte/object/identity binding changed in a run."""

    if before != after:
        fail(f"{label} changed during exact16 materialization")


def _runtime_source_cli_pins(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "materialize_elal3_simulator_c2_vae_v1": {
            "sha256": args.expected_materializer_source_sha256,
            "size": args.expected_materializer_source_size,
        },
        "elal3_simulator_c2_label_v1": {
            "sha256": args.expected_label_source_sha256,
            "size": args.expected_label_source_size,
        },
        "elal3_c0_v1": {
            "sha256": args.expected_elal3_c0_source_sha256,
            "size": args.expected_elal3_c0_source_size,
        },
        "train_lora": {
            "sha256": args.expected_train_lora_source_sha256,
            "size": args.expected_train_lora_source_size,
        },
        "tools.materialize_vae": {
            "sha256": args.expected_materialize_vae_source_sha256,
            "size": args.expected_materialize_vae_source_size,
        },
        "tools.build_renderer_dataset": {
            "sha256": args.expected_build_renderer_dataset_source_sha256,
            "size": args.expected_build_renderer_dataset_source_size,
        },
    }


def validate_runtime_sources(
    cli_pins: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind the actual files backing every local module used by this CLI."""

    module_rows = {
        "materialize_elal3_simulator_c2_vae_v1": (
            sys.modules[__name__],
            Path(__file__).resolve(strict=True),
            None,
        ),
        "elal3_simulator_c2_label_v1": (
            labels,
            (METHOD_ROOT / "elal3_simulator_c2_label_v1.py").resolve(strict=True),
            RUNTIME_SOURCE_PINS["elal3_simulator_c2_label_v1"],
        ),
        "elal3_c0_v1": (
            labels.elal3,
            (METHOD_ROOT / "elal3_c0_v1.py").resolve(strict=True),
            RUNTIME_SOURCE_PINS["elal3_c0_v1"],
        ),
        "train_lora": (
            legacy,
            (METHOD_ROOT / "train_lora.py").resolve(strict=True),
            RUNTIME_SOURCE_PINS["train_lora"],
        ),
        "tools.materialize_vae": (
            materialize_vae,
            (METHOD_ROOT / "tools/materialize_vae.py").resolve(strict=True),
            RUNTIME_SOURCE_PINS["tools.materialize_vae"],
        ),
        "tools.build_renderer_dataset": (
            materialize_vae.raw_builder,
            (METHOD_ROOT / "tools/build_renderer_dataset.py").resolve(strict=True),
            RUNTIME_SOURCE_PINS["tools.build_renderer_dataset"],
        ),
    }
    if tuple(cli_pins) != tuple(module_rows):
        fail("runtime source CLI pin order/closure differs")
    bindings: list[dict[str, Any]] = []
    for index, (name, (module, registered, embedded)) in enumerate(
        module_rows.items()
    ):
        pin = cli_pins[name]
        expected_sha = pin.get("sha256")
        expected_size = pin.get("size")
        if (
            not isinstance(expected_sha, str)
            or _SHA256.fullmatch(expected_sha) is None
            or type(expected_size) is not int
            or expected_size <= 0
        ):
            fail(f"runtime source CLI pin differs: {name}")
        if embedded is not None and (
            expected_sha != embedded["sha256"]
            or expected_size != embedded["size"]
        ):
            fail(f"runtime source embedded/CLI pin differs: {name}")
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            fail(f"runtime module has no source file: {name}")
        try:
            actual = Path(module_file).resolve(strict=True)
        except OSError as error:
            raise ELAL3SimulatorC2VAEError(
                f"runtime module source is unavailable: {name}"
            ) from error
        if actual != registered:
            fail(f"runtime module __file__ is not registered: {name}")
        relative_path = str(actual.relative_to(METHOD_ROOT))
        binding = stable_stream_hash_path(
            actual,
            label=f"runtime source {name}",
            expected_sha256=expected_sha,
            expected_size=expected_size,
            expected_mode=0o644,
            allowed_root=METHOD_ROOT,
        )
        bindings.append(
            {
                "source_index": index,
                "module_name": name,
                "relative_path": relative_path,
                "actual_module_file_verified": True,
                **binding,
            }
        )
    return bindings


def load_control_closure(
    *,
    derivative_path: Path,
    derivative_sha256: str,
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    """Load both external controls and retain their exact byte bindings."""

    derivative = labels.load_external_authority_v1(
        derivative_path, expected_sha256=derivative_sha256
    )
    contract = labels.load_experiment_contract_v1(
        contract_path, expected_sha256=contract_sha256
    )
    derivative_payload, derivative_binding = labels.stable_read_path(
        derivative_path.resolve(strict=True),
        label="external C2 derivative authority materializer replay",
        expected_sha256=derivative_sha256,
        expected_mode=0o644,
        allowed_root=derivative_path.resolve(strict=True).parent,
    )
    contract_payload, contract_binding = labels.stable_read_path(
        contract_path.resolve(strict=True),
        label="C2 experiment contract materializer replay",
        expected_sha256=contract_sha256,
        expected_mode=0o644,
        allowed_root=contract_path.resolve(strict=True).parent,
    )
    if (
        labels._strict_json_bytes(
            derivative_payload, label="external C2 derivative authority replay"
        )
        != derivative
        or labels._strict_json_bytes(
            contract_payload, label="C2 experiment contract replay"
        )
        != contract
    ):
        fail("external C2 control parsed bytes differ")
    return {
        "derivative_value": derivative,
        "derivative_payload": derivative_payload,
        "derivative_binding": derivative_binding,
        "contract_value": contract,
        "contract_payload": contract_payload,
        "contract_binding": contract_binding,
    }


def validate_model_authority(
    path: Path,
    expected_sha256: str,
    *,
    bernini_root: Path,
    checkpoint_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if expected_sha256 != MODEL_AUTHORITY_SHA256:
        fail("C2 real-model authority SHA literal differs")
    requested = path.expanduser()
    if not requested.is_absolute():
        fail("C2 real-model authority path must be absolute")
    resolved = requested.resolve(strict=True)
    if resolved != MODEL_AUTHORITY_PATH.resolve(strict=True):
        fail("C2 real-model authority is not the registered file")
    payload, binding = labels.stable_read_path(
        resolved,
        label="C2 real-model authority",
        expected_sha256=expected_sha256,
        expected_mode=0o644,
        allowed_root=resolved.parent,
    )
    value = labels._strict_json_bytes(payload, label="C2 real-model authority")
    unsigned = dict(value)
    stored_digest = unsigned.pop("authority_digest", None)
    constraints = value.get("constraints")
    rows = value.get("files")
    if (
        set(value)
        != {
            "authority_digest",
            "authorized_row_ids",
            "bernini_root",
            "checkpoint_root",
            "constraints",
            "file_count",
            "files",
            "model_family",
            "python_env_root",
            "runtime_versions",
            "schema_version",
        }
        or value.get("schema_version") != MODEL_AUTHORITY_SCHEMA
        or value.get("authorized_row_ids") != list(labels.C2_ROW_IDS)
        or stored_digest != MODEL_AUTHORITY_DIGEST
        or object_sha256(unsigned) != stored_digest
        or str(bernini_root.resolve(strict=True)) != value.get("bernini_root")
        or str(checkpoint_root.resolve(strict=True)) != value.get("checkpoint_root")
        or value.get("file_count") != 9
        or not isinstance(rows, list)
        or len(rows) != 9
        or not isinstance(constraints, Mapping)
        or constraints.get("allowed_operation")
        != "elal3_c2_simulator_oracle_q_optimizer_diagnostic"
        or constraints.get("max_optimizer_updates_per_arm") != 10
        or constraints.get("formal_c2_authorized") is not False
        or constraints.get("exact160_authorized") is not False
        or constraints.get("scientific_claim_authorized") is not False
        or constraints.get("real_video_claim_authorized") is not False
        or constraints.get("source_instruction_inference_claim_authorized")
        is not False
    ):
        fail("C2 real-model authority closure/digest differs")
    roots = {
        "bernini": Path(str(value["bernini_root"])),
        "checkpoint": Path(str(value["checkpoint_root"])),
        "python_env": Path(str(value["python_env_root"])),
    }
    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "mode",
            "relative_path",
            "root",
            "sha256",
            "size",
        }:
            fail("C2 model file row fields differ")
        root_name = row.get("root")
        relative = row.get("relative_path")
        identity = (str(root_name), str(relative))
        if (
            root_name not in roots
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or identity in seen
            or not isinstance(row.get("sha256"), str)
            or not isinstance(row.get("size"), int)
            or not isinstance(row.get("mode"), int)
        ):
            fail("C2 model file row value differs")
        seen.add(identity)
        bindings.append(
            {
                "authority_row_index": index,
                "authority_root": root_name,
                "authority_relative_path": relative,
                **stable_stream_hash_path(
                    roots[str(root_name)] / relative,
                    label=f"C2 model file {index}",
                    expected_sha256=str(row["sha256"]),
                    expected_size=int(row["size"]),
                    expected_mode=int(row["mode"]),
                    allowed_root=roots[str(root_name)],
                ),
            }
        )
    return value, binding, bindings


def validate_imported_model_modules(
    *,
    model_authority: Mapping[str, Any],
    model_file_bindings: Sequence[Mapping[str, Any]],
    pipeline_module: Any,
    diffusers_module: Any,
    wan_module: Any,
) -> list[dict[str, Any]]:
    """Prove that imported VAE code is the code named by model authority."""

    rows = model_authority.get("files")
    if not isinstance(rows, list):
        fail("C2 model authority file rows unavailable for import binding")
    row_map = {
        (str(row["root"]), str(row["relative_path"])): (index, row)
        for index, row in enumerate(rows)
    }
    roots = {
        "bernini": Path(str(model_authority["bernini_root"])),
        "checkpoint": Path(str(model_authority["checkpoint_root"])),
        "python_env": Path(str(model_authority["python_env_root"])),
    }
    import_rows = (
        ("bernini.pipeline", pipeline_module, "bernini", "bernini/pipeline.py"),
        ("diffusers", diffusers_module, "python_env", "diffusers/__init__.py"),
        (
            "diffusers.models.autoencoders.autoencoder_kl_wan",
            wan_module,
            "python_env",
            "diffusers/models/autoencoders/autoencoder_kl_wan.py",
        ),
    )
    results: list[dict[str, Any]] = []
    for import_index, (import_name, module, root_name, relative_path) in enumerate(
        import_rows
    ):
        authority_item = row_map.get((root_name, relative_path))
        if authority_item is None:
            fail(f"imported model module is absent from authority: {import_name}")
        authority_index, authority_row = authority_item
        expected_path = (roots[root_name] / relative_path).resolve(strict=True)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            fail(f"imported model module has no __file__: {import_name}")
        actual_path = Path(module_file).resolve(strict=True)
        if actual_path != expected_path:
            fail(f"imported model module path differs: {import_name}")
        binding = stable_stream_hash_path(
            actual_path,
            label=f"imported model module {import_name}",
            expected_sha256=str(authority_row["sha256"]),
            expected_size=int(authority_row["size"]),
            expected_mode=int(authority_row["mode"]),
            allowed_root=roots[root_name],
        )
        authoritative_binding = model_file_bindings[authority_index]
        for field in (
            "path",
            "sha256",
            "size",
            "mode",
            "device",
            "inode",
            "nlink",
        ):
            if binding[field] != authoritative_binding[field]:
                fail(f"imported model module binding differs: {import_name}")
        results.append(
            {
                "import_index": import_index,
                "import_name": import_name,
                "authority_row_index": authority_index,
                "authority_root": root_name,
                "authority_relative_path": relative_path,
                "actual_module_file_verified": True,
                **binding,
            }
        )
    vae_encode = getattr(pipeline_module, "_vae_encode", None)
    vae_encode_code = getattr(vae_encode, "__code__", None)
    autoencoder_class = getattr(wan_module, "AutoencoderKLWan", None)
    if (
        not callable(vae_encode)
        or getattr(vae_encode, "__module__", None) != "bernini.pipeline"
        or vae_encode_code is None
        or not isinstance(getattr(vae_encode_code, "co_filename", None), str)
        or Path(vae_encode_code.co_filename).resolve(strict=True)
        != Path(pipeline_module.__file__).resolve(strict=True)
        or getattr(autoencoder_class, "__module__", None)
        != "diffusers.models.autoencoders.autoencoder_kl_wan"
        or getattr(diffusers_module, "AutoencoderKLWan", None)
        is not autoencoder_class
    ):
        fail("imported Bernini VAE callable/class ownership differs")
    return results


def _strict_safetensors_header(payload: bytes) -> Mapping[str, Any]:
    if len(payload) < 8:
        fail("safetensors payload is truncated before header length")
    header_size = int.from_bytes(payload[:8], byteorder="little", signed=False)
    if header_size <= 0 or header_size % 8 != 0 or 8 + header_size > len(payload):
        fail("safetensors header length differs")

    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate safetensors header key: {key!r}")
            result[key] = value
        return result

    try:
        header = json.loads(
            payload[8 : 8 + header_size].decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: fail(
                f"non-finite safetensors header number: {value}"
            ),
        )
    except ELAL3SimulatorC2VAEError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3SimulatorC2VAEError(
            "safetensors header is not strict UTF-8 JSON"
        ) from error
    if not isinstance(header, Mapping):
        fail("safetensors header is not an object")
    return header


def verify_bundle_payload_v1(
    payload: bytes,
    *,
    expected_tensors: Mapping[str, Any],
    tensor_rows: Sequence[Mapping[str, Any]],
    expected_metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Reload serialized bytes and compare every exact16 tensor to memory."""

    import torch
    from safetensors.torch import load as load_safetensors

    header = _strict_safetensors_header(payload)
    metadata = header.get("__metadata__")
    tensor_headers = {key: value for key, value in header.items() if key != "__metadata__"}
    if metadata != dict(expected_metadata):
        fail("published safetensors metadata differs")
    if set(tensor_headers) != set(TENSOR_ORDER) or len(tensor_headers) != 16:
        fail("published safetensors exact16 key closure differs")
    if tuple(expected_tensors) != TENSOR_ORDER:
        fail("in-memory exact16 tensor order differs")
    if tuple(row.get("tensor_key") for row in tensor_rows) != TENSOR_ORDER:
        fail("tensor receipt row order differs")
    for key in TENSOR_ORDER:
        tensor_header = tensor_headers[key]
        if (
            not isinstance(tensor_header, Mapping)
            or set(tensor_header) != {"data_offsets", "dtype", "shape"}
            or tensor_header.get("dtype") != "F32"
            or tensor_header.get("shape") != list(EXPECTED_LATENT_SHAPE)
            or not isinstance(tensor_header.get("data_offsets"), list)
            or len(tensor_header["data_offsets"]) != 2
            or any(type(item) is not int for item in tensor_header["data_offsets"])
            or tensor_header["data_offsets"][0] < 0
            or tensor_header["data_offsets"][1]
            <= tensor_header["data_offsets"][0]
        ):
            fail(f"published safetensors header row differs: {key}")
    try:
        loaded = load_safetensors(payload)
    except Exception as error:
        raise ELAL3SimulatorC2VAEError(
            "published safetensors bytes cannot be reloaded"
        ) from error
    if set(loaded) != set(TENSOR_ORDER) or len(loaded) != 16:
        fail("reloaded safetensors exact16 key closure differs")
    verified_rows: list[dict[str, Any]] = []
    for index, key in enumerate(TENSOR_ORDER):
        actual = loaded[key]
        expected = expected_tensors[key]
        row = tensor_rows[index]
        if (
            not isinstance(actual, torch.Tensor)
            or actual.dtype != torch.float32
            or tuple(map(int, actual.shape)) != EXPECTED_LATENT_SHAPE
            or not bool(torch.isfinite(actual).all().item())
            or not torch.equal(actual.cpu(), expected.detach().cpu())
        ):
            fail(f"reloaded safetensors tensor differs: {key}")
        actual_sha = tensor_sha256(actual)
        if actual_sha != row.get("tensor_sha256"):
            fail(f"reloaded safetensors tensor SHA differs: {key}")
        verified_rows.append(
            {
                "tensor_key": key,
                "tensor_sha256": actual_sha,
                "shape": list(EXPECTED_LATENT_SHAPE),
                "dtype": "torch.float32",
                "equals_prewrite_memory_tensor": True,
            }
        )
    return {
        "serialized_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "serialized_payload_size": len(payload),
        "metadata": dict(expected_metadata),
        "exact16_keys_verified": True,
        "all_tensors_reloaded_from_serialized_bytes": True,
        "tensor_rows": verified_rows,
    }


def tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        fail("tensor hash input differs")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(
        canonical_json_bytes(
            {"dtype": str(tensor.dtype), "shape": list(map(int, tensor.shape))}
        )
        + b"\0"
    )
    byte_view = tensor.view(torch.uint8).reshape(-1)
    for offset in range(0, int(byte_view.numel()), 1 << 20):
        digest.update(bytes(byte_view[offset : offset + (1 << 20)].tolist()))
    return digest.hexdigest()


def _packet_binding(packet: labels.VerifiedC2PacketV1) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row_id in labels.C2_ROW_IDS:
        row = packet.rows[row_id]
        for variant in labels.MEDIA_ORDER:
            rows.append(
                {
                    "row_id": row_id,
                    "variant": variant,
                    "media": dict(row.file_bindings[variant]["media"]),
                    "annotation": dict(row.file_bindings[variant]["annotation"]),
                    "annotation_receipt": dict(
                        row.file_bindings[variant]["annotation_receipt"]
                    ),
                }
            )
    result = {
        "manifest_file_sha256": PACKET_MANIFEST_SHA256,
        "manifest_digest": labels.EXPECTED_MANIFEST_DIGEST,
        "row_ids": list(labels.C2_ROW_IDS),
        "variant_order": list(labels.MEDIA_ORDER),
        "exact_media_count": len(rows),
        "file_rows": rows,
    }
    return {**result, "packet_binding_digest": object_sha256(result)}


def _write_exact_file(path: Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(
        str(path),
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                fail("create-only write made no progress")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--expected-materializer-source-sha256", required=True)
    value.add_argument("--expected-materializer-source-size", required=True, type=int)
    value.add_argument("--expected-label-source-sha256", required=True)
    value.add_argument("--expected-label-source-size", required=True, type=int)
    value.add_argument("--expected-elal3-c0-source-sha256", required=True)
    value.add_argument("--expected-elal3-c0-source-size", required=True, type=int)
    value.add_argument("--expected-train-lora-source-sha256", required=True)
    value.add_argument("--expected-train-lora-source-size", required=True, type=int)
    value.add_argument("--expected-materialize-vae-source-sha256", required=True)
    value.add_argument("--expected-materialize-vae-source-size", required=True, type=int)
    value.add_argument(
        "--expected-build-renderer-dataset-source-sha256", required=True
    )
    value.add_argument(
        "--expected-build-renderer-dataset-source-size", required=True, type=int
    )
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--packet-root", required=True)
    value.add_argument("--packet-manifest-sha256", required=True)
    value.add_argument("--derivative-authority", required=True)
    value.add_argument("--derivative-authority-sha256", required=True)
    value.add_argument("--model-authority", required=True)
    value.add_argument("--model-authority-sha256", required=True)
    value.add_argument("--experiment-contract", required=True)
    value.add_argument("--experiment-contract-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--ack-simulator-c2-oracle-diagnostic", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if not args.ack_simulator_c2_oracle_diagnostic:
        fail("explicit simulator C2 oracle diagnostic acknowledgement is required")
    if args.packet_manifest_sha256 != PACKET_MANIFEST_SHA256:
        fail("packet manifest SHA literal differs")
    runtime_source_cli_pins = _runtime_source_cli_pins(args)
    runtime_source_pre = validate_runtime_sources(runtime_source_cli_pins)
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("output must be one fresh absolute path")
    derivative_path = Path(args.derivative_authority).expanduser()
    contract_path = Path(args.experiment_contract).expanduser()
    control_pre = load_control_closure(
        derivative_path=derivative_path,
        derivative_sha256=args.derivative_authority_sha256,
        contract_path=contract_path,
        contract_sha256=args.experiment_contract_sha256,
    )
    derivative = control_pre["derivative_value"]
    contract = control_pre["contract_value"]
    bernini_arg = Path(args.bernini_root).expanduser()
    checkpoint_arg = Path(args.checkpoint).expanduser()
    model_authority, model_authority_binding, model_pre = validate_model_authority(
        Path(args.model_authority).expanduser(),
        args.model_authority_sha256,
        bernini_root=bernini_arg,
        checkpoint_root=checkpoint_arg,
    )
    packet_root = Path(args.packet_root).expanduser().resolve(strict=True)
    packet_pre = labels.load_verified_c2_packet(packet_root)
    packet_binding_pre = _packet_binding(packet_pre)

    bernini_root, veomni_root, bernini_revision, veomni_revision = (
        legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
        )
    )
    checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import diffusers
    import diffusers.models.autoencoders.autoencoder_kl_wan as wan_module
    import bernini.pipeline as pipeline_module
    from safetensors.torch import save as save_safetensors

    AutoencoderKLWan = wan_module.AutoencoderKLWan
    _vae_encode = pipeline_module._vae_encode
    imported_model_modules_pre = validate_imported_model_modules(
        model_authority=model_authority,
        model_file_bindings=model_pre,
        pipeline_module=pipeline_module,
        diffusers_module=diffusers,
        wan_module=wan_module,
    )

    if not torch.cuda.is_available():
        fail("C2 exact16 VAE materialization requires one GPU")
    device = torch.device("cuda", 0)
    vae = (
        AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    tensors: dict[str, Any] = {}
    tensor_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="elal3-c2-exact16-") as temporary:
        temporary_root = Path(temporary)
        for row_index, row_id in enumerate(labels.C2_ROW_IDS):
            row = packet_pre.rows[row_id]
            for variant in labels.MEDIA_ORDER:
                key = bundle_key(row_index, row_id, variant)
                if key != TENSOR_ORDER[len(tensor_rows)]:
                    fail("exact16 tensor key/order differs")
                media_payload = row.media_bytes[variant]
                media_sha = row.row["media"][variant]["sha256"]
                if hashlib.sha256(media_payload).hexdigest() != media_sha:
                    fail("authenticated in-memory media payload SHA differs")
                media_copy = temporary_root / f"{key}.mp4"
                _write_exact_file(media_copy, media_payload, mode=0o400)
                frames, reported_fps, source_hw = materialize_vae._decode_exact_video(
                    media_copy
                )
                if (
                    int(frames.shape[0]) != FRAME_COUNT
                    or abs(float(reported_fps) - 25.0) > 1e-3
                    or tuple(source_hw) != (96, 128)
                ):
                    fail(f"C2 decoded video metadata differs: {row_id}/{variant}")
                bucket_hw = materialize_vae.source_aspect_bucket(
                    *source_hw,
                    max_pixels=MAX_PIXELS,
                    stride=SPATIAL_STRIDE,
                )
                if bucket_hw != EXPECTED_BUCKET_HW:
                    fail("C2 Bernini source-aspect bucket differs")
                pixels = materialize_vae._resize_video(
                    frames, bucket_hw, None
                ).unsqueeze(0)
                with torch.no_grad():
                    latent = (
                        _vae_encode(
                            vae, pixels.to(device=device, dtype=torch.float32)
                        )
                        .float()
                        .cpu()
                        .contiguous()
                    )
                if (
                    tuple(map(int, latent.shape)) != EXPECTED_LATENT_SHAPE
                    or latent.dtype != torch.float32
                    or not bool(torch.isfinite(latent).all().item())
                ):
                    fail(f"C2 VAE latent differs: {row_id}/{variant}")
                tensors[key] = latent
                tensor_rows.append(
                    {
                        "tensor_key": key,
                        "row_index": row_index,
                        "row_id": row_id,
                        "variant": variant,
                        "shape": list(EXPECTED_LATENT_SHAPE),
                        "dtype": "torch.float32",
                        "tensor_sha256": tensor_sha256(latent),
                        "source_media_sha256": media_sha,
                    }
                )
                del pixels, latent, frames
                torch.cuda.empty_cache()
    if tuple(tensors) != TENSOR_ORDER or len(tensor_rows) != 16:
        fail("C2 exact16 latent tensor closure differs")

    packet_post = labels.load_verified_c2_packet(packet_root)
    packet_binding_post = _packet_binding(packet_post)
    model_authority_post, model_authority_binding_post, model_post = (
        validate_model_authority(
            Path(args.model_authority).expanduser(),
            args.model_authority_sha256,
            bernini_root=bernini_arg,
            checkpoint_root=checkpoint_arg,
        )
    )
    control_post = load_control_closure(
        derivative_path=derivative_path,
        derivative_sha256=args.derivative_authority_sha256,
        contract_path=contract_path,
        contract_sha256=args.experiment_contract_sha256,
    )
    runtime_source_post = validate_runtime_sources(runtime_source_cli_pins)
    imported_model_modules_post = validate_imported_model_modules(
        model_authority=model_authority_post,
        model_file_bindings=model_post,
        pipeline_module=pipeline_module,
        diffusers_module=diffusers,
        wan_module=wan_module,
    )
    require_identical_replay(
        packet_binding_pre, packet_binding_post, label="packet closure"
    )
    require_identical_replay(
        (model_authority, model_authority_binding, model_pre),
        (model_authority_post, model_authority_binding_post, model_post),
        label="real-model authority closure",
    )
    require_identical_replay(
        control_pre, control_post, label="derivative authority/experiment contract"
    )
    require_identical_replay(
        runtime_source_pre, runtime_source_post, label="runtime source closure"
    )
    require_identical_replay(
        imported_model_modules_pre,
        imported_model_modules_post,
        label="imported model module closure",
    )

    output.mkdir(mode=0o700)
    bundle_path = output / "c2-exact16-latents.safetensors"
    receipt_path = output / "latent-bundle-receipt.json"
    bundle_metadata = expected_safetensors_metadata()
    bundle_payload = save_safetensors(tensors, metadata=bundle_metadata)
    serialized_prewrite_verification = verify_bundle_payload_v1(
        bundle_payload,
        expected_tensors=tensors,
        tensor_rows=tensor_rows,
        expected_metadata=bundle_metadata,
    )
    _write_exact_file(bundle_path, bundle_payload, mode=0o400)
    os.chmod(bundle_path, 0o444)
    bundle_sha = hashlib.sha256(bundle_payload).hexdigest()
    published_bundle_payload, bundle_binding = labels.stable_read_path(
        bundle_path,
        label="C2 exact16 latent bundle",
        expected_sha256=bundle_sha,
        expected_mode=0o444,
        allowed_root=output,
    )
    if published_bundle_payload != bundle_payload:
        fail("published bundle bytes differ from serialized in-memory bytes")
    published_bundle_verification = verify_bundle_payload_v1(
        published_bundle_payload,
        expected_tensors=tensors,
        tensor_rows=tensor_rows,
        expected_metadata=bundle_metadata,
    )
    require_identical_replay(
        serialized_prewrite_verification,
        published_bundle_verification,
        label="serialized/published exact16 bundle",
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "ELAL3_SIMULATOR_C2_EXACT16_VAE_GO",
        "bundle": bundle_binding,
        "bundle_format": "safetensors-exact16-fp32-v1",
        "tensor_order": list(TENSOR_ORDER),
        "tensor_order_digest": object_sha256(list(TENSOR_ORDER)),
        "tensor_rows": tensor_rows,
        "row_ids": list(labels.C2_ROW_IDS),
        "variant_order": list(labels.MEDIA_ORDER),
        "exact_media_count": 16,
        "bucket_hw": list(EXPECTED_BUCKET_HW),
        "latent_shape_each": list(EXPECTED_LATENT_SHAPE),
        "safetensors_metadata": bundle_metadata,
        "published_bundle_verification": published_bundle_verification,
        "packet_binding": packet_binding_pre,
        "derivative_authority_binding": {
            "relative_path": labels.EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH,
            "file": control_pre["derivative_binding"],
            "file_sha256": DERIVATIVE_AUTHORITY_SHA256,
            "authority_digest": derivative["authority_digest"],
            "schema_version": derivative["schema_version"],
            "verified_before_and_after_encoding": True,
        },
        "experiment_contract_binding": {
            "relative_path": labels.EXPECTED_EXPERIMENT_CONTRACT_RELATIVE_PATH,
            "file": control_pre["contract_binding"],
            "file_sha256": EXPERIMENT_CONTRACT_SHA256,
            "contract_digest": contract["contract_digest"],
            "schema_version": contract["schema_version"],
            "renderer_timestep_dtype": "torch.int64",
            "renderer_timestep_value": 999,
            "sigma_float32": 1.0,
            "x_sigma": "epsilon",
            "verified_before_and_after_encoding": True,
        },
        "real_model_authority_binding": {
            "relative_path": MODEL_AUTHORITY_RELATIVE_PATH,
            "file": model_authority_binding,
            "file_sha256": MODEL_AUTHORITY_SHA256,
            "authority_digest": model_authority["authority_digest"],
            "schema_version": model_authority["schema_version"],
            "verified_file_bindings": model_pre,
            "verified_before_and_after_encoding": True,
        },
        "runtime_source_bindings": {
            "source_count": len(runtime_source_pre),
            "sources": runtime_source_pre,
            "verified_actual_import_module_files": True,
            "verified_before_and_after_encoding": True,
            "self_pin_caller_supplied_not_standalone_authority": True,
            "trainer_consumption_requires_external_release_pin": True,
        },
        "imported_model_module_bindings": {
            "module_count": len(imported_model_modules_pre),
            "modules": imported_model_modules_pre,
            "verified_before_and_after_encoding": True,
        },
        "runtime": {
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "checkpoint_path": str(checkpoint),
            "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
            "transformer_layers": transformer_config.get("num_layers"),
        },
        "encoding": {
            "vae_encode_count": 16,
            "each_media_independently_full_video_vae_encoded": True,
            "source_media_consumed_from_authenticated_held_fd_bytes": True,
            "packet_replayed_before_and_after_encoding": True,
            "model_files_double_hashed_before_and_after_encoding": True,
            "derivative_authority_replayed_before_and_after_encoding": True,
            "experiment_contract_replayed_before_and_after_encoding": True,
            "runtime_sources_replayed_before_and_after_encoding": True,
            "bundle_serialized_to_bytes_then_create_only_written": True,
            "published_bundle_reloaded_and_exact16_verified": True,
        },
        "authority": {
            "teacher_forced_oracle_q_required_for_optimizer_use": True,
            "training_authority_is_external_and_narrow": True,
            "formal_c2_authorized": False,
            "exact160_authorized": False,
            "scientific_claim_authorized": False,
            "real_video_data": False,
            "source_instruction_inference_authorized": False,
            "materializer_source_independently_authorized_here": False,
            "trainer_consumption_requires_external_release_pin": True,
        },
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    receipt_payload = canonical_json_bytes(receipt) + b"\n"
    _write_exact_file(receipt_path, receipt_payload, mode=0o400)
    os.chmod(receipt_path, 0o444)
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    os.chmod(output, 0o555)
    final_bundle_payload, final_bundle_binding = labels.stable_read_path(
        bundle_path,
        label="final C2 exact16 latent bundle replay",
        expected_sha256=bundle_sha,
        expected_mode=0o444,
        allowed_root=output,
    )
    final_receipt_payload, final_receipt_binding = labels.stable_read_path(
        receipt_path,
        label="final C2 exact16 latent receipt replay",
        expected_sha256=receipt_sha,
        expected_mode=0o444,
        allowed_root=output,
    )
    if (
        final_bundle_payload != published_bundle_payload
        or final_bundle_binding != bundle_binding
        or final_receipt_payload != receipt_payload
        or labels._canonical_json_payload(
            final_receipt_payload, label="final C2 exact16 latent receipt"
        )
        != receipt
    ):
        fail("final exact16 bundle/receipt stable replay differs")
    final_bundle_verification = verify_bundle_payload_v1(
        final_bundle_payload,
        expected_tensors=tensors,
        tensor_rows=tensor_rows,
        expected_metadata=bundle_metadata,
    )
    require_identical_replay(
        published_bundle_verification,
        final_bundle_verification,
        label="published/final exact16 bundle",
    )
    print(
        json.dumps(
            {
                "status": "ELAL3_SIMULATOR_C2_EXACT16_VAE_GO",
                "bundle": str(bundle_path),
                "bundle_sha256": bundle_sha,
                "bundle_size": bundle_binding["size"],
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "receipt_size": final_receipt_binding["size"],
                "receipt_digest": receipt["receipt_digest"],
                "tensor_count": 16,
                "training_started": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ELAL3SimulatorC2VAEError",
    "EXPECTED_BUCKET_HW",
    "EXPECTED_LATENT_SHAPE",
    "MODEL_AUTHORITY_SHA256",
    "RECEIPT_SCHEMA",
    "RUNTIME_SOURCE_PINS",
    "SCHEMA_VERSION",
    "TENSOR_ORDER",
    "bundle_key",
    "expected_safetensors_metadata",
    "load_control_closure",
    "main",
    "require_identical_replay",
    "stable_stream_hash_path",
    "validate_imported_model_modules",
    "validate_model_authority",
    "validate_runtime_sources",
    "verify_bundle_payload_v1",
]
