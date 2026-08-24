#!/usr/bin/env python3
"""Fail-closed BOX-EXP-014 controller on retained 136141/gpu299.

The controller writes the frozen plan, executes the dedicated exact7
source-only materializer, then reopens every published posterior and seals a
completion receipt.  It never creates an optimizer and has no train command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for _root in (METHOD_ROOT, TOOLS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import full30_action_source7_reencode_plan_v1 as plan_contract  # noqa: E402
import materialize_full30_action_source7_reencode_v1 as materializer  # noqa: E402
import build_full30_action_source7_reencode_release_v1 as release  # noqa: E402
import full30_action_source7_reencode_runtime_cache_v1 as runtime_cache  # noqa: E402


SCHEMA_VERSION = "bernini-full30-action-source7-reencode-controller-completion-v4"
RUN_GENERATION = "r4"
HOLDER_JOB = "136141"
HOLDER_NODE = "auh7-1b-gpu-299"
HOLDER_USER = "guangyi.chen"
EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    materializer.pinned.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
)
EXPECTED_VAE_CONFIG_SHA256 = materializer.pinned.EXPECTED_VAE_CONFIG_SHA256
EXPECTED_TORCH_VERSION = runtime_cache.EXPECTED_TORCH_VERSION
EXPECTED_TORCH_HIP_VERSION = runtime_cache.EXPECTED_TORCH_HIP_VERSION
EXPECTED_MIOPEN_BACKEND_VERSION = runtime_cache.EXPECTED_MIOPEN_BACKEND_VERSION
EXPECTED_MIOPEN_LIBRARY_PATH = Path(runtime_cache.EXPECTED_MIOPEN_LIBRARY_PATH)
EXPECTED_MIOPEN_LIBRARY_SIZE = runtime_cache.EXPECTED_MIOPEN_LIBRARY_SIZE
EXPECTED_MIOPEN_LIBRARY_SHA256 = runtime_cache.EXPECTED_MIOPEN_LIBRARY_SHA256
EXPECTED_MIOPEN_EMBEDDED_VERSION = runtime_cache.EXPECTED_MIOPEN_EMBEDDED_VERSION
EXPECTED_MIOPEN_LIBRARY_OWNER_UID = 2012
EXPECTED_MIOPEN_LIBRARY_MODE = 0o755
WAN_RESAMPLE_CANDIDATE_GEOMETRIES = (
    (96, 544, 432, 545, 433, 272, 216),
    (192, 272, 216, 273, 217, 136, 108),
    (384, 136, 108, 137, 109, 68, 54),
)
FORMAL_NEGATIVE_ACCESS_FIELDS = frozenset(
    {
        "source_only_reencode_from_source_video",
        "vae_encode_calls_per_source",
        "paired_dataset_accessed",
        "legacy_source_target_container_opened",
        "synthetic_target_index1_path_read",
        "synthetic_target_index1_bytes_read",
        "synthetic_target_index1_decoded",
        "synthetic_target_index1_filtered_on",
        "synthetic_target_index1_hashed",
        "target_video_path_present",
        "target_video_accessed",
    }
)


class Source7ReencodeControllerError(RuntimeError):
    """Raised before BOX-EXP-014 completion can be admitted."""


def fail(message: str) -> NoReturn:
    raise Source7ReencodeControllerError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    return plan_contract.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return plan_contract.object_sha256(value)


def file_sha256(path: Path) -> str:
    return materializer.file_sha256(path)


def _hash_and_find_token(path: Path, token: bytes) -> tuple[str, bool]:
    digest = hashlib.sha256()
    found = False
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            if token in overlap + chunk:
                found = True
            overlap = (overlap + chunk)[-max(0, len(token) - 1) :]
    return digest.hexdigest(), found


def _closed_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute() and not path.is_symlink(), f"{label} must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise Source7ReencodeControllerError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def _load_json(path_value: str | Path, expected_sha256: str, *, label: str) -> tuple[Mapping[str, Any], Path, str]:
    path = _plain_file(path_value, label=label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    require(observed == expected_sha256, f"{label} SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_closed_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Source7ReencodeControllerError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Source7ReencodeControllerError(f"{label} is not valid JSON") from error
    require(type(value) is dict, f"{label} must be one object")
    require(raw == canonical_json_bytes(value) + b"\n", f"{label} is not canonical JSON")
    return value, path, observed


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    require(
        path.is_absolute()
        and path.parent.is_dir()
        and not path.parent.is_symlink()
        and not path.exists()
        and not path.is_symlink(),
        "controller output must be a fresh absolute file",
    )
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    require(path.read_bytes() == raw, "controller output physical reopen differs")
    return hashlib.sha256(raw).hexdigest()


def _validate_pinned_torch_miopen_runtime(torch: Any) -> Mapping[str, Any]:
    require(str(torch.__version__) == EXPECTED_TORCH_VERSION, "torch version differs")
    hip_version = getattr(torch.version, "hip", None)
    require(hip_version == EXPECTED_TORCH_HIP_VERSION, "torch HIP build version differs")
    require(
        torch.backends.cudnn.enabled and torch.backends.cudnn.is_available(),
        "ROCm MIOpen backend is unavailable",
    )
    backend_version = torch.backends.cudnn.version()
    require(
        backend_version == EXPECTED_MIOPEN_BACKEND_VERSION,
        "bundled MIOpen backend version differs",
    )
    torch_package_root = Path(torch.__file__).resolve(strict=True).parent
    library_path = _plain_file(
        torch_package_root / "lib/libMIOpen.so", label="bundled MIOpen library"
    )
    library_metadata = library_path.stat()
    require(library_path == EXPECTED_MIOPEN_LIBRARY_PATH, "bundled MIOpen path differs")
    require(
        library_metadata.st_size == EXPECTED_MIOPEN_LIBRARY_SIZE
        and library_metadata.st_uid == EXPECTED_MIOPEN_LIBRARY_OWNER_UID
        and stat.S_IMODE(library_metadata.st_mode) == EXPECTED_MIOPEN_LIBRARY_MODE,
        "bundled MIOpen physical metadata differs",
    )
    library_digest, embedded_version_found = _hash_and_find_token(
        library_path, EXPECTED_MIOPEN_EMBEDDED_VERSION.encode("ascii")
    )
    require(library_digest == EXPECTED_MIOPEN_LIBRARY_SHA256, "bundled MIOpen SHA-256 differs")
    require(embedded_version_found, "bundled MIOpen embedded version differs")
    return {
        "torch_version": str(torch.__version__),
        "torch_hip_version": hip_version,
        "miopen_backend_version": int(backend_version),
        "miopen_library_resolved_path": str(library_path),
        "miopen_library_size": library_metadata.st_size,
        "miopen_library_sha256": library_digest,
        "miopen_embedded_version": EXPECTED_MIOPEN_EMBEDDED_VERSION,
    }


def _run_cuda_miopen_conv_smoke(
    *, prepare_receipt: Mapping[str, Any], prepare_receipt_path: Path,
    prepare_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Import torch after cache validation and replay all r2 WanResample candidates."""

    require("torch" not in sys.modules, "torch was imported before runtime cache validation")
    expected_environment = prepare_receipt["environment"]
    require(
        expected_environment
        == {
            "HOME": os.environ.get("HOME"),
            "MIOPEN_USER_DB_PATH": os.environ.get("MIOPEN_USER_DB_PATH"),
            "MIOPEN_CUSTOM_CACHE_DIR": os.environ.get("MIOPEN_CUSTOM_CACHE_DIR"),
            "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
            "TMPDIR": os.environ.get("TMPDIR"),
        },
        "MIOpen smoke environment differs",
    )
    cache_root = Path(prepare_receipt["cache_root"])
    pre_conv_inventory = runtime_cache.inventory_cache_directories(cache_root)
    global_lock_before_torch = runtime_cache.observe_global_miopen_lock_root()
    require(
        pre_conv_inventory
        == prepare_receipt["post_probe_empty_inventory"]
        == {name: [] for name in runtime_cache.SUBDIRECTORY_NAMES},
        "MIOpen smoke requires an exactly empty fresh cache before torch import",
    )
    require(
        global_lock_before_torch
        == prepare_receipt["global_miopen_lock_root_before_torch"],
        "global MIOpen lock root metadata changed before torch import",
    )
    import torch

    require(torch.cuda.is_available(), "CUDA/ROCm is unavailable for MIOpen smoke")
    pinned_runtime = _validate_pinned_torch_miopen_runtime(torch)
    library_path = Path(pinned_runtime["miopen_library_resolved_path"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    input_tensor = padded = weight = bias = output = None
    geometry_receipts: list[dict[str, Any]] = []
    peak_allocated = None
    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for index, (
                channels,
                input_height,
                input_width,
                padded_height,
                padded_width,
                output_height,
                output_width,
            ) in enumerate(WAN_RESAMPLE_CANDIDATE_GEOMETRIES):
                input_tensor = torch.ones(
                    (1, channels, input_height, input_width),
                    dtype=torch.float32,
                    device=device,
                )
                weight = torch.ones(
                    (channels, channels, 3, 3),
                    dtype=torch.float32,
                    device=device,
                )
                bias = torch.ones((channels,), dtype=torch.float32, device=device)
                require(
                    input_tensor.dtype == weight.dtype == bias.dtype == torch.float32,
                    "WanResample smoke operands must enter autocast as FP32",
                )
                padded = torch.nn.functional.pad(input_tensor, (0, 1, 0, 1))
                require(
                    list(padded.shape)
                    == [1, channels, padded_height, padded_width],
                    "WanResample smoke padded input shape differs",
                )
                output = torch.nn.functional.conv2d(
                    padded, weight, bias=bias, stride=(2, 2), padding=0
                )
                torch.cuda.synchronize(device)
                require(
                    list(output.shape)
                    == [1, channels, output_height, output_width],
                    "WanResample smoke output shape differs",
                )
                require(output.dtype == torch.bfloat16, "WanResample smoke autocast dtype differs")
                require(bool(torch.isfinite(output).all().item()), "WanResample smoke output is non-finite")
                observed_samples_tensor = torch.stack(
                    (
                        output[0, 0, 0, 0],
                        output[0, 0, -1, 0],
                        output[0, 0, 0, -1],
                        output[0, 0, -1, -1],
                    )
                )
                expected_samples_tensor = torch.tensor(
                    [
                        channels * 9 + 1,
                        channels * 6 + 1,
                        channels * 6 + 1,
                        channels * 4 + 1,
                    ],
                    dtype=torch.float32,
                    device=device,
                ).to(dtype=torch.bfloat16)
                require(
                    bool(torch.equal(observed_samples_tensor, expected_samples_tensor)),
                    "WanResample smoke boundary numerics differ",
                )
                observed_samples = [
                    float(item)
                    for item in observed_samples_tensor.to(
                        device="cpu", dtype=torch.float32
                    ).tolist()
                ]
                geometry_receipts.append(
                    {
                        "candidate_index": index,
                        "pre_pad_input_shape": [
                            1,
                            channels,
                            input_height,
                            input_width,
                        ],
                        "conv_input_shape": [
                            1,
                            channels,
                            padded_height,
                            padded_width,
                        ],
                        "weight_shape": [channels, channels, 3, 3],
                        "bias_shape": [channels],
                        "stride": [2, 2],
                        "padding": [0, 0],
                        "output_shape": [
                            1,
                            channels,
                            output_height,
                            output_width,
                        ],
                        "module_and_input_declared_dtype": "torch.float32",
                        "cuda_autocast_dtype": "torch.bfloat16",
                        "output_dtype": "torch.bfloat16",
                        "boundary_samples_fp32": observed_samples,
                        "finite": True,
                        "cuda_synchronized": True,
                    }
                )
                del observed_samples_tensor, expected_samples_tensor
                del output, bias, weight, padded, input_tensor
                output = bias = weight = padded = input_tensor = None
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            require(peak_allocated > 0, "MIOpen smoke did not allocate GPU memory")
    finally:
        del output, bias, weight, padded, input_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    allocated_after_clear = int(torch.cuda.memory_allocated(device))
    require(allocated_after_clear == 0, "MIOpen smoke GPU allocations remain live")
    post_conv_inventory, kernel_db_evidence = (
        runtime_cache.validate_miopen_kernel_cache_activity(cache_root)
    )
    lock_inventory, temp_lock_evidence = (
        runtime_cache.validate_scoped_miopen_temp_lock_activity(cache_root)
    )
    require(
        lock_inventory == post_conv_inventory,
        "runtime cache changed between kernel DB and scoped lock validation",
    )
    user_db_evidence = runtime_cache.miopen_user_db_evidence(
        post_conv_inventory
    )
    user_db_claim = (
        "path-bound;expected-plaintext-write-observed-not-required"
        if user_db_evidence["plaintext_main_write_observed"]
        else "path-bound;no-write-observed-and-no-write-claim"
    )
    global_lock_after_smoke = runtime_cache.observe_global_miopen_lock_root()
    require(
        global_lock_after_smoke == global_lock_before_torch,
        "global MIOpen lock root metadata changed during redirected smoke",
    )
    try:
        maps_lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Source7ReencodeControllerError("cannot inspect loaded MIOpen mapping") from error
    loaded_miopen_paths = sorted(
        {
            line.split(maxsplit=5)[5]
            for line in maps_lines
            if len(line.split(maxsplit=5)) == 6
            and "libMIOpen.so" in line.split(maxsplit=5)[5]
        }
    )
    require(
        loaded_miopen_paths == [str(library_path)],
        "loaded MIOpen library path differs or is non-unique",
    )
    unsigned: dict[str, Any] = {
        "schema_version": "bernini-full30-action-source7-reencode-miopen-conv-smoke-v3",
        "experiment_id": plan_contract.EXPERIMENT_ID,
        "run_generation": RUN_GENERATION,
        "complete": True,
        "prepare_receipt_path": str(prepare_receipt_path),
        "prepare_receipt_file_sha256": prepare_receipt_sha256,
        "prepare_digest": prepare_receipt["prepare_digest"],
        "cache_root": prepare_receipt["cache_root"],
        "environment": dict(expected_environment),
        "miopen_user_db_path_kind": "directory",
        "miopen_custom_cache_dir_kind": "directory",
        "miopen_cache_paths_remained_canonical_0700_directories": True,
        "pre_conv_cache_inventory": pre_conv_inventory,
        "post_conv_cache_inventory": post_conv_inventory,
        "miopen_kernel_db_activity_required": True,
        "miopen_kernel_db_activity_observed": True,
        "miopen_kernel_db_evidence": kernel_db_evidence,
        "kernel_cache_claim": "path-bound;fresh-ukdb-write-required-and-observed",
        "miopen_user_db_claim": user_db_claim,
        "miopen_user_db_evidence": user_db_evidence,
        "scoped_miopen_temp_lock_activity_required": True,
        "scoped_miopen_temp_lock_activity_observed": True,
        "scoped_miopen_temp_lock_evidence": temp_lock_evidence,
        "tmpdir_cpp_temp_directory_path_redirect_observed": True,
        "global_miopen_lock_root_before_torch": global_lock_before_torch,
        "global_miopen_lock_root_after_smoke": global_lock_after_smoke,
        "global_miopen_lock_root_metadata_unchanged": True,
        "global_miopen_lock_root_members_scanned": False,
        "global_miopen_lock_root_mutation_attempted": False,
        "torch_import_after_cache_validation": True,
        "pinned_runtime": pinned_runtime,
        "loaded_miopen_library_paths": loaded_miopen_paths,
        "loaded_miopen_library_unique_exact_path": True,
        "backend": "ROCm-MIOpen-via-torch.backends.cudnn",
        "device_index": int(device.index or 0),
        "device_name": str(torch.cuda.get_device_name(device)),
        "operation": "WanResample-downsample-ZeroPad2d-right-bottom-1-then-Conv2d",
        "r2_failure_step": "136141.115",
        "r2_failure_stack_location": "diffusers/models/autoencoders/autoencoder_kl_wan.py:298",
        "r2_failure_stack_candidate_closure": "all-three-first-temporal-chunk-spatial-downsample-convs",
        "vae_config_sha256": EXPECTED_VAE_CONFIG_SHA256,
        "geometry_count": len(geometry_receipts),
        "geometries": geometry_receipts,
        "module_and_input_declared_dtype": "torch.float32",
        "cuda_autocast_dtype": "torch.bfloat16",
        "peak_allocated_bytes": peak_allocated,
        "gpu_cache_cleared": True,
        "gpu_memory_allocated_after_clear": allocated_after_clear,
        "source_video_opened": False,
        "source_video_decoded": False,
        "vae_encode_calls": 0,
    }
    receipt = {**unsigned, "smoke_digest": object_sha256(unsigned)}
    require(
        set(receipt) == runtime_cache.SMOKE_FIELDS,
        "MIOpen smoke receipt field closure differs",
    )
    return receipt


def validate_release_tree(
    *, method_root: str | Path, manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    root = Path(method_root)
    require(root.is_absolute() and not root.is_symlink(), "release method root differs")
    try:
        resolved_root = root.resolve(strict=True)
        metadata = root.lstat()
    except OSError as error:
        raise Source7ReencodeControllerError("release method root is unavailable") from error
    require(
        resolved_root == root and stat.S_ISDIR(metadata.st_mode),
        "release method root must be one canonical directory",
    )
    manifest, _, _ = _load_json(
        manifest_path, expected_manifest_sha256, label="release manifest"
    )
    try:
        release.validate_manifest(manifest)
    except release.Source7ReencodeReleaseError as error:
        raise Source7ReencodeControllerError(str(error)) from error
    expected_paths = {row["path"] for row in manifest["files"]}
    observed_paths: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        require(candidate.is_file() and not candidate.is_symlink(), "release tree contains a non-plain member")
        observed_paths.add(candidate.relative_to(root).as_posix())
    require(observed_paths == expected_paths, "release tree exact member closure differs")
    for row in manifest["files"]:
        member = _plain_file(root / row["path"], label=f"release member {row['path']}")
        require(
            member.stat().st_size == row["size"]
            and stat.S_IMODE(member.stat().st_mode) == row["mode"]
            and file_sha256(member) == row["sha256"],
            f"release member identity differs: {row['path']}",
        )
    return manifest


def _required_negative_access_closure(value: Mapping[str, Any], *, label: str) -> None:
    expected = materializer._negative_access_closure()
    require(
        set(expected) == FORMAL_NEGATIVE_ACCESS_FIELDS
        == materializer.NEGATIVE_ACCESS_FIELDS,
        "formal negative-access field registry differs",
    )
    for key, item in expected.items():
        observed = value.get(key)
        require(
            observed is item
            if type(item) is bool
            else type(observed) is type(item) and observed == item,
            f"{label}.{key} differs",
        )


def validate_published_materialization(
    *, output_root: str | Path, plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    import torch

    root = Path(output_root)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "materialization root differs")
    root = root.resolve(strict=True)
    receipt_path = root / "materialization_receipt.json"
    receipt_sha = file_sha256(_plain_file(receipt_path, label="materialization receipt"))
    receipt, _, _ = _load_json(receipt_path, receipt_sha, label="materialization receipt")
    require(
        set(receipt) == materializer.RECEIPT_FIELDS,
        "materialization receipt top-level field closure differs",
    )
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    require(declared == object_sha256(unsigned), "materialization receipt digest differs")
    require(
        receipt.get("schema_version") == materializer.SCHEMA_VERSION
        and receipt.get("complete") is True
        and receipt.get("row_count") == 7
        and receipt.get("total_vae_encode_calls") == 7
        and receipt.get("distinct_source_mp4_count") == 7,
        "materialization receipt exact7 closure differs",
    )
    plan_binding = receipt.get("plan")
    require(
        type(plan_binding) is dict
        and set(plan_binding) == {"path", "file_sha256", "plan_digest"},
        "materialization receipt plan binding closure differs",
    )
    bound_plan_path = _plain_file(plan_binding["path"], label="receipt-bound plan")
    expected_plan_file_sha256 = hashlib.sha256(
        canonical_json_bytes(plan) + b"\n"
    ).hexdigest()
    require(
        plan_binding
        == {
            "path": str(bound_plan_path),
            "file_sha256": expected_plan_file_sha256,
            "plan_digest": plan["plan_digest"],
        }
        and file_sha256(bound_plan_path) == expected_plan_file_sha256
        and receipt.get("output_root") == str(root),
        "materialization receipt plan/output binding differs",
    )
    vae_identity = receipt.get("vae_identity")
    require(
        type(vae_identity) is dict
        and vae_identity.get("checkpoint_content_manifest_sha256")
        == EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        and vae_identity.get("every_vae_file_sha256_verified") is True
        and vae_identity.get("posterior_representation")
        == "latent_dist.parameters_fp32"
        and vae_identity.get("posterior_sample_materialized") is False,
        "materialization pinned VAE identity differs",
    )
    _required_negative_access_closure(receipt, label="materialization receipt")
    expected_names = [row["output_filename"] for row in plan["rows"]]
    require(receipt.get("output_filenames") == expected_names, "materialization output order differs")
    require(
        {item.name for item in root.iterdir()}
        == set(expected_names) | {"materialization_receipt.json"},
        "published materialization member closure differs",
    )
    receipt_rows = receipt.get("rows")
    require(type(receipt_rows) is list and len(receipt_rows) == 7, "materialization row closure differs")
    require(
        all(
            type(row) is dict and set(row) == materializer.ROW_RECEIPT_FIELDS
            for row in receipt_rows
        ),
        "materialization per-row field closure differs",
    )
    by_iid = {row["iid"]: row for row in receipt_rows}
    require(len(by_iid) == 7, "materialization receipt IIDs differ")
    post_publish: list[dict[str, Any]] = []
    for planned in plan["rows"]:
        iid = planned["iid"]
        row = by_iid.get(iid)
        require(type(row) is dict, f"{iid} receipt row differs")
        _required_negative_access_closure(row, label=f"row {iid}")
        require(
            row.get("analysis_split") == planned["analysis_split"]
            and row.get("event_id") == planned["event_id"]
            and row.get("actor_kind") == planned["actor_kind"]
            and row.get("q0_id") == planned["q0_id"]
            and row.get("group_id") == planned["group_id"]
            and row.get("actor_id") == planned["actor_id"]
            and row.get("scene_id") == planned["scene_id"]
            and row.get("source_video_path") == planned["source_video_path"]
            and row.get("source_video_sha256") == planned["source_video_sha256"]
            and row.get("source_video_sha256_before_decode")
            == planned["source_video_sha256"]
            and row.get("source_video_sha256_after_decode")
            == planned["source_video_sha256"]
            and row.get("source_video_pre_post_stat_and_hash_stable") is True
            and row.get("frame_count") == planned["frame_count"]
            and row.get("expected_fps") == planned["fps"]
            and row.get("posterior_parameters_shape")
            == planned["expected_posterior_shape"]
            and row.get("posterior_parameters_dtype") == "torch.float32"
            and row.get("posterior_parameters_device") == "cpu"
            and row.get("posterior_parameters_layout") == "torch.strided"
            and row.get("posterior_parameters_contiguous") is True
            and row.get("posterior_parameters_finite") is True
            and row.get("posterior_parameters_bare_tensor") is True
            and row.get("posterior_sample_materialized") is False
            and row.get("physical_file_reopened_after_write") is True
            and row.get("physical_tensor_reopened_after_write") is True
            and row.get("physical_tensor_equal_to_encoded_tensor") is True,
            f"{iid} source/posterior receipt binding differs",
        )
        path = _plain_file(root / planned["output_filename"], label=f"{iid} posterior")
        require(str(path) == row["posterior_parameters_path"], f"{iid} output path binding differs")
        raw = path.read_bytes()
        file_digest = hashlib.sha256(raw).hexdigest()
        require(file_digest == row["posterior_parameters_file_sha256"], f"{iid} file SHA differs")
        tensor = materializer._decode_bare_tensor(
            raw,
            planned["expected_posterior_shape"],
            label=f"{iid} post-publish posterior",
        )
        require(
            materializer.base._tensor_sha256(tensor)
            == row["posterior_parameters_tensor_sha256"]
            and materializer._tensor_raw_sha256(tensor)
            == row["posterior_parameters_tensor_raw_sha256"]
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
            and bool(torch.isfinite(tensor).all().item()),
            f"{iid} post-publish tensor replay differs",
        )
        post_publish.append(
            {
                "iid": iid,
                "path": str(path),
                "file_sha256": file_digest,
                "tensor_sha256": row["posterior_parameters_tensor_sha256"],
                "tensor_raw_sha256": row["posterior_parameters_tensor_raw_sha256"],
                "shape": planned["expected_posterior_shape"],
                "physical_file_and_tensor_reopened_post_publish": True,
            }
        )
    external = receipt.get("external_existing_index0")
    expected_external = {
        **plan["external_existing_index0"],
        "opened_by_materializer": False,
        "included_in_exact7_output_files": False,
        "reencoded": False,
    }
    require(
        external == expected_external,
        "external existing 2d2 index0 scope differs",
    )
    return {
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "post_publish_rows": post_publish,
        "all_seven_physical_files_and_tensors_reopened": True,
    }


def run(
    *,
    method_root: str | Path,
    release_manifest: str | Path,
    expected_release_manifest_sha256: str,
    plan_output: str | Path,
    checkpoint: str | Path,
    checkpoint_content_manifest: str | Path,
    expected_checkpoint_content_manifest_sha256: str,
    materialization_output_root: str | Path,
    completion_output: str | Path,
    runtime_cache_receipt: str | Path,
    expected_runtime_cache_receipt_sha256: str,
    enforce_live_holder: bool = True,
) -> Mapping[str, Any]:
    require("torch" not in sys.modules, "torch was imported before r4 child cache validation")
    if enforce_live_holder:
        require(os.environ.get("SLURM_JOB_ID") == HOLDER_JOB, "controller must run inside retained holder 136141")
        require(socket.gethostname().split(".")[0] == HOLDER_NODE, "controller holder node differs")
    require(
        expected_checkpoint_content_manifest_sha256
        == EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "checkpoint content manifest authority differs",
    )
    manifest = validate_release_tree(
        method_root=method_root,
        manifest_path=release_manifest,
        expected_manifest_sha256=expected_release_manifest_sha256,
    )
    require(manifest["release_generation"] == RUN_GENERATION, "controller release generation differs")
    prepare_receipt_path = Path(runtime_cache_receipt)
    prepare_receipt = runtime_cache.validate_prepare_receipt(
        receipt_path=prepare_receipt_path,
        expected_sha256=expected_runtime_cache_receipt_sha256,
        environ=os.environ,
        require_cache_present=True,
    )
    require("torch" not in sys.modules, "torch was imported during r4 cache validation")
    cuda_miopen_smoke = _run_cuda_miopen_conv_smoke(
        prepare_receipt=prepare_receipt,
        prepare_receipt_path=prepare_receipt_path,
        prepare_receipt_sha256=expected_runtime_cache_receipt_sha256,
    )
    require(
        cuda_miopen_smoke["complete"] is True
        and cuda_miopen_smoke["geometry_count"] == 3
        and cuda_miopen_smoke["miopen_kernel_db_activity_required"] is True
        and cuda_miopen_smoke["miopen_kernel_db_activity_observed"] is True
        and cuda_miopen_smoke["miopen_kernel_db_evidence"]["kern_db_nonempty"]
        is True
        and cuda_miopen_smoke["scoped_miopen_temp_lock_activity_required"]
        is True
        and cuda_miopen_smoke["scoped_miopen_temp_lock_activity_observed"]
        is True
        and cuda_miopen_smoke["global_miopen_lock_root_metadata_unchanged"]
        is True
        and cuda_miopen_smoke["loaded_miopen_library_unique_exact_path"] is True
        and cuda_miopen_smoke["source_video_opened"] is False
        and cuda_miopen_smoke["source_video_decoded"] is False
        and cuda_miopen_smoke["vae_encode_calls"] == 0,
        "pre-source MIOpen smoke closure differs",
    )
    plan = plan_contract.validate_plan(plan_contract.canonical_plan())
    plan_path = Path(plan_output)
    plan_sha256 = _write_json_create_only(plan_path, plan)
    receipt = materializer.materialize(
        plan_path=plan_path,
        expected_plan_sha256=plan_sha256,
        checkpoint=checkpoint,
        checkpoint_content_manifest=checkpoint_content_manifest,
        expected_checkpoint_content_manifest_sha256=(
            expected_checkpoint_content_manifest_sha256
        ),
        output_root=materialization_output_root,
        device="cuda:0",
    )
    require(receipt.get("complete") is True, "materializer did not complete")
    physical = validate_published_materialization(
        output_root=materialization_output_root, plan=plan
    )
    final_cache_inventory, final_kernel_db_evidence = (
        runtime_cache.validate_miopen_kernel_cache_activity(
            Path(prepare_receipt["cache_root"])
        )
    )
    final_lock_inventory, final_temp_lock_evidence = (
        runtime_cache.validate_scoped_miopen_temp_lock_activity(
            Path(prepare_receipt["cache_root"])
        )
    )
    require(
        final_lock_inventory == final_cache_inventory,
        "runtime cache changed between final kernel DB and lock validation",
    )
    final_user_db_evidence = runtime_cache.miopen_user_db_evidence(
        final_cache_inventory
    )
    global_lock_after_exact7 = runtime_cache.observe_global_miopen_lock_root()
    require(
        global_lock_after_exact7
        == cuda_miopen_smoke["global_miopen_lock_root_before_torch"],
        "global MIOpen lock root metadata changed during exact7 materialization",
    )
    require(
        final_kernel_db_evidence["kern_db_row_count"]
        >= cuda_miopen_smoke["miopen_kernel_db_evidence"]["kern_db_row_count"],
        "MIOpen kernel DB regressed after exact7 materialization",
    )
    runtime_cache_post_materialization = {
        "captured_after_exact7_materialization": True,
        "cache_root": prepare_receipt["cache_root"],
        "inventory": final_cache_inventory,
        "miopen_kernel_db_evidence": final_kernel_db_evidence,
        "miopen_user_db_evidence": final_user_db_evidence,
        "scoped_miopen_temp_lock_evidence": final_temp_lock_evidence,
        "global_miopen_lock_root_after_exact7": global_lock_after_exact7,
        "global_miopen_lock_root_metadata_unchanged": True,
        "global_miopen_lock_root_members_scanned": False,
        "global_miopen_lock_root_mutation_attempted": False,
    }
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": plan_contract.EXPERIMENT_ID,
        "run_generation": RUN_GENERATION,
        "complete": True,
        "purpose": plan["purpose"],
        "scientific_target": plan["scientific_target"],
        "learning_target": plan["learning_target"],
        "numeric_target": plan["numeric_target"],
        "dataset": plan["dataset"],
        "steps": plan["steps"],
        "baseline": plan["baseline"],
        "core_validation": plan["core_validation"],
        "holder": {
            "job_id": int(HOLDER_JOB),
            "step_id": int(prepare_receipt["slurm_step_id"]),
            "node": HOLDER_NODE,
            "parent_retained": True,
            "parent_cancelled": False,
            "parent_released": False,
            "parent_requeued": False,
        },
        "release": {
            "manifest_path": str(Path(release_manifest)),
            "manifest_file_sha256": expected_release_manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
            "content_closure_sha1": manifest["content_closure_sha1"],
            "release_generation": manifest["release_generation"],
        },
        "runtime_cache": {
            "prepare_receipt_path": str(prepare_receipt_path),
            "prepare_receipt_file_sha256": expected_runtime_cache_receipt_sha256,
            "prepare_digest": prepare_receipt["prepare_digest"],
            "cache_root": prepare_receipt["cache_root"],
            "hostname": prepare_receipt["hostname"],
            "cache_root_device": prepare_receipt["cache_root_device"],
            "cache_root_inode": prepare_receipt["cache_root_inode"],
            "filesystem": dict(prepare_receipt["filesystem"]),
            "directories": dict(prepare_receipt["directories"]),
            "environment": dict(prepare_receipt["environment"]),
            "home_unchanged": True,
            "created_fresh_create_only": True,
            "exclusive_fsync_probe": dict(prepare_receipt["exclusive_fsync_probe"]),
            "sqlite_commit_reopen_probe": dict(prepare_receipt["sqlite_commit_reopen_probe"]),
            "post_probe_empty_inventory": dict(
                prepare_receipt["post_probe_empty_inventory"]
            ),
            "global_miopen_lock_root_before_torch": dict(
                prepare_receipt["global_miopen_lock_root_before_torch"]
            ),
            "validated_before_torch_import": True,
            "cache_reusable": False,
            "cleanup_policy": prepare_receipt["cleanup_policy"],
        },
        "cuda_miopen_smoke": cuda_miopen_smoke,
        "runtime_cache_post_materialization": runtime_cache_post_materialization,
        "plan": {
            "path": str(plan_path),
            "file_sha256": plan_sha256,
            "plan_digest": plan["plan_digest"],
        },
        "materialization": physical,
        "external_existing_index0": dict(plan["external_existing_index0"]),
        "external_existing_index0_reencoded": False,
        "inventory_snapshot_only": True,
        "exact8_authority_go_claimed": False,
        "teacher_cross_disjointness_pending": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "training_authorized": False,
        **materializer._negative_access_closure(),
    }
    completion = {**unsigned, "completion_digest": object_sha256(unsigned)}
    _write_json_create_only(Path(completion_output), completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--expected-release-manifest-sha256", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--materialization-output-root", required=True)
    parser.add_argument("--completion-output", required=True)
    parser.add_argument("--runtime-cache-receipt", required=True)
    parser.add_argument("--expected-runtime-cache-receipt-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        method_root=args.method_root,
        release_manifest=args.release_manifest,
        expected_release_manifest_sha256=args.expected_release_manifest_sha256,
        plan_output=args.plan_output,
        checkpoint=args.checkpoint,
        checkpoint_content_manifest=args.checkpoint_content_manifest,
        expected_checkpoint_content_manifest_sha256=(
            args.expected_checkpoint_content_manifest_sha256
        ),
        materialization_output_root=args.materialization_output_root,
        completion_output=args.completion_output,
        runtime_cache_receipt=args.runtime_cache_receipt,
        expected_runtime_cache_receipt_sha256=(
            args.expected_runtime_cache_receipt_sha256
        ),
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
