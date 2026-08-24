#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the Round37 native activation canary.

This process never imports Torch, initializes distributed state, loads model
weights, or writes output.  A caller cannot provide an expected trust hash.
The moving candidate keeps every compiled release pin ``None`` and therefore
always exits 3, even if a caller supplies locally authored packet bytes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-oracle-regeneration-native-activation-v2-preflight-v1"
SPEC_SCHEMA_VERSION = "bernini-oracle-regeneration-native-activation-v2-spec-v1"
METHOD_ROOT = Path(__file__).resolve().parent
SPEC_PATH = METHOD_ROOT / "assets/oracle_regeneration_native_activation_v2_spec.json"
CORE_PATH = METHOD_ROOT / "oracle_regeneration_activation_v2.py"
RUNNER_PATH = METHOD_ROOT / "infer_oracle_regeneration_native_activation_v2.py"
VAE_TOOL_PATH = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_vae_refs_activation_v2.py"
)
PROMPT_TOOL_PATH = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_prompts_activation_v2.py"
)
_COMPONENT_RELATIVE_PATHS = {
    "activation_core": "oracle_regeneration_activation_v2.py",
    "native_runner": "infer_oracle_regeneration_native_activation_v2.py",
    "vae_reference_materializer": (
        "tools/materialize_oracle_regeneration_vae_refs_activation_v2.py"
    ),
    "prompt_materializer": (
        "tools/materialize_oracle_regeneration_prompts_activation_v2.py"
    ),
}
_RUNTIME_DEPENDENCY_RELATIVE_PATHS = {
    "oracle_regeneration_canary_v1.py": "oracle_regeneration_canary_v1.py",
    "native_branch_homotopy_runtime_v1.py": "native_branch_homotopy_runtime_v1.py",
    "native_branch_homotopy_v1.py": "native_branch_homotopy_v1.py",
    "self_guided_action_field_v1.py": "self_guided_action_field_v1.py",
    "tri_branch_unipc.py": "tri_branch_unipc.py",
    "infer_native_identity_generation_canary.py": (
        "infer_native_identity_generation_canary.py"
    ),
    "infer_native_branch_homotopy_canary.py": (
        "infer_native_branch_homotopy_canary.py"
    ),
    "infer_source_kv_carrier_oracle.py": "infer_source_kv_carrier_oracle.py",
    "infer_source_value_residual_oracle.py": (
        "infer_source_value_residual_oracle.py"
    ),
    "infer_native_self_guided_action_field_canary.py": (
        "infer_native_self_guided_action_field_canary.py"
    ),
    "infer_lora.py": "infer_lora.py",
    "tools/materialize_vae.py": "tools/materialize_vae.py",
}
_SPEC_KEYS = {
    "schema_version",
    "status",
    "launch_ready",
    "scope",
    "authority_kind",
    "formal_authority",
    "training_authority",
    "training",
    "optimizer",
    "automatic_replacement",
    "selection_authority",
    "native_only",
    "flowedit",
    "connected_route",
    "learned_gate",
    "world_size",
    "sequence_parallel_size",
    "one_node",
    "candidate_count_per_arm",
    "runner_allowlist",
    "components",
    "frozen_runtime_dependencies",
    "compiled_authority_packet_sha256",
    "compiled_external_ledger_receipt_sha256",
    "cases",
    "scientific_boundary",
    "post_run_contract",
    "mandatory_blockers",
}

# Set only in a later reviewed byte revision, after packet, ledger, runner, and
# spec have all been finalized.  There is deliberately no CLI/env override.
COMPILED_SPEC_SHA256: Optional[str] = None
COMPILED_CORE_SHA256: Optional[str] = None
COMPILED_RUNNER_SHA256: Optional[str] = None
COMPILED_VAE_TOOL_SHA256: Optional[str] = None
COMPILED_PROMPT_TOOL_SHA256: Optional[str] = None
COMPILED_AUTHORITY_PACKET_SHA256: Optional[str] = None
COMPILED_EXTERNAL_LEDGER_SHA256: Optional[str] = None

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ActivationPreflightError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _owned_bytes(path: Path, *, label: str) -> tuple[bytes, Mapping[str, Any]]:
    if not path.is_absolute() or path.is_symlink():
        raise ActivationPreflightError(f"{label} must be an absolute non-symlink file")
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise ActivationPreflightError(f"{label} open failed") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ActivationPreflightError(f"{label} must be a one-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mode,
        row.st_nlink,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(named):
        raise ActivationPreflightError(f"{label} changed during owned read")
    return b"".join(chunks), {
        "sha256": hashlib.sha256(b"".join(chunks)).hexdigest(),
        "size": int(after.st_size),
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": int(after.st_nlink),
    }


def _strict_json_bytes(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ActivationPreflightError(f"{label} duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except ActivationPreflightError:
        raise
    except Exception as error:
        raise ActivationPreflightError(f"{label} JSON differs") from error
    if not isinstance(value, Mapping):
        raise ActivationPreflightError(f"{label} root must be an object")
    return value


def _all_compiled() -> bool:
    values = (
        COMPILED_SPEC_SHA256,
        COMPILED_CORE_SHA256,
        COMPILED_RUNNER_SHA256,
        COMPILED_VAE_TOOL_SHA256,
        COMPILED_PROMPT_TOOL_SHA256,
        COMPILED_AUTHORITY_PACKET_SHA256,
        COMPILED_EXTERNAL_LEDGER_SHA256,
    )
    return all(isinstance(value, str) and _SHA256.fullmatch(value) for value in values)


def _binding(
    value: Any, *, relative_path: str, sha256: str, label: str
) -> None:
    expected = {"path": relative_path, "sha256": sha256}
    if value != expected:
        raise ActivationPreflightError(f"{label} binding differs")


def _validate_runner_ast(raw: bytes) -> Mapping[str, Any]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(RUNNER_PATH))
    except Exception as error:
        raise ActivationPreflightError("native runner syntax differs") from error
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "main" not in functions or "cpu_preflight" not in functions:
        raise ActivationPreflightError("native runner entry ABI differs")
    source = raw.decode("utf-8")
    required_markers = (
        "LocalOracleNativeBranchRuntimePatchV2",
        "_sample_with_native_initial_noise_observer",
        "official-v2v-base",
        "local-source-reference-r2v4-in-manual-G",
        "ABSTAIN_KEEP_BASE",
        "self_generated_anchor_tensor_used",
    )
    if any(marker not in source for marker in required_markers):
        raise ActivationPreflightError("native runner required ABI marker is absent")
    return {"top_level_functions": sorted(functions), "required_markers": list(required_markers)}


def validate_release(
    *,
    packet_path: Path,
    ledger_path: Path,
) -> Mapping[str, Any]:
    """Return a ready receipt only for the later exact compiled release."""

    files = {
        "spec": SPEC_PATH,
        "core": CORE_PATH,
        "runner": RUNNER_PATH,
        "vae_tool": VAE_TOOL_PATH,
        "prompt_tool": PROMPT_TOOL_PATH,
    }
    observed: dict[str, Mapping[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for label, path in files.items():
        value, identity = _owned_bytes(path.resolve(strict=True), label=label)
        raw[label] = value
        observed[label] = identity
    spec = _strict_json_bytes(raw["spec"], label="activation spec")
    if not _all_compiled():
        raise ActivationPreflightError(
            "activation release pins are intentionally not compiled in this candidate"
        )
    expected_hashes = {
        "spec": COMPILED_SPEC_SHA256,
        "core": COMPILED_CORE_SHA256,
        "runner": COMPILED_RUNNER_SHA256,
        "vae_tool": COMPILED_VAE_TOOL_SHA256,
        "prompt_tool": COMPILED_PROMPT_TOOL_SHA256,
    }
    if any(observed[label]["sha256"] != digest for label, digest in expected_hashes.items()):
        raise ActivationPreflightError("compiled component bytes differ")
    if (
        set(spec) != _SPEC_KEYS
        or spec.get("schema_version") != SPEC_SCHEMA_VERSION
        or spec.get("status") != "ACTIVATED_INDEPENDENT_MODEL_REVIEWED_DIAGNOSTIC_CANARY"
        or spec.get("launch_ready") is not True
        or spec.get("scope") != "experimental diagnostic canary only"
        or spec.get("authority_kind")
        != "diagnostic_exact_packet_and_code_review_trust_root"
        or spec.get("formal_authority") is not False
        or spec.get("training_authority") is not False
        or spec.get("training") is not False
        or spec.get("optimizer") is not False
        or spec.get("automatic_replacement") is not False
        or spec.get("selection_authority") is not None
        or spec.get("native_only") is not True
        or spec.get("flowedit") is not False
        or spec.get("connected_route") is not False
        or spec.get("learned_gate") is not False
        or spec.get("world_size") != 4
        or spec.get("sequence_parallel_size") != 4
        or spec.get("one_node") is not True
        or spec.get("candidate_count_per_arm") != 1
        or spec.get("runner_allowlist") != [RUNNER_PATH.name]
        or spec.get("mandatory_blockers") != []
        or spec.get("compiled_authority_packet_sha256")
        != COMPILED_AUTHORITY_PACKET_SHA256
        or spec.get("compiled_external_ledger_receipt_sha256")
        != COMPILED_EXTERNAL_LEDGER_SHA256
    ):
        raise ActivationPreflightError("activation spec is not exact launch-ready policy")
    components = spec.get("components")
    if (
        not isinstance(components, Mapping)
        or set(components) != set(_COMPONENT_RELATIVE_PATHS)
    ):
        raise ActivationPreflightError("activation component map differs")
    for key, label in (
        ("activation_core", "core"),
        ("native_runner", "runner"),
        ("vae_reference_materializer", "vae_tool"),
        ("prompt_materializer", "prompt_tool"),
    ):
        _binding(
            components.get(key),
            relative_path=_COMPONENT_RELATIVE_PATHS[key],
            sha256=str(expected_hashes[label]),
            label=key,
        )
    runtime_dependencies = spec.get("frozen_runtime_dependencies")
    if (
        not isinstance(runtime_dependencies, Mapping)
        or set(runtime_dependencies) != set(_RUNTIME_DEPENDENCY_RELATIVE_PATHS)
    ):
        raise ActivationPreflightError("runtime dependency allowlist differs")
    for declared_name, relative_path in _RUNTIME_DEPENDENCY_RELATIVE_PATHS.items():
        expected_digest = runtime_dependencies.get(declared_name)
        if not isinstance(expected_digest, str) or _SHA256.fullmatch(expected_digest) is None:
            raise ActivationPreflightError(
                f"runtime dependency {declared_name} digest differs"
            )
        dependency_path = (METHOD_ROOT / relative_path).resolve(strict=True)
        dependency_raw, dependency_identity = _owned_bytes(
            dependency_path, label=f"runtime dependency {declared_name}"
        )
        del dependency_raw
        if dependency_identity["sha256"] != expected_digest:
            raise ActivationPreflightError(
                f"runtime dependency {declared_name} bytes differ"
            )
        observed[f"dependency:{declared_name}"] = dependency_identity
    runner_abi = _validate_runner_ast(raw["runner"])

    # The core has already passed an exact byte check, so it is safe to import.
    while str(METHOD_ROOT) in sys.path:
        sys.path.remove(str(METHOD_ROOT))
    sys.path.insert(0, str(METHOD_ROOT))
    import oracle_regeneration_activation_v2 as activation

    if (
        activation.COMPILED_AUTHORITY_PACKET_SHA256
        != COMPILED_AUTHORITY_PACKET_SHA256
        or activation.COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256
        != COMPILED_EXTERNAL_LEDGER_SHA256
    ):
        raise ActivationPreflightError("preflight/core trust anchors differ")
    authority = activation.load_compiled_activation_authority_v2(
        packet_path.resolve(strict=True), ledger_path.resolve(strict=True)
    )
    if authority.packet_sha256 != COMPILED_AUTHORITY_PACKET_SHA256 or authority.ledger_sha256 != COMPILED_EXTERNAL_LEDGER_SHA256:
        raise ActivationPreflightError("loaded authority roots differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": True,
        "cpu_only": True,
        "torch_imported": "torch" in sys.modules,
        "distributed_initialized": False,
        "model_loaded": False,
        "spec_sha256": observed["spec"]["sha256"],
        "component_identities": observed,
        "authority_packet_sha256": authority.packet_sha256,
        "external_ledger_sha256": authority.ledger_sha256,
        "packet_id": authority.packet_id,
        "cases": list(authority.cases),
        "runner_abi": runner_abi,
        "training": False,
        "optimizer": False,
        "flowedit": False,
        "connected_route": False,
        "automatic_replacement": False,
        "selection_authority": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-packet", required=True)
    parser.add_argument("--external-ledger", required=True)
    parser.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_release(
            packet_path=Path(args.authority_packet),
            ledger_path=Path(args.external_ledger),
        )
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "ready": False,
            "reason": str(error),
            "cpu_only": True,
            "torch_imported": "torch" in sys.modules,
            "distributed_initialized": False,
            "model_loaded": False,
            "training": False,
            "optimizer": False,
            "flowedit": False,
            "connected_route": False,
            "automatic_replacement": False,
            "selection_authority": None,
        }
        print(_canonical(result).decode("utf-8"))
        return 3
    print(_canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
