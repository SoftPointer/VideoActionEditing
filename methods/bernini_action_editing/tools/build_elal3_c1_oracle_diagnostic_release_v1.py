#!/usr/bin/env python3
"""Build the narrow, deterministic ELAL-3 C1 oracle-q diagnostic release.

The release is intentionally fail-closed.  It packages only the exact runtime
sources and three independently issued evidence objects needed by the one-row
simulator optimizer diagnostic.  The large latent bundle remains external but
is bound by its exact SHA-256 and byte count.  Nothing in this release grants
formal-C1, exact160, source+instruction inference, real-video, production, or
scientific-claim authority.

The trainer source is independently reviewed and frozen by exact SHA-256.  A
source mismatch therefore prevents release publication before any archive is
written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-elal3-c1-oracle-diagnostic-release-v1"
ARCHIVE_FORMAT = "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1"
ROW_ID = "c1-two-entity-push-to-goal"
TRAINER_SHA256 = (
    "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3"
)

RUNTIME_PINS: Mapping[str, str] = {
    "methods/bernini_action_editing/elal3_c0_v1.py": (
        "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862"
    ),
    "methods/bernini_action_editing/elal3_simulator_label_v1.py": (
        "4fecea53a55376545614edfca8603184f5e6f91dc86baccf6fb980f8b8124aa2"
    ),
    "methods/bernini_action_editing/inference_sigma_strata.py": (
        "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3"
    ),
    "methods/bernini_action_editing/packed_preservation_lora_v2.py": (
        "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6"
    ),
    "methods/bernini_action_editing/source_self_runtime.py": (
        "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f"
    ),
    "methods/bernini_action_editing/train_elal3_c1_simulator_overfit_v1.py": (
        TRAINER_SHA256
    ),
    "methods/bernini_action_editing/train_lora.py": (
        "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5"
    ),
}

DERIVATIVE_AUTHORITY_RELATIVE = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c1_simulator_optimizer_diagnostic_authority_v1.json"
)
MODEL_AUTHORITY_RELATIVE = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c1_real_model_authority_v1.json"
)
LATENT_RECEIPT_MEMBER = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c1_latent_bundle_receipt_authorized_v1.json"
)

DERIVATIVE_AUTHORITY_SHA256 = (
    "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b"
)
DERIVATIVE_AUTHORITY_DIGEST = (
    "c1706ee5b3f8a3fa4c037dfa6dbdbc7d0b088d3682128e50e712e311dae35043"
)
MODEL_AUTHORITY_SHA256 = (
    "4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed"
)
MODEL_AUTHORITY_DIGEST = (
    "25255902f4c5ce6de94ce6c3666bcf85eae4bf8e360a217f327c6febd049d21b"
)
LATENT_RECEIPT_SHA256 = (
    "a400d11d0d1337daa61d74a25e040aab27b83cc75e62038b81b83f56075e4fcb"
)
LATENT_RECEIPT_DIGEST = (
    "81f0ab734249651b00571e94a616de5a04fb13aa53fd711e45554b5a76251d61"
)
LATENT_BUNDLE_SHA256 = (
    "8fbd27abf7b6eea0593b236a0594dcfad38b3bedf46cf42e77391ec5648fdedf"
)
LATENT_BUNDLE_SIZE = 39_138_208
PACKET_MANIFEST_SHA256 = (
    "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
)

RUN_ASSIGNMENTS = (
    {
        "holder_job_id": "141620",
        "node": "auh7-1b-gpu-226",
        "seed": 20260817,
        "arm": "main-full-w64",
    },
    {
        "holder_job_id": "141618",
        "node": "auh7-1b-gpu-249",
        "seed": 20260818,
        "arm": "replicate-full-w64-seed2",
    },
    {
        "holder_job_id": "141619",
        "node": "auh7-1b-gpu-257",
        "seed": 20260819,
        "arm": "replicate-full-w64-seed3",
    },
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ELAL3C1ReleaseError(RuntimeError):
    """The source/evidence closure is not the preregistered narrow release."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ELAL3C1ReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3C1ReleaseError("value is not finite canonical ASCII JSON") from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_file(path: Path, *, maximum_bytes: int) -> bytes:
    _require(path.is_absolute() and not path.is_symlink(), f"non-canonical source: {path}")
    before_name = path.lstat()
    _require(stat.S_ISREG(before_name.st_mode), f"not a regular file: {path}")
    _require(before_name.st_nlink == 1, f"file has multiple links: {path}")
    _require(before_name.st_size <= maximum_bytes, f"file is too large: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        first = bytearray()
        while len(first) < before.st_size:
            block = os.read(descriptor, min(1 << 20, before.st_size - len(first)))
            _require(bool(block), f"file was truncated: {path}")
            first.extend(block)
        _require(os.read(descriptor, 1) == b"", f"file grew while reading: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = bytearray()
        while len(second) < before.st_size:
            block = os.read(descriptor, min(1 << 20, before.st_size - len(second)))
            _require(bool(block), f"file replay was truncated: {path}")
            second.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_name = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_rdev,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    _require(
        identity(before_name) == identity(before) == identity(after) == identity(after_name),
        f"file identity changed while reading: {path}",
    )
    _require(first == second, f"file bytes changed while reading: {path}")
    return bytes(first)


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3C1ReleaseError(f"{label} is not UTF-8 JSON") from error
    _require(type(value) is dict, f"{label} must contain one JSON object")
    return value


def _validate_evidence(
    derivative_raw: bytes,
    model_raw: bytes,
    latent_raw: bytes,
) -> dict[str, Any]:
    _require(
        hashlib.sha256(derivative_raw).hexdigest() == DERIVATIVE_AUTHORITY_SHA256,
        "derivative authority SHA-256 differs",
    )
    _require(
        hashlib.sha256(model_raw).hexdigest() == MODEL_AUTHORITY_SHA256,
        "model authority SHA-256 differs",
    )
    _require(
        hashlib.sha256(latent_raw).hexdigest() == LATENT_RECEIPT_SHA256,
        "latent receipt SHA-256 differs",
    )
    derivative = _strict_json(derivative_raw, label="derivative authority")
    model = _strict_json(model_raw, label="model authority")
    latent = _strict_json(latent_raw, label="latent receipt")
    for value, key, expected, label in (
        (derivative, "authority_digest", DERIVATIVE_AUTHORITY_DIGEST, "derivative authority"),
        (model, "authority_digest", MODEL_AUTHORITY_DIGEST, "model authority"),
        (latent, "receipt_digest", LATENT_RECEIPT_DIGEST, "latent receipt"),
    ):
        unsigned = dict(value)
        stored = unsigned.pop(key, None)
        _require(stored == expected and object_digest(unsigned) == expected, f"{label} self-digest differs")
    _require(
        derivative.get("schema_version")
        == "bernini-elal3-simulator-optimizer-derivative-authority-v1"
        and derivative.get("status") == "AUTHORIZED_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
        and derivative.get("authorized_row_id") == ROW_ID
        and derivative.get("max_optimizer_updates_per_arm") == 20
        and derivative.get("oracle_q_teacher_forced_required") is True
        and derivative.get("fresh_optimizer_run_required") is True
        and derivative.get("packet_manifest_sha256") == PACKET_MANIFEST_SHA256,
        "derivative authority scope differs",
    )
    _require(
        derivative.get("allowed_nodes")
        == [
            {"holder_job_id": row["holder_job_id"], "node": row["node"]}
            for row in RUN_ASSIGNMENTS
        ],
        "derivative authority node closure differs",
    )
    _require(
        derivative.get("disallowed_claims")
        == {
            "exact160": True,
            "formal_c1": True,
            "production_model": True,
            "real_video_generalization": True,
            "scientific_promotion": True,
            "source_instruction_inference": True,
        },
        "derivative authority claim boundary differs",
    )
    _require(
        derivative.get("training_objective_restrictions")
        == {
            "frozen_base_velocity_reference_forbidden": True,
            "frozen_teacher_self_distillation_forbidden": True,
            "hand_tuned_reward_scalar_forbidden": True,
            "target_grounded_event_and_context_flow_only": True,
        },
        "derivative authority objective boundary differs",
    )
    constraints = model.get("constraints")
    _require(
        model.get("schema_version") == "bernini-elal3-c1-real-model-authority-v1"
        and model.get("row_id") == ROW_ID
        and model.get("model_family") == "Bernini-R-1.3B-Diffusers"
        and model.get("file_count") == 9
        and type(model.get("files")) is list
        and len(model["files"]) == 9
        and type(constraints) is dict
        and constraints.get("allowed_operation")
        == "elal3_c1_simulator_oracle_q_optimizer_diagnostic"
        and constraints.get("max_optimizer_updates_per_arm") == 20
        and all(
            constraints.get(key) is False
            for key in (
                "exact160_authorized",
                "formal_c1_authorized",
                "real_video_claim_authorized",
                "scientific_claim_authorized",
                "source_instruction_inference_claim_authorized",
            )
        ),
        "model authority scope differs",
    )
    bundle = latent.get("bundle")
    derivative_binding = latent.get("derivative_optimizer_authority")
    model_binding = latent.get("real_model_authority")
    _require(
        latent.get("schema_version")
        == "bernini-elal3-simulator-c1-latent-bundle-receipt-v1"
        and latent.get("row_id") == ROW_ID
        and latent.get("bundle_format") == "safetensors-exact8-fp32-v1"
        and latent.get("latent_shape") == [1, 16, 21, 52, 70]
        and latent.get("vae_encode_count") == 8
        and latent.get("simulator_optimizer_diagnostic_authorized") is True
        and latent.get("oracle_q_required_for_training") is True
        and latent.get("source_instruction_inference") is False
        and latent.get("formal_c1_authorized") is False
        and latent.get("exact160_authorized") is False
        and latent.get("scientific_claim_authorized") is False
        and latent.get("real_video_data") is False
        and type(bundle) is dict
        and bundle.get("sha256") == LATENT_BUNDLE_SHA256
        and bundle.get("size") == LATENT_BUNDLE_SIZE
        and bundle.get("mode") == 0o444
        and bundle.get("nlink") == 1
        and type(derivative_binding) is dict
        and derivative_binding.get("authority_digest") == DERIVATIVE_AUTHORITY_DIGEST
        and derivative_binding.get("file", {}).get("sha256") == DERIVATIVE_AUTHORITY_SHA256,
        "latent receipt authority/bundle binding differs",
    )
    _require(
        type(model_binding) is dict
        and model_binding.get("schema_version")
        == "bernini-elal3-c1-real-model-authority-v1"
        and model_binding.get("authority_digest") == MODEL_AUTHORITY_DIGEST
        and model_binding.get("file", {}).get("sha256") == MODEL_AUTHORITY_SHA256
        and model_binding.get("verified_before_and_after_encoding") is True
        and type(model_binding.get("verified_file_bindings")) is list
        and len(model_binding["verified_file_bindings"]) == 9,
        "latent receipt real-model authority binding differs",
    )
    roots = {
        "bernini": model.get("bernini_root"),
        "checkpoint": model.get("checkpoint_root"),
        "python_env": model.get("python_env_root"),
    }
    expected_model_bindings = []
    for row in model.get("files", []):
        _require(
            type(row) is dict
            and row.get("root") in roots
            and type(row.get("relative_path")) is str,
            "model authority file row differs",
        )
        expected_model_bindings.append(
            {
                "path": str(Path(str(roots[row["root"]])) / row["relative_path"]),
                "sha256": row.get("sha256"),
                "size": row.get("size"),
                "mode": row.get("mode"),
                "nlink": 1,
            }
        )
    for actual, expected in zip(
        model_binding["verified_file_bindings"], expected_model_bindings
    ):
        _require(
            type(actual) is dict
            and all(actual.get(key) == value for key, value in expected.items()),
            "latent receipt verified model file binding differs",
        )
    return {"derivative": derivative, "model": model, "latent": latent}


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def build_payload(
    repo_root: Path,
    latent_receipt_path: Path,
    train_lora_source_path: Path,
    *,
    runtime_pins: Mapping[str, str] = RUNTIME_PINS,
) -> tuple[bytes, dict[str, Any]]:
    root = repo_root.resolve(strict=True)
    _require(root.is_dir() and not repo_root.is_symlink(), "repo root must be canonical")
    expected_runtime = set(RUNTIME_PINS)
    _require(set(runtime_pins) == expected_runtime, "runtime source closure differs")
    trainer_pin = runtime_pins[
        "methods/bernini_action_editing/train_elal3_c1_simulator_overfit_v1.py"
    ]
    _require(
        _SHA256.fullmatch(trainer_pin) is not None,
        "trainer SHA-256 pin is invalid; release publication is forbidden",
    )
    payloads: dict[str, bytes] = {}
    for relative in sorted(runtime_pins, key=lambda value: value.encode("ascii")):
        _require(_SHA256.fullmatch(runtime_pins[relative]) is not None, f"invalid runtime pin: {relative}")
        if relative == "methods/bernini_action_editing/train_lora.py":
            _require(
                train_lora_source_path.is_absolute(),
                "train_lora source path must be absolute",
            )
            path = train_lora_source_path
        else:
            path = root.joinpath(*PurePosixPath(relative).parts)
        raw = _stable_plain_file(path, maximum_bytes=8 << 20)
        _require(hashlib.sha256(raw).hexdigest() == runtime_pins[relative], f"runtime SHA-256 differs: {relative}")
        compile(raw, relative, "exec")
        payloads[relative] = raw
    derivative_raw = _stable_plain_file(
        root / DERIVATIVE_AUTHORITY_RELATIVE, maximum_bytes=1 << 20
    )
    model_raw = _stable_plain_file(
        root / MODEL_AUTHORITY_RELATIVE, maximum_bytes=1 << 20
    )
    _require(latent_receipt_path.is_absolute(), "latent receipt path must be absolute")
    latent_raw = _stable_plain_file(
        latent_receipt_path, maximum_bytes=1 << 20
    )
    evidence = _validate_evidence(derivative_raw, model_raw, latent_raw)
    payloads[DERIVATIVE_AUTHORITY_RELATIVE] = derivative_raw
    payloads[MODEL_AUTHORITY_RELATIVE] = model_raw
    payloads[LATENT_RECEIPT_MEMBER] = latent_raw
    names = sorted(payloads, key=lambda value: value.encode("ascii"))
    rows = [
        {
            "path": name,
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "size": len(payloads[name]),
            "mode": "0444",
        }
        for name in names
    ]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in names:
            archive.addfile(_tar_info(name, len(payloads[name])), io.BytesIO(payloads[name]))
    archive_raw = stream.getvalue()
    _require(len(archive_raw) % 10240 == 0, "archive record size differs")
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archive_size": len(archive_raw),
        "files": rows,
        "runtime_pins": dict(runtime_pins),
        "row_id": ROW_ID,
        "representation_variant": "full",
        "attention_width": 64,
        "lora_rank": 256,
        "optimizer_update_sequence": [0, 1, 10],
        "maximum_authorized_optimizer_updates": 20,
        "distributed_topology": {
            "world_size": 8,
            "data_parallel_size": 2,
            "sequence_parallel_size": 4,
            "one_node_per_run": True,
        },
        "run_assignments": list(RUN_ASSIGNMENTS),
        "authority_bindings": {
            "derivative_authority_sha256": DERIVATIVE_AUTHORITY_SHA256,
            "derivative_authority_digest": evidence["derivative"]["authority_digest"],
            "model_authority_sha256": MODEL_AUTHORITY_SHA256,
            "model_authority_digest": evidence["model"]["authority_digest"],
            "latent_receipt_sha256": LATENT_RECEIPT_SHA256,
            "latent_receipt_digest": evidence["latent"]["receipt_digest"],
            "packet_manifest_sha256": PACKET_MANIFEST_SHA256,
        },
        "external_latent_bundle": {
            "sha256": LATENT_BUNDLE_SHA256,
            "size": LATENT_BUNDLE_SIZE,
            "mode": "0444",
            "nlink": 1,
        },
        "execution_scope": "simulator_oracle_q_exact_one_row_optimizer_diagnostic_only",
        "simulator_optimizer_diagnostic_authorized": True,
        "teacher_forced_oracle_q_required": True,
        "formal_c1_authorized": False,
        "exact160_authorized": False,
        "source_instruction_inference_authorized": False,
        "real_video_generalization_authorized": False,
        "production_model_authorized": False,
        "scientific_claim_authorized": False,
    }
    manifest = {**unsigned, "manifest_digest": object_digest(unsigned)}
    return archive_raw, manifest


def _write_create_only(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            _require(written > 0, f"short write made no progress: {path}")
            remaining = remaining[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _require(
        _stable_plain_file(path, maximum_bytes=max(len(payload), 1)) == payload,
        f"published bytes differ: {path}",
    )


def publish(
    repo_root: Path,
    latent_receipt_path: Path,
    train_lora_source_path: Path,
    output: Path,
    *,
    runtime_pins: Mapping[str, str] = RUNTIME_PINS,
) -> dict[str, Any]:
    _require(output.is_absolute() and not output.exists() and not output.is_symlink(), "output must be a fresh absolute path")
    archive_raw, manifest = build_payload(
        repo_root,
        latent_receipt_path,
        train_lora_source_path,
        runtime_pins=runtime_pins,
    )
    os.mkdir(output, 0o700)
    _write_create_only(output / "source.tar", archive_raw, 0o444)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(output / "source.manifest.json", manifest_raw, 0o444)
    os.chmod(output, 0o555)
    return {
        "output": str(output),
        "archive_sha256": manifest["archive_sha256"],
        "archive_size": manifest["archive_size"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "scope": manifest["execution_scope"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--latent-receipt", type=Path, required=True)
    parser.add_argument("--train-lora-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = publish(
            args.repo_root,
            args.latent_receipt,
            args.train_lora_source,
            args.output,
        )
    except (ELAL3C1ReleaseError, OSError, SyntaxError) as error:
        print(f"[elal3-c1-release] ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
