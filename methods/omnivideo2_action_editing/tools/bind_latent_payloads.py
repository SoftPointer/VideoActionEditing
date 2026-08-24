#!/usr/bin/env python3
"""Bind validated latent payloads to training-authorized PACT atoms.

The atomizer deliberately emits no tensor paths.  This create-only publication
step closes that gap only after proving that the payload directory is a complete
one-to-one realization of the atomic manifest.  Payload bytes are loaded once
with ``weights_only=True``; the SHA-256 recorded in the training manifest is
computed from those exact bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pact.dataset import (  # noqa: E402
    PAYLOAD_FORMAT,
    PAYLOAD_PROVENANCE_BINDINGS,
    DatasetContractError,
    _safe_torch_load_bytes,
    encoder_contract_sha256,
    validate_precomputed_payload,
)
from pact.manifest import (  # noqa: E402
    ManifestError,
    canonical_json_bytes,
    file_sha256,
    load_jsonl,
    validate_atomic_row,
)


class PayloadBindingError(ValueError):
    """Raised when an atomic manifest and payload set cannot be bound safely."""


ATOMIC_INPUT_FILE_BINDINGS = (
    ("source_video_path", "source_video_sha256"),
    (
        "global_counterfactual_target_video_path",
        "global_counterfactual_target_video_sha256",
    ),
    ("source_component_mask_path", "source_component_mask_sha256"),
    ("target_component_mask_path", "target_component_mask_sha256"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a digest-bound PACT training manifest."
    )
    parser.add_argument("--atomic-manifest", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _relative_path(path: Path, *, start: Path) -> str:
    return Path(os.path.relpath(path, start=start)).as_posix()


def _load_validated_payload_bytes(
    path: Path,
) -> tuple[Mapping[str, Any], str, int]:
    """Safely load and hash exactly the same immutable byte snapshot."""

    try:
        payload_bytes = path.read_bytes()
    except OSError as exc:
        raise PayloadBindingError(f"cannot read payload {path}: {exc}") from exc
    if not payload_bytes:
        raise PayloadBindingError(f"payload is empty: {path}")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    try:
        value = _safe_torch_load_bytes(payload_bytes, path=path)
        payload = validate_precomputed_payload(value)
    except (
        DatasetContractError,
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PayloadBindingError(f"invalid safe tensor payload {path}: {exc}") from exc
    # Detect an ordinary concurrent replacement before publishing the digest.
    if file_sha256(path) != digest:
        raise PayloadBindingError(f"payload changed while it was being validated: {path}")
    return payload, digest, len(payload_bytes)


def _validate_source_rows(atomic_manifest: Path) -> list[dict[str, Any]]:
    try:
        raw_rows = load_jsonl(atomic_manifest)
    except (ManifestError, OSError) as exc:
        raise PayloadBindingError(f"cannot load atomic manifest: {exc}") from exc
    if not raw_rows:
        raise PayloadBindingError("atomic manifest is empty")

    rows: list[dict[str, Any]] = []
    seen_atom_ids: set[str] = set()
    for index, raw in enumerate(raw_rows, start=1):
        try:
            row = validate_atomic_row(raw)
        except ManifestError as exc:
            raise PayloadBindingError(
                f"invalid atomic manifest row {index}: {exc}"
            ) from exc
        atom_id = row["atom_id"]
        if atom_id in seen_atom_ids:
            raise PayloadBindingError(f"duplicate atom_id: {atom_id}")
        seen_atom_ids.add(atom_id)
        if row["training_authorized"] is not True:
            raise PayloadBindingError(f"atom {atom_id} is not authorized for training")
        if any(
            field in row
            for field in (
                "latent_payload_format",
                "latent_payload_path",
                "latent_payload_sha256",
            )
        ):
            raise PayloadBindingError(
                f"atom {atom_id} is already payload-bound; refusing silent rebinding"
            )
        for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS:
            digest = row.get(atomic_field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise PayloadBindingError(
                    f"atom {atom_id} has invalid {atomic_field} provenance digest "
                    f"required by payload {payload_field}"
                )
        rows.append(row)
    return rows


def _file_fingerprint(path: Path) -> tuple[int, int, int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PayloadBindingError(f"atomic input file is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PayloadBindingError(
            f"atomic input must be a regular non-symlink file: {path}"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _verify_atomic_input_files(
    rows: list[dict[str, Any]], *, manifest_root: Path
) -> dict[Path, tuple[str, tuple[int, int, int, int]]]:
    """Re-hash every unique video/mask input declared by the atomic rows."""

    verified: dict[Path, tuple[str, tuple[int, int, int, int]]] = {}
    for row in rows:
        for path_field, digest_field in ATOMIC_INPUT_FILE_BINDINGS:
            raw_path = Path(row[path_field])
            if not raw_path.is_absolute():
                raw_path = manifest_root / raw_path
            # Normalize dot segments without following a final symlink before
            # the explicit lstat-style regular-file check below.
            input_path = Path(os.path.abspath(raw_path))
            if input_path.is_symlink():
                raise PayloadBindingError(
                    f"atomic input must be a regular non-symlink file: {input_path}"
                )
            expected_digest = row[digest_field]
            prior = verified.get(input_path)
            if prior is not None:
                if prior[0] != expected_digest:
                    raise PayloadBindingError(
                        f"atomic rows declare conflicting digests for {input_path}"
                    )
                continue

            before = _file_fingerprint(input_path)
            actual_digest = file_sha256(input_path)
            after = _file_fingerprint(input_path)
            if before != after or input_path.is_symlink():
                raise PayloadBindingError(
                    f"atomic input changed while it was being hashed: {input_path}"
                )
            if actual_digest != expected_digest:
                raise PayloadBindingError(
                    f"atomic input {digest_field} differs for atom {row['atom_id']}: "
                    f"{input_path}"
                )
            verified[input_path] = (expected_digest, after)
    return verified


def _recheck_atomic_input_files(
    verified: Mapping[Path, tuple[str, tuple[int, int, int, int]]]
) -> None:
    for path, (_, expected_fingerprint) in verified.items():
        if path.is_symlink() or _file_fingerprint(path) != expected_fingerprint:
            raise PayloadBindingError(
                f"atomic input changed before manifest publication: {path}"
            )


def bind_latent_payloads(
    atomic_manifest: os.PathLike[str] | str,
    payload_dir: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Validate and publish a create-only digest-bound training manifest."""

    atomic_path = Path(atomic_manifest).resolve()
    payload_root = Path(payload_dir).resolve()
    output_root = Path(output_dir).resolve()

    if output_root.exists():
        raise PayloadBindingError(f"output directory already exists: {output_root}")
    if not atomic_path.is_file():
        raise PayloadBindingError(f"atomic manifest does not exist: {atomic_path}")
    if not payload_root.is_dir():
        raise PayloadBindingError(f"payload directory does not exist: {payload_root}")
    if not output_root.parent.is_dir():
        raise PayloadBindingError(
            f"output parent directory does not exist: {output_root.parent}"
        )

    rows = _validate_source_rows(atomic_path)
    atomic_manifest_digest = file_sha256(atomic_path)
    rows.sort(key=lambda row: row["atom_id"])
    verified_atomic_inputs = _verify_atomic_input_files(
        rows, manifest_root=atomic_path.parent
    )
    expected_names = {f"{row['atom_id']}.pt" for row in rows}
    actual_paths = {
        path.relative_to(payload_root).as_posix(): path
        for path in payload_root.rglob("*.pt")
        if path.is_file()
    }
    actual_names = set(actual_paths)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise PayloadBindingError(
            f"payload set differs from atomic manifest; missing={missing}, "
            f"unexpected={unexpected}"
        )

    bound_rows: list[dict[str, Any]] = []
    payload_digests: dict[str, str] = {}
    payload_bytes_total = 0
    common_encoder_contract_sha256: str | None = None
    for row in rows:
        atom_id = row["atom_id"]
        payload_path = payload_root / f"{atom_id}.pt"
        if payload_path.is_symlink():
            raise PayloadBindingError(
                f"payload symlinks are forbidden for rooted binding: {payload_path}"
            )
        resolved_payload = payload_path.resolve(strict=True)
        if resolved_payload.parent != payload_root:
            raise PayloadBindingError(
                f"payload resolves outside the declared payload directory: {payload_path}"
            )
        payload, digest, payload_byte_count = _load_validated_payload_bytes(
            resolved_payload
        )
        if payload.get("atom_id") != atom_id:
            raise PayloadBindingError(
                f"payload atom_id {payload.get('atom_id')!r} does not match {atom_id!r}"
            )
        for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS:
            if payload.get(payload_field) != row.get(atomic_field):
                raise PayloadBindingError(
                    f"payload {payload_field} does not match atomic "
                    f"{atomic_field} for {atom_id}"
                )

        payload_encoder_digest = encoder_contract_sha256(
            payload["encoder_contract"]
        )
        if common_encoder_contract_sha256 is None:
            common_encoder_contract_sha256 = payload_encoder_digest
        elif payload_encoder_digest != common_encoder_contract_sha256:
            raise PayloadBindingError(
                "payload set mixes incompatible offline encoder contracts"
            )

        bound = dict(row)
        bound["latent_payload_format"] = PAYLOAD_FORMAT
        bound["latent_payload_path"] = resolved_payload.relative_to(
            payload_root
        ).as_posix()
        bound["latent_payload_sha256"] = digest
        bound_rows.append(bound)
        payload_digests[f"{atom_id}.pt"] = digest
        payload_bytes_total += payload_byte_count

    final_payload_names = {
        path.relative_to(payload_root).as_posix()
        for path in payload_root.rglob("*.pt")
        if path.is_file()
    }
    if final_payload_names != expected_names:
        raise PayloadBindingError("payload set changed while it was being validated")
    for filename, expected_digest in payload_digests.items():
        payload_path = payload_root / filename
        if payload_path.is_symlink() or file_sha256(payload_path) != expected_digest:
            raise PayloadBindingError(
                f"payload changed while the set was being validated: {payload_path}"
            )
    if file_sha256(atomic_path) != atomic_manifest_digest:
        raise PayloadBindingError("atomic manifest changed while it was being validated")
    _recheck_atomic_input_files(verified_atomic_inputs)

    output_root.mkdir(parents=False)
    training_manifest_path = output_root / "training_manifest.jsonl"
    with training_manifest_path.open("wb") as handle:
        for row in bound_rows:
            handle.write(canonical_json_bytes(row) + b"\n")

    summary = {
        "schema_version": "pact-latent-payload-binding-summary-v1",
        "atomic_manifest_path": _relative_path(atomic_path, start=output_root),
        "atomic_manifest_sha256": atomic_manifest_digest,
        "atomic_input_files_verified": len(verified_atomic_inputs),
        "atomic_input_bytes_verified": sum(
            fingerprint[2] for _, fingerprint in verified_atomic_inputs.values()
        ),
        "payload_root": str(payload_root),
        "payload_format": PAYLOAD_FORMAT,
        "encoder_contract_sha256": common_encoder_contract_sha256,
        "payload_provenance_bindings": {
            payload_field: atomic_field
            for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS
        },
        "payload_files": len(bound_rows),
        "payload_bytes": payload_bytes_total,
        "strict_one_to_one_payload_set": True,
        "training_authorized_rows": len(bound_rows),
        "training_manifest_sha256": file_sha256(training_manifest_path),
    }
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    done = {
        "schema_version": "pact-latent-payload-binding-done-v1",
        "summary_sha256": file_sha256(summary_path),
        "training_manifest_sha256": summary["training_manifest_sha256"],
        "complete": True,
    }
    _write_json(output_root / "done.json", done)
    return summary


def main() -> int:
    args = parse_args()
    summary = bind_latent_payloads(
        args.atomic_manifest, args.payload_dir, args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
