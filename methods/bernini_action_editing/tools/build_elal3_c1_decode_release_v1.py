#!/usr/bin/env python3
"""Build the deterministic, simulator-only ELAL-3 C1 decode release.

The archive contains exactly the independently reviewed WORLD4 decoder and the
23-row content manifest for every base-checkpoint file it may read.  Its
manifest binds the completed three-seed training diagnostics and all external
authority objects needed to render the exact-nine review packet.  It grants no
source+instruction, formal-C1, exact160, real-video, production, or scientific
claim authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-elal3-c1-simulator-oracle-q-decode-release-v3"
ARCHIVE_FORMAT = "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1"
SCOPE = "simulator_oracle_q_exact_one_row_checkpoint_decode_only"
ROW_ID = "c1-two-entity-push-to-goal"
DECODER_MEMBER = (
    "methods/bernini_action_editing/decode_elal3_c1_simulator_oracle_q_v1.py"
)
CHECKPOINT_MANIFEST_MEMBER = (
    "methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
PENDING_DECODER_SHA256 = "PENDING_DECODER_SHA256_FAIL_CLOSED"
DECODER_SHA256 = (
    "977fa4e6a91d432e57ecaa59dae87419c734d7038fbe92f47369020d65d41c52"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_MANIFEST_SIZE = 2350
CHECKPOINT_MANIFEST_ROW_COUNT = 23

TRAINING_RELEASE = {
    "manifest_sha256": "bb56f175f205b626f003c855260243a5c1a5fa3d8c7f0464ddea49931006a9f3",
    "manifest_digest": "48988cd555dbb6b01c1242772c5837a9168d19a5b824f7adb3c3aa3b088cd799",
    "archive_sha256": "631611a96a744025eb6e5b223958908c7dfccfb69bfaefa7432ea9c20afc8194",
    "trainer_source_sha256": "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3",
}
TRAINING_ARTIFACTS_BY_SEED = {
    "20260817": {
        "seed": 20260817,
        "holder_job_id": "141620",
        "holder_node": "auh7-1b-gpu-226",
        "training_receipt_sha256": "a7ca5e4ec2fd04ccd77bfd943bee48cb4978561787a62f5d21175d9846b3af71",
        "step0_adapter_sha256": "0369c6dd3dfa5b58e2eb67984955babe4ab637edef1d50a7eb60628b07be1f38",
        "trained_adapter_sha256": "c38ba270b0ff2736c06ec4733b1b9bf4858a7654adb0f020b55a98a406282ac9",
    },
    "20260818": {
        "seed": 20260818,
        "holder_job_id": "141618",
        "holder_node": "auh7-1b-gpu-249",
        "training_receipt_sha256": "37b51f4f0003e0e4418664906106dd2ad25b5a09b1be866df3eeac0e0f3362d8",
        "step0_adapter_sha256": "96158cb165f1f3c0d151c27f79bcb71439cbd42a024e04e30b94214428d33dbb",
        "trained_adapter_sha256": "1108680084e976904c6c33586af556327100cfd94bb0d3891212800a9b0dea69",
    },
    "20260819": {
        "seed": 20260819,
        "holder_job_id": "141619",
        "holder_node": "auh7-1b-gpu-257",
        "training_receipt_sha256": "a67aa4b7235ad130cdb20b4060865fd9014a0c10437828cc1d3bc0b8a6eccb7c",
        "step0_adapter_sha256": "6a0abffed80bcf3d5021a05dbb080a8c39e785076e1fb162d88a5ffad8ddb4cd",
        "trained_adapter_sha256": "888f14297cbac3523cd0eb1ccd53892118e739692a5a4319a5ce1d4dd35be4d9",
    },
}
REGISTERED_PLACEMENT_BY_SEED = {
    "20260817": ("141620", "auh7-1b-gpu-226"),
    "20260818": ("141618", "auh7-1b-gpu-249"),
    "20260819": ("141619", "auh7-1b-gpu-257"),
}
RUNTIME = {
    "world_size": 4,
    "ulysses_size": 4,
    "num_inference_steps": 40,
    "authorized_training_seeds": [20260817, 20260818, 20260819],
    "sampling_seed_equals_training_seed": True,
    "branch_order": [
        "source",
        "gt_target",
        "appearance_anchor",
        "frozen_base",
        "step0_correct_q",
        "trained_correct_q",
        "trained_zero_q",
        "trained_phase_reverse_q",
        "trained_role_swap_q",
    ],
}
AUTHORITY_BINDINGS = {
    "model_authority_sha256": "4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed",
    "derivative_authority_sha256": "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b",
    "packet_manifest_sha256": "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc",
    "latent_bundle_sha256": "8fbd27abf7b6eea0593b236a0594dcfad38b3bedf46cf42e77391ec5648fdedf",
    "latent_bundle_receipt_sha256": "a400d11d0d1337daa61d74a25e040aab27b83cc75e62038b81b83f56075e4fcb",
    "checkpoint_content_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
}
FORBIDDEN_CLAIMS = (
    "formal_c1_authorized",
    "exact160_authorized",
    "source_instruction_inference_authorized",
    "real_video_generalization_authorized",
    "production_model_authorized",
    "scientific_claim_authorized",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ELAL3C1DecodeReleaseError(RuntimeError):
    """The decode release does not match its exact reviewed closure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ELAL3C1DecodeReleaseError(message)


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
        raise ELAL3C1DecodeReleaseError("manifest is not canonicalizable") from error


def validate_release_constants() -> None:
    expected_seeds = list(REGISTERED_PLACEMENT_BY_SEED)
    require(
        list(TRAINING_ARTIFACTS_BY_SEED) == expected_seeds,
        "training seed registry differs",
    )
    require(
        RUNTIME.get("authorized_training_seeds")
        == [int(seed) for seed in expected_seeds]
        and RUNTIME.get("sampling_seed_equals_training_seed") is True,
        "runtime training seed registry differs",
    )
    exact_row_keys = {
        "seed",
        "holder_job_id",
        "holder_node",
        "training_receipt_sha256",
        "step0_adapter_sha256",
        "trained_adapter_sha256",
    }
    for seed, placement in REGISTERED_PLACEMENT_BY_SEED.items():
        row = TRAINING_ARTIFACTS_BY_SEED[seed]
        require(set(row) == exact_row_keys, f"training artifact row keys differ: {seed}")
        require(row.get("seed") == int(seed), f"training artifact seed differs: {seed}")
        require(
            (row.get("holder_job_id"), row.get("holder_node")) == placement,
            f"training artifact placement differs: {seed}",
        )
        for key in (
            "training_receipt_sha256",
            "step0_adapter_sha256",
            "trained_adapter_sha256",
        ):
            require(
                _SHA256.fullmatch(str(row.get(key))) is not None,
                f"training artifact digest differs: {seed}/{key}",
            )


def stable_plain_file(path: Path, *, maximum_bytes: int) -> bytes:
    require(path.is_absolute() and not path.is_symlink(), f"unsafe source path: {path}")
    before = path.stat()
    require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and 0 < before.st_size <= maximum_bytes,
        f"source is not an exact plain file: {path}",
    )
    raw = path.read_bytes()
    after = path.stat()
    require(
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(raw)
        and before.st_mtime_ns == after.st_mtime_ns,
        f"source changed while read: {path}",
    )
    return raw


def build_payload(
    decoder_source: Path,
    checkpoint_manifest_source: Path,
    *,
    decoder_sha256: str = DECODER_SHA256,
) -> tuple[bytes, dict[str, Any]]:
    validate_release_constants()
    require(
        _SHA256.fullmatch(decoder_sha256) is not None,
        "decoder SHA-256 is still PENDING; publication is forbidden",
    )
    raw = stable_plain_file(decoder_source, maximum_bytes=8 << 20)
    require(
        hashlib.sha256(raw).hexdigest() == decoder_sha256,
        "decoder source SHA-256 differs",
    )
    checkpoint_manifest_raw = stable_plain_file(
        checkpoint_manifest_source, maximum_bytes=CHECKPOINT_MANIFEST_SIZE
    )
    require(
        len(checkpoint_manifest_raw) == CHECKPOINT_MANIFEST_SIZE
        and hashlib.sha256(checkpoint_manifest_raw).hexdigest()
        == CHECKPOINT_MANIFEST_SHA256,
        "checkpoint content manifest bytes differ",
    )
    try:
        checkpoint_lines = checkpoint_manifest_raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ELAL3C1DecodeReleaseError(
            "checkpoint content manifest is not ASCII"
        ) from error
    require(
        len(checkpoint_lines) == CHECKPOINT_MANIFEST_ROW_COUNT
        and checkpoint_manifest_raw.endswith(b"\n"),
        "checkpoint content manifest row closure differs",
    )
    seen_checkpoint_paths: set[str] = set()
    for line in checkpoint_lines:
        match = re.fullmatch(r"([0-9a-f]{64})  \./([^\x00-\x1f]+)", line)
        require(match is not None, "checkpoint content manifest row differs")
        relative = match.group(2)
        parts = relative.split("/")
        require(
            relative not in seen_checkpoint_paths
            and all(part not in ("", ".", "..") for part in parts),
            "checkpoint content manifest path closure differs",
        )
        seen_checkpoint_paths.add(relative)
    compile(raw, DECODER_MEMBER, "exec")
    members = sorted(
        (
            (CHECKPOINT_MANIFEST_MEMBER, checkpoint_manifest_raw),
            (DECODER_MEMBER, raw),
        )
    )
    archive_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=archive_buffer,
        mode="w",
        format=tarfile.USTAR_FORMAT,
        encoding="ascii",
        errors="strict",
    ) as archive:
        for member, payload in members:
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(payload))
    archive_raw = archive_buffer.getvalue()
    require(len(archive_raw) % 10_240 == 0, "USTAR record size differs")
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "scope": SCOPE,
        "row_id": ROW_ID,
        "decoder_member": DECODER_MEMBER,
        "decoder_source_sha256": decoder_sha256,
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archive_size": len(archive_raw),
        "files": [
            {
                "path": member,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "archive_mode": 0o444,
            }
            for member, payload in members
        ],
        "checkpoint_content_manifest": {
            "member": CHECKPOINT_MANIFEST_MEMBER,
            "sha256": CHECKPOINT_MANIFEST_SHA256,
            "size": CHECKPOINT_MANIFEST_SIZE,
            "row_count": CHECKPOINT_MANIFEST_ROW_COUNT,
        },
        "training_release": dict(TRAINING_RELEASE),
        "training_artifacts_by_seed": {
            seed: dict(artifacts)
            for seed, artifacts in TRAINING_ARTIFACTS_BY_SEED.items()
        },
        "runtime": {
            **RUNTIME,
            "branch_order": list(RUNTIME["branch_order"]),
        },
        "authority_bindings": dict(AUTHORITY_BINDINGS),
        **{key: False for key in FORBIDDEN_CLAIMS},
    }
    manifest = {
        **unsigned,
        "manifest_digest": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    return archive_raw, manifest


def write_create_only(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, f"write made no progress: {path}")
            remaining = remaining[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(
        stable_plain_file(path, maximum_bytes=max(1, len(payload))) == payload,
        f"published bytes differ: {path}",
    )


def publish(
    decoder_source: Path,
    checkpoint_manifest_source: Path,
    output: Path,
    *,
    decoder_sha256: str = DECODER_SHA256,
) -> Mapping[str, Any]:
    require(
        output.is_absolute() and not output.exists() and not output.is_symlink(),
        "output must be a fresh absolute path",
    )
    archive_raw, manifest = build_payload(
        decoder_source,
        checkpoint_manifest_source,
        decoder_sha256=decoder_sha256,
    )
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    os.mkdir(output, 0o700)
    write_create_only(output / "decode-source.tar", archive_raw, 0o444)
    write_create_only(output / "decode-manifest.json", manifest_raw, 0o444)
    os.chmod(output, 0o555)
    return {
        "output": str(output),
        "archive_sha256": manifest["archive_sha256"],
        "archive_size": manifest["archive_size"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "decoder_source_sha256": decoder_sha256,
        "checkpoint_content_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "scope": SCOPE,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-source", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = publish(
            args.decoder_source,
            args.checkpoint_manifest_source,
            args.output,
        )
    except (ELAL3C1DecodeReleaseError, OSError, SyntaxError) as error:
        print(f"[elal3-c1-decode-release] ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
