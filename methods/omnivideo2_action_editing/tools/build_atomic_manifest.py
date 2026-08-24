#!/usr/bin/env python3
"""Build a create-only PACT atomic manifest from accepted global pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pact.manifest import (  # noqa: E402
    AtomizeOptions,
    ManifestError,
    atomize_global_row,
    canonical_json_bytes,
    load_jsonl,
    sign_post_generation_release,
    verify_post_generation_release,
)


def _add_manifest_field_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--row-schema-version", required=True)
    parser.add_argument("--source-path-key", default="source_video_path")
    parser.add_argument("--target-path-key", default="target_video_path")
    parser.add_argument("--source-sha256-key", default="source_video_sha256")
    parser.add_argument("--target-sha256-key", default="target_video_sha256")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    sign = commands.add_parser(
        "sign-release",
        help="create an immutable SSHSIG receipt for a final global manifest",
    )
    sign.add_argument("--global-manifest", required=True)
    sign.add_argument("--release-receipt", required=True)
    sign.add_argument("--signing-key", required=True)
    sign.add_argument("--signer-public-key", required=True)
    sign.add_argument("--expected-signer-fingerprint", required=True)
    sign.add_argument("--release-id", required=True)
    sign.add_argument("--issued-at-utc", required=True)
    _add_manifest_field_args(sign)

    build = commands.add_parser(
        "build", help="verify the signed final release and publish atomic rows"
    )
    build.add_argument("--global-manifest", required=True)
    build.add_argument("--release-receipt", required=True)
    build.add_argument("--signer-public-key", required=True)
    build.add_argument("--expected-signer-fingerprint", required=True)
    build.add_argument("--track-manifest", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--minimum-track-confidence", type=float, default=0.75)
    build.add_argument("--allow-rejections", action="store_true")
    _add_manifest_field_args(build)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _sign_release(args: argparse.Namespace) -> int:
    envelope = sign_post_generation_release(
        global_manifest_path=args.global_manifest,
        output_path=args.release_receipt,
        signing_key_path=args.signing_key,
        public_key_path=args.signer_public_key,
        expected_signer_fingerprint=args.expected_signer_fingerprint,
        release_id=args.release_id,
        issued_at_utc=args.issued_at_utc,
        row_schema_version=args.row_schema_version,
        source_path_key=args.source_path_key,
        target_path_key=args.target_path_key,
        source_sha256_key=args.source_sha256_key,
        target_sha256_key=args.target_sha256_key,
    )
    print(
        json.dumps(
            {
                "schema_version": envelope["schema_version"],
                "release_id": envelope["signed"]["release_id"],
                "global_manifest_sha256": envelope["signed"]["global_manifest"][
                    "sha256"
                ],
                "rows": envelope["signed"]["global_manifest"]["rows"],
                "signer_fingerprint": envelope["signature"]["key_fingerprint"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _build_atomic_manifest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ManifestError(f"output directory already exists: {output_dir}")
    if not 0 <= args.minimum_track_confidence <= 1:
        raise ManifestError("--minimum-track-confidence must be in [0, 1]")

    verified_release = verify_post_generation_release(
        global_manifest_path=args.global_manifest,
        release_receipt_path=args.release_receipt,
        public_key_path=args.signer_public_key,
        expected_signer_fingerprint=args.expected_signer_fingerprint,
        row_schema_version=args.row_schema_version,
        source_path_key=args.source_path_key,
        target_path_key=args.target_path_key,
        source_sha256_key=args.source_sha256_key,
        target_sha256_key=args.target_sha256_key,
    )
    parents = load_jsonl(args.global_manifest)
    tracks = load_jsonl(args.track_manifest)
    parent_iids = [parent.get("iid") for parent in parents]
    if any(not isinstance(iid, str) or not iid for iid in parent_iids):
        raise ManifestError("every parent row must have a non-empty iid")
    if len(set(parent_iids)) != len(parent_iids):
        raise ManifestError("global manifest contains duplicate parent IIDs")
    if parent_iids != [iid for iid, _row_sha256 in verified_release.authorized_rows]:
        raise ManifestError(
            "global manifest rows/order changed after signed-release verification"
        )
    parent_iid_set = set(parent_iids)
    tracks_by_iid: dict[str, list[dict]] = defaultdict(list)
    for track in tracks:
        iid = track.get("iid")
        if not isinstance(iid, str) or not iid:
            raise ManifestError("every track row must have a non-empty iid")
        if iid not in parent_iid_set:
            raise ManifestError(f"orphan track row references unknown parent: {iid}")
        tracks_by_iid[iid].append(track)

    options = AtomizeOptions(
        source_path_key=args.source_path_key,
        target_path_key=args.target_path_key,
        source_sha256_key=args.source_sha256_key,
        target_sha256_key=args.target_sha256_key,
        allow_preview=False,
        verify_mask_files=True,
        verify_media_files=True,
        minimum_track_confidence=args.minimum_track_confidence,
        verified_release=verified_release,
    )
    atoms = []
    rejected = Counter()
    for parent in parents:
        iid = parent.get("iid", "<missing>")
        try:
            atoms.extend(atomize_global_row(parent, tracks_by_iid.get(iid, []), options=options))
        except ManifestError as exc:
            rejected[str(exc)] += 1

    if not atoms:
        details = dict(rejected.most_common(10))
        raise ManifestError(f"no atomic rows were produced; rejections={details}")
    if rejected and not args.allow_rejections:
        details = dict(rejected.most_common(10))
        raise ManifestError(
            "atomic publication rejected one or more parents; rerun only after "
            f"repair, or explicitly use --allow-rejections; rejections={details}"
        )
    atoms.sort(key=lambda row: row["atom_id"])
    if len({row["atom_id"] for row in atoms}) != len(atoms):
        raise ManifestError("duplicate atom_id after publication merge")
    if _sha256(Path(args.global_manifest)) != verified_release.global_manifest_sha256:
        raise ManifestError(
            "global manifest bytes changed after signed-release verification"
        )

    output_dir.mkdir(parents=False)
    manifest_path = output_dir / "atomic_manifest.jsonl"
    with manifest_path.open("wb") as handle:
        for row in atoms:
            handle.write(canonical_json_bytes(row) + b"\n")
    summary = {
        "schema_version": "pact-atomic-manifest-summary-v1",
        "parent_rows": len(parents),
        "track_rows": len(tracks),
        "atomic_rows": len(atoms),
        "training_authorized_rows": sum(row["training_authorized"] for row in atoms),
        "preview_rows": sum(row["parent_preview_only"] for row in atoms),
        "rejected_parent_rows": sum(rejected.values()),
        "rejection_reasons": dict(rejected.most_common()),
        "complete": not bool(rejected),
        "global_manifest_sha256": _sha256(Path(args.global_manifest)),
        "post_generation_release_receipt_sha256": (
            verified_release.receipt_sha256
        ),
        "post_generation_release_payload_sha256": (
            verified_release.payload_sha256
        ),
        "post_generation_release_id": verified_release.release_id,
        "post_generation_release_signer_fingerprint": (
            verified_release.signer_fingerprint
        ),
        "track_manifest_sha256": _sha256(Path(args.track_manifest)),
        "atomic_manifest_sha256": _sha256(manifest_path),
        "options": vars(args),
    }
    _write_json(output_dir / "summary.json", summary)
    done = {
        "schema_version": "pact-atomic-manifest-done-v1",
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "atomic_manifest_sha256": summary["atomic_manifest_sha256"],
        "post_generation_release_receipt_sha256": (
            verified_release.receipt_sha256
        ),
        "complete": summary["complete"],
    }
    _write_json(output_dir / "done.json", done)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "sign-release":
        return _sign_release(args)
    if args.command == "build":
        return _build_atomic_manifest(args)
    raise ManifestError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
