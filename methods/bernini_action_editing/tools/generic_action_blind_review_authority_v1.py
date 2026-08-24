#!/usr/bin/env python3
"""Create and ingest a sealed, branch-blind full-first8 review packet.

This module never generates a semantic label.  ``build-packet`` authenticates
and fully decodes all 160 preregistered authoring clips, copies them under
keyed opaque names, and seals a private mapping before review.  ``ingest``
only accepts a separately authored, hash-bound external response.  It emits
legacy-compatible per-clip review receipts plus one stronger packet-bound
authority; no partial population can authorize Phi extraction.

Generated RGB and latents remain teacher/review evidence.  They are never an
editor source, target, condition, donor, pseudo-target, or optimizer input.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import hmac
import html
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import build_generic_action_phi_v1_authority_release_v1 as authority_release  # noqa: E402


PACKET_SCHEMA = "bernini-generic-action-blind-full81-packet-v1"
PRIVATE_MAP_SCHEMA = "bernini-generic-action-blind-full81-private-map-v1"
RESPONSE_SCHEMA = "bernini-generic-action-external-full81-response-v1"
REVIEWER_AUTHORITY_SCHEMA = "bernini-generic-action-external-reviewer-authority-v1"
EXECUTION_CREDENTIAL_SCHEMA = "bernini-generic-action-external-review-execution-credential-v1"
AUTHORITY_SCHEMA = "bernini-generic-action-external-review-authority-v1"
GAP_SCHEMA = "bernini-generic-action-external-review-gap-v1"
FULL_CLIP_COUNT = 160
CORE4_CLIP_COUNT = 80
RESERVE4_CLIP_COUNT = 80
SEED_CELL_COUNT = 16
BRANCHES = (
    "action", "noop", "incomplete", "reverse", "shuffle", "wrong_actor",
    "wrong_object", "camera_only", "appearance_only", "generic_wrong_motion",
)
PHASES = 21
CLASS_UNJUDGEABLE = "unjudgeable"
AXIS_FIELDS = (
    "action", "noop", "incomplete", "reverse", "shuffle", "wrong_actor",
    "wrong_object", "camera_only", "appearance_only", "generic_wrong_motion",
)
AXIS_DEFINITIONS = {
    "action": "The registered main subject completes the forward target event and holds its terminal state.",
    "noop": "The registered main subject holds q0; the forward and reverse target events remain absent.",
    "incomplete": "The registered main subject begins the forward target transition but does not reach or hold q1.",
    "reverse": "The registered main subject completes q1-to-q0 in reverse order and holds q0.",
    "shuffle": "Target-event fragments occur in the wrong temporal order without the registered terminal completion.",
    "wrong_actor": "A non-owner performs the target event while the registered main subject does not.",
    "wrong_object": "The main subject changes the wrong object/contact while the registered target event remains absent.",
    "camera_only": "Only viewpoint/camera geometry changes; subject pose/action and appearance stay at q0.",
    "appearance_only": "Only subject appearance changes; camera geometry and subject pose/action stay at q0.",
    "generic_wrong_motion": "A non-target body motion occurs without completing the forward or reverse target event.",
}
PROFILE_PINS = {
    "core4-v2": {
        "seed1": "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95",
        "seed2": "900c0dece65ee2f075765571b39d62e45ceb1b3c8b5c883443ea09d1876e18f3",
    },
    "reserve4-v1": {
        "seed1": "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab",
        "seed2": "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e",
    },
}
_SHA = re.compile(r"[0-9a-f]{64}")
_OPAQUE = re.compile(r"clip-[0-9a-f]{32}")
_EXECUTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}")
_SIGNATURE = re.compile(r"[0-9a-f]{128}")

# Base modules are intentionally absent at import time.  Every build, ingest,
# or audit entrypoint validates the exact installed overlay/base closure first.
manifests: Any = None
legacy_phi: Any = None


class BlindReviewAuthorityError(RuntimeError):
    """A media, seal, review, or population boundary failed closed."""


def fail(message: str) -> NoReturn:
    raise BlindReviewAuthorityError(message)


def _activate_installed_closure(
    release_manifest: str | Path,
    expected_release_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate the overlay/base bytes before importing either base module."""

    try:
        installed = authority_release.validate_installed_closure(
            METHOD_ROOT,
            Path(release_manifest),
            expected_release_manifest_sha256,
        )
    except authority_release.PhiAuthorityReleaseError as error:
        raise BlindReviewAuthorityError(str(error)) from error
    global manifests, legacy_phi
    manifests = importlib.import_module("generic_action_manifest_v1")
    legacy_phi = importlib.import_module("materialize_phi_v1_sidecars_sp4")
    _require(tuple(legacy_phi.ALL_BRANCHES) == BRANCHES, "installed branch order differs")
    _require(
        manifests.AUTHORING_SHA256 == "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
        and manifests.POPULATION_SHA256 == "71906510d162e6626338b5785fd1cf55b437de5ba77d9b9b122ad761694f8e62",
        "installed registry pins differ",
    )
    return installed


def _require_base_modules() -> tuple[Any, Any]:
    _require(manifests is not None and legacy_phi is not None, "installed release closure was not activated")
    return manifests, legacy_phi


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise BlindReviewAuthorityError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def _sha(value: Any, label: str) -> str:
    _require(type(value) is str and _SHA.fullmatch(value) is not None, f"{label} must be SHA-256")
    return value


def _closed(value: Any, fields: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(fields)
    _require(type(value) is dict and set(value) == expected, f"{label} field closure differs")
    return value


def _plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise BlindReviewAuthorityError(f"{label} unavailable: {path}") from error
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} must be a plain file")
    return path.resolve(strict=True)


def _plain_dir(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise BlindReviewAuthorityError(f"{label} unavailable: {path}") from error
    _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} must be a plain directory")
    return path.resolve(strict=True)


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path_value: str | Path, label: str, expected_sha256: Optional[str] = None, *, require_canonical: bool = True) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(path_value, label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require(observed == _sha(expected_sha256, f"expected {label}"), f"{label} SHA-256 differs")
    try:
        value = json.loads(
            raw, object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise BlindReviewAuthorityError(f"cannot decode {label}") from error
    _require(type(value) is dict, f"{label} root must be an object")
    if require_canonical:
        _require(raw == canonical_json_bytes(value) + b"\n", f"{label} is not canonical sealed JSON")
    return value, path, observed


def _verify_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    declared = _sha(value.get(field), f"{label}.{field}")
    unsigned = dict(value)
    del unsigned[field]
    _require(object_sha256(unsigned) == declared, f"{label} digest differs")


def _atomic_create(path: Path, raw: bytes, mode: int) -> str:
    _require(path.is_absolute() and path.parent.is_dir(), f"output parent unavailable: {path}")
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any], mode: int = 0o400) -> str:
    return _atomic_create(path, canonical_json_bytes(value) + b"\n", mode)


def _copy_plain(source: Path, target: Path, expected_sha256: str) -> None:
    _require(not target.exists() and not target.is_symlink(), f"refusing to overwrite {target}")
    temporary: Optional[Path] = None
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_handle:
                for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        _require(digest.hexdigest() == expected_sha256, "media changed while copying")
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reviewer_authority_template() -> Mapping[str, Any]:
    """Return a deliberately non-authorizing template; no key is fabricated."""

    return {
        "schema_version": REVIEWER_AUTHORITY_SCHEMA,
        "authority_id": None,
        "reviewer_id": None,
        "review_method": None,
        "reviewer_tool_source_sha256": None,
        "verification_key": {
            "algorithm": "ed25519",
            "public_key_raw_hex": None,
            "public_key_sha256": None,
        },
        "independent_of_generation_runner": None,
        "independent_of_packet_builder": None,
        "independent_of_phi_runner": None,
        "private_key_embedded": False,
        "response_key_override_allowed": False,
        "signed_execution_credential_required": True,
        "authority_digest": None,
    }


def _validate_reviewer_authority(
    path_value: str | Path,
    expected_sha256: str,
    reviewer_tool_source_artifact: str | Path,
) -> tuple[Mapping[str, Any], Path, str, Path]:
    base_manifests, _ = _require_base_modules()
    value, path, observed_sha = _load_json(
        path_value, "external reviewer authority", expected_sha256,
    )
    _closed(
        value,
        {
            "schema_version", "authority_id", "reviewer_id", "review_method",
            "reviewer_tool_source_sha256", "verification_key",
            "independent_of_generation_runner", "independent_of_packet_builder",
            "independent_of_phi_runner", "private_key_embedded",
            "response_key_override_allowed", "signed_execution_credential_required",
            "authority_digest",
        },
        "external reviewer authority",
    )
    _verify_seal(value, "authority_digest", "external reviewer authority")
    _require(value["schema_version"] == REVIEWER_AUTHORITY_SCHEMA, "reviewer authority schema differs")
    for field in ("authority_id", "reviewer_id"):
        _require(
            type(value[field]) is str and _EXECUTION.fullmatch(value[field]) is not None,
            f"reviewer authority {field} differs",
        )
    _require(value["review_method"] in base_manifests.ALLOWED_REVIEW_METHODS, "reviewer authority method differs")
    _sha(value["reviewer_tool_source_sha256"], "reviewer tool source")
    _require(
        all(
            value[field] is True
            for field in (
                "independent_of_generation_runner", "independent_of_packet_builder",
                "independent_of_phi_runner", "signed_execution_credential_required",
            )
        )
        and value["private_key_embedded"] is False
        and value["response_key_override_allowed"] is False,
        "reviewer authority independence/key boundary differs",
    )
    key = _closed(
        value["verification_key"],
        {"algorithm", "public_key_raw_hex", "public_key_sha256"},
        "reviewer verification key",
    )
    _require(key["algorithm"] == "ed25519", "reviewer signature algorithm differs")
    public_hex = key["public_key_raw_hex"]
    _require(type(public_hex) is str and _SHA.fullmatch(public_hex) is not None, "reviewer public key differs")
    try:
        public_raw = bytes.fromhex(public_hex)
    except ValueError as error:
        raise BlindReviewAuthorityError("reviewer public key differs") from error
    _require(
        len(public_raw) == 32 and key["public_key_sha256"] == hashlib.sha256(public_raw).hexdigest(),
        "reviewer public key binding differs",
    )
    tool_path = _plain_file(reviewer_tool_source_artifact, "reviewer tool source artifact")
    tool_raw = tool_path.read_bytes()
    _require(tool_raw and hashlib.sha256(tool_raw).hexdigest() == value["reviewer_tool_source_sha256"], "reviewer tool source artifact SHA-256 differs")
    _require(
        b"-----BEGIN PRIVATE KEY-----" not in tool_raw
        and b"-----BEGIN OPENSSH PRIVATE KEY-----" not in tool_raw,
        "reviewer tool artifact embeds a private key",
    )
    return value, path, observed_sha, tool_path


def _credential_unsigned(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: item for key, item in value.items() if key != "signature_hex"}


def _verify_execution_credential(
    credential: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    packet_manifest_file_sha256: str,
    response_digest: str,
    reviewer_authority: Mapping[str, Any],
) -> None:
    _closed(
        credential,
        {
            "schema_version", "signature_algorithm", "reviewer_authority_file_sha256",
            "reviewer_authority_digest", "packet_manifest_file_sha256", "packet_digest",
            "reviewer_tool_source_sha256", "review_execution_id", "response_digest",
            "signature_hex",
        },
        "review execution credential",
    )
    fixed = packet["reviewer_authority"]
    _require(
        credential["schema_version"] == EXECUTION_CREDENTIAL_SCHEMA
        and credential["signature_algorithm"] == "ed25519"
        and credential["reviewer_authority_file_sha256"] == fixed["file_sha256"]
        and credential["reviewer_authority_digest"] == fixed["authority_digest"]
        and credential["packet_manifest_file_sha256"] == packet_manifest_file_sha256
        and credential["packet_digest"] == packet["packet_digest"]
        and credential["reviewer_tool_source_sha256"] == fixed["reviewer_tool_source_sha256"]
        and credential["response_digest"] == response_digest,
        "signed review credential binding differs",
    )
    _require(
        _EXECUTION.fullmatch(credential["review_execution_id"]) is not None
        and credential["review_execution_id"] != packet["packet_builder_execution_id"],
        "review execution is not external",
    )
    signature_hex = credential["signature_hex"]
    _require(type(signature_hex) is str and _SIGNATURE.fullmatch(signature_hex) is not None, "review credential signature encoding differs")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_raw = bytes.fromhex(reviewer_authority["verification_key"]["public_key_raw_hex"])
        signature = bytes.fromhex(signature_hex)
        Ed25519PublicKey.from_public_bytes(public_raw).verify(
            signature,
            canonical_json_bytes(_credential_unsigned(credential)),
        )
    except (ImportError, ValueError, InvalidSignature, binascii.Error) as error:
        raise BlindReviewAuthorityError("review credential signature verification failed") from error


def _probe_full81(path: Path) -> Mapping[str, Any]:
    """Decode integer frames 0..80 via the release/runtime Python decoder."""

    try:
        from tools import materialize_vae
        frames, fps, hw = materialize_vae._decode_exact_video(path)
        shape = tuple(int(item) for item in frames.shape)
        dtype = str(frames.dtype)
    except Exception as error:
        raise BlindReviewAuthorityError(f"cannot fully decode exact81 media: {path}") from error
    _require(
        len(shape) == 4 and shape[0] == 81 and shape[-1] == 3
        and shape[1:3] == tuple(hw) and all(int(item) > 0 for item in hw)
        and dtype == "uint8" and math.isfinite(float(fps))
        and abs(float(fps) - 25.0) <= 1.0e-3,
        f"media is not fully decodable exact81/25fps RGB: {path}",
    )
    decoder_path = Path(materialize_vae.__file__).resolve(strict=True)
    return {
        "decoder": "tools.materialize_vae._decode_exact_video",
        "decoder_source_sha256": file_sha256(decoder_path),
        "all_integer_frames_0_through_80_decoded": True,
        "frame_count": 81, "fps": 25, "height": int(hw[0]),
        "width": int(hw[1]), "channels": 3, "dtype": "uint8",
    }


def _population_context(authoring: Mapping[str, Any], population: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    cells = {row["iid"]: row for row in authoring.get("cells", [])}
    profiles = {row["profile_id"]: row for row in population.get("inherited_bank_profiles", [])}
    inherited: dict[str, Mapping[str, Any]] = {}
    for family in population.get("action_families", []):
        for row in family.get("inherited_identity_scenes", []):
            inherited[row["source_iid"]] = row
    _require(len(cells) == len(inherited) == 8 and set(cells) == set(inherited), "first8 population closure differs")
    expected: list[dict[str, Any]] = []
    for split in ("fit", "confirmation"):
        for iid in sorted(key for key, row in cells.items() if row["analysis_split"] == split):
            cell = cells[iid]
            population_row = inherited[iid]
            profile_id = population_row["source_bank_profile"]
            profile = profiles[profile_id]
            _require(profile_id in PROFILE_PINS and len(population_row["seeds"]) == 2, "bank profile differs")
            for seed_index, seed in enumerate(population_row["seeds"]):
                seed_kind = "seed1" if seed_index == 0 else "seed2"
                _require(profile[f"{seed_kind}_root_spec_raw_sha256"] == PROFILE_PINS[profile_id][seed_kind], "population spec pin differs")
                prefix = profile[f"{seed_kind}_candidate_prefix"]
                for branch in BRANCHES:
                    expected.append({
                        "candidate_id": f"{prefix}{iid}-{branch}", "source_iid": iid,
                        "analysis_split": split, "seed": seed, "seed_kind": seed_kind,
                        "profile_id": profile_id, "branch": branch,
                        "root_spec_raw_sha256": PROFILE_PINS[profile_id][seed_kind],
                    })
    _require(len(expected) == FULL_CLIP_COUNT and len({row["candidate_id"] for row in expected}) == FULL_CLIP_COUNT, "full-first8 candidate closure differs")
    return expected, cells


def _scan_generation(roots: Sequence[str | Path]) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    _, base_legacy_phi = _require_base_modules()
    indexed: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for root_value in roots:
        root = _plain_dir(root_value, "generation root")
        for path in sorted(root.rglob("pair-v5-t2v-calibration-receipt.json")):
            receipt = base_legacy_phi._candidate_receipt(path.resolve(strict=True))
            candidate_id = receipt["candidate"]["candidate_id"]
            _require(candidate_id not in indexed, f"duplicate candidate receipt: {candidate_id}")
            indexed[candidate_id] = (path.resolve(strict=True), receipt)
    return indexed


def _validate_generation(expected: Mapping[str, Any], path: Path, receipt: Mapping[str, Any]) -> tuple[Path, str, Mapping[str, Any]]:
    candidate = receipt["candidate"]
    _require(
        receipt["root_spec_raw_sha256"] == expected["root_spec_raw_sha256"]
        and candidate["candidate_id"] == expected["candidate_id"]
        and candidate["semantic_branch"] == expected["branch"]
        and candidate["analysis_split"] == expected["analysis_split"]
        and candidate["seed"] == expected["seed"]
        and candidate["calibration_group_id"] == f"cell-{expected['source_iid']}-s{expected['seed']}",
        f"candidate/spec coordinate differs: {expected['candidate_id']}",
    )
    media = receipt["artifacts"]["mp4"]
    media_path = _plain_file(media["path"], "generated review media")
    media_sha = _sha(media["sha256"], "generated media")
    _require(file_sha256(media_path) == media_sha, "generated media SHA differs")
    probe = _probe_full81(media_path)
    return media_path, media_sha, probe


def _opaque(key: bytes, domain: str, value: str) -> str:
    return hmac.new(key, f"{domain}\0{value}".encode("utf-8"), hashlib.sha256).hexdigest()


def _packet_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"packet_core_digest", "private_map_file_sha256", "packet_digest"}}


def _html_packet(packet_id: str, rows: Sequence[Mapping[str, Any]], rubrics: Mapping[str, Mapping[str, Any]]) -> bytes:
    cards = []
    for ordinal, row in enumerate(rows, 1):
        rubric = rubrics[row["rubric_id"]]
        cards.append(
            f'<article><h2>Blind clip {ordinal:03d}</h2><video controls muted playsinline preload="metadata" src="{html.escape(row["media_path"])}"></video>'
            f'<p><code>{html.escape(row["opaque_id"])}</code></p><details><summary>Registered observation rubric (not the hidden branch)</summary>'
            f'<p>{html.escape(rubric["scene_context"])}</p><p><strong>Forward target:</strong> {html.escape(rubric["forward_target_event"])}</p></details></article>'
        )
    definitions = "".join(f"<dt>{html.escape(axis)}</dt><dd>{html.escape(AXIS_DEFINITIONS[axis])}</dd>" for axis in AXIS_FIELDS)
    raw = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Blind full81 review</title><style>body{{font:16px system-ui;margin:2rem;background:#101317;color:#edf2f7}}main{{max-width:1100px;margin:auto}}article{{border:1px solid #445;padding:1rem;margin:1rem 0;background:#181d24}}video{{width:100%;max-height:620px;background:#000}}code{{font-size:.8rem}}dt{{font-weight:700;margin-top:.7rem}}dd{{margin-left:1rem}}</style></head><body><main><h1>Blind full81 external review</h1><p>Packet <code>{html.escape(packet_id)}</code>. Watch every clip through all 81 frames. Candidate IDs, prompts, seeds, splits, and requested branches are absent. Fill a copy of the blank response outside this packet and seal it before Phi extraction.</p><p>The ten axes below are separate categorical observations, never a weighted score. Phase labels are observation metadata, never optimizer or loss weights.</p><dl>{definitions}</dl>{''.join(cards)}</main></body></html>"""
    return raw.encode("utf-8")


def _gap(expected: Sequence[Mapping[str, Any]], observed: Mapping[str, Any], *, stage: str, failed: Sequence[str] = ()) -> dict[str, Any]:
    expected_ids = {row["candidate_id"] for row in expected}
    found_ids = set(observed)
    unsigned = {
        "schema_version": GAP_SCHEMA, "stage": stage,
        "existing_core4_expected": CORE4_CLIP_COUNT,
        "reserve4_expected": RESERVE4_CLIP_COUNT,
        "full_first8_expected": FULL_CLIP_COUNT,
        "observed_expected_candidate_count": len(expected_ids & found_ids),
        "missing_candidate_ids": sorted(expected_ids - found_ids),
        "unexpected_candidate_ids": sorted(found_ids - expected_ids),
        "failed_review_opaque_ids": sorted(failed),
        "full_first8_authority_authorized": False,
        "phi_v1_materialization_authorized": False,
        "planner_or_operator_optimizer_authorized": False,
        "generated_rgb_or_latent_is_editor_input_or_target": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def build_packet(
    *, authoring_path: str | Path, population_path: str | Path,
    generation_roots: Sequence[str | Path], blind_key_path: str | Path,
    packet_builder_execution_id: str, reviewer_authority_path: str | Path,
    expected_reviewer_authority_sha256: str,
    reviewer_tool_source_artifact: str | Path,
    authority_release_manifest: str | Path,
    expected_authority_release_manifest_sha256: str,
    public_output_dir: str | Path, private_map_output: str | Path,
    gap_output: str | Path,
) -> Mapping[str, Any]:
    _activate_installed_closure(
        authority_release_manifest, expected_authority_release_manifest_sha256,
    )
    base_manifests, _ = _require_base_modules()
    _require(_EXECUTION.fullmatch(packet_builder_execution_id) is not None, "packet builder execution id differs")
    reviewer_authority, reviewer_authority_source, reviewer_authority_sha, reviewer_tool_source = _validate_reviewer_authority(
        reviewer_authority_path,
        expected_reviewer_authority_sha256,
        reviewer_tool_source_artifact,
    )
    source_path = Path(__file__).resolve(strict=True)
    source_sha = file_sha256(source_path)
    _require(
        reviewer_authority["reviewer_tool_source_sha256"] != source_sha
        and reviewer_tool_source != source_path,
        "packet builder cannot be its own external reviewer tool",
    )
    authoring, _, _ = _load_json(authoring_path, "authoring", base_manifests.AUTHORING_SHA256, require_canonical=False)
    population, _, _ = _load_json(population_path, "population", base_manifests.POPULATION_SHA256, require_canonical=False)
    expected, cells = _population_context(authoring, population)
    generation = _scan_generation(generation_roots)
    missing = [row["candidate_id"] for row in expected if row["candidate_id"] not in generation]
    unexpected = sorted(set(generation) - {row["candidate_id"] for row in expected})
    if missing or unexpected or len(generation) != FULL_CLIP_COUNT:
        _write_json(Path(gap_output), _gap(expected, generation, stage="packet-build-preflight"))
    _require(not missing and not unexpected and len(generation) == FULL_CLIP_COUNT, "full-first8 generation closure is incomplete; gap receipt written")
    key_path = _plain_file(blind_key_path, "blind key")
    key = key_path.read_bytes()
    _require(len(key) == 32, "blind key must contain exactly 32 bytes")
    output = Path(public_output_dir)
    private_path = Path(private_map_output)
    _require(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink(), "public output must be a fresh absolute directory")
    _require(private_path.is_absolute() and private_path.parent.is_dir() and output not in private_path.parents and private_path != output, "private map must be fresh and outside public packet")
    _require(not private_path.exists() and not private_path.is_symlink(), "private map output already exists")
    staging = Path(tempfile.mkdtemp(prefix=".blind-review-", dir=output.parent))
    try:
        media_dir = staging / "media"
        media_dir.mkdir()
        public_rows: list[dict[str, Any]] = []
        private_rows: list[dict[str, Any]] = []
        rubrics: dict[str, dict[str, Any]] = {}
        for registered in expected:
            receipt_path, receipt = generation[registered["candidate_id"]]
            media_path, media_sha, probe = _validate_generation(registered, receipt_path, receipt)
            opaque_id = f"clip-{_opaque(key, 'clip', registered['candidate_id'] + ':' + media_sha)[:32]}"
            rubric_id = f"rubric-{_opaque(key, 'rubric', registered['source_iid'])[:24]}"
            cell = cells[registered["source_iid"]]
            rubrics.setdefault(rubric_id, {
                "rubric_id": rubric_id, "scene_context": cell["scene_caption"],
                "forward_target_event": cell["branch_descriptions"]["action"],
                "classification_axes_are_independent_not_weighted": True,
            })
            relative = f"media/{opaque_id}.mp4"
            _copy_plain(media_path, media_dir / f"{opaque_id}.mp4", media_sha)
            public_rows.append({
                "opaque_id": opaque_id, "rubric_id": rubric_id,
                "media_path": relative, "media_sha256": media_sha,
                "exact81": probe,
            })
            private_rows.append({
                **registered, "opaque_id": opaque_id, "rubric_id": rubric_id,
                "source_media_path": str(media_path), "media_sha256": media_sha,
                "generation_receipt_path": str(receipt_path),
                "generation_receipt_file_sha256": receipt["_file_sha256"],
            })
        public_rows.sort(key=lambda row: row["opaque_id"])
        private_rows.sort(key=lambda row: row["opaque_id"])
        _require(len({row["opaque_id"] for row in public_rows}) == FULL_CLIP_COUNT, "opaque identifiers collide")
        _require(len({row["media_sha256"] for row in public_rows}) == FULL_CLIP_COUNT, "generated media bytes alias across registered candidates")
        packet_id = f"generic-action-first8-full160-{object_sha256([row['opaque_id'] for row in public_rows])[:16]}"
        html_raw = _html_packet(packet_id, public_rows, rubrics)
        blank = {
            "schema_version": RESPONSE_SCHEMA, "packet_manifest_file_sha256": None,
            "packet_digest": None,
            "rows": [{
                "opaque_id": row["opaque_id"], "media_sha256": row["media_sha256"],
                "entire_exact81_video_viewed": None, "frame_count": 81, "fps": 25,
                "technical_quality_pass": None, "observed_semantic_class": None,
                "independent_axes": {axis: None for axis in AXIS_FIELDS},
                "phase_labels": [None] * PHASES,
            } for row in public_rows], "sealed_before_phi_extraction": None,
            "response_digest": None, "execution_credential": None,
        }
        blank_raw = canonical_json_bytes(blank) + b"\n"
        reviewer_authority_public_raw = reviewer_authority_source.read_bytes()
        reviewer_tool_public_raw = reviewer_tool_source.read_bytes()
        fixed_reviewer = {
            "path": "reviewer-authority.json",
            "file_sha256": reviewer_authority_sha,
            "authority_digest": reviewer_authority["authority_digest"],
            "authority_id": reviewer_authority["authority_id"],
            "reviewer_id": reviewer_authority["reviewer_id"],
            "review_method": reviewer_authority["review_method"],
            "reviewer_tool_source_path": "reviewer-tool-source.artifact",
            "reviewer_tool_source_sha256": reviewer_authority["reviewer_tool_source_sha256"],
            "signature_algorithm": "ed25519",
            "public_key_sha256": reviewer_authority["verification_key"]["public_key_sha256"],
        }
        core = {
            "schema_version": PACKET_SCHEMA, "packet_id": packet_id,
            "packet_builder_execution_id": packet_builder_execution_id,
            "packet_builder_source_sha256": source_sha,
            "authority_release_manifest_file_sha256": expected_authority_release_manifest_sha256,
            "authoring_registry_sha256": base_manifests.AUTHORING_SHA256,
            "population_registry_sha256": base_manifests.POPULATION_SHA256,
            "row_count": FULL_CLIP_COUNT, "seed_cell_count": SEED_CELL_COUNT,
            "scope_counts": {"core4": CORE4_CLIP_COUNT, "reserve4": RESERVE4_CLIP_COUNT, "fit": 80, "confirmation": 80},
            "public_field_exclusions": ["candidate_id", "semantic_branch", "prompt", "seed", "analysis_split", "source_iid"],
            "reviewer_blinded_to_prompt_and_requested_branch": True,
            "reviewer_authority": fixed_reviewer,
            "packet_sealed_before_review": True,
            "classification_axes_are_independent_not_weighted": True,
            "classification_axis_definitions": AXIS_DEFINITIONS,
            "phase_labels_are_observation_metadata_not_loss_weights": True,
            "generated_rgb_or_latent_is_editor_input_or_target": False,
            "public_artifact_sha256": {
                "blind-review.html": hashlib.sha256(html_raw).hexdigest(),
                "external-review-response.blank.json": hashlib.sha256(blank_raw).hexdigest(),
                "reviewer-authority.json": hashlib.sha256(reviewer_authority_public_raw).hexdigest(),
                "reviewer-tool-source.artifact": hashlib.sha256(reviewer_tool_public_raw).hexdigest(),
            },
            "rubrics": [rubrics[key] for key in sorted(rubrics)],
            "rows": public_rows,
        }
        core_digest = object_sha256(core)
        private_unsigned = {
            "schema_version": PRIVATE_MAP_SCHEMA, "packet_id": packet_id,
            "packet_core_digest": core_digest,
            "packet_builder_execution_id": packet_builder_execution_id,
            "blind_key_sha256": hashlib.sha256(key).hexdigest(),
            "row_count": FULL_CLIP_COUNT, "rows": private_rows,
            "sealed_before_external_review": True,
            "generated_rgb_or_latent_is_editor_input_or_target": False,
        }
        private_value = {**private_unsigned, "map_digest": object_sha256(private_unsigned)}
        private_raw = canonical_json_bytes(private_value) + b"\n"
        private_sha = hashlib.sha256(private_raw).hexdigest()
        packet_unsigned = {**core, "packet_core_digest": core_digest, "private_map_file_sha256": private_sha}
        packet = {**packet_unsigned, "packet_digest": object_sha256(packet_unsigned)}
        _atomic_create(staging / "blind-review.html", html_raw, 0o444)
        _atomic_create(staging / "external-review-response.blank.json", blank_raw, 0o444)
        _atomic_create(staging / "reviewer-authority.json", reviewer_authority_public_raw, 0o444)
        _atomic_create(staging / "reviewer-tool-source.artifact", reviewer_tool_public_raw, 0o444)
        _write_json(staging / "packet-manifest.json", packet, 0o444)
        _atomic_create(private_path, private_raw, 0o400)
        os.rename(staging, output)
        os.chmod(output / "media", 0o555)
        os.chmod(output, 0o555)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return packet


def _validate_packet(
    packet_path: str | Path, packet_sha256: str,
    private_map_path: str | Path, private_map_sha256: str,
    *, expected_authority_release_manifest_sha256: str,
    decode_media: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    base_manifests, _ = _require_base_modules()
    packet, path, observed_packet_sha = _load_json(packet_path, "packet manifest", packet_sha256)
    _closed(packet, {"schema_version", "packet_id", "packet_builder_execution_id", "packet_builder_source_sha256", "authority_release_manifest_file_sha256", "authoring_registry_sha256", "population_registry_sha256", "row_count", "seed_cell_count", "scope_counts", "public_field_exclusions", "reviewer_blinded_to_prompt_and_requested_branch", "reviewer_authority", "packet_sealed_before_review", "classification_axes_are_independent_not_weighted", "classification_axis_definitions", "phase_labels_are_observation_metadata_not_loss_weights", "generated_rgb_or_latent_is_editor_input_or_target", "public_artifact_sha256", "rubrics", "rows", "packet_core_digest", "private_map_file_sha256", "packet_digest"}, "packet")
    _verify_seal(packet, "packet_digest", "packet")
    _require(packet["schema_version"] == PACKET_SCHEMA and packet["row_count"] == FULL_CLIP_COUNT and packet["seed_cell_count"] == SEED_CELL_COUNT, "packet population differs")
    _require(packet["authority_release_manifest_file_sha256"] == expected_authority_release_manifest_sha256, "packet release closure binding differs")
    _require(packet["authoring_registry_sha256"] == base_manifests.AUTHORING_SHA256 and packet["population_registry_sha256"] == base_manifests.POPULATION_SHA256, "packet registry authority differs")
    _require(packet["scope_counts"] == {"core4": 80, "reserve4": 80, "fit": 80, "confirmation": 80}, "packet scope counts differ")
    _require(packet["public_field_exclusions"] == ["candidate_id", "semantic_branch", "prompt", "seed", "analysis_split", "source_iid"], "packet blind-field exclusions differ")
    _require(packet["packet_core_digest"] == object_sha256(_packet_core(packet)), "packet core digest differs")
    _require(packet["packet_builder_source_sha256"] == file_sha256(Path(__file__).resolve(strict=True)), "packet builder source identity differs")
    _require(packet["packet_sealed_before_review"] is True and packet["reviewer_blinded_to_prompt_and_requested_branch"] is True and packet["classification_axes_are_independent_not_weighted"] is True and packet["classification_axis_definitions"] == AXIS_DEFINITIONS and packet["phase_labels_are_observation_metadata_not_loss_weights"] is True and packet["generated_rgb_or_latent_is_editor_input_or_target"] is False, "packet authority boundary differs")
    fixed_reviewer = _closed(packet["reviewer_authority"], {"path", "file_sha256", "authority_digest", "authority_id", "reviewer_id", "review_method", "reviewer_tool_source_path", "reviewer_tool_source_sha256", "signature_algorithm", "public_key_sha256"}, "packet reviewer authority")
    _require(fixed_reviewer["path"] == "reviewer-authority.json" and fixed_reviewer["reviewer_tool_source_path"] == "reviewer-tool-source.artifact" and fixed_reviewer["signature_algorithm"] == "ed25519", "packet reviewer authority path/algorithm differs")
    reviewer_authority, _, reviewer_authority_sha, reviewer_tool_path = _validate_reviewer_authority(
        path.parent / fixed_reviewer["path"], fixed_reviewer["file_sha256"],
        path.parent / fixed_reviewer["reviewer_tool_source_path"],
    )
    _require(
        reviewer_authority_sha == fixed_reviewer["file_sha256"]
        and reviewer_authority["authority_digest"] == fixed_reviewer["authority_digest"]
        and reviewer_authority["authority_id"] == fixed_reviewer["authority_id"]
        and reviewer_authority["reviewer_id"] == fixed_reviewer["reviewer_id"]
        and reviewer_authority["review_method"] == fixed_reviewer["review_method"]
        and reviewer_authority["reviewer_tool_source_sha256"] == fixed_reviewer["reviewer_tool_source_sha256"]
        and reviewer_authority["verification_key"]["public_key_sha256"] == fixed_reviewer["public_key_sha256"]
        and fixed_reviewer["reviewer_tool_source_sha256"] != packet["packet_builder_source_sha256"]
        and reviewer_tool_path != Path(__file__).resolve(strict=True),
        "packet fixed reviewer authority binding differs",
    )
    _require(packet["public_artifact_sha256"] == {
        "blind-review.html": file_sha256(path.parent / "blind-review.html"),
        "external-review-response.blank.json": file_sha256(path.parent / "external-review-response.blank.json"),
        "reviewer-authority.json": file_sha256(path.parent / "reviewer-authority.json"),
        "reviewer-tool-source.artifact": file_sha256(path.parent / "reviewer-tool-source.artifact"),
    }, "public review artifact bytes differ")
    private, _, observed_private_sha = _load_json(private_map_path, "private map", private_map_sha256)
    _closed(private, {"schema_version", "packet_id", "packet_core_digest", "packet_builder_execution_id", "blind_key_sha256", "row_count", "rows", "sealed_before_external_review", "generated_rgb_or_latent_is_editor_input_or_target", "map_digest"}, "private map")
    _verify_seal(private, "map_digest", "private map")
    _require(private["schema_version"] == PRIVATE_MAP_SCHEMA and private["row_count"] == FULL_CLIP_COUNT and private["sealed_before_external_review"] is True and private["generated_rgb_or_latent_is_editor_input_or_target"] is False, "private map boundary differs")
    _require(packet["private_map_file_sha256"] == observed_private_sha and private["packet_id"] == packet["packet_id"] and private["packet_core_digest"] == packet["packet_core_digest"] and private["packet_builder_execution_id"] == packet["packet_builder_execution_id"], "packet/private map binding differs")
    public_rows = packet["rows"]
    private_rows = private["rows"]
    _require(type(public_rows) is list and type(private_rows) is list and len(public_rows) == len(private_rows) == FULL_CLIP_COUNT, "packet row closure differs")
    _require([row["opaque_id"] for row in public_rows] == [row["opaque_id"] for row in private_rows], "public/private opaque order differs")
    authoring, _, _ = _load_json(METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json", "installed authoring", base_manifests.AUTHORING_SHA256, require_canonical=False)
    population, _, _ = _load_json(METHOD_ROOT / "assets/mosaic_event_population_compact6_topup20_v1.json", "installed population", base_manifests.POPULATION_SHA256, require_canonical=False)
    expected, _ = _population_context(authoring, population)
    expected_by_id = {row["candidate_id"]: row for row in expected}
    _require({row.get("candidate_id") for row in private_rows} == set(expected_by_id), "private map candidate population differs")
    private_fields = {"candidate_id", "source_iid", "analysis_split", "seed", "seed_kind", "profile_id", "branch", "root_spec_raw_sha256", "opaque_id", "rubric_id", "source_media_path", "media_sha256", "generation_receipt_path", "generation_receipt_file_sha256"}
    for row in private_rows:
        _closed(row, private_fields, "private map row")
        registered = expected_by_id[row["candidate_id"]]
        _require(all(row[field] == registered[field] for field in registered), "private map registered coordinate differs")
        _sha(row["media_sha256"], "private media"); _sha(row["generation_receipt_file_sha256"], "generation receipt")
    rubrics = packet["rubrics"]
    _require(type(rubrics) is list and len(rubrics) == 8 and len({row.get("rubric_id") for row in rubrics}) == 8, "packet rubric closure differs")
    for rubric in rubrics:
        _closed(rubric, {"rubric_id", "scene_context", "forward_target_event", "classification_axes_are_independent_not_weighted"}, "packet rubric")
        _require(rubric["classification_axes_are_independent_not_weighted"] is True and type(rubric["scene_context"]) is str and type(rubric["forward_target_event"]) is str, "packet rubric differs")
    rubric_ids = {row["rubric_id"] for row in rubrics}
    expected_paths = {"blind-review.html", "external-review-response.blank.json", "reviewer-authority.json", "reviewer-tool-source.artifact", "packet-manifest.json"}
    for row, hidden in zip(public_rows, private_rows):
        _closed(row, {"opaque_id", "rubric_id", "media_path", "media_sha256", "exact81"}, "public row")
        _require(_OPAQUE.fullmatch(row["opaque_id"]) is not None and row["rubric_id"] in rubric_ids and row["media_path"] == f"media/{row['opaque_id']}.mp4", "public opaque media path differs")
        _require(row["media_sha256"] == hidden["media_sha256"] and row["rubric_id"] == hidden["rubric_id"], "public/private media or rubric differs")
        exact = _closed(row["exact81"], {"decoder", "decoder_source_sha256", "all_integer_frames_0_through_80_decoded", "frame_count", "fps", "height", "width", "channels", "dtype"}, "packet exact81")
        _require(exact["decoder"] == "tools.materialize_vae._decode_exact_video" and exact["all_integer_frames_0_through_80_decoded"] is True and exact["frame_count"] == 81 and exact["fps"] == 25 and type(exact["height"]) is int and exact["height"] > 0 and type(exact["width"]) is int and exact["width"] > 0 and exact["channels"] == 3 and exact["dtype"] == "uint8", "packet exact81 contract differs")
        _sha(exact["decoder_source_sha256"], "packet decoder source")
        media = _plain_file((path.parent / row["media_path"]).resolve(), "packet media")
        _require(media.parent == (path.parent / "media").resolve() and file_sha256(media) == row["media_sha256"], "packet media binding differs")
        if decode_media:
            _require(_probe_full81(media) == row["exact81"], "packet exact81 replay differs")
        expected_paths.add(row["media_path"])
    actual_paths = {str(item.relative_to(path.parent)) for item in path.parent.rglob("*") if item.is_file()}
    _require(actual_paths == expected_paths, "public packet exact member closure differs")
    _require(observed_packet_sha == packet_sha256, "packet SHA replay differs")
    return packet, private


def _validate_response(
    value: Mapping[str, Any], packet: Mapping[str, Any],
    packet_manifest_file_sha256: str, packet_manifest_parent: Path,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    base_manifests, _ = _require_base_modules()
    _closed(value, {"schema_version", "packet_manifest_file_sha256", "packet_digest", "rows", "sealed_before_phi_extraction", "response_digest", "execution_credential"}, "external response")
    unsigned_response = {
        key: item for key, item in value.items()
        if key not in {"response_digest", "execution_credential"}
    }
    _require(object_sha256(unsigned_response) == _sha(value["response_digest"], "external response digest"), "external response digest differs")
    _require(value["schema_version"] == RESPONSE_SCHEMA and value["packet_manifest_file_sha256"] == packet_manifest_file_sha256 and value["packet_digest"] == packet["packet_digest"], "external response packet binding differs")
    fixed = packet["reviewer_authority"]
    reviewer_authority, _, _, _ = _validate_reviewer_authority(
        packet_manifest_parent / fixed["path"],
        fixed["file_sha256"],
        packet_manifest_parent / fixed["reviewer_tool_source_path"],
    )
    credential = value["execution_credential"]
    _verify_execution_credential(
        credential,
        packet=packet,
        packet_manifest_file_sha256=packet_manifest_file_sha256,
        response_digest=value["response_digest"],
        reviewer_authority=reviewer_authority,
    )
    rows = value["rows"]
    _require(type(rows) is list and len(rows) == FULL_CLIP_COUNT and [row.get("opaque_id") for row in rows] == [row["opaque_id"] for row in packet["rows"]], "external response row order/closure differs")
    public_by_id = {row["opaque_id"]: row for row in packet["rows"]}
    for row in rows:
        _closed(row, {"opaque_id", "media_sha256", "entire_exact81_video_viewed", "frame_count", "fps", "technical_quality_pass", "observed_semantic_class", "independent_axes", "phase_labels"}, "external response row")
        public = public_by_id[row["opaque_id"]]
        _require(row["media_sha256"] == public["media_sha256"] and row["frame_count"] == 81 and row["fps"] == 25, "response media binding differs")
        _require(type(row["entire_exact81_video_viewed"]) is bool and type(row["technical_quality_pass"]) is bool, "response review booleans differ")
        _require(row["observed_semantic_class"] in set(BRANCHES) | {CLASS_UNJUDGEABLE}, "observed semantic class differs")
        axes = _closed(row["independent_axes"], AXIS_FIELDS, "independent axes")
        _require(all(type(axes[field]) is bool for field in AXIS_FIELDS), "independent axes must be booleans")
        enabled = [field for field in AXIS_FIELDS if axes[field]]
        if row["observed_semantic_class"] == CLASS_UNJUDGEABLE:
            _require(not enabled, "unjudgeable row may not assert a semantic axis")
        else:
            _require(enabled == [row["observed_semantic_class"]], "semantic axes are mixed or weighted")
        base_manifests.validate_phase_labels(row["phase_labels"], "external response phase labels")
    _require(value["sealed_before_phi_extraction"] is True, "external response was not sealed before Phi extraction")
    reviewer = {
        "authority_id": reviewer_authority["authority_id"],
        "reviewer_id": reviewer_authority["reviewer_id"],
        "review_method": reviewer_authority["review_method"],
        "reviewer_tool_source_sha256": reviewer_authority["reviewer_tool_source_sha256"],
        "review_execution_id": credential["review_execution_id"],
        "credential_signature_sha256": hashlib.sha256(bytes.fromhex(credential["signature_hex"])).hexdigest(),
        "independent_of_generation_runner": True,
        "independent_of_packet_builder": True,
        "independent_of_phi_runner": True,
        "response_authored_outside_packet": True,
    }
    return rows, reviewer, credential


def ingest_external_review(
    *, packet_manifest: str | Path, expected_packet_sha256: str,
    private_map: str | Path, expected_private_map_sha256: str,
    external_response: str | Path, expected_response_sha256: str,
    authority_release_manifest: str | Path,
    expected_authority_release_manifest_sha256: str,
    output_dir: str | Path, authority_output: str | Path,
    gap_output: str | Path,
) -> Mapping[str, Any]:
    _activate_installed_closure(
        authority_release_manifest, expected_authority_release_manifest_sha256,
    )
    base_manifests, _ = _require_base_modules()
    packet, private = _validate_packet(
        packet_manifest, expected_packet_sha256, private_map,
        expected_private_map_sha256,
        expected_authority_release_manifest_sha256=expected_authority_release_manifest_sha256,
        decode_media=True,
    )
    response, response_path, observed_response_sha = _load_json(external_response, "external response", expected_response_sha256)
    packet_parent = _plain_file(packet_manifest, "packet manifest").parent
    rows, reviewer, credential = _validate_response(
        response, packet, expected_packet_sha256, packet_parent,
    )
    private_by_id = {row["opaque_id"]: row for row in private["rows"]}
    failed = [row["opaque_id"] for row in rows if not row["entire_exact81_video_viewed"] or not row["technical_quality_pass"] or row["observed_semantic_class"] != private_by_id[row["opaque_id"]]["branch"]]
    if failed:
        _write_json(Path(gap_output), _gap(private["rows"], {row["candidate_id"]: row for row in private["rows"]}, stage="external-review-ingestion", failed=failed))
    _require(not failed, "external review has failed/unjudgeable/mismatched rows; gap receipt written")
    output = Path(output_dir)
    authority_path = Path(authority_output)
    _require(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink(), "review receipt output must be fresh")
    _require(authority_path.is_absolute() and authority_path.parent.is_dir() and not authority_path.exists() and not authority_path.is_symlink(), "authority output must be fresh")
    output.mkdir()
    receipt_rows: list[dict[str, Any]] = []
    try:
        for ordinal, response_row in enumerate(rows):
            hidden = private_by_id[response_row["opaque_id"]]
            branch = hidden["branch"]
            observations = {
                "start_state_present": True,
                "transition_present": branch in {"action", "reverse", "incomplete"},
                "requested_terminal_present": branch in {"action", "reverse"},
                "terminal_hold_present": branch in {"action", "reverse"},
                "full_target_event_present": branch in {"action", "reverse"},
            }
            unsigned = {
                "schema_version": base_manifests.REVIEW_SCHEMA,
                "candidate_id": hidden["candidate_id"], "branch": branch,
                "media_sha256": hidden["media_sha256"],
                "review_method": reviewer["review_method"],
                "entire_exact81_video_viewed": True, "frame_count": 81, "fps": 25,
                "reviewer_blinded_to_prompt_and_requested_branch": True,
                "sealed_before_phi_extraction": True, "quality_pass": True,
                "branch_semantics_pass": True,
                "phase_labels": response_row["phase_labels"],
                "observations": observations,
            }
            receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
            receipt_path = output / f"review-{ordinal:03d}.json"
            receipt_sha = _write_json(receipt_path, receipt, 0o400)
            base_manifests.validate_review_receipt(receipt_path, receipt_sha)
            receipt_rows.append({
                "opaque_id": hidden["opaque_id"], "candidate_id": hidden["candidate_id"],
                "source_iid": hidden["source_iid"], "analysis_split": hidden["analysis_split"],
                "seed": hidden["seed"], "profile_id": hidden["profile_id"], "branch": branch,
                "media_sha256": hidden["media_sha256"],
                "generation_receipt_path": hidden["generation_receipt_path"],
                "generation_receipt_file_sha256": hidden["generation_receipt_file_sha256"],
                "review_receipt_path": str(receipt_path), "review_receipt_file_sha256": receipt_sha,
                "review_receipt_digest": receipt["receipt_digest"],
            })
        _require(len(receipt_rows) == FULL_CLIP_COUNT, "review receipt closure differs")
        authority_unsigned = {
            "schema_version": AUTHORITY_SCHEMA,
            "authority_id": f"external-full160-{packet['packet_digest'][:16]}-{response['response_digest'][:16]}",
            "packet_manifest_path": str(_plain_file(packet_manifest, "packet manifest")),
            "packet_manifest_file_sha256": expected_packet_sha256,
            "packet_digest": packet["packet_digest"],
            "authority_release_manifest_file_sha256": expected_authority_release_manifest_sha256,
            "private_map_path": str(_plain_file(private_map, "private map")),
            "private_map_file_sha256": expected_private_map_sha256,
            "private_map_digest": private["map_digest"],
            "external_response_path": str(response_path),
            "external_response_file_sha256": observed_response_sha,
            "external_response_digest": response["response_digest"],
            "reviewer_authority": dict(packet["reviewer_authority"]),
            "reviewer": dict(reviewer),
            "execution_credential": dict(credential),
            "row_count": FULL_CLIP_COUNT, "seed_cell_count": SEED_CELL_COUNT,
            "scope_counts": {"core4": 80, "reserve4": 80, "fit": 80, "confirmation": 80},
            "rows": receipt_rows,
            "sealed_before_phi_extraction": True,
            "same_runner_self_certification": False,
            "classification_axes_are_independent_not_weighted": True,
            "phase_labels_are_observation_metadata_not_loss_weights": True,
            "full_first8_external_review_authorized": True,
            "phi_v1_materialization_authorized": True,
            "generated_rgb_or_latent_is_editor_input_or_target": False,
            "planner_or_operator_optimizer_authorized": False,
        }
        authority = {**authority_unsigned, "authority_digest": object_sha256(authority_unsigned)}
        _write_json(authority_path, authority, 0o400)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return authority


def load_authority(
    path: str | Path, expected_sha256: str,
    *, authority_release_manifest: str | Path,
    expected_authority_release_manifest_sha256: str,
    replay_packet: bool = True,
) -> Mapping[str, Any]:
    _activate_installed_closure(
        authority_release_manifest, expected_authority_release_manifest_sha256,
    )
    base_manifests, _ = _require_base_modules()
    _require(replay_packet is True, "external authority replay cannot be disabled")
    authority, _, _ = _load_json(path, "external review authority", expected_sha256)
    _verify_seal(authority, "authority_digest", "external review authority")
    required = {"schema_version", "authority_id", "packet_manifest_path", "packet_manifest_file_sha256", "packet_digest", "authority_release_manifest_file_sha256", "private_map_path", "private_map_file_sha256", "private_map_digest", "external_response_path", "external_response_file_sha256", "external_response_digest", "reviewer_authority", "reviewer", "execution_credential", "row_count", "seed_cell_count", "scope_counts", "rows", "sealed_before_phi_extraction", "same_runner_self_certification", "classification_axes_are_independent_not_weighted", "phase_labels_are_observation_metadata_not_loss_weights", "full_first8_external_review_authorized", "phi_v1_materialization_authorized", "generated_rgb_or_latent_is_editor_input_or_target", "planner_or_operator_optimizer_authorized", "authority_digest"}
    _closed(authority, required, "external review authority")
    _require(authority["schema_version"] == AUTHORITY_SCHEMA and authority["row_count"] == 160 and authority["seed_cell_count"] == 16 and authority["scope_counts"] == {"core4": 80, "reserve4": 80, "fit": 80, "confirmation": 80}, "external review authority population differs")
    _require(authority["authority_release_manifest_file_sha256"] == expected_authority_release_manifest_sha256, "external review release closure binding differs")
    _require(authority["sealed_before_phi_extraction"] is True and authority["same_runner_self_certification"] is False and authority["classification_axes_are_independent_not_weighted"] is True and authority["phase_labels_are_observation_metadata_not_loss_weights"] is True and authority["full_first8_external_review_authorized"] is True and authority["phi_v1_materialization_authorized"] is True and authority["generated_rgb_or_latent_is_editor_input_or_target"] is False and authority["planner_or_operator_optimizer_authorized"] is False, "external review authority boundary differs")
    rows = authority["rows"]
    _require(type(rows) is list and len(rows) == 160 and len({row["candidate_id"] for row in rows}) == 160, "authority row closure differs")
    _require(sum(row["profile_id"] == "core4-v2" for row in rows) == 80 and sum(row["profile_id"] == "reserve4-v1" for row in rows) == 80 and sum(row["analysis_split"] == "fit" for row in rows) == 80 and sum(row["analysis_split"] == "confirmation" for row in rows) == 80, "authority scope partition differs")
    row_fields = {"opaque_id", "candidate_id", "source_iid", "analysis_split", "seed", "profile_id", "branch", "media_sha256", "generation_receipt_path", "generation_receipt_file_sha256", "review_receipt_path", "review_receipt_file_sha256", "review_receipt_digest"}
    for row in rows:
        _closed(row, row_fields, "external authority row")
        receipt = base_manifests.validate_review_receipt(row["review_receipt_path"], row["review_receipt_file_sha256"])
        _require(receipt["candidate_id"] == row["candidate_id"] and receipt["branch"] == row["branch"] and receipt["media_sha256"] == row["media_sha256"], "authority/review receipt binding differs")
    if replay_packet:
        packet, private = _validate_packet(
            authority["packet_manifest_path"], authority["packet_manifest_file_sha256"],
            authority["private_map_path"], authority["private_map_file_sha256"],
            expected_authority_release_manifest_sha256=expected_authority_release_manifest_sha256,
            decode_media=True,
        )
        _require(packet["packet_digest"] == authority["packet_digest"] and private["map_digest"] == authority["private_map_digest"], "authority packet replay differs")
        response, _, _ = _load_json(authority["external_response_path"], "external response", authority["external_response_file_sha256"])
        _require(response["response_digest"] == authority["external_response_digest"], "authority external response replay differs")
        response_rows, reviewer, credential = _validate_response(
            response, packet, authority["packet_manifest_file_sha256"],
            _plain_file(authority["packet_manifest_path"], "packet manifest").parent,
        )
        _require(dict(packet["reviewer_authority"]) == authority["reviewer_authority"], "authority fixed reviewer binding differs")
        _require(dict(reviewer) == authority["reviewer"] and dict(credential) == authority["execution_credential"], "authority reviewer credential binding differs")
        private_by_id = {row["opaque_id"]: row for row in private["rows"]}
        authority_by_id = {row["opaque_id"]: row for row in rows}
        _require(set(private_by_id) == set(authority_by_id), "authority/private opaque population differs")
        for response_row in response_rows:
            hidden = private_by_id[response_row["opaque_id"]]
            registered = authority_by_id[response_row["opaque_id"]]
            _require(response_row["entire_exact81_video_viewed"] is True and response_row["technical_quality_pass"] is True and response_row["observed_semantic_class"] == hidden["branch"], "authority external review verdict differs")
            for field in ("candidate_id", "source_iid", "analysis_split", "seed", "profile_id", "branch", "media_sha256", "generation_receipt_path", "generation_receipt_file_sha256"):
                _require(registered[field] == hidden[field], "authority/private registered row differs")
    return authority


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-packet")
    build.add_argument("--authoring", required=True); build.add_argument("--population", required=True)
    build.add_argument("--generation-root", action="append", required=True); build.add_argument("--blind-key", required=True)
    build.add_argument("--reviewer-authority", required=True); build.add_argument("--expected-reviewer-authority-sha256", required=True)
    build.add_argument("--reviewer-tool-source-artifact", required=True)
    build.add_argument("--packet-builder-execution-id", required=True); build.add_argument("--public-output-dir", required=True)
    build.add_argument("--private-map-output", required=True); build.add_argument("--gap-output", required=True)
    build.add_argument("--authority-release-manifest", required=True); build.add_argument("--expected-authority-release-manifest-sha256", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--packet-manifest", required=True); ingest.add_argument("--expected-packet-sha256", required=True)
    ingest.add_argument("--private-map", required=True); ingest.add_argument("--expected-private-map-sha256", required=True)
    ingest.add_argument("--external-response", required=True); ingest.add_argument("--expected-response-sha256", required=True)
    ingest.add_argument("--output-dir", required=True); ingest.add_argument("--authority-output", required=True); ingest.add_argument("--gap-output", required=True)
    ingest.add_argument("--authority-release-manifest", required=True); ingest.add_argument("--expected-authority-release-manifest-sha256", required=True)
    audit = commands.add_parser("audit-authority")
    audit.add_argument("--authority", required=True); audit.add_argument("--expected-authority-sha256", required=True)
    audit.add_argument("--authority-release-manifest", required=True); audit.add_argument("--expected-authority-release-manifest-sha256", required=True)
    commands.add_parser("reviewer-authority-template")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reviewer-authority-template":
        value = reviewer_authority_template()
    elif args.command == "build-packet":
        value = build_packet(authoring_path=args.authoring, population_path=args.population, generation_roots=args.generation_root, blind_key_path=args.blind_key, packet_builder_execution_id=args.packet_builder_execution_id, reviewer_authority_path=args.reviewer_authority, expected_reviewer_authority_sha256=args.expected_reviewer_authority_sha256, reviewer_tool_source_artifact=args.reviewer_tool_source_artifact, authority_release_manifest=args.authority_release_manifest, expected_authority_release_manifest_sha256=args.expected_authority_release_manifest_sha256, public_output_dir=args.public_output_dir, private_map_output=args.private_map_output, gap_output=args.gap_output)
    elif args.command == "ingest":
        value = ingest_external_review(packet_manifest=args.packet_manifest, expected_packet_sha256=args.expected_packet_sha256, private_map=args.private_map, expected_private_map_sha256=args.expected_private_map_sha256, external_response=args.external_response, expected_response_sha256=args.expected_response_sha256, authority_release_manifest=args.authority_release_manifest, expected_authority_release_manifest_sha256=args.expected_authority_release_manifest_sha256, output_dir=args.output_dir, authority_output=args.authority_output, gap_output=args.gap_output)
    else:
        value = load_authority(args.authority, args.expected_authority_sha256, authority_release_manifest=args.authority_release_manifest, expected_authority_release_manifest_sha256=args.expected_authority_release_manifest_sha256)
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
