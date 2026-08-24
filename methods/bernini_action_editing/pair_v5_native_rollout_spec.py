#!/usr/bin/env python3
"""Closed, hash-bound specification for PAIR-v5 native RV2V-4 rollouts.

The proposal bank is deliberately absent from this schema.  Every candidate is
sampled from the deploy-time Bernini path: one full source video, four source
frames, one complete source-content/new-action caption, and native Gaussian
noise.  This module is dependency-free so a Slurm launcher can fail closed
before importing Torch or allocating model memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pair-v5-native-rv2v4-rollout-spec-v1"
CANDIDATE_SCHEMA_VERSION = "pair-v5-native-rv2v4-candidate-v1"
RECEIPT_SCHEMA_VERSION = "pair-v5-native-rv2v4-rollout-receipt-v1"
CAPTION_CONTRACT = "complete_source_content_caption_with_requested_new_action"
TARGET_INITIALIZATION = "official_gen_wanx22_fresh_gaussian"
REFERENCE_INDICES = [0, 27, 53, 80]
DEFAULT_GUIDANCE = {"omega_txt": 4.0, "omega_vid": 1.25, "omega_img": 4.5}
SEMANTIC_INPUT_CLOSURE = {
    "accepted": ["source_video", "complete_caption"],
    "target_video": False,
    "t2v_proposal_media": False,
    "donor_video": False,
    "external_reference": False,
    "mask": False,
    "flow": False,
    "pose": False,
    "track": False,
    "trajectory": False,
}
SAMPLING_CONTRACT = {
    "condition_mode": "rv2v4",
    "num_frames": 81,
    "latent_frames": 21,
    "fps": 25,
    "num_inference_steps": 40,
    "source_reference_indices": REFERENCE_INDICES,
    "target_initialization": TARGET_INITIALIZATION,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


class PairRolloutSpecError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _closed(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PairRolloutSpecError(
            f"{label} keys differ: expected={sorted(expected)!r}, "
            f"actual={sorted(value) if isinstance(value, Mapping) else type(value).__name__!r}"
        )


def _finite_nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairRolloutSpecError(f"{label} must be numeric")
    result = float(value)
    if not (result >= 0.0 and result < float("inf")):
        raise PairRolloutSpecError(f"{label} must be finite and nonnegative")
    return result


def validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    _closed(
        candidate,
        {
            "candidate_id",
            "source_video",
            "source_video_sha256",
            "complete_caption",
            "complete_caption_sha256",
            "caption_contract",
            "seed",
            "guidance",
        },
        "candidate",
    )
    candidate_id = candidate["candidate_id"]
    if not isinstance(candidate_id, str) or _SAFE_ID.fullmatch(candidate_id) is None:
        raise PairRolloutSpecError("candidate_id is not path-safe")
    source = Path(str(candidate["source_video"]))
    if not source.is_absolute() or source == Path("/"):
        raise PairRolloutSpecError("source_video must be absolute and non-root")
    source_sha = candidate["source_video_sha256"]
    if not isinstance(source_sha, str) or _SHA256.fullmatch(source_sha) is None:
        raise PairRolloutSpecError("source_video_sha256 must be lowercase SHA-256")
    caption = candidate["complete_caption"]
    if not isinstance(caption, str) or not caption.strip() or "\x00" in caption:
        raise PairRolloutSpecError("complete_caption must be non-empty text without NUL")
    if candidate["caption_contract"] != CAPTION_CONTRACT:
        raise PairRolloutSpecError("candidate does not attest the complete-caption contract")
    caption_sha = candidate["complete_caption_sha256"]
    if sha256_bytes(caption.encode("utf-8")) != caption_sha:
        raise PairRolloutSpecError("complete caption SHA-256 differs")
    seed = candidate["seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise PairRolloutSpecError("seed must be an integer in [0,2^63)")
    guidance = candidate["guidance"]
    _closed(guidance, set(DEFAULT_GUIDANCE), "candidate.guidance")
    normalized_guidance = {
        key: _finite_nonnegative_float(guidance[key], f"guidance.{key}")
        for key in DEFAULT_GUIDANCE
    }
    return {
        "candidate_id": candidate_id,
        "source_video": str(source),
        "source_video_sha256": source_sha,
        "complete_caption": caption,
        "complete_caption_sha256": caption_sha,
        "caption_contract": CAPTION_CONTRACT,
        "seed": seed,
        "guidance": normalized_guidance,
    }


def validate_root_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    _closed(
        value,
        {
            "schema_version",
            "sampling_contract",
            "semantic_input_closure",
            "groups",
        },
        "root spec",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise PairRolloutSpecError("root spec schema_version differs")
    if value["sampling_contract"] != SAMPLING_CONTRACT:
        raise PairRolloutSpecError("sampling contract is not exact81/40 native RV2V-4")
    if value["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE:
        raise PairRolloutSpecError("semantic input closure differs")
    groups = value["groups"]
    if not isinstance(groups, list) or len(groups) != 2:
        raise PairRolloutSpecError("exactly two SP4 groups are required")
    normalized_groups = []
    candidate_ids: set[str] = set()
    expected_layout = (("sp4-a", [0, 1, 2, 3]), ("sp4-b", [4, 5, 6, 7]))
    for group, (expected_id, expected_gpus) in zip(groups, expected_layout):
        _closed(group, {"group_id", "visible_gpus", "candidates"}, "group")
        if group["group_id"] != expected_id or group["visible_gpus"] != expected_gpus:
            raise PairRolloutSpecError("groups must be sp4-a=0..3 and sp4-b=4..7")
        candidates = group["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise PairRolloutSpecError("each SP4 group requires at least one candidate")
        normalized_candidates = []
        for candidate in candidates:
            normalized = validate_candidate(candidate)
            if normalized["candidate_id"] in candidate_ids:
                raise PairRolloutSpecError("candidate_id values must be globally unique")
            candidate_ids.add(normalized["candidate_id"])
            normalized_candidates.append(normalized)
        normalized_groups.append(
            {
                "group_id": expected_id,
                "visible_gpus": expected_gpus,
                "candidates": normalized_candidates,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "sampling_contract": dict(SAMPLING_CONTRACT),
        "semantic_input_closure": dict(SEMANTIC_INPUT_CLOSURE),
        "groups": normalized_groups,
    }


def load_sealed_spec(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise PairRolloutSpecError("expected spec SHA-256 is invalid")
    spec_path = Path(path)
    if not spec_path.is_absolute() or not spec_path.is_file() or spec_path.is_symlink():
        raise PairRolloutSpecError("sealed spec must be an absolute plain file")
    raw = spec_path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise PairRolloutSpecError("sealed spec raw SHA-256 differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairRolloutSpecError("sealed spec is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise PairRolloutSpecError("sealed spec root must be an object")
    normalized = validate_root_spec(value)
    return normalized, actual


def materialize_plan(
    *, spec_path: str | Path, expected_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    spec, digest = load_sealed_spec(spec_path, expected_sha256)
    output = Path(output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise PairRolloutSpecError("plan output must be a fresh absolute non-root path")
    output.mkdir(parents=False, exist_ok=False)
    records = []
    for group in spec["groups"]:
        group_dir = output / group["group_id"]
        group_dir.mkdir()
        for ordinal, candidate in enumerate(group["candidates"]):
            envelope = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "root_spec_raw_sha256": digest,
                "group_id": group["group_id"],
                "visible_gpus": group["visible_gpus"],
                "ordinal": ordinal,
                "sampling_contract": SAMPLING_CONTRACT,
                "semantic_input_closure": SEMANTIC_INPUT_CLOSURE,
                "candidate": candidate,
            }
            candidate_path = group_dir / f"{ordinal:04d}-{candidate['candidate_id']}.json"
            candidate_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
            os.chmod(candidate_path, 0o400)
            records.append(
                {
                    "group_id": group["group_id"],
                    "candidate_id": candidate["candidate_id"],
                    "path": str(candidate_path),
                    "sha256": sha256_bytes(candidate_path.read_bytes()),
                }
            )
    manifest = {
        "schema_version": "pair-v5-native-rv2v4-materialized-plan-v1",
        "root_spec_raw_sha256": digest,
        "candidate_records": records,
    }
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest))
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    os.chmod(output / "manifest.json", 0o400)
    return manifest


def load_candidate_envelope(path: str | Path, expected_root_sha256: str) -> dict[str, Any]:
    candidate_path = Path(path)
    if not candidate_path.is_absolute() or not candidate_path.is_file() or candidate_path.is_symlink():
        raise PairRolloutSpecError("candidate envelope must be an absolute plain file")
    value = json.loads(candidate_path.read_bytes())
    if not isinstance(value, Mapping):
        raise PairRolloutSpecError("candidate envelope root differs")
    _closed(
        value,
        {
            "schema_version", "root_spec_raw_sha256", "group_id", "visible_gpus",
            "ordinal", "sampling_contract", "semantic_input_closure", "candidate",
        },
        "candidate envelope",
    )
    if value["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise PairRolloutSpecError("candidate envelope schema differs")
    if value["root_spec_raw_sha256"] != expected_root_sha256:
        raise PairRolloutSpecError("candidate envelope root binding differs")
    if value["sampling_contract"] != SAMPLING_CONTRACT:
        raise PairRolloutSpecError("candidate sampling contract differs")
    if value["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE:
        raise PairRolloutSpecError("candidate semantic input closure differs")
    layout = {"sp4-a": [0, 1, 2, 3], "sp4-b": [4, 5, 6, 7]}
    if layout.get(value["group_id"]) != value["visible_gpus"]:
        raise PairRolloutSpecError("candidate SP4 binding differs")
    if type(value["ordinal"]) is not int or value["ordinal"] < 0:
        raise PairRolloutSpecError("candidate ordinal differs")
    normalized = dict(value)
    normalized["candidate"] = validate_candidate(value["candidate"])
    normalized["candidate_envelope_sha256"] = sha256_bytes(candidate_path.read_bytes())
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = materialize_plan(
        spec_path=args.spec,
        expected_sha256=args.expected_sha256,
        output_dir=args.output_dir,
    )
    print(canonical_json_bytes(manifest).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
