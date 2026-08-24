"""Prepare a provenance-bound R5-lite pseudo-label pilot.

This command does not synthesize human approval.  It retains strict legacy
Qwen positives and a deterministic audit set of clean negative examples, then
removes the legacy non-content split so the endpoint feature finalizer must
assign a diagnostic near-pHash source cluster split.  A formal production run
still requires a stronger source-visual-cluster split and human labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .qwen_filter import (
    _object_digest,
    _validate_observation,
    _validate_visual,
)


R5_PILOT_SCHEMA = "motive-r5-pseudo-pilot-v1"
R5_PILOT_PROFILE = "strict-legacy-qwen-original-v1"
POSITIVE_VERDICTS = {"valid_action", "valid_suppression"}
NEGATIVE_VERDICTS = {
    "static",
    "instruction_mismatch",
    "endpoint_only",
}
DEFAULT_NEGATIVE_CAPS = {
    "static": 40,
    "instruction_mismatch": 40,
    "endpoint_only": 10,
}


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _strict_visual_evidence(
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    visual = ((row.get("qwen_evidence") or {}).get("visual") or {})
    if (
        visual.get("status") != "ok"
        or visual.get("observation_validated_from") != "original"
        or visual.get("result_validated_from") != "original"
        or visual.get("observation_repairs")
        or visual.get("alignment_repairs")
    ):
        return None
    observation = visual.get("observation")
    result = visual.get("result")
    if not isinstance(observation, dict) or not isinstance(result, dict):
        return None
    try:
        _validate_observation(observation)
        _validate_visual(result, observation=observation)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        observation.get("camera_dominance") != "low"
        or observation.get("background_dominance") != "low"
        or observation.get("artifact_level") != "low"
        or observation.get("preservation_quality") != "acceptable"
    ):
        return None
    return observation, result


def _classify(
    row: dict[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any]] | None:
    evidence = _strict_visual_evidence(row)
    if evidence is None:
        return None
    observation, result = evidence
    verdict = str(result["verdict"])
    if verdict in POSITIVE_VERDICTS:
        if observation.get("target_actor_motion") != "clear":
            return None
        action_signature = str(result.get("action_signature") or "").strip()
        if not action_signature or action_signature == "unknown":
            return None
        return "positive", action_signature, observation, result
    if verdict in NEGATIVE_VERDICTS:
        return "negative", f"negative:{verdict}", observation, result
    return None


def _priority(seed: int, iid: str, bucket: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{int(seed)}\0{bucket}\0{iid}".encode("utf-8")
    ).hexdigest()
    return digest, iid


def prepare(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    summary_path = (
        args.summary.expanduser()
        if args.summary is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path == summary_path:
        raise ValueError("output and summary paths must differ")
    negative_caps = {
        "static": int(args.static_negatives),
        "instruction_mismatch": int(args.mismatch_negatives),
        "endpoint_only": int(args.endpoint_negatives),
    }
    if any(value < 0 for value in negative_caps.values()):
        raise ValueError("negative caps must be non-negative")
    positive_limit = int(args.positive_limit)
    if positive_limit < 0:
        raise ValueError("--positive-limit must be non-negative")

    source_sha256 = _file_digest(input_path)
    positives: list[
        tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]
    ] = []
    negative_buckets: dict[
        str,
        list[tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]],
    ] = {name: [] for name in negative_caps}
    scanned = 0
    rejected = Counter()
    seen_iids: set[str] = set()
    for row in _iter_jsonl(input_path):
        scanned += 1
        iid = str(row.get("iid") or "")
        if not iid or iid in seen_iids:
            raise ValueError(f"missing/duplicate iid={iid!r}")
        seen_iids.add(iid)
        classified = _classify(row)
        if classified is None:
            rejected["not_strict_clean_pilot_row"] += 1
            continue
        label_class, signature, observation, result = classified
        item = (row, signature, observation, result)
        if label_class == "positive":
            positives.append(item)
        else:
            negative_buckets[str(result["verdict"])].append(item)

    positives.sort(
        key=lambda item: _priority(
            args.data_seed,
            str(item[0]["iid"]),
            "positive",
        )
    )
    if positive_limit:
        positives = positives[:positive_limit]
    selected = list(positives)
    selected_negative_counts: Counter[str] = Counter()
    for verdict in sorted(negative_buckets):
        values = sorted(
            negative_buckets[verdict],
            key=lambda item: _priority(
                args.data_seed,
                str(item[0]["iid"]),
                verdict,
            ),
        )
        chosen = values[: negative_caps[verdict]]
        selected.extend(chosen)
        selected_negative_counts[verdict] = len(chosen)
    if not positives:
        raise RuntimeError("strict pilot selection produced no positive rows")

    selected.sort(
        key=lambda item: _priority(
            args.data_seed,
            str(item[0]["iid"]),
            "final",
        )
    )
    output_rows: list[dict[str, Any]] = []
    with _atomic_text_writer(output_path) as handle:
        for row, signature, observation, result in selected:
            output_row = dict(row)
            legacy_split = {
                "split": output_row.pop("split", None),
                "group_id": output_row.pop("group_id", None),
                "split_provenance": output_row.pop(
                    "split_provenance",
                    None,
                ),
            }
            verdict = str(result["verdict"])
            label_class = (
                "positive" if verdict in POSITIVE_VERDICTS else "negative"
            )
            output_row["r5_legacy_split_audit"] = legacy_split
            output_row["r5_pilot_label"] = {
                "schema_version": R5_PILOT_SCHEMA,
                "profile": R5_PILOT_PROFILE,
                "class": label_class,
                "negative_type": (
                    None if label_class == "positive" else verdict
                ),
                "action_signature": signature,
                "human_approved": False,
                "production_eligible": False,
                "legacy_result_digest_missing": (
                    "result_digest"
                    not in (
                        (output_row.get("qwen_evidence") or {}).get("visual")
                        or {}
                    )
                ),
                "source_manifest_sha256": source_sha256,
                "observation_digest": _object_digest(observation),
                "result_object_digest": _object_digest(result),
            }
            output_rows.append(output_row)
            handle.write(
                json.dumps(output_row, ensure_ascii=False) + "\n"
            )

    output_sha256 = _file_digest(output_path)
    summary = {
        "schema_version": R5_PILOT_SCHEMA,
        "profile": R5_PILOT_PROFILE,
        "status": "pseudo_label_r5_lite_only",
        "production_eligible": False,
        "source_manifest": str(input_path),
        "source_manifest_sha256": source_sha256,
        "source_rows": scanned,
        "selected_rows": len(output_rows),
        "positive_rows": len(positives),
        "negative_rows": int(sum(selected_negative_counts.values())),
        "negative_type_counts": dict(sorted(selected_negative_counts.items())),
        "data_seed": int(args.data_seed),
        "negative_caps": negative_caps,
        "output": str(output_path),
        "output_sha256": output_sha256,
        "selected_iid_digest": _canonical_digest(
            sorted(str(row["iid"]) for row in output_rows)
        ),
        "limitations": [
            "No human verdict or production eligibility is asserted.",
            "The legacy Qwen run predates post-generation result_digest.",
            "The downstream near-pHash split is diagnostic, not a production "
            "source-visual-cluster split.",
            "Negative rows are false-activation audits, not action-delta positives.",
        ],
    }
    with _atomic_text_writer(summary_path) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        "[motive-r5-pilot] "
        f"source={scanned} selected={len(output_rows)} "
        f"positive={len(positives)} negatives={dict(selected_negative_counts)} "
        f"output={output_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a strict legacy-Qwen R5-lite pseudo pilot.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--data-seed", type=int, default=260127)
    parser.add_argument("--positive-limit", type=int, default=0)
    parser.add_argument(
        "--static-negatives",
        type=int,
        default=DEFAULT_NEGATIVE_CAPS["static"],
    )
    parser.add_argument(
        "--mismatch-negatives",
        type=int,
        default=DEFAULT_NEGATIVE_CAPS["instruction_mismatch"],
    )
    parser.add_argument(
        "--endpoint-negatives",
        type=int,
        default=DEFAULT_NEGATIVE_CAPS["endpoint_only"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return prepare(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
