#!/usr/bin/env python3
"""Bind a diagnostic Qwen audit to a bitwise-equivalent fresh SAIC rerun.

This receipt is deliberately narrow.  It can show that candidate specifications
and generated artifacts are byte-identical across two exact60 runs and therefore
that an observation-only audit saw the same pixels.  It never grants event,
human-review, selection, training, optimizer, or scientific authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, NoReturn


SCHEMA = "saic-t2v-cross-run-equivalence-v2"
RECORD_SCHEMA = "saic-t2v-branch-semantics-qwen-record-v6"
SUMMARY_SCHEMA = "saic-t2v-branch-semantics-qwen-summary-v6"
GENERATION_RECEIPT = "saic-event-topup-generation-receipt.json"
EXPECTED_COUNT = 60
FALSE_AUTHORITY = {
    "event_verified": False,
    "human_review": False,
    "data_selection": False,
    "training": False,
    "optimizer": False,
    "scientific_claim": False,
}


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def plain_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        die(f"{label} must resolve to a plain file")
    return resolved


def load_attempts(root: Path) -> dict[str, Mapping[str, Any]]:
    root = root.resolve(strict=True)
    rows: dict[str, Mapping[str, Any]] = {}
    for receipt_path in sorted(root.glob(f"*/{GENERATION_RECEIPT}")):
        receipt_path = plain_file(receipt_path, label="generation receipt")
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        candidate = receipt.get("candidate")
        if not isinstance(candidate, dict):
            die(f"candidate missing: {receipt_path}")
        candidate_id = candidate.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in rows
            or receipt_path.parent.name != candidate_id
            or candidate.get("event_verified") is not False
        ):
            die(f"candidate boundary differs: {receipt_path}")
        artifact_root = receipt.get("artifacts", {})
        artifacts = (
            artifact_root.get("mp4", {})
            if isinstance(artifact_root, dict) else {}
        )
        declared = {
            "mp4": artifacts,
            "normalized_clean_latent": artifacts.get(
                "normalized_clean_latent", {}
            ),
            "official_initial_gaussian": artifact_root.get(
                "official_initial_gaussian", {}
            ),
        }
        actual: dict[str, Mapping[str, str]] = {}
        for kind, binding in declared.items():
            raw_path = binding.get("path") if isinstance(binding, dict) else None
            if not isinstance(raw_path, str):
                die(f"{kind} path missing: {receipt_path}")
            artifact = plain_file(Path(raw_path), label=kind)
            if artifact.parent != receipt_path.parent:
                die(f"{kind} escapes attempt directory: {receipt_path}")
            digest = file_sha256(artifact)
            if binding.get("sha256") != digest:
                die(f"{kind} declared digest differs: {receipt_path}")
            actual[kind] = {
                "path": str(artifact),
                "sha256": digest,
            }
            if kind == "official_initial_gaussian":
                tensor_digest = binding.get("tensor_value_sha256")
                if (
                    not isinstance(tensor_digest, str)
                    or len(tensor_digest) != 64
                    or binding.get("raw_value_sha256") != tensor_digest
                    or binding.get("all_rank_identity", {})
                    .get("identity", {}).get("raw_storage_sha256")
                    != tensor_digest
                ):
                    die(f"{kind} tensor digest differs: {receipt_path}")
                actual[kind]["tensor_value_sha256"] = tensor_digest
        rows[candidate_id] = {
            "candidate": candidate,
            "generation_receipt_path": str(receipt_path),
            "generation_receipt_sha256": file_sha256(receipt_path),
            "artifacts": actual,
        }
    if len(rows) != EXPECTED_COUNT:
        die(f"attempt count differs: {len(rows)}")
    return rows


def load_qwen(
    records_path: Path, summary_path: Path
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    records_path = plain_file(records_path, label="Qwen records")
    summary_path = plain_file(summary_path, label="Qwen summary")
    rows: dict[str, Mapping[str, Any]] = {}
    for line in records_path.read_text(encoding="ascii").splitlines():
        row = json.loads(line)
        candidate_id = row.get("candidate_id")
        if (
            row.get("schema_version") != RECORD_SCHEMA
            or not isinstance(candidate_id, str)
            or candidate_id in rows
            or row.get("authority") != {
                "human_review": False,
                "data_selection": False,
                "training": False,
                "optimizer": False,
                "scientific_claim": False,
            }
        ):
            die("Qwen record boundary differs")
        rows[candidate_id] = row
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if (
        len(rows) != EXPECTED_COUNT
        or summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("record_count") != EXPECTED_COUNT
        or summary.get("output_jsonl_sha256") != file_sha256(records_path)
        or summary.get("authority") != {
            "human_review": False,
            "data_selection": False,
            "training": False,
            "optimizer": False,
            "scientific_claim": False,
        }
    ):
        die("Qwen summary boundary differs")
    return rows, {
        "records_path": str(records_path),
        "records_sha256": file_sha256(records_path),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "summary_receipt_digest": summary.get("receipt_digest"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-attempts", type=Path, required=True)
    parser.add_argument("--fresh-attempts", type=Path, required=True)
    parser.add_argument("--qwen-records", type=Path, required=True)
    parser.add_argument("--qwen-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        die(f"output already exists: {args.output}")
    reference = load_attempts(args.reference_attempts)
    fresh = load_attempts(args.fresh_attempts)
    qwen, qwen_binding = load_qwen(args.qwen_records, args.qwen_summary)
    if set(reference) != set(fresh) or set(reference) != set(qwen):
        die("candidate set differs")
    comparisons = []
    for candidate_id in sorted(reference):
        old = reference[candidate_id]
        new = fresh[candidate_id]
        artifact_equal = {
            kind: old["artifacts"][kind]["sha256"]
            == new["artifacts"][kind]["sha256"]
            for kind in sorted(old["artifacts"])
        }
        gaussian_tensor_equal = (
            old["artifacts"]["official_initial_gaussian"][
                "tensor_value_sha256"
            ]
            == new["artifacts"]["official_initial_gaussian"][
                "tensor_value_sha256"
            ]
        )
        comparisons.append({
            "candidate_id": candidate_id,
            "candidate_spec_equal": old["candidate"] == new["candidate"],
            "artifact_bytes_equal": artifact_equal,
            "official_initial_gaussian_tensor_value_equal": (
                gaussian_tensor_equal
            ),
            "reference": old,
            "fresh": new,
            "qwen_record_receipt_digest": qwen[candidate_id]["receipt_digest"],
            "qwen_video_sha256_matches_reference": (
                qwen[candidate_id].get("video_sha256")
                == old["artifacts"]["mp4"]["sha256"]
            ),
        })
    all_specs = all(row["candidate_spec_equal"] for row in comparisons)
    equality = {
        kind: all(row["artifact_bytes_equal"][kind] for row in comparisons)
        for kind in ("mp4", "normalized_clean_latent", "official_initial_gaussian")
    }
    gaussian_tensor_values_equal = all(
        row["official_initial_gaussian_tensor_value_equal"]
        for row in comparisons
    )
    qwen_complete = all(
        row["qwen_video_sha256_matches_reference"] for row in comparisons
    )
    unsigned = {
        "schema_version": SCHEMA,
        "status": "diagnostic_cross_run_equivalence_no_authority",
        "candidate_count": len(comparisons),
        "candidate_specs_all_equal": all_specs,
        "artifact_kinds_all_byte_equal": equality,
        "official_initial_gaussian_tensor_values_all_equal": (
            gaussian_tensor_values_equal
        ),
        "qwen_observation_binding_complete": qwen_complete,
        "observation_reuse_by_bitwise_video_identity": (
            all_specs and equality["mp4"] and qwen_complete
        ),
        "qwen_binding": qwen_binding,
        "comparisons": comparisons,
        "authority": FALSE_AUTHORITY,
    }
    output = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "output": str(args.output),
        "receipt_digest": output["receipt_digest"],
        "candidate_specs_all_equal": all_specs,
        "artifact_kinds_all_byte_equal": equality,
        "official_initial_gaussian_tensor_values_all_equal": (
            gaussian_tensor_values_equal
        ),
        "observation_reuse_by_bitwise_video_identity": output[
            "observation_reuse_by_bitwise_video_identity"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
