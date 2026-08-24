#!/usr/bin/env python3
"""Build the exact644 source/action-anchor engineering training manifest.

The frozen VAE dataset stores exactly two posterior blobs per row.  This
materializer binds index 0 as the source identity anchor and index 1 as the
self-generated action anchor.  It does not call either blob paired ground
truth and it does not authorize a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-full644-self-generated-action-anchor-manifest-v1"
INDEX_SCHEMA = "bernini-r-action-vae-index-row-v2"
ROW_SCHEMA = "bernini-r-action-vae-row-v2"
RECEIPT_SCHEMA = "bernini-r-action-vae-sample-receipt-v2"
EXPECTED_ROWS = 644
AUTHORIZATION = "user_explicit_self_generated_action_anchor_training_20260818"
NOOP_INSTRUCTION = (
    "Keep the source video unchanged. Preserve every identity, appearance, "
    "pose, object, background detail, camera parameter, lighting condition, "
    "and motion exactly as in the source video."
)
HEX64 = re.compile(r"[0-9a-f]{64}")


class ManifestError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ManifestError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def strict_json(value: Any, *, label: str) -> Any:
    if not isinstance(value, str):
        fail(f"{label} must be JSON text")

    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                fail(f"{label} contains a duplicate key")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=reject_duplicate)


def instruction_from_inputs(value: Any) -> str:
    messages = strict_json(value, label="inputs")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [item.get("type") for item in messages]
        != ["video", "text", "video_gen"]
        or [item.get("has_loss") for item in messages] != [0, 0, 1]
    ):
        fail("renderer input role closure differs")
    instruction = messages[1].get("text")
    if not isinstance(instruction, str) or not instruction.strip() or "\x00" in instruction:
        fail("edit instruction differs")
    return instruction.strip()


def load_json(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        fail(f"{label} SHA-256 differs")
    value = json.loads(raw)
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", required=True)
    parser.add_argument("--dataset-index-sha256", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--dataset-summary-sha256", required=True)
    parser.add_argument("--authorization-label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.authorization_label != AUTHORIZATION:
        fail("explicit engineering authorization label differs")
    for label, value in (
        ("dataset index", args.dataset_index_sha256),
        ("dataset summary", args.dataset_summary_sha256),
    ):
        if not HEX64.fullmatch(value):
            fail(f"{label} SHA-256 differs")

    import pyarrow.parquet as pq

    index_path = Path(args.dataset_index).resolve(strict=True)
    summary_path = Path(args.dataset_summary).resolve(strict=True)
    if file_sha256(index_path) != args.dataset_index_sha256:
        fail("dataset index byte identity differs")
    summary = load_json(
        summary_path,
        expected_sha256=args.dataset_summary_sha256,
        label="dataset summary",
    )
    if summary.get("expected_sample_count") != EXPECTED_ROWS:
        fail("dataset summary exact644 count differs")

    raw_lines = index_path.read_text(encoding="utf-8").splitlines()
    index_rows = [json.loads(line) for line in raw_lines]
    if len(index_rows) != EXPECTED_ROWS:
        fail("dataset index must contain exact644 rows")
    if any(row.get("schema_version") != INDEX_SCHEMA for row in index_rows):
        fail("dataset index row schema differs")
    iids = [row.get("iid") for row in index_rows]
    if any(not isinstance(iid, str) or not iid for iid in iids):
        fail("dataset IID differs")
    if iids != sorted(iids) or len(set(iids)) != EXPECTED_ROWS:
        fail("dataset IID order/uniqueness differs")

    manifest_rows: list[dict[str, Any]] = []
    strict_count = 0
    for index_row in index_rows:
        iid = index_row["iid"]
        parquet_path = Path(index_row["parquet_path"]).resolve(strict=True)
        receipt_path = Path(index_row["sample_receipt_path"]).resolve(strict=True)
        parquet_raw = parquet_path.read_bytes()
        parquet_sha = sha256_bytes(parquet_raw)
        if parquet_sha != index_row.get("parquet_sha256"):
            fail(f"parquet SHA-256 differs: {iid}")
        receipt = load_json(
            receipt_path,
            expected_sha256=index_row["sample_receipt_sha256"],
            label=f"sample receipt {iid}",
        )
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("iid") != iid
            or receipt.get("parquet_sha256") != parquet_sha
            or receipt.get("complete") is not True
            or receipt.get("frame_count") != 81
            or receipt.get("fps") != 25.0
            or receipt.get("latent_frame_count") != 21
        ):
            fail(f"sample receipt closure differs: {iid}")
        import pyarrow as pa

        table = pq.read_table(
            pa.BufferReader(parquet_raw),
            columns=[
                "schema_version", "inputs", "videos", "iid", "group_id",
                "family", "edit_instruction_sha256", "source_video_path",
                "source_video_sha256", "target_video_path",
                "target_video_sha256", "strict_selection_gates_all_true",
                "preview_only", "training_authorized", "training_use_forbidden",
                "production_claim_forbidden", "video_vae_latents",
                "materialized_row_digest",
            ],
        )
        if table.num_rows != 1:
            fail(f"parquet row count differs: {iid}")
        row = table.to_pylist()[0]
        if row.get("schema_version") != ROW_SCHEMA or row.get("iid") != iid:
            fail(f"renderer row identity differs: {iid}")
        instruction = instruction_from_inputs(row.get("inputs"))
        if sha256_bytes(instruction.encode("utf-8")) != row.get("edit_instruction_sha256"):
            fail(f"instruction SHA-256 differs: {iid}")
        videos = row.get("videos")
        if (
            not isinstance(videos, list)
            or len(videos) != 2
            or videos[0].get("video_path") != row.get("source_video_path")
            or videos[1].get("video_path") != row.get("target_video_path")
        ):
            fail(f"source/action-anchor video role closure differs: {iid}")
        latents = row.get("video_vae_latents")
        if not isinstance(latents, list) or len(latents) != 2:
            fail(f"posterior role container differs: {iid}")
        source_blob = bytes(latents[0])
        anchor_blob = bytes(latents[1])
        source_sha = sha256_bytes(source_blob)
        anchor_sha = sha256_bytes(anchor_blob)
        if (
            source_sha != receipt.get("source_latent_blob_sha256")
            or anchor_sha != receipt.get("target_latent_blob_sha256")
        ):
            fail(f"posterior blob receipt join differs: {iid}")
        strict = row.get("strict_selection_gates_all_true")
        if type(strict) is not bool:
            fail(f"strict-selection flag type differs: {iid}")
        strict_count += int(strict)
        manifest_rows.append(
            {
                "iid": iid,
                "group_id": row.get("group_id"),
                "family": row.get("family"),
                "instruction": instruction,
                "instruction_sha256": row["edit_instruction_sha256"],
                "noop_instruction": NOOP_INSTRUCTION,
                "posterior_pair": {
                    "parquet_path": str(parquet_path),
                    "parquet_sha256": parquet_sha,
                    "source_role_index": 0,
                    "source_blob_sha256": source_sha,
                    "action_anchor_role_index": 1,
                    "action_anchor_blob_sha256": anchor_sha,
                },
                "media_binding": {
                    "source_video_path": row["source_video_path"],
                    "source_video_sha256": row["source_video_sha256"],
                    "action_anchor_video_path": row["target_video_path"],
                    "action_anchor_video_sha256": row["target_video_sha256"],
                    "frame_count": 81,
                    "fps": 25.0,
                },
                "upstream_sample_receipt": {
                    "path": str(receipt_path),
                    "sha256": index_row["sample_receipt_sha256"],
                },
                "upstream_materialized_row_digest": row["materialized_row_digest"],
                "strict_selection_gates_all_true": strict,
            }
        )
    if strict_count != 359:
        fail("strict/broad partition differs")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "authorization_label": AUTHORIZATION,
        "dataset_index": {"path": str(index_path), "sha256": args.dataset_index_sha256},
        "dataset_summary": {"path": str(summary_path), "sha256": args.dataset_summary_sha256},
        "row_count": EXPECTED_ROWS,
        "strict_row_count": strict_count,
        "broad_row_count": EXPECTED_ROWS - strict_count,
        "rows": manifest_rows,
        "source_anchor_role": "identity_appearance_background_camera_and_non_target_preservation",
        "self_generated_action_anchor_role": "dense_action_trajectory_supervision",
        "paired_ground_truth_claimed": False,
        "qwen_or_other_verifier_controls_optimizer_admission": False,
        "optimizer_schedule": "exact644_unique_rows_once",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "experimental_training": True,
    }
    manifest["manifest_digest"] = sha256_bytes(canonical(manifest))
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        fail(f"refusing to overwrite output: {output}")
    write_create_only(output, canonical(manifest) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
