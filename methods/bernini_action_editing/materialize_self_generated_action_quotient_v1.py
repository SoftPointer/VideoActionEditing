#!/usr/bin/env python3
"""Create a source-only training manifest for the four-program quotient canary.

The historical reward parquet contains ``[source, selected_target]``.  This
one-shot isolator indexes element zero and publishes only that byte string.
The selected target is never decoded, hashed, copied, named in the output, or
made reachable by the training process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-self-generated-action-quotient-source-manifest-v1"
IIDS = (
    "7b88a1ca1f804f41",
    "841b5e0080a1441d",
    "a35b590961d24694",
    "a66e6818e4144928",
)
BRANCHES = ("action", "noop", "camera_only", "appearance_only")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_messages(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 3:
        raise RuntimeError("source row messages differ")
    if [item.get("type") for item in value] != ["video", "text", "video_gen"]:
        raise RuntimeError("source row message roles differ")
    return value


def caption(anchor_root: Path, iid: str, branch: str) -> tuple[str, Path, str]:
    directory = anchor_root / f"pair5-t2v-core4-v2-{iid}-{branch}"
    receipt = directory / "pair-v5-t2v-calibration-receipt.json"
    value = json.loads(receipt.read_text(encoding="utf-8"))
    text = value.get("candidate", {}).get("full_t2v_caption")
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise RuntimeError(f"missing {branch} caption for {iid}")
    return text.strip(), receipt, file_sha256(receipt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq
    from safetensors import safe_open

    parquet_dir = Path(args.parquet_dir).resolve(strict=True)
    anchor_root = Path(args.anchor_root).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    rows = []
    for iid in IIDS:
        candidates = (parquet_dir / f"{iid}.parquet", parquet_dir / "shards" / f"{iid}.parquet")
        matches = [path.resolve(strict=True) for path in candidates if path.is_file()]
        if len(set(matches)) != 1:
            raise RuntimeError(f"source parquet absent or ambiguous: {iid}: {matches}")
        parquet = matches[0]
        table = pq.read_table(parquet, columns=["iid", "inputs", "video_vae_latents"])
        if table.num_rows != 1:
            raise RuntimeError(f"source parquet must have one row: {iid}")
        row = table.to_pylist()[0]
        if str(row.get("iid")) != iid:
            raise RuntimeError(f"IID differs in parquet: {iid}")
        messages = parse_messages(row.get("inputs"))
        instruction = messages[1].get("text")
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError(f"instruction absent: {iid}")
        latents = row.get("video_vae_latents")
        if not isinstance(latents, list) or len(latents) != 2:
            raise RuntimeError(f"latent role container differs: {iid}")
        # Important: element 1 is intentionally never decoded, hashed, copied,
        # inspected, or mentioned in the released source-only manifest.
        source_blob = bytes(latents[0])
        source_name = f"{iid}.source-posterior.pt"
        write_exclusive(output / source_name, source_blob)

        captions = {}
        receipt_bindings = {}
        for branch in BRANCHES:
            text, receipt, receipt_sha = caption(anchor_root, iid, branch)
            captions[branch] = text
            receipt_bindings[branch] = {"path": str(receipt), "sha256": receipt_sha}
        action_dir = anchor_root / f"pair5-t2v-core4-v2-{iid}-action"
        latent = (action_dir / "t2v.normalized-clean-latent.safetensors").resolve(strict=True)
        video = (action_dir / "t2v.mp4").resolve(strict=True)
        with safe_open(str(latent), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            tensor = handle.get_tensor("normalized_clean_latent")
        if keys != ["normalized_clean_latent"] or tuple(tensor.shape[:3]) != (1, 16, 21):
            raise RuntimeError(f"action anchor latent differs: {iid}")
        rows.append(
            {
                "iid": iid,
                "instruction": instruction.strip(),
                "source_posterior": {
                    "path": str((output / source_name).resolve()),
                    "sha256": sha256_bytes(source_blob),
                },
                "source_parquet_provenance": {
                    "path": str(parquet),
                    "sha256": file_sha256(parquet),
                    "selected_role_index": 0,
                },
                "action_anchor": {
                    "latent_path": str(latent),
                    "latent_sha256": file_sha256(latent),
                    "video_path": str(video),
                    "video_sha256": file_sha256(video),
                },
                "teacher_captions": captions,
                "caption_receipts": receipt_bindings,
            }
        )
    manifest = {
        "schema_version": SCHEMA,
        "rows": rows,
        "training_process_can_reach_historical_selected_target": False,
        "self_generated_anchor_role": "detached_action_phase_representation_only",
        "self_generated_anchor_is_rv2v_supervision_target": False,
    }
    manifest["manifest_digest"] = sha256_bytes(canonical(manifest))
    write_exclusive(output / "manifest.json", canonical(manifest) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
