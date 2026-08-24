from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_wan22_sources.py"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object_sha(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha(canonical)


def test_materializes_real_copy_and_refuses_mismatched_existing_file(
    tmp_path: Path,
) -> None:
    source_payload = b"source-video-payload"
    preview_payload = b"generated-video-payload"
    source = tmp_path / "upstream" / "source.mp4"
    source.parent.mkdir()
    source.write_bytes(source_payload)

    manifest = tmp_path / "generation_manifest.jsonl"
    row = {
        "action_change_substantive": "yes",
        "edit_instruction": "Make the person wave.",
        "group_id": "group001",
        "iid": "sample001",
        "resolved_source_video": str(source),
        "source_video_sha256": _sha(source_payload),
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output_root = tmp_path / "full"
    sample_dir = output_root / "samples" / "sample001"
    sample_dir.mkdir(parents=True)
    preview = sample_dir / "preview.mp4"
    preview.write_bytes(preview_payload)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result = {
        "action_change_substantive": "yes",
        "group_id": "group001",
        "iid": "sample001",
        "inputs": {
            "source_video_resolved_path": str(source),
            "source_video_sha256": _sha(source_payload),
        },
        "manifest_row_digest": _object_sha(row),
        "manifest_sha256": manifest_sha,
        "outputs": {
            "preview_mp4": "preview.mp4",
            "preview_mp4_sha256": _sha(preview_payload),
        },
        "prompt": {
            "field": "edit_instruction",
            "sha256": _sha(row["edit_instruction"].encode("utf-8")),
            "text": row["edit_instruction"],
        },
        "sample_index": 0,
        "schema_version": "motive-wan22-i2v-sample-v1",
    }
    result["result_digest"] = _object_sha(result)
    (sample_dir / "result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
            "--expected-manifest-sha256",
            manifest_sha,
            "--require-all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    copied = sample_dir / "source.mp4"
    assert summary["copied"] == 1
    assert copied.read_bytes() == source_payload
    assert not copied.is_symlink()
    assert os.stat(copied).st_ino != os.stat(source).st_ino
    assert (sample_dir / "source_copy.json").is_file()
    instruction_file = sample_dir / "edit_instruction.txt"
    assert instruction_file.read_text(encoding="utf-8") == "Make the person wave.\n"
    pair = json.loads((output_root / "source_target_pairs.jsonl").read_text())
    assert pair["edit_instruction_file"] == str(instruction_file)
    assert pair["source_video"] == str(copied)
    assert pair["generated_video"] == str(preview)

    copied.write_bytes(b"tampered")
    failed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert "refusing to overwrite mismatched source copy" in failed.stderr
    assert copied.read_bytes() == b"tampered"
