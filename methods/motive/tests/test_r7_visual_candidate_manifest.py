from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from motive.qwen_filter import _object_digest
from motive.r7_build_expansion_manifest import build_expansion_manifest
from motive.r7_visual_candidate_manifest import (
    CANDIDATE_ROW_FIELDS,
    ROW_SCHEMA,
    build_visual_candidate_manifest,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    ).encode()


def _row(index: int, *, kind: str) -> dict[str, object]:
    iid = f"iid-{index:05d}"
    if kind in {"pseudo_negative", "audit_only"}:
        verdict = "static"
        confidence = "high"
        target_motion = "none"
        target_action = "no visible action"
        edit_effect = "none"
        signature = "unknown"
    elif kind == "review":
        verdict = "valid_action"
        confidence = "low"
        target_motion = "clear"
        target_action = "person walks right"
        edit_effect = "started"
        signature = "walk_right"
    elif kind == "pseudo_positive":
        verdict = "valid_action"
        confidence = "high"
        target_motion = "clear"
        target_action = "person walks right"
        edit_effect = "started"
        signature = "walk_right"
    else:
        raise ValueError(kind)
    observation: dict[str, object] = {
        "schema_version": "qwen-motion-observation-v2",
        "source_action": "no visible action",
        "target_action": target_action,
        "source_actor_motion": "none",
        "target_actor_motion": target_motion,
        "camera_dominance": "low",
        "background_dominance": "low",
        "artifact_level": "low",
        "preservation_quality": "acceptable",
        "temporal_evidence": ["Ordered frames show the target actor."],
        "uncertainty_codes": [],
    }
    result: dict[str, object] = {
        "schema_version": "qwen-motion-judge-v4",
        "verdict": verdict,
        "edit_effect": edit_effect,
        "action_signature": signature,
        "reason_codes": [f"judge_{verdict}"],
        "uncertainty_codes": [],
        "confidence": confidence,
    }
    observation_digest = _object_digest(observation)
    result_digest = _object_digest(result)
    result_source = (
        "original_sanitized" if kind == "audit_only" else "original"
    )
    repairs: list[dict[str, object]] = []
    if kind == "audit_only":
        repairs = [
            {
                "attempt": 0,
                "status": "ok",
                "repair_generation_called": False,
                "authoritative_context_digest": observation_digest,
                "repair_sanitizations": [
                    {
                        "action": (
                            "downgrade_instruction_mismatch_to_static"
                        )
                    }
                ],
            }
        ]
    input_digest = _sha(f"input-{iid}".encode())
    visual: dict[str, object] = {
        "iid": iid,
        "input_digest": input_digest,
        "mode": "visual",
        "status": "ok",
        "observation": observation,
        "observation_digest": observation_digest,
        "observation_repairs": [],
        "observation_validated_from": "original",
        "alignment_repairs": repairs,
        "result": result,
        "result_digest": result_digest,
        "result_validated_from": result_source,
        "visual_input_digest": _sha(f"visual-{iid}".encode()),
        "execution_shard_index": index % 8,
        "execution_shard_count": 8,
        "execution_manifest": f"/synthetic/shard-{index % 8:03d}.jsonl",
        "execution_manifest_sha256": _sha(
            f"manifest-{index % 8}".encode()
        ),
        "run_config_digest": "a" * 64,
        "config_digest": _sha(f"config-{index % 8}".encode()),
        "implementation_digest": "b" * 64,
        "model_revision": "c" * 40,
        "transformers_version": "4.51.3",
    }
    return {
        "schema_version": "goku-action-pair-v1",
        "iid": iid,
        "input_digest": input_digest,
        "prompt": f"make person perform action {index}",
        "src_video": f"videos/{iid}/source.mp4",
        "tgt_video": f"videos/{iid}/edited.mp4",
        "auto_rule": {
            "action_families": ["walk"],
            "actors": ["person"],
            "label": "temporal_action",
            "score": 0.8,
            "tier": "high",
        },
        "r7_expansion_selection": {
            "schema_version": "motive-r7-expansion-selection-v1",
            "primary_family": "walk",
            "split_assigned": False,
            "legacy_split_quarantine": {
                "present": False,
                "canonical_sha256": None,
            },
        },
        "qwen_evidence": {"visual": visual},
    }


def _interleaved_kinds(counts: dict[str, int]) -> list[str]:
    remaining = dict(counts)
    result: list[str] = []
    while any(remaining.values()):
        for kind in (
            "review",
            "pseudo_positive",
            "audit_only",
            "pseudo_negative",
        ):
            if remaining[kind]:
                result.append(kind)
                remaining[kind] -= 1
    return result


def _make_manifest(
    root: Path,
    counts: dict[str, int],
) -> tuple[Path, list[dict[str, object]], list[str]]:
    kinds = _interleaved_kinds(counts)
    rows = [_row(index, kind=kind) for index, kind in enumerate(kinds)]
    fused = root / "fused.jsonl"
    fused.write_bytes(_jsonl_bytes(rows))
    manifest = root / "manifest_v2"
    build_expansion_manifest(input_path=fused, output_dir=manifest)
    return manifest, rows, kinds


def _rechain_manifest(manifest: Path) -> None:
    data_names = ("positives.jsonl", "negatives.jsonl", "review.jsonl")
    summary_path = manifest / "summary.json"
    summary = json.loads(summary_path.read_text())
    for name in data_names:
        raw = (manifest / name).read_bytes()
        summary["outputs"][name]["sha256"] = _sha(raw)
        summary["outputs"][name]["rows"] = len(raw.splitlines())
    summary_raw = _json_bytes(summary)
    summary_path.write_bytes(summary_raw)
    output_sha = {
        name: _sha((manifest / name).read_bytes()) for name in data_names
    }
    output_sha["summary.json"] = _sha(summary_raw)
    done_path = manifest / "done.json"
    done = json.loads(done_path.read_text())
    done["output_sha256"] = dict(sorted(output_sha.items()))
    done["output_rows"] = {
        name: len((manifest / name).read_bytes().splitlines())
        for name in sorted(data_names)
    }
    done["artifact_digest"] = _object_digest(done["output_sha256"])
    done_path.write_bytes(_json_bytes(done))


class R7VisualCandidateManifestTests(unittest.TestCase):
    def test_real_scale_analogue_selects_only_3167_candidates(self) -> None:
        counts = {
            "pseudo_positive": 947,
            "pseudo_negative": 2220,
            "audit_only": 3610,
            "review": 6223,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, source_rows, kinds = _make_manifest(root, counts)
            output = root / "visual_candidates_v1"
            summary = build_visual_candidate_manifest(
                manifest_dir=manifest,
                output_dir=output,
            )

            self.assertEqual(
                summary["cohort_counts"],
                {
                    "pseudo_negative": 2220,
                    "pseudo_positive": 947,
                },
            )
            self.assertEqual(
                summary["excluded_counts"],
                {"audit_only_negative": 3610, "review": 6223},
            )
            self.assertEqual(summary["output"]["rows"], 3167)
            self.assertTrue(summary["conservation"]["exact"])
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode),
                0o555,
            )
            for artifact in output.iterdir():
                self.assertEqual(
                    stat.S_IMODE(artifact.stat().st_mode),
                    0o444,
                )

            candidates = [
                json.loads(line)
                for line in (output / "candidates.jsonl").read_text().splitlines()
            ]
            expected_iids = [
                source_rows[index]["iid"]
                for index, kind in enumerate(kinds)
                if kind in {"pseudo_positive", "pseudo_negative"}
            ]
            excluded_iids = {
                source_rows[index]["iid"]
                for index, kind in enumerate(kinds)
                if kind in {"audit_only", "review"}
            }
            self.assertEqual(
                [row["iid"] for row in candidates],
                expected_iids,
            )
            self.assertTrue(
                excluded_iids.isdisjoint(row["iid"] for row in candidates)
            )
            self.assertEqual(
                Counter(row["cohort"] for row in candidates),
                Counter(
                    {"pseudo_positive": 947, "pseudo_negative": 2220}
                ),
            )
            source_digest = json.loads(
                (manifest / "done.json").read_text()
            )["artifact_digest"]
            for row in candidates:
                self.assertEqual(set(row), set(CANDIDATE_ROW_FIELDS))
                self.assertEqual(row["schema_version"], ROW_SCHEMA)
                self.assertEqual(
                    row["source_artifact_digest"],
                    source_digest,
                )
                self.assertFalse(row["split_assigned"])
                self.assertFalse(row["human_label"])
                self.assertFalse(row["training_eligible"])

            resumed = build_visual_candidate_manifest(
                manifest_dir=manifest,
                output_dir=output,
                resume=True,
            )
            self.assertTrue(resumed["resume_verified"])
            with self.assertRaises(FileExistsError):
                build_visual_candidate_manifest(
                    manifest_dir=manifest,
                    output_dir=output,
                )

    def test_source_hash_tamper_and_output_resume_tamper_fail_closed(
        self,
    ) -> None:
        counts = {
            "pseudo_positive": 2,
            "pseudo_negative": 2,
            "audit_only": 1,
            "review": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = _make_manifest(root, counts)
            positive = manifest / "positives.jsonl"
            original = positive.read_bytes()
            positive.write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                build_visual_candidate_manifest(
                    manifest_dir=manifest,
                    output_dir=root / "rejected",
                )
            positive.write_bytes(original)

            output = root / "visual_candidates_v1"
            build_visual_candidate_manifest(
                manifest_dir=manifest,
                output_dir=output,
            )
            candidate_path = output / "candidates.jsonl"
            os.chmod(candidate_path, 0o644)
            candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
            os.chmod(candidate_path, 0o444)
            with self.assertRaises(RuntimeError):
                build_visual_candidate_manifest(
                    manifest_dir=manifest,
                    output_dir=output,
                    resume=True,
                )

    def test_rechained_source_bucket_reordering_is_rejected(self) -> None:
        counts = {
            "pseudo_positive": 3,
            "pseudo_negative": 3,
            "audit_only": 1,
            "review": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, source_rows, kinds = _make_manifest(root, counts)

            clean_output = root / "clean_candidates"
            build_visual_candidate_manifest(
                manifest_dir=manifest,
                output_dir=clean_output,
            )
            expected_iids = [
                source_rows[index]["iid"]
                for index, kind in enumerate(kinds)
                if kind in {"pseudo_positive", "pseudo_negative"}
            ]
            actual_iids = [
                json.loads(line)["iid"]
                for line in (
                    clean_output / "candidates.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(actual_iids, expected_iids)

            positive_path = manifest / "positives.jsonl"
            positive_lines = positive_path.read_bytes().splitlines(
                keepends=True
            )
            self.assertGreaterEqual(len(positive_lines), 2)
            positive_lines[0], positive_lines[1] = (
                positive_lines[1],
                positive_lines[0],
            )
            positive_path.write_bytes(b"".join(positive_lines))
            _rechain_manifest(manifest)
            with self.assertRaisesRegex(
                ValueError,
                "source-line ordering",
            ):
                build_visual_candidate_manifest(
                    manifest_dir=manifest,
                    output_dir=root / "reordered_rejected",
                )


if __name__ == "__main__":
    unittest.main()
