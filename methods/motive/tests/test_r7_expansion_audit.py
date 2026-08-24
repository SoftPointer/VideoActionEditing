from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from motive.qwen_filter import _object_digest
from motive.r7_build_expansion_manifest import build_expansion_manifest
from motive.r7_expansion_audit import audit_expansion


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    ).encode()


def _observation(
    *,
    target_motion: str = "clear",
    confidence_problem: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "qwen-motion-observation-v2",
        "source_action": "no visible action",
        "target_action": (
            "person walks right"
            if target_motion != "none"
            else "no visible action"
        ),
        "source_actor_motion": "none",
        "target_actor_motion": target_motion,
        "camera_dominance": "low",
        "background_dominance": "low",
        "artifact_level": "low",
        "preservation_quality": "acceptable",
        "temporal_evidence": ["Ordered frames show the target actor."],
        "uncertainty_codes": (
            ["occlusion"] if confidence_problem else []
        ),
    }


def _result(
    verdict: str,
    *,
    confidence: str,
    signature: str = "Walk_Right",
) -> dict[str, object]:
    if verdict == "valid_action":
        edit_effect = "started"
    elif verdict == "instruction_mismatch":
        edit_effect = "changed_action"
        signature = "unknown"
    elif verdict == "uncertain":
        edit_effect = "unclear"
        signature = "unknown"
    else:
        edit_effect = "none"
        signature = "unknown"
    return {
        "schema_version": "qwen-motion-judge-v4",
        "verdict": verdict,
        "edit_effect": edit_effect,
        "action_signature": signature,
        "reason_codes": [f"judge_{verdict}"],
        "uncertainty_codes": (
            ["ambiguous"] if verdict == "uncertain" else []
        ),
        "confidence": confidence,
    }


def _row(
    index: int,
    *,
    verdict: str,
    target_motion: str = "clear",
    result_source: str = "original",
    confidence: str = "high",
    score: float = 0.61,
    signature: str = "Walk_Right",
) -> dict[str, object]:
    iid = f"iid-{index:03d}"
    observation = _observation(target_motion=target_motion)
    result = _result(
        verdict,
        confidence=confidence,
        signature=signature,
    )
    observation_digest = _object_digest(observation)
    result_digest = _object_digest(result)
    alignment_repairs: list[dict[str, object]] = []
    if result_source == "repair_1":
        alignment_repairs = [
            {
                "attempt": 1,
                "status": "ok",
                "repair_generation_called": True,
                "authoritative_context_digest": observation_digest,
            }
        ]
    elif result_source == "original_sanitized":
        alignment_repairs = [
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
        "alignment_repairs": alignment_repairs,
        "result": result,
        "result_digest": result_digest,
        "result_validated_from": result_source,
        "visual_input_digest": _sha(f"visual-{iid}".encode()),
        "execution_shard_index": index % 8,
        "execution_shard_count": 8,
        "execution_manifest": f"/audit/shard-{index % 8:03d}.jsonl",
        "execution_manifest_sha256": _sha(
            f"manifest-{index % 8}".encode()
        ),
        "run_config_digest": "a" * 64,
        "config_digest": _sha(f"config-{index % 8}".encode()),
        "implementation_digest": "b" * 64,
        "model_revision": "c" * 40,
        "transformers_version": "4.51.3",
    }
    if result_source == "fallback_uncertain":
        visual["result_fallback"] = {
            "fallback_digest": result_digest,
            "authoritative_context_digest": observation_digest,
        }
    return {
        "iid": iid,
        "input_digest": input_digest,
        "prompt": "make the person move",
        "src_video": f"{index}/source.mp4",
        "tgt_video": f"{index}/edited.mp4",
        "auto_rule": {
            "action_families": ["walk"],
            "actors": ["person"],
            "label": "temporal_action",
            "score": score,
            "tier": "high" if score >= 0.8 else "possible",
        },
        "r7_expansion_selection": {
            "schema_version": "motive-r7-expansion-selection-v1",
            "primary_family": "walk",
            "split_assigned": False,
        },
        "qwen_evidence": {"visual": visual},
    }


def _write_fused_commit(
    fused_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    fused_raw = _jsonl_bytes(rows)
    fused_sha = _sha(fused_raw)
    input_sha = "d" * 64
    verdict_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    verdict_family: dict[str, Counter[str]] = {}
    fallback_counts: Counter[str] = Counter(
        {"observation": 0, "result": 0}
    )
    repair_counts: Counter[str] = Counter(
        {
            "observation_rows": 0,
            "observation_attempts": 0,
            "alignment_rows": 0,
            "alignment_attempts": 0,
        }
    )
    validation_counts: Counter[str] = Counter()
    repair_generation_counts: Counter[str] = Counter()
    sanitization_counts: Counter[str] = Counter()
    shard_rows: Counter[int] = Counter()
    shard_metadata: dict[int, dict[str, str]] = {}
    for row in rows:
        visual = row["qwen_evidence"]["visual"]
        result = visual["result"]
        verdict = result["verdict"]
        family = row["r7_expansion_selection"]["primary_family"]
        verdict_counts[verdict] += 1
        family_counts[family] += 1
        verdict_family.setdefault(family, Counter())[verdict] += 1
        observation_source = visual["observation_validated_from"]
        result_source = visual["result_validated_from"]
        fallback_counts["observation"] += int(
            observation_source == "fallback_uncertain"
        )
        fallback_counts["result"] += int(
            result_source == "fallback_uncertain"
        )
        observation_repairs = visual["observation_repairs"]
        alignment_repairs = visual["alignment_repairs"]
        repair_counts["observation_rows"] += int(bool(observation_repairs))
        repair_counts["observation_attempts"] += len(observation_repairs)
        repair_counts["alignment_rows"] += int(bool(alignment_repairs))
        repair_counts["alignment_attempts"] += len(alignment_repairs)
        validation_counts[f"observation:{observation_source}"] += 1
        validation_counts[f"result:{result_source}"] += 1
        for stage, attempts in (
            ("observation", observation_repairs),
            ("alignment", alignment_repairs),
        ):
            for attempt in attempts:
                generated = attempt.get("repair_generation_called")
                if generated is True:
                    repair_generation_counts[f"{stage}:generated"] += 1
                elif generated is False:
                    repair_generation_counts[
                        f"{stage}:deterministic"
                    ] += 1
                for event in attempt.get("repair_sanitizations", []):
                    sanitization_counts[
                        f"{stage}:{event['action']}"
                    ] += 1
        shard = visual["execution_shard_index"]
        shard_rows[shard] += 1
        shard_metadata[shard] = {
            "manifest_sha256": visual["execution_manifest_sha256"],
            "config_digest": visual["config_digest"],
            "run_config_digest": visual["run_config_digest"],
        }
    first_visual = rows[0]["qwen_evidence"]["visual"]
    summary: dict[str, object] = {
        "schema_version": "motive-r7-qwen-visual-merge-v2",
        "partition_version": "line_modulo_v1",
        "shard_marker_schema": "motive-qwen-shard-manifest-v2",
        "input": {
            "path": "/synthetic/selected.jsonl",
            "rows": len(rows),
            "sha256": input_sha,
        },
        "qwen_root": "/synthetic/qwen",
        "shard_count": 8,
        "shards": [
            {
                "shard_index": index,
                "manifest_rows": shard_rows[index],
                "manifest_sha256": shard_metadata[index][
                    "manifest_sha256"
                ],
                "marker_sha256": _sha(f"marker-{index}".encode()),
                "output_rows": shard_rows[index],
                "output_sha256": _sha(f"output-{index}".encode()),
                "config_digest": shard_metadata[index]["config_digest"],
                "run_config_digest": shard_metadata[index][
                    "run_config_digest"
                ],
            }
            for index in range(8)
        ],
        "qwen_contract": {
            "implementation_digest": first_visual[
                "implementation_digest"
            ],
            "model_revision": first_visual["model_revision"],
            "transformers_version": first_visual["transformers_version"],
            "mode": "visual",
            "run_config_digest": first_visual["run_config_digest"],
            "observation_schema_version": "qwen-motion-observation-v2",
            "visual_schema_version": "qwen-motion-judge-v4",
        },
        "fused": {
            "name": "fused.jsonl",
            "rows": len(rows),
            "sha256": fused_sha,
        },
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "verdict_family_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(verdict_family.items())
        },
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
        "validation_source_counts": dict(
            sorted(validation_counts.items())
        ),
        "repair_generation_counts": dict(
            sorted(repair_generation_counts.items())
        ),
        "sanitization_counts": dict(sorted(sanitization_counts.items())),
    }
    summary_raw = _json_bytes(summary)
    done: dict[str, object] = {
        "schema_version": "motive-r7-qwen-visual-merge-done-v2",
        "status": "complete",
        "input_rows": len(rows),
        "input_sha256": input_sha,
        "fused_rows": len(rows),
        "fused_sha256": fused_sha,
        "summary_sha256": _sha(summary_raw),
        "artifact_digest": _object_digest(
            {
                "fused.jsonl": fused_sha,
                "summary.json": _sha(summary_raw),
            }
        ),
    }
    fused_dir.mkdir()
    (fused_dir / "fused.jsonl").write_bytes(fused_raw)
    (fused_dir / "summary.json").write_bytes(summary_raw)
    (fused_dir / "done.json").write_bytes(_json_bytes(done))


def _make_pair(root: Path) -> tuple[Path, Path]:
    rows = [
        _row(
            0,
            verdict="valid_action",
            score=0.95,
            signature="Walk_Right",
        ),
        _row(
            1,
            verdict="valid_action",
            score=0.85,
            signature="walk-right",
        ),
        _row(
            2,
            verdict="instruction_mismatch",
            score=0.75,
        ),
        _row(
            3,
            verdict="static",
            target_motion="none",
            result_source="original_sanitized",
            score=0.65,
        ),
        _row(
            4,
            verdict="valid_action",
            result_source="repair_1",
            score=0.55,
        ),
        _row(
            5,
            verdict="uncertain",
            result_source="fallback_uncertain",
            confidence="low",
            score=1.1,
        ),
        _row(
            6,
            verdict="valid_action",
            confidence="low",
            score=0.61,
        ),
        _row(
            7,
            verdict="static",
            target_motion="none",
            score=0.70,
        ),
    ]
    rows[0]["r7_expansion_selection"]["legacy_split_quarantine"] = {
        "present": False,
        "canonical_sha256": None,
    }
    rows[1]["r7_expansion_selection"]["legacy_split_quarantine"] = {
        "present": True,
        "canonical_sha256": "e" * 64,
    }
    rows[2]["split"] = "train"
    rows[2]["split_provenance"] = {
        "seed": 260108828,
        "version": "caption-or-path-fallback-v1",
    }
    # The manifest builder permits missing diagnostic auto_rule metadata.
    # The auditor must retain the row and expose an unstratifiable sentinel.
    rows[7].pop("auto_rule")
    fused = root / "fused_v2"
    # Build the manifest while fused.jsonl is the only fused artifact; the
    # builder consumes a file, while the auditor later requires the commit.
    fused.mkdir()
    (fused / "fused.jsonl").write_bytes(_jsonl_bytes(rows))
    manifest = root / "manifest_v2"
    build_expansion_manifest(
        input_path=fused / "fused.jsonl",
        output_dir=manifest,
    )
    (fused / "fused.jsonl").unlink()
    fused.rmdir()
    _write_fused_commit(fused, rows)
    return fused, manifest


def _rechain_manifest(manifest: Path) -> None:
    names = ("positives.jsonl", "negatives.jsonl", "review.jsonl")
    summary_path = manifest / "summary.json"
    summary = json.loads(summary_path.read_text())
    for name in names:
        raw = (manifest / name).read_bytes()
        summary["outputs"][name]["sha256"] = _sha(raw)
        summary["outputs"][name]["rows"] = len(raw.splitlines())
    summary_raw = _json_bytes(summary)
    summary_path.write_bytes(summary_raw)
    output_sha = {
        name: _sha((manifest / name).read_bytes()) for name in names
    }
    output_sha["summary.json"] = _sha(summary_raw)
    done_path = manifest / "done.json"
    done = json.loads(done_path.read_text())
    done["output_sha256"] = dict(sorted(output_sha.items()))
    done["output_rows"] = {
        name: len((manifest / name).read_bytes().splitlines())
        for name in sorted(names)
    }
    done["artifact_digest"] = _object_digest(done["output_sha256"])
    done_path.write_bytes(_json_bytes(done))


class R7ExpansionAuditTests(unittest.TestCase):
    def test_happy_path_recomputes_requested_strata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fused, manifest = _make_pair(root)
            output = root / "audit.json"
            summary = audit_expansion(
                fused_dir=fused,
                manifest_dir=manifest,
                output_path=output,
            )

            self.assertEqual(summary["conservation"]["fused_rows"], 8)
            self.assertTrue(
                summary["conservation"]["bucket_disjoint_union_exact"]
            )
            self.assertEqual(
                summary["bucket_verdict_family_counts"]["positive"][
                    "valid_action"
                ],
                {"walk": 2},
            )
            self.assertEqual(
                summary["negative_roles"]["counts"],
                {"audit_only": 1, "pseudo_negative": 2},
            )
            safety = summary["safety_assertions"]
            self.assertTrue(safety["all_passed"])
            self.assertEqual(safety["sanitized_rows"], 1)
            self.assertEqual(safety["repair_rows"], 1)
            self.assertEqual(safety["fallback_rows"], 1)
            signatures = summary["positive_signatures"]
            self.assertEqual(signatures["exact"]["unique"], 2)
            self.assertEqual(
                signatures["diagnostic_normalized"]["unique"],
                1,
            )
            self.assertEqual(
                summary["review_quality_failures"]["counts"],
                {"confidence=low": 1},
            )
            self.assertEqual(
                summary["auto_rule_strata"]["score_bin_counts"][
                    "gt_1.00"
                ],
                1,
            )
            self.assertEqual(
                summary["auto_rule_strata"]["unstratifiable_score_rows"],
                1,
            )
            quarantine = summary["legacy_split_quarantine"]
            self.assertTrue(
                quarantine[
                    "all_manifest_top_level_split_fields_absent"
                ]
            )
            self.assertEqual(quarantine["rows_removed"], 2)
            self.assertEqual(
                quarantine["quarantine_stage_counts"],
                {
                    "builder_legacy": 1,
                    "none": 6,
                    "selection_upstream": 1,
                },
            )
            self.assertTrue(output.is_file())

    def test_hash_tamper_and_self_consistent_safety_tamper_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fused, manifest = _make_pair(root)
            fused_path = fused / "fused.jsonl"
            fused_path.write_bytes(
                fused_path.read_bytes().replace(
                    b"make the person move",
                    b"make the person MOVE",
                    1,
                )
            )
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                audit_expansion(fused_dir=fused, manifest_dir=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fused, manifest = _make_pair(root)
            negatives_path = manifest / "negatives.jsonl"
            negatives = [
                json.loads(line)
                for line in negatives_path.read_text().splitlines()
            ]
            sanitized = next(
                row
                for row in negatives
                if row["r7_expansion_manifest"]["negative_role"]
                == "audit_only"
            )
            sanitized["r7_expansion_manifest"][
                "negative_role"
            ] = "pseudo_negative"
            negatives_path.write_bytes(_jsonl_bytes(negatives))
            summary_path = manifest / "summary.json"
            producer_summary = json.loads(summary_path.read_text())
            producer_summary["negative_role_counts"] = {
                "pseudo_negative": 2
            }
            summary_path.write_bytes(_json_bytes(producer_summary))
            _rechain_manifest(manifest)
            with self.assertRaisesRegex(
                ValueError,
                "negative_role mismatch",
            ):
                audit_expansion(fused_dir=fused, manifest_dir=manifest)

    def test_resume_is_exact_verification_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fused, manifest = _make_pair(root)
            output = root / "audit.json"
            expected = audit_expansion(
                fused_dir=fused,
                manifest_dir=manifest,
                output_path=output,
            )
            resumed = audit_expansion(
                fused_dir=fused,
                manifest_dir=manifest,
                output_path=output,
                resume=True,
            )
            self.assertEqual(resumed, expected)
            with self.assertRaises(FileExistsError):
                audit_expansion(
                    fused_dir=fused,
                    manifest_dir=manifest,
                    output_path=output,
                )
            output.write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                audit_expansion(
                    fused_dir=fused,
                    manifest_dir=manifest,
                    output_path=output,
                    resume=True,
                )

    def test_missing_resume_fails_before_inputs_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "--resume requires",
            ):
                audit_expansion(
                    fused_dir=root / "also-missing-fused",
                    manifest_dir=root / "also-missing-manifest",
                    output_path=root / "missing-audit.json",
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
