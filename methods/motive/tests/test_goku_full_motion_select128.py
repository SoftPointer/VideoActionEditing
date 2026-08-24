from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.motive.tests import test_goku_full_motion_postcheck as fixture
from methods.motive.tests import test_goku_full_motion_shard_manifest as shard_fixture
from motive import goku_full_motion_contract as contract
from motive import goku_full_motion_finalize as finalizer
from motive import goku_full_motion_qwen as qwen
from motive import goku_full_motion_select128 as selector


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_sha(value: object) -> str:
    return _sha(_canonical(value))


def _self_digested(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value["result_digest"] = _object_sha(value)
    return value


def _pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite_self_consistent_dataset_root(output: Path, row: dict) -> None:
    """Rewrite root digests after a deliberate sample/row mutation."""

    manifest_raw = _canonical(row) + b"\n"
    (output / selector.MANIFEST_NAME).write_bytes(manifest_raw)
    sample_artifact_digest = _object_sha(
        {row["iid"]: selector._dataset_row_artifact_map(row)}
    )
    summary_path = output / selector.SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["sample_artifact_digest"] = sample_artifact_digest
    summary["dataset_manifest"].update(
        {
            "sha256": _sha(manifest_raw),
            "bytes": len(manifest_raw),
            "rows": 1,
        }
    )
    summary_raw = _pretty(summary)
    summary_path.write_bytes(summary_raw)

    done_path = output / selector.DONE_NAME
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["sample_artifact_digest"] = sample_artifact_digest
    done["artifacts"][selector.MANIFEST_NAME].update(
        {"sha256": _sha(manifest_raw), "bytes": len(manifest_raw), "rows": 1}
    )
    done["artifacts"][selector.SUMMARY_NAME].update(
        {"sha256": _sha(summary_raw), "bytes": len(summary_raw), "rows": 1}
    )
    done["artifact_digest"] = _object_sha(done["artifacts"])
    done_payload = dict(done)
    done_payload.pop("done_digest", None)
    done["done_digest"] = _object_sha(done_payload)
    done_path.write_bytes(_pretty(done))


def _actual_generation_row(media_root: Path) -> dict:
    source = media_root / "source.mp4"
    anchor = media_root / "anchor.png"
    source.write_bytes(b"source-video-bytes")
    anchor.write_bytes(b"anchor-image-bytes")
    row = fixture._generation_row(source.resolve(), anchor.resolve())
    media = {"frame_count": 81, "fps": 25.0, "width": 832, "height": 480}
    temporal = {
        "schema_version": finalizer.TEMPORAL_GEOMETRY_SCHEMA,
        "source_frame_count": 81,
        "source_frame_rate": "25/1",
        "source_timeline_span_seconds": 3.2,
        "target_frame_count": 81,
        "target_frame_rate": "25/1",
        "target_timeline_span_seconds": 3.2,
        "requires_exact_frame_count_and_rate_match": True,
    }
    spec = row["motion_spec"]
    receipt_digest = "3" * 64
    row.update(
        {
            "selected_media_evidence": media,
            "selected_media_evidence_sha256": _object_sha(media),
            "strict_temporal_geometry": temporal,
            "full_motion_finalization": {
                "schema_version": finalizer.FINALIZATION_ROW_SCHEMA,
                "policy_version": finalizer.POLICY_VERSION,
                "candidate_rank": 1,
                "review_rank": 1,
                "selection_bucket": "primary",
                "dynamic_unit_count": 1,
                "target_action_signatures": [
                    spec["target_plan"]["dynamic_unit_targets"][0][
                        "target_action_signature"
                    ]
                ],
                "family": row["family"],
                "required_canary": False,
                "qwen_shard_index": 0,
                "qwen_receipt_digest": receipt_digest,
            },
            "authorization_interface_available": False,
        }
    )
    finalizer.validate_generation_row(row)
    return row


def _write_finalizer_closure(root: Path, row: dict) -> Path:
    manifest = root / "primary_1.jsonl"
    raw_outputs = {
        manifest.name: _canonical(row) + b"\n",
        "reserve_0.jsonl": b"",
        finalizer.REVIEW_NAME: _canonical(row) + b"\n",
        finalizer.SUMMARY_NAME: b"{}\n",
    }
    for name, raw in raw_outputs.items():
        (root / name).write_bytes(raw)
    artifacts = {
        name: {
            "sha256": _sha(raw),
            "bytes": len(raw),
            "rows": 1 if name in {manifest.name, finalizer.REVIEW_NAME} else 0
            if name == "reserve_0.jsonl"
            else 1,
        }
        for name, raw in raw_outputs.items()
    }
    payload = {
        "schema_version": finalizer.DONE_SCHEMA,
        "status": "complete",
        "artifacts": artifacts,
        "artifact_digest": _object_sha(artifacts),
        "input_digest": "1" * 64,
        "implementation_digest": "2" * 64,
    }
    done = dict(payload)
    done["done_digest"] = _object_sha(payload)
    (root / finalizer.DONE_NAME).write_bytes(_canonical(done) + b"\n")
    return manifest


class ExactSelectionTests(unittest.TestCase):
    def test_generation_shard_root_alias_is_repeatable(self) -> None:
        args = selector.build_parser().parse_args(
            [
                "--generation-manifest",
                "/tmp/primary.jsonl",
                "--generation-shard-manifest-dir",
                "/tmp/generation-shards",
                "--generation-shard-root",
                "/tmp/shard-000",
                "--wan-run-root",
                "/tmp/shard-001",
                "--postcheck-output",
                "/tmp/post-000.jsonl",
                "--output-dir",
                "/tmp/exact128",
            ]
        )
        self.assertEqual(
            args.wan_run_root,
            [Path("/tmp/shard-000"), Path("/tmp/shard-001")],
        )

    def test_generation_shard_index_dir_discovers_sorted_complete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory)
            (index / "jobs.tsv").write_text("audit-only metadata\n")
            for name in ("shard_010", "shard_002"):
                root = index / name
                (root / "samples").mkdir(parents=True)
                for artifact in (
                    "run_contract.json",
                    "generated_manifest.jsonl",
                    "run_complete.json",
                ):
                    (root / artifact).write_text("{}\n")
            roots = selector.discover_generation_shard_roots(index)
            self.assertEqual(
                [root.name for root in roots], ["shard_002", "shard_010"]
            )

            incomplete = index / "shard_011"
            incomplete.mkdir()
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error, "incomplete"
            ):
                selector.discover_generation_shard_roots(index)

    def test_exact_128_prefers_three_then_two_then_primary_order(self) -> None:
        rows = []
        for index in range(150):
            unit_count = 3 if index < 40 else 2 if index < 90 else 1
            rows.append(
                {
                    "iid": f"iid-{index:03d}",
                    "primary_index": 149 - index,
                    "dynamic_unit_count": unit_count,
                }
            )
        selected = selector.select_exact_candidates(rows)
        self.assertEqual(len(selected), 128)
        self.assertEqual(
            [row["dynamic_unit_count"] for row in selected[:40]], [3] * 40
        )
        self.assertEqual(
            [row["dynamic_unit_count"] for row in selected[40:90]], [2] * 50
        )
        self.assertGreaterEqual(
            sum(row["dynamic_unit_count"] >= 2 for row in selected), 32
        )
        for unit_count in (3, 2, 1):
            indices = [
                row["primary_index"]
                for row in selected
                if row["dynamic_unit_count"] == unit_count
            ]
            self.assertEqual(indices, sorted(indices))

    def test_insufficient_total_or_multi_unit_pool_fails(self) -> None:
        rows = [
            {
                "iid": f"iid-{index:03d}",
                "primary_index": index,
                "dynamic_unit_count": 2 if index < 31 else 1,
            }
            for index in range(128)
        ]
        with self.assertRaisesRegex(
            selector.GokuFullMotionSelect128Error, "insufficient multi-unit"
        ):
            selector.select_exact_candidates(rows)
        with self.assertRaisesRegex(
            selector.GokuFullMotionSelect128Error, "insufficient postcheck"
        ):
            selector.select_exact_candidates(
                rows[:100], exact_size=101, min_multi_unit=0
            )


class GenerationClosureTests(unittest.TestCase):
    def test_postcheck_v6_qwen_a0_i0_binding_is_not_downgradable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _actual_generation_row(root)
            normalized = selector.postcheck._normalize_contract(
                row, manifest_root=root
            )
            census = fixture._target_census()
            judgment = fixture._judgment()
            aggregate = selector.postcheck.aggregate_postcheck(
                census,
                judgment,
                expected_contract=selector.postcheck._expected_judge_contract(
                    normalized
                ),
            )
            record = {
                "schema_version": "motive-goku-full-motion-postcheck-v6",
                "iid": row["iid"],
                "change_region_proposals_digest": normalized[
                    "change_region_proposals_digest"
                ],
                "coverage_authority_inventory_digest": normalized[
                    "coverage_authority_inventory_digest"
                ],
                "coverage_authority_assignments_digest": normalized[
                    "coverage_authority_assignments_digest"
                ],
                "coverage_authority_digest": normalized[
                    "coverage_authority_digest"
                ],
                "coverage_authority_alignment_digest": normalized[
                    "coverage_authority_alignment_digest"
                ],
                "source_census_digest": normalized["source_census_digest"],
                "target_plan_digest": normalized["target_plan_digest"],
                "motion_spec_digest": normalized["motion_spec_digest"],
                "compiled_instruction_digest": normalized[
                    "compiled_instruction_digest"
                ],
                "coverage_critic_digest": normalized[
                    "coverage_critic_digest"
                ],
                "full_motion_contract_digest": normalized[
                    "full_motion_contract_digest"
                ],
                "qwen_result_digest": normalized["qwen_result_digest"],
                "qwen_provenance_digest": normalized[
                    "qwen_provenance_digest"
                ],
                "qwen_record_payload_sha256": normalized[
                    "qwen_record_payload_sha256"
                ],
                "qwen_evidence_binding": normalized[
                    "qwen_evidence_binding"
                ],
                "instruction_sha256": normalized["instruction_sha256"],
                "target_census": census,
                "clause_judgment": judgment,
                "aggregate": aggregate,
                "decision": aggregate["decision"],
                "eligible": aggregate["eligible"],
            }
            observed, expected = selector._validate_postcheck_semantics(
                record,
                generation_row=row,
                generation_manifest_path=root / "primary.jsonl",
            )
            self.assertEqual(
                observed["i0_grounding_digest"],
                row["qwen_evidence"]["i0_grounding_digest"],
            )
            for field in (
                "change_region_proposals_digest",
                "coverage_authority_inventory_digest",
                "coverage_authority_assignments_digest",
                "coverage_authority_digest",
                "coverage_authority_alignment_digest",
            ):
                self.assertEqual(observed[field], row["qwen_evidence"][field])
            self.assertEqual(expected, aggregate)

            legacy = copy.deepcopy(record)
            legacy["schema_version"] = "motive-goku-full-motion-postcheck-v5"
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "postcheck v6 schema differs",
            ):
                selector._validate_postcheck_semantics(
                    legacy,
                    generation_row=row,
                    generation_manifest_path=root / "primary.jsonl",
                )

            shadow = copy.deepcopy(record)
            shadow["qwen_evidence_binding"]["hard_gate_schema_version"] = (
                "goku-full-motion-hard-gate-v5"
            )
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "qwen_evidence_binding differs",
            ):
                selector._validate_postcheck_semantics(
                    shadow,
                    generation_row=row,
                    generation_manifest_path=root / "primary.jsonl",
                )

            missing_a0 = copy.deepcopy(record)
            del missing_a0["coverage_authority_digest"]
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "coverage_authority_digest differs",
            ):
                selector._validate_postcheck_semantics(
                    missing_a0,
                    generation_row=row,
                    generation_manifest_path=root / "primary.jsonl",
                )

            tampered_a0 = copy.deepcopy(record)
            tampered_a0["coverage_authority_alignment_digest"] = "0" * 64
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "coverage_authority_alignment_digest differs",
            ):
                selector._validate_postcheck_semantics(
                    tampered_a0,
                    generation_row=row,
                    generation_manifest_path=root / "primary.jsonl",
                )

    def test_v6_a0_i0_evidence_record_and_hard_gate_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = _actual_generation_row(Path(directory))
            accepted = selector._validate_generation_row(row)
            a0_bindings = {
                "change_region_proposals": (
                    "change_region_proposals_digest",
                    "change_region_proposals_sha256",
                ),
                "coverage_authority": (
                    "coverage_authority_digest",
                    "coverage_authority_sha256",
                ),
                "coverage_authority_alignment": (
                    "coverage_authority_alignment_digest",
                    "coverage_authority_alignment_sha256",
                ),
            }
            authority = row["motion_spec"]["coverage_authority"]
            a0_bindings.update(
                {
                    "coverage_authority.inventory": (
                        "coverage_authority_inventory_digest",
                        "coverage_authority_inventory_sha256",
                    ),
                    "coverage_authority.assignments": (
                        "coverage_authority_assignments_digest",
                        "coverage_authority_assignments_sha256",
                    ),
                }
            )
            for spec_field, (evidence_field, hard_gate_field) in a0_bindings.items():
                if spec_field.startswith("coverage_authority."):
                    digest = _object_sha(authority[spec_field.split(".", 1)[1]])
                else:
                    digest = _object_sha(row["motion_spec"][spec_field])
                self.assertEqual(
                    accepted["qwen_evidence"][evidence_field], digest
                )
                self.assertEqual(
                    accepted["qwen_evidence"]["hard_gate"][hard_gate_field],
                    digest,
                )
            grounding_sha = _object_sha(row["motion_spec"]["i0_grounding"])
            self.assertEqual(
                accepted["qwen_evidence"]["i0_grounding_digest"],
                grounding_sha,
            )
            self.assertEqual(
                accepted["qwen_evidence"]["hard_gate"][
                    "i0_grounding_sha256"
                ],
                grounding_sha,
            )
            self.assertEqual(
                _object_sha(
                    accepted["qwen_evidence"]["qwen_record_payload"]
                ),
                selector.postcheck._normalize_contract(
                    row, manifest_root=Path(directory)
                )["qwen_record_payload_sha256"],
            )

            for case in (
                "generation_v5",
                "motion_spec_v5",
                "qwen_evidence_v5",
                "qwen_record_v5",
                "hard_gate_v5",
            ):
                with self.subTest(case=case):
                    bad = copy.deepcopy(row)
                    if case == "generation_v5":
                        bad["schema_version"] = (
                            "motive-goku-full-motion-generation-v5"
                        )
                    elif case == "motion_spec_v5":
                        bad["motion_spec"]["schema_version"] = (
                            "motive-goku-full-motion-generation-spec-v5"
                        )
                        bad["motion_spec_sha256"] = _object_sha(
                            bad["motion_spec"]
                        )
                    elif case == "qwen_evidence_v5":
                        bad["qwen_evidence"]["schema_version"] = (
                            "motive-goku-full-motion-qwen-evidence-v5"
                        )
                    elif case == "qwen_record_v5":
                        bad["qwen_evidence"]["record_schema_version"] = (
                            "goku-full-motion-qwen-record-v5"
                        )
                    else:
                        bad["qwen_evidence"]["hard_gate"][
                            "schema_version"
                        ] = "goku-full-motion-hard-gate-v5"
                    with self.assertRaises(
                        selector.GokuFullMotionSelect128Error
                    ):
                        selector._validate_generation_row(bad)

            for case in (
                "missing_proposals",
                "missing_authority_digest",
                "missing_authority_alignment_gate",
            ):
                with self.subTest(case=case):
                    bad = copy.deepcopy(row)
                    if case == "missing_proposals":
                        del bad["motion_spec"]["change_region_proposals"]
                        bad["motion_spec_sha256"] = _object_sha(
                            bad["motion_spec"]
                        )
                    elif case == "missing_authority_digest":
                        del bad["qwen_evidence"]["coverage_authority_digest"]
                    else:
                        del bad["qwen_evidence"]["hard_gate"][
                            "coverage_authority_alignment_sha256"
                        ]
                    with self.assertRaises(
                        selector.GokuFullMotionSelect128Error
                    ):
                        selector._validate_generation_row(bad)

            shadow = copy.deepcopy(row)
            shadow_grounding = shadow["motion_spec"]["i0_grounding"]
            shadow_grounding["subjects"][0]["i0_state"] = (
                "The person kneels with both arms down at I0"
            )
            shadow_grounding_sha = _object_sha(shadow_grounding)
            shadow["motion_spec_sha256"] = _object_sha(shadow["motion_spec"])
            shadow["qwen_evidence"][
                "i0_grounding_digest"
            ] = shadow_grounding_sha
            shadow["qwen_evidence"]["hard_gate"][
                "i0_grounding_sha256"
            ] = shadow_grounding_sha
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "exact-I0",
            ):
                selector._validate_generation_row(shadow)

            missing_payload = copy.deepcopy(row)
            del missing_payload["qwen_evidence"]["qwen_record_payload"]
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "qwen_evidence is not closed",
            ):
                selector._validate_generation_row(missing_payload)

            payload_shadow = copy.deepcopy(row)
            payload_shadow["qwen_evidence"]["qwen_record_payload"][
                "shadow_provenance"
            ] = "0" * 64
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "closed Qwen v6 record",
            ):
                selector._validate_generation_row(payload_shadow)

    def test_self_redigested_qwen_payload_and_consistent_digest_forgery_reject(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = _actual_generation_row(Path(directory))

            forged = copy.deepcopy(row)
            payload = forged["qwen_evidence"]["qwen_record_payload"]
            payload["compiled_instruction"] = copy.deepcopy(
                payload["compiled_instruction"]
            )
            payload["compiled_instruction"]["edit_instruction"] += (
                " and spin in place"
            )
            payload["result_digest"] = _object_sha(
                qwen.qwen_result_payload(payload)
            )
            payload["provenance_digest"] = qwen.qwen_provenance_digest(
                payload
            )
            for field in ("result_digest", "provenance_digest"):
                forged["qwen_evidence"][field] = payload[field]
                forged["motion_spec"][f"qwen_{field}"] = payload[field]
            forged["motion_spec_sha256"] = _object_sha(
                forged["motion_spec"]
            )
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "semantic artifact binding differs",
            ):
                selector._validate_generation_row(forged)

            original_normalized = selector.postcheck._normalize_contract(
                row, manifest_root=Path(directory)
            )
            forged_binding = copy.deepcopy(
                original_normalized["qwen_evidence_binding"]
            )
            forged_binding.update(
                {
                    "result_digest": forged["qwen_evidence"][
                        "result_digest"
                    ],
                    "provenance_digest": forged["qwen_evidence"][
                        "provenance_digest"
                    ],
                    "qwen_evidence_digest": _object_sha(
                        forged["qwen_evidence"]
                    ),
                    "qwen_record_payload_sha256": _object_sha(payload),
                }
            )
            forged_postcheck = {
                "schema_version": selector.postcheck.POSTCHECK_SCHEMA,
                "iid": row["iid"],
                "qwen_result_digest": forged["qwen_evidence"][
                    "result_digest"
                ],
                "qwen_provenance_digest": forged["qwen_evidence"][
                    "provenance_digest"
                ],
                "qwen_record_payload_sha256": _object_sha(payload),
                "qwen_evidence_binding": forged_binding,
            }
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "generation/Qwen-record closure differs",
            ):
                selector._validate_postcheck_semantics(
                    forged_postcheck,
                    generation_row=forged,
                    generation_manifest_path=Path(directory) / "primary.jsonl",
                )

            equal_self_reports = copy.deepcopy(row)
            for field, digest in (
                ("result_digest", "a" * 64),
                ("provenance_digest", "b" * 64),
            ):
                equal_self_reports["qwen_evidence"][field] = digest
                equal_self_reports["qwen_evidence"]["qwen_record_payload"][
                    field
                ] = digest
                equal_self_reports["motion_spec"][f"qwen_{field}"] = digest
            equal_self_reports["motion_spec_sha256"] = _object_sha(
                equal_self_reports["motion_spec"]
            )
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "canonical payload",
            ):
                selector._validate_generation_row(equal_self_reports)

    def test_same_schema_top_level_additions_and_motion_spec_additions_reject(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = _actual_generation_row(Path(directory))
            future = copy.deepcopy(row)
            future["future_top_level_evidence"] = {"version": 2}
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "closed schema",
            ):
                selector._validate_generation_row(future)

            shadow = copy.deepcopy(row)
            shadow["secondary_source_census"] = copy.deepcopy(
                row["motion_spec"]["secondary_source_census"]
            )
            shadow["secondary_source_census"]["scene_description"] = (
                "A top-level shadow must never override motion_spec"
            )
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "closed schema",
            ):
                selector._validate_generation_row(shadow)

            a0_shadow = copy.deepcopy(row)
            a0_shadow["coverage_authority"] = copy.deepcopy(
                row["motion_spec"]["coverage_authority"]
            )
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "closed schema",
            ):
                selector._validate_generation_row(a0_shadow)

            bad = copy.deepcopy(row)
            bad["motion_spec"]["future_nested_field"] = True
            bad["motion_spec_sha256"] = _object_sha(bad["motion_spec"])
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error, "motion_spec is not closed"
            ):
                selector._validate_generation_row(bad)

    def test_finalizer_parent_is_byte_bound_and_exactly_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            media = base / "media"
            final = base / "final"
            media.mkdir()
            final.mkdir()
            row = _actual_generation_row(media)
            manifest = _write_finalizer_closure(final, row)
            rows, closure = selector.load_generation_manifest(manifest)
            self.assertEqual([value["iid"] for value in rows], [row["iid"]])
            self.assertEqual(closure["manifest"]["sha256"], _sha(manifest.read_bytes()))

            intruder = final / "unreceipted.txt"
            intruder.write_text("not in done", encoding="utf-8")
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error, "directory closure"
            ):
                selector.load_generation_manifest(manifest)
            intruder.unlink()

            (final / finalizer.SUMMARY_NAME).write_bytes(b"tampered\n")
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error, "artifact SHA"
            ):
                selector.load_generation_manifest(manifest)


class ShardedTopologyTests(unittest.TestCase):
    def _topology(self, root: Path):
        parent = shard_fixture._build_parent(root)
        sharded = root / "generation_shards"
        shard_fixture._materialize(parent, sharded)
        primary = parent / "primary_256.jsonl"
        rows, finalizer_closure = selector.load_generation_manifest(primary)
        by_iid, by_path, descriptors, closure = (
            selector.load_generation_shard_manifest(
                sharded,
                generation_manifest_path=primary,
                generation_rows=rows,
                finalizer_closure=finalizer_closure,
            )
        )
        return (
            parent,
            sharded,
            primary,
            rows,
            finalizer_closure,
            by_iid,
            by_path,
            descriptors,
            closure,
        )

    def test_real_primary256_is_reconstructed_as_exact_32x8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                _parent,
                sharded,
                primary,
                rows,
                _finalizer_closure,
                by_iid,
                by_path,
                descriptors,
                closure,
            ) = self._topology(root)
            self.assertEqual(len(rows), 256)
            self.assertEqual(len(descriptors), 32)
            self.assertEqual(len(by_path), 32)
            self.assertEqual(len(by_iid), 256)
            self.assertEqual(
                [by_iid[str(row["iid"])]["root_row_index"] for row in rows],
                list(range(256)),
            )
            reconstructed = b"".join(
                (sharded / descriptor["path"]).read_bytes()
                for descriptor in descriptors
            )
            self.assertEqual(reconstructed, primary.read_bytes())
            self.assertEqual(closure["root"], str(sharded.resolve()))

            first = sharded / descriptors[0]["path"]
            first.write_bytes(first.read_bytes() + b"\n")
            with self.assertRaises(selector.GokuFullMotionSelect128Error):
                selector.load_generation_shard_manifest(
                    sharded,
                    generation_manifest_path=primary,
                    generation_rows=rows,
                    finalizer_closure=_finalizer_closure,
                )

    def test_wan_and_postcheck_close_each_real_descriptor_before_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                _parent,
                _sharded,
                _primary,
                rows,
                _finalizer_closure,
                by_iid,
                by_path,
                descriptors,
                _closure,
            ) = self._topology(root)
            wan_parent = root / "wan_shards"
            wan_parent.mkdir()
            roots: list[Path] = []
            descriptor_by_index = {
                int(value["descriptor"]["shard_index"]): value
                for value in by_path.values()
            }
            for index in range(32):
                binding = descriptor_by_index[index]
                run_root = wan_parent / f"shard_{index:03d}"
                run_root.mkdir()
                contract = {
                    "manifest": {
                        "path": str(binding["manifest_path"]),
                        "sha256": binding["manifest_sha256"],
                        "bytes": binding["manifest_bytes"],
                        "row_count": 8,
                    },
                    "contract_digest": f"{index:064x}",
                }
                (run_root / "run_contract.json").write_bytes(
                    _canonical(contract) + b"\n"
                )
                generated = [
                    {"iid": iid}
                    for iid in binding["descriptor"]["ordered_iids"]
                ]
                (run_root / "generated_manifest.jsonl").write_bytes(
                    b"".join(_canonical(row) + b"\n" for row in generated)
                )
                complete = {"complete_digest": f"{index + 32:064x}"}
                (run_root / "run_complete.json").write_bytes(
                    _canonical(complete) + b"\n"
                )
                roots.append(run_root)

            def fake_contract(run_root, **kwargs):
                value = json.loads((run_root / "run_contract.json").read_text())
                self.assertEqual(Path(value["manifest"]["path"]), kwargs["manifest_path"])
                self.assertEqual(value["manifest"]["sha256"], kwargs["manifest_sha256"])
                self.assertEqual(kwargs["manifest_rows"], 8)
                return value, _sha((run_root / "run_contract.json").read_bytes())

            def fake_generated(run_root, **kwargs):
                path = kwargs["generated_manifest_path"]
                generated = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(
                    [row["iid"] for row in generated],
                    [row["iid"] for row in kwargs["generation_rows"]],
                )
                complete_path = run_root / "run_complete.json"
                complete = json.loads(complete_path.read_text())
                return (
                    generated,
                    _sha(path.read_bytes()),
                    complete,
                    _sha(complete_path.read_bytes()),
                )

            with mock.patch.object(
                selector.postcheck,
                "_validate_run_contract",
                side_effect=fake_contract,
            ), mock.patch.object(
                selector.postcheck,
                "_validate_generated_manifest",
                side_effect=fake_generated,
            ):
                wan_by_iid, wan_closures = selector._load_wan_runs(
                    roots,
                    generated_manifests=None,
                    generation_shards_by_path=by_path,
                    generation_shards_by_iid=by_iid,
                )
            self.assertEqual(len(wan_by_iid), 256)
            self.assertEqual(
                [value["generation_shard"]["shard_index"] for value in wan_closures],
                list(range(32)),
            )

            post_root = root / "postcheck"
            post_root.mkdir()
            outputs: list[Path] = []
            for index in range(32):
                binding = descriptor_by_index[index]
                wan = wan_by_iid[binding["descriptor"]["ordered_iids"][0]]
                output = post_root / f"postcheck_shard_{index:03d}.jsonl"
                records = [
                    {"iid": iid, "status": "ok"}
                    for iid in binding["descriptor"]["ordered_iids"]
                ]
                output.write_bytes(
                    b"".join(_canonical(record) + b"\n" for record in records)
                )
                receipt = {
                    "schema_version": "synthetic-receipt-v1",
                    "status": "complete",
                    "manifest": str(binding["manifest_path"]),
                    "manifest_sha256": binding["manifest_sha256"],
                    "generation_root": str(wan["root"]),
                    "run_contract_sha256": wan["run_contract_sha256"],
                    "generated_manifest": wan["closure"]["generated_manifest"]["path"],
                    "generated_manifest_sha256": wan["closure"]["generated_manifest"]["sha256"],
                    "run_complete_sha256": wan["closure"]["run_complete"]["sha256"],
                    "config_digest": f"{index + 64:064x}",
                    "assigned_iids": binding["descriptor"]["ordered_iids"],
                    "output": {"path": str(output)},
                    "receipt_digest": f"{index + 96:064x}",
                }
                selector.postcheck.shard_receipt_path(output).write_bytes(
                    _canonical(receipt) + b"\n"
                )
                outputs.append(output)
            generation_by_iid = {str(row["iid"]): row for row in rows}
            with mock.patch.object(
                selector.postcheck, "validate_shard_receipt"
            ), mock.patch.object(
                selector.postcheck, "_validate_output_record"
            ), mock.patch.object(
                selector,
                "_validate_postcheck_semantics",
                return_value=(
                    {"dynamic_units": [{"unit_id": "unit_01"}]},
                    {"decision": "pass", "eligible": True, "failure_codes": []},
                ),
            ):
                post_by_iid, post_closures = selector._load_postcheck_shards(
                    outputs,
                    receipts=None,
                    generation_rows_by_iid=generation_by_iid,
                    generation_shards_by_iid=by_iid,
                    generation_shards_by_path=by_path,
                    wan_by_iid=wan_by_iid,
                )
            self.assertEqual(len(post_by_iid), 256)
            self.assertEqual(
                [value["generation_shard"]["shard_index"] for value in post_closures],
                list(range(32)),
            )


class AtomicMaterializationTests(unittest.TestCase):
    def _candidate(self, root: Path) -> dict:
        media = root / "media"
        media.mkdir()
        row = _actual_generation_row(media)
        iid = row["iid"]
        wan_root = root / "wan"
        sample = wan_root / "samples" / iid
        sample.mkdir(parents=True)

        source = Path(row["resolved_source_video"])
        target = sample / "generated.mp4"
        target.write_bytes(b"generated-target-video")
        result_payload = {
            "schema_version": "motive-wan22-i2v-sample-v1",
            "iid": iid,
            "geometry": {"frames": 81, "fps": "25/1"},
        }
        result = _self_digested(result_payload)
        result_path = sample / "result.json"
        result_path.write_bytes(_canonical(result) + b"\n")

        conditioning = {
            "conditioning_anchor_original": sample / "anchor_original.png",
            "conditioning_frame0_float32": sample / "frame0.npy",
            "conditioning_frame0_png": sample / "frame0.png",
            "conditioning_latent": sample / "conditioning_latent.pt",
        }
        for index, path in enumerate(conditioning.values(), start=1):
            path.write_bytes(f"conditioning-{index}".encode("utf-8"))
        generated = {"iid": iid}
        for field, path in conditioning.items():
            generated[field] = str(path)
            generated[f"{field}_sha256"] = _sha(path.read_bytes())

        normalized = selector.postcheck._normalize_contract(
            row, manifest_root=media
        )
        aggregate = {"decision": "pass", "eligible": True, "failure_codes": []}
        post_payload = {
            "schema_version": selector.postcheck.POSTCHECK_SCHEMA,
            "iid": iid,
            "status": "ok",
            "change_region_proposals_digest": normalized[
                "change_region_proposals_digest"
            ],
            "coverage_authority_inventory_digest": normalized[
                "coverage_authority_inventory_digest"
            ],
            "coverage_authority_assignments_digest": normalized[
                "coverage_authority_assignments_digest"
            ],
            "coverage_authority_digest": normalized[
                "coverage_authority_digest"
            ],
            "coverage_authority_alignment_digest": normalized[
                "coverage_authority_alignment_digest"
            ],
            "qwen_record_payload_sha256": normalized[
                "qwen_record_payload_sha256"
            ],
            "qwen_evidence_binding": copy.deepcopy(
                normalized["qwen_evidence_binding"]
            ),
            "decision": "pass",
            "eligible": True,
            "aggregate": aggregate,
        }
        post_record = _self_digested(post_payload)
        manifest_path = root / "primary.jsonl"
        manifest_path.write_bytes(_canonical(row) + b"\n")
        manifest_sha = _sha(manifest_path.read_bytes())
        return {
            "iid": iid,
            "primary_index": 0,
            "dynamic_unit_count": 1,
            "generation_row": row,
            "generation_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha,
            },
            "generation_shard": {
                "manifest_path": manifest_path.resolve(),
                "manifest_sha256": manifest_sha,
                "descriptor_digest": "8" * 64,
                "shard_index": 0,
                "shard_row_index": 0,
                "root_row_index": 0,
            },
            "wan": {
                "root": wan_root.resolve(),
                "generated_row": generated,
                "closure": {
                    "generated_manifest": {
                        "path": str((wan_root / "generated_manifest.jsonl").resolve()),
                        "sha256": "1" * 64,
                        "rows": 1,
                    },
                    "run_contract": {
                        "path": str((wan_root / "run_contract.json").resolve()),
                        "sha256": "2" * 64,
                        "contract_digest": "3" * 64,
                    },
                    "run_complete": {
                        "path": str((wan_root / "run_complete.json").resolve()),
                        "sha256": "4" * 64,
                        "complete_digest": "5" * 64,
                    },
                },
            },
            "postcheck": {
                "record": post_record,
                "normalized": normalized,
                "closure": {
                    "output": {"path": str(root / "post.jsonl"), "sha256": "6" * 64},
                    "receipt": {"path": str(root / "post.receipt.json"), "sha256": "7" * 64},
                },
            },
            "verified_media": {
                "source": {"path": str(source), "sha256": _sha(source.read_bytes())},
                "target": {"path": str(target), "sha256": _sha(target.read_bytes())},
                "sample_result": {
                    "path": str(result_path),
                    "sha256": _sha(result_path.read_bytes()),
                },
            },
        }

    def test_atomic_create_only_materialization_and_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            output = root / "dataset"
            summary = selector._publish_dataset(
                [candidate],
                output_dir=output,
                exact_size=1,
                min_multi_unit=0,
                finalizer_closure={"manifest": candidate["generation_manifest"]},
                generation_shard_closure={"status": "synthetic"},
                wan_closures=[candidate["wan"]["closure"]],
                postcheck_closures=[candidate["postcheck"]["closure"]],
            )
            self.assertEqual(summary["counts"]["selected"], 1)
            validated = selector.validate_materialized_dataset(output)
            self.assertEqual(validated["rows"], 1)
            manifest_row = json.loads(
                (output / selector.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest_row["schema_version"],
                "motive-goku-full-motion-dataset-row-v6",
            )
            self.assertNotEqual(
                selector.DATASET_ROW_SCHEMA,
                "motive-goku-full-motion-dataset-row-v4",
            )
            self.assertEqual(
                json.loads((output / selector.DONE_NAME).read_text())[
                    "selector_schema_version"
                ],
                "motive-goku-full-motion-select128-v6",
            )
            sample = output / "samples" / candidate["iid"]
            self.assertEqual(
                (sample / "edit_instruction.txt").read_bytes(),
                candidate["generation_row"]["edit_instruction"].encode("utf-8"),
            )
            materialized_generation = json.loads(
                (sample / "generation_row.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                materialized_generation["qwen_evidence"],
                candidate["generation_row"]["qwen_evidence"],
            )
            materialized_qwen_record = json.loads(
                (sample / "qwen_record_payload.json").read_text(
                    encoding="utf-8"
                )
            )
            expected_qwen_record = candidate["generation_row"][
                "qwen_evidence"
            ]["qwen_record_payload"]
            self.assertEqual(materialized_qwen_record, expected_qwen_record)
            self.assertEqual(
                manifest_row["i0_grounding_sha256"],
                _object_sha(
                    candidate["generation_row"]["motion_spec"][
                        "i0_grounding"
                    ]
                ),
            )
            a0_bindings = {
                "change_region_proposals": "change_region_proposals_sha256",
                "coverage_authority": "coverage_authority_sha256",
                "coverage_authority_alignment": (
                    "coverage_authority_alignment_sha256"
                ),
            }
            for spec_field, row_field in a0_bindings.items():
                expected_digest = _object_sha(
                    candidate["generation_row"]["motion_spec"][spec_field]
                )
                self.assertEqual(manifest_row[row_field], expected_digest)
                self.assertEqual(
                    manifest_row["generation_binding"][row_field],
                    expected_digest,
                )
                self.assertEqual(
                    manifest_row["postcheck_binding"][row_field],
                    expected_digest,
                )
            for authority_field, row_field in (
                (
                    "inventory",
                    "coverage_authority_inventory_sha256",
                ),
                (
                    "assignments",
                    "coverage_authority_assignments_sha256",
                ),
            ):
                expected_digest = _object_sha(
                    candidate["generation_row"]["motion_spec"][
                        "coverage_authority"
                    ][authority_field]
                )
                self.assertEqual(manifest_row[row_field], expected_digest)
                self.assertEqual(
                    manifest_row["generation_binding"][row_field],
                    expected_digest,
                )
                self.assertEqual(
                    manifest_row["postcheck_binding"][row_field],
                    expected_digest,
                )
            self.assertEqual(
                manifest_row["qwen_evidence_sha256"],
                _object_sha(candidate["generation_row"]["qwen_evidence"]),
            )
            expected_qwen_record_sha = _object_sha(expected_qwen_record)
            self.assertEqual(
                manifest_row["qwen_record_payload_sha256"],
                expected_qwen_record_sha,
            )
            self.assertEqual(
                manifest_row["generation_binding"][
                    "qwen_record_payload_sha256"
                ],
                expected_qwen_record_sha,
            )
            self.assertEqual(
                manifest_row["postcheck_binding"][
                    "qwen_record_payload_sha256"
                ],
                expected_qwen_record_sha,
            )
            self.assertEqual(
                manifest_row["qwen_hard_gate_sha256"],
                _object_sha(
                    candidate["generation_row"]["qwen_evidence"][
                        "hard_gate"
                    ]
                ),
            )
            finalizer.validate_generation_row(materialized_generation)
            self.assertEqual(
                Path(manifest_row["artifacts"]["source"]["path"]),
                (sample / "source.mp4").resolve(),
            )
            self.assertIn(
                "conditioning_latent",
                manifest_row["artifacts"]["conditioning"],
            )

            postcheck_tamper_output = root / "dataset_postcheck_a0_tamper"
            selector._publish_dataset(
                [candidate],
                output_dir=postcheck_tamper_output,
                exact_size=1,
                min_multi_unit=0,
                finalizer_closure={"manifest": candidate["generation_manifest"]},
                generation_shard_closure={"status": "synthetic"},
                wan_closures=[candidate["wan"]["closure"]],
                postcheck_closures=[candidate["postcheck"]["closure"]],
            )
            postcheck_tampered_row = json.loads(
                (postcheck_tamper_output / selector.MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            postcheck_path = (
                postcheck_tamper_output
                / "samples"
                / candidate["iid"]
                / "postcheck.json"
            )
            postcheck_value = json.loads(
                postcheck_path.read_text(encoding="utf-8")
            )
            postcheck_value["coverage_authority_digest"] = "0" * 64
            postcheck_value.pop("result_digest")
            postcheck_value["result_digest"] = _object_sha(postcheck_value)
            postcheck_raw = _pretty(postcheck_value)
            postcheck_path.write_bytes(postcheck_raw)
            postcheck_tampered_row["postcheck_result_digest"] = (
                postcheck_value["result_digest"]
            )
            postcheck_tampered_row["postcheck_binding"]["record_digest"] = (
                postcheck_value["result_digest"]
            )
            postcheck_tampered_row["artifacts"]["postcheck"].update(
                {"sha256": _sha(postcheck_raw), "bytes": len(postcheck_raw)}
            )
            postcheck_tampered_row["artifact_digest"] = _object_sha(
                postcheck_tampered_row["artifacts"]
            )
            _rewrite_self_consistent_dataset_root(
                postcheck_tamper_output, postcheck_tampered_row
            )
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "postcheck record binding differs",
            ):
                selector.validate_materialized_dataset(
                    postcheck_tamper_output
                )

            payload_tamper_output = root / "dataset_qwen_payload_tamper"
            selector._publish_dataset(
                [candidate],
                output_dir=payload_tamper_output,
                exact_size=1,
                min_multi_unit=0,
                finalizer_closure={"manifest": candidate["generation_manifest"]},
                generation_shard_closure={"status": "synthetic"},
                wan_closures=[candidate["wan"]["closure"]],
                postcheck_closures=[candidate["postcheck"]["closure"]],
            )
            payload_tampered_row = json.loads(
                (payload_tamper_output / selector.MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            payload_path = (
                payload_tamper_output
                / "samples"
                / candidate["iid"]
                / "qwen_record_payload.json"
            )
            payload_value = json.loads(payload_path.read_text(encoding="utf-8"))
            payload_value["shadow_provenance"] = "0" * 64
            payload_raw = _pretty(payload_value)
            payload_path.write_bytes(payload_raw)
            forged_payload_sha = _object_sha(payload_value)
            payload_tampered_row["qwen_record_payload_sha256"] = (
                forged_payload_sha
            )
            for binding_name in ("generation_binding", "postcheck_binding"):
                payload_tampered_row[binding_name][
                    "qwen_record_payload_sha256"
                ] = forged_payload_sha
            payload_tampered_row["artifacts"]["qwen_record_payload"].update(
                {"sha256": _sha(payload_raw), "bytes": len(payload_raw)}
            )
            payload_tampered_row["artifact_digest"] = _object_sha(
                payload_tampered_row["artifacts"]
            )
            _rewrite_self_consistent_dataset_root(
                payload_tamper_output, payload_tampered_row
            )
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "generation row artifact binding differs",
            ):
                selector.validate_materialized_dataset(payload_tamper_output)

            tamper_output = root / "dataset_self_consistent_tamper"
            selector._publish_dataset(
                [candidate],
                output_dir=tamper_output,
                exact_size=1,
                min_multi_unit=0,
                finalizer_closure={"manifest": candidate["generation_manifest"]},
                generation_shard_closure={"status": "synthetic"},
                wan_closures=[candidate["wan"]["closure"]],
                postcheck_closures=[candidate["postcheck"]["closure"]],
            )
            tampered_row = json.loads(
                (tamper_output / selector.MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            tampered_row["schema_version"] = (
                "motive-goku-full-motion-dataset-row-v4"
            )
            _rewrite_self_consistent_dataset_root(tamper_output, tampered_row)
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "dataset row schema differs",
            ):
                selector.validate_materialized_dataset(tamper_output)

            tampered_row["schema_version"] = selector.DATASET_ROW_SCHEMA
            real_authority_sha = tampered_row.pop(
                "coverage_authority_sha256"
            )
            _rewrite_self_consistent_dataset_root(tamper_output, tampered_row)
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "dataset row schema differs",
            ):
                selector.validate_materialized_dataset(tamper_output)

            tampered_row["coverage_authority_sha256"] = real_authority_sha
            for binding_name in ("generation_binding", "postcheck_binding"):
                tampered_row[binding_name][
                    "coverage_authority_sha256"
                ] = "0" * 64
            tampered_row["coverage_authority_sha256"] = "0" * 64
            _rewrite_self_consistent_dataset_root(tamper_output, tampered_row)
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "motion_spec object binding differs",
            ):
                selector.validate_materialized_dataset(tamper_output)

            tampered_row["coverage_authority_sha256"] = real_authority_sha
            for binding_name in ("generation_binding", "postcheck_binding"):
                tampered_row[binding_name][
                    "coverage_authority_sha256"
                ] = real_authority_sha
            real_i0_sha = tampered_row["i0_grounding_sha256"]
            tampered_row["i0_grounding_sha256"] = "0" * 64
            tampered_row["generation_binding"][
                "i0_grounding_sha256"
            ] = "0" * 64
            _rewrite_self_consistent_dataset_root(tamper_output, tampered_row)
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "motion_spec object binding differs",
            ):
                selector.validate_materialized_dataset(tamper_output)

            tampered_row["i0_grounding_sha256"] = real_i0_sha
            tampered_row["generation_binding"][
                "i0_grounding_sha256"
            ] = real_i0_sha
            generation_path = tamper_output / "samples" / candidate["iid"] / (
                "generation_row.json"
            )
            generation_value = json.loads(
                generation_path.read_text(encoding="utf-8")
            )
            generation_value["qwen_evidence"]["record_schema_version"] = (
                "goku-full-motion-qwen-record-v4"
            )
            generation_raw = _pretty(generation_value)
            generation_path.write_bytes(generation_raw)
            tampered_row["artifacts"]["generation_row"].update(
                {"sha256": _sha(generation_raw), "bytes": len(generation_raw)}
            )
            tampered_row["artifact_digest"] = _object_sha(
                tampered_row["artifacts"]
            )
            _rewrite_self_consistent_dataset_root(tamper_output, tampered_row)
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error,
                "generation row artifact validation failed",
            ):
                selector.validate_materialized_dataset(tamper_output)

            with self.assertRaises(FileExistsError):
                selector._publish_dataset(
                    [candidate],
                    output_dir=output,
                    exact_size=1,
                    min_multi_unit=0,
                    finalizer_closure={},
                    generation_shard_closure={},
                    wan_closures=[],
                    postcheck_closures=[],
                )

            (sample / "target.mp4").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                selector.GokuFullMotionSelect128Error, "file SHA differs"
            ):
                selector.validate_materialized_dataset(output)


if __name__ == "__main__":
    unittest.main()
