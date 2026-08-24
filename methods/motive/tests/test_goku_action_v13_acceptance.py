from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import unittest
from unittest import mock

from motive import goku_action_v13_acceptance as acceptance
from motive.goku_action_anchor_finalize import finalize_action_anchors
from motive import goku_action_anchor_qwen as qwen_module
from methods.motive.tests.test_goku_action_anchor_finalize import (
    _make_judge_a_reject,
    _refresh_shard_receipt,
    _write_fixture,
    _write_real_repair_fixture,
)


REPO = Path(__file__).resolve().parents[3]
QWEN_SOURCE = (
    REPO / "methods" / "motive" / "motive" / "goku_action_anchor_qwen.py"
)
FINALIZER_SOURCE = (
    REPO / "methods" / "motive" / "motive" / "goku_action_anchor_finalize.py"
)
VERIFIER_SOURCE = (
    REPO / "methods" / "motive" / "motive"
    / "goku_action_v13_acceptance.py"
)
SBATCH_SOURCE = (
    REPO / "methods" / "motive" / "scripts"
    / "auh_goku_action_anchor_qwen.sbatch"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact(path: Path, value: object) -> None:
    path.write_bytes(acceptance._canonical_bytes(value) + b"\n")


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_source_snapshot(
    root: Path,
    *,
    model_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    snapshot = root / "source_snapshot"
    sources = (
        QWEN_SOURCE,
        FINALIZER_SOURCE,
        VERIFIER_SOURCE,
        SBATCH_SOURCE,
    )
    rows: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(REPO)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o444)
        rows.append(
            {
                "mode": "0444",
                "path": relative.as_posix(),
                "sha256": _sha(target),
                "size": target.stat().st_size,
                "type": "file",
            }
        )
    model_files = sorted(
        path for path in model_path.rglob("*") if path.is_file()
    )
    closure_value = {
        "schema_version": acceptance.MODEL_CLOSURE_SCHEMA,
        "model_id": acceptance.EXPECTED_MODEL_ID,
        "revision": acceptance.EXPECTED_MODEL_REVISION,
        "model_path": "/frozen/models/test-qwen",
        "hash_algorithm": "sha256",
        "file_count": len(model_files),
        "total_bytes": sum(path.stat().st_size for path in model_files),
        "files": [
            {
                "relative_path": path.relative_to(model_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in model_files
        ],
    }
    closure_path = snapshot / acceptance.MODEL_CLOSURE_CANONICAL_RELPATH
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    closure_path.write_text(
        json.dumps(closure_value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    closure_path.chmod(0o444)
    closure_relative = closure_path.relative_to(snapshot)
    rows.append(
        {
            "mode": "0444",
            "path": closure_relative.as_posix(),
            "sha256": _sha(closure_path),
            "size": closure_path.stat().st_size,
            "type": "file",
        }
    )
    rows.sort(key=lambda item: str(item["path"]))
    manifest = snapshot / "SOURCE_FILES.jsonl"
    manifest_raw = b"".join(
        acceptance._canonical_bytes(row) + b"\n" for row in rows
    )
    manifest.write_bytes(manifest_raw)
    manifest.chmod(0o444)
    manifest_sha = _sha(manifest)
    tree_sha = hashlib.sha256(manifest_raw).hexdigest()
    provenance = {
        "schema": "motive-action-source-snapshot-v1",
        "created_at_utc": "2026-07-30T00:00:00+00:00",
        "repo_root": str(REPO),
        "source_roots": ["methods/motive"],
        "source_file_count": len(rows),
        "source_tree_sha256": tree_sha,
        "source_manifest_sha256": manifest_sha,
        "git_base_commit": "unit-test",
        "git_status_short": [],
    }
    provenance_path = snapshot / "SOURCE_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance_path.chmod(0o444)
    for directory, _, _ in os.walk(snapshot, topdown=False):
        Path(directory).chmod(0o555)

    archive = root / "source_snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(snapshot, arcname=snapshot.name, recursive=True)
    contract: dict[str, object] = {
        "path": str(snapshot.resolve()),
        "tree_sha256": tree_sha,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "archive_path": str(archive.resolve()),
        "archive_sha256": _sha(archive),
        "qwen_relpath": str(QWEN_SOURCE.relative_to(REPO)),
        "qwen_implementation_sha256": _sha(QWEN_SOURCE),
        "finalizer_relpath": str(FINALIZER_SOURCE.relative_to(REPO)),
        "finalizer_implementation_sha256": _sha(FINALIZER_SOURCE),
        "verifier_relpath": str(VERIFIER_SOURCE.relative_to(REPO)),
        "verifier_implementation_sha256": _sha(VERIFIER_SOURCE),
        "sbatch_relpath": str(SBATCH_SOURCE.relative_to(REPO)),
        "sbatch_sha256": _sha(SBATCH_SOURCE),
    }
    closure_contract: dict[str, object] = {
        "path": str(closure_path.resolve()),
        "sha256": _sha(closure_path),
        "file_count": closure_value["file_count"],
        "total_bytes": closure_value["total_bytes"],
    }
    return snapshot, archive, contract, closure_contract


def _gold_value(
    selected: Path,
    rows: list[dict[str, object]],
    *,
    inadmissible_iids: set[str],
) -> dict[str, object]:
    raw = selected.read_bytes()
    iids = [str(row["iid"]) for row in rows]
    return {
        "schema_version": acceptance.SMOKE_GOLD_SCHEMA,
        "gold_authority": "codex_visual_audit_not_generation_approval",
        "review_method": (
            "manual inspection of exact I0 and chronological source frames"
        ),
        "reviewed_at_utc": "2026-07-30T20:06:15Z",
        "semantic_contract_policy": dict(
            acceptance.EXPECTED_SEMANTIC_CONTRACT_POLICY
        ),
        "policy": {
            "admissible": "Substantive new motion is executable from I0.",
            "inadmissible": "The request is not a new executable motion.",
            "writer_route_is_not_a_gold_label": True,
            "positive_acceptance": (
                "Judge A and final Judge B pass by either valid route."
            ),
            "negative_acceptance": (
                "Judge A rejects before writer or Judge B."
            ),
            "wan_generation_authorized": False,
        },
        "parent_selected": {
            "path": "/frozen/parent-selected.jsonl",
            "sha256": "a" * 64,
            "rows": len(rows),
            "bytes": len(raw),
        },
        "selected_smoke": {
            "relative_path": "audits/synthetic-selected.jsonl",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(rows),
            "bytes": len(raw),
            "ordered_iids_sha256": acceptance._ordered_iids_digest(iids),
            "iid_set_sha256": acceptance._iid_set_digest(iids),
            "num_shards": 8,
            "expected_shard_rows": [
                sum(acceptance._iid_shard(iid) == index for iid in iids)
                for index in range(8)
            ],
        },
        "labels": [
            {
                "iid": iid,
                "label": (
                    "inadmissible"
                    if iid in inadmissible_iids
                    else "admissible"
                ),
                "target_contract": {
                    "schema_version": acceptance.TARGET_CONTRACT_SCHEMA,
                    "instruction_sha256": hashlib.sha256(
                        str(rows[index]["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "expected_target_change_class": (
                        "source_action_restatement"
                        if iid in inadmissible_iids
                        else "new_articulated_action"
                    ),
                    "expected_source_target_relation": (
                        "repeats_source_future"
                        if iid in inadmissible_iids
                        else "novel_future"
                    ),
                    "expected_atomic_tuple": (
                        {
                            "target_already_true": "yes",
                            "target_start_state_visually_verifiable": "yes",
                            "prerequisite_grounded": "yes",
                            "novel_trajectory": "no",
                            "scalar_or_endpoint_only": "no",
                        }
                        if iid in inadmissible_iids
                        else {
                            "target_already_true": "no",
                            "target_start_state_visually_verifiable": "yes",
                            "prerequisite_grounded": "yes",
                            "novel_trajectory": "yes",
                            "scalar_or_endpoint_only": "no",
                        }
                    ),
                    "target_token_groups": [
                        {
                            "group_id": "action",
                            "any_of": (
                                [["move", "forward"]]
                                if iid in inadmissible_iids
                                else [
                                    [
                                        "locomotion",
                                        "verb",
                                        str(index),
                                    ]
                                ]
                            ),
                        }
                    ],
                },
                "reason_code": (
                    "source_future_restatement"
                    if iid in inadmissible_iids
                    else "new_motion_from_visible_i0"
                ),
                "visual_evidence": (
                    "Chronological frames establish the binary gold decision."
                ),
            }
            for index, iid in enumerate(iids)
        ],
        "quarantine_stress_iids_not_in_gating_smoke": [
            {
                "iid": "synthetic-quarantine-only",
                "reason": "Ontology-boundary stress case excluded from gating.",
            }
        ],
    }


def _small_model_closure_fixture(
    root: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        '{"model_type":"unit"}\n', encoding="utf-8"
    )
    (model / "weights.bin").write_bytes(b"unit-weights")
    files = sorted(path for path in model.rglob("*") if path.is_file())
    manifest_value = {
        "schema_version": acceptance.MODEL_CLOSURE_SCHEMA,
        "model_id": acceptance.EXPECTED_MODEL_ID,
        "revision": acceptance.EXPECTED_MODEL_REVISION,
        "model_path": str(model.resolve()),
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [
            {
                "relative_path": path.relative_to(model).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in files
        ],
    }
    snapshot = root / "snapshot"
    manifest = snapshot / acceptance.MODEL_CLOSURE_CANONICAL_RELPATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(manifest_value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    binding: dict[str, object] = {
        "path": str(manifest.resolve()),
        "sha256": _sha(manifest),
        "file_count": len(files),
        "total_bytes": manifest_value["total_bytes"],
    }
    model_contract: dict[str, object] = {
        "path": str(model.resolve()),
        "config_path": str((model / "config.json").resolve()),
        "config_sha256": _sha(model / "config.json"),
    }
    return snapshot, model, binding, model_contract


class GokuActionV16AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        run = cls.root / "run"
        run.mkdir()
        cls.selected, cls.qwen_root, selected_rows = _write_fixture(
            run, row_count=4, medium_index=0
        )
        cls.selected_rows = selected_rows
        cls.negative_iids = {
            str(selected_rows[2]["iid"]),
            str(selected_rows[3]["iid"]),
        }
        for row in selected_rows[2:]:
            iid = str(row["iid"])
            shard_index = acceptance._iid_shard(iid)
            shard = (
                cls.qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
            )
            shard_rows = _jsonl(shard)
            record = next(item for item in shard_rows if item["iid"] == iid)
            _make_judge_a_reject(record, row)
            shard.write_bytes(
                b"".join(
                    acceptance._canonical_bytes(item) + b"\n"
                    for item in shard_rows
                )
            )
            _refresh_shard_receipt(cls.qwen_root, shard_index)

        cls.final = run / "final"
        finalize_action_anchors(
            selected_path=cls.selected,
            qwen_root=cls.qwen_root,
            output_dir=cls.final,
            seed=260730,
            allow_partial=True,
        )
        cls.gold_path = (
            cls.root / acceptance.SMOKE_GOLD_CANONICAL_RELPATH
        )
        cls.gold_path.parent.mkdir(parents=True)
        _compact(
            cls.gold_path,
            _gold_value(
                cls.selected,
                selected_rows,
                inadmissible_iids=cls.negative_iids,
            ),
        )
        cls.model_dir = cls.root / "model"
        cls.model_dir.mkdir()
        cls.model_config = cls.model_dir / "config.json"
        cls.model_config.write_text(
            '{"model_type":"qwen3_vl"}\n', encoding="utf-8"
        )
        (cls.model_dir / "tokenizer.json").write_text(
            '{"unit_test":true}\n', encoding="utf-8"
        )
        (
            cls.snapshot,
            cls.archive,
            source_contract,
            model_closure_contract,
        ) = _write_source_snapshot(
            cls.root,
            model_path=cls.model_dir,
        )
        model_contract = {
            "path": "/frozen/models/test-qwen",
            "config_path": str(cls.model_config.resolve()),
            "config_sha256": _sha(cls.model_config),
        }

        cls.submission_path = cls.root / "submission.json"
        submission = {
            "schema_version": acceptance.SUBMISSION_CONTRACT_SCHEMA,
            "selected": {
                "path": str(cls.selected.resolve()),
                "sha256": _sha(cls.selected),
                "rows": len(selected_rows),
            },
            "smoke_gold": {
                "path": str(cls.gold_path.resolve()),
                "sha256": _sha(cls.gold_path),
            },
            "source_snapshot": source_contract,
            "model": model_contract,
            "model_closure": model_closure_contract,
            "runtime": {
                **acceptance.EXPECTED_QWEN_EXECUTION,
                "final_seed": 260730,
                "allow_partial": True,
            },
            "outputs": {
                "qwen_root": str(cls.qwen_root.resolve()),
                "final_output": str(cls.final.resolve()),
            },
        }
        _compact(cls.submission_path, submission)

        cls.completion_path = cls.root / "completion.json"
        completion_shards = []
        for index in range(8):
            shard = cls.qwen_root / f"qwen_shard_{index:03d}.jsonl"
            receipt = (
                cls.qwen_root / f"qwen_shard_{index:03d}.receipt.json"
            )
            completion_shards.append(
                {
                    "index": index,
                    "path": str(shard.resolve()),
                    "sha256": _sha(shard),
                    "bytes": shard.stat().st_size,
                    "receipt_path": str(receipt.resolve()),
                    "receipt_sha256": _sha(receipt),
                }
            )
        final_artifacts = {
            name: {
                "path": str((cls.final / name).resolve()),
                "sha256": _sha(cls.final / name),
                "bytes": (cls.final / name).stat().st_size,
            }
            for name in acceptance.FINAL_NAMES
        }
        completion = {
            "schema_version": acceptance.COMPLETION_RECEIPT_SCHEMA,
            "status": "complete",
            "job_id": "unit-test-v16",
            "submission_contract_path": str(cls.submission_path.resolve()),
            "submission_contract_sha256": _sha(cls.submission_path),
            "selected_sha256": _sha(cls.selected),
            "smoke_gold_sha256": _sha(cls.gold_path),
            "model_closure": model_closure_contract,
            "qwen_root": str(cls.qwen_root.resolve()),
            "final_output": str(cls.final.resolve()),
            "qwen_shards": completion_shards,
            "final_artifacts": final_artifacts,
        }
        _compact(cls.completion_path, completion)

        gold = json.loads(cls.gold_path.read_text(encoding="utf-8"))
        cls.contract_path = cls.root / "acceptance.json"
        contract = {
            "schema_version": acceptance.ACCEPTANCE_CONTRACT_SCHEMA,
            "selected": {
                "rows": len(selected_rows),
                "sha256": _sha(cls.selected),
                "ordered_iids_sha256": gold["selected_smoke"][
                    "ordered_iids_sha256"
                ],
            },
            "smoke_gold": {
                "path": str(cls.gold_path.resolve()),
                "sha256": _sha(cls.gold_path),
            },
            "expected_shard_counts": gold["selected_smoke"][
                "expected_shard_rows"
            ],
            "source_snapshot": source_contract,
            "model": model_contract,
            "model_closure": model_closure_contract,
            "execution": dict(acceptance.EXPECTED_QWEN_EXECUTION),
            "final": dict(acceptance.EXPECTED_FINAL),
            "bindings": {
                "submission_contract_sha256": _sha(cls.submission_path),
                "completion_receipt_sha256": _sha(cls.completion_path),
            },
        }
        _compact(cls.contract_path, contract)

        mutable = [
            *cls.qwen_root.iterdir(),
            *cls.final.iterdir(),
            cls.selected,
            cls.gold_path,
            cls.submission_path,
            cls.completion_path,
            cls.contract_path,
            cls.archive,
        ]
        cls.baseline = {path: path.read_bytes() for path in mutable}
        cls.snapshot_modes = {
            path: stat.S_IMODE(path.stat().st_mode)
            for path in [cls.snapshot, *cls.snapshot.rglob("*")]
        }

    @classmethod
    def tearDownClass(cls) -> None:
        for path in sorted(
            [cls.snapshot, *cls.snapshot.rglob("*")],
            key=lambda item: len(item.parts),
        ):
            if path.exists():
                path.chmod(0o755 if path.is_dir() else 0o644)
        cls.temporary.cleanup()

    def tearDown(self) -> None:
        for path, raw in self.baseline.items():
            if path.exists():
                path.chmod(0o644)
                path.write_bytes(raw)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        for path, mode in sorted(
            self.snapshot_modes.items(),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if path.exists():
                path.chmod(mode)
        for extra in self.qwen_root.iterdir():
            if extra not in self.baseline:
                if extra.is_dir():
                    shutil.rmtree(extra)
                else:
                    extra.unlink()

    def _verify(
        self,
        *,
        gold_sha: str | None = None,
        smoke_gold_path: Path | None = None,
    ) -> dict:
        with mock.patch.multiple(
            acceptance,
            EXPECTED_SMOKE_GOLD_SHA256=gold_sha or _sha(self.gold_path),
            EXPECTED_QWEN_IMPLEMENTATION_SHA256=_sha(QWEN_SOURCE),
            EXPECTED_FINALIZER_IMPLEMENTATION_SHA256=_sha(FINALIZER_SOURCE),
            EXPECTED_SBATCH_SHA256=_sha(SBATCH_SOURCE),
            EXPECTED_MODEL_PATH="/frozen/models/test-qwen",
            EXPECTED_MODEL_CONFIG_SHA256=_sha(self.model_config),
            EXPECTED_MODEL_CLOSURE_SHA256=json.loads(
                self.contract_path.read_text(encoding="utf-8")
            )["model_closure"]["sha256"],
            EXPECTED_MODEL_CLOSURE_FILE_COUNT=2,
            EXPECTED_MODEL_CLOSURE_TOTAL_BYTES=sum(
                path.stat().st_size
                for path in self.model_dir.iterdir()
                if path.is_file()
            ),
        ):
            with mock.patch.object(
                acceptance,
                "_verify_model",
                return_value={"path": "/frozen/models/test-qwen"},
            ), mock.patch.object(
                acceptance,
                "_verify_model_closure",
                return_value={
                    "manifest_path": str(
                        self.snapshot
                        / acceptance.MODEL_CLOSURE_CANONICAL_RELPATH
                    ),
                    "file_count": 2,
                    "total_bytes": sum(
                        path.stat().st_size
                        for path in self.model_dir.iterdir()
                        if path.is_file()
                    ),
                },
            ):
                return acceptance.verify_acceptance(
                    contract_path=self.contract_path,
                    smoke_gold_path=smoke_gold_path or self.gold_path,
                    selected_path=self.selected,
                    qwen_root=self.qwen_root,
                    final_dir=self.final,
                    source_snapshot=self.snapshot,
                    submission_contract_path=self.submission_path,
                    completion_receipt_path=self.completion_path,
                )

    def test_acceptance_rejects_edited_caption_target_evidence(self) -> None:
        judge = {
            "schema_version": acceptance.TARGET_ADMISSIBILITY_SCHEMA,
            "target_change_class": "new_articulated_action",
            "source_target_relation": "novel_future",
            "target_action_normalized": "perform a new action",
            "target_action_verb": "new_action",
            "target_already_true": "no",
            "target_start_state_visually_verifiable": "yes",
            "prerequisite_grounded": "yes",
            "novel_trajectory": "yes",
            "novel_trajectory_description": "perform a new action",
            "scalar_or_endpoint_only": "no",
            "source_evidence_ref": "source_action",
            "target_evidence_ref": "edited_caption",
            "uncertainty_codes": [],
            "confidence": "high",
        }
        with self.assertRaisesRegex(
            acceptance.AcceptanceError,
            "target_evidence_ref",
        ):
            acceptance._validate_judge_a(
                judge,
                "edited-caption-selector",
            )
        with self.assertRaisesRegex(
            acceptance.AcceptanceError,
            "immutable instruction",
        ):
            acceptance._judge_a_evidence(
                judge,
                self.selected_rows[0],
                {
                    "source_action": "source action",
                    "temporal_evidence": [],
                },
                "edited-caption-selector",
            )

    def test_initial_state_selector_and_judge_a_policy_mirror(self) -> None:
        selected = self.selected_rows[0]
        iid = str(selected["iid"])
        shard = (
            self.qwen_root
            / f"qwen_shard_{acceptance._iid_shard(iid):03d}.jsonl"
        )
        record = next(row for row in _jsonl(shard) if row["iid"] == iid)
        observation = record["anchor_observation"]
        judge = copy.deepcopy(record["target_admissibility"])
        judge["source_evidence_ref"] = "initial_state"

        self.assertIs(
            qwen_module.validate_target_admissibility(judge),
            judge,
        )
        self.assertIs(
            acceptance._validate_judge_a(judge, "initial-state-selector"),
            judge,
        )
        expected_evidence = {
            "source_evidence_ref": "initial_state",
            "source_evidence": observation["initial_state"],
            "target_evidence_ref": "instruction",
            "target_evidence": selected["prompt"],
        }
        self.assertEqual(
            qwen_module.resolve_target_admissibility_evidence(
                judge,
                row=selected,
                observation=observation,
            ),
            expected_evidence,
        )
        self.assertEqual(
            acceptance._judge_a_evidence(
                judge,
                selected,
                observation,
                "initial-state-selector",
            ),
            expected_evidence,
        )

        exact_copy = copy.deepcopy(judge)
        exact_copy["novel_trajectory_description"] = exact_copy[
            "target_action_normalized"
        ]
        self.assertEqual(
            qwen_module.aggregate_target_admissibility(
                exact_copy,
                row=selected,
                observation=observation,
            ),
            acceptance._aggregate_a(
                exact_copy,
                selected,
                observation,
            ),
        )

        mismatch = copy.deepcopy(exact_copy)
        mismatch["novel_trajectory_description"] = (
            "the subject performs an unrelated circular run"
        )
        qwen_aggregate = qwen_module.aggregate_target_admissibility(
            mismatch,
            row=selected,
            observation=observation,
        )
        acceptance_aggregate = acceptance._aggregate_a(
            mismatch,
            selected,
            observation,
        )
        self.assertEqual(qwen_aggregate, acceptance_aggregate)
        self.assertEqual(qwen_aggregate["decision"], "reject")
        self.assertIn(
            "judge_a:novel_trajectory_description_target_mismatch",
            qwen_aggregate["risk_codes"],
        )

        invalid = copy.deepcopy(judge)
        invalid["source_evidence_ref"] = "initial_state:0"
        with self.assertRaisesRegex(
            qwen_module.GokuActionAnchorQwenError,
            "source_evidence_ref",
        ):
            qwen_module.validate_target_admissibility(invalid)
        with self.assertRaisesRegex(
            acceptance.AcceptanceError,
            "source_evidence_ref",
        ):
            acceptance._validate_judge_a(
                invalid,
                "invalid-initial-state-selector",
            )

    def test_judge_a_dynamic_selector_prompt_and_range_mirror(self) -> None:
        selected = self.selected_rows[0]
        iid = str(selected["iid"])
        shard = (
            self.qwen_root
            / f"qwen_shard_{acceptance._iid_shard(iid):03d}.jsonl"
        )
        record = next(row for row in _jsonl(shard) if row["iid"] == iid)
        observation = {
            **record["anchor_observation"],
            "temporal_evidence": [
                "S0: the subject is still",
                "S2: the subject starts moving",
                "S4: the subject continues",
                "S6: the subject changes direction",
                "S10: the subject passes the marker",
                "S11: the subject reaches the endpoint",
            ],
        }
        expected_allowlist = json.dumps(
            [
                "initial_state",
                "source_action",
                "temporal_evidence:0",
                "temporal_evidence:1",
                "temporal_evidence:2",
                "temporal_evidence:3",
                "temporal_evidence:4",
                "temporal_evidence:5",
            ],
            separators=(",", ":"),
        )
        qwen_prompt = qwen_module.build_target_admissibility_prompt(
            row=selected,
            observation=observation,
        )
        constants = acceptance._source_constants(QWEN_SOURCE)
        acceptance_prompt, acceptance_digest = acceptance._render_prompt(
            "judge_a",
            constants,
            row=selected,
            observation=observation,
        )
        self.assertEqual(acceptance_prompt, qwen_prompt)
        self.assertEqual(
            acceptance_digest,
            qwen_module._rendered_prompt_digest(
                qwen_module.JUDGE_A_SYSTEM,
                qwen_prompt,
            ),
        )
        self.assertIn(expected_allowlist, qwen_prompt)
        self.assertNotIn("temporal_evidence:10", qwen_prompt)
        self.assertIn("JSON array position", qwen_prompt)
        self.assertIn("not an embedded S-frame label", qwen_prompt)

        out_of_range = copy.deepcopy(record["target_admissibility"])
        out_of_range["source_evidence_ref"] = "temporal_evidence:10"
        self.assertIs(
            qwen_module.validate_target_admissibility(out_of_range),
            out_of_range,
        )
        self.assertIs(
            acceptance._validate_judge_a(
                out_of_range,
                "dynamic-selector-range",
            ),
            out_of_range,
        )
        with self.assertRaisesRegex(
            qwen_module.GokuActionAnchorQwenError,
            "out of range",
        ):
            qwen_module.resolve_target_admissibility_evidence(
                out_of_range,
                row=selected,
                observation=observation,
            )
        with self.assertRaisesRegex(
            acceptance.AcceptanceError,
            "out of range",
        ):
            acceptance._judge_a_evidence(
                out_of_range,
                selected,
                observation,
                "dynamic-selector-range",
            )

    def test_writer_prompt_and_exact_target_core_mirror_producer(self) -> None:
        records = [
            row
            for shard in sorted(self.qwen_root.glob("qwen_shard_*.jsonl"))
            for row in _jsonl(shard)
        ]
        record = next(
            row
            for row in records
            if isinstance(row.get("compatibility"), dict)
        )
        iid = str(record["iid"])
        selected = next(
            row for row in self.selected_rows if str(row["iid"]) == iid
        )
        observation = record["anchor_observation"]
        judge_a = record["target_admissibility"]
        compatibility = record["compatibility"]
        producer_prompt = qwen_module.build_compatibility_prompt(
            row=selected,
            observation=observation,
            judge_a=judge_a,
        )
        constants = acceptance._source_constants(QWEN_SOURCE)
        verifier_prompt, verifier_digest = acceptance._render_prompt(
            "writer",
            constants,
            row=selected,
            observation=observation,
            judge_a=judge_a,
        )
        self.assertEqual(verifier_prompt, producer_prompt)
        self.assertEqual(
            verifier_digest,
            qwen_module._rendered_prompt_digest(
                qwen_module.COMPATIBILITY_SYSTEM,
                producer_prompt,
            ),
        )

        producer_evidence = qwen_module.target_core_agreement_evidence(
            judge_a,
            compatibility,
            selected,
        )
        verifier_evidence = acceptance._target_core_agreement(
            judge_a,
            compatibility,
            selected,
        )
        self.assertEqual(verifier_evidence, producer_evidence)
        self.assertTrue(verifier_evidence["agreement_verified"])
        self.assertTrue(verifier_evidence["normalized_exact_match"])
        self.assertTrue(verifier_evidence["verb_exact_match"])

        for field, suffix in (
            ("target_action_normalized", " "),
            ("target_action_verb", "_changed"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(compatibility)
                changed[field] += suffix
                producer_changed = (
                    qwen_module.target_core_agreement_evidence(
                        judge_a,
                        changed,
                        selected,
                    )
                )
                verifier_changed = acceptance._target_core_agreement(
                    judge_a,
                    changed,
                    selected,
                )
                self.assertEqual(verifier_changed, producer_changed)
                self.assertFalse(
                    verifier_changed["agreement_verified"]
                )

    def _refresh_completion_binding(self, completion: dict) -> None:
        _compact(self.completion_path, completion)
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["bindings"]["completion_receipt_sha256"] = _sha(
            self.completion_path
        )
        _compact(self.contract_path, contract)

    def _resign_shard(self, index: int, rows: list[dict]) -> None:
        shard = self.qwen_root / f"qwen_shard_{index:03d}.jsonl"
        shard.write_bytes(
            b"".join(
                acceptance._canonical_bytes(row) + b"\n" for row in rows
            )
        )
        receipt_path = (
            self.qwen_root / f"qwen_shard_{index:03d}.receipt.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        counts = Counter(str(row["status"]) for row in rows)
        receipt["output"] = {
            "path": str(shard.resolve()),
            "sha256": _sha(shard),
            "bytes": shard.stat().st_size,
            "rows": len(rows),
            "status_counts": dict(sorted(counts.items())),
        }
        receipt.pop("receipt_digest", None)
        receipt["receipt_digest"] = acceptance._object_digest(receipt)
        _compact(receipt_path, receipt)
        completion = json.loads(
            self.completion_path.read_text(encoding="utf-8")
        )
        completion["qwen_shards"][index].update(
            {
                "sha256": _sha(shard),
                "bytes": shard.stat().st_size,
                "receipt_sha256": _sha(receipt_path),
            }
        )
        self._refresh_completion_binding(completion)

    def _mutate_judge_a_and_resign(
        self,
        iid: str,
        mutate,
    ) -> None:
        selected = next(
            row for row in self.selected_rows if str(row["iid"]) == iid
        )
        index = acceptance._iid_shard(iid)
        shard = self.qwen_root / f"qwen_shard_{index:03d}.jsonl"
        rows = _jsonl(shard)
        record = next(row for row in rows if row["iid"] == iid)
        judge_a = dict(record["target_admissibility"])
        mutate(judge_a)
        record["target_admissibility"] = judge_a
        record["target_admissibility_raw"] = json.dumps(judge_a)
        aggregate = qwen_module.aggregate_target_admissibility(
            judge_a,
            row=selected,
            observation=record["anchor_observation"],
        )
        record["target_admissibility_aggregate"] = aggregate
        if record["pipeline_stage"] == "judge_a":
            record["deterministic_risk_codes"] = aggregate["risk_codes"]
            record["pipeline_decision"] = aggregate["decision"]
        record["result_digest"] = acceptance._object_digest(
            acceptance._result_payload(record)
        )
        record["provenance_digest"] = acceptance._object_digest(
            acceptance._provenance_payload(record)
        )
        self._resign_shard(index, rows)

    def test_passes_semantic_gold_and_pending_v8_finalizer(self) -> None:
        result = self._verify()
        self.assertTrue(result["passed"])
        self.assertTrue(result["full_123_authorized"])
        self.assertFalse(result["generation_authorized"])
        self.assertFalse(result["production_eligible"])
        self.assertFalse(result["wan_generation_authorized"])
        self.assertFalse(result["authorization_interface_available"])
        self.assertEqual(
            result["qwen"]["route_counts"],
            {"direct": 2, "judge_a_reject": 2},
        )
        self.assertEqual(
            result["qwen"]["expected_shard_counts"],
            json.loads(self.gold_path.read_text())["selected_smoke"][
                "expected_shard_rows"
            ],
        )

    def test_gold_quarantine_report_never_authorizes_wan(self) -> None:
        result = self._verify()
        self.assertEqual(
            result["smoke_gold"][
                "quarantine_stress_iids_not_in_gating_smoke"
            ][0]["iid"],
            "synthetic-quarantine-only",
        )
        self.assertFalse(result["wan_generation_authorized"])

    def test_rejects_correct_binary_route_with_wrong_semantic_class(self) -> None:
        iid = str(self.selected_rows[0]["iid"])
        self._mutate_judge_a_and_resign(
            iid,
            lambda judge: judge.__setitem__(
                "target_change_class", "new_posture_transition"
            ),
        )
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "gold target_change_class"
        ):
            self._verify()

    def test_rejects_correct_binary_route_with_wrong_semantic_relation(
        self,
    ) -> None:
        iid = str(self.selected_rows[0]["iid"])
        self._mutate_judge_a_and_resign(
            iid,
            lambda judge: judge.__setitem__(
                "source_target_relation", "shared_base_with_novel_action"
            ),
        )
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "gold source_target_relation"
        ):
            self._verify()

    def test_rejects_correct_label_with_wrong_atomic_tuple(self) -> None:
        iid = str(self.selected_rows[2]["iid"])
        self._mutate_judge_a_and_resign(
            iid,
            lambda judge: judge.__setitem__(
                "scalar_or_endpoint_only", "yes"
            ),
        )
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "gold atomic tuple"
        ):
            self._verify()

    def test_rejects_missing_compound_target_token_group(self) -> None:
        iid = str(self.selected_rows[0]["iid"])

        def remove_compound_component(judge: dict) -> None:
            judge["target_action_normalized"] = "perform locomotion_verb"
            judge["target_action_verb"] = "locomotion_verb"
            judge["novel_trajectory_description"] = (
                "perform locomotion_verb"
            )

        self._mutate_judge_a_and_resign(iid, remove_compound_component)
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "target token groups missing"
        ):
            self._verify()

    def test_rejects_resigned_gold_mutation_against_source_anchor(self) -> None:
        value = json.loads(self.gold_path.read_text(encoding="utf-8"))
        value["labels"][0]["reason_code"] = "resigned_gold_mutation"
        _compact(self.gold_path, value)
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["smoke_gold"]["sha256"] = _sha(self.gold_path)
        _compact(self.contract_path, contract)
        submission = json.loads(
            self.submission_path.read_text(encoding="utf-8")
        )
        submission["smoke_gold"]["sha256"] = _sha(self.gold_path)
        _compact(self.submission_path, submission)
        completion = json.loads(
            self.completion_path.read_text(encoding="utf-8")
        )
        completion["submission_contract_sha256"] = _sha(
            self.submission_path
        )
        completion["smoke_gold_sha256"] = _sha(self.gold_path)
        self._refresh_completion_binding(completion)
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["bindings"]["submission_contract_sha256"] = _sha(
            self.submission_path
        )
        _compact(self.contract_path, contract)
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "non-resignable source trust anchor"
        ):
            self._verify(
                gold_sha=hashlib.sha256(
                    self.baseline[self.gold_path]
                ).hexdigest()
            )

    def test_rejects_label_flip_even_under_hypothetical_new_anchor(self) -> None:
        value = json.loads(self.gold_path.read_text(encoding="utf-8"))
        value["labels"][0]["label"] = "inadmissible"
        _compact(self.gold_path, value)
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["smoke_gold"]["sha256"] = _sha(self.gold_path)
        _compact(self.contract_path, contract)
        submission = json.loads(
            self.submission_path.read_text(encoding="utf-8")
        )
        submission["smoke_gold"]["sha256"] = _sha(self.gold_path)
        _compact(self.submission_path, submission)
        completion = json.loads(
            self.completion_path.read_text(encoding="utf-8")
        )
        completion["submission_contract_sha256"] = _sha(
            self.submission_path
        )
        completion["smoke_gold_sha256"] = _sha(self.gold_path)
        self._refresh_completion_binding(completion)
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["bindings"]["submission_contract_sha256"] = _sha(
            self.submission_path
        )
        _compact(self.contract_path, contract)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "route"):
            self._verify(gold_sha=_sha(self.gold_path))

    def test_rejects_negative_to_positive_label_flip(self) -> None:
        value = json.loads(self.gold_path.read_text(encoding="utf-8"))
        negative = next(
            item for item in value["labels"]
            if item["label"] == "inadmissible"
        )
        negative["label"] = "admissible"
        _compact(self.gold_path, value)
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["smoke_gold"]["sha256"] = _sha(self.gold_path)
        _compact(self.contract_path, contract)
        submission = json.loads(
            self.submission_path.read_text(encoding="utf-8")
        )
        submission["smoke_gold"]["sha256"] = _sha(self.gold_path)
        _compact(self.submission_path, submission)
        completion = json.loads(
            self.completion_path.read_text(encoding="utf-8")
        )
        completion["submission_contract_sha256"] = _sha(
            self.submission_path
        )
        completion["smoke_gold_sha256"] = _sha(self.gold_path)
        self._refresh_completion_binding(completion)
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["bindings"]["submission_contract_sha256"] = _sha(
            self.submission_path
        )
        _compact(self.contract_path, contract)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "route"):
            self._verify(gold_sha=_sha(self.gold_path))

    def test_rejects_gold_argument_path_swap(self) -> None:
        swapped = self.root / "swapped-gold.json"
        swapped.write_bytes(self.gold_path.read_bytes())
        try:
            with self.assertRaisesRegex(
                acceptance.AcceptanceError, "canonical v16 relative path"
            ):
                self._verify(smoke_gold_path=swapped)
        finally:
            swapped.unlink(missing_ok=True)

    def test_rejects_tampered_resolved_evidence_after_resigning(self) -> None:
        positive = next(
            row
            for row in self.selected_rows
            if str(row["iid"]) not in self.negative_iids
        )
        iid = str(positive["iid"])
        index = acceptance._iid_shard(iid)
        shard = self.qwen_root / f"qwen_shard_{index:03d}.jsonl"
        rows = _jsonl(shard)
        record = next(row for row in rows if row["iid"] == iid)
        record["target_admissibility_resolved_evidence"][
            "source_evidence"
        ] = "tampered resolved source evidence"
        record["result_digest"] = acceptance._object_digest(
            acceptance._result_payload(record)
        )
        record["provenance_digest"] = acceptance._object_digest(
            acceptance._provenance_payload(record)
        )
        self._resign_shard(index, rows)
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "resolved evidence"
        ):
            self._verify()

    def test_rejects_frame_metadata_tamper_after_resigning(self) -> None:
        iid = str(self.selected_rows[0]["iid"])
        index = acceptance._iid_shard(iid)
        shard = self.qwen_root / f"qwen_shard_{index:03d}.jsonl"
        rows = _jsonl(shard)
        record = next(row for row in rows if row["iid"] == iid)
        record["media_verification"]["width"] += 1
        record["provenance_digest"] = acceptance._object_digest(
            acceptance._provenance_payload(record)
        )
        self._resign_shard(index, rows)
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "width"
        ):
            self._verify()

    def test_rejects_extra_qwen_root_file(self) -> None:
        (self.qwen_root / "evil.txt").write_text(
            "not bound\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "exact file closure"
        ):
            self._verify()

    def test_contract_rejects_legacy_fixed_routes(self) -> None:
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["routes"] = {
            "direct": [],
            "repair_once": [],
            "judge_a_reject": [],
        }
        _compact(self.contract_path, contract)
        with self.assertRaisesRegex(
            acceptance.UnauditableError, "closed keys differ"
        ):
            self._verify()

    def test_contract_rejects_old_v13_schema(self) -> None:
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        contract["schema_version"] = (
            "motive-goku-action-v13-acceptance-contract-v1"
        )
        _compact(self.contract_path, contract)
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "acceptance contract schema"
        ):
            self._verify()

    def test_route_classifier_is_iid_independent_and_permutable(self) -> None:
        direct = {
            "compatibility_validated_from": "original",
            "compatibility_semantic_repairs": [],
        }
        repaired = {
            "compatibility_validated_from": "semantic_repair_1",
            "compatibility_semantic_repairs": [{"attempt": 1}],
        }
        self.assertEqual(
            acceptance._positive_route(direct, context="permutation-a"),
            "direct",
        )
        self.assertEqual(
            acceptance._positive_route(repaired, context="permutation-b"),
            "repair_once",
        )
        self.assertEqual(
            acceptance._positive_route(repaired, context="permutation-a"),
            "repair_once",
        )
        self.assertEqual(
            acceptance._positive_route(direct, context="permutation-b"),
            "direct",
        )

    def test_real_repair_route_is_verified_without_gold_route_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, selected_rows = _write_real_repair_fixture(
                root, iid="synthetic-permutable-repair"
            )
            row = selected_rows[0]
            iid = str(row["iid"])
            shard = (
                qwen_root
                / f"qwen_shard_{acceptance._iid_shard(iid):03d}.jsonl"
            )
            [record] = _jsonl(shard)
            self.assertEqual(
                acceptance._positive_route(record, context=iid),
                "repair_once",
            )
            constants = acceptance._source_constants(QWEN_SOURCE)
            observation = acceptance._validate_observation(
                record["anchor_observation"], iid
            )
            judge_a = acceptance._validate_judge_a(
                record["target_admissibility"], iid
            )
            repaired = acceptance._verify_repair_route(
                record,
                selected=row,
                observation=observation,
                judge_a=judge_a,
                constants=constants,
            )
            self.assertEqual(
                acceptance._target_core(repaired),
                acceptance._target_core(
                    record["compatibility_semantic_repairs"][0][
                        "draft_compatibility"
                    ]
                ),
            )

    def test_source_has_no_legacy_iid_or_fixed_route_matrix(self) -> None:
        source = VERIFIER_SOURCE.read_text(encoding="utf-8")
        forbidden = (
            "EXPECTED_" + "DIRECT",
            "EXPECTED_" + "REPAIR",
            "EXPECTED_" + "REJECT",
            "V" + "12_SMOKE",
            "3-" + "direct",
            "3-" + "repair",
            "10-" + "reject",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_prompt_mutation_cannot_reuse_frozen_instruction_contract(
        self,
    ) -> None:
        gold = json.loads(self.gold_path.read_text(encoding="utf-8"))
        labels = {str(item["iid"]): item for item in gold["labels"]}
        rows = [dict(row) for row in self.selected_rows]
        rows[0]["prompt"] = str(rows[0]["prompt"]) + " Changed."
        raw = b"".join(
            acceptance._canonical_bytes(row) + b"\n" for row in rows
        )
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "immutable instruction SHA"
        ):
            acceptance._bind_gold_to_selected(
                gold,
                labels,
                selected_rows=rows,
                selected_raw=raw,
            )

    def test_semantic_token_policy_is_nfkc_casefold_and_contiguous(
        self,
    ) -> None:
        tokens = acceptance._semantic_contract_tokens(
            "LOOK_Back—At，THE＿Camera!"
        )
        self.assertEqual(
            tokens,
            ["look", "back", "at", "the", "camera"],
        )
        self.assertTrue(
            acceptance._contains_contiguous_tokens(
                tokens, ["back", "at", "the"]
            )
        )
        self.assertFalse(
            acceptance._contains_contiguous_tokens(
                tokens, ["look", "camera"]
            )
        )

    def test_target_contract_cannot_be_omitted_or_filled_from_result(
        self,
    ) -> None:
        value = json.loads(self.gold_path.read_text(encoding="utf-8"))
        value["labels"][0].pop("target_contract")
        _compact(self.gold_path, value)
        with mock.patch.object(
            acceptance,
            "EXPECTED_SMOKE_GOLD_SHA256",
            _sha(self.gold_path),
        ):
            with self.assertRaisesRegex(
                acceptance.UnauditableError, "closed keys differ"
            ):
                acceptance._load_smoke_gold(self.gold_path)

    def test_submission_has_no_approval_interface(self) -> None:
        submission = json.loads(
            self.submission_path.read_text(encoding="utf-8")
        )
        self.assertNotIn("approval_path", submission["runtime"])
        submission["runtime"]["approval_path"] = None
        _compact(self.submission_path, submission)
        with self.assertRaisesRegex(
            acceptance.UnauditableError, "closed keys differ"
        ):
            self._verify()

    def test_pending_outputs_expose_no_authorization_interface(self) -> None:
        result = self._verify()
        self.assertFalse(result["generation_authorized"])
        self.assertFalse(result["production_eligible"])
        self.assertFalse(result["wan_generation_authorized"])
        summary = json.loads(
            (self.final / "summary.json").read_text(encoding="utf-8")
        )
        self.assertFalse(
            summary["semantics"]["authorization_interface_available"]
        )
        for name in (
            "review_candidates.jsonl",
            "proposed_128.jsonl",
            "generation_manifest.jsonl",
        ):
            for row in _jsonl(self.final / name):
                finalization = row.get("action_anchor_finalization", row)
                self.assertFalse(
                    finalization["authorization_interface_available"]
                )
                self.assertFalse(finalization["generation_authorized"])
                self.assertFalse(finalization["production_eligible"])
                self.assertIsNone(finalization["approval"])

    def test_stale_v15_and_finalizer_hashes_are_rejected(self) -> None:
        self.assertEqual(
            acceptance.EXPECTED_QWEN_IMPLEMENTATION_SHA256,
            "f5535e0f68e515609a1b578b494197ae0c45a5ca79030ba9ceaa25ba0d7b772e",
        )
        self.assertEqual(
            acceptance.EXPECTED_FINALIZER_IMPLEMENTATION_SHA256,
            "63d98952f400dd30a069fee72f169a2d512b8d3b0b9b7c4779475663e26758e3",
        )
        for context, digest in (
            (
                "stale-v15",
                "3678586e4f33af88e1452fb0345f3c3d9c35caf090be65bba6af3639b3d2638d",
            ),
            (
                "superseded-v16-v8",
                "d626f1d7297081e48620ab9ef49223236b015406c0e421b6c090f0e42b2a2210",
            ),
            (
                "superseded-auto-attention-runtime",
                "6e0d80f6e436eb127ae0b2d6d718ab2173245826125e5b628915c774a5ef8d3a",
            ),
        ):
            with self.subTest(context=context):
                source = json.loads(
                    self.contract_path.read_text(encoding="utf-8")
                )["source_snapshot"]
                source["finalizer_implementation_sha256"] = digest
                with self.assertRaisesRegex(
                    acceptance.AcceptanceError,
                    "finalizer_implementation_sha256",
                ):
                    acceptance._validate_source_snapshot_shape(
                        source, context=context
                    )

    def test_old_qwen25_7b_trust_anchors_are_rejected(self) -> None:
        source = json.loads(
            self.contract_path.read_text(encoding="utf-8")
        )["source_snapshot"]
        stale_source_hashes = {
            "qwen_implementation_sha256": (
                "59c3c9de6eb9e4b2f3aad631e00359bb6195bd93436d8a22edc20ae64dfd13dc"
            ),
            "sbatch_sha256": (
                "c4e4ebfec4e7f8eb11829f8268c939ba5b611d58d611e075dd6280c9487214d7"
            ),
        }
        for field, stale_value in stale_source_hashes.items():
            with self.subTest(field=field):
                mutated = dict(source)
                mutated[field] = stale_value
                with self.assertRaisesRegex(
                    acceptance.AcceptanceError,
                    field,
                ):
                    acceptance._validate_source_snapshot_shape(
                        mutated, context="old-qwen25-7b"
                    )

        for field, stale_value in (
            ("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"),
            ("revision", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    snapshot, _, binding, model_contract = (
                        _small_model_closure_fixture(root)
                    )
                    manifest = (
                        snapshot
                        / acceptance.MODEL_CLOSURE_CANONICAL_RELPATH
                    )
                    value = json.loads(
                        manifest.read_text(encoding="utf-8")
                    )
                    value[field] = stale_value
                    manifest.write_text(
                        json.dumps(value, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    binding["sha256"] = _sha(manifest)
                    with self.assertRaisesRegex(
                        acceptance.AcceptanceError,
                        f"model closure {field}",
                    ):
                        acceptance._verify_model_closure(
                            binding,
                            source_snapshot=snapshot,
                            model_contract=model_contract,
                        )

    def test_model_closure_rehashes_and_rejects_all_closure_attacks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, model, binding, model_contract = (
                _small_model_closure_fixture(root)
            )
            result = acceptance._verify_model_closure(
                binding,
                source_snapshot=snapshot,
                model_contract=model_contract,
            )
            self.assertEqual(result["file_count"], 2)
            (model / "weights.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                acceptance.AcceptanceError, "weights.bin (?:bytes|SHA)"
            ):
                acceptance._verify_model_closure(
                    binding,
                    source_snapshot=snapshot,
                    model_contract=model_contract,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, model, binding, model_contract = (
                _small_model_closure_fixture(root)
            )
            (model / "extra.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(
                acceptance.AcceptanceError, "exact file set"
            ):
                acceptance._verify_model_closure(
                    binding,
                    source_snapshot=snapshot,
                    model_contract=model_contract,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, model, binding, model_contract = (
                _small_model_closure_fixture(root)
            )
            (model / "weights.bin").unlink()
            with self.assertRaisesRegex(
                acceptance.AcceptanceError, "exact file set"
            ):
                acceptance._verify_model_closure(
                    binding,
                    source_snapshot=snapshot,
                    model_contract=model_contract,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, _, binding, model_contract = (
                _small_model_closure_fixture(root)
            )
            swapped = (
                root / "swapped"
                / acceptance.MODEL_CLOSURE_CANONICAL_RELPATH
            )
            swapped.parent.mkdir(parents=True)
            source_manifest = (
                snapshot / acceptance.MODEL_CLOSURE_CANONICAL_RELPATH
            )
            swapped.write_bytes(source_manifest.read_bytes())
            binding["path"] = str(swapped.resolve())
            with self.assertRaisesRegex(
                acceptance.AcceptanceError, "canonical snapshot path"
            ):
                acceptance._verify_model_closure(
                    binding,
                    source_snapshot=snapshot,
                    model_contract=model_contract,
                )

    def test_selected_media_bytes_are_rehashed(self) -> None:
        media_root = self.root / "media-unit"
        media_root.mkdir(exist_ok=True)
        source = media_root / "source.mp4"
        anchor = media_root / "anchor.png"
        source.write_bytes(b"source-v1")
        anchor.write_bytes(b"anchor-v1")
        row = {
            "iid": "unit-media",
            "resolved_src_video": str(source.resolve()),
            "resolved_anchor_image": str(anchor.resolve()),
            "source_video_sha256": _sha(source),
            "anchor_sha256": _sha(anchor),
        }
        self.assertEqual(
            acceptance._verify_selected_media([row])["rows"], 1
        )
        source.write_bytes(b"source-v2")
        with self.assertRaisesRegex(
            acceptance.AcceptanceError, "source media SHA"
        ):
            acceptance._verify_selected_media([row])
        shutil.rmtree(media_root)


if __name__ == "__main__":
    unittest.main()
