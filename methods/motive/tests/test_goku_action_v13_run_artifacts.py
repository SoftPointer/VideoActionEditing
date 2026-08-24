from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest import mock

from motive import goku_action_v13_run_artifacts as artifacts
from motive import goku_action_v13_acceptance as independent_acceptance


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class GokuActionV13RunArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        audit_root = (
            Path(__file__).resolve().parents[1] / "audits"
        )
        self.gold_raw = (
            audit_root / "goku_action_v16_smoke_gold.json"
        ).read_bytes()
        self.gold_value = json.loads(self.gold_raw)
        self.selected_raw = (
            audit_root / "goku_action_v15_smoke_selected.jsonl"
        ).read_bytes()
        self.selected_rows = [
            json.loads(line) for line in self.selected_raw.splitlines()
        ]
        self.ordered_iids = [
            str(row["iid"]) for row in self.selected_rows
        ]
        self.shard_counts = [
            sum(artifacts._iid_shard(iid) == index for iid in self.ordered_iids)
            for index in range(8)
        ]
        self.admissible_iids = [
            str(item["iid"])
            for item in self.gold_value["labels"]
            if item["label"] == "admissible"
        ]
        self.selected_relative = self.gold_value["selected_smoke"][
            "relative_path"
        ]
        (
            self.model,
            self.model_config_sha256,
        ) = self._make_model()
        self.model_closure_value = self._make_model_closure()
        self.model_closure_raw = _canonical(
            self.model_closure_value
        )
        self.patchers: tuple[mock._patch, ...] = (
            mock.patch.object(
                artifacts,
                "FROZEN_SMOKE_GOLD_SHA256",
                _sha256(self.gold_raw),
            ),
            mock.patch.object(
                artifacts,
                "FROZEN_MODEL_CLOSURE_SHA256",
                _sha256(self.model_closure_raw),
            ),
        )
        for patcher in self.patchers:
            patcher.start()
        (
            self.snapshot,
            self.archive,
            self.tree_sha256,
            self.manifest_sha256,
            self.archive_sha256,
        ) = self._make_snapshot()
        self.gold = self.snapshot / artifacts.SMOKE_GOLD_RELPATH
        self.selected = self.snapshot / self.selected_relative
        self.model_closure = (
            self.snapshot / artifacts.MODEL_CLOSURE_RELPATH
        )
        self.trust_patchers = (
            mock.patch.object(
                artifacts,
                "FROZEN_QWEN_IMPLEMENTATION_SHA256",
                artifacts._sha256_file(
                    self.snapshot
                    / artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["qwen"]
                ),
            ),
            mock.patch.object(
                artifacts,
                "FROZEN_FINALIZER_IMPLEMENTATION_SHA256",
                artifacts._sha256_file(
                    self.snapshot
                    / artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["finalizer"]
                ),
            ),
            mock.patch.object(
                artifacts,
                "FROZEN_ACCEPTANCE_VERIFIER_SHA256",
                artifacts._sha256_file(
                    self.snapshot
                    / artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["verifier"]
                ),
            ),
            mock.patch.object(
                artifacts,
                "FROZEN_SBATCH_SHA256",
                artifacts._sha256_file(
                    self.snapshot
                    / artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["sbatch"]
                ),
            ),
            mock.patch.object(
                artifacts,
                "FROZEN_MODEL_PATH",
                str(self.model),
            ),
            mock.patch.object(
                artifacts,
                "FROZEN_MODEL_CONFIG_SHA256",
                self.model_config_sha256,
            ),
        )
        for patcher in self.trust_patchers:
            patcher.start()
        self.run_root = self.root / "run_v16"

    def tearDown(self) -> None:
        for patcher in reversed(self.trust_patchers):
            patcher.stop()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def _make_snapshot(
        self,
    ) -> tuple[Path, Path, str, str, str]:
        snapshot = self.root / "snapshot-v16"
        sources = {
            artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["qwen"]: (
                b"# frozen qwen\n"
            ),
            artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["finalizer"]: (
                b"# frozen finalizer\n"
            ),
            artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["verifier"]: (
                b"# frozen independent verifier\n"
            ),
            artifacts.CONTRACT_SOURCE_RELATIVE_PATHS["sbatch"]: (
                b"#!/usr/bin/env bash\nexit 0\n"
            ),
            artifacts.RUN_ARTIFACT_BUILDER_RELPATH: Path(
                artifacts.__file__
            ).resolve().read_bytes(),
            artifacts.SMOKE_GOLD_RELPATH: self.gold_raw,
            artifacts.MODEL_CLOSURE_RELPATH: self.model_closure_raw,
            self.selected_relative: self.selected_raw,
        }
        rows: list[dict[str, object]] = []
        for relative, payload in sorted(sources.items()):
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(0o444)
            rows.append(
                {
                    "mode": "0444",
                    "path": relative,
                    "sha256": _sha256(payload),
                    "size": len(payload),
                    "type": "file",
                }
            )
        manifest_raw = b"".join(_canonical(row) for row in rows)
        manifest = snapshot / "SOURCE_FILES.jsonl"
        manifest.write_bytes(manifest_raw)
        manifest.chmod(0o444)
        manifest_sha256 = _sha256(manifest_raw)
        provenance = {
            "schema": artifacts.SOURCE_SNAPSHOT_SCHEMA,
            "created_at_utc": "2026-07-30T00:00:00+00:00",
            "repo_root": "/frozen/source",
            "source_roots": ["lucy", "methods/motive"],
            "source_file_count": len(rows),
            "source_tree_sha256": manifest_sha256,
            "source_manifest_sha256": manifest_sha256,
            "git_base_commit": "0" * 40,
            "git_status_short": [],
        }
        provenance_path = snapshot / "SOURCE_PROVENANCE.json"
        provenance_path.write_bytes(_canonical(provenance))
        provenance_path.chmod(0o444)
        for directory in sorted(
            (path for path in snapshot.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        snapshot.chmod(0o555)

        archive = self.root / "snapshot-v16.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(snapshot, arcname=snapshot.name)
        return (
            snapshot,
            archive,
            manifest_sha256,
            manifest_sha256,
            artifacts._sha256_file(archive),
        )

    def _make_model(self) -> tuple[Path, str]:
        model = self.root / artifacts.FROZEN_MODEL_REVISION
        model.mkdir()
        config = model / "config.json"
        config.write_bytes(_canonical({"model_type": "qwen3_vl"}))
        weight = model / "model-00001-of-00001.safetensors"
        weight.write_bytes(b"synthetic-safetensors")
        index = {
            "metadata": {"total_size": weight.stat().st_size},
            "weight_map": {
                "model.synthetic.weight": weight.name,
            },
        }
        (model / "model.safetensors.index.json").write_bytes(
            _canonical(index)
        )
        return model, artifacts._sha256_file(config)

    def _make_model_closure(self) -> dict[str, object]:
        files = [
            {
                "relative_path": path.relative_to(self.model).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": artifacts._sha256_file(path),
            }
            for path in sorted(self.model.rglob("*"))
            if path.is_file()
        ]
        return {
            "schema_version": artifacts.MODEL_CLOSURE_SCHEMA,
            "model_id": artifacts.FROZEN_MODEL_ID,
            "revision": artifacts.FROZEN_MODEL_REVISION,
            "model_path": str(self.model),
            "hash_algorithm": "sha256",
            "file_count": len(files),
            "total_bytes": sum(int(item["bytes"]) for item in files),
            "files": files,
        }

    def _prepare(self) -> dict[str, object]:
        return artifacts.prepare_run(
            run_root=self.run_root,
            frozen_selected=self.selected,
            smoke_gold=self.gold,
            model_closure=self.model_closure,
            source_snapshot=self.snapshot,
            source_archive=self.archive,
            source_tree_sha256=self.tree_sha256,
            source_manifest_sha256=self.manifest_sha256,
            source_archive_sha256=self.archive_sha256,
            model_path=self.model,
            model_config_sha256=self.model_config_sha256,
        )

    def _make_terminal_outputs(self, submission: dict[str, object]) -> None:
        qwen_root = Path(submission["outputs"]["qwen_root"])
        final_output = Path(submission["outputs"]["final_output"])
        qwen_root.mkdir()
        selected_path = str(submission["selected"]["path"])
        selected_sha256 = str(submission["selected"]["sha256"])
        source = submission["source_snapshot"]
        runtime = submission["runtime"]
        model = submission["model"]
        selected_rows = self.selected_rows
        completion_shards: list[dict[str, object]] = []
        for index in range(8):
            assigned = [
                row["iid"]
                for row in selected_rows
                if artifacts._iid_shard(row["iid"]) == index
            ]
            run_config = {
                "model_path": model["path"],
                "model_revision": self.model.name,
                "transformers_version": "5.5.4",
                "max_samples": runtime["max_samples"],
                "num_shards": runtime["num_shards"],
                "max_new_tokens": runtime["max_new_tokens"],
                "nframes": runtime["nframes"],
                "max_pixels": runtime["max_pixels"],
                "attn_implementation": runtime[
                    "attn_implementation"
                ],
                "allow_download": runtime["allow_download"],
                "repair_attempts": runtime["repair_attempts"],
                "implementation_digest": source[
                    "qwen_implementation_sha256"
                ],
            }
            run_config_digest = artifacts._object_digest(run_config)
            config_digest = artifacts._object_digest(
                {"run_config": run_config, "shard_index": index}
            )
            rows = [
                {
                    "iid": iid,
                    "status": "ok",
                    "execution_manifest": selected_path,
                    "execution_manifest_sha256": selected_sha256,
                    "shard_index": index,
                    "num_shards": 8,
                    "implementation_digest": source[
                        "qwen_implementation_sha256"
                    ],
                    "config_digest": config_digest,
                    "run_config_digest": run_config_digest,
                    "model_path": model["path"],
                    "model_revision": self.model.name,
                    "transformers_version": "5.5.4",
                }
                for iid in assigned
            ]
            raw = b"".join(_canonical(row) for row in rows)
            shard = qwen_root / f"qwen_shard_{index:03d}.jsonl"
            shard.write_bytes(raw)
            receipt = {
                "schema_version": artifacts.SHARD_RECEIPT_SCHEMA,
                "status": "complete",
                "execution_manifest": selected_path,
                "execution_manifest_sha256": selected_sha256,
                "root": str(Path(selected_path).parent),
                "shard_index": index,
                "num_shards": 8,
                "assigned_iids": assigned,
                "implementation_digest": source[
                    "qwen_implementation_sha256"
                ],
                "config_digest": config_digest,
                "run_config_digest": run_config_digest,
                "run_config": run_config,
                "model_path": model["path"],
                "model_revision": self.model.name,
                "transformers_version": "5.5.4",
                "output": {
                    "path": str(shard.resolve()),
                    "sha256": _sha256(raw),
                    "bytes": len(raw),
                    "rows": len(rows),
                    "status_counts": {"ok": len(rows)} if rows else {},
                },
            }
            receipt["receipt_digest"] = artifacts._self_digest(
                receipt,
                digest_field="receipt_digest",
            )
            receipt_path = (
                qwen_root / f"qwen_shard_{index:03d}.receipt.json"
            )
            receipt_raw = _canonical(receipt)
            receipt_path.write_bytes(receipt_raw)
            completion_shards.append(
                {
                    "index": index,
                    "path": str(shard.resolve()),
                    "sha256": _sha256(raw),
                    "bytes": len(raw),
                    "receipt_path": str(receipt_path.resolve()),
                    "receipt_sha256": _sha256(receipt_raw),
                }
            )

        final_output.mkdir()
        hard_pass = list(self.admissible_iids)
        finalization = {
            "hard_gate_passed": True,
            "hard_gate_failures": [],
            "human_review_status": "pending",
            "human_label": False,
            "generation_authorized": False,
            "manifest_role": "review_proposal",
            "production_eligible": False,
            "approval": None,
            "authorization_interface_available": False,
        }
        review_rows = [
            {
                "iid": iid,
                "action_anchor_finalization": dict(finalization),
            }
            for iid in hard_pass
        ]
        generation_rows = [
            {
                "iid": iid,
                "human_review_status": "pending",
                "generation_authorized": False,
                "manifest_role": "review_proposal",
                "production_eligible": False,
                "approval": None,
                "authorization_interface_available": False,
            }
            for iid in hard_pass
        ]
        payloads = {
            "review_candidates.jsonl": b"".join(
                _canonical(row) for row in review_rows
            ),
            "proposed_128.jsonl": b"".join(
                _canonical(row) for row in review_rows
            ),
            "reserve_32.jsonl": b"",
            "generation_manifest.jsonl": b"".join(
                _canonical(row) for row in generation_rows
            ),
        }
        for name, payload in payloads.items():
            (final_output / name).write_bytes(payload)
        summary = {
            "schema_version": (
                "motive-goku-action-anchor-finalize-v8"
            ),
            "input": {
                "selected_path": selected_path,
                "selected_rows": 16,
                "selected_sha256": selected_sha256,
                "qwen_num_shards": 8,
                "qwen_implementation_digest": source[
                    "qwen_implementation_sha256"
                ],
                "qwen_shards": [
                    {
                        **item,
                        "rows": self.shard_counts[
                            item["index"]
                        ],
                    }
                    for item in completion_shards
                ],
            },
            "hard_gate": {
                "passed_rows": 6,
                "rejected_rows": 10,
            },
            "selection": {
                "allow_partial": True,
                "review_rows": 6,
                "proposed_rows": 6,
                "reserve_rows": 0,
                "generation_rows": 6,
            },
            "semantics": {
                "manifest_role": "review_proposal",
                "human_review_status": "pending",
                "human_labels_asserted": False,
                "generation_authorized": False,
                "production_eligible": False,
                "approval": None,
                "authorization_interface_available": False,
            },
            "implementation_sha256": source[
                "finalizer_implementation_sha256"
            ],
            "output_sha256": {
                name: _sha256(payload)
                for name, payload in sorted(payloads.items())
            },
        }
        summary_raw = _canonical(summary)
        (final_output / "summary.json").write_bytes(summary_raw)
        done_outputs = {
            **summary["output_sha256"],
            "summary.json": _sha256(summary_raw),
        }
        done = {
            "schema_version": (
                "motive-goku-action-anchor-finalize-done-v8"
            ),
            "status": "complete",
            "summary_sha256": _sha256(summary_raw),
            "implementation_sha256": source[
                "finalizer_implementation_sha256"
            ],
            "output_sha256": dict(sorted(done_outputs.items())),
        }
        (final_output / "done.json").write_bytes(_canonical(done))

    def _complete(self) -> dict[str, object]:
        return artifacts.complete_run(
            submission_contract=self.run_root / "submission_contract.json",
            job_id="123456",
            output=self.run_root / "completion_receipt.json",
        )

    def test_schema_names_match_independent_verifier(self) -> None:
        self.assertEqual(
            artifacts.SUBMISSION_SCHEMA,
            "motive-goku-action-v16-submission-contract-v1",
        )
        self.assertEqual(
            artifacts.COMPLETION_SCHEMA,
            "motive-goku-action-v16-completion-receipt-v1",
        )
        self.assertEqual(
            artifacts.ACCEPTANCE_CONTRACT_SCHEMA,
            "motive-goku-action-v16-acceptance-contract-v1",
        )

    def test_prepare_publishes_exact_closed_contract_and_fresh_closure(
        self,
    ) -> None:
        contract = self._prepare()
        self.assertEqual(
            set(contract),
            {
                "schema_version",
                "selected",
                "smoke_gold",
                "model_closure",
                "source_snapshot",
                "model",
                "runtime",
                "outputs",
            },
        )
        self.assertEqual(
            set(contract["selected"]),
            {"path", "sha256", "rows"},
        )
        self.assertEqual(
            contract["smoke_gold"],
            {
                "path": str(self.gold),
                "sha256": _sha256(self.gold_raw),
            },
        )
        self.assertEqual(
            contract["model_closure"],
            {
                "path": str(self.model_closure),
                "sha256": _sha256(self.model_closure_raw),
                "file_count": self.model_closure_value["file_count"],
                "total_bytes": self.model_closure_value["total_bytes"],
            },
        )
        self.assertEqual(
            set(contract["outputs"]),
            {"qwen_root", "final_output"},
        )
        self.assertEqual(
            (self.run_root / "input" / "selected_smoke.jsonl").read_bytes(),
            self.selected_raw,
        )
        self.assertTrue((self.run_root / "logs").is_dir())
        for name in (
            "qwen8",
            "final",
            "jobs.tsv",
            "completion_receipt.json",
            "acceptance_contract.json",
            "acceptance_result.json",
        ):
            self.assertFalse((self.run_root / name).exists())
        raw = (self.run_root / "submission_contract.json").read_bytes()
        self.assertEqual(raw, _canonical(contract))
        for path in (
            self.run_root / "input" / "selected_smoke.jsonl",
            self.run_root / "input" / "subset_provenance.json",
            self.run_root / "submission_contract.json",
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o222, 0)
        self.assertEqual(
            stat.S_IMODE(
                (self.run_root / "input").stat().st_mode
            )
            & 0o222,
            0,
        )

    def test_prepare_subset_provenance_binds_order_and_shards(self) -> None:
        self._prepare()
        path = self.run_root / "input" / "subset_provenance.json"
        provenance = json.loads(path.read_bytes())
        self.assertEqual(
            provenance["ordered_iids"],
            self.ordered_iids,
        )
        self.assertEqual(
            provenance["qwen_shard_row_counts"],
            self.shard_counts,
        )
        self.assertEqual(
            provenance["provenance_digest"],
            artifacts._self_digest(
                provenance,
                digest_field="provenance_digest",
            ),
        )

    def test_prepare_refuses_existing_run_root(self) -> None:
        self.run_root.mkdir()
        with self.assertRaises(FileExistsError):
            self._prepare()

    def test_prepare_rejects_selected_mutation_without_partial_root(
        self,
    ) -> None:
        mutated = self.root / "mutated_selected.jsonl"
        mutated.write_bytes(self.selected_raw + b"\n")
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts.prepare_run(
                run_root=self.run_root,
                frozen_selected=mutated,
                smoke_gold=self.gold,
                model_closure=self.model_closure,
                source_snapshot=self.snapshot,
                source_archive=self.archive,
                source_tree_sha256=self.tree_sha256,
                source_manifest_sha256=self.manifest_sha256,
                source_archive_sha256=self.archive_sha256,
                model_path=self.model,
                model_config_sha256=self.model_config_sha256,
            )
        self.assertFalse(self.run_root.exists())

    def test_prepare_requires_smoke_gold_cli_argument(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                artifacts.build_parser().parse_args(
                    [
                        "prepare",
                        "--run-root",
                        str(self.run_root),
                        "--frozen-selected",
                        str(self.selected),
                        "--model-closure",
                        str(self.model_closure),
                        "--source-snapshot",
                        str(self.snapshot),
                        "--source-archive",
                        str(self.archive),
                        "--source-tree-sha256",
                        self.tree_sha256,
                        "--source-manifest-sha256",
                        self.manifest_sha256,
                        "--source-archive-sha256",
                        self.archive_sha256,
                        "--model",
                        str(self.model),
                        "--model-config-sha256",
                        self.model_config_sha256,
                    ]
                )

    def test_prepare_requires_model_closure_cli_argument(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                artifacts.build_parser().parse_args(
                    [
                        "prepare",
                        "--run-root",
                        str(self.run_root),
                        "--frozen-selected",
                        str(self.selected),
                        "--smoke-gold",
                        str(self.gold),
                        "--source-snapshot",
                        str(self.snapshot),
                        "--source-archive",
                        str(self.archive),
                        "--source-tree-sha256",
                        self.tree_sha256,
                        "--source-manifest-sha256",
                        self.manifest_sha256,
                        "--source-archive-sha256",
                        self.archive_sha256,
                        "--model",
                        str(self.model),
                        "--model-config-sha256",
                        self.model_config_sha256,
                    ]
                )

    def test_gold_counts_are_computed_and_not_fixed_routes(self) -> None:
        gold = artifacts._validate_smoke_gold(
            self.gold,
            source_snapshot=self.snapshot,
        )
        rows, _, _ = artifacts._validated_selected(
            self.selected,
            smoke_gold=gold,
        )
        iids = [str(row["iid"]) for row in rows]
        computed = [
            sum(artifacts._iid_shard(iid) == index for iid in iids)
            for index in range(8)
        ]
        self.assertEqual(
            gold["selected"]["expected_shard_rows"],
            computed,
        )
        self.assertFalse(
            any(
                hasattr(artifacts, name)
                for name in (
                    "EXPECTED_DIRECT_IIDS",
                    "EXPECTED_REPAIR_ONCE_IIDS",
                    "EXPECTED_JUDGE_A_REJECT_IIDS",
                    "FROZEN_ORDERED_IIDS",
                )
            )
        )
        bad_gold = dict(gold)
        bad_selected = dict(gold["selected"])
        bad_selected["expected_shard_rows"] = [0] * 8
        bad_gold["selected"] = bad_selected
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validated_selected(
                self.selected,
                smoke_gold=bad_gold,
            )

    def test_submission_rejects_old_schema_even_if_resigned(self) -> None:
        self._prepare()
        path = self.run_root / "submission_contract.json"
        value = json.loads(path.read_bytes())
        value["schema_version"] = (
            "motive-goku-action-v15-submission-contract-v1"
        )
        path.chmod(0o600)
        path.write_bytes(_canonical(value))
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validate_submission_contract(path)

    def test_submission_rejects_v15_gold_hash_if_resigned(self) -> None:
        self._prepare()
        path = self.run_root / "submission_contract.json"
        value = json.loads(path.read_bytes())
        value["smoke_gold"]["sha256"] = (
            "0541e800a0c9fbbafffe04829292008ab03812e7916436f5e498033b7b988162"
        )
        path.chmod(0o600)
        path.write_bytes(_canonical(value))
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validate_submission_contract(path)

    def test_submission_rejects_any_approval_interface(self) -> None:
        self._prepare()
        path = self.run_root / "submission_contract.json"
        value = json.loads(path.read_bytes())
        value["runtime"]["approval_path"] = None
        path.chmod(0o600)
        path.write_bytes(_canonical(value))
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validate_submission_contract(path)

    def test_submission_rejects_gold_at_different_path_if_resigned(
        self,
    ) -> None:
        self._prepare()
        copied_gold = self.root / "resigned-gold.json"
        copied_gold.write_bytes(self.gold_raw)
        path = self.run_root / "submission_contract.json"
        value = json.loads(path.read_bytes())
        value["smoke_gold"] = {
            "path": str(copied_gold),
            "sha256": _sha256(self.gold_raw),
        }
        path.chmod(0o600)
        path.write_bytes(_canonical(value))
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validate_submission_contract(path)

    def test_submission_rejects_gold_tamper_even_if_resigned(self) -> None:
        self._prepare()
        gold_value = json.loads(self.gold_raw)
        gold_value["review_method"] += " tampered"
        tampered_raw = json.dumps(
            gold_value,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8") + b"\n"
        self.gold.chmod(0o600)
        self.gold.write_bytes(tampered_raw)
        path = self.run_root / "submission_contract.json"
        value = json.loads(path.read_bytes())
        value["smoke_gold"]["sha256"] = _sha256(tampered_raw)
        path.chmod(0o600)
        path.write_bytes(_canonical(value))
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validate_submission_contract(path)

    def test_prepare_rejects_archive_digest_without_partial_root(self) -> None:
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts.prepare_run(
                run_root=self.run_root,
                frozen_selected=self.selected,
                smoke_gold=self.gold,
                model_closure=self.model_closure,
                source_snapshot=self.snapshot,
                source_archive=self.archive,
                source_tree_sha256=self.tree_sha256,
                source_manifest_sha256=self.manifest_sha256,
                source_archive_sha256="0" * 64,
                model_path=self.model,
                model_config_sha256=self.model_config_sha256,
            )
        self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_tampered_model_weight(self) -> None:
        weight = self.model / "model-00001-of-00001.safetensors"
        weight.write_bytes(weight.read_bytes() + b"-tampered")
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            self._prepare()
        self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_missing_model_file(self) -> None:
        (self.model / "model-00001-of-00001.safetensors").unlink()
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            self._prepare()
        self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_extra_model_file(self) -> None:
        (self.model / "untracked.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            self._prepare()
        self.assertFalse(self.run_root.exists())

    def test_prepare_allows_hf_style_file_symlink(self) -> None:
        weight = self.model / "model-00001-of-00001.safetensors"
        blob = self.root / "model-blob"
        blob.write_bytes(weight.read_bytes())
        weight.unlink()
        weight.symlink_to(blob)
        contract = self._prepare()
        self.assertEqual(
            contract["model_closure"]["sha256"],
            _sha256(self.model_closure_raw),
        )

    def test_model_closure_manifest_resign_is_rejected(self) -> None:
        value = json.loads(self.model_closure_raw)
        value["files"][0]["sha256"] = "0" * 64
        resigned = _canonical(value)
        self.model_closure.chmod(0o600)
        self.model_closure.write_bytes(resigned)
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validate_model_closure(
                self.model_closure,
                source_snapshot=self.snapshot,
                model_path=self.model,
            )

    def test_submission_model_closure_resign_is_rejected(self) -> None:
        self._prepare()
        path = self.run_root / "submission_contract.json"
        value = json.loads(path.read_bytes())
        value["model_closure"]["sha256"] = "0" * 64
        path.chmod(0o600)
        path.write_bytes(_canonical(value))
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts._validate_submission_contract(path)

    def test_prepare_rejects_nonfrozen_producer_trust_anchors(self) -> None:
        constant_names = (
            "FROZEN_QWEN_IMPLEMENTATION_SHA256",
            "FROZEN_FINALIZER_IMPLEMENTATION_SHA256",
            "FROZEN_ACCEPTANCE_VERIFIER_SHA256",
            "FROZEN_SBATCH_SHA256",
        )
        for constant_name in constant_names:
            with self.subTest(constant_name=constant_name):
                with mock.patch.object(
                    artifacts,
                    constant_name,
                    "0" * 64,
                ):
                    with self.assertRaises(
                        artifacts.GokuActionV13RunArtifactError
                    ):
                        self._prepare()
                self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_v15_implementation_hashes(self) -> None:
        stale = {
            "FROZEN_QWEN_IMPLEMENTATION_SHA256": (
                "7aa9ec1ba202922cd4e3241a6c6e6bcc63b7f6c6cf17260fba7a1405930b73ce"
            ),
            "FROZEN_FINALIZER_IMPLEMENTATION_SHA256": (
                "3678586e4f33af88e1452fb0345f3c3d9c35caf090be65bba6af3639b3d2638d"
            ),
            "FROZEN_ACCEPTANCE_VERIFIER_SHA256": (
                "76f1394058f466789d480dc7b4dd394ebb3ef4a6bd49d3c6683feca948a682a4"
            ),
            "FROZEN_SBATCH_SHA256": (
                "c4e4ebfec4e7f8eb11829f8268c939ba5b611d58d611e075dd6280c9487214d7"
            ),
        }
        for constant_name, stale_sha256 in stale.items():
            with self.subTest(constant_name=constant_name):
                with mock.patch.object(
                    artifacts,
                    constant_name,
                    stale_sha256,
                ):
                    with self.assertRaises(
                        artifacts.GokuActionV13RunArtifactError
                    ):
                        self._prepare()
                self.assertFalse(self.run_root.exists())

    def test_old_qwen25_7b_model_identity_is_rejected(self) -> None:
        old_identity = {
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        }
        for field, stale_value in old_identity.items():
            with self.subTest(field=field):
                value = json.loads(self.model_closure_raw)
                value[field] = stale_value
                stale_raw = _canonical(value)
                self.model_closure.chmod(0o600)
                self.model_closure.write_bytes(stale_raw)
                with mock.patch.object(
                    artifacts,
                    "FROZEN_MODEL_CLOSURE_SHA256",
                    _sha256(stale_raw),
                ):
                    with self.assertRaisesRegex(
                        artifacts.GokuActionV13RunArtifactError,
                        "model closure identity/header differs",
                    ):
                        artifacts._validate_model_closure(
                            self.model_closure,
                            source_snapshot=self.snapshot,
                            model_path=self.model,
                        )
                self.model_closure.write_bytes(self.model_closure_raw)
                self.model_closure.chmod(0o444)

    def test_prepare_rejects_stale_model_closure_anchor(self) -> None:
        with mock.patch.object(
            artifacts,
            "FROZEN_MODEL_CLOSURE_SHA256",
            "1" * 64,
        ):
            with self.assertRaises(
                artifacts.GokuActionV13RunArtifactError
            ):
                self._prepare()
        self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_stale_acceptance_verifier_hash(self) -> None:
        with mock.patch.object(
            artifacts,
            "FROZEN_ACCEPTANCE_VERIFIER_SHA256",
            "1" * 64,
        ):
            with self.assertRaises(
                artifacts.GokuActionV13RunArtifactError
            ):
                self._prepare()
        self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_superseded_v16_acceptance_verifier_hash(
        self,
    ) -> None:
        superseded = (
            "930f4b5ab4268e157a6b5a1c001decacded5211521ef15451cd3d2fef4389990",
            "1c3ef873fddc0c7395d618c1b23ba4edb8fb7f7f50a10f953cd2c63c0536fc10",
            "b157dba0fe2c859c8a7d68b19383b32ac4673bbcf25dce151a54f2fe9081e48b",
            "e01df3501aa67789defb9b2ff40d16f427c822bfcc4f63d5f603ddb7d8a62a04",
        )
        for digest in superseded:
            with self.subTest(digest=digest):
                with mock.patch.object(
                    artifacts,
                    "FROZEN_ACCEPTANCE_VERIFIER_SHA256",
                    digest,
                ):
                    with self.assertRaises(
                        artifacts.GokuActionV13RunArtifactError
                    ):
                        self._prepare()
                self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_superseded_v16_finalizer_hash(
        self,
    ) -> None:
        for digest in (
            "d626f1d7297081e48620ab9ef49223236b015406c0e421b6c090f0e42b2a2210",
            "6e0d80f6e436eb127ae0b2d6d718ab2173245826125e5b628915c774a5ef8d3a",
        ):
            with self.subTest(digest=digest):
                with mock.patch.object(
                    artifacts,
                    "FROZEN_FINALIZER_IMPLEMENTATION_SHA256",
                    digest,
                ):
                    with self.assertRaises(
                        artifacts.GokuActionV13RunArtifactError
                    ):
                        self._prepare()
                self.assertFalse(self.run_root.exists())

    def test_prepare_rejects_nonfrozen_model_identity(self) -> None:
        mutations = (
            ("FROZEN_MODEL_PATH", str(self.root / "other-model")),
            ("FROZEN_MODEL_CONFIG_SHA256", "0" * 64),
        )
        for constant_name, replacement in mutations:
            with self.subTest(constant_name=constant_name):
                with mock.patch.object(
                    artifacts,
                    constant_name,
                    replacement,
                ):
                    with self.assertRaises(
                        artifacts.GokuActionV13RunArtifactError
                    ):
                        self._prepare()
                self.assertFalse(self.run_root.exists())

    def test_complete_publishes_exact_closed_receipt(self) -> None:
        submission = self._prepare()
        self._make_terminal_outputs(submission)
        receipt = self._complete()
        self.assertEqual(
            set(receipt),
            {
                "schema_version",
                "status",
                "job_id",
                "submission_contract_path",
                "submission_contract_sha256",
                "selected_sha256",
                "smoke_gold_sha256",
                "model_closure",
                "qwen_root",
                "final_output",
                "qwen_shards",
                "final_artifacts",
            },
        )
        self.assertEqual(
            [item["index"] for item in receipt["qwen_shards"]],
            list(range(8)),
        )
        self.assertEqual(
            set(receipt["final_artifacts"]),
            set(artifacts.FINAL_ARTIFACT_NAMES),
        )
        self.assertEqual(
            (self.run_root / "completion_receipt.json").read_bytes(),
            _canonical(receipt),
        )
        self.assertEqual(
            stat.S_IMODE(
                (self.run_root / "completion_receipt.json").stat().st_mode
            )
            & 0o222,
            0,
        )

    def test_complete_rejects_bad_job_id_before_publication(self) -> None:
        submission = self._prepare()
        self._make_terminal_outputs(submission)
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            artifacts.complete_run(
                submission_contract=(
                    self.run_root / "submission_contract.json"
                ),
                job_id="not-slurm",
                output=self.run_root / "completion_receipt.json",
            )
        self.assertFalse(
            (self.run_root / "completion_receipt.json").exists()
        )

    def test_complete_rejects_extra_qwen_file(self) -> None:
        submission = self._prepare()
        self._make_terminal_outputs(submission)
        (self.run_root / "qwen8" / "stale.tmp").write_text("stale")
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            self._complete()

    def test_complete_rejects_receipt_tampering(self) -> None:
        submission = self._prepare()
        self._make_terminal_outputs(submission)
        path = self.run_root / "qwen8" / "qwen_shard_000.receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt["receipt_digest"] = "0" * 64
        path.write_bytes(_canonical(receipt))
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            self._complete()

    def test_acceptance_contract_matches_exact_frozen_schema(self) -> None:
        submission = self._prepare()
        self._make_terminal_outputs(submission)
        self._complete()
        contract = artifacts.build_acceptance_contract(
            submission_contract=self.run_root / "submission_contract.json",
            completion_receipt=self.run_root / "completion_receipt.json",
            output=self.run_root / "acceptance_contract.json",
        )
        self.assertEqual(
            set(contract),
            {
                "schema_version",
                "selected",
                "smoke_gold",
                "model_closure",
                "expected_shard_counts",
                "source_snapshot",
                "model",
                "execution",
                "final",
                "bindings",
            },
        )
        self.assertEqual(
            set(contract["final"]),
            {
                "seed",
                "allow_partial",
                "manifest_role",
                "human_review_status",
                "generation_authorized",
                "production_eligible",
                "wan_generation_authorized",
            },
        )
        self.assertFalse(contract["final"]["wan_generation_authorized"])
        self.assertNotIn("routes", contract)
        self.assertEqual(
            contract["model_closure"],
            submission["model_closure"],
        )
        self.assertEqual(
            contract["smoke_gold"],
            {
                "path": str(self.gold),
                "sha256": _sha256(self.gold_raw),
            },
        )
        self.assertEqual(
            contract["expected_shard_counts"],
            self.shard_counts,
        )
        self.assertEqual(
            (self.run_root / "acceptance_contract.json").read_bytes(),
            _canonical(contract),
        )
        self.assertEqual(
            stat.S_IMODE(
                (self.run_root / "acceptance_contract.json").stat().st_mode
            )
            & 0o222,
            0,
        )

    def test_independent_verifier_closed_schema_validators_accept_contracts(
        self,
    ) -> None:
        submission = self._prepare()
        self._make_terminal_outputs(submission)
        completion = self._complete()
        contract = artifacts.build_acceptance_contract(
            submission_contract=self.run_root / "submission_contract.json",
            completion_receipt=self.run_root / "completion_receipt.json",
            output=self.run_root / "acceptance_contract.json",
        )
        snapshot = submission["source_snapshot"]
        model = submission["model"]
        # The verifier pins production trust anchors.  This unit fixture uses
        # tiny synthetic source/model bytes, so replace only those constants
        # within this test while exercising the real closed-schema validators.
        with mock.patch.multiple(
            independent_acceptance,
            EXPECTED_QWEN_IMPLEMENTATION_SHA256=snapshot[
                "qwen_implementation_sha256"
            ],
            EXPECTED_FINALIZER_IMPLEMENTATION_SHA256=snapshot[
                "finalizer_implementation_sha256"
            ],
            EXPECTED_SBATCH_SHA256=snapshot["sbatch_sha256"],
            EXPECTED_SMOKE_GOLD_SHA256=_sha256(self.gold_raw),
            EXPECTED_MODEL_CLOSURE_SHA256=_sha256(
                self.model_closure_raw
            ),
            EXPECTED_MODEL_CLOSURE_FILE_COUNT=(
                self.model_closure_value["file_count"]
            ),
            EXPECTED_MODEL_CLOSURE_TOTAL_BYTES=(
                self.model_closure_value["total_bytes"]
            ),
            EXPECTED_MODEL_PATH=model["path"],
            EXPECTED_MODEL_CONFIG_SHA256=model["config_sha256"],
        ):
            independent_acceptance._validate_submission_contract(submission)
            independent_acceptance._validate_completion_receipt(completion)
            independent_acceptance._validate_acceptance_contract(
                contract
            )

    def test_acceptance_contract_refuses_existing_output(self) -> None:
        submission = self._prepare()
        self._make_terminal_outputs(submission)
        self._complete()
        output = self.run_root / "acceptance_contract.json"
        output.write_text("do-not-overwrite")
        with self.assertRaises(FileExistsError):
            artifacts.build_acceptance_contract(
                submission_contract=(
                    self.run_root / "submission_contract.json"
                ),
                completion_receipt=(
                    self.run_root / "completion_receipt.json"
                ),
                output=output,
            )
        self.assertEqual(output.read_text(), "do-not-overwrite")

    def test_submission_contract_extra_key_is_rejected(self) -> None:
        submission = self._prepare()
        path = self.run_root / "submission_contract.json"
        value = json.loads(path.read_bytes())
        value["wan_generation_authorized"] = False
        path.chmod(0o600)
        path.write_bytes(_canonical(value))
        self._make_terminal_outputs(submission)
        with self.assertRaises(artifacts.GokuActionV13RunArtifactError):
            self._complete()

    def test_atomic_writer_never_overwrites(self) -> None:
        path = self.root / "existing.json"
        path.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            artifacts._write_new_json(path, {"replacement": True})
        self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
