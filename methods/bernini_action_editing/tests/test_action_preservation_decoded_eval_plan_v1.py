from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_plan_v1 as evaluation


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sign(row: dict, field: str) -> dict:
    result = copy.deepcopy(row)
    result.pop(field, None)
    result[field] = evaluation.object_sha256(result)
    return result


def pins() -> dict[str, str]:
    return {key: digest(key) for key in evaluation.PIN_FIELDS}


def action_contract(iid: str) -> dict:
    description = f"Complete the fitted action for source {iid}, then hold the terminal pose."
    row = {
        "schema_version": evaluation.ACTION_REVIEW_CONTRACT_SCHEMA,
        "action_order_description": description,
        "action_order_description_sha256": evaluation.text_sha256(description),
        "expected_onset_frame_min": 4,
        "expected_onset_frame_max": 20,
        "terminal_hold_start_frame_min": 65,
        "terminal_hold_end_frame": 80,
        "full_video_frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
    }
    row["contract_digest"] = evaluation.object_sha256(row)
    return row


def sources() -> list[dict]:
    values = []
    for index, iid in enumerate(evaluation.FITTED_IIDS):
        instruction = f"Perform the fitted action for source {iid}."
        values.append(
            {
                "iid": iid,
                "source_video_sha256": digest(f"source-video:{iid}"),
                "source_receipt_sha256": digest(f"source-receipt:{iid}"),
                "instruction": instruction,
                "instruction_sha256": evaluation.text_sha256(instruction),
                "action_review_contract": action_contract(iid),
                "seed": 2026081801 + index,
            }
        )
    return values


def checkpoints() -> list[dict]:
    return [
        {
            "arm": arm,
            "checkpoint_step": step,
            "checkpoint_receipt_sha256": digest(f"checkpoint-receipt:{arm}:{step}"),
            "adapter_sha256": digest(f"adapter:{arm}:{step}"),
        }
        for arm in evaluation.ARMS
        for step in evaluation.CHECKPOINT_STEPS
    ]


class DecodedEvaluationPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = pathlib.Path(self.temporary.name).resolve()
        self.root = self.parent / "decoded-eval"
        self.input_spec = evaluation.build_input_spec(
            evaluation_id="preservation-v2-decoded-eval-r1",
            evaluation_root=self.root,
            pins=pins(),
            sources=sources(),
            checkpoints=checkpoints(),
        )
        self.bundle = evaluation.build_bundle(self.input_spec)
        self.manifest = self.bundle["manifest"]

    def test_exact_256_candidate_matrix_and_eight_base_controls(self):
        candidates = self.manifest["candidates"]
        controls = self.manifest["frozen_base_controls"]
        self.assertEqual(len(candidates), 256)
        self.assertEqual(len(controls), 8)
        self.assertEqual(self.manifest["matrix"]["total_decode_count"], 264)
        keys = {
            (
                row["arm"],
                row["checkpoint_step"],
                row["iid"],
                row["onset_policy"]["name"],
            )
            for row in candidates
        }
        expected = {
            (arm, step, iid, policy)
            for arm in evaluation.ARMS
            for step in evaluation.CHECKPOINT_STEPS
            for iid in evaluation.FITTED_IIDS
            for policy in evaluation.POLICIES
        }
        self.assertEqual(keys, expected)
        self.assertTrue(
            all(row["media_contract"]["frame_count"] == 81 for row in candidates + controls)
        )
        self.assertTrue(
            all(
                (row["media_contract"]["fps_num"], row["media_contract"]["fps_den"])
                == (25, 1)
                for row in candidates + controls
            )
        )

    def test_no_loss_based_filtering_or_checkpoint_selection(self):
        policy = self.manifest["selection_policy"]
        self.assertEqual(policy["candidate_subset_selection"], "none_full_cartesian")
        self.assertFalse(policy["training_loss_read"])
        self.assertFalse(policy["training_loss_filtering"])
        self.assertFalse(policy["checkpoint_loss_ranking"])
        self.assertTrue(
            all(
                row["training_loss_read_or_used_for_selection"] is False
                for row in self.manifest["candidates"]
                + self.manifest["frozen_base_controls"]
            )
        )
        per_checkpoint = {}
        for row in self.manifest["candidates"]:
            key = (row["arm"], row["checkpoint_step"])
            per_checkpoint[key] = per_checkpoint.get(key, 0) + 1
        self.assertEqual(set(per_checkpoint.values()), {8})

        tampered = copy.deepcopy(self.manifest)
        tampered["selection_policy"]["training_loss_filtering"] = True
        tampered = sign(tampered, "manifest_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "loss-free"):
            evaluation.validate_manifest(tampered, input_spec=self.input_spec)

        hostile_input = copy.deepcopy(self.input_spec)
        hostile_input["training_losses"] = [0.1]
        hostile_input = sign(hostile_input, "input_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "field closure"):
            evaluation.validate_input_spec(hostile_input)

    def test_source_and_fitted_iid_completeness_is_closed(self):
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "exactly four"):
            evaluation.build_input_spec(
                evaluation_id="missing-source",
                evaluation_root=self.parent / "missing-source",
                pins=pins(),
                sources=sources()[:-1],
                checkpoints=checkpoints(),
            )
        self.assertEqual(
            {row["iid"] for row in self.manifest["candidates"]},
            set(evaluation.FITTED_IIDS),
        )
        tampered = copy.deepcopy(self.manifest)
        tampered["candidates"].pop()
        tampered = sign(tampered, "manifest_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "candidate matrix"):
            evaluation.validate_manifest(tampered, input_spec=self.input_spec)

    def test_public_instruction_and_action_contract_cannot_leak_private_arm(self):
        hostile_sources = sources()
        description = "Use v2_onset_all and checkpoint-00000020 for this review."
        contract = hostile_sources[0]["action_review_contract"]
        contract["action_order_description"] = description
        contract["action_order_description_sha256"] = evaluation.text_sha256(
            description
        )
        contract["contract_digest"] = evaluation.object_sha256(
            {key: value for key, value in contract.items() if key != "contract_digest"}
        )
        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError, "leaks method/arm"
        ):
            evaluation.build_input_spec(
                evaluation_id="hostile-private-leak",
                evaluation_root=self.parent / "hostile-private-leak",
                pins=pins(),
                sources=hostile_sources,
                checkpoints=checkpoints(),
            )

    def test_seed_pairing_is_exact_across_policy_arm_checkpoint_and_base(self):
        seed_by_iid = {row["iid"]: row["seed"] for row in self.input_spec["sources"]}
        self.assertTrue(
            all(row["seed"] == seed_by_iid[row["iid"]] for row in self.manifest["candidates"])
        )
        self.assertTrue(
            all(
                row["seed"] == seed_by_iid[row["iid"]]
                for row in self.manifest["frozen_base_controls"]
            )
        )
        pairing = {}
        for row in self.manifest["candidates"]:
            key = (row["arm"], row["checkpoint_step"], row["iid"])
            pairing.setdefault(key, set()).add(
                (row["onset_policy"]["name"], row["seed"])
            )
        self.assertTrue(
            all(
                values
                == {
                    ("none", seed_by_iid[key[2]]),
                    ("hard1_every_step", seed_by_iid[key[2]]),
                }
                for key, values in pairing.items()
            )
        )

        tampered = copy.deepcopy(self.manifest)
        tampered["candidates"][0]["seed"] += 1
        tampered["candidates"][0] = sign(
            tampered["candidates"][0], "record_digest"
        )
        tampered = sign(tampered, "manifest_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "candidate matrix"):
            evaluation.validate_manifest(tampered, input_spec=self.input_spec)

    def test_checkpoint_adapter_and_release_provenance_is_exact(self):
        first = self.manifest["candidates"][0]
        source = self.input_spec["sources"][0]
        checkpoint = self.input_spec["checkpoints"][0]
        self.assertEqual(first["source_video_sha256"], source["source_video_sha256"])
        self.assertEqual(first["instruction_sha256"], source["instruction_sha256"])
        self.assertEqual(
            first["checkpoint_receipt_sha256"],
            checkpoint["checkpoint_receipt_sha256"],
        )
        self.assertEqual(first["adapter_sha256"], checkpoint["adapter_sha256"])
        for key in (
            "adapter_release_manifest_sha256",
            "model_release_manifest_sha256",
            "inference_source_sha256",
            "inference_release_manifest_sha256",
            "inference_config_sha256",
            "source_preprocessing_sha256",
            "calibration_digest",
        ):
            self.assertEqual(first[key], self.input_spec["pins"][key])

        invalid = checkpoints()
        invalid[0]["adapter_sha256"] = None
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "SHA-256"):
            evaluation.build_input_spec(
                evaluation_id="missing-adapter-hash",
                evaluation_root=self.parent / "missing-adapter-hash",
                pins=pins(),
                sources=sources(),
                checkpoints=invalid,
            )

        tampered = copy.deepcopy(self.manifest)
        tampered["candidates"][0]["adapter_sha256"] = digest("wrong-adapter")
        tampered["candidates"][0] = sign(
            tampered["candidates"][0], "record_digest"
        )
        tampered = sign(tampered, "manifest_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "provenance"):
            evaluation.validate_manifest(tampered, input_spec=self.input_spec)

    def test_frozen_base_controls_are_deduplicated_and_separately_bound(self):
        controls = self.manifest["frozen_base_controls"]
        keys = {
            (row["iid"], row["seed"], row["onset_policy"]["name"])
            for row in controls
        }
        expected = {
            (source["iid"], source["seed"], policy)
            for source in self.input_spec["sources"]
            for policy in evaluation.POLICIES
        }
        self.assertEqual(keys, expected)
        self.assertTrue(all(row["adapter_sha256"] is None for row in controls))
        self.assertEqual(len({row["deduplication_key"] for row in controls}), 8)
        control_ids = {row["control_id"] for row in controls}
        self.assertTrue(
            all(
                row["matched_frozen_base_control_id"] in control_ids
                for row in self.manifest["candidates"]
            )
        )
        tampered = copy.deepcopy(self.manifest)
        tampered["frozen_base_controls"][-1] = copy.deepcopy(
            tampered["frozen_base_controls"][0]
        )
        tampered = sign(tampered, "manifest_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "control closure"):
            evaluation.validate_manifest(tampered, input_spec=self.input_spec)

    def test_four_holder_shards_are_deterministic_complete_and_local_only(self):
        shards = self.bundle["shards"]
        self.assertEqual(set(shards), {row["job_id"] for row in evaluation.HOLDER_ROWS})
        candidate_ids = []
        control_ids = []
        for shard in shards.values():
            self.assertEqual(shard["candidate_task_count"], 64)
            self.assertEqual(shard["base_control_task_count"], 2)
            self.assertEqual(shard["total_task_count"], 66)
            self.assertFalse(shard["training_loss_read_or_used_for_sharding"])
            self.assertFalse(shard["remote_launch_performed"])
            self.assertFalse(shard["upload_performed"])
            for task in shard["tasks"]:
                if task["task_kind"] == "adapter_candidate":
                    candidate_ids.append(task["record"]["candidate_id"])
                else:
                    control_ids.append(task["record"]["control_id"])
        self.assertEqual(len(candidate_ids), len(set(candidate_ids)) if candidate_ids else 0)
        self.assertEqual(len(candidate_ids), 256)
        self.assertEqual(len(control_ids), len(set(control_ids)))
        self.assertEqual(len(control_ids), 8)
        second = evaluation.build_bundle(self.input_spec)
        self.assertEqual(second["manifest"], self.manifest)
        self.assertEqual(second["shards"], shards)

        hostile = copy.deepcopy(shards["136719"])
        hostile["holder"]["node"] = "wrong-node"
        hostile = sign(hostile, "shard_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "holder"):
            evaluation.validate_shard(
                hostile, manifest=self.manifest, input_spec=self.input_spec
            )

    def test_review_packet_is_blinded_axis_separate_and_fail_closed(self):
        contract = self.bundle["review_contract"]
        self.assertEqual(
            [row["name"] for row in contract["axes"]], list(evaluation.REVIEW_AXES)
        )
        self.assertEqual(evaluation.REVIEW_AXES, evaluation.gate.AXES)
        self.assertTrue(
            all("undetermined" in row["states"] for row in contract["axes"])
        )
        self.assertIsNone(contract["weighted_score"])
        self.assertTrue(contract["weighted_compensation_forbidden"])
        self.assertTrue(contract["machine_gate"]["calibrated_evidence_required"])
        self.assertTrue(contract["machine_gate"]["machine_abstain_blocks_promotion"])
        self.assertTrue(contract["promotion"]["blind_full_video_review_required"])
        self.assertEqual(
            contract["promotion"]["outcome_if_any_abstain"], "STOP"
        )
        self.assertFalse(contract["promotion"]["automatic_model_update"])
        self.assertEqual(
            contract["submission_schema"]["unresolved_or_disagreeing_axis_becomes"],
            "undetermined",
        )
        self.assertEqual(
            contract["submission_schema"]["schema_version"],
            evaluation.gate.BLIND_REVIEW_SCHEMA,
        )
        self.assertTrue(
            contract["submission_schema"]["reviewer_ids_must_be_unique"]
        )
        self.assertTrue(
            contract["submission_schema"][
                "axis_resolution_must_equal_ballot_consensus"
            ]
        )
        self.assertTrue(
            contract["submission_schema"]["weighted_score_field_forbidden"]
        )

        hostile = copy.deepcopy(contract)
        hostile["promotion"]["automatic_model_update"] = True
        hostile = sign(hostile, "contract_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "promotion"):
            evaluation.validate_review_packet_contract(hostile)

    def test_media_contract_tamper_is_rejected_even_when_resigned(self):
        tampered = copy.deepcopy(self.manifest)
        tampered["candidates"][0]["media_contract"]["frame_count"] = 80
        tampered["candidates"][0] = sign(
            tampered["candidates"][0], "record_digest"
        )
        tampered = sign(tampered, "manifest_digest")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "candidate matrix"):
            evaluation.validate_manifest(tampered, input_spec=self.input_spec)

    def test_publication_is_create_only_and_missing_shard_is_hostile(self):
        hostile = copy.deepcopy(self.bundle)
        hostile["shards"].pop("136140")
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "shard closure"):
            evaluation.build_publication_receipt(hostile)

        receipt_path = evaluation.publish_bundle(self.bundle)
        self.assertEqual(receipt_path, self.root / evaluation.PUBLICATION_FILENAME)
        self.assertTrue((self.root / evaluation.INPUT_FILENAME).is_file())
        self.assertTrue((self.root / evaluation.MANIFEST_FILENAME).is_file())
        self.assertTrue((self.root / evaluation.REVIEW_CONTRACT_FILENAME).is_file())
        self.assertEqual(
            {path.name for path in (self.root / evaluation.SHARD_DIRECTORY).iterdir()},
            {f"{holder['job_id']}.json" for holder in evaluation.HOLDER_ROWS},
        )
        authority = json.loads(
            (self.root / evaluation.DIRECTORY_AUTHORITY_FILENAME).read_text()
        )
        receipt = json.loads(receipt_path.read_text())
        validated = evaluation.validate_publication_receipt(
            receipt,
            bundle=self.bundle,
            directory_authority=authority,
            verify_directory_authority=True,
        )
        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError,
            "materialized directory authority verification is required",
        ):
            evaluation.validate_publication_receipt(
                receipt,
                bundle=self.bundle,
            )
        self.assertTrue(validated["directory_authority_materialized"])
        self.assertEqual(len(validated["holder_completion_reservations"]), 4)
        self.assertEqual(
            {
                row["relative_path"]
                for row in validated["holder_completion_reservations"]
            },
            {
                evaluation.holder_completion_reservation_relative(
                    holder["job_id"]
                )
                for holder in evaluation.HOLDER_ROWS
            },
        )
        for row in validated["holder_completion_reservations"]:
            path = pathlib.Path(row["path"])
            self.assertEqual(path.read_bytes(), b"")
            self.assertEqual(
                stat.S_IMODE(path.stat().st_mode),
                evaluation.HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE,
            )
            self.assertEqual(row["identity"], evaluation._identity_row(path.stat()))
        self.assertEqual(authority["row_count"], 585)
        self.assertEqual(
            [row["relative_path"] for row in authority["rows"]],
            sorted(row["relative_path"] for row in authority["rows"]),
        )
        self.assertTrue(
            all(row["identity"] is not None for row in authority["rows"])
        )
        topology_by_relative = {
            row["relative_path"]: row for row in authority["rows"]
        }
        task_directory_count = 0
        for holder in evaluation.HOLDER_ROWS:
            holder_job_id = holder["job_id"]
            tasks = evaluation._shard_tasks(self.manifest, holder)
            task_ids = [
                task["record"]["candidate_id"]
                if task["task_kind"] == "adapter_candidate"
                else task["record"]["control_id"]
                for task in tasks
            ]
            task_parent = (
                f"{evaluation.EXECUTION_SHARD_DIRECTORY}/"
                f"{holder_job_id}/tasks"
            )
            self.assertEqual(
                topology_by_relative[task_parent]["expected_entries"],
                sorted(task_ids),
            )
            self.assertEqual(len(task_ids), 66)
            for task_id in task_ids:
                task_row = topology_by_relative[f"{task_parent}/{task_id}"]
                self.assertEqual(
                    task_row["owner_holder_job_id"], holder_job_id
                )
                self.assertEqual(task_row["expected_entries"], [])
                task_directory_count += 1
        self.assertEqual(task_directory_count, 264)
        self.assertEqual(
            {
                row["relative_path"]
                for row in authority["rows"]
                if not row["expected_entries"]
            },
            {
                str(pathlib.Path(record["output_relpath"]).parent)
                for record in self.manifest["candidates"]
                + self.manifest["frozen_base_controls"]
            }
            | {
                (
                    f"{evaluation.EXECUTION_SHARD_DIRECTORY}/"
                    f"{holder['job_id']}/tasks/"
                    + (
                        task["record"]["candidate_id"]
                        if task["task_kind"] == "adapter_candidate"
                        else task["record"]["control_id"]
                    )
                )
                for holder in evaluation.HOLDER_ROWS
                for task in evaluation._shard_tasks(self.manifest, holder)
            }
            | {
                f"{evaluation.EXECUTION_SHARD_DIRECTORY}/"
                f"{holder['job_id']}/"
                f"{evaluation.CONSUMPTION_AUTHORITY_DIRECTORY}"
                for holder in evaluation.HOLDER_ROWS
            },
        )
        with self.assertRaisesRegex(evaluation.DecodedEvaluationPlanError, "not fresh"):
            evaluation.publish_bundle(self.bundle)

    def test_retained_root_barrier_rejects_rename_out_replacement(self):
        displaced = self.parent / "decoded-eval-displaced"

        def barrier(event: str, root: pathlib.Path, relative: str) -> None:
            if event == "after-root-create" and relative == ".":
                os.rename(root, displaced)
                os.mkdir(root, 0o700)

        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError,
            "identity|drift|closure",
        ):
            evaluation.publish_bundle(
                self.bundle, publication_barrier=barrier
            )
        self.assertTrue(displaced.is_dir())
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertFalse((self.root / evaluation.PUBLICATION_FILENAME).exists())

    def test_retained_file_replay_rejects_prior_file_name_replacement(self):
        hostile_directory = tempfile.TemporaryDirectory()
        self.addCleanup(hostile_directory.cleanup)
        displaced = (
            pathlib.Path(hostile_directory.name)
            / "evaluation-input-displaced.json"
        )

        def barrier(event: str, root: pathlib.Path, relative: str) -> None:
            if event == "after-file-reserve" and relative == evaluation.MANIFEST_FILENAME:
                source = root / evaluation.INPUT_FILENAME
                os.rename(source, displaced)
                source.write_bytes(displaced.read_bytes())
                source.chmod(0o400)

        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError,
            "file identity|closure",
        ):
            evaluation.publish_bundle(
                self.bundle, publication_barrier=barrier
            )
        self.assertTrue(displaced.is_file())

    def test_retained_file_replay_rejects_external_hardlink(self):
        hostile_directory = tempfile.TemporaryDirectory()
        self.addCleanup(hostile_directory.cleanup)
        hostile_link = (
            pathlib.Path(hostile_directory.name)
            / "evaluation-input-hostile-hardlink.json"
        )

        def barrier(event: str, root: pathlib.Path, relative: str) -> None:
            if event == "after-file-reserve" and relative == evaluation.MANIFEST_FILENAME:
                os.link(root / evaluation.INPUT_FILENAME, hostile_link)

        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError,
            "file identity|closure",
        ):
            evaluation.publish_bundle(
                self.bundle, publication_barrier=barrier
            )
        self.assertTrue(hostile_link.is_file())

    def test_completion_reservation_rejects_replacement_and_hardlink(self):
        target_relative = evaluation.holder_completion_reservation_relative(
            evaluation.HOLDER_ROWS[0]["job_id"]
        )
        for attack in ("replace", "hardlink"):
            with self.subTest(attack=attack):
                root = self.parent / f"decoded-eval-{attack}"
                input_spec = evaluation.build_input_spec(
                    evaluation_id=f"reservation-{attack}",
                    evaluation_root=root,
                    pins=pins(), sources=sources(), checkpoints=checkpoints(),
                )
                bundle = evaluation.build_bundle(input_spec)
                attack_directory = self.parent / f"reservation-{attack}-attack"
                attack_directory.mkdir()
                displaced = attack_directory / "displaced"

                def barrier(event: str, observed_root: pathlib.Path, relative: str) -> None:
                    if event != "after-file-reserve" or relative != target_relative:
                        return
                    target = observed_root / relative
                    if attack == "replace":
                        os.rename(target, displaced)
                        target.write_bytes(b"")
                        target.chmod(
                            evaluation.HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE
                        )
                    else:
                        os.link(target, displaced)

                with self.assertRaisesRegex(
                    evaluation.DecodedEvaluationPlanError,
                    "file identity|closure|drift",
                ):
                    evaluation.publish_bundle(
                        bundle, publication_barrier=barrier
                    )
                self.assertTrue(displaced.is_file())

    def test_materialized_authority_retains_exact_topology_and_holder_scope(self):
        evaluation.publish_bundle(self.bundle)
        directory_authority = json.loads(
            (self.root / evaluation.DIRECTORY_AUTHORITY_FILENAME).read_text()
        )
        topology = evaluation.build_directory_topology(
            self.manifest, input_spec=self.input_spec
        )
        holder = evaluation.HOLDER_ROWS[0]["job_id"]
        owned = next(
            row["relative_path"]
            for row in topology
            if row["owner_holder_job_id"] == holder
            and row["expected_entries"] == []
        )
        unowned = next(
            row["relative_path"]
            for row in topology
            if row["owner_holder_job_id"] not in (None, holder)
            and row["expected_entries"] == []
        )
        retained = evaluation.RetainedPublicationRoot.open_materialized(
            self.root,
            directory_authority=directory_authority,
            topology=topology,
            holder_job_id=holder,
        )
        try:
            self.assertGreaterEqual(retained.directory_fd("."), 0)
            self.assertGreaterEqual(retained.directory_fd(owned), 0)
            retained.write_json(f"{owned}/result.json", {"status": "ok"})
            with self.assertRaisesRegex(
                evaluation.DecodedEvaluationPlanError,
                "does not own directory",
            ):
                retained.write_json(
                    f"{unowned}/forbidden.json", {"status": "hostile"}
                )
            with self.assertRaisesRegex(
                evaluation.DecodedEvaluationPlanError,
                "does not own directory",
            ):
                retained.set_directory_mode(unowned, 0o555)
        finally:
            retained.close()
        self.assertTrue((self.root / owned / "result.json").is_file())
        self.assertFalse((self.root / unowned / "forbidden.json").exists())
        self.assertEqual(stat.S_IMODE((self.root / unowned).stat().st_mode), 0o700)

    def test_captured_completion_reservation_rejects_named_replacement(self):
        evaluation.publish_bundle(self.bundle)
        directory_authority = json.loads(
            (self.root / evaluation.DIRECTORY_AUTHORITY_FILENAME).read_text()
        )
        publication_receipt = json.loads(
            (self.root / evaluation.PUBLICATION_FILENAME).read_text()
        )
        topology = evaluation.build_directory_topology(
            self.manifest, input_spec=self.input_spec
        )
        holder = evaluation.HOLDER_ROWS[0]["job_id"]
        target_relative = evaluation.holder_completion_reservation_relative(holder)
        attack_directory = self.parent / "captured-reservation-attack"
        attack_directory.mkdir()
        displaced = attack_directory / "displaced"

        def barrier(event: str, root: pathlib.Path, relative: str) -> None:
            if event == "after-holder-completion-capture":
                target = root / relative
                os.rename(target, displaced)
                target.write_bytes(b"")
                target.chmod(
                    evaluation.HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE
                )

        retained = evaluation.RetainedPublicationRoot.open_materialized(
            self.root,
            directory_authority=directory_authority,
            topology=topology,
            holder_job_id=holder,
            barrier=barrier,
        )
        try:
            with self.assertRaisesRegex(
                evaluation.DecodedEvaluationPlanError,
                "identity|closure|drift",
            ):
                retained.capture_holder_completion_reservation(
                    publication_receipt
                )
        finally:
            retained.close()
        self.assertTrue(displaced.is_file())
        self.assertEqual(
            target_relative,
            f"{evaluation.EXECUTION_SHARD_DIRECTORY}/{holder}"
            f"{evaluation.HOLDER_DIRECTORY_COMPLETION_SUFFIX}",
        )

    def test_materialized_reopen_uses_held_work_root_and_rejects_rename(self):
        authority_parent = self.parent / "held-authority-parent"
        authority_parent.mkdir()
        work_root = authority_parent / "held-work-root"
        work_root.mkdir(mode=0o700)
        evaluation_root = work_root / "evaluation"
        input_spec = evaluation.build_input_spec(
            evaluation_id="held-work-root-reopen",
            evaluation_root=evaluation_root,
            pins=pins(), sources=sources(), checkpoints=checkpoints(),
        )
        bundle_value = evaluation.build_bundle(input_spec)
        grand_fd = os.open(
            authority_parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        work_fd = os.open(
            work_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        parent_immutable = evaluation._immutable_directory_row(
            os.fstat(work_fd)
        )
        parent_parent_immutable = evaluation._immutable_directory_row(
            os.fstat(grand_fd)
        )
        publication = evaluation.publish_bundle_authorized(
            bundle_value,
            retained_parent_fd=work_fd,
            retained_parent_parent_fd=grand_fd,
            expected_parent_immutable_identity=parent_immutable,
            expected_parent_parent_immutable_identity=parent_parent_immutable,
        )
        directory_authority = publication["directory_authority"]
        topology = evaluation.build_directory_topology(
            bundle_value["manifest"], input_spec=input_spec
        )
        root_row = next(
            row for row in directory_authority["rows"]
            if row["relative_path"] == "."
        )
        root_authority = {
            "schema_version": "bernini-retained-directory-authority-v1",
            "path": str(evaluation_root),
            "identity": root_row["identity"],
            "parent_identity": root_row["parent_identity"],
            "entries": root_row["expected_entries"],
            "retained_parent_fd": True,
            "retained_root_fd": True,
        }
        displaced = authority_parent / "held-work-root-displaced"

        def barrier(event: str, root: pathlib.Path, relative: str) -> None:
            if event == "after-materialized-open":
                os.rename(work_root, displaced)
                work_root.mkdir(mode=0o700)

        try:
            with self.assertRaisesRegex(
                evaluation.DecodedEvaluationPlanError,
                "parent identity|identity|closure",
            ):
                evaluation.RetainedPublicationRoot.open_materialized(
                    evaluation_root,
                    directory_authority=directory_authority,
                    topology=topology,
                    holder_job_id=None,
                    retained_parent_fd=work_fd,
                    retained_parent_parent_fd=grand_fd,
                    expected_parent_immutable_identity=parent_immutable,
                    expected_parent_parent_immutable_identity=(
                        parent_parent_immutable
                    ),
                    expected_root_authority=root_authority,
                    barrier=barrier,
                )
        finally:
            os.close(work_fd)
            os.close(grand_fd)
        self.assertTrue(displaced.is_dir())
        self.assertEqual(list(work_root.iterdir()), [])

    def test_materialized_holders_accept_foreign_owned_leaf_progress_only(self):
        evaluation.publish_bundle(self.bundle)
        directory_authority = json.loads(
            (self.root / evaluation.DIRECTORY_AUTHORITY_FILENAME).read_text()
        )
        topology = evaluation.build_directory_topology(
            self.manifest, input_spec=self.input_spec
        )
        first_holder = evaluation.HOLDER_ROWS[0]["job_id"]
        second_holder = evaluation.HOLDER_ROWS[1]["job_id"]
        first_leaf = next(
            row["relative_path"]
            for row in topology
            if row["owner_holder_job_id"] == first_holder
            and row["expected_entries"] == []
        )
        hostile_directory = tempfile.TemporaryDirectory()
        self.addCleanup(hostile_directory.cleanup)
        first = evaluation.RetainedPublicationRoot.open_materialized(
            self.root,
            directory_authority=directory_authority,
            topology=topology,
            holder_job_id=first_holder,
        )
        second = evaluation.RetainedPublicationRoot.open_materialized(
            self.root,
            directory_authority=directory_authority,
            topology=topology,
            holder_job_id=second_holder,
        )
        try:
            first.write_json(f"{first_leaf}/result.json", {"status": "ok"})
            self.assertGreaterEqual(second.directory_fd("."), 0)
            displaced = pathlib.Path(hostile_directory.name) / "foreign-leaf-displaced"
            os.rename(self.root / first_leaf, displaced)
            (self.root / first_leaf).mkdir(mode=0o700)
            with self.assertRaisesRegex(
                evaluation.DecodedEvaluationPlanError,
                "identity|closure",
            ):
                second.directory_fd(".")
        finally:
            first.close()
            second.close()

    def test_holder_completions_merge_to_final_exact585_audit_authority(self):
        evaluation.publish_bundle(self.bundle)
        base = json.loads(
            (self.root / evaluation.DIRECTORY_AUTHORITY_FILENAME).read_text()
        )
        topology = evaluation.build_directory_topology(
            self.manifest, input_spec=self.input_spec
        )
        publication_receipt = json.loads(
            (self.root / evaluation.PUBLICATION_FILENAME).read_text()
        )
        first_holder = evaluation.HOLDER_ROWS[0]["job_id"]
        first_authority = evaluation.RetainedPublicationRoot.open_materialized(
            self.root,
            directory_authority=base,
            topology=topology,
            holder_job_id=first_holder,
        )
        self.addCleanup(first_authority.close)
        first_reservation = (
            first_authority.capture_holder_completion_reservation(
                publication_receipt
            )
        )
        holder_rows: dict[str, list[dict]] = {
            holder["job_id"]: [] for holder in evaluation.HOLDER_ROWS
        }
        for item in topology:
            holder = item["owner_holder_job_id"]
            if (
                holder is None
                or item not in evaluation._holder_mutable_topology_rows(
                    topology, holder
                )
            ):
                continue
            leaf = self.root / item["relative_path"]
            if item["relative_path"].startswith(
                (
                    evaluation.OUTPUT_CANDIDATE_DIRECTORY + "/",
                    evaluation.OUTPUT_BASE_DIRECTORY + "/",
                )
            ):
                added_names = [
                    f"{policy}.mp4" for policy in evaluation.POLICIES
                ]
            elif item["relative_path"].endswith("/tasks"):
                added_names = []
            else:
                added_names = ["holder-summary.json"]
            names = sorted(set(item["expected_entries"]) | set(added_names))
            for name in added_names:
                output = leaf / name
                output.write_bytes(name.encode("utf-8"))
                output.chmod(0o400)
            if holder == first_holder:
                first_authority.refresh_owned_directory(
                    item["relative_path"], expected_entries=set(names)
                )
            holder_rows[holder].append(
                {
                    "relative_path": item["relative_path"],
                    "path": str(leaf),
                    "owner_holder_job_id": holder,
                    "expected_mode": item["expected_mode"],
                    "expected_entries": sorted(names),
                    "identity": evaluation._identity_row(leaf.stat()),
                    "parent_identity": evaluation._identity_row(
                        leaf.parent.stat()
                    ),
                }
            )
        completions = {}
        for holder in (item["job_id"] for item in evaluation.HOLDER_ROWS):
            value = {
                "schema_version": evaluation.HOLDER_DIRECTORY_COMPLETION_SCHEMA,
                "evaluation_root": str(self.root),
                "base_authority_digest": base["authority_digest"],
                "base_topology_digest": base["topology_digest"],
                "holder_job_id": holder,
                "holder_summary_digest": digest(f"holder-summary:{holder}"),
                "rows": sorted(
                    holder_rows[holder], key=lambda item: item["relative_path"]
                ),
                "row_count": len(holder_rows[holder]),
            }
            value["completion_digest"] = evaluation.object_sha256(value)
            completions[holder] = value
        first_binding = first_authority.fill_holder_completion_reservation(
            first_reservation,
            completions[first_holder],
            topology=topology,
            base_directory_authority=base,
        )
        self.assertEqual(
            first_binding["path"],
            str(
                self.root
                / evaluation.holder_completion_reservation_relative(first_holder)
            ),
        )
        self.assertEqual(
            first_binding["mode"],
            evaluation.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE,
        )
        self.assertEqual(
            json.loads(pathlib.Path(first_binding["path"]).read_text()),
            completions[first_holder],
        )
        first_authority.close()
        reservation_by_holder = {
            row["holder_job_id"]: row
            for row in publication_receipt["holder_completion_reservations"]
        }
        for holder, completion in completions.items():
            if holder == first_holder:
                continue
            row = reservation_by_holder[holder]
            descriptor = os.open(row["path"], os.O_WRONLY | os.O_NOFOLLOW)
            try:
                self.assertEqual(
                    evaluation._identity_row(os.fstat(descriptor)),
                    row["identity"],
                )
                payload = evaluation.canonical_json_bytes(completion) + b"\n"
                self.assertEqual(os.write(descriptor, payload), len(payload))
                os.fchmod(
                    descriptor,
                    evaluation.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE,
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        merged = evaluation.merge_holder_directory_completions(
            topology=topology,
            base_directory_authority=base,
            completions=completions,
        )
        self.assertEqual(merged["mutable_row_count"], 404)
        self.assertEqual(len(merged["topology"]), 585)
        final = evaluation.RetainedPublicationRoot.open_materialized(
            self.root,
            directory_authority=merged["directory_authority"],
            topology=merged["topology"],
            holder_job_id=None,
        )
        try:
            self.assertGreaterEqual(final.directory_fd("."), 0)
            first_task = evaluation._shard_tasks(
                self.manifest, evaluation.HOLDER_ROWS[0]
            )[0]
            first_task_id = first_task["record"]["candidate_id"]
            first_task_relative = (
                f"{evaluation.EXECUTION_SHARD_DIRECTORY}/"
                f"{evaluation.HOLDER_ROWS[0]['job_id']}/tasks/"
                f"{first_task_id}"
            )
            self.assertGreaterEqual(
                final.directory_fd(first_task_relative), 0
            )
            captured = final.capture_filled_holder_completions(
                publication_receipt,
                topology=topology,
                base_directory_authority=base,
            )
            self.assertEqual(set(captured), set(completions))
            self.assertEqual(
                {
                    holder: row["completion"]["completion_digest"]
                    for holder, row in captured.items()
                },
                {
                    holder: completion["completion_digest"]
                    for holder, completion in completions.items()
                },
            )
            with self.assertRaisesRegex(
                evaluation.DecodedEvaluationPlanError, "read-only"
            ):
                final.write_json("forbidden.json", {"hostile": True})
        finally:
            final.close()

        hostile = copy.deepcopy(completions)
        hostile[first_holder]["rows"][0]["unexpected"] = True
        hostile[first_holder] = sign(
            hostile[first_holder], "completion_digest"
        )
        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError, "field closure differs"
        ):
            evaluation.merge_holder_directory_completions(
                topology=topology,
                base_directory_authority=base,
                completions=hostile,
            )

    def test_reserve_file_closes_descriptor_on_preregistration_failure(self):
        authority = evaluation.RetainedPublicationRoot.create(self.root)
        captured: list[int] = []
        real_open = os.open

        def tracking_open(*args, **kwargs):
            descriptor = real_open(*args, **kwargs)
            if args and args[0] == "leak-probe.json":
                captured.append(descriptor)
            return descriptor

        try:
            with mock.patch.object(os, "open", side_effect=tracking_open), mock.patch.object(
                os, "set_inheritable", side_effect=OSError("hostile failure")
            ):
                with self.assertRaisesRegex(OSError, "hostile failure"):
                    authority.reserve_file("leak-probe.json")
            self.assertEqual(len(captured), 1)
            with self.assertRaises(OSError):
                os.fstat(captured[0])
        finally:
            authority.close()

    def test_materialized_authority_rejects_root_replacement_and_open_objects(self):
        evaluation.publish_bundle(self.bundle)
        authority_path = self.root / evaluation.DIRECTORY_AUTHORITY_FILENAME
        directory_authority = json.loads(authority_path.read_text())
        topology = evaluation.build_directory_topology(
            self.manifest, input_spec=self.input_spec
        )
        hostile = copy.deepcopy(directory_authority)
        hostile["rows"][0]["unexpected"] = True
        hostile = sign(hostile, "authority_digest")
        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError, "field closure differs"
        ):
            evaluation.validate_directory_authority(
                hostile, topology=topology, materialized_required=True
            )

        displaced = self.parent / "decoded-eval-materialized-displaced"

        def barrier(event: str, root: pathlib.Path, relative: str) -> None:
            if event == "after-materialized-open" and relative == ".":
                os.rename(root, displaced)
                os.mkdir(root, 0o700)

        with self.assertRaisesRegex(
            evaluation.DecodedEvaluationPlanError,
            "identity|drift|closure",
        ):
            evaluation.RetainedPublicationRoot.open_materialized(
                self.root,
                directory_authority=directory_authority,
                topology=topology,
                holder_job_id=evaluation.HOLDER_ROWS[0]["job_id"],
                barrier=barrier,
            )
        self.assertTrue(displaced.is_dir())
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
