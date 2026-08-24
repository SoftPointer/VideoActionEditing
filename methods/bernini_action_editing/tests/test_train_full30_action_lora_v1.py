#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = METHOD_ROOT / "train_full30_action_lora_v1.py"
SPEC = importlib.util.spec_from_file_location(
    "train_full30_action_lora_v1_test_subject", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def sha(character: str) -> str:
    return character * 64


@dataclass(frozen=True)
class FakeStepRecord:
    marker: str


@dataclass(frozen=True)
class FakeStepResult:
    receipt: object
    optimizer_receipt: object


class FakeStepModule:
    SCHEMA_VERSION = "fake-step-core-v1"
    Full30LocalMicroRecordV1 = FakeStepRecord
    Full30ActionTrainingStepResultV1 = FakeStepResult

    def __init__(self, *, fail=False, mismatch=False):
        self.fail = fail
        self.mismatch = mismatch
        self.calls = []

    @staticmethod
    def canonical_receipt_bytes(receipt):
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest", None)
        if declared != trainer.object_sha256(unsigned):
            raise ValueError("bad receipt")
        return trainer.canonical_json_bytes(dict(receipt))

    def execute_full30_action_training_step_v1(self, **kwargs):
        self.calls.append(kwargs)
        optimizer = kwargs["optimizer"]
        before = optimizer.update_count
        if self.fail:
            raise RuntimeError("injected sole-step failure")
        optimizer_receipt = {
            "schema_version": "fake-optimizer-v1",
            "status": "committed",
            "arm": kwargs["arm"],
            "update_count_before": before,
            "update_count_after": before + 1,
        }
        optimizer_receipt = {
            **optimizer_receipt,
            "receipt_digest": trainer.object_sha256(optimizer_receipt),
        }
        optimizer.update_count += 1
        embedded_digest = (
            sha("f") if self.mismatch else optimizer_receipt["receipt_digest"]
        )
        step_receipt = {
            "schema_version": "fake-step-v1",
            "status": "committed",
            "arm": kwargs["arm"],
            "update_count_before": before,
            "update_count_after": before + 1,
            "optimizer": {
                "step_call_count": 1,
                "receipt_digest": embedded_digest,
            },
        }
        step_receipt = {
            **step_receipt,
            "receipt_digest": trainer.object_sha256(step_receipt),
        }
        return FakeStepResult(step_receipt, optimizer_receipt)


class Full30TrainerThinAdapterTests(unittest.TestCase):
    def _records(self):
        teacher = trainer.TeacherPacketV1(
            teacher_unit=None,
            minimum_amplitude=None,
            minimum_amplitude_float32_le_sha256=sha("a"),
            minimum_amplitude_bundle_digest=sha("b"),
            minimum_amplitude_calibration_id="calibration",
            nuisance_packet=None,
            authority_receipt={},
        )
        return tuple(
            trainer.PreparedRecordV1(
                schedule_row=SimpleNamespace(update=0, microbatch=index),
                runtime_record=SimpleNamespace(row_id=f"row-{index}"),
                teacher=teacher,
                step_record=FakeStepRecord(str(index)),
                authority_receipt={"global_index": index},
            )
            for index in range(4)
        )

    def test_adapter_invokes_only_registered_step_and_checks_exact_receipts(self):
        module = FakeStepModule()
        optimizer = SimpleNamespace(update_count=0)
        records = self._records()
        result = trainer.execute_one_update_v1(
            arm="action+retain",
            records=records,
            runtime=object(),
            optimizer=optimizer,
            full_schedule=("sealed-schedule",),
            rank=3,
            training_step_module=module,
            gradient_mean=lambda request: {},
            world_consensus=lambda request: {},
            optimizer_all_reduce_sum=lambda value: None,
        )
        self.assertEqual(optimizer.update_count, 1)
        self.assertEqual(len(module.calls), 1)
        call = module.calls[0]
        self.assertEqual(call["rank"], 3)
        self.assertEqual(call["update_index"], 0)
        self.assertEqual(
            call["local_records"],
            tuple(item.step_record for item in records),
        )
        self.assertEqual(
            result.receipt["optimizer"]["receipt_digest"],
            result.optimizer_receipt["receipt_digest"],
        )

    def test_adapter_failure_does_not_count_an_update(self):
        module = FakeStepModule(fail=True)
        optimizer = SimpleNamespace(update_count=0)
        with self.assertRaisesRegex(RuntimeError, "injected sole-step failure"):
            trainer.execute_one_update_v1(
                arm="action-only",
                records=self._records(),
                runtime=object(),
                optimizer=optimizer,
                full_schedule=("sealed-schedule",),
                rank=0,
                training_step_module=module,
                gradient_mean=lambda request: {},
                world_consensus=lambda request: {},
                optimizer_all_reduce_sum=lambda value: None,
            )
        self.assertEqual(optimizer.update_count, 0)
        self.assertEqual(len(module.calls), 1)

    def test_adapter_rejects_receipt_drift_between_step_and_optimizer(self):
        module = FakeStepModule(mismatch=True)
        optimizer = SimpleNamespace(update_count=0)
        with self.assertRaisesRegex(
            trainer.Full30ActionTrainingError,
            "not exactly bound",
        ):
            trainer.execute_one_update_v1(
                arm="action+retain",
                records=self._records(),
                runtime=object(),
                optimizer=optimizer,
                full_schedule=("sealed-schedule",),
                rank=0,
                training_step_module=module,
                gradient_mean=lambda request: {},
                world_consensus=lambda request: {},
                optimizer_all_reduce_sum=lambda value: None,
            )
        self.assertEqual(len(module.calls), 1)

    def test_adapter_rejects_nonexact_local_record_closure_before_step(self):
        module = FakeStepModule()
        optimizer = SimpleNamespace(update_count=0)
        with self.assertRaisesRegex(
            trainer.Full30ActionTrainingError, "exactly four"
        ):
            trainer.execute_one_update_v1(
                arm="action-only",
                records=self._records()[:3],
                runtime=object(),
                optimizer=optimizer,
                full_schedule=("sealed-schedule",),
                rank=0,
                training_step_module=module,
                gradient_mean=lambda request: {},
                world_consensus=lambda request: {},
                optimizer_all_reduce_sum=lambda value: None,
            )
        self.assertEqual(module.calls, [])
        self.assertEqual(optimizer.update_count, 0)


class Full30TrainerGlobalReceiptTests(unittest.TestCase):
    @staticmethod
    def _optimizer_receipt():
        value = {
            "schema_version": "fake-optimizer-v1",
            "status": "committed",
            "arm": "action+retain",
            "update_count_before": 0,
            "update_count_after": 1,
        }
        return {**value, "receipt_digest": trainer.object_sha256(value)}

    def _world_envelopes(self):
        optimizer = self._optimizer_receipt()
        envelopes = []
        for rank in range(8):
            dp_rank = rank // 4
            indices = [microbatch * 2 + dp_rank for microbatch in range(4)]
            records = [
                {
                    "global_index": index,
                    "row_id": f"row-{index}",
                    "objective_authority_digest": sha(
                        "a" if index % 2 == 0 else "b"
                    ),
                }
                for index in indices
            ]
            authority = [
                {
                    "global_index": row["global_index"],
                    "row_id": row["row_id"],
                    "objective_authority_digest": row[
                        "objective_authority_digest"
                    ],
                }
                for row in records
            ]
            step = {
                "schema_version": "fake-step-receipt-v1",
                "status": "committed",
                "arm": "action+retain",
                "rank": rank,
                "world_size": 8,
                "dp_size": 2,
                "sp_size": 4,
                "update_count_before": 0,
                "update_count_after": 1,
                "global_batch": 8,
                "schedule": {"local_global_indices": indices},
                "inventory": {"inventory_sha256": sha("c")},
                "records": records,
                "noop_replay_records": [
                    {"global_index": index, "row_id": f"row-{index}"}
                    for index in indices
                ],
                "runtime": {"formal_physical_evaluation_count": 32},
                "gradients": {
                    "action_sha256": sha("d"),
                    "noop_sha256": sha("e"),
                    "coverage_gate": {"passed": True},
                },
                "world_consensus": {"consensus_digest": sha("f")},
                "optimizer": {
                    "step_call_count": 1,
                    "receipt_digest": optimizer["receipt_digest"],
                },
                "objective_contract": {
                    "noop_target_is_epsilon_minus_real_source": True
                },
            }
            step = {
                **step,
                "receipt_digest": trainer.object_sha256(step),
            }
            envelope = {
                "rank": rank,
                "step_receipt": step,
                "optimizer_receipt": copy.deepcopy(optimizer),
                "record_authority": authority,
            }
            envelope["envelope_digest"] = trainer.object_sha256(envelope)
            envelopes.append(envelope)
        return envelopes

    def _call(self, envelopes):
        class FakeDist:
            @staticmethod
            def all_gather_object(output, _local, group=None):
                output[:] = copy.deepcopy(envelopes)

        module = FakeStepModule()
        modules = SimpleNamespace(training_step=module)
        prepared = []
        teacher = trainer.TeacherPacketV1(
            teacher_unit=None,
            minimum_amplitude=None,
            minimum_amplitude_float32_le_sha256=sha("1"),
            minimum_amplitude_bundle_digest=sha("2"),
            minimum_amplitude_calibration_id="calibration",
            nuisance_packet=None,
            authority_receipt={},
        )
        for authority in envelopes[0]["record_authority"]:
            prepared.append(
                trainer.PreparedRecordV1(
                    schedule_row=SimpleNamespace(),
                    runtime_record=SimpleNamespace(),
                    teacher=teacher,
                    step_record=FakeStepRecord("record"),
                    authority_receipt=authority,
                )
            )
        parallel = SimpleNamespace(
            contract=SimpleNamespace(rank=0), world_group=object()
        )
        result = FakeStepResult(
            envelopes[0]["step_receipt"],
            envelopes[0]["optimizer_receipt"],
        )
        original_import = trainer.importlib.import_module

        def import_module(name):
            return FakeDist if name == "torch.distributed" else original_import(name)

        with mock.patch.object(
            trainer.importlib, "import_module", side_effect=import_module
        ):
            return trainer.gather_global_update_receipt_v1(
                step_result=result,
                prepared_records=prepared,
                parallel=parallel,
                modules=modules,
            )

    def test_global_receipt_closes_eight_records_and_exact_optimizer_receipts(self):
        receipt = self._call(self._world_envelopes())
        self.assertEqual(receipt["global_record_count"], 8)
        self.assertEqual(receipt["global_physical_evaluations"], 32)
        self.assertEqual(
            [item["step"]["global_index"] for item in receipt["records"]],
            list(range(8)),
        )
        self.assertTrue(receipt["WORLD8_optimizer_receipts_exactly_equal"])
        self.assertEqual(len(receipt["dp_leader_step_receipts"]), 2)

    def test_global_receipt_rejects_one_rank_optimizer_receipt_drift(self):
        envelopes = self._world_envelopes()
        envelopes[7]["optimizer_receipt"]["arm"] = "action-only"
        unsigned = dict(envelopes[7]["optimizer_receipt"])
        unsigned.pop("receipt_digest")
        envelopes[7]["optimizer_receipt"]["receipt_digest"] = (
            trainer.object_sha256(unsigned)
        )
        envelope_unsigned = dict(envelopes[7])
        envelope_unsigned.pop("envelope_digest")
        envelopes[7]["envelope_digest"] = trainer.object_sha256(
            envelope_unsigned
        )
        with self.assertRaisesRegex(
            trainer.Full30ActionTrainingError,
            "binding differs|not exactly equal",
        ):
            self._call(envelopes)


class Full30TrainerAdmissionTests(unittest.TestCase):
    def setUp(self):
        trainer._RELEASE_VALIDATED = False
        trainer._VALIDATED_RELEASE_RECEIPT = None
        trainer._BOUND_RELEASE_MODULES.clear()
        trainer._BUSINESS_MODULES = None

    def tearDown(self):
        for name, module in tuple(trainer._BOUND_RELEASE_MODULES.items()):
            if sys.modules.get(name) is module:
                del sys.modules[name]
        trainer._RELEASE_VALIDATED = False
        trainer._VALIDATED_RELEASE_RECEIPT = None
        trainer._BOUND_RELEASE_MODULES.clear()
        trainer._BUSINESS_MODULES = None

    def test_sealed_mapping_receipts_are_canonical_json_serializable(self):
        sealed = trainer._seal(
            {"nested": MappingProxyType({"value": 1})}
        )
        raw = trainer.canonical_json_bytes(sealed)
        self.assertEqual(
            raw,
            b'{"nested":{"value":1},"receipt_digest":"'
            + sealed["receipt_digest"].encode("ascii")
            + b'"}',
        )

    def test_boundary_profiles_are_closed(self):
        canary = trainer.build_boundary_plan_v1(
            "disposable-canary-2", start_update=0, stop_update=2
        )
        self.assertEqual(canary.checkpoint_updates, (0, 1, 2))
        self.assertEqual(canary.receipt()["maximum_updates"], 2)
        fresh = trainer.build_boundary_plan_v1(
            "review-gated-segment", start_update=0, stop_update=20
        )
        self.assertEqual(fresh.checkpoint_updates, (0, 5, 10, 20))
        self.assertEqual(fresh.receipt()["maximum_updates"], 160)
        resumed = trainer.build_boundary_plan_v1(
            "review-gated-segment", start_update=20, stop_update=40
        )
        self.assertTrue(resumed.requires_resume)
        self.assertEqual(resumed.checkpoint_updates, (40,))
        for start, stop in ((0, 40), (10, 20), (20, 60), (140, 161)):
            with self.subTest(start=start, stop=stop):
                with self.assertRaises(trainer.Full30ActionTrainingError):
                    trainer.build_boundary_plan_v1(
                        "review-gated-segment",
                        start_update=start,
                        stop_update=stop,
                    )

    def test_business_imports_are_blocked_before_release_validation(self):
        before = set(sys.modules)
        with self.assertRaisesRegex(
            trainer.Full30ActionTrainingError, "before executed release"
        ):
            trainer.load_business_modules_v1()
        after = set(sys.modules)
        self.assertNotIn("full30_action_checkpoint_v1", after - before)
        self.assertNotIn("full30_action_training_step_v1", after - before)
        self.assertNotIn("full30_action_data_teacher_authority_v1", after - before)
        self.assertNotIn("full30_action_amplitude_authority_v1", after - before)

    def test_production_release_hard_pins_unique_step_runtime_and_authority_apis(self):
        self.assertEqual(
            trainer.FROZEN_RELEASE_MEMBER_SHA256[
                "full30_action_training_step_v1.py"
            ],
            "c3cf9b5f51a0247de20ad687b51109e8088e297d20c17fec722362da4c4c7ee2",
        )
        self.assertEqual(
            trainer.FROZEN_RELEASE_MEMBER_SHA256[
                "full30_action_data_teacher_authority_v1.py"
            ],
            "d210628791e6861b3cfb141bd9bca930966b9dfc9050d54460e69c18d0883e2a",
        )
        self.assertEqual(
            trainer.FROZEN_RELEASE_MEMBER_SHA256[
                "full30_action_amplitude_authority_v1.py"
            ],
            "103f9f676b8126615d6fa7916b9c9e4dd37003fbacda0055f046d6a8de8f0f93",
        )
        self.assertEqual(
            trainer.FROZEN_RELEASE_MEMBER_SHA256[
                "full30_action_psiout_materializer_v1.py"
            ],
            "a7daf7f81956818669f2d23e806034ab902aa34bcbb8e76315f1d2ee89c53b45",
        )
        self.assertEqual(
            trainer.FROZEN_RELEASE_MEMBER_SHA256[
                "full30_action_mechanism_canary_authority_v1.py"
            ],
            hashlib.sha256(
                (METHOD_ROOT / "full30_action_mechanism_canary_authority_v1.py").read_bytes()
            ).hexdigest(),
        )
        self.assertIn(
            "full30_action_training_step_v1.py",
            trainer.REQUIRED_RELEASE_FILES,
        )
        self.assertIn(
            "full30_action_amplitude_authority_v1.py",
            trainer.REQUIRED_RELEASE_FILES,
        )
        self.assertIn(
            "full30_action_psiout_materializer_v1.py",
            trainer.REQUIRED_RELEASE_FILES,
        )
        self.assertIn(
            "full30_action_mechanism_canary_authority_v1.py",
            trainer.REQUIRED_RELEASE_FILES,
        )
        self.assertTrue(
            {
                "infer_dclr_reward_runtime_smoke.py",
                "graft_phase_a_native_training_closure_v1.py",
                "infer_source_kv_carrier_oracle.py",
                "infer_lora.py",
                "source_kv_replay.py",
                "source_kv_route_batches.py",
            }.issubset(trainer.REQUIRED_RELEASE_FILES)
        )

    def test_amplitude_runtime_binding_requires_every_actual_identity(self):
        actual = {
            "schema_version": trainer.AMPLITUDE_RUNTIME_IDENTITY_SCHEMA_VERSION,
            "bernini_revision": "1" * 40,
            "veomni_revision": "6" * 40,
            "official_checkpoint_tree_sha256": sha("2"),
            "transformer_config_sha256": sha("3"),
            "sigma_table_sha256": sha("4"),
            "psiout_protocol_sha256": sha("5"),
            "official_provider_source_sha256": sha("7"),
            "official_provider_abi": trainer.OFFICIAL_PROVIDER_ABI,
            "compute_contract": dict(trainer.AMPLITUDE_COMPUTE_CONTRACT),
            "compute_contract_digest": trainer.object_sha256(
                trainer.AMPLITUDE_COMPUTE_CONTRACT
            ),
            "frame_count": 81,
            "fps": 25.0,
            "sampler_steps": 40,
        }
        runtime_digest = trainer.object_sha256(actual)
        authority = SimpleNamespace(
            amplitude_runtime_identity={
                **actual,
                "runtime_digest": runtime_digest,
            },
            amplitude=SimpleNamespace(
                manifest_file_sha256=sha("8"),
                frozen_runtime_digest=runtime_digest,
                validation_receipt={"validation_digest": sha("8")},
            ),
        )
        receipt = trainer.validate_amplitude_runtime_binding_v1(
            authority_index=authority,
            bernini_revision=actual["bernini_revision"],
            veomni_revision=actual["veomni_revision"],
            model_sha256=actual["official_checkpoint_tree_sha256"],
            transformer_config_sha256=actual["transformer_config_sha256"],
            sigma_table_sha256=actual["sigma_table_sha256"],
            psiout_protocol_sha256=actual["psiout_protocol_sha256"],
            official_provider_source_sha256=actual[
                "official_provider_source_sha256"
            ],
            executed_runtime_source_sha256=trainer.FROZEN_RELEASE_MEMBER_SHA256[
                "full30_action_runtime_v1.py"
            ],
        )
        self.assertTrue(receipt["all_runtime_identities_exactly_equal"])
        with self.assertRaisesRegex(
            trainer.Full30ActionTrainingError, "different runtime"
        ):
            trainer.validate_amplitude_runtime_binding_v1(
                authority_index=authority,
                bernini_revision=actual["bernini_revision"],
                veomni_revision=actual["veomni_revision"],
                model_sha256=actual["official_checkpoint_tree_sha256"],
                transformer_config_sha256=actual["transformer_config_sha256"],
                sigma_table_sha256=actual["sigma_table_sha256"],
                psiout_protocol_sha256=sha("9"),
                official_provider_source_sha256=actual[
                    "official_provider_source_sha256"
                ],
                executed_runtime_source_sha256=trainer.FROZEN_RELEASE_MEMBER_SHA256[
                    "full30_action_runtime_v1.py"
                ],
            )

    def test_teacher_checkpoint_identity_is_canonical_amplitude_composite(self):
        manifest = {
            "sources": [
                {
                    "source_iid": "source",
                    "analysis_split": "fit",
                    "source_video_sha256": sha("1"),
                    "source_posterior_index0_sha256": sha("2"),
                    "source_digest": sha("3"),
                }
            ],
            "pairs": [
                {
                    "pair_id": "pair",
                    "analysis_split": "fit",
                    "source_iid": "source",
                    "branch": "action",
                    "teacher_cell_id": "cell",
                    "instruction_utf8_sha256": sha("4"),
                    "pair_digest": sha("5"),
                }
            ],
            "teacher_origins": [
                {
                    "teacher_cell_id": "cell",
                    "analysis_split": "fit",
                    "origin_digest": sha("6"),
                }
            ],
            "representation_admissions": [
                {
                    "admission_id": "admission",
                    "teacher_cell_id": "cell",
                    "analysis_split": "fit",
                    "branch": "action",
                    "origin_evidence": {
                        "psiout_sidecar_sha256": sha("7"),
                        "nuisance_packet_sha256": sha("8"),
                    },
                    "cross_anchor_evidence": {
                        "nuisance_packet_sha256": sha("9")
                    },
                    "admission_digest": sha("a"),
                }
            ],
        }
        first = trainer._authority_projection_digests(
            manifest,
            amplitude_manifest_sha256=sha("b"),
            amplitude_validation_digest=sha("c"),
        )
        second = trainer._authority_projection_digests(
            manifest,
            amplitude_manifest_sha256=sha("d"),
            amplitude_validation_digest=sha("c"),
        )
        self.assertEqual(first["data_sha256"], second["data_sha256"])
        self.assertEqual(first["parent_teacher_sha256"], second["parent_teacher_sha256"])
        self.assertNotEqual(first["teacher_sha256"], second["teacher_sha256"])
        expected = trainer.object_sha256(
            {
                "schema_version": trainer.TEACHER_COMPOSITE_SCHEMA_VERSION,
                "parent_teacher_projection_sha256": first[
                    "parent_teacher_sha256"
                ],
                "amplitude_manifest_file_sha256": sha("b"),
                "amplitude_validation_digest": sha("c"),
            }
        )
        self.assertEqual(first["teacher_sha256"], expected)

    def _release(self, parent: Path):
        root = parent / "method"
        root.mkdir()
        files = {
            "alpha.py": b"VALUE = 'alpha'\n",
            "nested/beta.py": b"VALUE = 'beta'\n",
        }
        for relative, raw in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        rows = [
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for relative, raw in files.items()
        ]
        release_sha = trainer.object_sha256(
            {
                "schema_version": trainer.RELEASE_SCHEMA_VERSION,
                "files": rows,
            }
        )
        unsigned = {
            "schema_version": trainer.RELEASE_SCHEMA_VERSION,
            "exact_member_closure": True,
            "files": rows,
            "release_sha256": release_sha,
        }
        manifest_value = {
            **unsigned,
            "manifest_digest": trainer.object_sha256(unsigned),
        }
        manifest = parent / "release.json"
        manifest.write_bytes(trainer.canonical_json_bytes(manifest_value) + b"\n")
        return (
            root.resolve(),
            manifest.resolve(),
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            release_sha,
        )

    def test_release_validator_closes_executed_files_before_unlocking_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, manifest_sha, release_sha = self._release(
                Path(directory)
            )
            receipt = trainer.validate_executed_release_v1(
                method_root=root,
                manifest=manifest,
                expected_manifest_sha256=manifest_sha,
                expected_release_sha256=release_sha,
                test_only_required_files={"alpha.py", "nested/beta.py"},
                test_only_require_current_entrypoint=False,
            )
            self.assertTrue(receipt["exact_member_closure_verified"])
            self.assertTrue(trainer._RELEASE_VALIDATED)

    def test_release_import_binder_rejects_hostile_preload_even_with_forged_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, manifest_sha, release_sha = self._release(
                Path(directory)
            )
            receipt = trainer.validate_executed_release_v1(
                method_root=root,
                manifest=manifest,
                expected_manifest_sha256=manifest_sha,
                expected_release_sha256=release_sha,
                test_only_required_files={"alpha.py", "nested/beta.py"},
                test_only_require_current_entrypoint=False,
            )
            forged = ModuleType("alpha")
            forged.__file__ = str(root / "alpha.py")
            forged.__spec__ = importlib.util.spec_from_file_location(
                "alpha", root / "alpha.py"
            )
            sys.modules["alpha"] = forged
            try:
                with self.assertRaisesRegex(
                    trainer.Full30ActionTrainingError, "preloaded outside"
                ):
                    trainer._secure_import_release_module_v1(
                        module_name="alpha",
                        relative_path="alpha.py",
                        release_receipt=receipt,
                    )
            finally:
                if sys.modules.get("alpha") is forged:
                    del sys.modules["alpha"]
            trusted = trainer._secure_import_release_module_v1(
                module_name="alpha",
                relative_path="alpha.py",
                release_receipt=receipt,
            )
            self.assertEqual(trusted.VALUE, "alpha")
            self.assertEqual(Path(trusted.__file__), root / "alpha.py")

    def test_release_validator_rejects_unlisted_or_mutated_executed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, manifest_sha, release_sha = self._release(
                Path(directory)
            )
            (root / "extra.py").write_text("extra\n", encoding="ascii")
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError, "exact release"
            ):
                trainer.validate_executed_release_v1(
                    method_root=root,
                    manifest=manifest,
                    expected_manifest_sha256=manifest_sha,
                    expected_release_sha256=release_sha,
                    test_only_required_files={"alpha.py"},
                    test_only_require_current_entrypoint=False,
                )
            trainer._RELEASE_VALIDATED = False
            (root / "extra.py").unlink()
            (root / "alpha.py").write_text("mutated\n", encoding="ascii")
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError, "member differs"
            ):
                trainer.validate_executed_release_v1(
                    method_root=root,
                    manifest=manifest,
                    expected_manifest_sha256=manifest_sha,
                    expected_release_sha256=release_sha,
                    test_only_required_files={"alpha.py"},
                    test_only_require_current_entrypoint=False,
                )

    def test_reduced_canary_transitions_to_independent_fresh_formal_u0(self):
        release_sha, model_sha = sha("1"), sha("2")
        optimizer_code_sha, training_step_code_sha = sha("0"), sha("1")
        runtime_sha, objective_sha = sha("b"), sha("c")
        inventory_sha = sha("7")
        architecture = {"scope": "all-attention", "digest": sha("a")}
        canary_authority_sha, canary_amplitude_sha = sha("3"), sha("4")
        canary_noise_sha, canary_schedule_sha = sha("5"), sha("6")
        canary_prefixes = {0: sha("8"), 1: sha("9"), 2: sha("a")}
        canary_projections = {
            "data_sha256": sha("b"),
            "parent_teacher_sha256": sha("c"),
            "amplitude_manifest_sha256": canary_amplitude_sha,
            "amplitude_validation_digest": sha("d"),
            "teacher_sha256": sha("e"),
            "nuisance_sha256": sha("f"),
        }
        formal_authority_sha, formal_amplitude_sha = sha("d"), sha("e")
        formal_noise_sha, formal_schedule_sha = sha("f"), sha("0")
        formal_prefix_u0 = sha("1")
        formal_projections = {
            "data_sha256": sha("2"),
            "parent_teacher_sha256": sha("3"),
            "amplitude_manifest_sha256": formal_amplitude_sha,
            "amplitude_validation_digest": None,
            "teacher_sha256": sha("4"),
            "nuisance_sha256": sha("5"),
        }
        canary_bindings = {
            "arm": "action+retain",
            "release_sha256": release_sha,
            "model_sha256": model_sha,
            "data_sha256": canary_projections["data_sha256"],
            "teacher_sha256": canary_projections["teacher_sha256"],
            "nuisance_sha256": canary_projections["nuisance_sha256"],
            "noise_sha256": canary_noise_sha,
            "runtime_sha256": runtime_sha,
            "objective_sha256": objective_sha,
        }
        formal_bindings = {
            **canary_bindings,
            "data_sha256": formal_projections["data_sha256"],
            "teacher_sha256": formal_projections["teacher_sha256"],
            "nuisance_sha256": formal_projections["nuisance_sha256"],
            "noise_sha256": formal_noise_sha,
        }
        canary_plan = trainer.build_boundary_plan_v1(
            "disposable-canary-2", start_update=0, stop_update=2
        )
        formal_plan = trainer.build_boundary_plan_v1(
            "review-gated-segment", start_update=0, stop_update=20
        )

        def validation(unsigned):
            return {
                **unsigned,
                "validation_digest": trainer.object_sha256(unsigned),
            }

        canary_data_validation = validation(
            {
                "schema_version": "bernini-full30-action-mechanism-canary-validation-v1",
                "manifest_file_sha256": canary_authority_sha,
                "population_profile": "same_origin_two_seed_mechanism_only_v1",
                "same_origin_profile_verified": True,
                "shared_origin_identities": 1,
                "formal_authority": False,
                "mechanism_only": True,
                "generalization": False,
                "identity_generalization": False,
                "event_family_generalization": False,
                "optimizer_authorized": True,
                "synthetic_target_bytes_read": False,
            }
        )
        canary_amplitude_validation = validation(
            {
                "schema_version": "bernini-full30-action-mechanism-canary-amplitude-validation-v1",
                "manifest_file_sha256": canary_amplitude_sha,
                "parent_manifest_file_sha256": canary_authority_sha,
                "population_profile": "same_origin_two_seed_mechanism_only_v1",
                "formal_authority": False,
                "mechanism_only": True,
                "generalization": False,
                "identity_generalization": False,
                "event_family_generalization": False,
                "optimizer_authorized": True,
            }
        )
        canary_projections["amplitude_validation_digest"] = (
            canary_amplitude_validation["validation_digest"]
        )
        formal_data_validation = validation(
            {
                "schema_version": "bernini-full30-action-data-teacher-validation-v3",
                "source_counts": {"fit": 64, "confirmation": 16, "heldout": 8},
                "pair_counts": {"fit": 128, "confirmation": 32, "heldout": 16},
                "teacher_origin_counts": {"fit": 8, "confirmation": 8},
                "representation_counts": {"fit": 16, "confirmation": 16},
                "synthetic_target_index1_bytes_read": False,
                "optimizer_authorized": True,
            }
        )
        formal_amplitude_validation = validation(
            {
                "schema_version": "bernini-full30-action-amplitude-validation-v2",
                "manifest_file_sha256": formal_amplitude_sha,
                "parent_manifest_file_sha256": formal_authority_sha,
                "optimizer_bundles": 16,
                "calibrator_evidence": 32,
                "frozen_fail_evidence": 32,
                "sigma_floor_rows": 96,
                "optimizer_authorized": True,
            }
        )
        formal_projections["amplitude_validation_digest"] = (
            formal_amplitude_validation["validation_digest"]
        )

        def write_json(path, value):
            path.write_bytes(trainer.canonical_json_bytes(value) + b"\n")
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            records, previous = [], None
            for update in (0, 1, 2):
                checkpoint_path = parent / f"u{update:08d}"
                checkpoint_path.mkdir()
                manifest_unsigned = {
                    "schema_version": "checkpoint-v2",
                    "bindings": canary_bindings,
                    "capacity": {
                        "authoritative_inventory_sha256": inventory_sha,
                        "production_capacity_authorized": True,
                    },
                    "trainables": {"state_sha256": sha(str(update + 1))},
                    "optimizer": {
                        "state_sha256": sha(str(update + 4)),
                        "update_count": update,
                    },
                    "progress": {"completed_updates": update},
                    "schedule": {
                        "full_sha256": canary_schedule_sha,
                        "prefix_sha256": canary_prefixes[update],
                    },
                    "history": {"sha256": sha("8")},
                    "rng": {"sha256": sha("9")},
                    "checkpoint_sequence": update,
                    "previous_checkpoint": previous,
                }
                manifest = {
                    **manifest_unsigned,
                    "manifest_digest": trainer.object_sha256(manifest_unsigned),
                }
                manifest_sha = write_json(
                    checkpoint_path / "manifest.json", manifest
                )
                reference = {
                    "checkpoint_sequence": update,
                    "completed_updates": update,
                    "manifest_sha256": manifest_sha,
                    "manifest_digest": manifest["manifest_digest"],
                    "history_sha256": manifest["history"]["sha256"],
                    "rng_sha256": manifest["rng"]["sha256"],
                    "schedule_prefix_sha256": canary_prefixes[update],
                    "trainable_state_sha256": manifest["trainables"]["state_sha256"],
                    "optimizer_v_state_sha256": manifest["optimizer"]["state_sha256"],
                }
                reference_path = parent / f"u{update:08d}.json"
                write_json(reference_path, reference)
                records.append(
                    {
                        "completed_updates": update,
                        "path": str(checkpoint_path),
                        "reference_path": str(reference_path),
                        "reference": reference,
                    }
                )
                previous = reference

            receipt_unsigned = {
                "schema_version": trainer.RECEIPT_SCHEMA_VERSION,
                "method": trainer.METHOD,
                "complete": True,
                "arm": "action+retain",
                "profile": "disposable-canary-2",
                "start_update": 0,
                "stop_update": 2,
                "optimizer_updates_executed": 2,
                "optimizer_update_count": 2,
                "boundary_plan": dict(canary_plan.receipt()),
                "segment_gate": None,
                "fresh_formal_canary_evidence": None,
                "bindings": canary_bindings,
                "release": {
                    "release_sha256": release_sha,
                    "files": [
                        {"path": "full30_action_optimizer_v1.py", "sha256": optimizer_code_sha},
                        {"path": "full30_action_training_step_v1.py", "sha256": training_step_code_sha},
                    ],
                },
                "authority_validation": canary_data_validation,
                "amplitude_authority_validation": canary_amplitude_validation,
                "amplitude_runtime_binding": {"canary_only": True},
                "data_teacher_authority_manifest_sha256": canary_authority_sha,
                "amplitude_authority_manifest_sha256": canary_amplitude_sha,
                "authority_projections": canary_projections,
                "schedule_full_sha256": canary_schedule_sha,
                "schedule_prefix_sha256": canary_prefixes[2],
                "noise_authority_sha256": canary_noise_sha,
                "architecture": architecture,
                "lora_installation": {},
                "trainable_parameter_count": 188_946_432,
                "authoritative_inventory_sha256": inventory_sha,
                "final_optimizer_identity": {"update_count": 2, "inventory_sha256": inventory_sha},
                "checkpoints": records,
                "official_model": {"model_sha256": model_sha},
                "objective": {},
                "distributed": {},
                "synthetic_target_bytes_read": False,
                "synthetic_target_index1_bytes_read": False,
                "parent_allocation_released": False,
            }
            receipt = dict(trainer._seal(receipt_unsigned))
            receipt_path = parent / "receipt.json"
            receipt_sha = write_json(receipt_path, receipt)
            u2_reference_path = Path(records[2]["reference_path"])
            u2_reference_sha = hashlib.sha256(u2_reference_path.read_bytes()).hexdigest()
            canary_evidence = {
                "profile": canary_plan.profile,
                "population_profile": "same_origin_two_seed_mechanism_only_v1",
                "boundary_plan_sha256": canary_plan.receipt()["boundary_plan_sha256"],
                "authority_manifest_sha256": canary_authority_sha,
                "amplitude_authority_manifest_sha256": canary_amplitude_sha,
                "authority_projections": canary_projections,
                "data_validation_digest": canary_data_validation["validation_digest"],
                "amplitude_validation_digest": canary_amplitude_validation["validation_digest"],
                "noise_authority_sha256": canary_noise_sha,
                "schedule_full_sha256": canary_schedule_sha,
                "schedule_prefix_sha256_by_update": {str(k): v for k, v in canary_prefixes.items()},
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "receipt_digest": receipt["receipt_digest"],
                "u2_checkpoint_path": records[2]["path"],
                "u2_reference_path": str(u2_reference_path),
                "u2_reference_sha256": u2_reference_sha,
                "u2_reference_digest": trainer.object_sha256(records[2]["reference"]),
                "completed_updates": 2,
                "review_outcome": "GO",
            }
            formal_evidence = {
                "profile": formal_plan.profile,
                "start_update": 0,
                "stop_update": 20,
                "boundary_plan_sha256": formal_plan.receipt()["boundary_plan_sha256"],
                "authority_manifest_sha256": formal_authority_sha,
                "amplitude_authority_manifest_sha256": formal_amplitude_sha,
                "authority_projections": formal_projections,
                "data_validation_digest": formal_data_validation["validation_digest"],
                "amplitude_validation_digest": formal_amplitude_validation["validation_digest"],
                "noise_authority_sha256": formal_noise_sha,
                "schedule_full_sha256": formal_schedule_sha,
                "schedule_prefix_u0_sha256": formal_prefix_u0,
                "source_population": {"fit": 64, "confirmation": 16, "heldout": 8},
                "optimizer_pair_population": 128,
                "optimizer_teacher_population": 16,
                "optimizer_amplitude_population": 16,
                "formal_authority": True,
            }
            shared = {
                "arm": "action+retain",
                "release_sha256": release_sha,
                "model_sha256": model_sha,
                "runtime_code_sha256": runtime_sha,
                "objective_code_sha256": objective_sha,
                "optimizer_code_sha256": optimizer_code_sha,
                "training_step_code_sha256": training_step_code_sha,
                "architecture_digest": architecture["digest"],
                "authoritative_inventory_sha256": inventory_sha,
                "trainable_parameter_count": 188_946_432,
            }
            fresh_start = {
                "formal_start_update": 0,
                "formal_requires_resume": False,
                "canary_checkpoint_manifest_only_reopened": True,
                "canary_checkpoint_loaded": False,
                "canary_trainable_state_loaded": False,
                "canary_optimizer_state_loaded": False,
                "canary_rng_state_loaded": False,
                "canary_schedule_used_for_formal": False,
            }
            gate_unsigned = {
                "schema_version": trainer.REDUCED_CANARY_TO_FRESH_FORMAL_GATE_SCHEMA_VERSION,
                "status": "GO",
                "transition": "reduced_canary_to_fresh_formal_v1",
                "arm": "action+retain",
                "canary": canary_evidence,
                "formal": formal_evidence,
                "shared_execution_identity": shared,
                "fresh_start": fresh_start,
            }
            gate = {**gate_unsigned, "gate_digest": trainer.object_sha256(gate_unsigned)}
            gate_path = parent / "gate.json"
            gate_sha = write_json(gate_path, gate)
            inputs = trainer.FreshFormalCanaryInputsV1(
                gate_path, gate_sha, receipt_path, receipt_sha,
                u2_reference_path, u2_reference_sha,
            )
            kwargs = {
                "inputs": inputs,
                "arm": "action+retain",
                "formal_plan": formal_plan,
                "release_sha256": release_sha,
                "model_sha256": model_sha,
                "authority_manifest_sha256": formal_authority_sha,
                "amplitude_authority_manifest_sha256": formal_amplitude_sha,
                "authority_projections": formal_projections,
                "noise_authority_sha256": formal_noise_sha,
                "schedule_full_sha256": formal_schedule_sha,
                "schedule_prefix_u0_sha256": formal_prefix_u0,
                "architecture": architecture,
                "authoritative_inventory_sha256": inventory_sha,
                "trainable_parameter_count": 188_946_432,
                "bindings": formal_bindings,
                "formal_authority_validation": formal_data_validation,
                "formal_amplitude_validation": formal_amplitude_validation,
                "optimizer_code_sha256": optimizer_code_sha,
                "training_step_code_sha256": training_step_code_sha,
            }
            evidence = trainer.validate_reduced_canary_to_fresh_formal_v1(**kwargs)
            self.assertNotEqual(
                evidence["canary"]["authority_manifest_sha256"],
                evidence["formal"]["authority_manifest_sha256"],
            )
            self.assertFalse(evidence["fresh_start"]["canary_checkpoint_loaded"])
            self.assertFalse(evidence["fresh_start"]["canary_optimizer_state_loaded"])
            self.assertTrue(evidence["canary_and_formal_authorities_independently_validated"])
            self.assertTrue(evidence["canary_state_is_not_formal_initial_state"])

            hostile_unsigned = dict(gate_unsigned)
            hostile_unsigned["fresh_start"] = {
                **fresh_start,
                "canary_checkpoint_loaded": True,
            }
            hostile = {**hostile_unsigned, "gate_digest": trainer.object_sha256(hostile_unsigned)}
            hostile_path = parent / "hostile-gate.json"
            hostile_sha = write_json(hostile_path, hostile)
            hostile_inputs = trainer.FreshFormalCanaryInputsV1(
                hostile_path, hostile_sha, receipt_path, receipt_sha,
                u2_reference_path, u2_reference_sha,
            )
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError, "gate binding/digest"
            ):
                trainer.validate_reduced_canary_to_fresh_formal_v1(
                    **{**kwargs, "inputs": hostile_inputs}
                )

            hostile_formal = validation(
                {**{k: v for k, v in formal_data_validation.items() if k != "validation_digest"},
                 "source_counts": {"fit": 8}}
            )
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError, "64/16/8 population"
            ):
                trainer.validate_reduced_canary_to_fresh_formal_v1(
                    **{**kwargs, "formal_authority_validation": hostile_formal}
                )

    def test_production_cli_requires_complete_transition_and_forbids_resume_at_u0(
        self,
    ) -> None:
        plan = trainer.build_boundary_plan_v1(
            "review-gated-segment", start_update=0, stop_update=20
        )
        parser_dests = {action.dest for action in trainer.parser()._actions}
        self.assertTrue(
            {
                "canary_gate",
                "expected_canary_gate_sha256",
                "canary_receipt",
                "expected_canary_receipt_sha256",
                "canary_u2_reference",
                "expected_canary_u2_reference_sha256",
            }
            <= parser_dests
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "fresh-output"
            args = SimpleNamespace(
                arm="action+retain",
                profile="review-gated-segment",
                start_update=0,
                stop_update=20,
                run_seed=20260815,
                expected_boundary_plan_sha256=plan.receipt()[
                    "boundary_plan_sha256"
                ],
                output=str(output),
                resume_checkpoint=None,
                resume_reference=None,
                expected_resume_reference_sha256=None,
                resume_previous_reference=None,
                expected_resume_previous_reference_sha256=None,
                canary_gate=None,
                expected_canary_gate_sha256=None,
                canary_receipt=None,
                expected_canary_receipt_sha256=None,
                canary_u2_reference=None,
                expected_canary_u2_reference_sha256=None,
                segment_review_gate=None,
                expected_segment_review_gate_sha256=None,
            )
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError,
                "requires sealed same-arm u2 canary evidence",
            ):
                trainer._validate_cli_bindings(
                    args, release_receipt={"release_sha256": sha("1")}
                )
            args.resume_checkpoint = str(Path(directory).resolve())
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError,
                "fresh execution may not consume a resume",
            ):
                trainer._validate_cli_bindings(
                    args, release_receipt={"release_sha256": sha("1")}
                )

    def test_review_gate_binds_arm_boundary_authority_model_release_and_resume(self):
        plan = trainer.build_boundary_plan_v1(
            "review-gated-segment", start_update=20, stop_update=40
        )
        unsigned = {
            "schema_version": trainer.SEGMENT_GATE_SCHEMA_VERSION,
            "status": "approved",
            "arm": "action+retain",
            "profile": plan.profile,
            "start_update": 20,
            "stop_update": 40,
            "boundary_plan_sha256": plan.receipt()["boundary_plan_sha256"],
            "authority_manifest_sha256": sha("a"),
            "amplitude_authority_manifest_sha256": sha("e"),
            "model_sha256": sha("b"),
            "release_sha256": sha("c"),
            "resume_reference_sha256": sha("d"),
            "fresh_formal_canary_gate_sha256": None,
            "full81_review_required_before_next_segment": True,
        }
        value = {**unsigned, "gate_digest": trainer.object_sha256(unsigned)}
        with tempfile.TemporaryDirectory() as directory:
            path = (Path(directory) / "gate.json").resolve()
            path.write_bytes(trainer.canonical_json_bytes(value) + b"\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = trainer.validate_segment_gate_v1(
                path=path,
                expected_sha256=digest,
                plan=plan,
                arm="action+retain",
                authority_manifest_sha256=sha("a"),
                amplitude_authority_manifest_sha256=sha("e"),
                model_sha256=sha("b"),
                release_sha256=sha("c"),
                resume_reference_sha256=sha("d"),
                fresh_formal_canary_gate_sha256=None,
            )
            self.assertEqual(result["status"], "approved")
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError, "binding/digest"
            ):
                trainer.validate_segment_gate_v1(
                    path=path,
                    expected_sha256=digest,
                    plan=plan,
                    arm="action-only",
                    authority_manifest_sha256=sha("a"),
                    amplitude_authority_manifest_sha256=sha("e"),
                    model_sha256=sha("b"),
                    release_sha256=sha("c"),
                    resume_reference_sha256=sha("d"),
                    fresh_formal_canary_gate_sha256=None,
                )


if __name__ == "__main__":
    unittest.main()
