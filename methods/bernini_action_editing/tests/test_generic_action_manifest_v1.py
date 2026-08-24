from __future__ import annotations

import hashlib
import json
import math
import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import generic_action_manifest_v1 as contract  # noqa: E402
import materialize_phi_v1_sidecars_sp4 as materializer  # noqa: E402
import reserve4_fixed_generation_sp4_v1 as reserve4_runner  # noqa: E402


AUTHORING = METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
POPULATION = METHOD_ROOT / "assets/mosaic_event_population_compact6_topup20_v1.json"
Q0 = METHOD_ROOT / "assets/action_source_q0_authority_first8_v1.json"
READINESS = METHOD_ROOT / "assets/generic_action_manifest_readiness_20260814_v1.json"
CORE4_SPEC = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")


def _text(value: str) -> dict:
    return {"text": value, "utf8_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}


PHASE_LABELS = ["onset", "transition", "terminal"] + ["hold"] * 18


class GenericActionManifestTests(unittest.TestCase):
    def test_pinned_authoring_population_and_readiness_are_honest(self) -> None:
        self.assertEqual(_sha(AUTHORING), contract.AUTHORING_SHA256)
        self.assertEqual(_sha(POPULATION), contract.POPULATION_SHA256)
        readiness = json.loads(READINESS.read_text())
        self.assertEqual(readiness["observed_phi_v1_block22_sidecars"], {"fit": 0, "confirmation": 0})
        self.assertEqual(readiness["observed_generated_media_for_selected_four_branches"]["fit"], 16)
        self.assertFalse(readiness["optimizer_authorized"])
        q0 = json.loads(Q0.read_text())
        self.assertEqual(len(q0["rows"]), 8)
        self.assertEqual(len({row["q0_source_video_sha256"] for row in q0["rows"]}), 8)

    def test_expected_row_order_is_shared_and_exact(self) -> None:
        authoring = json.loads(AUTHORING.read_text())
        population = json.loads(POPULATION.read_text())
        rows = contract._expected_rows(authoring, population)
        self.assertEqual(len(rows), 64)
        self.assertEqual(sum(row["analysis_split"] == "fit" for row in rows), 32)
        self.assertEqual(sum(row["analysis_split"] == "confirmation" for row in rows), 32)
        self.assertEqual(tuple(row["branch"] for row in rows[:4]), contract.BRANCH_ORDER)
        self.assertEqual(len({row["row_id"] for row in rows}), 64)
        self.assertNotIn("action_family_id", rows[0])

    def test_tensor_contract_rejects_fake_zero_and_nonunit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "code.f32le"
            path.write_bytes(b"\x00" * contract.RAW_CODE_BYTES)
            binding = {
                "path": str(path.resolve()),
                "raw_sha256": _sha(path),
                "dtype": "float32",
                "byte_order": "little",
                "shape": [21, 32],
                "normalization": "global_l2_unit",
            }
            with self.assertRaisesRegex(contract.GenericActionManifestError, "L2 norm"):
                contract.validate_code_tensor(binding, is_noop=False, label="fixture")
            binding["normalization"] = "exact_zero_not_normalized"
            contract.validate_code_tensor(binding, is_noop=True, label="fixture")
            path.write_bytes(struct.pack("<672f", *([0.0] * 671 + [1.0])))
            binding["raw_sha256"] = _sha(path)
            with self.assertRaisesRegex(contract.GenericActionManifestError, "noop bytes"):
                contract.validate_code_tensor(binding, is_noop=True, label="fixture")

    def test_review_requires_blind_full81_and_branch_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "review.json"
            value = {
                "schema_version": contract.REVIEW_SCHEMA,
                "candidate_id": "candidate-action",
                "branch": "action",
                "media_sha256": "1" * 64,
                "review_method": "human_blind_video_review_v1",
                "entire_exact81_video_viewed": True,
                "frame_count": 81,
                "fps": 25,
                "reviewer_blinded_to_prompt_and_requested_branch": True,
                "sealed_before_phi_extraction": True,
                "quality_pass": True,
                "branch_semantics_pass": True,
                "phase_labels": PHASE_LABELS,
                "observations": {
                    "start_state_present": True,
                    "transition_present": True,
                    "requested_terminal_present": True,
                    "terminal_hold_present": True,
                    "full_target_event_present": True,
                },
            }
            value = {**value, "receipt_digest": contract.object_sha256(value)}
            _write_json(path, value)
            contract.validate_review_receipt(path, _sha(path))
            changed = dict(value)
            changed["entire_exact81_video_viewed"] = False
            unsigned = dict(changed)
            del unsigned["receipt_digest"]
            changed["receipt_digest"] = contract.object_sha256(unsigned)
            _write_json(path, changed)
            with self.assertRaisesRegex(contract.GenericActionManifestError, "entire_exact81"):
                contract.validate_review_receipt(path, _sha(path))

    def _fixture_evidence(self, root: Path) -> Path:
        authoring = json.loads(AUTHORING.read_text())
        population = json.loads(POPULATION.read_text())
        expected = contract._expected_rows(authoring, population)
        generator = root / "p32_generator.py"
        generator.write_text("# pinned fixture generator\n")
        p32 = root / "p32.f32le"
        values = [0.0] * (contract.HIDDEN_WIDTH * contract.CODE_WIDTH)
        for index in range(contract.CODE_WIDTH):
            values[index * contract.CODE_WIDTH + index] = 1.0
        p32.write_bytes(struct.pack(f"<{len(values)}f", *values))
        p32_sha = _sha(p32)
        rows = []
        for ordinal, registered in enumerate(expected):
            row_root = root / f"row-{ordinal:02d}"
            row_root.mkdir()
            tensor = row_root / "quotient.f32le"
            if registered["branch"] == "noop":
                code = [0.0] * 672
            else:
                code = [0.0] * 672
                code[32] = 1.0 / math.sqrt(2.0)
                code[64] = -1.0 / math.sqrt(2.0)
            tensor.write_bytes(struct.pack("<672f", *code))
            review = {
                "schema_version": contract.REVIEW_SCHEMA,
                "candidate_id": registered["candidate_id"],
                "branch": registered["branch"],
                "media_sha256": hashlib.sha256(registered["candidate_id"].encode()).hexdigest(),
                "review_method": "human_blind_video_review_v1",
                "entire_exact81_video_viewed": True,
                "frame_count": 81,
                "fps": 25,
                "reviewer_blinded_to_prompt_and_requested_branch": True,
                "sealed_before_phi_extraction": True,
                "quality_pass": True,
                "branch_semantics_pass": True,
                "phase_labels": PHASE_LABELS,
                "observations": {
                    "start_state_present": True,
                    "transition_present": registered["branch"] in {"action", "reverse", "incomplete"},
                    "requested_terminal_present": registered["branch"] in {"action", "reverse"},
                    "terminal_hold_present": registered["branch"] in {"action", "reverse"},
                    "full_target_event_present": registered["branch"] in {"action", "reverse"},
                },
            }
            review = {**review, "receipt_digest": contract.object_sha256(review)}
            review_path = row_root / "review.json"
            _write_json(review_path, review)
            sidecar = {
                "schema_version": contract.SIDECAR_SCHEMA,
                "row_id": registered["row_id"],
                "candidate_id": registered["candidate_id"],
                "source_iid": registered["source_iid"],
                "analysis_split": registered["analysis_split"],
                "seed": registered["seed"],
                "branch": registered["branch"],
                "phi_v1": {
                    "hook": "transformer_1.blocks[22].output",
                    "block_index": 22,
                    "teacher_exact40_index": 29,
                    "sp_world": 4,
                    "sp_order": "rank0_rank1_rank2_rank3_contiguous_global_target_indices",
                    "append_padding_removed": True,
                    "target_layout": "phase_major_21_then_patch_y_x",
                    "pooling": "fixed_spatial_mean",
                    "phase0": "exact_positive_zero",
                    "temporal_dc": "phases_1_20_per_channel_mean_subtracted",
                    "p32_seed": contract.P32_SEED,
                    "p32_shape": [1536, 32],
                    "p32_raw_path": str(p32.resolve()),
                    "p32_raw_sha256": p32_sha,
                    "p32_generator_path": str(generator.resolve()),
                    "p32_generator_source_sha256": _sha(generator),
                    "nuisance_order": ["camera_only", "appearance_only_gram_schmidt_off_camera"],
                },
                "tensor": {
                    "path": str(tensor.resolve()),
                    "raw_sha256": _sha(tensor),
                    "dtype": "float32",
                    "byte_order": "little",
                    "shape": [21, 32],
                    "normalization": "exact_zero_not_normalized" if registered["branch"] == "noop" else "global_l2_unit",
                },
                "nuisance_projection": {
                    "camera_raw_sha256": "2" * 64,
                    "appearance_raw_sha256": "3" * 64,
                    "camera_norm": 1.0,
                    "appearance_after_gs_norm": 1.0,
                    "pre_projection_norm": 0.0 if registered["branch"] == "noop" else 1.0,
                    "post_projection_norm": 0.0 if registered["branch"] == "noop" else 1.0,
                    "survival_cosine": 1.0,
                    "finite_non_degenerate": True,
                },
                "review_status": "PASS_SEALED_BEFORE_EXTRACTION",
                "generated_media_is_optimizer_input_or_target": False,
                "optimizer_authorized": False,
            }
            sidecar = {**sidecar, "receipt_digest": contract.object_sha256(sidecar)}
            sidecar_path = row_root / "sidecar.json"
            _write_json(sidecar_path, sidecar)
            instruction = _text(registered["instruction_text"])
            semantic = {
                "q0_state": _text(f"q0 {registered['source_iid']}"),
                "q1_state": _text(f"q1 {registered['source_iid']}"),
                "owner": _text(f"owner {registered['source_iid']}"),
                "object_contact": _text("no registered object contact"),
            }
            rows.append(
                {
                    "row_id": registered["row_id"],
                    "candidate_id": registered["candidate_id"],
                    "instruction": instruction,
                    "phase_labels": PHASE_LABELS,
                    "semantic_binding": semantic,
                    "sidecar_receipt": {"path": str(sidecar_path.resolve()), "file_sha256": _sha(sidecar_path)},
                    "review_receipt": {"path": str(review_path.resolve()), "file_sha256": _sha(review_path)},
                }
            )
        evidence = {"schema_version": contract.EVIDENCE_SCHEMA, "rows": rows}
        evidence = {**evidence, "index_digest": contract.object_sha256(evidence)}
        evidence_path = root / "evidence.json"
        _write_json(evidence_path, evidence)
        return evidence_path

    def test_full_builder_produces_shared_64_row_order_and_24_16_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence = self._fixture_evidence(root)
            representation = root / "representation_train_manifest_v1.json"
            pairs = root / "action_source_pair_manifest_v1.json"
            contract.build_manifests(
                authoring_path=AUTHORING,
                population_path=POPULATION,
                evidence_index_path=evidence,
                q0_authority_path=Q0,
                representation_output=representation,
                pair_output=pairs,
            )
            rep, pair = contract.validate_manifest_pair(representation, pairs)
            self.assertEqual(sum(row["planner_optimizer_eligible"] for row in rep["rows"]), 24)
            self.assertEqual(sum(row["operator_optimizer_eligible"] for row in pair["rows"]), 16)
            reverse = [row for row in pair["rows"] if row["branch"] == "reverse"]
            self.assertTrue(all(row["real_source_available"] is False for row in reverse))
            self.assertTrue(all(row["operator_optimizer_eligible"] is False for row in reverse))
            self.assertEqual([row["row_id"] for row in rep["rows"]], [row["row_id"] for row in pair["rows"]])

    def test_build_plan_writes_gap_and_refuses_empty_generation_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            media = root / "media"
            media.mkdir()
            output = root / "plan.json"
            gap = root / "gap.json"
            with self.assertRaisesRegex(materializer.PhiV1MaterializationError, "generation media closure"):
                materializer.build_plan(
                    authoring_path=AUTHORING,
                    population_path=POPULATION,
                    split="fit",
                    generation_roots=[media],
                    review_root=None,
                    output=output,
                    gap_output=gap,
                    allow_unreviewed_technical_only=True,
                )
            self.assertFalse(output.exists())
            value = json.loads(gap.read_text())
            self.assertEqual(len(value["missing_generation_candidate_ids"]), 80)
            self.assertFalse(value["optimizer_authorized"])


class Reserve4FixedGenerationTests(unittest.TestCase):
    def _reserve_specs(self, root: Path) -> tuple[Path, str, Path, str]:
        source = copy.deepcopy(json.loads(CORE4_SPEC.read_text(encoding="utf-8")))
        profile = reserve4_runner.seed2_builder.REPLICATION_PROFILES["reserve4-v1"]
        default = reserve4_runner.seed2_builder.REPLICATION_PROFILES["core4-v2"]
        seed_map = dict(zip(sorted(default.seed_map), sorted(profile.seed_map)))
        for group in source["groups"]:
            for candidate in group["candidates"]:
                old_seed = candidate["seed"]
                new_seed = seed_map[old_seed]
                candidate["seed"] = new_seed
                candidate["candidate_id"] = (
                    profile.source_id_prefix
                    + candidate["candidate_id"][len(default.source_id_prefix) :]
                )
                suffix = f"-s{old_seed}"
                self.assertTrue(candidate["calibration_group_id"].endswith(suffix))
                candidate["calibration_group_id"] = (
                    candidate["calibration_group_id"][: -len(suffix)]
                    + f"-s{new_seed}"
                )
        reserve4_runner.bank_contract.validate_root_spec(source)
        seed1 = root / "reserve4-seed1.json"
        seed1.write_bytes(
            reserve4_runner.bank_contract.canonical_json_bytes(source) + b"\n"
        )
        derived = reserve4_runner.seed2_builder.derive_seed2_spec(
            source, "reserve4-v1"
        )
        seed2 = root / "reserve4-seed2.json"
        seed2.write_bytes(
            reserve4_runner.bank_contract.canonical_json_bytes(derived) + b"\n"
        )
        return seed1, _sha(seed1), seed2, _sha(seed2)

    def test_fit_plan_is_two_seed_complete_ten_branch_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            seed1, sha1, seed2, sha2 = self._reserve_specs(root)
            output = root / "plan"
            with mock.patch.dict(
                reserve4_runner.SPEC_AUTHORITIES,
                {"seed1": sha1, "seed2": sha2},
                clear=True,
            ):
                plan = reserve4_runner.build_plan(
                    seed1_spec=seed1,
                    seed2_spec=seed2,
                    split="fit",
                    output_dir=output,
                )
                validated, _, _ = reserve4_runner.load_plan(
                    plan["_path"], plan["_file_sha256"]
                )
            self.assertEqual(validated["generation_invocation_count"], 40)
            self.assertEqual(validated["seed_cell_count"], 4)
            self.assertEqual(len(validated["cell_proofs"]), 4)
            self.assertTrue(
                all(
                    proof["branch_order"]
                    == list(reserve4_runner.bank_contract.MACE_BRANCH_ORDER)
                    for proof in validated["cell_proofs"]
                )
            )
            self.assertFalse(
                validated["execution_contract"][
                    "generated_media_is_editor_input_or_target"
                ]
            )
            gap = json.loads(
                (output / "reserve4-generation-gap-before-run-v1.json").read_text()
            )
            self.assertEqual(len(gap["missing_candidate_ids"]), 40)
            self.assertFalse(gap["phi_v1_extraction_authorized"])

    def test_seed2_non_seed_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            seed1, sha1, seed2, _ = self._reserve_specs(root)
            changed = json.loads(seed2.read_text(encoding="utf-8"))
            candidate = changed["groups"][0]["candidates"][0]
            candidate["full_t2v_caption"] += " The lighting remains stable."
            candidate["full_t2v_caption_utf8_sha256"] = hashlib.sha256(
                candidate["full_t2v_caption"].encode("utf-8")
            ).hexdigest()
            seed2.write_bytes(
                reserve4_runner.bank_contract.canonical_json_bytes(changed) + b"\n"
            )
            sha2 = _sha(seed2)
            with mock.patch.dict(
                reserve4_runner.SPEC_AUTHORITIES,
                {"seed1": sha1, "seed2": sha2},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    reserve4_runner.Reserve4GenerationError, "seed-only derivation"
                ):
                    reserve4_runner.build_plan(
                        seed1_spec=seed1,
                        seed2_spec=seed2,
                        split="fit",
                        output_dir=root / "plan",
                    )

    def test_empty_generation_audit_writes_exact_gap_and_refuses_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            seed1, sha1, seed2, sha2 = self._reserve_specs(root)
            plan_root = root / "plan"
            media = root / "media"
            media.mkdir()
            with mock.patch.dict(
                reserve4_runner.SPEC_AUTHORITIES,
                {"seed1": sha1, "seed2": sha2},
                clear=True,
            ):
                plan = reserve4_runner.build_plan(
                    seed1_spec=seed1,
                    seed2_spec=seed2,
                    split="fit",
                    output_dir=plan_root,
                )
                completion = root / "completion.json"
                gap = root / "gap.json"
                with self.assertRaisesRegex(
                    reserve4_runner.Reserve4GenerationError,
                    "generation closure is incomplete",
                ):
                    reserve4_runner.audit_plan(
                        plan_path=plan["_path"],
                        expected_plan_sha256=plan["_file_sha256"],
                        generation_roots=[media],
                        output=completion,
                        gap_output=gap,
                    )
            self.assertFalse(completion.exists())
            value = json.loads(gap.read_text())
            self.assertEqual(value["observed_candidate_count"], 0)
            self.assertEqual(len(value["missing_candidate_ids"]), 40)
            self.assertFalse(value["optimizer_authorized"])


if __name__ == "__main__":
    unittest.main()
