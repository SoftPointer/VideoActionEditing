from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_t2v_factorial_bank as bank  # noqa: E402


ASSET_ROOT = METHOD_ROOT / "assets"
MICRO_ASSET = ASSET_ROOT / "dmiq_cdf_dog_t2v_micro_spec_v2.json"
SCIENTIFIC_ASSET = ASSET_ROOT / "dmiq_cdf_dog_t2v_scientific_spec_v2.json"
TEST_METHOD_REVISION = "2" * 40
TEST_METHOD_ARCHIVE_SHA256 = "3" * 64


def _build_manifest(spec: dict, *, attempt_rung: int = 0) -> dict:
    return bank.build_manifest(
        spec,
        method_source_revision=TEST_METHOD_REVISION,
        method_source_archive_sha256=TEST_METHOD_ARCHIVE_SHA256,
        attempt_rung=attempt_rung,
    )


def _sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(value: dict, field: str) -> dict:
    value.pop(field, None)
    value[field] = bank.object_sha256(value)
    return value


def _write_fake_initial_gaussian(path: Path, *, first_byte: int = 0) -> str:
    byte_count = 4
    for dimension in bank.LATENT_SHAPE:
        byte_count *= dimension
    raw = bytes([first_byte]) + bytes(byte_count - 1)
    header = {
        "__metadata__": {
            "coordinate": "bernini_native_target_latent_before_rearrange",
            "source": "observed_return_of_official_module_global_randn_tensor",
            "observer_only": "true",
            "external_initial_noise_injection": "false",
        },
        "official_initial_gaussian": {
            "dtype": "F32",
            "shape": list(bank.LATENT_SHAPE),
            "data_offsets": [0, byte_count],
        },
    }
    header_bytes = bank.canonical_json_bytes(header)
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + raw)
    return _sha(raw)


def _minimal_bank_receipt(manifest: dict) -> dict:
    receipt = {
        "schema_version": bank.BANK_RECEIPT_SCHEMA,
        "bank_id": manifest["bank_id"],
        "profile": manifest["profile"],
        "attempt_rung": manifest["attempt_rung"],
        "manifest_digest": manifest["manifest_digest"],
        "entry_count": len(manifest["entries"]),
        "proposal_cell_count": len(manifest["registered_design_cells"]),
        "entries": [
            {
                "entry_id": entry["entry_id"],
                "video_sha256": _sha(f"video:{entry['entry_id']}"),
            }
            for entry in manifest["entries"]
        ],
        "condition_closure": {
            "all_native_entry_audits_pass": True,
            "all_cells_share_exact_initial_noise_across_ten_branches": True,
        },
        "interpretation": {
            "factorial_render_complete": True,
            "fitq_confirmation_eligible": False,
        },
    }
    return _seal(receipt, "receipt_digest")


def _passing_event_row(entry: dict, video_sha256: str) -> dict:
    branch = entry["semantic_branch"]
    row = {
        "entry_id": entry["entry_id"],
        "video_sha256": video_sha256,
        "semantic_branch": branch,
        "analysis_split": entry["analysis_split"],
        "initial_preconditions_realized": [True, True],
        "ordered_milestone_realized": [False, False, False, False],
        "ordered_milestone_first_frame": [None, None, None, None],
        "terminal_hold_start_frame": None,
        "terminal_hold_through_final_frame": False,
        "correct_actor_performs_target": False,
        "distractor_actor_performs_target": False,
        "correct_object_is_target": False,
        "distractor_object_is_target": False,
        "registered_branch_realized": True,
        "gross_motion_energy_class": "not-applicable",
        "atomic_event_count": 0,
        "assessor_confidence": "high",
        "reverse_opposite_state_transition_realized": False,
        "object_remained_ground_supported_throughout": True,
        "acting_dog_empty_mouthed_at_final_frame": True,
    }
    if branch == "full_action":
        row.update(
            ordered_milestone_realized=[True] * 4,
            ordered_milestone_first_frame=[8, 24, 40, 56],
            terminal_hold_start_frame=64,
            terminal_hold_through_final_frame=True,
            correct_actor_performs_target=True,
            correct_object_is_target=True,
            object_remained_ground_supported_throughout=False,
            acting_dog_empty_mouthed_at_final_frame=False,
            atomic_event_count=4,
        )
    elif branch == "incomplete_action":
        row.update(
            ordered_milestone_realized=[True, True, False, False],
            ordered_milestone_first_frame=[10, 28, None, None],
            correct_actor_performs_target=True,
            correct_object_is_target=True,
            atomic_event_count=2,
        )
    elif branch == "reverse_action":
        row.update(
            ordered_milestone_realized=[True, True, False, False],
            ordered_milestone_first_frame=[12, 28, None, None],
            correct_actor_performs_target=True,
            correct_object_is_target=True,
            reverse_opposite_state_transition_realized=True,
            gross_motion_energy_class="matched-to-full-action",
            atomic_event_count=4,
        )
    elif branch == "shuffled_action":
        row.update(
            ordered_milestone_realized=[True, True, True, False],
            ordered_milestone_first_frame=[8, 40, 24, None],
            correct_actor_performs_target=True,
            correct_object_is_target=True,
            atomic_event_count=3,
        )
    elif branch in {"camera_only", "appearance_only"}:
        row["atomic_event_count"] = 1
    elif branch == "generic_wrong_motion":
        row.update(
            gross_motion_energy_class="matched-to-full-action",
            atomic_event_count=4,
        )
    elif branch == "wrong_actor":
        row.update(
            ordered_milestone_realized=[True] * 4,
            ordered_milestone_first_frame=[8, 24, 40, 56],
            terminal_hold_start_frame=64,
            terminal_hold_through_final_frame=True,
            distractor_actor_performs_target=True,
            correct_object_is_target=True,
            object_remained_ground_supported_throughout=False,
            acting_dog_empty_mouthed_at_final_frame=False,
            atomic_event_count=4,
        )
    elif branch == "wrong_object":
        row.update(
            ordered_milestone_realized=[True] * 4,
            ordered_milestone_first_frame=[8, 24, 40, 56],
            terminal_hold_start_frame=64,
            terminal_hold_through_final_frame=True,
            correct_actor_performs_target=True,
            distractor_object_is_target=True,
            object_remained_ground_supported_throughout=False,
            acting_dog_empty_mouthed_at_final_frame=False,
            atomic_event_count=4,
        )
    return row


def _event_audit(manifest: dict, receipt: dict) -> dict:
    videos = {row["entry_id"]: row["video_sha256"] for row in receipt["entries"]}
    rows = [_passing_event_row(entry, videos[entry["entry_id"]]) for entry in manifest["entries"]]
    discovery = [row for row in rows if row["analysis_split"] == "discovery"]
    audit = {
        "schema_version": bank.EVENT_AUDIT_SCHEMA,
        "bank_manifest_digest": manifest["manifest_digest"],
        "bank_receipt_digest": receipt["receipt_digest"],
        "assessor_contract": {
            "assessor_id": "blind-assessor-01",
            "organization_id": "independent-audit-lab",
            "independent_of_renderer_and_method": True,
            "no_fitq_or_training_outputs_seen": True,
            "proposal_videos_are_only_model_outputs_seen": True,
            "blinded_fields": list(bank.BLINDED_FIELDS),
            "attestation": "I assessed only the preregistered visible events before receiving split or FITQ information.",
        },
        "split_isolation": {
            "discovery_decisions_frozen_before_confirmation_opened": True,
            "discovery_decisions_sha256": bank.object_sha256(discovery),
            "confirmation_opened_after_freeze": True,
            "confirmation_rows_never_used_for_prompt_seed_or_design_selection": True,
            "row_order_matches_manifest": True,
        },
        "rows": rows,
    }
    return _seal(audit, "audit_digest")


class DMIQPureT2VFactorialBankV2Tests(unittest.TestCase):
    def test_checked_in_specs_are_valid_and_scientific_has_32_by_10(self) -> None:
        micro = _build_manifest(_load(MICRO_ASSET))
        scientific = _build_manifest(_load(SCIENTIFIC_ASSET))
        self.assertEqual(len(micro["registered_design_cells"]), 2)
        self.assertEqual(len(micro["entries"]), 20)
        self.assertFalse(micro["factorial_contract"]["scientific_shape_and_rank_preregistered"])
        self.assertEqual(len(scientific["registered_design_cells"]), 32)
        self.assertEqual(len(scientific["entries"]), 320)
        self.assertEqual(scientific["factorial_contract"]["semantic_branches"], list(bank.BRANCH_ORDER))
        import internal_temporal_quotient as fitq

        self.assertEqual(
            tuple(scientific["factorial_contract"]["negative_branches"]),
            fitq.SCIENTIFIC_REQUIRED_NEGATIVE_LABELS,
        )
        self.assertEqual(scientific["factorial_contract"]["requested_discovery_full_action_count"], 24)
        self.assertEqual(scientific["factorial_contract"]["requested_confirmation_full_action_count"], 8)
        self.assertIsNone(scientific["factorial_contract"]["independently_verified_discovery_full_action_count"])
        self.assertIsNone(scientific["factorial_contract"]["independently_verified_confirmation_full_action_count"])
        self.assertFalse(scientific["factorial_contract"]["fitq_eligible_after_render"])
        self.assertEqual([row["entry_count"] for row in scientific["execution_topology"]["parallel_groups"]], [160, 160])
        self.assertFalse(
            scientific["execution_topology"]["launcher_readiness"][
                "scientific_scale_ready"
            ]
        )
        self.assertFalse(
            scientific["execution_topology"]["launcher_readiness"][
                "persistent_model_worker_implemented"
            ]
        )

    def test_l16_has_pairwise_balance_and_truthful_separate_discovery_rank(self) -> None:
        manifest = _build_manifest(_load(SCIENTIFIC_ASSET))
        rows = bank.scientific_oa_rows()
        factors = ("actor_index", "scene_index", "camera_index", "wording_index")
        for left_index, left in enumerate(factors):
            for right in factors[left_index + 1 :]:
                counts = Counter((row[left], row[right]) for row in rows)
                self.assertEqual(len(counts), 16)
                self.assertEqual(set(counts.values()), {1})
        full = manifest["design_diagnostics"]["full_design"]
        discovery = manifest["design_diagnostics"]["discovery_only"]
        self.assertEqual((full["rank"], full["columns"]), (13, 13))
        self.assertEqual((discovery["rank"], discovery["columns"]), (12, 12))
        self.assertTrue(full["full_column_rank"])
        self.assertTrue(discovery["full_column_rank"])
        self.assertTrue(manifest["design_diagnostics"]["confirmation_only"]["main_effect_rank_not_claimed"])

    def test_every_cell_is_group_closed_balanced_and_replicates_cross_over(self) -> None:
        manifest = _build_manifest(_load(SCIENTIFIC_ASSET))
        entries = manifest["entries"]
        for cell in manifest["registered_design_cells"]:
            rows = [row for row in entries if row["proposal_cell_id"] == cell["proposal_cell_id"]]
            self.assertEqual([row["semantic_branch"] for row in rows], list(bank.BRANCH_ORDER))
            self.assertEqual(len({row["execution_group"] for row in rows}), 1)
            self.assertEqual(len({row["seed"] for row in rows}), 1)
        for oa_index in range(16):
            cells = [row for row in manifest["registered_design_cells"] if row["oa_row_index"] == oa_index]
            self.assertEqual({row["seed_replicate_id"] for row in cells}, {"rep-a", "rep-b"})
            self.assertEqual({row["execution_group"] for row in cells}, set(bank.GROUPS))
        for group in bank.GROUPS:
            local = [row["group_local_order"] for row in entries if row["execution_group"] == group]
            self.assertEqual(local, list(range(160)))

    def test_prompts_predeclare_both_actors_objects_and_registered_contrasts(self) -> None:
        spec = _load(SCIENTIFIC_ASSET)
        manifest = _build_manifest(spec)
        cells = {row["proposal_cell_id"]: row for row in manifest["registered_design_cells"]}
        axes = manifest["axis_levels"]
        actors = {row["actor_id"]: row for row in axes["actor_levels"]}
        obj = axes["object_pair"]
        for entry in manifest["entries"]:
            actor = actors[cells[entry["proposal_cell_id"]]["actor_id"]]
            for phrase in (
                actor["actor_phrase"], actor["actor_reference"],
                actor["distractor_actor_phrase"], actor["distractor_actor_reference"],
                obj["object_phrase"], obj["object_reference"],
                obj["distractor_object_phrase"], obj["distractor_object_reference"],
            ):
                self.assertIn(phrase, entry["prompt"])
        branches = {row["semantic_branch"]: row["prompt"] for row in manifest["entries"][:10]}
        self.assertIn("that initial condition", branches["reverse_action"])
        self.assertIn("pushes it away along the ground", branches["reverse_action"])
        self.assertNotIn("rises before", branches["reverse_action"])
        self.assertIn("four clear events", branches["generic_wrong_motion"])
        self.assertIn("not the tan pit bull", branches["wrong_actor"])
        self.assertIn("blue rubber ball", branches["wrong_object"])
        for wording in spec["wording_levels"]:
            self.assertEqual(list(wording["templates"]), list(bank.BRANCH_ORDER))

    def test_scientific_patch_sketch_reconstructs_exact_signed_matrix(self) -> None:
        spec = _load(SCIENTIFIC_ASSET)
        sketch = spec["spatial_sketch"]
        raw = bank.reconstruct_spatial_sketch(sketch)
        self.assertEqual(len(raw), 16 * 930 * 4)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), sketch["matrix_raw_bytes_sha256"]
        )
        header = b"fitq-canonical-fp32-little-endian-v1|shape=16,930|"
        self.assertEqual(
            hashlib.sha256(header + raw).hexdigest(),
            sketch["matrix_value_sha256"],
        )
        self.assertEqual(bank.validate_spatial_sketch(sketch), sketch["matrix_value_sha256"])
        self.assertEqual(sketch["matrix_shape"], [16, 930])
        self.assertEqual((sketch["patch_height"], sketch["patch_width"]), (31, 30))
        self.assertEqual(
            sketch["sketch_id"],
            "dmiq-fitq-patch31x30-rademacher-s20260808017-v1",
        )
        self.assertEqual(bank.spatial_sketch_exact_row_rank(sketch), 16)
        self.assertNotIn("latent_height", sketch)
        mutated = bytearray(raw)
        mutated[101] ^= 1
        self.assertNotEqual(
            hashlib.sha256(mutated).hexdigest(), sketch["matrix_raw_bytes_sha256"]
        )
        corrupt = deepcopy(spec)
        corrupt["spatial_sketch"]["matrix_value_sha256"] = "0" * 64
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "matrix digest"):
            _build_manifest(corrupt)

    def test_seed_topup_is_whole_cohort_preregistered_and_deterministic(self) -> None:
        spec = _load(SCIENTIFIC_ASSET)
        rung0 = _build_manifest(spec, attempt_rung=0)
        rung1 = _build_manifest(spec, attempt_rung=1)
        self.assertEqual(rung0["topup_contract"]["next_attempt_rung"], 1)
        self.assertEqual(rung1["topup_contract"]["next_attempt_rung"], 2)
        self.assertTrue(rung0["topup_contract"]["individual_cell_or_winner_topup_forbidden"])
        self.assertTrue(rung0["topup_contract"]["previous_failed_and_successful_attempts_retained"])
        slots0 = [row["design_slot_id"] for row in rung0["registered_design_cells"]]
        slots1 = [row["design_slot_id"] for row in rung1["registered_design_cells"]]
        self.assertEqual(slots0, slots1[:32])
        self.assertEqual(slots0, slots1[32:])
        self.assertEqual(len(slots1), 64)
        self.assertEqual(
            rung0["registered_design_cells"],
            rung1["registered_design_cells"][:32],
        )
        self.assertEqual(rung0["entries"], rung1["entries"][:320])
        for cell in rung1["registered_design_cells"][:32]:
            self.assertEqual(cell["seed"], cell["seed_ladder"][0])
        for cell in rung1["registered_design_cells"][32:]:
            self.assertEqual(cell["seed"], cell["seed_ladder"][1])
        self.assertTrue(
            rung1["topup_contract"][
                "previous_attempt_entries_included_in_this_manifest"
            ]
        )
        self.assertFalse(
            rung1["topup_contract"][
                "cumulative_prior_artifact_reuse_worker_implemented"
            ]
        )
        self.assertFalse(
            rung1["topup_contract"][
                "attempt_rung_greater_than_zero_launch_authorized"
            ]
        )
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "outside"):
            _build_manifest(spec, attempt_rung=3)

    def test_scientific_spec_rejects_nonindependent_or_unregistered_design(self) -> None:
        bad = _load(SCIENTIFIC_ASSET)
        bad["actor_levels"][3]["split"] = "discovery"
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "actor levels"):
            _build_manifest(bad)
        bad = _load(SCIENTIFIC_ASSET)
        bad["spatial_sketch"]["status"] = "pending"
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "spatial sketch"):
            _build_manifest(bad)
        bad = _load(SCIENTIFIC_ASSET)
        bad["seed_replicates"] = bad["seed_replicates"][:1]
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "two seed replicates"):
            _build_manifest(bad)

    def test_manifest_is_deterministic_closed_and_tamper_evident(self) -> None:
        spec = _load(MICRO_ASSET)
        manifest = _build_manifest(spec)
        self.assertEqual(
            manifest["renderer_contract"]["method_source_revision"],
            TEST_METHOD_REVISION,
        )
        self.assertEqual(
            manifest["renderer_contract"]["method_source_archive_sha256"],
            TEST_METHOD_ARCHIVE_SHA256,
        )
        self.assertTrue(
            manifest["renderer_contract"][
                "method_source_preregistered_before_render"
            ]
        )
        self.assertEqual(bank.canonical_json_bytes(manifest), bank.canonical_json_bytes(_build_manifest(spec)))
        self.assertEqual(bank.validate_manifest(manifest), manifest)
        with self.assertRaisesRegex(
            bank.T2VFactorialBankError, "method source revision"
        ):
            bank.build_manifest(
                spec,
                method_source_revision="short",
                method_source_archive_sha256=TEST_METHOD_ARCHIVE_SHA256,
            )
        tampered = deepcopy(manifest)
        tampered["entries"][0]["seed"] += 1
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "digest differs"):
            bank.validate_manifest(tampered)
        tampered["manifest_digest"] = bank.object_sha256({key: value for key, value in tampered.items() if key != "manifest_digest"})
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "reconstruction"):
            bank.validate_manifest(tampered)

    def test_source_ast_has_no_training_and_group_assignment_is_not_branch_ordinal(self) -> None:
        source = inspect.getsource(bank)
        ast.parse(source)
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "loss.backward",
            "torch.optim",
            "GROUPS[ordinal",
            "GROUPS[len(entries)",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn(
            'group = GROUPS[(row["oa_row_index"] + replicate_index) % 2]',
            source,
        )
        finalizer_source = inspect.getsource(bank.finalize_bank)
        self.assertIn('"fitq_confirmation_eligible": False', finalizer_source)
        self.assertIn('"optimizer_update": "null"', finalizer_source)
        self.assertNotIn("fitq_confirmation_eligible\": True", finalizer_source)

    def test_duplicate_json_and_missing_negative_fail_closed(self) -> None:
        bad = _load(MICRO_ASSET)
        del bad["wording_levels"][0]["templates"]["wrong_object"]
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "fields differ"):
            _build_manifest(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "duplicate.json"
            path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
            with self.assertRaisesRegex(bank.T2VFactorialBankError, "duplicate JSON key"):
                bank.load_json_file(path, label="test")

    def test_render_entry_dispatches_only_native_t2v(self) -> None:
        source_bytes = b"exact81-source"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.mp4"
            source.write_bytes(source_bytes)
            spec = _load(MICRO_ASSET)
            spec["source_geometry_video"]["sha256"] = _sha(source_bytes)
            manifest_value = _build_manifest(spec)
            manifest = root / "manifest.json"
            bank.write_json_atomically(manifest, manifest_value)
            output = root / "output"
            output.mkdir()
            (output / "entries").mkdir()
            captured: list[list[str]] = []
            fake_native = types.ModuleType("infer_native_identity_generation_canary")
            fake_native.main = lambda argv: captured.append(list(argv)) or 0  # type: ignore[attr-defined]
            prior = sys.modules.get("infer_native_identity_generation_canary")
            sys.modules["infer_native_identity_generation_canary"] = fake_native
            try:
                result = bank.render_entry(
                    manifest_path=str(manifest), manifest_file_sha256=bank.file_sha256(manifest),
                    entry_id=manifest_value["entries"][0]["entry_id"], output_root=str(output),
                    bernini_root="/model/bernini", veomni_root="/model/veomni",
                    checkpoint="/model/checkpoint", checkpoint_content_manifest="/model/checkpoint.sha256",
                    source_video=str(source), method_source_revision="2" * 40,
                    method_source_archive_sha256="3" * 64,
                )
            finally:
                if prior is None:
                    sys.modules.pop("infer_native_identity_generation_canary", None)
                else:
                    sys.modules["infer_native_identity_generation_canary"] = prior
            self.assertEqual(result, 0)
            argv = captured[0]
            self.assertEqual(argv[argv.index("--arms") + 1], "t2v")
            self.assertEqual(argv[argv.index("--num-inference-steps") + 1], "40")
            for forbidden in ("--target-video", "--mask", "--flow", "--pose", "--track", "--trajectory", "--reference-image", "--reference-video", "--initial-latent", "--initial-noise"):
                self.assertNotIn(forbidden, argv)
            with self.assertRaisesRegex(
                bank.T2VFactorialBankError,
                "runtime method source differs",
            ):
                bank.render_entry(
                    manifest_path=str(manifest),
                    manifest_file_sha256=bank.file_sha256(manifest),
                    entry_id=manifest_value["entries"][0]["entry_id"],
                    output_root=str(output),
                    bernini_root="/model/bernini",
                    veomni_root="/model/veomni",
                    checkpoint="/model/checkpoint",
                    checkpoint_content_manifest="/model/checkpoint.sha256",
                    source_video=str(source),
                    method_source_revision="4" * 40,
                    method_source_archive_sha256=TEST_METHOD_ARCHIVE_SHA256,
                )

    def _write_fake_native_bank(
        self,
        root: Path,
        manifest: dict,
        *,
        bad_latent_shape: bool = False,
        missing_noise: bool = False,
        break_cell_noise: bool = False,
        spoof_noise_raw_digest: bool = False,
    ) -> None:
        renderer = manifest["renderer_contract"]
        canonical_noise: Path | None = None
        canonical_noise_raw_sha: str | None = None
        canonical_noise_file_sha: str | None = None
        for index, entry in enumerate(manifest["entries"]):
            entry_root = root / entry["output_subdir"]
            entry_root.mkdir(parents=True)
            video = entry_root / "t2v.mp4"
            latent = entry_root / "t2v.normalized-clean-latent.safetensors"
            noise = entry_root / "t2v.official-initial-gaussian.safetensors"
            video.write_bytes(f"video-{index}".encode())
            latent.write_bytes(f"latent-{index}".encode())
            if break_cell_noise and index == 1:
                noise_raw_sha = _write_fake_initial_gaussian(
                    noise, first_byte=1
                )
                noise_file_sha = _sha(noise.read_bytes())
            elif canonical_noise is None:
                noise_raw_sha = _write_fake_initial_gaussian(noise)
                canonical_noise = noise
                canonical_noise_raw_sha = noise_raw_sha
                canonical_noise_file_sha = _sha(noise.read_bytes())
                noise_file_sha = canonical_noise_file_sha
            else:
                os.link(canonical_noise, noise)
                assert canonical_noise_raw_sha is not None
                assert canonical_noise_file_sha is not None
                noise_raw_sha = canonical_noise_raw_sha
                noise_file_sha = canonical_noise_file_sha
            sampling = {
                "num_frames": 81,
                "num_inference_steps": 40,
                "guidance_mode": "t2v_apg",
                "omega_vid": 1.25,
                "omega_img": 4.5,
                "omega_txt": 4.0,
                "omega_scale": 0.8,
                "flow_shift": 5.0,
                "seed": entry["seed"],
                "eta": 0.5,
                "norm_threshold": [50.0, 50.0],
                "momentum": 0.0,
                "target_initialization": "official_gen_wanx22_fresh_gaussian",
                "target_mixed_with_source_latent": False,
                "custom_sampler_or_scheduler": False,
                "single_expert": "transformer_1",
                "ulysses_size": 4,
            }
            receipt = {
                "schema_version": bank.NATIVE_RECEIPT_SCHEMA,
                "method_source_revision": TEST_METHOD_REVISION,
                "method_source_archive_sha256": TEST_METHOD_ARCHIVE_SHA256,
                "arms": ["t2v"],
                "input": {
                    "source_video_sha256": manifest["source_geometry_video"]["sha256"],
                    "action_prompt_utf8_sha256": entry["prompt_utf8_sha256"],
                    "accepted_external_conditions": ["source_video", "action_prompt"],
                    "target_video": False,
                    "external_reference_image_or_video": False,
                    "external_mask_flow_pose_track_trajectory": False,
                    "external_first_frame_anchor": False,
                },
                "sampling": {"t2v": sampling},
                "conditioning": {"t2v": {"full_source_video_count": 0, "source_derived_reference_count": 0, "source_frame_indices": [], "source_ids": {"target_source_id": 0, "conditioning_source_count": 0, "video_source_ids": [], "reference_source_ids": []}}},
                "latent_geometry": {
                    "video_latent_shape": list(bank.LATENT_SHAPE),
                    "target_patch_tokens": 19_530,
                    "one_reference_patch_tokens": 930,
                },
                "source_condition_artifact": None,
                "bernini_commit": bank.BERNINI_COMMIT,
                "veomni_commit": bank.VEOMNI_COMMIT,
                "checkpoint": {"tree_sha256": bank.CHECKPOINT_TREE_SHA256},
                "freeze_certificate": {"base_frozen": True},
                "interpretation": {"training_performed": False, "best_arm_selected": False},
                "initial_noise_artifacts": {} if missing_noise and index == 0 else {"t2v": {
                    "path": str(noise),
                    "sha256": noise_file_sha,
                    "tensor_key": "official_initial_gaussian",
                    "tensor_value_sha256": (
                        "f" * 64
                        if spoof_noise_raw_digest and index == 0
                        else noise_raw_sha
                    ),
                    "raw_value_sha256": (
                        "f" * 64
                        if spoof_noise_raw_digest and index == 0
                        else noise_raw_sha
                    ),
                    "shape": list(bank.LATENT_SHAPE),
                    "stored_dtype": "torch.float32",
                    "numel": int(bank.LATENT_SHAPE[0] * bank.LATENT_SHAPE[1] * bank.LATENT_SHAPE[2] * bank.LATENT_SHAPE[3] * bank.LATENT_SHAPE[4]),
                    "byte_count": int(4 * bank.LATENT_SHAPE[0] * bank.LATENT_SHAPE[1] * bank.LATENT_SHAPE[2] * bank.LATENT_SHAPE[3] * bank.LATENT_SHAPE[4]),
                    "official_randn_tensor_call_count": 1,
                    "captured_from_native_sampler": True,
                    "observer_changed_return_value": False,
                    "source_or_target_derived": False,
                }},
                "outputs": {"t2v": {
                    "path": str(video), "sha256": _sha(video.read_bytes()),
                    "frame_count": 81, "fps": 25, "height": 496, "width": 480,
                    "normalized_clean_latent": {
                        "path": str(latent), "sha256": _sha(latent.read_bytes()),
                        "shape": [1, 16, 21, 2, 2] if bad_latent_shape and index == 0 else list(bank.LATENT_SHAPE),
                        "stored_dtype": "torch.float32", "native_sampler_before_vae_decode": True,
                        "source_video_vae_encode_before_any_decode": False, "mp4_decode_reencode_used": False,
                    },
                }},
            }
            _seal(receipt, "receipt_digest")
            (entry_root / "receipt.json").write_bytes(bank.canonical_json_bytes(receipt) + b"\n")

    def _finalize_micro(self, root: Path, **kwargs: bool) -> tuple[dict, dict]:
        manifest_value = _build_manifest(_load(MICRO_ASSET))
        manifest = root / "manifest.json"
        bank.write_json_atomically(manifest, manifest_value)
        output = root / "run"
        output.mkdir()
        self._write_fake_native_bank(output, manifest_value, **kwargs)
        receipt = bank.finalize_bank(
            manifest_path=str(manifest), manifest_file_sha256=bank.file_sha256(manifest),
            output_root=str(output), output_receipt=str(output / "bank.receipt.json"),
        )
        return manifest_value, receipt

    def test_finalizer_requires_exact_latent_and_same_initial_noise_within_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, receipt = self._finalize_micro(Path(directory).resolve())
            self.assertEqual(receipt["entry_count"], 20)
            self.assertTrue(receipt["condition_closure"]["all_cells_share_exact_initial_noise_across_ten_branches"])
            self.assertEqual(receipt["requested_counts"], {"discovery_full_action": 2, "confirmation_full_action": 0})
            self.assertIsNone(receipt["independently_event_verified_counts"]["discovery_full_action"])
            self.assertFalse(receipt["interpretation"]["fitq_confirmation_eligible"])
            self.assertEqual(receipt["interpretation"]["optimizer_update"], "null")
        for kwargs, error in (
            ({"bad_latent_shape": True}, "clean latent contract"),
            ({"missing_noise": True}, "initial_noise_artifacts"),
            ({"break_cell_noise": True}, "byte-identical initial Gaussian"),
            ({"spoof_noise_raw_digest": True}, "provenance differs"),
        ):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(bank.T2VFactorialBankError, error):
                    self._finalize_micro(Path(directory).resolve(), **kwargs)

    def test_scientific_renderer_completion_still_cannot_self_authorize_fitq(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest_value = _build_manifest(_load(SCIENTIFIC_ASSET))
            manifest = root / "manifest.json"
            bank.write_json_atomically(manifest, manifest_value)
            output = root / "run"
            output.mkdir()
            self._write_fake_native_bank(output, manifest_value)
            receipt = bank.finalize_bank(
                manifest_path=str(manifest),
                manifest_file_sha256=bank.file_sha256(manifest),
                output_root=str(output),
                output_receipt=str(output / "bank.receipt.json"),
            )
            self.assertEqual(receipt["entry_count"], 320)
            self.assertFalse(receipt["interpretation"]["fitq_confirmation_eligible"])
            self.assertFalse(receipt["interpretation"]["optimizer_update_authorized"])
            self.assertEqual(receipt["interpretation"]["optimizer_update"], "null")
            self.assertEqual(
                receipt["interpretation"]["fitq_eligibility_status"],
                "pending_independent_split_isolated_event_audit",
            )

    def test_legacy_event_audit_counts_but_never_authorizes_fitq(self) -> None:
        manifest = _build_manifest(_load(SCIENTIFIC_ASSET))
        receipt = _minimal_bank_receipt(manifest)
        audit = _event_audit(manifest, receipt)
        result = bank.validate_event_audit(manifest, receipt, audit)
        self.assertEqual(result["requested_counts"], {"discovery_full_action": 24, "confirmation_full_action": 8})
        self.assertEqual(result["independently_event_verified_counts"], {
            "discovery_full_action": 24,
            "confirmation_full_action": 8,
            "fully_realized_discovery_cells": 24,
            "fully_realized_confirmation_cells": 8,
        })
        self.assertTrue(result["event_count_minimum_met"])
        self.assertFalse(result["fitq_bank_eligible"])
        self.assertTrue(result["legacy_plain_json_audit_only"])
        self.assertFalse(
            result["external_discovery_assessor_signature_verified"]
        )
        self.assertFalse(
            result["same_state_owner_by_prompt_cross_query_verified"]
        )
        self.assertFalse(result["optimizer_update_authorized"])
        self.assertEqual(result["optimizer_update"], "null")
        self.assertIsNone(result["next_preregistered_whole_cohort_topup_rung"])
        self.assertEqual(
            result["status"],
            "ineligible-unsealed-audit-and-same-state-cross-query-missing",
        )

    def test_legacy_failed_audit_cannot_trigger_confirmation_topup(self) -> None:
        manifest = _build_manifest(_load(SCIENTIFIC_ASSET))
        receipt = _minimal_bank_receipt(manifest)
        audit = _event_audit(manifest, receipt)
        discovery_full = [row for row in audit["rows"] if row["analysis_split"] == "discovery" and row["semantic_branch"] == "full_action"]
        for row in discovery_full[:17]:
            row["registered_branch_realized"] = False
        audit["split_isolation"]["discovery_decisions_sha256"] = bank.object_sha256([row for row in audit["rows"] if row["analysis_split"] == "discovery"])
        _seal(audit, "audit_digest")
        result = bank.validate_event_audit(manifest, receipt, audit)
        self.assertEqual(result["independently_event_verified_counts"]["discovery_full_action"], 7)
        self.assertFalse(result["event_count_minimum_met"])
        self.assertFalse(result["fitq_bank_eligible"])
        self.assertIsNone(result["next_preregistered_whole_cohort_topup_rung"])
        self.assertFalse(result["individual_winner_selection_authorized"])
        self.assertTrue(result["all_failed_attempts_retained_required"])

    def test_event_audit_fails_on_split_leak_or_wrong_actor_semantic_mismatch(self) -> None:
        manifest = _build_manifest(_load(SCIENTIFIC_ASSET))
        receipt = _minimal_bank_receipt(manifest)
        audit = _event_audit(manifest, receipt)
        leaked = deepcopy(audit)
        leaked["split_isolation"]["confirmation_opened_after_freeze"] = False
        _seal(leaked, "audit_digest")
        with self.assertRaisesRegex(bank.T2VFactorialBankError, "split isolation"):
            bank.validate_event_audit(manifest, receipt, leaked)

        semantic = deepcopy(audit)
        row = next(row for row in semantic["rows"] if row["semantic_branch"] == "wrong_actor")
        row["correct_actor_performs_target"] = True
        row["distractor_actor_performs_target"] = False
        semantic["split_isolation"]["discovery_decisions_sha256"] = bank.object_sha256([row for row in semantic["rows"] if row["analysis_split"] == "discovery"])
        _seal(semantic, "audit_digest")
        result = bank.validate_event_audit(manifest, receipt, semantic)
        self.assertFalse(result["row_pass_by_entry_id"][row["entry_id"]])
        self.assertEqual(
            result["independently_event_verified_counts"][
                "fully_realized_discovery_cells"
            ],
            23,
        )

        low_confidence = deepcopy(audit)
        low_row = next(
            row
            for row in low_confidence["rows"]
            if row["semantic_branch"] == "full_action"
        )
        low_row["assessor_confidence"] = "low"
        low_confidence["split_isolation"]["discovery_decisions_sha256"] = bank.object_sha256(
            [
                row
                for row in low_confidence["rows"]
                if row["analysis_split"] == "discovery"
            ]
        )
        _seal(low_confidence, "audit_digest")
        low_result = bank.validate_event_audit(
            manifest, receipt, low_confidence
        )
        self.assertFalse(low_result["row_pass_by_entry_id"][low_row["entry_id"]])

        missing_contact = deepcopy(audit)
        incomplete_row = next(
            row
            for row in missing_contact["rows"]
            if row["semantic_branch"] == "incomplete_action"
        )
        incomplete_row["ordered_milestone_realized"][1] = False
        incomplete_row["ordered_milestone_first_frame"][1] = None
        missing_contact["split_isolation"][
            "discovery_decisions_sha256"
        ] = bank.object_sha256(
            [
                row
                for row in missing_contact["rows"]
                if row["analysis_split"] == "discovery"
            ]
        )
        _seal(missing_contact, "audit_digest")
        missing_contact_result = bank.validate_event_audit(
            manifest, receipt, missing_contact
        )
        self.assertFalse(
            missing_contact_result["row_pass_by_entry_id"][
                incomplete_row["entry_id"]
            ]
        )


if __name__ == "__main__":
    unittest.main()
