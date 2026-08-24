from __future__ import annotations

import importlib.util
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("numpy unavailable") from error


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_object_proposal_role_probe_v15c.py"
ASSET = ROOT / "assets/e00_source_sam2_proposal_role_probe_v15c.json"
MATERIALIZER = ROOT / "materialize_source_sam2_proposal_tracks_v15c.py"
RUNNER = ROOT / "run_source_object_proposal_role_probe_v15c.py"
POSTFLIGHT = ROOT / "postflight_source_sam2_proposal_role_probe_v15c_r3.py"
OVERLAY = ROOT / "tools/build_source_object_proposal_role_v15c_r3_review.py"
FINALIZER = ROOT / "finalize_source_sam2_proposal_role_probe_v15c_r3.py"
LAUNCHER = ROOT / "scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r3_sealed.sh"
RELEASE = ROOT / "assets/e00_source_sam2_proposal_role_probe_v15c_r3_release.json"


def load_module():
    spec = importlib.util.spec_from_file_location("proposal_role_v15c", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve the module through sys.modules while decorating.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_module()


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MAT = load_path("proposal_tracks_v15c", MATERIALIZER)
FIN = load_path("proposal_role_v15c_r3_finalizer", FINALIZER)


def proposal_id(label: str) -> str:
    return "sam2-f000-" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def rectangle(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    value = np.zeros((M.GRID_HEIGHT, M.GRID_WIDTH), dtype=np.float32)
    value[y0:y1, x0:x1] = 1.0
    return value


def synthetic_inputs(signal_phases: int = 21):
    generator = np.random.default_rng(20260821)
    masks = np.stack(
        [
            rectangle(4, 9, 3, 7),
            rectangle(14, 20, 10, 15),
            rectangle(27, 32, 1, 5),
        ],
        axis=0,
    )
    coverage = np.repeat(masks[:, None], M.PHASE_COUNT, axis=1)
    real = generator.normal(
        0.0,
        1.0,
        size=(5, 3, M.PHASE_COUNT, M.GRID_HEIGHT, M.GRID_WIDTH),
    ).astype(np.float32)
    shuffled = generator.normal(
        0.0,
        1.0,
        size=real.shape,
    ).astype(np.float32)
    null = generator.normal(
        0.0,
        1.0,
        size=(5, M.NULL_COUNT, M.PHASE_COUNT, M.GRID_HEIGHT, M.GRID_WIDTH),
    ).astype(np.float32)
    for block in range(5):
        for role in range(3):
            for phase in range(signal_phases):
                real[block, role, phase] += np.float32(9.0) * masks[role]
    tracks = M.ProposalTrackInputV15C(
        proposal_ids=tuple(
            proposal_id(label) for label in ("p-old", "p-new", "p-recipient")
        ),
        phase_coverage=np.ascontiguousarray(coverage),
        track_gate_pass=(True, True, True),
    )
    affinity = M.R6AffinityInputV15C(
        real=np.ascontiguousarray(real),
        shuffled=np.ascontiguousarray(shuffled),
        null_bank=np.ascontiguousarray(null),
    )
    return tracks, affinity


class SourceProposalRoleProbeV15CTests(unittest.TestCase):
    def test_clear_three_role_regions_pass_without_forcing(self):
        tracks, affinity = synthetic_inputs()
        result = M.run_source_object_proposal_role_probe_v15c(
            tracks=tracks, affinity=affinity
        )
        self.assertTrue(result["mechanical_candidate_qualified"])
        self.assertEqual(
            result["assignments"],
            {
                "old_actor": proposal_id("p-old"),
                "new_actor": proposal_id("p-new"),
                "recipient": proposal_id("p-recipient"),
            },
        )
        self.assertFalse(result["forced_assignment"])
        self.assertFalse(result["route_authorized"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["affinity_used_pointwise_as_mask"])

    def test_token_permutation_equal_to_real_is_no_go(self):
        tracks, affinity = synthetic_inputs()
        mutated = M.R6AffinityInputV15C(
            real=affinity.real,
            shuffled=affinity.real.copy(),
            null_bank=affinity.null_bank,
        )
        result = M.run_source_object_proposal_role_probe_v15c(
            tracks=tracks, affinity=mutated
        )
        self.assertFalse(result["mechanical_candidate_qualified"])
        self.assertTrue(all(value is None for value in result["assignments"].values()))

    def test_temporally_sparse_signal_is_no_go(self):
        tracks, affinity = synthetic_inputs(signal_phases=5)
        result = M.run_source_object_proposal_role_probe_v15c(
            tracks=tracks, affinity=affinity
        )
        self.assertFalse(result["mechanical_candidate_qualified"])
        self.assertTrue(all(value is None for value in result["assignments"].values()))

    def test_duplicate_proposal_family_is_not_arbitrarily_selected(self):
        tracks, affinity = synthetic_inputs()
        duplicate_coverage = np.concatenate(
            [tracks.phase_coverage, tracks.phase_coverage[0:1]], axis=0
        )
        duplicate_tracks = M.ProposalTrackInputV15C(
            proposal_ids=tuple(
                proposal_id(label)
                for label in ("p-old", "p-new", "p-recipient", "p-old-duplicate")
            ),
            phase_coverage=duplicate_coverage,
            track_gate_pass=(True, True, True, True),
        )
        result = M.run_source_object_proposal_role_probe_v15c(
            tracks=duplicate_tracks, affinity=affinity
        )
        self.assertIsNone(result["assignments"]["old_actor"])
        self.assertIn(
            proposal_id("p-old-duplicate"),
            result["duplicate_proposal_adjacency"][proposal_id("p-old")],
        )
        self.assertFalse(result["forced_assignment"])

    def test_cross_role_same_proposal_conflict_unassigns_every_conflicting_role(self):
        tracks, affinity = synthetic_inputs()
        real = affinity.real.copy()
        first = tracks.phase_coverage[0, 0]
        # All roles receive their only strong evidence on proposal zero.
        generator = np.random.default_rng(9)
        real[:] = generator.normal(0.0, 1.0, size=real.shape)
        for block in range(5):
            for role in range(3):
                for phase in range(M.PHASE_COUNT):
                    real[block, role, phase] += np.float32(9.0) * first
        conflict_affinity = M.R6AffinityInputV15C(
            real=np.ascontiguousarray(real),
            shuffled=affinity.shuffled,
            null_bank=affinity.null_bank,
        )
        result = M.run_source_object_proposal_role_probe_v15c(
            tracks=tracks, affinity=conflict_affinity
        )
        self.assertFalse(result["mechanical_candidate_qualified"])
        self.assertEqual(
            result["cross_role_conflicts"][proposal_id("p-old")],
            list(M.ROLE_NAMES),
        )
        self.assertTrue(all(value is None for value in result["assignments"].values()))

    def test_proposal_order_permutation_preserves_id_assignments(self):
        tracks, affinity = synthetic_inputs()
        reference = M.run_source_object_proposal_role_probe_v15c(
            tracks=tracks, affinity=affinity
        )
        order = np.asarray([2, 0, 1])
        permuted = M.ProposalTrackInputV15C(
            proposal_ids=tuple(tracks.proposal_ids[index] for index in order),
            phase_coverage=np.ascontiguousarray(tracks.phase_coverage[order]),
            track_gate_pass=tuple(tracks.track_gate_pass[index] for index in order),
        )
        observed = M.run_source_object_proposal_role_probe_v15c(
            tracks=permuted, affinity=affinity
        )
        self.assertEqual(reference["assignments"], observed["assignments"])

    def test_invalid_track_gate_cannot_receive_role(self):
        tracks, affinity = synthetic_inputs()
        invalid = M.ProposalTrackInputV15C(
            proposal_ids=tracks.proposal_ids,
            phase_coverage=tracks.phase_coverage,
            track_gate_pass=(True, False, True),
        )
        result = M.run_source_object_proposal_role_probe_v15c(
            tracks=invalid, affinity=affinity
        )
        self.assertIsNone(result["assignments"]["new_actor"])
        self.assertFalse(result["mechanical_candidate_qualified"])

    def test_proposal_max_null_blocks_own_null_false_positive(self):
        real = np.full((3, 2, M.PHASE_COUNT), -5.0, dtype=np.float32)
        shuffled = np.full_like(real, -6.0)
        real[0, 0] = 5.0
        own_null = np.zeros((2, M.NULL_COUNT, M.PHASE_COUNT), dtype=np.float32)
        proposal_max_phase = np.linspace(
            -2.0, 2.0, M.NULL_COUNT, dtype=np.float64
        )[:, None].repeat(M.PHASE_COUNT, axis=1)
        proposal_max_track = np.linspace(-2.0, 2.0, M.NULL_COUNT, dtype=np.float64)
        proposal_max_track[-1] = 6.0
        row = M._candidate_evidence(
            role_index=0,
            proposal_index=0,
            real_scores=real,
            shuffled_scores=shuffled,
            null_scores=own_null,
            proposal_max_null_phase=proposal_max_phase,
            proposal_max_null_track=proposal_max_track,
            track_valid=True,
            duplicate_neighbors=(),
            thresholds=M.ProbeThresholdsV15C(),
        )
        self.assertEqual(row["own_proposal_null_median"], 0.0)
        self.assertFalse(
            row["gates"][
                "track_above_proposal_max_null_with_three_role_fwer"
            ]
        )
        self.assertGreater(row["three_role_bonferroni_fwer_upper_p"], 0.05)

    def test_three_role_fwer_requires_beating_all_64_proposal_max_nulls(self):
        real = np.full((3, 1, M.PHASE_COUNT), -5.0, dtype=np.float32)
        shuffled = np.full_like(real, -6.0)
        real[0, 0] = 5.0
        own_null = np.zeros((1, M.NULL_COUNT, M.PHASE_COUNT), dtype=np.float32)
        proposal_max_track = np.linspace(-2.0, 2.0, M.NULL_COUNT, dtype=np.float64)
        proposal_max_phase = proposal_max_track[:, None].repeat(
            M.PHASE_COUNT, axis=1
        )
        row = M._candidate_evidence(
            role_index=0,
            proposal_index=0,
            real_scores=real,
            shuffled_scores=shuffled,
            null_scores=own_null,
            proposal_max_null_phase=proposal_max_phase,
            proposal_max_null_track=proposal_max_track,
            track_valid=True,
            duplicate_neighbors=(),
            thresholds=M.ProbeThresholdsV15C(),
        )
        self.assertAlmostEqual(row["proposal_max_null_raw_upper_p"], 1.0 / 65.0)
        self.assertAlmostEqual(
            row["three_role_bonferroni_fwer_upper_p"], 3.0 / 65.0
        )
        self.assertTrue(
            row["gates"][
                "track_above_proposal_max_null_with_three_role_fwer"
            ]
        )

    def test_winner_must_dominate_every_eligible_proposal(self):
        ids = tuple(proposal_id(f"eligible-{index}") for index in range(3))
        evidence = [
            {"eligible_before_proposal_competition": True, "evidence_margin": 3.0},
            {"eligible_before_proposal_competition": True, "evidence_margin": 2.0},
            {"eligible_before_proposal_competition": True, "evidence_margin": 1.0},
        ]
        scores = np.zeros((3, 3, M.PHASE_COUNT), dtype=np.float32)
        scores[0, 0] = 2.0
        scores[0, 1] = 1.0
        scores[0, 2] = 1.0
        scores[0, 2, :6] = 3.0
        winner, detail = M._choose_without_forcing(
            role_index=0,
            evidence=evidence,
            proposal_ids=ids,
            real_scores=scores,
            thresholds=M.ProbeThresholdsV15C(),
        )
        self.assertIsNone(winner)
        self.assertEqual(
            detail["status"],
            "unassigned_winner_failed_all_eligible_temporal_dominance",
        )
        self.assertEqual(detail["dominance_phase_count_by_competitor_id"][ids[2]], 15)

    def test_exact_top_margin_tie_is_unassigned(self):
        ids = tuple(proposal_id(f"tie-{index}") for index in range(2))
        winner, detail = M._choose_without_forcing(
            role_index=0,
            evidence=[
                {"eligible_before_proposal_competition": True, "evidence_margin": 1.0},
                {"eligible_before_proposal_competition": True, "evidence_margin": 1.0},
            ],
            proposal_ids=ids,
            real_scores=np.zeros((3, 2, M.PHASE_COUNT), dtype=np.float32),
            thresholds=M.ProbeThresholdsV15C(),
        )
        self.assertIsNone(winner)
        self.assertEqual(detail["status"], "unassigned_non_unique_top_evidence_margin")

    def test_proposal_count_and_strict_bool_registry_fail_closed(self):
        empty = np.zeros(
            (0, M.PHASE_COUNT, M.GRID_HEIGHT, M.GRID_WIDTH), dtype=np.float32
        )
        with self.assertRaisesRegex(M.SourceProposalRoleProbeV15CError, "registry"):
            M.ProposalTrackInputV15C((), empty, ())
        count = 65
        over = np.zeros(
            (count, M.PHASE_COUNT, M.GRID_HEIGHT, M.GRID_WIDTH), dtype=np.float32
        )
        with self.assertRaisesRegex(M.SourceProposalRoleProbeV15CError, "registry"):
            M.ProposalTrackInputV15C(
                tuple(proposal_id(f"over-{index}") for index in range(count)),
                over,
                (True,) * count,
            )
        with self.assertRaisesRegex(M.SourceProposalRoleProbeV15CError, "registry"):
            M.ProposalTrackInputV15C(
                (proposal_id("bad-bool"),),
                np.zeros(
                    (1, M.PHASE_COUNT, M.GRID_HEIGHT, M.GRID_WIDTH),
                    dtype=np.float32,
                ),
                (np.bool_(True),),
            )

    def test_asset_is_roi_free_fail_closed_and_exact_e00(self):
        value = json.loads(ASSET.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(ASSET.read_bytes()).hexdigest(),
            MAT.EXPECTED_SPEC_RAW_SHA256,
        )
        self.assertEqual(
            M.object_sha256(value), MAT.EXPECTED_SPEC_CANONICAL_SHA256
        )
        self.assertEqual(
            value["source"]["sha256"],
            "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
        )
        self.assertEqual(value["r6"]["null_span_count"], 64)
        self.assertFalse(value["role_assignment"]["forced_assignment"])
        self.assertFalse(value["role_assignment"]["roi_or_manual_box_consumed"])
        self.assertFalse(value["claim_limits"]["route_authorized"])
        self.assertFalse(value["claim_limits"]["training_authorized"])
        self.assertEqual(
            value["sam2"]["proposal_admission"]["overflow_policy"],
            "fail_closed_without_ranking_or_truncation",
        )
        self.assertEqual(
            value["sam2"]["automatic_generator"]["min_mask_region_area"], 0
        )
        self.assertEqual(value["sam2"]["tracking"]["fill_hole_area"], 0)
        self.assertFalse(value["sam2"]["hydra"]["image_apply_postprocessing"])
        self.assertFalse(value["sam2"]["hydra"]["video_apply_postprocessing"])
        self.assertEqual(value["execution"]["parent_job_id"], 143808)
        self.assertEqual(value["execution"]["required_node"], "auh7-1b-gpu-292")
        self.assertEqual(value["execution"]["required_visible_gpu_count"], 1)
        MAT.validate_spec(value)
        mutated = json.loads(json.dumps(value))
        mutated["unexpected"] = False
        with self.assertRaisesRegex(
            MAT.SourceSAM2ProposalTracksV15CError, "canonical spec"
        ):
            MAT.validate_spec(mutated)

    def test_generic_proposal_admission_deduplicates_but_never_truncates(self):
        first = np.zeros((20, 20), dtype=np.bool_)
        first[2:8, 2:8] = True
        duplicate = first.copy()
        second = np.zeros_like(first)
        second[10:16, 10:16] = True

        def row(mask, quality):
            return {
                "segmentation": mask,
                "area": int(mask.sum()),
                "predicted_iou": quality,
                "stability_score": quality,
            }

        observed = MAT.admit_automatic_proposals(
            [row(first, 0.95), row(duplicate, 0.90), row(second, 0.94)],
            image_area=400,
            minimum_area_pixels=4,
            maximum_area_fraction=0.5,
            near_duplicate_iou=0.9,
            maximum_distinct_proposals=2,
        )
        self.assertEqual(len(observed), 2)
        observed_ids = [
            MAT.array_sha256(row["segmentation"].astype(np.uint8))
            for row in observed
        ]
        self.assertEqual(observed_ids, sorted(observed_ids))
        with self.assertRaisesRegex(
            MAT.SourceSAM2ProposalTracksV15CError, "without truncation"
        ):
            MAT.admit_automatic_proposals(
                [row(first, 0.95), row(second, 0.94)],
                image_area=400,
                minimum_area_pixels=4,
                maximum_area_fraction=0.5,
                near_duplicate_iou=0.9,
                maximum_distinct_proposals=1,
            )

    def test_track_geometry_is_fail_closed(self):
        mask = np.zeros((64, 64), dtype=np.bool_)
        mask[20:36, 18:34] = True
        spec = json.loads(ASSET.read_text(encoding="utf-8"))["sam2"]["tracking"]
        passed = MAT.track_geometry_receipt(
            [mask.copy() for _ in range(81)],
            mask,
            width=64,
            height=64,
            tracking_spec=spec,
        )
        self.assertTrue(passed["automatic_track_geometry_gate_pass"])
        failed_masks = [mask.copy() for _ in range(81)]
        failed_masks[40][:] = False
        failed = MAT.track_geometry_receipt(
            failed_masks,
            mask,
            width=64,
            height=64,
            tracking_spec=spec,
        )
        self.assertFalse(failed["automatic_track_geometry_gate_pass"])
        self.assertFalse(
            failed["automatic_track_geometry_gates"]["all_81_frames_visible"]
        )
        highlight = np.zeros((64, 64), dtype=np.bool_)
        highlight[8:56, 31:32] = True
        fragment = MAT.track_geometry_receipt(
            [highlight.copy() for _ in range(81)],
            highlight,
            width=64,
            height=64,
            tracking_spec=spec,
        )
        self.assertFalse(fragment["automatic_track_geometry_gate_pass"])
        self.assertFalse(
            fragment["automatic_track_geometry_gates"]["whole_object_area_extent"]
        )
        self.assertEqual(
            fragment["whole_object_observer_scope"],
            "source_mask_geometry_only_no_material_or_transparency_label",
        )

    def test_track_loader_does_not_coerce_integer_gate_to_bool(self):
        try:
            from safetensors.numpy import save_file
        except ImportError as error:  # pragma: no cover
            self.skipTest(f"safetensors unavailable: {error}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = {
                "schema_version": M.TRACK_SCHEMA_VERSION,
                "proposals": [
                    {
                        "proposal_id": proposal_id("loader-strict-bool"),
                        "automatic_track_geometry_gate_pass": 1,
                    }
                ],
            }
            metadata = root / "track_receipt.json"
            metadata.write_text(json.dumps(receipt), encoding="utf-8")
            tensors = root / "phase_coverage.safetensors"
            save_file(
                {
                    "phase_coverage": np.zeros(
                        (1, M.PHASE_COUNT, M.GRID_HEIGHT, M.GRID_WIDTH),
                        dtype=np.float32,
                    )
                },
                str(tensors),
            )
            with self.assertRaisesRegex(
                M.SourceProposalRoleProbeV15CError, "track gate registry"
            ):
                M.load_tracks_for_v15c(metadata, tensors)

    def test_materializer_has_exact_hydra_freeze_repeat_and_manifest_gates(self):
        source = MATERIALIZER.read_text(encoding="utf-8")
        self.assertNotIn("apply_postprocessing=True", source)
        self.assertIn("apply_postprocessing=False", source)
        self.assertIn("full SAM2 repeat/RNG gate", source)
        self.assertIn("parameter_and_buffer_bytes_unchanged", source)
        self.assertIn("actual_hydra_config_samefile_as_authority", source)
        self.assertIn("output_manifest.json", source)
        self.assertIn("prompts", source)
        self.assertIn("mask_array_sha256_by_frame", source)

    def test_programs_contain_no_text_detector_training_route_or_renderer_call(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (SOURCE, MATERIALIZER, RUNNER, POSTFLIGHT, OVERLAY)
        )
        lowered = combined.lower()
        self.assertNotIn("qwen", lowered)
        self.assertNotIn("groundingdino", lowered)
        self.assertNotIn("gdino", lowered)
        called = set()
        for path in (SOURCE, MATERIALIZER, RUNNER, POSTFLIGHT, OVERLAY):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            called.update(
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            )
        for forbidden in ("backward", "step", "zero_grad", "decode"):
            self.assertNotIn(forbidden, called)
        self.assertNotIn("optimizer =", lowered)

    def test_overlay_is_reject_only_and_synchronized(self):
        source = OVERLAY.read_text(encoding="utf-8")
        self.assertIn("同步播放全部", source)
        self.assertIn("reject_only", source)
        self.assertIn("approve_action_available", source)
        self.assertNotIn("Approve candidate", source)
        self.assertIn("DISPLAY_FRAMES = (0, 20, 40, 60, 80)", source)
        self.assertIn("all proposal evidence", source)

    def test_r3_source_only_property_and_family_conflict_contract(self):
        value = json.loads(ASSET.read_text(encoding="utf-8"))
        observer = value["source_object_observer"]
        self.assertTrue(observer["source_pixels_and_sam2_masks_only"])
        self.assertFalse(observer["anchor_consumed"])
        self.assertFalse(observer["target_instruction_consumed"])
        self.assertFalse(observer["material_or_transparency_classification"])
        self.assertEqual(
            value["role_assignment"]["family_overlap_nesting_policy"],
            "all_members_unassigned_before_role_competition",
        )
        tracks, affinity = synthetic_inputs()
        coverage = tracks.phase_coverage.copy()
        coverage[1] = coverage[0]
        conflicted = M.ProposalTrackInputV15C(
            proposal_ids=tracks.proposal_ids,
            phase_coverage=coverage,
            track_gate_pass=tracks.track_gate_pass,
        )
        result = M.run_source_object_proposal_role_probe_v15c(
            tracks=conflicted, affinity=affinity
        )
        first = tracks.proposal_ids[0]
        second = tracks.proposal_ids[1]
        adjacency = result["source_proposal_family_overlap_nesting_adjacency"]
        self.assertIn(second, adjacency[first])
        self.assertIn(first, adjacency[second])
        for role in M.ROLE_NAMES:
            rows = {row["proposal_id"]: row for row in result["evidence"][role]}
            self.assertFalse(
                rows[first]["gates"][
                    "no_source_family_overlap_or_nesting_conflict"
                ]
            )
            self.assertFalse(
                rows[second]["gates"][
                    "no_source_family_overlap_or_nesting_conflict"
                ]
            )

    def test_r3_launcher_and_finalizer_are_assert_free_and_sealed(self):
        launcher = LAUNCHER.read_text(encoding="utf-8")
        finalizer = FINALIZER.read_text(encoding="utf-8")
        self.assertNotIn("assert ", launcher)
        self.assertNotIn("assert ", finalizer)
        self.assertIn("-E -s -B", launcher)
        self.assertIn("sealed_code_snapshot", launcher)
        self.assertIn("COMPLETE.manifest.json", launcher)
        self.assertIn("verify_release", finalizer)
        release_file_sha = hashlib.sha256(RELEASE.read_bytes()).hexdigest()
        observed = FIN.verify_release(ROOT.parents[1], RELEASE, release_file_sha)
        self.assertEqual(observed["member_count"], 8)
        self.assertEqual(
            release_file_sha,
            "d1aedce0f95786603afcddf7a004aef012fd888efaea038e40f9dfc2100787eb",
        )

    def test_r3_repeat_transcript_exact_schema_and_second_publication_contract(self):
        def freeze(kind):
            return {
                "schema_version": MAT.FREEZE_SCHEMA,
                "model_kind": kind,
                "eval_mode_before": True,
                "eval_mode_after": True,
                "requires_grad_true_count_before": 0,
                "requires_grad_true_count_after": 0,
                "non_none_grad_count_before": 0,
                "non_none_grad_count_after": 0,
                "parameter_sha256_before": "1" * 64,
                "parameter_sha256_after": "1" * 64,
                "buffer_sha256_before": "2" * 64,
                "buffer_sha256_after": "2" * 64,
                "parameter_tensor_count": 1,
                "parameter_element_count": 1,
                "buffer_tensor_count": 1,
                "buffer_element_count": 1,
                "parameter_and_buffer_bytes_unchanged": True,
                "all_freeze_gates_pass": True,
            }

        propagation = [
            {
                "schema_version": MAT.PROPAGATION_LOGIT_SCHEMA,
                "frame_index": frame,
                "out_ids": [0],
                "shape": [1, 1, 1056, 704],
                "dtype": "torch.float32",
                "finite": True,
                "logits_sha256": hashlib.sha256(f"frame-{frame}".encode()).hexdigest(),
            }
            for frame in range(81)
        ]
        batch = {
            "schema_version": MAT.TRACKING_BATCH_SCHEMA,
            "batch_index": 0,
            "batch_start": 0,
            "batch_stop": 1,
            "object_ids": [0],
            "prompt_calls": [
                {
                    "schema_version": MAT.PROMPT_LOGIT_SCHEMA,
                    "inserted_object_id": 0,
                    "frame_index": 0,
                    "out_ids": [0],
                    "shape": [1, 1, 1056, 704],
                    "dtype": "torch.float32",
                    "finite": True,
                    "logits_sha256": "3" * 64,
                }
            ],
            "propagation_frames": propagation,
        }
        transcript = MAT.build_repeat_transcript(
            run_ordinal=2,
            proposal_rows=[{"proposal_id": proposal_id("transcript")}],
            prompt_signatures=[{"proposal_id": proposal_id("transcript")}],
            mask_signatures=[{"proposal_id": proposal_id("transcript")}],
            phase_coverage=np.zeros((1, 21, 37, 25), dtype=np.float32),
            tracking_batches=[batch],
            freeze_receipts={
                "image_model": freeze("image_model"),
                "video_model": freeze("video_model"),
            },
            sam2__C_imported=False,
        )
        MAT.validate_repeat_transcript(transcript, proposal_count=1, run_ordinal=2)
        self.assertEqual(transcript["run_ordinal"], 2)
        mutated = dict(transcript)
        mutated["unexpected"] = False
        with self.assertRaisesRegex(
            MAT.SourceSAM2ProposalTracksV15CError, "exact keys"
        ):
            MAT.validate_repeat_transcript(mutated, proposal_count=1, run_ordinal=2)

    def test_normal_and_optimized_interpreters_emit_identical_core_receipt(self):
        code = f"""
import importlib.util,json,sys,numpy as np
p={str(SOURCE)!r}
s=importlib.util.spec_from_file_location('v15c_opt_probe',p)
m=importlib.util.module_from_spec(s)
sys.modules[s.name]=m
s.loader.exec_module(m)
pid='sam2-f000-'+'0'*64
t=m.ProposalTrackInputV15C((pid,),np.ones((1,21,37,25),dtype=np.float32),(True,))
a=m.R6AffinityInputV15C(np.zeros((5,3,21,37,25),dtype=np.float32),np.zeros((5,3,21,37,25),dtype=np.float32),np.zeros((5,64,21,37,25),dtype=np.float32))
r=m.run_source_object_proposal_role_probe_v15c(tracks=t,affinity=a)
print(r['receipt_sha256'])
"""
        normal = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        optimized = subprocess.run(
            [sys.executable, "-O", "-c", code],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        self.assertRegex(normal, r"^[0-9a-f]{64}$")
        self.assertEqual(normal, optimized)


if __name__ == "__main__":
    unittest.main()
