#!/usr/bin/env python3

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import validate_v15a_zero_update_upper_bound_sidecar_v1 as subject


def fixture(
    video: Path,
    transport: str = subject.TRANSPORTS[0],
    arm_role: str = "route_on_temporal",
):
    route_off = arm_role == "route_off_plain_frozen"
    transport_steps = 0 if route_off else 40
    expected_cells = 0 if route_off else subject.EXPECTED_CELLS
    expected_captures = 0 if route_off else subject.EXPECTED_CAPTURES
    expected_replays = 0 if route_off else subject.EXPECTED_REPLAYS
    schedule = [
        {
            "step_index": i,
            "candidate_count": 5 if i < 3 else 1,
            "anchor_timestep": float(999 - i),
            "outer_timestep": float(999 - i),
            "anchor_sigma": float(40 - i) / 40,
            "outer_sigma": float(40 - i) / 40,
            "cap_applied": False,
        }
        for i in range(transport_steps)
    ]
    freeze = {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
        "adapter_modules_absent": True,
    }
    cache = {
        "method": "bernini-online-pure-t2v-anchor-qk-transport-v3",
        "capture_count": expected_captures,
        "qk_only_capture_count": expected_captures,
        "replay_count": expected_replays,
        "qk_only_replay_count": expected_replays,
        "pending_entries": 0,
        "qk_only_cached_fields": ["query", "key"],
        "qk_only_forbidden_cached_fields": subject.FORBIDDEN_CACHE_FIELDS,
        "selected_block_indices": subject.BLOCKS,
    }
    trace = {
        "anchor_state_mode": "clean_noised",
        "anchor_contrast_mode": "dynamic_static_same_caption",
        "anchor_reference_is_static_phase0_video": True,
        "anchor_initial_gaussian_used_at_step0_candidate0": True,
        "anchor_model_forwards": 2 * expected_cells,
        "anchor_candidate_cells": expected_cells,
        "target_owned_qk_route_v14r2": not route_off,
        "anchor_donor_cached_fields": None if route_off else ["query", "key"],
        "anchor_donor_value_hidden_output_or_coordinate_used": None if route_off else False,
        "anchor_to_target_appearance_correspondence_used": None if route_off else False,
        "anchor_temporal_attention_kernel_contrast": not route_off,
        "anchor_temporal_kernel_applied_to_target_value_only": not route_off,
        "anchor_target_activity_gated_hard_kernel": transport == subject.TRANSPORTS[1] and not route_off,
        "anchor_route_shared_by_target_negative_and_condition": True,
        "anchor_route_target_conditional_only": False,
        "initial_latent_phase_clamped_after_every_update": True,
        "anchor_value_stream_copied": False,
        "source_value_stream_retained": True,
        "anchor_present_in_every_active_target_candidate": not route_off,
        "anchor_present_after_active_interval": False,
        "anchor_active_schedule": schedule,
        "anchor_action_reward_used_for_sga": False,
        "sga_weights_forced_to_anchor_candidate0": False,
        "candidate_counts": [5, 5, 5] + [1] * 37,
        "attention_cache": cache,
    }
    mechanism = {
        "arm": "AQK_SGA5",
        "transport": transport,
        "transport_strength": 1.0,
        "transport_steps": transport_steps,
        "initial_phase_clamp": True,
        "field_guidance": "raw_cfg",
        "field_model": "first_phase_caption_i2v",
        "source_cfg_scale": 4.5,
        "target_cfg_scale": 4.5,
        "early_candidate_count": 5,
        "initial_noise_proposal_mode": "anchor_candidate0",
        "anchor_state_mode": "clean_noised",
        "anchor_cfg_scope": "shared",
        "anchor_contrast_mode": "dynamic_static_same_caption",
        "anchor_sigma_cap": 1.0,
        "preservation_mode": "none",
        "preservation_residual_fraction": 0.0,
        "preservation_object_identity_strength": 0.0,
        "anchor_candidate_mode": "single_shared",
        "anchor_bank_size": 1,
        "sga_score_mode": "global_source_cosine",
        "anchor_spatial_alignment": "none",
        "event01_forced_role_proposal_index": -1,
        "selected_blocks": subject.BLOCKS,
        "pure_t2v_anchor_online_block_transport_enabled": not route_off,
        "pure_t2v_anchor_online_velocity_transport_enabled": False,
        "pure_t2v_anchor_values_or_pixels_copied_to_output": False,
        "decode_audit_contract": {
            "transport_steps": transport_steps,
            "anchor_state_mode": "clean_noised",
            "anchor_cfg_scope": "shared",
            "source_cfg_scale": 4.5,
            "target_cfg_scale": 4.5,
            "source_and_target_cfg_equal": True,
            "pure_t2v_teacher_adapter_policy": "plain_frozen_base",
            "target_source_editor_adapter_policy": "plain_frozen_base",
            "trained_route_off_control_explicitly_allowed": False,
            "same_checkpoint_route_off_causal_control": False,
            "anchor_injection_enabled": not route_off,
        },
        "trace": trace,
    }
    return {
        "schema_version": subject.SOURCE_SCHEMA,
        "complete": True,
        "training_performed": False,
        "optimization_steps": 0,
        "loaded_trained_attention_checkpoint": False,
        "trained_attention_checkpoint": None,
        "freeze_before": freeze,
        "freeze_after": copy.deepcopy(freeze),
        "causal_control": {
            "enabled": False,
            "kind": None,
            "explicit_opt_in": False,
            "trained_adapter_loaded": False,
            "adapter_enabled_for_target_source_calls": False,
            "anchor_injection_enabled": not route_off,
            "transport_steps": transport_steps,
        },
        "source": {"sha256": subject.SOURCE_SHA256, "role": "clean_edit_state_identity_appearance_scene_authority"},
        "pure_t2v_anchor": {"sha256": subject.ANCHOR_SHA256, "active_solver_steps": transport_steps, "model_forward_at_every_active_solver_step_and_candidate": not route_off},
        "anchor_generation_initial_gaussian": {
            "file_sha256": subject.GAUSSIAN_FILE_SHA256,
            "role": "dynaedit_step0_candidate0_native_generation_noise",
            "tensor_identity": {"raw_storage_sha256": subject.GAUSSIAN_RAW_SHA256},
        },
        "prompts": dict(subject.PROMPT_SHA256),
        "mechanism": mechanism,
        "output": {"path": str(video), "sha256": subject._sha256(video), "frames": 81, "fps": 25},
    }


class V15AReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / "result.mp4"
        self.video.write_bytes(b"synthetic-video-for-receipt-unit-test")

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self, receipt, transport=subject.TRANSPORTS[0], arm_role="route_on_temporal"):
        return subject.validate_receipt(
            receipt,
            video=self.video,
            transport=transport,
            arm_role=arm_role,
            enforce_output_binding=False,
        )

    def test_temporal_contract_passes(self):
        proof = self.validate(fixture(self.video))
        self.assertTrue(proof["same_caption"])
        self.assertTrue(proof["same_model_timestep"])
        self.assertTrue(proof["same_exact_candidate_noise"])

    def test_activity25_contract_passes(self):
        receipt = fixture(self.video, subject.TRANSPORTS[1], "route_on_activity25")
        self.validate(receipt, subject.TRANSPORTS[1], "route_on_activity25")

    def test_routeoff_contract_passes_with_zero_anchor_and_qk(self):
        receipt = fixture(self.video, arm_role="route_off_plain_frozen")
        proof = self.validate(receipt, arm_role="route_off_plain_frozen")
        self.assertTrue(proof["route_injection_off"])
        self.assertFalse(proof["contrast_pair_executed"])

    def test_rejects_any_optimizer_update(self):
        receipt = fixture(self.video)
        receipt["optimization_steps"] = 1
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_adapter_or_checkpoint(self):
        receipt = fixture(self.video)
        receipt["loaded_trained_attention_checkpoint"] = True
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_caption_noop_contrast(self):
        receipt = fixture(self.video)
        receipt["mechanism"]["anchor_contrast_mode"] = "caption_noop_same_video"
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_nonshared_timestep(self):
        receipt = fixture(self.video)
        receipt["mechanism"]["trace"]["anchor_active_schedule"][7]["anchor_timestep"] += 1
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_keyed_only_instead_of_official_gaussian(self):
        receipt = fixture(self.video)
        receipt["mechanism"]["initial_noise_proposal_mode"] = "keyed_only"
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_incomplete_qk_replay(self):
        receipt = fixture(self.video)
        receipt["mechanism"]["trace"]["attention_cache"]["replay_count"] -= 1
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_prompt_hash_mutation(self):
        for name in (
            "action_mv2v_sha256",
            "source_noop_mv2v_sha256",
            "anchor_t2v_sha256",
            "anchor_noop_t2v_sha256",
            "source_t2v_sha256",
            "target_t2v_sha256",
            "negative_sha256",
        ):
            with self.subTest(name=name):
                receipt = fixture(self.video)
                receipt["prompts"][name] = "0" * 64
                with self.assertRaises(subject.V15AValidationError):
                    self.validate(receipt)

    def test_rejects_candidate_schedule_mutation(self):
        receipt = fixture(self.video)
        receipt["mechanism"]["trace"]["candidate_counts"][2] = 4
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_sga_shortcut_or_reward_mutation(self):
        mutations = (
            ("mechanism_score", "anchor_action_cosine"),
            ("action_reward", True),
            ("forced_candidate0", True),
        )
        for name, value in mutations:
            with self.subTest(name=name):
                receipt = fixture(self.video)
                if name == "mechanism_score":
                    receipt["mechanism"]["sga_score_mode"] = value
                elif name == "action_reward":
                    receipt["mechanism"]["trace"][
                        "anchor_action_reward_used_for_sga"
                    ] = value
                else:
                    receipt["mechanism"]["trace"][
                        "sga_weights_forced_to_anchor_candidate0"
                    ] = value
                with self.assertRaises(subject.V15AValidationError):
                    self.validate(receipt)

    def test_rejects_forced_event01_proposal(self):
        receipt = fixture(self.video)
        receipt["mechanism"]["event01_forced_role_proposal_index"] = 0
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt)

    def test_rejects_routeoff_anchor_forward(self):
        receipt = fixture(self.video, arm_role="route_off_plain_frozen")
        receipt["mechanism"]["trace"]["anchor_model_forwards"] = 1
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt, arm_role="route_off_plain_frozen")

    def test_rejects_routeoff_qk_capture(self):
        receipt = fixture(self.video, arm_role="route_off_plain_frozen")
        receipt["mechanism"]["trace"]["attention_cache"]["capture_count"] = 1
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt, arm_role="route_off_plain_frozen")

    def test_rejects_routeoff_qk_replay(self):
        receipt = fixture(self.video, arm_role="route_off_plain_frozen")
        receipt["mechanism"]["trace"]["attention_cache"]["replay_count"] = 1
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt, arm_role="route_off_plain_frozen")

    def test_rejects_routeoff_anchor_forward_scope(self):
        receipt = fixture(self.video, arm_role="route_off_plain_frozen")
        receipt["pure_t2v_anchor"][
            "model_forward_at_every_active_solver_step_and_candidate"
        ] = True
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt, arm_role="route_off_plain_frozen")

    def test_rejects_routeoff_injection_enabled(self):
        receipt = fixture(self.video, arm_role="route_off_plain_frozen")
        receipt["causal_control"]["anchor_injection_enabled"] = True
        with self.assertRaises(subject.V15AValidationError):
            self.validate(receipt, arm_role="route_off_plain_frozen")

    def test_exact_output_binding_rejects_arbitrary_label(self):
        with self.assertRaises(subject.V15AValidationError):
            subject.validate_receipt(
                fixture(self.video), video=self.video, transport=subject.TRANSPORTS[0]
            )


class V15ALauncherStaticTest(unittest.TestCase):
    def test_launcher_contains_node_manifest_triplet_and_compute_assertions(self):
        launcher = METHOD_ROOT / "scripts/auh_launch_v15a_zero_update_upper_bound_e00_v1.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("node=auh7-1b-gpu-292", text)
        self.assertNotIn("node=auh7-1b-gpu-315", text)
        self.assertIn('test "$(hostname -s)" = "$EXPECTED_COMPUTE_NODE"', text)
        self.assertIn("route_off_plain_frozen", text)
        self.assertIn("MATCHED_PLAIN_FROZEN_ROUTEOFF_K0_A100", text)
        self.assertIn(subject.AUTHORING_MANIFEST_SHA256, text)
        self.assertIn(subject.EXPERIMENT_TAG, text)
        self.assertIn("dynaedit_maxstrength_routeprobe_v15a_r3", text)
        self.assertIn("ANCHOR_CAPTION_OVERRIDE", text)
        self.assertIn("v15a_launch_receipt.json", text)
        self.assertIn('validator_candidates=("$script_dir/../$validator_name" "$script_dir/$validator_name")', text)
        self.assertIn('test "$validator_matches" = 1', text)
        self.assertIn("verify_video_geometry \"$source_video\"", text)
        self.assertIn("verify_video_geometry \"$anchor_video\"", text)
        for label in subject.ARM_LABELS.values():
            self.assertTrue(label.endswith(".mp4"))
            self.assertIn(label[:-4], text)
        for digest in subject.PROMPT_SHA256.values():
            self.assertIn(digest, text)
        routeoff = text.index('run_arm "$routeoff_label"')
        temporal = text.index('run_arm "$temporal_label"')
        activity = text.index('run_arm "$activity_label"')
        self.assertLess(routeoff, temporal)
        self.assertLess(temporal, activity)
        self.assertNotIn("scancel", text)
        self.assertNotIn("frozen_upperbound", text)
        self.assertNotIn(
            "v15a_zero_update_dynamic_static_e00_maxstrength_routeprobe_20260820",
            text,
        )
        self.assertNotIn(
            "v15a_zero_update_dynamic_static_e00_maxstrength_routeprobe_r2_20260820",
            text,
        )
        self.assertNotIn("VALIDATOR_SHA256_PLACEHOLDER", text)

    def test_launcher_waits_fail_closed_for_lustre_visibility(self):
        launcher = METHOD_ROOT / "scripts/auh_launch_v15a_zero_update_upper_bound_e00_v1.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("wait_sha_complete_json_visible", text)
        self.assertIn('launch_receipt_compute_sha="$(srun', text)
        self.assertIn('receipt_tmp="${LAUNCH_RECEIPT}.compute-step-', text)
        self.assertIn('mv -- "$receipt_tmp" "$LAUNCH_RECEIPT"', text)
        self.assertIn("attempt <= 60", text)
        self.assertIn(
            'wait_sha_file_visible "$video" "$sidecar_video_sha" "$arm_role MP4"',
            text,
        )
        self.assertIn(
            'wait_complete_json_visible "$sidecar" "$arm_role native sidecar"',
            text,
        )
        self.assertIn(
            '"$launch_receipt" "$launch_receipt_compute_sha" "v15a-r3 launch receipt"',
            text,
        )

    def test_compute_launch_receipt_payload_executes_and_returns_sha(self):
        launcher = METHOD_ROOT / "scripts/auh_launch_v15a_zero_update_upper_bound_e00_v1.sh"
        text = launcher.read_text(encoding="utf-8")
        region_start = text.index('launch_receipt_compute_sha="$(srun')
        payload_start = text.index("    bash -c '\n", region_start) + len(
            "    bash -c '\n"
        )
        payload_end = text.index("\n    ')\"", payload_start)
        payload = text[payload_start:payload_end].replace("'\\''", "'")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            root.mkdir()
            fixture_bin = Path(directory) / "bin"
            fixture_bin.mkdir()
            sha256sum = fixture_bin / "sha256sum"
            sha256sum.write_text(
                "#!/usr/bin/env bash\nexec shasum -a 256 \"$@\"\n",
                encoding="ascii",
            )
            sha256sum.chmod(0o755)
            receipt = root / "v15a_launch_receipt.json"
            host = subprocess.run(
                ["hostname", "-s"], check=True, capture_output=True, text=True
            ).stdout.strip()
            zero = "0" * 64
            env = dict(os.environ)
            env["PATH"] = str(fixture_bin) + os.pathsep + env["PATH"]
            env.update(
                {
                    "EXPECTED_COMPUTE_NODE": host,
                    "EXPECTED_PARENT_JOB": "143808",
                    "SLURM_JOB_ID": "143808",
                    "SLURM_STEP_ID": "fixture",
                    "EXPERIMENT_TAG": subject.EXPERIMENT_TAG,
                    "FIXED_OUTPUT_ROOT": str(root),
                    "LAUNCH_RECEIPT": str(receipt),
                    "LAUNCHER_PATH": "/fixture/launcher.sh",
                    "LAUNCHER_SHA": zero,
                    "VALIDATOR_PATH": "/fixture/validator.py",
                    "VALIDATOR_SHA": zero,
                    "BRIDGE_PATH": "/fixture/bridge.sh",
                    "BRIDGE_SHA": zero,
                    "CONTROLLER_SHA": zero,
                    "QK_SHA": zero,
                    "INFER_SHA": zero,
                    "DECODE_SHA": zero,
                    "DEPLOYMENT_VALIDATOR_PATH": "/fixture/deployment-validator.py",
                    "DEPLOYMENT_VALIDATOR_SHA": zero,
                    "AUTHORING_MANIFEST": "/fixture/authoring.json",
                    "AUTHORING_SHA": zero,
                    "MARKER_PATH": "/fixture/marker.json",
                    "MARKER_SHA": zero,
                    "ARCHIVE_PATH": "/fixture/archive.tar",
                    "ARCHIVE_SHA": zero,
                    "REVISION_PATH": "/fixture/revision",
                    "REVISION_SHA": zero,
                    "CONTENT_PATH": "/fixture/content.json",
                    "CONTENT_SHA": zero,
                    "ROUTEOFF_LABEL": subject.ARM_LABELS[
                        "route_off_plain_frozen"
                    ][:-4],
                    "TEMPORAL_LABEL": subject.ARM_LABELS["route_on_temporal"][:-4],
                    "ACTIVITY_LABEL": subject.ARM_LABELS["route_on_activity25"][:-4],
                }
            )
            result = subprocess.run(
                ["bash", "-c", payload],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            expected_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
            self.assertEqual(result.stdout.strip(), expected_sha)
            data = json.loads(receipt.read_text(encoding="ascii"))
            self.assertTrue(data["complete"])
            self.assertEqual(data["experiment_tag"], subject.EXPERIMENT_TAG)
            self.assertEqual(
                [row["role"] for row in data["arms"]],
                [
                    "route_off_plain_frozen",
                    "route_on_temporal",
                    "route_on_activity25",
                ],
            )
            self.assertEqual(data["matched_contract"]["candidate_counts"], [5, 5, 5] + [1] * 37)
            self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
