from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
SOURCE = METHOD_ROOT / "decode_elal3_c2_simulator_oracle_q_v1.py"
ANALYZER = METHOD_ROOT / "analyze_elal3_c2_decoded_role_effect_v1.py"
SPEC = importlib.util.spec_from_file_location("decode_elal3_c2_simulator_oracle_q_v1", SOURCE)
assert SPEC is not None and SPEC.loader is not None
decoder = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = decoder
SPEC.loader.exec_module(decoder)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def noise_receipt(rank: int, *, seed: int = 20260821, tensor_sha: str = "9" * 64):
    return {
        "call_count": 1,
        "requested_shape": list(decoder.LATENT_SHAPE),
        "requested_device": f"cuda:{rank}",
        "requested_dtype": "torch.float32",
        "generator_device": "cpu",
        "generator_initial_seed": seed,
        "returned_object_forwarded_by_identity": True,
        "external_initial_noise_injection": False,
        "spatial_tensor_sha256": tensor_sha,
        "noise_factory": "diffusers.utils.torch_utils.randn_tensor",
        "native_observation_only_not_injection": True,
    }


def numeric_receipt(branch: str):
    return {
        "branch": branch,
        "forward_autocast_dtype": "torch.bfloat16",
        "forward_autocast_scope": "diff_dec.shared_step_only",
        "checkpoint_master_parameter_dtype": "torch.float32",
        "elal3_parameters_cast_to_bfloat16": False,
        "transformer_block_input_dtype_gate": "torch.bfloat16",
        "transformer_block_output_dtype_gate": "torch.bfloat16",
        "shared_step_output_dtype_gate": "torch.bfloat16",
        "shared_step_calls": 80,
        "expected_shared_step_calls": 80,
        "scheduler_outside_autocast": True,
        "scheduler_sample_dtype_gate": "torch.float32",
        "scheduler_output_dtype_gate": "torch.float32",
        "scheduler_step_calls": 40,
        "expected_scheduler_step_calls": 40,
        "transformer_block_input_calls": 2400,
        "transformer_block_output_calls": 2400,
        "expected_transformer_block_calls": 2400,
    }


def generated_receipt(branch: str, step, q_key, *, seed: int = 20260821):
    q_sources = {
        "target": "authenticated_full_target_annotation",
        "role_swap": "authenticated_full_role_swap_annotation",
        "target_role_mismatch": "target_fixed_fields_opposite_entity_relation_only",
        "wrong_agent": "authenticated_full_wrong_agent_annotation",
        "wrong_object": "authenticated_full_wrong_object_annotation",
        "zero_target": "all_zero_intervention_on_authenticated_target_q",
        "reverse": "authenticated_full_reverse_annotation",
        "phase_shuffle": "authenticated_full_phase_shuffle_annotation",
    }
    if q_key is None:
        local = {
            "checkpoint_step": None,
            "q_intervention": None,
            "oracle_q_teacher_forced": False,
            "q_ignored_because_elal_absent": True,
        }
    else:
        label_key = "target" if q_key == "zero_target" else q_key
        q_binding = {
            "q_source": q_sources[q_key],
            "label_digest": sha(label_key.encode()),
        }
        if q_key == "target_role_mismatch":
            q_binding["only_q_entity_and_q_relation_changed"] = True
        local = {
            "checkpoint_step": step,
            "q_intervention": q_key,
            "q_binding": q_binding,
            "oracle_q_teacher_forced": True,
            "elal_hook_audit": {
                "all30_used": True,
                "source_and_padding_bit_exact": True,
                "calls_by_block": {str(index): 80 for index in range(30)},
            },
            "renderer_numeric_path": numeric_receipt(branch),
        }
    latent_sha = sha(f"latent:{branch}".encode())
    ranks = [
        {
            "world_rank": rank,
            "latent_sha256": latent_sha,
            **copy.deepcopy(local),
            "initial_sampling_noise": noise_receipt(rank, seed=seed),
        }
        for rank in range(4)
    ]
    return {
        **copy.deepcopy(local),
        "initial_sampling_noise": noise_receipt(0, seed=seed),
        "generated_latent_sha256": latent_sha,
        "world4_full_latent_consensus": True,
        "world4_initial_sampling_noise_sha256_consensus": True,
        "world4_rank_receipts": ranks,
    }


def exact14_rows(*, seed: int = 20260821):
    probe_ref = {
        "frame_count": 81,
        "fps": 25.0,
        "fps_numerator": 25,
        "fps_denominator": 1,
        "height": 96,
        "width": 128,
        "pixel_format": "yuv420p",
        "stream_count": 1,
        "video_stream_count": 1,
        "audio_stream_count": 0,
        "full_decode_verified": True,
        "held_file_identity_stable_across_full_decode": True,
        "retained_fd_spans_full_decode": True,
        "pyav_opened_dup_of_retained_fd": True,
    }
    rows = []
    for index, (key, title, variant) in enumerate(decoder.REFERENCE_BRANCHES):
        digest = sha(f"media:{key}".encode())
        rows.append({
            "key": key,
            "label": title,
            "kind": "registered_simulator_reference",
            "q_condition": f"simulator {variant}; not model output",
            "relative_path": f"{index:02d}_{key}.mp4",
            "sha256": digest,
            "size": index + 1,
            "create_only_copy": True,
            "source_sha256": digest,
            "retained_fd_pre_post_sha256": digest,
            **probe_ref,
        })
    for index, (key, title, step, q_key) in enumerate(decoder.GENERATED_BRANCHES, start=4):
        rows.append({
            "key": key,
            "label": title,
            "kind": "real_bernini_generated_simulator_conditioned",
            "q_condition": (
                "q ignored: frozen base has no ELAL route"
                if q_key is None
                else f"teacher-forced simulator oracle q={q_key}"
            ),
            "checkpoint_step": step,
            "relative_path": f"{index:02d}_{key}.mp4",
            "sha256": sha(f"media:{key}".encode()),
            "size": index + 1,
            "create_only_generated_video": True,
            "retained_fd_pre_post_sha256": sha(f"media:{key}".encode()),
            **{**probe_ref, "height": 416, "width": 560},
            "branch_receipt": generated_receipt(key, step, q_key, seed=seed),
        })
    return rows


class ELAL3C2DecoderTests(unittest.TestCase):
    def test_source_parses_and_contains_origin_physical_closure(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        ast.parse(source)
        for marker in (
            "WORLD_SIZE = 4",
            "SP_SIZE = 4",
            "FRAME_COUNT = 81",
            "FPS = 25.0",
            "def validate_training_and_origin_v1",
            "trainer._validate_exact10_receipt_value_v1",
            "trainer.validate_checkpoint_record_v1",
            "trainer.seal_and_validate_checkpoint_tree_v1",
            '"origin_holder_physical_checkpoint_replay_required": True',
            '"login_node_checkpoint_dereference_forbidden": True',
            '"same_sampling_noise_for_all_matched_comparisons": True',
            "trainer.validate_checkpoint_exact23_world8_v1",
            "trainer.validate_bernini_execution_sources_world8_v1",
            "trainer.replay_strong_model_authority_world8_v1",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("scp", source)
        self.assertNotIn("srun", source)

    def test_exact14_branch_order_and_exact10_generated_interventions(self) -> None:
        self.assertEqual(len(decoder.REFERENCE_BRANCHES), 4)
        self.assertEqual(len(decoder.GENERATED_BRANCHES), 10)
        self.assertEqual(len(decoder.BRANCH_ORDER), 14)
        self.assertEqual(
            decoder.BRANCH_ORDER,
            (
                "source",
                "gt_target",
                "gt_role_swap",
                "appearance_anchor",
                "frozen_base",
                "step0_correct_q",
                "trained_correct_q",
                "trained_full_role_swap_q",
                "trained_role_only_mismatch_q",
                "trained_wrong_agent_q",
                "trained_wrong_object_q",
                "trained_zero_q",
                "trained_reverse_q",
                "trained_phase_shuffle_q",
            ),
        )
        generated_q = {row[0]: row[3] for row in decoder.GENERATED_BRANCHES}
        self.assertEqual(generated_q["trained_correct_q"], "target")
        self.assertEqual(generated_q["trained_full_role_swap_q"], "role_swap")
        self.assertEqual(
            generated_q["trained_role_only_mismatch_q"], "target_role_mismatch"
        )

    def test_world4_and_exact40_seed_contracts_fail_closed(self) -> None:
        contract = decoder.distributed_contract_v1(
            {
                "WORLD_SIZE": "4",
                "LOCAL_WORLD_SIZE": "4",
                "RANK": "2",
                "LOCAL_RANK": "2",
            }
        )
        self.assertEqual((contract.world_size, contract.rank, contract.local_rank), (4, 2, 2))
        for hostile in (
            {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "RANK": "2", "LOCAL_RANK": "2"},
            {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4", "RANK": "1", "LOCAL_RANK": "0"},
            {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"},
            {"WORLD_SIZE": "x", "LOCAL_WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"},
        ):
            with self.assertRaises(decoder.ELAL3C2DecodeError):
                decoder.distributed_contract_v1(hostile)
        sampling = decoder.sampler_contract_v1(steps=40, seed=20260821)
        self.assertEqual(sampling["num_frames"], 81)
        self.assertEqual(sampling["seed"], 20260821)
        with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "exact40"):
            decoder.sampler_contract_v1(steps=20, seed=20260821)

    def _release_value(self, source_shas, source_sizes):
        sources = [
            {
                "relative_path": name,
                "sha256": source_shas[name],
                "size": source_sizes[name],
                "archive_mode": 0o444,
            }
            for name in decoder.DECODE_SOURCE_ORDER
        ]
        runtime_sources = {}
        for name, (relative, digest, size) in decoder.RUNTIME_SOURCE_BINDINGS.items():
            runtime_sources[name] = {
                "relative_path": relative,
                "sha256": digest,
                "size": size,
                "mode": 0o444,
                "nlink": 1,
                "held_fd_double_hash_verified": True,
                "held_openat_parent_chain_replayed": True,
                "actual_imported_module_file_verified": True,
            }
        runtime_unsigned = {
            "source_count": decoder.RUNTIME_SOURCE_COUNT,
            "sources": runtime_sources,
            "all_modes": "0444",
            "all_nlink1_no_follow_held_openat_double_hash": True,
            "actual_imported_module_files_verified": True,
            "callable_ownership_verified": True,
            "runtime_absolute_paths_devices_inodes_excluded": True,
        }
        runtime_pins = {
            **runtime_unsigned,
            "release_pin_digest": decoder.object_sha256(runtime_unsigned),
        }
        origins = {
            arm: {
                "arm_id": arm,
                "holder_job_id": decoder.ARM_PLACEMENT[arm][0],
                "node": decoder.ARM_PLACEMENT[arm][1],
                "seed": decoder.ARM_PLACEMENT[arm][2],
                "status": "EXACT10_ORIGIN_PHYSICAL_REPLAY_PASS",
                "attestation_sha256": sha(f"{arm}:attestation".encode()),
                "attestation_digest": sha(f"{arm}:attestation-digest".encode()),
                "training_receipt_sha256": sha(f"{arm}:receipt".encode()),
                "training_receipt_digest": sha(f"{arm}:receipt-digest".encode()),
                "runner_source_sha256": decoder.FINAL_C2_TRAINER_SHA256,
                "portable_checkpoint_tree_digest": sha(f"{arm}:tree".encode()),
                "physical_checkpoint_path_embedded": False,
                "origin_holder_decode_required": True,
            }
            for arm in decoder.ARM_IDS
        }
        unsigned = {
            "schema_version": decoder.RELEASE_SCHEMA,
            "status": "FINAL_C2_SIMULATOR_ORACLE_Q_DECODE_RELEASE",
            "method": decoder.METHOD,
            "source_files": sources,
            "runtime_source_pins": runtime_pins,
            "exact10_origin_attestations": origins,
            "attestation_tool_bindings": decoder.ATTESTATION_TOOL_BINDINGS,
            "authority_bindings": decoder.AUTHORITY_BINDINGS,
            "decode_contract": {
                "row_ids": list(decoder.ROW_IDS),
                "arm_ids": list(decoder.ARM_IDS),
                "world_size": 4,
                "sequence_parallel_size": 4,
                "generated_branch_order": [row[0] for row in decoder.GENERATED_BRANCHES],
                "review_branch_order": list(decoder.BRANCH_ORDER),
                "reference_media_count": 4,
                "generated_media_count": 10,
                "review_media_count": 14,
                "frame_count": 81,
                "fps": 25.0,
                "latent_shape": list(decoder.LATENT_SHAPE),
                "bucket_hw": list(decoder.BUCKET_HW),
                "patch_grid": list(decoder.PATCH_GRID),
                "num_inference_steps": 40,
                "same_sampling_noise_for_all_matched_comparisons": True,
                "native_initial_sampling_noise_observed_not_injected": True,
                "origin_holder_physical_checkpoint_replay_required": True,
                "origin_checkpoint_root_cli_required": True,
                "portable_release_contains_checkpoint_path": False,
                "login_node_checkpoint_dereference_forbidden": True,
            },
            "claim_boundaries": dict(decoder.CLAIM_BOUNDARIES),
        }
        return {**unsigned, "manifest_digest": decoder.object_sha256(unsigned)}

    def test_decode_release_manifest_exact3_sources_and_origin_pin(self) -> None:
        source_shas = {
            "decode_elal3_c2_simulator_oracle_q_v1.py": "1" * 64,
            "decode_elal3_c1_simulator_oracle_q_v1.py": "2" * 64,
            "analyze_elal3_c2_decoded_role_effect_v1.py": "3" * 64,
        }
        source_sizes = {key: index + 10 for index, key in enumerate(source_shas)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "manifest.json"
            value = self._release_value(source_shas, source_sizes)
            raw = decoder.canonical_json_bytes(value) + b"\n"
            path.write_bytes(raw)
            release = decoder.validate_decode_release_v1(
                path,
                expected_sha256=sha(raw),
                arm_id=decoder.ARM_IDS[0],
                expected_decoder_sha256="1" * 64,
                expected_helper_sha256="2" * 64,
                expected_analyzer_sha256="3" * 64,
            )
            self.assertEqual(release.digest, value["manifest_digest"])
            tampered = json.loads(raw)
            tampered["decode_contract"]["world_size"] = 8
            unsigned = dict(tampered)
            unsigned.pop("manifest_digest")
            tampered["manifest_digest"] = decoder.object_sha256(unsigned)
            bad_raw = decoder.canonical_json_bytes(tampered) + b"\n"
            bad = root / "bad.json"
            bad.write_bytes(bad_raw)
            with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "semantic closure"):
                decoder.validate_decode_release_v1(
                    bad,
                    expected_sha256=sha(bad_raw),
                    arm_id=decoder.ARM_IDS[0],
                    expected_decoder_sha256="1" * 64,
                    expected_helper_sha256="2" * 64,
                    expected_analyzer_sha256="3" * 64,
                )

            reordered = copy.deepcopy(value)
            reordered["source_files"] = list(reversed(reordered["source_files"]))
            unsigned = dict(reordered)
            unsigned.pop("manifest_digest")
            reordered["manifest_digest"] = decoder.object_sha256(unsigned)
            reordered_raw = decoder.canonical_json_bytes(reordered) + b"\n"
            reordered_path = root / "reordered.json"
            reordered_path.write_bytes(reordered_raw)
            with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "source order"):
                decoder.validate_decode_release_v1(
                    reordered_path,
                    expected_sha256=sha(reordered_raw),
                    arm_id=decoder.ARM_IDS[0],
                    expected_decoder_sha256="1" * 64,
                    expected_helper_sha256="2" * 64,
                    expected_analyzer_sha256="3" * 64,
                )

            stale = copy.deepcopy(value)
            stale["exact10_origin_attestations"][decoder.ARM_IDS[0]][
                "runner_source_sha256"
            ] = "a" * 64
            unsigned = dict(stale)
            unsigned.pop("manifest_digest")
            stale["manifest_digest"] = decoder.object_sha256(unsigned)
            stale_raw = decoder.canonical_json_bytes(stale) + b"\n"
            stale_path = root / "stale.json"
            stale_path.write_bytes(stale_raw)
            with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "origin row"):
                decoder.validate_decode_release_v1(
                    stale_path,
                    expected_sha256=sha(stale_raw),
                    arm_id=decoder.ARM_IDS[0],
                    expected_decoder_sha256="1" * 64,
                    expected_helper_sha256="2" * 64,
                    expected_analyzer_sha256="3" * 64,
                )

            stale_runtime = copy.deepcopy(value)
            stale_runtime["runtime_source_pins"]["sources"]["c1_trainer"][
                "sha256"
            ] = "b" * 64
            runtime_unsigned = dict(stale_runtime["runtime_source_pins"])
            runtime_unsigned.pop("release_pin_digest")
            stale_runtime["runtime_source_pins"]["release_pin_digest"] = (
                decoder.object_sha256(runtime_unsigned)
            )
            unsigned = dict(stale_runtime)
            unsigned.pop("manifest_digest")
            stale_runtime["manifest_digest"] = decoder.object_sha256(unsigned)
            stale_runtime_raw = decoder.canonical_json_bytes(stale_runtime) + b"\n"
            stale_runtime_path = root / "stale-runtime.json"
            stale_runtime_path.write_bytes(stale_runtime_raw)
            with self.assertRaisesRegex(
                decoder.ELAL3C2DecodeError, "frozen runtime source pin"
            ):
                decoder.validate_decode_release_v1(
                    stale_runtime_path,
                    expected_sha256=sha(stale_runtime_raw),
                    arm_id=decoder.ARM_IDS[0],
                    expected_decoder_sha256="1" * 64,
                    expected_helper_sha256="2" * 64,
                    expected_analyzer_sha256="3" * 64,
                )

            v7_tools = copy.deepcopy(value)
            v7_tools["attestation_tool_bindings"]["origin_verifier_binding"][
                "sha256"
            ] = "468ba565aa778dd4b6002f8615808b716a046223b6b26878165bfafe4535d371"
            v7_tools["attestation_tool_bindings"]["gate_controller_binding"][
                "sha256"
            ] = "2a468be3ad06b7efa05e400e67340f33d720c17a1a883cbe1088ccf81cbeb2ca"
            unsigned = dict(v7_tools)
            unsigned.pop("manifest_digest")
            v7_tools["manifest_digest"] = decoder.object_sha256(unsigned)
            v7_tools_raw = decoder.canonical_json_bytes(v7_tools) + b"\n"
            v7_tools_path = root / "v7-tools.json"
            v7_tools_path.write_bytes(v7_tools_raw)
            with self.assertRaisesRegex(
                decoder.ELAL3C2DecodeError, "semantic closure"
            ):
                decoder.validate_decode_release_v1(
                    v7_tools_path,
                    expected_sha256=sha(v7_tools_raw),
                    arm_id=decoder.ARM_IDS[0],
                    expected_decoder_sha256="1" * 64,
                    expected_helper_sha256="2" * 64,
                    expected_analyzer_sha256="3" * 64,
                )

            v8_resigned = copy.deepcopy(value)
            v8_trainer = v8_resigned["runtime_source_pins"]["sources"][
                "c2_trainer"
            ]
            v8_trainer.update(
                {
                    "sha256": "190fdd12c2613b818bd7a85facf84f5f2a7d5747cc865aae3621618fc4ccdc80",
                    "size": 447_090,
                }
            )
            runtime_unsigned = dict(v8_resigned["runtime_source_pins"])
            runtime_unsigned.pop("release_pin_digest")
            v8_resigned["runtime_source_pins"]["release_pin_digest"] = (
                decoder.object_sha256(runtime_unsigned)
            )
            for row in v8_resigned["exact10_origin_attestations"].values():
                row["runner_source_sha256"] = v8_trainer["sha256"]
            v8_resigned["attestation_tool_bindings"]["origin_verifier_binding"][
                "sha256"
            ] = "6423a126904cfa8a604b87dc5770a6e029f6a5a7c2d6975ce2ce6ff26f397619"
            v8_resigned["attestation_tool_bindings"]["gate_controller_binding"][
                "sha256"
            ] = "8b30aa204b5eef56a180fb25d676e3b19f25f7eb7d281164354e388ba3a5e7f4"
            unsigned = dict(v8_resigned)
            unsigned.pop("manifest_digest")
            v8_resigned["manifest_digest"] = decoder.object_sha256(unsigned)
            v8_raw = decoder.canonical_json_bytes(v8_resigned) + b"\n"
            v8_path = root / "fully-resigned-v8.json"
            v8_path.write_bytes(v8_raw)
            with self.assertRaisesRegex(
                decoder.ELAL3C2DecodeError,
                "(frozen runtime source pin|semantic closure|origin row)",
            ):
                decoder.validate_decode_release_v1(
                    v8_path,
                    expected_sha256=sha(v8_raw),
                    arm_id=decoder.ARM_IDS[0],
                    expected_decoder_sha256="1" * 64,
                    expected_helper_sha256="2" * 64,
                    expected_analyzer_sha256="3" * 64,
                )

    def test_static_cli_requires_acks_arm_seed_exact40_and_fresh_output(self) -> None:
        digest = "1" * 64
        parser = decoder.parser()
        common = [
            "--arm-id", decoder.ARM_IDS[0],
            "--row-id", decoder.ROW_IDS[0],
            "--bernini-root", "/b",
            "--veomni-root", "/v",
            "--checkpoint", "/c",
            "--checkpoint-exact23-manifest", "/m23",
            "--runtime-root", "/runtime",
            "--decode-release-manifest", "/release.json",
            "--expected-decode-release-manifest-sha256", digest,
            "--helper-source", "/helper.py",
            "--expected-helper-source-sha256", digest,
            "--analyzer-source", "/analyzer.py",
            "--expected-analyzer-source-sha256", digest,
            "--expected-decoder-source-sha256", digest,
            "--training-receipt", "/training.json",
            "--expected-training-receipt-sha256", digest,
            "--exact10-origin-attestation", "/origin.json",
            "--expected-exact10-origin-attestation-sha256", digest,
            "--origin-checkpoint-root", "/tmp/origin-checkpoints",
            "--packet-root", "/packet",
            "--latent-bundle", "/bundle",
            "--latent-bundle-receipt", "/bundle.json",
            "--materializer-run-complete", "/run.json",
            "--experiment-contract", "/experiment.json",
            "--external-authority", "/authority.json",
            "--model-authority", "/model.json",
            "--output-root", "/tmp/elal3-c2-decoder-static-fresh",
            "--sampling-seed", "20260821",
            "--num-inference-steps", "40",
        ]
        acknowledgements = [
            "--ack-simulator-oracle-q-only",
            "--ack-origin-holder-physical-checkpoint-replay",
            "--ack-not-source-instruction-inference",
            "--ack-not-formal-c2",
            "--ack-not-exact160",
            "--ack-no-real-video-or-scientific-claim",
        ]
        args = parser.parse_args(common + acknowledgements)
        decoder.validate_static_args_v1(args)
        with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "acknowledgements"):
            decoder.validate_static_args_v1(parser.parse_args(common + acknowledgements[:-1]))
        wrong_seed = list(common)
        wrong_seed[wrong_seed.index("20260821")] = "20260822"
        with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "arm seed"):
            decoder.validate_static_args_v1(parser.parse_args(wrong_seed + acknowledgements))

    def test_q_branch_builder_binds_full_role_and_role_only_mismatch(self) -> None:
        class LabelModule:
            @staticmethod
            def build_role_only_hybrid_v1(target, role):
                self.assertIs(target, labels["target"])
                self.assertIs(role, labels["role_swap"])
                return SimpleNamespace(
                    latent="hybrid-latent",
                    receipt={"hybrid_digest": "f" * 64},
                )

        class Elal:
            @staticmethod
            def intervene_elal3_v1(latent, intervention):
                self.assertEqual((latent, intervention), ("target-latent", "zero"))
                return "zero-latent"

        labels = {
            variant: SimpleNamespace(
                latent=f"{variant}-latent",
                receipt={"label_digest": sha(variant.encode())},
            )
            for variant in (
                "target",
                "role_swap",
                "wrong_agent",
                "wrong_object",
                "reverse",
                "phase_shuffle",
            )
        }
        branches = decoder.build_q_branches_v1(
            labels=labels, label_module=LabelModule, elal_module=Elal
        )
        self.assertEqual(branches["target"]["latent"], "target-latent")
        self.assertEqual(branches["role_swap"]["latent"], "role_swap-latent")
        self.assertEqual(branches["target_role_mismatch"]["latent"], "hybrid-latent")
        self.assertTrue(
            branches["target_role_mismatch"]["only_q_entity_and_q_relation_changed"]
        )
        self.assertEqual(branches["zero_target"]["latent"], "zero-latent")

    def test_exact14_closed_branch_receipts_and_hostiles(self) -> None:
        rows = exact14_rows()
        decoder.validate_exact14_media_rows_v1(rows, sampling_seed=20260821)
        hostiles = []
        wrong_device = copy.deepcopy(rows)
        wrong_device[4]["branch_receipt"]["world4_rank_receipts"][2][
            "initial_sampling_noise"
        ]["requested_device"] = "cuda:999"
        hostiles.append((wrong_device, "rank2"))
        wrong_q = copy.deepcopy(rows)
        wrong_q[6]["branch_receipt"]["q_binding"]["q_source"] = "self-signed"
        wrong_q[6]["branch_receipt"]["world4_rank_receipts"][0]["q_binding"][
            "q_source"
        ] = "self-signed"
        hostiles.append((wrong_q, "q-binding"))
        wrong_hook = copy.deepcopy(rows)
        wrong_hook[6]["branch_receipt"]["elal_hook_audit"]["calls_by_block"]["0"] = 79
        hostiles.append((wrong_hook, "hook"))
        wrong_condition = copy.deepcopy(rows)
        wrong_condition[7]["q_condition"] = "teacher-forced simulator oracle q=target"
        hostiles.append((wrong_condition, "branch receipt"))
        wrong_reference = copy.deepcopy(rows)
        wrong_reference[0]["source_sha256"] = "a" * 64
        hostiles.append((wrong_reference, "reference media"))
        extra = copy.deepcopy(rows)
        extra[6]["branch_receipt"]["world4_rank_receipts"][0]["decoy"] = True
        hostiles.append((extra, "rank receipt"))
        for hostile, message in hostiles:
            with self.subTest(message=message):
                with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, message):
                    decoder.validate_exact14_media_rows_v1(
                        hostile, sampling_seed=20260821
                    )

    def test_preimport_exact12_source_authentication_and_mutation(self) -> None:
        names = tuple(decoder.RUNTIME_SOURCE_BINDINGS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sources = {}
            for index, name in enumerate(names):
                relative = decoder.RUNTIME_SOURCE_BINDINGS[name][0]
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b"" if name == "tools_package" else f"VALUE = {index}\n".encode()
                path.write_bytes(raw)
                path.chmod(0o444)
                sources[name] = {
                    "relative_path": relative,
                    "sha256": sha(raw),
                    "size": len(raw),
                }
            result = decoder.prevalidate_runtime_source_files_v1(
                method_root=root, source_pins={"sources": sources}
            )
            self.assertEqual(len(result), decoder.RUNTIME_SOURCE_COUNT)
            victim = root / decoder.RUNTIME_SOURCE_BINDINGS["c2_materializer"][0]
            victim.chmod(0o644)
            victim.write_text("VALUE = 999\n", encoding="ascii")
            victim.chmod(0o444)
            with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "held-file replay"):
                decoder.prevalidate_runtime_source_files_v1(
                    method_root=root, source_pins={"sources": sources}
                )

    def test_pinned_import_rejects_prepopulated_module_cache(self) -> None:
        name = "train_elal3_c2_simulator_role_pair_v1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            previous = sys.modules.get(name)
            sys.modules[name] = SimpleNamespace(__file__="/decoy.py")
            try:
                with self.assertRaisesRegex(
                    decoder.ELAL3C2DecodeError, "module cache"
                ):
                    decoder._import_from_method_root(root, lease=None)
            finally:
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    def test_runtime_source_import_lease_replays_full_identity_across_import(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "runtime.py"
            raw = b"VALUE = 1\n"
            path.write_bytes(raw)
            path.chmod(0o444)
            source_paths = {"runtime:test": (path, sha(raw), len(raw))}
            lease = decoder.RuntimeSourceImportLeaseV1.open(
                method_root=root, source_paths=source_paths
            )
            receipt = lease.verify_and_close()
            self.assertTrue(receipt["retained_method_root_fd_across_import"])
            self.assertTrue(receipt["retained_exact_source_fds_across_import"])

            hostile = decoder.RuntimeSourceImportLeaseV1.open(
                method_root=root, source_paths=source_paths
            )
            path.chmod(0o644)
            path.write_bytes(raw)
            path.chmod(0o444)
            with self.assertRaisesRegex(
                decoder.ELAL3C2DecodeError, "changed across import"
            ):
                hostile.verify_and_close()

    def test_exact12_loader_executes_held_sources_not_timestamp_valid_pyc(
        self,
    ) -> None:
        bodies = {
            "tools_package": b"",
            "tools_build_renderer_dataset": b"VALUE = 1\n",
            "tools_materialize_vae": (
                b"from tools import build_renderer_dataset as raw_builder\n"
            ),
            "elal3_core": b"VALUE = 1\n",
            "c2_label": (
                b"import elal3_c0_v1 as elal3\n"
                b"def load_oracle_q_label_v1(): return 1\n"
                b"def load_verified_c2_packet(): return 1\n"
            ),
            "train_lora": b"def validate_checkpoint(): return 1\n",
            "c2_materializer": (
                b"import elal3_simulator_c2_label_v1 as labels\n"
                b"import train_lora as legacy\n"
                b"from tools import materialize_vae\n"
                b"def verify_bundle_payload_v1(): return 1\n"
            ),
            "c1_trainer": b"VALUE = 1\n",
            "c2_trainer": (
                b"import train_elal3_c1_simulator_overfit_v1 as c1\n"
                b"def _validate_exact10_receipt_value_v1(): return 1\n"
                b"def validate_checkpoint_exact23_world8_v1(): return 1\n"
                b"def seal_and_validate_checkpoint_tree_v1(): return 1\n"
            ),
            "packed_lora": b"VALUE = 1\n",
            "world8_runtime": b"VALUE = 1\n",
            "sigma_strata": b"VALUE = 1\n",
        }
        self.assertEqual(set(bodies), set(decoder.RUNTIME_SOURCE_BINDINGS))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_rows = {}
            source_paths = {}
            for name, raw in bodies.items():
                relative = decoder.RUNTIME_SOURCE_BINDINGS[name][0]
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                path.chmod(0o444)
                source_rows[name] = {
                    "relative_path": relative,
                    "sha256": sha(raw),
                    "size": len(raw),
                }
                source_paths[f"runtime:{name}"] = (path, sha(raw), len(raw))

            sigma_path = root / decoder.RUNTIME_SOURCE_BINDINGS["sigma_strata"][0]
            sigma_path.chmod(0o644)
            sigma_path.write_bytes(b"VALUE = 9\n")
            malicious_stat = sigma_path.stat()
            cache_path = Path(importlib.util.cache_from_source(str(sigma_path)))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(str(sigma_path), cfile=str(cache_path), doraise=True)
            sigma_path.write_bytes(bodies["sigma_strata"])
            os.utime(
                sigma_path,
                ns=(malicious_stat.st_atime_ns, malicious_stat.st_mtime_ns),
            )
            sigma_path.chmod(0o444)
            sys.path.insert(0, str(root))
            try:
                imported_from_pyc = importlib.import_module(
                    "inference_sigma_strata"
                )
                self.assertEqual(imported_from_pyc.VALUE, 9)
            finally:
                sys.modules.pop("inference_sigma_strata", None)
                sys.path.remove(str(root))

            validated = decoder.prevalidate_runtime_source_files_v1(
                method_root=root, source_pins={"sources": source_rows}
            )
            self.assertEqual(validated, source_paths)
            lease = decoder.RuntimeSourceImportLeaseV1.open(
                method_root=root, source_paths=source_paths
            )
            module_names = [
                value[0] for value in decoder.RUNTIME_MODULE_BINDINGS.values()
            ]
            try:
                modules = decoder._import_from_method_root(root, lease=lease)
                self.assertEqual(modules["sigma"].VALUE, 1)
                receipt = lease.verify_and_close()
                self.assertTrue(
                    receipt["executed_exact_sources_from_held_fd_bytes"]
                )
                self.assertEqual(
                    receipt["runtime_source_count"], decoder.RUNTIME_SOURCE_COUNT
                )
            finally:
                lease.close()
                for module_name in module_names:
                    sys.modules.pop(module_name, None)

    def test_decode_source_replay_uses_exact3_control_mode_projection(self) -> None:
        class FakeDist:
            @staticmethod
            def all_gather_object(gathered, value):
                for rank in range(decoder.WORLD_SIZE):
                    gathered[rank] = {
                        "world_rank": rank,
                        "fixed_binding_digest": value["fixed_binding_digest"],
                    }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = {}
            names = (
                *sorted(decoder.DECODE_MUTABLE_CONTROL_SOURCE_NAMES),
                "runtime:c2_trainer",
            )
            for index, name in enumerate(names):
                raw = f"source-{index}\n".encode("ascii")
                path = root / f"source-{index}.bin"
                path.write_bytes(raw)
                mode = (
                    0o644
                    if name in decoder.DECODE_MUTABLE_CONTROL_SOURCE_NAMES
                    else 0o444
                )
                path.chmod(mode)
                paths[name] = (path, sha(raw), len(raw))
            replay = decoder.replay_decode_sources_world4_v1(
                paths=paths,
                distributed=SimpleNamespace(rank=0),
                dist=FakeDist(),
                stage="pre_load",
            )
            observed = {
                row["name"]: row["mode"]
                for row in replay["fixed_binding"]["sources"]
            }
            self.assertEqual(
                {
                    observed[name]
                    for name in decoder.DECODE_MUTABLE_CONTROL_SOURCE_NAMES
                },
                {0o644},
            )
            self.assertEqual(observed["runtime:c2_trainer"], 0o444)
            victim = paths["artifact:model_authority"][0]
            victim.chmod(0o444)
            with self.assertRaisesRegex(
                decoder.ELAL3C2DecodeError, "source size/mode"
            ):
                decoder.replay_decode_sources_world4_v1(
                    paths=paths,
                    distributed=SimpleNamespace(rank=0),
                    dist=FakeDist(),
                    stage="pre_load",
                )

    def test_model_authority_replay_projects_trainer_world8_to_actual_world4(self) -> None:
        trainer_value = {
            "stage": "post_deserialize",
            "authority_sha256": decoder.MODEL_AUTHORITY_SHA256,
            "authority_digest": decoder.MODEL_AUTHORITY_DIGEST,
            "strong_replay_digest": "4" * 64,
            "exact9_held_openat_replayed": True,
            "actual_imported_modules_and_callable_ownership_replayed": True,
            "world8_broadcast_identity_verified": True,
        }
        projected = decoder.project_strong_model_authority_world4_v1(
            trainer_value, expected_stage="post_deserialize"
        )
        self.assertTrue(projected["world4_broadcast_identity_verified"])
        self.assertTrue(projected["trainer_world8_claim_not_republished"])
        self.assertNotIn("world8_broadcast_identity_verified", projected)
        for hostile in (
            {**trainer_value, "world8_broadcast_identity_verified": False},
            {**trainer_value, "world4_broadcast_identity_verified": True},
        ):
            with self.assertRaisesRegex(
                decoder.ELAL3C2DecodeError, "strong-model replay closure"
            ):
                decoder.project_strong_model_authority_world4_v1(
                    hostile, expected_stage="post_deserialize"
                )

    def test_release_helper_executes_held_bytes_and_rejects_import_time_swap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "helper.py"
            raw = b"VALUE = 7\n"
            path.write_bytes(raw)
            path.chmod(0o444)
            row = {"sha256": sha(raw), "size": len(raw)}
            module_name = "unit_c2_held_helper"
            module, binding = decoder.load_release_python_source_from_held_bytes_v1(
                path,
                row=row,
                module_name=module_name,
                label="unit held helper",
            )
            self.assertEqual(module.VALUE, 7)
            self.assertEqual(binding["sha256"], sha(raw))
            sys.modules.pop(module_name, None)

            original_compile = compile
            swapped = False

            def swap_during_compile(source, filename, mode, **kwargs):
                nonlocal swapped
                if not swapped and filename == str(path):
                    swapped = True
                    path.chmod(0o644)
                    path.write_bytes(raw)
                    path.chmod(0o444)
                return original_compile(source, filename, mode, **kwargs)

            with mock.patch("builtins.compile", side_effect=swap_during_compile):
                with self.assertRaisesRegex(
                    decoder.ELAL3C2DecodeError,
                    "changed across held-byte execution",
                ):
                    decoder.load_release_python_source_from_held_bytes_v1(
                        path,
                        row=row,
                        module_name=module_name,
                        label="unit swapped held helper",
                    )
            self.assertNotIn(module_name, sys.modules)

    def test_video_probe_decodes_dup_of_one_retained_fd_and_rejects_path_swap(
        self,
    ) -> None:
        class Frame:
            width = 8
            height = 6
            format = SimpleNamespace(name="yuv420p")

            @staticmethod
            def to_ndarray(*, format):
                self.assertEqual(format, "rgb24")
                return None

        class Streams(list):
            def __init__(self):
                stream = SimpleNamespace(
                    average_rate=SimpleNamespace(numerator=25, denominator=1)
                )
                super().__init__([stream])
                self.video = [stream]
                self.audio = []

        class Container:
            streams = Streams()

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            @staticmethod
            def decode(*, video):
                self.assertEqual(video, 0)
                return [Frame() for _ in range(decoder.FRAME_COUNT)]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "video.mp4"
            raw = b"held-video-fixture"
            path.write_bytes(raw)
            opened = []

            def open_retained(stream, *, mode):
                self.assertFalse(isinstance(stream, (str, Path)))
                self.assertEqual(mode, "r")
                opened.append(stream)
                return Container()

            previous = sys.modules.get("av")
            sys.modules["av"] = SimpleNamespace(open=open_retained)
            try:
                receipt = decoder.probe_exact_video_v1(
                    path, expected_hw=(6, 8)
                )
                self.assertTrue(receipt["retained_fd_spans_full_decode"])
                self.assertTrue(receipt["pyav_opened_dup_of_retained_fd"])
                self.assertEqual(receipt["retained_fd_pre_post_sha256"], sha(raw))
                self.assertEqual(len(opened), 1)

                def swap_path_during_open(stream, *, mode):
                    path.unlink()
                    path.write_bytes(raw)
                    return Container()

                sys.modules["av"] = SimpleNamespace(open=swap_path_during_open)
                with self.assertRaisesRegex(
                    decoder.ELAL3C2DecodeError, "not exact81"
                ):
                    decoder.probe_exact_video_v1(path, expected_hw=(6, 8))
            finally:
                if previous is None:
                    sys.modules.pop("av", None)
                else:
                    sys.modules["av"] = previous

    def test_portable_checkpoint_and_fixed_replay_strip_physical_telemetry(self) -> None:
        origin = Path("/tmp/c2-origin-exact2")
        records = [{"step": 0}, {"step": 10}]
        portable = {
            "schema_version": "bernini-elal3-c2-sealed-checkpoint-tree-v1",
            "expected_steps": [0, 10],
            "directory_entries": ["checkpoint-00000000", "checkpoint-00000010"],
            "directory_mode": 0o500,
            "portable_checkpoint_records": records,
            "portable_checkpoint_tree_digest": decoder.object_sha256(records),
            "physical_origin_replay_passed": True,
        }
        full = {
            **portable,
            "origin_path": str(origin),
            "origin_device": 41,
            "origin_inode": 73,
            "tree_binding_digest": decoder.object_sha256(portable),
        }
        projected = decoder.portable_checkpoint_tree_replay_v1(
            full, expected_origin_root=origin, label="test"
        )
        self.assertEqual(projected, portable)
        self.assertFalse(any(key.startswith("origin_") for key in projected))
        hostile = copy.deepcopy(full)
        hostile["origin_path"] = "/tmp/foreign"
        with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "full origin"):
            decoder.portable_checkpoint_tree_replay_v1(
                hostile, expected_origin_root=origin, label="test"
            )

        fixed = {"file_count": 23, "files": [{"sha256": "a" * 64}]}
        physical = {
            "stage": "pre_load",
            "fixed_release_binding": fixed,
            "fixed_release_binding_digest": decoder.object_sha256(fixed),
            "runtime_telemetry": {
                "path": "/private/holder/model",
                "device": 9,
                "inode": 10,
            },
        }
        fixed_portable = decoder.portable_fixed_release_replay_v1(
            physical, stage="pre_load", label="test fixed"
        )
        self.assertEqual(
            set(fixed_portable),
            {
                "stage",
                "fixed_release_binding",
                "fixed_release_binding_digest",
                "physical_runtime_replay_passed",
            },
        )
        self.assertNotIn("runtime_telemetry", fixed_portable)
        resigned = copy.deepcopy(physical)
        resigned["fixed_release_binding_digest"] = "b" * 64
        with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "closure"):
            decoder.portable_fixed_release_replay_v1(
                resigned, stage="pre_load", label="test fixed"
            )

    def test_origin_checkpoint_exact2_retained_descriptor_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            records = []
            for step in (0, 10):
                child = root / f"checkpoint-{step:08d}"
                child.mkdir(mode=0o700)
                child.chmod(0o500)
                records.append({
                    "step": step,
                    "path": str(child),
                    "directory_entries": [],
                })
            root.chmod(0o500)
            lease = decoder.OriginCheckpointLeaseV1.open(root, records=records)
            replay = lease.snapshot(
                stage="before_step0_reload", reference=lease.initial_snapshot
            )
            self.assertEqual(
                replay["fixed_identity_digest"],
                lease.initial_snapshot["fixed_identity_digest"],
            )
            (root / "checkpoint-00000010").chmod(0o700)
            with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "identity changed"):
                lease.snapshot(
                    stage="before_step10_reload", reference=lease.initial_snapshot
                )
            (root / "checkpoint-00000010").chmod(0o500)
            lease.close()
            with self.assertRaisesRegex(decoder.ELAL3C2DecodeError, "closed twice"):
                lease.close()


if __name__ == "__main__":
    unittest.main()
