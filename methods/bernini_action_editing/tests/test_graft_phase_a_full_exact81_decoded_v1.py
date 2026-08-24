#!/usr/bin/env python3

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import graft_phase_a_full_exact81_decoded_v1 as core


SHA = "a" * 64
SHA_B = "b" * 64
HEIGHT = 480
WIDTH = 496
LATENT_SHAPE = [1, 16, 21, HEIGHT // 8, WIDTH // 8]


def tensor_identity(tag: int, *, shape=LATENT_SHAPE, dtype="torch.float32"):
    digest = f"{tag:064x}"[-64:]
    elements = 1
    for item in shape:
        elements *= item
    size = 4 if dtype == "torch.float32" else 2
    return {
        "shape": list(shape),
        "dtype": dtype,
        "device_type_at_observation": "cuda",
        "finite": True,
        "byte_count": elements * size,
        "raw_sha256": digest,
        "content_sha256": digest,
    }


def route(index: int):
    sigma = 0.8 if index < 26 else 0.5
    gate = 0.0 if index < 26 else 0.5
    value = {
        "schema_version": "bernini-graft-phase-a-full-exact81-route-projection-v1",
        "schedule_index": index,
        "branch_name": "V",
        "total_tokens": 160,
        "condition_tokens": 32,
        "target_tokens": 128,
        "sequence_parallel_size": 4,
        "sigma_hex": sigma.hex(),
        "gate_hex": gate.hex(),
        "atlas_receipt_digest": SHA,
        "source_memory_owned_by_V_VI_only": True,
        "enabled": True,
        "all_sp4_ranks_apply_same_global_route": True,
        "local_rank_validated_before_projection": True,
        "rank_specific_receipt_digest_not_cross_rank_comparable": True,
    }
    return {**value, "digest": core.object_sha256(value)}


def trace():
    states = [tensor_identity(index + 1) for index in range(41)]
    rows = []
    for index in range(40):
        packet = route(index)
        rows.append(
            {
                "schedule_index": index,
                "timestep": 1000 - index,
                "sigma_float32_be_hex": (
                    "3f4ccccd" if index < 26 else "3f000000"
                ),
                "state_before": states[index],
                "native_visual_pack": tensor_identity(
                    100 + index,
                    shape=[1, 100, 1536],
                    dtype="torch.bfloat16",
                ),
                "native_rotary_pack": tensor_identity(
                    200 + index,
                    shape=[1, 1, 100, 64],
                    dtype="torch.bfloat16",
                ),
                "negative_raw": tensor_identity(
                    300 + index,
                    shape=[1, 100, 64],
                    dtype="torch.bfloat16",
                ),
                "action_raw": tensor_identity(
                    400 + index,
                    shape=[1, 100, 64],
                    dtype="torch.bfloat16",
                ),
                "guided_velocity": tensor_identity(500 + index),
                "state_after": states[index + 1],
                "route_receipts": {"negative": packet, "action": packet},
                "scheduler_step_call_count": 1,
                "source_conditioned": True,
                "action_positive_condition": True,
                "target_video_used": False,
                "clean_source_initial_latent_used": False,
            }
        )
    return core.seal_mapping(
        {
            "schema_version": core.TRACE_SCHEMA_VERSION,
            "rows": rows,
            "official_unipc_step_count": 40,
            "initial_state_role": "fresh-source-keyed-standard-gaussian",
            "source_condition_role": "full-confirmation-source-v-pack",
            "positive_condition_role": "preregistered-action-text",
            "same_gaussian_seed_across_sp4": True,
            "cross_index_selection_used": False,
            "checkpoint_loaded_from_dependency": False,
            "checkpoint_written": False,
        }
    )


def media():
    return {
        "schema_version": core.MEDIA_SCHEMA_VERSION,
        "frame_count": 81,
        "fps_numerator": 25,
        "fps_denominator": 1,
        "reported_fps_numerator": 25,
        "reported_fps_denominator": 1,
        "height": HEIGHT,
        "width": WIDTH,
        "decoded_tensor_shape": [81, HEIGHT, WIDTH, 3],
        "codec_content_interpreted_for_semantics": False,
    }


def artifact(family: str, role: str, name: str, endpoint):
    if role == "normalized-clean-latent":
        content_binding = {
            "kind": "safetensors-exact-endpoint-tensor",
            "tensor_key": "normalized_clean_latent",
            "tensor_shape": endpoint["shape"],
            "tensor_dtype": endpoint["dtype"],
            "tensor_raw_sha256": endpoint["raw_sha256"],
            "tensor_content_sha256": endpoint["content_sha256"],
            "endpoint_raw_sha256": endpoint["raw_sha256"],
            "endpoint_content_sha256": endpoint["content_sha256"],
            "safetensors_roundtrip_verified": True,
        }
    else:
        content_binding = {
            "kind": "same-call-vae-decode-from-sealed-endpoint",
            "decoded_from_endpoint_raw_sha256": endpoint["raw_sha256"],
            "decoded_from_endpoint_content_sha256": endpoint["content_sha256"],
            "endpoint_unchanged_after_decode": True,
            "semantic_content_interpreted": False,
        }
    return {
        "schema_version": core.ARTIFACT_SCHEMA_VERSION,
        "role": role,
        "path": f"/sealed/run/{family}/{name}",
        "relative_path": f"{family}/{name}",
        "size_bytes": 100,
        "mode": "0444",
        "regular_file": True,
        "link_count_one": True,
        "sha256": SHA,
        "opened_nofollow_and_revalidated": True,
        "content_binding": content_binding,
    }


def local_result(rank: int, family_trace=None, *, rank_local_upstream=False):
    arm = rank // 4
    family = core.FAMILY_ORDER[arm]
    selected_trace = family_trace or trace()
    return core.build_local_result(
        global_rank=rank,
        dp_arm=arm,
        family=family,
        confirmation_iid=("dog-confirm" if arm == 0 else "human-confirm"),
        source_sha256=SHA,
        action_prompt_sha256=SHA_B,
        seed=2308110001 + arm,
        short_receipt_digest=(f"{rank + 1000:064x}" if rank_local_upstream else SHA),
        field14_receipt_digest=(f"{rank + 2000:064x}" if rank_local_upstream else SHA),
        active14_precommit_digest=(
            f"{rank + 3000:064x}" if rank_local_upstream else SHA
        ),
        initial_gaussian_identity=selected_trace["rows"][0]["state_before"],
        endpoint_identity=selected_trace["rows"][-1]["state_after"],
        exact40_trace=selected_trace,
        media=media(),
        latent_artifact=artifact(
            family,
            "normalized-clean-latent",
            "endpoint.safetensors",
            selected_trace["rows"][-1]["state_after"],
        ),
        video_artifact=artifact(
            family,
            "decoded-exact81-video",
            "decoded-exact81.mp4",
            selected_trace["rows"][-1]["state_after"],
        ),
        output_root="/sealed/run",
        expected_height=HEIGHT,
        expected_width=WIDTH,
        trainable_sha256_before_decode=SHA,
        trainable_sha256_after_decode=SHA,
        base_sha256_before_decode=SHA_B,
        base_sha256_after_decode=SHA_B,
    )


class FullExact81CoreTests(unittest.TestCase):
    def test_local_result_and_world8_close_without_authority(self):
        dog_trace = trace()
        human_trace = trace()
        packets = [
            local_result(rank, dog_trace if rank < 4 else human_trace)
            for rank in range(8)
        ]
        result = core.assemble_world8_result(packets)
        self.assertTrue(result["both_exact81_decoded"])
        self.assertTrue(result["both_sp4_arms_exact"])
        self.assertFalse(result["action_authority"])
        self.assertFalse(result["identity_authority"])
        self.assertFalse(result["quality_authority"])
        self.assertFalse(result["scientific_success_claimed"])

    def test_world8_binds_rank_local_upstream_receipts_without_false_sp4_equality(self):
        packets = [
            local_result(rank, rank_local_upstream=True) for rank in range(8)
        ]
        result = core.assemble_world8_result(packets)
        rows = result["rank_local_upstream_receipt_bindings"]
        self.assertEqual([row["global_rank"] for row in rows], list(range(8)))
        self.assertEqual(
            result["rank_local_upstream_receipt_bindings_digest"],
            core.object_sha256(rows),
        )
        self.assertEqual(len({row["short"] for row in rows}), 8)

    def test_trace_rejects_discontinuous_state(self):
        value = dict(trace())
        value.pop("digest")
        value["rows"] = copy.deepcopy(value["rows"])
        value["rows"][7]["state_before"] = tensor_identity(9999)
        with self.assertRaisesRegex(core.FullExact81ContractError, "discontinuous"):
            core.validate_exact40_trace(core.seal_mapping(value))

    def test_trace_rejects_nonzero_gate_at_inactive_index(self):
        value = dict(trace())
        value.pop("digest")
        value["rows"] = copy.deepcopy(value["rows"])
        bad = route(26)
        value["rows"][0]["route_receipts"] = {
            "negative": bad,
            "action": bad,
        }
        with self.assertRaisesRegex(core.FullExact81ContractError, "route differs"):
            core.validate_exact40_trace(core.seal_mapping(value))

    def test_trace_rejects_rank_specific_route_receipt(self):
        value = dict(trace())
        value.pop("digest")
        value["rows"] = copy.deepcopy(value["rows"])
        bad = dict(value["rows"][0]["route_receipts"]["negative"])
        bad.pop("digest")
        bad["sequence_parallel_rank"] = 0
        bad["digest"] = core.object_sha256(
            {key: item for key, item in bad.items() if key != "digest"}
        )
        value["rows"][0]["route_receipts"] = {
            "negative": bad,
            "action": bad,
        }
        with self.assertRaisesRegex(core.FullExact81ContractError, "route differs"):
            core.validate_exact40_trace(core.seal_mapping(value))

    def test_media_rejects_wrong_frame_count(self):
        value = media()
        value["frame_count"] = 80
        with self.assertRaisesRegex(core.FullExact81ContractError, "media record differs"):
            core.validate_media_record(
                value, expected_height=HEIGHT, expected_width=WIDTH
            )

    def test_local_result_rejects_latent_not_bound_to_endpoint_bytes(self):
        selected_trace = trace()
        endpoint = selected_trace["rows"][-1]["state_after"]
        latent = artifact(
            "dog", "normalized-clean-latent", "endpoint.safetensors", endpoint
        )
        latent["content_binding"]["tensor_raw_sha256"] = SHA_B
        with self.assertRaisesRegex(
            core.FullExact81ContractError, "latent content binding differs"
        ):
            core.build_local_result(
                global_rank=0,
                dp_arm=0,
                family="dog",
                confirmation_iid="dog-confirm",
                source_sha256=SHA,
                action_prompt_sha256=SHA_B,
                seed=2308110001,
                short_receipt_digest=SHA,
                field14_receipt_digest=SHA,
                active14_precommit_digest=SHA,
                initial_gaussian_identity=selected_trace["rows"][0]["state_before"],
                endpoint_identity=endpoint,
                exact40_trace=selected_trace,
                media=media(),
                latent_artifact=latent,
                video_artifact=artifact(
                    "dog", "decoded-exact81-video", "decoded-exact81.mp4", endpoint
                ),
                output_root="/sealed/run",
                expected_height=HEIGHT,
                expected_width=WIDTH,
                trainable_sha256_before_decode=SHA,
                trainable_sha256_after_decode=SHA,
                base_sha256_before_decode=SHA_B,
                base_sha256_after_decode=SHA_B,
            )

    def test_local_result_rejects_parameter_change(self):
        selected_trace = trace()
        with self.assertRaisesRegex(core.FullExact81ContractError, "parameter bytes"):
            core.build_local_result(
                global_rank=0,
                dp_arm=0,
                family="dog",
                confirmation_iid="dog-confirm",
                source_sha256=SHA,
                action_prompt_sha256=SHA_B,
                seed=2308110001,
                short_receipt_digest=SHA,
                field14_receipt_digest=SHA,
                active14_precommit_digest=SHA,
                initial_gaussian_identity=selected_trace["rows"][0]["state_before"],
                endpoint_identity=selected_trace["rows"][-1]["state_after"],
                exact40_trace=selected_trace,
                media=media(),
                latent_artifact=artifact(
                    "dog",
                    "normalized-clean-latent",
                    "endpoint.safetensors",
                    selected_trace["rows"][-1]["state_after"],
                ),
                video_artifact=artifact(
                    "dog",
                    "decoded-exact81-video",
                    "decoded-exact81.mp4",
                    selected_trace["rows"][-1]["state_after"],
                ),
                output_root="/sealed/run",
                expected_height=HEIGHT,
                expected_width=WIDTH,
                trainable_sha256_before_decode=SHA,
                trainable_sha256_after_decode=SHA_B,
                base_sha256_before_decode=SHA,
                base_sha256_after_decode=SHA,
            )

    def test_world8_rejects_one_sp_rank_artifact_disagreement(self):
        dog_trace = trace()
        human_trace = trace()
        packets = [
            local_result(rank, dog_trace if rank < 4 else human_trace)
            for rank in range(8)
        ]
        changed = dict(packets[2])
        changed.pop("digest")
        changed["artifacts"] = copy.deepcopy(changed["artifacts"])
        changed["artifacts"]["video"]["sha256"] = SHA_B
        packets[2] = core.seal_mapping(changed)
        with self.assertRaisesRegex(core.FullExact81ContractError, "SP4 result"):
            core.assemble_world8_result(packets)

    def test_authority_walker_rejects_nested_true(self):
        core.assert_no_elevated_authority({"authority": core.false_authority()})
        with self.assertRaisesRegex(core.FullExact81ContractError, "elevated"):
            core.assert_no_elevated_authority(
                {"nested": {"semantic_action_editing_success_claimed": True}}
            )
        malformed = core.false_authority()
        malformed.pop("quality_authority")
        with self.assertRaisesRegex(core.FullExact81ContractError, "elevated"):
            core.assert_no_elevated_authority({"authority": malformed})


if __name__ == "__main__":
    unittest.main()
