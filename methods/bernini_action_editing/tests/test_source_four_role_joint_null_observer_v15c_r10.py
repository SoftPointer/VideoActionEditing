#!/usr/bin/env python3
"""Registry, broadcast, SP4-repeat, and strict-loader tests for v15c-r10."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
RELEASE_PATH = (
    METHOD_ROOT
    / "assets/e00_source_four_role_joint_null_observer_v15c_r10_release.json"
)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_four_role_joint_null_observer_v15c_r10 as M  # noqa: E402
import source_role_authority_v15c_r9 as r9  # noqa: E402
import validate_source_four_role_joint_null_observer_v15c_r10 as V  # noqa: E402


def r9_validation_fixture():
    return {
        "schema_version": M.SCHEMA_VERSION,
        "validation_sha256": "a" * 64,
        "capture_channel_registry_sha256": M.CAPTURE_CHANNEL_REGISTRY_SHA256,
        "capture_channel_value_binding_sha256": "b" * 64,
        "independent_capture_channel_value_binding_pinned": False,
        "actual_sp4_rank_shard_files_replayed": False,
        "official_r10_runner_present": False,
        "role_assignment_mechanical_candidate_qualified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }


def make_arrays():
    real = np.zeros(M.REAL_SHAPE, dtype=np.float32)
    shuffled = np.zeros(M.REAL_SHAPE, dtype=np.float32)
    null = np.empty(M.NULL_SHAPE, dtype=np.float32)
    # Values are intentionally role-distinct.  No scientific meaning is
    # claimed; this is a mechanical consumer fixture only.
    for block in range(len(M.SELECTED_BLOCK_INDICES)):
        real[block].fill(np.float32(block + 1))
        shuffled[block].fill(np.float32(-(block + 1)))
        for role in range(len(M.ROLE_NAMES)):
            null[block, role].fill(np.float32(100 * block + 10 * role + 1))
            null[block, role, :, 0, 0, 0] = (
                np.arange(M.NULL_COUNT, dtype=np.float32) + np.float32(role / 8.0)
            )
    return real, shuffled, null


def make_transcript(repeat_real, repeat_shuffled, repeat_null):
    blocks = []
    for block_position, block_index in enumerate(M.SELECTED_BLOCK_INDICES):
        repeats = []
        for repeat_index in range(M.REPEAT_COUNT):
            real = repeat_real[repeat_index]
            shuffled = repeat_shuffled[repeat_index]
            null = repeat_null[repeat_index]
            assembled = {
                "real_array_sha256": M.array_sha256(real[block_position]),
                "shuffled_array_sha256": M.array_sha256(shuffled[block_position]),
                "role_null_array_sha256": [
                    M.array_sha256(null[block_position, role])
                    for role in range(len(M.ROLE_NAMES))
                ],
            }
            pass_label = f"source-observer-pass-{repeat_index}"
            ranks = []
            for rank in range(M.SP_SIZE):
                start = rank * M.PADDED_LOCAL_TOKENS
                stop = min(M.GLOBAL_VISUAL_TOKENS, start + M.PADDED_LOCAL_TOKENS)
                collective = M._expected_padded_collective_tensor(  # noqa: SLF001
                    real_block=np.ascontiguousarray(real[block_position]),
                    shuffled_block=np.ascontiguousarray(shuffled[block_position]),
                    null_block=np.ascontiguousarray(null[block_position]),
                    rank=rank,
                )
                row = {
                    "sp_rank": rank,
                    "global_start": start,
                    "global_stop": stop,
                    "padded_local_tokens": M.PADDED_LOCAL_TOKENS,
                    "valid_local_tokens": stop - start,
                    "collective_channel_count": M.COLLECTIVE_CHANNEL_COUNT,
                    "collective_tensor_sha256": M.array_sha256(collective),
                    "padded_collective_shape": [
                        M.COLLECTIVE_CHANNEL_COUNT, M.PADDED_LOCAL_TOKENS
                    ],
                    "append_padding_all_zero": True,
                    "channel_layout_sha256": M.object_sha256(
                        M.COLLECTIVE_CHANNEL_LAYOUT
                    ),
                    "capture_channel_registry_sha256": (
                        M.CAPTURE_CHANNEL_REGISTRY_SHA256
                    ),
                }
                metadata_payload = {
                    "block_index": block_index,
                    "repeat_index": repeat_index,
                    "capture_pass_label": pass_label,
                    "registry_sha256": M.REGISTRY_SHA256,
                    **row,
                }
                ranks.append(
                    {**row, "metadata_sha256": M.object_sha256(metadata_payload)}
                )
            repeat_payload = {
                "repeat_index": repeat_index,
                "capture_pass_label": pass_label,
                "ranks": ranks,
                "assembled": assembled,
            }
            repeats.append(
                {
                    **repeat_payload,
                    "repeat_receipt_sha256": M.object_sha256(repeat_payload),
                }
            )
        blocks.append(
            {
                "block_index": block_index,
                "repeats": repeats,
                "repeat_bit_exact": True,
            }
        )
    payload = {
        "blocks": blocks,
        "required_separate_repeat_artifacts": M.REPEAT_COUNT,
        "repeat_equality": "bit_exact",
        "implicit_collective_calls_inside_attn2": 0,
        "collective_channel_layout": M.COLLECTIVE_CHANNEL_LAYOUT,
        "capture_channel_registry_sha256": M.CAPTURE_CHANNEL_REGISTRY_SHA256,
        "append_padding_value": 0.0,
        "reconstruction_scope": (
            "hypothetical_global_array_to_SP4_shard_reconstruction_only"
        ),
        "actual_rank_shard_files_consumed": False,
        "producer_process_independence_verified": False,
    }
    return {**payload, "transcript_sha256": M.object_sha256(payload)}


def make_receipt(repeat_real, repeat_shuffled, repeat_null, *, file_rows=None):
    if file_rows is None:
        file_rows = {}
        for repeat_index in range(M.REPEAT_COUNT):
            arrays = {
                "real": repeat_real[repeat_index],
                "shuffled": repeat_shuffled[repeat_index],
                "joint_null": repeat_null[repeat_index],
            }
            rows = {}
            for position, (name, array) in enumerate(arrays.items(), start=1):
                row = {
                    "filename": f"repeat_{repeat_index}_{name}_affinity.npy",
                    "shape": list(M.REAL_SHAPE if name != "joint_null" else M.NULL_SHAPE),
                    "dtype": "float32",
                    "file_size": 1,
                    "file_sha256": str(position + repeat_index * 3) * 64,
                    "array_sha256": M.array_sha256(array),
                    "capture_repeat_index": repeat_index,
                }
                if name == "joint_null":
                    row["role_array_sha256"] = [
                        M.array_sha256(array[:, role])
                        for role in range(len(M.ROLE_NAMES))
                    ]
                rows[name] = row
            file_rows[f"repeat_{repeat_index}"] = rows
    registry = M.load_joint_null_registry_v15c_r10()
    channel_binding = M.capture_channel_value_binding_payload_v15c_r10(
        registry=registry,
        repeat_real=repeat_real,
        repeat_shuffled=repeat_shuffled,
        repeat_null_bank=repeat_null,
    )
    payload = {
        "schema_version": M.ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "status": "REAL_SOURCE_OBSERVER_TENSOR_LOCAL_CONTRACT_PASS_ONLY",
        "event_id": "pour-liquid-into-cup",
        "source_video_sha256": M.SOURCE_VIDEO_SHA256,
        "source_text_provenance_sha256": M.SOURCE_TEXT_PROVENANCE_SHA256,
        "registry_sha256": M.REGISTRY_SHA256,
        "role_control_registry_sha256": list(M.ROLE_CONTROL_REGISTRY_SHA256),
        "joint_index_registry_sha256": M.JOINT_INDEX_REGISTRY_SHA256,
        "capture_channel_registry_sha256": M.CAPTURE_CHANNEL_REGISTRY_SHA256,
        "capture_channel_value_binding_sha256": M.object_sha256(channel_binding),
        "selected_block_indices": list(M.SELECTED_BLOCK_INDICES),
        "role_names": list(M.ROLE_NAMES),
        "null_index_alignment": (
            "same_control_index_j_is_alignment_only_across_blocks_phases_"
            "proposals_with_distinct_role_controls"
        ),
        "tensor_files": file_rows,
        "sp4_repeat_transcript": make_transcript(
            repeat_real, repeat_shuffled, repeat_null
        ),
        "source_only": True,
        "frozen_model": True,
        "eval_mode": True,
        "all_adapters_off": True,
        "attn2_observer_output_modified": False,
        "anchor_consumed": False,
        "target_edit_instruction_consumed": False,
        "four_role_joint_null_available": True,
        "common_null_broadcast_used": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "observer_tensor_contract_pass_only": True,
        "global_role_proposal_control_max_rank_gate_pass": False,
        "pre_affinity_ordered_proposal_family_binding_verified": False,
        "control_exchangeability_proven": False,
        "same_index_common_randomization_transform_proven": False,
        "statistical_error_control_available": False,
        "actual_sp4_rank_shard_files_present": False,
        "sp4_gather_execution_verified": False,
        "remote_execution_verified": False,
        "scientific_claim_authorized": False,
    }
    return {**payload, "receipt_sha256": M.object_sha256(payload)}


def reseal_receipt(receipt):
    payload = deepcopy(receipt)
    payload.pop("receipt_sha256", None)
    return {**payload, "receipt_sha256": M.object_sha256(payload)}


def artifact_from(repeat_real, repeat_shuffled, repeat_null, receipt=None):
    receipt = receipt or make_receipt(repeat_real, repeat_shuffled, repeat_null)
    return M.LoadedJointNullArtifactV15CR10(
        repeat_real=repeat_real,
        repeat_shuffled=repeat_shuffled,
        repeat_null_bank=repeat_null,
        receipt=receipt,
        receipt_sha256=receipt["receipt_sha256"],
    )


class JointNullRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = M.load_joint_null_registry_v15c_r10()

    def test_registry_expands_four_distinct_exact64_matched_wrong_banks(self):
        registry = self.registry
        self.assertEqual(registry.registry_sha256, M.REGISTRY_SHA256)
        self.assertEqual(
            tuple(item.registry_sha256 for item in registry.roles),
            M.ROLE_CONTROL_REGISTRY_SHA256,
        )
        self.assertEqual(len(set(M.ROLE_CONTROL_REGISTRY_SHA256)), 4)
        locked = set(registry.locked_role_token_indices)
        for role in registry.roles:
            self.assertEqual(len(role.controls), 64)
            self.assertEqual(len({item.source_token_indices for item in role.controls}), 64)
            for control in role.controls:
                self.assertEqual(len(control.source_token_indices), len(role.token_ids))
                self.assertFalse(set(control.source_token_indices) & locked)
        for index in range(M.NULL_COUNT):
            self.assertEqual(
                len({role.controls[index].source_token_indices for role in registry.roles}),
                4,
            )

    def test_joint_registry_dataclass_fields_are_exact_and_not_shadowed(self):
        self.assertEqual(
            tuple(item.name for item in fields(M.JointNullRegistryV15CR10)),
            (
                "active_token_ids", "locked_role_token_indices",
                "eligible_wrong_token_indices", "roles",
                "joint_index_registry_sha256", "capture_channel_registry_sha256",
                "registry_sha256", "raw",
            ),
        )

    def test_negative_control_and_proposal_family_terms_are_exact(self):
        consumer = self.registry.raw["consumer_contract"]
        self.assertEqual(
            consumer["control_bank_kind"], "aligned_negative_control_bank"
        )
        self.assertEqual(
            consumer["global_gate_name"],
            "global_role_proposal_control_max_rank_gate_pass",
        )
        self.assertEqual(
            consumer["finite_control_order_statistic_name"],
            "plus_one_control_exceedance_rank",
        )
        self.assertFalse(consumer["control_exchangeability_proven"])
        self.assertFalse(
            consumer["same_index_common_randomization_transform_proven"]
        )
        self.assertFalse(consumer["statistical_error_control_available"])
        self.assertEqual(
            consumer["proposal_family_binding_fields"],
            [
                "ordered_proposal_ids",
                "mask_sha256_by_proposal",
                "track_sha256_by_proposal",
                "geometry_sha256_by_proposal",
                "family_sha256",
            ],
        )

    def test_production_dict_literals_have_no_duplicate_string_keys(self):
        tree = ast.parse(
            (METHOD_ROOT / "source_four_role_joint_null_observer_v15c_r10.py")
            .read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            self.assertEqual(len(keys), len(set(keys)), f"duplicate keys: {keys}")

    def test_registry_reseal_cannot_change_one_role_control_root(self):
        raw = deepcopy(dict(self.registry.raw))
        raw["roles"][0]["control_registry_sha256"] = "f" * 64
        payload = dict(raw)
        payload.pop("registry_sha256")
        raw["registry_sha256"] = M.object_sha256(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "registry.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(M.FourRoleJointNullObserverV15CR10Error):
                M.load_joint_null_registry_v15c_r10(path)

    def test_runtime_tokenization_is_exact_and_source_only(self):
        receipt = M.validate_runtime_tokenization_v15c_r10(
            input_ids=self.registry.active_token_ids,
            attention_mask=(1,) * len(self.registry.active_token_ids),
            registry=self.registry,
        )
        self.assertTrue(receipt["runtime_exact"])
        self.assertFalse(receipt["route_authorized"])
        mutated = list(self.registry.active_token_ids)
        mutated[0] += 1
        with self.assertRaisesRegex(
            M.FourRoleJointNullObserverV15CR10Error, "tokenization differs"
        ):
            M.validate_runtime_tokenization_v15c_r10(
                input_ids=mutated,
                attention_mask=(1,) * len(mutated),
                registry=self.registry,
            )

    def test_checked_in_state_has_no_tensor_and_is_no_go(self):
        status = M.current_no_tensor_status_v15c_r10()
        self.assertEqual(
            status["status"],
            "PREREGISTERED_PLAN_ONLY_NO_REAL_TENSOR_"
            "NO_STATISTICAL_FWER_NO_GO",
        )
        for key in (
            "real_tensor_present", "four_role_joint_null_verified",
            "r9_future_affinity_constructed",
            "plus_one_control_exceedance_rank_computed",
            "global_role_proposal_control_max_rank_gate_pass",
            "separate_repeat_artifacts_present",
            "producer_process_independence_verified",
            "mechanical_candidate_qualified", "remote_execution_verified",
            "route_authorized", "decode_authorized", "training_authorized",
            "scientific_claim_authorized",
        ):
            self.assertIs(status[key], False)
        self.assertEqual(status["renderer_forward_calls"], 0)
        self.assertEqual(status["optimizer_updates"], 0)

    def test_no_argument_validator_is_no_go_and_modules_have_no_execution_calls(self):
        status = V.validate()
        self.assertFalse(status["mechanical_candidate_qualified"])
        self.assertFalse(
            status["global_role_proposal_control_max_rank_gate_pass"]
        )
        self.assertFalse(status["remote_execution_verified"])
        called = set()
        for path in (
            METHOD_ROOT / "source_four_role_joint_null_observer_v15c_r10.py",
            METHOD_ROOT / "validate_source_four_role_joint_null_observer_v15c_r10.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            called.update(
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            )
        for forbidden in (
            "backward", "step", "zero_grad", "decode", "all_gather",
            "all_gather_object",
        ):
            self.assertNotIn(forbidden, called)


class JointNullStrictConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = M.load_joint_null_registry_v15c_r10()
        cls.real, cls.shuffled, cls.null = make_arrays()
        cls.repeat_real = (cls.real, cls.real.copy())
        cls.repeat_shuffled = (cls.shuffled, cls.shuffled.copy())
        cls.repeat_null = (cls.null, cls.null.copy())
        cls.receipt = make_receipt(
            cls.repeat_real, cls.repeat_shuffled, cls.repeat_null
        )

    def test_complete_local_fixture_replays_but_adapter_remains_no_go(self):
        artifact = artifact_from(
            self.repeat_real, self.repeat_shuffled, self.repeat_null, self.receipt
        )
        validation = M.validate_loaded_joint_null_artifact_v15c_r10(
            artifact, registry=self.registry
        )
        self.assertTrue(validation["four_role_joint_null_tensor_contract_verified"])
        self.assertEqual(
            validation["control_bank_kind"], "aligned_negative_control_bank"
        )
        self.assertFalse(
            validation["global_role_proposal_control_max_rank_gate_pass"]
        )
        self.assertFalse(validation["control_exchangeability_proven"])
        self.assertTrue(validation["separate_repeat_artifacts_verified"])
        self.assertFalse(validation["producer_process_independence_verified"])
        self.assertFalse(
            validation["independent_capture_channel_value_binding_pinned"]
        )
        with self.assertRaisesRegex(
            M.FourRoleJointNullObserverV15CR10Error,
            "no independent capture binding is pinned",
        ):
            M.adapt_joint_null_artifact_to_r9_v15c_r10(
                artifact, registry=self.registry
            )
        expected = self.receipt["capture_channel_value_binding_sha256"]
        pinned = M.validate_loaded_joint_null_artifact_v15c_r10(
            artifact,
            registry=self.registry,
            expected_capture_channel_value_binding_sha256=expected,
        )
        self.assertTrue(
            pinned["caller_expected_capture_channel_value_binding_matched"]
        )
        self.assertFalse(
            pinned["independent_capture_channel_value_binding_pinned"]
        )
        self.assertFalse(pinned["actual_sp4_rank_shard_files_replayed"])
        with self.assertRaisesRegex(
            M.FourRoleJointNullObserverV15CR10Error,
            "no independent capture binding is pinned",
        ):
            M.adapt_joint_null_artifact_to_r9_v15c_r10(
                artifact,
                registry=self.registry,
                expected_capture_channel_value_binding_sha256=expected,
            )
        with patch.object(
            M, "PINNED_CAPTURE_CHANNEL_VALUE_BINDING_SHA256", expected
        ):
            with self.assertRaisesRegex(
                M.FourRoleJointNullObserverV15CR10Error,
                "actual rank shards and fresh runner/postflight are absent",
            ):
                M.adapt_joint_null_artifact_to_r9_v15c_r10(
                    artifact,
                    registry=self.registry,
                    expected_capture_channel_value_binding_sha256=expected,
                )

    def test_common_null_broadcast_claiming_true_is_rejected_before_r9(self):
        common = self.null[:, 0:1]
        copied = np.ascontiguousarray(np.broadcast_to(common, M.NULL_SHAPE))
        self.assertFalse(M.role_null_slices_pairwise_distinct_v15c_r10(copied))
        with self.assertRaisesRegex(
            r9.SourceRoleAuthorityV15CR9Error, "common/broadcast"
        ):
            role_sha = tuple(
                r9.array_sha256(copied[:, role]) for role in range(len(M.ROLE_NAMES))
            )
            r9.joint_null_binding_payload_v15c_r9(
                real=self.real,
                shuffled=self.shuffled,
                null_bank=copied,
                null_registry_sha256=M.REGISTRY_SHA256,
                role_null_registry_sha256=M.ROLE_CONTROL_REGISTRY_SHA256,
                role_null_tensor_sha256=role_sha,
                upstream_validation=r9_validation_fixture(),
            )

    def test_r9_future_affinity_rejects_float64_complex_and_noncontiguous(self):
        role_sha = tuple(
            r9.array_sha256(self.null[:, role]) for role in range(len(M.ROLE_NAMES))
        )
        binding = r9.joint_null_binding_payload_v15c_r9(
            real=self.real,
            shuffled=self.shuffled,
            null_bank=self.null,
            null_registry_sha256=M.REGISTRY_SHA256,
            role_null_registry_sha256=M.ROLE_CONTROL_REGISTRY_SHA256,
            role_null_tensor_sha256=role_sha,
            upstream_validation=r9_validation_fixture(),
        )
        base = {
            "real": self.real,
            "shuffled": self.shuffled,
            "null_bank": self.null,
            "null_registry_sha256": M.REGISTRY_SHA256,
            "null_index_alignment_verified": True,
            "four_role_joint_null_available": True,
            "role_null_registry_sha256": M.ROLE_CONTROL_REGISTRY_SHA256,
            "role_null_tensor_sha256": role_sha,
            "joint_null_upstream_validation": r9_validation_fixture(),
            "joint_null_binding_sha256": r9.object_sha256(binding),
        }
        mutations = (
            {**base, "real": self.real.astype(np.float64)},
            {**base, "shuffled": self.shuffled.astype(np.complex64)},
            {**base, "null_bank": self.null[..., ::-1]},
        )
        for kwargs in mutations:
            with self.assertRaisesRegex(
                r9.SourceRoleAuthorityV15CR9Error, "tensor contract"
            ):
                r9.R6AffinityInputV15CR9(**kwargs)

    def test_r9_local_unit_binding_recomputes_real_shuffled_and_null(self):
        role_sha = tuple(
            r9.array_sha256(self.null[:, role]) for role in range(len(M.ROLE_NAMES))
        )
        validation = r9_validation_fixture()
        binding = r9.joint_null_binding_payload_v15c_r9(
            real=self.real,
            shuffled=self.shuffled,
            null_bank=self.null,
            null_registry_sha256=M.REGISTRY_SHA256,
            role_null_registry_sha256=M.ROLE_CONTROL_REGISTRY_SHA256,
            role_null_tensor_sha256=role_sha,
            upstream_validation=validation,
        )
        base = {
            "real": self.real,
            "shuffled": self.shuffled,
            "null_bank": self.null,
            "null_registry_sha256": M.REGISTRY_SHA256,
            "null_index_alignment_verified": True,
            "four_role_joint_null_available": True,
            "role_null_registry_sha256": M.ROLE_CONTROL_REGISTRY_SHA256,
            "role_null_tensor_sha256": role_sha,
            "joint_null_upstream_validation": validation,
            "joint_null_binding_sha256": r9.object_sha256(binding),
        }
        mutations = []
        for name in ("real", "shuffled", "null_bank"):
            changed = base[name].copy()
            changed.reshape(-1)[0] += np.float32(0.5)
            mutations.append({**base, name: changed})
        for kwargs in mutations:
            with self.assertRaises(r9.SourceRoleAuthorityV15CR9Error):
                r9.R6AffinityInputV15CR9(**kwargs)

    def test_one_role_copied_and_receipt_resealed_is_rejected(self):
        copied = self.null.copy()
        copied[:, 3] = copied[:, 2]
        copied_repeats = (copied, copied.copy())
        receipt = make_receipt(
            self.repeat_real, self.repeat_shuffled, copied_repeats
        )
        artifact = artifact_from(
            self.repeat_real, self.repeat_shuffled, copied_repeats, receipt
        )
        with self.assertRaisesRegex(
            M.FourRoleJointNullObserverV15CR10Error, "byte-identical"
        ):
            M.validate_loaded_joint_null_artifact_v15c_r10(
                artifact, registry=self.registry
            )

    def test_role_and_null_index_permutations_cannot_cross_original_channel_pin(self):
        expected = self.receipt["capture_channel_value_binding_sha256"]
        role_permutation = [1, 0, 3, 2]
        null_permutation = list(range(M.NULL_COUNT))
        null_permutation[0], null_permutation[1] = 1, 0

        all_roles_real = np.ascontiguousarray(self.real[:, role_permutation])
        all_roles_shuffled = np.ascontiguousarray(
            self.shuffled[:, role_permutation]
        )
        all_roles_null = np.ascontiguousarray(
            self.null[:, role_permutation]
        )
        null_roles_only = np.ascontiguousarray(self.null[:, role_permutation])
        all_j = np.ascontiguousarray(self.null[:, :, null_permutation])
        one_role_j = self.null.copy()
        one_role_j[:, 2] = np.ascontiguousarray(
            one_role_j[:, 2, null_permutation]
        )
        fixtures = (
            (all_roles_real, all_roles_shuffled, all_roles_null),
            (self.real, self.shuffled, null_roles_only),
            (self.real, self.shuffled, all_j),
            (self.real, self.shuffled, one_role_j),
        )
        for real, shuffled, null in fixtures:
            repeat_real = (real, real.copy())
            repeat_shuffled = (shuffled, shuffled.copy())
            repeat_null = (null, null.copy())
            receipt = make_receipt(repeat_real, repeat_shuffled, repeat_null)
            artifact = artifact_from(
                repeat_real, repeat_shuffled, repeat_null, receipt
            )
            with self.assertRaisesRegex(
                M.FourRoleJointNullObserverV15CR10Error,
                "differs from caller expectation",
            ):
                M.validate_loaded_joint_null_artifact_v15c_r10(
                    artifact,
                    registry=self.registry,
                    expected_capture_channel_value_binding_sha256=expected,
                )

    def test_sp4_repeat_tensor_mismatch_fails_even_after_nested_reseal(self):
        receipt = deepcopy(self.receipt)
        repeat = receipt["sp4_repeat_transcript"]["blocks"][0]["repeats"][1]
        repeat["ranks"][2]["collective_tensor_sha256"] = "e" * 64
        rank = repeat["ranks"][2]
        metadata_payload = {
            "block_index": M.SELECTED_BLOCK_INDICES[0],
            "repeat_index": 1,
            "capture_pass_label": repeat["capture_pass_label"],
            "registry_sha256": M.REGISTRY_SHA256,
            **{key: value for key, value in rank.items() if key != "metadata_sha256"},
        }
        rank["metadata_sha256"] = M.object_sha256(metadata_payload)
        repeat_payload = dict(repeat)
        repeat_payload.pop("repeat_receipt_sha256")
        repeat["repeat_receipt_sha256"] = M.object_sha256(repeat_payload)
        transcript = receipt["sp4_repeat_transcript"]
        transcript_payload = dict(transcript)
        transcript_payload.pop("transcript_sha256")
        transcript["transcript_sha256"] = M.object_sha256(transcript_payload)
        receipt = reseal_receipt(receipt)
        artifact = artifact_from(
            self.repeat_real, self.repeat_shuffled, self.repeat_null, receipt
        )
        with self.assertRaisesRegex(
            M.FourRoleJointNullObserverV15CR10Error, "rank gather geometry"
        ):
            M.validate_loaded_joint_null_artifact_v15c_r10(
                artifact, registry=self.registry
            )

    def test_role_registry_provenance_mutation_and_global_reseal_fails(self):
        receipt = deepcopy(self.receipt)
        receipt["role_control_registry_sha256"][1] = "d" * 64
        receipt = reseal_receipt(receipt)
        artifact = artifact_from(
            self.repeat_real, self.repeat_shuffled, self.repeat_null, receipt
        )
        with self.assertRaises(M.FourRoleJointNullObserverV15CR10Error):
            M.validate_loaded_joint_null_artifact_v15c_r10(
                artifact, registry=self.registry
            )

    def test_strict_npy_loader_binds_file_bytes_and_replays_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = {}
            for repeat_index in range(M.REPEAT_COUNT):
                paths[f"repeat_{repeat_index}"] = {
                    "real": root / f"repeat_{repeat_index}_real_affinity.npy",
                    "shuffled": root / f"repeat_{repeat_index}_shuffled_affinity.npy",
                    "joint_null": root / f"repeat_{repeat_index}_joint_null_affinity.npy",
                }
                np.save(
                    paths[f"repeat_{repeat_index}"]["real"],
                    self.repeat_real[repeat_index],
                    allow_pickle=False,
                )
                np.save(
                    paths[f"repeat_{repeat_index}"]["shuffled"],
                    self.repeat_shuffled[repeat_index],
                    allow_pickle=False,
                )
                np.save(
                    paths[f"repeat_{repeat_index}"]["joint_null"],
                    self.repeat_null[repeat_index],
                    allow_pickle=False,
                )
            file_rows = deepcopy(self.receipt["tensor_files"])
            for repeat_key, repeat_paths in paths.items():
                for name, path in repeat_paths.items():
                    file_rows[repeat_key][name]["file_size"] = path.stat().st_size
                    file_rows[repeat_key][name]["file_sha256"] = M.file_sha256(path)
            receipt = make_receipt(
                self.repeat_real,
                self.repeat_shuffled,
                self.repeat_null,
                file_rows=file_rows,
            )
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            loaded = M.load_joint_null_artifact_v15c_r10(
                root, receipt_path, registry=self.registry
            )
            self.assertEqual(loaded.null_bank.shape, M.NULL_SHAPE)
            self.assertEqual(loaded.receipt_sha256, receipt["receipt_sha256"])
            repeated_path = paths["repeat_1"]["real"]
            repeated_path.unlink()
            os.link(paths["repeat_0"]["real"], repeated_path)
            hardlink_rows = deepcopy(file_rows)
            hardlink_rows["repeat_1"]["real"]["file_size"] = (
                repeated_path.stat().st_size
            )
            hardlink_rows["repeat_1"]["real"]["file_sha256"] = M.file_sha256(
                repeated_path
            )
            hardlink_receipt = make_receipt(
                self.repeat_real,
                self.repeat_shuffled,
                self.repeat_null,
                file_rows=hardlink_rows,
            )
            receipt_path.write_text(json.dumps(hardlink_receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                M.FourRoleJointNullObserverV15CR10Error, "reused a path/inode"
            ):
                M.load_joint_null_artifact_v15c_r10(
                    root, receipt_path, registry=self.registry
                )

    def test_same_in_memory_object_cannot_impersonate_two_capture_repeats(self):
        receipt = make_receipt(
            (self.real, self.real),
            (self.shuffled, self.shuffled),
            (self.null, self.null),
        )
        artifact = artifact_from(
            (self.real, self.real),
            (self.shuffled, self.shuffled),
            (self.null, self.null),
            receipt,
        )
        with self.assertRaisesRegex(
            M.FourRoleJointNullObserverV15CR10Error, "not separate artifacts"
        ):
            M.validate_loaded_joint_null_artifact_v15c_r10(
                artifact, registry=self.registry
            )

    def test_bool_or_float_cannot_impersonate_exact_integer_receipts(self):
        for target, value in (("optimizer", False), ("block", 4.0)):
            receipt = deepcopy(self.receipt)
            if target == "optimizer":
                receipt["optimizer_updates"] = value
            else:
                receipt["sp4_repeat_transcript"]["blocks"][0]["block_index"] = value
                transcript = receipt["sp4_repeat_transcript"]
                transcript_payload = dict(transcript)
                transcript_payload.pop("transcript_sha256")
                transcript["transcript_sha256"] = M.object_sha256(transcript_payload)
            receipt = reseal_receipt(receipt)
            artifact = artifact_from(
                self.repeat_real, self.repeat_shuffled, self.repeat_null, receipt
            )
            with self.assertRaises(M.FourRoleJointNullObserverV15CR10Error):
                M.validate_loaded_joint_null_artifact_v15c_r10(
                    artifact, registry=self.registry
                )


class LocalReleaseClosureTests(unittest.TestCase):
    def test_local_release_exactly_closes_code_and_no_go_boundaries(self):
        release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            release["schema_version"],
            "bernini-source-four-role-joint-null-observer-v15c-r10-local-release",
        )
        self.assertEqual(
            release["status"],
            "LOCAL_SCHEMA_ONLY_NO_REAL_TENSOR_NO_ACTUAL_SP4_SHARDS_"
            "NO_RUNNER_NO_STATISTICAL_FWER_NO_GO",
        )
        self.assertEqual(release["member_count"], 5)
        self.assertEqual(release["member_count"], len(release["members"]))
        paths = [row["path"] for row in release["members"]]
        self.assertEqual(paths, sorted(paths))
        for row in release["members"]:
            path = REPO_ROOT / row["path"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(path.stat().st_size, row["size"])
        self.assertEqual(release["registry_sha256"], M.REGISTRY_SHA256)
        self.assertEqual(
            release["role_control_registry_sha256"],
            list(M.ROLE_CONTROL_REGISTRY_SHA256),
        )
        self.assertEqual(
            release["joint_index_registry_sha256"],
            M.JOINT_INDEX_REGISTRY_SHA256,
        )
        self.assertEqual(
            release["capture_channel_registry_sha256"],
            M.CAPTURE_CHANNEL_REGISTRY_SHA256,
        )
        self.assertIsNone(release["pinned_capture_channel_value_binding_sha256"])
        for key in (
            "control_exchangeability_proven",
            "same_index_common_randomization_transform_proven",
            "statistical_error_control_available",
            "pre_affinity_ordered_proposal_family_binding_verified",
            "real_tensor_present",
            "separate_repeat_artifacts_present",
            "producer_process_independence_verified",
            "actual_sp4_rank_shard_files_present",
            "official_r10_runner_present",
            "official_r10_postflight_present",
            "r9_future_affinity_constructed",
            "global_role_proposal_control_max_rank_gate_pass",
            "mechanical_candidate_qualified",
            "route_authorized",
            "decode_authorized",
            "training_authorized",
            "scientific_claim_authorized",
        ):
            self.assertIs(release[key], False)
        self.assertIs(release["source_only"], True)
        self.assertEqual(type(release["renderer_forward_calls"]), int)
        self.assertEqual(release["renderer_forward_calls"], 0)
        self.assertEqual(type(release["optimizer_updates"]), int)
        self.assertEqual(release["optimizer_updates"], 0)
        payload = dict(release)
        claimed = payload.pop("release_sha256")
        expected = hashlib.sha256(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(claimed, expected)


if __name__ == "__main__":
    unittest.main()
