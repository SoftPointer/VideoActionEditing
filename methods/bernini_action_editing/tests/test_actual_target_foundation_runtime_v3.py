#!/usr/bin/env python3
"""Adversarial CPU contracts for the frozen actual-target V3 runtime."""

from __future__ import annotations

import ast
import ctypes
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import actual_target_foundation_graph_v3 as graph
import actual_target_foundation_runtime_v3 as runtime

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import torch
except ModuleNotFoundError:
    torch = None


def torch_numpy_bridge_available():
    if torch is None or np is None:
        return False
    try:
        torch.zeros(1).numpy()
    except (RuntimeError, TypeError):
        return False
    return True


TORCH_NUMPY_BRIDGE_AVAILABLE = torch_numpy_bridge_available()


def produce_and_own(inventory, value, category):
    inventory.mark_opportunity(category)
    inventory.mark_produced(category)
    return inventory.own(value, category)


def phase_vector(shift: int, width: int = 8) -> tuple[float, ...]:
    return tuple(
        1.0 if index == (phase + shift) % width else 0.0
        for phase in range(runtime.PHASES)
        for index in range(width)
    )


def track_vector(shift: int) -> tuple[float, ...]:
    values = [0.0] * (96 * 12)
    values[(0 * 12) + (0 + shift) % 8] = 1.0
    values[(1 * 12) + (1 + shift) % 8] = 1.0
    return tuple(values)


def edge_vector(shift: int) -> tuple[float, ...]:
    return tuple(
        1.0 if channel == (phase + shift) % 5 else 0.0
        for phase in range(runtime.PHASES)
        for channel in range(5)
    )


@unittest.skipIf(np is None, "numpy unavailable")
class FakeFrozenBackend:
    model_names = ("fake_sam2", "fake_cotracker", "fake_dinov2", "fake_vjepa2")

    def __init__(self, fail_node: bool = False):
        self.scrubs = 0
        self.fail_node = fail_node
        self.hooks = {name: 0 for name in runtime.EXPECTED_HOOK_COUNTS}

    def decode(self, _path, _expected_sha256):
        return tuple(range(64))

    @staticmethod
    def _node_payload(source: bool = False):
        phases = []
        for phase in range(runtime.PHASES):
            left = np.zeros((20, 20), dtype=bool)
            right = np.zeros((20, 20), dtype=bool)
            left[4:9, 2 + phase : 5 + phase] = True
            right[10:15, 11 - phase : 14 - phase] = True
            first = np.zeros(8); first[(phase + (4 if source else 0)) % 8] = 1.0
            second = np.zeros(8); second[(phase + (6 if source else 2)) % 8] = 1.0
            phases.append(
                (
                    graph.AnonymousNodeV3(
                        left,
                        first,
                        float(left.mean()),
                        ((3.0 + phase) / 19.0, 6.0 / 19.0),
                        0,
                    ),
                    graph.AnonymousNodeV3(
                        right,
                        second,
                        float(right.mean()),
                        ((12.0 - phase) / 19.0, 12.0 / 19.0),
                        1,
                    ),
                )
            )
        return tuple(phases)

    def node(self, _frames, view):
        if self.fail_node:
            raise ValueError("synthetic node failure")
        self.hooks["sam2_image_encoder"] += runtime.PHASES
        self.hooks["dinov2"] += runtime.PHASES
        payload = self._node_payload(source=view == "source_noop")
        return runtime.NodeSketch(
            signature=graph.canonical_node_signature(payload),
            cardinalities=(2,) * runtime.PHASES,
            mechanically_valid_phases=runtime.PHASES,
            dustbin_used=True,
            private_payload=payload,
            unbalanced_phase_pair_count=7,
            dustbin_unmatched_count=2,
            dustbin_transport_mass=0.25,
        )

    def motion(self, _frames, view, _nodes):
        self.hooks["cotracker"] += 1
        shifts = {
            "target_forward_reference": 0,
            "target_forward_eval": 0,
            "target_reverse": 2,
            "target_deterministic_shuffle": 4,
            "source_noop": 6,
        }
        shift = shifts[view]
        positive_edge = edge_vector(shift)
        drop = list(positive_edge)
        if view == "target_forward_eval":
            drop[0], drop[1] = 0.0, 1.0
            drop[5], drop[6] = 0.0, 1.0
        return runtime.MotionSketch(
            track_signature=track_vector(shift),
            edge_signature=positive_edge,
            drop_edge_signature=tuple(drop),
            assigned_track_count=2,
            assigned_point_count=8,
            minimum_same_track_member_phases_observed=4,
            visible_and_member_fraction=0.875,
            per_phase_visible_member_counts=(8,) * runtime.PHASES,
            assignment_diagnostics={
                "ambiguous_overlap_observation_count": 1,
                "out_of_bounds_observation_count": 1,
                "nonfinite_observation_count": 0,
                "vote_tie_abstain_count": 1,
                "insufficient_membership_abstain_count": 1,
            },
            state_counts={
                "ABSENT": 1,
                "VISIBLE_MEMBER": 12,
                "OCCLUDED": 1,
                "VISIBLE_OUTSIDE_MASK": 2,
            },
            lifecycle_counts={
                "entry": 2,
                "occlusion": 1,
                "membership_loss": 1,
                "reentry": 2,
                "death": 0,
            },
            valid_adjacent_velocity_count=6,
            per_phase_active_counts=(0, 1, 1, 0, 0, 0, 0, 0),
            per_phase_birth_counts=(0, 1, 0, 0, 0, 0, 0, 0),
            per_phase_persist_counts=(0, 0, 1, 0, 0, 0, 0, 0),
            per_phase_death_counts=(0, 0, 0, 1, 0, 0, 0, 0),
            per_phase_valid_velocity_counts=(0, 1, 1, 0, 0, 0, 0, 0),
            per_phase_qualified_lifecycle_counts=(0, 0, 1, 1, 0, 0, 0, 0),
            evaluated_pairwise_edge_count=2,
            drop_edge_removed_count=2,
        )

    def phase(self, _frames, view):
        self.hooks["vjepa2"] += 1
        shifts = {
            "target_forward_reference": 0,
            "target_forward_eval": 0,
            "target_reverse": 2,
            "target_deterministic_shuffle": 4,
            "source_noop": 6,
        }
        return runtime.PhaseSketch(phase_vector(shifts[view]))

    def frozen_receipt(self):
        return {
            "all_models_eval_frozen": True,
            "source_and_weight_closure_unchanged": True,
            "parameter_updates": 0,
            "generator_forward_calls": 0,
        }

    def begin_case(self):
        return None

    def scrub_case(self):
        self.scrubs += 1

    def actual_forward_counts(self):
        return dict(self.hooks)


class AbstainingFakeFrozenBackend(FakeFrozenBackend):
    """Mechanically valid execution whose object branches legitimately abstain."""

    def node(self, _frames, _view):
        self.hooks["sam2_image_encoder"] += runtime.PHASES
        self.hooks["dinov2"] += runtime.PHASES
        payload = tuple(() for _ in range(runtime.PHASES))
        return runtime.NodeSketch(
            signature=graph.canonical_node_signature(payload),
            cardinalities=(0,) * runtime.PHASES,
            mechanically_valid_phases=0,
            dustbin_used=True,
            private_payload=payload,
            unbalanced_phase_pair_count=7,
            dustbin_unmatched_count=0,
            dustbin_transport_mass=0.0,
        )

    def motion(self, _frames, _view, _nodes):
        self.hooks["cotracker"] += 1
        return runtime.MotionSketch(
            track_signature=(0.0,) * (96 * 12),
            edge_signature=(0.0,) * (runtime.PHASES * 5),
            drop_edge_signature=(0.0,) * (runtime.PHASES * 5),
            assigned_track_count=0,
            assigned_point_count=0,
            minimum_same_track_member_phases_observed=0,
            visible_and_member_fraction=None,
            per_phase_visible_member_counts=(0,) * runtime.PHASES,
            assignment_diagnostics={
                "ambiguous_overlap_observation_count": 0,
                "out_of_bounds_observation_count": 0,
                "nonfinite_observation_count": 0,
                "vote_tie_abstain_count": 0,
                "insufficient_membership_abstain_count": 144,
            },
            state_counts={
                "ABSENT": 0,
                "VISIBLE_MEMBER": 0,
                "OCCLUDED": 0,
                "VISIBLE_OUTSIDE_MASK": 0,
            },
            lifecycle_counts={
                "entry": 0,
                "occlusion": 0,
                "membership_loss": 0,
                "reentry": 0,
                "death": 0,
            },
            valid_adjacent_velocity_count=0,
            per_phase_active_counts=(0,) * runtime.PHASES,
            per_phase_birth_counts=(0,) * runtime.PHASES,
            per_phase_persist_counts=(0,) * runtime.PHASES,
            per_phase_death_counts=(0,) * runtime.PHASES,
            per_phase_valid_velocity_counts=(0,) * runtime.PHASES,
            per_phase_qualified_lifecycle_counts=(0,) * runtime.PHASES,
            evaluated_pairwise_edge_count=0,
            drop_edge_removed_count=0,
        )

    def phase(self, _frames, _view):
        self.hooks["vjepa2"] += 1
        return runtime.PhaseSketch((0.0,) * (runtime.PHASES * 8))


class MutableRaw:
    def __init__(self, *, fail_count: int = 0):
        self.value = 7
        self.fail_count = fail_count
        self.calls = 0

    def zero_(self):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError("injected zeroize failure")
        self.value = 0
        return self

    def any(self):
        return self.value != 0


class RuntimeMechanicalTests(unittest.TestCase):
    def test_real_launch_is_authorized_and_old_failed_root_cannot_be_reused(self):
        self.assertTrue(runtime.REAL_GPU_LAUNCH_AUTHORIZED)
        old_root = Path("/tmp/immutable-failed-v3")
        new_root = Path("/tmp/fresh-v3r2")
        fixed = {
            "fresh_formal_run_root": str(new_root),
            "candidate_filename": "candidate.json",
            "cache_dirname": "cache",
        }
        with mock.patch.object(runtime, "REAL_GPU_LAUNCH_AUTHORIZED", True), mock.patch.object(
            runtime.authority, "load_authority", return_value={"fixed_paths": fixed}
        ), mock.patch.object(runtime, "RealFrozenBackend") as backend:
            with self.assertRaises(runtime.RuntimeV3Error):
                runtime.main(
                    [
                        "--run-real",
                        "--output",
                        str(old_root / "candidate.json"),
                        "--cache-dir",
                        str(old_root / "cache"),
                    ]
                )
        backend.assert_not_called()

    def test_nonfinite_signatures_abstain_as_json_safe_null(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=repr(invalid)):
                self.assertIsNone(runtime._cosine((1.0, invalid), (1.0, 2.0)))
                self.assertIsNone(runtime._margin((1.0, invalid), (1.0, 2.0), (2.0, 1.0)))
                self.assertIsNone(runtime._l2((1.0, invalid), (1.0, 2.0)))
                self.assertIsNone(runtime._norm((1.0, invalid)))

    @unittest.skipIf(np is None, "numpy unavailable")
    def test_numpy_metric_inputs_are_total_and_fail_closed(self):
        nonempty = np.asarray([3.0, 4.0], dtype=np.float64)
        same = np.asarray([3.0, 4.0], dtype=np.float64)
        zero = np.zeros(2, dtype=np.float64)

        self.assertAlmostEqual(runtime._cosine(nonempty, same), 1.0)
        self.assertAlmostEqual(runtime._l2(nonempty, zero), 5.0)
        self.assertAlmostEqual(runtime._norm(nonempty), 5.0)
        self.assertEqual(runtime._cosine(nonempty, zero), 0.0)
        self.assertIsNone(runtime._cosine(zero, zero))

        empty = np.asarray([], dtype=np.float64)
        self.assertIsNone(runtime._cosine(empty, empty))
        self.assertEqual(runtime._l2(empty, empty), 0.0)
        self.assertEqual(runtime._norm(empty), 0.0)

        mismatched = np.asarray([3.0], dtype=np.float64)
        self.assertIsNone(runtime._cosine(nonempty, mismatched))
        self.assertIsNone(runtime._l2(nonempty, mismatched))

        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=repr(invalid)):
                nonfinite = np.asarray([1.0, invalid], dtype=np.float64)
                self.assertIsNone(runtime._cosine(nonfinite, same))
                self.assertIsNone(runtime._margin(nonfinite, same, zero))
                self.assertIsNone(runtime._l2(nonfinite, same))
                self.assertIsNone(runtime._norm(nonfinite))

        scalar = np.asarray(1.0, dtype=np.float64)
        self.assertIsNone(runtime._cosine(scalar, scalar))
        self.assertIsNone(runtime._l2(scalar, scalar))
        self.assertIsNone(runtime._norm(scalar))

        huge = np.asarray([1e308, 1e308], dtype=np.float64)
        self.assertIsNone(runtime._cosine(huge, huge))
        self.assertIsNone(runtime._l2(huge, zero))
        self.assertIsNone(runtime._norm(huge))

    @unittest.skipIf(np is None, "numpy unavailable")
    def test_numpy_signatures_complete_the_full_case_evidence_path(self):
        class NumpySignatureBackend(FakeFrozenBackend):
            def node(self, frames, view):
                value = super().node(frames, view)
                return replace(
                    value,
                    signature=np.asarray(value.signature, dtype=np.float64),
                )

            def motion(self, frames, view, nodes):
                value = super().motion(frames, view, nodes)
                return replace(
                    value,
                    track_signature=np.asarray(
                        value.track_signature, dtype=np.float64
                    ),
                    edge_signature=np.asarray(
                        value.edge_signature, dtype=np.float64
                    ),
                    drop_edge_signature=np.asarray(
                        value.drop_edge_signature, dtype=np.float64
                    ),
                )

            def phase(self, frames, view):
                value = super().phase(frames, view)
                return replace(
                    value,
                    signature=np.asarray(value.signature, dtype=np.float64),
                )

        receipt = runtime.run_canary(NumpySignatureBackend())
        self.assertEqual(len(receipt["mechanical_case_evidence"]), 4)
        self.assertTrue(receipt["aggregate"]["diagnostic_canary_pass"])

    def test_zero_production_category_is_valid_without_placeholder_ownership(self):
        inventory = runtime.RawInventoryV3(("conditional",))
        receipt = inventory.receipt(require_all_categories=True)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["opportunity_by_category"], {"conditional": 0})
        self.assertEqual(receipt["produced_by_category"], {"conditional": 0})
        self.assertEqual(receipt["registered_by_category"], {"conditional": 0})
        self.assertEqual(receipt["zeroized_by_category"], {"conditional": 0})
        self.assertEqual(receipt["zero_produced_categories"], ["conditional"])

    def test_views_are_disjoint_and_exact_controls(self):
        values = runtime._views(tuple(range(100, 180)), tuple(range(80)))
        self.assertFalse(
            set(values["target_forward_reference"])
            & set(values["target_forward_eval"])
        )
        self.assertEqual(
            values["target_reverse"], tuple(reversed(values["target_forward_eval"]))
        )
        self.assertEqual(
            values["target_deterministic_shuffle"],
            tuple(values["target_forward_eval"][index] for index in runtime.SHUFFLE),
        )
        phase = runtime._phase_views(tuple(range(100, 180)), tuple(range(80)))
        self.assertEqual(len(phase["target_forward_reference"]), 16)
        self.assertEqual(len(phase["target_forward_eval"]), 16)
        self.assertFalse(
            set(phase["target_forward_reference"])
            & set(phase["target_forward_eval"])
        )

    def test_raw_release_failure_remains_owned_and_later_objects_scrub(self):
        inventory = runtime.RawInventoryV3(("raw",))
        flaky = produce_and_own(inventory, MutableRaw(fail_count=1), "raw")
        later = produce_and_own(inventory, MutableRaw(), "raw")
        with self.assertRaises(runtime.RuntimeV3Error):
            inventory.release(flaky)
        self.assertEqual(len(inventory._owned), 2)
        inventory.scrub_all()
        self.assertEqual(flaky.value, 0)
        self.assertEqual(later.value, 0)
        receipt = inventory.receipt(require_all_categories=True)
        self.assertEqual(receipt["outstanding_count"], 0)
        self.assertEqual(receipt["failure_attempts_by_category"]["raw"], 1)
        self.assertFalse(receipt["verified"])

    def test_recursive_scrub_does_not_short_circuit_after_failure(self):
        inventory = runtime.RawInventoryV3(("raw",))
        bad = MutableRaw(fail_count=99)
        good = MutableRaw()
        produce_and_own(inventory, [bad, good], "raw")
        with self.assertRaises(runtime.RuntimeV3Error):
            inventory.scrub_all()
        self.assertGreaterEqual(bad.calls, 2)
        self.assertEqual(good.value, 0)

    def test_immutable_raw_leaf_cannot_be_reported_as_zeroized(self):
        inventory = runtime.RawInventoryV3(("raw",))
        produce_and_own(inventory, b"immutable-target-payload", "raw")
        with self.assertRaises(runtime.RuntimeV3Error):
            inventory.scrub_all()
        receipt = inventory.receipt(require_all_categories=True)
        self.assertFalse(receipt["verified"])
        self.assertEqual(receipt["outstanding_count"], 1)
        self.assertGreaterEqual(receipt["failure_attempts_by_category"]["raw"], 2)

    def test_nested_container_cannot_launder_leaf_storage_alias(self):
        inventory = runtime.RawInventoryV3(("raw",))
        leaf = MutableRaw()
        produce_and_own(inventory, [leaf], "raw")
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "storage alias"):
            produce_and_own(inventory, {"same_leaf": leaf}, "raw")
        inventory.scrub_all()
        self.assertEqual(leaf.value, 0)
        self.assertFalse(inventory.receipt(require_all_categories=True)["verified"])

    def test_compressed_media_hash_uses_owned_mutable_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "video.bin"
            payload = (b"mutable-stream-contract" * 1024) + b"tail"
            path.write_bytes(payload)
            backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
            backend.raw = runtime.RawInventoryV3(("compressed_video_hash_buffer",))
            self.assertEqual(
                backend._compressed_video_sha256(path),
                hashlib.sha256(payload).hexdigest(),
            )
            receipt = backend.raw.receipt(require_all_categories=True)
            self.assertTrue(receipt["verified"])
            self.assertEqual(
                receipt["registered_by_category"]["compressed_video_hash_buffer"],
                1,
            )

    def test_hash_buffer_registration_failure_scrubs_before_any_fd_open(self):
        class RejectBuffer(runtime.RawInventoryV3):
            rejected = None

            def own(self, value, category):
                self.rejected = value
                raise runtime.RuntimeV3Error("injected hash-buffer ownership failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "video.bin"
            path.write_bytes(b"compressed")
            backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
            backend.raw = RejectBuffer(("compressed_video_hash_buffer",))
            with mock.patch.object(
                runtime.os,
                "open",
                side_effect=AssertionError("fd opened before buffer ownership"),
            ) as open_mock:
                with self.assertRaisesRegex(
                    runtime.RuntimeV3Error, "hash-buffer ownership"
                ):
                    backend._compressed_video_sha256(path)
            open_mock.assert_not_called()
            self.assertIsNotNone(backend.raw.rejected)
            self.assertFalse(any(backend.raw.rejected))

    def test_single_cpu_copy_uses_one_device_branch_and_scrubs_registration_failure(self):
        class Device:
            def __init__(self, kind):
                self.type = kind

        class FakeTensor:
            def __init__(self, kind, created):
                self.device = Device(kind)
                self.created = created
                self.to_calls = []
                self.clone_calls = []

            def detach(self):
                return self

            def to(self, **kwargs):
                self.to_calls.append(kwargs)
                return self.created

            def clone(self, **kwargs):
                self.clone_calls.append(kwargs)
                return self.created

        class RejectInventory(runtime.RawInventoryV3):
            def __init__(self):
                super().__init__(("model_hash_copy",))
                self.seen = None

            def own(self, value, category):
                self.seen = (value, category)
                raise runtime.RuntimeV3Error("injected ownership failure")

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        contiguous = object()
        backend.torch = type("FakeTorch", (), {"contiguous_format": contiguous})
        for kind in ("cuda", "cpu"):
            with self.subTest(kind):
                created = MutableRaw()
                source = FakeTensor(kind, created)
                backend.raw = RejectInventory()
                with self.assertRaisesRegex(runtime.RuntimeV3Error, "injected"):
                    backend._own_single_cpu_tensor_copy(source)
                self.assertEqual(created.value, 0)
                self.assertEqual(backend.raw.seen, (created, "model_hash_copy"))
                if kind == "cuda":
                    self.assertEqual(source.clone_calls, [])
                    self.assertEqual(
                        source.to_calls,
                        [
                            {
                                "device": "cpu",
                                "copy": True,
                                "memory_format": contiguous,
                            }
                        ],
                    )
                else:
                    self.assertEqual(source.to_calls, [])
                    self.assertEqual(
                        source.clone_calls, [{"memory_format": contiguous}]
                    )

    def test_hook_callback_increments_exactly_once(self):
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend._hook_counts = {name: 0 for name in runtime.EXPECTED_HOOK_COUNTS}
        hook = backend._hook("dinov2")
        hook(None, None, None)
        self.assertEqual(backend._hook_counts["dinov2"], 1)
        self.assertEqual(sum(backend._hook_counts.values()), 1)

    def test_real_cli_rejects_non_authority_paths_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "forbidden.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "actual_target_foundation_runtime_v3.py"),
                    "--run-real",
                    "--output",
                    str(output),
                    "--cache-dir",
                    str(Path(directory).resolve()),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_create_only_rejects_overwrite_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "one.json"
            runtime.create_only_json(output, {"finite": 1.0})
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(runtime.RuntimeV3Error):
                runtime.create_only_json(output, {"finite": 2.0})
            with self.assertRaises(Exception):
                runtime.create_only_json(root / "nan.json", {"bad": float("nan")})

    def test_real_runtime_has_no_chained_raw_tensor_allocation(self):
        methods = {
            name: inspect.getsource(getattr(runtime.RealFrozenBackend, name))
            for name in ("node", "motion", "phase")
        }
        joined = "\n".join(methods.values())
        for forbidden in (
            ".to(self.device).clone(",
            ".float().cpu().clone(",
            ".bool().cpu().clone(",
            ".cpu().numpy().copy(",
            ".flatten().clone(",
            ".sum().clone(",
            ".float().clone(",
        ):
            self.assertNotIn(forbidden, joined)
        self.assertGreaterEqual(methods["node"].count("_owned_tensor_copy("), 2)
        self.assertGreaterEqual(methods["motion"].count("_owned_tensor_copy("), 4)
        self.assertGreaterEqual(methods["phase"].count("_owned_tensor_copy("), 3)

        helper_tree = ast.parse(
            inspect.cleandoc(
                inspect.getsource(runtime.RealFrozenBackend._owned_tensor_copy)
            )
        )
        calls = [node for node in ast.walk(helper_tree) if isinstance(node, ast.Call)]
        to_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to"
        ]
        own_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "_own_created"
        ]
        self.assertEqual(len(to_calls), 1)
        self.assertEqual(len(own_calls), 1)
        self.assertIs(own_calls[0].args[0], to_calls[0])


@unittest.skipIf(np is None, "numpy unavailable")
class RuntimeNumpyTests(unittest.TestCase):
    def test_numpy_storage_alias_double_ownership_is_rejected(self):
        inventory = runtime.RawInventoryV3(("raw",))
        array = np.ones(8, dtype=np.float32)
        produce_and_own(inventory, array, "raw")
        with self.assertRaises(runtime.RuntimeV3Error):
            produce_and_own(inventory, array[1:], "raw")
        inventory.scrub_all()
        self.assertFalse(array.any())
        self.assertFalse(inventory.receipt(require_all_categories=True)["verified"])

    def test_nonzero_coordinate_aliases_are_aggregate_owned_once(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 3:7] = True
        coordinate_pair = tuple(mask.nonzero())
        self.assertEqual(len(coordinate_pair), 2)
        self.assertIs(coordinate_pair[0].base, coordinate_pair[1].base)

        inventory = runtime.RawInventoryV3(("sam_mask_coordinate_indices",))
        inventory.mark_opportunity("sam_mask_coordinate_indices")
        inventory.mark_produced("sam_mask_coordinate_indices")
        owned_pair = inventory.own(
            coordinate_pair, "sam_mask_coordinate_indices"
        )
        self.assertEqual(tuple(owned_pair[0]), tuple(mask.nonzero()[0]))
        self.assertEqual(tuple(owned_pair[1]), tuple(mask.nonzero()[1]))
        inventory.release(owned_pair)
        self.assertFalse(bool(coordinate_pair[0].any()))
        self.assertFalse(bool(coordinate_pair[1].any()))
        receipt = inventory.receipt(require_all_categories=True)
        self.assertEqual(receipt["produced_by_category"], {
            "sam_mask_coordinate_indices": 1
        })
        self.assertEqual(receipt["registered_by_category"], {
            "sam_mask_coordinate_indices": 1
        })
        self.assertEqual(receipt["zeroized_by_category"], {
            "sam_mask_coordinate_indices": 1
        })
        self.assertTrue(receipt["verified"])

    def test_bgr_decode_intermediate_is_owned_and_immediately_zeroized(self):
        class FakeCV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def cvtColor(value, _code):
                return value[:, :, ::-1].copy()

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.np = np
        backend.raw = runtime.RawInventoryV3(
            ("decoded_bgr_frame", "decoded_rgb_frame")
        )
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        bgr[0, 0] = [1, 2, 3]
        rgb = backend._own_decoded_rgb(bgr, FakeCV2)
        self.assertFalse(bgr.any())
        self.assertEqual(rgb[0, 0].tolist(), [3, 2, 1])
        self.assertEqual(len(backend.raw._owned), 1)
        backend.raw.scrub_all()
        self.assertFalse(rgb.any())
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["registered_by_category"], {
            "decoded_bgr_frame": 1,
            "decoded_rgb_frame": 1,
        })

    def test_bgr_is_scrubbed_if_initial_external_ownership_fails(self):
        class NeverCalledCV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def cvtColor(_value, _code):
                raise AssertionError("conversion ran before BGR ownership")

        class RejectBGR(runtime.RawInventoryV3):
            def own(self, value, category):
                raise runtime.RuntimeV3Error("injected BGR ownership failure")

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.np = np
        backend.raw = RejectBGR(("decoded_bgr_frame", "decoded_rgb_frame"))
        bgr = np.ones((720, 1280, 3), dtype=np.uint8)
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "BGR ownership"):
            backend._own_decoded_rgb(bgr, NeverCalledCV2)
        self.assertFalse(bgr.any())
        self.assertEqual(backend.raw._owned, [])

    def test_cv2_result_is_scrubbed_if_inventory_registration_fails(self):
        class FakeCV2:
            COLOR_BGR2RGB = 1
            converted = None

            @classmethod
            def cvtColor(cls, value, _code):
                cls.converted = value[:, :, ::-1].copy()
                return cls.converted

        class RejectRGB(runtime.RawInventoryV3):
            def own(self, value, category):
                if category == "decoded_rgb_frame":
                    raise runtime.RuntimeV3Error("injected RGB registration failure")
                return super().own(value, category)

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.np = np
        backend.raw = RejectRGB(("decoded_bgr_frame", "decoded_rgb_frame"))
        bgr = np.ones((720, 1280, 3), dtype=np.uint8)
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "injected"):
            backend._own_decoded_rgb(bgr, FakeCV2)
        self.assertFalse(bgr.any())
        self.assertIsNotNone(FakeCV2.converted)
        self.assertFalse(FakeCV2.converted.any())
        self.assertEqual(backend.raw._owned, [])

    def test_sam_external_mask_batch_mid_claim_failure_scrubs_every_mask(self):
        class RejectSecond(runtime.RawInventoryV3):
            def __init__(self):
                super().__init__(("sam_ann_mask_pre_filter",))
                self.calls = 0

            def own(self, value, category):
                self.calls += 1
                if self.calls == 2:
                    raise runtime.RuntimeV3Error("injected SAM mid-list failure")
                return super().own(value, category)

        masks = tuple(np.ones((4, 4), dtype=bool) for _ in range(3))
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.raw = RejectSecond()
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "SAM mid-list"):
            backend._own_external_batch(
                tuple(
                    (mask, "sam_ann_mask_pre_filter") for mask in masks
                )
        )
        self.assertTrue(all(not mask.any() for mask in masks))
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertEqual(receipt["outstanding_count"], 0)
        self.assertEqual(
            receipt["produced_by_category"],
            receipt["zeroized_by_category"],
        )
        self.assertFalse(receipt["verified"])

    def test_sam_c_input_is_unconditionally_copied_without_alias(self):
        original = np.zeros((6, 5), dtype=bool, order="C")
        original[1:4, 2:5] = True
        expected = original.copy(order="C")
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.np = np
        backend.raw = runtime.RawInventoryV3(
            ("sam_ann_mask_pre_filter", "sam_mask_c_contiguous_copy")
        )
        normalized = backend._normalize_sam_annotations(
            ({"segmentation": original},)
        )
        retained = normalized[0]["segmentation"]
        self.assertTrue(retained.flags.c_contiguous)
        self.assertFalse(np.shares_memory(retained, original))
        self.assertTrue(np.array_equal(retained, expected))
        self.assertFalse(original.any())
        backend.raw.release(retained)
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["produced_by_category"], {
            "sam_ann_mask_pre_filter": 1,
            "sam_mask_c_contiguous_copy": 1,
        })

    def test_pinned_sam_rle_full_storage_transpose_is_copied_and_root_scrubbed(self):
        # Exact layout shape produced by pinned sam2.utils.amg.rle_to_mask:
        # one owning 1-D bool array, reshape(w, h), then transpose().
        root = np.zeros(30, dtype=bool)
        returned = root.reshape(5, 6).transpose()
        returned[1:5, 2:4] = True
        expected = returned.copy(order="C")
        self.assertTrue(returned.flags.f_contiguous)
        self.assertIsNot(returned, root)
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.np = np
        backend.raw = runtime.RawInventoryV3(
            ("sam_ann_mask_pre_filter", "sam_mask_c_contiguous_copy")
        )
        normalized = backend._normalize_sam_annotations(
            ({"segmentation": returned},)
        )
        retained = normalized[0]["segmentation"]
        self.assertFalse(root.any())
        self.assertTrue(retained.flags.c_contiguous)
        self.assertFalse(np.shares_memory(retained, root))
        self.assertTrue(np.array_equal(retained, expected))
        backend.raw.release(retained)
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["produced_by_category"], {
            "sam_ann_mask_pre_filter": 1,
            "sam_mask_c_contiguous_copy": 1,
        })

    def test_sam_multiple_masks_are_claimed_copied_and_scrubbed_per_mask(self):
        first = np.zeros((6, 5), dtype=bool, order="F")
        second = np.zeros((6, 5), dtype=bool, order="F")
        first[1:3, 1:4] = True
        second[3:5, 0:2] = True

        class OrderedInventory(runtime.RawInventoryV3):
            def __init__(self):
                super().__init__(
                    ("sam_ann_mask_pre_filter", "sam_mask_c_contiguous_copy")
                )
                self.events = []
                self.copy_count = 0

            def own(self, value, category):
                if category == "sam_ann_mask_pre_filter":
                    label = "external0" if value is first else "external1"
                else:
                    label = f"copy{self.copy_count}"
                    if self.copy_count == 1:
                        if first.any():
                            raise AssertionError(
                                "first external mask survived until second copy"
                            )
                    self.copy_count += 1
                self.events.append(("own", label))
                return super().own(value, category)

            def release(self, value):
                row = next(item for item in self._owned if item[0] is value)
                if row[1] == "sam_ann_mask_pre_filter":
                    label = "external0" if value is first else "external1"
                    self.events.append(("release", label))
                return super().release(value)

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.np = np
        backend.raw = OrderedInventory()
        normalized = backend._normalize_sam_annotations(
            ({"segmentation": first}, {"segmentation": second})
        )
        self.assertEqual(backend.raw.events, [
            ("own", "external0"),
            ("own", "copy0"),
            ("release", "external0"),
            ("own", "external1"),
            ("own", "copy1"),
            ("release", "external1"),
        ])
        self.assertFalse(first.any())
        self.assertFalse(second.any())
        self.assertTrue(all(
            row["segmentation"].flags.c_contiguous for row in normalized
        ))
        self.assertTrue(all(
            not np.shares_memory(row["segmentation"], original)
            for row, original in zip(normalized, (first, second))
        ))
        for row in normalized:
            backend.raw.release(row["segmentation"])
        self.assertTrue(
            backend.raw.receipt(require_all_categories=True)["verified"]
        )

    def test_second_sam_copy_failure_scrubs_prior_current_and_pending_storage(self):
        roots = tuple(np.zeros(30, dtype=bool) for _ in range(3))
        masks = tuple(root.reshape(5, 6).transpose() for root in roots)
        for index, mask in enumerate(masks):
            mask[index : index + 2, 1:4] = True

        class RejectSecondCopy(runtime.RawInventoryV3):
            def __init__(self):
                super().__init__(
                    ("sam_ann_mask_pre_filter", "sam_mask_c_contiguous_copy")
                )
                self.copy_attempts = 0
                self.copies = []

            def own(self, value, category):
                if category == "sam_mask_c_contiguous_copy":
                    self.copy_attempts += 1
                    self.copies.append(value)
                    if self.copy_attempts == 2:
                        raise runtime.RuntimeV3Error(
                            "injected second SAM C-copy failure"
                        )
                return super().own(value, category)

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.np = np
        backend.raw = RejectSecondCopy()
        with self.assertRaisesRegex(
            runtime.RuntimeV3Error, "second SAM C-copy failure"
        ):
            backend._normalize_sam_annotations(
                tuple({"segmentation": mask} for mask in masks)
            )
        self.assertTrue(all(not root.any() for root in roots))
        self.assertEqual(len(backend.raw.copies), 2)
        self.assertTrue(all(not copy.any() for copy in backend.raw.copies))
        self.assertEqual(backend.raw._owned, [])
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertEqual(receipt["produced_by_category"], {
            "sam_ann_mask_pre_filter": 3,
            "sam_mask_c_contiguous_copy": 2,
        })
        self.assertEqual(receipt["registered_by_category"], {
            "sam_ann_mask_pre_filter": 3,
            "sam_mask_c_contiguous_copy": 1,
        })
        self.assertEqual(receipt["zeroized_by_category"], {
            "sam_ann_mask_pre_filter": 3,
            "sam_mask_c_contiguous_copy": 2,
        })

    def test_sam_strided_or_partial_backing_views_are_rejected_and_root_scrubbed(self):
        cases = {
            "strided_slice": lambda root: root[:, ::2],
            "negative_stride": lambda root: root[::-1, :],
            "contiguous_partial_base": lambda root: root[2:6, :],
        }
        for label, make_view in cases.items():
            with self.subTest(label=label):
                root = np.ones((8, 8), dtype=bool, order="C")
                view = make_view(root)
                backend = runtime.RealFrozenBackend.__new__(
                    runtime.RealFrozenBackend
                )
                backend.np = np
                backend.raw = runtime.RawInventoryV3(
                    (
                        "sam_ann_mask_pre_filter",
                        "sam_mask_c_contiguous_copy",
                    )
                )
                with self.assertRaisesRegex(
                    runtime.RuntimeV3Error,
                    "C- or F-contiguous|complete writable ndarray backing",
                ):
                    backend._normalize_sam_annotations(
                        ({"segmentation": view},)
                    )
                self.assertFalse(root.any())
                self.assertEqual(backend.raw._owned, [])
                receipt = backend.raw.receipt(require_all_categories=True)
                self.assertEqual(
                    receipt["registered_by_category"][
                        "sam_ann_mask_pre_filter"
                    ],
                    1,
                )
                self.assertEqual(
                    receipt["zeroized_by_category"][
                        "sam_ann_mask_pre_filter"
                    ],
                    1,
                )
                self.assertEqual(
                    receipt["produced_by_category"][
                        "sam_mask_c_contiguous_copy"
                    ],
                    0,
                )

    def test_sam_c_copy_registration_failure_scrubs_external_and_copy(self):
        class RejectCopy(runtime.RawInventoryV3):
            def __init__(self):
                super().__init__(
                    (
                        "sam_ann_mask_pre_filter",
                        "sam_mask_c_contiguous_copy",
                    )
                )
                self.rejected_copy = None

            def own(self, value, category):
                if category == "sam_mask_c_contiguous_copy":
                    self.rejected_copy = value
                    raise runtime.RuntimeV3Error("injected SAM C-copy failure")
                return super().own(value, category)

        class FakeSAM:
            original = None

            @classmethod
            def generate(cls, _frame):
                cls.original = np.zeros((16, 12), dtype=bool, order="F")
                cls.original[3:8, 4:9] = True
                return [
                    {
                        "segmentation": cls.original,
                        "area": 25,
                        "predicted_iou": 0.99,
                        "stability_score": 0.99,
                        "bbox": [4, 3, 5, 5],
                    }
                ]

        class NeverDinoProcessor:
            @staticmethod
            def __call__(**_kwargs):
                raise AssertionError("DINO must not run after SAM copy failure")

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.torch = torch
        backend.np = np
        backend.device = "cpu"
        backend.raw = RejectCopy()
        backend.sam = FakeSAM()
        backend.dino_processor = NeverDinoProcessor()
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "C-copy failure"):
            backend.node((np.zeros((16, 12, 3), dtype=np.uint8),), "copy_fail")
        self.assertIsNotNone(FakeSAM.original)
        self.assertTrue(FakeSAM.original.flags.f_contiguous)
        self.assertFalse(FakeSAM.original.any())
        self.assertIsNotNone(backend.raw.rejected_copy)
        self.assertTrue(backend.raw.rejected_copy.flags.c_contiguous)
        self.assertFalse(backend.raw.rejected_copy.any())
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertEqual(receipt["outstanding_count"], 0)
        self.assertEqual(
            receipt["produced_by_category"], receipt["zeroized_by_category"]
        )

    def test_fake_end_to_end_complete_mechanics_and_zero_generator(self):
        backend = FakeFrozenBackend()
        receipt = runtime.run_canary(backend)
        self.assertEqual(backend.scrubs, 4)
        self.assertEqual(receipt["forward_closure"]["logical_counts"], runtime.EXPECTED_LOGICAL_COUNTS)
        self.assertEqual(receipt["forward_closure"]["actual_forward_hook_counts"], runtime.EXPECTED_HOOK_COUNTS)
        self.assertEqual(len(receipt["mechanical_case_evidence"]), 4)
        self.assertTrue(receipt["aggregate"]["diagnostic_canary_pass"])
        self.assertFalse(receipt["training_performed"])
        self.assertFalse(receipt["optimizer_created"])
        self.assertEqual(receipt["parameter_updates"], 0)
        self.assertFalse(receipt["generator_loaded"])
        self.assertEqual(receipt["generator_forward_calls"], 0)

    def test_no_object_opportunity_writes_complete_json_safe_rejected_candidate(self):
        backend = AbstainingFakeFrozenBackend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "candidate.json"
            cache = root / "cache"
            cache.mkdir()
            receipt = runtime.run_canary(
                backend, output=output, cache_dir=cache
            )
            payload = output.read_text(encoding="ascii")
            self.assertNotIn("NaN", payload)
            self.assertNotIn("Infinity", payload)
            persisted = json.loads(payload)
            self.assertEqual(persisted, receipt)
            self.assertFalse(receipt["aggregate"]["diagnostic_canary_pass"])
            self.assertEqual(receipt["aggregate"]["passed_case_count"], 0)
            self.assertTrue(all(not row["case_pass"] for row in receipt["cases"]))
            self.assertTrue(
                all(
                    row["branch_pass"]["node"] is False
                    and row["branch_pass"]["track"] is False
                    and row["branch_pass"]["edge"] is False
                    for row in receipt["cases"]
                )
            )
            self.assertTrue(
                any(
                    row["branches"]["track"]["visible_and_member_fraction"]
                    is None
                    for row in receipt["mechanical_case_evidence"]
                )
            )
        self.assertTrue(receipt["representation_admission_hard_false"])
        self.assertFalse(receipt["scientific_evidence_claimed"])
        text = json.dumps(receipt, sort_keys=True)
        for key in runtime.FORBIDDEN_RECEIPT_KEYS:
            self.assertNotIn(f'"{key}"', text)

    def test_fake_output_and_four_cache_rows_are_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache = root / "cache"
            cache.mkdir()
            output = root / "candidate.json"
            receipt = runtime.run_canary(
                FakeFrozenBackend(), output=output, cache_dir=cache
            )
            self.assertEqual(json.loads(output.read_text())["digest"], receipt["digest"])
            self.assertEqual(len(list(cache.glob("*.json"))), 4)
            for path in cache.glob("*.json"):
                row = json.loads(path.read_text())
                self.assertIn("case_evidence", row)
                self.assertIn("evaluated_case", row)
                self.assertFalse(row["raw_teacher_payload_persisted"])
            with self.assertRaises(runtime.RuntimeV3Error):
                runtime.run_canary(FakeFrozenBackend(), output=output)

    def test_exception_path_invokes_case_scrub(self):
        backend = FakeFrozenBackend(fail_node=True)
        with self.assertRaises(ValueError):
            runtime.run_canary(backend)
        self.assertEqual(backend.scrubs, 1)


@unittest.skipIf(torch is None, "torch unavailable")
class RuntimeTorchTests(unittest.TestCase):
    def test_inference_tensor_requires_inference_context_for_exception_scrub(self):
        with torch.inference_mode():
            value = torch.ones(7, dtype=torch.float32)
        self.assertTrue(value.is_inference())
        inventory = runtime.RawInventoryV3(("raw",))
        produce_and_own(inventory, value, "raw")
        with self.assertRaises(runtime.RuntimeV3Error):
            inventory.release(value)
        self.assertEqual(
            inventory.receipt(require_all_categories=True)["outstanding_count"],
            1,
        )
        with torch.inference_mode():
            inventory.scrub_all()
        self.assertFalse(bool(value.any()))
        receipt = inventory.receipt(require_all_categories=True)
        self.assertEqual(receipt["outstanding_count"], 0)
        self.assertEqual(receipt["registered_by_category"], {"raw": 1})
        self.assertEqual(receipt["zeroized_by_category"], {"raw": 1})
        self.assertEqual(receipt["failure_attempts_by_category"], {"raw": 1})
        self.assertFalse(receipt["verified"])

    @unittest.skipIf(np is None, "numpy unavailable")
    def test_dino_nonsquare_output_is_claimed_and_scrubbed_in_context(self):
        class ModelOutput(dict):
            def __getattr__(self, name):
                return self[name]

        class FakeSAM:
            @staticmethod
            def generate(_frame):
                return []

        class FakeProcessor:
            last_input = None

            @classmethod
            def __call__(cls, **_kwargs):
                cls.last_input = torch.ones((1, 3, 8, 8))
                return {"pixel_values": cls.last_input}

        class FakeDino:
            last_output = None

            @classmethod
            def __call__(cls, **_kwargs):
                cls.last_output = ModelOutput(
                    last_hidden_state=torch.ones((1, 6, 8))
                )
                return cls.last_output

        frames = tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(8))
        for inject_input_release in (False, True):
            with self.subTest(inject_input_release=inject_input_release):
                backend = runtime.RealFrozenBackend.__new__(
                    runtime.RealFrozenBackend
                )
                backend.torch = torch
                backend.np = np
                backend.device = "cpu"
                backend.sam = FakeSAM()
                backend.dino_processor = FakeProcessor()
                backend.dino = FakeDino()
                backend.raw = runtime.RawInventoryV3(
                    ("dino_processor_input", "dino_tokens")
                )
                if inject_input_release:
                    original_release = backend.raw.release
                    release_count = {"dino_processor_input": 0}

                    def injected_release(value):
                        category = next(
                            category
                            for item, category, _ in backend.raw._owned
                            if item is value
                        )
                        if category == "dino_processor_input":
                            release_count[category] += 1
                            if release_count[category] == 2:
                                raise runtime.RuntimeV3Error(
                                    "injected device-input release failure"
                                )
                        return original_release(value)

                    backend.raw.release = injected_release
                expected = (
                    "best-effort owned batch release failed"
                    if inject_input_release
                    else "not square"
                )
                with self.assertRaisesRegex(runtime.RuntimeV3Error, expected):
                    backend.node(frames, "shape_failure")
                if inject_input_release:
                    backend.raw.release = original_release
                backend.scrub_case()
                self.assertTrue(
                    FakeDino.last_output.last_hidden_state.is_inference()
                )
                self.assertFalse(
                    bool(FakeDino.last_output.last_hidden_state.any())
                )
                self.assertFalse(bool(FakeProcessor.last_input.any()))
                self.assertEqual(
                    backend.raw.receipt(require_all_categories=True)[
                        "outstanding_count"
                    ],
                    0,
                )

    @unittest.skipIf(np is None, "numpy unavailable")
    def test_vjepa_shape_and_input_release_failures_scrub_outputs(self):
        class ModelOutput(dict):
            def __getattr__(self, name):
                return self[name]

        class Object:
            pass

        class FakeProcessor:
            @staticmethod
            def __call__(**_kwargs):
                return {"pixel_values_videos": torch.ones((1, 16, 3, 4, 4))}

        class FakeVJEPA:
            last_output = None

            def __init__(self):
                self.config = Object()
                self.config.tubelet_size = 2
                self.config.image_size = 4
                self.config.patch_size = 2

            @classmethod
            def __call__(cls, **_kwargs):
                cls.last_output = ModelOutput(
                    last_hidden_state=torch.ones((1, 31, 20)),
                    masked_hidden_state=torch.full((1, 32, 20), 2.0),
                    predictor_output=ModelOutput(
                        last_hidden_state=torch.full((1, 32, 20), 3.0),
                        target_hidden_state=torch.full((1, 32, 20), 4.0),
                    ),
                )
                return cls.last_output

        frames = tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(16))
        for inject_input_release in (False, True):
            with self.subTest(inject_input_release=inject_input_release):
                backend = runtime.RealFrozenBackend.__new__(
                    runtime.RealFrozenBackend
                )
                backend.torch = torch
                backend.np = np
                backend.device = "cpu"
                backend.vjepa_processor = FakeProcessor()
                backend.vjepa = FakeVJEPA()
                backend.raw = runtime.RawInventoryV3(
                    ("vjepa_processor_input", "vjepa_hidden", "vjepa_phase_signature")
                )
                if inject_input_release:
                    original_release = backend.raw.release
                    release_count = {"vjepa_processor_input": 0}

                    def injected_release(value):
                        category = next(
                            category
                            for item, category, _ in backend.raw._owned
                            if item is value
                        )
                        if category == "vjepa_processor_input":
                            release_count[category] += 1
                            if release_count[category] == 2:
                                raise runtime.RuntimeV3Error(
                                    "injected device-input release failure"
                                )
                        return original_release(value)

                    backend.raw.release = injected_release
                expected = (
                    "best-effort owned batch release failed"
                    if inject_input_release
                    else "exactly eight real tubelet2 blocks"
                )
                with self.assertRaisesRegex(runtime.RuntimeV3Error, expected):
                    backend.phase(frames, "failure")
                if inject_input_release:
                    backend.raw.release = original_release
                backend.scrub_case()
                output_leaves = (
                    FakeVJEPA.last_output["last_hidden_state"],
                    FakeVJEPA.last_output["masked_hidden_state"],
                    FakeVJEPA.last_output["predictor_output"]["last_hidden_state"],
                    FakeVJEPA.last_output["predictor_output"]["target_hidden_state"],
                )
                self.assertTrue(all(value.is_inference() for value in output_leaves))
                self.assertTrue(all(not bool(value.any()) for value in output_leaves))
                self.assertEqual(
                    backend.raw.observed[
                        "vjepa_model_output_unique_storages"
                    ],
                    4,
                )
                self.assertEqual(
                    backend.raw.receipt(require_all_categories=True)["outstanding_count"],
                    0,
                )

    @unittest.skipUnless(
        TORCH_NUMPY_BRIDGE_AVAILABLE, "torch/numpy bridge unavailable"
    )
    def test_malformed_cotracker_output_is_scrubbed_before_schema_failure(self):
        frames = tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(8))
        nodes = runtime.NodeSketch(
            signature=(),
            cardinalities=(0,) * runtime.PHASES,
            mechanically_valid_phases=0,
            dustbin_used=True,
            private_payload=tuple(() for _ in range(runtime.PHASES)),
        )

        for leaf_count in (1, 3):
            with self.subTest(leaf_count=leaf_count):
                class FakeCoTracker:
                    last_output = None

                    @classmethod
                    def __call__(cls, _video, **_kwargs):
                        cls.last_output = tuple(
                            torch.ones((1, 8, 2, 2))
                            for _ in range(leaf_count)
                        )
                        return cls.last_output

                backend = runtime.RealFrozenBackend.__new__(
                    runtime.RealFrozenBackend
                )
                backend.torch = torch
                backend.np = np
                backend.device = "cpu"
                backend.cotracker = FakeCoTracker()
                backend.raw = runtime.RawInventoryV3(
                    ("cotracker_video", "cotracker_tracks", "cotracker_visibility")
                )
                with self.assertRaisesRegex(
                    runtime.RuntimeV3Error, "exact two-leaf sequence"
                ):
                    backend.motion(frames, "malformed", nodes)
                backend.scrub_case()
                self.assertTrue(
                    all(value.is_inference() for value in FakeCoTracker.last_output)
                )
                self.assertTrue(
                    all(not bool(value.any()) for value in FakeCoTracker.last_output)
                )
                self.assertEqual(
                    backend.raw.receipt(require_all_categories=True)["outstanding_count"],
                    0,
                )

    @staticmethod
    def backend():
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.torch = torch
        backend.raw = runtime.RawInventoryV3(("model_hash_copy",))
        return backend

    def test_tensor_hash_handles_scalar_bfloat16_empty_noncontiguous_and_stride0(self):
        tensors = [
            torch.tensor(3.0),
            torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            torch.empty(0),
            torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
            torch.tensor([5.0]).expand(7),
        ]
        for tensor in tensors:
            backend = self.backend()
            digest = backend._tensor_value_sha256(tensor)
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, backend._tensor_value_sha256(tensor))
            self.assertTrue(backend.raw.receipt(require_all_categories=True)["verified"])
        source = inspect.getsource(runtime.RealFrozenBackend._tensor_value_sha256)
        self.assertNotIn("bytes(", source)

    def test_large_tensor_hash_uses_owned_direct_buffer_without_immutable_chunk(self):
        tensor = torch.arange(3 * 1024 * 1024, dtype=torch.float32)
        backend = self.backend()
        observed = backend._tensor_value_sha256(tensor)
        byte_count = int(tensor.numel() * tensor.element_size())
        direct = memoryview(
            (ctypes.c_ubyte * byte_count).from_address(int(tensor.data_ptr()))
        ).cast("B")
        self.assertEqual(observed, hashlib.sha256(direct).hexdigest())
        self.assertTrue(backend.raw.receipt(require_all_categories=True)["verified"])

    def test_cpu_hash_copy_is_one_owned_contiguous_storage_and_zeroizes(self):
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        expected = source.clone()
        backend = self.backend()
        owned = backend._own_single_cpu_tensor_copy(source)
        self.assertTrue(owned.is_contiguous())
        self.assertNotEqual(owned.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
        self.assertEqual(len(backend.raw._owned), 1)
        backend.raw.release(owned)
        self.assertFalse(bool(owned.any()))
        self.assertTrue(torch.equal(source, expected))
        self.assertTrue(backend.raw.receipt(require_all_categories=True)["verified"])

    def test_real_storage_trace_requires_ownership_before_next_allocation(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        def tensor_pointers(value):
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return set()
                return {int(value.untyped_storage().data_ptr())}
            if isinstance(value, dict):
                return {
                    pointer
                    for child in value.values()
                    for pointer in tensor_pointers(child)
                }
            if isinstance(value, (list, tuple)):
                return {
                    pointer
                    for child in value
                    for pointer in tensor_pointers(child)
                }
            return set()

        class ImmediateOwnershipTrace(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.pending = set()
                self.allocated = set()
                self.registered = set()

            def register(self, value):
                pointers = tensor_pointers(value)
                self.registered.update(pointers)
                self.pending.difference_update(pointers)

            def __torch_dispatch__(self, function, _types, args=(), kwargs=None):
                if self.pending:
                    raise AssertionError(
                        "new tensor storage was used before inventory ownership"
                    )
                kwargs = kwargs or {}
                input_pointers = tensor_pointers((args, kwargs))
                output = function(*args, **kwargs)
                function_name = str(function)
                if not function_name.startswith("aten.any."):
                    new_pointers = tensor_pointers(output) - input_pointers
                    self.pending.update(new_pointers)
                    self.allocated.update(new_pointers)
                return output

        class TracedInventory(runtime.RawInventoryV3):
            def __init__(self, required_categories, trace):
                super().__init__(required_categories)
                self.trace = trace

            def own(self, value, category):
                result = super().own(value, category)
                self.trace.register(result)
                return result

        source = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        trace = ImmediateOwnershipTrace()
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.torch = torch
        backend.raw = TracedInventory(("raw", "model_hash_copy"), trace)
        with trace:
            copied = backend._owned_tensor_copy(
                source, "raw", device="cpu", dtype=torch.float32
            )
            support = backend._own_created(copied.sum(), "raw")
            weighted = backend._own_created(copied * support, "raw")
            normalized = backend._own_created(weighted / support, "raw")
            backend.raw.release(normalized)
            backend.raw.release(weighted)
            backend.raw.release(support)
            backend.raw.release(copied)
            backend._tensor_value_sha256(source)
        self.assertFalse(trace.pending)
        self.assertTrue(trace.allocated)
        self.assertEqual(trace.allocated, trace.registered)
        self.assertTrue(backend.raw.receipt(require_all_categories=True)["verified"])

        bad_trace = ImmediateOwnershipTrace()
        with self.assertRaisesRegex(
            AssertionError, "used before inventory ownership"
        ):
            with bad_trace:
                source.clone().clone()

    def test_new_storage_is_zeroized_when_immediate_registration_fails(self):
        class RejectInventory(runtime.RawInventoryV3):
            def own(self, value, category):
                self.rejected = value
                raise runtime.RuntimeV3Error("injected immediate ownership failure")

        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.torch = torch
        backend.raw = RejectInventory(("raw",))
        created = torch.ones(8)
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "injected immediate"):
            backend._own_created(created, "raw")
        self.assertFalse(bool(created.any()))

    def test_processor_external_mapping_mid_claim_failure_scrubs_all_values(self):
        class RejectSecond(runtime.RawInventoryV3):
            def __init__(self):
                super().__init__(("processor",))
                self.calls = 0

            def own(self, value, category):
                self.calls += 1
                if self.calls == 2:
                    raise runtime.RuntimeV3Error(
                        "injected processor mid-mapping failure"
                    )
                return super().own(value, category)

        mapping = {
            "pixel_values": torch.ones(4),
            "mask": torch.ones(3),
            "metadata": torch.ones(2),
        }
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.raw = RejectSecond()
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "mid-mapping"):
            backend._own_external_batch(
                tuple((value, "processor") for value in mapping.values())
            )
        self.assertTrue(all(not bool(value.any()) for value in mapping.values()))
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertFalse(receipt["verified"])
        self.assertEqual(receipt["outstanding_count"], 0)
        self.assertEqual(
            receipt["produced_by_category"], receipt["zeroized_by_category"]
        )

    def test_dino_multi_tensor_output_second_leaf_failure_scrubs_every_leaf(self):
        class RejectSecond(runtime.RawInventoryV3):
            def __init__(self):
                super().__init__(("dino_tokens",))
                self.calls = 0

            def own(self, value, category):
                self.calls += 1
                if self.calls == 2:
                    raise runtime.RuntimeV3Error(
                        "injected DINO second-output failure"
                    )
                return super().own(value, category)

        hidden = torch.ones((1, 5, 8))
        pooler = torch.ones((1, 8))
        output = {
            "last_hidden_state": hidden,
            "pooler_output": pooler,
            "hidden_states": None,
        }
        backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        backend.torch = torch
        backend.raw = RejectSecond()
        with self.assertRaisesRegex(runtime.RuntimeV3Error, "second-output"):
            backend._own_external_tensor_tree(output, "dino_tokens")
        self.assertFalse(bool(hidden.any()))
        self.assertFalse(bool(pooler.any()))
        receipt = backend.raw.receipt(require_all_categories=True)
        self.assertFalse(receipt["verified"])
        self.assertEqual(receipt["outstanding_count"], 0)
        self.assertEqual(
            receipt["produced_by_category"], receipt["zeroized_by_category"]
        )

    @unittest.skipUnless(
        TORCH_NUMPY_BRIDGE_AVAILABLE, "torch/numpy bridge unavailable"
    )
    def test_fake_real_backend_stages_register_and_zeroize_every_category(self):
        class Object:
            pass

        class ModelOutput(dict):
            def __getattr__(self, name):
                return self[name]

        frames8 = tuple(
            np.zeros((32, 32, 3), dtype=np.uint8)
            for _ in range(runtime.PHASES)
        )

        class FakeSAM:
            originals = []

            @classmethod
            def generate(cls, _frame):
                mask = np.zeros((32, 32), dtype=bool, order="F")
                mask[8:16, 8:16] = True
                cls.originals.append(mask)
                return [
                    {
                        "segmentation": mask,
                        "area": 64,
                        "predicted_iou": 0.99,
                        "stability_score": 0.99,
                        "bbox": [8, 8, 8, 8],
                    }
                ]

        class FakeDinoProcessor:
            @staticmethod
            def __call__(**_kwargs):
                return {"pixel_values": torch.ones((1, 3, 224, 224))}

        class FakeDino:
            last_output = None

            @classmethod
            def __call__(cls, **_kwargs):
                hidden = torch.arange(
                    40, dtype=torch.float32
                ).view(1, 5, 8)
                cls.last_output = ModelOutput(
                    last_hidden_state=hidden,
                    pooler_output=hidden[:, 0, :],
                )
                return cls.last_output

        node_categories = (
            "sam_ann_mask_pre_filter",
            "sam_mask_c_contiguous_copy",
            "sam_mask_coordinate_indices",
            "dino_processor_input",
            "dino_tokens",
            "dino_mask_input",
            "dino_mask_resized",
            "dino_mask_cropped",
            "dino_patch_weights",
            "dino_patch_support",
            "dino_pooled_descriptor",
            "dino_pooled_descriptor_cpu",
            "node_signature",
        )
        node_backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        node_backend.torch = torch
        node_backend.np = np
        node_backend.device = "cpu"
        node_backend.raw = runtime.RawInventoryV3(node_categories)
        node_backend.sam = FakeSAM()
        node_backend.dino_processor = FakeDinoProcessor()
        node_backend.dino = FakeDino()
        nodes = node_backend.node(frames8, "cpu_fake")
        self.assertEqual(nodes.cardinalities, (1,) * runtime.PHASES)
        self.assertEqual(len(FakeSAM.originals), runtime.PHASES)
        self.assertTrue(
            all(mask.flags.f_contiguous for mask in FakeSAM.originals)
        )
        self.assertTrue(all(not mask.any() for mask in FakeSAM.originals))
        retained_masks = [
            node.mask
            for phase in nodes.private_payload
            for node in phase
        ]
        self.assertEqual(len(retained_masks), runtime.PHASES)
        self.assertTrue(all(mask.flags.c_contiguous for mask in retained_masks))
        self.assertTrue(all(not mask.flags.f_contiguous for mask in retained_masks))
        self.assertTrue(all(int(mask.sum()) == 64 for mask in retained_masks))
        self.assertEqual(
            node_backend.raw.registered["sam_ann_mask_pre_filter"],
            runtime.PHASES,
        )
        self.assertEqual(
            node_backend.raw.registered["sam_mask_c_contiguous_copy"],
            runtime.PHASES,
        )
        self.assertIsNotNone(FakeDino.last_output)
        self.assertFalse(bool(FakeDino.last_output["last_hidden_state"].any()))
        self.assertFalse(bool(FakeDino.last_output["pooler_output"].any()))
        self.assertEqual(
            node_backend.raw.observed["dino_model_output_unique_storages"],
            runtime.PHASES,
        )
        node_backend.raw.scrub_all()
        node_receipt = node_backend.raw.receipt(require_all_categories=True)
        self.assertTrue(node_receipt["verified"])
        self.assertEqual(
            node_receipt["registered_by_category"],
            node_receipt["zeroized_by_category"],
        )

        phases = []
        for phase in range(runtime.PHASES):
            mask = np.zeros((32, 32), dtype=bool)
            mask[8:16, 8:16] = True
            phases.append(
                (
                    graph.AnonymousNodeV3(
                        mask=mask,
                        descriptor=np.arange(8, dtype=np.float64),
                        area_fraction=float(mask.mean()),
                        centroid_xy=(11.5 / 31.0, 11.5 / 31.0),
                        track_id=0,
                    ),
                )
            )
        private_payload = tuple(phases)
        motion_nodes = runtime.NodeSketch(
            signature=graph.canonical_node_signature(private_payload),
            cardinalities=(1,) * runtime.PHASES,
            mechanically_valid_phases=runtime.PHASES,
            dustbin_used=True,
            private_payload=private_payload,
        )

        class FakeCoTracker:
            last_tracks = None
            last_visible = None

            @staticmethod
            def __call__(_video, **_kwargs):
                tracks = torch.full(
                    (1, runtime.PHASES, 2, 2), 11.0, dtype=torch.float32
                )
                visible = torch.ones(
                    (1, runtime.PHASES, 2), dtype=torch.bool
                )
                FakeCoTracker.last_tracks = tracks
                FakeCoTracker.last_visible = visible
                return tracks, visible

        motion_categories = (
            "cotracker_video",
            "cotracker_tracks",
            "cotracker_visibility",
            "cotracker_coordinates_cpu",
            "cotracker_visibility_cpu",
            "cotracker_group_coordinates",
            "cotracker_group_visibility",
            "track_signature",
            "edge_signature",
            "drop_edge_signature",
        )
        motion_backend = runtime.RealFrozenBackend.__new__(
            runtime.RealFrozenBackend
        )
        motion_backend.torch = torch
        motion_backend.np = np
        motion_backend.device = "cpu"
        motion_backend.raw = runtime.RawInventoryV3(motion_categories)
        motion_backend.cotracker = FakeCoTracker()
        motion = motion_backend.motion(frames8, "cpu_fake", motion_nodes)
        self.assertEqual(motion.assigned_track_count, 1)
        self.assertEqual(motion.assigned_point_count, 2)
        self.assertTrue(FakeCoTracker.last_tracks.is_inference())
        self.assertTrue(FakeCoTracker.last_visible.is_inference())
        self.assertFalse(bool(FakeCoTracker.last_tracks.any()))
        self.assertFalse(bool(FakeCoTracker.last_visible.any()))
        motion_backend.raw.scrub_all()
        motion_receipt = motion_backend.raw.receipt(require_all_categories=True)
        self.assertTrue(motion_receipt["verified"])
        self.assertEqual(
            motion_receipt["registered_by_category"],
            motion_receipt["zeroized_by_category"],
        )

        class FakeVJEPAProcessor:
            @staticmethod
            def __call__(**_kwargs):
                return {
                    "pixel_values_videos": torch.ones(
                        (1, 16, 3, 4, 4), dtype=torch.float32
                    )
                }

        class FakeVJEPA:
            last_output = None

            def __init__(self):
                self.config = Object()
                self.config.tubelet_size = 2
                self.config.image_size = 4
                self.config.patch_size = 2

            @classmethod
            def __call__(cls, **_kwargs):
                cls.last_output = ModelOutput(
                    last_hidden_state=torch.arange(
                        1 * 32 * 20, dtype=torch.float32
                    ).view(1, 32, 20),
                    masked_hidden_state=torch.ones((1, 32, 20)),
                    predictor_output=ModelOutput(
                        last_hidden_state=torch.full((1, 32, 20), 2.0),
                        target_hidden_state=torch.full((1, 32, 20), 3.0),
                    ),
                )
                return cls.last_output

        phase_categories = (
            "vjepa_processor_input",
            "vjepa_hidden",
            "vjepa_phase_signature",
        )
        phase_backend = runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend)
        phase_backend.torch = torch
        phase_backend.np = np
        phase_backend.device = "cpu"
        phase_backend.raw = runtime.RawInventoryV3(phase_categories)
        phase_backend.vjepa_processor = FakeVJEPAProcessor()
        phase_backend.vjepa = FakeVJEPA()
        phase = phase_backend.phase(frames8 + frames8, "cpu_fake")
        self.assertEqual(len(phase.signature), runtime.PHASES * 16)
        self.assertIsNotNone(FakeVJEPA.last_output)
        self.assertTrue(
            all(
                value.is_inference()
                for value in (
                    FakeVJEPA.last_output["last_hidden_state"],
                    FakeVJEPA.last_output["masked_hidden_state"],
                    FakeVJEPA.last_output["predictor_output"]["last_hidden_state"],
                    FakeVJEPA.last_output["predictor_output"]["target_hidden_state"],
                )
            )
        )
        self.assertTrue(
            all(
                not bool(value.any())
                for value in (
                    FakeVJEPA.last_output["last_hidden_state"],
                    FakeVJEPA.last_output["masked_hidden_state"],
                    FakeVJEPA.last_output["predictor_output"]["last_hidden_state"],
                    FakeVJEPA.last_output["predictor_output"]["target_hidden_state"],
                )
            )
        )
        self.assertEqual(
            phase_backend.raw.observed[
                "vjepa_model_output_unique_storages"
            ],
            4,
        )
        phase_backend.raw.scrub_all()
        phase_receipt = phase_backend.raw.receipt(require_all_categories=True)
        self.assertTrue(phase_receipt["verified"])
        self.assertEqual(
            phase_receipt["registered_by_category"],
            phase_receipt["zeroized_by_category"],
        )

    def test_torch_storage_alias_double_ownership_is_rejected(self):
        inventory = runtime.RawInventoryV3(("raw",))
        tensor = torch.ones(8)
        produce_and_own(inventory, tensor, "raw")
        with self.assertRaises(runtime.RuntimeV3Error):
            produce_and_own(inventory, tensor[1:], "raw")
        inventory.scrub_all()
        self.assertFalse(bool(tensor.any()))
        self.assertFalse(inventory.receipt(require_all_categories=True)["verified"])


if __name__ == "__main__":
    unittest.main()
