from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from torch import nn

    import saic_joint_composition_v1 as joint
    import saic_online_motion_field_v1 as online_motion
    import saic_source_anchor_adapter_v1 as source_anchor
    import saic_temporal_action_operator_v2 as temporal_action

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    joint = None  # type: ignore[assignment]
    online_motion = None  # type: ignore[assignment]
    source_anchor = None  # type: ignore[assignment]
    temporal_action = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _Attention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)
            self.to_v = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.attn1 = _Attention(hidden)
            self.attn2 = _Attention(hidden)
            self.gradient_checkpointing = False


    class _Transformer(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(16, hidden, kernel_size=(1, 2, 2))
            self.blocks = nn.ModuleList(
                [_Block(hidden) for _ in range(source_anchor.TOTAL_BLOCKS_1P3B)]
            )
            self.gradient_checkpointing = False
            self.is_gradient_checkpointing = False

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


def _source_state(hidden: int = 8) -> dict[str, "torch.Tensor"]:
    state: dict[str, torch.Tensor] = {}
    for index in source_anchor.SOURCE_ANCHOR_BLOCK_INDICES:
        for projection in ("attn1.to_q", "attn1.to_out.0"):
            state[f"blocks.{index}.{projection}.state_down.weight"] = torch.full(
                (source_anchor.SOURCE_ANCHOR_RANK, hidden),
                0.01 * (index + 1),
                dtype=torch.float32,
            )
            state[f"blocks.{index}.{projection}.output_up.weight"] = torch.full(
                (hidden, source_anchor.SOURCE_ANCHOR_RANK),
                0.001 * (index + 1),
                dtype=torch.float32,
            )
    return state


def _motion_state(hidden: int = 8) -> dict[str, "torch.Tensor"]:
    state: dict[str, torch.Tensor] = {}
    for index in temporal_action.ACTION_BLOCK_INDICES:
        for projection in ("attn2.to_q", "attn2.to_out.0"):
            prefix = f"blocks.{index}.{projection}"
            state[f"{prefix}.state_down.weight"] = torch.full(
                (temporal_action.ACTION_OPERATOR_RANK, hidden),
                0.002 * (index + 1),
                dtype=torch.float32,
            )
            state[f"{prefix}.phase_gate.weight"] = torch.full(
                (
                    temporal_action.ACTION_OPERATOR_RANK,
                    online_motion.PHASE_CODE_DIM,
                ),
                0.003 * (index + 1),
                dtype=torch.float32,
            )
            state[f"{prefix}.output_up.weight"] = torch.full(
                (hidden, temporal_action.ACTION_OPERATOR_RANK),
                0.004 * (index + 1),
                dtype=torch.float32,
            )
    return state


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICJointCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(809)
        self.model = _Transformer(hidden=8)
        self.model.requires_grad_(False)
        self.original_parameter_rows = tuple(
            (name, id(parameter))
            for name, parameter in self.model.named_parameters(
                remove_duplicate=False
            )
        )
        self.original_module_rows = tuple(
            (name, id(module))
            for name, module in self.model.named_modules(remove_duplicate=False)
        )
        self.handle = None

    def tearDown(self) -> None:
        if self.handle is not None and not self.handle.restored:
            # Tests that deliberately tamper cannot pass the fail-closed public
            # restore path.  Repair their local mutation in the test itself.
            self.handle.restore()

    def _install(self, *, mode: str = joint.STAGE_B_TRAIN, load: bool = True):
        self.handle = joint.install_saic_joint_composition(
            self.model,
            mode=mode,
            source_state=_source_state() if load else None,
            motion_state=_motion_state() if load else None,
        )
        return self.handle

    def test_legacy_exclusive_trainable_contract_reproduces_conflict(self) -> None:
        source = source_anchor.install_saic_source_anchor_adapter(self.model)
        try:
            self.assertTrue(source.base_parameters_frozen())
            with self.assertRaisesRegex(
                temporal_action.SAICTemporalActionOperatorError,
                "freeze the complete Bernini transformer",
            ):
                temporal_action.install_saic_temporal_action_operator(self.model)
        finally:
            source.restore()

    def test_joint_stage_b_install_load_receipts_and_strict_union(self) -> None:
        handle = self._install()
        audit = handle.audit()
        self.assertTrue(audit["strict_parameter_id_union"])
        self.assertTrue(audit["strict_module_id_union"])
        self.assertTrue(audit["adapter_parameter_sets_disjoint"])
        self.assertTrue(audit["only_motion_trainable"])
        self.assertFalse(audit["end_to_end_training_authorized"])
        self.assertFalse(audit["gradient_state_rollback_authorized"])
        self.assertFalse(audit["optimizer_or_scaler_state_rollback_authorized"])
        self.assertFalse(audit["rng_state_rollback_authorized"])
        self.assertEqual(
            audit["source_parameter_count"],
            len(source_anchor.SOURCE_ANCHOR_BLOCK_INDICES) * 4,
        )
        self.assertEqual(
            audit["motion_parameter_count"],
            len(temporal_action.ACTION_BLOCK_INDICES) * 6,
        )
        self.assertEqual(
            {id(parameter) for parameter in self.model.parameters() if parameter.requires_grad},
            {id(parameter) for _, parameter in handle.motion_parameters},
        )
        receipt = handle.receipt()
        self.assertIsNotNone(receipt["source_load_receipt_digest"])
        self.assertIsNotNone(receipt["motion_load_receipt_digest"])
        self.assertFalse(receipt["native_runtime_lifecycle_managed"])
        self.assertEqual(
            receipt["native_runtime_status"],
            joint.NATIVE_RUNTIME_STATUS,
        )
        self.assertFalse(receipt["scoped_motion_parameter_update_authorized"])
        self.assertFalse(receipt["parameter_update_authorized"])
        self.assertFalse(receipt["optimizer_parameter_access_authorized"])
        self.assertFalse(receipt["end_to_end_training_authorized"])
        self.assertIn("process_local_public_api_integrity", receipt["classification"])
        self.assertIn("no_parameter_update", receipt["classification"])
        self.assertTrue(receipt["process_local_registry_integrity_root"])
        self.assertFalse(
            receipt["arbitrary_same_process_reflection_resistance_claim"]
        )
        self.assertFalse(receipt["gradient_state_rollback_authorized"])
        self.assertFalse(receipt["optimizer_or_scaler_state_rollback_authorized"])
        self.assertFalse(receipt["rng_state_rollback_authorized"])
        self.assertEqual(
            receipt["motion_update_rollback_scope"],
            "none_non_authoritative_lifecycle",
        )
        self.assertFalse(receipt["training_authorized"])
        self.assertEqual(
            receipt["training_authorized_legacy_field_semantics"],
            "end_to_end_training_workflow_only",
        )

    def test_inference_gauge_freezes_both_adapters(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        audit = handle.audit()
        self.assertTrue(audit["all_parameters_frozen"])
        self.assertEqual(audit["trainable_parameter_count"], 0)
        self.assertFalse(audit["scoped_motion_parameter_update_authorized"])
        self.assertFalse(audit["end_to_end_training_authorized"])
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "does not authorize optimizer"
        ):
            handle.trainable_named_parameters()
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "does not authorize parameter updates"
        ):
            with handle.motion_update():
                pass

    def test_unknown_parameter_and_unknown_trainable_fail_closed(self) -> None:
        handle = self._install()
        self.model.unknown_joint_parameter = nn.Parameter(torch.ones(1))
        try:
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError,
                "parameter name-to-object-ID binding changed",
            ):
                handle.audit()
        finally:
            del self.model.unknown_joint_parameter
        handle.audit()

        vendor_parameter = handle.vendor_parameter_rows[0][1]
        vendor_parameter.requires_grad_(True)
        try:
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "trainable gauge differs"
            ):
                handle.audit()
        finally:
            vendor_parameter.requires_grad_(False)
        handle.audit()

    def test_parameter_alias_is_rejected(self) -> None:
        handle = self._install()
        wrapper = self.model.blocks[0].attn2.to_q
        original_output = wrapper.output_up.weight
        wrapper.output_up.weight = wrapper.state_down.weight
        try:
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "parameter alias detected"
            ):
                handle.audit()
        finally:
            wrapper.output_up.weight = original_output
        handle.audit()

    def test_same_id_set_module_path_swap_is_rejected(self) -> None:
        handle = self._install()
        left = self.model.blocks[0].attn1.to_out[1]
        right = self.model.blocks[1].attn1.to_out[1]
        self.assertIsInstance(left, nn.Identity)
        self.assertIsInstance(right, nn.Identity)
        self.model.blocks[0].attn1.to_out[1] = right
        self.model.blocks[1].attn1.to_out[1] = left
        try:
            self.assertEqual(
                {id(module) for module in self.model.modules()},
                set(handle.allowed_module_ids),
            )
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError,
                "active module name-to-object-ID binding changed",
            ):
                handle.audit()
        finally:
            self.model.blocks[0].attn1.to_out[1] = left
            self.model.blocks[1].attn1.to_out[1] = right
        handle.audit()

    def test_same_id_set_parameter_path_swap_is_rejected(self) -> None:
        handle = self._install()
        left_module = self.model.blocks[0].attn1.to_k
        right_module = self.model.blocks[1].attn1.to_k
        left = left_module.weight
        right = right_module.weight
        left_module.weight = right
        right_module.weight = left
        try:
            self.assertEqual(
                {id(parameter) for parameter in self.model.parameters()},
                set(handle.allowed_parameter_ids),
            )
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError,
                "active parameter name-to-object-ID binding changed",
            ):
                handle.audit()
        finally:
            left_module.weight = left
            right_module.weight = right
        handle.audit()

    def test_adapter_identical_byte_storage_rebinding_is_rejected(self) -> None:
        handle = self._install()
        for label, rows, message in (
            (
                "source",
                handle.source_parameters,
                "source parameter binding changed",
            ),
            (
                "motion",
                handle.motion_parameters,
                "motion parameter binding changed",
            ),
        ):
            parameter = rows[0][1]
            original_data = parameter.detach()
            original_pointer = joint._storage_pointer(parameter)
            original_bytes = parameter.detach().clone()
            parameter.data = parameter.detach().clone()
            try:
                self.assertNotEqual(
                    joint._storage_pointer(parameter), original_pointer, msg=label
                )
                self.assertTrue(torch.equal(parameter, original_bytes), msg=label)
                with self.assertRaisesRegex(
                    joint.SAICJointCompositionError, message
                ):
                    handle.audit()
            finally:
                parameter.data = original_data
            handle.audit()

    def test_parameter_and_optimizer_access_are_never_authorized(self) -> None:
        handle = self._install()
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "does not authorize optimizer"
        ):
            handle.trainable_named_parameters()
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "does not authorize parameter updates"
        ):
            with handle.motion_update():
                pass
        audit = handle.audit()
        self.assertFalse(audit["parameter_update_authorized"])
        self.assertFalse(audit["optimizer_parameter_access_authorized"])

    def test_data_byte_tamper_is_detected_without_version_change(self) -> None:
        handle = self._install()
        birth_receipt_digest = handle.receipt()["digest"]
        for label, rows, message in (
            ("vendor", handle.vendor_parameter_rows, "vendor parameter bytes changed"),
            ("source", handle.source_parameters, "source parameter bytes changed"),
            ("motion", handle.motion_parameters, "motion parameter bytes changed"),
        ):
            parameter = rows[0][1]
            before = parameter.detach().clone()
            before_version = int(parameter._version)
            parameter.data.add_(1.0)
            try:
                self.assertEqual(int(parameter._version), before_version, msg=label)
                with self.assertRaisesRegex(
                    joint.SAICJointCompositionError, message
                ):
                    handle.audit()
                with self.assertRaises(joint.SAICJointCompositionError):
                    handle.receipt()
            finally:
                parameter.data.copy_(before)
            self.assertEqual(int(parameter._version), before_version, msg=label)
            handle.audit()
            self.assertEqual(handle.receipt()["digest"], birth_receipt_digest)

    def test_public_state_hash_and_binding_resign_attempt_is_rejected(self) -> None:
        handle = self._install()
        for label, rows, state_field, binding_field in (
            (
                "source",
                handle.source_parameters,
                "source_state_sha256",
                "source_parameter_bindings",
            ),
            (
                "motion",
                handle.motion_parameters,
                "motion_state_sha256",
                "motion_parameter_bindings",
            ),
        ):
            parameter = rows[0][1]
            before = parameter.detach().clone()
            original_state = getattr(handle, state_field)
            original_bindings = getattr(handle, binding_field)
            parameter.data.add_(0.5)
            setattr(handle, state_field, joint._state_sha256(rows, label=label))
            setattr(handle, binding_field, joint._parameter_binding_map(rows))
            try:
                with self.assertRaisesRegex(
                    joint.SAICJointCompositionError, "private birth seal"
                ):
                    handle.audit()
            finally:
                parameter.data.copy_(before)
                setattr(handle, state_field, original_state)
                setattr(handle, binding_field, original_bindings)
            handle.audit()

    def test_forged_install_receipts_cannot_resign_public_provenance(self) -> None:
        handle = self._install()
        for field in ("source_install_receipt", "motion_install_receipt"):
            original = getattr(handle, field)
            forged = dict(original)
            forged.pop("digest")
            forged["classification"] = "forged_same_process_public_receipt"
            forged["digest"] = joint._object_sha256(forged)
            setattr(handle, field, forged)
            try:
                with self.assertRaisesRegex(
                    joint.SAICJointCompositionError,
                    "public joint handle fields differ from private birth seal",
                ):
                    handle.audit()
                with self.assertRaises(joint.SAICJointCompositionError):
                    handle.receipt()
            finally:
                setattr(handle, field, original)
            handle.audit()

    def test_erased_or_forged_load_receipt_presence_is_rejected(self) -> None:
        handle = self._install()
        for field in ("source_load_receipt", "motion_load_receipt"):
            original = getattr(handle, field)
            self.assertIsNotNone(original)
            setattr(handle, field, None)
            try:
                with self.assertRaisesRegex(
                    joint.SAICJointCompositionError,
                    "public joint handle fields differ from private birth seal",
                ):
                    handle.audit()
                with self.assertRaises(joint.SAICJointCompositionError):
                    handle.receipt()
            finally:
                setattr(handle, field, original)

            forged = dict(original)
            forged.pop("digest")
            forged["state_tensor_sha256"] = "f" * 64
            forged["digest"] = joint._object_sha256(forged)
            setattr(handle, field, forged)
            try:
                with self.assertRaisesRegex(
                    joint.SAICJointCompositionError,
                    "public joint handle fields differ from private birth seal",
                ):
                    handle.audit()
            finally:
                setattr(handle, field, original)
            handle.audit()

    def test_copied_handle_is_not_a_registry_issued_identity(self) -> None:
        handle = self._install()
        forged = copy.copy(handle)
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "not issued by the private registry"
        ):
            forged.audit()

    def test_mutable_mode_cannot_escalate_inference_to_update_authority(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        parameter = handle.motion_parameters[0][1]
        handle.mode = joint.STAGE_B_TRAIN
        parameter.requires_grad_(True)
        try:
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "private birth seal"
            ):
                handle.audit()
            with self.assertRaises(joint.SAICJointCompositionError):
                handle.receipt()
            with self.assertRaises(joint.SAICJointCompositionError):
                handle.trainable_named_parameters()
            with self.assertRaises(joint.SAICJointCompositionError):
                with handle.motion_update():
                    pass
        finally:
            parameter.requires_grad_(False)
            handle.mode = joint.INFERENCE
        audit = handle.audit()
        self.assertEqual(audit["mode"], joint.INFERENCE)
        self.assertFalse(audit["parameter_update_authorized"])

    def test_runtime_lease_blocks_restore_before_child_mutation(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        order: list[str] = []
        original_motion_restore = handle.motion_handle.restore
        original_source_restore = handle.source_handle.restore

        def motion_restore() -> None:
            order.append("motion")
            original_motion_restore()

        def source_restore() -> None:
            order.append("source")
            original_source_restore()

        with mock.patch.object(
            handle.motion_handle, "restore", side_effect=motion_restore
        ), mock.patch.object(
            handle.source_handle, "restore", side_effect=source_restore
        ):
            lease = handle.acquire_runtime_lease()
            lease_audit = lease.audit()
            self.assertTrue(
                lease_audit["composition_restore_blocked_before_mutation"]
            )
            self.assertFalse(lease_audit["parameter_update_authorized"])
            self.assertFalse(lease_audit["end_to_end_training_authorized"])
            self.assertTrue(handle.audit()["runtime_lease_active"])
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "runtime lease is active"
            ):
                handle.restore()
            self.assertEqual(order, [])
            self.assertFalse(handle.motion_handle.restored)
            self.assertFalse(handle.source_handle.restored)
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "runtime lease is active"
            ):
                handle.acquire_runtime_lease()
            release = lease.release()
            self.assertTrue(release["released"])
            self.assertFalse(handle.audit()["runtime_lease_active"])
            handle.restore()
        self.assertEqual(order, ["motion", "source"])

    def test_runtime_lease_is_inference_only(self) -> None:
        handle = self._install(mode=joint.STAGE_B_TRAIN)
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError,
            "immutable inference birth mode",
        ):
            handle.acquire_runtime_lease()

    def test_runtime_lease_acquisition_is_private_immutable_and_sealed(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        lease = handle.acquire_runtime_lease()
        self.assertFalse(hasattr(lease, "pre_lease_audit_digest"))
        acquisition = lease._acquisition
        original_pre_digest = acquisition.pre_lease_audit_digest
        original_digest = acquisition.digest
        with self.assertRaises(FrozenInstanceError):
            acquisition.pre_lease_audit_digest = "forged"  # type: ignore[misc]

        # Even Python's explicit frozen-dataclass bypass cannot produce a valid
        # mutable re-seal: the independently retained acquisition seal rejects it.
        object.__setattr__(acquisition, "pre_lease_audit_digest", "forged")
        object.__setattr__(
            acquisition,
            "digest",
            joint._object_sha256(acquisition._payload()),
        )
        try:
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "private registry birth"
            ):
                lease.audit()
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "private registry birth"
            ):
                handle.audit()
        finally:
            object.__setattr__(
                acquisition, "pre_lease_audit_digest", original_pre_digest
            )
            object.__setattr__(acquisition, "digest", original_digest)
        lease.audit()
        lease.release()

    def test_runtime_lease_forge_and_cross_handle_rebind_are_rejected(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        lease = handle.acquire_runtime_lease()
        forged = joint.SAICJointCompositionRuntimeLease(
            _handle=handle,
            _acquisition=lease._acquisition,
            _construction_token=joint._RUNTIME_LEASE_MINT,
        )
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "not issued by the private registry"
        ):
            forged.audit()

        other_model = _Transformer(hidden=8)
        other_model.requires_grad_(False)
        other_handle = joint.install_saic_joint_composition(
            other_model,
            mode=joint.INFERENCE,
            source_state=_source_state(),
            motion_state=_motion_state(),
        )
        original_handle = lease._handle
        lease._handle = other_handle
        try:
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "cross-handle"
            ):
                lease.audit()
            with self.assertRaisesRegex(
                joint.SAICJointCompositionError, "cross-handle"
            ):
                handle.audit()
        finally:
            lease._handle = original_handle
        lease.audit()
        lease.release()
        other_handle.restore()

    def test_clearing_public_lease_provenance_cannot_enable_restore(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        lease = handle.acquire_runtime_lease()
        calls: list[str] = []
        original_motion_restore = handle.motion_handle.restore
        original_source_restore = handle.source_handle.restore

        def motion_restore() -> None:
            calls.append("motion")
            original_motion_restore()

        def source_restore() -> None:
            calls.append("source")
            original_source_restore()

        # These attacker-created attributes mimic the obsolete public
        # provenance fields.  Registry state remains the sole lease authority.
        handle._runtime_lease_acquisition = None
        handle._runtime_lease_acquisition_digest = None
        handle._runtime_lease_generation = 0
        try:
            with mock.patch.object(
                handle.motion_handle, "restore", side_effect=motion_restore
            ), mock.patch.object(
                handle.source_handle, "restore", side_effect=source_restore
            ):
                with self.assertRaisesRegex(
                    joint.SAICJointCompositionError, "runtime lease is active"
                ):
                    handle.restore()
            self.assertEqual(calls, [])
        finally:
            del handle._runtime_lease_acquisition
            del handle._runtime_lease_acquisition_digest
            del handle._runtime_lease_generation
        lease.audit()
        lease.release()

    def test_runtime_lease_release_failure_retains_exclusive_lease(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        lease = handle.acquire_runtime_lease()
        acquisition = lease._acquisition
        original_audit = handle.audit
        audit_calls = 0

        def fail_only_post_release_audit():
            nonlocal audit_calls
            audit_calls += 1
            if audit_calls == 2:
                raise RuntimeError("post-release-audit")
            return original_audit()

        with mock.patch.object(
            handle, "audit", side_effect=fail_only_post_release_audit
        ):
            with self.assertRaisesRegex(RuntimeError, "post-release-audit"):
                lease.release()
        self.assertFalse(lease.released)
        self.assertIs(lease._acquisition, acquisition)
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "runtime lease is active"
        ):
            handle.restore()
        lease.audit()
        lease.release()

    def test_runtime_lease_cleanup_error_does_not_mask_body_corruption(self) -> None:
        handle = self._install(mode=joint.INFERENCE)
        parameter = handle.source_parameters[0][1]
        original_data = parameter.detach()
        lease = None
        with self.assertRaisesRegex(ValueError, "body-root") as caught:
            with handle.runtime_lease() as acquired:
                lease = acquired
                parameter.data = parameter.detach().clone()
                raise ValueError("body-root")
        cleanup = getattr(
            caught.exception, "saic_runtime_lease_release_error", None
        )
        self.assertIsInstance(cleanup, joint.SAICJointCompositionError)
        self.assertIsNotNone(lease)
        self.assertFalse(lease.released)
        parameter.data = original_data
        lease.audit()
        lease.release()

    def test_joint_context_restore_error_does_not_mask_body_exception(self) -> None:
        handle = self._install()
        with self.assertRaisesRegex(ValueError, "body-root") as caught:
            with handle:
                self.model.restore_unknown = nn.Parameter(
                    torch.ones(1), requires_grad=False
                )
                raise ValueError("body-root")
        cleanup = getattr(caught.exception, "saic_joint_restore_error", None)
        self.assertIsInstance(cleanup, joint.SAICJointCompositionRestoreError)
        del self.model.restore_unknown
        handle.restore()

    def test_restore_is_motion_then_source_and_recovers_exact_vendor_tree(self) -> None:
        handle = self._install(load=False)
        receipt = handle.restore()
        self.assertTrue(receipt["original_vendor_parameter_ids_restored"])
        self.assertTrue(receipt["original_vendor_module_ids_restored"])
        self.assertEqual(
            tuple(
                (name, id(parameter))
                for name, parameter in self.model.named_parameters(
                    remove_duplicate=False
                )
            ),
            self.original_parameter_rows,
        )
        self.assertEqual(
            tuple(
                (name, id(module))
                for name, module in self.model.named_modules(remove_duplicate=False)
            ),
            self.original_module_rows,
        )
        self.assertTrue(handle.motion_handle.restored)
        self.assertTrue(handle.source_handle.restored)
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "was restored"
        ):
            handle.audit()

    def test_restore_motion_mutates_then_raises_but_source_cleanup_completes(self) -> None:
        handle = self._install(load=False)
        motion_restore = handle.motion_handle.restore
        source_restore = handle.source_handle.restore
        calls = {"motion": 0, "source": 0}

        def motion_then_error() -> None:
            calls["motion"] += 1
            motion_restore()
            raise RuntimeError("motion-after-mutation")

        def counted_source_restore() -> None:
            calls["source"] += 1
            source_restore()

        with mock.patch.object(
            handle.motion_handle, "restore", side_effect=motion_then_error
        ), mock.patch.object(
            handle.source_handle, "restore", side_effect=counted_source_restore
        ):
            with self.assertRaises(
                joint.SAICJointCompositionRestoreError
            ) as caught:
                handle.restore()
        self.assertTrue(handle.restored)
        self.assertEqual(calls, {"motion": 1, "source": 1})
        self.assertIsInstance(caught.exception.root_cause, RuntimeError)
        self.assertEqual(
            caught.exception.receipt["root_cause"]["stage"], "restore_motion"
        )
        self.assertEqual(
            caught.exception.receipt["final_restore_state"],
            "complete_vendor_restored",
        )
        self.assertTrue(caught.exception.receipt["vendor_verified"])

    def test_restore_partial_motion_mutation_is_ambiguous_and_stops_source(self) -> None:
        handle = self._install(load=False)
        motion_restore = handle.motion_handle.restore
        source_restore = handle.source_handle.restore
        index, _ = handle.motion_handle.q_wrappers[0]
        original_q = dict(handle.motion_handle.original_q)[index]
        source_calls = 0

        def partial_motion_restore() -> None:
            self.model.blocks[index].attn2.to_q = original_q
            raise RuntimeError("motion-partial-mutation")

        def counted_source_restore() -> None:
            nonlocal source_calls
            source_calls += 1
            source_restore()

        with mock.patch.object(
            handle.motion_handle, "restore", side_effect=partial_motion_restore
        ), mock.patch.object(
            handle.source_handle, "restore", side_effect=counted_source_restore
        ):
            with self.assertRaises(
                joint.SAICJointCompositionRestoreError
            ) as caught:
                handle.restore()
        self.assertEqual(source_calls, 0)
        self.assertFalse(caught.exception.receipt["retryable"])
        self.assertEqual(
            caught.exception.receipt["final_restore_state"],
            "ambiguous_registered_slots",
        )

        # Test-only repair proves subsequent cleanup does not require invoking
        # the source child before the motion family is fully original again.
        motion_restore()
        handle.restore()

    def test_restore_after_motion_union_failure_cleans_source_and_retries_vendor(self) -> None:
        handle = self._install(load=False)
        motion_restore = handle.motion_handle.restore
        source_restore = handle.source_handle.restore
        calls = {"motion": 0, "source": 0}

        def motion_then_inject_unknown() -> None:
            calls["motion"] += 1
            motion_restore()
            self.model.restore_unknown = nn.Parameter(
                torch.ones(1), requires_grad=False
            )

        def counted_source_restore() -> None:
            calls["source"] += 1
            source_restore()

        with mock.patch.object(
            handle.motion_handle,
            "restore",
            side_effect=motion_then_inject_unknown,
        ), mock.patch.object(
            handle.source_handle, "restore", side_effect=counted_source_restore
        ):
            with self.assertRaises(
                joint.SAICJointCompositionRestoreError
            ) as caught:
                handle.restore()
            first_receipt = caught.exception.receipt
            self.assertFalse(handle.restored)
            self.assertEqual(
                first_receipt["final_restore_state"],
                "source_and_motion_removed_vendor_unverified",
            )
            self.assertEqual(calls, {"motion": 1, "source": 1})
            self.assertIn(
                "audit_after_motion",
                [row["stage"] for row in first_receipt["errors"]],
            )
            self.assertIn(
                "verify_vendor_identity",
                [row["stage"] for row in first_receipt["errors"]],
            )
            del self.model.restore_unknown
            receipt = handle.restore()
        self.assertTrue(handle.restored)
        self.assertEqual(calls, {"motion": 1, "source": 1})
        self.assertEqual(
            receipt["initial_restore_state"],
            "source_and_motion_removed_vendor_unverified",
        )

    def test_restore_source_failure_retries_without_reinvoking_motion(self) -> None:
        handle = self._install(load=False)
        motion_restore = handle.motion_handle.restore
        source_restore = handle.source_handle.restore
        calls = {"motion": 0, "source": 0}

        def counted_motion_restore() -> None:
            calls["motion"] += 1
            motion_restore()

        def transient_source_restore() -> None:
            calls["source"] += 1
            if calls["source"] == 1:
                raise RuntimeError("source-before-mutation")
            source_restore()

        with mock.patch.object(
            handle.motion_handle, "restore", side_effect=counted_motion_restore
        ), mock.patch.object(
            handle.source_handle, "restore", side_effect=transient_source_restore
        ):
            with self.assertRaises(
                joint.SAICJointCompositionRestoreError
            ) as caught:
                handle.restore()
            self.assertFalse(handle.restored)
            self.assertEqual(
                caught.exception.receipt["final_restore_state"],
                "motion_removed_source_active",
            )
            receipt = handle.restore()
        self.assertTrue(receipt["complete"])
        self.assertEqual(calls, {"motion": 1, "source": 2})

    def test_restore_source_mutates_then_raises_still_verifies_vendor(self) -> None:
        handle = self._install(load=False)
        source_restore = handle.source_handle.restore
        source_calls = 0

        def source_then_error() -> None:
            nonlocal source_calls
            source_calls += 1
            source_restore()
            raise RuntimeError("source-after-mutation")

        with mock.patch.object(
            handle.source_handle, "restore", side_effect=source_then_error
        ):
            with self.assertRaises(
                joint.SAICJointCompositionRestoreError
            ) as caught:
                handle.restore()
        self.assertTrue(handle.restored)
        self.assertEqual(source_calls, 1)
        self.assertTrue(caught.exception.receipt["vendor_verified"])
        self.assertEqual(
            caught.exception.receipt["root_cause"]["stage"], "restore_source"
        )
        self.assertIsInstance(caught.exception.root_cause, RuntimeError)

    def test_bad_state_key_closure_rolls_back_to_exact_vendor_tree(self) -> None:
        bad_motion = _motion_state()
        bad_motion["unexpected"] = torch.zeros(1, dtype=torch.float32)
        with self.assertRaisesRegex(
            joint.SAICJointCompositionError, "state key closure differs"
        ):
            joint.install_saic_joint_composition(
                self.model,
                mode=joint.STAGE_B_TRAIN,
                source_state=_source_state(),
                motion_state=bad_motion,
            )
        self.assertEqual(
            tuple(
                (name, id(parameter))
                for name, parameter in self.model.named_parameters(
                    remove_duplicate=False
                )
            ),
            self.original_parameter_rows,
        )
        self.assertEqual(
            tuple(
                (name, id(module))
                for name, module in self.model.named_modules(remove_duplicate=False)
            ),
            self.original_module_rows,
        )
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
