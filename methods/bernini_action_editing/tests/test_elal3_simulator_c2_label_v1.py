from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = METHOD_ROOT / "elal3_simulator_c2_label_v1.py"
PACKET_ROOT = (
    REPO_ROOT / "md/action_editing/20260817_box/simulator_gt_canary_v1"
)
AUTHORITY_PATH = (
    REPO_ROOT
    / "md/action_editing/20260817_box/evidence/"
    "elal3_c2_simulator_optimizer_diagnostic_authority_v1.json"
)
CONTRACT_PATH = (
    REPO_ROOT
    / "md/action_editing/20260817_box/evidence/"
    "elal3_c2_role_binding_experiment_contract_v1.json"
)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    import elal3_simulator_c2_label_v1 as subject

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    subject = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class ELAL3SimulatorC2LabelStaticTests(unittest.TestCase):
    def test_source_parses_and_closes_narrow_scope(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for fragment in (
            'LABEL_SCHEMA_VERSION = "elal3-simulator-c2-oracle-q-label-v1"',
            '"c2-three-entity-blocking-response"',
            '"c2-three-entity-handover-occlusion"',
            'EXPECTED_EXTERNAL_AUTHORITY_SHA256 = (',
            'EXPECTED_EXPERIMENT_CONTRACT_SHA256 = (',
            '"92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"',
            'evaluation_energy_abi.get("renderer_timestep_value") != 999',
            "def stable_read_path(",
            'getattr(os, "O_NOFOLLOW", 0)',
            "dir_fd=parent_descriptor",
            "def load_verified_c2_packet(",
            "def load_oracle_q_label_v1(",
            "def role_only_slot_swap_v1(",
            '"formal_c2_authorized": False',
            '"source_instruction_inference_authorized": False',
        ):
            self.assertIn(fragment, source)


@unittest.skipUnless(TORCH_AVAILABLE, "torch runtime is required")
class ELAL3SimulatorC2LabelFunctionalTests(unittest.TestCase):
    def test_exact2_exact16_packet_and_held_fd_bindings(self) -> None:
        packet = subject.load_verified_c2_packet(PACKET_ROOT)
        self.assertEqual(tuple(packet.rows), subject.C2_ROW_IDS)
        for row_id, row in packet.rows.items():
            self.assertEqual(row.row_id, row_id)
            self.assertEqual(tuple(row.annotations), subject.MEDIA_ORDER)
            self.assertEqual(tuple(row.file_bindings), subject.MEDIA_ORDER)
            for variant in subject.MEDIA_ORDER:
                bindings = row.file_bindings[variant]
                self.assertEqual(
                    set(bindings),
                    {"media", "annotation", "annotation_receipt"},
                )
                for binding in bindings.values():
                    self.assertTrue(binding["held_fd_double_read_verified"])
                    self.assertTrue(
                        binding["held_openat_parent_chain_replayed"]
                    )
                    self.assertEqual(binding["nlink"], 1)

    def test_target_and_role_swap_are_fixed_k3_e6_and_semantically_rebound(self) -> None:
        labels = {}
        for row_id in subject.C2_ROW_IDS:
            for variant in ("target", "role_swap"):
                labels[(row_id, variant)] = subject.load_oracle_q_label_v1(
                    PACKET_ROOT,
                    row_id=row_id,
                    media_variant=variant,
                    patch_grid=(21, 6, 8),
                    external_authority_path=AUTHORITY_PATH,
                    experiment_contract_path=CONTRACT_PATH,
                )
        for row_id in subject.C2_ROW_IDS:
            target = labels[(row_id, "target")]
            swapped = labels[(row_id, "role_swap")]
            self.assertEqual(
                target.receipt["slot_entity_ids"],
                ["agent", "patient", "object"],
            )
            self.assertEqual(
                swapped.receipt["slot_entity_ids"],
                ["patient", "agent", "object"],
            )
            for label in (target, swapped):
                label.latent.validate()
                self.assertEqual(tuple(label.latent.q_local.shape), (1, 21, 6, 8, 64))
                self.assertEqual(tuple(label.latent.q_entity.shape), (1, 3, 21, 256))
                self.assertEqual(tuple(label.latent.q_relation.shape), (1, 6, 21, 128))
                self.assertTrue(label.latent.entity_presence.all())
                self.assertTrue(label.latent.temporal_valid.all())
                self.assertTrue(label.latent.relation_valid.all())
                self.assertTrue(label.latent.phase_valid.all())
                self.assertEqual(tuple(label.event_mask_patch.shape), (1, 21, 6, 8))
                self.assertEqual(tuple(label.event_mask_vae.shape), (1, 1, 21, 12, 16))
                self.assertEqual(tuple(label.role_event_mask_patch.shape), (1, 3, 21, 6, 8))
                self.assertEqual(tuple(label.role_event_mask_vae.shape), (1, 3, 21, 12, 16))
                self.assertTrue(
                    torch.equal(
                        label.role_event_mask_vae.any(dim=1),
                        label.event_mask_vae[:, 0],
                    )
                )
                self.assertTrue(
                    torch.logical_xor(
                        label.event_mask_patch, label.context_mask_patch
                    ).all()
                )
                self.assertEqual(
                    label.receipt["experiment_contract_binding"]["file_sha256"],
                    subject.EXPECTED_EXPERIMENT_CONTRACT_SHA256,
                )
                self.assertEqual(
                    label.receipt["experiment_contract_binding"][
                        "renderer_timestep_value"
                    ],
                    999,
                )
                self.assertEqual(
                    label.receipt["experiment_contract_binding"]["sigma_float32"],
                    1.0,
                )
                self.assertEqual(
                    label.receipt["experiment_contract_binding"]["x_sigma"],
                    "epsilon",
                )
                self.assertFalse(label.receipt["formal_c2_authorized"])
                self.assertFalse(
                    label.receipt["source_instruction_inference_authorized"]
                )
            self.assertTrue(
                torch.equal(target.latent.q_phase, swapped.latent.q_phase)
            )
            self.assertFalse(
                torch.equal(target.latent.q_entity, swapped.latent.q_entity)
            )
            self.assertFalse(
                torch.equal(target.latent.q_relation, swapped.latent.q_relation)
            )
        handover_target = labels[(subject.C2_ROW_IDS[1], "target")]
        handover_swapped = labels[(subject.C2_ROW_IDS[1], "role_swap")]
        self.assertEqual(
            handover_target.receipt["slot_roles"],
            ["agent", "co_agent", "patient_object"],
        )
        self.assertEqual(
            handover_swapped.receipt["slot_roles"],
            ["agent", "receiver", "patient_object"],
        )
        self.assertEqual(
            handover_target.receipt["role_code_order"],
            list(subject.ROLE_CODE_ORDER),
        )
        self.assertEqual(
            handover_swapped.receipt["role_code_order"],
            list(subject.ROLE_CODE_ORDER),
        )

    def test_role_only_helper_changes_only_entity_and_relation(self) -> None:
        label = subject.load_oracle_q_label_v1(
            PACKET_ROOT,
            row_id=subject.C2_ROW_IDS[0],
            media_variant="target",
            patch_grid=(21, 6, 8),
            external_authority_path=AUTHORITY_PATH,
            experiment_contract_path=CONTRACT_PATH,
        )
        hybrid = subject.build_role_only_slot_swap_v1(label)
        for name in (
            "q_local",
            "q_phase",
            "q_terminal",
            "q_camera",
            "entity_presence",
            "temporal_valid",
            "relation_valid",
            "phase_valid",
        ):
            self.assertTrue(
                torch.equal(getattr(label.latent, name), getattr(hybrid.latent, name)),
                name,
            )
        self.assertFalse(torch.equal(label.latent.q_entity, hybrid.latent.q_entity))
        self.assertFalse(torch.equal(label.latent.q_relation, hybrid.latent.q_relation))
        self.assertTrue(
            hybrid.receipt["only_q_entity_and_q_relation_changed"]
        )
        for edge_index, (source, target) in enumerate(subject.RELATION_EDGES):
            self.assertTrue(
                torch.equal(
                    hybrid.latent.q_relation[:, edge_index, :, 9],
                    torch.full_like(
                        hybrid.latent.q_relation[:, edge_index, :, 9],
                        source / 2.0,
                    ),
                )
            )
            self.assertTrue(
                torch.equal(
                    hybrid.latent.q_relation[:, edge_index, :, 10],
                    torch.full_like(
                        hybrid.latent.q_relation[:, edge_index, :, 10],
                        target / 2.0,
                    ),
                )
            )

        opposite = subject.load_oracle_q_label_v1(
            PACKET_ROOT,
            row_id=subject.C2_ROW_IDS[0],
            media_variant="role_swap",
            patch_grid=(21, 6, 8),
            external_authority_path=AUTHORITY_PATH,
            experiment_contract_path=CONTRACT_PATH,
        )
        opposite_hybrid = subject.build_role_only_hybrid_v1(label, opposite)
        self.assertTrue(
            torch.equal(opposite_hybrid.latent.q_entity, opposite.latent.q_entity)
        )
        self.assertTrue(
            torch.equal(
                opposite_hybrid.latent.q_relation, opposite.latent.q_relation
            )
        )
        for name in (
            "q_local",
            "q_phase",
            "q_terminal",
            "q_camera",
            "entity_presence",
            "temporal_valid",
            "relation_valid",
            "phase_valid",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(opposite_hybrid.latent, name),
                    getattr(label.latent, name),
                ),
                name,
            )
        for name in (
            "event_mask_patch",
            "context_mask_patch",
            "event_mask_vae",
            "context_mask_vae",
            "role_amodal_mask_patch",
            "role_visible_mask_patch",
            "role_event_mask_patch",
            "role_event_mask_vae",
        ):
            self.assertTrue(
                torch.equal(getattr(opposite_hybrid, name), getattr(label, name)),
                name,
            )

    def test_literal_authority_contract_and_invalid_selectors_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            subject.ELAL3SimulatorC2LabelError,
            "external authority SHA literal differs",
        ):
            subject.load_external_authority_v1(
                AUTHORITY_PATH, expected_sha256="0" * 64
            )
        with self.assertRaisesRegex(
            subject.ELAL3SimulatorC2LabelError,
            "experiment contract SHA literal differs",
        ):
            subject.load_experiment_contract_v1(
                CONTRACT_PATH, expected_sha256="0" * 64
            )
        for row_id, variant, grid in (
            ("c2-missing", "target", (21, 6, 8)),
            (subject.C2_ROW_IDS[0], "missing", (21, 6, 8)),
            (subject.C2_ROW_IDS[0], "target", (20, 6, 8)),
        ):
            with self.subTest(row_id=row_id, variant=variant, grid=grid):
                with self.assertRaises(subject.ELAL3SimulatorC2LabelError):
                    subject.load_oracle_q_label_v1(
                        PACKET_ROOT,
                        row_id=row_id,
                        media_variant=variant,
                        patch_grid=grid,
                        external_authority_path=AUTHORITY_PATH,
                        experiment_contract_path=CONTRACT_PATH,
                    )

    def test_manifest_tamper_fails_before_annotation_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "packet"
            shutil.copytree(PACKET_ROOT, copied)
            manifest = copied / "manifest.json"
            manifest.chmod(0o644)
            manifest.write_bytes(manifest.read_bytes() + b"x")
            manifest.chmod(0o444)
            with self.assertRaisesRegex(
                subject.ELAL3SimulatorC2LabelError,
                "held-FD identity/double-read/SHA replay differs",
            ):
                subject.load_verified_c2_packet(copied)

    def test_stable_read_exception_closes_file_and_root_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            path = root / "source.json"
            payload = b'{}\n'
            path.write_bytes(payload)
            path.chmod(0o644)
            expected_sha = hashlib.sha256(payload).hexdigest()
            original_fstat = subject.os.fstat
            original_close = subject.os.close
            fstat_calls = 0
            closed = []

            def injected_fstat(descriptor):
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 4:
                    raise OSError("injected held-root replay failure")
                return original_fstat(descriptor)

            def tracked_close(descriptor):
                closed.append(descriptor)
                return original_close(descriptor)

            with mock.patch.object(
                subject.os, "fstat", side_effect=injected_fstat
            ), mock.patch.object(subject.os, "close", side_effect=tracked_close):
                with self.assertRaisesRegex(
                    OSError, "injected held-root replay failure"
                ):
                    subject.stable_read_path(
                        path.resolve(strict=True),
                        label="exception cleanup probe",
                        expected_sha256=expected_sha,
                        expected_mode=0o644,
                        allowed_root=root,
                    )
            self.assertEqual(len(closed), 2)
            for descriptor in closed:
                with self.assertRaises(OSError):
                    original_fstat(descriptor)


if __name__ == "__main__":
    unittest.main()
