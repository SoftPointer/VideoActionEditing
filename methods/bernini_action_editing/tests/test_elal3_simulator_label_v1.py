from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = METHOD_ROOT / "elal3_simulator_label_v1.py"
PACKET_ROOT = (
    REPOSITORY_ROOT
    / "md"
    / "action_editing"
    / "20260817_box"
    / "simulator_gt_canary_v1"
)
EXTERNAL_AUTHORITY_PATH = (
    REPOSITORY_ROOT
    / "md"
    / "action_editing"
    / "20260817_box"
    / "evidence"
    / "elal3_c1_simulator_optimizer_diagnostic_authority_v1.json"
)
EXTERNAL_AUTHORITY_SHA256 = (
    "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b"
)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    import elal3_simulator_label_v1 as labels

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    labels = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _load_authorized_label(*args, **kwargs):
    return labels.load_oracle_q_label_v1(
        *args,
        external_authority_path=EXTERNAL_AUTHORITY_PATH,
        external_authority_sha256=EXTERNAL_AUTHORITY_SHA256,
        **kwargs,
    )


class ELAL3SimulatorLabelStaticTests(unittest.TestCase):
    def test_source_parses_and_contains_narrow_authority_markers(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for fragment in (
            'EXPECTED_ROW_ID = "c1-two-entity-push-to-goal"',
            'EXPECTED_MANIFEST_SHA256 = (',
            'EXPECTED_EXTERNAL_AUTHORITY_SHA256 = (',
            'EXPECTED_EXTERNAL_AUTHORITY_DIGEST = (',
            'EXPECTED_MEDIA_PINS: Mapping[str, Mapping[str, str]]',
            "class ELAL3SimulatorOracleLabelV1",
            "def load_oracle_q_label_v1(",
            '"simulator_optimizer_diagnostic_authorized": True',
            '"external_optimizer_authority_verified": True',
            '"formal_training_authorized": False',
            '"exact160_eligible": False',
            '"scientific_claim_authorized": False',
            '"real_video_data": False',
            '"source_instruction_inference_authorized": False',
            "os.O_EXCL",
        ):
            self.assertIn(fragment, source)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class ELAL3SimulatorLabelFunctionalTests(unittest.TestCase):
    def _packet_snapshot(self, root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        }

    def test_exact_packet_builds_deterministic_fixed_abi_and_masks(self) -> None:
        before = self._packet_snapshot(PACKET_ROOT)
        with self.assertRaises(TypeError):
            labels.load_oracle_q_label_v1(
                PACKET_ROOT, patch_grid=(21, 6, 8)
            )
        first = _load_authorized_label(
            PACKET_ROOT,
            patch_grid=(21, 6, 8),
            device="cpu",
            dtype=torch.float32,
        )
        second = _load_authorized_label(
            PACKET_ROOT,
            patch_grid=(phase for phase in (21, 6, 8)),
            device="cpu",
            dtype=torch.float32,
        )
        after = self._packet_snapshot(PACKET_ROOT)

        self.assertEqual(before, after)
        self.assertEqual(first.verified_row.row_id, labels.EXPECTED_ROW_ID)
        self.assertEqual(
            set(first.verified_row.annotations), set(labels.EXPECTED_MEDIA_PINS)
        )
        first.latent.validate()
        expected_shapes = {
            "q_local": (1, 21, 6, 8, 64),
            "q_entity": (1, 3, 21, 256),
            "q_relation": (1, 6, 21, 128),
            "q_phase": (1, 21, 128),
            "q_terminal": (1, 9, 256),
            "q_camera": (1, 21, 128),
        }
        for name, shape in expected_shapes.items():
            value = getattr(first.latent, name)
            self.assertEqual(tuple(value.shape), shape)
            self.assertTrue(torch.equal(value, getattr(second.latent, name)))
            self.assertFalse(value.requires_grad)
        self.assertEqual(
            first.latent.entity_presence.tolist(), [[True, True, False]]
        )
        self.assertTrue(first.latent.temporal_valid[:, :2].all())
        self.assertFalse(first.latent.temporal_valid[:, 2].any())
        self.assertEqual(
            first.latent.relation_valid[0, :, 0].tolist(),
            [True, False, True, False, False, False],
        )
        self.assertTrue(first.latent.phase_valid.all())
        self.assertEqual(float(first.latent.q_camera.abs().sum()), 0.0)

        self.assertEqual(tuple(first.event_mask_patch.shape), (1, 21, 6, 8))
        self.assertEqual(tuple(first.context_mask_patch.shape), (1, 21, 6, 8))
        self.assertEqual(tuple(first.event_mask_vae.shape), (1, 1, 21, 12, 16))
        self.assertEqual(
            tuple(first.context_mask_vae.shape), (1, 1, 21, 12, 16)
        )
        self.assertTrue(
            torch.logical_xor(
                first.event_mask_patch, first.context_mask_patch
            ).all()
        )
        self.assertFalse(
            torch.logical_and(
                first.event_mask_patch, first.context_mask_patch
            ).any()
        )
        expected_vae = (
            first.event_mask_patch[:, None]
            .repeat_interleave(2, dim=3)
            .repeat_interleave(2, dim=4)
        )
        self.assertTrue(torch.equal(first.event_mask_vae, expected_vae))
        self.assertTrue(
            torch.equal(first.context_mask_vae, ~first.event_mask_vae)
        )
        self.assertGreater(int(first.event_mask_patch.sum()), 0)
        self.assertGreater(int(first.context_mask_patch.sum()), 0)
        self.assertGreater(float(first.signed_motion_patch.abs().sum()), 0.0)
        self.assertIs(first.target_flow, first.signed_motion_patch)
        self.assertEqual(
            first.receipt["label_digest"],
            "2a41cbd1f79779d65a6c92bfae99a5398b85d15c43b668679f4dc516c6f6260a",
        )
        self.assertTrue(first.receipt["external_optimizer_authority_verified"])
        self.assertEqual(
            first.receipt["external_authority_binding"]["file_sha256"],
            EXTERNAL_AUTHORITY_SHA256,
        )
        self.assertEqual(
            first.receipt["external_authority_binding"]["object_digest"],
            "c1706ee5b3f8a3fa4c037dfa6dbdbc7d0b088d3682128e50e712e311dae35043",
        )
        self.assertEqual(first.receipt, second.receipt)

    def test_manifest_media_annotation_and_receipt_tamper_fail_closed(self) -> None:
        relative_paths = (
            Path("manifest.json"),
            Path("media/c1-two-entity-push-to-goal/source.mp4"),
            Path(
                "annotations/c1-two-entity-push-to-goal/"
                "target.annotations.json.gz"
            ),
            Path(
                "annotations/c1-two-entity-push-to-goal/"
                "reverse.annotation-receipt.json"
            ),
        )
        for relative_path in relative_paths:
            with self.subTest(
                path=relative_path.as_posix()
            ), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "packet"
                shutil.copytree(PACKET_ROOT, copied)
                target = copied / relative_path
                target.chmod(0o600)
                target.write_bytes(target.read_bytes() + b"x")
                with self.assertRaisesRegex(
                    labels.ELAL3SimulatorLabelError, "SHA-256 differs"
                ):
                    _load_authorized_label(
                        copied, patch_grid=(21, 6, 8)
                    )

    def test_invalid_row_and_grid_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            labels.ELAL3SimulatorLabelError, "registered C1 row"
        ):
            _load_authorized_label(
                PACKET_ROOT,
                row_id="c2-three-entity-occlusion",
                patch_grid=(21, 6, 8),
            )
        for grid in ((20, 6, 8), (21, 0, 8), (21, True, 8), "21,6,8"):
            with self.subTest(grid=grid), self.assertRaises(
                labels.ELAL3SimulatorLabelError
            ):
                _load_authorized_label(
                    PACKET_ROOT, patch_grid=grid  # type: ignore[arg-type]
                )

    def test_derivative_authority_is_narrow_sealed_and_create_only(self) -> None:
        label = _load_authorized_label(
            PACKET_ROOT, patch_grid=(21, 6, 8)
        )
        with self.assertRaises(TypeError):
            labels.build_derivative_authority_v1(label)
        with self.assertRaisesRegex(
            labels.ELAL3SimulatorLabelError,
            "external authority SHA literal differs",
        ):
            labels.build_derivative_authority_v1(
                label,
                external_authority_path=labels.EXPECTED_EXTERNAL_AUTHORITY_PATH,
                external_authority_sha256="0" * 64,
            )
        authority = labels.build_derivative_authority_v1(
            label,
            external_authority_path=labels.EXPECTED_EXTERNAL_AUTHORITY_PATH,
            external_authority_sha256=labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256,
        )
        self.assertEqual(
            set(authority),
            {
                "schema_version",
                "status",
                "row_id",
                "source_packet",
                "external_authority_binding",
                "label_binding",
                "scope",
                "authority",
                "authority_digest",
            },
        )
        unsigned = dict(authority)
        digest = unsigned.pop("authority_digest")
        self.assertEqual(digest, labels.object_sha256(unsigned))
        self.assertEqual(
            digest,
            "7a5a44abb60dfa6f19a005018c4ef5562f57fa404924b755cb464d42703b6aab",
        )
        flags = authority["authority"]
        self.assertTrue(flags["simulator_optimizer_diagnostic_authorized"])
        self.assertTrue(flags["training_authorized"])
        self.assertTrue(flags["external_optimizer_authority_verified"])
        self.assertEqual(
            flags["training_authority_source"],
            "separately-issued-pinned-local-authority",
        )
        for key in (
            "formal_training_authorized",
            "formal_c0_c1_c2_go_authorized",
            "exact160_eligible",
            "exact160_claim_authorized",
            "scientific_claim_authorized",
            "real_video_data",
            "source_instruction_inference_authorized",
            "model_output_claim_authorized",
            "action_encoder_qualified",
            "action_predictor_present",
            "upstream_packet_mutated",
        ):
            self.assertFalse(flags[key], key)
        self.assertEqual(authority["scope"]["allowed_optimizer_updates_max"], 20)
        self.assertEqual(authority["scope"]["allowed_representation_variant"], "full")
        self.assertEqual(authority["scope"]["allowed_attention_width"], 64)
        external_binding = authority["external_authority_binding"]
        self.assertEqual(
            external_binding["file_sha256"],
            "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b",
        )
        self.assertEqual(
            external_binding["object_digest"],
            "c1706ee5b3f8a3fa4c037dfa6dbdbc7d0b088d3682128e50e712e311dae35043",
        )
        self.assertEqual(
            external_binding["training_objective_restrictions"],
            {
                "frozen_base_velocity_reference_forbidden": True,
                "frozen_teacher_self_distillation_forbidden": True,
                "hand_tuned_reward_scalar_forbidden": True,
                "target_grounded_event_and_context_flow_only": True,
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "authority.json"
            missing = Path(temporary).resolve() / "missing-authority.json"
            with self.assertRaisesRegex(
                labels.ELAL3SimulatorLabelError, "file is unavailable"
            ):
                labels.build_derivative_authority_v1(
                    label,
                    external_authority_path=missing,
                    external_authority_sha256=(
                        labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256
                    ),
                )
            written = labels.write_derivative_authority_create_only_v1(
                output,
                label,
                external_authority_path=labels.EXPECTED_EXTERNAL_AUTHORITY_PATH,
                external_authority_sha256=(
                    labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256
                ),
            )
            self.assertEqual(written, authority)
            self.assertEqual(
                output.read_bytes(), labels.canonical_json_bytes(authority) + b"\n"
            )
            self.assertEqual(output.stat().st_mode & 0o222, 0)
            with self.assertRaisesRegex(
                labels.ELAL3SimulatorLabelError, "refusing to overwrite"
            ):
                labels.write_derivative_authority_create_only_v1(
                    output,
                    label,
                    external_authority_path=(
                        labels.EXPECTED_EXTERNAL_AUTHORITY_PATH
                    ),
                    external_authority_sha256=(
                        labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256
                    ),
                )

            copied = Path(temporary) / "copied-authority.json"
            shutil.copyfile(labels.EXPECTED_EXTERNAL_AUTHORITY_PATH, copied)
            with self.assertRaisesRegex(
                labels.ELAL3SimulatorLabelError, "registered local file"
            ):
                labels.build_derivative_authority_v1(
                    label,
                    external_authority_path=copied,
                    external_authority_sha256=(
                        labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256
                    ),
                )

            resigned = Path(temporary) / "resigned-authority.json"
            resigned_value = json.loads(
                labels.EXPECTED_EXTERNAL_AUTHORITY_PATH.read_text(
                    encoding="utf-8"
                )
            )
            resigned_value["max_optimizer_updates_per_arm"] = 21
            unsigned_resigned = dict(resigned_value)
            unsigned_resigned.pop("authority_digest")
            resigned_value["authority_digest"] = labels.object_sha256(
                unsigned_resigned
            )
            resigned.write_text(
                json.dumps(resigned_value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                labels, "EXPECTED_EXTERNAL_AUTHORITY_PATH", resigned
            ), self.assertRaisesRegex(
                labels.ELAL3SimulatorLabelError, "file SHA-256 differs"
            ):
                labels.load_oracle_q_label_v1(
                    PACKET_ROOT,
                    patch_grid=(21, 6, 8),
                    external_authority_path=resigned,
                    external_authority_sha256=(
                        labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256
                    ),
                )

        label.event_mask_patch.zero_()
        with self.assertRaisesRegex(
            labels.ELAL3SimulatorLabelError, "authenticated oracle label differs"
        ):
            labels.build_derivative_authority_v1(
                label,
                external_authority_path=labels.EXPECTED_EXTERNAL_AUTHORITY_PATH,
                external_authority_sha256=(
                    labels.EXPECTED_EXTERNAL_AUTHORITY_SHA256
                ),
            )


if __name__ == "__main__":
    unittest.main()
