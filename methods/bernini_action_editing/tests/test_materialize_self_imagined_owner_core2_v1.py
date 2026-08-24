from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import generate_self_imagined_owner_core2_v1 as generation  # noqa: E402
import materialize_self_imagined_owner_core2_v1 as materializer  # noqa: E402
import self_imagined_motion_cotangent_v1 as cotangent  # noqa: E402


REGISTRY = METHOD_ROOT / "assets/self_imagined_motion_cotangent_core2_v1.json"

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:  # pragma: no cover
    serialization = None
    Ed25519PrivateKey = None


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(materializer.canonical_json_bytes(value) + b"\n")


class SignedOwnerFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry = root / "registry.json"
        self.registry.write_bytes(REGISTRY.read_bytes())
        self.registry_sha = materializer.file_sha256(self.registry)
        self.registry_value = cotangent.load_probe_registry(
            self.registry.resolve(), expected_file_sha256=self.registry_sha
        )
        self.owner_root = root / "job131524"
        self.owner_root.mkdir()
        self.children: dict[str, dict] = {}
        self.child_paths: dict[str, Path] = {}
        for cell_id in generation.CELL_IDS:
            cell = self.registry_value.cell(cell_id)
            cell_root = self.owner_root / cell_id
            cell_root.mkdir()
            artifacts = {}
            for key, suffix in (
                ("mp4", ".mp4"),
                ("predecode_clean_latent", ".safetensors"),
                ("official_initial_gaussian", ".safetensors"),
            ):
                path = cell_root / f"{key}{suffix}"
                path.write_bytes(f"{cell_id}:{key}".encode("ascii"))
                artifacts[key] = {
                    "path": str(path.resolve()),
                    "sha256": materializer.file_sha256(path),
                }
            native = cell_root / "receipt.json"
            native.write_bytes(f"{cell_id}:native".encode("ascii"))
            unsigned = {
                "schema_version": generation.SCHEMA_VERSION,
                "probe_id": self.registry_value.probe_id,
                "cell_id": cell_id,
                "registry_path": str(self.registry.resolve()),
                "registry_file_sha256": self.registry_sha,
                "source_iid": cell.source_iid,
                "geometry_source_video_sha256": cell.source_video_sha256,
                "geometry_source_role": "bucket_shape_only_never_transformer_condition",
                "action_family_id": cell.action_family_id,
                "action_caption_utf8_sha256": cell.action_caption_utf8_sha256,
                "owner_generation_seed": cell.owner_generation_seed,
                "native_receipt_path": str(native.resolve()),
                "native_receipt_file_sha256": materializer.file_sha256(native),
                "native_receipt_digest": hashlib.sha256(
                    f"{cell_id}:native-receipt".encode("ascii")
                ).hexdigest(),
                "bucket_hw": [480, 496],
                "latent_shape": list(cell.latent_shape),
                "artifacts": artifacts,
                "method_source_revision": "1" * 40,
                "method_source_archive_sha256": "2" * 64,
                "runtime_topology": {
                    "world_size": 4,
                    "ulysses_size": 4,
                    "rocr_visible_devices": generation.VISIBLE_GPUS_BY_CELL[cell_id],
                },
                "owner_source_condition_used": False,
                "owner_exact81_action_audit_status":
                "pending_detached_full_video_review",
                "owner_template_materialization_authorized": False,
                "editor_condition_or_target_authorized": False,
                "optimizer_or_parameter_update_authorized": False,
            }
            receipt = {**unsigned, "receipt_digest": materializer.object_sha256(unsigned)}
            child_path = cell_root / generation.OWNER_RECEIPT_BASENAME
            _write_json(child_path, receipt)
            self.children[cell_id] = receipt
            self.child_paths[cell_id] = child_path
        master_unsigned = {
            "schema_version": generation.MASTER_SCHEMA_VERSION,
            "probe_id": self.registry_value.probe_id,
            "registry_path": str(self.registry.resolve()),
            "registry_file_sha256": self.registry_sha,
            "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
            "cell_order": list(generation.CELL_IDS),
            "children": [
                {
                    "cell_id": cell_id,
                    "receipt_path": str(self.child_paths[cell_id].resolve()),
                    "receipt_file_sha256": materializer.file_sha256(
                        self.child_paths[cell_id]
                    ),
                    "receipt_digest": self.children[cell_id]["receipt_digest"],
                    "mp4_path": self.children[cell_id]["artifacts"]["mp4"]["path"],
                    "mp4_sha256": self.children[cell_id]["artifacts"]["mp4"]["sha256"],
                }
                for cell_id in generation.CELL_IDS
            ],
            "exact81_owner_count": 2,
            "all8_used": True,
            "semantic_action_audit_complete": False,
            "owner_template_materialization_authorized": False,
            "optimizer_or_parameter_update_authorized": False,
        }
        self.master = {
            **master_unsigned,
            "receipt_digest": materializer.object_sha256(master_unsigned),
        }
        self.master_path = self.owner_root / generation.MASTER_RECEIPT_BASENAME
        _write_json(self.master_path, self.master)
        self.master_sha = materializer.file_sha256(self.master_path)
        self.evidence = root / "external-full81-review.md"
        self.evidence.write_text(
            "Detached human review: dog and human owners pass all 81 frames.\n",
            encoding="ascii",
        )
        assert Ed25519PrivateKey is not None and serialization is not None
        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.public_key = root / "audit-authority-ed25519.pem"
        self.public_key.write_bytes(public)
        self.public_key_sha = materializer.file_sha256(self.public_key)
        sidecar_unsigned = {
            "schema_version": materializer.AUDIT_SCHEMA,
            "owner_generation_job_id": materializer.AUDIT_JOB_ID,
            "owner_master_binding": {
                "path": str(self.master_path.resolve()),
                "file_sha256": self.master_sha,
                "receipt_digest": self.master["receipt_digest"],
            },
            "audit_evidence_binding": {
                "path": str(self.evidence.resolve()),
                "file_sha256": materializer.file_sha256(self.evidence),
            },
            "audit_authority_public_key_sha256": self.public_key_sha,
            "authority_signature_scheme": materializer.AUDIT_SIGNATURE_SCHEME,
            "approval_scope": {
                "decision": "approve_owner_template_materialization",
                "semantic_action_audit_complete": True,
                "owner_template_materialization_authorized": True,
                "allowed_persistent_tensor_channel":
                cotangent.ALLOWED_OWNER_TO_EDITOR_CHANNEL,
                "forbidden_owner_to_editor_channels": list(
                    cotangent.FORBIDDEN_OWNER_TO_EDITOR_CHANNELS
                ),
                "optimizer_or_parameter_update_authorized": False,
            },
            "cells": [
                {
                    "cell_id": cell_id,
                    "owner_child_receipt_file_sha256": materializer.file_sha256(
                        self.child_paths[cell_id]
                    ),
                    "owner_child_receipt_digest": self.children[cell_id][
                        "receipt_digest"
                    ],
                    "owner_mp4_sha256": self.children[cell_id]["artifacts"]["mp4"][
                        "sha256"
                    ],
                    "action_family_id": self.registry_value.cell(cell_id).action_family_id,
                    "exact81_frame_count": 81,
                    "review_scope": "all_81_frames_start_transition_terminal_hold",
                    "owner_exact81_action_audit_passed": True,
                    "owner_source_condition_used": False,
                    "materialize_template": True,
                }
                for cell_id in generation.CELL_IDS
            ],
        }
        signed = {
            **sidecar_unsigned,
            "receipt_digest": materializer.object_sha256(sidecar_unsigned),
        }
        signature = private.sign(materializer.canonical_json_bytes(signed))
        self.sidecar = {
            **signed,
            "authority_signature_ed25519_base64": base64.b64encode(signature).decode(
                "ascii"
            ),
        }
        self.sidecar_path = root / "signed-full81-audit.json"
        _write_json(self.sidecar_path, self.sidecar)
        self.sidecar_sha = materializer.file_sha256(self.sidecar_path)

    def kwargs(self) -> dict:
        return {
            "registry": self.registry.resolve(),
            "expected_registry_sha256": self.registry_sha,
            "owner_root": self.owner_root.resolve(),
            "owner_master_receipt": self.master_path.resolve(),
            "expected_owner_master_receipt_sha256": self.master_sha,
            "audit_sidecar": self.sidecar_path.resolve(),
            "expected_audit_sidecar_sha256": self.sidecar_sha,
            "audit_evidence": self.evidence.resolve(),
            "audit_public_key": self.public_key.resolve(),
            "expected_audit_public_key_sha256": self.public_key_sha,
        }


@unittest.skipIf(Ed25519PrivateKey is None, "cryptography Ed25519 is unavailable")
class SignedAuditBoundaryTests(unittest.TestCase):
    def test_gpu_entrypoint_uses_historical_container_compatibility_verifier(self) -> None:
        source = inspect.getsource(materializer.materialize_cell)
        self.assertIn(
            "starc.verify_authenticated_native_clean_tensor_identity", source
        )
        self.assertNotIn(
            "frozen.verify_native_tensor_value_identity(\n            clean", source
        )

    def test_public_pending_loader_authenticates_but_grants_no_use_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SignedOwnerFixture(Path(temporary))
            pending = materializer.load_pending_owner_generation_inputs(
                registry=fixture.registry.resolve(),
                expected_registry_sha256=fixture.registry_sha,
                owner_root=fixture.owner_root.resolve(),
                owner_master_receipt=fixture.master_path.resolve(),
                expected_owner_master_receipt_sha256=fixture.master_sha,
            )
            self.assertEqual(tuple(pending.child_receipts), generation.CELL_IDS)
            self.assertFalse(pending.semantic_action_audit_complete)
            self.assertFalse(pending.template_materialization_authorized)
            self.assertFalse(pending.clean_latent_editor_input_authorized)

    def test_pending_receipts_reject_without_external_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SignedOwnerFixture(Path(temporary))
            kwargs = fixture.kwargs()
            kwargs.update(
                audit_sidecar=None,
                expected_audit_sidecar_sha256=None,
                audit_evidence=None,
                audit_public_key=None,
                expected_audit_public_key_sha256=None,
            )
            with self.assertRaisesRegex(
                materializer.OwnerTemplateMaterializationError,
                "require a signed external full81 audit",
            ):
                materializer.load_authorized_owner_inputs(**kwargs)

    def test_signed_job131524_sidecar_unlocks_but_does_not_mutate_pending_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SignedOwnerFixture(Path(temporary))
            authorized = materializer.load_authorized_owner_inputs(**fixture.kwargs())
            self.assertEqual(authorized.audit_sidecar_file_sha256, fixture.sidecar_sha)
            self.assertEqual(tuple(authorized.child_receipts), generation.CELL_IDS)
            for child in authorized.child_receipts.values():
                self.assertEqual(
                    child["owner_exact81_action_audit_status"],
                    "pending_detached_full_video_review",
                )
                self.assertFalse(child["owner_template_materialization_authorized"])

    def test_sidecar_signature_tamper_rejects_even_with_new_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SignedOwnerFixture(Path(temporary))
            changed = dict(fixture.sidecar)
            cells = [dict(row) for row in changed["cells"]]
            cells[0]["owner_exact81_action_audit_passed"] = False
            changed["cells"] = cells
            _write_json(fixture.sidecar_path, changed)
            kwargs = fixture.kwargs()
            kwargs["expected_audit_sidecar_sha256"] = materializer.file_sha256(
                fixture.sidecar_path
            )
            with self.assertRaisesRegex(
                materializer.OwnerTemplateMaterializationError, "signature verification failed"
            ):
                materializer.load_authorized_owner_inputs(**kwargs)

    def test_child_or_external_evidence_tamper_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SignedOwnerFixture(Path(temporary))
            fixture.evidence.write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(
                materializer.OwnerTemplateMaterializationError, "audit authority differs"
            ):
                materializer.load_authorized_owner_inputs(**fixture.kwargs())
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SignedOwnerFixture(Path(temporary))
            Path(
                fixture.children["dog"]["artifacts"]["predecode_clean_latent"]["path"]
            ).write_bytes(b"tampered")
            with self.assertRaisesRegex(
                materializer.OwnerTemplateMaterializationError, "bytes changed"
            ):
                materializer.load_authorized_owner_inputs(**fixture.kwargs())

    def test_parser_has_no_runtime_sidecar_authoring_command(self) -> None:
        parser = materializer.build_parser()
        help_text = parser.format_help()
        self.assertIn("preflight", help_text)
        self.assertIn("materialize-cell", help_text)
        self.assertNotIn("author-sidecar", help_text)
        self.assertNotIn("materialize_cell_from_hidden_callback", materializer.__all__)
        self.assertNotIn("persist_cell_materialization", materializer.__all__)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class FakeHiddenMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = REGISTRY.read_bytes()
        self.registry = cotangent.load_probe_registry(
            REGISTRY.resolve(), expected_file_sha256=hashlib.sha256(raw).hexdigest()
        )
        self.cell = self.registry.cell("dog")
        self.clean = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)

    @staticmethod
    def gaussian_factory(seed: int, *, shape: tuple[int, ...], device: object):
        value = float(seed % 997 + 1) / 997.0
        return torch.full(shape, value, dtype=torch.float32, device=device)

    @staticmethod
    def pattern(k: int) -> "torch.Tensor":
        phase = torch.linspace(-1.0, 1.0, 21, dtype=torch.float32).reshape(1, 21, 1, 1)
        channel = torch.zeros((1, 1, 1, 1536), dtype=torch.float32)
        channel[..., 0] = 1.0
        return (phase * channel).expand(1, 21, k, 1536).contiguous()

    def test_same_object_distinct_seeds_spatial_orderless_and_specific(self) -> None:
        calls: list[tuple[int, str, int, str]] = []

        def callback(*, x_sigma, prompt_role, prompt_caption, query_seed):
            self.assertIn(prompt_caption, {
                self.cell.action_caption,
                self.cell.reverse_wrong_family_caption,
                self.cell.noop_caption,
            })
            calls.append(
                (
                    query_seed,
                    prompt_role,
                    id(x_sigma),
                    materializer.tensor_sha256(x_sigma, label="fake x_sigma"),
                )
            )
            k = 2 if query_seed == self.cell.query_seeds[0] else 5
            pattern = self.pattern(k)
            if prompt_role == "action":
                return pattern, {"fake_role": prompt_role}
            if prompt_role == "reverse_wrong_family":
                return -pattern, {"fake_role": prompt_role}
            return torch.zeros_like(pattern), {"fake_role": prompt_role}

        result = materializer._materialize_cell_from_hidden_callback_unsafe(
            cell=self.cell,
            owner_clean_latent=self.clean,
            hidden_forward=callback,
            gaussian_factory=self.gaussian_factory,
            specificity_margin=0.1,
            minimum_template_cosine=0.05,
        )
        self.assertEqual(result.ordered_query_seeds, self.cell.query_seeds)
        self.assertTrue(result.two_seed_audit.passed)
        self.assertEqual([row.owner_spatial_coordinates for row in result.rows], [2, 5])
        for index, seed in enumerate(self.cell.query_seeds):
            rows = calls[index * 3 : (index + 1) * 3]
            self.assertEqual([row[1] for row in rows], list(materializer.PROMPT_ORDER))
            self.assertEqual({row[0] for row in rows}, {seed})
            self.assertEqual(len({row[2] for row in rows}), 1)
            self.assertEqual(len({row[3] for row in rows}), 1)
            self.assertTrue(result.rows[index].specificity.passed)
        self.assertNotEqual(calls[0][3], calls[3][3])
        self.assertNotEqual(
            result.rows[0].official_gaussian_tensor_digest,
            result.rows[1].official_gaussian_tensor_digest,
        )

        # Phi is spatial-orderless: an editor candidate may use a third K.
        scorer = cotangent.make_frozen_per_query_scorer(result.rows[0].template)
        candidate_leaf = self.pattern(7).requires_grad_(True)
        candidate = candidate_leaf * 1.0
        score = scorer.forward_sketched_residual(
            candidate, require_input_grad=True
        ).score
        score.backward()
        self.assertIsNotNone(candidate_leaf.grad)
        self.assertGreater(float(score.detach().item()), 0.99)

    def test_generic_motion_or_prompt_alias_fails_specificity(self) -> None:
        def callback(*, x_sigma, prompt_role, prompt_caption, query_seed):
            del x_sigma, prompt_caption, query_seed
            pattern = self.pattern(3)
            if prompt_role == "common_scene_noop":
                return torch.zeros_like(pattern)
            return pattern

        with self.assertRaisesRegex(
            materializer.OwnerTemplateMaterializationError, "failed prompt specificity"
        ):
            materializer._materialize_cell_from_hidden_callback_unsafe(
                cell=self.cell,
                owner_clean_latent=self.clean,
                hidden_forward=callback,
                gaussian_factory=self.gaussian_factory,
                specificity_margin=0.1,
                minimum_template_cosine=0.05,
            )

    def test_bundle_retains_only_templates_and_scalar_receipts(self) -> None:
        def callback(*, x_sigma, prompt_role, prompt_caption, query_seed):
            del x_sigma, prompt_caption, query_seed
            pattern = self.pattern(2)
            if prompt_role == "action":
                return pattern
            if prompt_role == "reverse_wrong_family":
                return -pattern
            return torch.zeros_like(pattern)

        result = materializer._materialize_cell_from_hidden_callback_unsafe(
            cell=self.cell,
            owner_clean_latent=self.clean,
            hidden_forward=callback,
            gaussian_factory=self.gaussian_factory,
            specificity_margin=0.1,
            minimum_template_cosine=0.05,
        )
        self.assertEqual(
            set(result.tensors()),
            {
                f"{materializer.TENSOR_KEY_PREFIX}{seed}"
                for seed in self.cell.query_seeds
            },
        )
        self.assertFalse(any(value.requires_grad for value in result.tensors().values()))
        self.assertFalse(hasattr(result, "clean_latent"))
        self.assertFalse(hasattr(result, "hidden_states"))
        self.assertFalse(hasattr(result, "velocity"))

    @unittest.skipIf(Ed25519PrivateKey is None, "cryptography Ed25519 is unavailable")
    def test_only_runtime_bound_bundle_can_persist_and_public_packet_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SignedOwnerFixture(root)
            authority = materializer.load_authorized_owner_inputs(**fixture.kwargs())
            cell = authority.registry.cell("dog")
            clean = self.clean.clone()
            clean_artifact = authority.cell("dog")["artifacts"][
                "predecode_clean_latent"
            ]

            def callback(*, x_sigma, prompt_role, prompt_caption, query_seed):
                del x_sigma, prompt_caption, query_seed
                pattern = self.pattern(2)
                proof = {
                    "prompt_order": list(materializer.PROMPT_ORDER),
                    "same_x_sigma_noisy_rotary_timestep_objects": True,
                    "shared_tensor_bytes_unchanged": True,
                    "target_suffix_only": True,
                    "source_condition_consumed": False,
                    "full_hidden_persisted": False,
                }
                if prompt_role == "action":
                    return pattern, proof
                if prompt_role == "reverse_wrong_family":
                    return -pattern, {"triplet_cache_reused": True}
                return torch.zeros_like(pattern), {"triplet_cache_reused": True}

            bundle = materializer._materialize_cell_from_hidden_callback_unsafe(
                cell=cell,
                owner_clean_latent=clean,
                hidden_forward=callback,
                gaussian_factory=self.gaussian_factory,
                specificity_margin=0.1,
                minimum_template_cosine=0.05,
                production_input_binding={
                    "owner_child_receipt_digest": authority.cell("dog")[
                        "receipt_digest"
                    ],
                    "external_full81_audit_sidecar_receipt_digest": (
                        authority.audit_sidecar_receipt_digest
                    ),
                    "owner_clean_latent_file_sha256": clean_artifact["sha256"],
                    "owner_clean_latent_tensor_digest": materializer.tensor_sha256(
                        clean, label="fake authenticated clean"
                    ),
                },
            )
            with self.assertRaisesRegex(
                materializer.OwnerTemplateMaterializationError,
                "raw callback bundles cannot cross",
            ):
                materializer.persist_cell_materialization(
                    output_dir=root.resolve() / "must_not_exist",
                    bundle=replace(bundle, production_input_binding=None),
                    authority=authority,
                    model_binding={},
                )
            self.assertFalse((root.resolve() / "must_not_exist").exists())

            tensor_store = {
                str(clean_artifact["path"]): {
                    "normalized_clean_latent": clean.clone()
                }
            }

            class FakeOpen:
                def __init__(self, path: str, **_: object) -> None:
                    self.path = path
                    self.values = tensor_store.get(path)
                    if self.values is None:
                        self.values = torch.load(path)

                def __enter__(self):
                    return self

                def __exit__(self, *_: object) -> None:
                    return None

                def keys(self):
                    return list(self.values)

                def get_tensor(self, key: str):
                    return self.values[key].clone()

            def fake_save_file(values, path: str, metadata=None):
                del metadata
                torch.save({key: value.clone() for key, value in values.items()}, path)

            fake_root = types.ModuleType("safetensors")
            fake_torch = types.ModuleType("safetensors.torch")
            fake_root.safe_open = FakeOpen
            fake_torch.save_file = fake_save_file
            output = root.resolve() / "published_dog"
            model_binding = {
                "native_schedule_index": cotangent.SCHEDULE_INDEX,
                "native_timestep": cotangent.NATIVE_TIMESTEP,
                "sigma": cotangent.NATIVE_SIGMA,
                "hook_coordinate": cotangent.HOOK_COORDINATE,
                "transformer_1_only": True,
                "all_parameters_frozen": True,
                "adapter_loaded": False,
            }
            modules = {
                "safetensors": fake_root,
                "safetensors.torch": fake_torch,
            }
            with mock.patch.dict(sys.modules, modules):
                receipt = materializer.persist_cell_materialization(
                    output_dir=output,
                    bundle=bundle,
                    authority=authority,
                    model_binding=model_binding,
                )
                checked = materializer.validate_published_cell_packet(
                    receipt, cell_root=output, authority=authority
                )
            self.assertEqual(checked["receipt_digest"], receipt["receipt_digest"])
            self.assertEqual(
                set(path.name for path in output.iterdir()),
                {
                    materializer.QUOTIENT_FILENAME,
                    materializer.CELL_RECEIPT_FILENAME,
                },
            )
            output.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
