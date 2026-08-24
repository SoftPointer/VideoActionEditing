#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import run_caper_dual_coordinate_stateless_canary_v1 as subject


ASSET_PATH = (
    METHOD_ROOT / "assets" / "caper_dual_coordinate_core4_canary_v1.json"
).resolve()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write_canonical(path: Path, value: dict) -> str:
    payload = subject.canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class AssetAuthorityAndWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.asset = subject.load_and_validate_asset(
            ASSET_PATH, subject.PINNED_ASSET_SHA256
        )

    def test_asset_bytes_and_external_authorities_are_fixed(self) -> None:
        self.assertEqual(
            subject.PINNED_ASSET_SHA256,
            "b8fe179905cf77951fda2fdc6cf18622b11510263c56df4b306457a3ce717f57",
        )
        self.assertEqual(subject.file_sha256(ASSET_PATH), subject.PINNED_ASSET_SHA256)
        self.assertEqual(
            subject.PINNED_CHECKPOINT_MANIFEST_SHA256,
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
        )
        self.assertEqual(
            subject.PINNED_EDITOR_RUNTIME_PUBLIC_KEY_SHA256,
            "b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d",
        )
        self.assertEqual(
            subject.PINNED_DIRECTION_GATE_PUBLIC_KEY_SHA256,
            "655befbbf0deea1006e33e9656127e11753d85a0ae22d84619bfbcb8185dcdbd",
        )
        fixed = {
            "registry": (
                subject.PINNED_REGISTRY_SHA256,
                "01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96",
            ),
            "owner master file": (
                subject.PINNED_OWNER_MASTER_FILE_SHA256,
                "c0d8f3e4a7f3b95269b5196c0d8844327d9e7296dda1828493683a9ae7d707de",
            ),
            "owner master digest": (
                subject.PINNED_OWNER_MASTER_RECEIPT_DIGEST,
                "b71d726c7001c57da80391b18c5c82b8fe0910a62f8cd99484d3a90d218347ab",
            ),
            "owner audit sidecar": (
                subject.PINNED_OWNER_AUDIT_SIDECAR_SHA256,
                "24746c91e88e4051c49fe18b06e0e58bb2c4b119b3d946586d9dd6092308030b",
            ),
            "owner audit evidence": (
                subject.PINNED_OWNER_AUDIT_EVIDENCE_SHA256,
                "3e2335d4d335a9ee8262aa319fc2790dbac3e59b20e554f54b4dc1273f259dc3",
            ),
            "owner audit public key": (
                subject.PINNED_OWNER_AUDIT_PUBLIC_KEY_SHA256,
                "d1bba83ca1d162128bda71e21c419c476b9328c7892bd1998adcd24c09c577ec",
            ),
            "quotient master file": (
                subject.PINNED_QUOTIENT_MASTER_FILE_SHA256,
                "fde8de229135bf46682681bbc83fc39d7554e144b6d97991741e39a2ebfe98c3",
            ),
            "quotient master digest": (
                subject.PINNED_QUOTIENT_MASTER_RECEIPT_DIGEST,
                "8fa7e4cf01d9fa49b506aa66b50932c2ac767faa4727dc772e407d41052652e6",
            ),
        }
        for label, (observed, expected) in fixed.items():
            with self.subTest(label=label):
                self.assertEqual(observed, expected)
        self.assertEqual(
            subject.PINNED_CELL_RECEIPT_FILE_SHA256,
            {
                "dog": "5630c0f511360a6ae0386855f4c00e78e226fea32f71d340773db83ab5c49bd2",
                "human": "fb6a37464e98841fe340e5a1411dffe8135640410fd0cef5c1f89b86fe81184e",
            },
        )
        self.assertEqual(
            subject.PINNED_CELL_RECEIPT_DIGEST,
            {
                "dog": "6970b785eda453afa7a382c2ab6638e6f286ba8115bdcfb632af4eefd02bdf90",
                "human": "5471955cfc8a67ec4aef0e414815fc1dec763db9322951e2f485bfa790fe97f3",
            },
        )
        self.assertEqual(
            subject.PINNED_GENERATION_RECEIPT_FILE_SHA256,
            {
                "dog": "e6e6cdcd7ffbb6c3fcbaad52ac3ed088429c842777238aa021fc07de1fa67cf2",
                "human": "b1fd8e8a8296bb8e688fb09e84141ce67a5f5416ccd0e1cdca39ca18f307ac9b",
            },
        )
        self.assertEqual(
            subject.PINNED_GENERATION_RECEIPT_DIGEST,
            {
                "dog": "10acf8d383185ae87f19a966ae2d9a524d793ca315ce7e1779a18de62661cafe",
                "human": "6ff703a713b92faf8e0e1ff83c76ceb45715ef821ce025b210a5e4e13e7b5b01",
            },
        )

    def test_asset_top_level_and_nested_authority_mutations_fail_closed(self) -> None:
        mutations = []
        changed = copy.deepcopy(self.asset)
        changed["unexpected"] = True
        mutations.append(("field closure", changed))
        changed = copy.deepcopy(self.asset)
        changed["owner_authority"]["master_receipt_digest"] = "0" * 64
        mutations.append(("owner authority", changed))
        changed = copy.deepcopy(self.asset)
        changed["owner_authority"]["unexpected"] = True
        mutations.append(("nested field closure", changed))
        changed = copy.deepcopy(self.asset)
        changed["editor_runtime_authority"]["public_key_sha256"] = "0" * 64
        mutations.append(("editor runtime authority", changed))
        changed = copy.deepcopy(self.asset)
        changed["later_update_phase"][
            "direction_gate_authority_public_key_sha256"
        ] = "0" * 64
        mutations.append(("direction-gate authority", changed))
        changed = copy.deepcopy(self.asset)
        changed["cells"][0]["generation_receipt_digest"] = "0" * 64
        mutations.append(("generation authority", changed))
        changed = copy.deepcopy(self.asset)
        changed["cells"][0]["editor_noise_seeds"][0] = 2026081502
        mutations.append(("unregistered editor-noise mapping", changed))
        changed = copy.deepcopy(self.asset)
        changed["direction_phase"]["optimizer"] = True
        mutations.append(("optimizer authority", changed))
        changed = copy.deepcopy(self.asset)
        changed["forbidden_inputs"].remove(
            "generation_gaussian_as_training_epsilon"
        )
        mutations.append(("Gaussian prohibition", changed))
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            for index, (label, value) in enumerate(mutations):
                with self.subTest(label=label):
                    path = root / f"mutated-{index}.json"
                    digest = _write_canonical(path, value)
                    # Patching the file pin isolates semantic closure from the
                    # independent byte-hash rejection.
                    with mock.patch.object(subject, "PINNED_ASSET_SHA256", digest):
                        with self.assertRaises(subject.CAPERRuntimeError):
                            subject.load_and_validate_asset(path, digest)

    def test_preflight_revalidates_plain_mapping_instead_of_trusting_schema(self) -> None:
        changed = copy.deepcopy(self.asset)
        changed["owner_authority"]["master_receipt_digest"] = "0" * 64
        with self.assertRaises(subject.CAPERRuntimeError):
            subject.build_preflight_decision(changed)

    def test_two_world8_waves_are_fixed_and_not_seed_selectable(self) -> None:
        expected = (
            (("dog", 2026081502), ("human", 2026081505)),
            (("dog", 2026081503), ("human", 2026081506)),
        )
        self.assertEqual(subject.WAVE_PLAN, expected)
        self.assertEqual(subject.two_wave_plan(self.asset), expected)
        expected_editor_noise = {
            "dog": (2026082502, 2026082503),
            "human": (2026082505, 2026082506),
        }
        self.assertEqual(subject.EDITOR_NOISE_SEEDS, expected_editor_noise)
        for cell_id, owner_seeds in subject.QUERY_SEEDS.items():
            for owner_seed, noise_seed in zip(
                owner_seeds, expected_editor_noise[cell_id]
            ):
                with self.subTest(cell_id=cell_id, owner_seed=owner_seed):
                    self.assertEqual(
                        subject.editor_noise_seed(cell_id, owner_seed), noise_seed
                    )
                    self.assertEqual(noise_seed, owner_seed + 1000)
                    self.assertNotEqual(noise_seed, owner_seed)
        for cell_id, owner_seed in (("dog", 2026081505), ("cat", 2026081502)):
            with self.subTest(cell_id=cell_id, owner_seed=owner_seed):
                with self.assertRaises(subject.CAPERRuntimeError):
                    subject.editor_noise_seed(cell_id, owner_seed)
        groups = self.asset["waves"]
        self.assertEqual(
            [row["cuda_visible_devices"] for row in groups[0]["groups"]],
            ["0,1,2,3", "4,5,6,7"],
        )
        changed = copy.deepcopy(self.asset)
        changed["waves"][0]["groups"][0]["query_seed"] = 2026081503
        with self.assertRaises(subject.CAPERRuntimeError):
            subject.two_wave_plan(changed)

    def test_zero_no_go_preflight_has_no_update_authority_or_graph(self) -> None:
        self.assertFalse(self.asset["direction_phase"]["optimizer"])
        self.assertFalse(self.asset["direction_phase"]["parameter_update"])
        self.assertFalse(self.asset["later_update_phase"]["authorized"])
        self.assertFalse(self.asset["later_update_phase"]["optimizer"])
        self.assertFalse(
            self.asset["hard_negative_authority"]["gradient_source_authorized"]
        )
        self.assertEqual(
            self.asset["hard_negative_authority"]["use"],
            "NO_GO_OR_HARD_NEGATIVE_ONLY",
        )
        decision = subject.build_preflight_decision(self.asset)
        self.assertEqual(
            decision["receipt_digest"],
            subject.object_sha256(
                {key: value for key, value in decision.items() if key != "receipt_digest"}
            ),
        )
        self.assertFalse(decision["semantic_direction_gate_materialized"])
        self.assertEqual(
            decision["editor_noise_domain_separation"],
            "owner_query_seed_plus_1000",
        )
        for wave in decision["waves"]:
            for row in wave:
                self.assertFalse(row["owner_editor_noise_seed_shared"])
                self.assertEqual(
                    row["editor_noise_seed"], row["owner_query_seed"] + 1000
                )
        for name in (
            "lora_b_vjp_authorized",
            "source_preservation_qp_authorized",
            "candidate_direct_add_authorized",
            "parameter_update_authorized",
            "training_claim_authorized",
        ):
            self.assertFalse(decision[name], name)
        source = inspect.getsource(subject.build_preflight_decision)
        for forbidden in (
            "torch",
            "requires_grad",
            "optimizer",
            "parameter.add_(",
            "optimizer.step(",
            "materialize_editor_runtime_packet(",
        ):
            self.assertNotIn(forbidden, source)


class GaussianObserverProvenanceTests(unittest.TestCase):
    @staticmethod
    def _provenance(capture) -> dict:
        return {
            "observer_call_count": capture.call_count,
            "observer_only": True,
            "observer_returned_original_tensor_object": True,
            "observer_replaced_or_injected_noise": False,
            "requested_shape": list(capture.requested_shape),
            "requested_dtype": capture.requested_dtype,
            "requested_device": capture.requested_device,
            "returned_dtype": capture.returned_dtype,
            "returned_device": capture.returned_device,
            "generator_device": capture.generator_device,
            "generator_initial_seed": capture.generator_initial_seed,
            "raw_value_sha256": capture.raw_value_sha256,
            "content_sha256": capture.content_sha256,
            "cpu_generator_replay_exact_equal": True,
            "persisted_tensor_is_observer_capture_not_replay": True,
            "runtime": {
                "torch": "test-torch",
                "torch_hip": "test-rocm",
                "diffusers": "test-diffusers",
                "transformers": "test-transformers",
            },
            "role": "native_sampler_initial_noise_only",
            "training_epsilon_reuse_authorized": False,
        }

    def test_actual_sampler_observer_forwards_original_tensor_and_only_clones_provenance(self) -> None:
        owner_query_seed = 2026081502
        noise_seed = subject.editor_noise_seed("dog", owner_query_seed)
        shape = (1, 16, 21, 2, 2)
        seen = {}

        def canonical_randn_tensor(
            requested_shape, *, generator, device, dtype
        ):
            value = torch.randn(
                requested_shape, generator=generator, device=device, dtype=dtype
            )
            seen["sampler_tensor"] = value
            return value

        module = SimpleNamespace(randn_tensor=canonical_randn_tensor)

        def sample_fn():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed)
            return module.randn_tensor(
                shape,
                generator=generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        result, capture = subject.native._sample_with_native_initial_noise_observer(
            sample_fn=sample_fn,
            wan_diffusion_module=module,
            expected_shape=shape,
            expected_device=torch.device("cpu"),
            expected_seed=noise_seed,
            canonical_randn_tensor=canonical_randn_tensor,
        )
        self.assertIs(result, seen["sampler_tensor"])
        self.assertIs(module.randn_tensor, canonical_randn_tensor)
        self.assertTrue(torch.equal(result, capture.tensor))
        self.assertNotEqual(result.data_ptr(), capture.tensor.data_ptr())
        replay_generator = torch.Generator(device="cpu")
        replay_generator.manual_seed(noise_seed)
        replay = torch.randn(shape, generator=replay_generator, dtype=torch.float32)
        self.assertTrue(torch.equal(capture.tensor, replay))
        validated = subject.validate_observed_gaussian_provenance(
            self._provenance(capture), editor_noise_seed=noise_seed
        )
        self.assertFalse(validated["training_epsilon_reuse_authorized"])
        self.assertNotEqual(
            validated["generator_initial_seed"], owner_query_seed
        )

    def test_gaussian_replay_injection_or_training_epsilon_reuse_fail_closed(self) -> None:
        owner_query_seed = 2026081502
        noise_seed = subject.editor_noise_seed("dog", owner_query_seed)
        shape = (1, 16, 21, 1, 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        identity = subject.native.value_audit.tensor_identity(
            tensor, label="Gaussian fixture"
        )
        capture = SimpleNamespace(
            call_count=1,
            requested_shape=shape,
            requested_dtype="torch.float32",
            requested_device="cpu",
            returned_dtype="torch.float32",
            returned_device="cpu",
            generator_device="cpu",
            generator_initial_seed=noise_seed,
            raw_value_sha256=identity["raw_storage_sha256"],
            content_sha256=identity["content_sha256"],
        )
        base = self._provenance(capture)
        mutations = {
            "training epsilon reuse": {"training_epsilon_reuse_authorized": True},
            "noise injection": {"observer_replaced_or_injected_noise": True},
            "replay persisted": {"persisted_tensor_is_observer_capture_not_replay": False},
            "wrong role": {"role": "training_epsilon"},
            "owner/generation seed reused": {
                "generator_initial_seed": owner_query_seed
            },
        }
        for label, patch in mutations.items():
            with self.subTest(label=label):
                changed = {**base, **patch}
                with self.assertRaises(subject.CAPERRuntimeError):
                    subject.validate_observed_gaussian_provenance(
                        changed, editor_noise_seed=noise_seed
                    )
        changed = dict(base)
        changed["extra"] = True
        with self.assertRaisesRegex(subject.CAPERRuntimeError, "field closure"):
            subject.validate_observed_gaussian_provenance(
                changed, editor_noise_seed=noise_seed
            )

    def test_cli_is_preflight_only_and_legacy_materializer_is_unreachable(self) -> None:
        parser = subject.build_parser()
        choices = [
            action.choices
            for action in parser._actions
            if isinstance(getattr(action, "choices", None), dict)
        ]
        self.assertEqual(len(choices), 1)
        self.assertEqual(set(choices[0]), {"preflight"})
        self.assertNotIn("materialize-editor", choices[0])
        self.assertFalse(hasattr(subject, "materialize_editor_runtime_packet"))
        with self.assertRaisesRegex(subject.CAPERRuntimeError, "unauthorized"):
            subject._unauthorized_legacy_materialize_editor_runtime_packet(
                object()
            )
        source = inspect.getsource(
            subject._unauthorized_legacy_materialize_editor_runtime_packet
        )
        self.assertLess(source.index("raise CAPERRuntimeError"), source.index("asset ="))


if __name__ == "__main__":
    unittest.main()
