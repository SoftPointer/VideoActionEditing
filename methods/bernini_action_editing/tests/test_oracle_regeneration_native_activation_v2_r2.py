#!/usr/bin/env python3
"""CPU/authority/runtime tests for the isolated Round37 native r2 candidate."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oracle_regeneration_native_runtime_activation_v2 as activation
import self_guided_action_field_v1 as sgaf


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object, *, frozen: bool = True) -> Path:
    path.write_bytes(activation.safe_core.canonical_json_bytes_v1(value))
    if frozen:
        path.chmod(0o444)
    return path


def _file(path: Path, payload: bytes, *, frozen: bool = False) -> tuple[Path, str]:
    path.write_bytes(payload)
    if frozen:
        path.chmod(0o444)
    return path, _sha(path)


@contextmanager
def _without_top_level_torch_module():
    """Exercise the pre-Torch gate even when another test loaded Torch."""

    sentinel = object()
    previous = sys.modules.pop("torch", sentinel)
    try:
        yield
    finally:
        if previous is not sentinel:
            sys.modules["torch"] = previous


def _scheduler_normalization_environment(step_id: str = "307") -> dict[str, str]:
    return {
        "SLURM_JOB_ID": "141620",
        "SLURM_STEP_ID": step_id,
        "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_JOB_ID": "141620",
        "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_STEP_ID": step_id,
        "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_HOSTNAME": "auh7-1b-gpu-226",
        "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_OBSERVED": "/tmp",
        "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_ACTION": (
            "UNSET_BEFORE_ANY_PYTHON"
        ),
    }


def _scheduler_normalization_receipt(step_id: str = "307") -> dict[str, object]:
    environment = _scheduler_normalization_environment(step_id)
    return {
        "schema_version": "bernini-slurm-tmpdir-normalization-r9",
        "slurm_job_id": "141620",
        "slurm_step_id": step_id,
        "hostname": "auh7-1b-gpu-226",
        "scheduler_observed_tmpdir": "/tmp",
        "normalization_action": "UNSET_BEFORE_ANY_PYTHON",
        "normalized_before_any_python_or_torch_import": True,
        "launcher_receipt_environment_keys": sorted(
            key
            for key in environment
            if key.startswith("ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_")
        ),
    }


def _strict_launcher_bundle_pins(source: str) -> dict[str, str]:
    section = source[
        source.index("declare -A expected_bundle_sha256=(") :
        source.index("check_frozen()")
    ]
    rows = re.findall(
        r'^\s*\["([^"]+)"\]="([^"]*)"\s*$',
        section,
        flags=re.MULTILINE,
    )
    if not rows or len(rows) != len({relative for relative, _ in rows}):
        raise ValueError("launcher bundle pins are absent or duplicated")
    malformed = [
        relative
        for relative, digest in rows
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ]
    if malformed:
        raise ValueError(f"launcher bundle digest is malformed: {malformed[0]}")
    return dict(rows)


class ActivationV2StaticTests(unittest.TestCase):
    def test_real_e02_vae_digest_diagnostic_identifies_exact_failed_predicates(
        self,
    ) -> None:
        expected_source = (
            "89511902ae5a7521485c4255321cee929dbf574e9b067b42d8d889c30f992775"
        )
        expected_references = (
            "1d0b29800e179391683c25c442dbbaad576759b5e3726c88a560fe68297b4791",
            "678dc00b4702b1b6aa9cc72d5956898589e9530b44a01ebf109f0fb709cf87d2",
            "7c74041b3826659fa414f22a0d4dca22d9d37dea16af0f6a24a2cefb9f843664",
            "4096e260254eeefa875e3655660b57a3bf60af9b2e851d972d1053ed3c8e28e2",
        )
        observed_references = list(expected_references)
        observed_references[1] = hashlib.sha256(b"live-ref-27-drift").hexdigest()
        observed_references[3] = hashlib.sha256(b"live-ref-80-drift").hexdigest()
        observed_slices = [
            hashlib.sha256(f"live-source-slice-{phase}".encode()).hexdigest()
            for phase in activation.REFERENCE_LATENT_PHASES
        ]
        observed_slices[2] = expected_references[0]
        diagnostic = activation._reference_provenance_diagnostic_v2(
            expected_source_sha256=expected_source,
            observed_source_sha256=hashlib.sha256(b"live-full-source-drift").hexdigest(),
            expected_reference_sha256=expected_references,
            observed_reference_sha256=observed_references,
            observed_source_slice_sha256=observed_slices,
        )
        self.assertFalse(diagnostic["contract_matches"])
        self.assertFalse(diagnostic["source_matches"])
        self.assertEqual(diagnostic["reference_mismatch_positions"], [1, 3])
        self.assertEqual(
            diagnostic["source_slice_collision_pairs"],
            [{"reference_position": 0, "source_phase": 13}],
        )
        self.assertEqual(
            tuple(diagnostic["expected_reference_sha256"]), expected_references
        )

    def test_independent_packet_roots_are_compiled_but_release_preflight_is_closed(
        self,
    ) -> None:
        self.assertEqual(
            activation.COMPILED_AUTHORITY_PACKET_SHA256,
            "6ae5602350d54696e0ddcd716a311f96a3569c6f062622840ad130fcbba0baeb",
        )
        self.assertEqual(
            activation.COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256,
            "5a9efae443bc8d3cb0886dee7f950204377f653f7dbc474f820d7abbbe437e51",
        )
        self.assertTrue(activation.compiled_activation_available_v2())
        with self.assertRaises((FileNotFoundError, activation.OracleActivationV2Error)):
            activation.load_compiled_activation_authority_v2(
                Path("/not/used"), Path("/not/used")
            )

    def test_new_python_sources_have_no_literal_duplicate_dict_keys(self) -> None:
        paths = (
            METHOD_ROOT / "oracle_regeneration_native_runtime_activation_v2.py",
            METHOD_ROOT / "infer_oracle_regeneration_native_activation_v2_r2.py",
            METHOD_ROOT / "preflight_oracle_regeneration_native_activation_v2_r2.py",
            METHOD_ROOT / "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py",
            METHOD_ROOT / "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py",
        )
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                keys = [
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, (str, int, float, bytes, bool))
                ]
                self.assertEqual(
                    len(keys), len(set(keys)), f"duplicate literal dict key in {path}:{node.lineno}"
                )

    def test_frozen_materializer_core_is_not_the_execution_runtime(self) -> None:
        frozen_core = METHOD_ROOT / "oracle_regeneration_activation_v2.py"
        runtime_copy = METHOD_ROOT / "oracle_regeneration_native_runtime_activation_v2.py"
        self.assertEqual(
            _sha(frozen_core),
            "ef97259dd181ff065267e32f1e5cca158e26ad5174457780163658a3db728bb0",
        )
        self.assertNotEqual(_sha(runtime_copy), _sha(frozen_core))
        self.assertEqual(Path(activation.__file__).resolve(strict=True), runtime_copy)

    def test_source_bound_template_instructions_use_newline_canonical_program_hashes(
        self,
    ) -> None:
        template_path = (
            METHOD_ROOT
            / "assets/oracle_regeneration_activation_v2_authoring_template.json"
        )
        template = json.loads(template_path.read_text(encoding="utf-8"))
        cases = {row["case_id"]: row for row in template["cases"]}
        self.assertEqual(tuple(cases), activation.ALLOWED_CASES)
        for case_id in activation.ALLOWED_CASES:
            row = dict(cases[case_id])
            row["structured_action_program_sha256"] = (
                activation.EXPECTED_CASE_BINDINGS[case_id][
                    "structured_action_program_sha256"
                ]
            )
            activation._validate_instruction(row, case_id=case_id)
            self.assertEqual(
                activation._canonical_object_sha256(
                    row["structured_action_program"]
                ),
                activation.EXPECTED_CASE_BINDINGS[case_id][
                    "structured_action_program_sha256"
                ],
            )

    def test_checkpoint_identity_uses_materializer_newline_canonical_domain(
        self,
    ) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner

        # Exact validate_checkpoint_content result from the accepted e02
        # materialization run.  Its independently frozen VAE and prompt
        # receipts both bind the newline-canonical digest below.
        checkpoint_identity = {
            "manifest_path": (
                "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
                "VideoEditing/VideoEdit_experiments/"
                "bernini_appearance_counterfactual_20260808_74ed30c/runtime/"
                "methods/bernini_action_editing/audits/"
                "bernini_r13_ff4c5d4_checkpoint.sha256"
            ),
            "manifest_sha256_computed": (
                "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
            ),
            "manifest_sha256_expected": (
                "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
            ),
            "verified_file_count": 23,
            "every_file_sha256_verified": True,
            "verified_entries_digest": (
                "676e6104eebee3ab1066c70f40af385346b013a3afcab8cafb06c5290994d9ba"
            ),
        }
        material_receipt_digest = (
            "6adb7fb0a9093a2ea4ff96e5899118bbeffc10fc1036916f10f2bf06317b11dd"
        )
        legacy_no_lf_digest = (
            "d860c1815517b5a711541d01014756e2ee36b684b266a62fe961400a7332ef38"
        )
        no_lf = json.dumps(
            checkpoint_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        with_lf = activation.safe_core.canonical_json_bytes_v1(
            checkpoint_identity
        )
        self.assertEqual(with_lf, no_lf + b"\n")
        self.assertEqual(hashlib.sha256(no_lf).hexdigest(), legacy_no_lf_digest)
        self.assertEqual(hashlib.sha256(with_lf).hexdigest(), material_receipt_digest)
        self.assertEqual(
            runner._checkpoint_content_identity_sha256(
                checkpoint_identity, activation=activation
            ),
            material_receipt_digest,
        )
        self.assertEqual(
            runner._canonical_sha256(checkpoint_identity), legacy_no_lf_digest
        )
        self.assertNotEqual(material_receipt_digest, legacy_no_lf_digest)

        runner_source = (
            METHOD_ROOT / "infer_oracle_regeneration_native_activation_v2_r2.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(
            runner_source.count("_checkpoint_content_identity_sha256("), 7
        )
        self.assertIsNone(
            re.search(
                r"_canonical_sha256\(\s*checkpoint_(?:identity|before|after|final)",
                runner_source,
            )
        )
        self.assertIn("_canonical_sha256(receipt)", runner_source)

        hostile_activation = SimpleNamespace(
            safe_core=activation.safe_core,
            _canonical_object_sha256=runner._canonical_sha256,
        )
        with self.assertRaisesRegex(
            runner.NativeActivationV2RunnerError,
            "checkpoint canonical identity digest differs",
        ):
            runner._checkpoint_content_identity_sha256(
                checkpoint_identity, activation=hostile_activation
            )

    def test_real_prompt_receipt_has_only_three_exact_copied_local_roles(
        self,
    ) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner

        materializer_root = (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "action_flow_noise_stage0_job140846_v1/stage1/"
            "oracle_activation_v2_materializer_r2_ef97259d_07b40cdd_"
            "5bba9f97_6da8c414/methods/bernini_action_editing"
        )
        expected = {
            "tokenizer_code": {
                "receipt_path": f"{materializer_root}/infer_lora.py",
                "release_relative_path": "infer_lora.py",
                "sha256": (
                    "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
                ),
            },
            "prompt_builder_code": {
                "receipt_path": (
                    f"{materializer_root}/infer_native_branch_homotopy_canary.py"
                ),
                "release_relative_path": "infer_native_branch_homotopy_canary.py",
                "sha256": (
                    "d6dab735ce52da151848c96f9e00775994dc281ace20afa6dcb9fb64709e5983"
                ),
            },
            "native_prompt_code": {
                "receipt_path": (
                    f"{materializer_root}/infer_native_identity_generation_canary.py"
                ),
                "release_relative_path": "infer_native_identity_generation_canary.py",
                "sha256": (
                    "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42"
                ),
            },
        }
        self.assertEqual(runner._COPIED_LOCAL_PROMPT_ROLE_BINDINGS, expected)
        for row in expected.values():
            self.assertEqual(
                _sha(METHOD_ROOT / row["release_relative_path"]), row["sha256"]
            )
        for external_role in (
            "vae_code",
            "autoencoder_class_module",
            "renderer_code",
            "prompt_cleaner_code",
            "auto_tokenizer_module",
            "text_encoder_class_module",
            "python_executable",
        ):
            self.assertNotIn(
                external_role, runner._COPIED_LOCAL_PROMPT_ROLE_BINDINGS
            )
        runner_source = (
            METHOD_ROOT / "infer_oracle_regeneration_native_activation_v2_r2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("or path != Path(expected_path)", runner_source)
        self.assertIn("or _sha256_file(path) != expected_sha", runner_source)

    def test_copied_local_prompt_dual_origin_closure_and_hostiles(self) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner

        real_bindings = runner._COPIED_LOCAL_PROMPT_ROLE_BINDINGS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            release_root = root / "release"
            receipt_root = root / "materializer"
            wrong_root = root / "wrong"
            release_root.mkdir()
            receipt_root.mkdir()
            wrong_root.mkdir()
            bindings = {}
            paths = {}
            for role, real in real_bindings.items():
                relative = real["release_relative_path"]
                payload = (METHOD_ROOT / relative).read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                self.assertEqual(digest, real["sha256"])
                receipt_path = receipt_root / relative
                current_path = release_root / relative
                wrong_path = wrong_root / relative
                for path in (receipt_path, current_path, wrong_path):
                    path.write_bytes(payload)
                    path.chmod(0o444)
                bindings[role] = {
                    "receipt_path": str(receipt_path),
                    "release_relative_path": relative,
                    "sha256": digest,
                }
                paths[role] = (receipt_path, current_path, wrong_path, payload)

            with mock.patch.object(runner, "METHOD_ROOT", release_root), mock.patch.object(
                runner, "_COPIED_LOCAL_PROMPT_ROLE_BINDINGS", bindings
            ):
                for role, row in bindings.items():
                    receipt_path, current_path, wrong_path, payload = paths[role]
                    self.assertEqual(
                        runner._certify_copied_local_prompt_role_v2(
                            activation=activation,
                            role=role,
                            receipt_path=str(receipt_path),
                            receipt_sha256=row["sha256"],
                            observed_current_path=current_path,
                            expected_current_pin_sha256=row["sha256"],
                        ),
                        row["sha256"],
                    )
                    with self.assertRaisesRegex(
                        runner.NativeActivationV2RunnerError, "origin differs"
                    ):
                        runner._certify_copied_local_prompt_role_v2(
                            activation=activation,
                            role=role,
                            receipt_path=str(receipt_path),
                            receipt_sha256=row["sha256"],
                            observed_current_path=wrong_path,
                            expected_current_pin_sha256=row["sha256"],
                        )
                    with self.assertRaisesRegex(
                        runner.NativeActivationV2RunnerError,
                        "receipt binding differs",
                    ):
                        runner._certify_copied_local_prompt_role_v2(
                            activation=activation,
                            role=role,
                            receipt_path=str(wrong_path),
                            receipt_sha256=row["sha256"],
                            observed_current_path=current_path,
                            expected_current_pin_sha256=row["sha256"],
                        )

                    receipt_path.chmod(0o644)
                    receipt_path.write_bytes(payload + b"tamper")
                    receipt_path.chmod(0o444)
                    with self.assertRaisesRegex(
                        runner.NativeActivationV2RunnerError, "bytes differ"
                    ):
                        runner._certify_copied_local_prompt_role_v2(
                            activation=activation,
                            role=role,
                            receipt_path=str(receipt_path),
                            receipt_sha256=row["sha256"],
                            observed_current_path=current_path,
                            expected_current_pin_sha256=row["sha256"],
                        )
                    receipt_path.chmod(0o644)
                    receipt_path.write_bytes(payload)
                    receipt_path.chmod(0o444)

                    current_path.chmod(0o644)
                    current_path.write_bytes(payload + b"current-tamper")
                    current_path.chmod(0o444)
                    with self.assertRaisesRegex(
                        runner.NativeActivationV2RunnerError, "bytes differ"
                    ):
                        runner._certify_copied_local_prompt_role_v2(
                            activation=activation,
                            role=role,
                            receipt_path=str(receipt_path),
                            receipt_sha256=row["sha256"],
                            observed_current_path=current_path,
                            expected_current_pin_sha256=row["sha256"],
                        )
                    current_path.chmod(0o644)
                    current_path.write_bytes(payload)
                    current_path.chmod(0o444)

                    with self.assertRaisesRegex(
                        runner.NativeActivationV2RunnerError,
                        "receipt binding differs",
                    ):
                        runner._certify_copied_local_prompt_role_v2(
                            activation=activation,
                            role=role,
                            receipt_path=str(receipt_path),
                            receipt_sha256=row["sha256"],
                            observed_current_path=current_path,
                            expected_current_pin_sha256="0" * 64,
                        )

                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError, "not allowlisted"
                ):
                    runner._certify_copied_local_prompt_role_v2(
                        activation=activation,
                        role="renderer_code",
                        receipt_path=str(next(iter(paths.values()))[0]),
                        receipt_sha256="0" * 64,
                        observed_current_path=next(iter(paths.values()))[1],
                        expected_current_pin_sha256="0" * 64,
                    )

    def test_r2_preflight_has_exact_roots_and_rejects_unbound_paths_before_torch(
        self,
    ) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        compiled = (
            preflight.COMPILED_SPEC_SHA256,
            preflight.COMPILED_RUNTIME_SHA256,
            preflight.COMPILED_RUNNER_SHA256,
            preflight.COMPILED_VAE_TOOL_SHA256,
            preflight.COMPILED_PROMPT_TOOL_SHA256,
            preflight.COMPILED_AUTHORITY_PACKET_SHA256,
            preflight.COMPILED_EXTERNAL_LEDGER_SHA256,
        )
        self.assertEqual(
            compiled,
            (
                "edadfa5be1758aaed8b8c4c5f72354bd6becf9a3b999ad116814886e08487d7e",
                "b8e0018893c9582d97d20446956c2bea0506fbc48c7d333a021f70f467edc0d0",
                "ee7fe068096231f222fe9cb153e754b35be3f9078dc00f3924bb7889142aa2f9",
                "07b40cdd67771d257ce546ca4166301980c6768269acc5f097fc08973656bbde",
                "5bba9f977fa40e5044053baaaf73eba779b3816ef6137457a06ceac82a3463af",
                "6ae5602350d54696e0ddcd716a311f96a3569c6f062622840ad130fcbba0baeb",
                "5a9efae443bc8d3cb0886dee7f950204377f653f7dbc474f820d7abbbe437e51",
            ),
        )
        self.assertEqual(
            runner.ARM_ORDER_BY_CASE,
            {
                "e02": (runner.ARM_OFFICIAL, runner.ARM_LOCAL),
                "e03": (),
            },
        )
        self.assertEqual(runner.EXECUTION_CASES, ("e02",))
        with self.assertRaises((FileNotFoundError, preflight.ActivationPreflightError)):
            preflight.validate_release(
                packet_path=Path("/not-consumed-packet"),
                ledger_path=Path("/not-consumed-ledger"),
            )
        self.assertNotIn("torch", sys.modules)

    def test_preflight_normalizes_authority_failure_after_full_component_gate(
        self,
    ) -> None:
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        def frozen_owned_bytes(path: Path, *, label: str):
            del label
            raw = path.read_bytes()
            return raw, {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "mode": 0o444,
                "nlink": 1,
            }

        with mock.patch.object(
            preflight, "_owned_bytes", side_effect=frozen_owned_bytes
        ), mock.patch.object(
            preflight,
            "_validate_host_load_environment",
            return_value={"required": True},
        ), mock.patch.object(
            preflight,
            "_validate_miopen_cache_environment",
            return_value={"root_path": "/unit-cache"},
        ):
            with self.assertRaisesRegex(
                preflight.ActivationPreflightError,
                "compiled authority graph failed validation",
            ):
                preflight.validate_release(
                    packet_path=Path("/not-consumed-packet"),
                    ledger_path=Path("/not-consumed-ledger"),
                )
        self.assertNotIn("torch", sys.modules)

    def test_runner_has_no_training_selection_or_connected_surface(self) -> None:
        runner = (
            METHOD_ROOT / "infer_oracle_regeneration_native_activation_v2_r2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_sample_with_native_initial_noise_observer", runner)
        self.assertIn("LocalOracleNativeBranchRuntimePatchV2", runner)
        self.assertIn('EXECUTION_CASES = ("e02",)', runner)
        self.assertIn('"e03": ()', runner)
        self.assertIn('"automatic_replacement": False', runner)
        self.assertIn('"selection_authority": None', runner)
        self.assertIn('"self_generated_anchor_tensor_used": False', runner)
        self.assertNotIn("optimizer.step", runner)
        self.assertNotIn("backward(", runner)
        self.assertNotIn("flowedit_step0_target_noise_v1(", runner)
        self.assertLess(
            runner.index("miopen_cache_initial = _activate_miopen_cache_pre_torch"),
            runner.index("\n    import torch\n", runner.index("def main(")),
        )

    def test_dynamic_native_schedule_is_bound_before_torch_in_all_release_layers(
        self,
    ) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        name = "source_self_native_ref_contrastive_v3.py"
        digest = "d8825bc167c64e497f8d29c807d9b0a69d9a9a59de09afee863b7fc9df2bdeb0"
        self.assertEqual(
            activation.FROZEN_DEPENDENCY_PINS[
                "native_unipc40_schedule_contract"
            ],
            (name, digest),
        )
        self.assertEqual(
            activation._EXPECTED_IMPORT_ORIGINS[name[:-3]], name
        )
        self.assertEqual(preflight._RUNTIME_DEPENDENCY_RELATIVE_PATHS[name], name)
        self.assertEqual(
            runner._LOCAL_RELEASE_PATHS[f"dependency:{name}"], name
        )
        spec = json.loads(
            (
                METHOD_ROOT
                / "assets/oracle_regeneration_native_activation_v2_r2_spec.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(spec["frozen_runtime_dependencies"][name], digest)
        with mock.patch.dict(sys.modules, {name[:-3]: SimpleNamespace()}):
            with self.assertRaisesRegex(
                preflight.ActivationPreflightError,
                "loaded before the CPU gate",
            ):
                preflight.validate_release(
                    packet_path=Path("/not-consumed-packet"),
                    ledger_path=Path("/not-consumed-ledger"),
                )

    def test_spec_is_exact_e02_execution_e03_policy_and_launcher_is_pinned(
        self,
    ) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        spec_path = (
            METHOD_ROOT
            / "assets/oracle_regeneration_native_activation_v2_r2_spec.json"
        )
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.assertEqual(spec["cases"], preflight._EXPECTED_SPEC_CASES)
        self.assertEqual(
            spec["host_load_contract"], preflight._EXPECTED_HOST_LOAD_CONTRACT
        )
        self.assertEqual(
            spec["allocation_contract"], preflight._EXPECTED_ALLOCATION_CONTRACT
        )
        self.assertEqual(
            spec["miopen_cache_contract"],
            preflight._EXPECTED_MIOPEN_CACHE_CONTRACT,
        )
        preflight_source = Path(preflight.__file__).read_text(encoding="utf-8")
        runner_source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn(
            '"scheduler_tmpdir_normalization": dict(\n'
            '            miopen_cache["scheduler_tmpdir_normalization"]',
            preflight_source,
        )
        self.assertIn(
            'scheduler_tmpdir_normalization = preflight_receipt.get(',
            runner_source,
        )
        self.assertIn(
            'label="post-runtime-init node-local tmp baseline"',
            runner_source,
        )
        self.assertNotIn(
            "initial WORLD4 node-local tmp is not empty",
            runner_source,
        )
        self.assertEqual(
            spec["scientific_boundary"], preflight._EXPECTED_SCIENTIFIC_BOUNDARY
        )
        self.assertEqual(
            spec["post_run_contract"], preflight._EXPECTED_POST_RUN_CONTRACT
        )
        self.assertEqual(spec["cases"]["e03"]["arms"], [])
        self.assertFalse(spec["cases"]["e03"]["executed"])
        self.assertIsNone(spec["cases"]["e03"]["vae_reference_receipt"])
        self.assertIsNone(spec["cases"]["e03"]["prompt_receipt"])

        launcher = (
            METHOD_ROOT
            / "scripts/auh_run_oracle_regeneration_native_activation_v2_r2.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"__[A-Z][A-Z0-9_]+__", source))
        self.assertIn(
            "bernini-launcher-node-local-tmp-fresh-empty-proof-r11", source
        )
        self.assertIn(
            "caller launcher local-tmp empty proof is forbidden", source
        )
        self.assertIn(
            "ORACLE_ACTIVATION_V2_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_SHA256",
            source,
        )
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        executable_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for forbidden in ("ssh ", "srun ", "sbatch "):
            self.assertNotIn(forbidden, executable_source)
        self.assertIn("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1", source)
        self.assertIn("NATIVE_V_AXIS_LOAD_LOCK=", source)
        self.assertIn("MIOPEN_USER_DB_PATH=", source)
        self.assertIn("MIOPEN_CUSTOM_CACHE_DIR=", source)
        self.assertIn(".miopen-cache-r9", source)
        self.assertIn("launcher-bootstrap-user-db", source)
        self.assertIn('grep \'^MIOPEN_\'', source)
        self.assertIn('local_tmp_root="/tmp/oracle-regeneration-native-v2-r9-u', source)
        self.assertIn('"${local_tmp_root}/rank-${rank}"', source)
        self.assertIn('--miopen-cache-root "${cache_root}"', source)
        self.assertIn('--miopen-local-tmp-root "${local_tmp_root}"', source)
        self.assertNotIn("MIOPEN_DISABLE_CACHE=", source)
        self.assertNotIn("MIOPEN_SYSTEM_DB_PATH=", source)
        first_python = source.index('"${python_bin}" -I -B -c')
        self.assertLess(
            source.index('export MIOPEN_USER_DB_PATH="${cache_root}/launcher-bootstrap-user-db"'),
            first_python,
        )
        self.assertLess(
            source.index('export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/launcher-bootstrap-kernel-cache"'),
            first_python,
        )
        self.assertLess(source.index("unset TMPDIR"), first_python)
        self.assertIn('[[ -v TMPDIR && "${TMPDIR}" == "/tmp" ]]', source)
        self.assertIn("for temp_name in TMP TEMP TEMPDIR", source)
        self.assertIn("unset TMPDIR", source)
        self.assertIn(
            "caller scheduler TMPDIR normalization receipt is forbidden", source
        )
        self.assertIn('SLURM_STEP_ID:-}', source)
        self.assertIn(
            'ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_ACTION="UNSET_BEFORE_ANY_PYTHON"',
            source,
        )
        self.assertIn('! -e "${cache_root}" && ! -L "${cache_root}"', source)
        self.assertIn('mkdir -m 700 -- "${cache_root}"', source)
        self.assertIn('[[ -d /tmp && ! -L /tmp', source)
        self.assertIn('"$(stat_mode /tmp)" == "1777"', source)
        self.assertIn('"$(stat_uid /tmp)" == "0"', source)
        self.assertIn('cache_uid="$(id -u)"', source)
        self.assertIn('"${cache_uid}" "${output_dir}" | sha256sum', source)
        self.assertIn('[[ ! -e "${local_tmp_root}" && ! -L "${local_tmp_root}" ]]', source)
        self.assertIn('mkdir -m 700 -- "${local_tmp_root}"', source)
        self.assertIn('"$(stat_device "${local_tmp_root}")" == "$(stat_device /tmp)"', source)
        self.assertIn('"$(stat_device "${local_tmp_root}")" != "$(stat_device "${cache_root}")"', source)
        self.assertNotIn('rm -rf -- "${local_tmp_root}"', source)
        self.assertLess(source.index('mkdir -m 700 -- "${local_tmp_root}"'), first_python)
        self.assertIn("output/cache must be outside frozen bundle", source)
        with self.assertRaisesRegex(
            runner.NativeActivationV2RunnerError, "outside the frozen release tree"
        ):
            runner._fresh_output_dir(
                str(METHOD_ROOT / "unit-output-must-not-live-in-release")
            )
        self.assertIn("--nproc_per_node=4", source)
        self.assertIn('visible_gpus}" == "0,1,2,3"', source)
        self.assertIn('SLURM_JOB_ID:-}" == "141620"', source)
        self.assertIn("auh7-1b-gpu-226)", source)
        for stale_node in (
            "auh7-1b-gpu-246|",
            "auh7-1b-gpu-247|",
            "auh7-1b-gpu-248|",
            "auh7-1b-gpu-279)",
        ):
            self.assertNotIn(stale_node, source)
        self.assertIn("6ae5602350d54696", source)
        self.assertIn("5a9efae443bc8d3c", source)

        pinned = _strict_launcher_bundle_pins(source)
        expected_bundle_paths = set(runner._LOCAL_RELEASE_PATHS.values()) | {
            "assets/oracle_regeneration_activation_v2_authoring_template.json",
            "assets/oracle_regeneration_native_activation_v2_r2_host_load.lock",
            "oracle_regeneration_activation_v2.py",
            "preflight_oracle_regeneration_native_activation_v2_r2.py",
            "tests/test_native_branch_homotopy_runtime_v1.py",
            "tests/test_oracle_regeneration_native_activation_v2_r2.py",
        }
        self.assertEqual(set(pinned), expected_bundle_paths)
        self.assertEqual(len(pinned), 31)
        schedule_path = "source_self_native_ref_contrastive_v3.py"
        self.assertEqual(
            pinned[schedule_path],
            "d8825bc167c64e497f8d29c807d9b0a69d9a9a59de09afee863b7fc9df2bdeb0",
        )
        for relative, digest in pinned.items():
            self.assertEqual(_sha(METHOD_ROOT / relative), digest, relative)
        malformed = source.replace(
            pinned[schedule_path], pinned[schedule_path][:-2], 1
        )
        with self.assertRaisesRegex(ValueError, "digest is malformed"):
            _strict_launcher_bundle_pins(malformed)

    def test_launcher_executes_the_real_local_bundle_gate(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts/auh_run_oracle_regeneration_native_activation_v2_r2.sh"
        )
        pins = _strict_launcher_bundle_pins(launcher.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve(strict=True) / "release"
            method = repo / "methods/bernini_action_editing"
            for relative in pins:
                destination = method / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(METHOD_ROOT / relative, destination)
                destination.chmod(0o444)
            launcher_copy = (
                method
                / "scripts/auh_run_oracle_regeneration_native_activation_v2_r2.sh"
            )
            launcher_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(launcher, launcher_copy)
            launcher_copy.chmod(0o555)
            frozen_directories = [method] + [
                path for path in method.rglob("*") if path.is_dir()
            ]
            for directory in sorted(
                frozen_directories,
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            result = subprocess.run(
                ["bash", str(launcher_copy), "--verify-local-bundle-only"],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "ORACLE_ACTIVATION_V2_REPO_ROOT": str(repo),
                },
            )
            for directory in frozen_directories:
                directory.chmod(0o755)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("local frozen bundle verified", result.stdout)


class RunnerNoReplacePublishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner

        cls.runner = runner

    def _enter_context(self, context):
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    @contextmanager
    def _miopen_layout(
        self, root: Path, output: Path, *, step_id: str = "307"
    ):
        """Build split persistent/local fixtures with synthetic devices.

        macOS exposes the workspace and ``/tmp`` on one device.  Production
        requires distinct NFS/ext4 devices, so only the device numbers in the
        owned directory identities are substituted here; paths, modes, uid,
        inode, emptiness, and all bytes remain real.
        """

        runner = self.runner
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        cache_root = Path(f"{output}.miopen-cache-r9")
        cache_root.mkdir(mode=0o700)
        cache_root.chmod(0o700)
        for name in sorted(runner._expected_miopen_cache_directory_names()):
            path = cache_root / name
            path.mkdir(mode=0o700)
            path.chmod(0o700)

        local_parent = root / "exact-node-local-tmp"
        local_parent.mkdir(mode=0o700)
        local_parent.chmod(0o700)
        scheduler = _scheduler_normalization_receipt(step_id)
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    runner, "MIOPEN_LOCAL_TMP_PARENT", local_parent
                )
            )
            stack.enter_context(
                mock.patch.object(
                    preflight, "_MIOPEN_LOCAL_TMP_PARENT", local_parent
                )
            )
            local_tmp_root = runner._expected_miopen_local_tmp_root(
                output, scheduler
            )
            local_tmp_root.mkdir(mode=0o700)
            local_tmp_root.chmod(0o700)
            for name in sorted(
                runner._expected_miopen_local_tmp_directory_names()
            ):
                path = local_tmp_root / name
                path.mkdir(mode=0o700)
                path.chmod(0o700)

            runner_private = runner._private_directory_identity
            preflight_private = preflight._private_directory_identity

            def on_or_below(path: Path, parent: Path) -> bool:
                try:
                    path.relative_to(parent)
                    return True
                except ValueError:
                    return False

            def runner_identity(path: Path, *, label: str):
                row = dict(runner_private(path, label=label))
                if on_or_below(path, cache_root):
                    row["device"] = 47
                elif on_or_below(path, local_tmp_root):
                    row["device"] = int(local_tmp_root.stat().st_dev)
                return row

            def preflight_identity(path: Path, *, label: str):
                row = dict(preflight_private(path, label=label))
                if on_or_below(path, cache_root):
                    row["device"] = 47
                elif on_or_below(path, local_tmp_root):
                    row["device"] = int(local_tmp_root.stat().st_dev)
                return row

            parent_row = {
                "path": str(local_parent),
                "mode": 0o1777,
                "uid": 0,
                "gid": 0,
                "device": int(local_tmp_root.stat().st_dev),
                "inode": int(local_parent.stat().st_ino),
            }
            stack.enter_context(
                mock.patch.object(
                    runner,
                    "_private_directory_identity",
                    side_effect=runner_identity,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    preflight,
                    "_private_directory_identity",
                    side_effect=preflight_identity,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    runner,
                    "_node_local_tmp_parent_identity",
                    return_value=parent_row,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    preflight,
                    "_node_local_tmp_parent_identity",
                    return_value=parent_row,
                )
            )
            local_tmp_root_identity = runner_identity(
                local_tmp_root, label="fixture local tmp root"
            )
            local_tmp_directory_identities = {
                name: runner_identity(
                    local_tmp_root / name, label=f"fixture local tmp {name}"
                )
                for name in sorted(
                    runner._expected_miopen_local_tmp_directory_names()
                )
            }
            launcher_local_tmp_empty_proof = (
                runner._launcher_local_tmp_empty_proof(
                    local_tmp_root_identity, local_tmp_directory_identities
                )
            )
            self.assertEqual(
                launcher_local_tmp_empty_proof,
                preflight._launcher_local_tmp_empty_proof(
                    local_tmp_root_identity, local_tmp_directory_identities
                ),
            )
            yield {
                "cache_root": cache_root,
                "local_tmp_root": local_tmp_root,
                "local_parent": local_parent,
                "parent_identity": parent_row,
                "local_tmp_root_identity": local_tmp_root_identity,
                "local_tmp_directory_identities": (
                    local_tmp_directory_identities
                ),
                "launcher_local_tmp_empty_proof": (
                    launcher_local_tmp_empty_proof
                ),
            }

    def test_scheduler_tmpdir_normalization_receipt_and_hostiles(self) -> None:
        runner = self.runner
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        expected = _scheduler_normalization_receipt()
        valid = _scheduler_normalization_environment()
        with mock.patch.dict(os.environ, valid, clear=True), mock.patch.object(
            runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
        ), mock.patch.object(
            preflight.socket, "gethostname", return_value="auh7-1b-gpu-226"
        ):
            self.assertEqual(
                runner._certify_scheduler_tmpdir_normalization(), expected
            )
            self.assertEqual(
                preflight._validate_scheduler_tmpdir_normalization(), expected
            )

        hostile_environments = []
        wrong_observed = dict(valid)
        wrong_observed[
            "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_OBSERVED"
        ] = "/private/tmp"
        hostile_environments.append(wrong_observed)
        missing_step = dict(valid)
        del missing_step["ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_STEP_ID"]
        hostile_environments.append(missing_step)
        wrong_step = dict(valid)
        wrong_step["SLURM_STEP_ID"] = "batch"
        wrong_step["ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_STEP_ID"] = "batch"
        hostile_environments.append(wrong_step)
        extra_receipt_field = dict(valid)
        extra_receipt_field[
            "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_UNREVIEWED"
        ] = "1"
        hostile_environments.append(extra_receipt_field)
        caller_tmp_survived = dict(valid)
        caller_tmp_survived["TMPDIR"] = "/tmp"
        hostile_environments.append(caller_tmp_survived)

        for hostile in hostile_environments:
            with self.subTest(hostile=hostile), mock.patch.dict(
                os.environ, hostile, clear=True
            ), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ), mock.patch.object(
                preflight.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ):
                if hostile is caller_tmp_survived:
                    self.assertEqual(
                        runner._certify_scheduler_tmpdir_normalization(), expected
                    )
                else:
                    with self.assertRaises(
                        runner.NativeActivationV2RunnerError
                    ):
                        runner._certify_scheduler_tmpdir_normalization()
                with self.assertRaises(preflight.ActivationPreflightError):
                    preflight._validate_scheduler_tmpdir_normalization()

    def test_miopen_pre_torch_bootstrap_switch_exact_allowlist_and_library(self) -> None:
        runner = self.runner
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            output = root / "result"
            layout = self._enter_context(self._miopen_layout(root, output))
            cache_root = layout["cache_root"]
            local_tmp_root = layout["local_tmp_root"]
            domain_payload = b"\x00".join(
                (
                    runner.MIOPEN_LOCAL_TMP_DOMAIN.encode("utf-8"),
                    b"141620",
                    b"307",
                    str(os.geteuid()).encode("ascii"),
                    str(output).encode("utf-8"),
                )
            )
            self.assertEqual(
                local_tmp_root.name,
                "oracle-regeneration-native-v2-r9-"
                f"u{os.geteuid()}-j141620-s307-o"
                f"{hashlib.sha256(domain_payload).hexdigest()}",
            )
            library = root / "libMIOpen.so"
            library.write_bytes(b"unit MIOpen 3.3 library")
            library.chmod(0o755)
            digest = _sha(library)
            environment = {
                **_scheduler_normalization_environment(),
                runner.MIOPEN_CACHE_ROOT_ENV: str(cache_root),
                runner.MIOPEN_LOCAL_TMP_ROOT_ENV: str(local_tmp_root),
                runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV: str(
                    layout["launcher_local_tmp_empty_proof"]["sha256"]
                ),
                "ORACLE_ACTIVATION_V2_OUTPUT_DIR": str(output),
                runner.MIOPEN_USER_DB_ENV: str(
                    cache_root / "launcher-bootstrap-user-db"
                ),
                runner.MIOPEN_KERNEL_CACHE_ENV: str(
                    cache_root / "launcher-bootstrap-kernel-cache"
                ),
                "LOCAL_RANK": "2",
                "RANK": "2",
                "WORLD_SIZE": "4",
                "LOCAL_WORLD_SIZE": "4",
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ), mock.patch.object(
                runner, "MIOPEN_LIBRARY_PATH", library
            ), mock.patch.object(
                runner, "MIOPEN_LIBRARY_SHA256", digest
            ), mock.patch.object(
                runner, "MIOPEN_LIBRARY_SIZE", library.stat().st_size
            ):
                os.environ["TMPDIR"] = "/tmp"
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "inherited MIOpen bootstrap environment differs",
                ), _without_top_level_torch_module():
                    runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(local_tmp_root),
                    )
                del os.environ["TMPDIR"]
                original_proof = os.environ[
                    runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV
                ]
                os.environ[
                    runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV
                ] = "0" * 64
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "launcher node-local tmp empty proof differs",
                ), _without_top_level_torch_module():
                    runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(local_tmp_root),
                    )
                os.environ[
                    runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV
                ] = original_proof
                with _without_top_level_torch_module():
                    initial = runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(local_tmp_root),
                    )
                self.assertEqual(
                    os.environ[runner.MIOPEN_USER_DB_ENV],
                    str(cache_root / "rank-2-user-db"),
                )
                self.assertEqual(
                    os.environ[runner.MIOPEN_KERNEL_CACHE_ENV],
                    str(cache_root / "rank-2-kernel-cache"),
                )
                self.assertEqual(
                    os.environ["TMPDIR"], str(local_tmp_root / "rank-2")
                )
                self.assertEqual(initial["local_tmp_parent_identity"], layout["parent_identity"])
                self.assertEqual(initial["library"]["mode"], 0o755)
                self.assertEqual(
                    initial["launcher_local_tmp_fresh_empty_proof"],
                    layout["launcher_local_tmp_empty_proof"],
                )
                runner._certify_active_miopen_environment(initial)
                os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "unexpected MIOpen solver environment",
                ):
                    runner._certify_active_miopen_environment(initial)
                del os.environ["MIOPEN_FIND_MODE"]
                marker_name = (
                    "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_OBSERVED"
                )
                os.environ[marker_name] = "/wrong"
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "scheduler TMPDIR normalization",
                ):
                    runner._certify_active_miopen_environment(initial)
                os.environ[marker_name] = "/tmp"
                cache_root.chmod(0o755)
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "path differs|canonical/private/effective-user-owned",
                ):
                    runner._certify_active_miopen_environment(initial)
                cache_root.chmod(0o700)

            preflight_environment = {
                **_scheduler_normalization_environment(),
                runner.MIOPEN_CACHE_ROOT_ENV: str(cache_root),
                runner.MIOPEN_LOCAL_TMP_ROOT_ENV: str(local_tmp_root),
                runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV: str(
                    layout["launcher_local_tmp_empty_proof"]["sha256"]
                ),
                "ORACLE_ACTIVATION_V2_OUTPUT_DIR": str(output),
                runner.MIOPEN_USER_DB_ENV: str(
                    cache_root / "launcher-bootstrap-user-db"
                ),
                runner.MIOPEN_KERNEL_CACHE_ENV: str(
                    cache_root / "launcher-bootstrap-kernel-cache"
                ),
            }
            with mock.patch.dict(os.environ, preflight_environment, clear=True), mock.patch.object(
                preflight.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ), mock.patch.object(
                preflight, "_MIOPEN_LIBRARY_PATH", library
            ), mock.patch.object(
                preflight, "_MIOPEN_LIBRARY_SHA256", digest
            ), mock.patch.object(
                preflight, "_MIOPEN_LIBRARY_SIZE", library.stat().st_size
            ):
                with _without_top_level_torch_module():
                    receipt = preflight._validate_miopen_cache_environment()
                self.assertEqual(receipt["library"]["mode"], 0o755)
                self.assertEqual(
                    receipt["local_tmp_parent_identity"],
                    layout["parent_identity"],
                )
                self.assertEqual(
                    receipt["scheduler_tmpdir_normalization"],
                    _scheduler_normalization_receipt(),
                )
                self.assertEqual(
                    receipt["launcher_local_tmp_fresh_empty_proof"],
                    layout["launcher_local_tmp_empty_proof"],
                )
                self.assertEqual(
                    receipt["cpu_preflight_local_tmp_empty_proof"],
                    {
                        "root_identity": layout["local_tmp_root_identity"],
                        "directory_identities": layout[
                            "local_tmp_directory_identities"
                        ],
                        "all_rank_directories_empty": True,
                    },
                )
                os.environ[
                    runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV
                ] = "f" * 64
                with self.assertRaisesRegex(
                    preflight.ActivationPreflightError,
                    "launcher node-local tmp empty proof differs",
                ):
                    with _without_top_level_torch_module():
                        preflight._validate_miopen_cache_environment()
                os.environ[
                    runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV
                ] = str(layout["launcher_local_tmp_empty_proof"]["sha256"])
                library.chmod(0o775)
                with self.assertRaisesRegex(
                    preflight.ActivationPreflightError,
                    "library differs",
                ):
                    with _without_top_level_torch_module():
                        preflight._validate_miopen_cache_environment()
                library.chmod(0o755)
                os.environ["MIOPEN_FIND_ENFORCE"] = "SEARCH"
                with self.assertRaisesRegex(
                    preflight.ActivationPreflightError,
                    "bootstrap environment differs",
                ):
                    with _without_top_level_torch_module():
                        preflight._validate_miopen_cache_environment()

    def test_miopen_known_33_cache_modes_and_rank_private_lock_inventory(self) -> None:
        runner = self.runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            output = root / "result"
            layout = self._enter_context(self._miopen_layout(root, output))
            cache_root = layout["cache_root"]
            local_tmp_root = layout["local_tmp_root"]
            user_db = cache_root / "rank-1-user-db"
            kernel = cache_root / "rank-1-kernel-cache"
            rank_tmp = local_tmp_root / "rank-1"
            main_name = sorted(runner.MIOPEN_USER_DB_MAIN_BASENAMES)[0]
            main = user_db / main_name
            main.write_bytes(b"plaintext-db")
            main.chmod(0o777)
            sidecar = user_db / f"{main_name}.time"
            sidecar.write_bytes(b"time")
            sidecar.chmod(0o666)
            ukdb = kernel / "gfx90a68.ukdb"
            ukdb.write_bytes(b"sqlite")
            ukdb.chmod(0o600)
            lock_dir = rank_tmp / "miopen-lockfiles"
            lock_dir.mkdir(mode=0o777)
            lock_dir.chmod(0o777)
            lock_name = sorted(runner._expected_miopen_lock_basenames(user_db))[0]
            lock = lock_dir / lock_name
            lock.write_bytes(b"")
            lock.chmod(0o777)
            initial = {
                "rank": 1,
                "local_rank": 1,
                "root": runner._private_directory_identity(
                    cache_root, label="unit cache root"
                ),
                "role_directory_identities": {
                    name: runner._private_directory_identity(
                        cache_root / name, label=f"unit cache {name}"
                    )
                    for name in sorted(
                        runner._expected_miopen_cache_directory_names()
                    )
                },
                "local_tmp_root": runner._private_directory_identity(
                    local_tmp_root, label="unit local tmp root"
                ),
                "local_tmp_parent_identity": layout["parent_identity"],
                "local_tmp_role_directory_identities": {
                    name: runner._private_directory_identity(
                        local_tmp_root / name,
                        label=f"unit local tmp {name}",
                    )
                    for name in sorted(
                        runner._expected_miopen_local_tmp_directory_names()
                    )
                },
                "launcher_local_tmp_fresh_empty_proof": layout[
                    "launcher_local_tmp_empty_proof"
                ],
                "user_db_path": str(user_db),
                "kernel_cache_path": str(kernel),
                "tmp_path": str(rank_tmp),
                "official_environment": {
                    runner.MIOPEN_USER_DB_ENV: str(user_db),
                    runner.MIOPEN_KERNEL_CACHE_ENV: str(kernel),
                    "TMPDIR": str(rank_tmp),
                    "TMP": None,
                    "TEMP": None,
                    "TEMPDIR": None,
                },
                "scheduler_tmpdir_normalization": (
                    _scheduler_normalization_receipt()
                ),
            }
            environment = {
                **_scheduler_normalization_environment(),
                runner.MIOPEN_USER_DB_ENV: str(user_db),
                runner.MIOPEN_KERNEL_CACHE_ENV: str(kernel),
                runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV: str(
                    layout["launcher_local_tmp_empty_proof"]["sha256"]
                ),
                "TMPDIR": str(rank_tmp),
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ):
                snapshot = runner._stable_local_miopen_cache_snapshot(
                    initial, label="unit MIOpen 3.3"
                )
                self.assertEqual(snapshot["trees"][0]["regular_file_count"], 2)
                self.assertEqual(snapshot["trees"][1]["regular_file_count"], 1)
                local_snapshot = runner._stable_local_tmp_snapshot(
                    initial, label="unit local MIOpen locks"
                )
                self.assertEqual(
                    local_snapshot["tree"]["regular_file_count"], 1
                )
                main.chmod(0o600)
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError, "name/mode differs"
                ):
                    runner._stable_local_miopen_cache_snapshot(
                        initial, label="unit wrong MIOpen mode"
                    )

    def test_miopen_fresh_layout_rejects_reuse_and_role_symlink(self) -> None:
        runner = self.runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            output = root / "result"
            layout = self._enter_context(self._miopen_layout(root, output))
            cache_root = layout["cache_root"]
            local_tmp_root = layout["local_tmp_root"]
            library = root / "libMIOpen.so"
            library.write_bytes(b"unit library")
            library.chmod(0o755)
            environment = {
                **_scheduler_normalization_environment(),
                runner.MIOPEN_CACHE_ROOT_ENV: str(cache_root),
                runner.MIOPEN_LOCAL_TMP_ROOT_ENV: str(local_tmp_root),
                runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV: str(
                    layout["launcher_local_tmp_empty_proof"]["sha256"]
                ),
                runner.MIOPEN_USER_DB_ENV: str(
                    cache_root / "launcher-bootstrap-user-db"
                ),
                runner.MIOPEN_KERNEL_CACHE_ENV: str(
                    cache_root / "launcher-bootstrap-kernel-cache"
                ),
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "4",
                "LOCAL_WORLD_SIZE": "4",
            }
            reused = cache_root / "rank-0-user-db" / "old.db"
            reused.write_bytes(b"reused")
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ), mock.patch.object(
                runner, "MIOPEN_LIBRARY_PATH", library
            ), mock.patch.object(
                runner, "MIOPEN_LIBRARY_SHA256", _sha(library)
            ), mock.patch.object(
                runner, "MIOPEN_LIBRARY_SIZE", library.stat().st_size
            ), _without_top_level_torch_module():
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError, "not empty"
                ):
                    runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(local_tmp_root),
                    )
                reused.unlink()
                role = local_tmp_root / "rank-0"
                role.rmdir()
                target = root / "outside"
                target.mkdir(mode=0o700)
                role.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "path differs|canonical/private/effective-user-owned",
                ):
                    runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(local_tmp_root),
                    )

                role.unlink()
                role.mkdir(mode=0o700)
                role.chmod(0o700)
                other_rank_reuse = local_tmp_root / "rank-3" / "stale.tmp"
                other_rank_reuse.write_bytes(b"must reject other-rank reuse")
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError, "rank-3 is not empty"
                ):
                    runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(local_tmp_root),
                    )
                other_rank_reuse.unlink()

                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "node-local tmp root binding differs",
                ):
                    runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(target),
                    )
                same_device_parent = {
                    **layout["parent_identity"],
                    "device": 47,
                }
                with mock.patch.object(
                    runner,
                    "_node_local_tmp_parent_identity",
                    return_value=same_device_parent,
                ), self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "filesystem separation differs",
                ):
                    runner._activate_miopen_cache_pre_torch(
                        output_dir=output,
                        cache_root_value=str(cache_root),
                        local_tmp_root_value=str(local_tmp_root),
                    )

                # The hostile symlink replacement above deliberately changed
                # one fixture inode.  Recreate the launcher's proof over the
                # restored exact empty layout before testing the worker gate.
                restored_root = runner._private_directory_identity(
                    local_tmp_root, label="restored unit local tmp root"
                )
                restored_roles = {
                    name: runner._private_directory_identity(
                        local_tmp_root / name,
                        label=f"restored unit local tmp {name}",
                    )
                    for name in sorted(
                        runner._expected_miopen_local_tmp_directory_names()
                    )
                }
                os.environ[
                    runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV
                ] = str(
                    runner._launcher_local_tmp_empty_proof(
                        restored_root, restored_roles
                    )["sha256"]
                )
                initial = runner._activate_miopen_cache_pre_torch(
                    output_dir=output,
                    cache_root_value=str(cache_root),
                    local_tmp_root_value=str(local_tmp_root),
                )
                other_rank = local_tmp_root / "rank-3"
                # AUH ext4 may immediately recycle an inode after rmdir.  Keep
                # the replacement live alongside the original first so this
                # same-mode replacement hostile has a deterministically new
                # inode on both ext4 and APFS.
                original_inode = other_rank.stat().st_ino
                replacement = local_tmp_root / "rank-3-replacement"
                replacement.mkdir(mode=0o700)
                replacement.chmod(0o700)
                replacement_inode = replacement.stat().st_ino
                self.assertNotEqual(original_inode, replacement_inode)
                other_rank.rmdir()
                replacement.rename(other_rank)
                self.assertEqual(other_rank.stat().st_ino, replacement_inode)
                self.assertEqual(
                    {entry.name for entry in local_tmp_root.iterdir()},
                    runner._expected_miopen_local_tmp_directory_names(),
                )
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "rank-3 identity changed",
                ):
                    runner._certify_active_miopen_environment(initial)

    def test_miopen_cache_receipt_is_exclusive_bound_and_tamper_detectable(self) -> None:
        runner = self.runner

        class FakeDist:
            @staticmethod
            def all_gather_object(rows, value) -> None:
                rows[:] = [copy.deepcopy(value) for _ in range(4)]

            @staticmethod
            def broadcast_object_list(rows, src=0) -> None:
                del rows, src

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            output = root / "result"
            layout = self._enter_context(self._miopen_layout(root, output))
            cache_root = layout["cache_root"]
            role_identities = {
                name: runner._private_directory_identity(
                    cache_root / name, label=f"unit cache {name}"
                )
                for name in sorted(runner._expected_miopen_cache_directory_names())
            }
            initial = {
                "rank": 0,
                "local_rank": 0,
                "root": runner._private_directory_identity(
                    cache_root, label="unit cache root"
                ),
                "role_directory_identities": role_identities,
                "launcher_bootstrap": {
                    "user_db_path": str(cache_root / "launcher-bootstrap-user-db"),
                    "kernel_cache_path": str(
                        cache_root / "launcher-bootstrap-kernel-cache"
                    ),
                    "torchrun_parent_tmpdir": None,
                    "workers_use_bootstrap_namespace": False,
                },
                "scheduler_tmpdir_normalization": (
                    _scheduler_normalization_receipt()
                ),
            }
            world_rows = [
                {"rank": rank, "snapshot_sha256": f"{rank:064x}"}
                for rank in range(4)
            ]
            bootstrap = {"snapshot_sha256": "1" * 64}
            with mock.patch.dict(
                os.environ, _scheduler_normalization_environment(), clear=True
            ), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ), mock.patch.object(
                runner,
                "_gather_miopen_cache_snapshot",
                return_value=world_rows,
            ), mock.patch.object(
                runner,
                "_gather_miopen_bootstrap_snapshot",
                return_value=bootstrap,
            ):
                sealed = runner._seal_miopen_cache_receipt_world4(
                    initial=initial,
                    activation_rows=[{"rank": rank} for rank in range(4)],
                    initial_world4=world_rows,
                    bootstrap_initial=bootstrap,
                    arm_snapshots={"e02": {}},
                    loaded_library={"sha256": "2" * 64},
                    output_dir=output,
                    torch=SimpleNamespace(),
                    dist=FakeDist(),
                    rank=0,
                )
            receipt_path = Path(sealed["path"])
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o444)
            self.assertEqual(receipt_path.stat().st_nlink, 1)
            self.assertEqual(
                json.loads(receipt_path.read_text(encoding="utf-8"))[
                    "scheduler_tmpdir_normalization"
                ],
                _scheduler_normalization_receipt(),
            )
            self.assertEqual(
                sealed["scheduler_tmpdir_normalization"],
                _scheduler_normalization_receipt(),
            )
            bound_sha = sealed["file_identity"]["sha256"]
            receipt_path.chmod(0o644)
            receipt_path.write_bytes(b"tampered retained cache receipt")
            receipt_path.chmod(0o444)
            self.assertNotEqual(
                runner._owned_file_identity(
                    receipt_path, label="tampered cache receipt"
                )["sha256"],
                bound_sha,
            )

    def test_node_local_tmp_receipt_is_exclusive_bound_ephemeral_and_tamper_detectable(
        self,
    ) -> None:
        runner = self.runner

        class FakeDist:
            @staticmethod
            def all_gather_object(rows, value) -> None:
                rows[:] = [copy.deepcopy(value) for _ in range(4)]

            @staticmethod
            def broadcast_object_list(rows, src=0) -> None:
                del rows, src

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            output = root / "result"
            layout = self._enter_context(self._miopen_layout(root, output))
            cache_root = layout["cache_root"]
            local_tmp_root = layout["local_tmp_root"]
            persistent_path = cache_root / runner.MIOPEN_CACHE_RECEIPT_NAME
            persistent_identity = runner._write_json_no_replace(
                persistent_path, {"scope": "unit persistent cache receipt"}
            )
            persistent = {
                "path": str(persistent_path),
                "file_identity": persistent_identity,
                "receipt_digest": "3" * 64,
            }
            initial = {
                "rank": 0,
                "local_rank": 0,
                "root": runner._private_directory_identity(
                    cache_root, label="unit persistent cache root"
                ),
                "local_tmp_root": runner._private_directory_identity(
                    local_tmp_root, label="unit local tmp root"
                ),
                "local_tmp_parent_identity": layout["parent_identity"],
                "local_tmp_role_directory_identities": layout[
                    "local_tmp_directory_identities"
                ],
                "launcher_local_tmp_fresh_empty_proof": layout[
                    "launcher_local_tmp_empty_proof"
                ],
                "worker_pre_torch_local_tmp_empty_proof": {
                    "root_identity": layout["local_tmp_root_identity"],
                    "directory_identities": layout[
                        "local_tmp_directory_identities"
                    ],
                    "all_rank_directories_empty": True,
                },
                "scheduler_tmpdir_normalization": (
                    _scheduler_normalization_receipt()
                ),
            }
            # Simulate legitimate runtime-created scratch after the worker's
            # strict pre-Torch empty gate, then build a real scanner-shaped
            # nonempty historical baseline for all four ranks.
            world_rows = []
            for rank in range(4):
                rank_path = local_tmp_root / f"rank-{rank}"
                runtime_file = rank_path / f"runtime-init-{rank}.tmp"
                runtime_file.write_bytes(f"runtime-{rank}".encode("ascii"))
                runtime_file.chmod(0o600)
                tree = runner._snapshot_node_local_tmp_tree(
                    rank_path,
                    user_db_path=cache_root / f"rank-{rank}-user-db",
                    label=f"unit post-runtime baseline rank-{rank}",
                )
                world_rows.append(
                    {
                        "rank": rank,
                        "local_rank": rank,
                        "tmp_path": str(rank_path),
                        "tree": tree,
                        "snapshot_sha256": runner._canonical_sha256(tree),
                    }
                )
            baseline_sha256 = runner._canonical_sha256(world_rows)

            def rehash_world_row(row):
                row["tree"]["tree_sha256"] = runner._canonical_sha256(
                    row["tree"]["entries"]
                )
                row["snapshot_sha256"] = runner._canonical_sha256(row["tree"])

            wrong_mode = copy.deepcopy(world_rows)
            wrong_mode[0]["tree"]["entries"][0]["mode"] = 0o777
            rehash_world_row(wrong_mode[0])
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError,
                "regular file identity differs|local file mode differs",
            ):
                runner._validate_local_tmp_world4_snapshot_rows(
                    wrong_mode, initial=initial, label="unit forged mode"
                )

            orphan = copy.deepcopy(world_rows)
            orphan[0]["tree"]["entries"][0]["path"] = "missing/child.tmp"
            rehash_world_row(orphan[0])
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError,
                "nested entry parent differs",
            ):
                runner._validate_local_tmp_world4_snapshot_rows(
                    orphan, initial=initial, label="unit forged parent"
                )

            too_many = copy.deepcopy(world_rows)
            too_many[0]["tree"]["entries"] = [
                copy.deepcopy(world_rows[0]["tree"]["entries"][0])
                for _ in range(4097)
            ]
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError,
                "entry list differs",
            ):
                runner._validate_local_tmp_world4_snapshot_rows(
                    too_many, initial=initial, label="unit forged count"
                )
            pre_torch_rows = [
                {
                    "rank": rank,
                    "local_rank": rank,
                    "initial_rank_directories_empty": True,
                    "launcher_local_tmp_fresh_empty_proof": layout[
                        "launcher_local_tmp_empty_proof"
                    ],
                    "worker_pre_torch_local_tmp_empty_proof": initial[
                        "worker_pre_torch_local_tmp_empty_proof"
                    ],
                }
                for rank in range(4)
            ]
            with mock.patch.dict(
                os.environ,
                {
                    **_scheduler_normalization_environment(),
                    runner.MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV: str(
                        layout["launcher_local_tmp_empty_proof"]["sha256"]
                    ),
                },
                clear=True,
            ), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
            ), mock.patch.object(
                runner,
                "_gather_local_tmp_snapshot",
                return_value=world_rows,
            ):
                tampered_cpu_proof = copy.deepcopy(
                    initial["worker_pre_torch_local_tmp_empty_proof"]
                )
                tampered_cpu_proof["all_rank_directories_empty"] = False
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "three-layer pre-Torch node-local tmp empty proof differs",
                ):
                    runner._seal_local_tmp_receipt_world4(
                        initial=initial,
                        activation_rows=pre_torch_rows,
                        cpu_preflight_local_tmp_empty_proof=tampered_cpu_proof,
                        post_runtime_init_baseline_world4=world_rows,
                        post_runtime_init_baseline_world4_sha256=baseline_sha256,
                        arm_snapshots={"e02": {}},
                        persistent_cache_receipt=persistent,
                        output_dir=output,
                        torch=SimpleNamespace(),
                        dist=FakeDist(),
                        rank=0,
                    )

                tampered_launcher_initial = copy.deepcopy(initial)
                tampered_launcher_initial[
                    "launcher_local_tmp_fresh_empty_proof"
                ]["sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "three-layer pre-Torch node-local tmp empty proof differs",
                ):
                    runner._seal_local_tmp_receipt_world4(
                        initial=tampered_launcher_initial,
                        activation_rows=pre_torch_rows,
                        cpu_preflight_local_tmp_empty_proof=initial[
                            "worker_pre_torch_local_tmp_empty_proof"
                        ],
                        post_runtime_init_baseline_world4=world_rows,
                        post_runtime_init_baseline_world4_sha256=baseline_sha256,
                        arm_snapshots={"e02": {}},
                        persistent_cache_receipt=persistent,
                        output_dir=output,
                        torch=SimpleNamespace(),
                        dist=FakeDist(),
                        rank=0,
                    )

                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "baseline digest differs",
                ):
                    runner._seal_local_tmp_receipt_world4(
                        initial=initial,
                        activation_rows=pre_torch_rows,
                        cpu_preflight_local_tmp_empty_proof=initial[
                            "worker_pre_torch_local_tmp_empty_proof"
                        ],
                        post_runtime_init_baseline_world4=world_rows,
                        post_runtime_init_baseline_world4_sha256="f" * 64,
                        arm_snapshots={"e02": {}},
                        persistent_cache_receipt=persistent,
                        output_dir=output,
                        torch=SimpleNamespace(),
                        dist=FakeDist(),
                        rank=0,
                    )

                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "pre-Torch WORLD4 node-local tmp empty proof differs",
                ):
                    runner._seal_local_tmp_receipt_world4(
                        initial=initial,
                        activation_rows=[
                            {
                                **row,
                                "initial_rank_directories_empty": False,
                            }
                            for row in pre_torch_rows
                        ],
                        cpu_preflight_local_tmp_empty_proof=initial[
                            "worker_pre_torch_local_tmp_empty_proof"
                        ],
                        post_runtime_init_baseline_world4=world_rows,
                        post_runtime_init_baseline_world4_sha256=(
                            baseline_sha256
                        ),
                        arm_snapshots={"e02": {}},
                        persistent_cache_receipt=persistent,
                        output_dir=output,
                        torch=SimpleNamespace(),
                        dist=FakeDist(),
                        rank=0,
                    )
                sealed = runner._seal_local_tmp_receipt_world4(
                    initial=initial,
                    activation_rows=pre_torch_rows,
                    cpu_preflight_local_tmp_empty_proof=initial[
                        "worker_pre_torch_local_tmp_empty_proof"
                    ],
                    post_runtime_init_baseline_world4=world_rows,
                    post_runtime_init_baseline_world4_sha256=baseline_sha256,
                    arm_snapshots={
                        "e02": {
                            "official-v2v-base": {
                                "before_world4": world_rows,
                                "after_world4": world_rows,
                            },
                            "local-source-reference-r2v4-in-manual-G": {
                                "before_world4": world_rows,
                                "after_world4": world_rows,
                            },
                        }
                    },
                    persistent_cache_receipt=persistent,
                    output_dir=output,
                    torch=SimpleNamespace(),
                    dist=FakeDist(),
                    rank=0,
                )
            receipt_path = Path(sealed["path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(sealed["receipt_content"], receipt)
            self.assertTrue(
                sealed["full_receipt_embedded_in_durable_output_receipt"]
            )
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o444)
            self.assertEqual(receipt_path.stat().st_nlink, 1)
            self.assertFalse(receipt["runner_cleanup_performed"])
            self.assertFalse(receipt["durability_guaranteed"])
            self.assertTrue(receipt["node_lifetime_only"])
            self.assertTrue(
                receipt["observed_and_replayed_before_WORLD4_step_exit"]
            )
            self.assertFalse(
                receipt["existence_after_process_or_step_exit_guaranteed"]
            )
            self.assertFalse(receipt["scientific_output_artifact"])
            self.assertTrue(
                receipt["pre_torch_rank_directories_empty_all_ranks"]
            )
            self.assertEqual(
                receipt["pre_torch_empty_activation_world4"], pre_torch_rows
            )
            self.assertTrue(receipt["post_runtime_init_baseline_may_be_nonempty"])
            self.assertTrue(
                receipt[
                    "post_runtime_init_baseline_strictly_scanned_and_quiescent"
                ]
            )
            self.assertEqual(
                receipt["post_runtime_init_baseline_world4"], world_rows
            )
            self.assertEqual(
                receipt["post_runtime_init_baseline_world4_sha256"],
                baseline_sha256,
            )
            self.assertTrue(
                receipt["post_runtime_init_baseline_is_observation_not_allowlist"]
            )
            self.assertTrue(
                receipt[
                    "post_runtime_init_baseline_is_not_immutable_or_monotonic_claim"
                ]
            )
            self.assertTrue(
                receipt[
                    "differences_present_at_observation_boundaries_are_recorded_not_forbidden"
                ]
            )
            self.assertFalse(receipt["continuous_monitoring_claimed"])
            self.assertTrue(
                receipt[
                    "transients_created_and_removed_between_observations_may_be_unrecorded"
                ]
            )
            self.assertTrue(
                receipt["launcher_cpu_preflight_worker_empty_proofs_exact"]
            )
            self.assertEqual(
                receipt["launcher_fresh_empty_proof"],
                layout["launcher_local_tmp_empty_proof"],
            )
            self.assertEqual(
                receipt["cpu_preflight_empty_proof"],
                initial["worker_pre_torch_local_tmp_empty_proof"],
            )
            self.assertEqual(
                receipt["worker_pre_torch_empty_proof"],
                initial["worker_pre_torch_local_tmp_empty_proof"],
            )
            self.assertEqual(receipt["parent_identity"], layout["parent_identity"])
            self.assertEqual(
                receipt["persistent_cache_receipt"]["file_identity"],
                persistent_identity,
            )
            bound_sha = sealed["file_identity"]["sha256"]
            receipt_path.chmod(0o644)
            receipt_path.write_bytes(b"tampered node-local receipt")
            receipt_path.chmod(0o444)
            self.assertNotEqual(
                runner._owned_file_identity(
                    receipt_path, label="tampered node-local receipt"
                )["sha256"],
                bound_sha,
            )

    def test_node_local_tmp_rank_failure_is_collective_safe(self) -> None:
        runner = self.runner

        class FakeCuda:
            @staticmethod
            def synchronize() -> None:
                return None

        class FakeDist:
            calls = 0

            def all_gather_object(self, rows, value) -> None:
                self.calls += 1
                rows[:] = [
                    value,
                    {
                        "ok": False,
                        "rank": 1,
                        "error_type": "OSError",
                        "error": "injected local tmp scan failure",
                    },
                    value,
                    value,
                ]

        dist = FakeDist()
        with mock.patch.object(
            runner, "_stable_local_tmp_snapshot", return_value={"rank": 0}
        ):
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError,
                "rank-local tmp certification failed",
            ):
                runner._gather_local_tmp_snapshot(
                    {"rank": 0},
                    torch=SimpleNamespace(cuda=FakeCuda()),
                    dist=dist,
                    label="unit collective-safe local tmp",
                )
        self.assertEqual(dist.calls, 1)

    def test_miopen_rank_local_snapshot_failure_is_collective_safe(self) -> None:
        runner = self.runner

        class FakeCuda:
            @staticmethod
            def synchronize() -> None:
                return None

        class FakeDist:
            calls = 0

            def all_gather_object(self, rows, value) -> None:
                self.calls += 1
                rows[:] = [
                    value,
                    value,
                    {
                        "ok": False,
                        "rank": 2,
                        "error_type": "OSError",
                        "error": "injected rank-local SQLite scan failure",
                    },
                    value,
                ]

        dist = FakeDist()
        with mock.patch.object(
            runner,
            "_stable_local_miopen_cache_snapshot",
            return_value={"rank": 0},
        ):
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError,
                "rank-local cache certification failed",
            ):
                runner._gather_miopen_cache_snapshot(
                    {"rank": 0},
                    torch=SimpleNamespace(cuda=FakeCuda()),
                    dist=dist,
                    label="unit collective-safe cache",
                )
        self.assertEqual(dist.calls, 1)

    def test_loaded_miopen_proc_maps_origin_is_exact(self) -> None:
        runner = self.runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            library = root / "libMIOpen.so"
            library.write_bytes(b"unit library")
            library.chmod(0o755)
            maps = root / "maps"
            maps.write_text(
                f"7f00-7f01 r-xp 00000000 00:00 1 {library}\n",
                encoding="utf-8",
            )
            with mock.patch.object(runner, "MIOPEN_LIBRARY_PATH", library), mock.patch.object(
                runner, "MIOPEN_LIBRARY_SHA256", _sha(library)
            ), mock.patch.object(
                runner, "MIOPEN_LIBRARY_SIZE", library.stat().st_size
            ):
                receipt = runner._certify_loaded_miopen_library(maps_path=maps)
                self.assertTrue(receipt["loaded_from_proc_maps"])
                maps.write_text(
                    "7f00-7f01 r-xp 00000000 00:00 1 /opt/rocm/lib/libMIOpen.so\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError, "origin differs"
                ):
                    runner._certify_loaded_miopen_library(maps_path=maps)

    def test_receipt_direct_exclusive_writer_and_short_eintr(self) -> None:
        runner = self.runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            path = root / "receipt.json"
            real_write = os.write
            calls = 0

            def short_eintr(descriptor: int, payload: bytes) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(descriptor, payload[:7])
                if calls == 2:
                    raise InterruptedError("unit EINTR")
                return real_write(descriptor, payload)

            with mock.patch.object(runner.os, "write", side_effect=short_eintr):
                identity = runner._write_json_no_replace(
                    path, {"diagnostic": True, "padding": "x" * 128}
                )
            before = path.read_bytes()
            before_inode = path.stat().st_ino
            self.assertGreaterEqual(calls, 3)
            self.assertEqual(identity["mode"], 0o444)
            self.assertEqual(identity["nlink"], 1)
            with self.assertRaises(runner.NativeActivationV2RunnerError):
                runner._write_json_no_replace(path, {"overwrite": True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_ino, before_inode)

    def test_staged_publish_streams_without_link_or_timestamp_contract(self) -> None:
        runner = self.runner
        source = Path(runner.__file__).read_text(encoding="utf-8")
        publish_source = source[
            source.index("def _publish_staged_file_no_replace"):
            source.index("def _write_json_no_replace")
        ]
        self.assertIn("os.O_EXCL", publish_source)
        self.assertNotIn("os.link(", publish_source)
        self.assertNotIn("unlink(", publish_source)
        self.assertNotIn("st_mtime", publish_source)
        self.assertNotIn("st_ctime", publish_source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            staging = root / "staging.bin"
            destination = root / "published.bin"
            payload = bytes(range(256)) * 8193
            staging.write_bytes(payload)
            identity = runner._publish_staged_file_no_replace(
                staging_path=staging,
                destination=destination,
                label="unit staged artifact",
            )
            self.assertEqual(staging.read_bytes(), payload)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(identity["mode"], 0o444)
            self.assertEqual(identity["nlink"], 1)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o444)
            with self.assertRaises(runner.NativeActivationV2RunnerError):
                runner._publish_staged_file_no_replace(
                    staging_path=staging,
                    destination=destination,
                    label="unit staged artifact replay",
                )

    def test_staged_partial_publish_is_preserved_fail_closed(self) -> None:
        runner = self.runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            staging = root / "staging.bin"
            destination = root / "published.bin"
            staging.write_bytes(b"z" * (2 * 1024 * 1024 + 17))
            real_write = os.write
            calls = 0

            def fail_after_prefix(descriptor: int, payload: bytes) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(descriptor, payload[:31])
                raise OSError("injected output failure")

            with mock.patch.object(runner.os, "write", side_effect=fail_after_prefix):
                with self.assertRaises(OSError):
                    runner._publish_staged_file_no_replace(
                        staging_path=staging,
                        destination=destination,
                        label="unit partial artifact",
                    )
            self.assertTrue(destination.is_file())
            partial = destination.read_bytes()
            inode = destination.stat().st_ino
            with self.assertRaises(runner.NativeActivationV2RunnerError):
                runner._publish_staged_file_no_replace(
                    staging_path=staging,
                    destination=destination,
                    label="unit partial artifact retry",
                )
            self.assertEqual(destination.read_bytes(), partial)
            self.assertEqual(destination.stat().st_ino, inode)

    def test_serialized_host_load_lock_is_required_frozen_and_live(self) -> None:
        runner = self.runner
        import preflight_oracle_regeneration_native_activation_v2_r2 as preflight

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            lock = root / "renderer-load.lock"
            lock.write_bytes(b"")
            lock.chmod(0o444)
            environment = {
                "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                "NATIVE_V_AXIS_LOAD_LOCK": str(lock),
                "SLURM_JOB_ID": "141620",
                "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                preflight.socket,
                "gethostname",
                return_value="auh7-1b-gpu-226",
            ):
                receipt = runner._certify_serialized_host_load_lock()
                self.assertEqual(receipt["path"], str(lock))
                self.assertEqual(receipt["size"], 0)
                self.assertEqual(receipt["mode"], 0o444)
                self.assertEqual(receipt["nlink"], 1)
                preflight_receipt = preflight._validate_host_load_environment()
                self.assertEqual(preflight_receipt["path"], str(lock))
                self.assertEqual(preflight_receipt["sha256"], hashlib.sha256(b"").hexdigest())
                lock.chmod(0o644)
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError, "empty/frozen/one-link"
                ):
                    runner._certify_serialized_host_load_lock()
            with mock.patch.dict(
                os.environ,
                {
                    "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "0",
                    "NATIVE_V_AXIS_LOAD_LOCK": str(lock),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError, "not enabled"
                ):
                    runner._certify_serialized_host_load_lock()

    def _complete_output_receipt(self, output_dir: Path) -> dict:
        runner = self.runner
        cases = {}
        for case_id in runner.CASE_ORDER:
            arms = {}
            for arm in runner.ARM_ORDER_BY_CASE[case_id]:
                roles = {}
                filenames = {
                    "official_gaussian": (
                        f"{case_id}.{arm}.official-gaussian.safetensors"
                    ),
                    "clean_latent": f"{case_id}.{arm}.clean-latent.safetensors",
                    "video": f"{case_id}.{arm}.mp4",
                }
                for role, filename in filenames.items():
                    path = output_dir / filename
                    path.write_bytes(f"{case_id}/{arm}/{role}".encode("utf-8"))
                    path.chmod(0o444)
                    identity = runner._owned_file_identity(
                        path, label=f"unit output {filename}"
                    )
                    file_identity = {
                        key: identity[key]
                        for key in ("sha256", "size", "mode", "nlink")
                    }
                    roles[role] = {
                        "path": str(path),
                        "sha256": file_identity["sha256"],
                        "file_identity": file_identity,
                    }
                arms[arm] = roles
            if case_id == "e03":
                cases[case_id] = {
                    "decision": "ABSTAIN_KEEP_BASE",
                    "executed": False,
                    "arms": [],
                    "condition_receipts": None,
                    "arm_receipts": {},
                    "runtime_traces": {},
                    "outputs": {},
                    "kept_frozen_base": {
                        "path": str(runner.E03_FROZEN_BASE_PATH),
                        "sha256": runner.E03_FROZEN_BASE_SHA256,
                    },
                    "selection": "ABSTAIN_KEEP_BASE",
                }
            else:
                cases[case_id] = {"outputs": arms}
        receipt = {"schema_version": "unit", "cases": cases}
        receipt["receipt_digest"] = runner._canonical_sha256(receipt)
        return receipt

    def test_final_receipt_rebinds_every_file_and_rejects_post_publish_swap(self) -> None:
        runner = self.runner
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary).resolve(strict=True) / "release"
            output_dir.mkdir(mode=0o700)
            receipt = self._complete_output_receipt(output_dir)
            real_writer = runner._write_json_no_replace

            def write_then_swap(path: Path, value: dict):
                identity = real_writer(path, value)
                victim = output_dir / (
                    "e02.official-v2v-base.official-gaussian.safetensors"
                )
                replacement = output_dir / "replacement.tmp"
                replacement.write_bytes(b"hostile replacement")
                replacement.chmod(0o444)
                os.replace(replacement, victim)
                return identity

            with mock.patch.object(
                runner, "_write_json_no_replace", side_effect=write_then_swap
            ):
                with self.assertRaisesRegex(
                    runner.NativeActivationV2RunnerError,
                    "differs from receipt",
                ):
                    runner._freeze_output_release(
                        output_dir=output_dir, receipt=receipt
                    )
            tampered = copy.deepcopy(receipt)
            tampered["receipt_digest"] = "0" * 64
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError, "receipt digest"
            ):
                runner._freeze_output_release(
                    output_dir=output_dir, receipt=tampered
                )

    def test_world4_sp4_topology_is_exact_and_duplicate_device_fails(self) -> None:
        runner = self.runner
        group = object()

        class FakeCuda:
            @staticmethod
            def device_count() -> int:
                return 4

            @staticmethod
            def current_device() -> int:
                return 0

        class FakeDist:
            def __init__(self, *, duplicate: bool = False) -> None:
                self.duplicate = duplicate

            @staticmethod
            def get_world_size(requested_group=None) -> int:
                return 4

            @staticmethod
            def get_rank(requested_group=None) -> int:
                return 0

            @staticmethod
            def get_backend(requested_group=None) -> str:
                return "nccl"

            def all_gather_object(self, rows, value, group=None) -> None:
                if group is not None:
                    rows[:] = [0, 1, 2, 3]
                    return
                rows[:] = [
                    {
                        "rank": rank,
                        "local_rank": rank,
                        "current_device": rank,
                        "device_count": 4,
                        "hostname": "auh7-1b-gpu-226",
                        "sp_rank": rank,
                    }
                    for rank in range(4)
                ]
                if self.duplicate:
                    rows[3]["local_rank"] = 2

        distributed = SimpleNamespace(rank=0, local_rank=0)
        parallel_state = SimpleNamespace(
            ulysses_group=group,
            ulysses_rank=0,
            ulysses_enabled=True,
            ulysses_size=4,
        )
        fake_torch = SimpleNamespace(cuda=FakeCuda())
        environment = {
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            "LOCAL_WORLD_SIZE": "4",
            "SLURM_JOB_ID": "141620",
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            runner.socket, "gethostname", return_value="auh7-1b-gpu-226"
        ):
            receipt = runner._certify_world4_sp4(
                distributed=distributed,
                torch=fake_torch,
                dist=FakeDist(),
                parallel_state=parallel_state,
            )
            self.assertEqual(
                receipt["sp_ordered_global_ranks"], [0, 1, 2, 3]
            )
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError, "topology differs"
            ):
                runner._certify_world4_sp4(
                    distributed=distributed,
                    torch=fake_torch,
                    dist=FakeDist(duplicate=True),
                    parallel_state=parallel_state,
                )


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class ActivationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def _tensor_identity(self, value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "content_sha256": activation.safe_core.tensor_content_sha256_v1(value),
        }

    def _make_gate(
        self,
        root: Path,
        *,
        case_row: dict,
        annotator: str,
        reviewer: str,
        geometry: tuple[int, int, int, int, int],
        contact_start: int = 4,
    ) -> tuple[Path, str, Path, str, str]:
        review_path = root / "e02-gate-review.json"
        delete = [[] for _ in range(21)]
        create = [[] for _ in range(21)]
        contact = [[] for _ in range(21)]
        for phase in range(1, 21):
            delete[phase] = [[0, 1]]
            if phase >= contact_start:
                contact[phase] = [[1, 4]]
            if phase >= 5:
                create[phase] = [[5, 1]]
        mask_payload = activation._manual_gate_payload_v2(
            geometry=geometry,
            delete_rle=delete,
            create_rle=create,
            contact_rle=contact,
        )
        mask_sha = hashlib.sha256(
            activation.safe_core.canonical_json_bytes_v1(mask_payload)
        ).hexdigest()
        leaf_payload = {
            "schema_version": activation.MANUAL_GATE_LEAF_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": case_row["source_video"]["sha256"],
            "anchor_sha256": case_row["self_generated_anchor"]["sha256"],
            "action_caption_sha256": case_row["action_caption_sha256"],
            "structured_action_program_sha256": case_row[
                "structured_action_program_sha256"
            ],
            "mask_sha256": mask_sha,
            "annotator": annotator,
            "reviewer": reviewer,
        }
        leaf_sha = hashlib.sha256(
            b"\x00" + activation.safe_core.canonical_json_bytes_v1(leaf_payload)
        ).hexdigest()
        gate = {
            "schema_version": activation.MANUAL_GATE_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": case_row["source_video"]["sha256"],
            "anchor_sha256": case_row["self_generated_anchor"]["sha256"],
            "action_caption_sha256": case_row["action_caption_sha256"],
            "structured_action_program_sha256": case_row[
                "structured_action_program_sha256"
            ],
            "latent_geometry": list(geometry),
            "flattening": "per_phase_row_major_yx",
            "dtype": "bool",
            "hard_support": True,
            "phase_zero_empty": True,
            "delete_rle": delete,
            "create_rle": create,
            "contact_rle": contact,
            "typed_semantics": {
                "delete_D": "obsolete_source_state_occupancy_to_delete",
                "create_C": "new_actor_object_state_occupancy_to_create",
                "contact_ownership_K": "contact_and_ownership_transition_permission_corridor",
                "execution_support_G": "exact_boolean_union_D_or_C_or_K",
                "coordinate_system": "source_latent_phase_y_x",
                "expected_nonempty_phase_windows": {
                    "delete_D": [1, 20],
                    "create_C": [5, 20],
                    "contact_ownership_K": [4, 20],
                    "execution_support_G": [1, 20],
                },
            },
            "mask_sha256": mask_sha,
            "annotation_authority": {
                "tree_shape": activation.safe_core.ANNOTATION_TREE_SHAPE,
                "ledger_root_sha256": leaf_sha,
                "leaf_sha256": leaf_sha,
                "leaf_index": 0,
                "tree_size": 1,
                "inclusion_proof": [],
            },
            "authority": {
                "role": "source_only_model_proposal_diagnostic_intervention_only",
                "training_target_authorized": False,
                "action_representation_claimed": False,
                "forbidden_inputs_absent": {
                    "failed_active_video_or_latent": True,
                    "raw_anchor_source_pixel_or_latent_difference": True,
                    "predicted_soft_gate": True,
                    "target_video_or_latent": True,
                    "self_generated_anchor_tensor": True,
                },
            },
            "qualification": {
                "status": "independent_model_reviewed_diagnostic_exact_gate",
                "annotator": annotator,
                "reviewer": reviewer,
                "author_kind": "AI_AGENT",
                "reviewer_kind": "AI_AGENT",
                "review_receipt_path": str(review_path),
            },
        }
        gate_path = root / "e02-gate.json"
        _write(gate_path, gate, frozen=False)
        gate_sha = _sha(gate_path)
        review = {
            "schema_version": activation.MANUAL_GATE_REVIEW_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": case_row["source_video"]["sha256"],
            "anchor_sha256": case_row["self_generated_anchor"]["sha256"],
            "action_caption_sha256": case_row["action_caption_sha256"],
            "structured_action_program_sha256": case_row[
                "structured_action_program_sha256"
            ],
            "gate_manifest_sha256": gate_sha,
            "mask_sha256": mask_sha,
            "annotation_authority_root_sha256": leaf_sha,
            "annotation_authority_leaf_sha256": leaf_sha,
            "annotator": annotator,
            "reviewer": reviewer,
            "author_kind": "AI_AGENT",
            "reviewer_kind": "AI_AGENT",
            "source_only_model_proposal": True,
            "independent_model_review": True,
            "accepted": True,
            "phase_zero_source_authority_checked": True,
            "source_coordinate_authoring_checked": True,
            "delete_create_contact_semantics_checked": True,
            "D_C_disjoint_checked": True,
            "K_preserved_as_independent_channel_checked": True,
            "G_exact_union_D_C_K_checked": True,
            "channel_active_windows_checked": True,
            "no_large_rectangle_shortcut_checked": True,
            "single_actor_object_component_checked": True,
            "duplicate_actor_or_object_rejected": True,
            "terminal_hold_semantics_checked": True,
            "anchor_terminal_disappearance_observed": True,
            "anchor_strict_target_pass": False,
            "anchor_used_only_as_review_context": True,
            "failed_active_used_to_author_mask": False,
            "anchor_difference_used_to_author_mask": False,
            "predicted_soft_gate_used_to_author_mask": False,
            "target_video_or_latent_used_to_author_mask": False,
            "self_generated_anchor_tensor_used_to_author_mask": False,
        }
        _write(review_path, review)
        gate_path.chmod(0o444)
        return gate_path, gate_sha, review_path, _sha(review_path), leaf_sha

    def _make_reference_receipt(
        self,
        root: Path,
        *,
        case_id: str,
        source_iid: str,
        source_sha: str,
        source,
        references,
        bucket_hw: tuple[int, int],
    ) -> tuple[Path, str]:
        dependency = {}
        for name in (
            "decode",
            "source-prepare",
            "checkpoint-manifest",
            "config",
            "vae-code",
            "autoencoder-class-module",
            "python-executable",
            "vae-materializer",
        ):
            path, digest = _file(
                root / f"{case_id}-vae-{name}.pin", name.encode(), frozen=True
            )
            dependency[name] = (path, digest)
        raw_sha = tuple(hashlib.sha256(f"raw-{index}".encode()).hexdigest() for index in range(4))
        pre_sha = tuple(hashlib.sha256(f"pre-{index}".encode()).hexdigest() for index in range(4))
        source_sha_tensor = activation.safe_core.tensor_content_sha256_v1(source)
        ref_sha = tuple(
            activation.safe_core.tensor_content_sha256_v1(value) for value in references
        )
        rows = [
            {
                "frame_index": frame_index,
                "raw_rgb_sha256": raw_sha[position],
                "preprocessed_rgb_sha256": pre_sha[position],
                "shape": list(references[position].shape),
                "dtype": str(references[position].dtype),
                "content_sha256": ref_sha[position],
                "independently_vae_encoded": True,
            }
            for position, frame_index in enumerate(activation.REFERENCE_RGB_INDICES)
        ]
        payload = {
            "schema_version": activation.REFERENCE_RECEIPT_SCHEMA_VERSION,
            "case_id": case_id,
            "source_iid": source_iid,
            "source_video_sha256": source_sha,
            "source_frame_count": 81,
            "source_fps_numerator": 25,
            "source_fps_denominator": 1,
            "source_input_frame_geometry": [16, 16, 3],
            "source_bucket_hw": list(bucket_hw),
            "reference_rgb_indices": list(activation.REFERENCE_RGB_INDICES),
            "reference_raw_rgb_sha256": list(raw_sha),
            "full_preprocessed_source_identity": {
                "shape": [1, 3, 81, *bucket_hw],
                "dtype": "torch.float32",
                "content_sha256": "a" * 64,
            },
            "reference_preprocessed_rgb_sha256": list(pre_sha),
            "preprocess_contract": {
                "frame_decode_backend": "decord_cpu0_num_threads1_private_source_snapshot",
                "frame_decode_code_path": str(dependency["decode"][0]),
                "frame_decode_code_sha256": dependency["decode"][1],
                "source_prepare_code_path": str(dependency["source-prepare"][0]),
                "source_prepare_code_sha256": dependency["source-prepare"][1],
                "rgb_dtype": "uint8",
                "rgb_channel_order": "RGB",
                "resize_policy": "torchvision_bicubic_antialias_true_source_aspect_bucket",
                "normalization": "uint8_div255_mul2_sub1_float32",
            },
            "vae_contract": {
                "checkpoint_content_manifest_path": str(
                    dependency["checkpoint-manifest"][0]
                ),
                "checkpoint_content_manifest_sha256": dependency[
                    "checkpoint-manifest"
                ][1],
                "checkpoint_content_identity_sha256": "c" * 64,
                "config_path": str(dependency["config"][0]),
                "config_sha256": dependency["config"][1],
                "vae_code_path": str(dependency["vae-code"][0]),
                "vae_code_sha256": dependency["vae-code"][1],
                "autoencoder_class_module_path": str(
                    dependency["autoencoder-class-module"][0]
                ),
                "autoencoder_class_module_sha256": dependency[
                    "autoencoder-class-module"
                ][1],
                "diffusers_version": "unit-diffusers",
                "torch_version": str(self.torch.__version__),
                "python_executable_path": str(dependency["python-executable"][0]),
                "python_executable_sha256": dependency["python-executable"][1],
                "python_version": sys.version,
                "rocm_version": "unit-no-rocm",
                "encode_function": "bernini.pipeline._vae_encode",
                "encode_dtype": "torch.float32",
                "latent_coordinate": "official_bernini_vae_encode_output",
            },
            "full_source_latent_identity": self._tensor_identity(source),
            "reference_latent_identities": rows,
            "materializer_code_path": str(dependency["vae-materializer"][0]),
            "materializer_code_sha256": dependency["vae-materializer"][1],
            "rank_world_receipt": {
                "world_size": 4,
                "sequence_parallel_size": 4,
                "rank0_only_vae_encode": True,
                "all_rank_vae_load_roles": [
                    {"rank": rank, "vae_loaded": rank == 0}
                    for rank in range(4)
                ],
                "broadcast_exact": True,
                "all_rank_full_source_latent_sha256": [source_sha_tensor] * 4,
                "all_rank_reference_latent_sha256": [list(ref_sha) for _ in range(4)],
            },
            "references_encoded_as_four_independent_rgb_frames": True,
            "references_not_sliced_from_full_source_latent": True,
            "source_reference_storage_alias_rejected": True,
            "reference_content_duplicates_rejected": True,
            "target_video_or_latent_used": False,
            "self_generated_anchor_tensor_used": False,
            "materialization_checks_passed": True,
        }
        path = _write(root / f"{case_id}-vae-receipt.json", payload)
        return path, _sha(path)

    def _make_prompt_receipt(
        self,
        root: Path,
        *,
        case_id: str,
        source_iid: str,
        caption: str,
        prompts,
    ) -> tuple[Path, str]:
        dependency = {}
        for name in (
            "tokenizer-config",
            "tokenizer-code",
            "checkpoint-manifest",
            "text-config",
            "renderer-code",
            "prompt-builder-code",
            "native-prompt-code",
            "prompt-cleaner-code",
            "auto-tokenizer-module",
            "resolved-tokenizer-class-module",
            "text-encoder-class-module",
            "python-executable",
            "prompt-materializer",
        ):
            path, digest = _file(
                root / f"{case_id}-prompt-{name}.pin", name.encode(), frozen=True
            )
            dependency[name] = (path, digest)
        modes = ("low-vr2v", "high-r2v4", "renderer-negative")
        keys = ("low_action", "high_action", "negative")
        role_rows = {}
        digests = []
        for key, mode, tensor in zip(keys, modes, prompts):
            rendered = f"{mode}:{caption}"
            identity = self._tensor_identity(tensor)
            digests.append(identity["content_sha256"])
            role_rows[key] = {
                "mode": mode,
                "rendered_text": rendered,
                "rendered_text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(f"ids-{mode}".encode()).hexdigest(),
                "attention_mask_sha256": hashlib.sha256(f"mask-{mode}".encode()).hexdigest(),
                "embedding_identity": identity,
            }
        payload = {
            "schema_version": activation.PROMPT_RECEIPT_SCHEMA_VERSION,
            "case_id": case_id,
            "source_iid": source_iid,
            "action_caption": caption,
            "action_caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
            "prompt_contract": {
                "tokenizer_config_path": str(dependency["tokenizer-config"][0]),
                "tokenizer_config_sha256": dependency["tokenizer-config"][1],
                "tokenizer_code_path": str(dependency["tokenizer-code"][0]),
                "tokenizer_code_sha256": dependency["tokenizer-code"][1],
                "checkpoint_content_manifest_path": str(
                    dependency["checkpoint-manifest"][0]
                ),
                "checkpoint_content_manifest_sha256": dependency[
                    "checkpoint-manifest"
                ][1],
                "checkpoint_content_identity_sha256": "c" * 64,
                "text_encoder_config_path": str(dependency["text-config"][0]),
                "text_encoder_config_sha256": dependency["text-config"][1],
                "renderer_code_path": str(dependency["renderer-code"][0]),
                "renderer_code_sha256": dependency["renderer-code"][1],
                "prompt_builder_code_path": str(
                    dependency["prompt-builder-code"][0]
                ),
                "prompt_builder_code_sha256": dependency["prompt-builder-code"][1],
                "native_prompt_code_path": str(dependency["native-prompt-code"][0]),
                "native_prompt_code_sha256": dependency["native-prompt-code"][1],
                "prompt_cleaner_code_path": str(
                    dependency["prompt-cleaner-code"][0]
                ),
                "prompt_cleaner_code_sha256": dependency["prompt-cleaner-code"][1],
                "auto_tokenizer_module_path": str(
                    dependency["auto-tokenizer-module"][0]
                ),
                "auto_tokenizer_module_sha256": dependency[
                    "auto-tokenizer-module"
                ][1],
                "resolved_tokenizer_class_module_path": str(
                    dependency["resolved-tokenizer-class-module"][0]
                ),
                "resolved_tokenizer_class_module_sha256": dependency[
                    "resolved-tokenizer-class-module"
                ][1],
                "text_encoder_class_module_path": str(
                    dependency["text-encoder-class-module"][0]
                ),
                "text_encoder_class_module_sha256": dependency[
                    "text-encoder-class-module"
                ][1],
                "transformers_version": "unit-transformers",
                "torch_version": str(self.torch.__version__),
                "python_executable_path": str(dependency["python-executable"][0]),
                "python_executable_sha256": dependency["python-executable"][1],
                "python_version": sys.version,
                "rocm_version": "unit-no-rocm",
                "tokenizer_function": "infer_lora._tokenize_training_prompt+_tokenize_renderer_negative",
                "text_encoder_function": "bernini.models.renderer.BerniniRendererModel.encode_prompt",
                "max_length": 512,
                "embedding_dtype": "torch.bfloat16",
            },
            **role_rows,
            "materializer_code_path": str(dependency["prompt-materializer"][0]),
            "materializer_code_sha256": dependency["prompt-materializer"][1],
            "rank_world_receipt": {
                "world_size": 4,
                "sequence_parallel_size": 4,
                "rank0_only_text_encode": True,
                "all_rank_text_encoder_load_roles": [
                    {
                        "rank": rank,
                        "real_t5_loaded": rank == 0,
                        "bypassed_t5_load": rank != 0,
                        "bypass_call_count": 0 if rank == 0 else 1,
                        "placeholder_retained": rank != 0,
                    }
                    for rank in range(4)
                ],
                "broadcast_exact": True,
                "all_rank_low_action_sha256": [digests[0]] * 4,
                "all_rank_high_action_sha256": [digests[1]] * 4,
                "all_rank_negative_sha256": [digests[2]] * 4,
            },
            "rank0_only_text_encoder_load": True,
            "nonzero_ranks_never_deserialized_text_encoder": True,
            "self_generated_anchor_tensor_used": False,
            "target_video_or_latent_used": False,
            "materialization_checks_passed": True,
        }
        path = _write(root / f"{case_id}-prompt-receipt.json", payload)
        return path, _sha(path)

    def _make_material_graph_receipts(
        self,
        root: Path,
        *,
        source_iid: str,
        source_sha: str,
        action_caption_sha: str,
        reference_path: Path,
        reference_sha: str,
        prompt_path: Path,
        prompt_sha: str,
    ) -> dict:
        vae_run = {
            "schema_version": (
                "bernini-oracle-regeneration-activation-v2-vae-authoring-run-r2"
            ),
            "receipt_sha256": reference_sha,
            "diagnostic_authoring_material_only": True,
            "full_model_or_sampler_loaded": False,
            "scheduler_loaded": False,
            "transformer_loaded": False,
            "training": False,
            "optimizer": False,
        }
        vae_run_path = _write(root / "e02-vae-run-receipt.json", vae_run)
        vae_run_sha = _sha(vae_run_path)
        prompt_run = {
            "schema_version": (
                "bernini-oracle-regeneration-activation-v2-prompt-authoring-run-r2"
            ),
            "prompt_receipt_sha256": prompt_sha,
            "diagnostic_authoring_material_only": True,
            "sampler_or_scheduler_called": False,
            "denoising_transformer_moved_to_gpu": False,
            "training": False,
            "optimizer": False,
        }
        prompt_run_path = _write(root / "e02-prompt-run-receipt.json", prompt_run)
        prompt_run_sha = _sha(prompt_run_path)

        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))

        def reviewed_row(path: Path, digest: str, schema: str) -> dict:
            status = path.stat()
            return {
                "remote_path": str(path),
                "sha256": digest,
                "bytes": status.st_size,
                "mode": "0444",
                "nlink": 1,
                "schema_version": schema,
            }

        review = {
            "schema_version": (
                "bernini-oracle-regeneration-e02-ai-agent-diagnostic-material-review-v1"
            ),
            "case_id": "e02",
            "decision": "ACCEPT_E02_DIAGNOSTIC_ONLY",
            "authority": False,
            "accepted_case_ids": ["e02"],
            "marker": (
                "AI_AGENT_REVIEW_NONHUMAN_NONFORMAL_NOT_TRAINING_AUTHORITY_"
                "DIAGNOSTIC_ONLY"
            ),
            "authority_flags": {
                "diagnostic_material_accepted": True,
                "executable_authority": False,
                "formal_authority": False,
                "human_annotation": False,
                "human_review": False,
                "optimizer_authorized": False,
                "runnable": False,
                "training_authorized": False,
                "automatic_model_replacement_authorized": False,
            },
            "audit_results": {
                "materializer_exact_receipt_closure_passed": True,
                "checkpoint_23_file_sha_member_identity_closure_passed": True,
                "world4_cross_rank_identity_passed": True,
                "source_only_materialization": True,
                "optimizer_used": False,
                "sampler_or_scheduler_called": False,
                "self_generated_anchor_tensor_used": False,
                "target_video_or_latent_used": False,
                "training_used": False,
            },
            "source_binding": {
                "source_iid": source_iid,
                "source_video_sha256": source_sha,
                "action_caption_sha256": action_caption_sha,
            },
            "checkpoint_binding": {
                "checkpoint_content_identity_sha256": reference["vae_contract"][
                    "checkpoint_content_identity_sha256"
                ],
                "checkpoint_content_manifest_sha256": reference["vae_contract"][
                    "checkpoint_content_manifest_sha256"
                ],
                "verified_file_count": 23,
                "every_file_sha256_independently_reverified": True,
            },
            "materializer_receipts": {
                "e02_vae_reference_receipt": reviewed_row(
                    reference_path,
                    reference_sha,
                    activation.REFERENCE_RECEIPT_SCHEMA_VERSION,
                ),
                "e02_vae_run_receipt": reviewed_row(
                    vae_run_path,
                    vae_run_sha,
                    vae_run["schema_version"],
                ),
                "e02_prompt_receipt": reviewed_row(
                    prompt_path,
                    prompt_sha,
                    activation.PROMPT_RECEIPT_SCHEMA_VERSION,
                ),
                "e02_prompt_run_receipt": reviewed_row(
                    prompt_run_path,
                    prompt_run_sha,
                    prompt_run["schema_version"],
                ),
            },
        }
        self.assertEqual(
            review["checkpoint_binding"]["checkpoint_content_identity_sha256"],
            prompt["prompt_contract"]["checkpoint_content_identity_sha256"],
        )
        self.assertEqual(
            review["checkpoint_binding"]["checkpoint_content_manifest_sha256"],
            prompt["prompt_contract"]["checkpoint_content_manifest_sha256"],
        )
        review_path = _write(root / "e02-material-review.json", review)
        return {
            "vae_run_path": vae_run_path,
            "vae_run_sha": vae_run_sha,
            "prompt_run_path": prompt_run_path,
            "prompt_run_sha": prompt_run_sha,
            "review_path": review_path,
            "review_sha": _sha(review_path),
        }

    @contextmanager
    def _authority_fixture(
        self,
        *,
        cryptographic_signature_claimed: bool = False,
        contact_start: int = 4,
    ):
        torch = self.torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            geometry = (1, 16, 21, 6, 6)
            gate_geometry = (1, 1, 21, 6, 6)
            reference_geometry = (1, 16, 1, 6, 6)
            source = torch.zeros(geometry, dtype=torch.float32).contiguous()
            references = tuple(
                torch.full(reference_geometry, float(index + 1), dtype=torch.float32)
                for index in range(4)
            )
            prompts = tuple(
                torch.full(
                    (1, 512, 4096), float(index + 1), dtype=torch.bfloat16
                ).contiguous()
                for index in range(3)
            )
            media = {}
            for case_id in activation.ALLOWED_CASES:
                source_path, source_sha = _file(
                    root / f"{case_id}-source.mp4", f"{case_id}-source".encode()
                )
                anchor_path, anchor_sha = _file(
                    root / f"{case_id}-anchor.mp4", f"{case_id}-anchor".encode()
                )
                media[case_id] = (source_path, source_sha, anchor_path, anchor_sha)
            kept_base_path, kept_base_sha = _file(
                root / "e03-kept-frozen-base.mp4",
                b"unit-e03-kept-frozen-base",
                frozen=True,
            )
            captions = {
                "e02": (
                    "The same pale bare hand firmly grips the same red mushroom at its "
                    "lower stem, twists and pulls it free from the same soil, lifts the "
                    "same intact mushroom above the newly empty root hole, and holds it "
                    "there. Exactly one hand and one mushroom remain visible; do not "
                    "duplicate, split, or fuse either one."
                ),
                "e03": (
                    "The same farm worker moves the same harvested root cluster over the "
                    "same woven basket, lowers it past the rim, opens the same hand, and "
                    "releases it. The cluster falls, bounces slightly, and settles inside "
                    "while the now-empty hand withdraws. Do not duplicate the hand or "
                    "cluster."
                ),
            }
            programs = {
                "e02": {
                    "entities": ["same_pale_bare_hand", "same_red_mushroom"],
                    "states": ["rooted", "same_hand_contact", "detached", "held"],
                },
                "e03": {
                    "entities": ["same_farm_worker_hand", "same_harvested_root_cluster"],
                    "states": ["held", "released", "settled", "empty_hand"],
                },
            }
            cases = []
            expected_bindings = {}
            material_review_sha = None
            for case_id in activation.ALLOWED_CASES:
                source_path, source_sha, anchor_path, anchor_sha = media[case_id]
                caption = captions[case_id]
                caption_sha = hashlib.sha256(caption.encode()).hexdigest()
                program_sha = activation._canonical_object_sha256(programs[case_id])
                source_iid = f"unit-{case_id}-source"
                expected_bindings[case_id] = {
                    "source_iid": source_iid,
                    "source_sha256": source_sha,
                    "anchor_sha256": anchor_sha,
                    "action_caption_sha256": caption_sha,
                    "structured_action_program_sha256": program_sha,
                }
                reference_path = reference_sha = None
                prompt_path = prompt_sha = None
                material_graph = None
                if case_id == "e02":
                    reference_path, reference_sha = self._make_reference_receipt(
                        root,
                        case_id=case_id,
                        source_iid=source_iid,
                        source_sha=source_sha,
                        source=source,
                        references=references,
                        bucket_hw=(48, 48),
                    )
                    prompt_path, prompt_sha = self._make_prompt_receipt(
                        root,
                        case_id=case_id,
                        source_iid=source_iid,
                        caption=caption,
                        prompts=prompts,
                    )
                    material_graph = self._make_material_graph_receipts(
                        root,
                        source_iid=source_iid,
                        source_sha=source_sha,
                        action_caption_sha=caption_sha,
                        reference_path=reference_path,
                        reference_sha=reference_sha,
                        prompt_path=prompt_path,
                        prompt_sha=prompt_sha,
                    )
                    material_review_sha = material_graph["review_sha"]
                common = {
                    "case_id": case_id,
                    "source_iid": source_iid,
                    "decision": (
                        "ACTIVE_DIAGNOSTIC" if case_id == "e02" else "ABSTAIN_KEEP_BASE"
                    ),
                    "source_video": {"path": str(source_path), "sha256": source_sha},
                    "self_generated_anchor": {
                        "path": str(anchor_path),
                        "sha256": anchor_sha,
                    },
                    "anchor_used_only_as_review_context": True,
                    "self_generated_anchor_tensor_used_by_native_expert": False,
                    "target_video_or_latent_used": False,
                    "failed_active_used_to_author_gate": False,
                    "anchor_source_difference_used_to_author_gate": False,
                    "predicted_soft_gate_used_to_author_gate": False,
                    "automatic_model_replacement_authorized": False,
                    "action_caption": caption,
                    "action_caption_sha256": caption_sha,
                    "structured_action_program": programs[case_id],
                    "structured_action_program_sha256": program_sha,
                    "seed": 0,
                    "full_source_latent_geometry": list(geometry),
                    "hard_gate_geometry": list(gate_geometry),
                    "reference_latent_geometry": list(reference_geometry),
                    "reference_rgb_indices": list(activation.REFERENCE_RGB_INDICES),
                    "run_arms": list(
                        activation.EXPECTED_ARMS_E02
                        if case_id == "e02"
                        else activation.EXPECTED_ARMS_E03
                    ),
                    "manual_gate_manifest": None,
                    "independent_review_receipt": None,
                    "annotation_authority_root_sha256": None,
                    "vae_reference_receipt": (
                        {"path": str(reference_path), "sha256": reference_sha}
                        if case_id == "e02"
                        else None
                    ),
                    "prompt_receipt": (
                        {"path": str(prompt_path), "sha256": prompt_sha}
                        if case_id == "e02"
                        else None
                    ),
                }
                if case_id == "e02":
                    assert material_graph is not None
                    common.update(
                        {
                            "anchor_terminal_disappearance_observed": True,
                            "anchor_strict_target_pass": False,
                            "vae_authoring_run_receipt": {
                                "path": str(material_graph["vae_run_path"]),
                                "sha256": material_graph["vae_run_sha"],
                            },
                            "prompt_authoring_run_receipt": {
                                "path": str(material_graph["prompt_run_path"]),
                                "sha256": material_graph["prompt_run_sha"],
                            },
                            "materialization_review_receipt": {
                                "path": str(material_graph["review_path"]),
                                "sha256": material_graph["review_sha"],
                            },
                        }
                    )
                    gate_path, gate_sha, review_path, review_sha, root_sha = (
                        self._make_gate(
                            root,
                            case_row=common,
                            annotator="annotator.unit",
                            reviewer="reviewer.unit",
                            geometry=gate_geometry,
                            contact_start=contact_start,
                        )
                    )
                    common["manual_gate_manifest"] = {
                        "path": str(gate_path),
                        "sha256": gate_sha,
                    }
                    common["independent_review_receipt"] = {
                        "path": str(review_path),
                        "sha256": review_sha,
                    }
                    common["annotation_authority_root_sha256"] = root_sha
                else:
                    common["local_regeneration_selection_authorized"] = False
                    common["kept_frozen_base"] = {
                        "path": str(kept_base_path),
                        "sha256": kept_base_sha,
                    }
                cases.append(common)
            packet = {
                "schema_version": activation.AUTHORITY_PACKET_SCHEMA_VERSION,
                "status": "INDEPENDENT_MODEL_REVIEWED_DIAGNOSTIC_EXPERIMENTAL_PACKET",
                "packet_id": "round37-unit-authority",
                "execution_contract": {
                    "native_only": True,
                    "flowedit_enabled": False,
                    "connected_runner_enabled": False,
                    "learned_gate_enabled": False,
                    "world_size": 4,
                    "sequence_parallel_size": 4,
                    "one_node": True,
                    "same_seed_and_official_gaussian": True,
                    "candidate_count_per_arm": 1,
                    "source_reference_r2v4_regeneration_expert": True,
                    "self_generated_anchor_tensor_used_by_native_expert": False,
                    "anchor_reference_or_quotient_arm_deferred": True,
                    "global_source_reference_r2v4_upper_bound_arm_deferred": True,
                },
                "safety_contract": {
                    "training_authorized": False,
                    "optimizer_authorized": False,
                    "automatic_model_replacement_authorized": False,
                    "background_cosine_selection_authorized": False,
                    "target_video_or_latent_used": False,
                },
                "cases": cases,
            }
            packet_path = _write(root / "authority.json", packet)
            packet_sha = _sha(packet_path)
            ledger = {
                "schema_version": activation.LEDGER_RECEIPT_SCHEMA_VERSION,
                "authority_packet_sha256": packet_sha,
                "packet_id": packet["packet_id"],
                "annotator": "annotator.unit",
                "reviewer": "reviewer.unit",
                "issuer": "issuer.unit",
                "annotator_kind": "AI_AGENT",
                "reviewer_kind": "AI_AGENT",
                "issuer_kind": "AI_AGENT",
                "trust_root_kind": "COMPILED_EXACT_PACKET_AND_LEDGER_SHA256_CODE_REVIEW",
                "accepted": True,
                "e02_exact_gate_reviewed": True,
                "e03_abstain_keep_base_reviewed": True,
                "authority_packet_contains_no_activation_code_hashes": True,
                "private_signing_material_present": False,
                "cryptographic_signature_claimed": cryptographic_signature_claimed,
                "diagnostic_experimental_canary_only": True,
                "formal_authority": False,
                "training_authority": False,
            }
            ledger_path = _write(root / "ledger.json", ledger)
            expected_latent = {case_id: geometry for case_id in activation.ALLOWED_CASES}
            expected_gate = {case_id: gate_geometry for case_id in activation.ALLOWED_CASES}
            expected_reference = {
                case_id: reference_geometry for case_id in activation.ALLOWED_CASES
            }
            expected_input_hw = {
                case_id: (16, 16) for case_id in activation.ALLOWED_CASES
            }
            with mock.patch.object(
                activation, "COMPILED_AUTHORITY_PACKET_SHA256", packet_sha
            ), mock.patch.object(
                activation,
                "COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256",
                _sha(ledger_path),
            ), mock.patch.object(
                activation, "EXPECTED_CASE_BINDINGS", expected_bindings
            ), mock.patch.object(
                activation, "EXPECTED_LATENT_GEOMETRY", expected_latent
            ), mock.patch.object(
                activation, "EXPECTED_GATE_GEOMETRY", expected_gate
            ), mock.patch.object(
                activation, "EXPECTED_REFERENCE_GEOMETRY", expected_reference
            ), mock.patch.object(
                activation, "EXPECTED_SOURCE_INPUT_HW", expected_input_hw
            ), mock.patch.object(
                activation,
                "EXPECTED_E02_MATERIAL_REVIEW_SHA256",
                material_review_sha,
            ), mock.patch.object(
                activation, "EXPECTED_E03_FROZEN_BASE_PATH", kept_base_path
            ), mock.patch.object(
                activation, "EXPECTED_E03_FROZEN_BASE_SHA256", kept_base_sha
            ):
                yield {
                    "root": root,
                    "packet_path": packet_path,
                    "ledger_path": ledger_path,
                    "source": source,
                    "references": references,
                    "prompts": prompts,
                    "media": media,
                }

    def test_compiled_packet_validates_dck_reference_prompt_and_mints_e02_only(self) -> None:
        with self._authority_fixture() as fixture:
            authority = activation.load_compiled_activation_authority_v2(
                fixture["packet_path"], fixture["ledger_path"]
            )
            e03 = authority.cases["e03"]
            self.assertEqual(e03.run_arms, ())
            self.assertIsNone(e03.reference_receipt_path)
            self.assertIsNone(e03.prompt_receipt_path)
            policy = activation.preflight_case_material_receipts_v2(
                authority, case_id="e03"
            )
            self.assertFalse(policy["executed"])
            self.assertFalse(policy["material_receipts_present"])
            self.assertEqual(policy["decision"], "ABSTAIN_KEEP_BASE")
            gate = activation.validate_manual_gate_v2(authority)
            self.assertEqual(gate.support_count, 104)
            self.assertEqual(gate.contact_count, 68)
            self.assertFalse(bool(gate.owned_gate.contact[:, :, 0:4].any().item()))
            self.assertEqual(
                {
                    phase
                    for phase in range(21)
                    if bool(gate.owned_gate.contact[:, :, phase].any().item())
                },
                set(range(4, 21)),
            )
            reference = activation.validate_reference_receipt_v2(
                authority,
                case_id="e02",
                source_video_latent=fixture["source"],
                source_reference_latents=fixture["references"],
            )
            prompt = activation.validate_prompt_receipt_v2(
                authority,
                case_id="e02",
                low_action_prompt_embeds=fixture["prompts"][0],
                high_action_prompt_embeds=fixture["prompts"][1],
                negative_prompt_embeds=fixture["prompts"][2],
            )
            capability = activation.mint_native_local_execution_capability_v2(
                authority,
                case_id="e02",
                source_video_latent=fixture["source"],
                source_reference_latents=fixture["references"],
                low_action_prompt_embeds=fixture["prompts"][0],
                high_action_prompt_embeds=fixture["prompts"][1],
                negative_prompt_embeds=fixture["prompts"][2],
            )
            self.assertEqual(capability.source_latent_sha256, reference.source_latent_sha256)
            self.assertEqual(capability.r2v_action_prompt_sha256, prompt.high_action_sha256)
            with self.assertRaisesRegex(activation.OracleActivationV2Error, "e03"):
                activation.mint_native_local_execution_capability_v2(
                    authority,
                    case_id="e03",
                    source_video_latent=fixture["source"],
                    source_reference_latents=fixture["references"],
                    low_action_prompt_embeds=fixture["prompts"][0],
                    high_action_prompt_embeds=fixture["prompts"][1],
                    negative_prompt_embeds=fixture["prompts"][2],
                )

    def test_wrong_media_prompt_ref_alias_and_authority_mutation_fail_closed(self) -> None:
        torch = self.torch
        with self._authority_fixture() as fixture:
            authority = activation.load_compiled_activation_authority_v2(
                fixture["packet_path"], fixture["ledger_path"]
            )
            wrong_source = fixture["source"].clone()
            wrong_source[0, 0, 1, 0, 0] = 9.0
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error,
                "live VAE source/reference provenance differs",
            ) as source_error:
                activation.validate_reference_receipt_v2(
                    authority,
                    case_id="e02",
                    source_video_latent=wrong_source,
                    source_reference_latents=fixture["references"],
                )
            source_diagnostic = json.loads(
                str(source_error.exception).split(": ", 1)[1]
            )
            self.assertFalse(source_diagnostic["contract_matches"])
            self.assertFalse(source_diagnostic["source_matches"])
            self.assertEqual(source_diagnostic["reference_mismatch_positions"], [])
            self.assertEqual(source_diagnostic["source_slice_collision_pairs"], [])
            self.assertEqual(
                source_diagnostic["observed_source_sha256"],
                activation.safe_core.tensor_content_sha256_v1(wrong_source),
            )
            wrong_references = tuple(value.clone() for value in fixture["references"])
            wrong_references[2][0, 0, 0, 0, 0] += 1.0
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error,
                "live VAE source/reference provenance differs",
            ) as reference_error:
                activation.validate_reference_receipt_v2(
                    authority,
                    case_id="e02",
                    source_video_latent=fixture["source"],
                    source_reference_latents=wrong_references,
                )
            reference_diagnostic = json.loads(
                str(reference_error.exception).split(": ", 1)[1]
            )
            self.assertTrue(reference_diagnostic["source_matches"])
            self.assertEqual(
                reference_diagnostic["reference_mismatch_positions"], [2]
            )
            self.assertEqual(
                reference_diagnostic["observed_reference_sha256"][2],
                activation.safe_core.tensor_content_sha256_v1(wrong_references[2]),
            )
            aliases = (
                fixture["references"][0],
                fixture["references"][0],
                fixture["references"][2],
                fixture["references"][3],
            )
            with self.assertRaises(activation.OracleActivationV2Error):
                activation.validate_reference_receipt_v2(
                    authority,
                    case_id="e02",
                    source_video_latent=fixture["source"],
                    source_reference_latents=aliases,
                )
            wrong_high = fixture["prompts"][1].clone()
            wrong_high[0, 0, 0] = torch.tensor(7.0, dtype=torch.bfloat16)
            with self.assertRaises(activation.OracleActivationV2Error):
                activation.validate_prompt_receipt_v2(
                    authority,
                    case_id="e02",
                    low_action_prompt_embeds=fixture["prompts"][0],
                    high_action_prompt_embeds=wrong_high,
                    negative_prompt_embeds=fixture["prompts"][2],
                )
            source_path = fixture["media"]["e02"][0]
            source_path.write_bytes(b"wrong-media")
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error, "graph|bytes|changed"
            ):
                activation.revalidate_compiled_activation_authority_v2(authority)

    def test_reference_provenance_diagnostic_reports_exact_slice_collision(self) -> None:
        expected_refs = tuple(hashlib.sha256(f"ref-{index}".encode()).hexdigest() for index in range(4))
        source_slices = tuple(
            expected_refs[1] if index == 2 else hashlib.sha256(f"slice-{index}".encode()).hexdigest()
            for index in range(4)
        )
        diagnostic = activation._reference_provenance_diagnostic_v2(
            expected_source_sha256="a" * 64,
            observed_source_sha256="a" * 64,
            expected_reference_sha256=expected_refs,
            observed_reference_sha256=expected_refs,
            observed_source_slice_sha256=source_slices,
        )
        self.assertFalse(diagnostic["contract_matches"])
        self.assertTrue(diagnostic["source_matches"])
        self.assertEqual(diagnostic["reference_mismatch_positions"], [])
        self.assertEqual(
            diagnostic["source_slice_collision_pairs"],
            [{"reference_position": 1, "source_phase": 13}],
        )

    def test_cryptographic_signature_claim_cannot_upgrade_code_review_root(self) -> None:
        with self._authority_fixture(cryptographic_signature_claimed=True) as fixture:
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error, "external ledger"
            ):
                activation.load_compiled_activation_authority_v2(
                    fixture["packet_path"], fixture["ledger_path"]
                )

    def test_K_active_window_is_exact_and_rejects_early_contact_permission(self) -> None:
        with self._authority_fixture(contact_start=3) as fixture:
            authority = activation.load_compiled_activation_authority_v2(
                fixture["packet_path"], fixture["ledger_path"]
            )
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error, "topology|phase semantics"
            ):
                activation.validate_manual_gate_v2(authority)

    def _gate_tensor(self, *, null: bool):
        torch = self.torch
        delete = torch.zeros(1, 1, 21, 2, 2, dtype=torch.bool)
        create = torch.zeros_like(delete)
        contact = torch.zeros_like(delete)
        if not null:
            delete[:, :, 1, 0, 0] = True
            contact[:, :, 1, 0, 1] = True
            create[:, :, 1, 1, 1] = True
        support = torch.logical_or(torch.logical_or(delete, create), contact).contiguous()
        preserve = torch.logical_not(support).contiguous()
        provisional = activation._OwnedHardStateChangeGateV2(
            delete=delete,
            create=create,
            contact=contact,
            support=support,
            preserve=preserve,
            source_mask_sha256="b" * 64,
            realized_gate_sha256="0" * 64,
            delete_count=int(delete.sum()),
            create_count=int(create.sum()),
            contact_count=int(contact.sum()),
            support_count=int(support.sum()),
        )
        return activation._OwnedHardStateChangeGateV2(
            **{
                **provisional.__dict__,
                "realized_gate_sha256": activation._realized_gate_sha256_v2(provisional),
            }
        )

    def test_null_G_returns_original_signed_zero_object_before_high_validation(self) -> None:
        torch = self.torch
        shape = (1, 16, 21, 2, 2)
        official = torch.full((1, 21, 64), -0.0, dtype=torch.float32)
        executed, trace = activation._scheduled_local_velocity_v2(
            sample=torch.zeros_like(official),
            high_r2v4_velocity=object(),
            official_v2v_velocity=official,
            sigma=torch.tensor(1.0, dtype=torch.float32),
            gate=self._gate_tensor(null=True),
            target_latent_shape=shape,
        )
        self.assertIs(executed, official)
        self.assertFalse(trace["scheduled_expert_evaluated"])
        self.assertTrue(trace["scheduler_received_original_official_object"])
        self.assertTrue(
            torch.equal(
                executed.contiguous().view(torch.uint8),
                official.contiguous().view(torch.uint8),
            )
        )

    def test_active_DCK_route_keeps_bytes_outside_G(self) -> None:
        torch = self.torch
        shape = (1, 16, 21, 2, 2)
        official = torch.full((1, 21, 64), -0.0, dtype=torch.float32)
        high = torch.ones_like(official)
        gate = self._gate_tensor(null=False)
        executed, trace = activation._scheduled_local_velocity_v2(
            sample=torch.zeros_like(official),
            high_r2v4_velocity=high,
            official_v2v_velocity=official,
            sigma=torch.tensor(1.0, dtype=torch.float32),
            gate=gate,
            target_latent_shape=shape,
        )
        packed = sgaf._spatial_to_packed(gate.support.expand(shape), shape)
        outside = torch.logical_not(packed)
        self.assertTrue(
            torch.equal(
                executed[outside].contiguous().view(torch.uint8),
                official[outside].contiguous().view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(executed[packed], high[packed]))
        self.assertEqual(trace["hard_support_definition"], "G=D_or_C_or_K")

    def test_authenticated_fake_patch_executes_five_forward_exact40(self) -> None:
        """Exercise the real local patch while authority tests stay separate."""

        fixture_path = METHOD_ROOT / "tests/test_native_branch_homotopy_runtime_v1.py"
        fixture_spec = importlib.util.spec_from_file_location(
            "_activation_v2_frozen_runtime_fixture", fixture_path
        )
        self.assertIsNotNone(fixture_spec)
        self.assertIsNotNone(fixture_spec.loader)
        frozen_test = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(frozen_test)

        torch = self.torch
        fixture = frozen_test.RuntimePatchTests(methodName="runTest")
        fixture.torch = torch
        fixture.setUp()
        try:
            diffusion = fixture._diffusion()
            config = fixture._config()
            sample_kwargs = fixture._sample_kwargs(diffusion)
            gate_shape = (1, 1, 21, 2, 2)
            delete = torch.zeros(gate_shape, dtype=torch.bool)
            delete[:, :, 1:, 0, 0] = True
            create = torch.zeros_like(delete)
            contact = torch.zeros_like(delete)
            support = delete.clone().contiguous()
            preserve = torch.logical_not(support).contiguous()
            provisional = activation._OwnedHardStateChangeGateV2(
                delete=delete,
                create=create,
                contact=contact,
                support=support,
                preserve=preserve,
                source_mask_sha256="a" * 64,
                realized_gate_sha256="0" * 64,
                delete_count=int(delete.sum().item()),
                create_count=0,
                contact_count=0,
                support_count=int(support.sum().item()),
            )
            gate = activation._OwnedHardStateChangeGateV2(
                **{
                    **provisional.__dict__,
                    "realized_gate_sha256": activation._realized_gate_sha256_v2(
                        provisional
                    ),
                }
            )
            source = sample_kwargs["multi_video_vae_latents"][0]
            references = tuple(sample_kwargs["multi_image_vae_latents"])
            low_action = sample_kwargs["prompt_embeds"]
            high_action = fixture.high_action
            negative = sample_kwargs["uncond_prompt_embeds"]
            authority = SimpleNamespace(
                packet_sha256="b" * 64,
                ledger_sha256="c" * 64,
                packet_path=Path("/unit-authority-packet"),
            )
            manifest = SimpleNamespace(annotation_authority_root_sha256="d" * 64)
            capability = activation.NativeLocalExecutionCapabilityV2(
                authority=authority,
                case_id="e02",
                sample_id="unit-five-forward",
                manifest=manifest,
                owned_gate=gate,
                realized_gate_sha256=gate.realized_gate_sha256,
                source_latent_sha256=activation.safe_core.tensor_content_sha256_v1(
                    source
                ),
                source_reference_latent_sha256=tuple(
                    activation.safe_core.tensor_content_sha256_v1(value)
                    for value in references
                ),
                source_reference_rgb_indices=(0, 27, 53, 80),
                low_action_prompt_sha256=activation.safe_core.tensor_content_sha256_v1(
                    low_action
                ),
                r2v_action_prompt_sha256=activation.safe_core.tensor_content_sha256_v1(
                    high_action
                ),
                negative_prompt_sha256=activation.safe_core.tensor_content_sha256_v1(
                    negative
                ),
                r2v_action_prompt_embeds=high_action,
                authority_packet_path=authority.packet_path,
                authority_packet_sha256=authority.packet_sha256,
                _validation_token=activation._CAPABILITY_TOKEN,
            )
            with mock.patch.object(
                activation, "revalidate_compiled_activation_authority_v2"
            ), mock.patch.object(
                activation, "validate_manual_gate_v2"
            ), mock.patch.object(
                activation, "validate_reference_receipt_v2"
            ), mock.patch.object(
                activation, "validate_prompt_receipt_v2"
            ):
                patch = activation.LocalOracleNativeBranchRuntimePatchV2(
                    diffusion,
                    config=config,
                    capability=capability,
                )
                patch.install()
                try:
                    result = diffusion.sample(**sample_kwargs)
                finally:
                    patch.restore()
                receipt = patch.finalize()
            self.assertEqual(tuple(result.shape), (1, 21, 64))
            self.assertEqual(receipt["transformer_forwards"], 200)
            self.assertEqual(receipt["low_vi_forwards"], 80)
            self.assertEqual(receipt["high_r2v4_forwards"], 120)
            self.assertEqual(receipt["patch_vae_latent_calls"], 400)
            self.assertEqual(receipt["original_scheduler_calls"], 40)
            self.assertTrue(receipt["outside_G_official_bytes_exact_all_steps"])
            self.assertEqual(
                receipt["exact40_scheduled_endpoint_partition"],
                {
                    "high_r2v4_apg_indices": list(range(0, 15)),
                    "transition_indices": list(range(15, 31)),
                    "low_official_v2v_apg_indices": list(range(31, 40)),
                },
            )
            packed_support = sgaf._spatial_to_packed(
                gate.support.expand(config.target_latent_shape),
                config.target_latent_shape,
            )
            outside = torch.logical_not(packed_support)
            for received, official in zip(
                diffusion.scheduler.received_objects, diffusion.official_outputs
            ):
                self.assertTrue(
                    torch.equal(
                        received[outside].contiguous().view(torch.uint8),
                        official[outside].contiguous().view(torch.uint8),
                    )
                )
            for index in range(31, 40):
                self.assertIs(
                    diffusion.scheduler.received_objects[index],
                    diffusion.official_outputs[index],
                )
        finally:
            fixture.tearDown()

    def test_local_release_import_objects_and_full_closure_are_exact(self) -> None:
        import infer_native_branch_homotopy_canary as prompt_builder
        import infer_native_identity_generation_canary as native
        import infer_native_self_guided_action_field_canary as freeze_provider
        import infer_oracle_regeneration_native_activation_v2_r2 as runner
        import infer_source_kv_carrier_oracle as source_audit
        import source_self_native_ref_contrastive_v3 as native_schedule_contract
        import tools as tools_package
        from tools import materialize_vae
        import tri_branch_unipc as sampler_contract

        identities = {
            label: runner._owned_file_identity(
                METHOD_ROOT / relative, label=f"unit closure {label}"
            )
            for label, relative in runner._LOCAL_RELEASE_PATHS.items()
        }
        receipt = {"component_identities": identities}
        observed = runner._certify_local_release_import_closure(
            preflight_receipt=receipt,
            activation=activation,
            native=native,
            prompt_builder=prompt_builder,
            source_audit=source_audit,
            freeze_provider=freeze_provider,
            sampler_contract=sampler_contract,
            native_schedule_contract=native_schedule_contract,
            tools_package=tools_package,
            materialize_vae=materialize_vae,
        )
        self.assertEqual(
            set(observed["release_sha256"]), set(runner._LOCAL_RELEASE_PATHS)
        )
        self.assertIs(
            activation._load_native_schedule_contract_v2(),
            native_schedule_contract,
        )
        with mock.patch.dict(
            sys.modules,
            {"source_self_native_ref_contrastive_v3": SimpleNamespace()},
        ):
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error,
                "schedule import origin differs",
            ):
                activation._load_native_schedule_contract_v2()
        same_path_fake = SimpleNamespace(__file__=source_audit.__file__)
        with mock.patch.object(native, "source_audit", same_path_fake):
            with self.assertRaisesRegex(
                runner.NativeActivationV2RunnerError,
                "runtime import objects",
            ):
                runner._certify_local_release_import_closure(
                    preflight_receipt=receipt,
                    activation=activation,
                    native=native,
                    prompt_builder=prompt_builder,
                    source_audit=source_audit,
                    freeze_provider=freeze_provider,
                    sampler_contract=sampler_contract,
                    native_schedule_contract=native_schedule_contract,
                    tools_package=tools_package,
                    materialize_vae=materialize_vae,
                )
        wrong = copy.deepcopy(receipt)
        wrong["component_identities"]["runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            runner.NativeActivationV2RunnerError, "local release closure"
        ):
            runner._certify_local_release_import_closure(
                preflight_receipt=wrong,
                activation=activation,
                native=native,
                prompt_builder=prompt_builder,
                source_audit=source_audit,
                freeze_provider=freeze_provider,
                sampler_contract=sampler_contract,
                native_schedule_contract=native_schedule_contract,
                tools_package=tools_package,
                materialize_vae=materialize_vae,
            )

    def test_official_fake_arm_is_observer_only_and_restores_call_surface(self) -> None:
        import infer_native_identity_generation_canary as native
        import infer_oracle_regeneration_native_activation_v2_r2 as runner

        fixture_path = METHOD_ROOT / "tests/test_native_branch_homotopy_runtime_v1.py"
        fixture_spec = importlib.util.spec_from_file_location(
            "_activation_v2_official_runtime_fixture", fixture_path
        )
        self.assertIsNotNone(fixture_spec)
        self.assertIsNotNone(fixture_spec.loader)
        frozen_test = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(frozen_test)
        torch = self.torch
        fixture = frozen_test.RuntimePatchTests(methodName="runTest")
        fixture.torch = torch
        fixture.setUp()
        try:
            diffusion = fixture._diffusion()
            sample_kwargs = fixture._sample_kwargs(diffusion)
            surface_before = runner._capture_call_surface(diffusion)
            original_values = []
            sample_values = []

            def canonical_randn_tensor(shape, *, generator, device, dtype):
                value = torch.randn(
                    shape, generator=generator, device=device, dtype=dtype
                )
                original_values.append(value)
                return value

            module = types.SimpleNamespace(randn_tensor=canonical_randn_tensor)

            def official_sample():
                value = module.randn_tensor(
                    (1, 16, 21, 2, 2),
                    generator=torch.Generator(device="cpu").manual_seed(7),
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                sample_values.append(value)
                return diffusion.sample(**sample_kwargs)

            # Torch 1.12 in the ``vd`` audit environment was compiled against
            # NumPy 1.x, while that environment now supplies NumPy 2.x.  The
            # production identity helper therefore cannot call ``.numpy()``
            # there even though Torch itself is fully usable.  Exercise the
            # observer with the production helper whenever available and with
            # a byte-exact, NumPy-free equivalent only for that ABI mismatch.
            tensor_identity = native.value_audit.tensor_identity
            try:
                tensor_identity(torch.zeros(1), label="numpy_abi_probe")
            except RuntimeError as error:
                if "Numpy is not available" not in str(error):
                    raise

                def tensor_identity(value, *, label):
                    detached = value.detach().contiguous()
                    if not bool(torch.isfinite(detached).all()):
                        raise AssertionError(f"{label} contains non-finite values")
                    cpu = detached.cpu().contiguous()
                    raw = bytes(cpu.reshape(-1).view(torch.uint8).tolist())
                    metadata = {
                        "shape": [int(item) for item in cpu.shape],
                        "dtype": str(cpu.dtype),
                        "numel": int(cpu.numel()),
                        "byte_count": len(raw),
                    }
                    payload = (
                        native.value_audit.legacy.canonical_json_bytes(metadata)
                        + b"\0"
                        + raw
                    )
                    return {
                        **metadata,
                        "content_sha256": hashlib.sha256(payload).hexdigest(),
                        "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
                        "finite": True,
                        "label": label,
                    }

            with mock.patch.object(
                native.value_audit,
                "tensor_identity",
                side_effect=tensor_identity,
            ):
                result, capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=official_sample,
                    wan_diffusion_module=module,
                    expected_shape=(1, 16, 21, 2, 2),
                    expected_device=torch.device("cpu"),
                    expected_seed=7,
                    canonical_randn_tensor=canonical_randn_tensor,
                )
            surface_after = runner._certify_call_surface(
                diffusion, surface_before, label="unit official arm"
            )
            self.assertIs(module.randn_tensor, canonical_randn_tensor)
            self.assertIs(sample_values[0], original_values[0])
            self.assertEqual(capture.call_count, 1)
            self.assertEqual(capture.generator_initial_seed, 7)
            self.assertEqual(tuple(result.shape), (1, 21, 64))
            self.assertEqual(diffusion.shared_call_count, 80)
            self.assertTrue(surface_after["all_instance_overrides_absent"])
        finally:
            fixture.tearDown()

    def test_fresh_arms_use_identical_scoped_unused_t5_bypass(self) -> None:
        import infer_oracle_regeneration_native_activation_v2_r2 as runner

        torch = self.torch
        constructor_rows = []
        checkpoint = Path("/unit/checkpoint")

        class FakeConfig:
            shift = 5.0
            use_unipc = True
            dtype = None

            @staticmethod
            def to_dict():
                return {"shift": 5.0, "use_unipc": True}

        class FakeConfigClass:
            @staticmethod
            def from_pretrained(path, **kwargs):
                constructor_rows.append(("config", path, dict(kwargs)))
                return FakeConfig()

        class FakeT5Encoder:
            real_calls = 0

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.real_calls += 1
                raise AssertionError("real all-rank T5 constructor must be bypassed")

        class FakeRenderer(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                constructor_rows.append(("renderer", config))
                self.t5_text_encoder = FakeT5Encoder.from_pretrained(
                    str(checkpoint),
                    subfolder="text_encoder",
                    torch_dtype=torch.bfloat16,
                )
                self.diff_dec = torch.nn.Linear(2, 2, bias=False)

        class FakeDist:
            barriers = 0

            def barrier(self):
                self.barriers += 1

        trainer = SimpleNamespace(
            validate_renderer_config_mapping=lambda mapping, checkpoint: None
        )

        def canonical_load(factory, config, device):
            model = factory(config)
            model.requires_grad_(False)
            model.eval()
            model.to(device)
            return model

        native = SimpleNamespace(
            legacy=SimpleNamespace(
                inference_renderer_config_overrides=lambda checkpoint: {},
                trainer=trainer,
            ),
            _load_frozen_renderer_gpu_resident_serialized=canonical_load,
        )
        dist = FakeDist()
        original = vars(FakeT5Encoder)["from_pretrained"]
        receipts = []
        for _arm in (runner.ARM_OFFICIAL, runner.ARM_LOCAL):
            model, _, receipt = runner._load_fresh_arm_renderer(
                native=native,
                renderer_config_class=FakeConfigClass,
                renderer_model_class=FakeRenderer,
                t5_encoder_class=FakeT5Encoder,
                bernini_root=Path("/unit/bernini"),
                checkpoint=checkpoint,
                torch=torch,
                dist=dist,
                device=torch.device("cpu"),
            )
            self.assertIsNone(model.t5_text_encoder)
            self.assertIs(vars(FakeT5Encoder)["from_pretrained"], original)
            receipts.append(receipt)
        self.assertEqual(
            [row[0] for row in constructor_rows],
            ["config", "renderer", "config", "renderer"],
        )
        self.assertEqual(FakeT5Encoder.real_calls, 0)
        self.assertEqual(dist.barriers, 2)
        self.assertEqual(receipts[0], receipts[1])
        self.assertTrue(receipts[0]["unused_t5_deserialization_bypassed_all_ranks"])
        self.assertEqual(receipts[0]["bypass_call_count"], 1)
        self.assertFalse(
            receipts[0]["official_denoiser_scheduler_surface_changed_by_bypass"]
        )
        runner_source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("_all_rank_t5_constructor_bypass", runner_source)


if __name__ == "__main__":
    unittest.main()
