from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from methods.bernini_action_editing.tools import (
    build_schedule_block_causal_localization_release_v1 as release,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
BASE_RELEASE = METHOD_ROOT / "releases/source_noised_carrier_stage_b_inference_r3"
RUNTIME = METHOD_ROOT / "infer_schedule_block_causal_localization_v1.py"
RUNTIME_CLOSURE_FROZEN = (
    RUNTIME.exists()
    and release.EXPECTED_FILE_SHA256["schedule_block_target_row_prompt_swap_v1.py"]
    != "0" * 64
    and release.EXPECTED_FILE_SHA256["infer_schedule_block_causal_localization_v1.py"]
    != "0" * 64
)
STAGE_B_CONTROLLER = (
    METHOD_ROOT / "scripts/auh_infer_source_noised_carrier_stage_b_two_holder_v5.sh"
)


class ScheduleBlockReleaseSourceStaticTests(unittest.TestCase):
    def test_declared_added_member_closure_contains_core_and_authorities(self) -> None:
        self.assertEqual(
            release.ADDED_FILES,
            (
                "schedule_block_causal_policy_v1.py",
                "schedule_block_target_row_prompt_swap_v1.py",
                "infer_schedule_block_causal_localization_v1.py",
                "train_source_self_identity_orbit_v4.py",
                "source_self_native_ref_contrastive_v3.py",
                "appearance_counterfactual_identity_orbit.py",
                "source_self_identity_orbit_v4.py",
                "source_self_native_rv2v_guidance.py",
                "source_self_native_target_adapter.py",
                "tools/materialize_appearance_counterfactual_identity_orbit.py",
                "mdr_exact_motion_analogy.py",
                "assets/pair_v5_t2v_calibration_first8_authoring_v1.json",
                "assets/appearance_identity_orbit_portrait2_review_v1.json",
            ),
        )
        self.assertEqual(
            release.EXPECTED_FILE_SHA256[
                "schedule_block_target_row_prompt_swap_v1.py"
            ],
            "385cc2321da888f75d5aff5017175b85acf06174969aaa39210b802cc14695c5",
        )
        self.assertEqual(
            release.EXPECTED_FILE_SHA256[
                "infer_schedule_block_causal_localization_v1.py"
            ],
            "913272f2d8c55dd0d07ebae26fb17788b82f95955b38421b1d9d5ae48b21ed1f",
        )

    def test_python_source_has_no_duplicate_literal_dict_keys(self) -> None:
        source_path = Path(release.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = set()
            for key in node.keys:
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                if key.value in seen:
                    duplicates.append((node.lineno, key.value))
                seen.add(key.value)
        self.assertEqual(duplicates, [])

    def test_every_nonzero_added_member_pin_matches_actual_bytes(self) -> None:
        for relative in release.ADDED_FILES:
            expected = release.EXPECTED_FILE_SHA256[relative]
            if expected == "0" * 64:
                continue
            with self.subTest(relative=relative):
                self.assertEqual(
                    hashlib.sha256((METHOD_ROOT / relative).read_bytes()).hexdigest(),
                    expected,
                )


@unittest.skipUnless(
    RUNTIME_CLOSURE_FROZEN, "independently-owned Stage-A runtime closure not frozen"
)
class ScheduleBlockCausalLocalizationReleaseTests(unittest.TestCase):
    def test_release_closes_all_repository_local_python_imports(self) -> None:
        available = set(release.RELEASE_FILES)
        root_modules = {
            path.stem: path.name for path in METHOD_ROOT.glob("*.py")
        }
        tool_modules = {
            path.stem: f"tools/{path.name}"
            for path in (METHOD_ROOT / "tools").glob("*.py")
        }
        missing = []

        def require(module: str, *, importer: str) -> None:
            parts = module.split(".")
            relative = None
            if parts[0] == "tools" and len(parts) >= 2:
                relative = tool_modules.get(parts[1])
            elif parts[0] in root_modules:
                relative = root_modules[parts[0]]
            if relative is not None and relative not in available:
                missing.append((importer, module, relative))

        for relative in release.RELEASE_FILES:
            if not relative.endswith(".py"):
                continue
            tree = ast.parse(
                (METHOD_ROOT / relative).read_text(encoding="utf-8"),
                filename=relative,
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        require(alias.name, importer=relative)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        require(node.module, importer=relative)
                    if node.level or node.module == "tools":
                        prefix = "tools." if node.module == "tools" else ""
                        for alias in node.names:
                            require(prefix + alias.name, importer=relative)
                elif (
                    isinstance(node, ast.Call)
                    and (
                        (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                        or (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "import_module"
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "importlib"
                        )
                    )
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    require(node.args[0].value, importer=relative)
        self.assertEqual(missing, [])

    def test_exact_diagnostic_release_is_deterministic(self) -> None:
        manifest, payloads = release.build_manifest(
            METHOD_ROOT,
            (BASE_RELEASE / "source.tar").resolve(),
            (BASE_RELEASE / "source.manifest.json").resolve(),
        )
        self.assertEqual(
            manifest["schema_version"],
            "bernini-schedule-block-causal-localization-release-v1",
        )
        self.assertEqual(manifest["release_generation"], "r1")
        self.assertEqual(manifest["file_count"], 27)
        self.assertEqual(
            [row["path"] for row in manifest["files"]],
            list(release.RELEASE_FILES),
        )
        self.assertIs(manifest["diagnostic_only"], True)
        self.assertIs(manifest["optimizer_authorized"], False)
        self.assertIs(manifest["parameter_update_authorized"], False)
        self.assertIs(manifest["scientific_selection_authorized"], False)
        self.assertEqual(manifest["formal_profile"], "smoke-then-full-fixed")
        self.assertIs(manifest["single_model_load_required"], True)
        self.assertEqual(manifest["engineering_c0_decoded_output_count"], 6)
        self.assertEqual(
            manifest["engineering_c0_plan_digest"],
            "d11dbd0cfca34f26ea5f72bdd2f5ed8b21c512387410b659ade9f217d866c923",
        )
        self.assertEqual(
            manifest["preregistered_full_grid_decoded_output_count"], 112
        )
        self.assertEqual(
            manifest["preregistered_full_grid_plan_digest"],
            "6fd3299a1af84968bebe12cd6f1b2a84feb0fb28a07d29619fbcfac66bf4d2e8",
        )
        self.assertEqual(manifest["formal_total_decoded_output_count"], 118)
        self.assertIs(
            manifest["formal_full_continuation_automatic_after_c0_pass"], True
        )
        self.assertIs(manifest["c0_failure_forbids_full_grid"], True)
        self.assertIs(manifest["c0_gate_engineering_only"], True)
        self.assertIs(
            manifest["engineering_c0_has_no_visual_or_scientific_selection"], True
        )
        self.assertIs(
            manifest["prompt_calibration_action_reverse_direction_passed"], True
        )
        self.assertIs(
            manifest["prompt_calibration_noop_incomplete_semantics_passed"], False
        )
        self.assertIs(manifest["negative_cluster_semantically_validated"], False)
        self.assertIs(
            manifest["negative_cluster_scientific_veto_authorized"], False
        )
        self.assertIs(manifest["full_grid_cells_retained_without_deletion"], True)
        self.assertEqual(
            hashlib.sha256(
                payloads[
                    "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
                ]
            ).hexdigest(),
            "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c",
        )
        self.assertEqual(
            hashlib.sha256(
                payloads[
                    "assets/appearance_identity_orbit_portrait2_review_v1.json"
                ]
            ).hexdigest(),
            "dc2d83322357196cec84418ddf4318d9fc7d1eb41269cb216739bae7c6169651",
        )
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        release.verify_archive_bytes(first, manifest)

    def test_publication_is_fresh_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = (root / "source.tar").resolve()
            manifest_path = (root / "source.manifest.json").resolve()
            result = release.build(
                METHOD_ROOT,
                (BASE_RELEASE / "source.tar").resolve(),
                (BASE_RELEASE / "source.manifest.json").resolve(),
                archive,
                manifest_path,
            )
            self.assertEqual(result["file_count"], 27)
            self.assertEqual(
                result["archive_sha256"],
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
            self.assertEqual(archive.stat().st_mode & 0o777, 0o444)
            self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(release.ReleaseError):
                release.build(
                    METHOD_ROOT,
                    (BASE_RELEASE / "source.tar").resolve(),
                    (BASE_RELEASE / "source.manifest.json").resolve(),
                    archive,
                    (root / "second.manifest.json").resolve(),
                )

    def test_pair_preflight_never_publishes_a_half_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            occupied_manifest = (root / "occupied.manifest.json").resolve()
            occupied_manifest.write_bytes(b"belongs-to-another-release\n")
            fresh_archive = (root / "must-remain-absent.tar").resolve()
            with self.assertRaises(release.ReleaseError):
                release.build(
                    METHOD_ROOT,
                    (BASE_RELEASE / "source.tar").resolve(),
                    (BASE_RELEASE / "source.manifest.json").resolve(),
                    fresh_archive,
                    occupied_manifest,
                )
            self.assertFalse(fresh_archive.exists())
            self.assertEqual(
                occupied_manifest.read_bytes(), b"belongs-to-another-release\n"
            )

            aliased_output = (root / "same-output").resolve()
            with self.assertRaises(release.ReleaseError):
                release.build(
                    METHOD_ROOT,
                    (BASE_RELEASE / "source.tar").resolve(),
                    (BASE_RELEASE / "source.manifest.json").resolve(),
                    aliased_output,
                    aliased_output,
                )
            self.assertFalse(aliased_output.exists())

    def test_tampered_base_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hostile_path = (Path(directory) / "hostile.manifest.json").resolve()
            value = json.loads(
                (BASE_RELEASE / "source.manifest.json").read_text(encoding="ascii")
            )
            value["files"][0]["sha256"] = "0" * 64
            hostile_path.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaises(release.ReleaseError):
                release.build_manifest(
                    METHOD_ROOT,
                    (BASE_RELEASE / "source.tar").resolve(),
                    hostile_path,
                )

    def test_stage_b_frozen_bytes_are_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256(STAGE_B_CONTROLLER.read_bytes()).hexdigest(),
            "5aa68c97c52cba9f2a2171b9ff98f6fc865c67ab641c11a07799369715e71f02",
        )
        self.assertEqual(
            hashlib.sha256(
                (METHOD_ROOT / "infer_source_noised_carrier_stage_b_v1.py").read_bytes()
            ).hexdigest(),
            "7e6cdba95c62d2ae9bbe81cfa123ac208c2ca890f134cfe6d0538cefea68db50",
        )
        self.assertEqual(
            hashlib.sha256((BASE_RELEASE / "source.tar").read_bytes()).hexdigest(),
            release.BASE_ARCHIVE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (BASE_RELEASE / "source.manifest.json").read_bytes()
            ).hexdigest(),
            release.BASE_MANIFEST_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
