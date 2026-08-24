from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = MODULE_ROOT / "tools"
for entry in (MODULE_ROOT, TOOLS_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import full644_exploratory_matched_r5f_cpu_consumption_probe_v1 as consumption
import full644_exploratory_matched_r5f_static_nomodel_probe_v1 as static
import full644_exploratory_matched_spooled_launcher_auh_r5f as launcher
import materialize_full644_exploratory_matched_r5f_case00_package_v1 as materializer


ADAPTER_SHA256 = (
    "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19"
)
LAUNCHER_SHA256 = (
    "d70eac5c0ee5fbcbfa84bc3a711fc2e836fa8cc0331555502d2b9b832e7c6b4e"
)
STATIC_SHA256 = (
    "242a48d6720ef4514f841c518a5a9dbdd864102e60d0aa12d845496b6a02b5cd"
)
CONSUMPTION_SHA256 = (
    "fd64fafc9580c8f25c88d79ca603a0dbf192ea98f77403e82f14d4e17c6905f6"
)
MATERIALIZER_SHA256 = (
    "4c6e8c9db7b92750be0e580d0e655b9009b300963983b33750d6697c72bdd0b1"
)
EXPECTED_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_"
    "case00_847b91a2_c91de7eb_d70eac5c_r1"
)
EXPECTED_CACHE = Path(
    "/tmp/bernini-full644-r5f-job143812-node293-r1-rank-cache"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R5FPackageContractTests(unittest.TestCase):
    def test_frozen_source_cascade_and_exact21_snapshot(self) -> None:
        expected = {
            "full644_exploratory_matched_infer_adapter_auh_r5f.py":
                ADAPTER_SHA256,
            "full644_exploratory_matched_spooled_launcher_auh_r5f.py":
                LAUNCHER_SHA256,
            "full644_exploratory_matched_r5f_static_nomodel_probe_v1.py":
                STATIC_SHA256,
            "full644_exploratory_matched_r5f_cpu_consumption_probe_v1.py":
                CONSUMPTION_SHA256,
            "tools/materialize_full644_exploratory_matched_r5f_case00_package_v1.py":
                MATERIALIZER_SHA256,
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(sha(MODULE_ROOT / relative), digest)
        self.assertEqual(
            len(materializer.RELEASE_FILES)
            + len(materializer.DIAGNOSTIC_SOURCE_PINS)
            + 1,
            21,
        )
        for relative, digest in materializer.DIAGNOSTIC_SOURCE_PINS.items():
            self.assertEqual(sha(MODULE_ROOT.parent.parent / relative), digest)

    def test_physical17_identity16_and_exact2_are_distinct(self) -> None:
        self.assertEqual(len(materializer.RELEASE_FILES), 17)
        self.assertEqual(materializer.RELEASE_FILES, static.RELEASE_FILES)
        identity_roles = set(launcher.EXPECTED_STATIC_SHA256) | {
            "python", "ffmpeg", "plan",
        }
        self.assertEqual(len(identity_roles), 16)
        self.assertEqual(
            materializer.SELECTED_TASK_IDS,
            ("shared8-00-base", "shared8-00-full644"),
        )
        self.assertEqual(len(materializer.SELECTED_TASK_IDS), 2)
        self.assertEqual(
            materializer.CAMPAIGN, launcher.CASE00_CANARY_CAMPAIGN
        )
        self.assertIn(
            "methods/bernini_action_editing/"
            "full644_exploratory_matched_infer_adapter_auh_r5f.py",
            materializer.RELEASE_FILES,
        )
        self.assertIn(
            "methods/bernini_action_editing/"
            "full644_exploratory_matched_spooled_launcher_auh_r5f.py",
            materializer.RELEASE_FILES,
        )
        self.assertFalse(
            any(
                name.endswith("infer_adapter_auh_r5d.py")
                or name.endswith("spooled_launcher_auh_r5d.py")
                for name in materializer.RELEASE_FILES
            )
        )

    def test_schema_source_identity_and_reused_root_probe_are_exact(self) -> None:
        self.assertEqual(
            launcher.SCHEMA,
            "full644-exploratory-matched-root-launch-release-auh-r5f",
        )
        self.assertEqual(
            launcher.INPUT_SCHEMA,
            "full644-exploratory-matched-root-launch-input-auh-r5f",
        )
        self.assertEqual(
            launcher.RECEIPT_SCHEMA,
            "full644-exploratory-matched-root-launch-receipt-auh-r5f",
        )
        self.assertEqual(
            hashlib.sha256(launcher.ROOT_BOOTSTRAP.encode("utf-8")).hexdigest(),
            static.ROOT_BOOTSTRAP_SHA256,
        )
        self.assertTrue(
            materializer.ROOT_BOOTSTRAP_PROBE.endswith(
                "full644_exploratory_matched_r5d_"
                "root_bootstrap_probe_runner_v1.py"
            )
        )
        self.assertTrue(
            materializer.STATIC_NOMODEL_PROBE.endswith(
                "full644_exploratory_matched_r5f_"
                "static_nomodel_probe_v1.py"
            )
        )
        self.assertTrue(
            materializer.CPU_CONSUMPTION_PROBE.endswith(
                "full644_exploratory_matched_r5f_"
                "cpu_consumption_probe_v1.py"
            )
        )
        self.assertEqual(
            launcher.EXPECTED_STATIC_SHA256["adapter"], ADAPTER_SHA256
        )
        self.assertEqual(
            consumption.SOURCE_SPECS["r5f_adapter"],
            (
                "full644_exploratory_matched_infer_adapter_auh_r5f.py",
                ADAPTER_SHA256,
                27_042,
            ),
        )

    def test_fresh_root_cache_and_legacy_runner_artifact_basenames(self) -> None:
        key = ("143812", "auh7-1b-gpu-293")
        self.assertEqual(set(materializer.TARGETS), {key})
        root = materializer.TARGETS[key]
        self.assertEqual(str(root), EXPECTED_ROOT)
        self.assertEqual(materializer.PRODUCTION_RANK_CACHE_ROOT, EXPECTED_CACHE)
        self.assertEqual(static.PRODUCTION_RANK_CACHE_ROOT, EXPECTED_CACHE)
        plan = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        value = materializer.launch_input(root, *key, plan)
        self.assertEqual(value["schema_version"], launcher.INPUT_SCHEMA)
        self.assertEqual(value["rank_cache_root"], str(EXPECTED_CACHE))
        self.assertEqual(
            value["adapter"],
            str(
                root / "release/methods/bernini_action_editing/"
                "full644_exploratory_matched_infer_adapter_auh_r5f.py"
            ),
        )
        self.assertEqual(
            value["output_report"],
            str(root / "final/case00_canary_report_auh_r5d.json"),
        )
        self.assertEqual(
            value["runner_attestation"],
            str(root / "final/case00_canary_runner_attestation_auh_r5d.json"),
        )
        self.assertNotEqual(
            os.path.commonpath((value["rank_cache_root"], str(root))),
            str(root),
        )

    def test_static_requires_explicit_lexists_false_for_cache(self) -> None:
        root = materializer.TARGETS[("143812", "auh7-1b-gpu-293")]
        plan = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        value = materializer.launch_input(
            root, "143812", "auh7-1b-gpu-293", plan
        )
        environment = {
            "SLURM_JOB_ID": "143812",
            "SLURM_JOB_NODELIST": "auh7-1b-gpu-293",
        }
        with mock.patch.dict(os.environ, environment, clear=False), \
             mock.patch.object(static.os.path, "lexists", return_value=False) as lexists:
            static.validate_input(value, root, plan)
            lexists.assert_called_once_with(str(EXPECTED_CACHE))
        with mock.patch.dict(os.environ, environment, clear=False), \
             mock.patch.object(static.os.path, "lexists", return_value=True):
            with self.assertRaisesRegex(
                static.R5FStaticProbeError,
                "launch input semantics differ",
            ):
                static.validate_input(value, root, plan)

    def test_existing_target_refuses_before_source_read_or_creation(self) -> None:
        key = ("143812", "auh7-1b-gpu-293")
        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(materializer.os, "geteuid", return_value=2012), \
             mock.patch.object(materializer.os, "getegid", return_value=2000), \
             mock.patch.object(materializer, "stable_file") as stable, \
             mock.patch.object(materializer, "mkdir_fresh") as mkdir:
            with self.assertRaisesRegex(
                materializer.R5FMaterializationError,
                "fresh r5f root exists",
            ):
                materializer._materialize(
                    types.SimpleNamespace(
                        job_id=key[0],
                        node=key[1],
                        source_root=str(MODULE_ROOT.parents[1]),
                    )
                )
            stable.assert_not_called()
            mkdir.assert_not_called()

    def test_old_r5d_chain_is_unchanged(self) -> None:
        frozen = {
            "full644_exploratory_matched_infer_adapter_auh_r5d.py":
                "5794e1f0e5ecb84ffdb37f618fe63696ee4f87176952ac083c8c91792a9d192a",
            "full644_exploratory_matched_spooled_launcher_auh_r5d.py":
                "85ccc17b30d97a7bf048702cd8a8ed10c3421e01721902fea7db6242eac45753",
            "full644_exploratory_matched_r5d_static_nomodel_probe_v4.py":
                "887a810560e4f2ece560e4e1a766c88845cf623547ab9230879cd2a6839e693b",
            "full644_exploratory_matched_r5d_cpu_consumption_probe_v1.py":
                "5c7f5caf5ad73aecacedda618e941308e4fc1b94218b71cdc44e88afc3d3f0ea",
            "tools/materialize_full644_exploratory_matched_r5d_case00_package_v5.py":
                "994e463c55cd73d151707da762f04a6150a15631bc4bff90dfe71359b61ff82a",
        }
        for relative, digest in frozen.items():
            with self.subTest(relative=relative):
                self.assertEqual(sha(MODULE_ROOT / relative), digest)


if __name__ == "__main__":
    unittest.main()
