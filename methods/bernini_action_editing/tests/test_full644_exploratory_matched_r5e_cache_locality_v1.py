from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = MODULE_ROOT / "tools"
for entry in (MODULE_ROOT, TOOLS_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import full644_exploratory_matched_r5d_static_nomodel_probe_v4 as static
import materialize_full644_exploratory_matched_r5d_case00_package_v5 as materializer


EXPECTED_CACHE = Path(
    "/tmp/bernini-full644-r5e-job143812-node293-r5-rank-cache"
)


class R5ECacheLocalityTests(unittest.TestCase):
    def test_production_cache_is_exact_node_local_path(self) -> None:
        root = materializer.TARGETS[("143812", "auh7-1b-gpu-293")]
        plan = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        value = materializer.launch_input(
            root, "143812", "auh7-1b-gpu-293", plan
        )
        self.assertEqual(materializer.PRODUCTION_RANK_CACHE_ROOT, EXPECTED_CACHE)
        self.assertEqual(static.PRODUCTION_RANK_CACHE_ROOT, EXPECTED_CACHE)
        self.assertEqual(value["rank_cache_root"], str(EXPECTED_CACHE))
        self.assertEqual(value["authority_root"], str(root / "runtime/model-authority"))
        self.assertEqual(
            value["output_report"],
            str(root / "final/case00_canary_report_auh_r5d.json"),
        )
        self.assertNotEqual(
            os.path.commonpath((value["rank_cache_root"], str(root))),
            str(root),
        )

    def test_static_gate_accepts_local_cache_and_rejects_vast_cache(self) -> None:
        root = materializer.TARGETS[("143812", "auh7-1b-gpu-293")]
        plan = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        value = materializer.launch_input(
            root, "143812", "auh7-1b-gpu-293", plan
        )
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "143812",
                "SLURM_JOB_NODELIST": "auh7-1b-gpu-293",
            },
            clear=False,
        ):
            static.validate_input(value, root, plan)
            for hostile_cache in (
                str(root / "runtime/rank-cache"),
                "/tmp/bernini-full644-r5e-job143812-node293-r6-rank-cache",
                "/var/tmp/bernini-full644-r5e-job143812-node293-r5-rank-cache",
                "tmp/bernini-full644-r5e-job143812-node293-r5-rank-cache",
            ):
                with self.subTest(hostile_cache=hostile_cache):
                    hostile = dict(value)
                    hostile["rank_cache_root"] = hostile_cache
                    with self.assertRaisesRegex(
                        static.R5DStaticProbeError,
                        "launch input semantics differ",
                    ):
                        static.validate_input(hostile, root, plan)

    def test_fresh_static_probe_pin_is_exact(self) -> None:
        source = MODULE_ROOT / Path(materializer.STATIC_NOMODEL_PROBE).name
        observed = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(
            observed,
            materializer.DIAGNOSTIC_SOURCE_PINS[
                materializer.STATIC_NOMODEL_PROBE
            ],
        )
        materializer.ensure_ready_pins()
        self.assertEqual(materializer.RELEASE_FILES, static.RELEASE_FILES)


if __name__ == "__main__":
    unittest.main()
