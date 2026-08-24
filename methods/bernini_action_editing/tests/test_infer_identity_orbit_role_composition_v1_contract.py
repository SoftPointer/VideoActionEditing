from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER = METHOD_ROOT / "infer_identity_orbit_role_composition_v1.py"
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_infer_identity_orbit_role_composition_dual4.sbatch"
)
SPEC = (
    METHOD_ROOT
    / "assets"
    / "identity_orbit_heldout_role_composition_core2_v1.json"
)


class IdentityOrbitRoleCompositionContractTests(unittest.TestCase):
    @staticmethod
    def _audit_helpers() -> Mapping[str, Any]:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
        wanted = {
            "RoleCompositionError",
            "canonical_json_bytes",
            "object_sha256",
            "_rank_invariant_route_payload",
        }
        nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
        ]
        namespace: dict[str, Any] = {
            "Any": Any,
            "Mapping": Mapping,
            "hashlib": hashlib,
            "json": json,
            "SP_SIZE": 4,
            "ARM_ORDER": ("base", "orbit-adapter"),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), namespace)
        return namespace

    @staticmethod
    def _route_traces(rank: int, object_sha256: Any) -> Mapping[str, Any]:
        traces: dict[str, Any] = {}
        for arm, enabled in (("base", False), ("orbit-adapter", True)):
            routes = []
            for branch in ("none", "V", "I", "VI"):
                payload = {
                    "branch_name": branch,
                    "total_tokens": 42780,
                    "condition_tokens": 23250,
                    "target_tokens": 19530,
                    "sequence_parallel_rank": rank,
                    "sequence_parallel_size": 4,
                    "padding_policy": "append_false_then_contiguous_rank_chunk",
                    "enabled": enabled,
                }
                routes.append({**payload, "digest": object_sha256(payload)})
            step_records = [
                {
                    "step_index": 0,
                    "patch_source_ids": [0.0, 1.0, 2.0, 3.0, 4.0],
                    "routes": routes,
                    "adapter_enabled": enabled,
                    "scheduler_original_return_forwarded": True,
                }
            ]
            traces[arm] = {
                "sample_calls": 1,
                "step_count": 1,
                "guidance_forward_count": 4,
                "step_records": step_records,
                "step_records_digest": object_sha256(step_records),
            }
        return traces

    def test_runner_is_syntactically_valid_and_declares_paired_exact_route(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        ast.parse(source, filename=str(RUNNER))
        for fragment in (
            'ARM_ORDER = ("base", "orbit-adapter")',
            'FRAME_COUNT = 81',
            'REFERENCE_INDICES = (0, 27, 53, 80)',
            '"adapter_active_on_every_native_coordinate"',
            '"adapter_explicitly_disabled_on_every_native_coordinate"',
            '"same_official_gaussian_base_and_adapter": True',
            '"same_donor_refs_text_scheduler_guidance_base_and_adapter": True',
            '"rho": 0.0',
            '"older_calibration_bank_consumed": False',
            '"generated_inside_same_invocation": True',
            '"condition_only": True',
            '"rv2v_target": False',
            '"rv2v_noise": False',
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("pair_v5_t2v_calibration_core4_bank", source)
        self.assertNotIn("t2v.normalized-clean-latent.safetensors", source)

    def test_layout_formula_covers_both_preregistered_buckets(self) -> None:
        def layout(height: int, width: int) -> tuple[int, int, list[int]]:
            target = 21 * (height // 2) * (width // 2)
            reference = (height // 2) * (width // 2)
            return target, reference, [
                target,
                2 * target,
                2 * target + 4 * reference,
                2 * target + 4 * reference,
            ]

        self.assertEqual(layout(60, 62), (19530, 930, [19530, 39060, 42780, 42780]))
        self.assertEqual(layout(64, 58), (19488, 928, [19488, 38976, 42688, 42688]))

    def test_route_audit_normalizes_only_verified_rank_local_receipts(self) -> None:
        helpers = self._audit_helpers()
        normalize = helpers["_rank_invariant_route_payload"]
        object_sha256 = helpers["object_sha256"]
        error = helpers["RoleCompositionError"]

        traces = [self._route_traces(rank, object_sha256) for rank in range(4)]
        shared_digests = {
            object_sha256(normalize(trace, expected_rank=rank))
            for rank, trace in enumerate(traces)
        }
        local_digests = {object_sha256(trace) for trace in traces}
        self.assertEqual(len(shared_digests), 1)
        self.assertEqual(len(local_digests), 4)

        wrong_rank = self._route_traces(1, object_sha256)
        with self.assertRaisesRegex(error, "expected SP rank"):
            normalize(wrong_rank, expected_rank=0)

        bad_digest = self._route_traces(0, object_sha256)
        bad_digest["base"]["step_records"][0]["routes"][0]["digest"] = "0" * 64
        bad_digest["base"]["step_records_digest"] = object_sha256(
            bad_digest["base"]["step_records"]
        )
        with self.assertRaisesRegex(error, "route receipt digest"):
            normalize(bad_digest, expected_rank=0)

    def test_spec_is_closed_fresh_and_caption_hashes_are_exact(self) -> None:
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        self.assertEqual(
            spec["schema_version"],
            "bernini-identity-orbit-heldout-role-composition-core2-spec-v1",
        )
        self.assertEqual([row["cell_id"] for row in spec["cells"]], ["dog-fit", "human-fit"])
        self.assertEqual([row["actor_kind"] for row in spec["cells"]], ["dog", "human"])
        self.assertEqual(spec["contract"]["frame_count"], 81)
        self.assertEqual(spec["contract"]["num_inference_steps"], 40)
        self.assertEqual(spec["contract"]["reference_indices"], [0, 27, 53, 80])
        self.assertEqual(spec["contract"]["source_rich_noise_rho"], 0.0)
        self.assertFalse(spec["contract"]["external_donor_media"])
        self.assertFalse(spec["contract"]["cell_selection_uses_generated_quality"])
        for row in spec["cells"]:
            self.assertFalse(row["identity_orbit_training_iid_overlap"])
            self.assertEqual(
                row["donor_policy"],
                "fresh_frozen_t2v_in_same_invocation_condition_only",
            )
            self.assertNotEqual(row["donor_seed"], row["target_seed"])
            self.assertEqual(
                hashlib.sha256(row["action_caption"].encode("utf-8")).hexdigest(),
                row["action_caption_utf8_sha256"],
            )

    def test_launcher_uses_all_eight_as_two_isolated_world4_exact40_groups(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --qos=gtqos",
            "--nproc_per_node=4",
            "launch_group dog dog-fit 0,1,2,3",
            "launch_group human human-fit 4,5,6,7",
            "--num-inference-steps 40",
            'routes.get("base", {}).get("step_count") == 40',
            'routes.get("base", {}).get("guidance_forward_count") == 160',
            'routes.get("orbit-adapter", {}).get("step_count") == 40',
            'routes.get("orbit-adapter", {}).get("guidance_forward_count") == 160',
            'set(outputs) != {"t2v-donor", "base", "orbit-adapter"}',
            'output.get("frame_count") != 81',
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("sbatch ", source)


if __name__ == "__main__":
    unittest.main()
