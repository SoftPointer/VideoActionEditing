from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # local contract environment may intentionally omit torch
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_t2v_factorial_bank as bank  # noqa: E402
import infer_fitq_owner_prompt_cross_query_micro as runtime  # noqa: E402


MICRO_SPEC = METHOD_ROOT / "assets/dmiq_cdf_dog_t2v_micro_spec_v2.json"
REVISION = "2" * 40
ARCHIVE_SHA256 = "3" * 64


def _manifest() -> dict:
    spec = json.loads(MICRO_SPEC.read_text(encoding="utf-8"))
    return bank.build_manifest(
        spec,
        method_source_revision=REVISION,
        method_source_archive_sha256=ARCHIVE_SHA256,
        attempt_rung=0,
    )


def _seal_receipt(receipt: dict) -> dict:
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = runtime.object_sha256(receipt)
    return receipt


def _bank_receipt(manifest: dict) -> dict:
    entries = []
    for index, entry in enumerate(manifest["entries"]):
        raw_noise = (
            "a" * 64
            if entry["proposal_cell_index"] == 0
            else "b" * 64
        )
        entries.append(
            {
                "entry_id": entry["entry_id"],
                "semantic_branch": entry["semantic_branch"],
                "proposal_cell_id": entry["proposal_cell_id"],
                "design_slot_id": entry["design_slot_id"],
                "analysis_split": entry["analysis_split"],
                "execution_group": entry["execution_group"],
                "seed_replicate_id": entry["seed_replicate_id"],
                "seed": entry["seed"],
                "attempt_rung": entry["attempt_rung"],
                "native_receipt_path": f"/bank/{entry['entry_id']}/receipt.json",
                "native_receipt_file_sha256": f"{index + 1:064x}",
                "native_receipt_digest": f"{index + 21:064x}",
                "video_path": f"/bank/{entry['entry_id']}/t2v.mp4",
                "video_sha256": f"{index + 41:064x}",
                "clean_latent_path": (
                    f"/bank/{entry['entry_id']}/"
                    "t2v.normalized-clean-latent.safetensors"
                ),
                "clean_latent_sha256": f"{index + 61:064x}",
                "initial_noise_path": (
                    f"/bank/{entry['entry_id']}/t2v.initial-noise.safetensors"
                ),
                "initial_noise_file_sha256": f"{index + 81:064x}",
                "initial_noise_tensor_value_sha256": raw_noise,
                "initial_noise_value_digest_independently_recomputed": True,
                "method_source_revision": REVISION,
                "method_source_archive_sha256": ARCHIVE_SHA256,
                "pure_t2v_condition_audit_pass": True,
            }
        )
    receipt = {
        "schema_version": bank.BANK_RECEIPT_SCHEMA,
        "bank_id": manifest["bank_id"],
        "profile": "engineering_micro",
        "attempt_rung": 0,
        "manifest_digest": manifest["manifest_digest"],
        "entry_count": 20,
        "proposal_cell_count": 2,
        "entries": entries,
        "native_method_provenance": {
            "method_source_revision": REVISION,
            "method_source_archive_sha256": ARCHIVE_SHA256,
            "preregistered_in_manifest_before_render": True,
            "all_entries_exact": True,
        },
        "condition_closure": {
            "renderer_arm": "t2v",
            "source_video_role": (
                "exact81_bucket_selection_and_hash_verification_only"
            ),
            "source_latent_or_reference_consumed": False,
            "target_video_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "all_native_entry_audits_pass": True,
            "all_cells_share_exact_initial_noise_across_ten_branches": True,
            "all_initial_noise_value_digests_independently_recomputed": True,
        },
        "interpretation": {
            "factorial_render_complete": True,
            "optimizer_update": "null",
            "training_performed": False,
        },
    }
    return _seal_receipt(receipt)


class _FakeScheduler:
    def step(self, *args, **kwargs):
        return args, kwargs


class _FakeDiffusion:
    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()

    def sample(self, *args, **kwargs):
        return args, kwargs

    def shared_step(self, *args, **kwargs):
        return args, kwargs


class OwnerPromptCrossQueryMicroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _manifest()
        cls.receipt = _bank_receipt(cls.manifest)
        cls.source_path = METHOD_ROOT / runtime.__file__.split("/")[-1]
        cls.source = cls.source_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_fixed_native_schedule_and_forward_counts(self) -> None:
        self.assertEqual(runtime.OWNER_BRANCH_ORDER, (
            "full_action", "noop", "reverse_action", "camera_only"
        ))
        self.assertEqual(runtime.PROMPT_BRANCH_ORDER, bank.BRANCH_ORDER)
        self.assertEqual(runtime.SELECTED_SCHEDULE_INDICES, (22, 31, 36, 39))
        self.assertEqual(
            [row["timestep"] for row in runtime.selected_schedule_contract()],
            [803, 593, 359, 117],
        )
        self.assertEqual(
            [row["sigma_float32_be_hex"] for row in runtime.selected_schedule_contract()],
            ["3f4dcdd4", "3f17da71", "3eb80796", "3df0f309"],
        )
        self.assertEqual(runtime.EXPECTED_OFFICIAL_FORWARDS_PER_OWNER, 80)
        self.assertEqual(runtime.EXPECTED_EXTRA_FORWARDS_PER_OWNER, 40)
        self.assertEqual(runtime.EXPECTED_TOTAL_FORWARDS_PER_OWNER, 120)
        self.assertEqual(4 * runtime.EXPECTED_TOTAL_FORWARDS_PER_OWNER, 480)

    @unittest.skipUnless(torch is not None, "PyTorch regression runs in AUH vace")
    def test_tensor_identity_materializes_stride_zero_logical_digest_only(self) -> None:
        assert torch is not None
        scalar = torch.tensor(803, dtype=torch.int64)
        expanded_one = scalar.expand(1)
        expanded_many = scalar.expand(5)
        for name, expanded in (
            ("expanded_one", expanded_one),
            ("expanded_many", expanded_many),
        ):
            original_object = expanded
            original_shape = tuple(expanded.shape)
            original_stride = tuple(expanded.stride())
            original_offset = expanded.storage_offset()
            self.assertEqual(original_stride, (0,))

            identity = runtime._tensor_identity(expanded, label=name)
            contiguous_identity = runtime._tensor_identity(
                expanded.clone(memory_format=torch.contiguous_format),
                label=f"{name}_contiguous",
            )

            self.assertIs(expanded, original_object)
            self.assertEqual(tuple(expanded.shape), original_shape)
            self.assertEqual(tuple(expanded.stride()), original_stride)
            self.assertEqual(expanded.storage_offset(), original_offset)
            self.assertTrue(runtime._same_raw_tensor(identity, contiguous_identity))
            self.assertEqual(identity["original_tensor"]["shape"], list(original_shape))
            self.assertEqual(identity["original_tensor"]["stride"], [0])
            self.assertEqual(
                identity["original_tensor"]["storage_offset"], original_offset
            )
            self.assertEqual(identity["original_tensor"]["layout"], "torch.strided")
            self.assertEqual(
                identity["logical_value_digest_source"],
                "detached_cpu_contiguous_clone",
            )
            self.assertTrue(identity["original_tensor_object_preserved"])
            self.assertFalse(identity["python_object_id_or_data_ptr_serialized"])
            serialized = json.dumps(identity, sort_keys=True)
            self.assertNotIn('"data_ptr":', serialized)
            self.assertNotIn('"python_object_id":', serialized)

        dtype_cases = (
            torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
            torch.tensor([1.0 + 2.0j, -3.0 + 0.5j], dtype=torch.complex64).conj(),
        )
        for index, value in enumerate(dtype_cases):
            identity = runtime._tensor_identity(value, label=f"dtype_case_{index}")
            clone_identity = runtime._tensor_identity(
                value.resolve_conj().clone(memory_format=torch.contiguous_format),
                label=f"dtype_case_{index}_clone",
            )
            self.assertTrue(runtime._same_raw_tensor(identity, clone_identity))
            self.assertEqual(identity["dtype"], str(value.dtype))
            self.assertEqual(identity["byte_count"], value.numel() * value.element_size())

    def test_each_sp4_binds_one_full_cell_and_fixed_owners(self) -> None:
        for group in bank.GROUPS:
            bound = runtime.validate_micro_bank_bindings(
                self.manifest, self.receipt, execution_group=group
            )
            self.assertEqual(len(bound["prompt_rows"]), 10)
            self.assertEqual(
                tuple(row["manifest"]["semantic_branch"] for row in bound["prompt_rows"]),
                bank.BRANCH_ORDER,
            )
            self.assertEqual(
                tuple(row["manifest"]["semantic_branch"] for row in bound["owner_rows"]),
                runtime.OWNER_BRANCH_ORDER,
            )
            self.assertEqual(
                len({row["bank"]["initial_noise_tensor_value_sha256"] for row in bound["prompt_rows"]}),
                1,
            )

    def test_receipt_tampering_and_cross_cell_noise_fail_closed(self) -> None:
        corrupted = deepcopy(self.receipt)
        corrupted["condition_closure"][
            "all_initial_noise_value_digests_independently_recomputed"
        ] = False
        _seal_receipt(corrupted)
        with self.assertRaises(runtime.OwnerPromptCrossQueryError):
            runtime.validate_micro_bank_bindings(
                self.manifest, corrupted, execution_group="sp4-a"
            )

        changed_noise = deepcopy(self.receipt)
        changed_noise["entries"][1]["initial_noise_tensor_value_sha256"] = "c" * 64
        _seal_receipt(changed_noise)
        with self.assertRaises(runtime.OwnerPromptCrossQueryError):
            runtime.validate_micro_bank_bindings(
                self.manifest, changed_noise, execution_group="sp4-a"
            )

        embedded_digest_tamper = deepcopy(self.receipt)
        embedded_digest_tamper["entries"][0]["seed"] += 1
        with self.assertRaises(runtime.OwnerPromptCrossQueryError):
            runtime.validate_micro_bank_bindings(
                self.manifest, embedded_digest_tamper, execution_group="sp4-a"
            )

    def test_full_cartesian_query_order_is_owner_state_prompt_major(self) -> None:
        bound = runtime.validate_micro_bank_bindings(
            self.manifest, self.receipt, execution_group="sp4-a"
        )
        expected = runtime.expected_cartesian_query_order(bound["owner_rows"])
        self.assertEqual(len(expected), 160)
        records = []
        for owner in bound["owner_rows"]:
            entry = owner["manifest"]
            records.append(
                {
                    "owner_id": entry["entry_id"],
                    "owner_branch": entry["semantic_branch"],
                    "selected_states": [
                        {
                            "schedule_index": index,
                            "prompt_order": list(bank.BRANCH_ORDER),
                        }
                        for index in runtime.SELECTED_SCHEDULE_INDICES
                    ],
                }
            )
        self.assertEqual(runtime.observed_cartesian_query_order(records), expected)
        records[-1]["selected_states"][-1]["prompt_order"].pop()
        self.assertNotEqual(runtime.observed_cartesian_query_order(records), expected)

    def test_wrapper_is_reversible_and_rejects_stacking(self) -> None:
        diffusion = _FakeDiffusion()
        prompts = {branch: object() for branch in bank.BRANCH_ORDER}
        lengths = {branch: 1 for branch in bank.BRANCH_ORDER}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runtime, "_nested_value_identity", return_value={"kind": "fake"}
        ):
            bridge = runtime.OwnerPromptCrossQueryBridge(
                diffusion,
                prompt_embeds=prompts,
                prompt_lengths=lengths,
                statistics_dir=Path(directory),
                distributed_rank=0,
            )
            bridge.install()
            self.assertIn("sample", vars(diffusion))
            self.assertIn("shared_step", vars(diffusion))
            self.assertIn("step", vars(diffusion.scheduler))
            with self.assertRaises(runtime.OwnerPromptCrossQueryError):
                bridge.install()
            bridge.restore()
            self.assertNotIn("sample", vars(diffusion))
            self.assertNotIn("shared_step", vars(diffusion))
            self.assertNotIn("step", vars(diffusion.scheduler))

    def test_parser_has_no_privileged_condition_or_training_inputs(self) -> None:
        destinations = {action.dest for action in runtime.build_parser()._actions}
        forbidden = {
            "target", "target_video", "mask", "flow", "pose", "track",
            "trajectory", "reference", "reference_video", "first_frame",
            "source_latent", "initial_noise", "optimizer", "learning_rate",
            "adapter", "lora", "resume",
        }
        self.assertTrue(destinations.isdisjoint(forbidden))

    def test_runtime_calls_no_backward_optimizer_or_weight_save(self) -> None:
        terminal_names = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute):
                terminal_names.append(function.attr)
            elif isinstance(function, ast.Name):
                terminal_names.append(function.id)
        self.assertNotIn("backward", terminal_names)
        self.assertNotIn("step_optimizer", terminal_names)
        self.assertNotIn("save_pretrained", terminal_names)
        self.assertIn(
            "result = self._original_scheduler_step(*args, **kwargs)", self.source
        )
        self.assertIn("return official_owner", self.source)
        self.assertIn('"training_authorized": False', self.source)
        self.assertIn('"scientific_claim_authorized": False', self.source)
        self.assertIn('"leave_one_owner_out_evaluated": False', self.source)


if __name__ == "__main__":
    unittest.main()
