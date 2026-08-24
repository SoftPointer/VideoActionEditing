from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_self_generated_action_lora_heldout_core4_v1 as heldout


SPEC = METHOD_ROOT / "assets/self_generated_action_lora_heldout_core4_v1.json"
SPEC_SHA256 = "82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c"
CONTROLLER = (
    METHOD_ROOT
    / "scripts"
    / "auh_eval_self_generated_action_lora_heldout_pair_existing_job_v1.sh"
)
MULTINODE_CONTROLLER = (
    METHOD_ROOT
    / "scripts"
    / "auh_eval_self_generated_action_lora_heldout_pair_multinode_v1.sh"
)
RANK_CACHE_EXEC = METHOD_ROOT / "scripts" / "auh_heldout_rank_cache_exec_v1.sh"
FRESH256_ARRAY = (
    METHOD_ROOT
    / "scripts"
    / "auh_eval_seer_fresh256_core4_array_20260813.sbatch"
)


class HeldoutCore4ContractTests(unittest.TestCase):
    def test_fresh256_array_serializes_arms_and_seals_both_methods(self) -> None:
        text = FRESH256_ARRAY.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --mem=256G", text)
        self.assertIn("#SBATCH --gres=gpu:mi210:4", text)
        self.assertIn("#SBATCH --array=0-1%2", text)
        self.assertIn("#SBATCH --exclusive", text)
        self.assertIn("${SLURM_TMPDIR:-/tmp}/seer-eval-", text)
        self.assertNotIn('local cache_root="${case_root}/cache"', text)
        self.assertIn("constrained)", text)
        self.assertIn("direct)", text)
        self.assertNotIn(
            'run_arm frozen_base 29511 >"${case_root}/frozen-base.log" 2>&1 &',
            text,
        )
        self.assertLess(
            text.index("run_arm frozen_base 29511"),
            text.index("run_arm trained_adapter 29512"),
        )

    def test_static_two_node_torchrun_prefix_is_exact_world4(self) -> None:
        prefix = heldout.torchrun_prefix(
            python_bin=Path("/frozen/python"),
            nnodes=2,
            nproc_per_node=2,
            node_rank=1,
            master_addr="auh7-1b-gpu-209",
            master_port=29441,
        )
        self.assertEqual(
            prefix,
            [
                "/frozen/python",
                "-m",
                "torch.distributed.run",
                "--nnodes=2",
                "--nproc_per_node=2",
                "--node_rank=1",
                "--master_addr=auh7-1b-gpu-209",
                "--master_port=29441",
            ],
        )
        with self.assertRaises(heldout.HeldoutEvalError):
            heldout.torchrun_prefix(
                python_bin=Path("/frozen/python"),
                nnodes=2,
                nproc_per_node=1,
                node_rank=0,
                master_addr="auh7-1b-gpu-209",
                master_port=29441,
            )

    def test_multinode_controller_splits_world4_and_host_memory(self) -> None:
        text = MULTINODE_CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("--mem=56G --gres=gpu:mi210:2", text)
        self.assertIn("--torchrun-nnodes 2 --torchrun-nproc-per-node 2", text)
        self.assertIn('--torchrun-node-rank "${node_rank}"', text)
        self.assertIn("NCCL_SOCKET_IFNAME=bond0", text)
        self.assertIn("NCCL_IB_DISABLE=1", text)
        self.assertLess(
            text.index("run_arm_pair frozen_base 29441"),
            text.index("run_arm_pair trained_adapter 29442"),
        )
        self.assertIn("wait_pair", text)
        self.assertIn("--torchrun-worker-prefix", text)
        self.assertIn("BERNINI_HELDOUT_RANK_CACHE_TOKEN", text)
        self.assertNotIn("BERNINI_HELDOUT_RANK_CACHE_PARENT", text)
        self.assertNotIn('cache_root="${output_root}/cache"', text)

    def test_rank_cache_worker_is_local_rank_scoped(self) -> None:
        text = RANK_CACHE_EXEC.read_text(encoding="utf-8")
        self.assertIn('local_rank="${LOCAL_RANK:', text)
        self.assertIn('global_rank="${RANK:', text)
        self.assertIn('world_size="${WORLD_SIZE:', text)
        self.assertIn('scratch_parent="${SLURM_TMPDIR:-/tmp}"', text)
        self.assertIn('rank_root="$(mktemp -d --', text)
        self.assertIn("shared filesystem forbidden for rank cache", text)
        self.assertIn('"${python_bin}" -B "$@" &', text)
        self.assertIn('find "${rank_root}" -xdev -depth -mindepth 1 -delete', text)
        for name in (
            "HOME",
            "TMPDIR",
            "XDG_CACHE_HOME",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "PYTHONPYCACHEPREFIX",
            "MIOPEN_USER_DB_PATH",
            "MIOPEN_CUSTOM_CACHE_DIR",
        ):
            self.assertIn(f"export {name}=", text)

    def test_worker_prefix_selects_torchrun_no_python(self) -> None:
        prefix = heldout.torchrun_prefix(
            python_bin=Path("/frozen/python"),
            nnodes=2,
            nproc_per_node=2,
            node_rank=0,
            master_addr="auh7-1b-gpu-209",
            master_port=29441,
            no_python=True,
        )
        self.assertEqual(prefix[-1], "--no-python")

    def test_controller_binds_arm_before_cache_expansion(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertNotIn(
            'local arm="$1" port="$2" cache="${cache_root}/${arm}"', text
        )
        self.assertIn('local arm="$1" port="$2" cache', text)
        self.assertIn('cache="${cache_root}/${arm}"', text)

    def test_controller_delays_sealed_directory_mode_restore(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("tar --delay-directory-restore", text)
        self.assertIn("--no-same-owner --no-same-permissions", text)

    def test_controller_runs_memory_safe_sequential_arms(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn('worker_mem="${BERNINI_HELDOUT_WORKER_MEM:-56G}"', text)
        self.assertIn("--gres=gpu:mi210:4", text)
        self.assertNotIn("--gpus-per-task=4", text)
        self.assertNotIn('frozen-base.log" 2>&1 &', text)
        self.assertNotIn('trained-adapter.log" 2>&1 &', text)
        self.assertLess(
            text.index("run_arm frozen_base 29431"),
            text.index("run_arm trained_adapter 29432"),
        )

    def test_trained_arm_admits_both_sealed_seer_receipt_readers(self) -> None:
        self.assertEqual(
            heldout.inference_runner_name(
                arm="trained_adapter",
                trained_runner=heldout.DEFAULT_TRAINED_INFER_RUNNER,
            ),
            "infer_seer_scoped_lora.py",
        )
        self.assertEqual(
            heldout.inference_runner_name(
                arm="frozen_base",
                trained_runner="infer_seer_same_state_lora.py",
            ),
            "infer_seer_scoped_lora.py",
        )
        self.assertEqual(
            heldout.inference_runner_name(
                arm="trained_adapter",
                trained_runner="infer_seer_same_state_lora.py",
            ),
            "infer_seer_same_state_lora.py",
        )
        self.assertEqual(
            heldout.inference_runner_name(
                arm="trained_adapter",
                trained_runner="infer_seer_same_state_full160_lora.py",
            ),
            "infer_seer_same_state_full160_lora.py",
        )
        self.assertEqual(
            heldout.inference_runner_name(
                arm="frozen_base",
                trained_runner="infer_seer_same_state_full160_lora.py",
            ),
            "infer_seer_scoped_lora.py",
        )
        with self.assertRaises(heldout.HeldoutEvalError):
            heldout.inference_runner_name(
                arm="trained_adapter", trained_runner="infer_lora.py"
            )

    def test_sealed_spec_is_two_dog_two_human_and_owner_disjoint(self) -> None:
        spec, digest = heldout.load_spec(SPEC, SPEC_SHA256)
        self.assertEqual(digest, SPEC_SHA256)
        self.assertEqual(
            [row["actor_family"] for row in spec["cases"]],
            ["dog", "dog", "human", "human"],
        )
        self.assertTrue(
            {row["iid"] for row in spec["cases"]}.isdisjoint(
                spec["training_exclusion"]["owner_fit_iids"]
            )
        )
        self.assertFalse(
            spec["decision_contract"]["training_completion_is_success"]
        )
        self.assertEqual(
            spec["inference_contract"]["runner"],
            "methods/bernini_action_editing/infer_seer_scoped_lora.py",
        )

    def test_spec_rejects_owner_fit_leakage(self) -> None:
        value = json.loads(SPEC.read_text(encoding="utf-8"))
        value["cases"][0]["iid"] = "7b88a1ca1f804f41"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(heldout.HeldoutEvalError, "exclusion"):
                heldout.load_spec(path, digest)

    def test_receipt_verifier_rejects_coordinate_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source.mp4"
            source.write_bytes(b"source")
            output = root / "frozen_base.mp4"
            output.write_bytes(b"output")
            case = {
                "iid": "0123456789abcdef",
                "source_video": str(source),
                "source_video_sha256": heldout.file_sha256(source),
                "instruction_utf8_sha256": hashlib.sha256(b"instruction").hexdigest(),
                "seed": 17,
            }
            receipt_path = root / "frozen_base.mp4.receipt.json"
            unsigned = {
                "schema_version": heldout.INFERENCE_RECEIPT_SCHEMA,
                "method_source_revision": "1" * 40,
                "method_source_archive_sha256": "2" * 64,
                "bernini_commit": heldout.BERNINI_COMMIT,
                "veomni_commit": heldout.VEOMNI_COMMIT,
                "checkpoint_tree_sha256": heldout.CHECKPOINT_TREE_SHA256,
                "adapter": {
                    "enabled": False,
                    "mode": "frozen_base_no_adapter",
                    "strictly_reloaded": False,
                    "safe_merged_for_inference": False,
                    "tensor_count": 0,
                },
                "input": {
                    "source_video_path": str(source),
                    "source_video_sha256": heldout.file_sha256(source),
                    "instruction_utf8_sha256": case["instruction_utf8_sha256"],
                    "accepted_model_conditions": ["source_video", "edit_instruction"],
                    "target_accessed_by_inference": False,
                },
                "preprocessing": {
                    "frame_count": 81,
                    "fps": 25.0,
                    "reported_fps": 25.0,
                    "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
                    "external_shared_i0": False,
                },
                "prompt_contract": {},
                "sampling": {
                    "seed": 17,
                    "num_frames": 81,
                    "num_inference_steps": 40,
                    "guidance_mode": "v2v_apg",
                    "ulysses_size": 4,
                },
                "output": {
                    "path": str(output),
                    "sha256": heldout.file_sha256(output),
                    "frame_count": 81,
                    "fps": 25.0,
                },
            }
            receipt = {**unsigned, "receipt_digest": heldout.object_sha256(unsigned)}
            receipt_path.write_bytes(heldout.canonical_json_bytes(receipt) + b"\n")
            checked = heldout._verify_inference_receipt(
                receipt_path, case=case, arm="frozen_base", adapter_checkpoint=None
            )
            self.assertEqual(checked["sampling"]["seed"], 17)

            receipt["sampling"]["seed"] = 18
            without_digest = dict(receipt)
            without_digest.pop("receipt_digest")
            receipt["receipt_digest"] = heldout.object_sha256(without_digest)
            receipt_path.write_bytes(heldout.canonical_json_bytes(receipt) + b"\n")
            with self.assertRaisesRegex(heldout.HeldoutEvalError, "contract differs"):
                heldout._verify_inference_receipt(
                    receipt_path, case=case, arm="frozen_base", adapter_checkpoint=None
                )


if __name__ == "__main__":
    unittest.main()
