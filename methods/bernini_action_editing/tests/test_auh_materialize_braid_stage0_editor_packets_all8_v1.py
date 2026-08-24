#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_materialize_braid_stage0_editor_packets_all8_v1.sbatch"
)

import braid_stage0_editor_packet_rank_guard_v1 as guard


class BraidStage0EditorPacketAll8LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")
        cls.guard_text = Path(guard.__file__).read_text(encoding="utf-8")
        cls.guard_sha256 = hashlib.sha256(
            Path(guard.__file__).read_bytes()
        ).hexdigest()
        cls.materializer_sha256 = "0" * 64

    def _environment(self, *, cell: str, rank: int) -> dict[str, str]:
        devices = "0,1,2,3" if cell == "dog" else "4,5,6,7"
        return {
            "ROCR_VISIBLE_DEVICES": devices,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "4",
        }

    def test_shell_is_one_all8_node_with_two_concurrent_world4_groups(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertNotRegex(self.text, r"(?m)^#SBATCH\s+--qos(?:=|\s)")
        self.assertIn(
            'launch_cell dog "${dog_seed}" "${dog_editor_seed}" 0,1,2,3',
            self.text,
        )
        self.assertIn(
            'launch_cell human "${human_seed}" "${human_editor_seed}" 4,5,6,7',
            self.text,
        )
        self.assertIn("& dog_pid=$!", self.text)
        self.assertIn("& human_pid=$!", self.text)
        self.assertEqual(self.text.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.text)
        self.assertIn('ROCR_VISIBLE_DEVICES="${devices}"', self.text)
        self.assertIn(
            "env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL",
            self.text,
        )

    def test_cells_owner_seeds_and_editor_seed_domains_are_fixed(self) -> None:
        for value in (
            "readonly dog_seed=2026081502",
            "readonly human_seed=2026081505",
            "readonly dog_editor_seed=2026082502",
            "readonly human_editor_seed=2026082505",
            '[[ "${editor_seed}" == "$((seed + 1000))" ]]',
            "BRAID_STAGE0_PACKET_ACK_FIXED_DOG_HUMAN_SEEDS",
        ):
            self.assertIn(value, self.text)
        self.assertNotRegex(self.text, r"for\s+(?:seed|query_seed)\s+in")
        self.assertNotRegex(self.text, r"(?:best|select|rerank).*seed")

    def test_existing_qmosaic_materializer_owns_the_only_gpu_phase(self) -> None:
        self.assertIn(
            'materializer="${method_root}/materialize_qmosaic_editor_runtime_v1.py"',
            self.text,
        )
        self.assertIn('--materializer "${materializer}"', self.text)
        self.assertIn('--output-dir "${packet_root}"', self.text)
        self.assertEqual(self.text.count('run_pair "${base_port}"'), 1)
        for forbidden in (
            "run_qmosaic_abba_parity_diagnostic_sp4_v1.py",
            "run_qmosaic_editor_direction_sp4_v1.py",
            "direction_runner=",
            "--no-lora-vjp",
            "--diagnostic-only",
            ".backward(",
            "optimizer.step(",
            "_vae_decode",
            "save_output(",
        ):
            self.assertNotIn(forbidden, self.text)
            self.assertNotIn(forbidden, self.guard_text)

    def test_owner_quotient_editor_and_checkpoint_authorities_are_literal_pins(self) -> None:
        for value in (
            "registry_sha256=01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96",
            "checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "owner_master_sha256=c0d8f3e4a7f3b95269b5196c0d8844327d9e7296dda1828493683a9ae7d707de",
            "owner_sidecar_sha256=24746c91e88e4051c49fe18b06e0e58bb2c4b119b3d946586d9dd6092308030b",
            "owner_evidence_sha256=3e2335d4d335a9ee8262aa319fc2790dbac3e59b20e554f54b4dc1273f259dc3",
            "owner_public_key_sha256=d1bba83ca1d162128bda71e21c419c476b9328c7892bd1998adcd24c09c577ec",
            "dog_quotient_receipt_sha256=5630c0f511360a6ae0386855f4c00e78e226fea32f71d340773db83ab5c49bd2",
            "human_quotient_receipt_sha256=fb6a37464e98841fe340e5a1411dffe8135640410fd0cef5c1f89b86fe81184e",
            "editor_public_key_sha256=b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d",
            "pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "pinned_materializer_source_sha256=9c7121f5f2de4ac048b1a5ea1c560ef58dde4c8c77a670380acad9357d451a52",
            f"pinned_rank_guard_source_sha256={self.guard_sha256}",
        ):
            self.assertIn(value, self.text)
        for label in (
            "checkpoint_manifest",
            "owner_master",
            "owner_sidecar",
            "owner_evidence",
            "owner_public_key",
            "dog_quotient_receipt",
            "human_quotient_receipt",
            "editor_public_key",
        ):
            self.assertRegex(
                self.text,
                rf'check_hash "\$\{{{label}\}}" "\$\{{{label}_sha256\}}"',
            )
        self.assertIn("editor authority private/public keys do not match", self.text)

    def test_rank_guard_observes_real_environment_before_materializer_exec(self) -> None:
        dog = guard.validate_live_environment(
            cell_id="dog",
            expected_rocr_visible_devices="0,1,2,3",
            expected_guard_source_sha256=self.guard_sha256,
            expected_materializer_source_sha256=self.materializer_sha256,
            environment=self._environment(cell="dog", rank=2),
            imported_modules={},
        )
        self.assertEqual(dog["rank"], 2)
        self.assertEqual(dog["physical_visible_devices"], [0, 1, 2, 3])
        self.assertTrue(dog["observed_before_torch_import"])
        self.assertFalse(dog["decode_backward_optimizer_update_authority"])
        guard.validate_live_environment_receipt(
            dog,
            cell_id="dog",
            rank=2,
            expected_guard_source_sha256=self.guard_sha256,
            expected_materializer_source_sha256=self.materializer_sha256,
        )
        self.assertLess(
            self.guard_text.index("validate_live_environment("),
            self.guard_text.index("os.execv("),
        )
        self.assertNotIn("import torch", self.guard_text)

    def test_rank_guard_rejects_alias_pollution_wrong_mapping_and_tamper(self) -> None:
        base = self._environment(cell="human", rank=3)
        for alias in guard.FORBIDDEN_VISIBILITY_ALIASES:
            with self.assertRaisesRegex(
                guard.BraidStage0EditorPacketRankGuardError,
                "forbidden aliases",
            ):
                guard.validate_live_environment(
                    cell_id="human",
                    expected_rocr_visible_devices="4,5,6,7",
                    expected_guard_source_sha256=self.guard_sha256,
                    expected_materializer_source_sha256=self.materializer_sha256,
                    environment={**base, alias: ""},
                    imported_modules={},
                )
        with self.assertRaisesRegex(
            guard.BraidStage0EditorPacketRankGuardError, "ROCR_VISIBLE"
        ):
            guard.validate_live_environment(
                cell_id="human",
                expected_rocr_visible_devices="4,5,6,7",
                expected_guard_source_sha256=self.guard_sha256,
                expected_materializer_source_sha256=self.materializer_sha256,
                environment={**base, "ROCR_VISIBLE_DEVICES": "0,1,2,3"},
                imported_modules={},
            )
        valid = guard.validate_live_environment(
            cell_id="human",
            expected_rocr_visible_devices="4,5,6,7",
            expected_guard_source_sha256=self.guard_sha256,
            expected_materializer_source_sha256=self.materializer_sha256,
            environment=base,
            imported_modules={},
        )
        tampered = copy.deepcopy(valid)
        tampered["rank"] = 2
        with self.assertRaisesRegex(
            guard.BraidStage0EditorPacketRankGuardError,
            "semantics or seal",
        ):
            guard.validate_live_environment_receipt(
                tampered,
                cell_id="human",
                rank=3,
                expected_guard_source_sha256=self.guard_sha256,
                expected_materializer_source_sha256=self.materializer_sha256,
            )

    def test_rank_evidence_is_create_only_and_reopens_exactly(self) -> None:
        receipt = guard.validate_live_environment(
            cell_id="dog",
            expected_rocr_visible_devices="0,1,2,3",
            expected_guard_source_sha256=self.guard_sha256,
            expected_materializer_source_sha256=self.materializer_sha256,
            environment=self._environment(cell="dog", rank=1),
            imported_modules={},
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            path = guard.write_create_only_evidence(root, receipt)
            reopened = json.loads(path.read_text(encoding="ascii"))
            self.assertEqual(reopened, receipt)
            with self.assertRaisesRegex(
                guard.BraidStage0EditorPacketRankGuardError, "not fresh"
            ):
                guard.write_create_only_evidence(root, receipt)

    def test_authenticated_archive_and_cpu_tests_precede_output_and_gpu(self) -> None:
        for name in (
            "materialize_qmosaic_editor_runtime_v1.py",
            "braid_stage0_editor_packet_rank_guard_v1.py",
            "auh_materialize_braid_stage0_editor_packets_all8_v1.sbatch",
            "test_materialize_qmosaic_editor_runtime_v1.py",
            "test_auh_materialize_braid_stage0_editor_packets_all8_v1.py",
        ):
            self.assertIn(name, self.text)
        for unsafe in (
            "member.issym()",
            "member.islnk()",
            "member.isdev()",
            "member.isfifo()",
            '".." in path.parts',
            "name in seen",
        ):
            self.assertIn(unsafe, self.text)
        archive = self.text.index("source archive contains an unsafe or duplicate member")
        source_identity = self.text.index(
            "running packet launcher differs from authenticated source archive"
        )
        cpu = self.text.index("# These CPU contracts execute")
        output = self.text.index('mkdir -- "${output_root}"')
        torchrun = self.text.index("-m torch.distributed.run")
        self.assertLess(archive, source_identity)
        self.assertLess(source_identity, cpu)
        self.assertLess(cpu, output)
        self.assertLess(output, torchrun)

    def test_all_embedded_python_boundaries_compile(self) -> None:
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.text, flags=re.DOTALL)
        self.assertEqual(len(blocks), 3)
        for index, block in enumerate(blocks):
            compile(block, f"{LAUNCHER}:heredoc-{index}", "exec")

    def test_final_paths_are_not_renamed_and_manifest_is_last_boundary(self) -> None:
        self.assertIn('packet_root="${output_root}/${cell}"', self.text)
        self.assertIn("signed child receipts embed it", self.text)
        self.assertIn(
            "persistent packet root must not be inside job-local scratch",
            self.text,
        )
        self.assertIn("per_cell_packet_atomic_rename", self.text)
        self.assertNotRegex(
            self.text,
            re.compile(r"(?:mv|os\.replace)[^\n]*(?:output_root|packet_root)"),
        )
        cleanup = self.text[
            self.text.index("cleanup() {") : self.text.index("trap cleanup EXIT")
        ]
        self.assertIn('if [[ "${published}" != true', cleanup)
        self.assertIn('rm -rf -- "${output_root}"', cleanup)
        validate = self.text.index("materializer.validate_published_files")
        manifest = self.text.index('target = root / "all8-editor-packets.manifest.json"')
        atomic_manifest = self.text.index("os.replace(staged, target)")
        chmod = self.text.index(
            'chmod a-w -- "${published_editor_public_key}" "${master_manifest}"'
        )
        published = self.text.rindex("published=true")
        self.assertLess(validate, manifest)
        self.assertLess(manifest, atomic_manifest)
        self.assertLess(atomic_manifest, chmod)
        self.assertLess(manifest, chmod)
        self.assertLess(chmod, published)
        self.assertEqual(
            self.text[published + len("published=true") :].strip(), ""
        )

    def test_manifest_hashes_every_file_and_exports_partial_runner_inputs(self) -> None:
        for value in (
            '"files": file_rows',
            '"file_sha256": sha_file(path)',
            '"file_size_bytes": path.stat().st_size',
            "pre_manifest_file_count",
            "guard.validate_live_environment_receipt",
            "materializer.validate_published_files",
            "BRAID_STAGE0_EDITOR_PUBLIC_KEY",
            "BRAID_STAGE0_EDITOR_PUBLIC_KEY_SHA256",
            "BRAID_STAGE0_DOG_EDITOR_RECEIPT",
            "BRAID_STAGE0_DOG_EDITOR_RECEIPT_SHA256",
            "BRAID_STAGE0_DOG_EDITOR_ROOT",
            "BRAID_STAGE0_HUMAN_EDITOR_RECEIPT",
            "BRAID_STAGE0_HUMAN_EDITOR_RECEIPT_SHA256",
            "BRAID_STAGE0_HUMAN_EDITOR_ROOT",
            "DOG_EDITOR_PACKET_ROOT",
            "HUMAN_EDITOR_PACKET_ROOT",
        ):
            self.assertIn(value, self.text)
        self.assertIn('cp -- "${editor_public_key}" "${published_editor_public_key}"', self.text)
        self.assertNotIn('cp -- "${editor_private_key}"', self.text)
        manifest_call = self.text[
            self.text.index('master_manifest="${output_root}') : self.text.index(
                "from pathlib import Path", self.text.index('master_manifest="${output_root}')
            )
        ]
        self.assertNotIn("editor_private_key", manifest_call)

    def test_publication_has_no_training_or_stage_authority(self) -> None:
        for value in (
            '"materialization_only": True',
            '"decode_executed": False',
            '"backward_executed": False',
            '"optimizer_created": False',
            '"parameter_update_performed": False',
            '"checkpoint_written": False',
            '"stage0_training_authority": False',
            '"stage_a_authority": False',
            '"semantic_action_editing_success_claim": False',
            '"private_key_published": False',
        ):
            self.assertIn(value, self.text)
        torchrun_call = self.text[
            self.text.index('"${rank_guard}"') : self.text.index("run_pair() {")
        ]
        self.assertNotRegex(
            torchrun_call, r"--(?:train|decode|optimizer|update)(?:\s|=)"
        )


if __name__ == "__main__":
    unittest.main()
