#!/usr/bin/env python3
"""CPU/static release tests for the activation-v2 authoring materializers."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CORE = METHOD_ROOT / "oracle_regeneration_activation_v2.py"
VAE_TOOL = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_vae_refs_activation_v2.py"
)
PROMPT_TOOL = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_prompts_activation_v2.py"
)
LAUNCHER = (
    METHOD_ROOT / "scripts/auh_materialize_oracle_regeneration_activation_v2.sh"
)
EXPECTED = {
    CORE: "ef97259dd181ff065267e32f1e5cca158e26ad5174457780163658a3db728bb0",
    VAE_TOOL: "57025333f00f4da9dfa0d74263597090a8c79546847f341dfd4441564043116a",
    PROMPT_TOOL: "57689bdfddef2702f834bf4e021f585b0fbb81cfbeee2a889f35f9efa0359f93",
}
E02_CAPTION = (
    "The same pale bare hand firmly grips the same red mushroom at its lower "
    "stem, twists and pulls it free from the same soil, lifts the same intact "
    "mushroom above the newly empty root hole, and holds it there. Exactly one "
    "hand and one mushroom remain visible; do not duplicate, split, or fuse "
    "either one."
)
E03_CAPTION = (
    "The same farm worker moves the same harvested root cluster over the same "
    "woven basket, lowers it past the rim, opens the same hand, and releases "
    "it. The cluster falls, bounces slightly, and settles inside while the "
    "now-empty hand withdraws. Do not duplicate the hand or cluster."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duplicate_literal_dict_keys(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    duplicates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        observed: set[str] = set()
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if key.value in observed:
                duplicates.append((node.lineno, key.value))
            observed.add(key.value)
    return duplicates


class MaterializerStaticTests(unittest.TestCase):
    def test_exact_authoring_code_tuple_and_hard_disabled_core(self) -> None:
        for path, expected in EXPECTED.items():
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(sha256(path), expected)
        namespace: dict[str, object] = {}
        core_source = CORE.read_text(encoding="utf-8")
        self.assertIn("COMPILED_AUTHORITY_PACKET_SHA256: Optional[str] = None", core_source)
        self.assertIn(
            "COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256: Optional[str] = None",
            core_source,
        )
        self.assertNotIn("20260818_regeneration_oracle_human_annotation_v1.md", core_source)
        self.assertNotIn("e02_round37_gate_proposal", core_source)
        self.assertNotIn("torch", sys.modules)
        del namespace

    def test_python_sources_have_no_duplicate_literal_dict_keys(self) -> None:
        for path in (CORE, VAE_TOOL, PROMPT_TOOL):
            self.assertEqual(duplicate_literal_dict_keys(path), [], str(path))

    def test_vae_tool_is_source_only_and_sampler_free(self) -> None:
        source = VAE_TOOL.read_text(encoding="utf-8")
        required = (
            "rank zero",
            "alone decodes the bound source",
            "references_encoded_as_four_independent_rgb_frames",
            "references_not_sliced_from_full_source_latent",
            "reference_content_duplicates_rejected",
            "target_video_or_latent_used\": False",
            "self_generated_anchor_tensor_used\": False",
            "full_model_or_sampler_loaded\": False",
            "transformer_loaded\": False",
            "scheduler_loaded\": False",
        )
        self.assertTrue(all(token in source for token in required))
        self.assertNotIn("diffusion.sample", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward(", source)

    def test_prompt_tool_rank0_only_t5_and_exact_captions(self) -> None:
        source = PROMPT_TOOL.read_text(encoding="utf-8")
        required = (
            "rank0_only_text_encoder_load\": True",
            "nonzero_ranks_never_deserialized_text_encoder\": True",
            "rank0_only_text_encode\": True",
            "broadcast_exact\": True",
            "sampler_or_scheduler_called\": False",
            "target_video_or_latent_used\": False",
            "self_generated_anchor_tensor_used\": False",
        )
        self.assertTrue(all(token in source for token in required))
        self.assertNotIn("diffusion.sample", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward(", source)
        core_source = CORE.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(E02_CAPTION.encode()).hexdigest(), core_source)
        self.assertIn(hashlib.sha256(E03_CAPTION.encode()).hexdigest(), core_source)

    def test_launcher_is_existing_allocation_world4_and_exact_pinned(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("--nproc_per_node=4", source)
        self.assertIn("ORACLE_MATERIALIZER_KIND", source)
        self.assertIn("ORACLE_CASE_ID", source)
        self.assertIn("ORACLE_VISIBLE_GPUS", source)
        self.assertIn("unset PYTHONPATH", source)
        self.assertIn("PYTHONNOUSERSITE=1", source)
        self.assertNotIn("srun ", source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("ssh ", source)
        for path, digest in EXPECTED.items():
            self.assertIn(path.name, source)
            self.assertEqual(len(re.findall(re.escape(digest), source)), 1)
        self.assertEqual(source.count(E02_CAPTION), 1)
        self.assertEqual(source.count(E03_CAPTION), 1)
        self.assertIn(
            "source-only diagnostic provenance inside an existing one-node allocation",
            source,
        )
        self.assertNotIn("human-reviewed", source.lower())
        self.assertNotIn("cryptographically signed", source.lower())

    def test_launcher_rejects_duplicate_gpu_map_and_out_of_range_port_early(self) -> None:
        base = {
            **os.environ,
            "ORACLE_ACTIVATION_V2_REPO_ROOT": "/",
            "ORACLE_ACTIVATION_V2_PYTHON_BIN": "/",
            "BERNINI_OFFICIAL_ROOT": "/",
            "BERNINI_VEOMNI_ROOT": "/",
            "BERNINI_ACTION_CHECKPOINT": "/",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST": "/",
            "ORACLE_MATERIALIZER_OUTPUT_DIR": "/not-created",
            "ORACLE_MATERIALIZER_KIND": "vae",
            "ORACLE_CASE_ID": "e02",
            "ORACLE_MASTER_PORT": "40001",
        }
        duplicate = subprocess.run(
            ["bash", str(LAUNCHER)],
            env={**base, "ORACLE_VISIBLE_GPUS": "0,0,0,0"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("visible GPU list differs", duplicate.stderr)
        invalid_port = subprocess.run(
            ["bash", str(LAUNCHER)],
            env={
                **base,
                "ORACLE_VISIBLE_GPUS": "0,1,2,3",
                "ORACLE_MASTER_PORT": "99999",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(invalid_port.returncode, 2)
        self.assertIn("master port exceeds 65535", invalid_port.stderr)

    def test_atomic_receipt_publish_never_replaces_existing_name(self) -> None:
        for index, tool_path in enumerate((VAE_TOOL, PROMPT_TOOL)):
            spec = importlib.util.spec_from_file_location(
                f"activation_v2_materializer_test_{index}", tool_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve(strict=True)
                path = directory / "receipt.json"
                module._write_receipt(path, {"first": True})
                before = path.read_bytes()
                before_stat = path.stat()
                with self.assertRaises(module.__dict__[next(
                    name
                    for name in module.__dict__
                    if name.endswith("MaterializationError")
                )]):
                    module._write_receipt(path, {"second": True})
                after_stat = path.stat()
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(stat.S_IMODE(after_stat.st_mode), 0o444)
                self.assertEqual(after_stat.st_nlink, 1)
                self.assertEqual(
                    (after_stat.st_dev, after_stat.st_ino),
                    (before_stat.st_dev, before_stat.st_ino),
                )


if __name__ == "__main__":
    unittest.main()
