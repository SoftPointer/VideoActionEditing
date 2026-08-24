from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import stat
import tarfile
import tempfile
import unittest
from unittest import mock
import re


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
BUILDER_PATH = METHOD_ROOT / "tools" / "build_elal3_c1_oracle_diagnostic_release_v1.py"
LAUNCHER_PATH = METHOD_ROOT / "scripts" / "auh_run_elal3_c1_oracle_diagnostic_release_v1.sh"
SPEC = importlib.util.spec_from_file_location(
    "build_elal3_c1_oracle_diagnostic_release_v1", BUILDER_PATH
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ELAL3C1OracleDiagnosticReleaseTests(unittest.TestCase):
    def _fake_tree(self, parent: Path) -> tuple[Path, Path, Path, dict[str, str]]:
        root = parent / "repo"
        pins: dict[str, str] = {}
        for index, relative in enumerate(builder.RUNTIME_PINS):
            path = root.joinpath(*Path(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = f"VALUE_{index} = {index}\n".encode("ascii")
            path.write_bytes(raw)
            pins[relative] = hashlib.sha256(raw).hexdigest()
        for relative in (
            builder.DERIVATIVE_AUTHORITY_RELATIVE,
            builder.MODEL_AUTHORITY_RELATIVE,
        ):
            path = root.joinpath(*Path(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}\n")
        latent = parent / "latent-receipt.json"
        latent.write_bytes(b"{}\n")
        train_lora = root / "methods/bernini_action_editing/train_lora.py"
        return root.resolve(), latent.resolve(), train_lora.resolve(), pins

    @staticmethod
    def _evidence_stub(*_args: bytes) -> dict[str, dict[str, str]]:
        return {
            "derivative": {"authority_digest": builder.DERIVATIVE_AUTHORITY_DIGEST},
            "model": {"authority_digest": builder.MODEL_AUTHORITY_DIGEST},
            "latent": {"receipt_digest": builder.LATENT_RECEIPT_DIGEST},
        }

    def test_production_builder_pins_independently_reviewed_trainer(self) -> None:
        relative = (
            "methods/bernini_action_editing/"
            "train_elal3_c1_simulator_overfit_v1.py"
        )
        expected = (
            "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3"
        )
        self.assertEqual(builder.TRAINER_SHA256, expected)
        self.assertEqual(builder.RUNTIME_PINS[relative], expected)
        self.assertEqual(hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_release_pins_reviewed_remote_train_lora_not_local_workspace_bytes(self) -> None:
        relative = "methods/bernini_action_editing/train_lora.py"
        reviewed = builder.RUNTIME_PINS[relative]
        local = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        self.assertEqual(
            reviewed,
            "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
        )
        self.assertNotEqual(local, reviewed)

    def test_deterministic_ustar_and_narrow_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, latent, train_lora, pins = self._fake_tree(parent)
            first = parent / "release-a"
            second = parent / "release-b"
            with mock.patch.object(
                builder, "_validate_evidence", side_effect=self._evidence_stub
            ):
                a = builder.publish(
                    root, latent, train_lora, first, runtime_pins=pins
                )
                b = builder.publish(
                    root, latent, train_lora, second, runtime_pins=pins
                )
            self.assertEqual(
                (first / "source.tar").read_bytes(),
                (second / "source.tar").read_bytes(),
            )
            self.assertEqual(
                (first / "source.manifest.json").read_bytes(),
                (second / "source.manifest.json").read_bytes(),
            )
            self.assertEqual(a["archive_sha256"], b["archive_sha256"])
            manifest_raw = (first / "source.manifest.json").read_bytes()
            manifest = json.loads(manifest_raw)
            self.assertEqual(manifest_raw, builder.canonical_json_bytes(manifest) + b"\n")
            unsigned = dict(manifest)
            digest = unsigned.pop("manifest_digest")
            self.assertEqual(digest, builder.object_digest(unsigned))
            self.assertTrue(manifest["simulator_optimizer_diagnostic_authorized"])
            self.assertTrue(manifest["teacher_forced_oracle_q_required"])
            for key in (
                "formal_c1_authorized",
                "exact160_authorized",
                "source_instruction_inference_authorized",
                "real_video_generalization_authorized",
                "production_model_authorized",
                "scientific_claim_authorized",
            ):
                self.assertFalse(manifest[key], key)
            self.assertEqual(manifest["optimizer_update_sequence"], [0, 1, 10])
            self.assertEqual(
                manifest["distributed_topology"],
                {
                    "world_size": 8,
                    "data_parallel_size": 2,
                    "sequence_parallel_size": 4,
                    "one_node_per_run": True,
                },
            )
            self.assertEqual(manifest["run_assignments"], list(builder.RUN_ASSIGNMENTS))
            self.assertEqual(
                manifest["external_latent_bundle"],
                {
                    "sha256": builder.LATENT_BUNDLE_SHA256,
                    "size": builder.LATENT_BUNDLE_SIZE,
                    "mode": "0444",
                    "nlink": 1,
                },
            )
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o555)
            self.assertEqual(stat.S_IMODE((first / "source.tar").stat().st_mode), 0o444)
            with tarfile.open(first / "source.tar", "r:") as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                self.assertEqual(names, sorted(names, key=lambda value: value.encode("ascii")))
                self.assertEqual(len(names), len(pins) + 3)
                for member in members:
                    self.assertTrue(member.isreg())
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.gid, 0)
                    self.assertEqual(member.mtime, 0)
                    self.assertEqual(member.mode, 0o444)
                extracted = parent / "extracted-repo-tree"
                extracted.mkdir()
                for member in members:
                    target = extracted.joinpath(*Path(member.name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    handle = archive.extractfile(member)
                    assert handle is not None
                    target.write_bytes(handle.read())
            label_path = (
                extracted
                / "methods/bernini_action_editing/elal3_simulator_label_v1.py"
            )
            registered_authority = (
                label_path.resolve().parents[2]
                / builder.DERIVATIVE_AUTHORITY_RELATIVE
            )
            self.assertEqual(
                registered_authority,
                (
                    extracted / builder.DERIVATIVE_AUTHORITY_RELATIVE
                ).resolve(),
            )
            self.assertTrue(registered_authority.is_file())

    def test_publish_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, latent, train_lora, pins = self._fake_tree(parent)
            output = parent / "release"
            with mock.patch.object(
                builder, "_validate_evidence", side_effect=self._evidence_stub
            ):
                builder.publish(
                    root, latent, train_lora, output, runtime_pins=pins
                )
                with self.assertRaises(builder.ELAL3C1ReleaseError):
                    builder.publish(
                        root, latent, train_lora, output, runtime_pins=pins
                    )

    def test_runtime_symlink_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, latent, train_lora, pins = self._fake_tree(parent)
            relative = next(iter(pins))
            source = root.joinpath(*Path(relative).parts)
            replacement = parent / "replacement.py"
            replacement.write_text("VALUE = 7\n", encoding="ascii")
            source.unlink()
            source.symlink_to(replacement)
            pins[relative] = hashlib.sha256(replacement.read_bytes()).hexdigest()
            with mock.patch.object(
                builder, "_validate_evidence", side_effect=self._evidence_stub
            ):
                with self.assertRaisesRegex(
                    builder.ELAL3C1ReleaseError, "non-canonical source"
                ):
                    builder.publish(
                        root,
                        latent,
                        train_lora,
                        parent / "release",
                        runtime_pins=pins,
                    )

    def test_launcher_is_syntax_valid_and_exactly_release_pinned(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(LAUNCHER_PATH)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        run = subprocess.run(
            ["bash", str(LAUNCHER_PATH)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(run.returncode, 2)
        self.assertIn("ELAL3_C1_SOURCE_ARCHIVE is required", run.stderr)
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("PENDING_", source)
        self.assertIn(builder.TRAINER_SHA256, source)
        self.assertIn(
            "631611a96a744025eb6e5b223958908c7dfccfb69bfaefa7432ea9c20afc8194",
            source,
        )
        self.assertIn(
            "bb56f175f205b626f003c855260243a5c1a5fa3d8c7f0464ddea49931006a9f3",
            source,
        )

    def test_launcher_preregisters_exact_nodes_v2_bundle_and_gate_order(self) -> None:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        for literal in (
            "141620:auh7-1b-gpu-226",
            "141618:auh7-1b-gpu-249",
            "141619:auh7-1b-gpu-257",
            "seed=20260817",
            "seed=20260818",
            "seed=20260819",
            "vae-c1-row-modelbound-v2/c1-latents.safetensors",
            builder.LATENT_BUNDLE_SHA256,
            builder.LATENT_RECEIPT_SHA256,
            "--nproc-per-node=8",
            "--no-python",
            "--max-steps \"${steps}\"",
            "--seed \"${seed}\"",
            "--preflight-only",
            "--ack-simulator-oracle-q-overfit-only",
            "--expected-train-lora-source-sha256",
            "run_stage elal3_c1_preflight_no_update 1 preflight",
            "run_stage elal3_c1_one_step_smoke 1 train",
            "run_stage elal3_c1_ten_step_overfit 10 train",
        ):
            self.assertIn(literal, source)
        for literal in (
            'rank_root="${cache_base}/rank_${rank}"',
            'MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"',
            'MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"',
            'TORCH_EXTENSIONS_DIR="${rank_root}/torch"',
            'XDG_CACHE_HOME="${rank_root}/xdg"',
            'TRITON_CACHE_DIR="${rank_root}/triton"',
            'failure log sealed',
        ):
            self.assertIn(literal, source)
        self.assertNotIn(
            "63bea174ca22f1c2aa784def3031a998db3108005a6954b72e1aeb85f5504fac",
            source,
        )
        self.assertNotIn(
            "6c790de134314c155a604e35a7f7625301d2a53bae12076af79e5891906be040",
            source,
        )
        no_update = source.index("NO_UPDATE_PREFLIGHT.json")
        world8_preflight = source.index(
            "run_stage elal3_c1_preflight_no_update 1 preflight"
        )
        one_step = source.index("run_stage elal3_c1_one_step_smoke 1 train")
        ten_step = source.index("run_stage elal3_c1_ten_step_overfit 10 train")
        self.assertLess(no_update, world8_preflight)
        self.assertLess(world8_preflight, one_step)
        self.assertLess(no_update, one_step)
        self.assertLess(one_step, ten_step)

    def test_every_launcher_python_heredoc_compiles(self) -> None:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        chunks = []
        remaining = source
        marker = "<<'PY'\n"
        while marker in remaining:
            _, remaining = remaining.split(marker, 1)
            chunk, remaining = remaining.split("\nPY\n", 1)
            chunks.append(chunk)
        self.assertEqual(len(chunks), 5)
        for index, chunk in enumerate(chunks):
            compile(chunk, f"launcher-heredoc-{index}.py", "exec")

    def test_generated_per_rank_cache_wrapper_is_bash_syntax_valid(self) -> None:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        match = re.search(r"raw = (b'''#!/usr/bin/env bash\n.*?\n''')", source, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        raw = ast.literal_eval(match.group(1))
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = Path(temporary) / "rank-wrapper.sh"
            wrapper.write_bytes(raw)
            result = subprocess.run(
                ["bash", "-n", str(wrapper)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
