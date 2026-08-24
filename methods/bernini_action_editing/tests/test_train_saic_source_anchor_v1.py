from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_saic_source_anchor_v1 as trainer  # noqa: E402
from tools import build_saic_source_anchor_manifest_v1 as builder  # noqa: E402


class TrainSAICSourceAnchorV1Tests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires Linux procfs")
    def test_exact80_parent_retained_fd_map_survives_leaf_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            descriptors: list[int] = []
            rows = []
            paths = []
            try:
                for index in range(80):
                    path = root / f"source-{index:02d}.mp4"
                    raw = f"original-{index:02d}".encode("ascii")
                    path.write_bytes(raw)
                    path.chmod(0o444)
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    descriptors.append(descriptor); paths.append(path)
                    info = os.fstat(descriptor)
                    rows.append({
                        "declared_path": str(path),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "descriptor": descriptor, "device": info.st_dev,
                        "inode": info.st_ino, "size": info.st_size,
                        "mtime_ns": info.st_mtime_ns, "uid": info.st_uid,
                        "mode": stat.S_IMODE(info.st_mode),
                    })
                retained = root / "source-00.retained.mp4"
                paths[0].rename(retained)
                paths[0].write_bytes(b"hostile replacement")
                paths[0].chmod(0o444)
                core = {
                    "schema_version": (
                        "saic-source-anchor-retained-source-fd-map-v1"
                    ),
                    "supervisor_pid": os.getpid(), "source_count": 80,
                    "rows": rows,
                    "source_descriptors_held_by_parent_supervisor": True,
                    "workers_open_only_parent_proc_fd_paths": True,
                }
                value = {
                    **core, "map_digest": trainer.object_sha256(core),
                }
                raw_map = trainer.canonical_json_bytes(value)
                loaded = trainer.load_retained_source_fd_map(
                    raw_map.decode("ascii"),
                    expected_sha256=hashlib.sha256(raw_map).hexdigest(),
                    expected_supervisor_pid=os.getpid(),
                )
                self.assertEqual(len(loaded), 80)
                self.assertEqual(
                    loaded[paths[0]].stable_bytes(label="replaced source"), b""
                )
                self.assertNotEqual(
                    loaded[paths[0]].inode, paths[0].lstat().st_ino
                )
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_cpu_and_gpu_launchers_bind_archive_revision_and_checkpoint_manifest(self) -> None:
        scripts = METHOD_ROOT / "scripts"
        cpu = (scripts / "auh_build_saic_source_anchor_manifest_v1.sbatch").read_text(
            "utf-8"
        )
        gpu = (scripts / "auh_train_saic_source_anchor_v1.sbatch").read_text(
            "utf-8"
        )
        self.assertIn('git get-tar-commit-id <"${source_archive}"', cpu)
        self.assertIn('== "${source_revision}"', cpu)
        self.assertNotIn("#SBATCH --qos=", cpu)
        self.assertIn(
            'git get-tar-commit-id <"/proc/$$/fd/${SAIC_ANCHOR_ARCHIVE_FD}"',
            gpu,
        )
        self.assertIn('== "${archive_revision}"', gpu)
        self.assertIn("#SBATCH --qos=bgqos", gpu)
        self.assertNotIn("#SBATCH --gres", cpu)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", gpu)
        self.assertIn("--checkpoint-content-manifest", gpu)
        self.assertIn("--expected-checkpoint-content-manifest-sha256", gpu)
        self.assertNotIn("tests.test_build_saic_source_anchor_manifest_v1", cpu)
        self.assertIn(
            '"${method_root}/tests/test_build_saic_source_anchor_manifest_v1.py"',
            cpu,
        )
        self.assertIn("SAIC_ANCHOR_CHECKPOINT_MANIFEST_FD", gpu)
        self.assertIn("--standalone --nnodes=1 --nproc-per-node=8", gpu)
        self.assertIn("formal full60 terminal admission pins are unresolved", gpu)
        self.assertIn("source manifest exact80 path closure differs", gpu)
        self.assertIn("SAIC_ANCHOR_SOURCE_FD_MAP", gpu)
        self.assertIn("/proc/{self.supervisor_pid}/fd/{self.descriptor}", Path(
            trainer.__file__).read_text("utf-8"))
        self.assertIn("after complete decode", Path(trainer.__file__).read_text("utf-8"))
        self.assertIn("source archive canonical manifest differs", gpu)
        self.assertIn("runtime recursive import closure differs", gpu)
        self.assertIn("formal full60 exact deep admission differs", gpu)

    def test_trainer_constructs_the_deployment_shift5_unipc_renderer(self) -> None:
        source = Path(trainer.__file__).read_text("utf-8")
        self.assertIn("infer_lora.inference_renderer_config_overrides(checkpoint)", source)
        self.assertNotIn("**legacy.renderer_config_overrides(checkpoint)", source)
        self.assertIn("infer_lora.DEFAULT_NEGATIVE_PROMPT", source)
        self.assertNotIn("legacy.DEFAULT_NEGATIVE_PROMPT", source)
        self.assertIn("audit_runtime_unipc_schedule", source)

    def test_schedule_is_only_low_sigma_35_through_39(self) -> None:
        self.assertEqual(
            [trainer.schedule_index_for_update(index) for index in range(32)],
            ([35, 36, 37, 38, 39] * 7)[:32],
        )
        with self.assertRaises(trainer.SAICSourceAnchorTrainingError):
            trainer.schedule_index_for_update(-1)

    def test_noise_seed_is_deterministic_arm_and_phase_bound(self) -> None:
        seed = trainer.noise_seed(seed=3, update_index=7, dp_arm=0, phase="train")
        self.assertEqual(
            seed,
            trainer.noise_seed(seed=3, update_index=7, dp_arm=0, phase="train"),
        )
        self.assertNotEqual(
            seed,
            trainer.noise_seed(seed=3, update_index=7, dp_arm=1, phase="train"),
        )
        self.assertNotEqual(
            seed,
            trainer.noise_seed(seed=3, update_index=7, dp_arm=0, phase="heldout-before"),
        )

    @staticmethod
    def _summary(*, error: float, advantage: float, fraction: float):
        return {
            "row_count": 16,
            "correct_flow_error_mean": error,
            "wrong_source_advantage_mean": advantage,
            "wrong_source_positive_fraction": fraction,
        }

    def test_heldout_gate_is_noncompensating(self) -> None:
        before = self._summary(error=1.0, advantage=0.1, fraction=0.75)
        passed = trainer.heldout_gate(
            before, self._summary(error=1.01, advantage=0.11, fraction=0.875)
        )
        self.assertTrue(passed["noncompensating_all_pass"])
        self.assertTrue(passed["checkpoint_publication_allowed"])
        noop_failure = trainer.heldout_gate(
            before, self._summary(error=1.03, advantage=0.2, fraction=1.0)
        )
        self.assertFalse(noop_failure["noncompensating_all_pass"])
        source_failure = trainer.heldout_gate(
            before, self._summary(error=0.5, advantage=0.09, fraction=1.0)
        )
        self.assertFalse(source_failure["noncompensating_all_pass"])
        exact_no_change = trainer.heldout_gate(before, dict(before))
        self.assertTrue(exact_no_change["no_op_reconstruction_noninferior"])
        self.assertTrue(
            exact_no_change["wrong_source_dependence_noninferior_and_positive"]
        )
        self.assertFalse(exact_no_change["at_least_one_strict_improvement"])
        self.assertFalse(exact_no_change["noncompensating_all_pass"])
        self.assertFalse(exact_no_change["checkpoint_publication_allowed"])

    def test_cli_requires_formal32_or_explicit_smoke(self) -> None:
        base = dict(
            num_frames=81,
            mode="formal",
            max_updates=32,
            learning_rate=1e-5,
            max_grad_norm=1.0,
            wrong_source_margin=0.01,
            ranking_weight=1.0,
            seed=trainer.DEFAULT_SEED,
            gradient_accumulation_steps=1,
            ack_source_anchor_only_no_action_claim=True,
            ack_incomplete_row_coverage_smoke=False,
            expected_bernini_commit="a" * 40,
            expected_veomni_commit="b" * 40,
            method_source_revision="c" * 40,
            expected_manifest_sha256="d" * 64,
            expected_checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
            expected_checkpoint_content_manifest_sha256="f" * 64,
            method_source_archive_sha256="e" * 64,
            trainer_source_sha256="1" * 64,
            release_manifest_sha256="2" * 64,
            release_manifest_digest="3" * 64,
            submission_receipt_sha256="4" * 64,
            submission_receipt_digest="5" * 64,
            python_executable_sha256="6" * 64,
            python_version_stdout_sha256="7" * 64,
            formal_full60_admission_sha256="8" * 64,
            formal_full60_admission_digest="9" * 64,
            source_fd_map_sha256="a" * 64,
            source_supervisor_pid=12345,
            archive_member_manifest_sha256=(
                "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
            ),
            extracted_tree_manifest_sha256="b" * 64,
            archive_binding_receipt_digest="c" * 64,
            runtime_origin_manifest_sha256=(
                "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
            ),
            runtime_origin_receipt_digest="d" * 64,
            slurm_job_id="123456",
        )
        contract = trainer.validate_cli(argparse.Namespace(**base))
        self.assertEqual(contract["optimizer_updates"], 32)
        self.assertTrue(contract["all_train_rows_used_once_as_clean_endpoint"])
        base["max_updates"] = 20
        with self.assertRaisesRegex(trainer.SAICSourceAnchorTrainingError, "exactly 32"):
            trainer.validate_cli(argparse.Namespace(**base))
        base["mode"] = "smoke"
        base["max_updates"] = 1
        base["ack_incomplete_row_coverage_smoke"] = True
        smoke = trainer.validate_cli(argparse.Namespace(**base))
        self.assertFalse(smoke["all_train_rows_used_once_as_clean_endpoint"])
        base["mode"] = "formal"
        base["max_updates"] = 32
        base["ack_incomplete_row_coverage_smoke"] = False
        base["ack_source_anchor_only_no_action_claim"] = False
        with self.assertRaisesRegex(trainer.SAICSourceAnchorTrainingError, "acknowledgement"):
            trainer.validate_cli(argparse.Namespace(**base))

    def test_manifest_loader_rejects_heldout_leak_and_action_iid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            media: dict[str, Path] = {}
            for index in range(80):
                iid = f"{index:016x}"
                path = root / f"{iid}.mp4"
                path.write_bytes(iid.encode("ascii"))
                media[iid] = path

            def sha(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            def make_rows(split: str, count: int, start: int):
                rows = []
                per_arm = count // 2
                for arm in range(2):
                    ids = [f"{start + arm * per_arm + offset:016x}" for offset in range(per_arm)]
                    for local, iid in enumerate(ids):
                        wrong = ids[(local + 1) % len(ids)]
                        value = {
                            "schema_version": builder.SCHEMA_VERSION,
                            "split": split,
                            "row_index": len(rows),
                            "dp_arm": arm,
                            "iid": iid,
                            "source_video_path": str(media[iid]),
                            "source_video_sha256": sha(media[iid]),
                            "wrong_iid": wrong,
                            "wrong_source_video_path": str(media[wrong]),
                            "wrong_source_video_sha256": sha(media[wrong]),
                            "frame_count": 81,
                            "fps": 25.0,
                            "reported_fps": 25.0,
                            "bucket_hw": [480, 496],
                            "scramble_seed": start + local,
                        }
                        rows.append({**value, "row_digest": trainer.object_sha256(value)})
                return rows

            value = {
                "schema_version": builder.SCHEMA_VERSION,
                "optimizer_authorized": False,
                "source_root": str(root),
                "selection_seed": 1,
                "frame_count": 81,
                "fps": 25.0,
                "train_count": 64,
                "holdout_count": 16,
                "selected_bucket_counts": {"480x496": 80},
                "eligible_bucket_counts": {"480x496": 80},
                "strict_action_iids_excluded": sorted(builder.STRICT_ACTION_IIDS),
                "wrong_source_policy": "same_split_same_bucket_same_dp_arm_fixed_point_free",
                "holdout_used_by_optimizer": False,
                "train_rows": make_rows("train", 64, 0),
                "holdout_rows": make_rows("holdout", 16, 64),
                "input_closure": {
                    "source_video_only": True,
                    "paired_target": False,
                    "action_instruction": False,
                    "proposal_video": False,
                    "mask_pose_flow_track_trajectory": False,
                },
            }
            manifest = {**value, "manifest_digest": trainer.object_sha256(value)}
            path = root / "manifest.json"
            path.write_bytes(trainer.canonical_json_bytes(manifest) + b"\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = trainer.load_manifest(path, expected_sha256=digest, verify_files=True)
            self.assertEqual(len(loaded.train_rows), 64)
            self.assertEqual(len(loaded.holdout_rows), 16)

            tampered = json.loads(path.read_text("ascii"))
            tampered["holdout_rows"][0]["iid"] = tampered["train_rows"][0]["iid"]
            row = tampered["holdout_rows"][0]
            unsigned_row = dict(row)
            unsigned_row.pop("row_digest")
            row["row_digest"] = trainer.object_sha256(unsigned_row)
            unsigned_root = dict(tampered)
            unsigned_root.pop("manifest_digest")
            tampered["manifest_digest"] = trainer.object_sha256(unsigned_root)
            other = root / "tampered.json"
            other.write_bytes(trainer.canonical_json_bytes(tampered) + b"\n")
            with self.assertRaises(trainer.SAICSourceAnchorTrainingError):
                trainer.load_manifest(
                    other,
                    expected_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
                    verify_files=False,
                )

    def test_condition_intervention_audit_binds_one_endpoint(self) -> None:
        import torch

        clean = torch.zeros(1, 16, 21, 2, 2)
        wrong_clean = torch.ones_like(clean)
        refs = tuple(torch.full((1, 16, 1, 2, 2), float(index)) for index in range(4))
        wrong_refs = tuple(
            torch.full((1, 16, 1, 2, 2), float(index + 10)) for index in range(4)
        )
        correct = trainer.EncodedSource("a", clean, refs, "a" * 64)
        wrong = trainer.EncodedSource("b", wrong_clean, wrong_refs, "b" * 64)
        epsilon = torch.full_like(clean, 2.0)
        sigma = torch.tensor([0.25], dtype=torch.float32)
        state = (0.75 * clean + 0.25 * epsilon).contiguous()
        target = (epsilon - clean).contiguous()
        kwargs = dict(
            correct=correct,
            wrong=wrong,
            correct_condition=clean.clone(),
            wrong_condition=wrong_clean.clone(),
            state=state,
            target=target,
            epsilon=epsilon,
            sigma=sigma,
            timestep=torch.tensor([117.0], dtype=torch.float32),
            cond_embeds=torch.zeros(1, 2, 3),
            uncond_embeds=torch.ones(1, 2, 3),
        )
        receipt = trainer._assert_only_source_condition_differs(**kwargs)
        self.assertTrue(receipt["same_clean_source_endpoint"])
        bad = dict(kwargs)
        bad["state"] = state + 0.01
        with self.assertRaisesRegex(
            trainer.SAICSourceAnchorTrainingError, "share endpoint"
        ):
            trainer._assert_only_source_condition_differs(**bad)

    def test_four_field_forward_and_correct_wrong_serial_vjp(self) -> None:
        """Toy graph proves none is skipped and both conditions reach all weights."""

        import torch
        from contextlib import nullcontext
        from types import SimpleNamespace

        shape = (1, 16, 21, 2, 2)
        state = torch.zeros(shape, dtype=torch.float32)
        condition_correct = torch.zeros_like(state)
        condition_wrong = torch.ones_like(state)
        references = tuple(torch.zeros(1, 16, 1, 2, 2) for _ in range(4))
        parameters = {
            "none_uncond": torch.nn.Parameter(torch.tensor(10.0)),
            "V_uncond": torch.nn.Parameter(torch.tensor(2.0)),
            "VI_uncond": torch.nn.Parameter(torch.tensor(3.0)),
            "VI_cond": torch.nn.Parameter(torch.tensor(4.0)),
        }
        branches = {
            "none_uncond": SimpleNamespace(name="none", key="none_uncond"),
            "V_uncond": SimpleNamespace(name="V", key="V_uncond"),
            "VI_uncond": SimpleNamespace(name="VI", key="VI_uncond"),
            "VI_cond": SimpleNamespace(name="VI", key="VI_cond"),
        }
        coefficients = {
            "none_uncond": -0.25,
            "V_uncond": -3.25,
            "VI_uncond": 0.5,
            "VI_cond": 4.0,
        }
        pack_calls: list[torch.Tensor] = []
        forward_calls: list[str] = []

        def fake_build(transformer, condition, refs, x_state):
            self.assertIs(x_state, state)
            self.assertEqual(len(refs), 4)
            pack_calls.append(condition)
            return object()

        def fake_rows(pack, *, cond, uncond):
            return tuple(
                (name, branches[name], cond if name == "VI_cond" else uncond, coefficient)
                for name, coefficient in coefficients.items()
            )

        def fake_forward(diffusion, branch, *, timestep, text, handle):
            forward_calls.append(branch.key)
            return parameters[branch.key].expand(shape)

        def identity_unpack(value, *, video_shape):
            self.assertEqual(tuple(video_shape), shape)
            return value

        patches = (
            mock.patch.object(trainer, "_build_pack", side_effect=fake_build),
            mock.patch.object(trainer, "_native_rows", side_effect=fake_rows),
            mock.patch.object(trainer, "_forward_branch", side_effect=fake_forward),
            mock.patch.object(
                trainer.native_bridge,
                "_unpack_spatial_velocity",
                side_effect=identity_unpack,
            ),
            mock.patch.object(torch, "autocast", side_effect=lambda **kwargs: nullcontext()),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            expected = trainer._guided_prediction_no_grad(
                object(),
                object(),
                condition=condition_correct,
                references=references,
                state=state,
                timestep=torch.tensor([117.0]),
                cond_embeds=torch.ones(1, 2, 3),
                uncond_embeds=torch.zeros(1, 2, 3),
                handle=object(),
            )
            expected_scalar = sum(
                float(parameters[name].detach()) * coefficient
                for name, coefficient in coefficients.items()
            )
            self.assertTrue(
                torch.equal(expected.guided, torch.full(shape, expected_scalar))
            )
            self.assertEqual(
                forward_calls,
                ["none_uncond", "V_uncond", "VI_uncond", "VI_cond"],
            )
            forward_calls.clear()
            trainer._serial_prediction_vjp(
                object(),
                object(),
                condition=condition_correct,
                references=references,
                state=state,
                timestep=torch.tensor([117.0]),
                cond_embeds=torch.ones(1, 2, 3),
                uncond_embeds=torch.zeros(1, 2, 3),
                handle=object(),
                output_cotangent=torch.ones_like(state),
                expected=expected,
            )
            trainer._serial_prediction_vjp(
                object(),
                object(),
                condition=condition_wrong,
                references=references,
                state=state,
                timestep=torch.tensor([117.0]),
                cond_embeds=torch.ones(1, 2, 3),
                uncond_embeds=torch.zeros(1, 2, 3),
                handle=object(),
                output_cotangent=torch.full_like(state, 2.0),
                expected=expected,
            )
        # One no-grad pack plus one correct and one wrong replay.
        self.assertIs(pack_calls[0], condition_correct)
        self.assertIs(pack_calls[1], condition_correct)
        self.assertIs(pack_calls[2], condition_wrong)
        # none is evaluated in the four-field prediction but never replayed.
        self.assertEqual(
            forward_calls,
            ["V_uncond", "VI_uncond", "VI_cond"] * 2,
        )
        self.assertIsNone(parameters["none_uncond"].grad)
        numel = state.numel()
        for name in ("V_uncond", "VI_uncond", "VI_cond"):
            self.assertIsNotNone(parameters[name].grad)
            expected_grad = coefficients[name] * 3.0 * numel
            self.assertAlmostEqual(
                float(parameters[name].grad), expected_grad, places=4
            )


if __name__ == "__main__":
    unittest.main()
