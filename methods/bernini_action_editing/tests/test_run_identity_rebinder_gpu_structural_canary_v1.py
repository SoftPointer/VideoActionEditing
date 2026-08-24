#!/usr/bin/env python3
"""CPU contracts for the real-GPU IdentityRebinder structural canary."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import identity_rebinder_v1 as rebinder  # noqa: E402
import run_identity_rebinder_gpu_structural_canary_v1 as runner  # noqa: E402


class FakeTransformer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls: list[tuple[int, float, int]] = []

    def patch_vae_latent(self, value: torch.Tensor, *, source_id: float):
        token_count = int(value.shape[2])
        self.calls.append((id(value), float(source_id), token_count))
        hidden = torch.full(
            (1, token_count, rebinder.HIDDEN_SIZE_1P3B),
            float(source_id + 1.0),
            dtype=torch.float32,
        )
        rotary = torch.full(
            (1, 1, token_count, 8), float(source_id), dtype=torch.float32
        )
        return hidden, rotary


def atlas() -> rebinder.IdentityAtlas:
    return rebinder.IdentityAtlas(
        tokens=torch.zeros(1, 2, rebinder.HIDDEN_SIZE_1P3B).contiguous(),
        source_video_sha256="a" * 64,
        source_frame_count=81,
        construction_digest="b" * 64,
    )


class ParserContractTests(unittest.TestCase):
    def test_no_duplicate_option_strings(self) -> None:
        parser = runner.build_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        options = [
            option
            for action in parser._actions
            for option in action.option_strings
        ]
        self.assertEqual(len(options), len(set(options)))
        self.assertEqual(options.count("--expected-rebinder-sha256"), 1)
        self.assertEqual(options.count("--ack-structural-only-no-semantic-claim"), 1)

    def test_parser_is_really_startable_and_ack_defaults_false(self) -> None:
        parsed = runner.build_parser().parse_args(
            [
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--checkpoint-content-manifest", "/m",
                "--cell-spec", "/s",
                "--expected-cell-spec-sha256", "1" * 64,
                "--cell-id", "dog",
                "--output-dir", "/o",
                "--expected-runner-sha256", "2" * 64,
                "--expected-rebinder-sha256", "3" * 64,
                "--expected-native-field-probe-sha256", "4" * 64,
                "--launcher-source-sha256", "5" * 64,
                "--runtime-closure-manifest", "/closure.json",
                "--expected-runtime-closure-manifest-sha256", "6" * 64,
                "--runtime-source-archive-sha256", "7" * 64,
            ]
        )
        self.assertFalse(parsed.ack_structural_only_no_semantic_claim)
        self.assertEqual(parsed.cell_id, "dog")


class VPackContractTests(unittest.TestCase):
    def test_exact_source1_target0_patch_once_and_concat(self) -> None:
        transformer = FakeTransformer()
        source = torch.zeros(1, 16, 21, 1, 1)
        target = torch.ones_like(source)
        pack = runner.build_vonly_pack(transformer, source, target)
        self.assertEqual(
            [(row[1], row[2]) for row in transformer.calls],
            [(1.0, 21), (0.0, 21)],
        )
        self.assertEqual(pack.condition_tokens, 21)
        self.assertEqual(pack.target_tokens, 21)
        self.assertEqual(pack.total_tokens, 42)
        self.assertEqual(tuple(pack.hidden.shape), (1, 42, 1536))
        self.assertEqual(tuple(pack.rotary.shape), (1, 1, 42, 8))
        self.assertTrue(torch.equal(pack.hidden[:, :21], pack.source_hidden))
        self.assertTrue(torch.equal(pack.hidden[:, 21:], pack.target_hidden))
        receipt = pack.receipt()
        self.assertEqual(receipt["patch_call_order"], ["source:1", "target:0"])
        self.assertTrue(receipt["negative_action_share_same_pack_object"])

    def test_route_derives_target_suffix_and_sp4_selectors(self) -> None:
        transformer = FakeTransformer()
        pack = runner.build_vonly_pack(
            transformer,
            torch.zeros(1, 16, 21, 1, 1),
            torch.ones(1, 16, 21, 1, 1),
        )
        selectors = []
        for rank in range(4):
            route = runner.make_route(pack, sp_rank=rank, sigma=0.5, atlas=atlas())
            self.assertEqual(route.branch_name, "V")
            self.assertEqual(route.condition_tokens, pack.condition_tokens)
            selectors.extend(route.local_target_selector(device=torch.device("cpu")).tolist())
        padded = ((pack.total_tokens + 3) // 4) * 4
        self.assertEqual(selectors[: pack.condition_tokens], [False] * 21)
        self.assertEqual(
            selectors[pack.condition_tokens : pack.total_tokens], [True] * 21
        )
        self.assertEqual(selectors[pack.total_tokens : padded], [False] * (padded - 42))

    def test_pack_rejects_non_exact81_or_reference_like_shape(self) -> None:
        transformer = FakeTransformer()
        with self.assertRaises(runner.IdentityRebinderStructuralCanaryError):
            runner.build_vonly_pack(
                transformer,
                torch.zeros(1, 16, 1, 1, 1),
                torch.zeros(1, 16, 1, 1, 1),
            )
        self.assertEqual(transformer.calls, [])

    def test_live_timestep_cells_preserve_int64_exact_schedule_values(self) -> None:
        scheduler = type("Scheduler", (), {})()
        scheduler.timesteps = torch.tensor(
            runner.sigma_strata.PINNED_TIMESTEPS, dtype=torch.int64
        )
        selected = runner.live_timestep_cells(
            scheduler, indices=(0, 29, 38), device=torch.device("cpu")
        )
        self.assertEqual(set(selected), {0, 29, 38})
        for index, value in selected.items():
            self.assertEqual(value.dtype, torch.int64)
            self.assertEqual(tuple(value.shape), (1,))
            self.assertEqual(int(value.item()), runner.sigma_strata.PINNED_TIMESTEPS[index])

    def test_live_timestep_cells_reject_float_or_changed_schedule(self) -> None:
        scheduler = type("Scheduler", (), {})()
        scheduler.timesteps = torch.tensor(
            runner.sigma_strata.PINNED_TIMESTEPS, dtype=torch.float32
        )
        with self.assertRaises(runner.IdentityRebinderStructuralCanaryError):
            runner.live_timestep_cells(scheduler, indices=(0,), device="cpu")
        scheduler.timesteps = torch.tensor(
            runner.sigma_strata.PINNED_TIMESTEPS, dtype=torch.int64
        )
        scheduler.timesteps[29] += 1
        with self.assertRaises(runner.IdentityRebinderStructuralCanaryError):
            runner.live_timestep_cells(scheduler, indices=(29,), device="cpu")


class ReceiptPublicationTests(unittest.TestCase):
    def test_create_only_no_overwrite_and_mode_0444(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "receipt.json"
            runner.write_receipt_create_only(path, {"z": 1, "a": False})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            self.assertEqual(json.loads(path.read_text("ascii")), {"a": False, "z": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write_receipt_create_only(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)

    def test_retained_output_dir_fd_openat_publication(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "fresh-output"
            directory_fd, identity = runner._create_output_dir(output)
            try:
                self.assertEqual(
                    (os.fstat(directory_fd).st_dev, os.fstat(directory_fd).st_ino),
                    identity,
                )
                runner.write_receipt_create_only(
                    output / "receipt.json",
                    {"directory": {"st_dev": identity[0], "st_ino": identity[1]}},
                    directory_fd=directory_fd,
                    expected_directory_identity=identity,
                )
                self.assertEqual(stat.S_IMODE((output / "receipt.json").stat().st_mode), 0o444)
                self.assertEqual(
                    (output.lstat().st_dev, output.lstat().st_ino), identity
                )
            finally:
                os.close(directory_fd)

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW")
    def test_create_only_refuses_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)
            target = parent / "target.json"
            link = parent / "receipt.json"
            target.write_text("sentinel", encoding="ascii")
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                runner.write_receipt_create_only(link, {"replacement": True})
            self.assertEqual(target.read_text("ascii"), "sentinel")

    def test_runtime_closure_requires_exact_all_python_file_set(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "method"
            root.mkdir()
            source = root / "a.py"
            source.write_text("value = 1\n", encoding="ascii")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = Path(temporary) / "closure.json"
            manifest.write_bytes(
                runner.canonical_json_bytes(
                    {
                        "schema_version": runner.RUNTIME_CLOSURE_SCHEMA,
                        "root": "methods/bernini_action_editing",
                        "selection": runner.RUNTIME_CLOSURE_SELECTION,
                        "files": {"a.py": digest},
                    }
                )
            )
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            result = runner.load_runtime_support_closure(
                manifest,
                expected_sha256=manifest_sha,
                source_archive_sha256="8" * 64,
                root=root,
                required=("a.py",),
            )
            self.assertEqual(result["files"], {"a.py": digest})
            self.assertTrue(result["exact_file_set_verified"])
            self.assertTrue(result["archive_manifest_exact"])
            self.assertFalse(
                result["authoritative_repository_complete_source_tree_claimed"]
            )
            (root / "extra.py").write_text("extra = True\n", encoding="ascii")
            with self.assertRaises(runner.IdentityRebinderStructuralCanaryError):
                runner.load_runtime_support_closure(
                    manifest,
                    expected_sha256=manifest_sha,
                    source_archive_sha256="8" * 64,
                    root=root,
                    required=("a.py",),
                )


class PreHighMemoryReleaseTests(unittest.TestCase):
    def test_cpu_two_step_adamw_cleanup_preserves_parameters_and_forward(self) -> None:
        torch.manual_seed(20260810)
        module = torch.nn.Sequential(
            torch.nn.Linear(5, 7, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(7, 3, bias=False),
        )
        inputs = torch.randn(4, 5)
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=1.0e-3, weight_decay=0.0
        )
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            module(inputs).square().mean().backward()
            optimizer.step()
        named = tuple(module.named_parameters())
        parameter_digest_before = runner._parameter_digest(named)
        forward_before = module(inputs).detach().clone()

        audit = runner.cleanup_adamw_after_two_steps(
            optimizer,
            named,
            forward_probe=lambda: module(inputs),
        )

        self.assertEqual(audit["observed_step_values"], [2])
        self.assertEqual(audit["parameter_device_type"], "cpu")
        self.assertGreater(audit["optimizer_state_tensor_bytes_before"], 0)
        self.assertGreater(audit["gradient_tensor_bytes_before"], 0)
        self.assertEqual(
            audit["optimizer_state_entries_before"], len(named)
        )
        self.assertEqual(audit["optimizer_state_entries_after_clear"], 0)
        self.assertTrue(audit["zero_grad_set_to_none_before_state_clear"])
        self.assertTrue(audit["all_gradients_none_after_zero_grad"])
        self.assertTrue(audit["parameters_byte_exact_after_cleanup"])
        self.assertTrue(audit["forward_probe_executed"])
        self.assertTrue(audit["forward_byte_exact_after_cleanup"])
        self.assertEqual(
            audit["forward_raw_identity_before"],
            audit["forward_raw_identity_after"],
        )
        self.assertEqual(runner._parameter_digest(named), parameter_digest_before)
        self.assertTrue(torch.equal(module(inputs).detach(), forward_before))
        self.assertTrue(all(parameter.grad is None for _, parameter in named))
        self.assertFalse(optimizer.state)

    def test_cleanup_rejects_adamw_before_exactly_two_completed_steps(self) -> None:
        module = torch.nn.Linear(3, 2, bias=False)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
        optimizer.zero_grad(set_to_none=True)
        module(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        with self.assertRaises(runner.IdentityRebinderStructuralCanaryError):
            runner.cleanup_adamw_after_two_steps(
                optimizer, tuple(module.named_parameters())
            )
        self.assertTrue(optimizer.state)
        self.assertTrue(
            all(parameter.grad is not None for parameter in module.parameters())
        )

    def test_allocator_telemetry_requires_per_rank_release_and_headroom(self) -> None:
        before = {
            "allocated_bytes": 900,
            "reserved_bytes": 950,
            "maximum_allocated_bytes": 1000,
            "device_free_bytes": 50,
            "device_total_bytes": 1000,
        }
        after = {
            "allocated_bytes": 600,
            "reserved_bytes": 650,
            "maximum_allocated_bytes": 1000,
            "device_free_bytes": 350,
            "device_total_bytes": 1000,
        }
        result = runner.validate_pre_high_memory_telemetry(
            before, after, minimum_free_bytes=300
        )
        self.assertEqual(result["allocated_bytes_released"], 300)
        self.assertEqual(result["reserved_bytes_released"], 300)
        self.assertEqual(result["device_free_bytes_gained"], 300)
        self.assertTrue(result["per_rank_live_allocation_release_verified"])
        self.assertTrue(result["per_rank_device_free_headroom_verified"])
        for bad_after in (
            {**after, "allocated_bytes": before["allocated_bytes"]},
            {**after, "reserved_bytes": before["reserved_bytes"] + 1},
            {**after, "device_free_bytes": before["device_free_bytes"]},
            {**after, "device_free_bytes": 299},
        ):
            with self.subTest(bad_after=bad_after):
                with self.assertRaises(
                    runner.IdentityRebinderStructuralCanaryError
                ):
                    runner.validate_pre_high_memory_telemetry(
                        before, bad_after, minimum_free_bytes=300
                    )

    def test_release_order_precedes_unweakened_high_graph_forward(self) -> None:
        cleanup_source = inspect.getsource(runner.cleanup_adamw_after_two_steps)
        self.assertLess(
            cleanup_source.index("optimizer.zero_grad(set_to_none=True)"),
            cleanup_source.index("optimizer.state.clear()"),
        )
        self.assertLess(
            cleanup_source.index("optimizer.state.clear()"),
            cleanup_source.index("parameter_digest_after ="),
        )

        source = inspect.getsource(runner.main)
        pack_receipt = source.index("pack_receipt =")
        latent_release = source.index("del full_source_latent, target_latent")
        baseline_release = source.index("baselines.pop(MID_INDEX)")
        high_detach = source.index("tokens=last_atlas.tokens.detach().contiguous()")
        optimizer_release = source.index("cleanup_adamw_after_two_steps(")
        graph_release = source.index(
            "del source_frames, source_tensor, step_atlases, last_atlas"
        )
        empty_cache = source.index("torch.cuda.empty_cache()", optimizer_release)
        high_forward = source.index("high_losses, high_pair, _ = _run_pair(")
        self.assertLess(pack_receipt, latent_release)
        self.assertLess(latent_release, baseline_release)
        self.assertLess(baseline_release, high_detach)
        self.assertLess(high_detach, optimizer_release)
        self.assertLess(optimizer_release, graph_release)
        self.assertLess(graph_release, empty_cache)
        self.assertLess(empty_cache, high_forward)
        high_call = source[high_forward : source.index("high_rows =", high_forward)]
        self.assertIn("fixed_atlas=high_atlas", high_call)
        self.assertIn("baselines=baselines[HIGH_INDEX]", high_call)
        self.assertIn("backward_each=False", high_call)
        self.assertIn('"pre_high_memory_release": pre_high_memory_release', source)
        self.assertIn(
            '"allocator_telemetry_cross_rank_equality_required": False', source
        )
        self.assertIn('"semantic_action_success": False', source)
        self.assertIn('"official_sampler_parity": False', source)

        pair_source = inspect.getsource(runner._run_pair)
        self.assertIn("with torch.enable_grad():", pair_source)
        self.assertIn("not prediction.requires_grad", pair_source)
        self.assertIn("raw_storage_sha256_exact_vs_uninstalled", pair_source)
        self.assertIn("del target_prediction, prediction, loss", pair_source)
        self.assertLess(
            pair_source.index("losses.append(float(loss.detach().item()))"),
            pair_source.index("del target_prediction, prediction, loss"),
        )
        self.assertLess(
            pair_source.index("del target_prediction, prediction, loss"),
            pair_source.index("if len(set(pack_ids)) != 1"),
        )
        self.assertIn("module.adapter_delta(hidden.detach())", inspect.getsource(runner.DirectResidualAudit))

    def test_forward_only_pair_releases_first_branch_graph_before_second_forward(
        self,
    ) -> None:
        class Marker:
            pass

        class GraphMarker(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, marker):
                ctx.marker = marker
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient, None

        class FakeHandle:
            @contextmanager
            def route(self, _route):
                yield

        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        hidden = hidden.clone().requires_grad_(True)
        pack = SimpleNamespace(hidden=hidden, condition_tokens=3)
        fixed_atlas = object()
        first_marker = Marker()
        first_marker_ref = weakref.ref(first_marker)
        calls = 0

        def fake_forward(_diffusion, observed_pack, *, timestep, embeds):
            nonlocal calls, first_marker
            self.assertIs(observed_pack, pack)
            self.assertEqual(timestep, "t")
            if calls == 0:
                marker = first_marker
                first_marker = None
            else:
                gc.collect()
                self.assertIsNone(
                    first_marker_ref(),
                    "first forward graph survived into the second native forward",
                )
                marker = Marker()
            calls += 1
            return GraphMarker.apply(observed_pack.hidden, marker)

        with mock.patch.object(runner, "make_route", return_value=object()), mock.patch.object(
            runner, "_forward_v", side_effect=fake_forward
        ):
            losses, audit, atlases = runner._run_pair(
                diffusion=object(),
                handle=FakeHandle(),
                pack=pack,
                atlas_factory=None,
                fixed_atlas=fixed_atlas,
                sp_rank=0,
                sigma=0.5,
                timestep="t",
                negative="negative",
                action="action",
                baselines=None,
                backward_each=False,
            )
        self.assertEqual(calls, 2)
        self.assertEqual(len(losses), 2)
        self.assertEqual(atlases, [fixed_atlas, fixed_atlas])
        self.assertTrue(audit["pack_hidden_leaf_graph_participation"])
        self.assertIsNone(hidden.grad)


class ClaimBoundaryTests(unittest.TestCase):
    def test_atlas_only_clamp_handles_bicubic_overshoot_without_mutating_vae_input(
        self,
    ) -> None:
        source = torch.zeros(1, 3, 81, 2, 2, dtype=torch.float32).contiguous()
        source[0, 0, 0, 0, 0] = -1.125
        source[0, 1, 1, 0, 1] = 1.25
        source[0, 2, 80, 1, 1] = 0.625
        before = source.clone()
        atlas_frames, audit = runner.prepare_atlas_source_frames(
            source, device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(source, before))
        self.assertEqual(tuple(atlas_frames.shape), (1, 81, 3, 2, 2))
        self.assertTrue(atlas_frames.is_contiguous())
        self.assertNotEqual(
            atlas_frames.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )
        # Lock the B,C,T,H,W -> B,T,C,H,W mapping.  A reshape has the same
        # output shape and value range but moves these sentinels to wrong axes.
        self.assertEqual(float(atlas_frames[0, 0, 0, 0, 0]), -1.0)
        self.assertEqual(float(atlas_frames[0, 1, 1, 0, 1]), 1.0)
        self.assertEqual(float(atlas_frames[0, 80, 2, 1, 1]), 0.625)
        self.assertEqual(float(atlas_frames.amin()), -1.0)
        self.assertEqual(float(atlas_frames.amax()), 1.0)
        self.assertEqual(audit["input_layout"], "B_C_T_H_W")
        self.assertEqual(audit["atlas_layout"], "B_T_C_H_W")
        self.assertEqual(
            audit["policy"],
            "atlas_view_only_clamp_closed_interval_minus1_plus1",
        )
        self.assertEqual(audit["preclamp_min_float_hex"], (-1.125).hex())
        self.assertEqual(audit["preclamp_max_float_hex"], (1.25).hex())
        self.assertEqual(audit["postclamp_min_float_hex"], (-1.0).hex())
        self.assertEqual(audit["postclamp_max_float_hex"], (1.0).hex())
        self.assertEqual(audit["max_abs_correction_float_hex"], (0.25).hex())
        self.assertEqual(audit["below_minus_one_count"], 1)
        self.assertEqual(audit["above_plus_one_count"], 1)
        self.assertEqual(audit["clipped_element_count"], 2)
        self.assertEqual(audit["total_element_count"], source.numel())
        self.assertFalse(audit["vae_source_tensor_clamped_or_replaced"])
        self.assertTrue(audit["postclamp_range_verified"])
        atlas_frames[0, 80, 2, 1, 1] = -0.5
        self.assertTrue(torch.equal(source, before))

    def test_atlas_only_clamp_rejects_nonfinite_source(self) -> None:
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_value=bad_value):
                source = torch.zeros(
                    1, 3, 81, 2, 2, dtype=torch.float32
                ).contiguous()
                source[0, 0, 0, 0, 0] = bad_value
                with self.assertRaises(
                    runner.IdentityRebinderStructuralCanaryError
                ):
                    runner.prepare_atlas_source_frames(
                        source, device=torch.device("cpu")
                    )

    def test_lazy_infer_lora_decoder_is_an_explicit_archive_dependency(self) -> None:
        self.assertIn("tools/materialize_vae.py", runner.REQUIRED_RUNTIME_SUPPORT)
        self.assertIn(
            "tools/build_renderer_dataset.py", runner.REQUIRED_RUNTIME_SUPPORT
        )

    def test_runner_does_not_use_incompatible_rv2v4_binder(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("NativeRV2VIdentityRouteBinder(", source)
        self.assertNotIn("build_native_rv2v_pack(", source)
        self.assertIn('"official_sampler_parity": False', source)
        self.assertIn('"native_binder_used": False', source)
        self.assertIn('"route_metadata_canary_constructed": True', source)
        self.assertIn("pinned_vendor_normalized_guidance_vjp_parity(", source)
        self.assertIn('"raw_full_source_atlas_encode_calls_per_rank": 4', source)
        self.assertIn('"atlas_encoder_optimizer_target": True', source)
        self.assertIn("timesteps[index : index + 1].to(device=device)", source)
        self.assertNotIn("dtype=torch.float32, device=device", source)
        self.assertIn('"trajectory_state_valid": False', source)
        self.assertIn('"official_full_sampler_parity_required_external": True', source)
        self.assertIn('"packed_raw_to_APG_adapter_chain_verified": False', source)
        self.assertIn("raw_storage_sha256_exact_vs_uninstalled", source)
        self.assertIn("pack.hidden.requires_grad_(True)", source)
        self.assertIn('"pack_hidden_leaf_graph_participation": True', source)
        self.assertIn(
            '"pack_hidden_gradient_checked_and_cleared_after_each_backward": True',
            source,
        )
        self.assertIn('"receipt_openat_directory_fd": True', source)
        self.assertNotIn("official sampler V pack", source)

    def test_no_oracle_channels_in_pack_builder(self) -> None:
        names = set(runner.build_vonly_pack.__code__.co_varnames)
        self.assertEqual(
            {"transformer", "source_latent", "target_latent"}.issubset(names), True
        )
        self.assertTrue(
            names.isdisjoint({"mask", "pose", "flow", "track", "target_video", "reward"})
        )


class LauncherContractTests(unittest.TestCase):
    def test_dual4_uses_all8_and_seals_cleanup_identity(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts"
            / "auh_run_identity_rebinder_gpu_structural_canary_dual4_all8_v1.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", launcher)
        self.assertIn("--nproc_per_node=4", launcher)
        self.assertIn("run_group dog 0,1,2,3", launcher)
        self.assertIn("run_group human 4,5,6,7", launcher)
        self.assertIn("task_scratch_identity=", launcher)
        self.assertIn("stat -Lc '%d:%i'", launcher)
        self.assertIn('[[ "${observed}" == "${task_scratch_identity}" ]]', launcher)
        self.assertGreaterEqual(
            launcher.count('[[ "${observed}" == "${task_scratch_identity}" ]]'), 3
        )
        self.assertIn('raise SystemExit("receipt parent dev:ino differs")', launcher)
        self.assertIn("--expected-native-field-probe-sha256", launcher)
        self.assertIn("GRAFT_REBINDER_SOURCE_ARCHIVE", launcher)
        self.assertIn("GRAFT_REBINDER_RUNTIME_CLOSURE_MANIFEST", launcher)
        self.assertIn("source archive/closure set differs", launcher)
        self.assertIn(runner.RUNTIME_CLOSURE_SCHEMA, launcher)
        self.assertIn(runner.RUNTIME_CLOSURE_SELECTION, launcher)
        self.assertIn("runtime closure lacks infer_lora lazy imports", launcher)
        self.assertIn("chmod -R a-w", launcher)
        self.assertIn("sealed infer_lora lazy-import preflight failed", launcher)
        self.assertIn("from tools import build_renderer_dataset, materialize_vae", launcher)
        self.assertIn('"tools/materialize_vae.py"', launcher)
        self.assertIn('"tools/build_renderer_dataset.py"', launcher)
        self.assertIn("--ack-structural-only-no-semantic-claim", launcher)
        self.assertNotIn("sbatch ", launcher)


if __name__ == "__main__":
    unittest.main()
