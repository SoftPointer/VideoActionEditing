from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action import TemporalMotionPlanPredictor, validate_action_config  # noqa: E402
from action.omni import (  # noqa: E402
    _load_special_token_payload,
    enable_action_lora,
)
from action.checkpoint_contract import (  # noqa: E402
    ACTION_ADAPTER_CHECKPOINT_FIELDS,
    OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
    OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS,
    action_activation_contract_record,
    special_token_layout_record,
)
from pact.lora import lora_state_dict  # noqa: E402
from tools.verify_action_real_ddp_smoke import (  # noqa: E402
    RealSmokeAuditError,
    parse_args,
    verify,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in module.state_dict().items()
    }


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)
        self.v = nn.Linear(4, 4)
        self.o = nn.Linear(4, 4)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.cross_attn = _Attention()


class _Wan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 1
        self.blocks = nn.ModuleList([_Block()])


class _TinyOfficialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wan_model = _Wan()


class VerifyActionRealDdpSmokeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, object]:
        sample_id = "clip001"
        source_revision = "a" * 40
        source_archive_digest = "d" * 64
        encoder_digest = "e" * 64
        materialized = root / "materialized"
        materialized.mkdir()
        manifest = materialized / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"sample_id": sample_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_digest = _sha256(manifest)
        materialization = {
            "schema_version": "omnivideo2-action-materialization-receipt-v2",
            "complete": True,
            "sample_count": 1,
            "preview_only": True,
            "training_authorized": False,
            "scientific_claim_authorized": False,
            "target_motion_tokens_usage": "planner_loss_only",
            "manifest_sha256": manifest_digest,
            "encoder_contract_sha256": encoder_digest,
            "temporal_mode": "full_81_25fps",
            "temporal_indices": list(range(81)),
            "temporal_sampling_policy": (
                "all_frames_in_order_no_temporal_subsampling"
            ),
            "temporal_subsampled": False,
            "source_frame_count": 81,
            "materialized_frame_count": 81,
            "source_fps": 25.0,
            "materialized_fps": 25.0,
            "spatial_profile": "full_480p",
            "landscape_bucket_hw": [480, 832],
            "portrait_bucket_hw": [832, 480],
        }
        materialization["receipt_digest"] = _object_sha256(materialization)
        (materialized / "materialization.json").write_text(
            json.dumps(materialization) + "\n", encoding="utf-8"
        )
        materialization_digest = _sha256(materialized / "materialization.json")
        (materialized / "verification.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "sample_id": sample_id,
                    "preview_only": True,
                    "scientific_quality_not_tested": True,
                    "source_revision": source_revision,
                    "source_archive_sha256": source_archive_digest,
                    "temporal_mode": "full_81_25fps",
                    "spatial_profile": "full_480p",
                    "temporal_indices_verified": True,
                    "temporal_subsampled": False,
                    "materialization_sha256": materialization_digest,
                    "source_latent_shape": [16, 21, 60, 104],
                    "target_latent_shape": [16, 21, 60, 104],
                    "target_motion_tokens_usage": "planner_loss_only",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        config_source = ROOT / "configs" / "marp_81f_fullres_real_one_step.json"
        config = validate_action_config(
            json.loads(config_source.read_text(encoding="utf-8"))
        ).to_dict()
        config_path = root / "exact_action_config.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_digest = _sha256(config_path)

        omnivideo_root = root / "Omni-Video"
        omnivideo_root.mkdir()
        checkpoint_dir = root / "OmniVideo2-1.3B"
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True)
        base_checkpoint = transformer_dir / "pytorch_model.pt"
        base_checkpoint.write_bytes(b"tiny official checkpoint identity\n")
        base_digest = _sha256(base_checkpoint)
        special_tokens = checkpoint_dir / "special_tokens.pkl"
        with special_tokens.open("wb") as handle:
            pickle.dump(
                {
                    "<img_st>": torch.zeros(6, 4096, dtype=torch.bfloat16),
                    "<img_ed>": torch.zeros(6, 4096, dtype=torch.bfloat16),
                    "<ipl_st>": torch.zeros(7, 4096, dtype=torch.bfloat16),
                    "<ipl_ed>": torch.zeros(7, 4096, dtype=torch.bfloat16),
                    "<prp_st>": torch.zeros(7, 4096, dtype=torch.bfloat16),
                    "<prp_ed>": torch.zeros(7, 4096, dtype=torch.bfloat16),
                },
                handle,
            )
        special_digest = _sha256(special_tokens)
        torch.manual_seed(17)
        adapted = _TinyOfficialModel()
        checked_config = validate_action_config(config)
        injected, _parameters = enable_action_lora(
            adapted,
            scope=checked_config.lora.scope,
            rank=checked_config.lora.rank,
            alpha=checked_config.lora.alpha,
            dropout=checked_config.lora.dropout,
        )
        with torch.no_grad():
            for module in adapted.modules():
                if hasattr(module, "lora_B"):
                    module.lora_B.weight.normal_()
        planner = TemporalMotionPlanPredictor(
            checked_config.planner.num_tokens,
            input_dim=checked_config.planner.input_dim,
            hidden_dim=checked_config.planner.hidden_dim,
            depth=checked_config.planner.depth,
        )
        adapter_payload = {
            "format": "marp-omnivideo2-action-adapters-v2",
            "step": 1,
            "validated_config": copy.deepcopy(config),
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "base_checkpoint_sha256": base_digest,
            "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
            "special_tokens_sha256": special_digest,
            "special_token_rows": 26,
            "special_token_serialized_rows": (
                OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
            ),
            "special_token_layout": special_token_layout_record(),
            "encoder_contract_sha256": encoder_digest,
            "world_size": 4,
            "preview_only": True,
            "temporal_smoke_only": False,
            "production_claim_forbidden": True,
            "source_revision": source_revision,
            "source_archive_sha256": source_archive_digest,
            "activation_contract": action_activation_contract_record(),
            "target_motion_tokens_used_by_renderer": False,
            "base_weights_saved": False,
            "lora_modules": injected,
            "lora_state_dict": lora_state_dict(adapted),
            "motion_planner_state_dict": _cpu_state(planner),
            "rank0_cpu_rng_state": torch.get_rng_state(),
            "rank0_device_rng_state": torch.arange(16, dtype=torch.uint8),
        }
        self.assertEqual(set(adapter_payload), ACTION_ADAPTER_CHECKPOINT_FIELDS)

        run_root = root / "run"
        run_root.mkdir()
        adapter = run_root / "action_adapters_final_step_00000001.pt"
        torch.save(adapter_payload, adapter)
        context_budget = {
            "source_shape": [1, 16, 21, 60, 104],
            "nonvisual_tokens": 420,
            "visual_tokens": 8190,
            "total_tokens": 8610,
            "budget_tokens": 9216,
            "fixed_budget_padding_tokens": 606,
            "fits": True,
            "source_truncated": False,
            "sample_id": sample_id,
            "task_type": "action_edit",
            "context_padding_mode": "fixed_budget",
            "effective_wan_text_len": 9216,
            "effective_padding_tokens": 606,
        }
        context_preflight = {
            **context_budget,
            "row_index": 0,
            "latent_shape": [16, 21, 60, 104],
            "raw_video_frames": 81,
            "materialized_video_frames": 81,
            "materialized_video_fps": 25.0,
            "latent_frames": 21,
        }
        context_path = run_root / "context_preflight.jsonl"
        context_path.write_text(
            json.dumps(context_preflight, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run = {
            "format": "marp-omnivideo2-action-run-v2",
            "validated_config": copy.deepcopy(config),
            "config": str(config_path.resolve()),
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "dataset_rows": 1,
            "preview_only": True,
            "production_claim_forbidden": True,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "base_checkpoint_sha256": base_digest,
            "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
            "special_tokens_sha256": special_digest,
            "special_token_rows": 26,
            "special_token_serialized_rows": (
                OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
            ),
            "special_token_layout": special_token_layout_record(),
            "encoder_contract_sha256": encoder_digest,
            "omnivideo_root": str(omnivideo_root.resolve()),
            "world_size": 4,
            "source_revision": source_revision,
            "source_archive_sha256": source_archive_digest,
            "target_motion_tokens_used_by_renderer": False,
            "raw_video_num_frames": 81,
            "raw_video_fps": 25.0,
            "materialized_video_num_frames": 81,
            "materialized_video_fps": 25.0,
            "temporal_mode": "full_81_25fps",
            "spatial_profile": "full_480p",
            "expected_latent_frames": 21,
            "allowed_latent_hw": [[60, 104], [104, 60]],
            "context_padding_mode": "fixed_budget",
            "context_budget_tokens": 9216,
            "official_context_padding_is_unmasked": True,
            "temporal_smoke_only": False,
            "source_temporal_compression_allowed": False,
            "source_truncation_allowed": False,
            "mask_or_tube_inputs": False,
            "context_preflight": context_path.name,
            "context_preflight_sha256": _sha256(context_path),
            "context_rows_preflighted": 1,
        }
        runtime = [
            {
                "rank": rank,
                "optimizer_step": 1,
                "microbatches": 1,
                "isolated_optimizer_window_seconds": 1.25 + rank / 10,
                "peak_memory_allocated_bytes": 1_000_000 + rank,
                "peak_memory_reserved_bytes": 2_000_000 + rank,
            }
            for rank in range(4)
        ]
        done = {
            "format": "marp-omnivideo2-action-training-done-v2",
            "complete": True,
            "optimizer_steps": 1,
            "world_size": 4,
            "source_revision": source_revision,
            "source_archive_sha256": source_archive_digest,
            "final_adapter_checkpoint": adapter.name,
            "final_adapter_sha256": _sha256(adapter),
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "base_checkpoint_sha256": base_digest,
            "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
            "special_tokens_sha256": special_digest,
            "special_token_rows": 26,
            "special_token_serialized_rows": (
                OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
            ),
            "special_token_layout": special_token_layout_record(),
            "encoder_contract_sha256": encoder_digest,
            "preview_only": True,
            "temporal_smoke_only": False,
            "production_claim_forbidden": True,
            "target_motion_tokens_used_by_renderer": False,
            "observed_task_types": ["action_edit"],
            "lora_module_count": len(injected),
            "runtime_maxima_all_ranks": [
                {
                    "rank": row["rank"],
                    "optimizer_windows": 1,
                    "max_isolated_optimizer_window_seconds": row[
                        "isolated_optimizer_window_seconds"
                    ],
                    "max_peak_memory_allocated_bytes": row[
                        "peak_memory_allocated_bytes"
                    ],
                    "max_peak_memory_reserved_bytes": row[
                        "peak_memory_reserved_bytes"
                    ],
                }
                for row in runtime
            ],
            "first_optimizer_step_runtime_all_ranks": runtime,
            "one_step_duration_seconds_all_ranks": {
                str(row["rank"]): row["isolated_optimizer_window_seconds"]
                for row in runtime
            },
            "rank0_peak_memory_allocated_bytes": runtime[0][
                "peak_memory_allocated_bytes"
            ],
            "rank0_peak_memory_reserved_bytes": runtime[0][
                "peak_memory_reserved_bytes"
            ],
        }
        metric = {
            "step": 1,
            "loss": {
                "total_world_mean": 1.0,
                "velocity_world_mean": 0.8,
                "motion_plan_world_mean": 0.2,
            },
            "gradient_groups_rank0": {
                "action_lora": {"l2_norm": 1.0},
                "motion_planner": {"l2_norm": 1.0},
            },
            "source_budgets_rank0": [
                {
                    **context_budget,
                    "sample_index": 0,
                    "compressed": False,
                    "first_frame_exact": True,
                    "original_visual_tokens": 8190,
                    "output_visual_tokens": 8190,
                    "output_total_tokens": 8610,
                }
            ],
            "runtime_all_ranks": runtime,
            "task_records_all_ranks": [
                {
                    "rank": rank,
                    "sample_id": sample_id,
                    "task_type": "action_edit",
                    "lora_gate": 1.0,
                    "plan_gate": 1.0,
                }
                for rank in range(4)
            ],
        }
        (run_root / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_root / "done.json").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_root / "metrics.jsonl").write_text(
            json.dumps(metric, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "sample_id": sample_id,
            "source_revision": source_revision,
            "materialized": materialized,
            "run_root": run_root,
            "omnivideo_root": omnivideo_root,
            "checkpoint_dir": checkpoint_dir,
            "base_checkpoint": base_checkpoint,
            "special_tokens": special_tokens,
            "adapter": adapter,
            "adapter_payload": adapter_payload,
            "done": done,
            "injected": injected,
        }

    @staticmethod
    def _rewrite_adapter(fixture: dict[str, object], payload: dict) -> None:
        adapter = fixture["adapter"]
        assert isinstance(adapter, Path)
        torch.save(payload, adapter)
        done = copy.deepcopy(fixture["done"])
        assert isinstance(done, dict)
        done["final_adapter_sha256"] = _sha256(adapter)
        run_root = fixture["run_root"]
        assert isinstance(run_root, Path)
        (run_root / "done.json").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _verify(self, fixture: dict[str, object], *, model_loader):
        def fixture_special_token_loader(
            checkpoint_dir, *, dtype, device, required
        ):
            path = Path(checkpoint_dir) / "special_tokens.pkl"
            if not path.is_file():
                if required:
                    raise FileNotFoundError(path)
                return None, 0, None
            payload = path.read_bytes()
            return _load_special_token_payload(
                payload,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                dtype=dtype,
                device=device,
            )

        return verify(
            fixture["materialized"],
            fixture["run_root"],
            fixture["omnivideo_root"],
            fixture["checkpoint_dir"],
            expected_sample_id=fixture["sample_id"],
            expected_world_size=4,
            expected_source_revision=fixture["source_revision"],
            model_loader=model_loader,
            special_token_loader=fixture_special_token_loader,
            _test_only_allow_unpinned_checkpoint=True,
        )

    def test_reconstructs_clean_tiny_base_and_strictly_reloads_both_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            calls = []

            def loader(omnivideo_root, checkpoint_dir, config):
                calls.append((omnivideo_root, checkpoint_dir, config))
                return _TinyOfficialModel(), object(), fixture["base_checkpoint"]

            summary = self._verify(fixture, model_loader=loader)
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary["status"], "verified")
            self.assertTrue(summary["official_model_reconstructed"])
            self.assertTrue(summary["adapter_strictly_reloaded"])
            self.assertEqual(summary["temporal_mode"], "full_81_25fps")
            self.assertEqual(summary["materialized_video_num_frames"], 81)
            self.assertEqual(summary["special_token_rows"], 26)
            self.assertEqual(len(summary["runtime_all_ranks"]), 4)
            self.assertEqual(summary["lora_module_count"], 8)
            self.assertEqual(summary["lora_injected_modules"], fixture["injected"])
            self.assertEqual(
                summary["base_checkpoint_sha256"],
                _sha256(fixture["base_checkpoint"]),
            )

    def test_default_verifier_rejects_unpinned_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            with self.assertRaisesRegex(
                RealSmokeAuditError, "pinned checkpoint contract"
            ):
                verify(
                    fixture["materialized"],
                    fixture["run_root"],
                    fixture["omnivideo_root"],
                    fixture["checkpoint_dir"],
                    expected_sample_id=fixture["sample_id"],
                    expected_world_size=4,
                    expected_source_revision=fixture["source_revision"],
                )

    def test_rejects_checkpoint_unknown_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            payload = copy.deepcopy(fixture["adapter_payload"])
            payload["unexpected_base_state"] = torch.zeros(1)
            self._rewrite_adapter(fixture, payload)
            with self.assertRaisesRegex(RealSmokeAuditError, "closed schema"):
                self._verify(
                    fixture,
                    model_loader=lambda _root, _checkpoint, _config: (
                        _TinyOfficialModel(),
                        object(),
                        fixture["base_checkpoint"],
                    ),
                )

    def test_rejects_checkpoint_injected_module_list_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            payload = copy.deepcopy(fixture["adapter_payload"])
            payload["lora_modules"] = list(reversed(payload["lora_modules"]))
            self._rewrite_adapter(fixture, payload)
            with self.assertRaisesRegex(
                RealSmokeAuditError, "injected module list differs"
            ):
                self._verify(
                    fixture,
                    model_loader=lambda _root, _checkpoint, _config: (
                        _TinyOfficialModel(),
                        object(),
                        fixture["base_checkpoint"],
                    ),
                )

    def test_strict_lora_loader_rejects_unexpected_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            payload = copy.deepcopy(fixture["adapter_payload"])
            payload["lora_state_dict"]["unexpected.lora_A.weight"] = torch.zeros(1)
            self._rewrite_adapter(fixture, payload)
            with self.assertRaisesRegex(
                RealSmokeAuditError, "strict load_lora_state_dict failed"
            ):
                self._verify(
                    fixture,
                    model_loader=lambda _root, _checkpoint, _config: (
                        _TinyOfficialModel(),
                        object(),
                        fixture["base_checkpoint"],
                    ),
                )

    def test_strict_planner_loader_rejects_missing_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            payload = copy.deepcopy(fixture["adapter_payload"])
            payload["motion_planner_state_dict"].pop("motion_queries")
            self._rewrite_adapter(fixture, payload)
            with self.assertRaisesRegex(RealSmokeAuditError, "strict planner load failed"):
                self._verify(
                    fixture,
                    model_loader=lambda _root, _checkpoint, _config: (
                        _TinyOfficialModel(),
                        object(),
                        fixture["base_checkpoint"],
                    ),
                )

    def test_base_digest_mismatch_fails_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture["base_checkpoint"].write_bytes(b"tampered base\n")
            called = False

            def forbidden_loader(_root, _checkpoint, _config):
                nonlocal called
                called = True
                raise AssertionError("must fail before model construction")

            with self.assertRaisesRegex(
                RealSmokeAuditError, "base checkpoint digest differs"
            ):
                self._verify(fixture, model_loader=forbidden_loader)
            self.assertFalse(called)

    def test_rejects_same_total_but_wrong_special_token_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            special_tokens = fixture["special_tokens"]
            assert isinstance(special_tokens, Path)
            with special_tokens.open("rb") as handle:
                value = pickle.load(handle)
            value["<img_st>"] = torch.zeros(5, 4096, dtype=torch.bfloat16)
            value["<img_ed>"] = torch.zeros(7, 4096, dtype=torch.bfloat16)
            with special_tokens.open("wb") as handle:
                pickle.dump(value, handle)
            run_path = fixture["run_root"] / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["special_tokens_sha256"] = _sha256(special_tokens)
            run_path.write_text(
                json.dumps(run, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            called = False

            def forbidden_loader(_root, _checkpoint, _config):
                nonlocal called
                called = True
                raise AssertionError("must fail before model construction")

            with self.assertRaisesRegex(
                RealSmokeAuditError, "special token.*must be"
            ):
                self._verify(fixture, model_loader=forbidden_loader)
            self.assertFalse(called)

    def test_rejects_missing_runtime_rank_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            metrics_path = fixture["run_root"] / "metrics.jsonl"
            metric = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric["runtime_all_ranks"].pop()
            metrics_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
            called = False

            def forbidden_loader(_root, _checkpoint, _config):
                nonlocal called
                called = True
                raise AssertionError("must fail before model construction")

            with self.assertRaisesRegex(
                RealSmokeAuditError, "exactly one row per rank"
            ):
                self._verify(fixture, model_loader=forbidden_loader)
            self.assertFalse(called)

    def test_rejects_zero_runtime_peak_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            metrics_path = fixture["run_root"] / "metrics.jsonl"
            metric = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric["runtime_all_ranks"][2]["peak_memory_allocated_bytes"] = 0
            metrics_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
            called = False

            def forbidden_loader(_root, _checkpoint, _config):
                nonlocal called
                called = True
                raise AssertionError("must fail before model construction")

            with self.assertRaisesRegex(RealSmokeAuditError, "peak memory is invalid"):
                self._verify(fixture, model_loader=forbidden_loader)
            self.assertFalse(called)

    def test_rejects_41_frame_latent_under_81_frame_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            verification_path = fixture["materialized"] / "verification.json"
            value = json.loads(verification_path.read_text(encoding="utf-8"))
            value["source_latent_shape"] = [16, 11, 60, 104]
            value["target_latent_shape"] = [16, 11, 60, 104]
            verification_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            called = False

            def forbidden_loader(_root, _checkpoint, _config):
                nonlocal called
                called = True
                raise AssertionError("must fail before model construction")

            with self.assertRaisesRegex(
                RealSmokeAuditError, "latent geometry differs"
            ):
                self._verify(fixture, model_loader=forbidden_loader)
            self.assertFalse(called)

    def test_checkpoint_config_mismatch_fails_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            payload = copy.deepcopy(fixture["adapter_payload"])
            payload["validated_config"]["lora"]["rank"] = 2
            self._rewrite_adapter(fixture, payload)
            called = False

            def forbidden_loader(_root, _checkpoint, _config):
                nonlocal called
                called = True
                raise AssertionError("must fail before model construction")

            with self.assertRaisesRegex(
                RealSmokeAuditError, "validated_config differs"
            ):
                self._verify(fixture, model_loader=forbidden_loader)
            self.assertFalse(called)

    def test_cli_requires_official_roots(self) -> None:
        args = parse_args(
            [
                "--materialized-root",
                "materialized",
                "--run-output-dir",
                "run",
                "--omnivideo-root",
                "Omni-Video",
                "--checkpoint-dir",
                "OmniVideo2-1.3B",
                "--expected-sample-id",
                "clip001",
                "--expected-source-revision",
                "a" * 40,
            ]
        )
        self.assertEqual(args.omnivideo_root, Path("Omni-Video"))
        self.assertEqual(args.checkpoint_dir, Path("OmniVideo2-1.3B"))


if __name__ == "__main__":
    unittest.main()
