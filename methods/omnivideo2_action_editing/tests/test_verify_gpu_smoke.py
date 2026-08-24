from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from pact.lora import inject_lora, lora_scope_target_regex, lora_state_dict
from pact.router import PromptConditionedMaskRouter
from pact.training import DiffSynthWanTrainingScheduler, validate_training_config
from tools.verify_gpu_smoke import (
    SUMMARY_FORMAT,
    SmokeVerificationError,
    verify_gpu_smoke,
    verify_smoke_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        self.ffn = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))


class _Wan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 2
        self.num_layers = 30
        self.blocks = nn.ModuleList([_Block() for _ in range(self.num_layers)])


class _FakeOfficialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wan_model = _Wan()
        self.visual_context_adapter = nn.Linear(3, 3)
        self.vlm_norm = nn.LayerNorm(4)
        self.vlm_proj = nn.Linear(4, 4)


def _module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


class VerifyGpuSmokeTest(unittest.TestCase):
    def _config(self) -> dict:
        path = Path(__file__).resolve().parents[1] / "configs" / "pact_1_3b.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["training"].update(
            {
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_workers": 0,
                "max_steps": 1,
                "checkpoint_every": 1,
                "log_every": 1,
            }
        )
        return validate_training_config(value)

    def _fixture(self, root: Path) -> dict[str, object]:
        output = root / "run_output"
        output.mkdir()
        upstream = root / "Omni-Video"
        upstream.mkdir()
        checkpoint_dir = root / "OmniVideo2-1.3B"
        transformer = checkpoint_dir / "transformer"
        transformer.mkdir(parents=True)
        checkpoint = transformer / "pytorch_model.pt"
        checkpoint.write_bytes(b"synthetic official checkpoint identity\n")
        special_tokens = checkpoint_dir / "special_tokens.pkl"
        special_tokens.write_bytes(b"synthetic special token identity\n")
        manifest = root / "training_manifest.jsonl"
        manifest.write_text('{"atom_id":"atom-1"}\n', encoding="utf-8")

        config = self._config()
        config_digest = hashlib.sha256(b"exact original config bytes").hexdigest()
        checkpoint_digest = _sha256(checkpoint)
        special_digest = _sha256(special_tokens)
        manifest_digest = _sha256(manifest)

        torch.manual_seed(41)
        adapted_model = _FakeOfficialModel()
        names = inject_lora(
            adapted_model,
            lora_scope_target_regex(config["lora"]["scope"]),
            rank=config["lora"]["rank"],
            alpha=config["lora"]["alpha"],
            dropout=config["lora"]["dropout"],
        )
        with torch.no_grad():
            for module in adapted_model.modules():
                if hasattr(module, "lora_B"):
                    module.lora_B.weight.normal_()
        router = PromptConditionedMaskRouter(
            in_channels=2,
            prompt_dim=2048,
            hidden_channels=config["router"]["hidden_channels"],
            depth=config["router"]["depth"],
        )
        adapter_payload = {
            "format": "pact-omnivideo2-adapters-v2",
            "step": 1,
            "config_sha256": config_digest,
            "validated_config": copy.deepcopy(config),
            "base_checkpoint_sha256": checkpoint_digest,
            "manifest_sha256": manifest_digest,
            "special_tokens_sha256": special_digest,
            "encoder_contract_sha256": "5" * 64,
            "lora_modules": names,
            "lora_state_dict": lora_state_dict(adapted_model),
            "router_state_dict": _module_state(router),
            "visual_context_adapter_state_dict": _module_state(
                adapted_model.visual_context_adapter
            ),
            "vlm_norm_state_dict": _module_state(adapted_model.vlm_norm),
            "vlm_proj_state_dict": _module_state(adapted_model.vlm_proj),
        }
        adapter = output / "adapters_final_step_00000001.pt"
        torch.save(adapter_payload, adapter)
        adapter_digest = _sha256(adapter)

        run = {
            "format": "pact-omnivideo2-run-v2",
            "config": config,
            "config_sha256": config_digest,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": manifest_digest,
            "payload_root": None,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "checkpoint_sha256": checkpoint_digest,
            "special_tokens_sha256": special_digest,
            "encoder_contract_sha256": "5" * 64,
            "diffsynth_reference_revision": "ab12bf4119b7c9a23ff3359eefb41ba54a658ccb",
            "flow_master_dtype": "float32",
            "trainable_master_dtype": "float32",
            "base_model_dtype": "bfloat16",
            "base_weights_saved": False,
            "single_gpu": 0,
        }
        done = {
            "format": "pact-omnivideo2-training-done-v2",
            "optimizer_steps": 1,
            "final_adapter_checkpoint": adapter.name,
            "final_adapter_sha256": adapter_digest,
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "base_checkpoint_sha256": checkpoint_digest,
            "special_tokens_sha256": special_digest,
            "encoder_contract_sha256": "5" * 64,
            "diffsynth_reference_revision": "ab12bf4119b7c9a23ff3359eefb41ba54a658ccb",
            "flow_master_dtype": "float32",
            "trainable_master_dtype": "float32",
            "base_weights_saved": False,
            "lora_module_count": len(names),
            "trainable_model_adapter_parameters": 1000,
            "trainable_router_parameters": 500,
            "elapsed_seconds": 2.75,
            "torch_version": torch.__version__,
            "torch_hip_version": None,
            "accelerator_name": "synthetic-test-device",
            "accelerator_peak_memory_allocated_bytes": 1024,
            "accelerator_peak_memory_reserved_bytes": 2048,
        }
        flow_sample = DiffSynthWanTrainingScheduler(
            shift=config["flow"]["shift"]
        ).at(500, 1)
        raw_flow_total = 1.0 + 2.0 + 0.25 * 3.0 + 0.25 * 4.0
        expected_total = float(flow_sample.flow_weight) * raw_flow_total + 0.1 * 0.7
        metric = {
            "step": 1,
            "epoch": 0,
            "batch": 0,
            "atom_ids": ["atom-1"],
            "loss": {
                "total": expected_total,
                "velocity_edit": 1.0,
                "velocity_preserve": 2.0,
                "x0_boundary": 3.0,
                "x0_temporal_outside": 4.0,
                "router": 0.7,
                "router_bce": 0.4,
                "router_dice": 0.3,
            },
            "grad_norm": 0.75,
            "gradient_groups": {
                name: {
                    "parameter_tensors": 2,
                    "parameter_elements": 16,
                    "pre_clip_l2_norm": 0.25,
                }
                for name in ("lora", "visual_adapter", "vlm_projection", "router")
            },
            "timestep_id": flow_sample.timestep_id,
            "timestep_mean": float(flow_sample.timestep.mean()),
            "sigma_mean": float(flow_sample.sigma.mean()),
            "flow_training_weight": float(flow_sample.flow_weight),
            "learning_rates": {
                "lora_router": config["optimizer"]["learning_rate"],
                "pretrained_condition_adapters": config["optimizer"][
                    "pretrained_adapter_learning_rate"
                ],
            },
            "source_visual_tokens": 12,
            "elapsed_seconds": 2.5,
        }
        (output / "run.json").write_text(
            json.dumps(run, indent=2) + "\n", encoding="utf-8"
        )
        (output / "done.json").write_text(
            json.dumps(done, indent=2) + "\n", encoding="utf-8"
        )
        (output / "metrics.jsonl").write_text(
            json.dumps(metric) + "\n", encoding="utf-8"
        )
        return {
            "output": output,
            "upstream": upstream,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint": checkpoint,
            "manifest": manifest,
            "adapter": adapter,
            "run": run,
            "done": done,
            "metric": metric,
        }

    def test_cpu_fake_official_base_strictly_restores_real_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            loader_calls = []

            def fake_loader(upstream, checkpoint_dir, config):
                loader_calls.append((upstream, checkpoint_dir, config))
                return _FakeOfficialModel(), object(), fixture["checkpoint"]

            summary = verify_gpu_smoke(
                fixture["output"],
                fixture["upstream"],
                fixture["checkpoint_dir"],
                model_loader=fake_loader,
            )

            self.assertEqual(len(loader_calls), 1)
            self.assertEqual(summary["format"], SUMMARY_FORMAT)
            self.assertEqual(summary["status"], "verified")
            self.assertEqual(summary["optimizer_steps"], 1)
            self.assertEqual(summary["metrics_records"], 1)
            self.assertEqual(summary["lora_modules"], 300)
            self.assertTrue(summary["official_model_reconstructed"])
            self.assertTrue(summary["adapter_strictly_reloaded"])
            self.assertEqual(
                summary["final_adapter_sha256"], _sha256(fixture["adapter"])
            )

    def test_pure_artifact_check_rejects_two_metric_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            metric_line = json.dumps(fixture["metric"]) + "\n"
            (fixture["output"] / "metrics.jsonl").write_text(
                metric_line + metric_line, encoding="utf-8"
            )
            with self.assertRaisesRegex(
                SmokeVerificationError, "exactly one record, found 2"
            ):
                verify_smoke_artifacts(
                    fixture["output"], fixture["checkpoint_dir"]
                )

    def test_rejects_nonfinite_metric_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            metric = copy.deepcopy(fixture["metric"])
            metric["grad_norm"] = float("nan")
            (fixture["output"] / "metrics.jsonl").write_text(
                json.dumps(metric) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                SmokeVerificationError, "non-finite JSON constant"
            ):
                verify_smoke_artifacts(
                    fixture["output"], fixture["checkpoint_dir"]
                )

    def test_provenance_mismatch_fails_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            done = copy.deepcopy(fixture["done"])
            done["manifest_sha256"] = "f" * 64
            (fixture["output"] / "done.json").write_text(
                json.dumps(done) + "\n", encoding="utf-8"
            )
            loader_called = False

            def forbidden_loader(upstream, checkpoint_dir, config):
                nonlocal loader_called
                loader_called = True
                raise AssertionError("model loader must not run")

            with self.assertRaisesRegex(
                SmokeVerificationError,
                "done.manifest_sha256 differs from run.json provenance",
            ):
                verify_gpu_smoke(
                    fixture["output"],
                    fixture["upstream"],
                    fixture["checkpoint_dir"],
                    model_loader=forbidden_loader,
                )
            self.assertFalse(loader_called)

    def test_adapter_metadata_must_match_recorded_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            adapter = fixture["adapter"]
            payload = torch.load(adapter, map_location="cpu", weights_only=True)
            payload["step"] = 2
            torch.save(payload, adapter)
            done = copy.deepcopy(fixture["done"])
            done["final_adapter_sha256"] = _sha256(adapter)
            (fixture["output"] / "done.json").write_text(
                json.dumps(done) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                SmokeVerificationError, "final adapter metadata differs"
            ):
                verify_smoke_artifacts(
                    fixture["output"], fixture["checkpoint_dir"]
                )


if __name__ == "__main__":
    unittest.main()
