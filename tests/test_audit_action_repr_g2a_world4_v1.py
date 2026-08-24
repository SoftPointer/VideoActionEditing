#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - host dependent
    raise unittest.SkipTest("PyTorch unavailable") from error


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "audit_action_repr_g2a_world4_v1.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "audit_action_repr_g2a_world4_v1_test", SUBJECT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.03125))

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states + hidden_states * self.scale


class _Transformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.blocks = nn.ModuleList([_Block(width) for _ in range(30)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _sha(character: str) -> str:
    return character * 64


def _authority(*, projection_width: int = 5) -> subject.TargetG1Authority:
    branch_refs = {
        "correct": {
            "flow": {"path": "/not-opened/fc", "sha256": _sha("1")},
            "middle": {"path": "/not-opened/mc", "sha256": _sha("2")},
        },
        "temporal_shuffle": {
            "flow": {"path": "/not-opened/fs", "sha256": _sha("3")},
            "middle": {"path": "/not-opened/ms", "sha256": _sha("4")},
        },
        "reverse": {
            "flow": {"path": "/not-opened/fr", "sha256": _sha("5")},
            "middle": {"path": "/not-opened/mr", "sha256": _sha("6")},
        },
        "incomplete": {
            "flow": {"path": "/not-opened/fi", "sha256": _sha("7")},
            "middle": {"path": "/not-opened/mi", "sha256": _sha("8")},
        },
        "wrong_action": {
            "flow": {"path": "/not-opened/fw", "sha256": _sha("9")},
            "middle": {"path": "/not-opened/mw", "sha256": _sha("a")},
        },
    }
    placeholder = Path("/not-opened/receipt.json")
    return subject.TargetG1Authority(
        case_id="case01",
        admission_path=placeholder,
        admission_sha256=_sha("a"),
        evaluation_path=placeholder,
        evaluation_sha256=_sha("b"),
        flow_cohort_path=placeholder,
        flow_cohort_sha256=_sha("c"),
        middle_cohort_path=placeholder,
        middle_cohort_sha256=_sha("d"),
        source_video_sha256=_sha("e"),
        anchor_video_sha256s=(_sha("f"),),
        instruction_sha256=_sha("9"),
        sigmas=(0.85, 0.55, 0.20),
        projection_width=projection_width,
        patch_grid=(21, 2, 3),
        branch_refs=branch_refs,
        flow_receipt={},
        middle_receipt={},
    )


def _cache_maps(*, projection_width: int = 5):
    flow_maps = {}
    middle_maps = {}
    for index, branch in enumerate(subject.REQUIRED_BRANCHES, start=1):
        raw = torch.full((20, 2, 4, 6), float(index))
        camera = torch.full((20, 2, 4, 6), float(index) / 2.0)
        validity = torch.ones((20, 1, 4, 6))
        flow_maps[branch] = {
            "backward_raw": raw,
            "backward_camera_residual": camera,
            "validity": validity,
        }
        tensors = {}
        for block_index in subject.BLOCK_INDICES:
            value = torch.full(
                (3, 21, 6, projection_width),
                float(index + block_index),
                dtype=torch.float16,
            )
            value[:, 0] = 0
            tensors[f"middle_block_{block_index:02d}"] = value.contiguous()
        middle_maps[branch] = tensors
    return flow_maps, middle_maps


class G2AWorld4AuditTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260824)

    def test_cache_assembly_is_global_target_only_sp4_abi(self):
        authority = _authority()
        flow_maps, middle_maps = _cache_maps()
        routes, facts = subject.assemble_global_route_payloads(
            authority=authority,
            flow_maps=flow_maps,
            middle_maps=middle_maps,
            sigma_index=1,
        )
        self.assertEqual(set(routes), set(subject.g2a.STEP0_REQUIRED_ROUTES))
        self.assertEqual(facts["layout"]["source_tokens"], 0)
        self.assertEqual(facts["layout"]["phase_count"], 21)
        self.assertEqual(facts["flow_abi"], "global_B_L_12")
        self.assertEqual(facts["middle_abi"], "selected_sigma_global_B_L_W")
        self.assertEqual(facts["middle_capture"], "post_transformer_block_output")
        self.assertEqual(
            facts["step0_required_routes"],
            list(subject.g2a.STEP0_REQUIRED_ROUTES),
        )
        self.assertTrue(
            facts["all_five_active_branches_loaded_from_authenticated_G1_cohort"]
        )
        self.assertEqual(
            facts["branches"]["reverse"]["g1_cohort_cache_role"],
            "external.reverse",
        )
        self.assertEqual(
            facts["branches"]["incomplete"]["g1_cohort_cache_role"],
            "generated.incomplete",
        )
        for branch in subject.REQUIRED_BRANCHES:
            route = routes[branch]
            self.assertEqual(tuple(route.flow.shape), (1, 126, 12))
            self.assertEqual(tuple(route.activity.shape), (1, 126, 1))
            self.assertEqual(set(route.middle_by_block), set(subject.BLOCK_INDICES))
            self.assertTrue(all(
                tuple(value.shape) == (1, 126, 5)
                for value in route.middle_by_block.values()
            ))
            self.assertFalse(route.flow.requires_grad)
            self.assertEqual(route.layout.source_tokens, 0)
        self.assertIsNone(routes["zero"].flow)
        self.assertFalse(routes["zero"].middle_by_block)

    def test_authenticated_loader_includes_reverse_and_incomplete(self):
        authority = _authority()
        observed = []

        def load(reference, *, label):
            observed.append((reference["sha256"], label))
            return {"bound_label": label}

        with mock.patch.object(subject, "load_safetensors_bound", side_effect=load):
            flow_maps, middle_maps = subject.load_authenticated_route_cache_maps(
                authority
            )
        self.assertEqual(set(flow_maps), set(subject.REQUIRED_BRANCHES))
        self.assertEqual(set(middle_maps), set(subject.REQUIRED_BRANCHES))
        labels = {label for _, label in observed}
        self.assertIn("reverse flow", labels)
        self.assertIn("reverse middle", labels)
        self.assertIn("incomplete flow", labels)
        self.assertIn("incomplete middle", labels)
        self.assertEqual(len(observed), 2 * len(subject.REQUIRED_BRANCHES))

    def test_real_30_block_step0_audit_restores_base_and_has_no_grad(self):
        authority = _authority()
        flow_maps, middle_maps = _cache_maps()
        routes, _ = subject.assemble_global_route_payloads(
            authority=authority,
            flow_maps=flow_maps,
            middle_maps=middle_maps,
            sigma_index=1,
        )
        model = _Transformer(8).eval().requires_grad_(False)
        hidden = torch.randn((1, 126, 8))
        input_sha = subject.object_sha256({"toy": "one immutable batch"})
        receipt, parameters = subject.run_native_step0_audit(
            model=model,
            forward_native=lambda: model(hidden),
            input_digest=lambda: input_sha,
            routes=routes,
            hidden_width=8,
            middle_width=5,
            bottleneck_width=4,
        )
        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["step0_noop_audit"]["all_routes_exact_native_bits"])
        self.assertEqual(
            receipt["step0_noop_audit"]["required_routes"],
            list(subject.g2a.STEP0_REQUIRED_ROUTES),
        )
        self.assertEqual(parameters["native_forward_count"], 8)
        self.assertTrue(parameters["six_step0_routes_bit_exact_native"])
        self.assertEqual(len(receipt["step0_noop_audit"]["route_outputs"]), 6)
        self.assertEqual(
            parameters["renderer_base_snapshot_digest_before"],
            parameters["renderer_base_snapshot_digest_after"],
        )
        self.assertEqual(
            parameters["adapter_state_digest_before"],
            parameters["adapter_state_digest_after"],
        )
        self.assertFalse(parameters["adapter_gradients_materialized"])
        self.assertTrue(all(
            not hasattr(block, subject.g2a.MODULE_NAME) for block in model.blocks
        ))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_world4_receipt_is_create_only_and_tamper_closed(self):
        authority = _authority()
        flow_maps, middle_maps = _cache_maps()
        routes, route_facts = subject.assemble_global_route_payloads(
            authority=authority,
            flow_maps=flow_maps,
            middle_maps=middle_maps,
            sigma_index=1,
        )
        model = _Transformer(8).eval().requires_grad_(False)
        hidden = torch.randn((1, 126, 8))
        batch_sha = subject.object_sha256({"batch": "source-only"})
        core, parameters = subject.run_native_step0_audit(
            model=model,
            forward_native=lambda: model(hidden),
            input_digest=lambda: batch_sha,
            routes=routes,
            hidden_width=8,
            middle_width=5,
            bottleneck_width=4,
        )
        source_input = {
            "source_video_sha256": authority.source_video_sha256,
            "source_video_sha256_verified_by_flow_cohort": True,
            "source_video_differs_from_all_anchor_videos": True,
            "source_posterior_tensor_sha256": _sha("7"),
            "matched_native_batch_sha256": batch_sha,
            "same_native_batch_used_for_all_routes": True,
            "source_rgb_used_by_frozen_vae_only": True,
            "source_vae_used_to_form_target_only_audit_FM_state": True,
            "target_or_anchor_media_accessed": False,
            "source_rgb_vae_or_clean_latent_persisted": False,
            "posterior_transport": {},
        }
        runtime = {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": "nccl",
            "exact_transformer_block_count": 30,
            "hidden_width": 8,
            "native_batch_kind": "source_owned_target_only_T2V_FM_state",
            "native_output_kind": "post_head_predicted_target_velocity",
            "bernini_revision": _sha("1"),
            "veomni_revision": _sha("2"),
            "checkpoint_tree_sha256": _sha("3"),
            "selected_sigma_index": 1,
            "selected_sigma": 0.55,
            "spatial_shape": [1, 16, 21, 4, 6],
            "patch_grid": [21, 2, 3],
        }
        with mock.patch.object(subject, "HIDDEN_WIDTH", 8):
            receipt = subject.build_world4_receipt(
                case_id=authority.case_id,
                g1_authority=authority.public_receipt(),
                representation_routes=route_facts,
                source_owned_native_input=source_input,
                runtime=runtime,
                parameter_firewall=parameters,
                core_g2a_receipt=core,
                source_lock={"runner.py": _sha("8")},
            )
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "receipt.json"
                subject.write_world4_receipt_create_only(path, receipt)
                self.assertTrue(path.is_file())
                with self.assertRaisesRegex(
                    subject.G2AWorld4AuditError, "fresh|create-only"
                ):
                    subject.write_world4_receipt_create_only(path, receipt)
            tampered = copy.deepcopy(receipt)
            tampered["training_authority"]["optimization_steps"] = 1
            with self.assertRaises(subject.G2AWorld4AuditError):
                subject.validate_world4_receipt(tampered)

    def test_geometry_mismatch_fails_before_renderer(self):
        authority = _authority()
        flow_maps, middle_maps = _cache_maps()
        middle_maps["wrong_action"]["middle_block_24"] = torch.zeros(
            (3, 21, 7, 5), dtype=torch.float16
        )
        with self.assertRaisesRegex(
            subject.G2AWorld4AuditError, "geometry|layouts"
        ):
            subject.assemble_global_route_payloads(
                authority=authority,
                flow_maps=flow_maps,
                middle_maps=middle_maps,
                sigma_index=1,
            )

    def test_missing_reverse_or_incomplete_cache_fails_before_renderer(self):
        authority = _authority()
        flow_maps, middle_maps = _cache_maps()
        for missing in ("reverse", "incomplete"):
            with self.subTest(missing=missing):
                reduced_flow = dict(flow_maps)
                reduced_middle = dict(middle_maps)
                reduced_flow.pop(missing)
                reduced_middle.pop(missing)
                with self.assertRaisesRegex(
                    subject.G2AWorld4AuditError, "branch closure"
                ):
                    subject.assemble_global_route_payloads(
                        authority=authority,
                        flow_maps=reduced_flow,
                        middle_maps=reduced_middle,
                        sigma_index=1,
                    )

    def test_cli_has_no_target_media_or_update_surface(self):
        source = SUBJECT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("--target-video", source)
        self.assertNotIn("--anchor-video", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn(".backward(", source)
        self.assertIn("target_blob=source_blob", source)
        self.assertIn("source_tokens=0", source)


if __name__ == "__main__":
    unittest.main()
