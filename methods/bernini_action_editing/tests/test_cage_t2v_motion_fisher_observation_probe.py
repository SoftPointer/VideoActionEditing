from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cage_t2v_motion_fisher_observation_probe as probe  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as events  # noqa: E402


BASE_SPEC_PATH = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"
BASE_SPEC_SHA = "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"


def load_population():
    base = json.loads(BASE_SPEC_PATH.read_text())
    registration = probe.load_registration()
    topup, rows = probe.load_topup_spec(
        probe.TOPUP_SPEC_PATH.resolve(),
        probe.TOPUP_SPEC_SHA256,
        base_spec=base,
        expected_base_spec_sha256=BASE_SPEC_SHA,
    )
    selection = probe.validate_population_coverage(
        base,
        registration,
        topup_rows=rows,
        topup_spec_sha256=probe.TOPUP_SPEC_SHA256,
    )
    return base, topup, rows, selection


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn2 = nn.Module()
        self.attn2.to_out = nn.ModuleList([nn.Linear(1536, 1536, bias=False)])
        self.attn2.to_out[0].requires_grad_(False)


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(30)])

    @property
    def dtype(self):
        return self.blocks[0].attn2.to_out[0].weight.dtype


class MotionFisherObservationProbeTests(unittest.TestCase):
    def test_registration_and_action_only_topup_close_2x2x2(self) -> None:
        base, topup, rows, selection = load_population()
        self.assertEqual(topup["file_sha256"], probe.TOPUP_SPEC_SHA256)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(selection.candidate_ids), 8)
        self.assertEqual(set(selection.action_families), {
            "dog-sit-facing-camera",
            "human-rise-to-stand",
        })
        self.assertEqual(
            {key: len(value) for key, value in selection.group_candidate_ids.items()},
            {"sp4-a": 4, "sp4-b": 4},
        )
        with self.assertRaises(probe.MotionFisherObservationProbeError):
            probe.validate_population_coverage(base, probe.load_registration())

    def test_topup_changes_only_seed_and_candidate_cell_ids(self) -> None:
        base, _, rows, _ = load_population()
        actions = {
            item["candidate_id"]: item
            for group in base["groups"]
            for item in group["candidates"]
            if item["semantic_branch"] == "action"
        }
        for row in rows:
            candidate = row["candidate"]
            original = actions[row["base_action_candidate_id"]]
            changed = {
                key
                for key in candidate
                if candidate[key] != original[key]
            }
            self.assertEqual(changed, {"candidate_id", "calibration_group_id", "seed"})
            self.assertNotEqual(candidate["seed"], original["seed"])

    def test_topup_plan_uses_standard_pair_envelopes_without_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            result = probe.materialize_topup_plan(
                topup_spec_path=probe.TOPUP_SPEC_PATH.resolve(),
                expected_topup_spec_sha256=probe.TOPUP_SPEC_SHA256,
                base_spec_path=BASE_SPEC_PATH.resolve(),
                expected_base_spec_sha256=BASE_SPEC_SHA,
                output_dir=output,
            )
            self.assertEqual(result["candidate_count"], 4)
            self.assertFalse(result["negative_branch_generation_required"])
            envelopes = list(output.glob("sp4-*/*.json"))
            self.assertEqual(len(envelopes), 4)
            self.assertTrue(all(json.loads(path.read_text())["candidate"]["semantic_branch"] == "action" for path in envelopes))

    def test_registered_temporal_views_are_exact_and_camera_appearance_share_media(self) -> None:
        value = torch.arange(21, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
        value = value.expand(1, 16, 21, 2, 2).contiguous()
        action = probe.apply_registered_transform(value, "action")
        reverse = probe.apply_registered_transform(value, "reverse")
        freeze = probe.apply_registered_transform(value, "freeze")
        shuffle = probe.apply_registered_transform(value, "shuffle")
        camera = probe.apply_registered_transform(value, "camera")
        appearance = probe.apply_registered_transform(value, "appearance")
        self.assertTrue(torch.equal(action, value))
        self.assertEqual(reverse[0, 0, :, 0, 0].tolist(), list(range(20, -1, -1)))
        self.assertEqual(freeze[0, 0, :, 0, 0].tolist(), [0.0] * 21)
        self.assertEqual(shuffle[0, 0, :, 0, 0].tolist(), [(8 * i) % 21 for i in range(21)])
        self.assertTrue(torch.equal(camera, value))
        self.assertTrue(torch.equal(appearance, value))

    def test_walsh_a_and_zero_b_hooks_leave_base_output_exact(self) -> None:
        fixed_a = probe.make_fixed_orthogonal_a(dtype=torch.float32)
        self.assertTrue(
            torch.allclose(fixed_a @ fixed_a.T, torch.eye(8), atol=1e-6, rtol=0.0)
        )
        transformer = FakeTransformer()
        bank = probe.FixedAZeroBProbeBank(transformer)
        target = transformer.blocks[0].attn2.to_out[0]
        value = torch.randn(1, 3, 1536)
        baseline = target(value)
        with bank.installed():
            observed = target(value)
            self.assertTrue(torch.equal(observed, baseline))
            scalar = observed.float().square().mean()
            gradients = torch.autograd.grad(scalar, tuple(bank.probe_b), allow_unused=True)
            self.assertIsNotNone(gradients[0])
            self.assertTrue(all(parameter.grad is None for parameter in bank.probe_b))
            self.assertTrue(all(torch.count_nonzero(parameter) == 0 for parameter in bank.probe_b))
        bank.assert_zero_clean()

    def test_event_index_requires_all_eight_positive_action_receipts(self) -> None:
        base, _, topup_spec_rows, selection = load_population()
        base_candidates = {
            item["candidate_id"]: item
            for group in base["groups"]
            for item in group["candidates"]
            if item["semantic_branch"] == "action"
        }
        topup_candidates = {
            row["candidate"]["candidate_id"]: row["candidate"]
            for row in topup_spec_rows
        }
        candidates = {**base_candidates, **topup_candidates}
        base_generation = {
            candidate_id: probe.object_sha256({"base": candidate_id})
            for candidate_id in base_candidates
        }
        topup_generation = {
            candidate_id: probe.object_sha256({"topup": candidate_id})
            for candidate_id in topup_candidates
        }
        bank = {
            "receipt_digest": "b" * 64,
            "candidate_receipts": [
                {"candidate_id": key, "receipt_digest": value}
                for key, value in base_generation.items()
            ],
        }
        topup_runtime_rows = [
            {"candidate": candidates[key], "generation_receipt_digest": value}
            for key, value in topup_generation.items()
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindings = []
            for candidate_id in selection.candidate_ids:
                candidate = candidates[candidate_id]
                generation = {**base_generation, **topup_generation}[candidate_id]
                receipt = events.seal_event_audit_receipt(
                    candidate_id=candidate_id,
                    analysis_split=candidate["analysis_split"],
                    action_family_id=candidate["action_family_id"],
                    calibration_group_id=candidate["calibration_group_id"],
                    actor_group_id=candidate["actor_group_id"],
                    scene_group_id=candidate["scene_group_id"],
                    action_group_id=candidate["action_group_id"],
                    semantic_branch="action",
                    generation_receipt_digest=generation,
                    audit_source_kind="manual_detached",
                    external_audit_artifact_sha256="a" * 64,
                    complete_target_transition_observed=True,
                    terminal_hold_observed=True,
                    full_target_action_observed=True,
                    full_target_action_false_confirmed=False,
                )
                path = root / f"{candidate_id}.json"
                path.write_bytes(probe.canonical_json_bytes(receipt) + b"\n")
                bindings.append(
                    {
                        "candidate_id": candidate_id,
                        "event_receipt_path": str(path),
                        "event_receipt_file_sha256": probe.file_sha256(path),
                        "event_receipt_digest": receipt["receipt_digest"],
                    }
                )
            index = probe.seal_event_index(
                root_spec_sha256=selection.population_digest,
                bank_receipt_digest=bank["receipt_digest"],
                rows=bindings,
            )
            checked = probe.validate_event_index(
                index,
                selection=selection,
                population_digest=selection.population_digest,
                bank=bank,
                topup_rows=topup_runtime_rows,
            )
            self.assertEqual(set(checked), set(selection.candidate_ids))
            broken = dict(index)
            broken["rows"] = broken["rows"][:-1]
            unsigned = dict(broken)
            unsigned.pop("receipt_digest")
            broken["receipt_digest"] = probe.object_sha256(unsigned)
            with self.assertRaises(probe.MotionFisherObservationProbeError):
                probe.validate_event_index(
                    broken,
                    selection=selection,
                    population_digest=selection.population_digest,
                    bank=bank,
                    topup_rows=topup_runtime_rows,
                )

    def test_contract_contains_no_optimizer_or_rv2v_tensor_input(self) -> None:
        source = Path(probe.__file__).read_text()
        self.assertIn("negative_control_only", source)
        self.assertIn("raw_parameter_gradient_no_manual_sign_flip", source)
        self.assertNotIn("torch.optim", source)
        signature = set(__import__("inspect").signature(probe.builtin_negative_control_vjp).parameters)
        self.assertTrue({"clean_view", "official_gaussian"}.issubset(signature))
        self.assertFalse({"rv2v_target", "source_video", "mask", "flow"} & signature)


if __name__ == "__main__":
    unittest.main()

