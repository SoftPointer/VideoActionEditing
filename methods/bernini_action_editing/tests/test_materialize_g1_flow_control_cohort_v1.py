#!/usr/bin/env python3

"""Tests for the G1 dense-flow control cohort."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBJECT_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "materialize_g1_flow_control_cohort_v1.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "materialize_g1_flow_control_cohort_v1_test", SUBJECT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


subject = _load_subject()

try:
    import torch
    from safetensors.torch import load_file, save_file
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    load_file = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@unittest.skipUnless(
    torch is not None and load_file is not None and save_file is not None,
    "PyTorch/safetensors are unavailable",
)
class G1FlowControlCohortTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None and save_file is not None
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.correct = self._bundle("correct", factor=1.0, offset=0.0, source="source-a")
        self.shuffle = self._bundle("shuffle", factor=0.8, offset=3.0, source="source-a")
        self.reverse = self._bundle("reverse", factor=-0.7, offset=2.0, source="source-a")
        self.wrong = self._bundle("wrong", factor=2.5, offset=-4.0, source="source-b")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _bundle(
        self,
        name: str,
        *,
        factor: float,
        offset: float,
        source: str,
    ) -> Path:
        assert torch is not None and save_file is not None
        path = self.root / f"{name}.safetensors"
        raw = torch.arange(20 * 2 * 2 * 3, dtype=torch.float32).reshape(20, 2, 2, 3)
        raw = raw.mul(factor).add(offset + 1.0)
        camera = raw.mul(0.37).add(factor)
        validity = torch.linspace(0.35, 1.0, 20, dtype=torch.float32).view(20, 1, 1, 1)
        validity = validity.expand(20, 1, 2, 3).contiguous()
        save_file(
            {
                "backward_raw": raw,
                "backward_camera_residual": camera,
                "validity": validity,
            },
            str(path),
            metadata={"fixture": name},
        )
        sidecar = {
            "schema_version": subject.EXTRACTOR_SCHEMA_VERSION,
            "source_sha256": _digest(source),
            "anchor_sha256": _digest(f"anchor-{name}"),
            "sampled_frame_indices": list(range(0, 81, 4)),
            "latent_hw": [2, 3],
        }
        path.with_suffix(".json").write_text(
            json.dumps(sidecar, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def _materialize(self, output_name: str = "cohort", anchor_kind: str = "target"):
        return subject.materialize_cohort(
            correct_path=self.correct,
            temporal_shuffle_path=self.shuffle,
            reverse_path=self.reverse,
            wrong_action_path=self.wrong,
            output_dir=self.root / output_name,
            case_id="case-a",
            anchor_kind=anchor_kind,
            action_family="head_turn",
            wrong_case_id="case-b",
            wrong_action_family="barbell_lift",
            incomplete_transitions=9,
        )

    def test_materializes_zero_incomplete_and_energy_matched_wrong_action(self) -> None:
        assert torch is not None and load_file is not None
        receipt = self._materialize()
        output = self.root / "cohort"
        self.assertEqual(receipt["anchor_kind"], "target")
        self.assertEqual(receipt["correct_role"], "real_forward")
        self.assertTrue(receipt["contracts"]["weighted_compensation_forbidden"])
        self.assertFalse(receipt["contracts"]["optimizer_created"])
        self.assertEqual(receipt["contracts"]["current_experiment_optimization_steps"], 0)

        zero = load_file(str(output / "zero_or_noop.safetensors"), device="cpu")
        for tensor in zero.values():
            self.assertEqual(int(torch.count_nonzero(tensor).item()), 0)

        correct = load_file(str(self.correct), device="cpu")
        incomplete = load_file(str(output / "incomplete.safetensors"), device="cpu")
        for key in subject.REQUIRED_TENSORS:
            self.assertTrue(torch.equal(incomplete[key][:9], correct[key][:9]))
            self.assertEqual(int(torch.count_nonzero(incomplete[key][9:]).item()), 0)

        wrong_matched = load_file(
            str(output / "wrong_action_energy_matched.safetensors"), device="cpu"
        )
        correct_energy = subject._effective_energy(correct)
        matched_energy = subject._effective_energy(wrong_matched)
        self.assertLessEqual(
            abs(matched_energy - correct_energy) / correct_energy,
            subject.ENERGY_MATCH_RTOL,
        )
        verified = subject.verify_cohort_receipt(output / "cohort_receipt.json")
        self.assertEqual(verified, receipt)
        for row in receipt["generated_controls"].values():
            self.assertTrue(str(row["path"]).startswith(str(output)))

    def test_selfgen_is_explicitly_separate(self) -> None:
        receipt = self._materialize("selfgen-cohort", "selfgen")
        self.assertEqual(receipt["anchor_kind"], "selfgen")
        self.assertEqual(receipt["correct_role"], "self_generated")
        self.assertTrue(receipt["contracts"]["target_and_selfgen_judged_separately"])

    def test_wrong_action_must_be_different_case_and_family(self) -> None:
        common = dict(
            correct_path=self.correct,
            temporal_shuffle_path=self.shuffle,
            reverse_path=self.reverse,
            wrong_action_path=self.wrong,
            output_dir=self.root / "blocked",
            case_id="case-a",
            anchor_kind="target",
            action_family="head_turn",
            incomplete_transitions=9,
        )
        with self.assertRaisesRegex(subject.G1FlowControlError, "different case"):
            subject.materialize_cohort(
                **common,
                wrong_case_id="case-a",
                wrong_action_family="barbell_lift",
            )
        self.assertFalse((self.root / "blocked").exists())
        with self.assertRaisesRegex(subject.G1FlowControlError, "different action family"):
            subject.materialize_cohort(
                **common,
                wrong_case_id="case-b",
                wrong_action_family="head_turn",
            )
        self.assertFalse((self.root / "blocked").exists())

    def test_temporal_controls_must_share_correct_source(self) -> None:
        alien = self._bundle("alien-shuffle", factor=0.6, offset=8.0, source="alien")
        with self.assertRaisesRegex(subject.G1FlowControlError, "correct source"):
            subject.materialize_cohort(
                correct_path=self.correct,
                temporal_shuffle_path=alien,
                reverse_path=self.reverse,
                wrong_action_path=self.wrong,
                output_dir=self.root / "blocked-source",
                case_id="case-a",
                anchor_kind="target",
                action_family="head_turn",
                wrong_case_id="case-b",
                wrong_action_family="barbell_lift",
            )
        self.assertFalse((self.root / "blocked-source").exists())

    def test_refuses_overwrite_and_detects_tampering(self) -> None:
        self._materialize()
        output = self.root / "cohort"
        with self.assertRaisesRegex(subject.G1FlowControlError, "refusing to overwrite"):
            self._materialize()
        zero = output / "zero_or_noop.safetensors"
        zero.write_bytes(zero.read_bytes() + b"tamper")
        with self.assertRaises(subject.G1FlowControlError):
            subject.verify_cohort_receipt(output / "cohort_receipt.json")

    def test_invalid_bundle_leaves_no_partial_publication(self) -> None:
        assert torch is not None and save_file is not None
        bad = self.root / "bad.safetensors"
        save_file(
            {
                "backward_raw": torch.zeros(20, 2, 2, 3),
                "backward_camera_residual": torch.zeros(20, 2, 2, 3),
                "validity": torch.ones(20, 1, 2, 3),
            },
            str(bad),
        )
        bad.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": subject.EXTRACTOR_SCHEMA_VERSION,
                    "source_sha256": _digest("bad-source"),
                    "anchor_sha256": _digest("bad-anchor"),
                    "sampled_frame_indices": list(range(0, 81, 4)),
                    "latent_hw": [2, 3],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(subject.G1FlowControlError, "motion energy"):
            subject.materialize_cohort(
                correct_path=self.correct,
                temporal_shuffle_path=self.shuffle,
                reverse_path=self.reverse,
                wrong_action_path=bad,
                output_dir=self.root / "never-published",
                case_id="case-a",
                anchor_kind="target",
                action_family="head_turn",
                wrong_case_id="case-b",
                wrong_action_family="barbell_lift",
            )
        self.assertFalse((self.root / "never-published").exists())


if __name__ == "__main__":
    unittest.main()
