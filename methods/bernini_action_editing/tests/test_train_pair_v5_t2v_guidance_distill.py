from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from safetensors.torch import save_file

    import pair_v5_t2v_guidance_distill as guidance
    import train_pair_v5_t2v_guidance_distill as trainer

    _RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]
    guidance = None  # type: ignore[assignment]
    trainer = None  # type: ignore[assignment]
    _RUNTIME_AVAILABLE = False


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompts(action_family: str) -> dict[str, str]:
    return {
        branch: (
            f"A complete standalone exact81 pure T2V caption for {action_family} "
            f"under semantic branch {branch}."
        )
        for branch in guidance.BRANCH_ORDER
    }


@unittest.skipUnless(_RUNTIME_AVAILABLE, "AUH torch+safetensors runtime is required")
class PairV5T2VGuidanceTrainerTests(unittest.TestCase):
    def _bundle(
        self,
        root: Path,
        *,
        event_count: int = 2,
        splits: tuple[str, ...] | None = None,
        action_families: tuple[str, ...] | None = None,
    ) -> tuple[Path, str]:
        events = []
        for index in range(event_count):
            action_family = (
                action_families[index]
                if action_families is not None
                else "dog-sit" if index % 2 == 0 else "human-stand"
            )
            prompts = _prompts(action_family)
            analysis_split = (
                splits[index] if splits is not None else "fit"
            )
            event_id = f"{action_family}-{analysis_split}-{index}"
            clean = torch.full((1, 16, 21, 2, 2), 0.1 + index, dtype=torch.float32)
            epsilon = torch.full_like(clean, -0.2 - index)
            clean_path = root / f"{event_id}-clean.safetensors"
            noise_path = root / f"{event_id}-noise.safetensors"
            save_file({"clean": clean}, str(clean_path))
            save_file({"epsilon": epsilon}, str(noise_path))
            eligibility = guidance.seal_eligibility(
                sample_id=event_id,
                action_family=action_family,
                analysis_split=analysis_split,
                event_latent=clean,
                official_epsilon=epsilon,
                official_gaussian_artifact_sha256=_file_sha(noise_path),
                checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
                prompt_by_branch=prompts,
                event_qualified=True,
                calibration_confirmation_passed=True,
                calibration_optimizer_authorized=True,
                event_qualification_receipt_digest="a" * 64,
                calibration_receipt_digest="b" * 64,
            )
            eligibility_value = {
                **dict(eligibility.payload()),
                "receipt_digest": eligibility.receipt_digest,
            }
            eligibility_path = root / f"{event_id}-eligibility.json"
            eligibility_path.write_bytes(
                guidance.canonical_json_bytes(eligibility_value) + b"\n"
            )
            event = {
                "schema_version": trainer.EVENT_SCHEMA,
                "event_id": event_id,
                "action_family": action_family,
                "analysis_split": analysis_split,
                "prompt_by_branch": prompts,
                "prompt_bank_sha256": guidance.prompt_bank_sha256(prompts),
                "clean_latent_path": str(clean_path),
                "clean_latent_file_sha256": _file_sha(clean_path),
                "clean_latent_tensor_key": "clean",
                "official_gaussian_path": str(noise_path),
                "official_gaussian_file_sha256": _file_sha(noise_path),
                "official_gaussian_tensor_key": "epsilon",
                "eligibility_receipt_path": str(eligibility_path),
                "eligibility_receipt_file_sha256": _file_sha(eligibility_path),
            }
            event["event_digest"] = trainer.object_sha256(event)
            events.append(event)
        manifest = {
            "schema_version": trainer.MANIFEST_SCHEMA,
            "optimizer_authorized": True,
            "checkpoint_tree_sha256": trainer.legacy.CHECKPOINT_TREE_SHA256,
            "event_count": len(events),
            "events": events,
            "input_closure": dict(trainer._INPUT_CLOSURE),
        }
        manifest["manifest_digest"] = trainer.object_sha256(manifest)
        path = root / "manifest.json"
        path.write_bytes(trainer.canonical_json_bytes(manifest) + b"\n")
        return path, _file_sha(path)

    def test_manifest_and_tensor_preflight_bind_event_and_own_gaussian(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._bundle(Path(temporary))
            manifest = trainer.load_manifest(str(path), digest)
            runtime = trainer.load_event_tensors(manifest)
            self.assertEqual(len(runtime), 2)
            self.assertEqual(
                [row.spec.action_family for row in runtime], ["dog-sit", "human-stand"]
            )
            self.assertEqual([row.spec.analysis_split for row in runtime], ["fit", "fit"])
            for row in runtime:
                self.assertEqual(
                    guidance.tensor_sha256(row.event_latent_cpu),
                    row.spec.eligibility.clean_t2v_latent_tensor_sha256,
                )
                self.assertEqual(
                    guidance.tensor_sha256(row.official_epsilon_cpu),
                    row.spec.eligibility.official_gaussian_tensor_sha256,
                )

    def test_manifest_requires_two_dp_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._bundle(Path(temporary), event_count=1)
            with self.assertRaisesRegex(
                trainer.PairV5T2VGuidanceTrainingError, "requires 2..16 events"
            ):
                trainer.load_manifest(str(path), digest)

    def test_confirmation_event_is_rejected_from_optimizer_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._bundle(
                Path(temporary), splits=("fit", "confirmation")
            )
            with self.assertRaisesRegex(
                trainer.PairV5T2VGuidanceTrainingError,
                "optimizer events must be fit",
            ):
                trainer.load_manifest(str(path), digest)

    def test_every_exact40_cycle_pairs_distinct_action_families(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._bundle(
                Path(temporary),
                event_count=4,
                action_families=("dog-sit", "human-stand", "dog-sit", "dog-sit"),
            )
            with self.assertRaisesRegex(
                trainer.PairV5T2VGuidanceTrainingError,
                "cycle1 DP0/DP1",
            ):
                trainer.load_manifest(str(path), digest)

    def test_cli_requires_complete_exact40_and_acknowledgement(self) -> None:
        parser = trainer.build_parser()
        common = [
            "--bernini-root", "/tmp/bernini",
            "--veomni-root", "/tmp/veomni",
            "--checkpoint", "/tmp/checkpoint",
            "--checkpoint-content-manifest", "/tmp/checkpoint.sha256",
            "--event-manifest", "/tmp/manifest.json",
            "--expected-event-manifest-sha256", "a" * 64,
            "--cagd-validator-evidence", "/tmp/evidence.json",
            "--expected-cagd-validator-evidence-sha256", "d" * 64,
            "--scorer-group-receipt", "/tmp/sp4-a.json",
            "--expected-scorer-group-receipt-sha256", "e" * 64,
            "--scorer-group-receipt", "/tmp/sp4-b.json",
            "--expected-scorer-group-receipt-sha256", "f" * 64,
            "--output", "/tmp/pair5-guidance",
            "--max-schedule-steps", "40",
            "--method-source-revision", "b" * 40,
            "--method-source-archive-sha256", "c" * 64,
        ]
        args = parser.parse_args(common)
        with self.assertRaisesRegex(
            trainer.PairV5T2VGuidanceTrainingError, "acknowledgement"
        ):
            trainer.validate_cli(args)
        args = parser.parse_args(common + ["--ack-experimental-no-action-success-claim"])
        trainer.validate_cli(args)
        args.max_schedule_steps = 41
        with self.assertRaisesRegex(trainer.PairV5T2VGuidanceTrainingError, "exact40"):
            trainer.validate_cli(args)
        args.max_schedule_steps = 40
        args.expected_checkpoint_content_manifest_sha256 = "0" * 64
        with self.assertRaisesRegex(
            trainer.PairV5T2VGuidanceTrainingError, "content manifest"
        ):
            trainer.validate_cli(args)

    def test_manifest_schema_has_no_rv2v_visual_or_motion_carrier_slot(self) -> None:
        fields = set(trainer._ROOT_FIELDS) | set(trainer._EVENT_FIELDS)
        forbidden = {
            "source_video",
            "source_latent",
            "rv2v_video",
            "target_video",
            "target_latent",
            "donor",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        }
        self.assertTrue(fields.isdisjoint(forbidden))
        self.assertFalse(trainer._INPUT_CLOSURE["cross_video_latent_or_residual"])


if __name__ == "__main__":
    unittest.main()
