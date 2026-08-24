from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_residual as motion  # noqa: E402
import train_seer_event_erasure_fm as seer  # noqa: E402


SHA1 = "1" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64
OWNER = METHOD_ROOT / "assets" / "seer_owner_core2_v1.json"
OWNER_SHA = hashlib.sha256(OWNER.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    owner = json.loads(OWNER.read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(owner["rows"]):
        rows.append(
            {
                "iid": item["iid"],
                "source_iid": item["source_iid"],
                "source_video": f"/sealed/source-{index}.mp4",
                "source_video_sha256": ("1" if index == 0 else "2") * 64,
                "target_video": item["target_video"],
                "target_video_sha256": item["target_video_sha256"],
                "shared_i0_path": f"/sealed/i0-{index}.npy",
                "shared_i0_sha256": ("3" if index == 0 else "4") * 64,
                "index_map_sha256": ("5" if index == 0 else "6") * 64,
                "prefix_rgb_exact": True,
                "transition_indices_absent": True,
            }
        )
    return {
        "schema_version": seer.MANIFEST_SCHEMA,
        "owner_spec": {"path": str(OWNER.resolve()), "sha256": OWNER_SHA},
        "rows": rows,
        "raw": {
            "parquet_path": "/sealed/raw.parquet",
            "parquet_sha256": "7" * 64,
            "receipt_path": "/sealed/raw.receipt.json",
            "receipt_sha256": "8" * 64,
            "job_done_path": "/sealed/raw.done.json",
            "job_done_sha256": "9" * 64,
        },
        "vae": {
            "parquet_directory": "/sealed/shards",
            "dataset_summary_path": "/sealed/summary.json",
            "dataset_summary_sha256": "a" * 64,
            "index_path": "/sealed/index.jsonl",
            "index_sha256": "b" * 64,
            "row_count": 2,
        },
        "authority": dict(seer.AUTHORITY),
    }


def _write_manifest(root: Path, value: dict[str, object]) -> tuple[Path, str]:
    path = root / "manifest.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _cli(**overrides: object):
    values = {
        "--bernini-root": "/b",
        "--veomni-root": "/v",
        "--checkpoint": "/c",
        "--preprocessed-parquet-dir": "/d",
        "--dataset-summary": "/s",
        "--routing-jsonl": "/r",
        "--expected-routing-jsonl-sha256": SHA_A,
        "--output": "/o",
        "--method-source-revision": SHA1,
        "--method-source-archive-sha256": SHA_B,
        "--seer-owner-spec": str(OWNER.resolve()),
        "--expected-seer-owner-spec-sha256": OWNER_SHA,
        "--seer-dataset-manifest": "/m",
        "--expected-seer-manifest-sha256": SHA_A,
    }
    values.update(overrides)
    argv: list[str] = []
    for key, value in values.items():
        argv.extend((key, str(value)))
    return seer.build_parser().parse_args(argv)


class SeerManifestContractTests(unittest.TestCase):
    def test_owner_and_derived_manifest_close_exact_core2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = _write_manifest(Path(directory), _manifest())
            binding = seer.validate_manifest(
                owner_path=OWNER.resolve(),
                expected_owner_sha256=OWNER_SHA,
                manifest_path=path.resolve(),
                expected_manifest_sha256=digest,
                verify_media=False,
            )
        self.assertEqual(binding.row_count, 2)
        self.assertEqual(
            binding.iids,
            (
                "seer-fit-dog-7b88a1ca1f804f41-truncate32",
                "seer-fit-human-a35b590961d24694-truncate32",
            ),
        )

    def test_authority_and_target_mutations_fail_closed(self) -> None:
        for mutate, pattern in (
            (
                lambda value: value["authority"].update(
                    {"training_completion_is_method_success": True}
                ),
                "authority",
            ),
            (
                lambda value: value["rows"][0].update(
                    {"target_video_sha256": "0" * 64}
                ),
                "event-erasure",
            ),
        ):
            value = _manifest()
            mutate(value)
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                path, digest = _write_manifest(Path(directory), value)
                with self.assertRaisesRegex(seer.SeerTrainingError, pattern):
                    seer.validate_manifest(
                        owner_path=OWNER.resolve(),
                        expected_owner_sha256=OWNER_SHA,
                        manifest_path=path.resolve(),
                        expected_manifest_sha256=digest,
                        verify_media=False,
                    )

    def test_full_pair_routes_are_exact_and_no_extra_row_is_allowed(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, digest = _write_manifest(root, manifest)
            binding = seer.validate_manifest(
                owner_path=OWNER.resolve(),
                expected_owner_sha256=OWNER_SHA,
                manifest_path=path.resolve(),
                expected_manifest_sha256=digest,
                verify_media=False,
            )
            routing = root / "routing.jsonl"
            payload = b"".join(
                json.dumps(
                    {
                        "schema_version": motion.ROUTING_SCHEMA,
                        "iid": iid,
                        "tier": "full_pair",
                        "full_target_weight": 1.0,
                        "review": "SEER exact same-coordinate accepted pair",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
                for iid in binding.iids
            )
            routing.write_bytes(payload)
            route_sha = hashlib.sha256(payload).hexdigest()
            seer.validate_full_pair_routing(
                binding=binding,
                routing_jsonl=routing.resolve(),
                expected_routing_sha256=route_sha,
            )
            routing.write_bytes(
                payload
                + json.dumps(
                    {
                        "schema_version": motion.ROUTING_SCHEMA,
                        "iid": "extra",
                        "tier": "reject",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            with self.assertRaisesRegex(seer.SeerTrainingError, "full_pair"):
                seer.validate_full_pair_routing(
                    binding=binding,
                    routing_jsonl=routing.resolve(),
                    expected_routing_sha256=hashlib.sha256(routing.read_bytes()).hexdigest(),
                )


class SeerObjectiveContractTests(unittest.TestCase):
    def test_defaults_are_real_same_state_fm_motion_copy_updates(self) -> None:
        args = _cli()
        seer.validate_cli(args)
        self.assertEqual(args.branch_state_mode, "shared_noisy_clean_field")
        self.assertEqual(args.lora_scope, "cross_q_out")
        self.assertEqual(args.learning_rate, 1e-6)
        self.assertEqual(args.fm_loss_weight, 1.0)
        self.assertEqual(args.motion_loss_weight, 0.5)
        self.assertEqual(args.copy_loss_weight, 0.5)
        self.assertEqual(args.motion_objective, "causal_boundary_charbonnier")
        self.assertEqual(args.unreviewed_tier, "reject")
        self.assertEqual(args.max_steps, 160)

    def test_objective_or_scope_drift_is_rejected(self) -> None:
        for flag, value in (
            ("--learning-rate", "2e-6"),
            ("--branch-state-mode", "separate_clean_paths"),
            ("--lora-scope", "cross_q"),
            ("--motion-loss-weight", "0.0"),
            ("--copy-loss-weight", "0.0"),
            ("--fm-loss-weight", "0.0"),
        ):
            with self.subTest(flag=flag):
                args = _cli(**{flag: value})
                with self.assertRaises(seer.SeerTrainingError):
                    seer.validate_cli(args)

    def test_receipt_records_nonzero_parameter_delta_but_no_method_claim(self) -> None:
        binding = seer.SeerBinding(
            owner_path=OWNER.resolve(),
            owner_sha256=OWNER_SHA,
            manifest_path=Path("/sealed/manifest.json"),
            manifest_sha256=SHA_A,
            row_count=2,
            iids=("a", "b"),
            manifest={},
        )
        base_receipt = {
            "adapter": {
                "initialization_digest": SHA_A,
                "checkpoint_parameter_digest": SHA_B,
            },
            "last_metrics": {"preclip_gradient_norm": 0.125},
            "receipt_digest": "0" * 64,
        }
        prior = seer._SEALED_BINDING
        try:
            seer._SEALED_BINDING = binding
            with mock.patch.object(
                seer, "_base_build_receipt", return_value=base_receipt
            ):
                receipt = seer._build_receipt()
        finally:
            seer._SEALED_BINDING = prior
        self.assertTrue(
            receipt["parameter_update_evidence"]["exact_parameter_bytes_changed"]
        )
        self.assertFalse(receipt["parameter_update_evidence"]["method_success_claimed"])
        self.assertFalse(receipt["seer"]["training_completion_is_method_success"])


if __name__ == "__main__":
    unittest.main()
