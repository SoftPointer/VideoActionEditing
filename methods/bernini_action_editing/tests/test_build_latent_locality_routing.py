from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for root in (METHOD_ROOT, TEST_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from test_build_strict_routing import FrozenReleaseFixture, _json_file  # noqa: E402
from tools import build_latent_locality_routing as locality  # noqa: E402
from tools import build_strict_routing as strict  # noqa: E402


def _zeros(*, channels: int = 16, phases: int = 21) -> list[object]:
    return [
        [[[0.0]] for _phase in range(phases)]
        for _channel in range(channels)
    ]


class LocalityReleaseFixture:
    """Rebind the frozen-contract fixture to tiny deterministic latent rows."""

    def __init__(self, root: Path):
        release_root = root / "release"
        release_root.mkdir()
        self.release = FrozenReleaseFixture(release_root)
        self.checkpoint = root / "checkpoint"
        (self.checkpoint / "vae").mkdir(parents=True)
        self.config = self.checkpoint / "vae" / "config.json"
        _json_file(
            self.config,
            {
                "z_dim": 16,
                "latents_mean": [0.0] * 16,
                "latents_std": [1.0] * 16,
            },
        )
        self.vae_identity = {
            "checkpoint_root": str(self.checkpoint),
            "vae_config_sha256": hashlib.sha256(self.config.read_bytes()).hexdigest(),
        }
        self.rows_by_shard_payload: dict[bytes, dict[str, object]] = {}
        self._rebind_rows()
        self.routing_receipt = self.release.build()

    def _rebind_rows(self) -> None:
        identity_json = strict.canonical_json_bytes(self.vae_identity).decode("utf-8")
        for ordinal, index_row in enumerate(self.release.index_rows):
            iid = str(index_row["iid"])
            source_blob = f"source:{iid}".encode()
            target_blob = f"target:{iid}".encode()
            source_sha = hashlib.sha256(source_blob).hexdigest()
            target_sha = hashlib.sha256(target_blob).hexdigest()
            row_without_digest: dict[str, object] = {
                "schema_version": locality.MATERIALIZED_ROW_SCHEMA,
                "iid": iid,
                "inputs": "[]",
                "bernini_vae_identity_json": identity_json,
                "fixture_ordinal": ordinal,
            }
            digest_input = dict(row_without_digest)
            digest_input["video_vae_latents_sha256"] = [source_sha, target_sha]
            materialized_digest = strict.object_sha256(digest_input)
            materialized_row = dict(row_without_digest)
            materialized_row.update(
                {
                    "video_vae_latents": [source_blob, target_blob],
                    "materialized_row_digest": materialized_digest,
                }
            )
            shard = Path(str(index_row["parquet_path"]))
            self.rows_by_shard_payload[shard.read_bytes()] = materialized_row

            receipt_path = Path(str(index_row["sample_receipt_path"]))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["vae_identity"] = self.vae_identity
            receipt["posterior_parameters_shape"] = [1, 32, 21, 1, 1]
            receipt["source_latent_blob_sha256"] = source_sha
            receipt["target_latent_blob_sha256"] = target_sha
            receipt["materialized_row_digest"] = materialized_digest
            receipt.pop("receipt_digest", None)
            receipt["receipt_digest"] = strict.object_sha256(receipt)
            _json_file(receipt_path, receipt)

            index_row["posterior_parameters_shape"] = [1, 32, 21, 1, 1]
            index_row["materialized_row_digest"] = materialized_digest
            index_row["sample_receipt_sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
        self.release._write_index()
        self.release.summary_value["vae_identity_digest"] = strict.object_sha256(
            self.vae_identity
        )
        self.release._write_summary()

    @property
    def routing_sha256(self) -> str:
        return hashlib.sha256(self.release.output.read_bytes()).hexdigest()

    def row_loader(self, payload: bytes) -> dict[str, object]:
        return dict(self.rows_by_shard_payload[payload])

    @staticmethod
    def mode_loader(
        blob: bytes,
        _vae: locality.VaeStatistics,
        expected_shape: tuple[int, ...],
    ) -> list[object]:
        if expected_shape != (1, 32, 21, 1, 1):
            raise AssertionError(expected_shape)
        role, iid = blob.decode("ascii").split(":", 1)
        value = _zeros()
        if role == "source":
            return value
        if role != "target":
            raise AssertionError(role)
        ordinal = int(iid[3:])
        if ordinal % 3 == 0:
            active_phases = range(1, 21)
        elif ordinal % 3 == 1:
            active_phases = (1,)
        else:
            active_phases = ()
        for channel in range(16):
            for phase in active_phases:
                value[channel][phase][0][0] = 0.2
        return value

    def build(self, output: Path) -> dict[str, object]:
        return locality.build_latent_locality_audit(
            dataset_summary=self.release.summary,
            expected_dataset_summary_sha256=self.release.summary_sha256,
            preprocessed_parquet_dir=self.release.shards,
            strict_routing_jsonl=self.release.output,
            expected_strict_routing_sha256=self.routing_sha256,
            amplitude_thresholds=[0.3, 0.1],
            coverage_caps=[0.5, 0.0],
            output_dir=output,
            _row_loader=self.row_loader,
            _mode_loader=self.mode_loader,
        )


class Q0MetricTests(unittest.TestCase):
    def test_static_delta_is_quotiented_and_only_changed_phase_is_local(self) -> None:
        source = _zeros()
        target = _zeros()
        for channel in range(16):
            for phase in range(21):
                target[channel][phase][0][0] = 5.0
            target[channel][1][0][0] = 7.0

        metric = locality.compute_q0_metrics(
            source, target, amplitude_thresholds=[2.0, 1.0]
        )
        self.assertEqual(metric["q0_boundary_max_abs"], 0.0)
        self.assertEqual(metric["phase0_delta_rms"], 5.0)
        self.assertAlmostEqual(metric["q0_residual_rms"], 2.0 / math.sqrt(21))
        self.assertAlmostEqual(
            metric["q0_nonboundary_residual_rms"], 2.0 / math.sqrt(20)
        )
        coverage = metric["amplitude_coverage_sweep"]
        self.assertEqual(
            [row["amplitude_threshold"] for row in coverage], [1.0, 2.0]
        )
        self.assertEqual(
            [row["nonboundary_cell_count_above"] for row in coverage], [1, 0]
        )
        self.assertEqual(coverage[0]["nonboundary_cell_fraction_above"], 0.05)
        self.assertEqual(
            metric["q0_cell_energy_support"][0][
                "minimum_nonboundary_cell_count"
            ],
            1,
        )

    def test_threshold_axes_are_explicit_finite_unique_and_bounded(self) -> None:
        with self.assertRaisesRegex(locality.LatentLocalityError, "at least one"):
            locality._validated_axis(
                [], context="axis", minimum=0.0, maximum=None
            )
        with self.assertRaisesRegex(locality.LatentLocalityError, "duplicate"):
            locality._validated_axis(
                [0.1, 0.1], context="axis", minimum=0.0, maximum=None
            )
        with self.assertRaisesRegex(locality.LatentLocalityError, "finite"):
            locality._validated_axis(
                [float("nan")], context="axis", minimum=0.0, maximum=None
            )
        with self.assertRaisesRegex(locality.LatentLocalityError, "<= 1.0"):
            locality._validated_axis(
                [1.01], context="axis", minimum=0.0, maximum=1.0
            )


class LocalityAuditPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = LocalityReleaseFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _file_payloads(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        }

    def test_publishes_deterministic_audit_only_sweep_receipts_and_hashes(self) -> None:
        first = self.root / "audit-a"
        second = self.root / "audit-b"
        receipt = self.fixture.build(first)
        self.fixture.build(second)

        self.assertEqual(self._file_payloads(first), self._file_payloads(second))
        self.assertTrue(receipt["audit_only"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["automatic_training_authorization"])
        self.assertIsNone(receipt["selected_candidate"])
        self.assertFalse(receipt["inference_conditions_added"])
        self.assertFalse(receipt["spatial_mask_generated"])
        self.assertEqual(
            [row["selected_count"] for row in receipt["candidate_subsets"]],
            [119, 239, 359, 359],
        )

        metrics = [
            json.loads(line)
            for line in (first / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(metrics), strict.EXPECTED_STRICT_ROWS)
        self.assertEqual({row["schema_version"] for row in metrics}, {locality.METRIC_SCHEMA})
        candidate = first / "candidates" / "candidate-0000.jsonl"
        candidate_rows = [
            json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(candidate_rows), 119)
        self.assertEqual(
            {row["schema_version"] for row in candidate_rows},
            {locality.CANDIDATE_ROW_SCHEMA},
        )
        self.assertNotEqual(locality.CANDIDATE_ROW_SCHEMA, strict.ROUTING_SCHEMA)
        self.assertTrue(all("tier" not in row for row in candidate_rows))

        candidate_receipt_path = (
            first / "candidates" / "candidate-0000.receipt.json"
        )
        candidate_receipt = json.loads(
            candidate_receipt_path.read_text(encoding="utf-8")
        )
        candidate_receipt_digest = candidate_receipt.pop("receipt_digest")
        self.assertEqual(
            strict.object_sha256(candidate_receipt), candidate_receipt_digest
        )
        sidecar = (
            first / "candidates" / "candidate-0000.sha256"
        ).read_text(encoding="ascii").splitlines()
        self.assertEqual(
            sidecar,
            [
                f"{hashlib.sha256(candidate.read_bytes()).hexdigest()}  {candidate.name}",
                (
                    f"{hashlib.sha256(candidate_receipt_path.read_bytes()).hexdigest()}  "
                    f"{candidate_receipt_path.name}"
                ),
            ],
        )

        for line in (first / "SHA256SUMS").read_text(encoding="ascii").splitlines():
            expected_sha, relative = line.split("  ", 1)
            self.assertEqual(
                hashlib.sha256((first / relative).read_bytes()).hexdigest(),
                expected_sha,
            )
        ready = json.loads((first / "audit_receipt.json").read_text(encoding="utf-8"))
        ready_digest = ready.pop("receipt_digest")
        self.assertEqual(strict.object_sha256(ready), ready_digest)

        with self.assertRaisesRegex(locality.LatentLocalityError, "create-only"):
            self.fixture.build(first)

    def test_fails_closed_if_a_bound_parquet_shard_changes(self) -> None:
        strict_shard = self.fixture.release.shards / "iid0000.parquet"
        strict_shard.write_bytes(strict_shard.read_bytes() + b"tamper")
        with self.assertRaisesRegex(locality.LatentLocalityError, "shard hash differs"):
            self.fixture.build(self.root / "tampered-audit")


if __name__ == "__main__":
    unittest.main()
