#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for path in (METHOD_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import numpy  # noqa: F401
    import torch

    import p_qmosaic_direction_envelope_v1 as profile
    import postflight_qmosaic_editor_direction_v1 as postflight
    import run_p_qmosaic_editor_direction_sp4_v1 as entrypoint
    import run_qmosaic_editor_direction_sp4_v1 as runner
    import self_imagined_native_rv2v_hidden_vjp_v1 as qmosaic
    import test_postflight_qmosaic_editor_direction_v1 as raw_postflight_tests
    import test_run_qmosaic_editor_direction_sp4_v1 as raw_runner_tests

    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - dependency-light hosts
    RUNTIME_AVAILABLE = False


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fixture() -> tuple[object, object, object, object, dict]:
    shape = (1, 16, 21, 60, 62)
    query_seed = 2026081502
    target = torch.randn(
        *shape,
        generator=torch.Generator().manual_seed(9911),
        dtype=torch.float32,
    )
    rows = []
    for sp_rank, fraction in enumerate((0.1, 0.2, 0.3, 0.4)):
        values = (fraction * target).contiguous()
        row = qmosaic.RankLocalVJPRow(
            query_seed=query_seed,
            sp_rank=sp_rank,
            vjp_target="clean_latent",
            values=values,
            score_cotangent_receipt_digest=_digest(f"score-{sp_rank}"),
            editor_packet_receipt_digest=_digest(f"editor-{sp_rank}"),
            global_cotangent_identity_digest=_digest("cotangent"),
            value_sha256=qmosaic.tensor_sha256(values, label="fixture row"),
            value_norm=float(torch.linalg.vector_norm(values.double()).item()),
            replay_max_abs=float(values.abs().max().item()),
            parameter_state_sha256=_digest("parameters"),
        )
        rows.append(qmosaic._seal_rank_local_vjp_row(row))  # noqa: SLF001
    clean_vjp = qmosaic._sum_rank_local_vjp_rows_unsafe_for_test(rows)  # noqa: SLF001
    base = torch.randn(
        *shape,
        generator=torch.Generator().manual_seed(9912),
        dtype=torch.float32,
    )
    base_arm, plus, minus, envelope = profile.construct(
        cell_id="dog", base_clean_latent=base, clean_vjp_row=clean_vjp
    )
    return base_arm, plus, minus, clean_vjp, envelope


@unittest.skipUnless(RUNTIME_AVAILABLE, "torch/numpy runtime is required")
class PQMosaicDirectionIntegrationTests(unittest.TestCase):
    def test_entrypoint_and_constructor_expose_no_variant_or_selection_input(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(profile.construct).parameters),
            ("cell_id", "base_clean_latent", "clean_vjp_row"),
        )
        parser_destinations = {action.dest for action in runner.build_parser()._actions}
        postflight_destinations = {
            action.dest for action in postflight.build_parser()._actions
        }
        for forbidden in ("variant", "dose", "sign", "arm", "callback"):
            self.assertNotIn(forbidden, parser_destinations)
            self.assertNotIn(forbidden, postflight_destinations)
        self.assertEqual(entrypoint.DIRECTION_VARIANT_ID, profile.VARIANT_ID)

    def test_live_adapter_and_all_tensor_hashes_close(self) -> None:
        base, plus, minus, clean_vjp, envelope = _fixture()
        hashes = envelope["tensor_sha256"]
        validated = profile.validate_envelope(
            envelope,
            cell_id="dog",
            query_seed=2026081502,
            clean_vjp_receipt_digest=clean_vjp.receipt()["digest"],
            clean_vjp_value_sha256=clean_vjp.receipt()["value_sha256"],
            base_tensor_sha256=qmosaic.tensor_sha256(base, label="base"),
            plus_tensor_sha256=qmosaic.tensor_sha256(plus, label="plus"),
            minus_tensor_sha256=qmosaic.tensor_sha256(minus, label="minus"),
        )
        self.assertTrue(
            validated["adapter_receipt_recomputed_after_live_tensor_rehash"]
        )
        self.assertEqual(
            hashes["projected_clean_latent_vjp"],
            envelope["runtime_adapter_receipt"]["tensor_sha256"][
                "projected_clean_latent_vjp"
            ],
        )

    def test_resealed_outer_or_nested_hash_tamper_fails_closed(self) -> None:
        base, plus, minus, clean_vjp, envelope = _fixture()
        tampered = deepcopy(envelope)
        tampered.pop("receipt_digest")
        tampered["tensor_sha256"]["projected_clean_latent_vjp"] = _digest("fake")
        tampered = profile._seal(tampered)  # noqa: SLF001 - adversarial fixture
        with self.assertRaises(profile.PQMosaicDirectionEnvelopeError):
            profile.validate_envelope(
                tampered,
                cell_id="dog",
                query_seed=2026081502,
                clean_vjp_receipt_digest=clean_vjp.receipt()["digest"],
                clean_vjp_value_sha256=clean_vjp.receipt()["value_sha256"],
                base_tensor_sha256=qmosaic.tensor_sha256(base, label="base"),
                plus_tensor_sha256=qmosaic.tensor_sha256(plus, label="plus"),
                minus_tensor_sha256=qmosaic.tensor_sha256(minus, label="minus"),
            )

        tampered = deepcopy(envelope)
        tampered.pop("receipt_digest")
        nested = dict(tampered["runtime_adapter_receipt"])
        nested.pop("receipt_digest")
        nested["unregistered_field"] = False
        tampered["runtime_adapter_receipt"] = profile._seal(nested)  # noqa: SLF001
        tampered = profile._seal(tampered)  # noqa: SLF001
        with self.assertRaisesRegex(
            profile.PQMosaicDirectionEnvelopeError, "field closure"
        ):
            profile.validate_envelope(
                tampered,
                cell_id="dog",
                query_seed=2026081502,
                clean_vjp_receipt_digest=clean_vjp.receipt()["digest"],
                clean_vjp_value_sha256=clean_vjp.receipt()["value_sha256"],
                base_tensor_sha256=qmosaic.tensor_sha256(base, label="base"),
                plus_tensor_sha256=qmosaic.tensor_sha256(plus, label="plus"),
                minus_tensor_sha256=qmosaic.tensor_sha256(minus, label="minus"),
            )

    def test_runner_builds_only_the_bumped_engineering_receipt(self) -> None:
        _base, _plus, _minus, clean_vjp, envelope = _fixture()
        hashes = envelope["tensor_sha256"]
        parity = raw_runner_tests.parity_fixture()
        parity["b0_tensor_sha256"] = hashes["adapter_base_clean_latent"]
        parity["z0_tensor_sha256"] = hashes["adapter_base_clean_latent"]
        arms = [
            {
                "role": role,
                "mp4_path": f"/out/{role}.mp4",
                "latent_tensor_sha256": tensor_hash,
            }
            for role, tensor_hash in zip(
                runner.ARM_ORDER,
                (
                    hashes["adapter_base_clean_latent"],
                    hashes["plus_clean_latent"],
                    hashes["minus_clean_latent"],
                ),
            )
        ]
        receipt = runner.build_run_receipt(
            cell={
                "cell_id": "dog",
                "source_iid": "7b88a1ca1f804f41",
                "source_video_sha256": _digest("source"),
                "action_family_id": "dog-stand-to-sit-facing-camera",
            },
            query_seed=2026081502,
            owner_receipt=raw_runner_tests.fake_receipt("owner"),
            editor_receipt=raw_runner_tests.fake_editor_receipt(),
            score_receipt=raw_runner_tests.fake_receipt("score"),
            clean_vjp_receipt=clean_vjp.receipt(),
            checkpoint_receipt=raw_runner_tests.fake_receipt("checkpoint"),
            collective_receipt=raw_runner_tests.fake_receipt("collective"),
            runner_contract=raw_runner_tests.fake_receipt("runner"),
            parity_evidence=parity,
            direction_evidence=envelope,
            terminal_full_seal_evidence=raw_runner_tests.terminal_rows(),
            arm_artifacts=arms,
            parameter_invariance={
                "parameter_bytes_unchanged": True,
                "lora_b_exact_zero_after": True,
            },
            method_source_revision="a" * 40,
            method_source_archive_sha256=_digest("archive"),
            _p_qmosaic=True,
        )
        self.assertEqual(receipt["schema_version"], profile.RUN_RECEIPT_SCHEMA)
        self.assertEqual(receipt["method_name"], profile.METHOD_NAME)
        self.assertEqual(receipt["direction_variant"], profile.variant_lock())
        self.assertEqual(
            receipt["experiment_scope"]["classification"], "ENGINEERING_ONLY"
        )
        self.assertFalse(receipt["authorization"]["lora_vjp_authorized"])
        self.assertFalse(receipt["authorization"]["parameter_update_authorized"])

    def test_p_run_receipt_and_postflight_remain_engineering_only(self) -> None:
        base, plus, minus, clean_vjp, envelope = _fixture()
        hashes = envelope["tensor_sha256"]

        def mutate(unsigned):
            unsigned["schema_version"] = profile.RUN_RECEIPT_SCHEMA
            unsigned["method_name"] = profile.METHOD_NAME
            unsigned["experiment_scope"]["classification"] = "ENGINEERING_ONLY"
            unsigned["direction_variant"] = dict(profile.variant_lock())
            unsigned["symmetric_direction"] = envelope
            unsigned["native_coordinate"]["sp4_clean_vjp_receipt_digest"] = (
                clean_vjp.receipt()["digest"]
            )
            unsigned["predecode_parity"]["b0_tensor_sha256"] = hashes[
                "adapter_base_clean_latent"
            ]
            unsigned["predecode_parity"]["z0_tensor_sha256"] = hashes[
                "adapter_base_clean_latent"
            ]
            for row, tensor_hash in zip(
                unsigned["published_arms"],
                (
                    hashes["adapter_base_clean_latent"],
                    hashes["plus_clean_latent"],
                    hashes["minus_clean_latent"],
                ),
            ):
                row["latent_tensor_sha256"] = tensor_hash

        helper = raw_postflight_tests.PostflightArtifactTests()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                receipt_path, probes, _ = helper.make_fixture(root, mutator=mutate)
                validated = postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runner._file_sha256(  # noqa: SLF001
                        receipt_path
                    ),
                    artifact_root=root,
                    probe_fn=helper.probe(probes),
                    _p_qmosaic=True,
                )
                receipt = postflight.build_postflight_receipt(
                    validated, _p_qmosaic=True
                )
                self.assertEqual(receipt["schema_version"], profile.POSTFLIGHT_SCHEMA)
                self.assertEqual(receipt["experiment_scope"], "ENGINEERING_ONLY")
                self.assertFalse(receipt["lora_vjp_authorized"])
                self.assertFalse(receipt["parameter_update_authorized"])
                self.assertFalse(receipt["scientific_action_editing_success_claim"])
        finally:
            helper.doCleanups()


if __name__ == "__main__":
    unittest.main()
