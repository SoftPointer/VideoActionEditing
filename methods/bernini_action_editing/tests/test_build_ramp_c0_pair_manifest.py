from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_ramp_c0_pair_manifest as builder  # noqa: E402
from tools import materialize_ramp_motion_analogy_vae as materializer  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Frames:
    def __init__(self, shape=(81, 32, 48, 3), dtype="uint8") -> None:
        self.shape = shape
        self.dtype = dtype


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(builder.canonical_json_bytes(value) + b"\n")


def _sealed(value: dict) -> dict:
    value = dict(value)
    value.pop("receipt_digest", None)
    value["receipt_digest"] = builder.object_sha256(value)
    return value


def _build_manifest(root: Path) -> tuple[dict, Path, Path, Path]:
    source = root / "source-a.mp4"
    donor = root / "donor-b.mp4"
    source.write_bytes(b"source-A-exact81-fixture")
    donor.write_bytes(b"donor-B-exact81-fixture")
    manifest = root / "ramp-c0.jsonl"
    with mock.patch.object(
        builder.materializer.base,
        "_decode_exact_video",
        side_effect=[
            (_Frames((81, 32, 48, 3)), 25.0, (32, 48)),
            (_Frames((81, 40, 40, 3)), 25.0, (40, 40)),
        ],
    ):
        result = builder.build_manifest_bundle(
            source_a_video=source.resolve(),
            expected_source_a_sha256=builder.file_sha256(source),
            donor_b_video=donor.resolve(),
            expected_donor_b_sha256=builder.file_sha256(donor),
            program_a_kind="reverse",
            program_a_parameter=0.0,
            program_b_kind="speed_up",
            program_b_parameter=0.5,
            output_manifest=manifest.resolve(),
        )
    return result, source.resolve(), donor.resolve(), manifest.resolve()


def _program_contract(kind: str, parameter_hex: str, salt: str) -> tuple[dict, str]:
    value = {
        "kind": kind,
        "parameter_hex": parameter_hex,
        "frame_count": 81,
        "coordinate_dtype": "torch.float64",
        "coordinate_sha256": _sha(f"coordinates-{salt}"),
        "rgb_resampling": "linear_output_to_input",
        "vae_phase_permutation_authorized": False,
    }
    return value, builder.object_sha256(value)


def _materialized_receipt(
    root: Path,
    *,
    manifest: Path,
    row_index: int,
    label: str,
) -> tuple[Path, str, Path]:
    loaded = materializer.load_manifest(
        manifest, expected_sha256=builder.file_sha256(manifest)
    )
    row = loaded.rows[row_index]
    parquet = root / f"{label}.parquet"
    parquet.write_bytes(f"sealed-parquet-{label}".encode("ascii"))
    parameter_hex = row.program_parameter_hex
    program_value, program_digest = _program_contract(
        row.program_kind, parameter_hex, label
    )
    instruction_sha = _sha(
        "Apply to the source video the temporal program demonstrated by the "
        "change between the two motion-donor videos."
    )
    motion = {
        "schema_version": "bernini-mdr-exact-motion-analogy-v1",
        "construction": "source=A,donor_packet=(B,T(B)),target=T(A)",
        "source_identity_sha256": row.source_sha256,
        "donor_identity_sha256": row.donor_sha256,
        "source_and_donor_identity_distinct": True,
        "program": program_value,
        "program_digest": program_digest,
        "instruction_sha256": instruction_sha,
        "instruction_is_generic_donor_follow": True,
        "target_origin": "deterministic_RGB_transform_of_source_inside_builder",
        "external_target_accepted": False,
        "donor_appearance_is_correct_target": False,
        "relative_donor_program_observable": True,
        "single_after_only_donor_is_main_training_input": False,
        "paired_action_dataset_used": False,
        "mask_flow_pose_track_trajectory_used": False,
        "frame_count": 81,
        "latent_frame_count_after_pinned_Wan_VAE": 21,
        "direct_21_phase_permutation_authorized": False,
    }
    motion = _sealed(motion)
    vae_identity = {
        "checkpoint_root": "/checkpoint",
        "checkpoint_content_manifest_path": "/checkpoint.sha256",
        "checkpoint_content_manifest_sha256": "a" * 64,
        "vae_config_sha256": "b" * 64,
        "vae_files": {
            "vae/config.json": "b" * 64,
            "vae/diffusion_pytorch_model.safetensors": "c" * 64,
        },
        "every_vae_file_sha256_verified": True,
        "posterior_representation": "latent_dist.parameters_fp32",
        "posterior_sample_materialized": False,
    }
    vae_identity["vae_identity_digest"] = builder.object_sha256(vae_identity)
    posterior = {
        "source_A": _sha("same-source-posterior"),
        "donor_before_B": _sha("same-donor-before-posterior"),
        "donor_after_TB": _sha(f"donor-after-{label}"),
        "target_TA": _sha(f"target-{label}"),
    }
    receipt = {
        "schema_version": builder.MATERIALIZER_RECEIPT_SCHEMA,
        "complete": True,
        "row_id": row.row_id,
        "manifest": {
            "path": str(manifest),
            "sha256": builder.file_sha256(manifest),
            "row_digest": row.manifest_row_digest,
        },
        "input": {
            "source_A": {
                "path": str(row.source_path),
                "sha256": row.source_sha256,
            },
            "donor_B": {
                "path": str(row.donor_path),
                "sha256": row.donor_sha256,
            },
            "source_and_donor_paths_distinct": True,
            "source_and_donor_sha256_distinct": True,
            "external_target": None,
        },
        "program": {
            "kind": row.program_kind,
            "parameter_hex": parameter_hex,
            "digest": program_digest,
        },
        "construction": "source=A,donor_packet=(B,T(B)),target=T(A)",
        "frame_count": 81,
        "fps": 25.0,
        "latent_frame_count": 21,
        "source_derived_bucket_hw": [32, 48],
        "donor_center_crop_tlbr": [0, 0, 40, 40],
        "rgb_tensor_sha256": {
            "source_A": _sha("source-rgb"),
            "donor_before_B": _sha("donor-rgb"),
            "donor_after_TB": _sha(f"donor-after-rgb-{label}"),
            "target_TA": _sha(f"target-rgb-{label}"),
        },
        "motion_analogy_builder_receipt": motion,
        "four_independent_VAE_encode_calls": True,
        "vae_identity": vae_identity,
        "vae_posterior_metadata": {},
        "vae_posterior_blob_sha256": posterior,
        "materialized_row_digest": _sha(f"row-{label}"),
        "parquet_path": str(parquet.resolve()),
        "parquet_sha256": builder.file_sha256(parquet),
        "create_only": True,
        "target_origin": "deterministic_RGB_transform_of_source_inside_committed_builder",
        "shared_i0_used": False,
        "external_target_accepted": False,
        "paired_action_dataset_used": False,
        "mask_flow_pose_track_box_trajectory_used": False,
        "direct_21_phase_permutation_authorized": False,
        "posterior_sample_materialized": False,
        "downstream_independent_posterior_sampling_authorized": False,
        "training_authorized": False,
        "training_use_forbidden": True,
        "action_training_authorized": False,
        "natural_semantic_action_learned": False,
        "scientific_claim_authorized": False,
    }
    receipt = _sealed(receipt)
    receipt_path = root / f"{label}.receipt.json"
    _write_json(receipt_path, receipt)
    return receipt_path.resolve(), builder.file_sha256(receipt_path), parquet.resolve()


class RAMPManifestBuilderTests(unittest.TestCase):
    def test_builds_exact_two_row_materializer_manifest_and_sealed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result, source, donor, manifest = _build_manifest(root)
            self.assertEqual(result["manifest_path"], str(manifest))
            self.assertEqual(result["manifest_sha256"], builder.file_sha256(manifest))
            self.assertEqual(result["row_ids"], [
                builder.DEFAULT_ROW_A_ID, builder.DEFAULT_ROW_B_ID
            ])
            loaded = materializer.load_manifest(
                manifest, expected_sha256=result["manifest_sha256"]
            )
            self.assertEqual(len(loaded.rows), 2)
            self.assertEqual(
                [(row.program_kind, row.program_parameter_hex) for row in loaded.rows],
                [("reverse", float(0.0).hex()), ("speed_up", float(0.5).hex())],
            )
            for row in loaded.rows:
                self.assertEqual(row.source_path, source)
                self.assertEqual(row.donor_path, donor)
                self.assertNotEqual(row.source_sha256, row.donor_sha256)
            receipt_path = Path(result["build_receipt_path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            declared = receipt.pop("receipt_digest")
            self.assertEqual(declared, builder.object_sha256(receipt))
            self.assertEqual(receipt["manifest"]["sha256"], result["manifest_sha256"])
            self.assertEqual(receipt["manifest"]["row_count"], 2)
            self.assertTrue(receipt["programs"]["distinct"])
            self.assertEqual(receipt["media_contract"]["frame_count"], 81)
            self.assertEqual(receipt["media_contract"]["fps"], 25.0)
            self.assertFalse(receipt["external_regression_target_accepted"])
            self.assertFalse(receipt["spatial_or_motion_side_channel_accepted"])
            for path in (manifest, receipt_path):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o222, 0)

    def test_rejects_same_identity_hash_mismatch_and_same_program(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source.mp4"
            donor = root / "donor.mp4"
            source.write_bytes(b"same")
            donor.write_bytes(b"same")
            kwargs = {
                "source_a_video": source.resolve(),
                "expected_source_a_sha256": builder.file_sha256(source),
                "donor_b_video": donor.resolve(),
                "expected_donor_b_sha256": builder.file_sha256(donor),
                "program_a_kind": "reverse",
                "program_a_parameter": 0.0,
                "program_b_kind": "speed_up",
                "program_b_parameter": 0.5,
                "output_manifest": (root / "one.jsonl").resolve(),
            }
            with self.assertRaisesRegex(
                builder.RAMPC0PairBuilderError, "identities must differ"
            ):
                builder.build_manifest_bundle(**kwargs)

            donor.write_bytes(b"different")
            kwargs["expected_donor_b_sha256"] = builder.file_sha256(donor)
            kwargs["expected_source_a_sha256"] = "f" * 64
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "SHA-256 differs"):
                builder.build_manifest_bundle(**kwargs)

            kwargs["expected_source_a_sha256"] = builder.file_sha256(source)
            kwargs["program_b_kind"] = "reverse"
            kwargs["program_b_parameter"] = 0.0
            with mock.patch.object(
                builder.materializer.base,
                "_decode_exact_video",
                side_effect=[
                    (_Frames(), 25.0, (32, 48)),
                    (_Frames(), 25.0, (32, 48)),
                ],
            ):
                with self.assertRaisesRegex(
                    builder.RAMPC0PairBuilderError, "distinct temporal programs"
                ):
                    builder.build_manifest_bundle(**kwargs)

    def test_default_probe_rejects_non_exact_frame_or_fps_contract(self) -> None:
        path = Path(__file__).resolve()
        cases = (
            ((_Frames((80, 32, 48, 3)), 25.0, (32, 48)), "exact81"),
            ((_Frames((81, 32, 48, 3)), 24.0, (32, 48)), "25fps"),
            ((_Frames((81, 32, 48, 4)), 25.0, (32, 48)), "RGB"),
        )
        for returned, label in cases:
            with self.subTest(label=label), mock.patch.object(
                builder.materializer.base,
                "_decode_exact_video",
                return_value=returned,
            ):
                with self.assertRaises(builder.RAMPC0PairBuilderError):
                    builder._probe_exact_video(path)

    def test_rejects_relative_and_symlink_media_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source.mp4"
            donor = root / "donor.mp4"
            source.write_bytes(b"source")
            donor.write_bytes(b"donor")
            link = root / "source-link.mp4"
            link.symlink_to(source)
            base = {
                "expected_source_a_sha256": builder.file_sha256(source),
                "donor_b_video": donor,
                "expected_donor_b_sha256": builder.file_sha256(donor),
                "program_a_kind": "reverse",
                "program_a_parameter": 0.0,
                "program_b_kind": "speed_up",
                "program_b_parameter": 0.5,
                "output_manifest": root / "manifest.jsonl",
            }
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "absolute"):
                builder.build_manifest_bundle(source_a_video="source.mp4", **base)
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "non-symlink"):
                builder.build_manifest_bundle(source_a_video=link, **base)

    def test_cli_has_no_external_target_or_side_channel_surface(self) -> None:
        parser = builder.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        for subparser_action in parser._actions:
            choices = getattr(subparser_action, "choices", None)
            if isinstance(choices, dict):
                for child in choices.values():
                    option_strings.update(
                        option
                        for action in child._actions
                        for option in action.option_strings
                    )
        for forbidden in (
            "--target", "--target-video", "--mask", "--flow", "--pose",
            "--track", "--box", "--trajectory", "--first-frame",
        ):
            self.assertNotIn(forbidden, option_strings)
        self.assertEqual(set(parser._subparsers._group_actions[0].choices), {
            "manifest", "pair-config"
        })


class RAMPPairConfigBuilderTests(unittest.TestCase):
    def test_builds_exact_trainer_config_from_two_sealed_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, _, _, manifest = _build_manifest(root)
            arm_a, arm_a_sha, parquet_a = _materialized_receipt(
                root, manifest=manifest, row_index=0, label="arm-a"
            )
            arm_b, arm_b_sha, parquet_b = _materialized_receipt(
                root, manifest=manifest, row_index=1, label="arm-b"
            )
            output = root / "pair-config.json"
            result = builder.build_pair_config(
                arm_a_receipt=arm_a,
                expected_arm_a_receipt_sha256=arm_a_sha,
                arm_b_receipt=arm_b,
                expected_arm_b_receipt_sha256=arm_b_sha,
                output_pair_config=output,
            )
            self.assertEqual(result["pair_config_sha256"], builder.file_sha256(output))
            self.assertEqual(result["pair_config_schema"], builder.PAIR_CONFIG_SCHEMA)
            self.assertTrue(result["pairing_audit"]["target_posteriors_distinct"])
            self.assertFalse(result["training_authorized_by_builder"])
            config = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(config), {"schema_version", "arm_a", "arm_b"})
            self.assertEqual(config["schema_version"], builder.PAIR_CONFIG_SCHEMA)
            self.assertEqual(
                config["arm_a"],
                {
                    "parquet_path": str(parquet_a),
                    "parquet_sha256": builder.file_sha256(parquet_a),
                    "receipt_path": str(arm_a),
                    "receipt_sha256": arm_a_sha,
                },
            )
            self.assertEqual(config["arm_b"]["parquet_path"], str(parquet_b))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode) & 0o222, 0)

    def test_pair_config_rejects_receipt_tamper_and_wrong_expected_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, _, _, manifest = _build_manifest(root)
            arm_a, arm_a_sha, _ = _materialized_receipt(
                root, manifest=manifest, row_index=0, label="arm-a"
            )
            arm_b, arm_b_sha, _ = _materialized_receipt(
                root, manifest=manifest, row_index=1, label="arm-b"
            )
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "file SHA-256"):
                builder.build_pair_config(
                    arm_a_receipt=arm_a,
                    expected_arm_a_receipt_sha256="0" * 64,
                    arm_b_receipt=arm_b,
                    expected_arm_b_receipt_sha256=arm_b_sha,
                    output_pair_config=root / "bad-sha.json",
                )
            receipt = json.loads(arm_b.read_text(encoding="utf-8"))
            receipt["program"]["kind"] = "identity"
            os.chmod(arm_b, 0o600)
            _write_json(arm_b, receipt)
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "embedded receipt digest"):
                builder.build_pair_config(
                    arm_a_receipt=arm_a,
                    expected_arm_a_receipt_sha256=arm_a_sha,
                    arm_b_receipt=arm_b,
                    expected_arm_b_receipt_sha256=builder.file_sha256(arm_b),
                    output_pair_config=root / "tampered.json",
                )

    def test_pair_config_rejects_resealed_nested_program_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, _, _, manifest = _build_manifest(root)
            arm_a, _, _ = _materialized_receipt(
                root, manifest=manifest, row_index=0, label="arm-a"
            )
            arm_b, arm_b_sha, _ = _materialized_receipt(
                root, manifest=manifest, row_index=1, label="arm-b"
            )
            receipt = json.loads(arm_a.read_text(encoding="utf-8"))
            motion = receipt["motion_analogy_builder_receipt"]
            motion["program"]["kind"] = "identity"
            motion = _sealed(motion)
            receipt["motion_analogy_builder_receipt"] = motion
            receipt = _sealed(receipt)
            _write_json(arm_a, receipt)
            with self.assertRaisesRegex(
                builder.RAMPC0PairBuilderError, "motion-builder receipt differs"
            ):
                builder.build_pair_config(
                    arm_a_receipt=arm_a,
                    expected_arm_a_receipt_sha256=builder.file_sha256(arm_a),
                    arm_b_receipt=arm_b,
                    expected_arm_b_receipt_sha256=arm_b_sha,
                    output_pair_config=root / "nested-program-tamper.json",
                )

    def test_pair_config_rejects_same_program_and_parquet_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, _, _, manifest = _build_manifest(root)
            arm_a, arm_a_sha, parquet_a = _materialized_receipt(
                root, manifest=manifest, row_index=0, label="arm-a"
            )
            arm_b, arm_b_sha, _ = _materialized_receipt(
                root, manifest=manifest, row_index=1, label="arm-b"
            )
            parquet_a.write_bytes(b"changed")
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "parquet SHA-256"):
                builder.build_pair_config(
                    arm_a_receipt=arm_a,
                    expected_arm_a_receipt_sha256=arm_a_sha,
                    arm_b_receipt=arm_b,
                    expected_arm_b_receipt_sha256=arm_b_sha,
                    output_pair_config=root / "parquet-tamper.json",
                )

            # Restore arm A, then create a distinct receipt file that still
            # points at the same materialized row/program as arm A.
            parquet_a.write_bytes(b"sealed-parquet-arm-a")
            duplicate = root / "duplicate-program.receipt.json"
            value = json.loads(arm_a.read_text(encoding="utf-8"))
            duplicate_parquet = root / "duplicate-program.parquet"
            duplicate_parquet.write_bytes(b"duplicate-parquet")
            value["parquet_path"] = str(duplicate_parquet)
            value["parquet_sha256"] = builder.file_sha256(duplicate_parquet)
            value = _sealed(value)
            _write_json(duplicate, value)
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "distinct rows/programs"):
                builder.build_pair_config(
                    arm_a_receipt=arm_a,
                    expected_arm_a_receipt_sha256=arm_a_sha,
                    arm_b_receipt=duplicate,
                    expected_arm_b_receipt_sha256=builder.file_sha256(duplicate),
                    output_pair_config=root / "same-program.json",
                )

    def test_pair_config_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, _, _, manifest = _build_manifest(root)
            arm_a, _, _ = _materialized_receipt(
                root, manifest=manifest, row_index=0, label="arm-a"
            )
            arm_b, arm_b_sha, _ = _materialized_receipt(
                root, manifest=manifest, row_index=1, label="arm-b"
            )
            text = arm_a.read_text(encoding="utf-8").strip()
            duplicate = text[:-1] + ',"row_id":"duplicate"}\n'
            arm_a.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(builder.RAMPC0PairBuilderError, "duplicate key"):
                builder.build_pair_config(
                    arm_a_receipt=arm_a,
                    expected_arm_a_receipt_sha256=builder.file_sha256(arm_a),
                    arm_b_receipt=arm_b,
                    expected_arm_b_receipt_sha256=arm_b_sha,
                    output_pair_config=root / "duplicate-key.json",
                )


if __name__ == "__main__":
    unittest.main()
