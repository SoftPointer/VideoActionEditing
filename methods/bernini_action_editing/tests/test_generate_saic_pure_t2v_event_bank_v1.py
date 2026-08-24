from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import generate_saic_pure_t2v_event_bank_v1 as generate  # noqa: E402
import saic_pure_t2v_event_bank_v1 as bank  # noqa: E402


SPEC_SHA = "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"


def make_envelope(root: Path) -> dict[str, object]:
    spec = json.loads(bank.ASSET_PATH.read_text(encoding="ascii"))
    candidate = spec["groups"][0]["candidates"][0]
    proxy = root / "black-exact81.mp4"
    proxy.write_bytes(b"black geometry only")
    return {
        "schema_version": bank.CANDIDATE_SCHEMA_VERSION,
        "root_spec_raw_sha256": SPEC_SHA,
        "source_manifest_content_sha256": bank.SOURCE_MANIFEST_CONTENT_SHA256,
        "group_id": "sp4-a",
        "actor_family": "dog",
        "visible_gpus": [0, 1, 2, 3],
        "sampling_contract": bank.SAMPLING_CONTRACT,
        "semantic_input_closure": bank.SEMANTIC_INPUT_CLOSURE,
        "geometry_proxy_contract": bank.GEOMETRY_PROXY_CONTRACT,
        "artifact_authority": bank.ARTIFACT_AUTHORITY,
        "candidate": candidate,
        "geometry_proxy": {
            "path": str(proxy),
            "sha256": bank.file_sha256(proxy),
            "height": candidate["source_geometry_hw"][0],
            "width": candidate["source_geometry_hw"][1],
            "source_media_read": False,
        },
    }


class SAICPureT2VGenerationTests(unittest.TestCase):
    def test_native_argv_reuses_real_t2v_entry_with_black_proxy_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            envelope = make_envelope(root)
            for name in ("bernini", "veomni", "checkpoint"):
                (root / name).mkdir()
            checkpoint_manifest = root / "checkpoint.sha256"
            checkpoint_manifest.write_text("manifest", encoding="ascii")
            args = SimpleNamespace(
                bernini_root=str(root / "bernini"),
                veomni_root=str(root / "veomni"),
                checkpoint=str(root / "checkpoint"),
                checkpoint_content_manifest=str(checkpoint_manifest),
                output_dir=str(root / "fresh-output"),
                method_source_revision="a" * 40,
                method_source_archive_sha256="b" * 64,
            )
            argv = generate.build_native_argv(args, envelope)
            self.assertEqual(argv[argv.index("--arms") + 1], "t2v")
            self.assertEqual(argv[argv.index("--num-inference-steps") + 1], "40")
            self.assertEqual(
                argv[argv.index("--source-video") + 1],
                str(Path(envelope["geometry_proxy"]["path"]).resolve()),
            )
            self.assertEqual(
                argv[argv.index("--expected-source-sha256") + 1],
                envelope["geometry_proxy"]["sha256"],
            )
            serialized = "\0".join(argv)
            source = source_set.load_manifest()
            for row in source["rows"]:
                self.assertNotIn(row["source_video"], serialized)
            for forbidden in ("--target-video", "--mask", "--flow", "--pose", "--track"):
                self.assertNotIn(forbidden, argv)

    def test_attempt_receipt_stays_pending_and_non_authorizing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            envelope = make_envelope(root)
            envelope_path = root / "candidate.json"
            envelope_path.write_bytes(bank.canonical_json_bytes(envelope) + b"\n")
            output = root / "attempt"
            output.mkdir()
            native_path = output / "receipt.json"
            native_path.write_text("{}\n", encoding="ascii")
            artifacts = {}
            for name in ("proposal.mp4", "clean.safetensors", "noise.safetensors"):
                path = output / name
                path.write_bytes(name.encode("ascii"))
                artifacts[name] = {"path": str(path), "sha256": bank.file_sha256(path)}
            verified = {
                "native_receipt_digest": "c" * 64,
                "bucket_hw": [480, 496],
                "latent_shape": [1, 16, 21, 60, 62],
                "mp4": artifacts["proposal.mp4"],
                "predecode_clean_latent": artifacts["clean.safetensors"],
                "official_initial_gaussian": artifacts["noise.safetensors"],
            }
            native_value = {
                "input": {
                    "source_video_path": str(
                        Path(envelope["geometry_proxy"]["path"]).resolve()
                    ),
                    "source_video_sha256": envelope["geometry_proxy"]["sha256"],
                }
            }
            args = SimpleNamespace(output_dir=str(output))
            with mock.patch.object(
                generate.native_receipts, "_load_json", return_value=native_value
            ), mock.patch.object(
                generate.native_receipts, "_verify_native_receipt", return_value=verified
            ):
                receipt_path = generate.bind_attempt_receipt(
                    args=args, envelope=envelope, envelope_path=envelope_path
                )
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            self.assertEqual(
                receipt["event_audit_status"], "pending_detached_full81_review"
            )
            self.assertFalse(receipt["event_verified"])
            self.assertFalse(receipt["identity_preservation_verified"])
            self.assertFalse(receipt["training_target_authorized"])
            self.assertFalse(receipt["optimizer_or_parameter_update_authorized"])
            self.assertFalse(
                receipt["real_source_nonuse_certificate"]["real_source_rgb_read"]
            )

    def test_generate_attempt_enforces_world4_sp4_and_create_only_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            envelope = make_envelope(root)
            envelope_path = root / "candidate.json"
            envelope_path.write_bytes(bank.canonical_json_bytes(envelope) + b"\n")
            args = SimpleNamespace(
                candidate_envelope=str(envelope_path),
                expected_root_spec_sha256=SPEC_SHA,
                output_dir=str(root / "fresh-output"),
            )
            with mock.patch.dict(
                os.environ,
                {"WORLD_SIZE": "4", "ROCR_VISIBLE_DEVICES": "0,1,2,3", "RANK": "1"},
                clear=False,
            ), mock.patch.object(generate.native, "main", return_value=0) as native_main, mock.patch.object(
                generate, "build_native_argv", return_value=["--fake"]
            ):
                self.assertEqual(generate.generate_attempt(args), 0)
            native_main.assert_called_once_with(["--fake"])
            (root / "fresh-output").mkdir()
            with mock.patch.dict(
                os.environ,
                {"WORLD_SIZE": "4", "ROCR_VISIBLE_DEVICES": "0,1,2,3", "RANK": "1"},
                clear=False,
            ), self.assertRaises(generate.SAICPureT2VGenerationError):
                generate.generate_attempt(args)

    def test_master_audit_never_self_qualifies_sixty_attempts(self) -> None:
        spec = json.loads(bank.ASSET_PATH.read_text(encoding="ascii"))
        source = source_set.load_manifest()
        gaussian = {
            "raw_value_sha256": "1" * 64,
            "content_sha256": "2" * 64,
            "shape": [1, 16, 21, 60, 62],
            "dtype": "torch.bfloat16",
            "stored_dtype": "torch.float32",
            "generator_initial_seed": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = root / "attempts"
            attempts.mkdir(parents=True)
            root_spec = root / "spec.json"
            root_spec.write_bytes(bank.ASSET_PATH.read_bytes())
            source_manifest = root / "source.json"
            source_manifest.write_bytes(source_set.ASSET_PATH.read_bytes())

            def fake_attempt(_path, *, candidate, **_kwargs):
                cell_gaussian = dict(gaussian)
                cell_gaussian["generator_initial_seed"] = candidate["seed"]
                return {
                    "receipt_digest": "d" * 64,
                    "artifacts": {
                        "official_initial_gaussian": cell_gaussian,
                        "mp4": {"path": "/tmp/proposal.mp4", "sha256": "e" * 64},
                    },
                }

            args = SimpleNamespace(
                root_spec=str(root_spec),
                source_manifest=str(source_manifest),
                expected_root_spec_sha256=SPEC_SHA,
                output_root=str(root),
            )
            with mock.patch.object(
                generate.contract, "load_sealed_spec", return_value=(spec, SPEC_SHA)
            ), mock.patch.object(
                generate.contract.source_set, "load_manifest", return_value=source
            ), mock.patch.object(
                generate.contract.source_set, "validate_manifest", return_value={}
            ), mock.patch.object(
                generate, "_load_attempt_receipt", side_effect=fake_attempt
            ), mock.patch.object(
                generate, "file_sha256", return_value="f" * 64
            ):
                self.assertEqual(generate.audit_bank(args), 0)
            master = json.loads(
                (root / generate.MASTER_RECEIPT_BASENAME).read_text(encoding="ascii")
            )
            self.assertEqual(master["attempt_count"], 60)
            self.assertEqual(len(master["same_seed_official_gaussian_proofs"]), 20)
            self.assertFalse(master["detached_full81_event_review_complete"])
            self.assertFalse(master["event_verified"])
            self.assertFalse(master["training_target_authorized"])
            self.assertFalse(master["optimizer_or_parameter_update_authorized"])


if __name__ == "__main__":
    unittest.main()
