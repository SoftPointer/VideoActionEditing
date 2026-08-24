from __future__ import annotations

import ast
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "motive" / "wan22_i2v_batch.py"
SMOKE = ROOT / "scripts" / "auh_wan22_i2v_smoke.sbatch"
FULL = ROOT / "scripts" / "auh_wan22_i2v_full.sbatch"
SUBMIT = ROOT / "scripts" / "auh_submit_wan22_i2v_chain.sh"
PARALLEL_SUBMIT = ROOT / "scripts" / "auh_submit_wan22_i2v_parallel.sh"
PARALLEL_RESUME = (
    ROOT / "scripts" / "auh_resume_wan22_i2v_parallel_published.sh"
)
FFPROBE_COMPAT = ROOT / "scripts" / "ffprobe_pyav_compat.py"

# Load the target file directly so this contract suite does not execute
# motive.__init__ (which intentionally imports NumPy-backed training helpers).
# The batch module itself must remain importable without NumPy, Torch, or Wan.
_SPEC = importlib.util.spec_from_file_location(
    "_wan22_i2v_batch_under_test",
    MODULE,
)
assert _SPEC is not None and _SPEC.loader is not None
_BATCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BATCH
_SPEC.loader.exec_module(_BATCH)

FIRST_FRAME_POLICY = _BATCH.FIRST_FRAME_POLICY
TEMPORAL_POLICY = _BATCH.TEMPORAL_POLICY
GENERATION_MANIFEST_SCHEMA = _BATCH.GENERATION_MANIFEST_SCHEMA
FULL_MOTION_GENERATION_SCHEMA = _BATCH.FULL_MOTION_GENERATION_SCHEMA
NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE = (
    _BATCH.NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
)
MODEL_HF_REVISION = _BATCH.MODEL_HF_REVISION
OFFICIAL_COMMIT = _BATCH.OFFICIAL_COMMIT
SAMPLE_SCHEMA = _BATCH.SAMPLE_SCHEMA
Wan22BatchError = _BATCH.Wan22BatchError
PINNED_MODEL_FILE_SPECS = _BATCH._PINNED_MODEL_FILE_SPECS
HF_METADATA_MTIME_TOLERANCE_SECONDS = (
    _BATCH.HF_METADATA_MTIME_TOLERANCE_SECONDS
)
EXPERT_NAMES = _BATCH._EXPERT_NAMES
EXPERT_SHARD_BASENAMES = _BATCH._EXPERT_SHARD_BASENAMES
EXPERT_INDEX_BASENAME = _BATCH._EXPERT_INDEX_BASENAME
_broadcast_rank0_payload = _BATCH._broadcast_rank0_payload
_collective_local_call = _BATCH._collective_local_call
_canonical_bytes = _BATCH._canonical_bytes
_object_digest = _BATCH._object_digest
build_run_contract = _BATCH.build_run_contract
ensure_run_contract = _BATCH.ensure_run_contract
inspect_hf_model_directory = _BATCH.inspect_hf_model_directory
load_generation_manifest = _BATCH.load_generation_manifest
load_non_production_preview_manifest = (
    _BATCH.load_non_production_preview_manifest
)
validate_generation_manifest_structure = (
    _BATCH.validate_generation_manifest_structure
)
normalize_ffprobe_payload = _BATCH.normalize_ffprobe_payload
sample_seed = _BATCH.sample_seed
validate_batch_temporal_grid = _BATCH.validate_batch_temporal_grid
validate_temporal_pair = _BATCH.validate_temporal_pair
validate_package_versions = _BATCH.validate_package_versions
validate_sample_commit = _BATCH.validate_sample_commit


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_row(
    *,
    iid: str = "sample-0001",
    group_id: str = "group-0001",
    authorized: bool = False,
) -> dict[str, object]:
    approval = (
        {
            "schema_version": _BATCH.APPROVAL_SCHEMA,
            "approval_digest": _sha(f"approval-{iid}".encode("utf-8")),
            "approval_file_sha256": _sha(
                f"approval-file-{iid}".encode("utf-8")
            ),
            "proposal_sha256": _sha(f"proposal-{iid}".encode("utf-8")),
            "reviewer_id": "unit-test-reviewer",
            "reviewed_at_utc": "2026-07-30T12:00:00+00:00",
            "decision": "approved",
            "reason": "target action was manually verified",
        }
        if authorized
        else None
    )
    return {
        "schema_version": GENERATION_MANIFEST_SCHEMA,
        "iid": iid,
        "group_id": group_id,
        "action_category": "interaction",
        "target_action_verb": "pick_up",
        "action_change_substantive": "yes",
        "source_video": "relative/source.mp4",
        "resolved_source_video": "/frozen/source.mp4",
        "source_video_sha256": "1" * 64,
        "anchor_image": "relative/anchor.png",
        "resolved_anchor_image": "/frozen/anchor.png",
        "anchor_sha256": "2" * 64,
        "edit_instruction": "Have the dog pick up the visible bone, then stand.",
        "absolute_target_prompt": (
            "The same seated dog first picks up the visible bone and then "
            "stands; identity, scene, and camera remain unchanged."
        ),
        "preservation_constraints": [
            "Preserve identity, appearance, background, and camera."
        ],
        "causal_stages": ["Pick up the visible bone.", "Stand while holding it."],
        "manifest_role": (
            _BATCH.APPROVED_MANIFEST_ROLE
            if authorized
            else "review_proposal"
        ),
        "production_eligible": authorized,
        "human_review_status": "approved" if authorized else "pending",
        "generation_authorized": authorized,
        "approval": approval,
    }


def _full_motion_preview_stub(
    *,
    iid: str = "full-motion-preview-0001",
) -> dict[str, object]:
    instruction = "Make every moving subject wave while keeping the camera locked."
    return {
        "schema_version": FULL_MOTION_GENERATION_SCHEMA,
        "iid": iid,
        "group_id": f"group-{iid}",
        "family": "motion_editing",
        "source_video": "/frozen/source.mp4",
        "resolved_source_video": "/frozen/source.mp4",
        "source_video_sha256": "1" * 64,
        "anchor_image": "/frozen/anchor.png",
        "resolved_anchor_image": "/frozen/anchor.png",
        "anchor_sha256": "2" * 64,
        "edit_instruction": instruction,
        "edit_instruction_sha256": _sha(instruction.encode("utf-8")),
        "qwen_evidence": {
            "result_digest": "3" * 64,
            "provenance_digest": "4" * 64,
        },
        "motion_spec": {"schema_version": "fixture"},
        "motion_spec_sha256": "5" * 64,
        "action_change_substantive": "yes",
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
    }


def _fake_full_motion_validator(
    validator: object,
) -> tuple[ModuleType, ModuleType]:
    package = ModuleType("motive")
    package.__path__ = []  # type: ignore[attr-defined]
    finalizer = ModuleType("motive.goku_full_motion_finalize")
    finalizer.validate_generation_row = validator  # type: ignore[attr-defined]
    package.goku_full_motion_finalize = finalizer  # type: ignore[attr-defined]
    return package, finalizer


def _video_probe(
    *,
    frames: int = 81,
    frame_rate: str = "25/1",
    duration: float = 3.24,
) -> dict[str, object]:
    rate = _BATCH.Fraction(frame_rate)
    nominal = float(_BATCH.Fraction(frames, 1) / rate)
    error = abs(duration - nominal)
    return {
        "probe_backend": "ffprobe",
        "codec": "h264",
        "configured_codec": None,
        "codec_family": "h264",
        "pixel_format": "yuv420p",
        "width": 704,
        "height": 1280,
        "frame_rate": frame_rate,
        "frame_rate_fields": {
            "avg_frame_rate": frame_rate,
            "r_frame_rate": frame_rate,
        },
        "frames": frames,
        "duration_seconds": duration,
        "nominal_duration_seconds": nominal,
        "nominal_duration_error_seconds": error,
        "nominal_duration_error_frames": error * float(rate),
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "container_bytes": 1234,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    )


def _write_exact_size(path: Path, payload: bytes, size: int) -> None:
    if len(payload) > size:
        raise AssertionError(f"fixture payload exceeds pinned size for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.write(b" " * (size - len(payload)))


def _write_hf_metadata(
    checkpoint: Path,
    relative: str,
    *,
    revision: str = MODEL_HF_REVISION,
    etag: str | None = None,
) -> None:
    expected_etag = PINNED_MODEL_FILE_SPECS[relative][1]
    metadata = (
        checkpoint
        / ".cache"
        / "huggingface"
        / "download"
        / f"{relative}.metadata"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        f"{revision}\n{etag or expected_etag}\n{time.time() + 5}\n",
        encoding="utf-8",
    )


def _create_sparse_pinned_checkpoint(checkpoint: Path) -> None:
    for relative, (size, _) in PINNED_MODEL_FILE_SPECS.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(EXPERT_INDEX_BASENAME):
            expert = relative.split("/", 1)[0]
            total_size = sum(
                PINNED_MODEL_FILE_SPECS[f"{expert}/{name}"][0]
                for name in EXPERT_SHARD_BASENAMES
            )
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": {
                    f"fixture.parameter.{index}": shard
                    for index, shard in enumerate(EXPERT_SHARD_BASENAMES)
                },
            }
            _write_exact_size(
                path,
                json.dumps(index, sort_keys=True).encode("utf-8"),
                size,
            )
        else:
            with path.open("wb") as handle:
                handle.truncate(size)
        _write_hf_metadata(checkpoint, relative)


class _SingleRankDist:
    def broadcast_object_list(self, container: list[object], src: int) -> None:
        if src != 0:
            raise AssertionError("fixture supports only rank zero")

    def all_gather_object(
        self,
        output: list[object],
        value: object,
    ) -> None:
        output[0] = value


class Wan22ManifestContractTests(unittest.TestCase):
    def test_preview_cli_is_mutually_exclusive_with_signed_release(self) -> None:
        common = [
            "--manifest",
            "/frozen/manifest.jsonl",
            "--output-root",
            "/frozen/output",
            "--wan-code-root",
            "/frozen/wan",
            "--ckpt-dir",
            "/frozen/checkpoint",
        ]
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                _BATCH._parser().parse_args(
                    common
                    + [
                        "--non-production-preview",
                        "--signed-release",
                        "/frozen/release.json",
                    ]
                )
        parsed = _BATCH._parser().parse_args(
            common + ["--non-production-preview"]
        )
        self.assertTrue(parsed.non_production_preview)
        self.assertIsNone(parsed.signed_release)

    def test_preview_loader_requires_exactly_one_deeply_validated_v6_row(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "preview.jsonl"
            row = _full_motion_preview_stub()
            _write_jsonl(manifest, [row])
            validator = mock.Mock(return_value=dict(row))
            package, finalizer = _fake_full_motion_validator(validator)
            with mock.patch.dict(
                sys.modules,
                {
                    "motive": package,
                    "motive.goku_full_motion_finalize": finalizer,
                },
            ):
                loaded = load_non_production_preview_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )
            validator.assert_called_once_with(row)
            prepared = loaded["selected_rows"][0]
            self.assertEqual(loaded["manifest_row_count"], 1)
            self.assertEqual(loaded["selected_row_count"], 1)
            self.assertEqual(
                prepared["_authorization_mode"],
                NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE,
            )
            self.assertEqual(prepared["_row_digest"], _object_digest(row))
            self.assertEqual(prepared["action_category"], "full_motion")

            _write_jsonl(manifest, [row, _full_motion_preview_stub(iid="two")])
            with self.assertRaisesRegex(Wan22BatchError, "exactly one"):
                load_non_production_preview_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )

            wrong_schema = dict(row, schema_version="old-full-motion-schema")
            _write_jsonl(manifest, [wrong_schema])
            with self.assertRaisesRegex(
                Wan22BatchError, FULL_MOTION_GENERATION_SCHEMA
            ):
                load_non_production_preview_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )

    def test_preview_loader_propagates_deep_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "preview.jsonl"
            row = _full_motion_preview_stub()
            _write_jsonl(manifest, [row])
            validator = mock.Mock(side_effect=ValueError("Qwen closure differs"))
            package, finalizer = _fake_full_motion_validator(validator)
            with mock.patch.dict(
                sys.modules,
                {
                    "motive": package,
                    "motive.goku_full_motion_finalize": finalizer,
                },
            ):
                with self.assertRaisesRegex(
                    Wan22BatchError,
                    "deep validation failed.*Qwen closure differs",
                ):
                    load_non_production_preview_manifest(
                        manifest,
                        allow_pending_review=False,
                        max_samples=1,
                    )

    def test_pending_review_and_legacy_override_are_both_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "generation_manifest.jsonl"
            _write_jsonl(manifest, [_manifest_row()])
            with self.assertRaisesRegex(
                Wan22BatchError,
                "signed generation release gate is unavailable",
            ):
                load_generation_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )
            with self.assertRaisesRegex(
                Wan22BatchError,
                "signed generation release gate is unavailable",
            ):
                load_generation_manifest(
                    manifest,
                    allow_pending_review=True,
                    max_samples=1,
                )

    def test_legacy_approved_generation_manifest_is_never_authorized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "generation_manifest.jsonl"
            _write_jsonl(manifest, [_manifest_row(authorized=True)])
            with self.assertRaisesRegex(
                Wan22BatchError,
                "signed generation release gate is unavailable",
            ):
                load_generation_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )

            audit = validate_generation_manifest_structure(
                manifest,
                allow_pending_review=False,
                max_samples=None,
            )
            self.assertEqual(
                audit["selected_rows"][0]["_authorization_mode"],
                "legacy_approval_record_untrusted",
            )

    def test_approval_record_is_closed_and_proposal_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "generation_manifest.jsonl"
            base = _manifest_row(authorized=True)
            mutations = []
            missing = json.loads(json.dumps(base))
            del missing["approval"]["proposal_sha256"]
            mutations.append(missing)
            extra = json.loads(json.dumps(base))
            extra["approval"]["unbound_note"] = "not allowed"
            mutations.append(extra)
            wrong_schema = json.loads(json.dumps(base))
            wrong_schema["approval"]["schema_version"] = "legacy"
            mutations.append(wrong_schema)
            wrong_digest = json.loads(json.dumps(base))
            wrong_digest["approval"]["approval_digest"] = "not-a-sha"
            mutations.append(wrong_digest)
            wrong_decision = json.loads(json.dumps(base))
            wrong_decision["approval"]["decision"] = "rejected"
            mutations.append(wrong_decision)

            for row in mutations:
                _write_jsonl(manifest, [row])
                with self.assertRaises(Wan22BatchError):
                    validate_generation_manifest_structure(
                        manifest,
                        allow_pending_review=False,
                        max_samples=None,
                    )

    def test_pending_v8_and_boolean_resign_cannot_bypass_release_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending_v8 = _manifest_row()
            pending_v8["qwen_provenance_schema"] = (
                "goku-action-anchor-qwen-provenance-v8"
            )
            boolean_resign = _manifest_row(authorized=True)
            boolean_resign["resigned_manifest_sha256"] = _sha(
                _canonical_bytes(boolean_resign)
            )
            for name, row in (
                ("pending-v8", pending_v8),
                ("boolean-resign", boolean_resign),
            ):
                with self.subTest(case=name):
                    manifest = root / f"{name}.jsonl"
                    _write_jsonl(manifest, [row])
                    with self.assertRaisesRegex(
                        Wan22BatchError,
                        "signed generation release gate is unavailable",
                    ):
                        load_generation_manifest(
                            manifest,
                            allow_pending_review=False,
                            max_samples=None,
                        )
            with mock.patch.object(
                _BATCH,
                "SIGNED_RELEASE_VERIFIER_AVAILABLE",
                True,
            ):
                with self.assertRaisesRegex(
                    Wan22BatchError,
                    "signed generation release gate is unavailable",
                ):
                    load_generation_manifest(
                        root / "boolean-resign.jsonl",
                        allow_pending_review=False,
                        max_samples=None,
                    )

    def test_cli_pending_override_is_disabled_before_runtime_imports(self) -> None:
        args = _BATCH._parser().parse_args(
            [
                "--manifest",
                "/frozen/manifest.jsonl",
                "--output-root",
                "/frozen/output",
                "--wan-code-root",
                "/frozen/wan",
                "--ckpt-dir",
                "/frozen/checkpoint",
                "--allow-pending-review",
            ]
        )
        with self.assertRaisesRegex(
            Wan22BatchError,
            "--allow-pending-review is disabled",
        ):
            _BATCH._validate_args(args)

    def test_signed_release_gate_fails_before_torch_or_model_load(self) -> None:
        args = _BATCH._parser().parse_args(
            [
                "--manifest",
                "/frozen/manifest.jsonl",
                "--output-root",
                "/frozen/output",
                "--wan-code-root",
                "/frozen/wan",
                "--ckpt-dir",
                "/frozen/checkpoint",
            ]
        )
        with mock.patch.dict(
            sys.modules,
            {"torch": None, "torch.distributed": None},
        ):
            with self.assertRaisesRegex(
                Wan22BatchError,
                "signed generation release gate is unavailable",
            ):
                _BATCH.run_batch(args)

    def test_duplicate_iid_and_missing_final_newline_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "duplicate.jsonl"
            row = _manifest_row(authorized=True)
            _write_jsonl(manifest, [row, row])
            with self.assertRaisesRegex(Wan22BatchError, "duplicate manifest iid"):
                validate_generation_manifest_structure(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )
            manifest.write_bytes(_canonical_bytes(row))
            with self.assertRaisesRegex(Wan22BatchError, "end with a newline"):
                validate_generation_manifest_structure(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                )

    def test_seed_is_order_independent_and_stable(self) -> None:
        first = sample_seed(260730, "sample-0001")
        self.assertEqual(first, sample_seed(260730, "sample-0001"))
        self.assertNotEqual(first, sample_seed(260730, "sample-0002"))
        self.assertNotEqual(first, sample_seed(260731, "sample-0001"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1 << 63)

    def test_run_contract_is_create_only_and_exact_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            contract = {
                "schema_version": "test",
                "contract_digest": _object_digest({"test": True}),
            }
            path = ensure_run_contract(output, contract)
            self.assertEqual(json.loads(path.read_text()), contract)
            ensure_run_contract(output, contract)
            changed = dict(contract)
            changed["extra"] = True
            with self.assertRaisesRegex(Wan22BatchError, "contract differs"):
                ensure_run_contract(output, changed)

    def test_nonempty_unbound_output_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "foreign.txt").write_text("not ours", encoding="utf-8")
            with self.assertRaisesRegex(Wan22BatchError, "no run contract"):
                ensure_run_contract(
                    output,
                    {"schema_version": "test", "contract_digest": "x"},
                )


class Wan22PreflightTests(unittest.TestCase):
    def test_package_inspection_does_not_shadow_importlib_metadata(self) -> None:
        versions = {
            "torch": "2.7.1+rocm6.3",
            "torchvision": "0.22.1",
            "transformers": "4.51.3",
            "diffusers": "0.38.0",
            "accelerate": "1.13.0",
            "numpy": "1.26.4",
            "Pillow": "11.3.0",
            "imageio": "2.37.3",
            "imageio-ffmpeg": "0.6.0",
            "easydict": "1.13",
            "einops": "0.8.2",
            "tqdm": "4.67.3",
            "safetensors": "0.8.0",
            "tokenizers": "0.21.4",
            "flash-attn": "2.7.4.post1",
            "ftfy": "6.3.1",
            "regex": "2026.4.4",
            "sentencepiece": "0.2.1",
        }

        def distribution_version(name: str) -> str:
            return versions[name]

        def import_module(name: str) -> SimpleNamespace:
            return SimpleNamespace(__file__=f"/frozen/site-packages/{name}.py")

        with mock.patch.object(
            _BATCH.importlib.metadata,
            "version",
            side_effect=distribution_version,
        ):
            with mock.patch.object(
                _BATCH.importlib,
                "import_module",
                side_effect=import_module,
            ):
                evidence = _BATCH.inspect_python_packages()

        self.assertEqual(evidence["packages"], versions)
        self.assertEqual(
            set(evidence["import_smoke_paths"]),
            set(versions) - {"torch"},
        )

    def test_official_dependency_bounds(self) -> None:
        valid = {
            "torch": "2.7.1+rocm6.3",
            "torchvision": "0.22.1",
            "transformers": "4.51.0",
            "diffusers": "0.31.0",
            "accelerate": "1.2.0",
            "numpy": "1.26.4",
            "Pillow": "10.4.0",
            "imageio": "2.37.0",
            "imageio-ffmpeg": "0.6.0",
            "easydict": "1.13",
            "einops": "0.8.0",
            "tqdm": "4.67.0",
            "safetensors": "0.5.0",
            "tokenizers": "0.21.0",
            "flash-attn": "2.7.4",
            "ftfy": "6.3.0",
            "regex": "2025.1.1",
            "sentencepiece": "0.2.0",
        }
        validate_package_versions(valid)
        too_new = dict(valid, transformers="5.5.4")
        with self.assertRaisesRegex(Wan22BatchError, "transformers"):
            validate_package_versions(too_new)
        numpy_two = dict(valid, numpy="2.0.0")
        with self.assertRaisesRegex(Wan22BatchError, "numpy<2"):
            validate_package_versions(numpy_two)

    def test_hf_payload_binds_every_size_etag_revision_and_expert_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            _create_sparse_pinned_checkpoint(checkpoint)
            evidence = inspect_hf_model_directory(checkpoint)
            self.assertEqual(evidence["hf_revision"], MODEL_HF_REVISION)
            self.assertEqual(
                evidence["metadata_file_count"],
                len(PINNED_MODEL_FILE_SPECS),
            )
            self.assertEqual(set(evidence["expert_indexes"]), set(EXPERT_NAMES))
            for expert in EXPERT_NAMES:
                self.assertEqual(
                    evidence["expert_indexes"][expert]["referenced_shards"],
                    sorted(EXPERT_SHARD_BASENAMES),
                )

            missing = (
                checkpoint
                / EXPERT_NAMES[0]
                / EXPERT_SHARD_BASENAMES[0]
            )
            missing.unlink()
            with self.assertRaisesRegex(Wan22BatchError, "pinned Wan payload"):
                inspect_hf_model_directory(checkpoint)

    def test_hf_payload_rejects_size_etag_revision_and_stale_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            _create_sparse_pinned_checkpoint(checkpoint)
            relative = f"{EXPERT_NAMES[1]}/{EXPERT_SHARD_BASENAMES[-1]}"
            payload = checkpoint / relative
            expected_size = PINNED_MODEL_FILE_SPECS[relative][0]
            with payload.open("r+b") as handle:
                handle.truncate(expected_size - 1)
            with self.assertRaisesRegex(Wan22BatchError, "size mismatch"):
                inspect_hf_model_directory(checkpoint)

            with payload.open("r+b") as handle:
                handle.truncate(expected_size)
            _write_hf_metadata(checkpoint, relative)
            _write_hf_metadata(checkpoint, relative, etag="f" * 64)
            with self.assertRaisesRegex(Wan22BatchError, "ETag mismatch"):
                inspect_hf_model_directory(checkpoint)

            _write_hf_metadata(
                checkpoint,
                relative,
                revision="a" * 40,
            )
            with self.assertRaisesRegex(Wan22BatchError, "revision mismatch"):
                inspect_hf_model_directory(checkpoint)

            _write_hf_metadata(checkpoint, relative)
            metadata = (
                checkpoint
                / ".cache"
                / "huggingface"
                / "download"
                / f"{relative}.metadata"
            )
            metadata.write_text(
                (
                    f"{MODEL_HF_REVISION}\n"
                    f"{PINNED_MODEL_FILE_SPECS[relative][1]}\n1\n"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Wan22BatchError, "stale"):
                inspect_hf_model_directory(checkpoint)

    def test_hf_metadata_allows_only_bounded_nfs_clock_skew(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            _create_sparse_pinned_checkpoint(checkpoint)
            relative = "google/umt5-xxl/special_tokens_map.json"
            payload = checkpoint / relative
            metadata = (
                checkpoint
                / ".cache"
                / "huggingface"
                / "download"
                / f"{relative}.metadata"
            )
            revision, etag, timestamp_text = metadata.read_text(
                encoding="utf-8"
            ).splitlines()
            timestamp = float(timestamp_text)

            tolerated_mtime = (
                timestamp + HF_METADATA_MTIME_TOLERANCE_SECONDS - 1.0
            )
            payload.touch()
            os.utime(payload, (tolerated_mtime, tolerated_mtime))
            evidence = inspect_hf_model_directory(checkpoint)
            matching = {
                row["path"]: row for row in evidence["verified_files"]
            }[relative]
            self.assertAlmostEqual(
                matching["payload_mtime_delta_seconds"],
                HF_METADATA_MTIME_TOLERANCE_SECONDS - 1.0,
            )

            stale_mtime = (
                timestamp + HF_METADATA_MTIME_TOLERANCE_SECONDS + 1.0
            )
            os.utime(payload, (stale_mtime, stale_mtime))
            with self.assertRaisesRegex(Wan22BatchError, "clock-skew"):
                inspect_hf_model_directory(checkpoint)

    def test_ffprobe_payload_is_normalized_and_frame_bound(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 832,
                    "height": 480,
                    "r_frame_rate": "16/1",
                    "avg_frame_rate": "16/1",
                    "nb_frames": "17",
                    "nb_read_frames": "17",
                }
            ],
            "format": {
                "duration": "1.0625",
                "size": "1234",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            },
        }
        normalized = normalize_ffprobe_payload(
            payload,
            expected_frames=17,
            expected_width=832,
            expected_height=480,
            expected_fps=16,
            expected_codec="libx264",
        )
        self.assertEqual(normalized["frames"], 17)
        self.assertEqual(normalized["frame_rate"], "16/1")
        self.assertEqual(normalized["nominal_duration_seconds"], 1.0625)
        self.assertEqual(normalized["nominal_duration_error_frames"], 0.0)
        self.assertEqual(normalized["codec_family"], "h264")
        self.assertEqual(normalized["configured_codec"], "libx264")
        with self.assertRaisesRegex(Wan22BatchError, "frame mismatch"):
            normalize_ffprobe_payload(payload, expected_frames=81)
        wrong_fps = json.loads(json.dumps(payload))
        wrong_fps["streams"][0]["avg_frame_rate"] = "15/1"
        with self.assertRaisesRegex(Wan22BatchError, "frame_rate mismatch"):
            normalize_ffprobe_payload(
                wrong_fps,
                expected_frames=17,
                expected_fps=16,
            )
        wrong_codec = json.loads(json.dumps(payload))
        wrong_codec["streams"][0]["codec_name"] = "hevc"
        with self.assertRaisesRegex(Wan22BatchError, "codec mismatch"):
            normalize_ffprobe_payload(
                wrong_codec,
                expected_frames=17,
                expected_codec="libx264",
            )
        variable_rate = json.loads(json.dumps(payload))
        variable_rate["streams"][0]["avg_frame_rate"] = "15/1"
        with self.assertRaisesRegex(
            Wan22BatchError,
            "variable frame-rate grid",
        ):
            normalize_ffprobe_payload(
                variable_rate,
                expected_frames=17,
            )
        wrong_duration = json.loads(json.dumps(payload))
        wrong_duration["format"]["duration"] = "2.0"
        with self.assertRaisesRegex(
            Wan22BatchError,
            "duration differs.*constant frame grid",
        ):
            normalize_ffprobe_payload(
                wrong_duration,
                expected_frames=17,
                max_nominal_duration_error_frames=1,
            )

    def test_source_bound_temporal_grid_and_pair_contract(self) -> None:
        rows = [
            {
                "_iid": "sample-a",
                "_input_media": {
                    "source_video_ffprobe": _video_probe(),
                },
            },
            {
                "_iid": "sample-b",
                "_input_media": {
                    "source_video_ffprobe": _video_probe(duration=3.241),
                },
            },
        ]
        policy = validate_batch_temporal_grid(
            rows,
            expected_frame_num=81,
        )
        self.assertEqual(policy["policy_version"], TEMPORAL_POLICY)
        self.assertEqual(policy["source_frame_count"], 81)
        self.assertEqual(policy["target_frame_count"], 81)
        self.assertEqual(policy["source_frame_rate"], "25/1")
        self.assertEqual(policy["target_container_frame_rate"], "25/1")
        self.assertEqual(policy["model_sample_fps"], 16)
        self.assertEqual(policy["duration_match_tolerance_frames"], 1)

        pair = validate_temporal_pair(
            source_probe=_video_probe(),
            target_probe=_video_probe(duration=3.25),
        )
        self.assertTrue(pair["frame_count_equal"])
        self.assertTrue(pair["frame_rate_equal"])
        self.assertTrue(pair["duration_within_tolerance"])
        self.assertAlmostEqual(pair["duration_delta_frames"], 0.25)

    def test_run_contract_names_model_and_container_rates_separately(self) -> None:
        row = _manifest_row(authorized=True)
        row.update(
            {
                "_iid": row["iid"],
                "_row_digest": "3" * 64,
                "_authorization_mode": "bound_human_approval",
                "_input_media": {
                    "anchor_rgb_sha256": "4" * 64,
                    "anchor_width": 704,
                    "anchor_height": 1280,
                    "source_video_ffprobe": _video_probe(),
                },
            }
        )
        temporal = validate_batch_temporal_grid(
            [row],
            expected_frame_num=81,
        )
        args = SimpleNamespace(
            base_seed=260730,
            max_samples=None,
            expected_world_size=8,
            max_new_samples=1,
            require_rocm=True,
            expected_gpu_name_substring="MI210",
            size="1280*720",
            frame_num=81,
            sample_steps=40,
            sample_shift=5.0,
            sample_solver="unipc",
            sample_guide_scale_low=3.5,
            sample_guide_scale_high=3.5,
            video_codec="libx264",
            video_quality=8,
            allow_pending_review=False,
        )
        contract = build_run_contract(
            manifest={
                "manifest_path": "/frozen/generation_manifest.jsonl",
                "manifest_sha256": "5" * 64,
                "manifest_bytes": 123,
                "manifest_row_count": 1,
                "selected_row_count": 1,
            },
            prepared_rows=[row],
            temporal_policy=temporal,
            args=args,
            official={"commit": OFFICIAL_COMMIT},
            model={"revision": MODEL_HF_REVISION},
            runtime={"world_size": 8},
        )
        parameters = contract["generation_parameters"]
        self.assertEqual(parameters["model_sample_fps"], 16)
        self.assertEqual(parameters["output_container_frame_rate"], "25/1")
        self.assertNotIn("fps", parameters)
        self.assertEqual(contract["temporal_policy"], temporal)
        self.assertNotIn("production_use_forbidden", contract)
        self.assertNotIn("non_production_preview", contract)
        self.assertNotIn("preview_bindings", contract["selected_inputs"][0])

    def test_preview_contract_and_generated_manifest_bind_provenance_and_ban_production(
        self,
    ) -> None:
        row = _full_motion_preview_stub()
        row["motion_spec_sha256"] = _object_digest(row["motion_spec"])
        row.update(
            {
                "_iid": row["iid"],
                "_row_digest": _object_digest(row),
                "_authorization_mode": (
                    NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
                ),
                "action_category": "full_motion",
                "target_action_verb": "multi_entity_action_edit",
                "_input_media": {
                    "anchor_rgb_sha256": "6" * 64,
                    "anchor_width": 704,
                    "anchor_height": 1280,
                    "source_video_ffprobe": _video_probe(),
                },
            }
        )
        temporal = validate_batch_temporal_grid([row], expected_frame_num=81)
        args = SimpleNamespace(
            base_seed=260730,
            max_samples=None,
            expected_world_size=8,
            max_new_samples=1,
            require_rocm=True,
            expected_gpu_name_substring="MI210",
            size="1280*720",
            frame_num=81,
            sample_steps=4,
            sample_shift=5.0,
            sample_solver="unipc",
            sample_guide_scale_low=3.5,
            sample_guide_scale_high=3.5,
            video_codec="libx264",
            video_quality=8,
            allow_pending_review=False,
        )
        manifest_sha = "7" * 64
        contract = build_run_contract(
            manifest={
                "manifest_path": "/frozen/full-motion-preview.jsonl",
                "manifest_sha256": manifest_sha,
                "manifest_bytes": 123,
                "manifest_row_count": 1,
                "selected_row_count": 1,
                "non_production_preview": True,
            },
            prepared_rows=[row],
            temporal_policy=temporal,
            args=args,
            official={"commit": OFFICIAL_COMMIT},
            model={"revision": MODEL_HF_REVISION},
            runtime={"world_size": 8},
        )
        expected_bindings = {
            "manifest_sha256": manifest_sha,
            "manifest_row_digest": row["_row_digest"],
            "edit_instruction_sha256": row["edit_instruction_sha256"],
            "qwen_result_digest": row["qwen_evidence"]["result_digest"],
            "qwen_provenance_digest": row["qwen_evidence"][
                "provenance_digest"
            ],
        }
        self.assertTrue(contract["production_use_forbidden"])
        self.assertEqual(
            contract["authorization"]["mode"],
            NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE,
        )
        self.assertTrue(contract["authorization"]["production_use_forbidden"])
        self.assertEqual(
            contract["non_production_preview"]["bindings"],
            expected_bindings,
        )
        self.assertEqual(
            contract["selected_inputs"][0]["preview_bindings"],
            expected_bindings,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            sample_dir = output_root / "samples" / str(row["iid"])
            sample_dir.mkdir(parents=True)
            source_copy = sample_dir / "source_video.mp4"
            instruction_file = sample_dir / "edit_instruction.txt"
            motion_file = sample_dir / "motion_spec.json"
            source_copy.write_bytes(b"source")
            instruction_file.write_bytes(
                str(row["edit_instruction"]).encode("utf-8")
            )
            motion_file.write_text(
                json.dumps(row["motion_spec"]), encoding="utf-8"
            )
            result = {
                "result_digest": "8" * 64,
                "seed": 9,
                "authorization_mode": (
                    NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
                ),
                "production_use_forbidden": True,
                "preview_bindings": expected_bindings,
                "temporal_policy": {},
                "outputs": {
                    "source_video": source_copy.name,
                    "source_video_sha256": "a" * 64,
                    "source_video_bytes": source_copy.stat().st_size,
                    "edit_instruction_file": instruction_file.name,
                    "edit_instruction_file_sha256": row[
                        "edit_instruction_sha256"
                    ],
                    "edit_instruction_file_bytes": instruction_file.stat().st_size,
                    "motion_spec_json": motion_file.name,
                    "motion_spec_json_sha256": _sha(motion_file.read_bytes()),
                    "motion_spec_json_bytes": motion_file.stat().st_size,
                    "motion_spec_object_sha256": row["motion_spec_sha256"],
                    "conditioning_anchor_original": "anchor.png",
                    "conditioning_anchor_original_sha256": "b" * 64,
                    "conditioning_frame0_float32": "frame0.npy",
                    "conditioning_frame0_float32_sha256": "c" * 64,
                    "conditioning_frame0_png": "frame0.png",
                    "conditioning_frame0_png_sha256": "d" * 64,
                    "preview_mp4": "preview.mp4",
                    "preview_mp4_sha256": "e" * 64,
                },
            }
            generated = _BATCH._generated_manifest_rows(
                output_root=output_root,
                rows=[row],
                results={0: result},
            )[0]
        self.assertTrue(generated["production_use_forbidden"])
        self.assertEqual(generated["preview_bindings"], expected_bindings)
        self.assertEqual(
            generated["authorization_mode"],
            NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE,
        )

    def test_temporal_preflight_rejects_unsupported_or_mixed_sources(self) -> None:
        unsupported = [
            {
                "_iid": "bad-count",
                "_input_media": {
                    "source_video_ffprobe": _video_probe(
                        frames=80,
                        duration=3.2,
                    ),
                },
            }
        ]
        with self.assertRaisesRegex(
            Wan22BatchError,
            "unsupported source frame count",
        ):
            validate_batch_temporal_grid(
                unsupported,
                expected_frame_num=81,
            )

        wrong_requested_count = [
            {
                "_iid": "wrong-request",
                "_input_media": {
                    "source_video_ffprobe": _video_probe(),
                },
            }
        ]
        with self.assertRaisesRegex(
            Wan22BatchError,
            "source frame count differs from --frame-num",
        ):
            validate_batch_temporal_grid(
                wrong_requested_count,
                expected_frame_num=17,
            )

        mixed_rate = [
            {
                "_iid": "rate-a",
                "_input_media": {
                    "source_video_ffprobe": _video_probe(),
                },
            },
            {
                "_iid": "rate-b",
                "_input_media": {
                    "source_video_ffprobe": _video_probe(
                        frame_rate="24/1",
                        duration=3.375,
                    ),
                },
            },
        ]
        with self.assertRaisesRegex(
            Wan22BatchError,
            "incompatible source frame rates",
        ):
            validate_batch_temporal_grid(
                mixed_rate,
                expected_frame_num=81,
            )

        with self.assertRaisesRegex(
            Wan22BatchError,
            "source/target frame rate mismatch",
        ):
            validate_temporal_pair(
                source_probe=_video_probe(),
                target_probe=_video_probe(
                    frame_rate="24/1",
                    duration=3.375,
                ),
            )
        with self.assertRaisesRegex(
            Wan22BatchError,
            "duration mismatch exceeds one frame",
        ):
            validate_temporal_pair(
                source_probe=_video_probe(duration=3.20),
                target_probe=_video_probe(duration=3.28),
            )

    def test_collective_helpers_propagate_rank_errors(self) -> None:
        dist = _SingleRankDist()

        def fail() -> None:
            raise ValueError("encoder commit failed")

        with self.assertRaisesRegex(
            Wan22BatchError,
            "sample commit.*failed on rank zero.*encoder commit failed",
        ):
            _broadcast_rank0_payload(
                dist,
                rank=0,
                producer=fail,
                stage="sample commit iid=fixture",
            )
        with self.assertRaisesRegex(
            Wan22BatchError,
            "kernel smoke failed collectively.*encoder commit failed",
        ):
            _collective_local_call(
                dist,
                rank=0,
                world_size=1,
                stage="kernel smoke",
                producer=fail,
            )


class Wan22ResumeCommitTests(unittest.TestCase):
    def test_preview_result_requires_bound_production_forbidden_marker(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample_dir = Path(temporary) / "sample"
            sample_dir.mkdir()
            row = _full_motion_preview_stub()
            row.update(
                {
                    "_iid": row["iid"],
                    "_row_digest": _object_digest(row),
                    "_authorization_mode": (
                        NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
                    ),
                }
            )
            manifest_sha = "6" * 64
            bindings = _BATCH._non_production_preview_bindings(
                row, manifest_sha256=manifest_sha
            )
            contract = {
                "contract_digest": "7" * 64,
                "manifest": {"sha256": manifest_sha},
                "generation_parameters": {"base_seed": 260730},
                "production_use_forbidden": True,
                "non_production_preview": {
                    "authorization_mode": (
                        NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
                    ),
                    "production_use_forbidden": True,
                    "bindings": bindings,
                },
            }
            result = {
                "schema_version": SAMPLE_SCHEMA,
                "iid": row["iid"],
                "sample_index": 0,
                "manifest_sha256": manifest_sha,
                "manifest_row_digest": row["_row_digest"],
                "contract_digest": contract["contract_digest"],
                "seed": sample_seed(260730, str(row["iid"])),
                "authorization_mode": (
                    NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
                ),
                "manifest_role": "review_proposal",
                "production_eligible": False,
                "approval": None,
                "human_review_status_at_generation": "pending",
                "generation_authorized_in_manifest": False,
                "prompt": {
                    "field": "edit_instruction",
                    "text": row["edit_instruction"],
                    "sha256": row["edit_instruction_sha256"],
                },
                "preview_bindings": bindings,
            }

            def write_result(value: dict[str, object]) -> None:
                payload = dict(value)
                payload["result_digest"] = _object_digest(payload)
                (sample_dir / "result.json").write_bytes(
                    json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
                )

            write_result(result)
            with self.assertRaisesRegex(
                Wan22BatchError, "bound approval provenance mismatch"
            ):
                validate_sample_commit(
                    sample_dir,
                    row=row,
                    contract=contract,
                    sample_index=0,
                )

            result["production_use_forbidden"] = True
            write_result(result)
            with self.assertRaisesRegex(Wan22BatchError, "outputs are missing"):
                validate_sample_commit(
                    sample_dir,
                    row=row,
                    contract=contract,
                    sample_index=0,
                )

    def test_self_contained_input_materialization_is_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "原视频.WebM"
            source.write_bytes(b"exact-source-video-bytes")
            staging = root / "staging"
            staging.mkdir()
            motion_spec = {
                "schema_version": "fixture-motion-spec-v1",
                "compiled_instruction": "all moving subjects are edited",
            }
            row = _manifest_row()
            row.update(
                {
                    "_iid": "sample-0001",
                    "source_video_sha256": _sha(source.read_bytes()),
                    "edit_instruction": "让两个人都改为挥手。",
                    "motion_spec": motion_spec,
                    "motion_spec_sha256": _object_digest(motion_spec),
                    "_input_media": {"source_video_path": str(source)},
                }
            )
            outputs = _BATCH._materialize_self_contained_inputs(
                staging,
                row=row,
            )
            copied = staging / "source_video.WebM"
            instruction = staging / "edit_instruction.txt"
            self.assertEqual(outputs["source_video"], copied.name)
            self.assertEqual(copied.read_bytes(), source.read_bytes())
            self.assertNotEqual(os.stat(copied).st_ino, os.stat(source).st_ino)
            self.assertEqual(
                instruction.read_bytes(), row["edit_instruction"].encode("utf-8")
            )
            self.assertFalse(instruction.read_bytes().endswith(b"\n"))
            self.assertEqual(
                json.loads((staging / "motion_spec.json").read_text("utf-8")),
                motion_spec,
            )
            self.assertEqual(
                outputs["motion_spec_object_sha256"],
                row["motion_spec_sha256"],
            )

    def test_valid_commit_skips_and_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            sample_dir = output_root / "samples" / "sample-0001"
            sample_dir.mkdir(parents=True)
            original_source = output_root / "original.MP4"
            original_source.write_bytes(b"source-video")
            files = {
                "source_video.MP4": original_source.read_bytes(),
                "edit_instruction.txt": (
                    "让左边的人挥手，并让右边的人也挥手。".encode("utf-8")
                ),
                "preview.mp4": b"preview",
                "conditioning_anchor_original.png": b"anchor",
                "conditioning_frame0_float32.npy": b"npy",
                "conditioning_frame0.png": b"png",
            }
            for name, payload in files.items():
                (sample_dir / name).write_bytes(payload)
            row = _manifest_row(authorized=True)
            row["edit_instruction"] = "让左边的人挥手，并让右边的人也挥手。"
            row.update(
                {
                    "_iid": "sample-0001",
                    "_row_digest": "3" * 64,
                    "_authorization_mode": "bound_human_approval",
                    "anchor_sha256": _sha(b"anchor"),
                    "source_video_sha256": _sha(original_source.read_bytes()),
                    "_input_media": {
                        "source_video_path": str(original_source),
                        "source_video_ffprobe": _video_probe(),
                    },
                }
            )
            contract_temporal = validate_batch_temporal_grid(
                [row],
                expected_frame_num=81,
            )
            contract = {
                "contract_digest": "4" * 64,
                "manifest": {"sha256": "5" * 64},
                "generation_parameters": {"base_seed": 260730},
                "temporal_policy": contract_temporal,
            }
            outputs = {
                "source_video": "source_video.MP4",
                "source_video_sha256": row["source_video_sha256"],
                "source_video_bytes": len(files["source_video.MP4"]),
                "edit_instruction_file": "edit_instruction.txt",
                "edit_instruction_file_sha256": _sha(
                    files["edit_instruction.txt"]
                ),
                "edit_instruction_file_bytes": len(
                    files["edit_instruction.txt"]
                ),
                "preview_mp4": "preview.mp4",
                "preview_mp4_sha256": _sha(b"preview"),
                "preview_mp4_ffprobe": _video_probe(),
                "conditioning_anchor_original": (
                    "conditioning_anchor_original.png"
                ),
                "conditioning_anchor_original_sha256": _sha(b"anchor"),
                "conditioning_frame0_float32": (
                    "conditioning_frame0_float32.npy"
                ),
                "conditioning_frame0_float32_sha256": _sha(b"npy"),
                "conditioning_frame0_png": "conditioning_frame0.png",
                "conditioning_frame0_png_sha256": _sha(b"png"),
            }
            result: dict[str, object] = {
                "schema_version": SAMPLE_SCHEMA,
                "iid": "sample-0001",
                "sample_index": 0,
                "manifest_sha256": "5" * 64,
                "manifest_row_digest": "3" * 64,
                "contract_digest": "4" * 64,
                "seed": sample_seed(260730, "sample-0001"),
                "authorization_mode": "bound_human_approval",
                "manifest_role": _BATCH.APPROVED_MANIFEST_ROLE,
                "production_eligible": True,
                "approval": row["approval"],
                "human_review_status_at_generation": "approved",
                "generation_authorized_in_manifest": True,
                "inputs": {
                    "source_video_resolved_path": str(original_source),
                    "source_video_committed_path": str(
                        sample_dir / "source_video.MP4"
                    ),
                    "source_video_sha256": row["source_video_sha256"],
                },
                "outputs": outputs,
                "first_frame_policy": {
                    "policy_version": FIRST_FRAME_POLICY,
                    "preencode_frame0_matches_png_pixels": True,
                    "mp4_decode_pixel_equality_claimed": False,
                },
                "temporal_policy": validate_temporal_pair(
                    source_probe=_video_probe(),
                    target_probe=_video_probe(),
                ),
            }
            result["result_digest"] = _object_digest(result)
            (sample_dir / "result.json").write_text(
                json.dumps(result, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validated = validate_sample_commit(
                sample_dir,
                row=row,
                contract=contract,
                sample_index=0,
            )
            self.assertEqual(validated["iid"], "sample-0001")
            generated = _BATCH._generated_manifest_rows(
                output_root=output_root,
                rows=[row],
                results={0: validated},
            )[0]
            self.assertEqual(
                generated["source_video"],
                str((sample_dir / "source_video.MP4").resolve()),
            )
            self.assertNotEqual(generated["source_video"], str(original_source))
            self.assertEqual(
                generated["edit_instruction_file"],
                str((sample_dir / "edit_instruction.txt").resolve()),
            )
            self.assertEqual(
                Path(generated["edit_instruction_file"]).read_bytes(),
                row["edit_instruction"].encode("utf-8"),
            )

            instruction_path = sample_dir / "edit_instruction.txt"
            instruction_path.write_bytes(files["edit_instruction.txt"] + b"\n")
            with self.assertRaisesRegex(
                Wan22BatchError, "(file hash mismatch|edit instruction content)"
            ):
                validate_sample_commit(
                    sample_dir,
                    row=row,
                    contract=contract,
                    sample_index=0,
                )
            instruction_path.write_bytes(files["edit_instruction.txt"])

            source_copy = sample_dir / "source_video.MP4"
            source_copy.write_bytes(b"mutated-source")
            with self.assertRaisesRegex(Wan22BatchError, "file hash mismatch"):
                validate_sample_commit(
                    sample_dir,
                    row=row,
                    contract=contract,
                    sample_index=0,
                )
            source_copy.write_bytes(files["source_video.MP4"])

            (sample_dir / "preview.mp4").write_bytes(b"mutated")
            with self.assertRaisesRegex(Wan22BatchError, "file hash mismatch"):
                validate_sample_commit(
                    sample_dir,
                    row=row,
                    contract=contract,
                    sample_index=0,
                )


class Wan22OrchestrationTests(unittest.TestCase):
    def test_module_has_no_top_level_torch_or_wan_import(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported)
        self.assertNotIn("wan", imported)

    def test_slurm_resources_and_quality_modes(self) -> None:
        smoke = SMOKE.read_text(encoding="utf-8")
        full = FULL.read_text(encoding="utf-8")
        for text in (smoke, full):
            self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
            self.assertIn("#SBATCH --ntasks=1", text)
            self.assertIn("--nproc_per_node=8", text)
            self.assertNotIn("--gpus-per-task", text)
            self.assertNotIn("\nsrun ", text)
            self.assertIn("--require-rocm", text)
            self.assertIn("--expected-gpu-name-substring MI210", text)
            self.assertIn("MOTIVE_WAN22_ALLOW_PENDING_REVIEW:?", text)
            self.assertIn("MOTIVE_WAN22_FFPROBE_BIN:?", text)
            self.assertIn('--ffprobe "${ffprobe_bin}"', text)
            self.assertIn(
                "only the signed exact-eight-row release authorizes generation",
                text,
            )
            self.assertIn("MOTIVE_WAN22_SIGNED_RELEASE:?", text)
            self.assertIn('--signed-release "${signed_release}"', text)
            self.assertNotIn("runner_args+=(--allow-pending-review)", text)
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", text)
            self.assertIn("HF_HUB_OFFLINE=1", text)
            self.assertIn("TRANSFORMERS_OFFLINE=1", text)
            self.assertIn("HF_DATASETS_OFFLINE=1", text)
            self.assertIn("PYTORCH_KERNEL_CACHE_PATH", text)
            self.assertIn("MIOPEN_CUSTOM_CACHE_DIR", text)
        self.assertIn('MOTIVE_WAN22_FRAME_NUM:-81', smoke)
        self.assertIn('MOTIVE_WAN22_SAMPLE_STEPS:-4', smoke)
        self.assertNotIn("--max-samples", smoke)
        self.assertIn('MOTIVE_WAN22_FRAME_NUM:-81', full)
        self.assertIn('MOTIVE_WAN22_SAMPLE_STEPS:-40', full)
        self.assertIn('MOTIVE_WAN22_MAX_NEW_SAMPLES:-16', full)

    def test_submit_chain_is_curation_smoke_then_configured_full_chunks(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8")
        self.assertIn('MOTIVE_WAN22_CHUNK_COUNT:-8', text)
        self.assertIn('MOTIVE_WAN22_CHUNK_SIZE:-16', text)
        self.assertIn("MOTIVE_WAN22_FFPROBE_BIN:?", text)
        self.assertIn("full chunk capacity is too small", text)
        self.assertIn("MOTIVE_WAN22_CURATION_DEPENDENCY_MODE", text)
        self.assertIn('curation_dependency_mode}" == "published"', text)
        self.assertIn(
            "published generation manifest SHA-256 differs",
            text,
        )
        self.assertNotIn(
            "the frozen 128-sample chain requires exactly 8 chunks of 16",
            text,
        )
        self.assertIn('smoke_dependency="afterok:${curation_job_id}"', text)
        self.assertIn('--dependency="${smoke_dependency}"', text)
        self.assertIn(
            '--dependency="afterok:${smoke_job}"',
            text,
        )
        self.assertIn('previous_job="${full_geometry_smoke_job}"', text)
        self.assertIn('--dependency="afterok:${previous_job}"', text)
        self.assertIn('MOTIVE_WAN22_MAX_NEW_SAMPLES="${chunk_size}"', text)
        self.assertIn(
            'full_geometry_smoke_output="${run_root}/full_geometry_smoke"',
            text,
        )
        self.assertIn("export MOTIVE_WAN22_FRAME_NUM=81", text)
        self.assertNotIn("export MOTIVE_WAN22_FRAME_NUM=17", text)
        self.assertIn("export MOTIVE_WAN22_SAMPLE_STEPS=1", text)
        self.assertIn("export MOTIVE_WAN22_SAMPLE_STEPS=40", text)
        self.assertIn("export MOTIVE_WAN22_SIZE='1280*720'", text)
        self.assertIn(
            "only the signed exact-eight-row release authorizes generation",
            text,
        )
        self.assertIn("MOTIVE_WAN22_SIGNED_RELEASE:?", text)

    def test_runner_separates_model_and_container_frame_rates(self) -> None:
        text = MODULE.read_text(encoding="utf-8")
        self.assertIn('"model_sample_fps": MODEL_SAMPLE_FPS', text)
        self.assertIn('"output_container_frame_rate": temporal_policy[', text)
        self.assertIn("fps=source_frame_rate", text)
        self.assertNotIn('"fps": DEFAULT_FPS', text)
        self.assertIn("validate_batch_temporal_grid(", text)
        self.assertIn("validate_temporal_pair(", text)

    def test_production_launchers_forbid_pending_review_override(self) -> None:
        for path in (
            SMOKE,
            FULL,
            SUBMIT,
            PARALLEL_SUBMIT,
            PARALLEL_RESUME,
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                "MOTIVE_WAN22_ALLOW_PENDING_REVIEW must be 0",
                text,
                msg=str(path),
            )
            self.assertNotIn("--allow-pending-review", text, msg=str(path))

    def test_runtime_launches_flash_attention_kernel_on_every_rank(self) -> None:
        text = MODULE.read_text(encoding="utf-8")
        self.assertIn("def _flash_attention_kernel_smoke(", text)
        self.assertIn("attention_module.flash_attention(", text)
        self.assertIn("torch_module.cuda.synchronize(local_rank)", text)
        self.assertIn(
            'stage="per-rank FlashAttention kernel smoke"',
            text,
        )
        self.assertIn("gather_values=True", text)
        self.assertNotIn('"output_checksum"', text)
        self.assertIn('stage=f"sample commit iid={row[', text)

    def test_shell_scripts_parse(self) -> None:
        for path in (
            SMOKE,
            FULL,
            SUBMIT,
            PARALLEL_SUBMIT,
            PARALLEL_RESUME,
        ):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{path}: {completed.stderr}",
            )

    def test_private_ffprobe_compat_is_executable_and_runtime_bound(self) -> None:
        self.assertTrue(os.access(FFPROBE_COMPAT, os.X_OK))
        text = FFPROBE_COMPAT.read_text(encoding="utf-8")
        self.assertIn("MOTIVE_WAN22_PYTHON_BIN", text)
        self.assertIn("container.decode(video=stream.index)", text)
        with tempfile.TemporaryDirectory() as pycache:
            environment = dict(os.environ)
            environment["PYTHONPYCACHEPREFIX"] = pycache
            completed = subprocess.run(
                [sys.executable, "-m", "py_compile", str(FFPROBE_COMPAT)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_pins_are_visible_in_runtime_and_scripts(self) -> None:
        module_text = MODULE.read_text(encoding="utf-8")
        for value in (OFFICIAL_COMMIT, MODEL_HF_REVISION):
            self.assertIn(value, module_text)
            self.assertIn(value, SMOKE.read_text(encoding="utf-8"))
            self.assertIn(value, FULL.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
