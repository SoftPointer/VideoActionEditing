from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_postvideo_quality_audit.sbatch"
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_postvideo_quality_routing as builder  # noqa: E402
import motion_residual as motion  # noqa: E402


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(builder.canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _quality(
    *,
    identity: str = "yes",
    species: str = "yes",
    clothing: str = "yes",
    non_edit: str = "yes",
    camera: str = "yes",
    blur: str = "none",
    flicker: str = "none",
    artifact: str = "none",
    action: str = "yes",
    confidence: str = "high",
) -> dict[str, object]:
    return {
        "schema_version": builder.QA_SCHEMA,
        "action_implemented": action,
        "identity_preserved": identity,
        "species_preserved": species,
        "clothing_preserved": clothing,
        "non_edited_content_preserved": non_edit,
        "camera_preserved": camera,
        "blur_level": blur,
        "flicker_level": flicker,
        "artifact_level": artifact,
        "confidence": confidence,
        "evidence": {
            "action": [
                {
                    "frames": ["T0", "T3"],
                    "observation": "the target arm moves through two ordered positions",
                }
            ],
            "identity": [
                {
                    "frames": ["S0", "T0", "T3"],
                    "observation": "literal face, species, and clothing comparison",
                }
            ],
            "preservation": [
                {
                    "frames": ["S0", "S3", "T0", "T3"],
                    "observation": "literal background and camera comparison",
                }
            ],
            "technical": [
                {
                    "frames": ["T0", "T3"],
                    "observation": "literal target sharpness and temporal stability",
                }
            ],
        },
        "uncertainty_codes": [],
    }


class FakeBackend:
    responses: dict[str, object] = {}
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object):
        self.model_revision = "fixture-revision"
        self.transformers_version = "fixture-transformers"
        self.kwargs = kwargs
        self.model = FakePlacementModel()
        self.processor = SimpleNamespace()

    def generate_postvideo_quality(self, **kwargs: object) -> tuple[str, str]:
        type(self).calls.append(dict(kwargs))
        iid = str(kwargs["iid"])
        value = type(self).responses[iid]
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, str):
            raw = value
        else:
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
        visual = hashlib.sha256(f"visual:{iid}".encode()).hexdigest()
        return raw, visual

    def generate_visual_observation(self, **kwargs: object) -> tuple[str, str]:
        source = Path(str(kwargs["source_path"]))
        suffix = ".source.mp4"
        iid = source.name[: -len(suffix)] if source.name.endswith(suffix) else source.stem
        value = type(self).responses[iid]
        if isinstance(value, BaseException):
            raise value
        raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        return raw, hashlib.sha256(f"visual:{iid}".encode()).hexdigest()


class FakeTensor:
    def __init__(self, name: str, *, device: str = "cuda:0", elements: int = 1):
        self.name = name
        self.device = device
        self.elements = elements

    def numel(self) -> int:
        return self.elements


class FakePlacementModel:
    def __init__(self, *, parameter_device: str = "cuda:0"):
        self.device = "cuda:0"
        self.hf_device_map = {"": 0}
        self.parameter_device = parameter_device

    def named_parameters(self):
        return iter((("weight", FakeTensor("weight", device=self.parameter_device, elements=7)),))

    def named_buffers(self):
        return iter((("position", FakeTensor("position", elements=3)),))


class FakeCuda:
    def __init__(self, *, devices: int = 1):
        self.devices = devices

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self.devices

    def current_device(self) -> int:
        return 0

    def get_device_name(self, index: int) -> str:
        if index != 0:
            raise AssertionError(index)
        return "fixture MI210"


class FakeTorch:
    def __init__(self, *, devices: int = 1):
        self.cuda = FakeCuda(devices=devices)


class FakePlacementBackend:
    def __init__(self, *, parameter_device: str = "cuda:0"):
        self.model = FakePlacementModel(parameter_device=parameter_device)


class PostVideoQualityRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / ("a" * 40)
        self.model.mkdir()
        (self.model / "config.json").write_text(
            '{"model_type":"qwen2_5_vl"}\n', encoding="utf-8"
        )
        self.manifest = self.root / "renderer.jsonl"
        self.input_receipt = self.root / "raw.receipt.json"
        self.output = self.root / "audit"
        FakeBackend.responses = {}
        FakeBackend.calls = []

    def _write_input_receipt(self, rows: list[dict[str, object]]) -> None:
        receipt: dict[str, object] = {
            "complete": True,
            "parquet_path": str(self.manifest.resolve()),
            "parquet_sha256": builder.file_sha256(self.manifest),
            "sample_count": len(rows),
            "sample_ids": [row["iid"] for row in rows],
            "renderer_row_digests_sha256": builder.object_sha256(
                [row["renderer_row_digest"] for row in rows]
            ),
            "non_strict_selection_rows": sum(
                row["strict_selection_gates_all_true"] is False for row in rows
            ),
            "preview_only": True,
            "production_eligible": False,
            "production_claim_forbidden": True,
        }
        receipt["receipt_digest"] = builder.object_sha256(receipt)
        self.input_receipt.write_bytes(builder._pretty_bytes(receipt))

    def _write_manifest(self, rows: list[dict[str, object]]) -> str:
        _jsonl(self.manifest, rows)
        self._write_input_receipt(rows)
        return builder.file_sha256(self.manifest)

    def _write_parquet_manifest(self, rows: list[dict[str, object]]) -> str:
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = []
        for name, physical_type, nullable in builder._EXPECTED_ARROW_FIELDS:
            if physical_type == "string":
                data_type = pa.string()
            elif physical_type == "bool":
                data_type = pa.bool_()
            elif name == "videos":
                data_type = pa.list_(
                    pa.field(
                        "element",
                        pa.struct(
                            [pa.field("video_path", pa.string(), nullable=False)]
                        ),
                    )
                )
            else:  # pragma: no cover - the frozen schema is intentionally closed
                raise AssertionError((name, physical_type))
            fields.append(pa.field(name, data_type, nullable=nullable))
        table = pa.Table.from_pylist(rows, schema=pa.schema(fields))
        pq.write_table(table, self.manifest)
        observed = tuple(
            (field.name, str(field.type), field.nullable) for field in table.schema
        )
        self.assertEqual(observed, builder._EXPECTED_ARROW_FIELDS)
        self._write_input_receipt(rows)
        return builder.file_sha256(self.manifest)

    def _audit_kwargs(self) -> dict[str, object]:
        return {
            "input_manifest": self.manifest,
            "expected_input_sha256": builder.file_sha256(self.manifest),
            "input_receipt": self.input_receipt,
            "expected_input_receipt_sha256": builder.file_sha256(
                self.input_receipt
            ),
            "model_path": self.model,
            "output_dir": self.output,
            "method_source_revision": "a" * 40,
            "method_source_archive_sha256": "b" * 64,
            "nframes": 4,
        }

    def _run_audit(
        self,
        *,
        expected_strict_rows: int,
        production: bool,
        production_parquet: bool = False,
    ) -> dict[str, object]:
        kwargs = {**self._audit_kwargs(), "expected_strict_rows": expected_strict_rows}
        if not production:
            return builder.run_audit(**kwargs, backend_factory=FakeBackend)
        runtime = SimpleNamespace(
            LocalQwenBackend=FakeBackend,
            VISUAL_SYSTEM="fixture",
            OBSERVATION_PROMPT="fixture",
        )
        execution = builder._production_backend_execution(
            FakePlacementBackend(), torch_module=FakeTorch()
        )
        bound_loader = builder._load_bound_input

        def fixture_bound_loader(*args: object, **loader_kwargs: object):
            loader_kwargs["require_parquet"] = False
            return bound_loader(*args, **loader_kwargs)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(builder, "_load_qwen_filter", return_value=runtime)
            )
            stack.enter_context(
                patch.object(
                    builder,
                    "_production_backend_execution",
                    return_value=execution,
                )
            )
            if not production_parquet:
                stack.enter_context(
                    patch.object(
                        builder,
                        "_load_bound_input",
                        side_effect=fixture_bound_loader,
                    )
                )
            return builder.run_audit(**kwargs)

    def _build_fixture_routing(
        self,
        audit: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        with patch.object(
            builder, "validate_published_audit", return_value=audit
        ):
            return builder.build_routing(**kwargs)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _renderer_row(self, iid: str, *, strict: bool = True) -> dict[str, object]:
        source = self.root / f"{iid}.source.mp4"
        target = self.root / f"{iid}.target.mp4"
        source.write_bytes(f"source-video:{iid}\n".encode())
        target.write_bytes(f"target-video:{iid}\n".encode())
        instruction = f"Make the subject wave in sample {iid}."
        messages = [
            {"type": "video", "has_loss": 0},
            {"type": "text", "text": instruction, "has_loss": 0},
            {"type": "video_gen", "has_loss": 1},
        ]
        row: dict[str, object] = {
            "schema_version": builder.RAW_ROW_SCHEMA,
            "inputs": builder.canonical_json_bytes(messages).decode("utf-8"),
            "videos": [
                {"video_path": str(source)},
                {"video_path": str(target)},
            ],
            "iid": iid,
            "edit_instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "source_video_path": str(source),
            "source_video_sha256": builder.file_sha256(source),
            "target_video_path": str(target),
            "target_video_sha256": builder.file_sha256(target),
            "strict_selection_gates_all_true": strict,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
        }
        row["renderer_row_digest"] = builder.object_sha256(row)
        return row

    def _release_renderer_row(
        self, iid: str, *, strict: bool = True
    ) -> dict[str, object]:
        fixture = self._renderer_row(iid, strict=strict)
        fixture.pop("renderer_row_digest")
        source_path = str(fixture["source_video_path"])
        source_sha256 = str(fixture["source_video_sha256"])
        target_path = str(fixture["target_video_path"])
        target_sha256 = str(fixture["target_video_sha256"])
        row: dict[str, object] = {
            "schema_version": builder.RAW_ROW_SCHEMA,
            "inputs": fixture["inputs"],
            "videos": fixture["videos"],
            "iid": iid,
            "group_id": f"group-{iid}",
            "family": "fixture-human",
            "edit_instruction_sha256": fixture["edit_instruction_sha256"],
            "source_video_path": source_path,
            "source_video_declared_path": source_path,
            "source_video_sha256": source_sha256,
            "target_video_path": target_path,
            "target_video_declared_path": target_path,
            "target_video_sha256": target_sha256,
            "shared_i0_path": source_path,
            "shared_i0_sha256": source_sha256,
            "preview_manifest_path": str(self.manifest.resolve()),
            "preview_manifest_sha256": "1" * 64,
            "preview_row_digest": "2" * 64,
            "preview_row_file_sha256": "3" * 64,
            "experimental_inclusion_policy": "fixture-strict-preview-v1",
            "selection_gates_json": '{"fixture_gate":true}',
            "strict_selection_gates_all_true": strict,
            "upstream_authorization_json": '{"training_authorized":false}',
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_eligible": False,
            "post_video_acceptance": "not_run",
            "experimental_training_acknowledged": False,
            "production_claim_forbidden": True,
        }
        self.assertEqual(
            tuple(row),
            tuple(name for name, _physical, _nullable in builder._EXPECTED_ARROW_FIELDS)
            [:-1],
        )
        row["renderer_row_digest"] = builder.object_sha256(row)
        return row

    def _source_archive(
        self,
        name: str,
        *,
        unsafe_member: tuple[str, bytes] | None = None,
    ) -> tuple[Path, str]:
        archive_path = self.root / name
        tool_payload = b"# fixture post-video tool\n"
        test_payload = b"# fixture post-video tests\n"
        members = (
            (
                "methods/bernini_action_editing/tools/"
                "build_postvideo_quality_routing.py",
                tool_payload,
            ),
            (
                "methods/bernini_action_editing/tests/"
                "test_build_postvideo_quality_routing.py",
                test_payload,
            ),
        )
        with tarfile.open(archive_path, mode="w") as archive:
            for member_name, payload in members:
                info = tarfile.TarInfo(member_name)
                info.mode = 0o600
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            if unsafe_member is not None:
                member_type, payload = unsafe_member
                if member_type == "absolute":
                    info = tarfile.TarInfo("/absolute_escape.py")
                elif member_type == "parent":
                    info = tarfile.TarInfo("../parent_escape.py")
                elif member_type == "symlink":
                    info = tarfile.TarInfo("methods/unsafe_symlink.py")
                    info.type = tarfile.SYMTYPE
                    info.linkname = "/tmp/escape.py"
                elif member_type == "hardlink":
                    info = tarfile.TarInfo("methods/unsafe_hardlink.py")
                    info.type = tarfile.LNKTYPE
                    info.linkname = members[0][0]
                elif member_type == "fifo":
                    info = tarfile.TarInfo("methods/unsafe_fifo")
                    info.type = tarfile.FIFOTYPE
                else:  # pragma: no cover - helper contract
                    raise AssertionError(member_type)
                if info.isreg():
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    archive.addfile(info)
        return archive_path, hashlib.sha256(tool_payload).hexdigest()

    def _run_archive_preflight(
        self, archive_path: Path, tool_sha256: str
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("BERNINI_POSTVIDEO_")
        }
        environment.pop("SLURM_JOB_ID", None)
        environment.update(
            {
                "BERNINI_POSTVIDEO_ARCHIVE_PREFLIGHT_ONLY": "1",
                "BERNINI_POSTVIDEO_SOURCE_ARCHIVE": str(archive_path.resolve()),
                "BERNINI_POSTVIDEO_SOURCE_ARCHIVE_SHA256": builder.file_sha256(
                    archive_path
                ),
                "BERNINI_POSTVIDEO_TOOL_SHA256": tool_sha256,
                "BERNINI_POSTVIDEO_PYTHON": sys.executable,
                "SLURM_TMPDIR": str(self.root.resolve()),
            }
        )
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def _prepare_three(self) -> tuple[list[dict[str, object]], str]:
        rows = [
            self._renderer_row("iid0001"),
            self._renderer_row("iid0002"),
            self._renderer_row("iid0003"),
            self._renderer_row("ignored_non_strict", strict=False),
        ]
        digest = self._write_manifest(rows)
        FakeBackend.responses = {
            "iid0001": _quality(),
            "iid0002": _quality(
                identity="no",
                clothing="no",
                non_edit="no",
                blur="medium",
                artifact="medium",
            ),
            "iid0003": "{not valid json",
        }
        return rows, digest

    def _run_three(self, *, production: bool = False) -> dict[str, object]:
        _rows, digest = self._prepare_three()
        self.assertEqual(digest, builder.file_sha256(self.manifest))
        return self._run_audit(expected_strict_rows=3, production=production)

    def _policy(self) -> dict[str, object]:
        return {
            "schema_version": builder.ROUTING_POLICY_SCHEMA,
            "policy_name": "fixture-conservative-v1",
            "unreviewed_default": "reject",
            "full_target_weight": 0.75,
            "full_pair": {
                "action_implemented": ["yes"],
                "identity_preserved": ["yes"],
                "species_preserved": ["yes", "not_applicable"],
                "clothing_preserved": ["yes", "not_applicable"],
                "non_edited_content_preserved": ["yes"],
                "camera_preserved": ["yes"],
                "max_blur": "low",
                "max_flicker": "low",
                "max_artifact": "low",
                "min_confidence": "medium",
            },
            "motion_only": {
                "action_implemented": ["yes"],
                "identity_preserved": ["yes", "no"],
                "species_preserved": ["yes", "not_applicable"],
                "clothing_preserved": ["yes", "no", "not_applicable"],
                "non_edited_content_preserved": ["yes", "no"],
                "camera_preserved": ["yes"],
                "max_blur": "medium",
                "max_flicker": "medium",
                "max_artifact": "medium",
                "min_confidence": "medium",
            },
        }

    def test_audit_is_three_file_audit_only_and_retains_all_required_evidence(self) -> None:
        result = self._run_three()
        self.assertEqual(result["status"], "VALID")
        self.assertTrue(result["audit_only"])
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["outcome_counts"], {"schema_error": 1, "success": 2})
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {builder.RECORDS_NAME, builder.SUMMARY_NAME, builder.DONE_NAME},
        )
        self.assertFalse(any("rout" in path.name for path in self.output.iterdir()))
        self.assertEqual(len(FakeBackend.calls), 3)
        self.assertTrue(all(call["nframes"] == 4 for call in FakeBackend.calls))

        records = result["records"]
        first = records[0]
        self.assertEqual(first["quality"]["action_implemented"], "yes")
        self.assertEqual(first["quality"]["identity_preserved"], "yes")
        self.assertEqual(first["quality"]["species_preserved"], "yes")
        self.assertEqual(first["quality"]["clothing_preserved"], "yes")
        self.assertEqual(first["quality"]["camera_preserved"], "yes")
        self.assertEqual(first["quality"]["blur_level"], "none")
        self.assertEqual(first["quality"]["flicker_level"], "none")
        self.assertEqual(first["quality"]["artifact_level"], "none")
        self.assertTrue(first["quality"]["evidence"]["action"])
        self.assertRegex(first["input"]["source_video"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(first["input"]["target_video"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(first["model_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(first["inference_input_forbidden"])
        self.assertTrue(first["target_video_as_inference_condition_forbidden"])

        failed = records[2]
        self.assertEqual(failed["audit_outcome"], "schema_error")
        self.assertEqual(failed["quality"]["action_implemented"], "unclear")
        self.assertEqual(failed["raw_response"], "{not valid json")

    def test_external_policy_builds_full_motion_reject_and_compatible_routes(self) -> None:
        audit = self._run_three(production=True)
        policy_path = self.root / "policy.json"
        policy_path.write_bytes(builder._pretty_bytes(self._policy()))
        output = self.root / "routing" / "postvideo.jsonl"
        receipt = self._build_fixture_routing(
            audit,
            audit_dir=self.output,
            expected_audit_done_sha256=audit["done_sha256"],
            policy_json=policy_path,
            expected_policy_sha256=builder.file_sha256(policy_path),
            output_jsonl=output,
        )
        routes = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            [(row["iid"], row["tier"]) for row in routes],
            [
                ("ignored_non_strict", "reject"),
                ("iid0001", "full_pair"),
                ("iid0002", "motion_only"),
                ("iid0003", "reject"),
            ],
        )
        self.assertEqual(routes[0]["full_target_weight"], 0.0)
        self.assertEqual(routes[1]["full_target_weight"], 0.75)
        self.assertEqual(routes[2]["full_target_weight"], 0.0)
        self.assertEqual(routes[3]["full_target_weight"], 0.0)
        self.assertEqual(
            receipt["route_counts"],
            {"full_pair": 1, "motion_only": 1, "reject": 2},
        )
        self.assertEqual(receipt["unreviewed_default"], "reject")
        self.assertTrue(receipt["route_thresholds_external_to_audit"])
        self.assertTrue(receipt["inference_input_forbidden"])
        self.assertTrue(Path(f"{output}.receipt.json").is_file())
        self.assertTrue(Path(f"{output}.sha256").is_file())
        router = motion.ReviewRouter.load(output, default_tier="reject")
        self.assertEqual(
            router.receipt()["explicit_route_counts"],
            {"full_pair": 1, "motion_only": 1, "reject": 2},
        )
        self.assertEqual(router.route("not-audited").tier, "reject")

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow is required for production parquet")
    def test_exact_arrow_production_audit_routes_without_validator_patch(self) -> None:
        self.manifest = self.root / "renderer.parquet"
        self.input_receipt = self.root / "renderer.receipt.json"
        rows = [
            self._release_renderer_row("iid0001"),
            self._release_renderer_row("iid0002"),
            self._release_renderer_row("iid0003"),
            self._release_renderer_row("ignored_non_strict", strict=False),
        ]
        self._write_parquet_manifest(rows)
        FakeBackend.responses = {
            "iid0001": _quality(),
            "iid0002": _quality(
                identity="no",
                clothing="no",
                non_edit="no",
                blur="medium",
                artifact="medium",
            ),
            "iid0003": "{not valid json",
        }
        audit = self._run_audit(
            expected_strict_rows=3,
            production=True,
            production_parquet=True,
        )
        self.assertTrue(audit["production_backend"])

        policy_path = self.root / "production-policy.json"
        policy_path.write_bytes(builder._pretty_bytes(self._policy()))
        output = self.root / "production-route.jsonl"
        receipt = builder.build_routing(
            audit_dir=self.output,
            expected_audit_done_sha256=audit["done_sha256"],
            policy_json=policy_path,
            expected_policy_sha256=builder.file_sha256(policy_path),
            output_jsonl=output,
        )
        routes = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(receipt["route_count"], 4)
        self.assertEqual(
            [(row["iid"], row["tier"]) for row in routes],
            [
                ("ignored_non_strict", "reject"),
                ("iid0001", "full_pair"),
                ("iid0002", "motion_only"),
                ("iid0003", "reject"),
            ],
        )

    def test_generation_failure_is_published_as_unclear_and_routes_reject(self) -> None:
        row = self._renderer_row("iid0001")
        self._write_manifest([row])
        FakeBackend.responses = {"iid0001": RuntimeError("fixture decoder failure")}
        audit = self._run_audit(expected_strict_rows=1, production=True)
        record = audit["records"][0]
        self.assertEqual(record["audit_outcome"], "generation_error")
        self.assertEqual(record["quality"]["artifact_level"], "unclear")
        self.assertEqual(record["visual_input_digest"], "")

        policy_path = self.root / "policy.json"
        policy_path.write_bytes(builder._pretty_bytes(self._policy()))
        output = self.root / "route.jsonl"
        self._build_fixture_routing(
            audit,
            audit_dir=self.output,
            expected_audit_done_sha256=audit["done_sha256"],
            policy_json=policy_path,
            expected_policy_sha256=builder.file_sha256(policy_path),
            output_jsonl=output,
        )
        route = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(route["tier"], "reject")
        self.assertIn("fail_closed_generation_error", route["review"])

    def test_frame_out_of_range_is_schema_error_not_positive_evidence(self) -> None:
        row = self._renderer_row("iid0001")
        self._write_manifest([row])
        bad = _quality()
        bad["evidence"]["action"][0]["frames"] = ["T0", "T4"]
        FakeBackend.responses = {"iid0001": bad}
        audit = self._run_audit(expected_strict_rows=1, production=False)
        self.assertEqual(audit["records"][0]["audit_outcome"], "schema_error")
        self.assertEqual(audit["records"][0]["quality"]["action_implemented"], "unclear")

    def test_caller_pinned_input_hash_and_exact_strict_count_are_required(self) -> None:
        self._prepare_three()
        with self.assertRaisesRegex(builder.PostVideoQualityError, "caller-pinned hash"):
            kwargs = {
                **self._audit_kwargs(),
                "expected_input_sha256": "0" * 64,
                "expected_strict_rows": 3,
            }
            builder.run_audit(**kwargs, backend_factory=FakeBackend)
        with self.assertRaisesRegex(builder.PostVideoQualityError, "expected 4"):
            builder.run_audit(
                **{**self._audit_kwargs(), "expected_strict_rows": 4},
                backend_factory=FakeBackend,
            )
        self.assertEqual(FakeBackend.calls, [])

    def test_media_change_invalidates_audit_receipt(self) -> None:
        result = self._run_three()
        source = Path(result["records"][0]["input"]["source_video"]["path"])
        source.write_bytes(source.read_bytes() + b"tamper")
        with self.assertRaisesRegex(builder.PostVideoQualityError, "hash differs"):
            builder.validate_published_audit(
                self.output,
                expected_done_sha256=result["done_sha256"],
            )

    def test_policy_is_pinned_and_unclear_can_never_be_allowed(self) -> None:
        audit = self._run_three(production=True)
        policy = self._policy()
        policy["full_pair"]["action_implemented"] = ["yes", "unclear"]
        policy_path = self.root / "policy.json"
        policy_path.write_bytes(builder._pretty_bytes(policy))
        with self.assertRaisesRegex(builder.PostVideoQualityError, "action_implemented"):
            self._build_fixture_routing(
                audit,
                audit_dir=self.output,
                expected_audit_done_sha256=audit["done_sha256"],
                policy_json=policy_path,
                expected_policy_sha256=builder.file_sha256(policy_path),
                output_jsonl=self.root / "route.jsonl",
            )
        with self.assertRaisesRegex(builder.PostVideoQualityError, "caller-pinned hash"):
            self._build_fixture_routing(
                audit,
                audit_dir=self.output,
                expected_audit_done_sha256=audit["done_sha256"],
                policy_json=policy_path,
                expected_policy_sha256="0" * 64,
                output_jsonl=self.root / "route2.jsonl",
            )

    def test_injected_test_backend_audit_can_never_authorize_routing(self) -> None:
        audit = self._run_three(production=False)
        policy_path = self.root / "policy.json"
        policy_path.write_bytes(builder._pretty_bytes(self._policy()))
        with self.assertRaisesRegex(
            builder.PostVideoQualityError, "production_local_qwen"
        ):
            builder.build_routing(
                audit_dir=self.output,
                expected_audit_done_sha256=audit["done_sha256"],
                policy_json=policy_path,
                expected_policy_sha256=builder.file_sha256(policy_path),
                output_jsonl=self.root / "forbidden.jsonl",
            )

    def test_positive_labels_require_temporal_and_cross_video_evidence(self) -> None:
        action = _quality()
        action["evidence"]["action"][0]["frames"] = ["T0"]
        with self.assertRaisesRegex(
            builder.PostVideoQualityError, "two distinct TARGET"
        ):
            builder.validate_quality_observation(action, nframes=4)

        identity = _quality()
        identity["evidence"]["identity"][0]["frames"] = ["T0", "T3"]
        with self.assertRaisesRegex(
            builder.PostVideoQualityError, "both SOURCE and TARGET"
        ):
            builder.validate_quality_observation(identity, nframes=4)

        preservation = _quality()
        preservation["evidence"]["preservation"][0]["frames"] = ["S0", "S3"]
        with self.assertRaisesRegex(
            builder.PostVideoQualityError, "both SOURCE and TARGET"
        ):
            builder.validate_quality_observation(preservation, nframes=4)

        aliased = _quality()
        aliased["evidence"]["action"][0]["frames"] = ["T0", "T00"]
        aliased["evidence"]["technical"][0]["frames"] = ["T0", "T00"]
        with self.assertRaisesRegex(
            builder.PostVideoQualityError, "frame label"
        ):
            builder.validate_quality_observation(aliased, nframes=4)

    def test_policy_cannot_override_core_positive_training_invariants(self) -> None:
        for tier in ("full_pair", "motion_only"):
            for allowed in (["no"], ["yes", "no"]):
                with self.subTest(tier=tier, invariant="action", allowed=allowed):
                    policy = self._policy()
                    policy[tier]["action_implemented"] = allowed
                    with self.assertRaisesRegex(
                        builder.PostVideoQualityError,
                        rf"{tier}\.action_implemented must require exactly yes",
                    ):
                        builder.validate_routing_policy(policy)

        for field in (
            "identity_preserved",
            "non_edited_content_preserved",
            "camera_preserved",
        ):
            with self.subTest(field=field):
                policy = self._policy()
                policy["full_pair"][field] = ["no"]
                with self.assertRaisesRegex(
                    builder.PostVideoQualityError,
                    rf"full_pair\.{field} must require exactly yes",
                ):
                    builder.validate_routing_policy(policy)

        for field in ("species_preserved", "clothing_preserved"):
            with self.subTest(field=field):
                policy = self._policy()
                policy["full_pair"][field] = ["no"]
                with self.assertRaisesRegex(
                    builder.PostVideoQualityError,
                    rf"full_pair\.{field} may allow only yes/not_applicable",
                ):
                    builder.validate_routing_policy(policy)

    def test_uncertainty_codes_fail_closed_even_with_positive_enums(self) -> None:
        quality = _quality()
        quality["uncertainty_codes"] = ["occluded_subject"]
        validated = builder.validate_quality_observation(quality, nframes=4)
        self.assertFalse(
            builder._passes_gate(validated, self._policy()["full_pair"])
        )

    def test_input_receipt_hash_is_caller_pinned(self) -> None:
        self._prepare_three()
        with self.assertRaisesRegex(
            builder.PostVideoQualityError, "receipt differs from caller-pinned"
        ):
            builder.run_audit(
                **{
                    **self._audit_kwargs(),
                    "expected_input_receipt_sha256": "0" * 64,
                    "expected_strict_rows": 3,
                },
                backend_factory=FakeBackend,
            )

    def test_large_qwen_files_are_content_hashed_and_detect_same_size_change(self) -> None:
        weight = self.model / "model-00001-of-00001.safetensors"
        with weight.open("wb") as handle:
            handle.seek(32 * 1024 * 1024)
            handle.write(b"A")
        before = builder._model_inventory(self.model)
        weight_row = next(
            row for row in before["files"] if row["path"] == weight.name
        )
        self.assertRegex(weight_row["sha256"], r"^[0-9a-f]{64}$")
        with weight.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"B")
        after = builder._model_inventory(self.model)
        changed_row = next(
            row for row in after["files"] if row["path"] == weight.name
        )
        self.assertEqual(weight_row["bytes"], changed_row["bytes"])
        self.assertNotEqual(weight_row["sha256"], changed_row["sha256"])
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_launcher_archive_preflight_contract_is_explicit(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "umask 077",
            "unset PYTHONPATH PYTHONHOME PYTHONSTARTUP",
            "mktemp -d",
            'archive_copy="${task_scratch}/source.archive"',
            'cp "${source_archive}" "${archive_copy}"',
            'tar -xf "${archive_copy}"',
            "path.is_absolute()",
            '".." in path.parts',
            "member.isfile() or member.isdir()",
            '"${python_bin}" -I - "${archive_copy}"',
            "archive-only preflight is forbidden inside an allocated Slurm job",
            "VALID_ARCHIVE_PREFLIGHT_ONLY",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_launcher_archive_preflight_executes_and_rejects_unsafe_members(self) -> None:
        valid_archive, tool_sha256 = self._source_archive("valid-source.tar")
        valid = self._run_archive_preflight(valid_archive, tool_sha256)
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
        self.assertIn("VALID_ARCHIVE_PREFLIGHT_ONLY", valid.stdout)

        for member_type in ("absolute", "parent", "symlink", "hardlink", "fifo"):
            with self.subTest(member_type=member_type):
                archive, digest = self._source_archive(
                    f"unsafe-{member_type}.tar",
                    unsafe_member=(member_type, b"unsafe\n"),
                )
                result = self._run_archive_preflight(archive, digest)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsafe source archive", result.stderr)

    def test_production_backend_requires_complete_cuda0_placement(self) -> None:
        execution = builder._production_backend_execution(
            FakePlacementBackend(), torch_module=FakeTorch()
        )
        self.assertTrue(execution["production_backend"])
        self.assertTrue(execution["cuda_only"])
        self.assertEqual(execution["parameter_devices"], ["cuda:0"])
        self.assertEqual(execution["parameter_elements"], 7)
        self.assertEqual(execution["buffer_elements"], 3)
        self.assertEqual(execution["hf_device_map_devices"], ["cuda:0"])

    def test_production_backend_rejects_offload_and_multiple_visible_gpus(self) -> None:
        with self.assertRaisesRegex(builder.PostVideoQualityError, "offload"):
            builder._production_backend_execution(
                FakePlacementBackend(parameter_device="cpu"),
                torch_module=FakeTorch(),
            )
        with self.assertRaisesRegex(builder.PostVideoQualityError, "exactly one"):
            builder._production_backend_execution(
                FakePlacementBackend(), torch_module=FakeTorch(devices=2)
            )


if __name__ == "__main__":
    unittest.main()
