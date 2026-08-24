from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "motive" / "wan22_parallel_shards.py"
SUBMIT = ROOT / "scripts" / "auh_submit_wan22_i2v_parallel.sh"
FULL = ROOT / "scripts" / "auh_wan22_i2v_parallel_full.sbatch"
FINALIZE = ROOT / "scripts" / "auh_wan22_i2v_parallel_finalize.sbatch"

_SPEC = importlib.util.spec_from_file_location(
    "_wan22_parallel_shards_under_test",
    MODULE,
)
assert _SPEC is not None and _SPEC.loader is not None
_PARALLEL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PARALLEL
_SPEC.loader.exec_module(_PARALLEL)

Wan22ParallelError = _PARALLEL.Wan22ParallelError
prepare_parallel_run = _PARALLEL.prepare_parallel_run
finalize_parallel_run = _PARALLEL.finalize_parallel_run
sample_seed = _PARALLEL.sample_seed


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(_PARALLEL._pretty_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(_PARALLEL._canonical_bytes(row) + b"\n" for row in rows)
    )


def _probe() -> dict[str, object]:
    return {
        "frames": 81,
        "frame_rate": "25/1",
        "duration_seconds": 3.24,
    }


def _contract_temporal_policy() -> dict[str, object]:
    return {
        "policy_version": _PARALLEL.TEMPORAL_POLICY,
        "source_frame_count": 81,
        "target_frame_count": 81,
        "source_frame_rate": "25/1",
        "target_container_frame_rate": "25/1",
        "nominal_duration_seconds": 3.24,
        "source_duration_range_seconds": [3.24, 3.24],
        "duration_match_tolerance_frames": 1,
        "duration_match_tolerance_seconds": 0.04,
        "model_sample_fps": 16,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "source_frame_count_must_be_4n_plus_1": True,
        "source_target_frame_count_equal": True,
        "source_target_frame_rate_equal": True,
        "source_target_duration_within_tolerance": True,
        "batch_time_grid_uniform": True,
    }


def _pair_temporal_policy() -> dict[str, object]:
    return {
        "policy_version": _PARALLEL.TEMPORAL_POLICY,
        "model_sample_fps": 16,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "source": {
            "frame_count": 81,
            "frame_rate": "25/1",
            "duration_seconds": 3.24,
        },
        "target": {
            "frame_count": 81,
            "frame_rate": "25/1",
            "duration_seconds": 3.24,
        },
        "frame_count_equal": True,
        "frame_rate_equal": True,
        "duration_delta_seconds": 0.0,
        "duration_delta_frames": 0.0,
        "duration_match_tolerance_frames": 1,
        "duration_match_tolerance_seconds": 0.04,
        "duration_within_tolerance": True,
    }


def _source_row(
    *,
    index: int,
    source_video: Path,
    source_payload: bytes,
) -> dict[str, object]:
    iid = f"iid-{index:03d}"
    anchor_payload = f"anchor-{iid}".encode("utf-8")
    return {
        "schema_version": _PARALLEL.GENERATION_MANIFEST_SCHEMA,
        "iid": iid,
        "group_id": f"group-{index:03d}",
        "action_category": "interaction",
        "target_action_verb": "pick_up",
        "action_change_substantive": "yes",
        "source_video": str(source_video),
        "resolved_source_video": str(source_video),
        "source_video_sha256": _sha(source_payload),
        "anchor_image": f"/frozen/anchor-{index:03d}.png",
        "resolved_anchor_image": f"/frozen/anchor-{index:03d}.png",
        "anchor_sha256": _sha(anchor_payload),
        "edit_instruction": f"Make actor {index} pick up the visible object.",
        "absolute_target_prompt": (
            f"The same actor {index} picks up the visible object while "
            "identity, scene, and camera remain unchanged."
        ),
        "preservation_constraints": ["Preserve identity and scene."],
        "causal_stages": ["Reach.", "Pick up."],
        "manifest_role": "approved_generation",
        "production_eligible": True,
        "human_review_status": "approved",
        "generation_authorized": True,
        "approval": {
            "schema_version": _PARALLEL.APPROVAL_SCHEMA,
            "approval_digest": _sha(f"approval-{iid}".encode("utf-8")),
            "approval_file_sha256": _sha(
                f"approval-file-{iid}".encode("utf-8")
            ),
            "proposal_sha256": _sha(f"proposal-{iid}".encode("utf-8")),
            "reviewer_id": "unit-test-reviewer",
            "reviewed_at_utc": "2026-07-30T12:00:00+00:00",
            "decision": "approved",
            "reason": "target action was manually verified",
        },
    }


def _create_source_manifest(
    root: Path,
    *,
    row_count: int,
) -> tuple[Path, list[dict[str, object]]]:
    rows = []
    media = root / "media"
    media.mkdir()
    for index in range(row_count):
        payload = f"source-video-{index}".encode("utf-8")
        source = media / f"source-{index:03d}.mp4"
        source.write_bytes(payload)
        rows.append(
            _source_row(
                index=index,
                source_video=source.resolve(),
                source_payload=payload,
            )
        )
    manifest = root / "generation_manifest.jsonl"
    _write_jsonl(manifest, rows)
    return manifest, rows


def _materialize_shard_outputs(
    parallel_root: Path,
    *,
    source_rows: list[dict[str, object]],
) -> None:
    plan = json.loads(
        (parallel_root / _PARALLEL.PLAN_NAME).read_text(encoding="utf-8")
    )
    for shard in plan["shards"]:
        start = shard["row_start_zero_based"]
        stop = shard["row_stop_exclusive"]
        rows = source_rows[start:stop]
        output_root = Path(shard["output_root"])
        samples = output_root / "samples"
        samples.mkdir(parents=True)

        selected_inputs = [
            {
                "index": index,
                "iid": row["iid"],
                "group_id": row["group_id"],
                "row_digest": _PARALLEL._object_digest(row),
                "seed": sample_seed(
                    _PARALLEL.EXPECTED_BASE_SEED,
                    str(row["iid"]),
                ),
                "source_video_ffprobe": _probe(),
                "authorization_mode": "bound_human_approval",
                "manifest_role": row["manifest_role"],
                "production_eligible": row["production_eligible"],
                "approval": row["approval"],
                "action_change_substantive": "yes",
            }
            for index, row in enumerate(rows)
        ]
        contract: dict[str, object] = {
            "schema_version": _PARALLEL.RUN_SCHEMA,
            "manifest": {
                "path": shard["manifest"]["path"],
                "sha256": shard["manifest"]["sha256"],
                "bytes": shard["manifest"]["bytes"],
                "row_count": shard["row_count"],
                "selected_row_count": shard["row_count"],
                "max_samples": None,
            },
            "selected_inputs": selected_inputs,
            "distributed_execution": {
                "world_size": 8,
                "cooperative_samples_per_step": 1,
                "independent_model_per_gpu": False,
                "t5_fsdp": True,
                "dit_fsdp": True,
                "ulysses_size": 8,
                "max_new_samples_per_allocation": shard["row_count"],
            },
            "generation_parameters": {
                "size": "1280*720",
                "frame_num": 81,
                "sample_steps": 40,
                "sample_shift": 5.0,
                "model_sample_fps": 16,
                "output_container_frame_rate": "25/1",
                "base_seed": 260730,
            },
            "authorization": {
                "allow_pending_review": False,
                "pending_review_override_supported": False,
                "requires_explicit_human_approval": True,
                "approved_manifest_role": "approved_generation",
                "approval_schema": _PARALLEL.APPROVAL_SCHEMA,
            },
            "temporal_policy": _contract_temporal_policy(),
        }
        contract["contract_digest"] = _PARALLEL._object_digest(contract)
        _write_json(output_root / _PARALLEL.RUN_CONTRACT_NAME, contract)

        generated_rows: list[dict[str, object]] = []
        result_digests: list[str] = []
        for sample_index, row in enumerate(rows):
            iid = str(row["iid"])
            sample_dir = samples / iid
            sample_dir.mkdir()
            payloads = {
                "preview.mp4": f"target-{iid}".encode("utf-8"),
                "conditioning_anchor_original.png": (
                    f"anchor-{iid}".encode("utf-8")
                ),
                "conditioning_frame0_float32.npy": (
                    f"float-{iid}".encode("utf-8")
                ),
                "conditioning_frame0.png": f"png-{iid}".encode("utf-8"),
            }
            for name, payload in payloads.items():
                (sample_dir / name).write_bytes(payload)
            outputs = {
                "preview_mp4": "preview.mp4",
                "preview_mp4_sha256": _sha(payloads["preview.mp4"]),
                "conditioning_anchor_original": (
                    "conditioning_anchor_original.png"
                ),
                "conditioning_anchor_original_sha256": _sha(
                    payloads["conditioning_anchor_original.png"]
                ),
                "conditioning_frame0_float32": (
                    "conditioning_frame0_float32.npy"
                ),
                "conditioning_frame0_float32_sha256": _sha(
                    payloads["conditioning_frame0_float32.npy"]
                ),
                "conditioning_frame0_png": "conditioning_frame0.png",
                "conditioning_frame0_png_sha256": _sha(
                    payloads["conditioning_frame0.png"]
                ),
                "preview_mp4_ffprobe": _probe(),
            }
            result: dict[str, object] = {
                "schema_version": _PARALLEL.SAMPLE_SCHEMA,
                "iid": iid,
                "group_id": row["group_id"],
                "sample_index": sample_index,
                "manifest_sha256": shard["manifest"]["sha256"],
                "manifest_row_digest": _PARALLEL._object_digest(row),
                "contract_digest": contract["contract_digest"],
                "seed": sample_seed(260730, iid),
                "authorization_mode": "bound_human_approval",
                "manifest_role": row["manifest_role"],
                "production_eligible": row["production_eligible"],
                "approval": row["approval"],
                "action_change_substantive": "yes",
                "human_review_status_at_generation": "approved",
                "generation_authorized_in_manifest": True,
                "prompt": {
                    "field": "absolute_target_prompt",
                    "text": row["absolute_target_prompt"],
                    "edit_instruction": row["edit_instruction"],
                },
                "inputs": {
                    "anchor_sha256": row["anchor_sha256"],
                    "source_video_resolved_path": row[
                        "resolved_source_video"
                    ],
                    "source_video_sha256": row["source_video_sha256"],
                    "source_video_ffprobe": _probe(),
                },
                "generation_parameters": contract["generation_parameters"],
                "first_frame_policy": {
                    "policy_version": _PARALLEL.FIRST_FRAME_POLICY,
                    "tensor_frame0_overridden_before_encoding": True,
                    "preencode_frame0_matches_png_pixels": True,
                    "mp4_codec_is_lossy": True,
                    "mp4_decode_pixel_equality_claimed": False,
                },
                "temporal_policy": _pair_temporal_policy(),
                "outputs": outputs,
            }
            result["result_digest"] = _PARALLEL._object_digest(result)
            _write_json(sample_dir / _PARALLEL.SAMPLE_RESULT_NAME, result)
            result_digests.append(str(result["result_digest"]))

            generated_rows.append(
                {
                    "schema_version": _PARALLEL.GENERATED_MANIFEST_SCHEMA,
                    "iid": iid,
                    "group_id": row["group_id"],
                    "action_category": row["action_category"],
                    "target_action_verb": row["target_action_verb"],
                    "absolute_target_prompt": row["absolute_target_prompt"],
                    "edit_instruction": row["edit_instruction"],
                    "source_video": row["resolved_source_video"],
                    "source_video_sha256": row["source_video_sha256"],
                    "conditioning_anchor_original": str(
                        sample_dir / "conditioning_anchor_original.png"
                    ),
                    "conditioning_anchor_original_sha256": outputs[
                        "conditioning_anchor_original_sha256"
                    ],
                    "conditioning_frame0_float32": str(
                        sample_dir / "conditioning_frame0_float32.npy"
                    ),
                    "conditioning_frame0_float32_sha256": outputs[
                        "conditioning_frame0_float32_sha256"
                    ],
                    "conditioning_frame0_png": str(
                        sample_dir / "conditioning_frame0.png"
                    ),
                    "conditioning_frame0_png_sha256": outputs[
                        "conditioning_frame0_png_sha256"
                    ],
                    "target_preview_mp4": str(sample_dir / "preview.mp4"),
                    "target_preview_mp4_sha256": outputs[
                        "preview_mp4_sha256"
                    ],
                    "result_json": str(
                        sample_dir / _PARALLEL.SAMPLE_RESULT_NAME
                    ),
                    "result_digest": result["result_digest"],
                    "seed": result["seed"],
                    "authorization_mode": result["authorization_mode"],
                    "manifest_role": row["manifest_role"],
                    "production_eligible": row["production_eligible"],
                    "approval": row["approval"],
                    "action_change_substantive": "yes",
                    "first_frame_policy": (
                        _PARALLEL.FIRST_FRAME_POLICY
                    ),
                    "mp4_decode_pixel_equality_claimed": False,
                    "temporal_policy": _pair_temporal_policy(),
                }
            )
        generated_path = output_root / _PARALLEL.GENERATED_MANIFEST_NAME
        _write_jsonl(generated_path, generated_rows)
        completion: dict[str, object] = {
            "schema_version": _PARALLEL.COMPLETE_SCHEMA,
            "contract_digest": contract["contract_digest"],
            "manifest_sha256": shard["manifest"]["sha256"],
            "selected_sample_count": len(rows),
            "completed_sample_count": len(rows),
            "generated_manifest": _PARALLEL.GENERATED_MANIFEST_NAME,
            "generated_manifest_sha256": _sha(generated_path.read_bytes()),
            "temporal_policy": contract["temporal_policy"],
            "sample_result_digests": result_digests,
        }
        completion["complete_digest"] = _PARALLEL._object_digest(completion)
        _write_json(output_root / _PARALLEL.RUN_COMPLETE_NAME, completion)


class Wan22ParallelPrepareTests(unittest.TestCase):
    def test_legacy_approved_manifest_cannot_create_submission_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, rows = _create_source_manifest(root, row_count=21)
            parallel_root = root / "parallel"
            with self.assertRaisesRegex(
                Wan22ParallelError,
                "signed generation release gate is unavailable",
            ):
                prepare_parallel_run(
                    manifest_path=manifest,
                    parallel_root=parallel_root,
                    geometry_job_id=113122,
                    shard_count=3,
                    allow_pending_review=False,
                )
            self.assertFalse(parallel_root.exists())

            _, _, _, structural_rows = _PARALLEL._strict_jsonl_file(
                manifest,
                context="test legacy manifest structure",
            )
            iids, groups = _PARALLEL._validate_source_rows(
                structural_rows,
                allow_pending_review=False,
            )
            self.assertEqual(iids, [str(row["iid"]) for row in rows])
            self.assertEqual(groups, [str(row["group_id"]) for row in rows])

    def test_pending_v8_boolean_resign_and_override_all_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, rows = _create_source_manifest(root, row_count=2)
            pending = dict(rows[0])
            pending.update(
                {
                    "manifest_role": "review_proposal",
                    "production_eligible": False,
                    "human_review_status": "pending",
                    "generation_authorized": False,
                    "approval": None,
                    "qwen_provenance_schema": (
                        "goku-action-anchor-qwen-provenance-v8"
                    ),
                }
            )
            boolean_resign = dict(rows[0])
            boolean_resign["generation_authorized"] = True
            boolean_resign["production_eligible"] = True
            boolean_resign["human_review_status"] = "approved"
            boolean_resign["manifest_role"] = "approved_generation"
            boolean_resign["resigned_manifest_sha256"] = _sha(
                _PARALLEL._canonical_bytes(boolean_resign)
            )
            pending_manifest = root / "pending-v8.jsonl"
            resigned_manifest = root / "boolean-resign.jsonl"
            _write_jsonl(pending_manifest, [pending])
            _write_jsonl(resigned_manifest, [boolean_resign])

            cases = (
                ("pending-v8", pending_manifest, False),
                ("boolean-resign", resigned_manifest, False),
                ("pending-override", manifest, True),
            )
            for name, candidate, allow_pending in cases:
                with self.subTest(case=name):
                    output = root / f"rejected-{name}"
                    with self.assertRaisesRegex(
                        Wan22ParallelError,
                        "signed generation release gate is unavailable",
                    ):
                        prepare_parallel_run(
                            manifest_path=candidate,
                            parallel_root=output,
                            geometry_job_id=113122,
                            shard_count=1,
                            allow_pending_review=allow_pending,
                        )
                    self.assertFalse(output.exists())
            with mock.patch.object(
                _PARALLEL,
                "SIGNED_RELEASE_VERIFIER_AVAILABLE",
                True,
            ):
                with self.assertRaisesRegex(
                    Wan22ParallelError,
                    "signed generation release gate is unavailable",
                ):
                    prepare_parallel_run(
                        manifest_path=resigned_manifest,
                        parallel_root=root / "boolean-gate-rejected",
                        geometry_job_id=113122,
                        shard_count=1,
                        allow_pending_review=False,
                    )


class Wan22ParallelFinalizeTests(unittest.TestCase):
    def _fixture(
        self,
        temporary: str,
    ) -> tuple[Path, list[dict[str, object]]]:
        root = Path(temporary)
        manifest, rows = _create_source_manifest(root, row_count=6)
        parallel_root = root / "parallel"
        # Finalization/geometry utilities remain testable, but this test-only
        # monkeypatch is never exposed through the CLI or production module.
        with mock.patch.object(
            _PARALLEL,
            "require_signed_generation_release",
            return_value=None,
        ):
            prepare_parallel_run(
                manifest_path=manifest,
                parallel_root=parallel_root,
                geometry_job_id=113122,
                shard_count=3,
                allow_pending_review=False,
            )
        _materialize_shard_outputs(parallel_root, source_rows=rows)
        return parallel_root, rows

    def test_finalize_validates_and_publishes_idempotent_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parallel_root, rows = self._fixture(temporary)
            aggregate = finalize_parallel_run(parallel_root=parallel_root)
            self.assertEqual(aggregate["generated_row_count"], 6)
            self.assertEqual(
                aggregate["generated_iids"],
                [row["iid"] for row in rows],
            )
            final_manifest = (
                parallel_root
                / "final"
                / _PARALLEL.GENERATED_MANIFEST_NAME
            )
            merged_rows = [
                json.loads(line)
                for line in final_manifest.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["iid"] for row in merged_rows],
                [row["iid"] for row in rows],
            )
            repeated = finalize_parallel_run(parallel_root=parallel_root)
            self.assertEqual(
                repeated["aggregate_digest"],
                aggregate["aggregate_digest"],
            )

    def test_finalize_rejects_parameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parallel_root, _ = self._fixture(temporary)
            contract_path = (
                parallel_root
                / "shards"
                / "shard_001"
                / _PARALLEL.RUN_CONTRACT_NAME
            )
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["generation_parameters"]["sample_steps"] = 39
            del contract["contract_digest"]
            contract["contract_digest"] = _PARALLEL._object_digest(contract)
            _write_json(contract_path, contract)
            with self.assertRaisesRegex(
                Wan22ParallelError,
                r"generation_parameters\.sample_steps differs",
            ):
                finalize_parallel_run(parallel_root=parallel_root)

    def test_finalize_rejects_generated_order_and_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parallel_root, _ = self._fixture(temporary)
            generated = (
                parallel_root
                / "shards"
                / "shard_000"
                / _PARALLEL.GENERATED_MANIFEST_NAME
            )
            rows = [
                json.loads(line)
                for line in generated.read_text(encoding="utf-8").splitlines()
            ]
            _write_jsonl(generated, list(reversed(rows)))
            complete_path = generated.parent / _PARALLEL.RUN_COMPLETE_NAME
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["generated_manifest_sha256"] = _sha(generated.read_bytes())
            del complete["complete_digest"]
            complete["complete_digest"] = _PARALLEL._object_digest(complete)
            _write_json(complete_path, complete)
            with self.assertRaisesRegex(
                Wan22ParallelError,
                "generated manifest IID order differs",
            ):
                finalize_parallel_run(parallel_root=parallel_root)

        with tempfile.TemporaryDirectory() as temporary:
            parallel_root, _ = self._fixture(temporary)
            target = next(
                (
                    parallel_root / "shards" / "shard_002" / "samples"
                ).glob("*/preview.mp4")
            )
            target.write_bytes(b"mutated")
            with self.assertRaisesRegex(
                Wan22ParallelError,
                "hash mismatch",
            ):
                finalize_parallel_run(parallel_root=parallel_root)

    def test_finalize_rejects_forged_extra_target_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parallel_root, _ = self._fixture(temporary)
            shard_root = parallel_root / "shards" / "shard_000"
            generated_path = shard_root / _PARALLEL.GENERATED_MANIFEST_NAME
            generated_rows = [
                json.loads(line)
                for line in generated_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            sample = generated_rows[0]
            result_path = Path(sample["result_json"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["temporal_policy"]["target"]["frame_count"] = 97
            result["outputs"]["preview_mp4_ffprobe"]["frames"] = 97
            del result["result_digest"]
            result["result_digest"] = _PARALLEL._object_digest(result)
            _write_json(result_path, result)

            sample["temporal_policy"]["target"]["frame_count"] = 97
            sample["result_digest"] = result["result_digest"]
            _write_jsonl(generated_path, generated_rows)

            complete_path = shard_root / _PARALLEL.RUN_COMPLETE_NAME
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["generated_manifest_sha256"] = _sha(
                generated_path.read_bytes()
            )
            complete["sample_result_digests"][0] = result["result_digest"]
            del complete["complete_digest"]
            complete["complete_digest"] = _PARALLEL._object_digest(complete)
            _write_json(complete_path, complete)

            with self.assertRaisesRegex(
                Wan22ParallelError,
                r"target\.frame_count differs from run contract",
            ):
                finalize_parallel_run(parallel_root=parallel_root)

    def test_finalize_rejects_completion_temporal_policy_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parallel_root, _ = self._fixture(temporary)
            complete_path = (
                parallel_root
                / "shards"
                / "shard_002"
                / _PARALLEL.RUN_COMPLETE_NAME
            )
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["temporal_policy"]["target_container_frame_rate"] = "16/1"
            del complete["complete_digest"]
            complete["complete_digest"] = _PARALLEL._object_digest(complete)
            _write_json(complete_path, complete)
            with self.assertRaisesRegex(
                Wan22ParallelError,
                "completion temporal_policy differs",
            ):
                finalize_parallel_run(parallel_root=parallel_root)


class Wan22ParallelOrchestrationTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        for path in (SUBMIT, FULL, FINALIZE):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{path.name}: {completed.stderr}",
            )

    def test_submitter_uses_independent_roots_and_afterok_finalizer(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8")
        for marker in (
            "MOTIVE_WAN22_GEOMETRY_JOB_ID:?",
            "MOTIVE_WAN22_PARALLEL_SHARD_COUNT:-3",
            "MOTIVE_WAN22_EXPECTED_ROW_COUNT:-8",
            "MOTIVE_WAN22_SIGNED_RELEASE",
            '--signed-release "${signed_release}"',
            '"${prepare_module}" prepare',
            '--dependency="afterok:${geometry_job_id}"',
            'export MOTIVE_WAN22_OUTPUT_ROOT="${shard_output}"',
            'export MOTIVE_WAN22_MAX_NEW_SAMPLES="${row_count}"',
            "parallel_jobs.tsv",
            'dependency+=":${job_id}"',
            '--dependency="${dependency}"',
            "auh_wan22_i2v_parallel_finalize.sbatch",
        ):
            self.assertIn(marker, text)
        self.assertNotIn('MOTIVE_WAN22_OUTPUT_ROOT="${parallel_root}/full"', text)

    def test_submitter_stops_before_first_sbatch_without_signed_release(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = _create_source_manifest(root, row_count=6)
            snapshot = root / "snapshot"
            snapshot_module = (
                snapshot
                / "methods"
                / "motive"
                / "motive"
                / "wan22_parallel_shards.py"
            )
            snapshot_scripts = snapshot / "methods" / "motive" / "scripts"
            snapshot_module.parent.mkdir(parents=True)
            snapshot_scripts.mkdir(parents=True)
            shutil.copy2(MODULE, snapshot_module)
            for source in (FULL, FINALIZE):
                shutil.copy2(source, snapshot_scripts / source.name)
            ordinary_full = snapshot_scripts / "auh_wan22_i2v_full.sbatch"
            ordinary_full.write_text("#!/usr/bin/env bash\nexit 0\n")

            wan = root / "Wan2.2"
            wan.mkdir()
            (wan / "generate.py").write_text("# fixture\n")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "Wan2.1_VAE.pth").write_bytes(b"fixture")
            ffprobe = root / "ffprobe"
            ffprobe.write_text("#!/usr/bin/env bash\nexit 0\n")
            ffprobe.chmod(0o700)
            python_target = root / "python-real"
            python_target.write_text(
                "#!/usr/bin/env bash\n"
                f'exec "{sys.executable!s}" "$@"\n'
            )
            python_target.chmod(0o700)
            python_bin = root / "python"
            python_bin.symlink_to(python_target)

            fake_bin = root / "bin"
            fake_bin.mkdir()
            counter = root / "sbatch.counter"
            calls = root / "sbatch.calls"
            counter.write_text("12000\n")
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                f"n=$(cat {counter!s})\n"
                "n=$((n + 1))\n"
                f"printf '%s\\n' \"$n\" > {counter!s}\n"
                f"printf '%s\\n' \"$*\" >> {calls!s}\n"
                "printf '%s\\n' \"$n;fixture-cluster\"\n"
            )
            fake_sbatch.chmod(0o700)
            fake_sha256sum = fake_bin / "sha256sum"
            fake_sha256sum.write_text(
                "#!/usr/bin/env bash\nexec shasum -a 256 \"$@\"\n"
            )
            fake_sha256sum.chmod(0o700)

            parallel_root = root / "parallel"
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("SBATCH_")
            }
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
                    "MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT": str(snapshot),
                    "MOTIVE_GOKU_ACTION_GENERATION_MANIFEST": str(manifest),
                    "MOTIVE_WAN22_CODE_ROOT": str(wan),
                    "MOTIVE_WAN22_CKPT_DIR": str(checkpoint),
                    "MOTIVE_WAN22_PYTHON_BIN": str(python_bin),
                    "MOTIVE_WAN22_FFPROBE_BIN": str(ffprobe),
                    "MOTIVE_WAN22_PARALLEL_ROOT": str(parallel_root),
                    "MOTIVE_WAN22_GEOMETRY_JOB_ID": "113122",
                    "MOTIVE_WAN22_ALLOW_PENDING_REVIEW": "0",
                    "MOTIVE_WAN22_PARALLEL_SHARD_COUNT": "3",
                    "MOTIVE_WAN22_EXPECTED_ROW_COUNT": "6",
                }
            )
            completed = subprocess.run(
                ["bash", str(SUBMIT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "signed generation release gate is unavailable",
                completed.stderr,
            )
            self.assertFalse(parallel_root.exists())
            self.assertFalse(calls.exists())

    def test_full_wrapper_freezes_socket_only_rccl(self) -> None:
        text = FULL.read_text(encoding="utf-8")
        for marker in (
            "#SBATCH --gres=gpu:mi210:8",
            "export NCCL_IB_DISABLE=1",
            "unset NCCL_IB_HCA NCCL_IB_GID_INDEX",
            "export NCCL_SOCKET_IFNAME=bond0",
            "export NCCL_SOCKET_FAMILY=AF_INET",
            "export GLOO_SOCKET_IFNAME=bond0",
            "export NCCL_ASYNC_ERROR_HANDLING=1",
            "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
            "export NCCL_DEBUG_SUBSYS=INIT,NET",
            'exec bash "${full_script}"',
        ):
            self.assertIn(marker, text)

    def test_finalizer_satisfies_qos_and_calls_fail_closed_module(self) -> None:
        text = FINALIZE.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:1", text)
        self.assertIn("bgqos enforces MinTRES=gres/gpu=1", text)
        self.assertIn('"${module}" finalize', text)
        self.assertIn("aggregate_complete.json", text)


if __name__ == "__main__":
    unittest.main()
