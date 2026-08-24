#!/usr/bin/env python3
"""Fail-closed release audit for the SSFT T1/IAVG/I1/I1A screen.

The auditor does not score or select media.  It independently reopens the two
fresh frame-0 coordinates and all eight runner bundles, verifies the registered
treatments and matched source/noise controls, and writes one canonical 0444
parent receipt.  A parent receipt is execution evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-saic-ssft-mechanism-screen-parent-receipt-v1"
RUNNER_SCHEMA = "bernini-saic-ssft-preregistered-mechanism-screen-v1"
RUNNER_METHOD = "frozen-bernini-saic-ssft-preregistered-mechanism-screen"
FRAME0_SCHEMA = "bernini-saic-frame0-latent-receipt-v1"
ARMS = ("T1", "IAVG", "I1", "I1A")
GROUPS = ("dog", "human")
ZERO_SHA256 = "0" * 64
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MechanismScreenReleaseError(RuntimeError):
    """Raised before an incomplete or ambiguous release can be certified."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise MechanismScreenReleaseError(
            f"value is not canonical JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: Path, *, mode_0444: bool = True) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise MechanismScreenReleaseError(f"cannot stat {path}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise MechanismScreenReleaseError(f"not a plain non-symlink file: {path}")
    if mode_0444 and stat.S_IMODE(info.st_mode) != 0o444:
        raise MechanismScreenReleaseError(f"file is not sealed 0444: {path}")
    return path.resolve(strict=True)


def load_canonical_receipt(path: Path) -> tuple[dict[str, Any], str]:
    resolved = _plain_file(path)
    payload = resolved.read_bytes()
    try:
        receipt = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MechanismScreenReleaseError(
            f"cannot parse canonical receipt: {resolved}"
        ) from error
    if type(receipt) is not dict or payload != canonical_json_bytes(receipt) + b"\n":
        raise MechanismScreenReleaseError(f"receipt is not canonical: {resolved}")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if _SHA256.fullmatch(str(declared or "")) is None:
        raise MechanismScreenReleaseError(f"receipt digest missing: {resolved}")
    if object_sha256(unsigned) != declared:
        raise MechanismScreenReleaseError(f"receipt digest differs: {resolved}")
    return receipt, file_sha256(resolved)


def load_plan(path: Path) -> tuple[dict[str, Any], str]:
    resolved = _plain_file(path)
    payload = resolved.read_bytes()
    try:
        plan = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MechanismScreenReleaseError("cannot parse mechanism plan") from error
    if type(plan) is not dict:
        raise MechanismScreenReleaseError("mechanism plan root differs")
    if plan.get("schema_version") != (
        "bernini-saic-ssft-preregistered-mechanism-screen-plan-v1"
    ):
        raise MechanismScreenReleaseError("mechanism plan schema differs")
    prereg = plan.get("preregistration")
    runtime = plan.get("runtime")
    authority = plan.get("authority")
    if (
        not isinstance(prereg, Mapping)
        or prereg.get("fixed_arm_order") != list(ARMS)
        or prereg.get("registered_before_job132387") is not True
        or prereg.get("scientific_treatment_extension") is not False
        or not isinstance(runtime, Mapping)
        or [runtime.get(key) for key in ("frames", "fps", "latent_frames", "inference_steps")]
        != [81, 25, 21, 40]
        or runtime.get("flow_shift") != 5.0
        or runtime.get("candidate_schedule") != [5, 5, 5] + [1] * 37
        or runtime.get("candidate_continuation") != "candidate_zero"
        or runtime.get("optimizer_steps") != 0
        or runtime.get("training_updates") != 0
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
    ):
        raise MechanismScreenReleaseError("mechanism plan contract differs")
    dependencies = plan.get("implementation_dependencies")
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies)
        != {
            "infer_saic_source_state_flow_transport_v2.py",
            "materialize_saic_frame0_latent_v1.py",
        }
        or any(_SHA256.fullmatch(str(value)) is None for value in dependencies.values())
    ):
        raise MechanismScreenReleaseError("implementation dependencies are not pinned")
    return plan, hashlib.sha256(payload).hexdigest()


def _validate_ffprobe(
    path: Path,
    *,
    expected_sha256: str,
    expected_version_stdout_sha256: str,
    expected_version_first_line: str,
) -> tuple[Path, dict[str, str]]:
    if not path.is_absolute():
        raise MechanismScreenReleaseError("ffprobe path is not absolute")
    try:
        info = path.lstat()
    except OSError as error:
        raise MechanismScreenReleaseError("cannot stat pinned ffprobe") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
        raise MechanismScreenReleaseError(
            "pinned ffprobe is not an executable plain non-symlink file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise MechanismScreenReleaseError("pinned ffprobe path is not canonical")
    if _SHA256.fullmatch(expected_sha256 or "") is None:
        raise MechanismScreenReleaseError("expected ffprobe SHA-256 differs")
    if _SHA256.fullmatch(expected_version_stdout_sha256 or "") is None:
        raise MechanismScreenReleaseError("expected ffprobe version SHA-256 differs")
    observed_sha256 = file_sha256(resolved)
    if observed_sha256 != expected_sha256:
        raise MechanismScreenReleaseError("pinned ffprobe bytes differ")
    try:
        completed = subprocess.run(
            [str(resolved), "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MechanismScreenReleaseError("pinned ffprobe version probe failed") from error
    observed_version_sha256 = hashlib.sha256(completed.stdout).hexdigest()
    try:
        version_stdout = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MechanismScreenReleaseError("pinned ffprobe version output is not UTF-8") from error
    version_lines = version_stdout.splitlines()
    observed_first_line = version_lines[0] if version_lines else ""
    if (
        observed_version_sha256 != expected_version_stdout_sha256
        or observed_first_line != expected_version_first_line
        or completed.stderr
    ):
        raise MechanismScreenReleaseError("pinned ffprobe version identity differs")
    return resolved, {
        "configured_and_resolved_path": str(resolved),
        "file_sha256": observed_sha256,
        "version_stdout_sha256": observed_version_sha256,
        "version_first_line": observed_first_line,
    }


def _ffprobe_exact81(path: Path, ffprobe_bin: Path) -> dict[str, Any]:
    command = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise MechanismScreenReleaseError(f"ffprobe failed for {path}") from error
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        raise MechanismScreenReleaseError(f"ffprobe stream closure differs: {path}")
    stream = streams[0]
    if (
        stream.get("codec_name") != "h264"
        or stream.get("avg_frame_rate") != "25/1"
        or stream.get("nb_read_frames") != "81"
        or type(stream.get("width")) is not int
        or type(stream.get("height")) is not int
        or stream["width"] <= 0
        or stream["height"] <= 0
    ):
        raise MechanismScreenReleaseError(f"media is not H.264 exact81/25fps: {path}")
    return {
        "codec": "h264",
        "frames": 81,
        "fps": 25,
        "width": stream["width"],
        "height": stream["height"],
    }


def _false_authority(receipt: Mapping[str, Any], *, label: str) -> None:
    authority = receipt.get("authority")
    allowed_true = {"frozen_inference_execution_receipt"}
    if not isinstance(authority, Mapping):
        raise MechanismScreenReleaseError(f"{label} authority map differs")
    for key, value in authority.items():
        expected = key in allowed_true
        if value is not expected:
            raise MechanismScreenReleaseError(
                f"{label} authority differs at {key}"
            )


def validate_frame0(
    root: Path,
    source: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    group = source["group"]
    artifact = _plain_file(root / "groups" / group / "frame0.latent.safetensors")
    receipt_path = artifact.with_name(f"{artifact.name}.receipt.json")
    receipt, receipt_file_sha = load_canonical_receipt(receipt_path)
    artifact_section = receipt.get("artifact")
    sealed = receipt.get("sealed_inputs")
    encoding = receipt.get("encoding")
    authority = receipt.get("authority")
    job132387_raw = source["job132387_ephemeral_frame0_tensor_raw_sha256"]
    actual_raw = (
        artifact_section.get("tensor_raw_sha256")
        if isinstance(artifact_section, Mapping)
        else None
    )
    observed_match = actual_raw == job132387_raw
    if (
        receipt.get("schema_version") != FRAME0_SCHEMA
        or receipt.get("method") != "frozen-bernini-frame0-latent-materializer"
        or not isinstance(artifact_section, Mapping)
        or artifact_section.get("path") != str(artifact)
        or artifact_section.get("file_sha256") != file_sha256(artifact)
        or _SHA256.fullmatch(str(actual_raw or "")) is None
        or artifact_section.get("mode") != "0444"
        or not isinstance(sealed, Mapping)
        or sealed.get("row_id") != source["row_id"]
        or sealed.get("source_video_sha256") != source["source_video_sha256"]
        or not isinstance(encoding, Mapping)
        or encoding.get("source_frame0_vae_encode_count") != 1
        or encoding.get("full_source_vae_encode_count") != 0
        or encoding.get("total_vae_encode_count") != 1
        or encoding.get("temporal_video_latent_slice_used") is not False
        or encoding.get("encoded_in_runner") is not False
        or encoding.get("expected_job132387_frame0_tensor_raw_sha256")
        != job132387_raw
        or encoding.get("actual_reference_frame0_tensor_raw_sha256")
        != actual_raw
        or encoding.get("job132387_frame0_tensor_raw_sha256_match")
        is not observed_match
        or source.get("fresh_frame0_must_match_job132387_ephemeral") is not False
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
    ):
        raise MechanismScreenReleaseError(f"{group} frame0 receipt differs")
    return {
        "group": group,
        "artifact_path": str(artifact),
        "artifact_sha256": file_sha256(artifact),
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_file_sha,
        "receipt_digest": receipt["receipt_digest"],
        "tensor_raw_sha256": actual_raw,
        "job132387_ephemeral_i0_tensor_raw_sha256": job132387_raw,
        "matches_job132387_ephemeral_i0_tensor": observed_match,
        "physical_vae_encode_count": 1,
        "consumer_arms": list(ARMS),
        "plan_sha256": plan_sha256,
    }


def validate_runner(
    root: Path,
    source: Mapping[str, Any],
    arm_plan: Mapping[str, Any],
    frame0: Mapping[str, Any],
    *,
    method_revision: str,
    method_archive_sha256: str,
    ffprobe_bin: Path,
) -> dict[str, Any]:
    group = source["group"]
    arm = arm_plan["arm"]
    media = _plain_file(root / "groups" / group / f"{arm}.mp4")
    receipt_path = _plain_file(media.with_name(f"{media.name}.receipt.json"))
    latent = _plain_file(
        media.with_name(f"{media.name}.normalized-clean-latent.safetensors")
    )
    receipt, receipt_file_sha = load_canonical_receipt(receipt_path)
    _false_authority(receipt, label=f"{group}/{arm}")
    arm_value = receipt.get("arm")
    schedule = receipt.get("schedule")
    noise = receipt.get("noise")
    output = receipt.get("output")
    transport = receipt.get("transport")
    distributed = receipt.get("distributed_execution")
    model = receipt.get("model")
    expected_arm = {
        "arm": arm,
        "field_regime": arm_plan["field_regime"],
        "task_name": arm_plan["task_name"],
        "guidance_mode": arm_plan["guidance_mode"],
        "candidate_schedule": [5, 5, 5] + [1] * 37,
        "anc_enabled": True,
        "aggregation_mode": arm_plan["aggregation_mode"],
        "temperature": arm_plan["temperature"],
        "anchor_latent_phase_zero": arm_plan["anchor_latent_phase_zero"],
    }
    if (
        receipt.get("schema_version") != RUNNER_SCHEMA
        or receipt.get("method") != RUNNER_METHOD
        or arm_value != expected_arm
        or not isinstance(schedule, Mapping)
        or [schedule.get(key) for key in ("num_frames", "fps", "latent_frames", "num_inference_steps")]
        != [81, 25, 21, 40]
        or schedule.get("flow_shift") != 5.0
        or not isinstance(noise, Mapping)
        or noise.get("candidate_schedule") != expected_arm["candidate_schedule"]
        or noise.get("candidate_continuation") != "candidate_zero"
        or noise.get("master_seed") != source["rollout_seed"]
        or _SHA256.fullmatch(str(noise.get("actual_ordered_noise_bank_sha256", ""))) is None
        or _SHA256.fullmatch(str(noise.get("actual_candidate_zero_subbank_sha256", ""))) is None
        or not isinstance(output, Mapping)
        or output.get("path") != str(media)
        or output.get("sha256") != file_sha256(media)
        or output.get("frame_count") != 81
        or output.get("fps") != 25
        or not isinstance(output.get("normalized_clean_latent"), Mapping)
        or output["normalized_clean_latent"].get("path") != str(latent)
        or output["normalized_clean_latent"].get("sha256") != file_sha256(latent)
        or not isinstance(transport, Mapping)
        or transport.get("complete_source_video_vae_encoded_in_runner") is not False
        or transport.get("source_outer_clean_state_loaded_from_sealed_coordinate") is not True
        or not isinstance(transport.get("sealed_source_coordinate"), Mapping)
        or transport["sealed_source_coordinate"].get("artifact_path")
        != source["source_clean_latent_path"]
        or transport["sealed_source_coordinate"].get("artifact_sha256")
        != source["source_clean_latent_sha256"]
        or transport["sealed_source_coordinate"].get("receipt_path")
        != source["source_clean_receipt_path"]
        or transport["sealed_source_coordinate"].get("receipt_file_sha256")
        != source["source_clean_receipt_sha256"]
        or transport["sealed_source_coordinate"].get("tensor_raw_sha256")
        != source["source_clean_tensor_raw_sha256"]
        or transport["sealed_source_coordinate"].get("source_video_sha256")
        != source["source_video_sha256"]
        or not isinstance(transport.get("sealed_frame0_coordinate"), Mapping)
        or transport["sealed_frame0_coordinate"].get("artifact_path")
        != frame0["artifact_path"]
        or transport["sealed_frame0_coordinate"].get("artifact_sha256")
        != frame0["artifact_sha256"]
        or transport["sealed_frame0_coordinate"].get("receipt_file_sha256")
        != frame0["receipt_file_sha256"]
        or transport["sealed_frame0_coordinate"].get("receipt_digest")
        != frame0["receipt_digest"]
        or transport["sealed_frame0_coordinate"].get("tensor_raw_sha256")
        != frame0["tensor_raw_sha256"]
        or transport["sealed_frame0_coordinate"].get(
            "job132387_ephemeral_i0_tensor_raw_sha256"
        )
        != frame0["job132387_ephemeral_i0_tensor_raw_sha256"]
        or transport["sealed_frame0_coordinate"].get(
            "matches_job132387_ephemeral_i0_tensor"
        )
        is not frame0["matches_job132387_ephemeral_i0_tensor"]
        or not isinstance(distributed, Mapping)
        or distributed.get("all_rank_exact") is not True
        or not isinstance(distributed.get("certificate"), Mapping)
        or distributed["certificate"].get("native_guided_query_success_count")
        != arm_plan["guided_queries_per_source"]
        or distributed["certificate"].get("native_guided_query_attempt_count")
        != arm_plan["guided_queries_per_source"]
        or distributed["certificate"].get("native_raw_transformer_forward_success_count")
        != arm_plan["raw_transformer_forwards_per_source"]
        or distributed["certificate"].get("native_raw_transformer_forward_attempt_count")
        != arm_plan["raw_transformer_forwards_per_source"]
        or distributed["certificate"].get("source_clean_encoded_in_runner") is not False
        or distributed["certificate"].get("shared_frame0_raw_sha256")
        != frame0["tensor_raw_sha256"]
        or not isinstance(model, Mapping)
        or not isinstance(model.get("method_provenance"), Mapping)
        or model["method_provenance"].get("revision") != method_revision
        or model["method_provenance"].get("archive_sha256") != method_archive_sha256
    ):
        raise MechanismScreenReleaseError(f"runner receipt differs: {group}/{arm}")
    visual_expected = arm != "T1"
    visual_observed = distributed["certificate"].get(
        "visual_i0_enabled_all_exact40_cells"
    )
    native_reference = distributed["certificate"].get(
        "native_field_reference_frame0_raw_sha256"
    )
    if (
        visual_observed is not visual_expected
        or native_reference
        != (frame0["tensor_raw_sha256"] if visual_expected else ZERO_SHA256)
    ):
        raise MechanismScreenReleaseError(
            f"visual-I0 treatment differs: {group}/{arm}"
        )
    media_probe = _ffprobe_exact81(media, ffprobe_bin)
    return {
        "group": group,
        "arm": arm,
        "media_path": str(media),
        "media_sha256": file_sha256(media),
        "media_size": media.stat().st_size,
        "media_probe": media_probe,
        "normalized_clean_latent_path": str(latent),
        "normalized_clean_latent_sha256": file_sha256(latent),
        "normalized_clean_latent_tensor_raw_sha256": output[
            "normalized_clean_latent"
        ]["tensor_raw_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_file_sha,
        "receipt_digest": receipt["receipt_digest"],
        "ordered_noise_bank_sha256": noise["actual_ordered_noise_bank_sha256"],
        "candidate_zero_noise_sha256": noise[
            "actual_candidate_zero_subbank_sha256"
        ],
        "source_clean_tensor_raw_sha256": source[
            "source_clean_tensor_raw_sha256"
        ],
        "shared_frame0_tensor_raw_sha256": frame0["tensor_raw_sha256"],
        "guided_query_attempt_and_success_count": arm_plan[
            "guided_queries_per_source"
        ],
        "raw_forward_attempt_and_success_count": arm_plan[
            "raw_transformer_forwards_per_source"
        ],
        "visual_i0_enabled": visual_expected,
    }


def _publish_parent(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MechanismScreenReleaseError("parent receipt already exists")
    payload = canonical_json_bytes(receipt) + b"\n"
    descriptor: Optional[int] = None
    temporary: Optional[Path] = None
    try:
        descriptor, text = tempfile.mkstemp(prefix=".parent.", dir=str(path.parent))
        temporary = Path(text)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = None
        _plain_file(path)
        if path.read_bytes() != payload:
            raise MechanismScreenReleaseError("published parent bytes differ")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def release(args: argparse.Namespace) -> dict[str, Any]:
    if _SHA1.fullmatch(args.method_source_revision or "") is None:
        raise MechanismScreenReleaseError("method source revision differs")
    if _SHA256.fullmatch(args.method_source_archive_sha256 or "") is None:
        raise MechanismScreenReleaseError("method archive digest differs")
    if not re.fullmatch(r"[0-9]+", args.job_id or ""):
        raise MechanismScreenReleaseError("Slurm job ID differs")
    ffprobe_bin, ffprobe_identity = _validate_ffprobe(
        Path(args.ffprobe_bin),
        expected_sha256=args.ffprobe_sha256,
        expected_version_stdout_sha256=args.ffprobe_version_stdout_sha256,
        expected_version_first_line=args.ffprobe_version_first_line,
    )
    root = Path(args.output_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise MechanismScreenReleaseError("output root differs")
    plan, plan_sha = load_plan(Path(args.plan))
    plan_sources = plan["sealed_inputs"]["sources"]
    arm_plans = plan["arms"]
    if [item.get("group") for item in plan_sources] != list(GROUPS):
        raise MechanismScreenReleaseError("source order differs")
    if [item.get("arm") for item in arm_plans] != list(ARMS):
        raise MechanismScreenReleaseError("arm plan order differs")
    frame0 = {
        item["group"]: validate_frame0(root, item, plan_sha256=plan_sha)
        for item in plan_sources
    }
    attempts = []
    for arm_plan in arm_plans:
        for source in plan_sources:
            attempts.append(
                validate_runner(
                    root,
                    source,
                    arm_plan,
                    frame0[source["group"]],
                    method_revision=args.method_source_revision,
                    method_archive_sha256=args.method_source_archive_sha256,
                    ffprobe_bin=ffprobe_bin,
                )
            )
    for group in GROUPS:
        rows = [item for item in attempts if item["group"] == group]
        if [item["arm"] for item in rows] != list(ARMS):
            raise MechanismScreenReleaseError(f"{group} fixed arm order differs")
        if len({item["source_clean_tensor_raw_sha256"] for item in rows}) != 1:
            raise MechanismScreenReleaseError(f"{group} source coordinate differs")
        if len({item["shared_frame0_tensor_raw_sha256"] for item in rows}) != 1:
            raise MechanismScreenReleaseError(f"{group} frame0 coordinate differs")
        if len({item["candidate_zero_noise_sha256"] for item in rows}) != 1:
            raise MechanismScreenReleaseError(f"{group} matched noise differs")
        if len({item["ordered_noise_bank_sha256"] for item in rows}) != 1:
            raise MechanismScreenReleaseError(f"{group} ordered noise bank differs")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "job_id": args.job_id,
        "release_id": plan["release_id"],
        "plan_path": str(Path(args.plan).resolve(strict=True)),
        "plan_sha256": plan_sha,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "media_probe_implementation": ffprobe_identity,
        "upstream_job_id": "132387",
        "fixed_arm_order": list(ARMS),
        "topology": plan["runtime"]["topology"],
        "exact_runtime": {
            "frames": 81,
            "fps": 25,
            "latent_frames": 21,
            "inference_steps": 40,
            "flow_shift": 5.0,
        },
        "frame0_materializations": [frame0[group] for group in GROUPS],
        "attempts": attempts,
        "verification": {
            "source_materialization_reused_only_from_job132387": True,
            "source_clean_consumer_count": 8,
            "fresh_frame0_materialization_count": 2,
            "frame0_vae_encode_count_per_source": 1,
            "frame0_consumer_count": 8,
            "runner_full_source_vae_encode_count": 0,
            "runner_frame0_vae_encode_count": 0,
            "runner_receipt_count": 8,
            "media_count": 8,
            "normalized_clean_latent_count": 8,
            "all_media_exact81_25fps_h264": True,
            "source_coordinate_equal_within_source": True,
            "frame0_coordinate_equal_within_source": True,
            "candidate_zero_noise_equal_within_source": True,
            "ordered_noise_bank_equal_within_source": True,
            "fixed_arm_order_verified": True,
        },
        "interpretation": {
            "mechanism_screen_only": True,
            "semantic_review_pending": True,
            "visual_quality_review_pending": True,
            "selection_performed": False,
            "training_performed": False,
            "optimizer_step_count": 0,
        },
        "authority": dict(plan["authority"]),
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _publish_parent(root / "mechanism-screen-parent-receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--ffprobe-bin", required=True)
    parser.add_argument("--ffprobe-sha256", required=True)
    parser.add_argument("--ffprobe-version-stdout-sha256", required=True)
    parser.add_argument("--ffprobe-version-first-line", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    receipt = release(build_parser().parse_args(argv))
    print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "GROUPS",
    "MechanismScreenReleaseError",
    "build_parser",
    "canonical_json_bytes",
    "file_sha256",
    "load_canonical_receipt",
    "load_plan",
    "main",
    "object_sha256",
    "release",
    "validate_frame0",
    "validate_runner",
]
