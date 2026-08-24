#!/usr/bin/env python3
"""Run and bind IID00435's sealed PAIR-v5 pure-T2V display anchor.

This is a topology/receipt overlay around the frozen native Bernini T2V
generator from source archive ``f9360f...``.  It deliberately does not call
the legacy 40-candidate bank auditor: the registered ``visible_gpus`` fields
describe an old single-node bank layout, whereas this overlay records and
requires one physical WORLD4 made from two protected holders with two local
ranks each.

The geometry video is used only to choose the exact81 spatial bucket.  No
source pixels, generated anchor pixels, latent, reference, or receipt is a
Stage-B condition.  Generated media is display-only and does not itself prove
that the requested action succeeded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any, Iterable, Mapping, NoReturn, Sequence


SCHEMA = "bernini-pair-v5-reserve-display-anchor-overlay-v1"
TOPOLOGY_SCHEMA = "bernini-pair-v5-reserve-display-anchor-physical2x2-v1"
SET_SCHEMA = "bernini-pair-v5-reserve-display-anchor-set-v1"
IID = "00435ad621c44fac"
BRANCH_ORDER = ("action", "noop", "incomplete", "reverse")
PROFILE_BRANCHES = {"action-only": ("action",), "family4": BRANCH_ORDER}
SOURCE_ARCHIVE_SHA256 = "f9360fcef6bdcb9e37345515fb85d18e4c444fd2b100de35aeb0c1a55a98ac55"
SOURCE_REVISION = "17cc2c73d774e14cdd10bd2ceea4afbaf4b0be26"
SELECTION_SHA256 = "a4baa1aea27f6497ca2dd615cc09b2b90eee37173f506e60ae7d630c41886be6"
REGISTRY_SHA256 = "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
ROW_DIGEST = "5da2592528633a4886d4f06946ba700ff69827c445748080de7de07e2b365245"
SOURCE_VIDEO_SHA256 = "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1"
SOURCE_VIDEO_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/"
    "00435ad621c44fac/samples/00435ad621c44fac/source_video.mp4"
)
SEED = 2026080821
EXPECTED_HW = [592, 400]
EXPECTED_LATENT_SHAPE = [1, 16, 21, 74, 50]
PROMPT_SHA256 = {
    "action": "b8b3f19e854c8c517549cdaf319af3cf5f07b719444c5468b5081dbf1507ded7",
    "noop": "8b88abdd980fbb6cff3397492fe647da8fa2fdd95b75346449e29fc257d6ffdc",
    "incomplete": "3dfba50307dba421f1eebe140008fd18e6ae80143d3caffd697bc0cc36ad6534",
    "reverse": "07de07b8afb40c8872cdbc38544875eda3f010622350c2333f5bc398338f15a9",
}
METHOD_FILE_SHA256 = {
    "infer_pair_v5_t2v_calibration_bank.py": "e19e353d7e83ce7a7fe37bc958dd67e58ae6ae772fafaba8cc40bfb2097e3db6",
    "infer_native_identity_generation_canary.py": "a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334",
    "infer_lora.py": "ce0bb91aa1850fa4568b6441cc4be4f41db8b8dcfc2afe2d9fcc76a6fca2ebe4",
    "pair_v5_t2v_calibration_bank_spec.py": "c8a81c4a1ab57aa9422d3d0ccf5084bf9ccbc60df0f69fe8849d04e132e25288",
    "tools/author_pair_v5_t2v_calibration_bank.py": "7ac617e9615fef9c488f4f0716915d60f06931afd2adcf2ea4c465fa7f6362ff",
}
HOLDER_NODE = {
    "135407": "auh7-1b-gpu-260",
    "135411": "auh7-1b-gpu-214",
    "135412": "auh7-1b-gpu-293",
}
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
CHECKPOINT_MANIFEST_SHA256 = "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROW_KEYS = {
    "iid", "analysis_split", "action_family_id", "actor_group_id",
    "scene_group_id", "action_group_id", "execution_group",
    "geometry_source_video", "seed", "scene_caption",
    "branch_descriptions", "camera_caption",
}


class DisplayAnchorError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise DisplayAnchorError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, *, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        fail(f"{label} must be an absolute plain file")
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        fail(f"{label} SHA-256 differs")
    try:
        value = json.loads(
            raw, object_pairs_hook=_reject_pairs,
            parse_constant=lambda token: fail(f"non-finite JSON constant: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DisplayAnchorError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def validate_embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or not _SHA256.fullmatch(declared):
        fail(f"{label} receipt digest is absent")
    if object_digest(unsigned) != declared:
        fail(f"{label} receipt digest differs")
    return declared


def _publish_create_only_bytes(path: Path, raw: bytes) -> None:
    """Atomically publish complete bytes without permitting replacement.

    The temporary inode is fully written and fsynced before ``link(2)`` makes
    the final name visible.  Unlike ``replace(2)``, the hard-link publication
    fails if another writer already owns the final path.
    """
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail(f"refusing non-fresh output: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        fail(f"output parent must be a plain directory: {path.parent}")
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    if temporary.exists() or temporary.is_symlink():
        fail(f"temporary publication path is not fresh: {temporary}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    linked = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
            linked = True
        except FileExistsError as error:
            raise DisplayAnchorError(
                f"refusing raced non-fresh output: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    if not linked or not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
        fail(f"atomically published output changed on reread: {path}")


def write_exclusive_json(path: Path, value: Mapping[str, Any]) -> tuple[str, str]:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail(f"refusing non-fresh JSON output: {path}")
    unsigned = dict(value)
    if "receipt_digest" in unsigned:
        fail("caller must not predeclare receipt_digest")
    receipt = {**unsigned, "receipt_digest": object_digest(unsigned)}
    raw = canonical_json_bytes(receipt) + b"\n"
    _publish_create_only_bytes(path, raw)
    return hashlib.sha256(raw).hexdigest(), receipt["receipt_digest"]


def validate_method_root(method_root: Path) -> None:
    if not method_root.is_absolute() or not method_root.is_dir() or method_root.is_symlink():
        fail("method root must be an absolute plain directory")
    for relative, expected in METHOD_FILE_SHA256.items():
        path = method_root / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            fail(f"frozen method file differs: {relative}")


def load_authoring(selection_path: Path, registry_path: Path) -> dict[str, Any]:
    selection = load_json(
        selection_path, label="reserve selection", expected_sha256=SELECTION_SHA256
    )
    registry = load_json(
        registry_path, label="authoring registry", expected_sha256=REGISTRY_SHA256
    )
    if selection != {
        "schema_version": "pair-v5-pure-t2v-calibration-authoring-selection-v1",
        "bank_id": "pair5-t2v-reserve4-v1",
        "expected_cell_count": 4,
        "registry_file": "pair_v5_t2v_calibration_first8_authoring_v1.json",
        "registry_raw_sha256": REGISTRY_SHA256,
        "selected_iids": [
            IID, "0c6915018a5f4d9b", "33322eb8ec1e4703", "71ba57892bd043df"
        ],
        "first_gpu_job_default": False,
    }:
        fail("reserve selection authority differs")
    if (
        registry.get("schema_version") != "pair-v5-pure-t2v-calibration-authoring-v1"
        or registry.get("bank_id") != "pair5-t2v-first8-v1"
        or registry.get("expected_cell_count") != 8
        or not isinstance(registry.get("cells"), list)
        or len(registry["cells"]) != 8
    ):
        fail("authoring registry authority differs")
    rows = [row for row in registry["cells"] if isinstance(row, dict) and row.get("iid") == IID]
    if len(rows) != 1 or set(rows[0]) != _ROW_KEYS or object_digest(rows[0]) != ROW_DIGEST:
        fail("IID00435 authoring row differs")
    row = dict(rows[0])
    if (
        row["analysis_split"] != "fit"
        or row["execution_group"] != "sp4-a"
        or row["geometry_source_video"] != SOURCE_VIDEO_PATH
        or row["seed"] != SEED
        or set(row["branch_descriptions"]) != set(BRANCH_ORDER) | {
            "shuffle", "wrong_actor", "wrong_object", "camera_only",
            "appearance_only", "generic_wrong_motion",
        }
    ):
        fail("IID00435 action authority differs")
    return row


def candidate_from_authoring(row: Mapping[str, Any], branch: str) -> dict[str, Any]:
    if branch not in BRANCH_ORDER:
        fail("display anchor branch is not registered")
    prompt = " ".join(
        (str(row["scene_caption"]).strip(), str(row["branch_descriptions"][branch]).strip(), str(row["camera_caption"]).strip())
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if prompt_sha != PROMPT_SHA256[branch]:
        fail("display anchor prompt authority differs")
    return {
        "iid": IID,
        "candidate_id": f"pair5-t2v-reserve4-v1-{IID}-{branch}",
        "analysis_split": "fit",
        "action_family_id": row["action_family_id"],
        "actor_group_id": row["actor_group_id"],
        "scene_group_id": row["scene_group_id"],
        "action_group_id": row["action_group_id"],
        "semantic_branch": branch,
        "geometry_source_video": SOURCE_VIDEO_PATH,
        "geometry_source_video_sha256": SOURCE_VIDEO_SHA256,
        "full_t2v_caption": prompt,
        "full_t2v_caption_utf8_sha256": prompt_sha,
        "seed": SEED,
    }


def validate_topology_rows(
    rows: Sequence[Mapping[str, Any]], *, job0: str, node0: str, job1: str, node1: str
) -> dict[str, Any]:
    if job0 == job1 or HOLDER_NODE.get(job0) != node0 or HOLDER_NODE.get(job1) != node1:
        fail("holder pair authority differs")
    expected = [
        (0, 0, 0, job0, node0), (1, 1, 0, job0, node0),
        (2, 0, 1, job1, node1), (3, 1, 1, job1, node1),
    ]
    by_rank = {row.get("rank"): row for row in rows}
    if len(rows) != 4 or set(by_rank) != {0, 1, 2, 3}:
        fail("physical topology rank closure differs")
    normalized = []
    for rank, local_rank, node_rank, job, node in expected:
        row = by_rank[rank]
        unsigned = dict(row)
        declared = unsigned.pop("row_digest", None)
        if declared != object_digest(unsigned):
            fail("physical topology row digest differs")
        exact = {
            "schema_version": "bernini-pair-v5-reserve-display-anchor-rank-v1",
            "world_size": 4,
            "local_world_size": 2,
            "rank": rank,
            "local_rank": local_rank,
            "node_rank": node_rank,
            "slurm_job_id": job,
            "hostname": node,
            "torch_cuda_device_count": 2,
        }
        if any(row.get(key) != value for key, value in exact.items()):
            fail(f"physical topology differs at rank {rank}")
        if not isinstance(row.get("slurm_step_id"), str) or not row["slurm_step_id"].isdigit():
            fail("numbered child step identity differs")
        if not isinstance(row.get("torch_hip"), str) or not row["torch_hip"]:
            fail("ROCm runtime identity differs")
        for name in ("rocr_visible_devices", "hip_visible_devices", "cuda_visible_devices"):
            if row.get(name) is not None and not isinstance(row[name], str):
                fail("GPU visibility environment differs")
        normalized.append(dict(row))
    if normalized[0]["slurm_step_id"] != normalized[1]["slurm_step_id"]:
        fail("node0 ranks do not share one child step")
    if normalized[2]["slurm_step_id"] != normalized[3]["slurm_step_id"]:
        fail("node1 ranks do not share one child step")
    for first, second in ((normalized[0], normalized[1]), (normalized[2], normalized[3])):
        for name in ("rocr_visible_devices", "hip_visible_devices", "cuda_visible_devices"):
            if first[name] != second[name]:
                fail("same-node GPU visibility differs across local ranks")
    return {
        "schema_version": TOPOLOGY_SCHEMA,
        "complete": True,
        "world_size": 4,
        "physical_nodes": 2,
        "local_ranks_per_node": 2,
        "ulysses_size": 4,
        "holder_pair": [job0, job1],
        "hostname_order": [node0, node1],
        "rank_rows": normalized,
        "two_mi210_visible_per_rank_process": True,
        "parent_allocations_are_holders_only": True,
    }


def capture_topology(args: argparse.Namespace) -> tuple[Path, str, str]:
    topology_dir = Path(args.topology_dir)
    if not topology_dir.is_absolute() or not topology_dir.is_dir() or topology_dir.is_symlink():
        fail("topology directory must be an absolute plain directory")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        node_rank = int(os.environ["GROUP_RANK"])
    except (KeyError, ValueError) as error:
        raise DisplayAnchorError("torchrun topology environment differs") from error
    hostname = socket.gethostname().split(".", 1)[0]
    job = os.environ.get("SLURM_JOB_ID")
    step = os.environ.get("SLURM_STEP_ID")
    import torch
    row = {
        "schema_version": "bernini-pair-v5-reserve-display-anchor-rank-v1",
        "world_size": world_size,
        "local_world_size": local_world_size,
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "slurm_job_id": job,
        "slurm_step_id": step,
        "hostname": hostname,
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "torch_hip": str(torch.version.hip or ""),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    row["row_digest"] = object_digest(row)
    row_path = topology_dir / f"rank-{rank}.json"
    _publish_create_only_bytes(row_path, canonical_json_bytes(row) + b"\n")
    deadline = time.monotonic() + 120.0
    paths = [topology_dir / f"rank-{item}.json" for item in range(4)]
    while not all(path.is_file() and not path.is_symlink() for path in paths):
        if time.monotonic() >= deadline:
            fail("timed out closing physical topology rows")
        time.sleep(0.2)
    rows = [load_json(path, label=f"topology rank {index}") for index, path in enumerate(paths)]
    receipt = validate_topology_rows(
        rows, job0=args.expected_job0, node0=args.expected_node0,
        job1=args.expected_job1, node1=args.expected_node1,
    )
    receipt_path = topology_dir / "physical2x2-receipt.json"
    if rank == 0:
        write_exclusive_json(receipt_path, receipt)
    while not receipt_path.is_file() or receipt_path.is_symlink():
        if time.monotonic() >= deadline:
            fail("timed out waiting for physical topology receipt")
        time.sleep(0.2)
    published = load_json(receipt_path, label="physical topology receipt")
    digest = validate_embedded_digest(published, label="physical topology")
    if {key: value for key, value in published.items() if key != "receipt_digest"} != receipt:
        fail("physical topology receipt changed during publication")
    return receipt_path, file_sha256(receipt_path), digest


def _plain_under(path_text: Any, root: Path, expected_name: str) -> Path:
    if not isinstance(path_text, str):
        fail(f"artifact path differs: {expected_name}")
    path = Path(path_text)
    if path != root / expected_name or not path.is_file() or path.is_symlink():
        fail(f"artifact closure differs: {expected_name}")
    return path


def bind_display_receipt(
    *, args: argparse.Namespace, candidate: Mapping[str, Any],
    topology_path: Path, topology_sha256: str, topology_digest: str,
) -> Path:
    output = Path(args.output_dir)
    native_path = _plain_under(str(output / "receipt.json"), output, "receipt.json")
    native_receipt = load_json(native_path, label="native T2V receipt")
    validate_embedded_digest(native_receipt, label="native T2V")
    sys.path.insert(0, str(Path(args.method_root)))
    import infer_pair_v5_t2v_calibration_bank as bank
    artifacts = bank._verify_native_receipt(native_receipt, candidate)
    mp4 = bank._verify_file_artifact(artifacts["mp4"], "display T2V MP4")
    clean = bank._verify_file_artifact(artifacts["predecode_clean_latent"], "display clean latent")
    gaussian = bank._verify_file_artifact(artifacts["official_initial_gaussian"], "display Gaussian")
    mp4_path = _plain_under(mp4.get("path"), output, "t2v.mp4")
    clean_path = _plain_under(clean.get("path"), output, "t2v.normalized-clean-latent.safetensors")
    gaussian_path = _plain_under(gaussian.get("path"), output, "t2v.official-initial-gaussian.safetensors")
    if (
        native_receipt.get("method_source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or native_receipt.get("method_source_revision") != SOURCE_REVISION
        or native_receipt.get("scientific_claim_authorized") is not False
        or native_receipt.get("production_claim_forbidden") is not True
        or native_receipt.get("source_condition_artifact") is not None
        or native_receipt.get("arms") != ["t2v"]
    ):
        fail("native display-only authority differs")
    output_row = native_receipt["outputs"]["t2v"]
    geometry = native_receipt["latent_geometry"]
    if (
        [output_row.get("height"), output_row.get("width")] != EXPECTED_HW
        or output_row.get("frame_count") != 81
        or output_row.get("fps") != 25
        or geometry.get("video_latent_shape") != EXPECTED_LATENT_SHAPE
        or clean.get("shape") != EXPECTED_LATENT_SHAPE
        or gaussian.get("shape") != EXPECTED_LATENT_SHAPE
        or gaussian.get("generator_initial_seed") != SEED
    ):
        fail("native exact81 display geometry differs")
    conditioning = native_receipt["conditioning"]["t2v"]
    if (
        conditioning.get("full_source_video_count") != 0
        or conditioning.get("source_derived_reference_count") != 0
        or conditioning.get("reference_encoding") != "none"
        or conditioning.get("source_ids", {}).get("conditioning_source_count") != 0
    ):
        fail("source content entered pure T2V display generation")
    receipt = {
        "schema_version": SCHEMA,
        "complete": True,
        "iid": IID,
        "semantic_branch": candidate["semantic_branch"],
        "candidate_id": candidate["candidate_id"],
        "analysis_split": "fit",
        "authoring_authority": {
            "selection_sha256": SELECTION_SHA256,
            "registry_sha256": REGISTRY_SHA256,
            "row_digest": ROW_DIGEST,
            "seed": SEED,
            "prompt_sha256": candidate["full_t2v_caption_utf8_sha256"],
            "source_video_sha256": SOURCE_VIDEO_SHA256,
            "source_video_role": "exact81_bucket_geometry_probe_only",
        },
        "method_authority": {
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_revision": SOURCE_REVISION,
            "native_generator_sha256": METHOD_FILE_SHA256["infer_native_identity_generation_canary.py"],
            "bank_wrapper_sha256": METHOD_FILE_SHA256["infer_pair_v5_t2v_calibration_bank.py"],
            "old_bank_wrapper_invoked": False,
        },
        "physical_topology": {
            "path": str(topology_path), "sha256": topology_sha256,
            "receipt_digest": topology_digest, "world_size": 4,
            "physical_nodes": 2, "local_ranks_per_node": 2,
        },
        "native_receipt": {
            "path": str(native_path), "sha256": file_sha256(native_path),
            "receipt_digest": native_receipt["receipt_digest"],
        },
        "output": {
            "mp4_path": str(mp4_path), "mp4_sha256": file_sha256(mp4_path),
            "frame_count": 81, "fps": 25, "height": EXPECTED_HW[0],
            "width": EXPECTED_HW[1], "latent_shape": EXPECTED_LATENT_SHAPE,
            "clean_latent_path": str(clean_path), "clean_latent_sha256": file_sha256(clean_path),
            "gaussian_path": str(gaussian_path), "gaussian_sha256": file_sha256(gaussian_path),
        },
        "use_contract": {
            "display_only": True,
            "stage_b_condition": False,
            "passed_to_stage_b_runtime": False,
            "used_as_model_condition": False,
            "anchor_pixels_or_latent_transplanted": False,
            "source_pixels_entered_t2v_transformer": False,
            "old40_bank_audit_claimed": False,
            "old40_bank_receipt_created": False,
            "action_success_claimed": False,
            "scientific_claim_authorized": False,
            "training_or_parameter_update_performed": False,
        },
    }
    path = output / "display-anchor-receipt.json"
    write_exclusive_json(path, receipt)
    return path


def run(args: argparse.Namespace) -> int:
    if args.method_source_archive_sha256 != SOURCE_ARCHIVE_SHA256 or args.method_source_revision != SOURCE_REVISION:
        fail("method source authority differs")
    method_root = Path(args.method_root)
    validate_method_root(method_root)
    row = load_authoring(Path(args.selection), Path(args.registry))
    candidate = candidate_from_authoring(row, args.branch)
    source = Path(args.source_video)
    if str(source) != SOURCE_VIDEO_PATH or not source.is_file() or source.is_symlink() or file_sha256(source) != SOURCE_VIDEO_SHA256:
        fail("IID00435 geometry source differs")
    manifest = Path(args.checkpoint_content_manifest)
    if not manifest.is_file() or manifest.is_symlink() or file_sha256(manifest) != CHECKPOINT_MANIFEST_SHA256:
        fail("checkpoint content manifest differs")
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink() or not output.parent.is_dir():
        fail("native output must be a fresh absolute directory")
    topology_path, topology_sha, topology_digest = capture_topology(args)
    sys.path.insert(0, str(method_root))
    import infer_native_identity_generation_canary as native
    native_argv = [
        "--bernini-root", args.bernini_root,
        "--veomni-root", args.veomni_root,
        "--checkpoint", args.checkpoint,
        "--checkpoint-content-manifest", args.checkpoint_content_manifest,
        "--source-video", str(source),
        "--expected-source-sha256", SOURCE_VIDEO_SHA256,
        "--action-prompt", candidate["full_t2v_caption"],
        "--expected-action-prompt-sha256", candidate["full_t2v_caption_utf8_sha256"],
        "--output-dir", str(output), "--arms", "t2v",
        "--num-inference-steps", "40", "--seed", str(SEED),
        "--expected-bernini-commit", BERNINI_COMMIT,
        "--expected-veomni-commit", VEOMNI_COMMIT,
        "--expected-checkpoint-tree-sha256", CHECKPOINT_TREE_SHA256,
        "--method-source-revision", SOURCE_REVISION,
        "--method-source-archive-sha256", SOURCE_ARCHIVE_SHA256,
    ]
    status = native.main(native_argv)
    if status == 0 and int(os.environ["RANK"]) == 0:
        bind_display_receipt(
            args=args, candidate=candidate, topology_path=topology_path,
            topology_sha256=topology_sha, topology_digest=topology_digest,
        )
    return int(status)


def validate_display_receipt(path: Path, *, branch: str) -> dict[str, Any]:
    receipt = load_json(path, label=f"{branch} display receipt")
    validate_embedded_digest(receipt, label=f"{branch} display")
    if (
        receipt.get("schema_version") != SCHEMA or receipt.get("complete") is not True
        or receipt.get("iid") != IID or receipt.get("semantic_branch") != branch
        or receipt.get("analysis_split") != "fit"
        or receipt.get("candidate_id") != f"pair5-t2v-reserve4-v1-{IID}-{branch}"
    ):
        fail(f"{branch} display receipt authority differs")
    if receipt.get("authoring_authority") != {
        "selection_sha256": SELECTION_SHA256,
        "registry_sha256": REGISTRY_SHA256,
        "row_digest": ROW_DIGEST,
        "seed": SEED,
        "prompt_sha256": PROMPT_SHA256[branch],
        "source_video_sha256": SOURCE_VIDEO_SHA256,
        "source_video_role": "exact81_bucket_geometry_probe_only",
    }:
        fail(f"{branch} authoring authority differs")
    if receipt.get("method_authority") != {
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_revision": SOURCE_REVISION,
        "native_generator_sha256": METHOD_FILE_SHA256["infer_native_identity_generation_canary.py"],
        "bank_wrapper_sha256": METHOD_FILE_SHA256["infer_pair_v5_t2v_calibration_bank.py"],
        "old_bank_wrapper_invoked": False,
    }:
        fail(f"{branch} method authority differs")
    use = receipt.get("use_contract")
    expected_use = {
        "display_only": True, "stage_b_condition": False,
        "passed_to_stage_b_runtime": False, "used_as_model_condition": False,
        "anchor_pixels_or_latent_transplanted": False,
        "source_pixels_entered_t2v_transformer": False,
        "old40_bank_audit_claimed": False, "old40_bank_receipt_created": False,
        "action_success_claimed": False, "scientific_claim_authorized": False,
        "training_or_parameter_update_performed": False,
    }
    if use != expected_use:
        fail(f"{branch} display-only contract differs")
    output = receipt.get("output")
    if not isinstance(output, dict) or [output.get("height"), output.get("width")] != EXPECTED_HW or output.get("frame_count") != 81 or output.get("fps") != 25 or output.get("latent_shape") != EXPECTED_LATENT_SHAPE:
        fail(f"{branch} exact81 output receipt differs")
    root = path.parent
    run_root = path.parents[2]
    if path != run_root / "outputs" / branch / "display-anchor-receipt.json":
        fail(f"{branch} display receipt path is non-canonical")
    for key, expected_name, sha_key in (
        ("mp4_path", "t2v.mp4", "mp4_sha256"),
        ("clean_latent_path", "t2v.normalized-clean-latent.safetensors", "clean_latent_sha256"),
        ("gaussian_path", "t2v.official-initial-gaussian.safetensors", "gaussian_sha256"),
    ):
        artifact = _plain_under(output.get(key), root, expected_name)
        if file_sha256(artifact) != output.get(sha_key):
            fail(f"{branch} artifact SHA differs: {expected_name}")
    topology = receipt.get("physical_topology")
    if not isinstance(topology, dict) or topology.get("world_size") != 4 or topology.get("physical_nodes") != 2 or topology.get("local_ranks_per_node") != 2:
        fail(f"{branch} physical topology binding differs")
    topology_path = Path(str(topology.get("path")))
    expected_topology_path = (
        run_root / "topology" / branch / "physical2x2-receipt.json"
    )
    if topology_path != expected_topology_path:
        fail(f"{branch} topology receipt path is non-canonical")
    topology_receipt = load_json(topology_path, label=f"{branch} topology")
    if file_sha256(topology_path) != topology.get("sha256") or validate_embedded_digest(topology_receipt, label=f"{branch} topology") != topology.get("receipt_digest"):
        fail(f"{branch} topology receipt SHA differs")
    validate_topology_rows(
        topology_receipt["rank_rows"],
        job0=topology_receipt["holder_pair"][0],
        node0=topology_receipt["hostname_order"][0],
        job1=topology_receipt["holder_pair"][1],
        node1=topology_receipt["hostname_order"][1],
    )
    native = receipt.get("native_receipt")
    if not isinstance(native, dict):
        fail(f"{branch} native receipt binding differs")
    native_path = _plain_under(native.get("path"), root, "receipt.json")
    native_receipt = load_json(native_path, label=f"{branch} native receipt")
    if (
        file_sha256(native_path) != native.get("sha256")
        or validate_embedded_digest(native_receipt, label=f"{branch} native")
        != native.get("receipt_digest")
        or native_receipt.get("method_source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or native_receipt.get("method_source_revision") != SOURCE_REVISION
        or native_receipt.get("arms") != ["t2v"]
        or native_receipt.get("scientific_claim_authorized") is not False
        or native_receipt.get("production_claim_forbidden") is not True
        or native_receipt.get("source_condition_artifact") is not None
    ):
        fail(f"{branch} native receipt authority differs")
    return receipt


def verify_set(args: argparse.Namespace) -> int:
    branches = PROFILE_BRANCHES[args.profile]
    root = Path(args.root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        fail("anchor set root must be an absolute plain directory")
    rows = []
    holder_pair = None
    hostname_order = None
    gaussian_identity = None
    for branch in branches:
        path = root / "outputs" / branch / "display-anchor-receipt.json"
        receipt = validate_display_receipt(path, branch=branch)
        topology = load_json(Path(receipt["physical_topology"]["path"]), label=f"{branch} topology")
        native = load_json(Path(receipt["native_receipt"]["path"]), label=f"{branch} native")
        noise_rows = native.get("initial_noise_artifacts")
        gaussian = noise_rows.get("t2v") if isinstance(noise_rows, dict) else None
        if not isinstance(gaussian, dict) or (
            _SHA256.fullmatch(str(gaussian.get("raw_value_sha256"))) is None
            or _SHA256.fullmatch(str(gaussian.get("content_sha256"))) is None
            or gaussian.get("tensor_value_sha256") != gaussian.get("raw_value_sha256")
            or gaussian.get("shape") != EXPECTED_LATENT_SHAPE
            or gaussian.get("dtype") != "torch.float32"
            or type(gaussian.get("generator_initial_seed")) is not int
            or gaussian.get("generator_initial_seed") != SEED
            or gaussian.get("path") != receipt["output"]["gaussian_path"]
            or gaussian.get("sha256") != receipt["output"]["gaussian_sha256"]
            or gaussian.get("captured_from_native_sampler") is not True
            or gaussian.get("external_initial_noise_injection") is not False
            or gaussian.get("source_or_target_derived") is not False
            or gaussian.get("observer_changed_return_value") is not False
            or gaussian.get("official_randn_tensor_call_count") != 1
        ):
            fail(f"{branch} official initial Gaussian authority differs")
        current_gaussian = {
            "raw_value_sha256": gaussian["raw_value_sha256"],
            "content_sha256": gaussian["content_sha256"],
            "shape": gaussian["shape"],
            "dtype": gaussian["dtype"],
            "generator_initial_seed": gaussian["generator_initial_seed"],
        }
        if gaussian_identity is None:
            gaussian_identity = current_gaussian
        elif current_gaussian != gaussian_identity:
            fail("anchor branches did not share one official initial Gaussian tensor value")
        current_pair = topology["holder_pair"]
        current_hosts = topology["hostname_order"]
        if holder_pair is None:
            holder_pair, hostname_order = current_pair, current_hosts
        elif current_pair != holder_pair or current_hosts != hostname_order:
            fail("anchor branches used different physical holder mapping")
        rows.append({
            "semantic_branch": branch,
            "directory": str(path.parent),
            "display_receipt_sha256": file_sha256(path),
            "display_receipt_digest": receipt["receipt_digest"],
            "mp4_path": receipt["output"]["mp4_path"],
            "mp4_sha256": receipt["output"]["mp4_sha256"],
            "official_initial_gaussian_tensor_identity": current_gaussian,
        })
    unsigned = {
        "schema_version": SET_SCHEMA,
        "complete": True,
        "iid": IID,
        "requested_profile": args.profile,
        "generation_order": list(branches),
        "action_canary_first": True,
        "same_official_initial_gaussian_tensor_value_across_requested_branches": True,
        "branches": rows,
        "physical_topology": {
            "world_size": 4, "physical_nodes": 2,
            "local_ranks_per_node": 2, "holder_pair": holder_pair,
            "hostname_order": hostname_order,
        },
        "use_contract": {
            "display_only": True, "stage_b_condition": False,
            "passed_to_stage_b_runtime": False,
            "old40_bank_audit_claimed": False,
            "action_success_claimed": False,
            "scientific_claim_authorized": False,
        },
    }
    output = root / "display-anchor-set-receipt.json"
    write_exclusive_json(output, unsigned)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    for name in (
        "method-root", "selection", "registry", "source-video", "output-dir",
        "topology-dir", "expected-job0", "expected-node0", "expected-job1",
        "expected-node1", "bernini-root", "veomni-root", "checkpoint",
        "checkpoint-content-manifest", "method-source-revision",
        "method-source-archive-sha256",
    ):
        run_parser.add_argument(f"--{name}", required=True)
    run_parser.add_argument("--branch", required=True, choices=BRANCH_ORDER)
    verify = sub.add_parser("verify-set")
    verify.add_argument("--root", required=True)
    verify.add_argument("--profile", required=True, choices=tuple(PROFILE_BRANCHES))
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "verify-set":
        return verify_set(args)
    fail("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRANCH_ORDER", "DisplayAnchorError", "PROFILE_BRANCHES",
    "candidate_from_authoring", "canonical_json_bytes", "load_authoring",
    "main", "validate_display_receipt", "validate_topology_rows",
]
