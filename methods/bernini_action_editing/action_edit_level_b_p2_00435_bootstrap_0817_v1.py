#!/usr/bin/env python3
"""Sealed bootstrap for the one-shot 0817 Level-B P2 product render.

This file is launcher authority, not a sixth member of the Level-B runtime
release.  It authenticates the exact-five Level-B source closure, then the
already frozen Level-A checkpoint-consumer closure, before constructing P2 in
one fresh WORLD8 process and invoking the source+instruction-only product API.

This checked-in launcher is frozen to the compatibility-QA-approved Level-B
renderer and exact-five runtime release.  Deployment and execution remain
separate, explicitly authorized operations.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, NoReturn, Sequence, Tuple


METHOD = "bernini-action-edit-level-b-p2-00435-bootstrap-0817-v1"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
TAG = "fresh-world8-level-b-p2-00435-v1"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
PARENT_JOB_ID = 140846
PINNED_NODE = "auh7-1b-gpu-279"

EXPERIMENT_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_action_editing_0817"
)
LEVEL_B_RELEASE_ROOT = EXPERIMENT_ROOT / "releases" / TAG
LEVEL_B_MANIFEST = LEVEL_B_RELEASE_ROOT / "RELEASE_MANIFEST.json"
LEVEL_B_MANIFEST_SHA256 = (
    "8cf24d45f64eed4d6bc3e02b60d68ab997ab8f56ccdfe33849e04dc7c6f684bf"
)
LEVEL_B_RENDERER_SHA256 = (
    "2b807aec19c17953a890ef76b9164b786af2b7f1912c32b2c33194c15ca29eed"
)
LEVEL_B_RENDERER_SIZE = 235823

LEVEL_A_RELEASE_ROOT = (
    EXPERIMENT_ROOT / "releases" / "fresh-world8-level-a-r2-p2-launchbound-v2"
)
LEVEL_A_MANIFEST = LEVEL_A_RELEASE_ROOT / "RELEASE_MANIFEST.json"
LEVEL_A_MANIFEST_SHA256 = (
    "f9e9f8542ec701cc9890fed919695980b989fd6d731eb914a5588edb1de4eeaa"
)
LEVEL_A_DRIVER_SHA256 = (
    "6435c6bb06a79cfcb407c137571404e5962e0de50e8082e7bd600e4618c05ea4"
)
LEVEL_A_CONSUMER_SHA256 = (
    "8bf0a9e48e0b2443a8e2f8e0744d08591226a167ac6ace45ee513481f5a97b3a"
)
LEVEL_A_PRODUCT_SHA256 = (
    "b16d8aef25b35df13e8294ef387e4d334170af65c2f43ece9894142d7cadac14"
)

R2_RELEASE_MANIFEST = (
    EXPERIMENT_ROOT
    / "releases"
    / "pre-d0-paired2-edf3d1d2a77c-r2"
    / "RELEASE_MANIFEST.json"
)
R2_RELEASE_MANIFEST_SHA256 = (
    "671179995a64f20ee773273e84b5eb3f1f0bbd018fbfa3c0c6dc41d56c5555f5"
)
CAMPAIGN_RECEIPT = (
    EXPERIMENT_ROOT
    / "runs"
    / "pre_d0_engineering_paired2-edf3d1d2a77c-r2"
    / "receipt.json"
)
CAMPAIGN_RECEIPT_SHA256 = (
    "8014b7b71413318d80162fba12b73d83d6b9d9de5ea57ad295643a238b0f8c0e"
)
CHECKPOINT_DIR = (
    EXPERIMENT_ROOT
    / "runs"
    / "pre_d0_engineering_paired2-edf3d1d2a77c-r2"
    / "checkpoints"
    / "checkpoint-00000002"
)
P2_PARAMETER_SHA256 = (
    "5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e"
)
P2_METADATA_SHA256 = (
    "b9a798e525666b352a64bb3030ebd8953b5847ba3d9a5de216735587cae0e5e4"
)

BERNINI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
)
VEOMNI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
)
BASE_CHECKPOINT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
BASE_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
CHECKPOINT_CONTENT_MANIFEST = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/"
    "runtime/methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)

SOURCE_VIDEO = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/"
    "00435ad621c44fac/samples/00435ad621c44fac/source_video.mp4"
)
SOURCE_VIDEO_SHA256 = (
    "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1"
)
SOURCE_INPUT_HW = (1056, 704)
SOURCE_BUCKET_HW = (592, 400)
SOURCE_FRAMES = 81
SOURCE_FPS = 25
EDIT_INSTRUCTION = (
    "The main woman slowly lowers both raised arms, places both hands firmly "
    "on her hips, and holds that completed pose while the camera stays locked."
)
EDIT_INSTRUCTION_SHA256 = (
    "cfe1e51a8b8ada76c5b1d6993cfb8d55cbc1f21fb0694a14ddb9c11133f74088"
)
FULL_TRAINING_PROMPT_SHA256 = (
    "cf88e9f947d47b6b8fa02db4cad8d40646d31b5839512aedb7736608eca55bff"
)
INFERENCE_SEED = 2026080821
OUTPUT_MP4 = (
    EXPERIMENT_ROOT
    / "runs"
    / TAG
    / "00435ad621c44fac_p2_seed2026080821.mp4"
)

PYTHON_PATH = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

LEVEL_B_MEMBER_PINS = {
    "action_preservation_decoded_eval_model_authority_v2.py": (
        "413508d42551fd1ab0ff83d9af7b29f144b1f6bcdccf29ed7c590417e3384ecb",
        114872,
    ),
    "infer_action_edit_level_b_renderer_0817_v1.py": (
        LEVEL_B_RENDERER_SHA256,
        LEVEL_B_RENDERER_SIZE,
    ),
    "infer_lora.py": (
        "c2e55a4ea41a21d0761e660ab630002b1bc569705e8c0bcafa1bc8c6c38ccc06",
        151393,
    ),
    "tools/build_renderer_dataset.py": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        31012,
    ),
    "tools/materialize_vae.py": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        32195,
    ),
}

LEVEL_A_MEMBER_PINS = {
    "action_edit_checkpoint_consumer_0817_v1.py": (LEVEL_A_CONSUMER_SHA256, 70074),
    "action_edit_fresh_world8_level_a_driver_0817_v1.py": (
        LEVEL_A_DRIVER_SHA256,
        90726,
    ),
    "action_plan_predictor_v1.py": (
        "464cd500f0ba1edb6cbe6d4f07287bfff346ae0ba7968c0d7c7f3cc7cb667308",
        58087,
    ),
    "clean_source_visual_context_stage_b_contract_v1.py": (
        "f782876fd2b90b7b1d517fc49db03b800f1d9924156575275a472e3ea79ff571",
        37557,
    ),
    "infer_action_edit_product_abi_0817_v1.py": (LEVEL_A_PRODUCT_SHA256, 89668),
    "inference_sigma_strata.py": (
        "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
        17956,
    ),
    "packed_preservation_lora_v2.py": (
        "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6",
        30419,
    ),
    "packed_preservation_release_v2.py": (
        "581e7314f9ca403bc8f0aa3d7e82adb57a9a202f8318f99256a002ecd255b99c",
        15953,
    ),
    "source_self_runtime.py": (
        "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
        36607,
    ),
    "train_action_edit_large_lora_0817_v1.py": (
        "edf3d1d2a77cb2f713968f537ce85a7d92f0b7347a0474419fe5562fbd319bd9",
        130923,
    ),
    "train_lora.py": (
        "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e",
        84216,
    ),
}


class BootstrapError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise BootstrapError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_finalized() -> None:
    pending = {
        "LEVEL_B_MANIFEST_SHA256": LEVEL_B_MANIFEST_SHA256,
        "LEVEL_B_RENDERER_SHA256": LEVEL_B_RENDERER_SHA256,
    }
    bad = sorted(
        name
        for name, value in pending.items()
        if not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
    )
    if bad or type(LEVEL_B_RENDERER_SIZE) is not int or LEVEL_B_RENDERER_SIZE <= 0:
        fail("Level-B launch authority is not fully frozen")


def _plain_file(path: Path, *, mode: int = None, nlink: int = 1) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail("pinned file is not one canonical plain absolute path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (
        nlink is not None and info.st_nlink != nlink
    ):
        fail("pinned file type/link closure differs")
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        fail("pinned file mode differs")
    return path


def _plain_dir(path: Path, *, mode: int = None) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail("pinned directory is not one canonical plain absolute path")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        fail("pinned directory type differs")
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        fail("pinned directory mode differs")
    return path


def _stable_bytes(path: Path, *, expected_sha: str, expected_size: int) -> bytes:
    _plain_file(path, mode=0o444)
    before = path.lstat()
    raw = path.read_bytes()
    after = path.lstat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest() != expected_sha
    ):
        fail("pinned source changed during stable read")
    return raw


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise BootstrapError("authority file is not strict JSON") from error
    if not isinstance(value, Mapping):
        fail("authority JSON root differs")
    return value


def _load_module(name: str, path: Path, raw: bytes, *, initial: Mapping[str, Any] = None) -> ModuleType:
    if name in sys.modules:
        fail("authenticated module was preimported")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__cached__ = None
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    if initial:
        module.__dict__.update(dict(initial))
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _authenticate_level_b_module() -> ModuleType:
    _require_finalized()
    root = _plain_dir(LEVEL_B_RELEASE_ROOT, mode=0o555)
    if sorted(item.name for item in root.iterdir()) != [
        "RELEASE_MANIFEST.json",
        "action_preservation_decoded_eval_model_authority_v2.py",
        "infer_action_edit_level_b_renderer_0817_v1.py",
        "infer_lora.py",
        "tools",
    ]:
        fail("Level-B exact-five physical root closure differs")
    manifest = _plain_file(LEVEL_B_MANIFEST, mode=0o444)
    manifest_raw = manifest.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != LEVEL_B_MANIFEST_SHA256:
        fail("Level-B manifest SHA differs")
    payload = _strict_json(manifest_raw)
    rows = payload.get("members")
    if (
        set(payload)
        != {
            "schema_version",
            "authority",
            "member_count",
            "members",
            "release_digest",
        }
        or payload.get("release_digest")
        != hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
        or payload.get("schema_version")
        != "bernini-action-edit-level-b-runtime-release-v1"
        or payload.get("authority") != AUTHORITY
        or payload.get("member_count") != 5
        or not isinstance(rows, list)
        or len(rows) != 5
    ):
        fail("Level-B manifest envelope differs")
    expected_paths = list(LEVEL_B_MEMBER_PINS)
    if [row.get("path") if isinstance(row, Mapping) else None for row in rows] != expected_paths:
        fail("Level-B exact member order differs")
    for row in rows:
        relative = row["path"]
        sha, size = LEVEL_B_MEMBER_PINS[relative]
        if row != {"path": relative, "sha256": sha, "size": size, "mode": 0o444}:
            fail("Level-B member row differs")
        _stable_bytes(root / relative, expected_sha=sha, expected_size=size)
    renderer_path = root / "infer_action_edit_level_b_renderer_0817_v1.py"
    renderer_raw = _stable_bytes(
        renderer_path,
        expected_sha=LEVEL_B_RENDERER_SHA256,
        expected_size=LEVEL_B_RENDERER_SIZE,
    )
    # The manifest pin exists in the new module globals before its first
    # source instruction executes; the Level-B module immediately consumes
    # and deletes this bootstrap-only authority.
    sealed_globals = {
        "_LEVEL_B_SEALED_LAUNCHER_EXPECTED_MANIFEST_SHA256": (
            LEVEL_B_MANIFEST_SHA256
        )
    }
    module = _load_module(
        "infer_action_edit_level_b_renderer_0817_v1",
        renderer_path,
        renderer_raw,
        initial=sealed_globals,
    )
    if tuple(module.LEVEL_B_RELEASE_MEMBER_PATHS) != tuple(LEVEL_B_MEMBER_PINS):
        fail("loaded Level-B exact-five member authority differs")
    return module


def _authenticate_level_a_consumer() -> Any:
    root = _plain_dir(LEVEL_A_RELEASE_ROOT, mode=0o555)
    manifest_raw = _plain_file(LEVEL_A_MANIFEST, mode=0o444).read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != LEVEL_A_MANIFEST_SHA256:
        fail("Level-A sealed release manifest differs")
    payload = _strict_json(manifest_raw)
    rows = payload.get("files")
    expected_paths = sorted(LEVEL_A_MEMBER_PINS)
    if (
        payload.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-release-manifest-v1"
        or payload.get("member_root") != "methods/bernini_action_editing"
        or not isinstance(rows, list)
        or [row.get("path") if isinstance(row, Mapping) else None for row in rows]
        != expected_paths
        or sorted(item.name for item in root.iterdir())
        != sorted(expected_paths + ["RELEASE_MANIFEST.json"])
    ):
        fail("Level-A exact-eleven release closure differs")
    sources = {}
    for row in rows:
        relative = row["path"]
        sha, size = LEVEL_A_MEMBER_PINS[relative]
        if row != {"path": relative, "sha256": sha, "size": size, "mode": 0o444}:
            fail("Level-A sealed member row differs")
        sources[relative] = _stable_bytes(
            root / relative, expected_sha=sha, expected_size=size
        )
    driver_path = root / "action_edit_fresh_world8_level_a_driver_0817_v1.py"
    driver = _load_module(
        "_level_b_authenticated_level_a_driver",
        driver_path,
        sources[driver_path.name],
    )
    release = driver.validate_deployment_release(
        LEVEL_A_MANIFEST,
        expected_manifest_sha256=LEVEL_A_MANIFEST_SHA256,
        expected_driver_sha256=LEVEL_A_DRIVER_SHA256,
    )
    driver.validate_executed_driver(release)
    consumer = driver.load_consumer_from_authenticated_bytes(release)
    if (
        consumer.PINNED_R2_RELEASE_MANIFEST_SHA256 != R2_RELEASE_MANIFEST_SHA256
        or consumer.PINNED_R2_CAMPAIGN_RECEIPT_SHA256 != CAMPAIGN_RECEIPT_SHA256
        or consumer.PINNED_R2_P_STATE_SHA256.get(2) != P2_PARAMETER_SHA256
    ):
        fail("loaded Level-A consumer trust roots differ")
    return consumer


def _verify_nonsemantic_authorities() -> None:
    if Path(sys.executable).resolve(strict=True) != PYTHON_PATH:
        fail("running interpreter path differs")
    python_raw = PYTHON_PATH.read_bytes()
    if hashlib.sha256(python_raw).hexdigest() != PYTHON_SHA256:
        fail("running interpreter bytes differ")
    fixed = (
        (R2_RELEASE_MANIFEST, R2_RELEASE_MANIFEST_SHA256),
        (CAMPAIGN_RECEIPT, CAMPAIGN_RECEIPT_SHA256),
        (CHECKPOINT_DIR / "metadata.json", P2_METADATA_SHA256),
        (CHECKPOINT_CONTENT_MANIFEST, CHECKPOINT_CONTENT_MANIFEST_SHA256),
    )
    for path, expected in fixed:
        if hashlib.sha256(_plain_file(path).read_bytes()).hexdigest() != expected:
            fail("fixed checkpoint/campaign authority differs")
    if not BASE_CHECKPOINT.is_dir() or not BERNINI_ROOT.is_dir() or not VEOMNI_ROOT.is_dir():
        fail("fixed model/vendor root is absent")


def run_world8() -> Mapping[str, Any]:
    _require_finalized()
    if hashlib.sha256(EDIT_INSTRUCTION.encode("utf-8")).hexdigest() != EDIT_INSTRUCTION_SHA256:
        fail("embedded edit instruction bytes differ")
    _verify_nonsemantic_authorities()
    level_b = _authenticate_level_b_module()
    verified_runtime = level_b.authenticate_level_b_runtime_release(LEVEL_B_MANIFEST)
    consumer = _authenticate_level_a_consumer()
    bundle = consumer.consume_frozen_r2_world8_checkpoint(
        release_manifest_path=R2_RELEASE_MANIFEST,
        campaign_receipt_path=CAMPAIGN_RECEIPT,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_step=2,
        bernini_root=BERNINI_ROOT,
        veomni_root=VEOMNI_ROOT,
        base_checkpoint=BASE_CHECKPOINT,
        checkpoint_content_manifest=CHECKPOINT_CONTENT_MANIFEST,
        expected_consumer_source_sha256=LEVEL_A_CONSUMER_SHA256,
        expected_product_source_sha256=LEVEL_A_PRODUCT_SHA256,
    )
    receipt = level_b.run_level_b_pre_d0_offline_inference(
        fresh_bundle=bundle,
        verified_runtime=verified_runtime,
        source_video_path=str(SOURCE_VIDEO),
        expected_source_video_sha256=SOURCE_VIDEO_SHA256,
        edit_instruction=EDIT_INSTRUCTION,
        inference_seed=INFERENCE_SEED,
        output_mp4_path=str(OUTPUT_MP4),
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("complete") is not True
        or receipt.get("authority") != AUTHORITY
        or receipt.get("full40_denoise_executed") is not True
        or receipt.get("full_bernini_renderer_denoise_verified") is not True
        or receipt.get("offline_product_inference_completed") is not True
        or receipt.get("mp4_emitted") is not True
        or receipt.get("promotion_authorized") is not False
        or receipt.get("counts_as_d0") is not False
    ):
        fail("Level-B WORLD8 terminal receipt differs")
    return receipt


def validate_product() -> Mapping[str, Any]:
    _verify_nonsemantic_authorities()
    level_b = _authenticate_level_b_module()
    level_b.authenticate_level_b_runtime_release(LEVEL_B_MANIFEST)
    committed = level_b.validate_committed_level_b_product(output_mp4_path=OUTPUT_MP4)
    receipt_path = OUTPUT_MP4.with_name(OUTPUT_MP4.name + ".receipt.json")
    marker_path = OUTPUT_MP4.with_name(OUTPUT_MP4.name + ".COMMITTED.json")
    receipt_file = _plain_file(receipt_path, mode=0o444, nlink=2)
    marker_file = _plain_file(marker_path, mode=0o444, nlink=2)
    receipt_info = receipt_file.lstat()
    marker_info = marker_file.lstat()
    payload = _strict_json(receipt_file.read_bytes())
    request = payload.get("request")
    policy = request.get("inference_policy") if isinstance(request, Mapping) else None
    output = payload.get("output_mp4")
    required_true = (
        "complete",
        "full40_denoise_executed",
        "full_bernini_renderer_denoise_verified",
        "offline_product_inference_completed",
        "mp4_emitted",
        "ffprobe_and_full_decode_verified",
        "clean_source_prefix_plus_evolving_noisy_target_verified",
        "exact30_target_only_action_hooks_once_per_denoise_step",
    )
    required_false = (
        "formal_training_started",
        "counts_as_d0",
        "scientific_claim_authorized",
        "action_quality_claim_authorized",
        "clean_target_or_anchor_consumed",
        "teacher_or_external_annotation_consumed",
        "hidden_user_callback_or_custom_denoiser_consumed",
        "promotion_authorized",
    )
    if (
        payload.get("authority") != AUTHORITY
        or any(payload.get(key) is not True for key in required_true)
        or any(payload.get(key) is not False for key in required_false)
        or not isinstance(request, Mapping)
        or request.get("source_video_path") != str(SOURCE_VIDEO)
        or request.get("source_video_sha256") != SOURCE_VIDEO_SHA256
        or request.get("instruction_utf8_sha256") != EDIT_INSTRUCTION_SHA256
        or request.get("clean_target_present") is not False
        or request.get("anchor_present") is not False
        or request.get("teacher_or_external_annotation_present") is not False
        or not isinstance(policy, Mapping)
        or policy.get("seed") != INFERENCE_SEED
        or not isinstance(output, Mapping)
        or output.get("mp4_path") != str(OUTPUT_MP4)
        or output.get("mp4_sha256") != committed.get("mp4_sha256")
        or output.get("ffprobe_exact81") is not True
        or output.get("full_decode_frame_count") != SOURCE_FRAMES
        or output.get("ffprobe_geometry_hw") != list(SOURCE_BUCKET_HW)
        or receipt_info.st_dev != marker_info.st_dev
        or receipt_info.st_ino != marker_info.st_ino
        or receipt_info.st_size != marker_info.st_size
        or receipt_info.st_mtime_ns != marker_info.st_mtime_ns
        or committed.get("receipt_inode_alias_marker_verified") is not True
        or committed.get("receipt_file_identity")
        != committed.get("commit_marker_file_identity")
        or committed.get("receipt_sha256")
        != committed.get("commit_marker_sha256")
    ):
        fail("committed Level-B receipt semantic/claim closure differs")
    return {
        "schema_version": "bernini-action-edit-level-b-p2-product-validation-v1",
        "method": METHOD,
        "authority": AUTHORITY,
        "output_mp4": str(OUTPUT_MP4),
        "validation": dict(committed),
        "receipt_claims_revalidated": True,
        "receipt_inode_alias_marker_revalidated": True,
        "committed_marker_required": True,
        "formal_training_started": False,
        "counts_as_d0": False,
        "promotion_authorized": False,
    }


def main(argv: Sequence[str] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["run"]:
        receipt = run_world8()
        print(
            "LEVEL_B_WORLD8_COMPLETE receipt_digest="
            + str(receipt.get("receipt_digest")),
            flush=True,
        )
        return 0
    if values == ["validate-product"]:
        sys.stdout.buffer.write(canonical_json_bytes(validate_product()))
        return 0
    fail("bootstrap accepts exactly one fixed subcommand: run or validate-product")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as error:
        print(f"Level-B P2 bootstrap refused: {error}", file=sys.stderr, flush=True)
        raise SystemExit(94)
